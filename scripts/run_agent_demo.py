"""Run the five demo conversations with the real LLM and write reports/agent_demo_transcripts.md.

Needs .env (Gemini key, optional Groq fallback). Expected ~6 LLM calls, ~2 min. The transcript
holds no keys: environment secrets are checked against the text before it is written.
Usage: uv run python scripts/run_agent_demo.py
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import rasterio

from quaketriage.agent.graph import Agent
from quaketriage.agent.llm import make_llm
from quaketriage.agent.tools import validate_citations
from quaketriage.config import get

ROOT = Path(__file__).parents[1]
OUT = ROOT / "reports" / "agent_demo_transcripts.md"


def degraded_post(src: Path, dst: Path) -> Path:
    """Synthetic 'bad image': 70 % of the Kilis post image covered by saturated cloud."""
    with rasterio.open(src) as s:
        arr, profile = s.read(), s.profile
    arr[:, : int(arr.shape[1] * 0.7), :] = 255
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **profile) as d:
        d.write(arr.astype(np.uint8))
    return dst


def fmt_trace(entries: list[dict]) -> str:
    rows = ["| tool | status | seconds | summary |", "|---|---|---|---|"]
    for e in entries:
        summary = (e.get("summary") or e.get("error", "")).replace("|", "/").replace("\n", " ")
        rows.append(f"| {e['tool']} | {e.get('status', 'error')} | {e.get('seconds', '')} | "
                    f"{summary[:140]} |")
    return "\n".join(rows)


def main() -> None:
    llm = make_llm()
    kilis = get("regions.kilis")
    bad_post = degraded_post(ROOT / kilis["post"],
                             ROOT / "outputs" / "demo" / "kilis_post_cloudy.tif")
    main_agent = Agent(llm)
    bad_agent = Agent(llm, regions={"antakya": {"detector": "simulated"},
                                    "kilis": {**kilis, "post": str(bad_post)}})
    script = [
        ("1. Normal plan question (TR)", main_agent,
         "Antakya için hasar değerlendirmesi yap, öncelikli binaları belirle ve rota planı "
         "hazırla"),
        ("2. What-if: team count (TR)", main_agent,
         "Ekip sayısını 6'ya çıkarsak plan nasıl değişir?"),
        ("3. Bad imagery / Kilis quality gate (TR)", bad_agent,
         "Kilis kesitindeki hasarı değerlendir"),
        ("4. Protocol question, cited (EN)", main_agent,
         "How do international USAR teams register when they arrive in the affected country?"),
        ("5. Off-topic (TR)", main_agent, "Bana akşam yemeği için bir kek tarifi verir misin?"),
    ]
    lines = ["# Agent demo transcripts", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} with the real LLM "
             f"(primary `{llm.backends[0].provider}`, "
             f"fallback `{llm.backends[1].provider if len(llm.backends) > 1 else 'none'}`). "
             "**Antakya results use the SIMULATED detector** (HOTOSM labels + noise); the depot, "
             "debris, walking and occupancy rules are scenario assumptions. In conversation 3 the "
             "'bad image' is a synthetic copy of the Kilis post image with 70 % saturated cloud "
             "(the real Kilis pair passes the guard).", ""]
    summary = ["| # | intent | language | tools | LLM calls | cache hits | provider | latency s | "
               "stop |", "|---|---|---|---|---|---|---|---|---|"]
    body, citation_report = [], None
    for title, agent, question in script:
        r = agent.ask(question)
        m = r.metrics
        trace = agent.ctx.trace[-len(r.tools_called):] if r.tools_called else []
        prov = ",".join(m["providers"]) or ("cache" if m["cache_hits"] else "none")
        summary.append(f"| {title[0]} | {r.intent} | {r.language} | {len(r.tools_called)} | "
                       f"{m['llm_calls']} | {m['cache_hits']} | {prov} | {m['latency_s']:.1f} | "
                       f"{r.stop_reason or '-'} |")
        body += [f"## {title}", "", f"**User:** {question}", "",
                 f"**Router:** intent=`{r.intent}`, language=`{r.language}`, region=`{r.region}`, "
                 f"keyword-fallback={m['router_fallback']}", "", "**Tool trace**", "",
                 fmt_trace(trace) if trace else "_no tools called_", "",
                 f"**LLM:** {m['llm_calls']} calls, {m['cache_hits']} cache hits, provider(s): "
                 f"{prov}, latency {m['latency_s']:.1f} s", "", "**Answer**", "",
                 "\n".join("> " + ln for ln in r.answer.splitlines()), ""]
        if title.startswith("1."):
            chunks = [json.loads(x) for x in (ROOT / get("paths.data_processed") / "rag" /
                                              "chunks.jsonl").read_text().splitlines()]
            citation_report = validate_citations(r.results["write_sitrep"]["markdown"],
                                                 agent.ctx.protocol_hits, chunks)
            body += ["**Citation check (sitrep)**", "", "```json",
                     json.dumps(citation_report, indent=1), "```", ""]
    text = "\n".join(lines + summary + [""] + body)
    secrets = [v for k, v in os.environ.items()
               if any(t in k.upper() for t in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
               and len(v) >= 8]
    assert not any(sec in text for sec in secrets), "secret found in transcript; not written"
    OUT.write_text(text)
    print(f"written {OUT.relative_to(ROOT)} ({len(text)} chars); calls this run: {llm.calls}, "
          f"cache hits {llm.cache_hits}; citations valid: {citation_report['valid']}")


if __name__ == "__main__":
    main()
