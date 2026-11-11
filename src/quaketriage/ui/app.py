"""Gradio app: ``uv run python -m quaketriage.ui.app``.

Left: region choice and chat with the agent. Right: tabs (map, before/after, priority table, sitrep,
tool trace, model card). Route plans are cached on disk and the UI uses a shorter solver limit, so
a repeated question answers in seconds. The app serves over http, which OSM map tiles need
(they do not load from file://). Every tab shows the SIMULATED-detector notice.
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

from quaketriage.agent.graph import Agent
from quaketriage.agent.llm import LLMError, make_llm
from quaketriage.agent.tools import ToolContext
from quaketriage.config import get
from quaketriage.ui import views

ROOT = Path(__file__).parents[3]
EXAMPLES = [
    "Antakya için hasar değerlendirmesi yap, öncelikli binaları belirle ve rota planı hazırla",
    "Ekip sayısını 6'ya çıkarsak plan nasıl değişir?",
    "Assess Antakya and write a situation report in English",
    "What do the INSARAG guidelines say about marking a worksite?",
]
_SHARED: dict = {}
_LOCK = threading.Lock()


@dataclass
class Session:
    agent: Agent | None = None
    log: list[str] = field(default_factory=list)


def shared_resources() -> ToolContext:
    """Road graph and RAG index are loaded once per process and shared between sessions."""
    with _LOCK:
        if "ctx" not in _SHARED:
            ctx = ToolContext()
            ctx.graph()
            try:
                ctx.rag()
            except Exception:  # RAG missing: protocol questions will say so; the rest still works
                pass
            _SHARED["ctx"] = ctx
        return _SHARED["ctx"]


def make_agent(region: str, llm=None) -> Agent:
    base = shared_resources()
    ctx = ToolContext(base_graph=base.base_graph, rag_index=base.rag_index,
                      embed_query=base.embed_query)
    ctx.constraints.solver_time_limit_s = int(get("ui.solver_time_limit_s"))
    ctx.plan_cache_dir = ROOT / get("ui.plan_cache_dir")
    return Agent(llm or make_llm(), ctx=ctx, region=region)


def _views(agent: Agent):
    """All tab contents from the agent's current state."""
    ctx = agent.ctx
    sit = views.sitrep_markdown(ctx)
    path = views.sitrep_file(ctx)
    return (views.map_html(ctx), views.priority_table(ctx), views.queue_summary(ctx),
            sit or "_No situation report yet. Ask the agent to write one._",
            str(path) if path else None, views.trace_table(ctx))


def _status(agent: Agent, t0: float, before: int) -> str:
    done = [e["tool"] for e in agent.ctx.trace[before:]]
    limit = get("ui.solver_time_limit_s")
    step = f"tools so far: {', '.join(done)}" if done else "routing the question"
    return (f"⏳ working… {time.time() - t0:.0f} s — {step} "
            f"(route solver limit {limit} s; repeated plans come from the cache)")


def build_app(llm_factory=None) -> gr.Blocks:
    """``llm_factory`` is injectable for tests; default builds the client from .env."""
    regions = list(get("regions"))
    default_kate = 0

    def respond(message, history, region, session: Session):
        history = list(history or [])
        if not message.strip():
            yield history, "", session, *([gr.update()] * 6), ""
            return
        history += [{"role": "user", "content": message}]
        try:
            if session.agent is None:
                session.agent = make_agent(region, llm=llm_factory() if llm_factory else None)
            elif session.agent.region != region:
                session.agent._reset_region(region)
        except (LLMError, OSError) as exc:
            history += [{"role": "assistant", "content": f"LLM not configured: {exc}"}]
            yield history, "", session, *([gr.update()] * 6), "error"
            return
        agent = session.agent
        box: dict = {}
        before = len(agent.ctx.trace)

        def work():
            try:
                box["r"] = agent.ask(message)
            except Exception as exc:  # shown in the chat, never crashes the app
                box["err"] = f"{type(exc).__name__}: {exc}"

        t0 = time.time()
        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():
            yield (history + [{"role": "assistant", "content": _status(agent, t0, before)}],
                   "", session, *([gr.update()] * 6), "working")
            th.join(timeout=1.0)
        if "err" in box:
            history += [{"role": "assistant", "content": f"Error: {box['err']}"}]
            yield history, "", session, *([gr.update()] * 6), "error"
            return
        r = box["r"]
        m = r.metrics
        prov = ",".join(m.get("providers", [])) or ("cache" if m.get("cache_hits") else "none")
        foot = (f"\n\n<sub>{r.intent} · tools: {', '.join(r.tools_called) or '-'} · "
                f"LLM calls {m['llm_calls']} ({prov}) · {m['latency_s']:.1f} s</sub>")
        history += [{"role": "assistant", "content": r.answer + foot}]
        yield history, "", session, *_views(agent), f"done in {m['latency_s']:.1f} s"

    def show_pair(source, idx):
        try:
            if source == "Kilis crop (real Siamese model)":
                pre, post, cap = views.kilis_pair()
            else:
                pre, post, cap = views.kate_cd_pair(idx)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            return None, (f"**No local imagery.** {views.NO_IMAGERY} ({type(exc).__name__})")
        return (pre, post), cap

    with gr.Blocks(title="QuakeTriage") as demo:
        gr.Markdown("# QuakeTriage\nAgentic post-earthquake damage assessment and rescue "
                    "prioritisation. " + views.notice_md())
        session = gr.State(Session())
        with gr.Row():
            with gr.Column(scale=2, min_width=380):
                region = gr.Dropdown(regions, value=regions[0], label="Demo region")
                chat = gr.Chatbot(height=520, label="Agent")
                box = gr.Textbox(placeholder="Ask in Turkish or English…", show_label=False)
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    status = gr.Textbox(show_label=False, interactive=False, max_lines=1)
                gr.Examples(EXAMPLES, inputs=box, label="Examples")
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Map"):
                        gr.Markdown(views.notice_md("Damage layer, top-priority markers, team "
                                                    "routes (toggle in the layer control)."))
                        map_out = gr.HTML()
                    with gr.Tab("Before / after"):
                        gr.Markdown(views.notice_md("Local-only tab: the imagery is not "
                                                    "in the repository (licences). "
                                                    + views.NO_IMAGERY))
                        with gr.Row():
                            src = gr.Radio(["KATE-CD sample", "Kilis crop (real Siamese model)"],
                                           value="KATE-CD sample", label="Image source")
                            idx = gr.Number(value=default_kate, precision=0, label="KATE-CD index")
                        slider = gr.ImageSlider(label="pre (left) / post (right)")
                        cap = gr.Markdown()
                    with gr.Tab("Priority table"):
                        gr.Markdown(views.notice_md())
                        qline = gr.Markdown()
                        table = gr.Dataframe(value=views.priority_table(ToolContext()),
                                             interactive=False)
                    with gr.Tab("Situation report"):
                        gr.Markdown(views.notice_md())
                        sit = gr.Markdown()
                        dl = gr.DownloadButton("Download sitrep (.md)")
                    with gr.Tab("Tool trace"):
                        gr.Markdown(views.notice_md("Every agent tool call of the session."))
                        trace = gr.Dataframe(value=views.trace_table(ToolContext()),
                                             interactive=False)
                    with gr.Tab("Model card"):
                        gr.Markdown(views.notice_md())
                        gr.Markdown(views.model_card_md())

        outs = [chat, box, session, map_out, table, qline, sit, dl, trace, status]
        for trigger in (box.submit, send.click):
            trigger(respond, [box, chat, region, session], outs)
        for trigger in (src.change, idx.change, demo.load):
            trigger(show_pair, [src, idx], [slider, cap])
    return demo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=get("ui.host"))
    ap.add_argument("--port", type=int, default=get("ui.port"))
    args = ap.parse_args(argv)
    build_app().queue().launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
