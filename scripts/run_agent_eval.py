"""Run eval/agent_scenarios.yaml against the real agent + LLM and write reports/agent_eval.json.

COLD run: the LLM disk cache is emptied first (it only holds regenerable replies), so latency and
LLM call counts are real. Needs .env (Gemini key, Groq fallback). ~55 LLM calls, 10-15 min.
Full answers go to outputs/logs/agent_eval_answers.jsonl (gitignored); the report keeps excerpts.
No keys are printed: every written text is checked against environment secrets.
Usage: uv run python scripts/run_agent_eval.py [--keep-cache] [--only ID,ID] [--out PATH]
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from quaketriage.agent.graph import Agent
from quaketriage.agent.llm import make_llm
from quaketriage.agent.tools import ToolContext
from quaketriage.config import get
from quaketriage.eval.runner import check_turn, detect_violations, env_secret_values, summarize
from quaketriage.eval.scenarios import load_scenarios

ROOT = Path(__file__).parents[1]
OUT = ROOT / "reports" / "agent_eval.json"
ANSWERS = ROOT / "outputs" / "logs" / "agent_eval_answers.jsonl"
sys.path.insert(0, str(ROOT / "scripts"))
from run_agent_demo import degraded_post  # noqa: E402


def run_scenario(sc, llm, shared: ToolContext, regions: dict) -> dict:
    ctx = ToolContext(base_graph=shared.base_graph, rag_index=shared.rag_index,
                      embed_query=shared.embed_query)
    reg = regions
    if "kilis_degraded" in sc.setup:
        bad = degraded_post(ROOT / regions["kilis"]["post"],
                            ROOT / "outputs" / "demo" / "kilis_post_cloudy.tif")
        reg = {**regions, "kilis": {**regions["kilis"], "post": str(bad)}}
    agent = Agent(llm, ctx=ctx, region=sc.region, regions=reg,
                  run_log=ROOT / "outputs" / "logs" / "agent_eval_runs.jsonl")
    if "quality_failed" in sc.setup:  # injected fault: the check "failed" before the first turn
        ctx.quality_ok = False
        ctx.quality_reasons = ["injected test fault: image quality check failed"]
    turns, answers, trace_all, refs, crash = [], [], [], [], None
    for t in sc.turns:
        try:
            r = agent.ask(t.say)
        except Exception as exc:  # a crash is a failure, never hidden
            crash = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(limit=3)
            turns.append({"say": t.say, "answer": "", "checks": {"no_crash": False},
                          "passed": False, "latency_s": 0.0, "llm_calls": 0, "cache_hits": 0,
                          "providers": [], "router_fallback": False, "expect_cited": False,
                          "citation_support": {"checked": 0, "supported": 0},
                          "tool_set_ok": None, "expected_language": t.expect.language,
                          "tools": [], "error": crash})
            answers.append("")
            refs.append(None)
            break
        m = r.metrics
        res = {"intent": r.intent, "language": r.language, "tools": r.tools_called,
               "stop_reason": r.stop_reason, "answer": r.answer,
               "constraints": ctx.constraints.model_dump(mode="json")}
        checks = check_turn(t.expect, res)
        trace_all += ctx.trace[len(trace_all):]
        answers.append(r.answer)
        refs.append({h["ref"] for h in ctx.protocol_hits})
        turns.append({
            "say": t.say, "intent": r.intent, "language": r.language, "tools": r.tools_called,
            "stop_reason": r.stop_reason, "answer": r.answer, "checks": checks,
            "passed": all(checks.values()), "latency_s": m["latency_s"],
            "llm_calls": m["llm_calls"], "cache_hits": m["cache_hits"],
            "providers": m["providers"], "router_fallback": m["router_fallback"],
            "expect_cited": bool(t.expect.cited_docs), "citation_support": m["citation_support"],
            "tool_set_ok": (None if t.expect.tools is None
                            else set(r.tools_called) == set(t.expect.tools)),
            "expected_language": t.expect.language})
    plan = ctx.plan or {}
    stops = {s["row"] for rt in plan.get("routes", []) for s in rt["stops"]}
    b = ctx.buildings
    queue = (set(b.index[b["needs_verification"]]) if b is not None and "needs_verification" in b
             else set())
    final = {"quality_ok": ctx.quality_ok, "uncertainty_tau": ctx.constraints.uncertainty_tau,
             "plan_stops": stops, "queue_rows": queue, "retrieved_refs": refs,
             "env_secrets": env_secret_values()}
    viol = detect_violations(sc, trace_all, answers, final)
    providers = [p for t in turns for p in t["providers"]]
    return {"id": sc.id, "type": sc.type, "region": sc.region, "setup": sc.setup,
            "turns": turns, "violations": viol, "providers": providers,
            "passed": crash is None and all(t["passed"] for t in turns) and not viol,
            "crash": crash}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(OUT), help="report path (default reports/agent_eval.json)")
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    answers_path = ANSWERS if out_path == OUT else (
        ROOT / "outputs" / "logs" / f"{out_path.stem}_answers.jsonl")
    scenarios = load_scenarios()
    if args.only:
        scenarios = [s for s in scenarios if s.id in args.only.split(",")]
    llm = make_llm()
    cache = llm.cache_dir
    if not args.keep_cache:
        files = list(cache.glob("*.json"))
        for f in files:
            f.unlink()
        print(f"cold run: cleared {len(files)} cached LLM replies")
    shared = ToolContext()
    shared.graph()
    shared.rag()
    regions = get("regions")
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    records, t0 = [], time.time()
    for i, sc in enumerate(scenarios, 1):
        rec = run_scenario(sc, llm, shared, regions)
        records.append(rec)
        failed = [k for t in rec["turns"] for k, ok in t["checks"].items() if not ok]
        print(f"[{i}/{len(scenarios)}] {sc.id} {'PASS' if rec['passed'] else 'FAIL'} "
              f"calls={sum(t['llm_calls'] for t in rec['turns'])} "
              f"lat={sum(t['latency_s'] for t in rec['turns']):.0f}s "
              f"prov={sorted(set(rec['providers'])) or '-'} "
              f"viol={len(rec['violations'])} failed={failed[:3]}", flush=True)
    with answers_path.open("w") as fh:
        for r in records:
            fh.write(json.dumps({"id": r["id"], "answers": [t["answer"] for t in r["turns"]]},
                                ensure_ascii=False) + "\n")
    metrics = summarize(records)  # needs the full answers (language heuristic)
    for r in records:  # the report keeps excerpts, not full answers
        for t in r["turns"]:
            t["answer_excerpt"] = " ".join(t.pop("answer").split())[:400]
    failed = [{"id": r["id"], "type": r["type"],
               "failed_checks": [{"turn": n + 1, "check": k} for n, t in enumerate(r["turns"])
                                 for k, ok in t["checks"].items() if not ok],
               "violations": r["violations"], "crash": r["crash"],
               "answer_excerpts": [t["answer_excerpt"] for t in r["turns"]]}
              for r in records if not r["passed"]]
    report = {
        "SIMULATED_DETECTOR": True,
        "caveats": [
            "Scenarios were written by the party that wrote the agent: not an independent "
            "evaluation.",
            "Single run; LLM output is not deterministic, so numbers can differ on a re-run.",
            "Antakya scenarios use the simulated detector; guard cases use injected faults.",
            "Citation support is a crude keyword check, not a judgement of correctness."],
        "run": {"date": time.strftime("%Y-%m-%d %H:%M"), "cold_cache": not args.keep_cache,
                "scenarios": len(records), "wall_clock_s": round(time.time() - t0),
                "llm_primary": f"{llm.backends[0].provider}:{llm.backends[0].model}",
                "llm_fallback": (f"{llm.backends[1].provider}:{llm.backends[1].model}"
                                 if len(llm.backends) > 1 else None),
                "llm_real_calls_total": llm.calls, "llm_cache_hits_total": llm.cache_hits},
        "metrics": metrics,
        "failed_scenarios": failed,
        "scenarios": records}
    text = json.dumps(report, indent=1, ensure_ascii=False, default=sorted)
    secrets = env_secret_values()
    assert not any(s in text for s in secrets), "secret found in report; not written"
    out_path.write_text(text)
    m = report["metrics"]
    viol = m["forbidden_behaviour"]["total_violations"]
    print(f"written {out_path.relative_to(ROOT)}: success {m['task_success']['passed']}/"
          f"{m['task_success']['total']}, violations {viol}, "
          f"LLM calls {m['llm']['calls_total']}, mean latency "
          f"{m['latency_s_per_turn']['mean']} s, failed: {[f['id'] for f in failed]}")


if __name__ == "__main__":
    main()
