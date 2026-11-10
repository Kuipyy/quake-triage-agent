"""Side-by-side v1 (router-rewritten query) vs v2 (original user message) protocol results.

Reads reports/archive/agent_eval_v1.json, reports/archive/agent_eval_protocol_v2.json and
eval/protocol_hand_review.json; writes reports/protocol_v1_vs_v2.json. v1 is kept as it was.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
v1 = json.loads((ROOT / "reports/archive/agent_eval_v1.json").read_text())
v2 = json.loads((ROOT / "reports/archive/agent_eval_protocol_v2.json").read_text())
rev = json.loads((ROOT / "eval/protocol_hand_review.json").read_text())


def strict(rep):
    return {s["id"]: bool(s["turns"][0]["checks"].get("cited_gold_page"))
            for s in rep["scenarios"] if s["type"] == "protocol"}


s1, s2 = strict(v1), strict(v2)
h1 = {i: s1[i] or rev["reviewed"].get(i, {}).get("acceptable", False) for i in s1}
h2 = {i: rev["v2"]["reviewed"][i]["acceptable"] for i in s2}


def block(strict_d, hand_d, rep):
    c = rep["metrics"]["citation"]
    return {"strict_gold_page": f"{sum(strict_d.values())}/{len(strict_d)}",
            "strict_passed": sum(strict_d.values()), "hand_reviewed_passed": sum(hand_d.values()),
            "total": len(strict_d), "dangling_citations": c["dangling_citations"],
            "rough_support": f"{c['rough_support_sentences']['passed']}/"
                             f"{c['rough_support_sentences']['total']}",
            "successful_llm_answers_by_provider": dict(Counter(
                p for s in rep["scenarios"] if s["type"] == "protocol" for p in s["providers"])),
            "llm_requests_incl_retries": sum(
                t["llm_calls"] for s in rep["scenarios"] if s["type"] == "protocol"
                for t in s["turns"])}


out = {"retrieval_query": {"v1": "router-rewritten keyword query", "v2": "original user message"},
       "v1": block(s1, h1, v1), "v2": block(s2, h2, v2),
       "per_scenario": {i: {"v1_strict": s1[i], "v2_strict": s2[i], "v1_hand": h1[i],
                            "v2_hand": h2[i]} for i in s1},
       "caveat": "Same 8 scenarios and gold pages (not changed); single run each; hand review "
                 "by the developer; v1 = part of the 38-scenario run, v2 = 8-scenario re-run."}
(ROOT / "reports/protocol_v1_vs_v2.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("v1", "v2")}, indent=0)[:900])
