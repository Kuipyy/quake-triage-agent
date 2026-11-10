"""Merge eval/protocol_hand_review.json into reports/agent_eval.json.

Adds metrics.citation.hand_reviewed; the strict gold-page numbers are left untouched. Idempotent.
Usage: uv run python scripts/apply_hand_review.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
rep = ROOT / "reports" / "agent_eval.json"
review = json.loads((ROOT / "eval" / "protocol_hand_review.json").read_text())["reviewed"]
data = json.loads(rep.read_text())
by_id = {s["id"]: s for s in data["scenarios"]}
cit = data["metrics"]["citation"]
strict_ok = {i for i, s in by_id.items() if s["type"] == "protocol"
             and s["turns"][0]["checks"].get("cited_gold_page")}
acceptable = strict_ok | {i for i, v in review.items() if v["acceptable"]}
total = cit["protocol_turns"]
cit["hand_reviewed"] = {
    "passed": len(acceptable), "total": total, "rate": round(len(acceptable) / total, 3),
    "wrong_answers": sum(v["verdict"] == "wrong_answer" for v in review.values()),
    "honest_no_answer_but_info_exists": sum(v["verdict"] == "miss_honest" for v in review.values()),
    "scope": "5 strict failures read by hand; the 3 strict passes were not re-read",
    "details": review}
rep.write_text(json.dumps(data, indent=1, ensure_ascii=False))
print(f"hand-reviewed acceptable: {len(acceptable)}/{total} (strict {len(strict_ok)}/{total})")
