"""Merge a hand-review file into reports/agent_eval.json.

Adds metrics.citation.hand_reviewed; the strict gold-page numbers are left untouched. Idempotent.
A review file with ``covers_all_protocol_scenarios: true`` judges every protocol answer; an older
file (default of ``--review`` before 2026-11-11) judges only the strict failures.
Usage: uv run python scripts/apply_hand_review.py [--review eval/<review file>.json]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
rep = ROOT / "reports" / "agent_eval.json"
ap = argparse.ArgumentParser()
ap.add_argument("--review", default="eval/protocol_hand_review_2026-11-11.json")
doc = json.loads((ROOT / ap.parse_args().review).read_text())
review = doc["reviewed"]
covers_all = bool(doc.get("covers_all_protocol_scenarios"))
data = json.loads(rep.read_text())
by_id = {s["id"]: s for s in data["scenarios"]}
cit = data["metrics"]["citation"]
strict_ok = {i for i, s in by_id.items() if s["type"] == "protocol"
             and s["turns"][0]["checks"].get("cited_gold_page")}
reviewed_ok = {i for i, v in review.items() if v["acceptable"]}
acceptable = reviewed_ok if covers_all else strict_ok | reviewed_ok
total = cit["protocol_turns"]
cit["hand_reviewed"] = {
    "passed": len(acceptable), "total": total, "rate": round(len(acceptable) / total, 3),
    "wrong_answers": sum(v["verdict"] == "wrong_answer" for v in review.values()),
    "honest_no_answer_but_info_exists": sum(v["verdict"] == "miss_honest" for v in review.values()),
    "scope": ("all 8 protocol answers read against the source text" if covers_all
              else "5 strict failures read by hand; the 3 strict passes were not re-read"),
    "details": review}
rep.write_text(json.dumps(data, indent=1, ensure_ascii=False))
print(f"hand-reviewed acceptable: {len(acceptable)}/{total} (strict {len(strict_ok)}/{total})")
