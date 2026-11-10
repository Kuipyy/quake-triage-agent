"""Ablation: uncertainty (verification-queue) filter ON vs OFF. SIMULATED detector.

The simulated uncertainty only depends on how close a probability is to the decision threshold,
NOT on whether the detector is right (see buildings/simulate.py). Therefore the base result
measures the noise model, not real model behaviour. Sensitivity: the ASSUMED error-aware variants
multiply the uncertainty of wrong detections by k (on KATE-CD real uncertainty is 14-21x higher on
wrong pixels, reports/uncertainty_*.json); k is an assumption, not a measurement.

Counts (mean +- std over 5 noise seeds): false/true positives among the reported damaged
buildings, among the top-N candidates by priority, and (seed 0, k=1, real OR-Tools solve) among
dispatched stops.
Usage: uv run python scripts/ablation_uncertainty_filter.py   (~1.5 min)
Writes reports/ablation_uncertainty_filter.json
"""
import time
from pathlib import Path

import numpy as np

from quaketriage.agent.tools import ToolContext
from quaketriage.buildings.queue import flag_verification
from quaketriage.buildings.simulate import SimulatorConfig, simulate_detector
from quaketriage.config import get
from quaketriage.planning.topsis import TopsisInput, prioritise
from quaketriage.provenance import write_json

ROOT = Path(__file__).parents[1]
SEEDS = range(5)
K_GRID = (1.0, 2.0, 3.0, 5.0)
SIM_COLS = ["damage_prob", "uncertainty", "damaged", "needs_verification", "verification_reason",
            "destroyed_truth"]


def counts(b, held, mask_reported, topn):
    truth = b["destroyed_truth"].to_numpy(bool)
    rep = mask_reported & ~held
    cand = b[~held].nlargest(topn, "priority")
    ct = cand["destroyed_truth"].to_numpy(bool)
    return {"reported_tp": int((rep & truth).sum()), "reported_fp": int((rep & ~truth).sum()),
            "held_back_tp": int((mask_reported & held & truth).sum()),
            "held_back_fp": int((mask_reported & held & ~truth).sum()),
            "queue_size": int(held.sum()),
            "topn_tp": int(ct.sum()), "topn_fp": int((~ct).sum())}


def agg(rows):
    return {k: {"mean": round(float(np.mean([r[k] for r in rows])), 1),
                "std": round(float(np.std([r[k] for r in rows])), 1)} for k in rows[0]}


def main() -> None:
    t0 = time.time()
    ctx = ToolContext()
    ctx.call("detect_damage", {"source": "simulated"})
    base = ctx.buildings.drop(columns=[c for c in SIM_COLS if c in ctx.buildings]).copy()
    destroyed = ctx.buildings["destroyed_truth"].to_numpy(bool)
    depot = [tuple(get("routing.depots")[0])]
    topn = get("routing.candidate_n")
    sims = {}
    for k in K_GRID:
        on_rows, off_rows = [], []
        for seed in SEEDS:
            s = simulate_detector(base, destroyed, SimulatorConfig(
                miss_rate=0.2, false_alarm_rate=0.03, size_exponent=1.0, seed=seed))
            wrong = s["damaged"].to_numpy(bool) != destroyed
            s["uncertainty"] = np.clip(s["uncertainty"].to_numpy() * np.where(wrong, k, 1.0), 0, 1)
            s = flag_verification(s)
            s = prioritise(s, TopsisInput(depots=depot))
            dmg = s["damaged"].to_numpy(bool)
            on_rows.append(counts(s, s["needs_verification"].to_numpy(bool), dmg, topn))
            off_rows.append(counts(s, np.zeros(len(s), bool), dmg, topn))
        sims[f"k={k:g}"] = {"filter_on": agg(on_rows), "filter_off": agg(off_rows),
                            "ASSUMED": k != 1.0}
        print(f"k={k:g}: FP reported off {sims[f'k={k:g}']['filter_off']['reported_fp']['mean']} "
              f"-> on {sims[f'k={k:g}']['filter_on']['reported_fp']['mean']}", flush=True)

    plans = {}
    for name, on in (("filter_on", True), ("filter_off", False)):
        c = ToolContext(base_graph=ctx.base_graph)
        c.call("detect_damage", {"source": "simulated"})
        c.call("score_buildings", {})
        if not on:
            c.buildings["needs_verification"] = False
            c.buildings["verification_reason"] = ""
        c.call("estimate_occupancy", {})
        c.call("prioritize_sites", {})
        out = c.call("plan_routes", {})
        truth = c.buildings["destroyed_truth"].to_numpy(bool)
        rows = [s["row"] for t in out["teams"] for s in t["stops"]]
        tp = int(truth[[c.buildings.index.get_loc(r) for r in rows]].sum()) if rows else 0
        plans[name] = {"n_visited": out["n_visited"], "dispatched_true_destroyed": tp,
                       "dispatched_false_positive": out["n_visited"] - tp,
                       "persons_reached": out["persons_reached"],
                       "priority_collected": out["priority_collected"],
                       "n_excluded_verification": out["n_excluded_verification"]}
        print(name, plans[name], flush=True)
        ctx.base_graph = c.base_graph

    write_json(ROOT / get("paths.reports") / "ablation_uncertainty_filter.json", {
        "SIMULATED_DETECTOR": True,
        "disclaimer": "Measures the simulated noise model, not real model behaviour: simulated "
                      "uncertainty peaks near the decision threshold and does not depend on "
                      "whether the detector is right. k>1 rows are ASSUMED error-aware variants.",
        "setup": {"miss_rate": 0.2, "false_alarm_rate": 0.03, "size_exponent": 1.0,
                  "seeds": list(SEEDS), "tau": get("thresholds.uncertainty_high"), "top_n": topn},
        "counts_by_k": sims, "vrp_seed0_k1": plans, "seconds": round(time.time() - t0)})
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
