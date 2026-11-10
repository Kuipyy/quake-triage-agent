"""Plan rescue routes for the Antakya demo set and test sensitivity to the candidate count N.

Writes reports/plan_demo.json (plan for routing.candidate_n) and reports/metrics_planning.json.
Detector output is SIMULATED (see metrics_building.json).
Usage: uv run python scripts/plan_routes.py
Estimated runtime ~4 min (graph + debris ~20 s, distance matrix ~30 s, 4 solves x time limit).
"""
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

from quaketriage.config import get
from quaketriage.planning.debris import DebrisInput, apply_debris_risk
from quaketriage.planning.distances import travel_time_matrix
from quaketriage.planning.roads import load_road_graph, snap_to_nodes
from quaketriage.planning.topsis import TopsisInput, prioritise
from quaketriage.planning.vrp import VrpInput, building_nodes, plan_routes, select_candidates

ROOT = Path(__file__).parents[1]
SWEEP = (200, 500, 1000)


def main() -> None:
    t0 = time.time()
    b = gpd.read_file(ROOT / get("paths.data_processed") / "buildings.gpkg")
    depot = tuple(get("routing.depots")[0])
    b = prioritise(b, TopsisInput(depots=[depot]))
    g = apply_debris_risk(load_road_graph(), b, DebrisInput())
    print(f"graph + debris ready ({time.time() - t0:.0f}s)")

    # reachability of every building (one matrix row/column from the depot)
    depot_node = int(snap_to_nodes(g, [depot])[0][0])
    all_nodes = building_nodes(g, b)
    row = travel_time_matrix(g, [depot_node, *all_nodes.tolist()], use_cache=False)
    reach = np.isfinite(row[0, 1:]) & np.isfinite(row[1:, 0])
    n_unreach_all = int((~reach).sum())
    print(f"unreachable from/to depot: {n_unreach_all}/{len(b)} buildings")

    # one cached matrix for the largest candidate set; smaller N reuse a sub-matrix
    n_max = max(SWEEP + (get("routing.candidate_n"),))
    cand = select_candidates(b, n_max)
    nodes = [depot_node, *building_nodes(g, cand).tolist()]
    t1 = time.time()
    matrix = travel_time_matrix(g, nodes)
    print(f"matrix {matrix.shape} ready ({time.time() - t1:.0f}s)")

    n_cfg = get("routing.candidate_n")
    sweep = {}
    for n in sorted(set(SWEEP) | {get("routing.candidate_n")}):
        t2 = time.time()
        plan = plan_routes(g, b, depot, VrpInput(candidate_n=n), matrix, nodes)
        sweep[str(n)] = {k: v for k, v in plan.items() if k != "routes"} | {
            "solve_s": round(time.time() - t2, 1)}
        print(f"N={n}: visited {plan['n_visited']}, priority {plan['priority_collected']}, "
              f"persons {plan['persons_reached']}, unreachable {plan['n_unreachable']}")
        if n == get("routing.candidate_n"):
            demo = plan
    # same N, but debris only slows roads (no closures): isolates the effect of the closure rule
    g_soft = apply_debris_risk(load_road_graph(), b, DebrisInput(block_count=10**9))
    soft_nodes = [depot_node, *building_nodes(g_soft, select_candidates(b, n_cfg)).tolist()]
    soft = plan_routes(g_soft, b, depot, VrpInput(candidate_n=n_cfg),
                       travel_time_matrix(g_soft, soft_nodes), soft_nodes)
    penalty_only = {k: v for k, v in soft.items() if k != "routes"}
    print(f"penalty-only N={n_cfg}: visited {soft['n_visited']}, "
          f"priority {soft['priority_collected']}, unreachable {soft['n_unreachable']}")
    demo.update({"depot_lonlat": depot, "disclaimer": "SIMULATED detector; scenario depot"})
    (ROOT / "reports" / "plan_demo.json").write_text(json.dumps(demo, indent=1))

    out = {"SIMULATED_DETECTOR": True,
           "disclaimer": "Damage probabilities come from a simulated detector (HOTOSM labels + "
                         "noise); depot is a scenario assumption; numbers show planning "
                         "behaviour, not real-world rescue outcomes.",
           "n_buildings": len(b), "n_unreachable_all": n_unreach_all,
           "params": {k: get(f"routing.{k}") for k in (
               "num_teams", "mission_duration_hours", "intervention_time_minutes",
               "solver_time_limit_s", "candidate_n", "debris_risk_buffer_meters",
               "debris_damage_prob_min", "debris_min_floors", "debris_penalty_factor",
               "debris_block_count")},
           "depot_lonlat": depot, "candidate_n_sensitivity": sweep,
           "closure_rule_effect": {"n": n_cfg, "with_closures": sweep[str(n_cfg)],
                                   "penalty_only": penalty_only},
           "note": "OR-Tools guided local search is time-limited, hence not exactly "
                   "reproducible; differences of a few percent between N values are noise."}
    (ROOT / "reports" / "metrics_planning.json").write_text(json.dumps(out, indent=1))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
