"""Plan rescue routes for the Antakya demo set; compare debris/walking variants and candidate N.

Writes reports/plan_demo.json (plan for routing.candidate_n, closures + walking) and
reports/metrics_planning.json. Detector output is SIMULATED (see metrics_building.json).
Usage: uv run python scripts/plan_routes.py
Estimated runtime ~4 min (graph + debris ~10 s, matrices ~5 s, 6 solves x limit, baselines <10 s).
"""
import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

from quaketriage.config import get
from quaketriage.planning.baselines import greedy_nearest, priority_first, random_dispatch
from quaketriage.planning.debris import DebrisInput, apply_debris_risk
from quaketriage.planning.distances import travel_time_matrix, walking_access
from quaketriage.planning.roads import load_road_graph, snap_to_nodes
from quaketriage.planning.topsis import TopsisInput, prioritise
from quaketriage.planning.vrp import (
    VrpInput,
    building_nodes,
    plan_routes,
    prepare_problem,
    select_candidates,
)

ROOT = Path(__file__).parents[1]
SWEEP = (200, 500, 1000)
RANDOM_SEEDS = 30
KEYS = ("n_candidates", "n_unreachable", "n_need_walking", "n_visited", "priority_collected",
        "persons_reached", "persons_in_candidates")


def summary(plan: dict, seconds: float) -> dict:
    return {k: plan[k] for k in KEYS} | {"solve_s": round(seconds, 1)}


def stop_nodes(g, depot_node, cand, max_walk: float) -> list:
    """Graph nodes the teams stop at for these candidates (access nodes within the walk limit)."""
    access, walk_m = walking_access(g, depot_node, building_nodes(g, cand).tolist())
    return [a for a, w in zip(access, walk_m, strict=True) if a is not None and w <= max_walk]


def compute_baselines(prob, vrp_summary: dict) -> dict:
    """Dispatch-rule baselines on the same problem as the VRP, with priority ratios vs random."""
    rand = [random_dispatch(prob, seed) for seed in range(RANDOM_SEEDS)]

    def mean_std(key: str) -> dict:
        v = np.array([r[key] for r in rand])
        return {"mean": round(float(v.mean()), 3), "std": round(float(v.std()), 3)}

    rkey = f"random_{RANDOM_SEEDS}_seeds"
    out = {"vrp": dict(vrp_summary),
           "greedy_nearest": summary(greedy_nearest(prob), 0.0),
           "priority_first": summary(priority_first(prob), 0.0),
           rkey: {"n_visited": mean_std("n_visited"),
                  "priority_collected": mean_std("priority_collected"),
                  "persons_reached": mean_std("persons_reached")}}
    rand_mean = out[rkey]["priority_collected"]["mean"]
    for name in ("vrp", "greedy_nearest", "priority_first"):
        ratio = out[name]["priority_collected"] / rand_mean
        out[name]["priority_vs_random_mean_x"] = round(ratio, 2)
    for name, v in out.items():
        print(f"{name}: {json.dumps(v)}")
    return out


def baselines_only() -> None:
    """Recompute only the baselines (deterministic) and keep the stored VRP numbers/plan."""
    path = ROOT / "reports" / "metrics_planning.json"
    stored = json.loads(path.read_text())
    b = gpd.read_file(ROOT / get("paths.data_processed") / "buildings.gpkg")
    depot = tuple(get("routing.depots")[0])
    b = prioritise(b, TopsisInput(depots=[depot]))
    g = apply_debris_risk(load_road_graph(), b, DebrisInput())
    depot_node = int(snap_to_nodes(g, [depot])[0][0])
    n_cfg = get("routing.candidate_n")
    nodes = [depot_node, *stop_nodes(g, depot_node, select_candidates(b, n_cfg),
                                     get("routing.max_walk_m"))]
    prob = prepare_problem(g, b, depot, VrpInput(candidate_n=n_cfg),
                           travel_time_matrix(g, nodes), nodes)
    vrp_sum = stored["variants_at_candidate_n"]["closures_with_walking"]
    stored["baselines_closures_with_walking"] = {"n": n_cfg, **compute_baselines(prob, vrp_sum)}
    path.write_text(json.dumps(stored, indent=1))


def main() -> None:
    t0 = time.time()
    b = gpd.read_file(ROOT / get("paths.data_processed") / "buildings.gpkg")
    depot = tuple(get("routing.depots")[0])
    b = prioritise(b, TopsisInput(depots=[depot]))
    base = load_road_graph()
    graphs = {"no_closures": apply_debris_risk(base, b, DebrisInput(block_count=10**9)),
              "closures": apply_debris_risk(base, b, DebrisInput())}
    depot_node = int(snap_to_nodes(base, [depot])[0][0])
    max_walk = get("routing.max_walk_m")
    n_cfg = get("routing.candidate_n")
    print(f"graphs ready ({time.time() - t0:.0f}s)")

    # buildings unreachable from the depot, over the whole table
    all_nodes = building_nodes(base, b).tolist()
    unreach_all = {}
    for name, g, walk in (("no_closures", graphs["no_closures"], True),
                          ("closures_no_walking", graphs["closures"], False),
                          ("closures_with_walking", graphs["closures"], True)):
        access, wm = walking_access(g, depot_node, all_nodes)
        lim = max_walk if walk else 0.0
        unreach_all[name] = int(sum(a is None or w > lim for a, w in zip(access, wm, strict=True)))
    print("unreachable over all buildings:", unreach_all)

    # one cached matrix per graph for the largest candidate set; smaller N use sub-matrices
    n_max = max((*SWEEP, n_cfg))
    cand_max = select_candidates(b, n_max)
    mats = {}
    for key, g in graphs.items():
        nodes = [depot_node, *stop_nodes(g, depot_node, cand_max, max_walk)]
        mats[key] = (travel_time_matrix(g, nodes), nodes)

    def run(graph_key: str, n: int, walking: bool) -> dict:
        t = time.time()
        m, nodes = mats[graph_key]
        plan = plan_routes(graphs[graph_key], b, depot, VrpInput(candidate_n=n, walking=walking),
                           m, nodes)
        return plan | {"_s": time.time() - t}

    variants, plans = {}, {}
    for name, key, walking in (("no_closures", "no_closures", True),
                               ("closures_no_walking", "closures", False),
                               ("closures_with_walking", "closures", True)):
        plan = run(key, n_cfg, walking)
        plans[name] = plan
        variants[name] = summary(plan, plan["_s"]) | {"n_unreachable_all_buildings":
                                                      unreach_all[name]}
        print(f"{name}: priority {plan['priority_collected']}, persons {plan['persons_reached']}, "
              f"unreachable {plan['n_unreachable']} (walking {plan['n_need_walking']})")

    sweep = {}
    for n in sorted(set(SWEEP) | {n_cfg}):
        plan = plans["closures_with_walking"] if n == n_cfg else run("closures", n, True)
        sweep[str(n)] = summary(plan, plan["_s"])
        print(f"N={n}: visited {plan['n_visited']}, priority {plan['priority_collected']}, "
              f"unreachable {plan['n_unreachable']}")

    prob = prepare_problem(graphs["closures"], b, depot, VrpInput(candidate_n=n_cfg),
                           *mats["closures"])
    vrp = plans["closures_with_walking"]
    baselines = compute_baselines(prob, summary(vrp, vrp["_s"]))

    demo = {k: v for k, v in plans["closures_with_walking"].items() if k != "_s"}
    demo |= {"depot_lonlat": depot, "SIMULATED_DETECTOR": True,
             "disclaimer": "SIMULATED detector; scenario depot; not an operational plan"}
    (ROOT / "reports" / "plan_demo.json").write_text(json.dumps(demo, indent=1))

    out = {"SIMULATED_DETECTOR": True,
           "disclaimer": "Damage probabilities come from a simulated detector (HOTOSM labels + "
                         "noise); depot, debris and walking rules are scenario assumptions; "
                         "numbers show planning behaviour, not real-world rescue outcomes.",
           "n_buildings": len(b),
           "params": {k: get(f"routing.{k}") for k in (
               "num_teams", "mission_duration_hours", "intervention_time_minutes",
               "solver_time_limit_s", "candidate_n", "debris_risk_buffer_meters",
               "debris_damage_prob_min", "debris_min_floors", "debris_penalty_factor",
               "debris_block_count", "walking_speed_kmh", "max_walk_m")},
           "depot_lonlat": depot,
           "variants_at_candidate_n": {"n": n_cfg, **variants},
           "candidate_n_sensitivity_closures_with_walking": sweep,
           "baselines_closures_with_walking": {"n": n_cfg, **baselines},
           "note": "OR-Tools guided local search is time-limited, hence not exactly "
                   "reproducible; differences of a few percent are noise."}
    (ROOT / "reports" / "metrics_planning.json").write_text(json.dumps(out, indent=1))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baselines-only", action="store_true",
                    help="recompute only the baselines, keep stored VRP results (~10 s)")
    baselines_only() if ap.parse_args().baselines_only else main()
