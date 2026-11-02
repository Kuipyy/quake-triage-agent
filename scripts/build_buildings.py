"""Build the Antakya building table for the planning demo.

Pipeline: OSM footprints -> HOTOSM ground-truth labels -> WorldPop occupancy -> SIMULATED detector
-> verification queue -> data/processed/buildings.gpkg, plus a noise-sensitivity grid in
reports/metrics_building.json.

IMPORTANT: damage_prob / uncertainty here are SIMULATED (no Antakya imagery pair exists); the
metrics describe the planning pipeline's robustness to detector noise, not model accuracy.

Usage:
    uv run python scripts/build_buildings.py [--miss 0.2 --false-alarm 0.03 --size-exponent 1.0]
"""
import argparse
import itertools
from pathlib import Path

import geopandas as gpd
import numpy as np

from quaketriage.buildings.population import coverage, estimate_population
from quaketriage.buildings.queue import flag_verification
from quaketriage.buildings.simulate import (
    SimulatorConfig,
    building_metrics,
    label_destroyed,
    simulate_detector,
)
from quaketriage.config import get
from quaketriage.data.geo import get_project_crs
from quaketriage.provenance import write_json

ROOT = Path(__file__).parents[1]
MISS_GRID = (0.0, 0.1, 0.2, 0.4)
FA_GRID = (0.0, 0.02, 0.05, 0.1)
SEEDS = range(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss", type=float, default=0.2)
    ap.add_argument("--false-alarm", type=float, default=0.03)
    ap.add_argument("--size-exponent", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = ROOT / get("paths.data_raw")
    buildings = gpd.read_file(raw / "demo/antakya_buildings.gpkg")
    hotosm = gpd.read_file(raw / "hotosm/hotosm_tur_destroyed_buildings_polygons_geojson.geojson")
    bbox = buildings.total_bounds
    hotosm = hotosm.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    destroyed = label_destroyed(buildings, hotosm)
    print(f"{len(buildings)} buildings, {len(hotosm)} HOTOSM polygons in area, "
          f"{int(destroyed.sum())} buildings labelled destroyed ({destroyed.mean():.1%})")

    buildings = estimate_population(buildings, raw / "demo/worldpop_turkey_100m.tif")
    cov = coverage(buildings)
    print(f"population: {cov['residents_day']:.0f} residents (day), "
          f"{cov['estimated_persons_night']:.0f} at night")

    base = SimulatorConfig(miss_rate=args.miss, false_alarm_rate=args.false_alarm,
                           size_exponent=args.size_exponent, seed=args.seed)
    sim = flag_verification(simulate_detector(buildings, destroyed, base))
    metrics = building_metrics(sim)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})

    out = ROOT / get("paths.data_processed") / "buildings.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sim.to_crs(get_project_crs()).to_file(out, driver="GPKG")

    grid = []
    for miss, fa in itertools.product(MISS_GRID, FA_GRID):
        runs = []
        for seed in SEEDS:
            cfg = SimulatorConfig(miss_rate=miss, false_alarm_rate=fa,
                                  size_exponent=args.size_exponent, seed=seed)
            s = flag_verification(simulate_detector(buildings, destroyed, cfg))
            runs.append({**building_metrics(s), "queue_share": float(s.needs_verification.mean())})
        agg = {k: (float(np.nanmean([r[k] for r in runs])), float(np.nanstd([r[k] for r in runs])))
               for k in ("precision", "recall", "f1", "person_recall", "queue_share")}
        grid.append({"miss_rate": miss, "false_alarm_rate": fa,
                     **{f"{k}_mean": round(v[0], 4) for k, v in agg.items()},
                     **{f"{k}_std": round(v[1], 4) for k, v in agg.items()}})

    write_json(ROOT / get("paths.reports") / "metrics_building.json", {
        "SIMULATED_DETECTOR": True,
        "disclaimer": "damage_prob/uncertainty are simulated from HOTOSM labels; these numbers "
                      "measure planning-pipeline robustness to detector noise, NOT model accuracy.",
        "n_buildings": int(len(buildings)), "n_destroyed_truth": int(destroyed.sum()),
        "base_scenario": {"miss_rate": args.miss, "false_alarm_rate": args.false_alarm,
                          "size_exponent": args.size_exponent, "seed": args.seed,
                          "metrics": metrics,
                          "queue_share": float(sim.needs_verification.mean())},
        "population": cov, "n_seeds": len(SEEDS), "sensitivity_grid": grid})
    print(f"buildings → {out.relative_to(ROOT)};  metrics → reports/metrics_building.json")


if __name__ == "__main__":
    main()
