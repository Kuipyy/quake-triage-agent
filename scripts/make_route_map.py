"""Write reports/route_map.html from reports/plan_demo.json (run plan_routes.py first).

View it over HTTP, not file:// (see README): python3 -m http.server 8000 --directory reports
"""
import json
from pathlib import Path

import geopandas as gpd

from quaketriage.config import get
from quaketriage.planning.debris import DebrisInput, apply_debris_risk
from quaketriage.planning.roads import load_road_graph, snap_to_nodes
from quaketriage.planning.topsis import TopsisInput, prioritise
from quaketriage.ui.route_map import build_route_map

ROOT = Path(__file__).parents[1]


def main() -> None:
    plan = json.loads((ROOT / "reports" / "plan_demo.json").read_text())
    depot = tuple(plan["depot_lonlat"])
    b = prioritise(gpd.read_file(ROOT / get("paths.data_processed") / "buildings.gpkg"),
                   TopsisInput(depots=[depot]))
    g = apply_debris_risk(load_road_graph(), b, DebrisInput())
    depot_node = int(snap_to_nodes(g, [depot])[0][0])
    out = ROOT / "reports" / "route_map.html"
    build_route_map(g, plan, b, depot_node, depot).save(str(out))
    size_mb = out.stat().st_size / 1e6
    print(f"{plan['n_visited']} stops → {out.relative_to(ROOT)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
