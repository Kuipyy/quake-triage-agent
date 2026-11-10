"""Apply debris risk to the real Antakya graph + building table and print a short summary."""
from pathlib import Path

import geopandas as gpd

from quaketriage.config import get
from quaketriage.planning.debris import DebrisInput, apply_debris_risk, debris_sources
from quaketriage.planning.roads import load_road_graph

ROOT = Path(__file__).parents[1]


def main() -> None:
    b = gpd.read_file(ROOT / get("paths.data_processed") / "buildings.gpkg")
    g = load_road_graph()
    g = apply_debris_risk(g, b)
    ed = [d for *_, d in g.edges(data=True)]
    n = len(ed)
    risk = sum(d["debris_sources"] > 0 for d in ed)
    closed = sum(d["blocked"] for d in ed)
    print(f"sources {len(debris_sources(b, DebrisInput()))}/{len(b)} buildings; "
          f"edges {n}: penalised {risk} ({risk / n:.1%}), closed {closed} ({closed / n:.1%})")


if __name__ == "__main__":
    main()
