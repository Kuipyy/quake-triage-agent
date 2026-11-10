"""How much does the TOPSIS ranking change when the depot moves?

Compares the configured depot with other real OSM objects in the demo area (hospital, town hall).
Writes reports/depot_sensitivity.json. Usage: uv run python scripts/depot_sensitivity.py
"""
import json
from pathlib import Path

import geopandas as gpd
from scipy.stats import spearmanr

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs
from quaketriage.planning.topsis import TopsisInput, prioritise

ROOT = Path(__file__).parents[1]
TOP_KS = (100, 500)
# alternative depots: OSM building centroids (ids from antakya_buildings.gpkg)
ALTERNATIVES = {"hospital (OSM way 366665647)": 366665647,
                "town hall (OSM way 311391399)": 311391399}


def main() -> None:
    b = gpd.read_file(ROOT / get("paths.data_processed") / "buildings.gpkg")
    raw = gpd.read_file(ROOT / get("paths.data_raw") / "demo" / "antakya_buildings.gpkg")
    base = prioritise(b, TopsisInput())
    base_top = {k: set(base.nlargest(k, "priority").index) for k in TOP_KS}
    out = {"top_ks": TOP_KS, "n_buildings": len(b),
           "base_depot": get("routing.depots")[0], "alternatives": {}}
    for name, osm_id in ALTERNATIVES.items():
        row = raw[raw["id"] == osm_id].to_crs(get_project_crs())
        c = row.geometry.centroid.to_crs(4326).iloc[0]
        alt = prioritise(b, TopsisInput(depots=[(c.x, c.y)]))
        rho = spearmanr(base["priority"], alt["priority"]).statistic
        overlaps = {}
        for k in TOP_KS:
            common = base_top[k] & set(alt.nlargest(k, "priority").index)
            overlaps[f"top{k}_overlap"] = round(len(common) / k, 3)
        out["alternatives"][name] = {"depot": [round(c.x, 5), round(c.y, 5)],
                                     "spearman": round(float(rho), 4), **overlaps}
        print(f"{name}: spearman {rho:.3f}, {overlaps}")
    (ROOT / "reports" / "depot_sensitivity.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
