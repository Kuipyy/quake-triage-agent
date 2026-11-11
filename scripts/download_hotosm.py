"""Download the HOTOSM Turkey destroyed-buildings polygons (ODbL) from HDX.

Looks the current GeoJSON resource up through the HDX API (the export URL changes over time) and
unpacks it to data/raw/hotosm/. Data: © OpenStreetMap contributors, Humanitarian OpenStreetMap
Team, ODbL. If this fails (HDX or its S3 export is down, or the dataset moved), download the
GeoJSON zip by hand from
https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings and unpack it into
data/raw/hotosm/ so that data/raw/hotosm/hotosm_tur_destroyed_buildings_polygons_geojson.geojson
exists.

Usage: uv run python scripts/download_hotosm.py
"""
import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).parents[1]
OUT = ROOT / "data" / "raw" / "hotosm"
API = "https://data.humdata.org/api/3/action/package_show?id=hotosm_tur_destroyed_buildings"
TARGET = "hotosm_tur_destroyed_buildings_polygons_geojson.geojson"


def main() -> int:
    if (OUT / TARGET).exists():
        print(f"already there: {OUT / TARGET}")
        return 0
    try:
        res = requests.get(API, timeout=60).json()["result"]["resources"]
        url = next(r["download_url"] for r in res
                   if r["name"].endswith("polygons_geojson.zip"))
        print("downloading", url[:90], "...")
        data = requests.get(url, timeout=120).content
        OUT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(OUT)
    except (requests.RequestException, KeyError, StopIteration, zipfile.BadZipFile) as exc:
        print(f"download failed ({type(exc).__name__}): {exc}")
        print("See the docstring of this script for the manual route.")
        return 1
    if not (OUT / TARGET).exists():
        print(f"zip did not contain {TARGET}; files: {[p.name for p in OUT.iterdir()]}")
        return 1
    print(f"OK → {OUT / TARGET} ({(OUT / TARGET).stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
