"""Download demo area data for Antakya:
- OSM building footprints (via OSMnx / Overpass)
- OSM road network (cached GraphML)
- WorldPop Turkey 100m raster (clipped to demo bbox)
"""
import time
from pathlib import Path

import osmnx as ox
from dotenv import load_dotenv

load_dotenv()
ox.settings.http_user_agent = "quaketriage-student-research/0.1 (non-commercial prototype)"
ox.settings.requests_timeout = 300

# Antakya demo bbox (minlon, minlat, maxlon, maxlat) — covers dense HOTOSM area
DEMO_BBOX = (36.10, 36.15, 36.25, 36.30)   # ~17 km²
DEMO_PLACE = "Antakya, Hatay, Turkey"

OUT_DIR = Path("data/raw/demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def check_inside_bbox(gdf, what: str, tol: float = 0.01) -> None:
    """Fail loudly if a download is not inside DEMO_BBOX (lesson from the osmnx bbox-order bug)."""
    minx, miny, maxx, maxy = gdf.total_bounds
    w, s_, e, n = DEMO_BBOX
    if not (minx >= w - tol and miny >= s_ - tol and maxx <= e + tol and maxy <= n + tol):
        raise RuntimeError(f"{what} lies outside DEMO_BBOX {DEMO_BBOX}: "
                           f"{[round(v, 4) for v in (minx, miny, maxx, maxy)]}")


def with_retries(fn, attempts: int = 8, base_delay: int = 60):
    """Public Overpass servers are often at capacity: retry with exponential back-off."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - any network/parse error is retryable here
            if i == attempts:
                raise
            delay = min(base_delay * 2 ** (i - 1), 900)
            print(f"  attempt {i}/{attempts} failed ({type(exc).__name__}); retry in {delay}s",
                  flush=True)
            time.sleep(delay)


def download_buildings():
    print("Downloading OSM building footprints for Antakya...")
    tags = {"building": True}
    gdf = ox.features_from_bbox(
        bbox=DEMO_BBOX,  # osmnx >= 2: (left, bottom, right, top) = (W, S, E, N)
        tags=tags,
    )
    # Keep only polygons (actual building footprints)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    check_inside_bbox(gdf, "buildings")
    out = OUT_DIR / "antakya_buildings.gpkg"
    gdf.to_file(out, driver="GPKG")
    print(f"  {len(gdf)} buildings → {out}")
    return gdf


def download_roads():
    print("Downloading OSM road network for Antakya...")
    G = ox.graph_from_bbox(
        bbox=DEMO_BBOX,  # osmnx >= 2: (left, bottom, right, top) = (W, S, E, N)
        network_type="drive",
    )
    out = OUT_DIR / "antakya_roads.graphml"
    ox.save_graphml(G, filepath=str(out))
    nodes, edges = ox.graph_to_gdfs(G)
    check_inside_bbox(nodes, "road nodes")
    print(f"  {len(nodes)} nodes, {len(edges)} edges → {out}")
    return G


def download_worldpop():
    """WorldPop Turkey 100m — download via direct URL (CC BY 4.0)."""
    import requests
    # WorldPop Turkey 2020 100m constrained individual countries
    # Direct download from worldpop.org FTP / S3 mirror
    url = "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/TUR/tur_ppp_2020_UNadj_constrained.tif"
    out = OUT_DIR / "worldpop_turkey_100m.tif"
    if out.exists():
        print(f"WorldPop already exists → {out}")
        return

    print("Downloading WorldPop Turkey 100m...")
    print("  (This is ~10 MB — clipping to demo area will happen in Week 4)")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"  {downloaded/1e6:.1f}/{total/1e6:.1f} MB", end="\r")
    print(f"\n  Done → {out} ({out.stat().st_size/1e6:.1f} MB)")


def main():
    with_retries(download_buildings)
    with_retries(download_roads)
    download_worldpop()
    print("\nAll demo area data downloaded.")


if __name__ == "__main__":
    main()
