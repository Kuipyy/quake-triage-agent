"""Download demo area data for Antakya:
- OSM building footprints (via OSMnx / Overpass)
- OSM road network (cached GraphML)
- WorldPop Turkey 100m raster (clipped to demo bbox)
"""
from pathlib import Path

import osmnx as ox
from dotenv import load_dotenv

load_dotenv()

# Antakya demo bbox (minlon, minlat, maxlon, maxlat) — covers dense HOTOSM area
DEMO_BBOX = (36.10, 36.15, 36.25, 36.30)   # ~17 km²
DEMO_PLACE = "Antakya, Hatay, Turkey"

OUT_DIR = Path("data/raw/demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download_buildings():
    print("Downloading OSM building footprints for Antakya...")
    tags = {"building": True}
    gdf = ox.features_from_bbox(
        bbox=(DEMO_BBOX[3], DEMO_BBOX[1], DEMO_BBOX[2], DEMO_BBOX[0]),  # N, S, E, W
        tags=tags,
    )
    # Keep only polygons (actual building footprints)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    out = OUT_DIR / "antakya_buildings.gpkg"
    gdf.to_file(out, driver="GPKG")
    print(f"  {len(gdf)} buildings → {out}")
    return gdf


def download_roads():
    print("Downloading OSM road network for Antakya...")
    G = ox.graph_from_bbox(
        bbox=(DEMO_BBOX[3], DEMO_BBOX[1], DEMO_BBOX[2], DEMO_BBOX[0]),  # N, S, E, W
        network_type="drive",
    )
    out = OUT_DIR / "antakya_roads.graphml"
    ox.save_graphml(G, filepath=str(out))
    nodes, edges = ox.graph_to_gdfs(G)
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
    download_buildings()
    download_roads()
    download_worldpop()
    print("\nAll demo area data downloaded.")


if __name__ == "__main__":
    main()
