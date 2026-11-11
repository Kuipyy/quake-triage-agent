"""Verify Week-1 data files exist, have reasonable sizes, and match their expected SHA-256.

Also writes (or refreshes) data/manifest.json with source URL, license, sha256,
size and download date for each file.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from quaketriage.provenance import file_sha256

# (rel_path, min_bytes, label, source_url, license, sha256_expected_or_None)
# sha256_expected: set to None on first run; populated from manifest.json afterwards.
REQUIRED = [
    (
        "data/raw/kate_cd/data/train-00000-of-00001.parquet",
        1_000_000,
        "KATE-CD train split",
        "https://huggingface.co/datasets/CSCRS/kate-cd",
        "No explicit license; cite Musaoglu et al., JARS 19(4) 044508 (2025)",
        None,
    ),
    (
        "data/raw/kate_cd/data/validation-00000-of-00001.parquet",
        100_000,
        "KATE-CD val split",
        "https://huggingface.co/datasets/CSCRS/kate-cd",
        "No explicit license; cite Musaoglu et al., JARS 19(4) 044508 (2025)",
        None,
    ),
    (
        "data/raw/kate_cd/data/test-00000-of-00001.parquet",
        100_000,
        "KATE-CD test split",
        "https://huggingface.co/datasets/CSCRS/kate-cd",
        "No explicit license; cite Musaoglu et al., JARS 19(4) 044508 (2025)",
        None,
    ),
    (
        "data/raw/hotosm/hotosm_tur_destroyed_buildings_polygons_geojson.geojson",
        100_000,
        "HOTOSM destroyed buildings",
        "https://data.humdata.org/dataset/hotosm_tur_destroyed_buildings",
        "ODbL",
        None,
    ),
    (
        "data/raw/demo/antakya_buildings.gpkg",
        10_000,
        "Antakya OSM buildings",
        "https://www.openstreetmap.org (via OSMnx/Overpass)",
        "ODbL",
        None,
    ),
    (
        "data/raw/demo/antakya_roads.graphml",
        10_000,
        "Antakya OSM roads",
        "https://www.openstreetmap.org (via OSMnx)",
        "ODbL",
        None,
    ),
    (
        "data/raw/demo/worldpop_turkey_100m.tif",
        1_000_000,
        "WorldPop Turkey 100m",
        "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/TUR/",
        "CC BY 4.0",
        None,
    ),
    (
        "data/raw/docs/AFAD_TAMP.pdf",
        500_000,
        "AFAD TAMP",
        "https://www.afad.gov.tr/kurumlar/afad.gov.tr/e_Kutuphane/Planlar/TAMP.pdf",
        "Public document",
        None,
    ),
    (
        "data/raw/docs/INSARAG_Vol2_ManB_Operations.pdf",
        100_000,
        "INSARAG Vol II Man B",
        "https://insarag.org/wp-content/uploads/2021/06/INSARAG20Guidelines20Vol20II2C20Man20B.pdf",
        "Public document",
        None,
    ),
    (
        "data/raw/docs/INSARAG_Vol3_FieldGuide.pdf",
        100_000,
        "INSARAG Vol III",
        "https://insarag.org/wp-content/uploads/2021/06/INSARAG20Guidelines20Vol20III.pdf",
        "Public document",
        None,
    ),
]

# Path 2 (model training) data: absence is reported but does not fail the check.
OPTIONAL_PREFIXES = ("data/raw/kate_cd/",)

BASE = Path(__file__).parent.parent
MANIFEST_PATH = BASE / "data" / "manifest.json"


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(entries: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")


def check_demo_extent(tol: float = 0.01) -> bool:
    """Guard against the Week-1 bbox-order bug: demo buildings must lie inside the demo bbox."""
    import geopandas as gpd

    from quaketriage.config import get

    w, s_, e, n = get("geo.demo_bbox")
    path = BASE / "data/raw/demo/antakya_buildings.gpkg"
    minx, miny, maxx, maxy = gpd.read_file(path).total_bounds
    inside = minx >= w - tol and miny >= s_ - tol and maxx <= e + tol and maxy <= n + tol
    print(f"  {'OK ' if inside else 'BAD'} Antakya buildings extent "
          f"[{minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}] vs demo bbox {[w, s_, e, n]}")
    return bool(inside)


def main() -> int:
    manifest = load_manifest()
    ok = True
    updated = {}

    print("Checking data files...\n")
    for rel_path, min_size, label, source_url, license_, _ in REQUIRED:
        p = BASE / rel_path
        if not p.exists():
            if rel_path.startswith(OPTIONAL_PREFIXES):
                print(f"  optional  {label}: not downloaded (training only)")
                continue
            print(f"  MISSING   {label}")
            print(f"            → {rel_path}")
            ok = False
            continue

        size = p.stat().st_size
        if size < min_size:
            print(f"  TOO SMALL {label} ({size} B < {min_size})")
            ok = False
            continue

        sha = file_sha256(p)
        prev = manifest.get(rel_path, {})

        if prev.get("sha256") and prev["sha256"] != sha:
            print(f"  SHA MISMATCH {label}")
            print(f"    expected: {prev['sha256']}")
            print(f"    actual:   {sha}")
            ok = False
        else:
            print(f"  OK  {label}  ({size / 1e3:.0f} KB)  sha256={sha[:12]}...")

        updated[rel_path] = {
            "label": label,
            "source_url": source_url,
            "license": license_,
            "sha256": sha,
            "size_bytes": size,
            "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    if ok and not check_demo_extent():
        ok = False

    save_manifest(updated)
    print(f"\nManifest written → {MANIFEST_PATH.relative_to(BASE)}")
    print()
    if ok:
        print("All files OK.")
        return 0
    print("Some required files are missing, too small, or have SHA-256 mismatches.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
