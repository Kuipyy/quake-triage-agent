"""Verify all Week-1 data files exist and have reasonable sizes."""
import sys
from pathlib import Path

REQUIRED = [
    # KATE-CD
    ("data/raw/kate_cd/data/train-00000-of-00001.parquet", 1e6, "KATE-CD train split"),
    ("data/raw/kate_cd/data/validation-00000-of-00001.parquet", 100_000, "KATE-CD val split"),
    ("data/raw/kate_cd/data/test-00000-of-00001.parquet", 100_000, "KATE-CD test split"),
    # HOTOSM
    (
        "data/raw/hotosm/hotosm_tur_destroyed_buildings_polygons_geojson.geojson",
        100_000,
        "HOTOSM destroyed buildings",
    ),
    # Demo area
    ("data/raw/demo/antakya_buildings.gpkg", 10_000, "Antakya OSM buildings"),
    ("data/raw/demo/antakya_roads.graphml", 10_000, "Antakya OSM roads"),
    ("data/raw/demo/worldpop_turkey_100m.tif", 1e6, "WorldPop Turkey 100m"),
    # Docs
    ("data/raw/docs/AFAD_TAMP.pdf", 500_000, "AFAD TAMP"),
    ("data/raw/docs/INSARAG_Vol2_ManB_Operations.pdf", 100_000, "INSARAG Vol II Man B"),
    ("data/raw/docs/INSARAG_Vol3_FieldGuide.pdf", 100_000, "INSARAG Vol III"),
]

BASE = Path(__file__).parent.parent
ok = True

print("Checking data files...\n")
for rel_path, min_size, label in REQUIRED:
    p = BASE / rel_path
    if not p.exists():
        print(f"  MISSING  {label}")
        print(f"           → {rel_path}")
        ok = False
    elif p.stat().st_size < min_size:
        print(f"  TOO SMALL {label} ({p.stat().st_size} bytes < {int(min_size)})")
        ok = False
    else:
        print(f"  OK  {label} ({p.stat().st_size/1e3:.0f} KB)")

print()
if ok:
    print("All files OK.")
    sys.exit(0)
else:
    print("Some files are missing or too small.")
    sys.exit(1)
