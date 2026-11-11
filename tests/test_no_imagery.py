"""Guard: satellite imagery (KATE-CD, Vantor/Maxar) must never be tracked in git."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".webp"}
FORBIDDEN = ("kate", "vantor", "maxar", "kilis", "worst10", "predictions_", "sample",
             "before_after")


def test_no_satellite_imagery_tracked():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.split()
    bad = [f for f in files if Path(f).suffix.lower() in IMAGE_EXT
           and any(w in f.lower() for w in FORBIDDEN)]
    assert not bad, f"imagery tracked in git: {bad}"
    tifs = [f for f in files if Path(f).suffix.lower() in {".tif", ".tiff"}]
    assert not tifs, f"GeoTIFFs tracked in git: {tifs}"
