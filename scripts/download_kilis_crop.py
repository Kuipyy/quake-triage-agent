"""Download a co-registered pre/post crop of the Kilis Vantor pair (COG range reads only).

Both tiles are EPSG:32637. The post image (0.33 m) is resampled onto the pre grid (0.58 m),
so the output pair shares one grid and can be fed to scripts/infer_area.py.

Imagery: Vantor (Maxar) Open Data, CC BY-NC 4.0. Keep attribution in the README.

Usage:
    uv run python scripts/download_kilis_crop.py [--size 2048]
"""
import argparse
from pathlib import Path

import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, from_bounds

from quaketriage.config import get

ROOT = Path(__file__).parents[1]
BASE = ("https://maxar-opendata.s3.amazonaws.com/events/Kahramanmaras-turkey-earthquake-23/"
        "ard/37/031133031132")
PRE = f"{BASE}/2023-01-26/10300100E18A1000-visual.tif"
POST = f"{BASE}/2023-02-07/1040010082B85E00-visual.tif"
CENTER_LONLAT = (37.115, 36.712)  # Kilis town, inside the shared tile footprint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048, help="crop size in pre pixels")
    ap.add_argument("--out", default=str(ROOT / get("paths.data_raw") / "demo"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(PRE) as pre, rasterio.open(POST) as post:
            assert str(pre.crs) == str(post.crs) == get("geo.crs"), (pre.crs, post.crs)
            cx, cy = Transformer.from_crs("EPSG:4326", pre.crs, always_xy=True).transform(
                *CENTER_LONLAT)
            half = args.size * pre.res[0] / 2
            win = from_bounds(cx - half, cy - half, cx + half, cy + half, pre.transform)
            win = Window(round(win.col_off), round(win.row_off), args.size, args.size)
            transform = pre.window_transform(win)
            pre_arr = pre.read([1, 2, 3], window=win, boundless=True, fill_value=0)
            with WarpedVRT(post, crs=pre.crs, transform=transform, width=args.size,
                           height=args.size, resampling=Resampling.average, nodata=0) as vrt:
                post_arr = vrt.read([1, 2, 3])
            profile = dict(driver="GTiff", count=3, dtype="uint8", crs=pre.crs,
                           transform=transform, width=args.size, height=args.size,
                           compress="deflate")
            for name, arr in (("pre", pre_arr), ("post", post_arr)):
                path = out / f"kilis_{name}.tif"
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(arr)
                print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1e6:.1f} MB  "
                      f"black={(arr.max(0) == 0).mean():.3f}")


if __name__ == "__main__":
    main()
