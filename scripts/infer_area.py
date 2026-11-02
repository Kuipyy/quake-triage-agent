"""Sliding-window damage inference over a georeferenced pre/post GeoTIFF pair.

Writes <out>/<name>_prob.tif (and <name>_uncertainty.tif with --tta) on the input grid.
Tiles failing the image-quality guard are left as NaN and listed in the summary.

Usage:
    uv run python scripts/infer_area.py --pre data/raw/demo/pre.tif --post data/raw/demo/post.tif \
        --name antakya --model early_fusion
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from quaketriage.config import get
from quaketriage.models.infer import infer_area, predict_tile
from quaketriage.models.unet import build_siamese_unet, build_unet

ROOT = Path(__file__).parents[1]


def read_rgb(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3])
        profile = src.profile
    return np.moveaxis(arr, 0, -1).astype(np.uint8), profile


def write_band(path: Path, data: np.ndarray, profile: dict) -> None:
    profile = {**profile, "count": 1, "dtype": "float32", "nodata": float("nan"),
               "compress": "deflate"}
    for k in ("photometric", "blockxsize", "blockysize", "tiled"):
        profile.pop(k, None)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)


def load_model(kind: str, checkpoint: Path, encoder: str, device: torch.device):
    builder = build_unet if kind == "early_fusion" else build_siamese_unet
    model = builder(encoder=encoder, encoder_weights=None).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt.get("threshold", get("thresholds.damage_prob"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--name", default="area")
    ap.add_argument("--model", choices=["early_fusion", "siamese"], default="early_fusion")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--encoder", default=get("model.encoder"))
    ap.add_argument("--tta", action="store_true", default=get("inference.tta_flips"),
                    help="flip-TTA std as uncertainty placeholder")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--out", default=str(ROOT / get("paths.outputs") / "rasters"))
    args = ap.parse_args()

    tag = "baseline" if args.model == "early_fusion" else "siamese"
    ckpt = Path(args.checkpoint) if args.checkpoint else ROOT / f"models/{tag}_best.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, thr = load_model(args.model, ckpt, args.encoder, device)

    pre, profile = read_rgb(Path(args.pre))
    post, post_profile = read_rgb(Path(args.post))
    if pre.shape != post.shape or profile["transform"] != post_profile["transform"]:
        raise SystemExit("pre/post must share the same grid (co-register first).")

    res = infer_area(
        pre, post,
        predict=lambda x: predict_tile(model, x.to(device), tta=args.tta),
        tile=get("inference.tile_size"), overlap=get("inference.overlap"),
        use_guard=not args.no_guard,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_band(out / f"{args.name}_prob.tif", res.prob, profile)
    if res.uncertainty is not None:
        write_band(out / f"{args.name}_uncertainty.tif", res.uncertainty, profile)

    valid = ~np.isnan(res.prob)
    damaged = float((res.prob[valid] > thr).mean()) if valid.any() else float("nan")
    summary = {"tiles": res.n_tiles, "rejected": len(res.rejected),
               "valid_pixel_ratio": round(float(valid.mean()), 4),
               "damaged_pixel_ratio": round(damaged, 4), "threshold": thr}
    print(json.dumps(summary))
    for y, x, reasons in res.rejected[:5]:
        print(f"  rejected tile (y={y}, x={x}): {'; '.join(reasons)}")


if __name__ == "__main__":
    main()
