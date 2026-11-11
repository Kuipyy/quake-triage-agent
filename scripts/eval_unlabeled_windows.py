# ruff: noqa: E501
"""False-positive rate of the Siamese model on windows without HOTOSM damage labels (Kilis crop).

The only demo area with real pre/post imagery is the Kilis Vantor crop (2048x2048 px). It is cut into
non-overlapping 512 px windows; windows that touch a HOTOSM destroyed-building polygon are excluded and the
rest are *presumed* undamaged. HOTOSM is incomplete, so a flagged pixel is not proof of an error: the numbers
are an upper bound on the false-positive rate only if the unlabelled windows really are intact, which was
not verified on the ground. The training set is not touched. KATE-CD background pixels (label = 0) give a
reference rate on the training domain.

Writes reports/metrics_unlabeled_windows.json (no imagery). Usage: uv run python scripts/eval_unlabeled_windows.py
"""
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from scipy import ndimage
from shapely.geometry import box

from quaketriage.config import get
from quaketriage.data.kate_cd import SPLITS, _decode_mask, _decode_rgb
from quaketriage.data.quality import QualityInput, check_image_quality
from quaketriage.models.infer import predict_tile, to_tensor
from quaketriage.models.loading import load_model
from quaketriage.provenance import write_json

ROOT = Path(__file__).parents[1]
TILE = int(get("inference.tile_size"))
MIN_COMPONENT_PX = 500  # ~45 m2 at 0.305 m/px: smaller blobs are not counted as a damaged building
WINDOW_FLAG_SHARE = 0.01  # a window counts as "flagged" if >= 1 % of its pixels are predicted damaged


def read(path: Path):
    with rasterio.open(path) as s:
        return np.moveaxis(s.read([1, 2, 3]), 0, -1).astype(np.uint8), s.transform, s.crs


def stats(prob: np.ndarray, thr: float) -> dict:
    m = prob > thr
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1)) if n else np.array([])
    return {"pred_pixel_share": round(float(m.mean()), 5),
            "components_ge_min": int((sizes >= MIN_COMPONENT_PX).sum())}


def main() -> None:
    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _ = load_model("siamese", ROOT / "models/siamese_best.pt", get("model.encoder"), device)
    thr = float(get("thresholds.damage_prob"))
    cfg = get("regions.kilis")
    pre, tf, crs = read(ROOT / cfg["pre"])
    post, tf2, _ = read(ROOT / cfg["post"])
    assert tf == tf2 and pre.shape == post.shape

    left, top = tf * (0, 0)
    right, bottom = tf * (pre.shape[1], pre.shape[0])
    hot = gpd.read_file(ROOT / "data/raw/hotosm/hotosm_tur_destroyed_buildings_polygons_geojson.geojson",
                        bbox=tuple(gpd.GeoSeries([box(left, bottom, right, top)], crs=crs)
                                   .to_crs(4326).total_bounds)).to_crs(crs)
    rows = []
    for y in range(0, pre.shape[0] - TILE + 1, TILE):
        for x in range(0, pre.shape[1] - TILE + 1, TILE):
            (x0, y0), (x1, y1) = tf * (x, y), tf * (x + TILE, y + TILE)
            win = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            p, q = pre[y:y + TILE, x:x + TILE], post[y:y + TILE, x:x + TILE]
            rep = check_image_quality(QualityInput(pre=p, post=q))
            row = {"y": y, "x": x, "touches_hotosm_destroyed": bool(hot.intersects(win).any()),
                   "guard_ok": bool(rep.ok), "guard_reasons": list(rep.reasons)}
            if rep.ok:
                prob, std = predict_tile(model, to_tensor(p, q).to(device), tta=True)
                row |= {"mean_prob": round(float(prob.mean()), 5),
                        "mean_uncertainty": round(float(std.mean()), 5),
                        "at_operating_threshold": stats(prob, thr), "at_0.5": stats(prob, 0.5)}
            rows.append(row)

    presumed = [r for r in rows if not r["touches_hotosm_destroyed"] and r["guard_ok"]]

    def agg(key: str) -> dict:
        shares = np.array([r[key]["pred_pixel_share"] for r in presumed])
        return {"mean_pred_pixel_share": round(float(shares.mean()), 5) if len(shares) else None,
                "max_pred_pixel_share": round(float(shares.max()), 5) if len(shares) else None,
                "windows_flagged": int((shares >= WINDOW_FLAG_SHARE).sum()),
                "windows_with_component": int(sum(r[key]["components_ge_min"] > 0 for r in presumed))}

    # reference: false-positive pixel rate on KATE-CD background (label = 0) pixels, same model/threshold
    import pandas as pd
    df = pd.read_parquet(ROOT / get("paths.data_raw") / "kate_cd" / SPLITS["test"])
    fp = neg = 0
    for _, r in df.iterrows():
        a, b, lab = _decode_rgb(r["pre_image"]), _decode_rgb(r["post_image"]), _decode_mask(r["label"]) > 127
        prob, _ = predict_tile(model, to_tensor(a, b).to(device), tta=True)
        fp += int(((prob > thr) & ~lab).sum())
        neg += int((~lab).sum())

    out = {
        "scope": ("Kilis Vantor crop, the only demo area with real pre/post imagery; Antakya has none. "
                  "Windows touching a HOTOSM destroyed polygon are excluded, the rest are PRESUMED undamaged "
                  "(not verified; HOTOSM is incomplete). Not a validated false-positive rate."),
        "model": "siamese_best.pt, 8-view TTA", "threshold": thr, "tile": TILE,
        "n_windows": len(rows), "n_touching_hotosm": sum(r["touches_hotosm_destroyed"] for r in rows),
        "n_guard_rejected": sum(not r["guard_ok"] for r in rows), "n_presumed_undamaged": len(presumed),
        "hotosm_destroyed_polygons_in_crop": int(len(hot)),
        "window_flag_share": WINDOW_FLAG_SHARE, "min_component_px": MIN_COMPONENT_PX,
        "presumed_undamaged_at_operating_threshold": agg("at_operating_threshold"),
        "presumed_undamaged_at_0.5": agg("at_0.5"),
        "kate_cd_test_background_fp_pixel_rate": {
            "value": round(fp / neg, 5), "n_background_pixels": neg,
            "note": "share of label-0 pixels predicted damaged; every KATE-CD tile contains damage"},
        "windows": rows, "seconds": round(time.time() - t0, 1)}
    path = ROOT / "reports" / "metrics_unlabeled_windows.json"
    write_json(path, out)
    s = out["presumed_undamaged_at_operating_threshold"]
    print(f"{out['n_presumed_undamaged']}/{len(rows)} windows presumed undamaged; "
          f"mean predicted-damaged share {s['mean_pred_pixel_share']}, flagged windows {s['windows_flagged']}, "
          f"KATE background FP rate {out['kate_cd_test_background_fp_pixel_rate']['value']} ({out['seconds']}s)")


if __name__ == "__main__":
    main()
