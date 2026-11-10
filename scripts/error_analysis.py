"""Error analysis: the 10 worst KATE-CD test tiles of the Siamese model (real) + the 10 worst
planning mistakes on the SIMULATED Antakya table (top-priority buildings that are not destroyed).

Writes reports/error_analysis.json and reports/figures/worst10_siamese.png.
Usage: uv run python scripts/error_analysis.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from quaketriage.agent.tools import ToolContext  # noqa: E402
from quaketriage.config import get  # noqa: E402
from quaketriage.data.kate_cd import KateCDDataset  # noqa: E402
from quaketriage.models.unet import build_siamese_unet  # noqa: E402
from quaketriage.provenance import write_json  # noqa: E402

ROOT = Path(__file__).parents[1]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def kate_worst(n: int = 10) -> tuple[list[dict], float, int, int]:
    thr = json.loads((ROOT / "reports/comparison_siamese_vs_baseline_f1.json").read_text())[
        "models"]["siamese"]["threshold"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(ROOT / "models/siamese_best.pt", map_location=device, weights_only=False)
    model = build_siamese_unet(encoder=get("model.encoder"), encoder_weights=None).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    data = KateCDDataset(str(ROOT / "data/raw/kate_cd"), "test")
    probs, masks = [], []
    with torch.no_grad():
        for x, m in DataLoader(data, batch_size=8, shuffle=False, num_workers=0):
            probs.append(torch.sigmoid(model(x.to(device))).cpu())
            masks.append(m)
    probs, masks = torch.cat(probs), torch.cat(masks)
    pred, gt = (probs > thr).bool(), masks.bool()
    rows = []
    for i in range(len(data)):
        tp = int((pred[i] & gt[i]).sum())
        fp = int((pred[i] & ~gt[i]).sum())
        fn = int((~pred[i] & gt[i]).sum())
        if tp + fp + fn == 0:
            continue  # nothing predicted, nothing to find
        rows.append({"tile": i, "iou": round(tp / (tp + fp + fn), 4),
                     "f1": round(2 * tp / (2 * tp + fp + fn), 4), "gt_px": tp + fn,
                     "pred_px": tp + fp, "fp_px": fp, "fn_px": fn,
                     "error_type": "false_alarm" if fp > fn else "missed_damage"})
    worst = sorted(rows, key=lambda r: (r["iou"], -(r["fp_px"] + r["fn_px"])))[:n]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    for r, ax in zip(worst, axes, strict=True):
        x, m = data[r["tile"]]
        pre = (x[:3] * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        post = (x[3:] * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        for a, img, title, kw in (
                (ax[0], pre, f"tile {r['tile']} pre", {}), (ax[1], post, "post", {}),
                (ax[2], pred[r["tile"], 0].numpy(), f"pred (IoU {r['iou']:.2f})",
                 {"cmap": "Reds", "vmin": 0, "vmax": 1}),
                (ax[3], gt[r["tile"], 0].numpy(), f"gt ({r['error_type']})",
                 {"cmap": "Reds", "vmin": 0, "vmax": 1})):
            a.imshow(img, **kw)
            a.set_title(title, fontsize=9)
            a.axis("off")
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures/worst10_siamese.png", dpi=45, bbox_inches="tight")
    plt.close()
    return worst, thr, len(rows), len(data)


def antakya_worst(n: int = 10) -> list[dict]:
    ctx = ToolContext()
    for step in ("detect_damage", "score_buildings", "estimate_occupancy", "prioritize_sites"):
        ctx.call(step, {"source": "simulated"} if step == "detect_damage" else {})
    b = ctx.buildings
    bad = b[(~b["destroyed_truth"]) & (~b["needs_verification"])].nlargest(n, "priority")
    return [{"building_id": (int(r["id"]) if "id" in b and np.isfinite(r["id"]) else None),
             "row": int(i), "priority": round(float(r["priority"]), 4),
             "damage_prob": round(float(r["damage_prob"]), 3),
             "flagged_damaged": bool(r["damaged"]),
             "uncertainty": round(float(r["uncertainty"]), 3),
             "estimated_persons": round(float(r["estimated_persons"]), 1),
             "building_tag": str(r.get("building", "")),
             "floors": None if r.get("building:levels") != r.get("building:levels") else
             r.get("building:levels")} for i, r in bad.iterrows()]


def main() -> None:
    worst, thr, n_eval, n_tiles = kate_worst()
    ant = antakya_worst()
    write_json(ROOT / "reports/error_analysis.json", {
        "kate_cd_siamese": {"split": "test", "threshold": thr, "n_tiles": n_tiles,
                            "n_tiles_with_any_damage_or_prediction": n_eval,
                            "n_worst_false_alarm": sum(r["error_type"] == "false_alarm"
                                                       for r in worst),
                            "n_worst_missed_damage": sum(r["error_type"] == "missed_damage"
                                                         for r in worst),
                            "worst_tiles": worst,
                            "note": "KATE-CD tiles, not buildings (no building labels there)."},
        "antakya_simulated": {"SIMULATED_DETECTOR": True,
                              "definition": "highest-priority buildings that are NOT destroyed in "
                                            "HOTOSM truth and not in the verification queue; "
                                            "truth is incomplete, so some may be real damage",
                              "worst_false_alarms": ant}})
    print(f"tiles evaluated {n_eval}/{n_tiles}; worst IoU {worst[0]['iou']}..{worst[-1]['iou']}; "
          f"types {[r['error_type'] for r in worst]}")


if __name__ == "__main__":
    main()
