"""Evaluate the baseline U-Net checkpoint on the KATE-CD test set.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --checkpoint models/baseline_best.pt --split test
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quaketriage.config import get
from quaketriage.data.kate_cd import KateCDDataset
from quaketriage.metrics import bootstrap_ci, compute_metrics
from quaketriage.models.unet import build_siamese_unet, build_unet
from quaketriage.provenance import write_json

ROOT = Path(__file__).parents[1]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def save_prediction_figure(logits: torch.Tensor, targets: torch.Tensor,
                            dataset: KateCDDataset, threshold: float, tag: str = "baseline",
                            n: int = 8) -> None:
    """Save a grid of pre/post/pred/gt images."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping figure.")
        return

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, 4, figsize=(14, 3 * n))
    axes = axes if n > 1 else [axes]

    for i in range(n):
        img_tensor, mask_tensor = dataset[i]
        pre_t = img_tensor[:3]
        post_t = img_tensor[3:]

        # Denormalize
        pre_rgb = ((pre_t * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy())
        post_rgb = ((post_t * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy())

        pred_mask = (torch.sigmoid(logits[i, 0]) > threshold).float().numpy()
        gt_mask = mask_tensor[0].numpy()

        panels = [
            (pre_rgb, "Pre", {}),
            (post_rgb, "Post", {}),
            (pred_mask, f"Pred (thr={threshold})", {"cmap": "Reds", "vmin": 0, "vmax": 1}),
            (gt_mask, "Ground Truth", {"cmap": "Reds", "vmin": 0, "vmax": 1}),
        ]
        for ax, (img, title, kw) in zip(axes[i], panels):
            ax.imshow(img, **kw)
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    out_path = FIGURES_DIR / f"predictions_{tag}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {out_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["early_fusion", "siamese"], default="early_fusion")
    parser.add_argument("--checkpoint", default=None,
                        help="default: models/baseline_best.pt or models/siamese_best.pt")
    parser.add_argument("--data-root", default=str(ROOT / "data/raw/kate_cd"))
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--batch-size", type=int, default=get("training.batch_size"))
    parser.add_argument("--encoder", default=get("model.encoder"))
    args = parser.parse_args()

    tag = "baseline" if args.model == "early_fusion" else "siamese"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else ROOT / f"models/{tag}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    threshold = ckpt.get("threshold", get("thresholds.damage_prob"))
    trained_epoch = ckpt.get("epoch", "?")
    best_val_f1 = ckpt.get("best_f1", None)
    print(f"  Epoch: {trained_epoch}  Val F1 (at save): {best_val_f1:.4f}  Threshold: {threshold}")

    builder = build_unet if args.model == "early_fusion" else build_siamese_unet
    model = builder(encoder=args.encoder).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = KateCDDataset(args.data_root, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset: {args.split} ({len(ds)} samples)")

    t0 = time.time()
    all_logits, all_masks = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())

    all_logits = torch.cat(all_logits)
    all_masks = torch.cat(all_masks)
    elapsed = time.time() - t0

    metrics = compute_metrics(all_logits, all_masks, threshold)
    ci = bootstrap_ci(all_logits, all_masks, threshold)

    print(f"\n--- {args.split.upper()} RESULTS (threshold={threshold}) ---")
    for k, v in metrics.items():
        lo, hi = ci[k]["ci95_lo"], ci[k]["ci95_hi"]
        print(f"  {k:12s}: {v:.4f}  [95% CI: {lo:.4f}–{hi:.4f}]")
    print(f"  Inference time: {elapsed:.1f}s")

    save_prediction_figure(all_logits, all_masks, ds, threshold, tag)

    out = {
        "checkpoint": str(ckpt_path.name),
        "model": args.model,
        "split": args.split,
        "encoder": args.encoder,
        "trained_epoch": trained_epoch,
        "best_val_f1": best_val_f1,
        "threshold": threshold,
        "n_samples": len(ds),
        "metrics": metrics,
        "bootstrap_ci": ci,
        "inference_secs": round(elapsed, 1),
    }
    out_path = REPORTS_DIR / f"metrics_{tag}.json"
    write_json(out_path, out)
    print(f"\nMetrics saved → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
