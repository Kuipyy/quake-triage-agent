"""Train the early-fusion U-Net baseline on KATE-CD.

Usage:
    uv run python scripts/train.py                    # full training
    uv run python scripts/train.py --max-epochs 1     # smoke / timing test
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quaketriage.config import get
from quaketriage.data.kate_cd import KateCDDataset
from quaketriage.models.unet import DiceFocalLoss, build_siamese_unet, build_unet
from quaketriage.provenance import write_json

ROOT = Path(__file__).parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float):
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT / "data/raw/kate_cd"))
    parser.add_argument("--max-epochs", type=int, default=get("training.max_epochs"))
    parser.add_argument("--batch-size", type=int, default=get("training.batch_size"))
    parser.add_argument("--lr", type=float, default=get("training.learning_rate"))
    parser.add_argument("--patience", type=int, default=get("training.early_stopping_patience"))
    parser.add_argument("--threshold", type=float, default=get("thresholds.damage_prob"))
    parser.add_argument("--encoder", default=get("model.encoder"))
    parser.add_argument("--model", choices=["early_fusion", "siamese"], default="early_fusion")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = KateCDDataset(args.data_root, "train")
    val_ds = KateCDDataset(args.data_root, "validation")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.model == "siamese":
        model = build_siamese_unet(encoder=args.encoder).to(device)
        ckpt_name = "siamese_best.pt"
    else:
        model = build_unet(encoder=args.encoder).to(device)
        ckpt_name = "baseline_best.pt"
    loss_fn = DiceFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    patience_left = args.patience
    history = []

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        epoch_secs = time.time() - t0

        # Validation
        model.eval()
        val_loss = 0.0
        all_logits, all_masks = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                val_loss += loss_fn(logits, masks).item()
                all_logits.append(logits.cpu())
                all_masks.append(masks.cpu())
        val_loss /= len(val_loader)
        val_logits = torch.cat(all_logits)
        val_masks = torch.cat(all_masks)
        metrics = compute_metrics(val_logits, val_masks, args.threshold)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "epoch_secs": round(epoch_secs, 1),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:3d}/{args.max_epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"F1={metrics['f1']:.4f}  IoU={metrics['iou']:.4f}  "
            f"({epoch_secs:.0f}s)"
        )

        if epoch == 1:
            print(f"\n>>> Epoch time: {epoch_secs:.0f}s. ", end="")
            if epoch_secs > 180:
                print("SLOW (>3 min) — switch to Kaggle GPU.\n")
            else:
                print("Fast enough for local training.\n")

        # Save best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_left = args.patience
            ckpt = MODELS_DIR / ckpt_name
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "best_f1": best_f1, "threshold": args.threshold}, ckpt)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nBest val F1: {best_f1:.4f}  checkpoint → {MODELS_DIR / ckpt_name}")
    write_json(ROOT / "outputs" / f"training_history_{args.model}.json", {"history": history})


if __name__ == "__main__":
    main()
