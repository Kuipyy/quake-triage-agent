"""Paired comparison of the Siamese model against the early-fusion baseline on KATE-CD.

Thresholds are chosen on the VALIDATION split (F1 or F2); the comparison itself uses the
TEST split with a paired bootstrap over images (both models see the same resamples).

Usage:
    uv run python scripts/compare_models.py --threshold-mode f1
Writes reports/comparison_siamese_vs_baseline_<mode>.json
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quaketriage.config import get
from quaketriage.data.kate_cd import KateCDDataset
from quaketriage.metrics import best_threshold, compute_metrics, paired_bootstrap_diff
from quaketriage.models.loading import MODEL_TAGS, load_model
from quaketriage.provenance import write_json

ROOT = Path(__file__).parents[1]


@torch.no_grad()
def logits_and_masks(model, split: str, data_root: str, device, batch_size: int):
    loader = DataLoader(KateCDDataset(data_root, split), batch_size=batch_size, shuffle=False)
    logits, masks = [], []
    for imgs, m in loader:
        logits.append(model(imgs.to(device)).cpu())
        masks.append(m)
    return torch.cat(logits), torch.cat(masks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-mode", choices=["f1", "f2"], default="f1")
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/kate_cd"))
    ap.add_argument("--encoder", default=get("model.encoder"))
    ap.add_argument("--batch-size", type=int, default=get("training.batch_size"))
    args = ap.parse_args()
    beta = 1.0 if args.threshold_mode == "f1" else 2.0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    test_logits, thr, per_model = {}, {}, {}
    for kind, tag in MODEL_TAGS.items():
        model, _ = load_model(kind, ROOT / f"models/{tag}_best.pt", args.encoder, device)
        val_l, val_m = logits_and_masks(model, "validation", args.data_root, device,
                                        args.batch_size)
        thr[kind], _ = best_threshold(val_l, val_m, beta=beta)
        test_logits[kind], masks = logits_and_masks(model, "test", args.data_root, device,
                                                    args.batch_size)
        per_model[kind] = {"threshold": thr[kind],
                           "metrics": compute_metrics(test_logits[kind], masks, thr[kind])}
        print(f"{kind:12s} thr={thr[kind]}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in per_model[kind]["metrics"].items()))

    diffs = {k: paired_bootstrap_diff(test_logits["early_fusion"], test_logits["siamese"], masks,
                                      thr["early_fusion"], thr["siamese"], key=k)
             for k in ("f1", "iou", "precision", "recall")}
    for k, d in diffs.items():
        print(f"siamese - baseline {k:9s} {d['mean_diff']:+.3f}  "
              f"[{d['ci95_lo']:+.3f}, {d['ci95_hi']:+.3f}]  P(not better)={d['p_not_better']:.3f}")

    out = ROOT / get("paths.reports") / f"comparison_siamese_vs_baseline_{args.threshold_mode}.json"
    write_json(out,
               {"threshold_mode": args.threshold_mode, "selected_on": "validation",
                "evaluated_on": "test", "n_test": int(masks.shape[0]),
                "models": per_model, "paired_bootstrap_siamese_minus_baseline": diffs})


if __name__ == "__main__":
    main()
