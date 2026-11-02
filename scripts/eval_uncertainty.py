"""Compare MC Dropout vs flip/rot90 TTA: calibration (ECE), reliability diagram, error-awareness.

Usage:
    uv run python scripts/eval_uncertainty.py --model early_fusion --split validation
Writes reports/uncertainty_<tag>_<split>.json and reports/figures/reliability_<tag>_<split>.png.
"""
import argparse
from pathlib import Path

import matplotlib
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quaketriage.config import get  # noqa: E402
from quaketriage.data.kate_cd import KateCDDataset  # noqa: E402
from quaketriage.metrics import metrics_from_probs  # noqa: E402
from quaketriage.models.unet import build_siamese_unet, build_unet  # noqa: E402
from quaketriage.provenance import write_json  # noqa: E402
from quaketriage.uncertainty import (  # noqa: E402
    DecoderDropout,
    predict_mc_dropout,
    predict_tta,
    reliability_bins,
    uncertainty_error_ratio,
)

ROOT = Path(__file__).parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["early_fusion", "siamese"], default="early_fusion")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/kate_cd"))
    ap.add_argument("--encoder", default=get("model.encoder"))
    ap.add_argument("--T", type=int, default=get("model.mc_dropout_samples"))
    ap.add_argument("--p", type=float, default=get("model.mc_dropout_p"))
    args = ap.parse_args()

    tag = "baseline" if args.model == "early_fusion" else "siamese"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else ROOT / f"models/{tag}_best.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    thr = ckpt.get("threshold", get("thresholds.damage_prob"))
    builder = build_unet if args.model == "early_fusion" else build_siamese_unet
    model = builder(encoder=args.encoder, encoder_weights=None).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dropout = DecoderDropout(model, p=args.p)

    loader = DataLoader(KateCDDataset(args.data_root, args.split), batch_size=4, shuffle=False)
    acc = {k: ([], [], []) for k in ("single", "mc_dropout", "tta")}  # mean, std, ...
    targets = []
    for imgs, masks in loader:
        x = imgs.to(device)
        targets.append(masks)
        with torch.no_grad():
            single = torch.sigmoid(model(x))
        mc = predict_mc_dropout(model, x, dropout, args.T)
        tta = predict_tta(model, x)
        for name, mean, std in (("single", single, torch.zeros_like(single)),
                                ("mc_dropout", mc.mean(0), mc.std(0)),
                                ("tta", tta.mean(0), tta.std(0))):
            acc[name][0].append(mean.cpu())
            acc[name][1].append(std.cpu())
    y = torch.cat(targets)

    results, fig_data = {}, {}
    for name, (means, stds, _) in acc.items():
        mean, std = torch.cat(means), torch.cat(stds)
        rb = reliability_bins(mean, y)
        results[name] = {
            "f1": round(metrics_from_probs(mean, y, thr)["f1"], 4),
            "ece": round(rb["ece"], 5),
            "brier": round(float(((mean - y) ** 2).mean()), 5),
            "uncertainty_error_ratio": (None if name == "single" else round(
                uncertainty_error_ratio(std, mean, y, thr), 3)),
            "mean_uncertainty": None if name == "single" else round(float(std.mean()), 5),
        }
        fig_data[name] = rb
        print(f"{name:11s} F1={results[name]['f1']:.4f}  ECE={results[name]['ece']:.5f}  "
              f"Brier={results[name]['brier']:.5f}  "
              f"unc/err ratio={results[name]['uncertainty_error_ratio']}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, rb in fig_data.items():
        pts = [(c, r) for c, r, n in zip(rb["mean_conf"], rb["pos_rate"], rb["count"]) if n > 0]
        ax.plot(*zip(*pts), marker="o", ms=4, label=f"{name} (ECE={results[name]['ece']:.4f})")
    ax.set_xlabel("mean predicted damage probability")
    ax.set_ylabel("observed damaged fraction")
    ax.set_title(f"Reliability — {tag}, {args.split}")
    ax.legend(fontsize=8)
    figs = ROOT / get("paths.reports") / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"reliability_{tag}_{args.split}.png", dpi=100, bbox_inches="tight")

    write_json(ROOT / get("paths.reports") / f"uncertainty_{tag}_{args.split}.json", {
        "model": args.model, "checkpoint": ckpt_path.name, "split": args.split,
        "threshold": thr, "T": args.T, "dropout_p": args.p, "n": len(y), "results": results,
        "note": "MC dropout applied at inference only (checkpoint trained without dropout).",
    })


if __name__ == "__main__":
    main()
