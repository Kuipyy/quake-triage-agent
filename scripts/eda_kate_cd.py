"""EDA for KATE-CD: class balance, coordinate check, sample grid figure."""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path("data/raw/kate_cd")
REPORTS_DIR = Path("reports/figures")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {"train": "data/train-*.parquet", "validation": "data/validation-*.parquet", "test": "data/test-*.parquet"}

def load_split(glob_pattern: str) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob(glob_pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def decode_image(cell) -> np.ndarray:
    """Decode bytes or dict-with-bytes image cell to HxWxC uint8 array."""
    if isinstance(cell, dict):
        raw = cell.get("bytes") or cell.get("path")
        if isinstance(raw, bytes):
            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    if isinstance(cell, bytes):
        return np.array(Image.open(io.BytesIO(cell)).convert("RGB"))
    return np.array(cell)


def decode_mask(cell) -> np.ndarray:
    """Decode mask cell to HxW uint8 array."""
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if isinstance(raw, bytes):
            img = Image.open(io.BytesIO(raw)).convert("L")
            return np.array(img)
    if isinstance(cell, bytes):
        img = Image.open(io.BytesIO(cell)).convert("L")
        return np.array(img)
    return np.array(cell)


def main():
    splits = {}
    for name, pattern in SPLITS.items():
        df = load_split(pattern)
        splits[name] = df
        print(f"{name}: {len(df)} samples | columns: {list(df.columns)}")

    # Check for coordinate / city columns
    all_cols = set()
    for df in splits.values():
        all_cols.update(df.columns)
    print(f"\nAll columns across splits: {sorted(all_cols)}")
    geo_hints = [c for c in all_cols if any(k in c.lower() for k in
                 ("lat", "lon", "coord", "city", "location", "geo", "place", "region"))]
    print(f"Geo/city-related columns: {geo_hints if geo_hints else 'NONE'}")

    # Class balance
    print("\nComputing class balance (damaged pixel ratio per split)...")
    metrics = {}
    for split_name, df in splits.items():
        if df.empty:
            continue
        mask_col = [c for c in df.columns if "mask" in c.lower() or "label" in c.lower()]
        if not mask_col:
            print(f"  {split_name}: no mask column found, skipping")
            continue
        mc = mask_col[0]
        damaged_ratios = []
        for cell in df[mc]:
            mask = decode_mask(cell)
            binary = (mask > 127).astype(np.float32)
            damaged_ratios.append(binary.mean())
        arr = np.array(damaged_ratios)
        metrics[split_name] = {
            "n_samples": len(arr),
            "mean_damaged_ratio": float(arr.mean()),
            "median_damaged_ratio": float(np.median(arr)),
            "pct_with_any_damage": float((arr > 0).mean()),
        }
        print(f"  {split_name}: mean damaged px = {arr.mean():.3f}, "
              f"samples with any damage = {(arr > 0).mean()*100:.1f}%")

    # Save metrics
    metrics_path = Path("reports/metrics_eda.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # Sample grid figure (4 triplets from train: pre / post / mask)
    train_df = splits.get("train", pd.DataFrame())
    if train_df.empty:
        print("No training data for figure.")
        return

    cols = list(train_df.columns)
    pre_col  = next((c for c in cols if "pre"  in c.lower()), None)
    post_col = next((c for c in cols if "post" in c.lower()), None)
    mask_col_name = next((c for c in cols if "mask" in c.lower() or "label" in c.lower()), None)

    if not all([pre_col, post_col, mask_col_name]):
        print(f"Could not identify pre/post/mask columns: {cols}")
        return

    n = 4
    indices = np.random.default_rng(42).choice(len(train_df), size=n, replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    col_titles = ["Pre-earthquake", "Post-earthquake", "Damage mask"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10)

    for row, idx in enumerate(indices):
        pre  = decode_image(train_df.iloc[idx][pre_col])
        post = decode_image(train_df.iloc[idx][post_col])
        mask = decode_mask(train_df.iloc[idx][mask_col_name])
        axes[row, 0].imshow(pre);  axes[row, 0].axis("off")
        axes[row, 1].imshow(post); axes[row, 1].axis("off")
        axes[row, 2].imshow(mask, cmap="Reds"); axes[row, 2].axis("off")

    plt.tight_layout()
    fig_path = REPORTS_DIR / "kate_cd_samples.png"
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f"Sample grid saved → {fig_path}")


if __name__ == "__main__":
    main()
