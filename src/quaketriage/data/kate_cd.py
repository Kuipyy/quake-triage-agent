"""KATE-CD dataset — Parquet → 6-channel (pre+post) tensor + binary mask."""
import io
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

# ImageNet mean/std (per channel, applied to both pre and post)
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _decode_rgb(cell) -> np.ndarray:
    """Decode HF image cell (bytes dict or raw bytes) → HxWx3 uint8."""
    raw = cell.get("bytes") if isinstance(cell, dict) else cell
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


def _decode_mask(cell) -> np.ndarray:
    """Decode mask cell → HxW uint8 (0 or 255)."""
    raw = cell.get("bytes") if isinstance(cell, dict) else cell
    return np.array(Image.open(io.BytesIO(raw)).convert("L"))


def build_transforms(split: str) -> A.Compose:
    """Albumentations pipeline. Geometric transforms apply to all three images jointly."""
    if split == "train":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    else:
        aug = []

    return A.Compose(
        aug + [ToTensorV2()],
        additional_targets={"post": "image", "mask": "mask"},
        is_check_shapes=False,
    )


def _normalize(t: torch.Tensor) -> torch.Tensor:
    """ImageNet normalize a CHW float32 tensor."""
    mean = torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_STD, dtype=torch.float32).view(3, 1, 1)
    return (t.float() / 255.0 - mean) / std


class KateCDDataset(Dataset):
    """KATE-CD change detection dataset.

    Returns:
        image  : FloatTensor [6, H, W]  (pre+post concatenated, ImageNet-normalised)
        mask   : FloatTensor [1, H, W]  (binary, 0/1)
    """

    def __init__(self, root: str | Path, split: str = "train") -> None:
        root = Path(root)
        parquet_path = root / SPLITS[split]
        self.df = pd.read_parquet(parquet_path)
        self.transforms = build_transforms(split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        pre = _decode_rgb(row["pre_image"])
        post = _decode_rgb(row["post_image"])
        mask = _decode_mask(row["label"])

        out = self.transforms(image=pre, post=post, mask=mask)
        pre_t = _normalize(out["image"])       # [3, H, W]
        post_t = _normalize(out["post"])       # [3, H, W]
        mask_t = (out["mask"].unsqueeze(0) > 127).float()  # [1, H, W]

        image = torch.cat([pre_t, post_t], dim=0)  # [6, H, W]
        return image, mask_t
