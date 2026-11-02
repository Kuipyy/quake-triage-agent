"""Verification-queue rule: which buildings need a human/field check before any dispatch decision.

A building is flagged when its uncertainty exceeds ``thresholds.uncertainty_high``, when it has
no (or too little) valid imagery, when its uncertainty is unknown, or when the imagery quality
guard warned about it (column ``guard_warning``, optional).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from pydantic import BaseModel, Field

from quaketriage.config import get


class QueueInput(BaseModel):
    """Input schema of the verification-queue tool."""

    tau: float = Field(default_factory=lambda: get("thresholds.uncertainty_high"), ge=0)


def flag_verification(buildings: gpd.GeoDataFrame, tau: float | None = None) -> gpd.GeoDataFrame:
    """Add boolean ``needs_verification`` and a short text ``verification_reason``."""
    tau = get("thresholds.uncertainty_high") if tau is None else tau
    out = buildings.copy()
    n = len(out)
    no_pixels = (out["no_valid_pixels"].to_numpy(bool) if "no_valid_pixels" in out
                 else np.zeros(n, bool))
    unc = out["uncertainty"].to_numpy(float) if "uncertainty" in out else np.full(n, np.nan)
    guard = out["guard_warning"].to_numpy(bool) if "guard_warning" in out else np.zeros(n, bool)

    unknown = ~no_pixels & np.isnan(unc)
    high = ~np.isnan(unc) & (unc > tau)
    out["needs_verification"] = no_pixels | unknown | high | guard

    reasons = np.full(n, "", dtype=object)
    for mask, text in ((guard, "image-quality warning"), (high, f"uncertainty > {tau:g}"),
                       (unknown, "uncertainty unknown"), (no_pixels, "no valid imagery")):
        reasons = np.where(mask, text, reasons)  # later entries (stronger) overwrite earlier
    out["verification_reason"] = reasons
    return out
