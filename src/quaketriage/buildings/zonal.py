"""Per-building zonal statistics from pixel damage-probability / uncertainty rasters.

Agent tool ``zonal_building_stats``: plain function + pydantic input schema.
Pixels that are NaN (e.g. tiles rejected by the quality guard) are ignored; a building whose
valid-pixel fraction is below ``min_valid_fraction`` gets NaN statistics and
``no_valid_pixels = True`` so the verification queue can pick it up.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pydantic import BaseModel, Field
from rasterio.features import rasterize

from quaketriage.config import get


class ZonalInput(BaseModel):
    """Input schema of the zonal-statistics tool."""

    buildings_path: Path
    prob_raster: Path
    uncertainty_raster: Path | None = None
    threshold: float = Field(default_factory=lambda: get("thresholds.damage_prob"), ge=0, le=1)
    min_valid_fraction: float = Field(default=0.5, ge=0, le=1)


def zonal_building_stats(
    buildings: gpd.GeoDataFrame,
    prob_raster: Path,
    uncertainty_raster: Path | None = None,
    threshold: float | None = None,
    min_valid_fraction: float = 0.5,
) -> gpd.GeoDataFrame:
    """Add damage_prob, uncertainty, damage_frac, damaged, valid_fraction, no_valid_pixels.

    ``buildings`` may be in any CRS; it is reprojected to the raster CRS for sampling and the
    result is returned in the original CRS, same row order.
    """
    thr = get("thresholds.damage_prob") if threshold is None else threshold
    out = buildings.copy()
    n = len(out)

    with rasterio.open(prob_raster) as src:
        geoms = out.to_crs(src.crs).geometry
        # label raster: building index + 1 (0 = background); later buildings win on overlap
        shapes = [(g, i + 1) for i, g in enumerate(geoms) if g is not None and not g.is_empty]
        labels = rasterize(shapes, out_shape=src.shape, transform=src.transform, fill=0,
                           dtype="int32")
        prob = src.read(1).astype("float64")
        if src.nodata is not None and not np.isnan(src.nodata):
            prob[prob == src.nodata] = np.nan
        unc = None
        if uncertainty_raster is not None:
            with rasterio.open(uncertainty_raster) as u:
                if u.shape != src.shape or u.transform != src.transform:
                    raise ValueError("uncertainty raster must share the probability raster grid")
                unc = u.read(1).astype("float64")

    flat = labels.ravel()
    k = n + 1
    total = np.bincount(flat, minlength=k)[1:].astype("float64")
    valid_mask = ~np.isnan(prob).ravel()
    valid = np.bincount(flat[valid_mask], minlength=k)[1:].astype("float64")
    psum = np.bincount(flat[valid_mask], weights=prob.ravel()[valid_mask], minlength=k)[1:]
    over = np.bincount(flat[valid_mask], weights=(prob.ravel()[valid_mask] > thr).astype(float),
                       minlength=k)[1:]

    with np.errstate(invalid="ignore", divide="ignore"):
        valid_fraction = np.where(total > 0, valid / total, 0.0)
        ok = (valid > 0) & (valid_fraction >= min_valid_fraction)
        out["damage_prob"] = np.where(ok, psum / valid, np.nan)
        out["damage_frac"] = np.where(ok, over / valid, np.nan)
        if unc is not None:
            um = ~np.isnan(unc).ravel() & valid_mask
            usum = np.bincount(flat[um], weights=unc.ravel()[um], minlength=k)[1:]
            ucnt = np.bincount(flat[um], minlength=k)[1:].astype("float64")
            out["uncertainty"] = np.where(ok & (ucnt > 0), usum / ucnt, np.nan)
    out["damaged"] = np.where(ok, out["damage_prob"] > thr, False).astype(bool)
    out["valid_fraction"] = valid_fraction
    out["no_valid_pixels"] = ~ok
    return out


def run(inp: ZonalInput) -> gpd.GeoDataFrame:
    """Tool entry point: load buildings from disk and compute statistics."""
    return zonal_building_stats(
        gpd.read_file(inp.buildings_path), inp.prob_raster, inp.uncertainty_raster,
        inp.threshold, inp.min_valid_fraction,
    )
