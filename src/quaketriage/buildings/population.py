"""Per-building occupancy estimate from WorldPop cells (people per cell, pre-quake).

Each WorldPop cell's population is distributed over the buildings whose centroid lies in the
cell, proportionally to floor area (footprint x floors; floors from ``building:levels`` or a
type-based fallback). The result is multiplied by a night factor: the Feb-2023 earthquake struck
at 04:17, so homes were occupied and non-residential buildings mostly empty. All factors are
ASSUMPTIONS from ``configs/default.yaml`` (population.*), not measurements.
Population in cells that contain no building centroid is not assigned (reported as coverage).
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pydantic import BaseModel
from rasterio.windows import Window, from_bounds

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs


class PopulationInput(BaseModel):
    """Input schema of the population tool."""

    buildings_path: Path
    worldpop_raster: Path


def _types(buildings: gpd.GeoDataFrame) -> pd.Series:
    if "building" in buildings:
        return buildings["building"]
    return pd.Series("yes", index=buildings.index)


def estimate_floors(buildings: gpd.GeoDataFrame) -> pd.Series:
    """Floors from ``building:levels`` if numeric and sensible, else by type, else default."""
    by_type = get("population.floors_by_type")
    default = get("population.default_floors")
    types = _types(buildings)
    floors = types.map(by_type).fillna(default).astype(float)
    if "building:levels" in buildings:
        lv = pd.to_numeric(buildings["building:levels"], errors="coerce")
        floors = lv.where(lv.between(1, 40), floors)
    return floors


def estimate_population(buildings: gpd.GeoDataFrame, worldpop_raster: Path) -> gpd.GeoDataFrame:
    """Add floors, floor_area_m2, residents_day (cell share) and estimated_persons (night)."""
    out = buildings.copy()
    proj = out.to_crs(get_project_crs())
    out["floors"] = estimate_floors(out)
    out["floor_area_m2"] = proj.geometry.area * out["floors"]

    with rasterio.open(worldpop_raster) as src:
        cent = proj.geometry.centroid
        cent_ll = gpd.GeoSeries(cent, crs=proj.crs).to_crs(src.crs)
        w = from_bounds(*cent_ll.total_bounds, transform=src.transform)
        col0, row0 = int(np.floor(w.col_off)) - 1, int(np.floor(w.row_off)) - 1
        window = Window(col0, row0, int(np.ceil(w.width)) + 3, int(np.ceil(w.height)) + 3)
        data = src.read(1, window=window, boundless=True, fill_value=src.nodata).astype("float64")
        wtransform = src.window_transform(window)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data = np.nan_to_num(data, nan=0.0)

    rows, cols = rasterio.transform.rowcol(wtransform, cent_ll.x.to_numpy(), cent_ll.y.to_numpy())
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    cell = np.where(inside, rows * data.shape[1] + cols, -1)

    weight = out["floor_area_m2"].to_numpy(float)
    cell_weight = pd.Series(weight).groupby(cell).transform("sum").to_numpy()
    cell_pop = np.where(inside, data.ravel()[np.clip(cell, 0, None)], 0.0)
    share = np.divide(weight, cell_weight, out=np.zeros_like(weight), where=cell_weight > 0)
    out["residents_day"] = cell_pop * share

    residential = set(get("population.residential_types"))
    types = _types(out)
    factor = np.where(types.isin(residential), get("population.night_factor"),
                      get("population.non_residential_night_factor"))
    out["estimated_persons"] = out["residents_day"] * factor
    return out


def coverage(buildings: gpd.GeoDataFrame) -> dict:
    """Total distributed population (day) and night estimate, for sanity reporting."""
    return {"residents_day": float(buildings["residents_day"].sum()),
            "estimated_persons_night": float(buildings["estimated_persons"].sum())}
