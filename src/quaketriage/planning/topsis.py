"""TOPSIS prioritisation (+ optional AHP weights) for the damaged-building table.

Criteria, directions and default weights come from ``configs/default.yaml`` (``topsis``).
``distance_to_depot`` is computed here (metres, in the project CRS) from the depot points.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs

# criterion name in the config -> column in the building table
CRITERION_COLUMNS = {"damage_prob": "damage_prob", "estimated_persons": "estimated_persons",
                     "floor_count": "floors"}

# Saaty random-consistency index by matrix size
_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def topsis_scores(matrix: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Closeness coefficient in [0, 1] (1 = best) for an (alternatives x criteria) matrix."""
    x = np.asarray(matrix, float)
    w = np.asarray(weights, float)
    if x.ndim != 2 or x.shape[1] != len(w) or len(w) != len(benefit):
        raise ValueError("matrix, weights and benefit flags must agree on the criteria count")
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    if not np.isfinite(x).all():
        raise ValueError("criteria matrix contains NaN/inf")
    w = w / w.sum()
    norm = np.linalg.norm(x, axis=0)
    norm[norm == 0] = 1.0
    v = x / norm * w
    ideal = np.where(benefit, v.max(axis=0), v.min(axis=0))
    anti = np.where(benefit, v.min(axis=0), v.max(axis=0))
    d_pos = np.linalg.norm(v - ideal, axis=1)
    d_neg = np.linalg.norm(v - anti, axis=1)
    denom = d_pos + d_neg
    return np.divide(d_neg, denom, out=np.full(len(x), 0.5), where=denom > 0)


def ahp_weights(pairwise: list[list[float]]) -> tuple[np.ndarray, float]:
    """Priority vector (principal eigenvector) and consistency ratio of an AHP matrix.

    ``pairwise[i][j]`` = how much more important criterion i is than j (Saaty 1-9, reciprocal).
    A consistency ratio above 0.1 means the judgements contradict each other.
    """
    a = np.asarray(pairwise, float)
    n = a.shape[0]
    if a.ndim != 2 or a.shape[1] != n or n < 2:
        raise ValueError("pairwise matrix must be square with at least 2 criteria")
    if (a <= 0).any() or not np.allclose(a, 1.0 / a.T, rtol=1e-6):
        raise ValueError("pairwise matrix must be positive and reciprocal (a_ij = 1/a_ji)")
    vals, vecs = np.linalg.eig(a)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    ci = (vals[k].real - n) / (n - 1)
    ri = _RI.get(n, 1.45)
    return w, (ci / ri if ri > 0 else 0.0)


class TopsisInput(BaseModel):
    """Input schema of the prioritisation tool."""

    depots: list[tuple[float, float]] = Field(
        default_factory=lambda: [tuple(d) for d in get("routing.depots")], min_length=1,
        description="Depot points as (lon, lat) in WGS84; default routing.depots from config.")
    weights: dict[str, float] | None = Field(
        default=None, description="Override config weights (same keys as topsis.weights).")
    ahp_pairwise: list[list[float]] | None = Field(
        default=None, description="AHP matrix in the order of topsis.weights keys; "
        "if given, replaces the weights.")

    @field_validator("depots")
    @classmethod
    def _valid_lonlat(cls, v):
        for lon, lat in v:
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f"depot ({lon}, {lat}) is not a valid lon/lat pair")
        return v


def distance_to_depot(buildings: gpd.GeoDataFrame, depots: list[tuple[float, float]]) -> np.ndarray:
    """Straight-line distance (m) from each building centroid to its nearest depot."""
    crs = get_project_crs()
    pts = gpd.GeoSeries(gpd.points_from_xy([d[0] for d in depots], [d[1] for d in depots]),
                        crs=4326).to_crs(crs)
    cen = buildings.to_crs(crs).geometry.centroid
    dists = np.column_stack([cen.distance(p).to_numpy() for p in pts])
    return dists.min(axis=1)


def prioritise(buildings: gpd.GeoDataFrame, params: TopsisInput) -> gpd.GeoDataFrame:
    """Add ``distance_to_depot_m`` and ``priority`` (TOPSIS closeness, higher = more urgent)."""
    cfg_w = dict(get("topsis.weights"))
    types = dict(get("topsis.criteria_types"))
    names = list(cfg_w)
    if params.ahp_pairwise is not None:
        w, cr = ahp_weights(params.ahp_pairwise)
        if len(w) != len(names):
            raise ValueError(f"AHP matrix must be {len(names)}x{len(names)} for {names}")
        if cr > 0.1:
            raise ValueError(f"AHP matrix is inconsistent (CR={cr:.2f} > 0.10)")
        weights = dict(zip(names, w, strict=True))
    else:
        weights = {**cfg_w, **(params.weights or {})}
        unknown = set(weights) - set(names)
        if unknown:
            raise ValueError(f"unknown criteria: {sorted(unknown)}")

    out = buildings.copy()
    out["distance_to_depot_m"] = distance_to_depot(out, params.depots)
    cols = {**CRITERION_COLUMNS, "distance_to_depot": "distance_to_depot_m"}
    missing = [c for n in names if (c := cols[n]) not in out]
    if missing:
        raise ValueError(f"building table lacks columns: {missing}")
    matrix = out[[cols[n] for n in names]].to_numpy(float)
    benefit = np.array([types[n] == "benefit" for n in names])
    out["priority"] = topsis_scores(matrix, np.array([weights[n] for n in names]), benefit)
    return out


def rank_table(buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """Buildings sorted by priority, for the plan JSON / agent."""
    cols = [c for c in ("id", "priority", "damage_prob", "estimated_persons", "floors",
                        "distance_to_depot_m", "needs_verification") if c in buildings]
    return buildings.sort_values("priority", ascending=False)[cols].reset_index(drop=True)
