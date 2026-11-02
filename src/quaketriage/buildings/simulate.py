"""Ground-truth labelling from HOTOSM and a SIMULATED detector for the Antakya demo.

Why simulated: there is no pre/post imagery pair covering Antakya, so no real detector output
exists there (docs/decisions.md). HOTOSM destroyed-building polygons serve as ground truth; the
detector's damage probability / uncertainty are drawn from documented, adjustable noise models.
Everything computed from this output measures the robustness of the PLANNING pipeline to detector
noise, never the accuracy of the damage model (that is reported on KATE-CD only).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs


class SimulatorConfig(BaseModel):
    """Noise model of the simulated detector."""

    miss_rate: float = Field(0.2, ge=0, le=1)  # P(missed | destroyed), median-size building
    false_alarm_rate: float = Field(0.03, ge=0, le=1)  # P(flagged | not destroyed)
    size_exponent: float = Field(0.0, ge=0)  # > 0: small buildings are missed more often
    seed: int = 0


def label_destroyed(
    buildings: gpd.GeoDataFrame, hotosm: gpd.GeoDataFrame, min_overlap: float = 0.5
) -> np.ndarray:
    """True where >= ``min_overlap`` of a footprint lies inside HOTOSM destroyed polygons."""
    crs = get_project_crs()
    b = buildings.to_crs(crs)
    h = hotosm.to_crs(crs)
    union_geoms = h.geometry.buffer(0).reset_index(drop=True)  # positional == label index
    pairs = gpd.sjoin(b[["geometry"]].reset_index(drop=True),
                      gpd.GeoDataFrame(geometry=union_geoms), predicate="intersects")
    overlap = np.zeros(len(b))
    area = b.geometry.area.to_numpy()
    for i, grp in pairs.groupby(pairs.index):
        cover = union_geoms.iloc[grp["index_right"].to_numpy()].union_all()
        inter = b.geometry.iloc[i].intersection(cover)
        overlap[i] = inter.area / area[i] if area[i] > 0 else 0.0
    return overlap >= min_overlap


def assemble_demo_buildings(
    osm: gpd.GeoDataFrame, hotosm: gpd.GeoDataFrame, zone_buffer_m: float = 100.0,
    min_overlap: float = 0.5,
) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Demo building set: HOTOSM destroyed polygons + nearby OSM buildings (not destroyed).

    The present-day OSM snapshot lacks most buildings that HOTOSM lists as destroyed (only a few
    dozen overlap), so the destroyed footprints are taken from HOTOSM itself. OSM buildings that
    overlap a HOTOSM polygon by >= ``min_overlap`` are dropped as duplicates; remaining OSM
    buildings whose centroid lies within ``zone_buffer_m`` of a HOTOSM polygon are kept as
    "not destroyed" (HOTOSM is incomplete, so this label is noisy). Returns (buildings in the
    project CRS, boolean destroyed array).
    """
    crs = get_project_crs()
    hot = gpd.GeoDataFrame(geometry=hotosm.to_crs(crs).geometry.buffer(0).reset_index(drop=True),
                           crs=crs)
    o = osm.to_crs(crs).reset_index(drop=True)
    duplicate = label_destroyed(o, hot, min_overlap)
    zone = hot.union_all().buffer(zone_buffer_m)
    keep = o.geometry.centroid.within(zone).to_numpy() & ~duplicate
    others = o[keep].copy()
    destroyed_polys = hot.copy()
    destroyed_polys["building"] = "yes"
    destroyed_polys["source"] = "HOTOSM destroyed-building footprint"
    out = gpd.GeoDataFrame(
        pd.concat([others, destroyed_polys], ignore_index=True), crs=crs)
    destroyed = np.r_[np.zeros(len(others), bool), np.ones(len(destroyed_polys), bool)]
    return out, destroyed


def simulate_detector(
    buildings: gpd.GeoDataFrame, destroyed: np.ndarray, cfg: SimulatorConfig
) -> gpd.GeoDataFrame:
    """Add damage_prob, uncertainty, damaged (+ ground-truth column ``destroyed_truth``).

    ``damaged`` is the detector decision, so miss/false-alarm rates hold exactly. Probabilities are
    consistent with it: detected -> thr + (1-thr)*Beta(2,2) (above the threshold), not detected ->
    thr*Beta(2,5) (below). Uncertainty peaks near the decision threshold (ASSUMED shape).
    """
    rng = np.random.default_rng(cfg.seed)
    out = buildings.copy()
    n = len(out)
    area = out.to_crs(get_project_crs()).geometry.area.to_numpy()
    scale = (np.median(area) / np.maximum(area, 1.0)) ** cfg.size_exponent
    miss = np.clip(cfg.miss_rate * scale, 0.0, 1.0)

    detected = np.where(destroyed, rng.random(n) >= miss, rng.random(n) < cfg.false_alarm_rate)
    thr = get("thresholds.damage_prob")
    prob = np.where(detected, thr + (1 - thr) * rng.beta(2, 2, n), thr * rng.beta(2, 5, n))
    unc = np.clip(0.02 + 0.3 * np.exp(-np.abs(prob - thr) / 0.1) * rng.uniform(0.5, 1.5, n), 0, 1)

    out["destroyed_truth"] = destroyed
    out["damage_prob"] = prob
    out["uncertainty"] = unc
    out["damaged"] = detected
    return out


def building_metrics(buildings: gpd.GeoDataFrame, persons_col: str = "estimated_persons") -> dict:
    """Count- and person-weighted precision/recall of ``damaged`` against ``destroyed_truth``."""
    truth = buildings["destroyed_truth"].to_numpy(bool)
    pred = buildings["damaged"].to_numpy(bool)
    tp, fp, fn = (truth & pred).sum(), (~truth & pred).sum(), (truth & ~pred).sum()
    out = {"n": int(len(truth)), "n_truth": int(truth.sum()), "tp": int(tp), "fp": int(fp),
           "fn": int(fn),
           "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
           "recall": float(tp / (tp + fn)) if tp + fn else float("nan")}
    out["f1"] = (2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
                 if tp else 0.0)
    if persons_col in buildings:
        p = buildings[persons_col].to_numpy(float)
        found, total = p[truth & pred].sum(), p[truth].sum()
        out["persons_in_destroyed"] = float(total)
        out["person_recall"] = float(found / total) if total > 0 else float("nan")
    return out
