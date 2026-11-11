"""Disk cache for solved route plans.

The OR-Tools solve takes ~30 s, so an identical question (same constraints, same priority
table) must not be solved twice. The key is a hash of the solver parameters, the closed areas, the
depot and a digest of the candidate table (row index, priority, damage probability, persons);
change any of them and the key changes. Plans are plain JSON (the dict ``plan_routes`` returns).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np

_COLUMNS = ("priority", "damage_prob", "estimated_persons")


def table_digest(buildings: gpd.GeoDataFrame) -> str:
    """Short hash of the numbers that decide the plan (not the geometry)."""
    h = hashlib.sha256()
    h.update(np.asarray(buildings.index, dtype=np.int64).tobytes())
    for col in _COLUMNS:
        if col in buildings:
            h.update(np.round(buildings[col].to_numpy(float), 6).tobytes())
    return h.hexdigest()[:16]


def plan_key(params: dict, closed_areas: list, depot: tuple[float, float], digest: str) -> str:
    payload = json.dumps({"params": params, "closed": closed_areas, "depot": list(depot),
                          "table": digest}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def load_plan(cache_dir: Path | None, key: str) -> dict | None:
    if cache_dir is None:
        return None
    path = Path(cache_dir) / f"{key}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def save_plan(cache_dir: Path | None, key: str, stored: dict) -> None:
    if cache_dir is None:
        return
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / f"{key}.json").write_text(json.dumps(stored))
    except OSError:
        pass  # a cache write must never break a turn
