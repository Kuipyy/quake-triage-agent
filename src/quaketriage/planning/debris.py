"""Debris risk on the road graph.

Road segments within ``routing.debris_risk_buffer_meters`` of a heavily damaged, multi-storey
building are penalised (travel time x ``debris_penalty_factor``); segments near
``debris_block_count`` or more such buildings are treated as closed. Parameters are assumptions
from ``configs/default.yaml``, not calibrated values (see docs/decisions.md).
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from pydantic import BaseModel, Field

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs


class DebrisInput(BaseModel):
    """Input schema of the debris-risk tool (defaults from config)."""

    buffer_m: float = Field(default_factory=lambda: get("routing.debris_risk_buffer_meters"), ge=0)
    damage_prob_min: float = Field(default_factory=lambda: get("routing.debris_damage_prob_min"),
                                   ge=0, le=1)
    min_floors: int = Field(default_factory=lambda: get("routing.debris_min_floors"), ge=1)
    penalty_factor: float = Field(default_factory=lambda: get("routing.debris_penalty_factor"),
                                  ge=1)
    block_count: int = Field(default_factory=lambda: get("routing.debris_block_count"), ge=1)


def debris_sources(buildings: gpd.GeoDataFrame, p: DebrisInput) -> gpd.GeoDataFrame:
    """Buildings likely to shed debris onto nearby streets (projected CRS)."""
    b = buildings.to_crs(get_project_crs())
    mask = (b["damage_prob"] > p.damage_prob_min) & (b["floors"] >= p.min_floors)
    return b[mask]


def apply_debris_risk(g: nx.MultiDiGraph, buildings: gpd.GeoDataFrame,
                      params: DebrisInput | None = None) -> nx.MultiDiGraph:
    """Return a copy of projected graph ``g`` with edge attributes ``debris_sources`` (count),
    ``blocked`` (bool) and ``travel_time_adj`` (s, inf if blocked).

    Edges need ``travel_time``; it is added from road-type speeds if missing.
    """
    p = params or DebrisInput()
    g = g.copy()
    if not all("travel_time" in d for _, _, d in g.edges(data=True)):
        g = ox.routing.add_edge_travel_times(ox.routing.add_edge_speeds(g))
    edges = ox.convert.graph_to_gdfs(g, nodes=False)[["geometry", "travel_time"]]
    src = debris_sources(buildings, p)
    count = np.zeros(len(edges), int)
    if len(src):
        zones = gpd.GeoDataFrame(geometry=src.geometry.buffer(p.buffer_m), crs=src.crs)
        hits = gpd.sjoin(edges.reset_index(drop=True)[["geometry"]], zones, predicate="intersects")
        counts = hits.groupby(level=0).size()
        count[counts.index.to_numpy()] = counts.to_numpy()
    blocked = count >= p.block_count
    factor = np.where(count > 0, p.penalty_factor, 1.0)
    adj = np.where(blocked, np.inf, edges["travel_time"].to_numpy(float) * factor)
    keys = list(edges.index)
    nx.set_edge_attributes(g, dict(zip(keys, count.tolist(), strict=True)), "debris_sources")
    nx.set_edge_attributes(g, dict(zip(keys, blocked.tolist(), strict=True)), "blocked")
    nx.set_edge_attributes(g, dict(zip(keys, adj.tolist(), strict=True)), "travel_time_adj")
    return g
