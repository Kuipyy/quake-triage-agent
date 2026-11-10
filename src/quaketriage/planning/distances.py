"""Shortest-path travel-time matrix between graph nodes, computed once and cached on disk."""
from __future__ import annotations

import hashlib
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from quaketriage.config import get
from quaketriage.planning.roads import ROOT


def _csr(g: nx.MultiDiGraph, weight: str) -> tuple[csr_matrix, dict]:
    """Sparse adjacency with the cheapest parallel edge; closed (inf) edges are dropped."""
    idx = {n: i for i, n in enumerate(g.nodes)}
    best: dict[tuple[int, int], float] = {}
    for u, v, d in g.edges(data=True):
        w = d.get(weight, np.inf)
        if np.isfinite(w) and w < best.get((idx[u], idx[v]), np.inf):
            best[(idx[u], idx[v])] = max(float(w), 1e-6)  # csgraph treats 0 as "no edge"
    rows, cols = zip(*best) if best else ((), ())
    n = len(idx)
    return csr_matrix((list(best.values()), (rows, cols)), shape=(n, n)), idx


def travel_time_matrix(g: nx.MultiDiGraph, nodes: list, weight: str = "travel_time_adj",
                       use_cache: bool = True) -> np.ndarray:
    """(len(nodes) x len(nodes)) seconds along shortest paths; ``inf`` where unreachable.

    The result is cached under ``routing.distance_cache_dir`` keyed by the node list and the
    graph's edge weights, so a changed debris rule recomputes automatically.
    """
    graph_csr, idx = _csr(g, weight)
    key = hashlib.sha1(
        np.asarray(nodes, dtype="int64").tobytes() + graph_csr.data.tobytes()
        + graph_csr.indices.tobytes()).hexdigest()[:16]
    path = Path(ROOT / get("routing.distance_cache_dir")) / f"tt_{key}.npy"
    if use_cache and path.exists():
        return np.load(path)
    src = [idx[n] for n in nodes]
    full = dijkstra(graph_csr, directed=True, indices=src)
    mat = full[:, src]
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mat)
    return mat
