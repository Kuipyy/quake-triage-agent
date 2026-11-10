"""Prize-collecting multi-team routing with OR-Tools.

Every candidate building is an optional node: skipping it costs its priority score, so the solver
maximises the priority collected within the mission time limit (travel time only breaks ties).
Teams start and end at one depot; each visit takes ``routing.intervention_time_minutes``.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from pydantic import BaseModel, Field

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs
from quaketriage.planning.distances import travel_time_matrix
from quaketriage.planning.roads import snap_to_nodes

PRIZE_SCALE = 1_000_000  # priority in [0,1] -> integer penalty; seconds of travel stay far below


class VrpInput(BaseModel):
    """Input schema of the route-planning tool (defaults from config)."""

    candidate_n: int = Field(default_factory=lambda: get("routing.candidate_n"), ge=1)
    num_teams: int = Field(default_factory=lambda: get("routing.num_teams"), ge=1)
    mission_hours: float = Field(default_factory=lambda: get("routing.mission_duration_hours"),
                                 gt=0)
    intervention_minutes: float = Field(
        default_factory=lambda: get("routing.intervention_time_minutes"), ge=0)
    time_limit_s: int = Field(default_factory=lambda: get("routing.solver_time_limit_s"), ge=1)
    depot_index: int = Field(default=0, ge=0)


def select_candidates(buildings: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    """Top-n buildings by TOPSIS ``priority`` (nested: top-200 is a subset of top-500)."""
    return buildings.sort_values("priority", ascending=False, kind="stable").head(n)


def building_nodes(g: nx.MultiDiGraph, buildings: gpd.GeoDataFrame) -> np.ndarray:
    """Nearest graph node of each building centroid."""
    c = buildings.to_crs(get_project_crs()).geometry.centroid.to_crs(4326)
    nodes, _ = snap_to_nodes(g, list(zip(c.x, c.y, strict=True)))
    return nodes


def plan_routes(g: nx.MultiDiGraph, buildings: gpd.GeoDataFrame, depot_lonlat: tuple[float, float],
                params: VrpInput | None = None, matrix: np.ndarray | None = None,
                matrix_nodes: list | None = None) -> dict:
    """Solve and return a plan dict. ``buildings`` must carry ``priority``.

    ``matrix``/``matrix_nodes`` let callers reuse one precomputed travel-time matrix
    (nodes = depot first, then the candidates) for several solves.
    """
    p = params or VrpInput()
    cand = select_candidates(buildings, p.candidate_n).copy()
    depot_node = int(snap_to_nodes(g, [depot_lonlat])[0][0])
    cand["node"] = building_nodes(g, cand)
    nodes = [depot_node, *cand["node"].tolist()]
    if matrix is None:
        matrix = travel_time_matrix(g, nodes)
    else:  # subset of a larger cached matrix
        pos = {n: i for i, n in enumerate(matrix_nodes)}
        sel = [pos[n] for n in nodes]
        matrix = matrix[np.ix_(sel, sel)]

    # buildings that cannot be reached from, or returned to, the depot are excluded
    ok = np.isfinite(matrix[0]) & np.isfinite(matrix[:, 0])
    ok[0] = True
    unreachable = int((~ok).sum())
    keep = np.flatnonzero(ok)
    cand = cand.iloc[keep[1:] - 1]
    mat = matrix[np.ix_(keep, keep)]
    n = len(keep)

    service = int(p.intervention_minutes * 60)
    horizon = int(p.mission_hours * 3600)
    # travel + service at the departure node; unreachable pairs get a prohibitive time
    big = horizon * 10
    trans = np.where(np.isfinite(mat), mat, big).round().astype(np.int64)
    trans = trans + np.where(np.arange(n)[:, None] == 0, 0, service)
    np.fill_diagonal(trans, 0)

    mgr = pywrapcp.RoutingIndexManager(n, p.num_teams, 0)
    routing = pywrapcp.RoutingModel(mgr)
    cb = routing.RegisterTransitMatrix(trans.tolist())
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.AddDimension(cb, 0, horizon, True, "time")
    prizes = (cand["priority"].to_numpy(float) * PRIZE_SCALE).round().astype(np.int64)
    for i in range(1, n):
        routing.AddDisjunction([mgr.NodeToIndex(i)], int(prizes[i - 1]))
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.time_limit.seconds = p.time_limit_s
    sol = routing.SolveWithParameters(sp)

    routes, visited = [], []
    if sol is not None:
        time_dim = routing.GetDimensionOrDie("time")
        for v in range(p.num_teams):
            idx, stops = routing.Start(v), []
            while not routing.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node != 0:
                    r = cand.iloc[node - 1]
                    stops.append({"building_id": (int(r["id"]) if "id" in r and np.isfinite(r["id"])
                                                  else None),
                                  "row": int(cand.index[node - 1]),
                                  "arrival_s": int(sol.Value(time_dim.CumulVar(idx))),
                                  "priority": round(float(r["priority"]), 4),
                                  "estimated_persons": round(float(r["estimated_persons"]), 1),
                                  "node": int(r["node"])})
                    visited.append(node - 1)
                idx = sol.Value(routing.NextVar(idx))
            routes.append({"team": v, "stops": stops,
                           "return_s": int(sol.Value(time_dim.CumulVar(idx)))})
    got = cand.iloc[visited]
    return {"SIMULATED_DETECTOR": True,
            "n_candidates": int(len(cand)), "n_unreachable": unreachable,
            "n_visited": len(visited),
            "priority_collected": round(float(got["priority"].sum()), 4),
            "persons_reached": round(float(got["estimated_persons"].sum()), 1),
            "persons_in_candidates": round(float(cand["estimated_persons"].sum()), 1),
            "solved": sol is not None, "routes": routes}
