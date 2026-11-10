"""Prize-collecting multi-team routing with OR-Tools.

Every candidate building is an optional node: skipping it costs its priority score, so the solver
maximises the priority collected within the mission time limit (travel time only breaks ties).
Teams start and end at one depot; each visit takes ``routing.intervention_time_minutes``.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from pydantic import BaseModel, Field

from quaketriage.config import get
from quaketriage.data.geo import get_project_crs
from quaketriage.planning.distances import travel_time_matrix, walking_access
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
    walking: bool = True  # allow a walking leg from the nearest drivable node
    walk_speed_kmh: float = Field(default_factory=lambda: get("routing.walking_speed_kmh"), gt=0)
    max_walk_m: float = Field(default_factory=lambda: get("routing.max_walk_m"), ge=0)


def select_candidates(buildings: gpd.GeoDataFrame, n: int) -> gpd.GeoDataFrame:
    """Top-n candidate buildings by TOPSIS ``priority`` (nested: top-200 is a subset of top-500).

    Buildings below the damage gate (``candidate`` is False) are never selected.
    """
    pool = buildings[buildings["candidate"]] if "candidate" in buildings else buildings
    return pool.sort_values("priority", ascending=False, kind="stable").head(n)


def building_nodes(g: nx.MultiDiGraph, buildings: gpd.GeoDataFrame) -> np.ndarray:
    """Nearest graph node of each building centroid."""
    c = buildings.to_crs(get_project_crs()).geometry.centroid.to_crs(4326)
    nodes, _ = snap_to_nodes(g, list(zip(c.x, c.y, strict=True)))
    return nodes


@dataclass
class Problem:
    """Candidate stops, integer transit-time matrix (travel + service at departure) and limits."""

    cand: gpd.GeoDataFrame  # reachable candidates; row i <-> matrix node i + 1 (node 0 = depot)
    trans: np.ndarray
    horizon: int
    num_teams: int
    n_unreachable: int
    n_need_walking: int


def prepare_problem(g: nx.MultiDiGraph, buildings: gpd.GeoDataFrame,
                    depot_lonlat: tuple[float, float], params: VrpInput | None = None,
                    matrix: np.ndarray | None = None, matrix_nodes: list | None = None) -> Problem:
    """Select candidates, resolve walking access, build the transit matrix.

    ``matrix``/``matrix_nodes`` let callers reuse one precomputed travel-time matrix
    (nodes = depot first, then stop nodes) for several solves. Shared by the VRP and baselines.
    """
    p = params or VrpInput()
    cand = select_candidates(buildings, p.candidate_n).copy()
    depot_node = int(snap_to_nodes(g, [depot_lonlat])[0][0])
    cand["node"] = building_nodes(g, cand)
    access, walk_m = walking_access(g, depot_node, cand["node"].tolist())
    limit = p.max_walk_m if p.walking else 0.0
    ok = np.array([a is not None for a in access]) & (walk_m <= limit)
    cand_ok = cand[ok].copy()
    cand_ok["stop_node"] = [a for a, k in zip(access, ok, strict=True) if k]
    cand_ok["walk_m"] = walk_m[ok]
    nodes = [depot_node, *cand_ok["stop_node"].tolist()]
    if matrix is None:
        mat = travel_time_matrix(g, nodes)
    else:  # subset of a larger cached matrix covering these nodes
        pos = {n: i for i, n in enumerate(matrix_nodes)}
        mat = matrix[np.ix_([pos[n] for n in nodes], [pos[n] for n in nodes])]
    # service at a stop = intervention + walking out and back
    walk_s = 2 * cand_ok["walk_m"].to_numpy() / (p.walk_speed_kmh / 3.6)
    service = np.concatenate([[0.0], p.intervention_minutes * 60 + walk_s]).round()
    horizon = int(p.mission_hours * 3600)
    trans = np.where(np.isfinite(mat), mat, horizon * 10).round().astype(np.int64)
    trans = trans + service.astype(np.int64)[:, None]
    np.fill_diagonal(trans, 0)
    return Problem(cand_ok, trans, horizon, p.num_teams, int((~ok).sum()),
                   int((ok & (walk_m > 0)).sum()))


def build_result(prob: Problem, routes: list[list[tuple[int, int]]], returns: list[int],
                 solved: bool = True) -> dict:
    """Plan dict from per-team lists of (matrix node, arrival_s) and return times."""
    cand = prob.cand
    out_routes, visited = [], []
    for team, (stops, ret) in enumerate(zip(routes, returns, strict=True)):
        rows = []
        for node, arrival in stops:
            r = cand.iloc[node - 1]
            rows.append({"building_id": (int(r["id"]) if "id" in r and np.isfinite(r["id"])
                                         else None),
                         "row": int(cand.index[node - 1]), "arrival_s": int(arrival),
                         "priority": round(float(r["priority"]), 4),
                         "estimated_persons": round(float(r["estimated_persons"]), 1),
                         "node": int(r["stop_node"]), "walk_m": round(float(r["walk_m"]), 0)})
            visited.append(node - 1)
        out_routes.append({"team": team, "stops": rows, "return_s": int(ret)})
    got = cand.iloc[visited]
    return {"SIMULATED_DETECTOR": True,
            "n_candidates": int(len(cand)), "n_unreachable": prob.n_unreachable,
            "n_visited": len(visited), "n_need_walking": prob.n_need_walking,
            "priority_collected": round(float(got["priority"].sum()), 4),
            "persons_reached": round(float(got["estimated_persons"].sum()), 1),
            "persons_in_candidates": round(float(cand["estimated_persons"].sum()), 1),
            "solved": solved, "routes": out_routes}


def plan_routes(g: nx.MultiDiGraph, buildings: gpd.GeoDataFrame, depot_lonlat: tuple[float, float],
                params: VrpInput | None = None, matrix: np.ndarray | None = None,
                matrix_nodes: list | None = None) -> dict:
    """Solve the prize-collecting VRP and return a plan dict. ``buildings`` needs ``priority``."""
    p = params or VrpInput()
    prob = prepare_problem(g, buildings, depot_lonlat, p, matrix, matrix_nodes)
    cand, trans = prob.cand, prob.trans
    n = len(trans)
    mgr = pywrapcp.RoutingIndexManager(n, p.num_teams, 0)
    routing = pywrapcp.RoutingModel(mgr)
    cb = routing.RegisterTransitMatrix(trans.tolist())
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    routing.AddDimension(cb, 0, prob.horizon, True, "time")
    prizes = (cand["priority"].to_numpy(float) * PRIZE_SCALE).round().astype(np.int64)
    for i in range(1, n):
        routing.AddDisjunction([mgr.NodeToIndex(i)], int(prizes[i - 1]))
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.time_limit.seconds = p.time_limit_s
    sol = routing.SolveWithParameters(sp)

    routes, returns = [], []
    if sol is not None:
        time_dim = routing.GetDimensionOrDie("time")
        for v in range(p.num_teams):
            idx, stops = routing.Start(v), []
            while not routing.IsEnd(idx):
                node = mgr.IndexToNode(idx)
                if node != 0:
                    stops.append((node, sol.Value(time_dim.CumulVar(idx))))
                idx = sol.Value(routing.NextVar(idx))
            routes.append(stops)
            returns.append(sol.Value(time_dim.CumulVar(idx)))
    else:
        routes, returns = [[] for _ in range(p.num_teams)], [0] * p.num_teams
    return build_result(prob, routes, returns, solved=sol is not None)
