"""Simple dispatch rules to compare against the VRP: nearest-first, priority-first and random.

Same candidates, travel times, service times, teams and mission limit as the VRP
(``prepare_problem``). Teams move in event order: the team that is free earliest picks its next
building among those it can still reach and return from within the mission time; when none fits,
that team goes home. Nearest-first and random ignore priority, priority-first ignores distance;
together they show what the optimiser buys.
"""
from __future__ import annotations

import numpy as np

from quaketriage.planning.vrp import Problem, build_result


def _simulate(prob: Problem, choose) -> dict:
    t = prob.trans
    n = len(t)
    free = [0] * prob.num_teams  # time each team is free
    at = [0] * prob.num_teams  # current matrix node
    routes = [[] for _ in range(prob.num_teams)]
    done = [False] * prob.num_teams
    todo = set(range(1, n))
    while not all(done) and todo:
        k = min((i for i in range(prob.num_teams) if not done[i]), key=lambda i: free[i])
        # t[a, b] = travel + service at a; arrival at b = free + t[at, b]
        feasible = [b for b in todo
                    if free[k] + t[at[k], b] + t[b, 0] <= prob.horizon]
        if not feasible:
            done[k] = True
            continue
        b = choose(at[k], feasible)
        free[k] += int(t[at[k], b])
        routes[k].append((b, free[k]))
        at[k] = b
        todo.discard(b)
    returns = [free[k] + int(t[at[k], 0]) for k in range(prob.num_teams)]
    return build_result(prob, routes, returns)


def greedy_nearest(prob: Problem) -> dict:
    """Always go to the closest (by travel + service time) feasible building."""
    return _simulate(prob, lambda a, c: min(c, key=lambda b: (prob.trans[a, b], b)))


def priority_first(prob: Problem) -> dict:
    """Always go to the highest-priority feasible building, ignoring distance (ties: lower row)."""
    prio = prob.cand["priority"].to_numpy(float)
    return _simulate(prob, lambda a, c: min(c, key=lambda b: (-prio[b - 1], b)))


def random_dispatch(prob: Problem, seed: int) -> dict:
    """Pick a random feasible building."""
    rng = np.random.default_rng(seed)
    return _simulate(prob, lambda a, c: sorted(c)[int(rng.integers(len(c)))])
