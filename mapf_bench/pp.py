# SPDX-License-Identifier: GPL-3.0-or-later
"""Prioritized planning with space-time A*.

Used to compute initial plans and, inside MAPF-LNS, to replan the agents of
a destroy set while the other agents keep their paths.
"""
import heapq
import random


class Reservations:
    """Space-time reservations of fixed paths (vertex, swap and parking)."""

    def __init__(self):
        self.vertex = {}      # (v, t) -> agent
        self.edge = set()     # (u, v, t): someone moves u -> v arriving at t
        self.park = {}        # v -> earliest time from which v is occupied forever
        self.last = {}        # v -> last time v is reserved (excluding parking)

    def add_path(self, path):
        for t, v in enumerate(path):
            self.vertex[(v, t)] = True
            self.last[v] = max(self.last.get(v, -1), t)
            if t > 0 and path[t - 1] != v:
                self.edge.add((path[t - 1], v, t))
        g = path[-1]
        self.park[g] = min(self.park.get(g, 10**9), len(path) - 1)

    def free(self, v, t):
        if (v, t) in self.vertex:
            return False
        p = self.park.get(v)
        return p is None or t < p

    def move_ok(self, u, v, t):
        """Moving u -> v arriving at t does not swap with a reserved move."""
        return u == v or (v, u, t) not in self.edge


def space_time_astar(graph, start, goal, res, h, horizon):
    """Shortest path from start to goal avoiding reservations.

    The agent stays at its goal after arrival, so the arrival time must be
    later than every reservation of the goal node.
    """
    if not res.free(start, 0):
        return None
    goal_after = res.last.get(goal, -1)
    if goal in res.park:
        return None  # another agent parks forever on our goal
    opened = [(h[start], 0, start)]
    parent = {(start, 0): None}
    while opened:
        f, t, v = heapq.heappop(opened)
        if v == goal and t > goal_after:
            path = []
            key = (v, t)
            while key is not None:
                path.append(key[0])
                key = parent[key]
            return path[::-1]
        if t >= horizon:
            continue
        for w in graph.out[v]:
            key = (w, t + 1)
            if key in parent:
                continue
            if not res.free(w, t + 1) or not res.move_ok(v, w, t + 1):
                continue
            parent[key] = (v, t)
            heapq.heappush(opened, (t + 1 + h[w], t + 1, w))
    return None


def prioritized_planning(graph, starts, goals, order=None, fixed=None,
                         horizon=None, h_tables=None):
    """Plan the agents in `order` one by one, avoiding `fixed` paths.

    Returns a dict agent -> path, or None if some agent cannot be planned.
    """
    n = len(starts)
    order = list(range(n)) if order is None else list(order)
    res = Reservations()
    longest = 0
    for k, p in (fixed or {}).items():
        res.add_path(p)
        longest = max(longest, len(p))
    horizon = horizon or (longest + 4 * graph.N)
    out = {}
    for k in order:
        h = h_tables[k] if h_tables else graph.bfs(goals[k], reverse=True)
        path = space_time_astar(graph, starts[k], goals[k], res, h, horizon)
        if path is None:
            return None
        res.add_path(path)
        out[k] = path
    return out


def initial_pp(graph, starts, goals, seed=0, attempts=50):
    """Prioritized planning with random restarts; returns per-agent paths."""
    rng = random.Random(seed)
    n = len(starts)
    h_tables = [graph.bfs(goals[k], reverse=True) for k in range(n)]
    order = sorted(range(n), key=lambda k: -h_tables[k][starts[k]])
    for _ in range(attempts):
        sol = prioritized_planning(graph, starts, goals, order, h_tables=h_tables)
        if sol is not None:
            return [sol[k] for k in range(n)]
        rng.shuffle(order)
    return None
