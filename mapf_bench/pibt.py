# SPDX-License-Identifier: GPL-3.0-or-later
"""PIBT (priority inheritance with backtracking; Okumura et al., 2019): a
rule-based MAPF solver, used to build initial plans.

At every step each agent, from the highest priority down, takes the neighbour
(or its own node) closest to its goal that is not reserved for the next step;
if another agent stands there and has not moved yet, it is pushed recursively
(priority inheritance) and, if it cannot move, the next candidate is tried
(backtracking). Swaps are avoided; following and rotations are allowed, as in
the MAPF model of this package. Priorities grow while an agent is away from
its goal and are reset when it reaches it. PIBT is not complete: in dense
tree-like corridors it may cycle, so it stops after `max_steps` and returns
None if not all the agents are at their goals.
"""
import random
from collections import deque


def _dist_to(G, goal):
    """Distances to `goal` (backward BFS on the arcs)."""
    d = [10 ** 9] * G.N
    d[goal] = 0
    q = deque([goal])
    while q:
        u = q.popleft()
        for v in G.inn[u]:
            if d[v] > d[u] + 1:
                d[v] = d[u] + 1; q.append(v)
    return d


def pibt(G, starts, goals, max_steps=None, seed=0):
    n = len(starts)
    rng = random.Random(seed)
    dist = [_dist_to(G, g) for g in goals]
    if any(dist[i][starts[i]] >= 10 ** 9 for i in range(n)):
        return None
    max_steps = max_steps or 20 * G.N + 100
    eps = [rng.random() * 1e-3 for _ in range(n)]     # distinct initial priorities
    prio = [dist[i][starts[i]] / max(1, G.N) + eps[i] for i in range(n)]
    cur = list(starts)
    plan = [tuple(cur)]
    for _ in range(max_steps):
        if all(cur[i] == goals[i] for i in range(n)):
            return plan
        for i in range(n):                              # priorities
            prio[i] = prio[i] + 1 if cur[i] != goals[i] else eps[i]
        occ_now = {cur[i]: i for i in range(n)}
        nxt = [None] * n
        occ_next = {}

        def move(i):
            cand = [v for v in G.out[cur[i]] if v != cur[i]] + [cur[i]]
            rng.shuffle(cand)
            cand.sort(key=lambda v: dist[i][v])         # closest to the goal first (random ties)
            for v in cand:
                if v in occ_next:
                    continue
                k = occ_now.get(v)
                if k is not None and nxt[k] == cur[i]:  # would be a swap
                    continue
                occ_next[v] = i; nxt[i] = v
                if k is not None and k != i and nxt[k] is None and not move(k):
                    continue
                return True
            occ_next[cur[i]] = i; nxt[i] = cur[i]
            return False

        for i in sorted(range(n), key=lambda i: -prio[i]):
            if nxt[i] is None:
                move(i)
        cur = nxt
        plan.append(tuple(cur))
    return None if not all(cur[i] == goals[i] for i in range(n)) else plan
