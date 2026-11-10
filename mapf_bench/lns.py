# SPDX-License-Identifier: GPL-3.0-or-later
"""A re-implementation of MAPF-LNS (Li et al., IJCAI 2021) for digraphs.

This is not the original C++ code. It follows its structure: an adaptive
choice among three destroy heuristics (random, agent-based, map-based),
replanning of the destroy set with randomized prioritized planning while the
other agents keep their paths, and acceptance of strict improvements of the
objective. The objective is the Sum-of-Costs as in MAPF-LNS, or the makespan
(an option that MAPF-LNS does not offer, included for a fairer comparison).
"""
import random
import time
from collections import deque

from .plan import paths_to_plan, plan_to_paths, sum_of_costs, makespan
from .pp import prioritized_planning


def _objective(paths, goals, kind):
    plan = paths_to_plan(paths)
    if kind == "soc":
        return (sum_of_costs(plan, goals),)
    if kind == "makespan":
        return (makespan(plan),)
    if kind == "lex":  # makespan first, then Sum-of-Costs
        return (makespan(plan), sum_of_costs(plan, goals))
    raise ValueError(kind)


def _random_destroy(rng, inst, paths, N, ctx):
    return rng.sample(range(inst.n), N)


def _agent_destroy(rng, inst, paths, N, ctx):
    """Agent-based: start from a delayed agent and collect the agents whose
    paths share nodes with its path (a simplified random walk)."""
    delays = [len(paths[k]) - 1 - ctx["sp"][k] for k in range(inst.n)]
    cand = [k for k in range(inst.n) if delays[k] > 0 and k not in ctx["tabu"]]
    if not cand:
        ctx["tabu"].clear()
        cand = [k for k in range(inst.n) if delays[k] > 0] or list(range(inst.n))
    a = max(cand, key=lambda k: (delays[k], rng.random()))
    ctx["tabu"].add(a)
    chosen = [a]
    nodes = set(paths[a])
    others = [k for k in range(inst.n) if k != a]
    rng.shuffle(others)
    for k in others:
        if len(chosen) >= N:
            break
        if nodes & set(paths[k]):
            chosen.append(k)
            nodes |= set(paths[k])
    for k in others:
        if len(chosen) >= N:
            break
        if k not in chosen:
            chosen.append(k)
    return chosen


def _map_destroy(rng, inst, paths, N, ctx):
    """Map-based: agents visiting the surroundings of a random intersection."""
    G = inst.graph
    inter = ctx["intersections"] or list(range(G.N))
    root = rng.choice(inter)
    visits = {}
    for k, p in enumerate(paths):
        for v in set(p):
            visits.setdefault(v, []).append(k)
    chosen, seen, q = [], {root}, deque([root])
    while q and len(chosen) < N:
        v = q.popleft()
        for k in visits.get(v, []):
            if k not in chosen and len(chosen) < N:
                chosen.append(k)
        for w in set(G.out[v]) | set(G.inn[v]):
            if w not in seen:
                seen.add(w)
                q.append(w)
    rest = [k for k in range(inst.n) if k not in chosen]
    rng.shuffle(rest)
    return chosen + rest[: N - len(chosen)]


HEURISTICS = {"random": _random_destroy, "agent": _agent_destroy, "map": _map_destroy}


def mapf_lns(inst, init_plan, objective="soc", N=5, max_iters=1000,
             time_limit=30.0, seed=0, reaction=0.01, accept_equal=False, stagnation=0):
    """Run MAPF-LNS from `init_plan`; returns (plan, stats)."""
    rng = random.Random(seed)
    G, goals = inst.graph, inst.goals
    N = min(N, inst.n)
    paths = plan_to_paths(init_plan, goals)
    h_tables = [G.bfs(goals[k], reverse=True) for k in range(inst.n)]
    ctx = {
        "sp": [int(h_tables[k][inst.starts[k]]) for k in range(inst.n)],
        "tabu": set(),
        "intersections": [v for v in range(G.N) if G.degree_und(v) >= 3],
    }
    best = _objective(paths, goals, objective)
    weights = {h: 1.0 for h in HEURISTICS}
    t0 = time.perf_counter()
    stats = {"iters": 0, "accepted": 0, "failed_repairs": 0}
    last_improvement = 0
    for it in range(max_iters):
        if time.perf_counter() - t0 > time_limit:
            break
        if stagnation and it - last_improvement > stagnation:
            break
        stats["iters"] += 1
        names = list(weights)
        h = rng.choices(names, weights=[weights[x] for x in names])[0]
        S = HEURISTICS[h](rng, inst, paths, N, ctx)
        fixed = {k: paths[k] for k in range(inst.n) if k not in S}
        order = list(S)
        rng.shuffle(order)
        sol = prioritized_planning(G, inst.starts, goals, order, fixed,
                                   h_tables=h_tables)
        gain = 0.0
        if sol is None:
            stats["failed_repairs"] += 1
        else:
            new_paths = [sol.get(k, paths[k]) for k in range(inst.n)]
            val = _objective(new_paths, goals, objective)
            if val < best or (accept_equal and val == best):
                gain = best[0] - val[0]
                paths, best = new_paths, val
                stats["accepted"] += 1
                last_improvement = it
        weights[h] = (1 - reaction) * weights[h] + reaction * max(gain, 0.0)
        weights[h] = max(weights[h], 1e-3)
    stats["time"] = time.perf_counter() - t0
    return paths_to_plan(paths), stats
