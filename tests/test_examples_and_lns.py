# SPDX-License-Identifier: GPL-3.0-or-later
"""The two examples of the paper, and basic properties of MAPF-LNS."""
import pytest

from mapf_bench.instances import merge, ring, grid_bottleneck
from mapf_bench.lns import mapf_lns
from mapf_bench.local_search import neighborhood_search, GroupedClock, local_search
from mapf_bench.plan import validate, makespan, sum_of_costs, paths_to_plan, serialize
from mapf_bench.pp import initial_pp


@pytest.mark.parametrize("k,ell", [(1, 4), (1, 8), (2, 6), (3, 6)])
def test_merge_soc_tie_and_lns_soc_stuck(k, ell):
    inst = merge(k, ell)
    p0 = inst.initial_plan
    validate(inst.graph, inst.starts, inst.goals, p0)
    # B-first plan: same Sum-of-Costs, makespan smaller by k
    s, b = inst.starts[:k], inst.starts[k]
    c = inst.graph.out[b][-1] if inst.graph.out[b][0] == b else inst.graph.out[b][0]
    paths = [[b] + _route(inst, k, b)]
    for i in range(k):
        paths.append([s[i]] * (i + 2) + _route(inst, i, s[i]))
    paths = [paths[1 + i] for i in range(k)] + [paths[0]]
    f = paths_to_plan(paths)
    validate(inst.graph, inst.starts, inst.goals, f)
    assert makespan(f) == makespan(p0) - k
    assert sum_of_costs(f, inst.goals) == sum_of_costs(p0, inst.goals)
    # MAPF-LNS with the Sum-of-Costs objective never leaves the reference
    p, st = mapf_lns(inst, p0, "soc", max_iters=300, seed=1)
    assert makespan(p) == makespan(p0) and st["accepted"] == 0


def _route(inst, k, start):
    """Shortest route of agent k (forward part, excluding the start)."""
    G = inst.graph
    h = G.bfs(inst.goals[k], reverse=True)
    out, v = [], start
    while v != inst.goals[k]:
        v = min((w for w in G.out[v] if w != v), key=lambda w: h[w])
        out.append(v)
    return out


def test_merge_route_restricted_finds_it():
    inst = merge(1, 6)
    new, _ = neighborhood_search(inst, inst.initial_plan,
                                 GroupedClock(R=2, restricted=True))
    assert makespan(new) == makespan(inst.initial_plan) - 1


@pytest.mark.parametrize("L,q", [(5, 2), (6, 3), (8, 4), (10, 5)])
def test_ring_grouped_clock_radius_zero(L, q):
    inst = ring(L, L + 2, q)
    p0 = inst.initial_plan
    validate(inst.graph, inst.starts, inst.goals, p0)
    new, _ = neighborhood_search(inst, p0, GroupedClock(R=0, groups="auto"))
    assert makespan(new) == q and makespan(p0) == 2 * q
    # one clock and radius 0 cannot do it
    new1, _ = neighborhood_search(inst, p0, GroupedClock(R=0))
    assert makespan(new1) == 2 * q


@pytest.mark.parametrize("objective", ["soc", "makespan"])
def test_ring_lns_cannot_improve_with_small_destroy_sets(objective):
    inst = ring(8, 10, 4)
    p, st = mapf_lns(inst, inst.initial_plan, objective, N=5, max_iters=300, seed=0)
    assert makespan(p) == makespan(inst.initial_plan)


def test_lns_valid_improving_and_deterministic():
    inst = grid_bottleneck(9, 9, K=1, n=8, seed=4)
    paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=0)
    p0 = serialize(inst.graph, paths_to_plan(paths))
    p1, st1 = mapf_lns(inst, p0, "soc", max_iters=200, seed=7)
    p2, st2 = mapf_lns(inst, p0, "soc", max_iters=200, seed=7)
    validate(inst.graph, inst.starts, inst.goals, p1)
    assert p1 == p2
    assert sum_of_costs(p1, inst.goals) <= sum_of_costs(p0, inst.goals)


def test_full_rings_longer_than_destroy_set():
    """Rings of 12 agents: MAPF-LNS with destroy sets of 5 or 10 agents
    cannot improve; grouped clocks at radius 0 can."""
    from mapf_bench.instances import full_rings
    from mapf_bench.local_search import local_search
    inst = full_rings((12, 12), (4, 5))
    p0 = inst.initial_plan
    validate(inst.graph, inst.starts, inst.goals, p0)
    for N in (5, 10):
        p, _ = mapf_lns(inst, p0, "makespan", N=N, max_iters=200, seed=0)
        assert makespan(p) == makespan(p0)
    p, _ = local_search(inst, p0, [GroupedClock(R=0, groups="auto", max_groups=4)])
    validate(inst.graph, inst.starts, inst.goals, p)
    assert makespan(p) < makespan(p0)


def test_convoy_route_restricted_matches_makespan_lns():
    from mapf_bench.instances import convoy
    from mapf_bench.local_search import local_search
    inst = convoy(2, 3, 6, seed=1)
    p0 = inst.initial_plan
    p_ls, _ = local_search(inst, p0, [GroupedClock(R=6, restricted=True)], tiebreak="summin")
    p_soc, _ = mapf_lns(inst, p0, "soc", max_iters=200, seed=0)
    p_mk, _ = mapf_lns(inst, p0, "makespan", max_iters=200, seed=0)
    assert makespan(p_soc) == makespan(p0)
    assert makespan(p_ls) == makespan(p_mk) < makespan(p0)


def test_dense_oneway_local_search_beats_lns():
    """At 90% density MAPF-LNS repairs almost always fail; the
    route-restricted local search still shortens the plan."""
    from mapf_bench.explore import FAMILIES
    from mapf_bench.local_search import local_search
    from mapf_bench.plan import trim
    inst = FAMILIES["dense_oneway"](4)          # density 0.9
    p0 = trim(inst.initial_plan)
    validate(inst.graph, inst.starts, inst.goals, p0)
    p_lns, st = mapf_lns(inst, p0, "soc", max_iters=200, seed=0)
    p_ls, _ = local_search(inst, p0, [GroupedClock(R=3, restricted=True)],
                           tiebreak="summin", time_limit=30)
    validate(inst.graph, inst.starts, inst.goals, p_ls)
    assert st["failed_repairs"] > 0.8 * st["iters"]
    assert makespan(p_ls) < makespan(p_lns)
