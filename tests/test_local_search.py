# SPDX-License-Identifier: GPL-3.0-or-later
import itertools
import random
from collections import deque

import pytest

from mapf_bench.graph import Graph
from mapf_bench.instances import Instance, random_digraph, oneway_grid
from mapf_bench.local_search import (neighborhood_search, local_search, PathDistance,
                                     Pointwise, AgentDistance, Combined, GroupedClock,
                                     SumMinMixin, interaction_groups)
from mapf_bench.plan import validate, makespan, paths_to_plan, serialize, trim
from mapf_bench.pp import initial_pp

ALL_NBHDS = [
    lambda: PathDistance(1), lambda: Pointwise(1), lambda: AgentDistance(1),
    lambda: Combined(2, 1), lambda: GroupedClock(1), lambda: GroupedClock(2, restricted=True),
    lambda: GroupedClock(1, groups="auto"),
    lambda: GroupedClock(2, restricted=True, groups="auto"),
]


def _instance(seed, n=4):
    inst = random_digraph(14, 3, n=n, seed=seed)
    paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=seed)
    plan = trim(serialize(inst.graph, paths_to_plan(paths)))
    return inst, plan


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("make", ALL_NBHDS)
def test_search_returns_valid_plan_not_worse(seed, make):
    inst, plan = _instance(seed)
    new, info = neighborhood_search(inst, plan, make())
    assert new is not None
    validate(inst.graph, inst.starts, inst.goals, new)
    assert makespan(new) <= makespan(plan)


def _joint_optimum(inst):
    """Brute-force minimum makespan by BFS on the joint state space."""
    G, n = inst.graph, inst.n
    start, goal = tuple(inst.starts), tuple(inst.goals)
    dist = {start: 0}
    q = deque([start])
    while q:
        z = q.popleft()
        if z == goal:
            return dist[z]
        for y in itertools.product(*[G.out[v] for v in z]):
            if len(set(y)) < n:
                continue
            if any(y[a] == z[b] and y[b] == z[a] for a in range(n) for b in range(a + 1, n)):
                continue
            if y not in dist:
                dist[y] = dist[z] + 1
                q.append(y)
    return None


@pytest.mark.parametrize("seed", range(5))
def test_unbounded_radius_gives_global_optimum(seed):
    """With an unbounded radius the sequentialized search is exact."""
    inst = random_digraph(8, 2, n=3, seed=seed)
    paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=seed)
    plan = paths_to_plan(paths)
    new, _ = neighborhood_search(inst, plan, PathDistance(R=10**5))
    assert makespan(new) == _joint_optimum(inst)


@pytest.mark.parametrize("seed", range(3))
def test_nesting_of_distances(seed):
    """d_sm <= d_gc <= d_lag <= d_seq on random sequential states."""
    inst, plan = _instance(seed, n=4)
    rng = random.Random(seed)
    seq, lag, gc = PathDistance(), GroupedClock(R=0), GroupedClock(R=0, groups="auto")
    for nb in (seq, lag, gc):
        nb.prepare(inst, plan)
    sm = SumMinMixin()
    sm.n, sm.inst, sm.ref, sm.D = inst.n, inst, plan, seq.D
    sm._prepare_summin()
    for _ in range(200):
        z = tuple(rng.sample(range(inst.graph.N), inst.n))
        p = rng.randrange(inst.n)
        y = tuple(rng.randrange(inst.graph.N) for _ in range(p))
        a, b, c, d = sm.summin(z, y), gc.value(z, y), lag.value(z, y), seq.value(z, y)
        assert a <= b + 1e-9 <= c + 2e-9 and c <= d + 1e-9


def test_reference_has_distance_zero():
    inst, plan = _instance(1)
    for nb in (PathDistance(), GroupedClock(R=0), GroupedClock(R=0, restricted=True)):
        nb.prepare(inst, plan)
        n = inst.n
        for t in range(1, len(plan)):
            for j in range(n):
                assert nb.value(plan[t - 1], plan[t][:j]) == 0


def test_local_search_terminates_and_improves_serialized_plans():
    inst = oneway_grid(6, 6, n=6, seed=0)
    paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=0)
    plan = trim(serialize(inst.graph, paths_to_plan(paths)))
    new, st = local_search(inst, plan, [GroupedClock(1, groups="auto"), Combined(2, 1)],
                           tiebreak="summin", time_limit=60)
    validate(inst.graph, inst.starts, inst.goals, new)
    assert makespan(new) < makespan(plan)


def test_interaction_groups_partition():
    inst, plan = _instance(2, n=5)
    groups = interaction_groups(plan, window=2, max_groups=3)
    assert sorted(k for g in groups for k in g) == list(range(inst.n))
    assert len(groups) <= 3


@pytest.mark.parametrize("seed", range(3))
def test_path_distance_matches_definition(seed):
    """Incremental d_seq equals min over the sequential reference Y_h."""
    inst, plan = _instance(seed, n=4)
    nb = PathDistance()
    nb.prepare(inst, plan)
    D, n, T = nb.D, inst.n, len(plan) - 1
    Y = [(plan[t - 1], plan[t][:j], j) for t in range(1, T + 1) for j in range(n)]
    Y.append((plan[T], (), 0))
    rng = random.Random(seed)
    for _ in range(200):
        z = tuple(rng.sample(range(inst.graph.N), n))
        p = rng.randrange(n)
        y = tuple(rng.randrange(inst.graph.N) for _ in range(p))
        brute = min(sum(D[a, b] for a, b in zip(z, z2)) + sum(D[a, b] for a, b in zip(y, y2))
                    for z2, y2, j in Y if j == p)
        assert nb.value(z, y) == brute


from mapf_bench.cpp_local_search import available as _cpp_available, cpp_local_search


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("make", ALL_NBHDS + [
    lambda: [Pointwise(1), AgentDistance(1)],
    lambda: [GroupedClock(3, restricted=True), GroupedClock(1, groups="auto", max_groups=2),
             Combined(1, 0)]])
def test_cpp_identical_to_python(seed, make):
    """The C++ binary returns exactly the same plan as the Python code."""
    inst, plan = _instance(seed)
    nbs = make()
    nbs = nbs if isinstance(nbs, list) else [nbs]
    tb = None if isinstance(nbs[0], Pointwise) else "summin"
    a, _ = local_search(inst, plan, nbs, tiebreak=tb, time_limit=60, max_expansions=10**6)
    b, _ = cpp_local_search(inst, plan, nbs, tiebreak=tb, time_limit=60, max_expansions=10**6)
    assert [tuple(x) for x in a] == [tuple(x) for x in b]


def _joint_soc_optimum(inst):
    """Brute-force minimum Sum-of-Costs: Dijkstra on (configuration, time of
    last arrival of each agent) would be exponential; instead enumerate
    joint plans by Dijkstra on (configuration, set of agents that have
    'finished'), where an agent may declare itself finished only at its goal
    and must then stay there. Each joint step costs the number of agents not
    yet finished, so the total cost is the Sum-of-Costs."""
    import heapq
    G, n = inst.graph, inst.n
    goals = tuple(inst.goals)
    start = (tuple(inst.starts), 0)
    best = {start: 0}
    heap = [(0, start)]
    full = (1 << n) - 1
    while heap:
        c, (z, fin) = heapq.heappop(heap)
        if c > best[(z, fin)]:
            continue
        if fin == full:
            return c
        # declare any subset of agents at their goals finished (free action)
        for k in range(n):
            if not fin >> k & 1 and z[k] == goals[k]:
                s2 = (z, fin | 1 << k)
                if c < best.get(s2, 1 << 60):
                    best[s2] = c
                    heapq.heappush(heap, (c, s2))
        step = n - bin(fin).count("1")
        opts = [[z[k]] if fin >> k & 1 else G.out[z[k]] for k in range(n)]
        for y in itertools.product(*opts):
            if len(set(y)) < n:
                continue
            if any(y[a] == z[b] and y[b] == z[a] for a in range(n) for b in range(a + 1, n)):
                continue
            s2 = (y, fin)
            if c + step < best.get(s2, 1 << 60):
                best[s2] = c + step
                heapq.heappush(heap, (c + step, s2))
    return None


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(6))
def test_cpp_soc_unbounded_radius_is_optimal(seed):
    from mapf_bench.plan import sum_of_costs
    inst = random_digraph(8, 2, n=3, seed=seed)
    plan = paths_to_plan(initial_pp(inst.graph, inst.starts, inst.goals, seed=seed))
    new, st = cpp_local_search(inst, plan, [PathDistance(R=10**5)], objective="soc",
                               max_expansions=10**7)
    validate(inst.graph, inst.starts, inst.goals, new)
    assert sum_of_costs(new, inst.goals) == _joint_soc_optimum(inst) == st["soc"]


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("make", ALL_NBHDS)
def test_cpp_soc_valid_and_not_worse(seed, make):
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    nbs = make()
    nbs = nbs if isinstance(nbs, list) else [nbs]
    new, st = cpp_local_search(inst, plan, nbs, tiebreak="summin", objective="soc")
    validate(inst.graph, inst.starts, inst.goals, new)
    assert sum_of_costs(new, inst.goals) <= sum_of_costs(plan, inst.goals)
    assert st["soc"] == sum_of_costs(new, inst.goals)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("objective", ["soc", "makespan"])
@pytest.mark.parametrize("seed", range(3))
def test_vnd_returns_a_local_optimum_of_every_neighborhood(seed, objective):
    from mapf_bench.algorithms import all_neighborhoods
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    obj = (lambda p: sum_of_costs(p, inst.goals)) if objective == "soc" else makespan
    new, st = cpp_local_search(inst, plan, all_neighborhoods(), tiebreak="summin",
                               objective=objective, vnd=True, max_expansions=10**6)
    validate(inst.graph, inst.starts, inst.goals, new)
    assert obj(new) <= obj(plan) and st["budget_exceeded"] == 0
    for nb in all_neighborhoods():
        again, st2 = cpp_local_search(inst, new, [nb], tiebreak="summin",
                                      objective=objective, max_expansions=10**6)
        assert st2["budget_exceeded"] > 0 or obj(again) == obj(new)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
def test_soc_proxy_never_increases_soc(seed):
    from mapf_bench.algorithms import all_neighborhoods
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    new, st = cpp_local_search(inst, plan, all_neighborhoods(), tiebreak="summin",
                               objective="soc_proxy", vnd=True)
    validate(inst.graph, inst.starts, inst.goals, new)
    assert st["soc"] == sum_of_costs(new, inst.goals) <= sum_of_costs(plan, inst.goals)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
def test_adaptive_radius_vnd(seed):
    """Adaptive VND: valid, not worse, and a local optimum of every
    neighborhood at its largest radius (unless the budget was exceeded)."""
    from mapf_bench.algorithms import adaptive_neighborhoods
    from mapf_bench.cpp_local_search import AdaptiveRadius
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    soc = lambda p: sum_of_costs(p, inst.goals)
    new, st = cpp_local_search(inst, plan, adaptive_neighborhoods(), tiebreak="summin",
                               objective="soc_proxy", vnd=True, max_expansions=10**6)
    validate(inst.graph, inst.starts, inst.goals, new)
    assert soc(new) <= soc(plan)
    for a in adaptive_neighborhoods():
        fixed = AdaptiveRadius(a.template, a.rmax, a.rmax)
        again, st2 = cpp_local_search(inst, new, [fixed], tiebreak="summin",
                                      objective="soc_proxy", max_expansions=10**6)
        assert st2["budget_exceeded"] > 0 or soc(again) == soc(new)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
def test_adaptive_radius_increases_only_when_needed():
    """Ring example: grouped clocks improve at radius 0 in the first search
    (then radii 0..3 confirm the local optimum); with a single clock radius 0
    does not suffice, and the adaptive radius still reaches the optimum."""
    from mapf_bench.instances import ring
    from mapf_bench.cpp_local_search import AdaptiveRadius
    inst = ring(6, 8, 3)
    p0 = inst.initial_plan
    _, st = cpp_local_search(inst, p0, [AdaptiveRadius("gc:{R}:1.0:auto:0:2:4", 0, 3)],
                             vnd=True)
    assert st["makespan"] == 3 and st["improvements"] == 1 and st["searches"] == 1 + 4
    _, st0 = cpp_local_search(inst, p0, [AdaptiveRadius("gc:{R}:1.0:none:0:2:4", 0, 0)])
    assert st0["makespan"] == 6
    _, st = cpp_local_search(inst, p0, [AdaptiveRadius("gc:{R}:1.0:none:0:2:4", 0, 6)],
                             vnd=True)
    assert st["makespan"] == 3 and st["radius_increases"] > 0


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
def test_adaptive_radius_keep_radius(seed):
    """keep_radius: valid, not worse, local optimum at the largest radii."""
    from mapf_bench.algorithms import adaptive_neighborhoods
    from mapf_bench.cpp_local_search import AdaptiveRadius
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    soc = lambda p: sum_of_costs(p, inst.goals)
    new, st = cpp_local_search(inst, plan, adaptive_neighborhoods(), tiebreak="summin",
                               objective="soc_proxy", vnd=True, keep_radius=True,
                               max_expansions=10**6)
    validate(inst.graph, inst.starts, inst.goals, new)
    assert soc(new) <= soc(plan)
    for a in adaptive_neighborhoods():
        again, st2 = cpp_local_search(inst, new, [AdaptiveRadius(a.template, a.rmax, a.rmax)],
                                      tiebreak="summin", objective="soc_proxy",
                                      max_expansions=10**6)
        assert st2["budget_exceeded"] > 0 or soc(again) == soc(new)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
def test_reduced_vnd_solves_the_paper_examples():
    """The neighborhoods of LS solve the examples of the paper; the serialized
    rings (Example "Serialized groups") by retiming, the first neighborhood."""
    from mapf_bench.algorithms import order_corridor_neighborhoods
    from mapf_bench.instances import ring, full_rings
    for inst, target in ((ring(6, 8, 3), 3), (full_rings((12, 12), (4, 5)), 5)):
        p, st = cpp_local_search(inst, inst.initial_plan, order_corridor_neighborhoods(),
                                 tiebreak="summin", objective="soc_proxy", vnd=True)
        validate(inst.graph, inst.starts, inst.goals, p)
        assert makespan(p) == target


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("objective", ["makespan", "soc", "soc_proxy"])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("make", ALL_NBHDS)
def test_astar_same_cost_as_dijkstra(seed, objective, make):
    """A single A* search finds the same optimal cost as Dijkstra."""
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    nbs = make()
    nbs = nbs if isinstance(nbs, list) else [nbs]
    obj = makespan if objective == "makespan" else (lambda p: sum_of_costs(p, inst.goals))
    a, sa = cpp_local_search(inst, plan, nbs, tiebreak="summin", objective=objective,
                             max_iters=1, max_expansions=10**7)
    b, sb = cpp_local_search(inst, plan, nbs, tiebreak="summin", objective=objective,
                             max_iters=1, max_expansions=10**7, astar=True)
    validate(inst.graph, inst.starts, inst.goals, b)
    assert obj(a) == obj(b)
    assert sb["states"] <= sa["states"] * 1.05 + 50


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(5))
def test_astar_unbounded_radius_is_optimal(seed):
    from mapf_bench.plan import sum_of_costs
    inst = random_digraph(8, 2, n=3, seed=seed)
    plan = paths_to_plan(initial_pp(inst.graph, inst.starts, inst.goals, seed=seed))
    new, _ = cpp_local_search(inst, plan, [PathDistance(R=10**5)], astar=True)
    assert makespan(new) == _joint_optimum(inst)
    new, _ = cpp_local_search(inst, plan, [PathDistance(R=10**5)], astar=True,
                              objective="soc", max_expansions=10**7)
    assert sum_of_costs(new, inst.goals) == _joint_soc_optimum(inst)


from mapf_bench.cpp_local_search import Window as _Window


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("astar", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_window_unbounded_is_optimal(seed, astar):
    """A free window as long as the plan contains every plan: global optimum."""
    from mapf_bench.plan import sum_of_costs
    inst = random_digraph(8, 2, n=3, seed=seed)
    plan = paths_to_plan(initial_pp(inst.graph, inst.starts, inst.goals, seed=seed))
    new, _ = cpp_local_search(inst, plan, [_Window(50)], astar=astar, max_expansions=10**7)
    validate(inst.graph, inst.starts, inst.goals, new)
    assert makespan(new) == _joint_optimum(inst)
    new, _ = cpp_local_search(inst, plan, [_Window(50)], astar=astar, objective="soc",
                              max_expansions=10**7)
    assert sum_of_costs(new, inst.goals) == _joint_soc_optimum(inst)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
def test_window_cuts_a_detour():
    """Reference a -> b -> a -> c with an arc a -> c: a window of one round
    from x^0 rejoins the reference at x^3 and cuts the detour."""
    from mapf_bench.graph import Graph
    from mapf_bench.instances import Instance
    # nodes: 0=a, 1=b, 2=c, 3 = other agent's node, 4 = spare
    G = Graph(5, [(0, 1), (1, 0), (0, 2), (2, 0), (3, 4), (4, 3)])
    inst = Instance("detour", "test", G, (0, 3), (2, 3))
    ref = [(0, 3), (1, 3), (0, 3), (2, 3)]
    validate(G, inst.starts, inst.goals, ref)
    new, st = cpp_local_search(inst, ref, [_Window(1)])
    assert makespan(new) == 1 and st["improvements"] == 1
    # with W = 0 the plan cannot leave the reference
    new0, _ = cpp_local_search(inst, ref, [_Window(0)])
    assert makespan(new0) == 3


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("W", [1, 2, 3])
def test_window_valid_not_worse_and_astar_same_cost(seed, W):
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    for objective in ("makespan", "soc_proxy"):
        a, _ = cpp_local_search(inst, plan, [_Window(W)], objective=objective, max_iters=1,
                                max_expansions=10**7)
        b, _ = cpp_local_search(inst, plan, [_Window(W)], objective=objective, max_iters=1,
                                max_expansions=10**7, astar=True)
        for p in (a, b):
            validate(inst.graph, inst.starts, inst.goals, p)
        obj = makespan if objective == "makespan" else (lambda p: sum_of_costs(p, inst.goals))
        assert obj(b) == obj(a) <= obj(plan)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("W", [1, 2])
def test_single_window_contained_in_multi(seed, W):
    """One window is a subset of several windows: never a lower cost; and the
    bound K = n is the same as no bound."""
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    soc = lambda p: sum_of_costs(p, inst.goals)
    one, _ = cpp_local_search(inst, plan, [_Window(W)], objective="soc_proxy", max_iters=1,
                              max_expansions=10**7)
    many, _ = cpp_local_search(inst, plan, [_Window(W, multi=True)], objective="soc_proxy",
                               max_iters=1, max_expansions=10**7)
    kn, _ = cpp_local_search(inst, plan, [_Window(W, K=inst.n)], objective="soc_proxy",
                             max_iters=1, max_expansions=10**7)
    for p in (one, many, kn):
        validate(inst.graph, inst.starts, inst.goals, p)
    assert soc(many) <= soc(one) <= soc(plan)
    assert soc(kn) == soc(one)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
def test_window_size_polynomial_for_W1_exponential_for_W2():
    """Neighborhood sizes on the pairs instance: W = 1 grows linearly with the
    number of agents, W = 2 at least doubles with every added agent, and
    W = 2 with K = 1 moving agent grows polynomially."""
    from mapf_bench.cpp_local_search import neighborhood_size
    from mapf_bench.instances import pairs_instance
    s1, s2, sk = [], [], []
    for m in (2, 4, 6, 8):
        inst = pairs_instance(m, T=6)
        s1.append(neighborhood_size(inst, inst.initial_plan, _Window(1))["states"])
        s2.append(neighborhood_size(inst, inst.initial_plan, _Window(2))["states"])
        sk.append(neighborhood_size(inst, inst.initial_plan, _Window(2, K=1))["states"])
    assert all(b >= 3 * a for a, b in zip(s2, s2[1:]))           # x4 per 2 agents
    assert all(b <= 3 * a for a, b in zip(s1, s1[1:]))
    assert all(b <= 3 * a for a, b in zip(sk, sk[1:]))


from mapf_bench.cpp_local_search import FreeSet as _FreeSet


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(5))
def test_free_set_all_agents_is_optimal(seed):
    """With every agent free the neighborhood contains every plan."""
    from mapf_bench.plan import sum_of_costs
    inst = random_digraph(8, 2, n=3, seed=seed)
    plan = paths_to_plan(initial_pp(inst.graph, inst.starts, inst.goals, seed=seed))
    for astar in (False, True):
        new, _ = cpp_local_search(inst, plan, [_FreeSet(K=3, R=0, candidates=1)], astar=astar,
                                  max_expansions=10**7)
        validate(inst.graph, inst.starts, inst.goals, new)
        assert makespan(new) == _joint_optimum(inst)
        new, _ = cpp_local_search(inst, plan, [_FreeSet(K=3, R=0, candidates=1)], astar=astar,
                                  objective="soc", max_expansions=10**7)
        assert sum_of_costs(new, inst.goals) == _joint_soc_optimum(inst)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("K,R,restricted", [(1, 0, True), (1, 1, False), (2, 1, True), (2, 2, False)])
def test_free_set_valid_not_worse_astar_same_cost(seed, K, R, restricted):
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    soc = lambda p: sum_of_costs(p, inst.goals)
    nb = _FreeSet(K=K, R=R, restricted=restricted, candidates=3)
    a, _ = cpp_local_search(inst, plan, [nb], objective="soc_proxy", max_iters=1,
                            max_expansions=10**7)
    b, _ = cpp_local_search(inst, plan, [nb], objective="soc_proxy", max_iters=1,
                            max_expansions=10**7, astar=True)
    for p in (a, b):
        validate(inst.graph, inst.starts, inst.goals, p)
    assert soc(a) == soc(b) <= soc(plan)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
def test_free_set_size_polynomial_in_n():
    """For K fixed the neighborhood grows polynomially with n (here at most
    about n^(K+1) between n = 4 and n = 12 on a 9x9 grid with fixed T)."""
    from mapf_bench.cpp_local_search import neighborhood_size
    from mapf_bench.instances import grid_bottleneck
    sizes = {1: [], 2: []}
    for n in (4, 8, 12):
        inst = grid_bottleneck(9, 9, K=0, n=n, seed=3)
        plan = trim(paths_to_plan(initial_pp(inst.graph, inst.starts, inst.goals, seed=3)))
        for K in (1, 2):
            r = neighborhood_size(inst, plan, _FreeSet(K=K, R=0, restricted=True, candidates=1),
                                  max_states=5 * 10**6)
            assert not r["budget_exceeded"]
            sizes[K].append(r["states"] / (r["T"] + 1))
    for K in (1, 2):
        assert sizes[K][-1] <= sizes[K][0] * (12 / 4) ** (K + 2)


from mapf_bench.cpp_local_search import Retime as _Retime, OrderSwap as _OrderSwap


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("make", [
    lambda s: random_digraph(30, 3, n=8, seed=s),
    lambda s: __import__("mapf_bench.instances", fromlist=["x"]).grid_bottleneck(9, 9, K=1, n=8, seed=s),
    lambda s: __import__("mapf_bench.instances", fromlist=["x"]).oneway_grid(6, 6, n=10, seed=s)])
@pytest.mark.parametrize("seed", range(3))
def test_retime_of_serialized_plan_is_at_least_as_good_as_the_original(make, seed):
    """Serializing keeps routes and visit orders, so retiming the serialized
    plan cannot be worse than the plan it came from (in every arrival)."""
    from mapf_bench.plan import sum_of_costs, arrival_times
    inst = make(seed)
    pp = paths_to_plan(initial_pp(inst.graph, inst.starts, inst.goals, seed=seed))
    ser = trim(serialize(inst.graph, pp))
    for objective in ("makespan", "soc"):
        new, _ = cpp_local_search(inst, ser, [_Retime()], objective=objective)
        validate(inst.graph, inst.starts, inst.goals, new)
        a_new, a_pp = arrival_times(new, inst.goals), arrival_times(pp, inst.goals)
        assert all(x <= y for x, y in zip(a_new, a_pp))
    again, _ = cpp_local_search(inst, new, [_Retime()], objective="soc")
    assert sum_of_costs(again, inst.goals) == sum_of_costs(new, inst.goals)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
def test_retime_parallelizes_the_ring_example():
    from mapf_bench.instances import ring, full_rings
    for inst, target in ((ring(6, 8, 3), 3), (full_rings((12, 12), (4, 5)), 5)):
        new, _ = cpp_local_search(inst, inst.initial_plan, [_Retime()])
        validate(inst.graph, inst.starts, inst.goals, new)
        assert makespan(new) == target


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("k,ell", [(1, 6), (2, 6), (3, 8)])
def test_order_swap_solves_the_convoy(k, ell):
    """Retiming alone cannot change who crosses the merge first; pair-wide
    swaps let the long-haul agent overtake (makespan objective)."""
    from mapf_bench.instances import merge
    inst = merge(k, ell)
    p0 = inst.initial_plan
    r, _ = cpp_local_search(inst, p0, [_Retime()])
    assert makespan(r) == makespan(p0)
    s, _ = cpp_local_search(inst, p0, [_OrderSwap()], vnd=True)
    validate(inst.graph, inst.starts, inst.goals, s)
    assert makespan(s) == makespan(p0) - k


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(4))
def test_order_neighborhoods_valid_and_not_worse(seed):
    from mapf_bench.plan import sum_of_costs
    inst, plan = _instance(seed)
    for nb in (_Retime(), _OrderSwap()):
        for objective in ("makespan", "soc"):
            new, _ = cpp_local_search(inst, plan, [nb], objective=objective, vnd=True)
            validate(inst.graph, inst.starts, inst.goals, new)
            obj = makespan if objective == "makespan" else (lambda p: sum_of_costs(p, inst.goals))
            assert obj(new) <= obj(plan)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
def test_free_set_corridor(seed):
    """A very wide corridor is the unrestricted free set; narrow corridors are
    valid, never worse, A* = Dijkstra, and give smaller neighborhoods."""
    from mapf_bench.plan import sum_of_costs
    from mapf_bench.cpp_local_search import neighborhood_size
    inst, plan = _instance(seed)
    soc = lambda p: sum_of_costs(p, inst.goals)
    base = _FreeSet(K=2, R=0, restricted=True, candidates=3)
    wide = _FreeSet(K=2, R=0, restricted=True, candidates=3, corridor=10**6)
    a, _ = cpp_local_search(inst, plan, [base], objective="soc_proxy", max_iters=1, max_expansions=10**7)
    b, _ = cpp_local_search(inst, plan, [wide], objective="soc_proxy", max_iters=1, max_expansions=10**7)
    assert soc(a) == soc(b)
    for c in (0, 1, 2):
        nb = _FreeSet(K=2, R=0, restricted=True, candidates=3, corridor=c)
        d, _ = cpp_local_search(inst, plan, [nb], objective="soc_proxy", max_iters=1, max_expansions=10**7)
        e, _ = cpp_local_search(inst, plan, [nb], objective="soc_proxy", max_iters=1, max_expansions=10**7,
                                astar=True)
        for p in (d, e):
            validate(inst.graph, inst.starts, inst.goals, p)
        assert soc(d) == soc(e) <= soc(plan)
    one = _FreeSet(K=2, R=0, restricted=True, candidates=1)
    sz = lambda nb: neighborhood_size(inst, plan, nb)["states"]
    assert sz(_FreeSet(K=2, R=0, restricted=True, candidates=1, corridor=1)) <= sz(one)


@pytest.mark.skipif(not _cpp_available(), reason="C++ local search not built")
@pytest.mark.parametrize("seed", range(3))
def test_hybrid_valid_not_worse_and_certified_local_optimum(seed):
    """The hybrid returns a valid plan, not worse than the initial one; when it
    is certified, no neighborhood improves it (checked on the order
    neighborhoods, which use the exact Sum-of-Costs)."""
    from mapf_bench.hybrid import hybrid
    from mapf_bench.plan import sum_of_costs
    from mapf_bench.cpp_local_search import Retime, OrderSwap
    inst, plan = _instance(seed)
    soc = lambda p: sum_of_costs(p, inst.goals)
    new, st = hybrid(inst, plan, time_limit=10, lns_phase=0.5, seed=seed)
    validate(inst.graph, inst.starts, inst.goals, new)
    assert soc(new) <= soc(plan)
    assert st["certified"]
    for nb in (Retime(), OrderSwap()):
        again, _ = cpp_local_search(inst, new, [nb], objective="soc")
        assert soc(again) == soc(new)
