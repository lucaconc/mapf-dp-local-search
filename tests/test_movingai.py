# SPDX-License-Identifier: GPL-3.0-or-later
"""MovingAI benchmark: parsing, scenarios and instances."""
import os

import pytest

from mapf_bench import movingai as M
from mapf_bench.plan import validate, sum_of_costs


def test_maps_parse():
    for name in M.SMALL_MAPS + M.LARGE_MAPS:
        h, w, free = M.load_map(name)
        assert h > 0 and w > 0 and 0 < len(free) <= h * w
    h, w, free = M.load_map("random-32-32-20")
    assert (h, w) == (32, 32)
    raw = open(M.map_path("random-32-32-20")).read().split("\n")[4:36]
    assert len(free) == sum(row.count(".") for row in raw)


@pytest.mark.parametrize("scen", ["random-32-32-20-random-1.scen", "random-32-32-10-random-1.scen"])
def test_official_scenarios(scen):
    path = os.path.join(M.ROOT, "scens", scen)
    pairs = M.load_scen(path)
    h, w, free = M.load_map(scen.rsplit("-random-", 1)[0])
    assert len(pairs) >= 100
    for s, g in pairs:
        assert s in free and g in free
    starts = [s for s, _ in pairs]
    assert len(set(starts)) == len(starts)


def test_generated_scenarios_reproducible_and_valid(tmp_path):
    a = M.generate_scen("maze-32-32-4", 7, count=50, path=str(tmp_path / "a.scen"))
    b = M.generate_scen("maze-32-32-4", 7, count=50, path=str(tmp_path / "b.scen"))
    assert open(a).read() == open(b).read()
    h, w, free = M.load_map("maze-32-32-4")
    pairs = M.load_scen(a)
    assert len(pairs) == 50
    assert len({s for s, _ in pairs}) == 50 and len({g for _, g in pairs}) == 50
    for line, (s, g) in zip(open(a).read().split("\n")[1:], pairs):
        assert int(line.split("\t")[8]) == M._bfs(free, s)[g]


@pytest.mark.parametrize("name", ["mai_empty-16-16_n10", "mai_random-32-32-20_n20",
                                  "mai_maze-32-32-2_n10", "mai_room-32-32-4_n20"])
def test_instances_and_initial_plans(name):
    inst = M.families()[name](0)
    assert len(set(inst.starts)) == inst.n and len(set(inst.goals)) == inst.n
    assert inst.graph.strongly_connected()
    assert inst.initial_plan is not None
    validate(inst.graph, inst.starts, inst.goals, inst.initial_plan)


def test_instance_uses_the_first_n_agents():
    scen = M.scenarios("random-32-32-20", 1)[0]
    inst = M.instance("random-32-32-20", scen, 5, initial=None)
    pairs = M.load_scen(scen)[:5]
    assert [inst.graph.coords[v] for v in inst.starts] == [s for s, _ in pairs]
    assert [inst.graph.coords[v] for v in inst.goals] == [g for _, g in pairs]


def test_hybrid_end_to_end_on_a_benchmark_instance(monkeypatch):
    from mapf_bench.cpp_local_search import available
    from mapf_bench.official_lns import available as official
    if not available():
        pytest.skip("C++ local search not built")
    if not official():          # LNS phases on a grid: re-implementation
        monkeypatch.setenv("MAPF_BENCH_LNS_FALLBACK", "1")
    from mapf_bench.hybrid import hybrid
    from mapf_bench.algorithms import order_reduced_neighborhoods, order_corridor_neighborhoods
    inst = M.families()["mai_empty-16-16_n10"](0)
    p, st = hybrid(inst, inst.initial_plan, time_limit=5, lns_first=True, lns_phase=1.0,
                   neighborhoods=order_reduced_neighborhoods,
                   final_neighborhoods=order_corridor_neighborhoods)
    validate(inst.graph, inst.starts, inst.goals, p)
    assert sum_of_costs(p, inst.goals) <= sum_of_costs(inst.initial_plan, inst.goals)


def test_paper_suite_protocol():
    """The protocol has 509 scenarios in nine groups, and the statistics helpers
    behave as expected."""
    from collections import Counter
    from mapf_bench import paper_suite as P
    sc = P.scenarios()
    assert len(sc) == 509
    assert Counter(g for g, _, _ in sc) == {"S": 15, "D": 20, "M": 24, "H": 36, "W": 5, "L": 96, "K": 40, "C": 72,
                                            "P": 201}
    assert P._sign(5, 0) == 0.0625 and P._sign(0, 0) == 1.0
    assert P._holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_reduced_connectivity_keeps_agents_and_connectivity():
    """Blocking cells keeps the map connected, never blocks a start or a goal,
    and the levels are nested; level 0 is the original map."""
    scen = M.scenarios("room-32-32-4", 1)[0]
    seq = M.removal_sequence("room-32-32-4", scen, 40)
    assert len(seq) == len(set(seq)) > 0
    h, w, free = M.load_map("room-32-32-4")
    comp = set(max(M._components(free), key=len))
    pairs = M.load_scen(scen)[:40]
    terminals = {s for s, _ in pairs} | {g for _, g in pairs}
    assert not terminals & set(seq)
    for k in (0, len(seq) // 2, len(seq)):
        cells = comp - set(seq[:k])
        assert M._connected(cells, next(iter(terminals)))
    # minimal: no further cell can be blocked
    cells = comp - set(seq)
    for c in list(cells - terminals)[:40]:
        assert not M._connected(cells - {c}, next(iter(terminals)))
    a = M.reduced_instance("room-32-32-4", scen, 40, 0.0, initial=None)
    b = M.reduced_instance("room-32-32-4", scen, 40, 1.0, initial=None)
    assert a.n == b.n == 40 and a.graph.N == len(comp) and b.graph.N == len(comp) - len(seq)
    assert [a.graph.coords[v] for v in a.starts] == [b.graph.coords[v] for v in b.starts]
