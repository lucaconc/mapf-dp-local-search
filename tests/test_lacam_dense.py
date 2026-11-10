# SPDX-License-Identifier: GPL-3.0-or-later
"""LaCAM3 adapter and high-occupancy MovingAI instances."""
import pytest

from mapf_bench import movingai as M
from mapf_bench.lacam import available, lacam_solve
from mapf_bench.plan import validate, sum_of_costs

needs_lacam = pytest.mark.skipif(not available(), reason="LaCAM3 not built")


def test_dense_scenarios_cover_the_component():
    for m in M.DENSE_MAPS:
        h, w, free = M.load_map(m)
        size = len(max(M._components(free), key=len))
        for p in M.dense_scenarios(m, 1):
            pairs = M.load_scen(p)
            assert len(pairs) == size
            assert len({s for s, _ in pairs}) == size and len({g for _, g in pairs}) == size


@needs_lacam
@pytest.mark.parametrize("m,occ", [("empty-8-8", 0.98), ("empty-16-16", 0.98), ("random-8-8-20", 0.9)])
def test_dense_instances_have_valid_initial_plans(m, occ):
    inst = M.dense_instance(m, M.dense_scenarios(m, 1)[0], occ)
    h, w, free = M.load_map(m)
    size = len(max(M._components(free), key=len))
    assert inst.n == round(occ * size)
    assert inst.initial_plan is not None
    validate(inst.graph, inst.starts, inst.goals, inst.initial_plan)


@needs_lacam
def test_lacam_anytime_not_worse_than_first_solution():
    inst = M.dense_instance("empty-8-8", M.dense_scenarios("empty-8-8", 1)[0], 0.8)
    p = lacam_solve(inst, time_limit=2, anytime=True)
    validate(inst.graph, inst.starts, inst.goals, p)
    assert sum_of_costs(p, inst.goals) <= sum_of_costs(inst.initial_plan, inst.goals)


def test_agent_distance_beyond_64_agents():
    """The agent distance works with more than 64 agents (it used a bit mask)."""
    from mapf_bench.cpp_local_search import available as cpp, cpp_local_search
    if not cpp() or not available():
        pytest.skip("C++ local search or LaCAM3 not built")
    from mapf_bench.local_search import AgentDistance
    inst = M.dense_instance("empty-16-16", M.dense_scenarios("empty-16-16", 1)[0], 0.5)
    assert inst.n > 64
    p, st = cpp_local_search(inst, inst.initial_plan, [AgentDistance(1)], tiebreak="summin",
                             objective="soc_proxy", time_limit=10)
    validate(inst.graph, inst.starts, inst.goals, p)
    assert sum_of_costs(p, inst.goals) <= sum_of_costs(inst.initial_plan, inst.goals)
