# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
from mapf_bench.graph import Graph
from mapf_bench.instances import grid_bottleneck, oneway_grid, random_digraph, multi_ring
from mapf_bench.plan import (validate, InvalidPlan, serialize, paths_to_plan,
                             makespan, sum_of_costs, arrival_times)
from mapf_bench.pp import initial_pp


def cycle(L):
    return Graph(L, [(j, (j + 1) % L) for j in range(L)])


def test_validate_rejects_swap_collision_and_missing_arc():
    G = Graph(3, [(0, 1), (1, 0), (1, 2), (2, 1)])
    validate(G, (0, 2), (1, 2), [(0, 2), (1, 2)])
    with pytest.raises(InvalidPlan):          # swap
        validate(G, (0, 1), (1, 0), [(0, 1), (1, 0)])
    with pytest.raises(InvalidPlan):          # vertex collision
        validate(G, (0, 2), (1, 1), [(0, 2), (1, 1)])
    with pytest.raises(InvalidPlan):          # no arc 0 -> 2
        validate(G, (0, 1), (2, 1), [(0, 1), (2, 1)])


def test_rotation_is_allowed():
    G = cycle(3)
    validate(G, (0, 1, 2), (1, 2, 0), [(0, 1, 2), (1, 2, 0)])


def test_metrics():
    plan = [(0, 5), (1, 5), (2, 5)]
    assert makespan(plan) == 2
    assert arrival_times(plan, (2, 5)) == [2, 0]
    assert sum_of_costs(plan, (2, 5)) == 2


@pytest.mark.parametrize("make", [
    lambda: grid_bottleneck(9, 9, K=1, n=8, seed=1),
    lambda: oneway_grid(6, 6, n=8, seed=2),
    lambda: random_digraph(30, 3, n=8, seed=3),
    lambda: multi_ring(2, 6, seed=0),
])
def test_pp_and_serialize_are_valid(make):
    inst = make()
    paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=0)
    assert paths is not None
    plan = paths_to_plan(paths)
    validate(inst.graph, inst.starts, inst.goals, plan)
    ser = serialize(inst.graph, plan)
    validate(inst.graph, inst.starts, inst.goals, ser)
    assert makespan(ser) >= makespan(plan)
