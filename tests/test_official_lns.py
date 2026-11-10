# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests of the adapter for the official MAPF-LNS (skipped if not built)."""
import pytest

from mapf_bench.official_lns import available, official_lns, is_grid_instance
from mapf_bench.instances import grid_bottleneck, grid_rings, oneway_grid
from mapf_bench.local_search import local_search, GroupedClock
from mapf_bench.plan import validate, makespan, sum_of_costs, paths_to_plan, serialize
from mapf_bench.pp import initial_pp

needs_bin = pytest.mark.skipif(not available(), reason="official MAPF-LNS not built")


def test_grid_detection():
    assert is_grid_instance(grid_bottleneck(9, 9, K=1, n=4, seed=0))
    assert is_grid_instance(grid_rings((4, 4), shifts=(2, 2)))
    assert not is_grid_instance(oneway_grid(6, 6, n=4, seed=0))


@needs_bin
def test_round_trip_from_given_plan():
    inst = grid_bottleneck(9, 9, K=1, n=8, seed=0)
    p0 = serialize(inst.graph, paths_to_plan(initial_pp(inst.graph, inst.starts, inst.goals)))
    p, st = official_lns(inst, p0, time_limit=1, seed=1)
    validate(inst.graph, inst.starts, inst.goals, p)
    assert sum_of_costs(p, inst.goals) <= sum_of_costs(p0, inst.goals)
    # zero iterations: the given plan is returned unchanged (up to padding)
    p_same, _ = official_lns(inst, p0, time_limit=1, max_iterations=0)
    assert sum_of_costs(p_same, inst.goals) == sum_of_costs(p0, inst.goals)


@needs_bin
def test_full_grid_rings_official_lns_stuck_local_search_not():
    inst = grid_rings((5, 5), shifts=(3, 4))        # two full rings of 12 cells
    p0 = inst.initial_plan
    for N in (5, 10):
        p, _ = official_lns(inst, p0, time_limit=2, N=N)
        assert makespan(p) == makespan(p0)
    p, _ = local_search(inst, p0, [GroupedClock(R=0, groups="auto", max_groups=4)])
    validate(inst.graph, inst.starts, inst.goals, p)
    assert makespan(p) < makespan(p0)


def test_lns_choice_without_the_official_binary(monkeypatch):
    """Without the binary: directed graphs use the re-implementation, grids
    raise a clear error unless the fallback is enabled."""
    from mapf_bench.official_lns import use_official
    monkeypatch.setenv("MAPF_LNS_BIN", "/nonexistent")
    monkeypatch.delenv("MAPF_BENCH_LNS_FALLBACK", raising=False)
    assert use_official(oneway_grid(6, 6, n=4, seed=0)) is False
    grid = grid_bottleneck(9, 9, K=1, n=4, seed=0)
    with pytest.raises(RuntimeError, match="MAPF_BENCH_LNS_FALLBACK"):
        use_official(grid)
    monkeypatch.setenv("MAPF_BENCH_LNS_FALLBACK", "1")
    assert use_official(grid) is False
