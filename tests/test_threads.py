# SPDX-License-Identifier: GPL-3.0-or-later
"""Parallel search of the candidates of the free-set neighborhoods (--threads)."""
import pytest

from mapf_bench.cpp_local_search import available, cpp_local_search
from mapf_bench.plan import sum_of_costs, trim, validate

needs_cpp = pytest.mark.skipif(not available(), reason="C++ local search not built")
from mapf_bench.lacam import available as lacam_available
# the deadlock-prone instances (group K) are built from the first LaCAM3 solution
needs_lacam = pytest.mark.skipif(not lacam_available(), reason="LaCAM3 not built")


def _inst(name):
    from mapf_bench.paper_suite import scenarios
    for g, fam, b in scenarios():
        if name.startswith(fam.split("_")[0]) or fam in name:
            inst, _ = b()
            if inst.name == name:
                return inst
    raise KeyError(name)


@needs_cpp
@needs_lacam
def test_parallel_candidates_valid_and_deterministic():
    from mapf_bench.algorithms import lnsx_neighborhoods
    inst = _inst("reversal_m6_p3_s1")
    p0 = trim(inst.initial_plan)
    kw = dict(tiebreak="summin", objective="soc_proxy", vnd=True, astar=True, keep_radius=True,
              time_limit=60, max_expansions=2_000_000, chain=True)
    runs = [cpp_local_search(inst, p0, lnsx_neighborhoods(), threads=t, **kw)[0] for t in (1, 3, 3)]
    for p in runs:
        validate(inst.graph, inst.starts, inst.goals, p)
    assert runs[1] == runs[2]                       # independent of the thread timing
    assert sum_of_costs(runs[1], inst.goals) <= sum_of_costs(p0, inst.goals)


@needs_cpp
def test_one_thread_is_the_sequential_search():
    from mapf_bench.algorithms import order_corridor_neighborhoods
    inst = _inst("fullrings_12_12_h0_s0")
    p0 = trim(inst.initial_plan)
    kw = dict(tiebreak="summin", objective="soc_proxy", vnd=True, astar=True, keep_radius=True,
              time_limit=30, max_expansions=2_000_000)
    a = cpp_local_search(inst, p0, order_corridor_neighborhoods(), **kw)[0]
    b = cpp_local_search(inst, p0, order_corridor_neighborhoods(), threads=1, **kw)[0]
    assert a == b
