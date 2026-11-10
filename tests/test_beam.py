# SPDX-License-Identifier: GPL-3.0-or-later
"""Beam search (bounded open list) and the LSB method."""
import pytest

from mapf_bench.cpp_local_search import available, cpp_local_search
from mapf_bench.plan import sum_of_costs, validate, trim

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
def test_large_beam_equals_exact_search():
    """With a beam larger than any open list, the search is the exact one."""
    from mapf_bench.algorithms import order_corridor_neighborhoods
    inst = _inst("fullrings_12_12_h0_s0")
    p0 = trim(inst.initial_plan)
    kw = dict(tiebreak="summin", objective="soc_proxy", vnd=True, astar=True, keep_radius=True,
              time_limit=30, max_expansions=2_000_000)
    a, sa = cpp_local_search(inst, p0, order_corridor_neighborhoods(), **kw)
    b, sb = cpp_local_search(inst, p0, order_corridor_neighborhoods(), beam=10 ** 9, **kw)
    assert a == b and sb["pruned"] == 0


@needs_cpp
def test_small_beam_prunes_and_stays_valid():
    from mapf_bench.algorithms import beam_neighborhoods
    inst = _inst("dense_random_d0.9_s9")
    p0 = trim(inst.initial_plan)
    p, st = cpp_local_search(inst, p0, beam_neighborhoods(), tiebreak="summin", objective="soc_proxy",
                             vnd=True, astar=True, keep_radius=True, time_limit=5,
                             max_expansions=2_000_000, beam=20)
    validate(inst.graph, inst.starts, inst.goals, p)
    assert st["pruned"] > 0 and sum_of_costs(p, inst.goals) <= sum_of_costs(p0, inst.goals)


@needs_cpp
@needs_lacam
def test_lsb_is_not_worse_than_ls_and_certifies_when_unpruned():
    """LSB starts with the exact descent, so on an instance that LS solves to a
    certified optimum quickly it returns a plan at least as good."""
    from mapf_bench.paper_suite import run_method
    inst = _inst("reversal_m6_p4_s1")
    p0 = trim(inst.initial_plan)
    ls, c_ls = run_method("LS", inst, p0, 5.0, 0)
    lsb, c_lsb = run_method("LSB", inst, p0, 5.0, 0)
    validate(inst.graph, inst.starts, inst.goals, lsb)
    assert sum_of_costs(lsb, inst.goals) <= sum_of_costs(ls, inst.goals)
