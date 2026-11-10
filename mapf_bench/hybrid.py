# SPDX-License-Identifier: GPL-3.0-or-later
"""Hybrid of MAPF-LNS and DP-based local search (objective: Sum-of-Costs).

The local search (a VND over the paper's neighborhoods) is run to local
optimality; then short LNS phases (destroy/repair, strict Sum-of-Costs
acceptance) are used as perturbation heuristics. An LNS improvement
invalidates local optimality, so the local search is run again from the
improved plan. The algorithm stops after `max_fail` consecutive LNS phases
without improvement, or when the time is over, and always ends with a local
search run (time is reserved for it), so the returned plan is the output of
the local search. `certified` is True when that last run stopped because no
neighborhood improves the plan (not because of the time limit or of a state
budget): the plan is then a local optimum of every neighborhood (for the DP
neighborhoods, with respect to the search cost; exact for retiming and swaps).
"""
import time

from .algorithms import order_corridor_neighborhoods, order_reduced_neighborhoods
from .cpp_local_search import cpp_local_search
from .lns import mapf_lns
from .official_lns import is_grid_instance, official_lns
from .plan import sum_of_costs, trim


def _local_search(inst, plan, time_limit, neighborhoods, expansions, keep_radius=False):
    p, st = cpp_local_search(inst, plan, neighborhoods(), tiebreak="summin",
                             objective="soc_proxy", vnd=True, astar=True,
                             time_limit=max(time_limit, 0.05), max_expansions=expansions,
                             keep_radius=keep_radius, chain=True)
    certified = ((not st.get("timeout")) and st.get("budget_exceeded", 0) == 0
                 and st.get("pruned", 0) == 0)
    return trim(p), st, certified


def _lns_phase(inst, plan, t, seed, stagnation=0):
    from .official_lns import use_official
    if use_official(inst):
        p, st = official_lns(inst, plan, time_limit=t, N=5, seed=seed, stagnation=stagnation)
    else:
        p, st = mapf_lns(inst, plan, objective="soc", N=5, max_iters=100000,
                         time_limit=t, seed=seed, stagnation=stagnation // 2)
    return trim(p)


def hybrid(inst, plan, time_limit=20.0, lns_phase=2.0, max_fail=2, seed=0,
           neighborhoods=order_corridor_neighborhoods, expansions=2_000_000,
           final_reserve=0.25, lns_first=False, final_neighborhoods=None,
           stagnation=0, keep_radius=False, adaptive_stagnation=False):
    """Returns (plan, stats). final_reserve: share of the time kept for the
    final local search run. lns_first: start with an LNS phase instead of a
    local search run. final_neighborhoods: neighborhoods of the final,
    certifying run (default: the same as during the alternation); with
    cheaper `neighborhoods`, the final run certifies local optimality for
    the larger set. stagnation: LNS phases stop after this many iterations
    without improvement (0: they last lns_phase seconds). keep_radius: the
    adaptive radii keep their last successful value (faster).
    adaptive_stagnation: every LNS phase without improvement doubles the
    stagnation limit of the next one (reset after an improvement), so hard
    instances get long phases and easy ones stay fast."""
    final_neighborhoods = final_neighborhoods or neighborhoods
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    reserve = final_reserve * time_limit
    goals = inst.goals
    stats = {"phases": 0, "lns_improvements": 0, "ls_runs": 0, "ls_improvements": 0,
             "time_ls": 0.0, "time_lns": 0.0}
    if lns_first:
        cur, certified = trim(plan), False
    else:
        t = time.perf_counter()
        cur, st, certified = _local_search(inst, plan, deadline - reserve - t, neighborhoods,
                                           expansions, keep_radius)
        stats["time_ls"] += time.perf_counter() - t
        stats["ls_runs"] += 1
        stats["ls_improvements"] += st.get("improvements", 0)
    fails = 0
    stag = stagnation
    while fails < max_fail and time.perf_counter() + lns_phase < deadline - reserve:
        t = time.perf_counter()
        new = _lns_phase(inst, cur, lns_phase, seed + 1000 * stats["phases"], stag)
        stats["time_lns"] += time.perf_counter() - t
        stats["phases"] += 1
        if sum_of_costs(new, goals) < sum_of_costs(cur, goals):
            stats["lns_improvements"] += 1
            fails = 0
            stag = stagnation
            t = time.perf_counter()
            cur, st, certified = _local_search(inst, new, deadline - reserve - t,
                                               neighborhoods, expansions, keep_radius)
            stats["time_ls"] += time.perf_counter() - t
            stats["ls_runs"] += 1
            stats["ls_improvements"] += st.get("improvements", 0)
        else:
            fails += 1
            if adaptive_stagnation and stag:
                stag *= 2
    if not certified or final_neighborhoods is not neighborhoods:
        # final local search (larger neighborhood set) with the remaining time
        t = time.perf_counter()
        cur, st, certified = _local_search(inst, cur, deadline - t, final_neighborhoods,
                                           expansions, keep_radius)
        stats["time_ls"] += time.perf_counter() - t
        stats["ls_runs"] += 1
        stats["ls_improvements"] += st.get("improvements", 0)
    stats["certified"] = certified
    stats["time"] = time.perf_counter() - t0
    return cur, stats
