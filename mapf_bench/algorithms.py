# SPDX-License-Identifier: GPL-3.0-or-later
"""Registry of the compared algorithms.

Each entry maps a name to a function (inst, plan, budget, seed) -> (plan, stats).
Pipelines ("a+b") run the second algorithm on the output of the first.
"""
from .lns import mapf_lns
from .official_lns import official_lns
from .cpp_local_search import (cpp_local_search, AdaptiveRadius, Window, FreeSet, Retime,
                               OrderSwap)
from .local_search import (local_search, PathDistance, Pointwise, AgentDistance,
                           Combined, GroupedClock)


def _ls(neighborhoods, tiebreak="summin"):
    def run(inst, plan, budget, seed):
        nbs = neighborhoods() if callable(neighborhoods) else neighborhoods
        return local_search(inst, plan, nbs, tiebreak=tiebreak,
                            time_limit=budget["time"],
                            max_expansions=budget["expansions"])
    run.neighborhoods, run.tiebreak = neighborhoods, tiebreak
    return run


def _cls(neighborhoods, tiebreak="summin"):
    """C++ local search (identical plans, ~10x faster): 10x the state budget."""
    def run(inst, plan, budget, seed):
        nbs = neighborhoods() if callable(neighborhoods) else neighborhoods
        return cpp_local_search(inst, plan, nbs, tiebreak=tiebreak,
                                time_limit=budget["time"],
                                max_expansions=10 * budget["expansions"])
    return run


def all_neighborhoods():
    """Every neighborhood of the paper, from the cheapest to the most expensive."""
    return [Pointwise(1), PathDistance(1), GroupedClock(R=2),
            GroupedClock(R=3, restricted=True),
            GroupedClock(R=1, groups="auto", max_groups=2),
            AgentDistance(1), Combined(R_ag=2, R_sm=1),
            GroupedClock(R=0, groups="auto", max_groups=4)]


def adaptive_neighborhoods():
    """Every neighborhood of the paper with an adaptive radius range."""
    A = AdaptiveRadius
    return [A("pointwise:{R}", 1, 2), A("path:{R}", 1, 3),
            A("gc:{R}:1.0:none:0:2:4", 1, 4),          # route-lag
            A("gc:{R}:1.0:none:1:2:4", 1, 8),          # route-restricted
            A("gc:{R}:1.0:auto:0:2:2", 0, 3),          # grouped clocks, 2 groups
            A("agent:{R}", 1, 3),
            A("combined:{R}:1", 1, 3),                 # R_ag adaptive, R_sm = 1
            A("gc:{R}:1.0:auto:0:2:4", 0, 1)]          # grouped clocks, 4 groups


# grouped-clock neighborhoods in the local search (LS, LSB, LSBX): off. On 18
# instances of groups S, D, H and K (20 s) they never changed the result of LS
# (18 equal) and LSBX was equal on 17 and better without them on one.
GROUPED_CLOCKS = False


def reduced_neighborhoods():
    """The three neighborhoods that the ablation found necessary: route-lag
    (whose neighborhood contains the path and route-restricted ones at equal
    radius), grouped clocks with 4 groups (which refine 2 groups), and the
    agent distance (which contains the combined neighborhood)."""
    A = AdaptiveRadius
    return [A("gc:{R}:1.0:none:0:2:4", 1, 8)] + (          # route-lag
        [A("gc:{R}:1.0:auto:0:2:4", 0, 1)] if GROUPED_CLOCKS else []) + [   # grouped clocks, 4 groups
        A("agent:{R}", 1, 3)]


def window_reduced_neighborhoods():
    """A free window of one round (cheap compressor of slack shared by all
    agents, with its own state budget) followed by the reduced set."""
    return [Window(1, budget=200000)] + reduced_neighborhoods()


def free_reduced_neighborhoods():
    """The reduced set followed by the free-set neighborhood: pairs of agents
    chosen LNS-style move freely, the other agents keep their routes
    (route-restricted, radius 0: a common time shift only)."""
    return reduced_neighborhoods() + [FreeSet(K=2, R=0, restricted=True, candidates=20,
                                              budget=300000)]


def order_reduced_neighborhoods():
    """Order-preserving retiming and order swaps (cheap, not DP), then the
    reduced set of DP neighborhoods."""
    return [Retime(), OrderSwap()] + reduced_neighborhoods()


def order_free_neighborhoods():
    """Retiming, order swaps, the reduced set and the free-set neighborhood."""
    return [Retime(), OrderSwap()] + free_reduced_neighborhoods()


def order_corridor_neighborhoods():
    """Retiming, order swaps, the reduced set and pairs of free agents
    restricted to corridors of width 1 around their routes."""
    return [Retime(), OrderSwap()] + reduced_neighborhoods() + [
        FreeSet(K=2, R=0, restricted=True, candidates=20, budget=300000, corridor=1)]


def lnsx_extra_neighborhoods():
    """The LNS/LaCAM free sets alone (4, then 8 agents; see lnsx_neighborhoods)."""
    return [FreeSet(K=4, R=0, restricted=True, candidates=12, budget=500000, beam=5000),
            FreeSet(K=8, R=0, restricted=True, candidates=12, budget=500000, beam=5000)]


def lnsx_neighborhoods():
    """The complete method plus a neighborhood with the behavior of MAPF-LNS and
    LaCAM: a free set of 4, then 8 agents chosen by LNS destroy heuristics
    (agent-based, intersection-based, random, interleaved), all the other agents
    frozen on their reference, and the free agents replanned jointly in their
    configuration space by A* with a bounded open list (5000 states)."""
    return order_corridor_neighborhoods() + [
        FreeSet(K=4, R=0, restricted=True, candidates=12, budget=500000, beam=5000),
        FreeSet(K=8, R=0, restricted=True, candidates=12, budget=500000, beam=5000)]


BEAM_RADII = "ladder"     # "ladder": radius by radius (default); "max": every beam neighborhood at its
                          # largest radius; "sparse": radii 10/13/16, 4/6, 3 -- both faster but they lose
                          # improvements on the dense directed graphs (see README)


def beam_neighborhoods(n=None):
    """The neighborhoods of order_corridor_neighborhoods with much larger
    radii, meant to be searched with a bounded open list (beam): route-lag up
    to 16, grouped clocks up to 4, up to 6 deviating agents, and free sets of
    three agents in corridors of width 2."""
    # LSB runs this descent only after the exact descent has certified the plan
    # for the radii of reduced_neighborhoods(): a search at a certified radius
    # explores a neighborhood with no better plan, so every neighborhood starts
    # just above its certified radius.
    A = AdaptiveRadius
    exact = {a.template: a.rmax for a in reduced_neighborhoods()}
    exact.setdefault("gc:{R}:1.0:auto:0:2:4", 1)
    rl, gc, ag = "gc:{R}:1.0:none:0:2:4", "gc:{R}:1.0:auto:0:2:4", "agent:{R}"
    if BEAM_RADII in ("opt", "opt13"):
        # optimized ladder, from the radii at which the beam phase found its
        # improvements on 42 instances: agent distance first (cheaper, about
        # as productive per second), then route-lag up to 15 (no improvement
        # at 16) or 13; grouped clocks and free sets never improved and are left out
        top = 15 if BEAM_RADII == "opt" else 13
        return [Retime(), OrderSwap(),
                A(ag, exact[ag] + 1, 6),
                A(rl, exact[rl] + 1, top)]
    if BEAM_RADII == "ladder+jump":  # the full ladder, then a few geometrically larger radii
        cap_ag = max(7, min(12, n if n else 12))
        return [Retime(), OrderSwap(),
                A(rl, exact[rl] + 1, 16, cap=64, gext=True),
                A(gc, exact[gc] + 1, 4, cap=8, gext=True),
                A(ag, exact[ag] + 1, 6, cap=cap_ag, gext=True),
                FreeSet(K=3, R=0, restricted=True, candidates=20, budget=300000, corridor=2)]
    if BEAM_RADII == "extend":       # the ladder, then beyond its limit while the searches are complete
        cap_ag = max(7, min(12, n if n else 12))
        return [Retime(), OrderSwap(),
                A(rl, exact[rl] + 1, 16, cap=64),
                A(gc, exact[gc] + 1, 4, cap=8),
                A(ag, exact[ag] + 1, 6, cap=cap_ag),
                FreeSet(K=3, R=0, restricted=True, candidates=20, budget=300000, corridor=2)]
    if BEAM_RADII == "adaptive":     # geometric growth from the certified radius, larger caps
        return [Retime(), OrderSwap(),
                A(rl, exact[rl] + 1, 64, geometric=True),
                A(gc, exact[gc] + 1, 8, geometric=True),
                A(ag, exact[ag] + 1, max(exact[ag] + 1, min(12, n if n else 12)), geometric=True),
                FreeSet(K=3, R=0, restricted=True, candidates=20, budget=300000, corridor=2)]
    if BEAM_RADII == "sparse":       # a few radii, from small to large, each searched once
        return [Retime(), OrderSwap()] + [A(rl, r, r) for r in (10, 13, 16)] + [A(gc, 3, 3)] + \
               [A(ag, r, r) for r in (4, 6)] + \
               [FreeSet(K=3, R=0, restricted=True, candidates=20, budget=300000, corridor=2)]
    lo = (lambda r_exact, r_max: r_max) if BEAM_RADII == "max" else (lambda r_exact, r_max: r_exact + 1)
    return [Retime(), OrderSwap(),
            A(rl, lo(exact[rl], 16), 16)] + (         # route-lag
        [A(gc, lo(exact[gc], 4), 4)] if GROUPED_CLOCKS else []) + [   # grouped clocks, 4 groups
            A(ag, lo(exact[ag], 6), 6),
            FreeSet(K=3, R=0, restricted=True, candidates=20, budget=300000, corridor=2)]


def _avnd(objective, keep_radius=False, neighborhoods=None, astar=False):
    """VND over all neighborhoods with adaptive radii (C++). With keep_radius,
    a neighborhood keeps the radius at which it last improved."""
    def run(inst, plan, budget, seed):
        nbs = (neighborhoods or adaptive_neighborhoods)()
        return cpp_local_search(inst, plan, nbs, tiebreak="summin",
                                time_limit=budget["time"],
                                max_expansions=10 * budget["expansions"],
                                objective=objective, vnd=True, keep_radius=keep_radius,
                                astar=astar)
    return run


def _vnd(objective):
    """Variable neighborhood descent over all neighborhoods (C++)."""
    def run(inst, plan, budget, seed):
        return cpp_local_search(inst, plan, all_neighborhoods(), tiebreak="summin",
                                time_limit=budget["time"],
                                max_expansions=10 * budget["expansions"],
                                objective=objective, vnd=True)
    return run


def _lns(objective, N=5):
    def run(inst, plan, budget, seed):
        return mapf_lns(inst, plan, objective=objective, N=N,
                        max_iters=budget["lns_iters"], time_limit=budget["time"],
                        seed=seed)
    return run


def _olns(N=5, replan="PP"):
    def run(inst, plan, budget, seed):
        p, st = official_lns(inst, plan, time_limit=budget["time"], N=N,
                             replan=replan, seed=seed)
        return p, st
    return run


ALGORITHMS = {
    # variable neighborhood descent over all the local searches
    "vnd_soc": _vnd("soc_proxy"),       # counter-free search, true-SoC acceptance
    "vnd_soc_exact": _vnd("soc"),       # exact SoC (Appendix A.3), larger state space
    "avnd_soc": _avnd("soc_proxy"),     # VND with adaptive radii
    "avndk_soc": _avnd("soc_proxy", keep_radius=True),  # ... keeping the last radius
    "avndr_soc": _avnd("soc_proxy", neighborhoods=reduced_neighborhoods),  # 3 neighborhoods
    "avndra_soc": _avnd("soc_proxy", neighborhoods=reduced_neighborhoods, astar=True),
    "avndraw_soc": _avnd("soc_proxy", neighborhoods=window_reduced_neighborhoods, astar=True),
    "avndrf_soc": _avnd("soc_proxy", neighborhoods=free_reduced_neighborhoods, astar=True),
    "avndro_soc": _avnd("soc_proxy", neighborhoods=order_reduced_neighborhoods, astar=True),
    "avndrof_soc": _avnd("soc_proxy", neighborhoods=order_free_neighborhoods, astar=True),
    "avndrofc_soc": _avnd("soc_proxy", neighborhoods=order_corridor_neighborhoods, astar=True),
    "avnda_soc": _avnd("soc_proxy", astar=True),
    "avnd_mk": _avnd("makespan"),
    "vnd_mk": _vnd("makespan"),
    # official MAPF-LNS (C++, patched to start from a given plan); grids only
    "olns": _olns(5, "PP"),
    "olns_N10": _olns(10, "PP"),
    "olns_eecbs": _olns(5, "EECBS"),
    # baselines
    "lns_soc": _lns("soc"),           # MAPF-LNS as published (Sum-of-Costs)
    "lns_mk": _lns("makespan"),       # MAPF-LNS with a makespan objective
    "lns_soc_N10": _lns("soc", N=10), # larger destroy sets
    "lns_mk_N10": _lns("makespan", N=10),
    # neighborhoods tested in the paper
    "ls_pointwise": _ls(lambda: [Pointwise(1)], tiebreak=None),
    "ls_agentB": _ls(lambda: [AgentDistance(1)]),
    "ls_alt": _ls(lambda: [Pointwise(1), AgentDistance(1)]),
    "ls_combined": _ls(lambda: [Combined(R_ag=2, R_sm=1)]),
    # new distances (route-lag, route-restricted, grouped clocks); the number
    # of groups is capped at 2, since the neighborhood grows like (T+1)^groups
    "ls_lag": _ls(lambda: [GroupedClock(R=2)]),
    "ls_rr": _ls(lambda: [GroupedClock(R=3, restricted=True)]),
    "ls_rr6": _ls(lambda: [GroupedClock(R=6, restricted=True)]),
    "ls_gc": _ls(lambda: [GroupedClock(R=1, groups="auto", max_groups=2)]),
    "ls_gc0": _ls(lambda: [GroupedClock(R=0, groups="auto", max_groups=4)]),
    "ls_new_alt": _ls(lambda: [GroupedClock(R=3, restricted=True),
                               GroupedClock(R=1, groups="auto", max_groups=2),
                               Combined(R_ag=1, R_sm=0)]),
}


# C++ versions of every local search: "cls_<x>" mirrors "ls_<x>"
for _name in [k for k in ALGORITHMS if k.startswith("ls_")]:
    _f = ALGORITHMS[_name]
    ALGORITHMS["c" + _name] = _cls(_f.neighborhoods, _f.tiebreak)


def run_algorithm(name, inst, plan, budget, seed=0):
    stats_all = {}
    for part in name.split("+"):
        plan, stats = ALGORITHMS[part](inst, plan, budget, seed)
        stats_all[part] = stats
    return plan, stats_all


DEFAULT_ALGOS = ["lns_soc", "lns_mk", "ls_combined", "ls_alt", "ls_lag",
                 "ls_rr", "ls_gc", "ls_new_alt",
                 "lns_soc+ls_combined", "lns_soc+ls_new_alt"]
