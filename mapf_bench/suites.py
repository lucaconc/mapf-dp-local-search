# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark suites: lists of (instance, initial-plan method)."""
from . import instances as I


def _grid(seeds, Ks=(0, 1, 2), n=8, init="pp"):
    return [(I.grid_bottleneck(9, 9, K=K, n=n, seed=s), init)
            for K in Ks for s in seeds]


SUITES = {
    # the two constructions of the paper (hand-built reference plans)
    "examples": lambda seeds: (
        [(I.merge(k, ell), "given") for k in (1, 2, 3) for ell in (4, 8)]
        + [(I.ring(L, L + 2, q), "given")
           for (L, q) in ((5, 2), (6, 3), (8, 4), (10, 5))]),
    # grid maps with bottlenecks, as in Section 5.3 (PP initial plans)
    "grid": lambda seeds: _grid(seeds),
    # same maps, initial plans serialized as by a rule-based solver
    "grid_serial": lambda seeds: _grid(seeds, init="pp_serial"),
    # one-way Manhattan grids
    "oneway": lambda seeds: [(I.oneway_grid(6, 6, n=n, seed=s), init)
                             for n in (6, 10) for s in seeds
                             for init in ("pp", "pp_serial")],
    # random strongly connected digraphs, as in Section 5.2
    "random": lambda seeds: [(I.random_digraph(30, 3, n=n, seed=s), init)
                             for n in (6, 10) for s in seeds
                             for init in ("pp", "pp_serial")],
    # dense one-way rings joined by bridges (independent groups)
    "multiring": lambda seeds: [(I.multi_ring(r, 6, seed=s), "pp_serial")
                                for r in (2, 3) for s in seeds],
}

SUITE_GROUPS = {
    "quick": ["examples"],
    "all": ["examples", "grid", "grid_serial", "oneway", "random", "multiring"],
}


def build(names, seeds):
    out = []
    for name in names:
        for sub in SUITE_GROUPS.get(name, [name]):
            out += [(sub, inst, init) for inst, init in SUITES[sub](seeds)]
    return out
