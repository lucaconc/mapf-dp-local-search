# SPDX-License-Identifier: GPL-3.0-or-later
"""Example animations of the README (docs/animations/).

    python scripts/make_animations.py

Two rings on a grid (perimeters of 4x6 rectangles) joined by a corridor of
three cells: 10 agents per ring, 8 of which rotate four positions along their
ring and 2 cross to the other ring. The initial plan is prioritized planning
serialized (one agent moves at a time, as in rule-based solvers); the plans of
MAPF-LNS (the re-implementation of the package, 10 s) and of the local search
(LS, 10 s) start from it. Every plan is validated.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mapf_bench.instances import Instance, grid_from_cells          # noqa: E402
from mapf_bench.lns import mapf_lns                                 # noqa: E402
from mapf_bench.paper_suite import run_method                       # noqa: E402
from mapf_bench.plan import paths_to_plan, serialize, trim, sum_of_costs, makespan, validate  # noqa: E402
from mapf_bench.plan_store import save_plan, load                   # noqa: E402
from mapf_bench.pp import initial_pp                                # noqa: E402
from mapf_bench.visualize import write_html, write_gif              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "animations")


def perimeter(r0, r1, c0, c1):
    top = [(r0, c) for c in range(c0, c1 + 1)]
    right = [(r, c1) for r in range(r0 + 1, r1 + 1)]
    bottom = [(r1, c) for c in range(c1 - 1, c0 - 1, -1)]
    left = [(r, c0) for r in range(r1 - 1, r0, -1)]
    return top + right + bottom + left


def two_rings_crossing(seed=10, n_ring=10, n_cross=2, shift=4):
    A = perimeter(0, 3, 0, 5)
    B = perimeter(0, 3, 9, 14)
    cells = set(A) | set(B) | {(1, 6), (1, 7), (1, 8)}
    rng = random.Random(seed)
    starts, goals, used = [], [], set()
    for ring in (A, B):
        idx = sorted(rng.sample(range(len(ring)), n_ring))
        cross = set(rng.sample(idx, n_cross))
        for i in idx:
            starts.append(ring[i])
            g = None if i in cross else ring[(i + shift) % len(ring)]
            goals.append(g)
            if g is not None:
                used.add(g)
    for k, (s, g) in enumerate(zip(starts, goals)):
        if g is None:                                   # crossing agents: a free cell of the other ring
            g = rng.choice([c for c in (B if s in A else A) if c not in used])
            goals[k] = g
            used.add(g)
    G = grid_from_cells(cells, "two_rings_crossing")
    idx = {rc: v for v, rc in G.coords.items()}
    return Instance(f"two_rings_crossing_s{seed}", "two_rings_crossing", G,
                    tuple(idx[s] for s in starts), tuple(idx[g] for g in goals), {"cells": sorted(cells)})


def main():
    inst = two_rings_crossing()
    paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=0, attempts=60)
    plans = {"initial": trim(serialize(inst.graph, paths_to_plan(paths)))}
    p, _ = mapf_lns(inst, plans["initial"], objective="soc", N=5, max_iters=10 ** 6, time_limit=10, seed=1)
    plans["LNS"] = trim(p)
    p, _ = run_method("LS", inst, plans["initial"], 10.0, 1)
    plans["LS"] = trim(p)
    os.makedirs(OUT, exist_ok=True)
    store = os.path.join(OUT, "rings_crossing.plans.xz")
    if os.path.exists(store):
        os.remove(store)
    for k, p in plans.items():
        validate(inst.graph, inst.starts, inst.goals, p)
        save_plan(store, inst, k, p)
        print(f"{k:8s} Sum-of-Costs {sum_of_costs(p, inst.goals):5d}, makespan {makespan(p):4d}")
    data = load(store)
    write_html(data, os.path.join(OUT, "rings_crossing.html"), methods=["initial", "LNS", "LS"], ref="initial",
               title="Two rings with crossing agents: initial plan, MAPF-LNS and the local search")
    write_gif(data, os.path.join(OUT, "rings_crossing.gif"), methods=["LNS", "LS"], substeps=2, fps=10)
    os.remove(store)
    print("written in", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
