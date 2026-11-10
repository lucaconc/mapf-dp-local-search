# SPDX-License-Identifier: GPL-3.0-or-later
"""Speed-up of the parallel candidate search (run on a multi-core machine).

    python scripts/bench_threads.py [THREADS] [--group H K] [--time 120]

For every instance, the local search with the LNS/LaCAM neighborhoods
(algorithms.lnsx_neighborhoods) runs to its local optimum (or the time limit)
with 1 thread and with THREADS threads; it prints the times, the speed-up and
the Sum-of-Costs (which may differ: the parallel search accepts the best
improvement of each batch of candidates, the sequential one the first).
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mapf_bench.paper_suite import scenarios
from mapf_bench.cpp_local_search import cpp_local_search
from mapf_bench.algorithms import lnsx_neighborhoods
from mapf_bench.plan import trim, sum_of_costs, validate

args = sys.argv[1:]
T = int(args[0]) if args and args[0].isdigit() else max(2, (os.cpu_count() or 2) - 1)
groups = ("H", "K")
if "--group" in args:
    i = args.index("--group"); groups = tuple(x for x in args[i + 1:] if not x.startswith("--"))
tl = float(args[args.index("--time") + 1]) if "--time" in args else 120.0
kw = dict(tiebreak="summin", objective="soc_proxy", vnd=True, astar=True, keep_radius=True,
          time_limit=tl, max_expansions=2_000_000, chain=True)
tot = [0.0, 0.0]
print(f"threads: {T}; groups: {', '.join(groups)}; time limit {tl} s")
for g, fam, b in scenarios():
    if g not in groups: continue
    inst, _ = b()
    if inst.initial_plan is None: continue
    p0 = trim(inst.initial_plan); row = []
    for k, t in enumerate((1, T)):
        t0 = time.perf_counter()
        p, st = cpp_local_search(inst, p0, lnsx_neighborhoods(), threads=t, **kw)
        dt = time.perf_counter() - t0; tot[k] += dt
        validate(inst.graph, inst.starts, inst.goals, p)
        row.append((dt, sum_of_costs(p, inst.goals)))
    print(f"{inst.name:30s} 1 thread {row[0][0]:7.1f}s SoC {row[0][1]:6d} | {T} threads {row[1][0]:7.1f}s SoC {row[1][1]:6d}"
          f" | speed-up {row[0][0] / max(row[1][0], 1e-9):4.2f}x", flush=True)
print(f"total {tot[0]:.1f}s -> {tot[1]:.1f}s, speed-up {tot[0] / max(tot[1], 1e-9):.2f}x")
