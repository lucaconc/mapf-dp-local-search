# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark of the C++ local search with a fixed amount of work.

    python scripts/bench_cpp.py prepare              # input files in bench_cpp/
    python scripts/bench_cpp.py run cpp/dpls [cpp/dpls_old]

Every instance is searched with the configuration of the paper and a fixed
number of neighborhood searches (--max-searches) instead of a time limit,
so that two binaries do exactly the same work: their running times are
comparable and their plans must be identical.
"""
import hashlib, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "bench_cpp")
NAMES = ["empty-32-32_gen1_n40", "random-32-32-20_random1_n40", "room-32-32-4_gen1_n40",
         "maze-32-32-4_gen1_n40", "empty-8-8_o0.90_1", "empty-16-16_o0.80_1",
         "dense_random_d0.9_s9", "dense_oneway_d0.9_s4", "reversal_m6_p4_s1",
         "crossroads_a10_k4_s1", "corridor_swap_L20_b3_m6_s1", "fullrings_12_12_h0_s0"]
SEARCHES = 12


def prepare():
    from mapf_bench.paper_suite import scenarios
    from mapf_bench.plan import trim
    os.makedirs(D, exist_ok=True)
    found = set()
    for g, fam, build in scenarios():
        if not any(fam.split("_")[0] in n or fam in n for n in NAMES):
            continue
        inst, _ = build()
        if inst.name not in NAMES or inst.initial_plan is None:
            continue
        G = inst.graph; plan = trim(inst.initial_plan)
        arcs = [(u, v) for u in range(G.N) for v in G.out[u] if v != u]
        with open(os.path.join(D, inst.name + ".txt"), "w") as fh:
            fh.write(f"{G.N} {len(arcs)}\n" + "".join(f"{u} {v}\n" for u, v in arcs))
            fh.write(f"{inst.n}\n{' '.join(map(str, inst.starts))}\n{' '.join(map(str, inst.goals))}\n{len(plan) - 1}\n")
            fh.write("".join(" ".join(map(str, x)) + "\n" for x in plan))
        found.add(inst.name)
        print("prepared", inst.name, inst.n, "agents", G.N, "nodes", len(plan) - 1, "steps", flush=True)
    print("missing:", sorted(set(NAMES) - found))


def args():
    from mapf_bench.algorithms import order_corridor_neighborhoods
    from mapf_bench.cpp_local_search import spec
    a = ["--time", "100000", "--expansions", "2000000", "--max-searches", str(SEARCHES), "--tiebreak", "summin",
         "--objective", "soc_proxy", "--vnd", "--keep-radius", "--astar"]
    for nb in order_corridor_neighborhoods():
        a += ["--nb", spec(nb)]
    return a


def run_one(binary, name):
    fin = os.path.join(D, name + ".txt"); fout = os.path.join(D, name + "." + os.path.basename(binary) + ".out")
    t = time.perf_counter()
    r = subprocess.run([binary, fin, fout] + args(), capture_output=True, text=True)
    dt = time.perf_counter() - t
    if r.returncode != 0:
        raise SystemExit(f"{binary} failed on {name}: {r.stderr[-300:]}")
    return dt, hashlib.md5(open(fout, "rb").read()).hexdigest()


def run(binaries, only=None):
    names = sorted(f[:-4] for f in os.listdir(D) if f.endswith(".txt"))
    if only:
        names = [n for n in names if any(o in n for o in only)]
    tot = [0.0] * len(binaries)
    print(f"{'instance':32s} " + " ".join(f"{os.path.basename(b):>14s}" for b in binaries) + "  identical")
    for n in names:
        res = [run_one(b, n) for b in binaries]
        for i, (dt, _) in enumerate(res): tot[i] += dt
        same = len({h for _, h in res}) == 1
        print(f"{n:32s} " + " ".join(f"{dt:13.2f}s" for dt, _ in res) + f"  {same}", flush=True)
    print(f"{'total':32s} " + " ".join(f"{t:13.2f}s" for t in tot)
          + ("" if len(tot) < 2 else f"   the second is {tot[0] / max(tot[1], 1e-9):.1f}x faster"))


if __name__ == "__main__":
    # run BIN [BIN2] [--only NAME ...] [--searches K]
    a = sys.argv[2:]
    only = None
    if "--searches" in a:
        i = a.index("--searches"); SEARCHES = int(a[i + 1]); del a[i:i + 2]
    if "--only" in a:
        i = a.index("--only"); only = a[i + 1:]; a = a[:i]
    prepare() if sys.argv[1] == "prepare" else run(a, only)
