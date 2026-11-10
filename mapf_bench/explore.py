# SPDX-License-Identifier: GPL-3.0-or-later
"""Sweeps over the families designed around the weaknesses of MAPF-LNS.

    python -m mapf_bench.explore --family rings --out explore
"""
import argparse, csv, os, random, time
from .instances import (full_rings, convoy, random_walk_instance, grid_graph,
                        random_digraph, grid_rings, grid_bottleneck)
from .pp import initial_pp
from .plan import paths_to_plan, serialize, trim


def _with_pp(inst, seed, serial):
    paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=seed)
    plan = paths_to_plan(paths)
    inst.initial_plan = trim(serialize(inst.graph, plan) if serial else plan)
    return inst

DENSITIES = (0.5, 0.6, 0.7, 0.8, 0.9)


def _dense(G, seed, family):
    d = DENSITIES[seed % len(DENSITIES)]
    n = max(2, min(G.N - 1, round(d * G.N)))
    return random_walk_instance(G, n, steps=30, seed=seed, family=family,
                                name=f"{family}_d{d:.1f}_s{seed}")
from .plan import validate, makespan, sum_of_costs
from .algorithms import run_algorithm

FAMILIES = {
    # full rings longer than the destroy set; rotate one ring at a time
    "rings": lambda seed: full_rings(
        lengths=tuple(random.Random(seed).choice((6, 8, 10, 12)) for _ in range(2 + seed % 2)),
        shifts=tuple(random.Random(seed + 100 + j).randrange(2, 5) for j in range(2 + seed % 2)),
        seed=seed),
    # full rings plus free agents on the hub, which move first
    "rings_hub": lambda seed: full_rings(
        lengths=tuple(random.Random(seed).choice((6, 8, 10, 12)) for _ in range(2)),
        shifts=tuple(random.Random(seed + 100 + j).randrange(2, 5) for j in range(2)),
        bridge=10, hub_agents=3, seed=seed),
    # dense instances, initial plan from a backward random walk
    "dense_grid": lambda seed: _dense(grid_graph(4, 5), seed, "dense_grid"),
    "dense_oneway": lambda seed: _dense(grid_graph(4, 4, oneway=True), seed, "dense_oneway"),
    "dense_random": lambda seed: _dense(random_digraph(16, 3, n=2, seed=2).graph, seed,
                                        "dense_random"),
    # fully occupied rings on undirected grids (testable with the official LNS)
    "grid_rings": lambda seed: grid_rings(
        widths=tuple(random.Random(seed).choice((3, 4, 5, 6)) for _ in range(2 + seed % 2)),
        shifts=tuple(random.Random(seed + 7 + j).randrange(2, 5) for j in range(2 + seed % 2)),
        hub_agents=(2 if seed % 4 >= 2 else 0), seed=seed),
    # Section 5.3-like grids with bottlenecks, PP and serialized PP plans
    "grid_pp": lambda seed: _with_pp(grid_bottleneck(9, 9, K=seed % 3, n=8, seed=seed), seed, False),
    "grid_serial": lambda seed: _with_pp(grid_bottleneck(9, 9, K=seed % 3, n=8, seed=seed), seed, True),
    # merges where long-haul agents queue behind short-haul ones
    "convoy": lambda seed: convoy(n_long=1 + seed % 3, n_short=2 + seed % 4,
                                  ell=4 + 2 * (seed % 3), seed=seed),
}
ALGOS = ["lns_soc", "lns_mk", "lns_soc_N10", "lns_mk_N10", "ls_gc0", "ls_gc",
         "ls_rr", "ls_new_alt", "lns_soc+ls_new_alt"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", nargs="+", default=list(FAMILIES))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=8)
    ap.add_argument("--out", default="explore")
    ap.add_argument("--algos", nargs="+", default=ALGOS)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.summary_only:
        return summarize(os.path.join(a.out, "explore.csv"), a.out)
    b = {"time": a.time_limit, "lns_iters": 400, "expansions": 300000}
    path = os.path.join(a.out, "explore.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["family", "instance", "n", "V", "algo", "makespan", "soc",
                        "makespan0", "soc0", "time"])
        for fam in a.family:
            for seed in range(a.seed_start, a.seed_start + a.seeds):
                inst = FAMILIES[fam](seed)
                p0 = inst.initial_plan
                validate(inst.graph, inst.starts, inst.goals, p0)
                mk0, soc0 = makespan(p0), sum_of_costs(p0, inst.goals)
                line = [f"init {mk0}"]
                for algo in a.algos:
                    t = time.perf_counter()
                    p, _ = run_algorithm(algo, inst, p0, b, seed)
                    dt = time.perf_counter() - t
                    validate(inst.graph, inst.starts, inst.goals, p)
                    w.writerow([fam, inst.name, inst.n, inst.graph.N, algo, makespan(p),
                                sum_of_costs(p, inst.goals), mk0, soc0, round(dt, 3)])
                    line.append(f"{algo} {makespan(p)}")
                fh.flush()
                print(fam, inst.name, " ".join(line), flush=True)


def summarize(path, out_dir):
    """Per family: mean makespan of every algorithm and wins/ties/losses of
    each local search against the published MAPF-LNS (lns_soc) and against
    the strongest LNS variant on each instance."""
    from collections import defaultdict
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(dict)
    for r in rows:
        by[(r["family"], r["instance"])][r["algo"]] = r
    fams = sorted({f for f, _ in by})
    lns = [x for x in ("lns_soc", "lns_mk", "lns_soc_N10", "lns_mk_N10")]
    lines = ["# Instances where local search beats MAPF-LNS", "",
             "Makespan; `best LNS` = best of the four LNS variants on each "
             "instance (Sum-of-Costs or makespan objective, destroy sets of 5 "
             "or 10 agents). W/T/L = shorter / equal / longer makespan.", ""]
    for f in fams:
        insts = [k for k in by if k[0] == f]
        algos = sorted({a for k in insts for a in by[k]})
        lines += [f"## {f} ({len(insts)} instances)", "",
                  "| algorithm | mean makespan | W/T/L vs lns_soc | W/T/L vs best LNS |",
                  "|---|---:|:---:|:---:|"]
        mk0 = sum(int(by[k][next(iter(by[k]))]["makespan0"]) for k in insts) / len(insts)
        lines.append(f"| initial plan | {mk0:.2f} | | |")
        for a in algos:
            ks = [k for k in insts if a in by[k]]
            mean = sum(int(by[k][a]["makespan"]) for k in ks) / len(ks)
            def wtl(ref):
                w = t = l = 0
                for k in ks:
                    refs = [int(by[k][x]["makespan"]) for x in ref if x in by[k]]
                    if not refs:
                        continue
                    m, rm = int(by[k][a]["makespan"]), min(refs)
                    w, t, l = w + (m < rm), t + (m == rm), l + (m > rm)
                return f"{w}/{t}/{l}"
            row = (f"| {a} | {mean:.2f} | " + ("" if a in lns else wtl(["lns_soc"]))
                   + " | " + ("" if a in lns else wtl(lns)) + " |")
            lines.append(row)
        lines.append("")
    with open(os.path.join(out_dir, "explore_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()


def density_table(path, out_dir):
    """Mean makespan by family and density for the dense families."""
    from collections import defaultdict
    rows = [r for r in csv.DictReader(open(path)) if r["family"].startswith("dense")]
    mk = defaultdict(lambda: defaultdict(list))
    best_lns = defaultdict(list)
    per_inst = defaultdict(dict)
    for r in rows:
        per_inst[(r["family"], r["instance"])][r["algo"]] = int(r["makespan"])
        per_inst[(r["family"], r["instance"])]["init"] = int(r["makespan0"])
    cols = ["init", "lns_soc", "best LNS", "ls_rr", "ls_new_alt", "ls_rr+lns_soc"]
    wins = defaultdict(lambda: [0, 0, 0])
    for (fam, inst), d in per_inst.items():
        dens = inst.split("_d")[1][:3]
        lnsv = [d[a] for a in ("lns_soc", "lns_mk", "lns_soc_N10") if a in d]
        d["best LNS"] = min(lnsv)
        for c in cols:
            if c in d:
                mk[(fam, dens)][c].append(d[c])
        m = d["ls_new_alt"]
        w = wins[(fam, dens)]
        w[0 if m < d["best LNS"] else 1 if m == d["best LNS"] else 2] += 1
    lines = ["# Dense instances: mean makespan by density", "",
             "Initial plan from a backward random walk of 30 steps. `best LNS` = "
             "best of MAPF-LNS with Sum-of-Costs (destroy sets of 5 and 10) and "
             "with makespan objective. W/T/L: `ls_new_alt` against best LNS. "
             "`ls_rr+lns_soc`: local search first, then MAPF-LNS (only at "
             "densities 0.8 and 0.9).", "",
             "| family | density | " + " | ".join(cols) + " | W/T/L |",
             "|---|---:|" + "---:|" * len(cols) + ":---:|"]
    for (fam, dens) in sorted(mk):
        vals = []
        for c in cols:
            xs = mk[(fam, dens)].get(c)
            vals.append(f"{sum(xs)/len(xs):.1f}" if xs else "")
        w = wins[(fam, dens)]
        lines.append(f"| {fam} | {dens} | " + " | ".join(vals) + f" | {w[0]}/{w[1]}/{w[2]} |")
    with open(os.path.join(out_dir, "dense_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def official_summary(path, out_dir, ref="olns", lns=("olns", "olns_N10", "olns_eecbs")):
    """Summary of the comparison with the official MAPF-LNS (duplicates
    from interrupted runs are dropped, keeping the last row)."""
    from collections import defaultdict
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(dict)
    for r in rows:
        by[(r["family"], r["instance"])][r["algo"]] = r
    groups = defaultdict(list)
    for (fam, inst) in by:
        key = fam
        if fam.startswith("dense"):
            key = f"{fam} d={inst.split('_d')[1][:3]}"
        groups[key].append((fam, inst))
    lines = ["# Local search vs the official MAPF-LNS", "",
             "Mean makespan. `olns`: official MAPF-LNS, default configuration "
             "(PP repair, destroy sets of 5); `olns_N10`: destroy sets of 10; "
             "`olns_eecbs`: EECBS repair. W/T/L = shorter / equal / longer "
             f"makespan than `{ref}` and than the best official variant on "
             "each instance.", ""]
    for key in sorted(groups):
        ks = groups[key]
        algos = sorted({a for k in ks for a in by[k]})
        mk0 = sum(int(by[k][next(iter(by[k]))]["makespan0"]) for k in ks) / len(ks)
        lines += [f"## {key} ({len(ks)} instances, initial makespan {mk0:.1f})", "",
                  f"| algorithm | mean makespan | W/T/L vs {ref} | W/T/L vs best official |",
                  "|---|---:|:---:|:---:|"]
        for a in algos:
            kk = [k for k in ks if a in by[k]]
            mean = sum(int(by[k][a]["makespan"]) for k in kk) / len(kk)

            def wtl(refs):
                w = t = l = 0
                for k in kk:
                    vals = [int(by[k][x]["makespan"]) for x in refs if x in by[k]]
                    if not vals:
                        continue
                    m, rm = int(by[k][a]["makespan"]), min(vals)
                    w, t, l = w + (m < rm), t + (m == rm), l + (m > rm)
                return f"{w}/{t}/{l}"
            off = a in lns
            lines.append(f"| {a} | {mean:.2f} | {'' if off else wtl([ref])} | "
                         f"{'' if off else wtl(list(lns))} |")
        lines.append("")
    with open(os.path.join(out_dir, "official_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines)
