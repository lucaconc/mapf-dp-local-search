# SPDX-License-Identifier: GPL-3.0-or-later
"""Variable neighborhood descent over all local searches (VND) vs MAPF-LNS.

    python -m mapf_bench.compare_vnd run --instances 5
    python -m mapf_bench.compare_vnd summary

Every method minimizes the Sum-of-Costs. `vnd`: VND over all local searches
with fixed radii; `avnd`: the same with adaptive radii (a radius is increased
when a search fails within its budget). On 4-connected undirected grids the
LNS is the official MAPF-LNS (default configuration: PP repair, destroy sets
of 5); on directed graphs, where the official code does not apply, it is the
re-implementation (mapf_bench/lns.py). Strategies: LNS, VND, LNS then VND,
VND then LNS. Results are appended to <out>/vnd.csv (runs resume).
"""
import argparse
import csv
import os
import random
import time
from collections import defaultdict

from .algorithms import run_algorithm
from .explore import FAMILIES as EXPLORE
from .instances import (random_digraph, oneway_grid, warehouse, rooms, maze,
                        random_obstacles)
from .official_lns import is_grid_instance, official_lns
from .lns import mapf_lns
from .plan import validate, makespan, sum_of_costs, trim
from .explore import _with_pp

FAMILIES = {
    # undirected grids: official MAPF-LNS
    "grid_pp": EXPLORE["grid_pp"],
    "grid_serial": EXPLORE["grid_serial"],
    "grid_rings": EXPLORE["grid_rings"],
    "dense_grid": EXPLORE["dense_grid"],
    # MovingAI-style map types (initial plan: prioritized planning)
    "warehouse": lambda s: _with_pp(warehouse(n=20, seed=s), s, False),
    "warehouse_serial": lambda s: _with_pp(warehouse(n=20, seed=s), s, True),
    "rooms": lambda s: _with_pp(rooms(n=16, seed=s), s, False),
    "maze": lambda s: _with_pp(maze(n=8, seed=s), s, False),
    "random_map": lambda s: _with_pp(random_obstacles(n=24, seed=s), s, False),
    # directed graphs: re-implemented MAPF-LNS
    "random_serial": lambda s: _with_pp(random_digraph(30, 3, n=8, seed=s), s, True),
    "oneway_serial": lambda s: _with_pp(oneway_grid(6, 6, n=10, seed=s), s, True),
    "full_rings": EXPLORE["rings"],
    "convoy": EXPLORE["convoy"],
    "dense_oneway": EXPLORE["dense_oneway"],
    "dense_random": EXPLORE["dense_random"],
}
FIELDS = ["family", "instance", "n", "V", "lns_impl", "strategy", "soc", "makespan",
          "soc0", "makespan0", "time"]


def _lns(inst, plan, t, seed):
    from .official_lns import use_official
    if use_official(inst):
        p, _ = official_lns(inst, plan, time_limit=t, N=5, seed=seed)
        return p, "official"
    p, _ = mapf_lns(inst, plan, objective="soc", N=5, max_iters=100000,
                    time_limit=t, seed=seed)
    return p, "reimpl"


def _ls(name, inst, plan, t, expansions, seed):
    p, _ = run_algorithm(name, inst, plan,
                         {"time": t, "expansions": expansions // 10, "lns_iters": 0}, seed)
    return p


# MovingAI benchmark families (maps of benchmarks/movingai, scenario index = seed)
try:
    from .movingai import families as _mai_families, dense_families as _dmai_families
    FAMILIES.update(_mai_families())
    FAMILIES.update(_dmai_families())
except Exception as _e:   # pragma: no cover - missing benchmark files
    pass

STEP = {"vnd": "vnd_soc", "avnd": "avnd_soc", "avndk": "avndk_soc", "avndr": "avndr_soc",
        "avndra": "avndra_soc", "avnda": "avnda_soc", "avndraw": "avndraw_soc",
        "avndrf": "avndrf_soc", "avndro": "avndro_soc", "avndrof": "avndrof_soc",
        "avndrofc": "avndrofc_soc"}
DEFAULT_STRATEGIES = ["lns", "vnd", "avnd", "lns+avnd", "avnd+lns"]


def run(out, instances, lns_time, vnd_time, expansions, families, strategies):
    """Strategies are chains: "lns", "vnd" (fixed radii), "avnd" (adaptive
    radii), and their compositions such as "avnd+lns" (the second method is
    applied to the result of the first). Prefixes are computed once."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "vnd.csv")
    done = set()
    if os.path.exists(path):
        done = {(r["family"], r["instance"], r["strategy"]) for r in csv.DictReader(open(path))}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for fam in families:
            for seed in range(instances):
                inst = FAMILIES[fam](seed)
                if inst.initial_plan is None:      # e.g. no solution found by LaCAM
                    print(fam, inst.name, "no initial plan: skipped", flush=True)
                    with open(os.path.join(out, "skipped.txt"), "a") as sk:
                        sk.write(f"{fam}\t{inst.name}\n")
                    continue
                p0 = trim(inst.initial_plan)
                validate(inst.graph, inst.starts, inst.goals, p0)
                base = {"family": fam, "instance": inst.name, "n": inst.n, "V": inst.graph.N,
                        "soc0": sum_of_costs(p0, inst.goals), "makespan0": makespan(p0)}
                todo = [s for s in strategies if (fam, inst.name, s) not in done]
                if not todo:
                    continue
                cache = {(): (p0, 0.0)}
                certified_log = []
                impl = "official" if is_grid_instance(inst) else "reimpl"
                line = [f"init {base['soc0']}/{base['makespan0']}"]
                for strat in todo:
                    steps = tuple(strat.split("+"))
                    for k in range(1, len(steps) + 1):
                        if steps[:k] in cache:
                            continue
                        prev, tprev = cache[steps[:k - 1]]
                        t = time.perf_counter()
                        if steps[k - 1] == "lns":
                            p, impl = _lns(inst, prev, lns_time, seed)
                        elif steps[k - 1] == "lacam":      # LaCAM* (anytime) from scratch
                            from .lacam import lacam_solve
                            p = lacam_solve(inst, time_limit=vnd_time, anytime=True, seed=seed)
                            if p is None:
                                p = prev
                        elif steps[k - 1] == "lnsT":       # LNS with the local-search time
                            p, impl = _lns(inst, prev, vnd_time, seed)
                        elif steps[k - 1] in ("hyb", "hyb2", "hyb3", "hyb4", "hyb5"):   # hybrid
                            from .hybrid import hybrid
                            from .algorithms import (order_reduced_neighborhoods,
                                                     order_corridor_neighborhoods)
                            if steps[k - 1] == "hyb":
                                p, hst = hybrid(inst, prev, time_limit=vnd_time, seed=seed)
                            elif steps[k - 1] == "hyb2":   # LNS first, cheap LS, full final LS
                                p, hst = hybrid(inst, prev, time_limit=vnd_time, seed=seed,
                                                lns_first=True, lns_phase=4.0,
                                                neighborhoods=order_reduced_neighborhoods,
                                                final_neighborhoods=order_corridor_neighborhoods)
                            else:   # hyb2 + LNS phases stopped on stagnation + kept radii
                                p, hst = hybrid(inst, prev, time_limit=vnd_time, seed=seed,
                                                lns_first=True, lns_phase=4.0,
                                                neighborhoods=order_reduced_neighborhoods,
                                                final_neighborhoods=order_corridor_neighborhoods,
                                                stagnation=500 if steps[k - 1] == "hyb3" else 2000,
                                                keep_radius=True,
                                                max_fail=4 if steps[k - 1] == "hyb5" else 2)
                            hst["variant"] = steps[k - 1]
                            impl = ("official" if is_grid_instance(inst) else "reimpl")
                            certified_log.append((fam, inst.name, hst))
                        else:
                            p = _ls(STEP[steps[k - 1]], inst, prev, vnd_time, expansions, seed)
                        cache[steps[:k]] = (p, tprev + time.perf_counter() - t)
                    p, dt = cache[steps]
                    validate(inst.graph, inst.starts, inst.goals, p)
                    w.writerow({**base, "lns_impl": impl, "strategy": strat,
                                "soc": sum_of_costs(p, inst.goals), "makespan": makespan(p),
                                "time": round(dt, 3)})
                    line.append(f"{strat} {sum_of_costs(p, inst.goals)}/{makespan(p)}")
                fh.flush()
                if certified_log:
                    with open(os.path.join(out, "hybrid_stats.jsonl"), "a") as hf:
                        import json
                        for f_, n_, h_ in certified_log:
                            hf.write(json.dumps({"family": f_, "instance": n_, **h_}) + "\n")
                print(fam, inst.name, " ".join(line), "(SoC/makespan)", flush=True)


def summary(out):
    rows = list(csv.DictReader(open(os.path.join(out, "vnd.csv"))))
    by = defaultdict(dict)
    for r in rows:
        by[(r["family"], r["instance"])][r["strategy"]] = r
    fams = list(dict.fromkeys(r["family"] for r in rows))
    strategies = list(dict.fromkeys(r["strategy"] for r in rows))
    others = [s for s in strategies if s != "lns"]
    lines = ["# Local search (VND, adaptive VND) vs MAPF-LNS (objective: Sum-of-Costs)", "",
             "Mean Sum-of-Costs (mean makespan in brackets). W/T/L = instances with lower / "
             "equal / higher Sum-of-Costs than LNS. LNS: official MAPF-LNS on undirected "
             "grids, re-implementation on directed graphs. `vnd`: fixed radii; `avnd`: "
             "adaptive radii; `a+b`: b applied to the result of a.", "",
             "| family | LNS | inst. | initial | " + " | ".join(strategies) + " | "
             + " | ".join(f"{s} vs lns" for s in others) + " |",
             "|---|---|---:|---:|" + "---:|" * len(strategies) + ":---:|" * len(others)]
    tot = defaultdict(lambda: [0, 0, 0])
    for fam in fams:
        ks = [k for k in by if k[0] == fam and all(s in by[k] for s in strategies)]
        if not ks:
            continue
        m = lambda s, f: sum(int(by[k][s][f]) for k in ks) / len(ks)
        cells = [f"{m(s, 'soc'):.1f} ({m(s, 'makespan'):.1f})" for s in strategies]
        r0 = by[ks[0]][strategies[0]]
        init = (f"{sum(int(by[k]['lns']['soc0']) for k in ks)/len(ks):.1f} "
                f"({sum(int(by[k]['lns']['makespan0']) for k in ks)/len(ks):.1f})")
        wt = []
        for s in others:
            w = t = l = 0
            for k in ks:
                a, b = int(by[k][s]["soc"]), int(by[k]["lns"]["soc"])
                w, t, l = w + (a < b), t + (a == b), l + (a > b)
            tot[s][0] += w; tot[s][1] += t; tot[s][2] += l
            wt.append(f"{w}/{t}/{l}")
        lines.append(f"| {fam} | {r0['lns_impl']} | {len(ks)} | {init} | "
                     + " | ".join(cells) + " | " + " | ".join(wt) + " |")
    lines += ["", "Totals (W/T/L against LNS): " +
              ", ".join(f"{s}: {tot[s][0]}/{tot[s][1]}/{tot[s][2]}" for s in others)]
    if "vnd" in strategies and "avnd" in strategies:
        w = t = l = 0
        for k, d in by.items():
            if "vnd" in d and "avnd" in d:
                a, b = int(d["avnd"]["soc"]), int(d["vnd"]["soc"])
                w, t, l = w + (a < b), t + (a == b), l + (a > b)
        lines.append(f"Adaptive vs fixed radii (avnd vs vnd): {w}/{t}/{l}")
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "vnd_summary.md"), "w").write(text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "summary"])
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--lns-time", type=float, default=10.0)
    ap.add_argument("--vnd-time", type=float, default=20.0)
    ap.add_argument("--expansions", type=int, default=2_000_000)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--out", default="vnd_results")
    ap.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    a = ap.parse_args(argv)
    if a.action == "run":
        run(a.out, a.instances, a.lns_time, a.vnd_time, a.expansions, a.families,
            a.strategies)
    summary(a.out)


if __name__ == "__main__":
    main()
