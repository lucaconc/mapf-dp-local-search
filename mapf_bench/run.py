# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line runner.

Example:
    python -m mapf_bench.run --suite quick --seeds 3 --out results
    python -m mapf_bench.run --suite grid oneway --algos lns_soc ls_combined
"""
import argparse
import csv
import json
import os
import time

from .algorithms import ALGORITHMS, DEFAULT_ALGOS, run_algorithm
from .plan import (validate, makespan, sum_of_costs, serialize, paths_to_plan,
                   trim, InvalidPlan)
from .pp import initial_pp
from .report import write_report
from .suites import build

FIELDS = ["suite", "family", "instance", "n", "V", "init", "seed", "algo",
          "makespan", "soc", "makespan0", "soc0", "lb_makespan", "lb_soc",
          "red_makespan", "red_soc", "time", "valid", "stats"]


def initial_plan(inst, method, seed):
    if method == "given":
        return trim(inst.initial_plan)
    paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=seed)
    if paths is None:
        return None
    plan = paths_to_plan(paths)
    if method == "pp_serial":
        plan = serialize(inst.graph, plan)
    return trim(plan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", nargs="+", default=["quick"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    ap.add_argument("--time-limit", type=float, default=30.0,
                    help="time limit per algorithm stage (s)")
    ap.add_argument("--lns-iters", type=int, default=1000)
    ap.add_argument("--expansions", type=int, default=300000,
                    help="state budget per neighborhood search")
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", action="store_true",
                    help="append to an existing results.csv, skipping done rows")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv)
    for a in args.algos:
        for part in a.split("+"):
            if part not in ALGORITHMS:
                ap.error(f"unknown algorithm {part}; known: {sorted(ALGORITHMS)}")

    os.makedirs(args.out, exist_ok=True)
    budget = {"time": args.time_limit, "lns_iters": args.lns_iters,
              "expansions": args.expansions}
    items = build(args.suite, list(range(args.seeds)))
    path = os.path.join(args.out, "results.csv")
    done = set()
    if args.resume and os.path.exists(path):
        with open(path) as fh:
            done = {(r["suite"], r["instance"], r["init"], r["algo"])
                    for r in csv.DictReader(fh)}
    mode = "a" if done else "w"
    with open(path, mode, newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            wr.writeheader()
        for idx, (suite, inst, init) in enumerate(items):
            seed = inst.params.get("seed", 0)
            plan0 = initial_plan(inst, init, seed)
            if plan0 is None:
                print(f"[{idx+1}/{len(items)}] {inst.name}: no initial plan, skipped")
                continue
            validate(inst.graph, inst.starts, inst.goals, plan0)
            G = inst.graph
            sp = [int(G.dist[s, g]) for s, g in zip(inst.starts, inst.goals)]
            mk0, soc0 = makespan(plan0), sum_of_costs(plan0, inst.goals)
            base = {"suite": suite, "family": inst.family, "instance": inst.name,
                    "n": inst.n, "V": G.N, "init": init, "seed": seed,
                    "makespan0": mk0, "soc0": soc0,
                    "lb_makespan": max(sp), "lb_soc": sum(sp)}
            if (suite, inst.name, init, "init") not in done:
                wr.writerow({**base, "algo": "init", "makespan": mk0, "soc": soc0,
                             "red_makespan": 0.0, "red_soc": 0.0, "time": 0.0,
                             "valid": True, "stats": ""})
            line = [f"init {mk0}"]
            for algo in args.algos:
                if (suite, inst.name, init, algo) in done:
                    continue
                t0 = time.perf_counter()
                plan, stats = run_algorithm(algo, inst, plan0, budget, seed)
                dt = time.perf_counter() - t0
                try:
                    valid = validate(G, inst.starts, inst.goals, plan)
                except InvalidPlan as e:
                    valid = False
                    stats = {"invalid": str(e)}
                mk, soc = makespan(plan), sum_of_costs(plan, inst.goals)
                wr.writerow({**base, "algo": algo, "makespan": mk, "soc": soc,
                             "red_makespan": (mk0 - mk) / mk0 if mk0 else 0.0,
                             "red_soc": (soc0 - soc) / soc0 if soc0 else 0.0,
                             "time": round(dt, 4), "valid": valid,
                             "stats": json.dumps(stats, default=str)})
                line.append(f"{algo} {mk}")
                fh.flush()
            print(f"[{idx+1}/{len(items)}] {suite}/{inst.name} ({init}): "
                  + ", ".join(line), flush=True)
    if not args.no_report:
        write_report(path, args.out)
    print(f"results: {path}\nsummary: {os.path.join(args.out, 'summary.md')}")


if __name__ == "__main__":
    main()
