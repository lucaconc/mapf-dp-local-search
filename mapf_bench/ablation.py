# SPDX-License-Identifier: GPL-3.0-or-later
"""Are all the local searches useful? Contribution and leave-one-out ablation
of the adaptive VND (objective: Sum-of-Costs).

    python -m mapf_bench.ablation run --instances 3
    python -m mapf_bench.ablation summary

For every instance: the full adaptive VND (with per-neighborhood statistics:
searches, improvements, Sum-of-Costs gain, time) and, for each neighborhood,
the adaptive VND without it.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

from .algorithms import adaptive_neighborhoods
from .compare_vnd import FAMILIES
from .cpp_local_search import cpp_local_search
from .plan import validate, sum_of_costs, makespan, trim

NAMES = ["pointwise", "path", "route-lag", "route-restricted", "grouped-2",
         "agent", "combined", "grouped-4"]
FIELDS = ["family", "instance", "variant", "soc0", "soc", "makespan", "time", "stats"]


def run(out, instances, time_limit, expansions, families):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "ablation.csv")
    done = set()
    if os.path.exists(path):
        done = {(r["family"], r["instance"], r["variant"]) for r in csv.DictReader(open(path))}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for fam in families:
            for seed in range(instances):
                inst = FAMILIES[fam](seed)
                p0 = trim(inst.initial_plan)
                full = adaptive_neighborhoods()
                variants = [("full", full)] + [(f"-{NAMES[i]}", full[:i] + full[i + 1:])
                                               for i in range(len(full))]
                line = []
                for name, nbs in variants:
                    if (fam, inst.name, name) in done:
                        continue
                    p, st = cpp_local_search(inst, p0, nbs, tiebreak="summin",
                                             objective="soc_proxy", vnd=True,
                                             time_limit=time_limit, max_expansions=expansions)
                    validate(inst.graph, inst.starts, inst.goals, p)
                    w.writerow({"family": fam, "instance": inst.name, "variant": name,
                                "soc0": sum_of_costs(p0, inst.goals),
                                "soc": sum_of_costs(p, inst.goals), "makespan": makespan(p),
                                "time": st["time"], "stats": json.dumps(st)})
                    line.append(f"{name} {sum_of_costs(p, inst.goals)}")
                fh.flush()
                print(fam, inst.name, " ".join(line), flush=True)


def summary(out):
    rows = list(csv.DictReader(open(os.path.join(out, "ablation.csv"))))
    by = defaultdict(dict)
    for r in rows:
        by[(r["family"], r["instance"])][r["variant"]] = r
    keys = [k for k in by if "full" in by[k] and all(f"-{n}" in by[k] for n in NAMES)]
    fams = list(dict.fromkeys(k[0] for k in keys))
    lines = ["# Are all local searches useful? (adaptive VND, Sum-of-Costs)", "",
             f"{len(keys)} instances. *Contribution*: improvements found by each "
             "neighborhood inside the full VND and the share of the total Sum-of-Costs "
             "reduction they account for (early neighborhoods get credit first). "
             "*Leave-one-out*: VND without the neighborhood vs the full VND: instances "
             "where the result gets worse / equal / better, mean relative increase of the "
             "Sum-of-Costs, and change of the running time.", "",
             "| neighborhood | improvements | share of reduction | searches | share of time | "
             "without it: worse/equal/better | mean SoC increase | worst case | time change |",
             "|---|---:|---:|---:|---:|:---:|---:|---:|---:|"]
    tot_gain = sum(int(by[k]["full"]["soc0"]) - int(by[k]["full"]["soc"]) for k in keys)
    tot_time = sum(sum(json.loads(by[k]["full"]["stats"])["slot_time"]) for k in keys)
    t_full = sum(float(by[k]["full"]["time"]) for k in keys)
    worst_by = {}
    for i, n in enumerate(NAMES):
        impr = sum(json.loads(by[k]["full"]["stats"])["slot_improvements"][i] for k in keys)
        gain = sum(json.loads(by[k]["full"]["stats"])["slot_gain"][i] for k in keys)
        srch = sum(json.loads(by[k]["full"]["stats"])["slot_searches"][i] for k in keys)
        tm = sum(json.loads(by[k]["full"]["stats"])["slot_time"][i] for k in keys)
        w = t = b = 0
        incs, worst = [], (0.0, "")
        for k in keys:
            a, f = int(by[k][f"-{n}"]["soc"]), int(by[k]["full"]["soc"])
            w, t, b = w + (a > f), t + (a == f), b + (a < f)
            inc = (a - f) / f
            incs.append(inc)
            if inc > worst[0]:
                worst = (inc, f"{k[0]} {f}->{a}")
        worst_by[n] = worst
        t_wo = sum(float(by[k][f"-{n}"]["time"]) for k in keys)
        lines.append(f"| {n} | {impr} | {gain / max(tot_gain, 1):.1%} | {srch} | "
                     f"{tm / max(tot_time, 1e-9):.1%} | {w}/{t}/{b} | "
                     f"{sum(incs) / len(incs):+.1%} | {worst[0]:+.0%} ({worst[1]}) | "
                     f"{(t_wo - t_full) / t_full:+.0%} |")
    lines += ["", "## Leave-one-out by family (mean Sum-of-Costs)", "",
              "| family | full | " + " | ".join(f"-{n}" for n in NAMES) + " |",
              "|---|---:|" + "---:|" * len(NAMES)]
    for f in fams:
        ks = [k for k in keys if k[0] == f]
        m = lambda v: sum(int(by[k][v]["soc"]) for k in ks) / len(ks)
        lines.append(f"| {f} | {m('full'):.1f} | " +
                     " | ".join(f"{m('-' + n):.1f}" for n in NAMES) + " |")
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "ablation_summary.md"), "w").write(text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "summary"])
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=15.0)
    ap.add_argument("--expansions", type=int, default=2_000_000)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--out", default="ablation_results")
    a = ap.parse_args(argv)
    if a.action == "run":
        run(a.out, a.instances, a.time_limit, a.expansions, a.families)
    summary(a.out)


if __name__ == "__main__":
    main()
