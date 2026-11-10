# SPDX-License-Identifier: GPL-3.0-or-later
"""Contribution of each neighborhood of the complete local search.

    python -m mapf_bench.ablation_ls run [--out ablation_ls] [--time 20]
    python -m mapf_bench.ablation_ls analyze [--out ablation_ls]

For every instance of the paper protocol (mapf_bench.paper_suite.scenarios)
and two starting plans -- the initial plan (as the LS method) and, on grids,
the LaCAM* solution after 10 s (as LaCAM*+Hybrid) -- the local search of the
paper (order_corridor_neighborhoods, same options and budget as in the
benchmark) is run once with all its neighborhoods, recording for each
neighborhood the searches, improvements, Sum-of-Costs removed and time
(credit). On one instance per family it is also run six more times, each time
without one neighborhood (leave-one-out), to see whether the others make up
for it and how much time it costs.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import zlib
from collections import defaultdict

from .algorithms import order_corridor_neighborhoods
from .cpp_local_search import cpp_local_search
from .official_lns import is_grid_instance
from .paper_suite import scenarios
from .plan import sum_of_costs, trim, validate

NAMES = ["retime", "swap", "route-lag", "grouped-clock", "agent", "free-pairs"]
FIELDS = ["idx", "group", "family", "instance", "n", "start", "variant", "soc_start", "soc", "time",
          "timeout", "certified"] + [f"{k}_{x}" for k in NAMES for x in ("searches", "impr", "gain", "time")]


def _ls(inst, plan, nbs, T):
    p, st = cpp_local_search(inst, plan, nbs, tiebreak="summin", objective="soc_proxy", vnd=True,
                             astar=True, keep_radius=True, time_limit=T, max_expansions=2_000_000)
    validate(inst.graph, inst.starts, inst.goals, p)
    return trim(p), st


def _lacam_start(inst, idx, out):
    path = os.path.join(out, "lacam_plans", f"{idx}.json")
    if os.path.exists(path):
        return [tuple(x) for x in json.load(open(path))]
    from .lacam import lacam_solve
    p = lacam_solve(inst, time_limit=10, anytime=True, seed=zlib.crc32(inst.name.encode()) % 1000)
    if p is None:
        return None
    p = trim(p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump([list(x) for x in p], open(path, "w"))
    return p


def run(out="ablation_ls", T=20.0):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.csv")
    done = set()
    if os.path.exists(path):
        done = {(r["idx"], r["start"], r["variant"]) for r in csv.DictReader(open(path))}
    new = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    sc = scenarios()
    first_of_family = {}
    for i, (g, fam, _) in enumerate(sc):
        first_of_family.setdefault(fam, i)
    loo = set(first_of_family.values())
    for i, (g, fam, build) in enumerate(sc):
        variants = ["full"] + ([f"-{k}" for k in NAMES] if i in loo else [])
        starts = ["initial", "lacam"]
        if all((str(i), s, v) in done for s in starts for v in variants):
            continue
        inst, _ = build()
        if inst.initial_plan is None:
            continue
        for s in starts:
            if s == "lacam" and not is_grid_instance(inst):
                continue
            if all((str(i), s, v) in done for v in variants):
                continue
            p0 = trim(inst.initial_plan) if s == "initial" else _lacam_start(inst, i, out)
            if p0 is None:
                continue
            soc0 = sum_of_costs(p0, inst.goals)
            for v in variants:
                if (str(i), s, v) in done:
                    continue
                nbs = order_corridor_neighborhoods()
                if v != "full":
                    del nbs[NAMES.index(v[1:])]
                p, st = _ls(inst, p0, nbs, T)
                row = {"idx": i, "group": g, "family": fam, "instance": inst.name, "n": inst.n,
                       "start": s, "variant": v, "soc_start": soc0, "soc": sum_of_costs(p, inst.goals),
                       "time": round(st["time"], 3), "timeout": int(bool(st.get("timeout"))),
                       "certified": int(not st.get("timeout") and st.get("budget_exceeded", 0) == 0)}
                names = NAMES if v == "full" else [k for k in NAMES if k != v[1:]]
                for j, k in enumerate(names):
                    for key, src in (("searches", "slot_searches"), ("impr", "slot_improvements"),
                                     ("gain", "slot_gain"), ("time", "slot_time")):
                        row[f"{k}_{key}"] = round(st[src][j], 3) if isinstance(st[src][j], float) else st[src][j]
                w.writerow(row)
                fh.flush()
                print(f"[{i}] {g} {inst.name} {s} {v}: {soc0} -> {row['soc']} ({row['time']} s)", flush=True)
    fh.close()


def analyze(out="ablation_ls"):
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    num = lambda r, k: float(r[k]) if r.get(k) not in ("", None) else 0.0
    full = [r for r in rows if r["variant"] == "full"]
    L = ["# Contribution of each neighborhood of the local search", ""]
    # ---- credit
    for start in ("initial", "lacam"):
        F = [r for r in full if r["start"] == start]
        if not F:
            continue
        title = "from the initial plan (LS)" if start == "initial" else "from the LaCAM* solution (LaCAM*+Hybrid)"
        L += [f"## Credit, {title}: {len(F)} instances", "",
              "Share of the total Sum-of-Costs reduction found by each neighborhood, instances on "
              "which it found at least one improvement, and share of the search time it used.", "",
              "| group | inst. | total reduction | " + " | ".join(NAMES) + " |",
              "|---|---:|---:|" + "---:|" * len(NAMES)]
        groups = sorted({r["group"] for r in F}, key="SDMHW".index) + ["all"]
        for g in groups:
            G = [r for r in F if g == "all" or r["group"] == g]
            red = sum(num(r, "soc_start") - num(r, "soc") for r in G)
            gains = {k: sum(num(r, f"{k}_gain") for r in G) for k in NAMES}
            tot_gain = sum(gains.values()) or 1
            tt = sum(sum(num(r, f"{k}_time") for k in NAMES) for r in G) or 1
            cells = []
            for k in NAMES:
                hit = sum(1 for r in G if num(r, f"{k}_impr") > 0)
                cells.append(f"{100 * gains[k] / tot_gain:.0f}% gain, {hit} inst., {100 * sum(num(r, f'{k}_time') for r in G) / tt:.0f}% time")
            L.append(f"| {g} | {len(G)} | {red:.0f} ({100 * red / max(sum(num(r, 'soc_start') for r in G), 1):.1f}%) | " + " | ".join(cells) + " |")
        L.append("")
    # ---- leave-one-out
    loo = [r for r in rows if r["variant"] != "full"]
    if loo:
        key = lambda r: (r["idx"], r["start"])
        F = {key(r): r for r in full}
        L += ["## Leave-one-out", "",
              "Each neighborhood removed in turn, on one instance per family, from both starting "
              "plans. Change of the final Sum-of-Costs relative to the full search (W/T/L: lower / "
              "equal / higher without the neighborhood), mean relative change, and change of the "
              "running time.", "",
              "| removed | runs | W/T/L | mean SoC change | worst | mean time change | timeouts (full -> without) |",
              "|---|---:|:---:|---:|---:|---:|:---:|"]
        for k in NAMES:
            R = [r for r in loo if r["variant"] == f"-{k}" and key(r) in F]
            if not R:
                continue
            d = [(num(r, "soc") - num(F[key(r)], "soc")) / max(num(F[key(r)], "soc"), 1) for r in R]
            w = sum(x < 0 for x in d); l = sum(x > 0 for x in d); t = len(d) - w - l
            dt = [num(r, "time") - num(F[key(r)], "time") for r in R]
            to_f = sum(int(F[key(r)]["timeout"]) for r in R); to_v = sum(int(r["timeout"]) for r in R)
            L.append(f"| {k} | {len(R)} | {w}/{t}/{l} | {100 * sum(d) / len(d):+.2f}% | {100 * max(d):+.1f}% | "
                     f"{sum(dt) / len(dt):+.1f} s | {to_f} -> {to_v} |")
        L.append("")
        # per family detail for the neighborhoods that matter
        L += ["### Leave-one-out by run (final Sum-of-Costs; full search first)", "",
              "| instance | start | full | " + " | ".join(f"-{k}" for k in NAMES) + " |",
              "|---|---|---:|" + "---:|" * len(NAMES)]
        for kk in sorted({key(r) for r in loo}, key=lambda x: (int(x[0]), x[1])):
            if kk not in F:
                continue
            cells = []
            for k in NAMES:
                r = next((r for r in loo if key(r) == kk and r["variant"] == f"-{k}"), None)
                if r is None:
                    cells.append(""); continue
                dv = int(num(r, "soc") - num(F[kk], "soc"))
                cells.append(f"{dv:+d}" if dv else "=")
            L.append(f"| {F[kk]['instance']} | {kk[1]} | {int(num(F[kk], 'soc'))} | " + " | ".join(cells) + " |")
    text = "\n".join(L) + "\n"
    open(os.path.join(out, "report.md"), "w").write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "analyze"])
    ap.add_argument("--out", default="ablation_ls")
    ap.add_argument("--time", type=float, default=20.0)
    a = ap.parse_args(argv)
    if a.action == "run":
        run(a.out, a.time)
    print(analyze(a.out))


if __name__ == "__main__":
    main()
