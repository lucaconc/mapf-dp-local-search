# SPDX-License-Identifier: GPL-3.0-or-later
"""Report of avndra_soc (adaptive VND, 3 neighborhoods, A*) vs MAPF-LNS by
type of MAPF instance, with sign tests.

    python -m mapf_bench.final_report final_results
"""
import csv
import os
import sys
from collections import defaultdict
from math import comb

CATEGORIES = [
    ("Grid maps, initial plan from PP", ["grid_pp", "warehouse", "rooms", "maze", "random_map"]),
    ("Serialized plans (rule-based-like)", ["grid_serial", "warehouse_serial", "random_serial",
                                            "oneway_serial"]),
    ("Dense instances (50-90% occupied)", ["dense_grid", "dense_oneway", "dense_random"]),
    ("Structured: full rings and merges", ["grid_rings", "full_rings", "convoy"]),
]
DIRECTED = {"random_serial", "oneway_serial", "full_rings", "convoy", "dense_oneway",
            "dense_random"}


def sign_test(w, l):
    """Two-sided exact sign test on wins/losses (ties dropped)."""
    m = w + l
    if m == 0:
        return 1.0
    k = min(w, l)
    p = 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m
    return min(1.0, p)


def report(out):
    rows = list(csv.DictReader(open(os.path.join(out, "vnd.csv"))))
    by = defaultdict(dict)
    for r in rows:
        by[(r["family"], r["instance"])][r["strategy"]] = r
    S = ["lns", "avndra", "lns+avndra", "avndra+lns"]
    keys = [k for k in by if all(s in by[k] for s in S)]
    lines = ["# avndra_soc vs MAPF-LNS by type of MAPF instance", "",
             f"{len(keys)} instances; objective Sum-of-Costs for every method. LNS: official "
             "MAPF-LNS (default configuration) on undirected grids, re-implementation on "
             "directed graphs (marked *). Mean Sum-of-Costs, mean makespan in brackets. "
             "W/T/L: lower / equal / higher Sum-of-Costs than LNS; p: two-sided sign test. "
             "Time limits: 10 s for LNS, 20 s for avndra.", ""]
    head = ("| family | n | inst. | initial | LNS | avndra | LNS+avndra | avndra+LNS | "
            "avndra vs LNS (p) | LNS+avndra vs LNS | avndra+LNS vs LNS | time avndra / LNS (s) |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|---:|"
    grand = defaultdict(lambda: [0, 0, 0])

    def wtl(ks, s):
        w = t = l = 0
        for k in ks:
            a, b = int(by[k][s]["soc"]), int(by[k]["lns"]["soc"])
            w, t, l = w + (a < b), t + (a == b), l + (a > b)
        return w, t, l

    for title, fams in CATEGORIES:
        lines += [f"## {title}", "", head, sep]
        cat = defaultdict(lambda: [0, 0, 0])
        for f in fams:
            ks = [k for k in keys if k[0] == f]
            if not ks:
                continue
            m = lambda s, c: sum(float(by[k][s][c]) for k in ks) / len(ks)
            cell = lambda s: f"{m(s, 'soc'):.1f} ({m(s, 'makespan'):.1f})"
            init = (f"{sum(int(by[k]['lns']['soc0']) for k in ks)/len(ks):.1f} "
                    f"({sum(int(by[k]['lns']['makespan0']) for k in ks)/len(ks):.1f})")
            res = {}
            for s in S[1:]:
                res[s] = wtl(ks, s)
                for j in range(3):
                    cat[s][j] += res[s][j]; grand[s][j] += res[s][j]
            w, t, l = res["avndra"]
            name = f + ("*" if f in DIRECTED else "")
            lines.append(
                f"| {name} | {by[ks[0]]['lns']['n']} | {len(ks)} | {init} | {cell('lns')} | "
                f"{cell('avndra')} | {cell('lns+avndra')} | {cell('avndra+lns')} | "
                f"{w}/{t}/{l} ({sign_test(w, l):.2f}) | "
                f"{'/'.join(map(str, res['lns+avndra']))} | {'/'.join(map(str, res['avndra+lns']))} | "
                f"{m('avndra', 'time'):.1f} / {m('lns', 'time'):.1f} |")
        w, t, l = cat["avndra"]
        lines += ["", f"Category: avndra vs LNS {w}/{t}/{l} (p = {sign_test(w, l):.3f}); "
                  f"LNS+avndra vs LNS {'/'.join(map(str, cat['lns+avndra']))} "
                  f"(p = {sign_test(cat['lns+avndra'][0], cat['lns+avndra'][2]):.3f}); "
                  f"avndra+LNS vs LNS {'/'.join(map(str, cat['avndra+lns']))} "
                  f"(p = {sign_test(cat['avndra+lns'][0], cat['avndra+lns'][2]):.3f})", ""]
    lines.append("## Overall")
    lines.append("")
    for s in S[1:]:
        w, t, l = grand[s]
        lines.append(f"- {s} vs LNS: {w}/{t}/{l} (p = {sign_test(w, l):.3f})")
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "final_report.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "final_results")
