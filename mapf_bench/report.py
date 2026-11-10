# SPDX-License-Identifier: GPL-3.0-or-later
"""Aggregate results.csv into summary tables and plots."""
import csv
import os
from collections import defaultdict


def _load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("makespan", "soc", "makespan0", "soc0", "lb_makespan", "lb_soc"):
            r[k] = int(r[k])
        for k in ("red_makespan", "red_soc", "time"):
            r[k] = float(r[k])
        r["valid"] = r["valid"] == "True"
    return rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows, ref_algo="lns_soc"):
    """Per (suite, init) and algorithm: mean reductions, gap to the lower
    bound, time, and wins/ties/losses against `ref_algo` on the makespan."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["suite"], r["init"], r["algo"])].append(r)
    ref = {(r["suite"], r["init"], r["instance"]): r for r in rows if r["algo"] == ref_algo}
    out = []
    for (suite, init, algo), rs in sorted(by_key.items()):
        w = t = l = 0
        for r in rs:
            rr = ref.get((suite, init, r["instance"]))
            if rr is None or algo == ref_algo:
                continue
            if r["makespan"] < rr["makespan"]:
                w += 1
            elif r["makespan"] == rr["makespan"]:
                t += 1
            else:
                l += 1
        out.append({
            "suite": suite, "init": init, "algo": algo, "instances": len(rs),
            "red_makespan": _mean([r["red_makespan"] for r in rs]),
            "red_soc": _mean([r["red_soc"] for r in rs]),
            "gap_lb": _mean([r["makespan"] / max(r["lb_makespan"], 1) for r in rs]),
            "time": _mean([r["time"] for r in rs]),
            "invalid": sum(not r["valid"] for r in rs),
            "vs_ref": f"{w}/{t}/{l}" if algo != ref_algo else "-",
        })
    return out


def write_report(csv_path, out_dir, ref_algo="lns_soc"):
    rows = _load(csv_path)
    summ = summarize(rows, ref_algo)
    lines = ["# MAPF local search vs MAPF-LNS", "",
             f"Reductions are relative to the initial plan; `gap LB` is makespan / "
             f"max shortest-path length (1.00 = provably optimal makespan); "
             f"`vs {ref_algo}` counts instances with shorter / equal / longer "
             f"makespan than {ref_algo}.", ""]
    cur = None
    for s in summ:
        key = (s["suite"], s["init"])
        if key != cur:
            cur = key
            lines += ["", f"## {s['suite']} (initial plan: {s['init']}, "
                          f"{s['instances']} instances)", "",
                      f"| algorithm | makespan red. | SoC red. | gap LB | time (s) | vs {ref_algo} | invalid |",
                      "|---|---:|---:|---:|---:|:---:|---:|"]
        lines.append(f"| {s['algo']} | {s['red_makespan']:.3f} | {s['red_soc']:.3f} | "
                     f"{s['gap_lb']:.2f} | {s['time']:.2f} | {s['vs_ref']} | {s['invalid']} |")
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    _plot(summ, out_dir)
    return summ


def _plot(summ, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    keys = sorted({(s["suite"], s["init"]) for s in summ})
    algos = [a for a in dict.fromkeys(s["algo"] for s in summ) if a != "init"]
    if not keys or not algos:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(keys)), 4))
    width = 0.8 / len(algos)
    val = {(s["suite"], s["init"], s["algo"]): s["red_makespan"] for s in summ}
    for j, a in enumerate(algos):
        xs = [i + j * width for i in range(len(keys))]
        ax.bar(xs, [val.get((k[0], k[1], a), 0) for k in keys], width, label=a)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(keys))])
    ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k in keys], fontsize=8)
    ax.set_ylabel("mean makespan reduction")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "makespan_reduction.png"), dpi=150)
    plt.close(fig)
