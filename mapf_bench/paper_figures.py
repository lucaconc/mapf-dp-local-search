# SPDX-License-Identifier: GPL-3.0-or-later
"""Recompute the experimental figures of the paper (Section 5).

    python -m mapf_bench.paper_figures run digraphs --instances 10
    python -m mapf_bench.paper_figures run lns --instances 30
    python -m mapf_bench.paper_figures plot

Results are appended to figures/*.csv (runs resume where they stopped);
`plot` writes the PDFs with the file names used in the paper
(images/tests/...) into figures/, and figures/numbers.md with the values
quoted in the text.

All methods minimize the Sum-of-Costs by default (--objective soc), as
MAPF-LNS does; the local search uses the formulation of Appendix A.3.
All methods minimize the Sum-of-Costs by default (--objective soc), as
MAPF-LNS does; the local search uses the formulation of Appendix A.3.
Differences from the paper: the initial plans of Section 5.2 come from
prioritized planning, serialized so that one agent moves at a time (diSC is
not available); instances where it fails are skipped and counted. The local
search is the C++ implementation (cpp/dpls). In Section 5.3 the official
MAPF-LNS is run from the same PP plan with the time limit given by --lns-time
and at most 5000 iterations.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

from .instances import random_digraph, grid_bottleneck
from .pp import initial_pp
from .plan import paths_to_plan, serialize, trim, makespan, sum_of_costs, validate
from .local_search import Pointwise, AgentDistance, Combined
from .cpp_local_search import cpp_local_search
from .official_lns import official_lns, official_initial_plan

OUT = "figures"
OBJECTIVE = "soc"   # objective of every method; reductions are of this objective
DIGRAPH_N = {50: list(range(2, 31, 2)), 100: list(range(4, 57, 4))}
METHODS = {   # name: (neighborhoods, tie-breaking)
    "pointwise": (lambda: [Pointwise(1)], None),
    "agentA": (lambda: [AgentDistance(1)], None),
    "agentB": (lambda: [AgentDistance(1)], "summin"),
    "alternating": (lambda: [Pointwise(1), AgentDistance(1)], "summin"),
    "combined": (lambda: [Combined(R_ag=1, R_sm=0)], "summin"),
}
LABEL = {"pointwise": "Pointwise ($R=1$)", "agentA": "Agent, Ordering A",
         "agentB": "Agent, Ordering B", "alternating": "Alternating",
         "combined": "Combined"}
PAPER_FILES = {   # method -> (reduction file stem, time file stem) without |V|
    "pointwise": ("reduction_{V}Nodes_R1_new", "time_{V}Nodes_R1_new"),
    "agentA": ("reduction_{V}Nodes_R2_new", "time_{V}Nodes_R2_new"),
    "agentB": ("reduction_{V}Nodes_R2_new_summinorder", "time_{V}Nodes_R2_new_summinorder"),
    "alternating": ("reduction_{V}Nodes_alternate_pointwise_agent_summinorder",
                    "time_{V}Nodes_alternate_pointwise_agent_summinorder"),
    "combined": ("reduction_{V}Nodes_R1_0_R2_1_combined_new",
                 "time_{V}Nodes_R1_0_R2_1_combined_new"),
}


def _done(path, keycols):
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {tuple(r[c] for c in keycols) for r in csv.DictReader(fh)}


def _writer(path, fields):
    new = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if new:
        w.writeheader()
    return fh, w


# ---------------------------------------------------------------------------
def run_digraphs(instances, time_limit, expansions, sizes=(50, 100)):
    path = os.path.join(OUT, "digraphs.csv")
    fields = ["V", "n", "seed", "method", "T0", "T", "soc0", "soc", "reduction", "time", "status"]
    done = _done(path, ["V", "n", "seed", "method"])
    fh, w = _writer(path, fields)
    for V in sizes:
        for n in DIGRAPH_N[V]:
            for seed in range(instances):
                key0 = (str(V), str(n), str(seed))
                if all(key0 + (m,) in done for m in METHODS) or key0 + ("none",) in done:
                    continue
                inst = random_digraph(V, 4, n=n, seed=1000 * V + 100 * n + seed)
                paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=seed, attempts=60)
                if paths is None:
                    w.writerow({"V": V, "n": n, "seed": seed, "method": "none",
                                "status": "no initial plan"})
                    fh.flush()
                    continue
                p0 = trim(serialize(inst.graph, paths_to_plan(paths)))
                T0 = makespan(p0)
                S0 = sum_of_costs(p0, inst.goals)
                for m, (nbs, tb) in METHODS.items():
                    if key0 + (m,) in done:
                        continue
                    t = time.perf_counter()
                    p, st = cpp_local_search(inst, p0, nbs(), tiebreak=tb,
                                             time_limit=time_limit, max_expansions=expansions,
                                             objective=OBJECTIVE)
                    dt = time.perf_counter() - t
                    T = makespan(p)
                    S = sum_of_costs(p, inst.goals)
                    status = "timeout" if st.get("timeout") else (
                        "budget" if st.get("budget_exceeded") else "ok")
                    obj0, obj = (S0, S) if OBJECTIVE == "soc" else (T0, T)
                    w.writerow({"V": V, "n": n, "seed": seed, "method": m, "T0": T0, "T": T,
                                "soc0": S0, "soc": S,
                                "reduction": (obj0 - obj) / obj0, "time": round(dt, 4),
                                "status": status})
                    fh.flush()
                print(f"V={V} n={n} seed={seed} T0={T0}", flush=True)
    fh.close()


# ---------------------------------------------------------------------------
def run_lns(instances, lns_time, ls_time, expansions, init="ours"):
    path = os.path.join(OUT, "lns.csv" if init == "ours" else "lns_official_init.csv")
    fields = ["K", "seed", "T0", "soc0", "lns_T", "lns_soc", "lns_time", "ls_T", "ls_soc",
              "ls_time", "lnsls_T", "lnsls_soc", "lnsls_time", "V"]
    done = _done(path, ["K", "seed"])
    fh, w = _writer(path, fields)
    nbs = lambda: [Combined(R_ag=2, R_sm=1)]
    for K in (0, 1, 2):
        for seed in range(instances):
            if (str(K), str(seed)) in done:
                continue
            inst = grid_bottleneck(11, 11, K=K, n=10, free_target=100, seed=10000 * K + seed)
            if init == "official":
                p0 = official_initial_plan(inst, "PP", seed=seed)
                if p0 is None:
                    continue
                p0 = trim(p0)
            else:
                paths = initial_pp(inst.graph, inst.starts, inst.goals, seed=seed, attempts=60)
                if paths is None:
                    continue
                p0 = trim(paths_to_plan(paths))
            t = time.perf_counter()
            pl, _ = official_lns(inst, p0, time_limit=lns_time, N=5, seed=seed,
                                 max_iterations=5000)
            tl = time.perf_counter() - t
            t = time.perf_counter()
            ps, _ = cpp_local_search(inst, p0, nbs(), tiebreak="summin",
                                     time_limit=ls_time, max_expansions=expansions,
                                     objective=OBJECTIVE)
            ts = time.perf_counter() - t
            t = time.perf_counter()
            pls, _ = cpp_local_search(inst, pl, nbs(), tiebreak="summin",
                                      time_limit=ls_time, max_expansions=expansions,
                                      objective=OBJECTIVE)
            tls = time.perf_counter() - t
            w.writerow({"K": K, "seed": seed, "V": inst.graph.N, "T0": makespan(p0),
                        "soc0": sum_of_costs(p0, inst.goals), "lns_T": makespan(pl),
                        "lns_soc": sum_of_costs(pl, inst.goals), "lns_time": round(tl, 3),
                        "ls_T": makespan(ps), "ls_soc": sum_of_costs(ps, inst.goals),
                        "ls_time": round(ts, 3),
                        "lnsls_T": makespan(pls), "lnsls_soc": sum_of_costs(pls, inst.goals),
                        "lnsls_time": round(tl + tls, 3)})
            fh.flush()
            print(f"K={K} seed={seed} SoC0={sum_of_costs(p0, inst.goals)} "
                  f"lns={sum_of_costs(pl, inst.goals)} ls={sum_of_costs(ps, inst.goals)} "
                  f"lns+ls={sum_of_costs(pls, inst.goals)}", flush=True)
    fh.close()
    # example maps for Figure fig:bottleneck-maps
    for K in (1, 2):
        inst = grid_bottleneck(11, 11, K=K, n=10, free_target=100, seed=10000 * K)
        _plot_map(inst, os.path.join(OUT, "one_bottle.pdf" if K == 1 else "two_bottle.pdf"))


# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    return plt


def _boxplot(ax, data, positions, labels, color="tab:blue", width=0.6):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                    showfliers=True, flierprops={"markersize": 2})
    for b in bp["boxes"]:
        b.set_facecolor("white")
    means = [sum(d) / len(d) if d else float("nan") for d in data]
    ax.plot(positions, means, "o-", color=color, markersize=3, linewidth=1, label="average")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    return means


def _plot_map(inst, path):
    plt = _plt()
    G = inst.graph
    rows = max(r for r, _ in G.coords.values()) + 1
    cols = max(c for _, c in G.coords.values()) + 1
    free = set(G.coords.values())
    fig, ax = plt.subplots(figsize=(3, 3))
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1,
                                       color="white" if (r, c) in free else "black"))
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot():
    plt = _plt()
    numbers = ["# Values for the text of Section 5 (recomputed)", ""]
    # ---- Section 5.2
    path = os.path.join(OUT, "digraphs.csv")
    if os.path.exists(path):
        rows = list(csv.DictReader(open(path)))
        red = defaultdict(list)
        tim = defaultdict(list)
        skipped = defaultdict(int)
        status = defaultdict(int)
        for r in rows:
            V, n, m = int(r["V"]), int(r["n"]), r["method"]
            if m == "none":
                skipped[(V, n)] += 1
                continue
            red[(V, m, n)].append(float(r["reduction"]))
            tim[(V, m, n)].append(float(r["time"]))
            status[(V, m, r["status"])] += 1
        means = {}
        for m in METHODS:
            for V in (50, 100):
                ns = [n for n in DIGRAPH_N[V] if red.get((V, m, n))]
                if not ns:
                    continue
                for kind, store, stem, ylab in (
                        ("red", red, PAPER_FILES[m][0],
                         "SoC reduction ratio" if OBJECTIVE == "soc" else "reduction ratio"),
                        ("time", tim, PAPER_FILES[m][1], "time (s)")):
                    fig, ax = plt.subplots(figsize=(4.2, 2.8))
                    mm = _boxplot(ax, [store[(V, m, n)] for n in ns], list(range(len(ns))),
                                  [str(n) for n in ns])
                    means[(kind, V, m)] = dict(zip(ns, mm))
                    ax.set_xlabel("|P|")
                    ax.set_ylabel(ylab)
                    ax.set_title(f"{LABEL[m]}, |V|={V}", fontsize=9)
                    ax.grid(alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(os.path.join(OUT, stem.format(V=V) + ".pdf"))
                    plt.close(fig)
        for kind, stem, ylab in (("red", "compare_reduction_{V}Nodes", "average reduction ratio"),
                                 ("time", "compare_time_{V}Nodes", "average time (s)")):
            for V in (50, 100):
                fig, ax = plt.subplots(figsize=(4.2, 2.8))
                for m in METHODS:
                    d = means.get((kind, V, m))
                    if d:
                        ax.plot(list(d), list(d.values()), "o-", markersize=3, label=LABEL[m])
                ax.set_xlabel("|P|"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
                ax.set_title(f"|V|={V}", fontsize=9)
                ax.legend(fontsize=6)
                fig.tight_layout()
                fig.savefig(os.path.join(OUT, stem.format(V=V) + ".pdf"))
                plt.close(fig)
        numbers += [f"## Section 5.2 (random digraphs; initial plans: serialized PP; "
                    f"objective and reductions: {OBJECTIVE})", "",
                    "| method | V | peak avg reduction (at n) | avg reduction at max n | max avg time (s) | runs ok/budget/timeout |",
                    "|---|---:|---|---:|---:|---|"]
        for m in METHODS:
            for V in (50, 100):
                r_ = means.get(("red", V, m)); t_ = means.get(("time", V, m))
                if not r_:
                    continue
                nb = max(r_, key=r_.get)
                nmax = max(r_)
                st = f"{status[(V, m, 'ok')]}/{status[(V, m, 'budget')]}/{status[(V, m, 'timeout')]}"
                numbers.append(f"| {LABEL[m]} | {V} | {r_[nb]:.2f} ({nb}) | {r_[nmax]:.3f} | "
                               f"{max(t_.values()):.2f} | {st} |")
        if skipped:
            numbers += ["", "Instances skipped (no PP plan): " +
                        ", ".join(f"V={V} n={n}: {c}" for (V, n), c in sorted(skipped.items()))]
        numbers.append("")
    # ---- Section 5.3: initial plans from the official code (protocol of the
    # paper, file names of the paper) and from our PP (suffix _ourpp)
    _plot_lns(plt, "lns_official_init.csv", "",
              "grids 11x11, |V|=100, n=10; initial plan from the PP of the official code", numbers)
    _plot_lns(plt, "lns.csv", "_ourpp",
              "grids 11x11, |V|=100, n=10; initial plan from our PP", numbers)
    with open(os.path.join(OUT, "numbers.md"), "w") as fh:
        fh.write("\n".join(numbers) + "\n")
    print("\n".join(numbers))


def _plot_lns(plt, csvname, suffix, title, numbers):
    path = os.path.join(OUT, csvname)
    if os.path.exists(path):
        rows = list(csv.DictReader(open(path)))
        byK = defaultdict(list)
        for r in rows:
            byK[int(r["K"])].append(r)
        Ks = sorted(byK)
        if OBJECTIVE == "soc":
            strategies = [("ls_soc", "Local Search"), ("lns_soc", "MAPF-LNS"),
                          ("lnsls_soc", "LNS+Local")]
            base = "soc0"
        else:
            strategies = [("ls_T", "Local Search"), ("lns_T", "MAPF-LNS"), ("lnsls_T", "LNS+Local")]
            base = "T0"
        red = {(K, s): [(int(r[base]) - int(r[s])) / int(r[base]) for r in byK[K]]
               for K in Ks for s, _ in strategies}
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ["tab:green", "tab:orange", "tab:purple"]
        for j, (s, lab) in enumerate(strategies):
            pos = [i * 4 + j for i in range(len(Ks))]
            bp = ax.boxplot([red[(K, s)] for K in Ks], positions=pos, widths=0.8,
                            patch_artist=True, flierprops={"markersize": 2})
            for b in bp["boxes"]:
                b.set_facecolor(colors[j]); b.set_alpha(0.5)
            ax.plot(pos, [sum(red[(K, s)]) / len(red[(K, s)]) for K in Ks], "kD", markersize=4)
            ax.plot([], [], "s", color=colors[j], alpha=0.5, label=lab)
        ax.set_xticks([i * 4 + 1 for i in range(len(Ks))])
        ax.set_xticklabels([f"K={K}" for K in Ks])
        ax.set_ylabel("reduction of the Sum-of-Costs" if OBJECTIVE == "soc"
                      else "reduction of the plan length")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "lns_reduction" + suffix + ".pdf")); plt.close(fig)
        tcols = [("ls_time", "Local Search"), ("lns_time", "MAPF-LNS"), ("lnsls_time", "LNS+Local")]
        fig, ax = plt.subplots(figsize=(6, 3))
        for j, (s, lab) in enumerate(tcols):
            pos = [i * 4 + j for i in range(len(Ks))]
            data = [[float(r[s]) for r in byK[K]] for K in Ks]
            bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True,
                            flierprops={"markersize": 2})
            for b in bp["boxes"]:
                b.set_facecolor(colors[j]); b.set_alpha(0.5)
            ax.plot(pos, [sum(d) / len(d) for d in data], "kD", markersize=4)
            ax.plot([], [], "s", color=colors[j], alpha=0.5, label=lab)
        ax.set_xticks([i * 4 + 1 for i in range(len(Ks))])
        ax.set_xticklabels([f"K={K}" for K in Ks])
        ax.set_ylabel("time (s)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "lns_time" + suffix + ".pdf")); plt.close(fig)
        cnt = {}
        for K in Ks:
            rs = byK[K]
            T0 = lambda r: int(r[base])
            lc, nc = strategies[0][0], strategies[1][0]
            cnt[K] = [sum(int(r[lc]) < T0(r) for r in rs),
                      sum(int(r[nc]) < T0(r) for r in rs),
                      sum(int(r[lc]) < int(r[nc]) for r in rs),
                      sum(int(r[nc]) >= T0(r) and int(r[lc]) < T0(r) for r in rs),
                      len(rs)]
        fig, ax = plt.subplots(figsize=(6, 3))
        labs = ["LS improves", "LNS improves", "LS better than LNS", "LNS fails, LS improves"]
        for j, lab in enumerate(labs):
            ax.bar([i * 5 + j for i in range(len(Ks))], [cnt[K][j] for K in Ks], label=lab)
        ax.set_xticks([i * 5 + 1.5 for i in range(len(Ks))])
        ax.set_xticklabels([f"K={K}" for K in Ks])
        ax.set_ylabel("instances"); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "localvslns" + suffix + ".pdf")); plt.close(fig)
        numbers += [f"## Section 5.3 ({title}; objective and reductions: {OBJECTIVE})", "",
                    "| K | instances | avg red. Local Search | avg red. MAPF-LNS | avg red. LNS+Local | "
                    "median time LNS (s) | avg time LS (s) | LS improves | LNS improves | LS < LNS | LNS fails & LS improves | LNS worsens makespan |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for K in Ks:
            rs = byK[K]
            med = sorted(float(r["lns_time"]) for r in rs)[len(rs) // 2]
            avg = lambda s: sum(red[(K, s)]) / len(rs)
            worse = sum(int(r["lns_T"]) > int(r["T0"]) for r in rs)
            a1, a2, a3 = (avg(x) for x, _ in strategies)
            numbers.append(f"| {K} | {len(rs)} | {a1:.3f} | {a2:.3f} | "
                           f"{a3:.3f} | {med:.2f} | "
                           f"{sum(float(r['ls_time']) for r in rs)/len(rs):.2f} | "
                           + " | ".join(str(x) for x in cnt[K][:4]) + f" | {worse} |")


def main(argv=None):
    global OUT, OBJECTIVE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "plot"])
    ap.add_argument("experiment", nargs="?", choices=["digraphs", "lns"])
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--expansions", type=int, default=5_000_000)
    ap.add_argument("--lns-time", type=float, default=30.0)
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--objective", choices=["soc", "makespan"], default="soc")
    ap.add_argument("--init", choices=["ours", "official"], default="ours",
                    help="Section 5.3: initial plan from our PP or from the official code")
    a = ap.parse_args(argv)
    OUT = a.out
    OBJECTIVE = a.objective
    os.makedirs(OUT, exist_ok=True)
    if a.action == "plot":
        return plot()
    if a.experiment == "digraphs":
        run_digraphs(a.instances, a.time_limit, a.expansions, a.sizes)
    elif a.experiment == "lns":
        run_lns(a.instances, a.lns_time, a.time_limit, a.expansions, a.init)


if __name__ == "__main__":
    main()
