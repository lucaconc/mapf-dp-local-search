# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental protocol for the paper.

    python -m mapf_bench.paper_suite run   [--time 20] [--groups S D M H W]
    python -m mapf_bench.paper_suite analyze

Five groups of scenarios (fixed seeds), six methods with the same total time
budget T on one thread, one CSV row per (instance, method); the analysis
writes Markdown and LaTeX tables, statistical tests and figures into
paper_results/.

Groups
  S  structured: full rings on grids and on directed graphs, convoys
  D  directed graphs: random digraphs and one-way grids with serialized
     plans, dense one-way grids and random digraphs (80-90% occupancy)
  M  MovingAI standard maps (empty, random, maze, room 32x32) with 20 and 40
     agents, initial plans from prioritized planning
  H  MovingAI small maps at 80-98% occupancy, initial plans = first LaCAM3
     solution
  W  dense 4x5 grids with plans from backward random walks (70-90%)
  C  grids with many width-1 corridors (warehouse, city blocks, braided maze)
     at 50-98% occupancy, initial plans = first LaCAM3 solution
  K  deadlock-prone instances: corridor swaps with few passing bays, dead-end
     corridors filled in random order, order reversal with small pockets,
     crossroads of single-width arms (mapf_bench.deadlock)
  L  the MovingAI instances of group M (and 100 agents) on maps with reduced
     connectivity: up to all the blockable cells blocked, the map staying
     connected; initial plans = first LaCAM3 solution

Methods (total budget T each)
  LNS            MAPF-LNS (official code on grids, re-implementation on
                 directed graphs), from the initial plan
  LaCAM*         LaCAM3 with its anytime refinement, from scratch (grids)
  LS             adaptive VND (retiming, order swaps, route-lag, grouped
                 clocks, agent distance, free pairs in corridors), from the
                 initial plan
  Hybrid         LNS phases alternated with LS, final LS (certifying)
  LNS+LS         LNS for T/2, then LS for T/2
  LaCAM*+Hybrid  LaCAM* for T/2, then Hybrid for T/2 (grids)
  LSB            the descent of LS with large radii and a bounded open list
                 (beam of BEAM states per search): not exact, larger moves
  LNS+LSB        LNS for T/2, then LSB for T/2
  LaCAM*+LNS+LSB LaCAM* for T/2 (as LaCAM*+LNS), MAPF-LNS for T/4, then LSB for
                 T/4 (grids)
  LaCAM*+LNS     LaCAM* for T/2, then MAPF-LNS for T/2 (grids): the baseline
                 that isolates what the local search adds after LaCAM*
"""
import argparse
import csv
import json
import math
import os
import time
import zlib
from collections import defaultdict

from .algorithms import order_corridor_neighborhoods, order_reduced_neighborhoods
from .compare_vnd import FAMILIES as CV
from .cpp_local_search import cpp_local_search
from .hybrid import hybrid
from .lns import mapf_lns
from .official_lns import is_grid_instance, official_lns
from .plan import makespan, sum_of_costs, trim, validate

OUT = "paper_results"
FIELDS = ["group", "family", "instance", "n", "V", "occupancy", "init", "soc0", "mk0", "lb",
          "method", "soc", "makespan", "time", "certified", "error"]
METHODS = ["LNS", "LaCAM*", "LS", "Hybrid", "LNS+LS", "LaCAM*+Hybrid", "LaCAM*+LNS", "LSB", "LNS+LSB",
           "LaCAM*+LNS+LSB", "LSBX"]
# also available (not in the default list): "LSX", "LNS+LSX", "LaCAM*+LNS+LSBX"
THREADS = 1          # parallel candidates of the free-set neighborhoods (1: sequential, as in the protocol)
LSBX_FREE_FIRST = True   # LSBX on undirected graphs: free sets before the route-lag/grouped/agent ladders
CHAIN = True         # chain forced moves in the C++ searches (faster; same optimal cost per search)
BEAM = 5000          # beam width of LSB (open-list states kept by every beam search)
GRID_ONLY = {"LaCAM*", "LaCAM*+Hybrid", "LaCAM*+LNS", "LaCAM*+LNS+LSB", "LaCAM*+LNS+LSBX"}
# methods that start from scratch (LaCAM*) and so do not depend on the initial
# plan: in group P (same instances, PIBT initial plans) they would repeat the
# results of the original groups, so they are not run there
PLAN_INDEPENDENT = GRID_ONLY


def skip_method(m, inst, group):
    from .lacam import is_grid_instance
    return (m in GRID_ONLY and not is_grid_instance(inst)) or (group == "P" and m in PLAN_INDEPENDENT)


def _pibt_initial(build):
    """The instance of `build` with the PIBT plan as initial plan (None if PIBT
    does not reach all the goals); the name gets the suffix _pibt."""
    import sys
    from .pibt import pibt
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    inst, _ = build()
    p = pibt(inst.graph, inst.starts, inst.goals, seed=zlib.crc32(inst.name.encode()) % 1000)
    inst.name = inst.name + "_pibt"
    inst.initial_plan = trim(p) if p is not None else None
    return inst, "PIBT"


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
def scenarios():
    """(group, family, builder) with builder() -> (Instance, init label)."""
    from . import movingai as M
    out = []
    for fam in ("grid_rings", "full_rings", "convoy"):
        for s in range(5):
            out.append(("S", fam, (lambda f=fam, s=s: (CV[f](s), "given"))))
    for fam in ("random_serial", "oneway_serial"):
        for s in range(5):
            out.append(("D", fam, (lambda f=fam, s=s: (CV[f](s), "serialized PP"))))
    for fam in ("dense_oneway", "dense_random"):
        for s in (3, 4, 8, 9, 13):              # densities 0.8 and 0.9
            out.append(("D", fam, (lambda f=fam, s=s: (CV[f](s), "random walk"))))
    for m in ("empty-32-32", "random-32-32-20", "maze-32-32-4", "room-32-32-4"):
        scens = M.scenarios(m, 3)
        for n in (20, 40):
            for i in range(3):
                out.append(("M", f"{m}_n{n}",
                            (lambda m=m, n=n, sc=scens[i]: (M.instance(m, sc, n), "PP"))))
    for m in M.DENSE_MAPS:
        scens = M.dense_scenarios(m, 3)
        for o in (0.8, 0.9, 0.95, 0.98):
            for i in range(3):
                out.append(("H", f"{m}_o{o:.2f}",
                            (lambda m=m, o=o, sc=scens[i]: (M.dense_instance(m, sc, o), "LaCAM"))))
    for s in (2, 3, 4, 7, 8):
        out.append(("W", "dense_grid", (lambda s=s: (CV["dense_grid"](s), "random walk"))))
    # L: the MovingAI instances of group M (same maps, scenarios and agents, plus
    # 100 agents) on maps with reduced connectivity: free cells are blocked in a
    # random order while the map stays connected and the starts and goals stay
    # free (movingai.removal_sequence); level r blocks the first fraction r of
    # the sequence (0: original map, 1: minimal connected map)
    for m in L_MAPS:
        scens = M.scenarios(m, 2)
        for n in L_AGENTS:
            for lv in L_LEVELS:
                for i in range(2):
                    out.append(("L", f"{m}_n{n}_r{lv:.2f}",
                                (lambda m=m, n=n, lv=lv, sc=scens[i]:
                                 (M.reduced_instance(m, sc, n, lv), "LaCAM"))))
    # K: deadlock-prone instances (corridor swaps, dead-end filling, order
    # reversal, crossroads; mapf_bench.deadlock)
    from . import deadlock as DL
    for fam, gen, args in DL.CONFIGS:
        for sd in DL.SEEDS:
            out.append(("K", f"{fam}_" + "_".join(map(str, args)),
                        (lambda fam=fam, gen=gen, args=args, sd=sd:
                         (DL.instance(fam, gen, args, sd), "LaCAM"))))
    # C: grids with many width-1 corridors (warehouse, city blocks, braided
    # maze) at 50-85% occupancy (mapf_bench.corridors)
    from . import corridors as CO
    for occs in (CO.OCCUPANCY, CO.OCCUPANCY_HIGH):      # 90-98% after the first block
        for fam, args in CO.CONFIGS:
            for occ in occs:
                for sd in CO.SEEDS:
                    out.append(("C", f"{fam}_" + "_".join(map(str, args)),
                                (lambda fam=fam, args=args, occ=occ, sd=sd:
                                 (CO.instance(fam, args, occ, sd), "LaCAM"))))
    # P: the instances of groups M, H, W, L, K with the plan of PIBT (a
    # rule-based solver) as initial plan, where PIBT reaches all the goals
    # (on the corridor grids of group C it never does)
    base = [(g, fam, b) for g, fam, b in out if g in ("M", "H", "W", "L", "K")]
    for g0, fam, b in base:
        out.append(("P", f"{g0}:{fam}", (lambda b=b: _pibt_initial(b))))
    return out


L_MAPS = ("empty-32-32", "random-32-32-20", "room-32-32-4", "maze-32-32-4")
L_AGENTS = (20, 40, 100)
L_LEVELS = (0.0, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------
def _lns(inst, plan, t, seed):
    from .official_lns import use_official
    if use_official(inst):
        p, _ = official_lns(inst, plan, time_limit=t, N=5, seed=seed)
    else:
        p, _ = mapf_lns(inst, plan, objective="soc", N=5, max_iters=10**6, time_limit=t, seed=seed)
    return trim(p), None


def _ls(inst, plan, t, seed, stats=None):
    p, st = cpp_local_search(inst, plan, order_corridor_neighborhoods(), tiebreak="summin",
                             objective="soc_proxy", vnd=True, astar=True, keep_radius=True,
                             time_limit=t, max_expansions=2_000_000, chain=CHAIN, threads=THREADS)
    if stats is not None:
        stats.update(st)
    return trim(p), (not st.get("timeout")) and st.get("budget_exceeded", 0) == 0


def _lsx(inst, plan, t, seed):
    """LS with the LNS/LaCAM neighborhood (algorithms.lnsx_neighborhoods)."""
    from .algorithms import lnsx_neighborhoods
    p, st = cpp_local_search(inst, plan, lnsx_neighborhoods(), tiebreak="summin",
                             objective="soc_proxy", vnd=True, astar=True, keep_radius=True,
                             time_limit=t, max_expansions=2_000_000, chain=CHAIN, threads=THREADS)
    return trim(p), ((not st.get("timeout")) and st.get("budget_exceeded", 0) == 0
                     and st.get("pruned", 0) == 0)


def _lsb(inst, plan, t, seed, extra=False):
    # extra (LSBX): the LNS/LaCAM free sets are added to the escape phase, which
    # starts whenever the exact descent has ended before the deadline (all its
    # neighborhoods tried without improvement), also when some of its searches
    # exceeded their budget; the plan is certified only if the exact descent
    # certified it, as in LSB
    """Exact descent, then large-radius beam descent as an escape step.

    The exact local search (LS: small adaptive radii, exact A*) runs until it
    certifies a local optimum; only then the descent over the same kinds of
    neighborhoods with much larger radii is run with a bounded open list (the
    BEAM most promising states per search), from that plan. If it finds an
    improvement, the exact descent restarts from the improved plan. So LSB
    keeps every improvement that LS finds, and uses the beam only to leave
    certified local optima; the returned plan is certified if the last exact
    descent certified it. Where LS never reaches a local optimum within the
    budget, LSB coincides with LS."""
    from .algorithms import beam_neighborhoods
    deadline = time.perf_counter() + t
    cur, cert = trim(plan), False
    while True:
        left = deadline - time.perf_counter()
        if left <= 0.05:
            break
        st_ls = {}
        cur, cert = _ls(inst, cur, left, seed, stats=st_ls)
        left = deadline - time.perf_counter()
        ended = not st_ls.get("timeout")
        if left <= 0.05 or not (cert or (extra and ended)):
            break                          # no (certified) local optimum, or no time left
        before = sum_of_costs(cur, inst.goals)
        nbs = beam_neighborhoods(inst.n)
        if extra:                          # LSBX: the LNS/LaCAM free sets in the escape phase too
            from .algorithms import lnsx_extra_neighborhoods
            from .cpp_local_search import FreeSet
            undirected = all(u in inst.graph.out[v] for u in range(inst.graph.N) for v in inst.graph.out[u])
            if LSBX_FREE_FIRST and undirected:
                # on grids: the free sets (cheap, and the ones that improve) before
                # the expensive ladders, which otherwise use the whole budget; on
                # directed graphs the route-lag improvements prepare those of the
                # other neighborhoods, so the order of LSB is kept
                # the free sets (cheap, and the ones that improve) before the
                # expensive ladders, which otherwise use the whole budget
                cheap = [x for x in nbs if not hasattr(x, "template")]
                free = [x for x in cheap if isinstance(x, FreeSet)]
                order = [x for x in cheap if not isinstance(x, FreeSet)]
                ladders = [x for x in nbs if hasattr(x, "template")]
                nbs = order + free + lnsx_extra_neighborhoods() + ladders
            else:
                nbs = nbs + lnsx_extra_neighborhoods()
        p, st = cpp_local_search(inst, cur, nbs, tiebreak="summin",
                                 objective="soc_proxy", vnd=True, astar=True, keep_radius=True,
                                 time_limit=left, max_expansions=2_000_000, beam=BEAM, chain=CHAIN, threads=THREADS)
        p = trim(p)
        if sum_of_costs(p, inst.goals) >= before:
            break                          # the beam found no escape: keep the certified plan
        cur, cert = p, False               # improved: certify again with the exact descent
    return cur, cert


def _hybrid(inst, plan, t, seed):
    p, st = hybrid(inst, plan, time_limit=t, seed=seed, lns_first=True, lns_phase=min(4.0, t / 5),
                   neighborhoods=order_reduced_neighborhoods,
                   final_neighborhoods=order_corridor_neighborhoods, keep_radius=True)
    return trim(p), st["certified"]


def _lacam(inst, t, seed):
    from .lacam import lacam_solve
    p = lacam_solve(inst, time_limit=t, anytime=True, seed=seed)
    return (trim(p) if p is not None else None), None


def run_method(name, inst, plan0, T, seed):
    if name == "LNS":
        return _lns(inst, plan0, T, seed)
    if name == "LS":
        return _ls(inst, plan0, T, seed)
    if name == "Hybrid":
        return _hybrid(inst, plan0, T, seed)
    if name == "LNS+LS":
        p, _ = _lns(inst, plan0, T / 2, seed)
        return _ls(inst, p, T / 2, seed)
    if name == "LSB":
        return _lsb(inst, plan0, T, seed)
    if name == "LSBX":             # LSB with the LNS/LaCAM free sets in its escape phase
        return _lsb(inst, plan0, T, seed, extra=True)
    if name == "LSX":              # LS with the LNS/LaCAM neighborhood (not in the default METHODS)
        return _lsx(inst, plan0, T, seed)
    if name == "LNS+LSX":
        p, _ = _lns(inst, plan0, T / 2, seed)
        return _lsx(inst, p, T / 2, seed)
    if name == "LNS+LSB":
        p, _ = _lns(inst, plan0, T / 2, seed)
        return _lsb(inst, p, T / 2, seed)
    if name == "LaCAM*":
        return _lacam(inst, T, seed)
    if name == "LaCAM*+Hybrid":
        p, _ = _lacam(inst, T / 2, seed)
        return _hybrid(inst, p if p is not None else plan0, T / 2, seed)
    if name == "LaCAM*+LNS+LSBX":  # LaCAM* for T/2, MAPF-LNS for T/4, LSBX for T/4
        p, _ = _lacam(inst, T / 2, seed)
        p, _ = _lns(inst, p if p is not None else plan0, T / 4, seed)
        return _lsb(inst, p, T / 4, seed, extra=True)
    if name == "LaCAM*+LNS+LSB":   # LaCAM* for T/2, MAPF-LNS for T/4, LSB for T/4
        p, _ = _lacam(inst, T / 2, seed)
        p, _ = _lns(inst, p if p is not None else plan0, T / 4, seed)
        return _lsb(inst, p, T / 4, seed)
    if name == "LaCAM*+LNS":       # baseline: what MAPF-LNS alone adds after LaCAM*
        p, _ = _lacam(inst, T / 2, seed)
        return _lns(inst, p if p is not None else plan0, T / 2, seed)
    raise ValueError(name)


def lower_bound(inst):
    G = inst.graph
    return sum(int(G.bfs(g, reverse=True)[s]) for s, g in zip(inst.starts, inst.goals))


def run(out=OUT, T=20.0, groups=("S", "D", "M", "H", "W", "L", "K", "C", "P"), methods=METHODS, only=None):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.csv")
    done = set()
    if os.path.exists(path):
        done = {(r["instance"], r["method"]) for r in csv.DictReader(open(path))}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for idx, (g, fam, build) in enumerate(scenarios()):
            if g not in groups or (only is not None and idx not in only):
                continue
            inst, init = build()
            if inst.initial_plan is None:
                if (inst.name, "none") not in done:
                    w.writerow({"group": g, "family": fam, "instance": inst.name, "n": inst.n,
                                "V": inst.graph.N, "method": "none", "error": "no initial plan"})
                    fh.flush()
                continue
            todo = [m for m in methods if (inst.name, m) not in done
                    and not skip_method(m, inst, g)]
            if not todo:
                continue
            p0 = trim(inst.initial_plan)
            validate(inst.graph, inst.starts, inst.goals, p0)
            base = {"group": g, "family": fam, "instance": inst.name, "n": inst.n,
                    "V": inst.graph.N, "occupancy": round(inst.n / inst.graph.N, 3), "init": init,
                    "soc0": sum_of_costs(p0, inst.goals), "mk0": makespan(p0),
                    "lb": lower_bound(inst)}
            seed = zlib.crc32(inst.name.encode()) % 1000
            line = []
            for m in todo:
                t0 = time.perf_counter()
                try:
                    p, cert = run_method(m, inst, p0, T, seed)
                    err = ""
                    if p is None:
                        raise RuntimeError("no plan")
                    validate(inst.graph, inst.starts, inst.goals, p)
                    soc, mk = sum_of_costs(p, inst.goals), makespan(p)
                except Exception as e:      # recorded, the run continues
                    soc = mk = ""
                    cert, err = None, f"{type(e).__name__}: {e}"[:200]
                w.writerow({**base, "method": m, "soc": soc, "makespan": mk,
                            "time": round(time.perf_counter() - t0, 3),
                            "certified": "" if cert is None else int(bool(cert)), "error": err})
                fh.flush()
                line.append(f"{m} {soc}")
            print(g, inst.name, f"init {base['soc0']} lb {base['lb']} |", " | ".join(line), flush=True)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
GROUP_NAMES = {"S": "Structured (full rings, convoys)",
               "D": "Directed graphs (serialized and dense)",
               "M": "MovingAI maps, 20-40 agents (PP plans)",
               "H": "MovingAI maps, 80-98% occupancy",
               "W": "Dense grids, random-walk plans",
               "L": "MovingAI maps with reduced connectivity",
               "K": "Deadlock-prone instances",
               "C": "Corridor grids at high density",
               "P": "PIBT initial plans"}
PAIRS = [("LS", "LNS"), ("Hybrid", "LNS"), ("LNS+LS", "LNS"), ("Hybrid", "LaCAM*"),
         ("LaCAM*+Hybrid", "LaCAM*"), ("LaCAM*+Hybrid", "LNS"),
         ("Hybrid", "LNS+LS"), ("LaCAM*+Hybrid", "LaCAM*+LNS"), ("LaCAM*+LNS", "LaCAM*"),
         ("LSB", "LS"), ("LNS+LSB", "LNS+LS"), ("LNS+LSB", "LNS"),
         ("LaCAM*+LNS+LSB", "LaCAM*+LNS"), ("LaCAM*+LNS+LSB", "LaCAM*"),
         ("LSBX", "LSB"), ("LSBX", "LS"),
         ("LaCAM*+LNS+LSBX", "LaCAM*+LNS"), ("LaCAM*+LNS+LSBX", "LaCAM*+LNS+LSB")]


def _load(out):
    rows = [r for r in csv.DictReader(open(os.path.join(out, "results.csv")))]
    inst = defaultdict(dict)
    meta = {}
    skipped = []
    for r in rows:
        if r["method"] == "none":
            skipped.append(r["instance"])
            continue
        meta[r["instance"]] = r
        if r["soc"] != "":
            inst[r["instance"]][r["method"]] = r
    return inst, meta, skipped


def _sign(w, l):
    m = w + l
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(min(w, l) + 1)) / 2 ** m)


def _wilcoxon(a, b):
    from scipy.stats import wilcoxon
    d = [x - y for x, y in zip(a, b)]
    if not any(d):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def _holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def analyze(out=OUT, T=None):
    inst, meta, skipped = _load(out)
    if not inst:
        rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
        errs = defaultdict(int)
        for r in rows:
            if r.get("error"):
                errs[r["error"][:120]] += 1
        text = ("# Experimental results\n\nNo successful results yet in this folder.\n\n"
                + ("Errors recorded (count, message):\n\n" + "\n".join(f"- {k} x {e}" for e, k in
                   sorted(errs.items(), key=lambda x: -x[1])) + "\n" if errs else ""))
        open(os.path.join(out, "results.md"), "w").write(text)
        return text
    present = {meta[i]["group"] for i in inst}
    groups = [g for g in "SDMHWLKCP" if g in present] + sorted(present - set("SDMHWLKCP"))
    md, tex = [], []
    md += ["# Experimental results", "",
           f"{len(inst)} instances in {len(groups)} groups; every method has the same budget of "
           f"T = {T if T is not None else 20:g} s on one thread. SoC/LB: Sum-of-Costs divided by the sum of shortest-path "
           "lengths (geometric mean). Gap: mean relative gap to the best method on the instance. "
           "Best: instances where the method attains the best Sum-of-Costs (ties counted for "
           "all). Cert.: final plans certified as local optima of all neighborhoods."]
    if skipped:
        md.append(f"Skipped (no initial plan found by LaCAM3 in 30 s): {', '.join(sorted(set(skipped)))}.")
    # ---- summary per group
    summary = {}
    for g in groups + ["all"]:
        ks = [i for i in inst if g == "all" or meta[i]["group"] == g]
        rows = []
        for m in METHODS:
            vals = [i for i in ks if m in inst[i]]
            if not vals:
                continue
            best = {i: min(int(r["soc"]) for r in inst[i].values()) for i in vals}
            ratio = math.exp(sum(math.log(int(inst[i][m]["soc"]) / max(int(meta[i]["lb"]), 1))
                                 for i in vals) / len(vals))
            gap = sum(int(inst[i][m]["soc"]) / best[i] - 1 for i in vals) / len(vals)
            nbest = sum(int(inst[i][m]["soc"]) == best[i] for i in vals)
            tm = sum(float(inst[i][m]["time"]) for i in vals) / len(vals)
            cert = [inst[i][m]["certified"] for i in vals if inst[i][m]["certified"] != ""]
            rows.append((m, len(vals), ratio, gap, nbest, tm,
                         (sum(int(c) for c in cert) / len(cert)) if cert else None))
        summary[g] = rows
        title = "All instances" if g == "all" else GROUP_NAMES.get(g, g)
        md += ["", f"## {title} ({len(ks)} instances)", "",
               "| method | inst. | SoC/LB | gap to best | best | time (s) | cert. |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for m, n, ratio, gap, nbest, tm, cert in rows:
            md.append(f"| {m} | {n} | {ratio:.3f} | {gap:.1%} | {nbest} | {tm:.1f} | "
                      f"{'' if cert is None else f'{cert:.0%}'} |")
    # ---- pairwise tests
    md += ["", "## Pairwise comparisons", "",
           "W/T/L: instances where the first method has lower / equal / higher Sum-of-Costs; "
           "p-values of the two-sided sign test and Wilcoxon signed-rank test, Holm-adjusted "
           "over all the comparisons of the table.", "",
           "| group | comparison | inst. | W/T/L | median rel. diff. | sign p | Wilcoxon p | Holm p |",
           "|---|---|---:|:---:|---:|---:|---:|---:|"]
    tests = []
    for g in groups + ["all"]:
        ks = [i for i in inst if g == "all" or meta[i]["group"] == g]
        for a, b in PAIRS:
            pk = [i for i in ks if a in inst[i] and b in inst[i]]
            if not pk:
                continue
            A = [int(inst[i][a]["soc"]) for i in pk]
            B = [int(inst[i][b]["soc"]) for i in pk]
            w = sum(x < y for x, y in zip(A, B)); t = sum(x == y for x, y in zip(A, B))
            l = len(pk) - w - t
            rel = sorted((x - y) / y for x, y in zip(A, B))
            med = rel[len(rel) // 2] if len(rel) % 2 else (rel[len(rel) // 2 - 1] + rel[len(rel) // 2]) / 2
            tests.append([g, a, b, len(pk), w, t, l, med, _sign(w, l), _wilcoxon(A, B)])
    holm = _holm([x[9] for x in tests])
    for x, h in zip(tests, holm):
        g, a, b, n, w, t, l, med, ps, pw = x
        md.append(f"| {g} | {a} vs {b} | {n} | {w}/{t}/{l} | {med:+.1%} | {ps:.3g} | {pw:.3g} | {h:.3g} |")
    # ---- per family (appendix)
    md += ["", "## Per family (mean Sum-of-Costs)", "",
           "| group | family | inst. | n | LB | initial | " + " | ".join(METHODS) + " |",
           "|---|---|---:|---:|---:|---:|" + "---:|" * len(METHODS)]
    fams = defaultdict(list)
    for i in inst:
        fams[(meta[i]["group"], meta[i]["family"])].append(i)
    for (g, f) in sorted(fams, key=lambda x: (groups.index(x[0]), x[1])):
        ks = fams[(g, f)]
        cells = []
        for m in METHODS:
            v = [int(inst[i][m]["soc"]) for i in ks if m in inst[i]]
            cells.append(f"{sum(v)/len(v):.1f}" if len(v) == len(ks) else ("" if not v else "*"))
        mean = lambda c: sum(float(meta[i][c]) for i in ks) / len(ks)
        md.append(f"| {g} | {f} | {len(ks)} | {mean('n'):.0f} | {mean('lb'):.1f} | {mean('soc0'):.1f} | "
                  + " | ".join(cells) + " |")
    text = "\n".join(md) + "\n"
    open(os.path.join(out, "results.md"), "w").write(text)
    _latex(summary, tests, holm, out)
    _figures(inst, meta, groups, out)
    return text


def _latex(summary, tests, holm, out):
    L = [r"% generated by mapf_bench.paper_suite.analyze", r"\begin{table}[t]", r"\centering\small",
         r"\caption{Results by group of scenarios ($T=20$\,s per method, one thread). "
         r"SoC/LB: geometric mean of the Sum-of-Costs over the sum of shortest-path lengths; "
         r"gap: mean relative gap to the best method on each instance; best: number of "
         r"instances where the method attains the best Sum-of-Costs.}",
         r"\label{tab:results}", r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"group & method & inst. & SoC/LB & gap & best & time (s)\\", r"\midrule"]
    for g, rows in summary.items():
        for j, (m, n, ratio, gap, nbest, tm, cert) in enumerate(rows):
            name = (("all" if g == "all" else GROUP_NAMES.get(g, g)).split(" (")[0] if j == 0 else "")
            L.append(f"{name} & {m} & {n} & {ratio:.3f} & {100*gap:.1f}\\% & {nbest} & {tm:.1f}\\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}", "",
          r"\begin{table}[t]", r"\centering\small",
          r"\caption{Pairwise comparisons of the Sum-of-Costs: wins/ties/losses of the first "
          r"method, median relative difference, and Holm-adjusted $p$-values of the Wilcoxon "
          r"signed-rank test.}", r"\label{tab:pairs}", r"\begin{tabular}{llrcrr}", r"\toprule",
          r"group & comparison & inst. & W/T/L & median diff. & $p$ (Holm)\\", r"\midrule"]
    for x, h in zip(tests, holm):
        g, a, b, n, w, t, l, med, ps, pw = x
        L.append(f"{g} & {a} vs {b} & {n} & {w}/{t}/{l} & {100*med:+.1f}\\% & {h:.2g}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out, "tables.tex"), "w").write("\n".join(L) + "\n")


# line style of each method: colour by family, dash pattern by variant
PROFILE_STYLE = {
    "LNS": ("#1f77b4", "-"), "LNS+LS": ("#1f77b4", "--"), "LNS+LSB": ("#1f77b4", ":"),
    "LNS+LSX": ("#1f77b4", "-."), "Hybrid": ("#9467bd", "-"),
    "LaCAM*": ("#d62728", "-"), "LaCAM*+LNS": ("#d62728", "--"), "LaCAM*+LNS+LSB": ("#d62728", ":"),
    "LaCAM*+Hybrid": ("#ff7f0e", "-"), "LaCAM*+LNS+LSBX": ("#ff7f0e", "--"),
    "LS": ("#2ca02c", "-"), "LSB": ("#2ca02c", "--"), "LSBX": ("#2ca02c", ":"), "LSX": ("#17becf", "-")}
PROFILE_TITLE = {"all": "All instances (except P)", "S": "S: structured", "D": "D: digraphs", "M": "M: MovingAI",
                 "H": "H: 80-98% occupancy", "W": "W: dense grids", "L": "L: reduced connectivity",
                 "K": "K: deadlock-prone", "C": "C: corridor grids", "P": "P: PIBT initial plans"}


def _figures(inst, meta, groups, out):
    """Performance profiles, drawn at the size of a printed page: a grid of
    three columns (all the instances first, then one panel per group), the
    legend in the free cells, and every method with its own colour and dash
    pattern (colour = family of methods)."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    taus = [1 + 0.01 * k for k in range(0, 101)]
    panels = ["all"] + list(groups)
    ncol = 3
    nrow = math.ceil((len(panels) + 1) / ncol)          # at least one free cell for the legend
    fig = plt.figure(figsize=(6.5, 1.85 * nrow))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.18)
    handles = {}
    for k, g in enumerate(panels):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        # group P repeats instances of other groups (other initial plans): it is
        # not counted again in the panel of all the instances
        ks = [i for i in inst if (g == "all" and meta[i]["group"] != "P") or meta[i]["group"] == g]
        for m in METHODS:
            vals = [i for i in ks if m in inst[i]]
            if not vals:
                continue
            ratios = [int(inst[i][m]["soc"]) / min(int(r["soc"]) for r in inst[i].values()) for i in vals]
            col, ls = PROFILE_STYLE.get(m, ("black", "-"))
            line, = ax.plot(taus, [sum(r <= t for r in ratios) / len(ks) for t in taus],
                            color=col, linestyle=ls, linewidth=1.3, label=m)
            handles.setdefault(m, line)
        ax.set_title(f"{PROFILE_TITLE.get(g, g)} ({len(ks)})")
        ax.set_xlim(1, 2); ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 1.5, 2]); ax.set_xticklabels(["1", "1.5", "2"]); ax.set_yticks([0, 0.5, 1])
        ax.grid(alpha=0.3)
        if k % ncol == 0:
            ax.set_ylabel("fraction")
        else:
            ax.set_yticklabels([])
        if k // ncol == nrow - 1 or k + ncol >= len(panels):
            ax.set_xlabel(r"$\tau$ (SoC / best)")
    # legend in the free cells of the last row
    first_free = len(panels)
    lax = fig.add_subplot(gs[first_free // ncol, first_free % ncol:])
    lax.axis("off")
    order = [m for m in METHODS if m in handles]
    lax.legend([handles[m] for m in order], order, loc="center", frameon=False,
               ncol=2 if ncol - first_free % ncol >= 2 else 1, handlelength=2.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"performance_profiles.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["run", "analyze"])
    ap.add_argument("--time", type=float, default=20.0)
    ap.add_argument("--groups", nargs="+", default=["S", "D", "M", "H", "W", "L", "K", "C", "P"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    if a.action == "run":
        run(a.out, a.time, a.groups)
    print(analyze(a.out))


if __name__ == "__main__":
    main()
