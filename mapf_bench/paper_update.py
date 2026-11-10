# SPDX-License-Identifier: GPL-3.0-or-later
"""Update the paper from a results folder.

    python -m mapf_bench.paper_update results/paper --paper path/to/LocalSearch_new.tex
    python -m mapf_bench.paper_update results/paper --paper paper.tex --compile --strict

or, at the end of a run,

    python -m mapf_bench.local_runner run configs/paper.json --workers 8 --update-paper paper.tex

What is updated
  * bench_numbers.tex, next to the paper: one macro for every number cited in
    the benchmark section and in the conclusion (e.g. \\BenchRatioSLs), plus a
    few generated sentences; the paper inputs it in its preamble;
  * the blocks of the paper between %<AUTO:name> and %</AUTO:name> (the tables
    of results and of paired comparisons), regenerated whole;
  * images/tests/performance_profiles.pdf, next to the paper.
Before writing, the paper is backed up and a unified diff is saved in
<paper dir>/paper_update/. Prose is never rewritten: the qualitative claims of
the section are checked against the new results, and the report
(<paper dir>/paper_update/report.md) lists each claim as confirmed or
contradicted; --strict exits with an error if a claim is contradicted, so the
sentence can be revised by hand.
"""
import argparse
import datetime
import difflib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

from .paper_suite import METHODS, PAIRS as SUITE_PAIRS, _holm, _load, _sign, _wilcoxon

CODE = {"LNS": "Lns", "LaCAM*": "Lacam", "LS": "Ls", "Hybrid": "Hyb", "LNS+LS": "LnsLs",
        "LaCAM*+Hybrid": "LacamHyb", "LaCAM*+LNS": "LacamLns", "LSB": "Lsb", "LNS+LSB": "LnsLsb", "LaCAM*+LNS+LSB": "LacamLnsLsb", "LSBX": "Lsbx", "LaCAM*+LNS+LSBX": "LacamLnsLsbx"}
TEX = {"LNS": r"\textsc{LNS}", "LaCAM*": r"\textsc{LaCAM}$^\ast$", "LS": r"\textsc{LS}",
       "Hybrid": r"\textsc{Hybrid}", "LNS+LS": r"\textsc{LNS+LS}",
       "LaCAM*+Hybrid": r"\textsc{LaCAM}$^\ast$\textsc{+Hybrid}",
       "LaCAM*+LNS": r"\textsc{LaCAM}$^\ast$\textsc{+LNS}",
       "LSB": r"\textsc{LSB}", "LNS+LSB": r"\textsc{LNS+LSB}",
       "LaCAM*+LNS+LSB": r"\textsc{LaCAM}$^\ast$\textsc{+LNS+LSB}",
       "LSBX": r"\textsc{LSBX}",
       "LaCAM*+LNS+LSBX": r"\textsc{LaCAM}$^\ast$\textsc{+LNS+LSBX}"}
TEX_SHORT = {**TEX, "LaCAM*+Hybrid": r"\textsc{LaCAM}$^\ast$+\textsc{Hyb.}",
             "LaCAM*+LNS+LSB": r"\textsc{LaCAM}$^\ast$+\textsc{LNS+LSB}",
             "LaCAM*+LNS+LSBX": r"\textsc{LaCAM}$^\ast$+\textsc{LNS+LSBX}",
             "LaCAM*+LNS": r"\textsc{LaCAM}$^\ast$+\textsc{LNS}"}
PAIRS = [("S", "LS", "LNS"), ("D", "LS", "LNS"), ("D", "Hybrid", "LNS"), ("D", "LNS+LS", "LNS"),
         ("M", "LS", "LNS"), ("M", "Hybrid", "LNS"), ("M", "LNS+LS", "LNS"),
         ("M", "LaCAM*+Hybrid", "LaCAM*"), ("H", "LaCAM*+Hybrid", "LNS"),
         ("H", "LaCAM*+Hybrid", "LaCAM*"), ("H", "Hybrid", "LaCAM*"), ("W", "LaCAM*+Hybrid", "LaCAM*"),
         ("All", "LaCAM*+Hybrid", "LNS"), ("All", "LaCAM*+Hybrid", "LaCAM*"), ("All", "Hybrid", "LNS"),
         ("All", "LNS+LS", "LNS"), ("All", "LS", "LNS"), ("All", "Hybrid", "LaCAM*")]
TABLE_PAIRS = [("All", "LaCAM*+Hybrid", "LNS"), ("All", "LaCAM*+Hybrid", "LaCAM*"),
               ("All", "Hybrid", "LNS"), ("All", "LNS+LS", "LNS"), ("All", "LS", "LNS"),
               ("S", "LS", "LNS"), ("M", "LS", "LNS"), ("M", "LaCAM*+Hybrid", "LaCAM*"),
               ("H", "LaCAM*+Hybrid", "LNS"), ("H", "LaCAM*+Hybrid", "LaCAM*"),
               ("H", "Hybrid", "LaCAM*"),
               ("All", "LaCAM*+Hybrid", "LaCAM*+LNS"), ("H", "LaCAM*+Hybrid", "LaCAM*+LNS"),
               ("All", "Hybrid", "LNS+LS")]


# ---------------------------------------------------------------------------
# statistics (as in paper_suite.analyze)
# ---------------------------------------------------------------------------
def statistics(out):
    inst, meta, skipped = _load(out)
    cfgp = os.path.join(out, "config.json")
    T = float(json.load(open(cfgp))["time"]) if os.path.exists(cfgp) else 20.0
    groups = [g for g in "SDMHWLKCP" if any(meta[i]["group"] == g for i in inst)]
    sel = lambda g: [i for i in inst if g == "All" or meta[i]["group"] == g]
    S = {"T": T, "groups": groups, "inst": inst, "meta": meta, "skipped": sorted(set(skipped)),
         "method": {}, "pair": {}}
    for g in groups + ["All"]:
        for m in METHODS:
            ks = [i for i in sel(g) if m in inst[i]]
            if not ks:
                continue
            best = {i: min(int(r["soc"]) for r in inst[i].values()) for i in ks}
            soc = {i: int(inst[i][m]["soc"]) for i in ks}
            cert = [int(inst[i][m]["certified"]) for i in ks if inst[i][m]["certified"] != ""]
            S["method"][(g, m)] = {
                "n": len(ks),
                "ratio": math.exp(sum(math.log(soc[i] / max(int(meta[i]["lb"]), 1)) for i in ks) / len(ks)),
                "best": sum(soc[i] == best[i] for i in ks),
                "gap": sum(soc[i] / best[i] - 1 for i in ks) / len(ks),
                "time": sum(float(inst[i][m]["time"]) for i in ks) / len(ks),
                "cert": (sum(cert) / len(cert)) if cert else None}
    tests = []          # the comparisons of paper_suite.analyze, in its order (same Holm adjustment)
    for g, a, b in [(g, a, b) for g in groups + ["All"] for a, b in SUITE_PAIRS]:
        ks = [i for i in sel(g) if a in inst[i] and b in inst[i]]
        if not ks:
            continue
        A = [int(inst[i][a]["soc"]) for i in ks]; B = [int(inst[i][b]["soc"]) for i in ks]
        w = sum(x < y for x, y in zip(A, B)); t = sum(x == y for x, y in zip(A, B)); l = len(ks) - w - t
        rel = sorted((x - y) / y for x, y in zip(A, B))
        med = rel[len(rel) // 2] if len(rel) % 2 else (rel[len(rel) // 2 - 1] + rel[len(rel) // 2]) / 2
        tests.append(((g, a, b), {"n": len(ks), "w": w, "t": t, "l": l, "med": med,
                                  "p_sign": _sign(w, l), "p": _wilcoxon(A, B)}))
    for (key, d), h in zip(tests, _holm([d["p"] for _, d in tests])):
        d["holm"] = h
        S["pair"][key] = d
    mj = os.path.join(out, "machine.json")
    S["machine"] = json.load(open(mj)) if os.path.exists(mj) else {}
    return S


# ---------------------------------------------------------------------------
# macros
# ---------------------------------------------------------------------------
def _p(p):
    if p < 1e-4:
        return r"p<10^{-4}"
    if p < 0.001:
        e = math.floor(math.log10(p)); return rf"p={p / 10 ** e:.1f}\cdot10^{{{e}}}"
    return f"p={p:.2g}" if p < 0.1 else f"p={p:.2f}"


def _pt(p):          # table format
    if p >= 0.995:
        return "$1$"
    if p < 1e-3:
        e = math.floor(math.log10(p)); return rf"${p / 10 ** e:.1f}\cdot10^{{{e}}}$"
    return f"${p:.2g}$"


def _pct(x, signed=False):
    if abs(x) < 0.0005:
        return r"0.0\%"
    return (f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}") + r"\%"


def _machine(m):
    cpu = m.get("cpu") or "an unspecified processor"
    cpu = re.sub(r"\((R|TM|tm|r)\)", "", cpu)
    cpu = re.sub(r"\bCPU\b", "", cpu)
    cpu = re.sub(r"\s+@\s+", " at ", cpu)
    cpu = re.sub(r"(\d+(\.\d+)?)\s*GHz", r"$\1$\\,GHz", cpu)
    cpu = re.sub(r"\s+", " ", cpu).strip().replace("Processor", "processor")
    art = "an" if cpu[:1].lower() in "aeiou" else "a"
    mem = m.get("memory_gb")
    return f"{art} {cpu}" + (f" with about ${round(mem)}$\\,GB of RAM" if mem else "")


def _family_label(f):
    m = re.match(r"(.+)_o(0\.\d+)$", f)
    return f"{m.group(1)} at ${round(100 * float(m.group(2)))}\\%$" if m else f.replace("_", r"\_")


def macros(S):
    M = {}
    T = S["T"]
    M["BenchT"] = f"{T:g}"
    M["BenchMachine"] = _machine(S["machine"])
    n_skip = len(S["skipped"])
    M["BenchNSolved"] = str(len(S["inst"]))
    M["BenchNInst"] = str(len(S["inst"]) + n_skip)
    for g in S["groups"] + ["All"]:
        M[f"BenchN{g}"] = str(sum(1 for i in S["inst"] if g == "All" or S["meta"][i]["group"] == g))
    M["BenchNComparisons"] = str(len(S["pair"]))
    for (g, m), d in S["method"].items():
        c = CODE[m]
        M[f"BenchRatio{g}{c}"] = f"{d['ratio']:.3f}"
        M[f"BenchBest{g}{c}"] = str(d["best"])
        M[f"BenchGap{g}{c}"] = _pct(d["gap"])
        M[f"BenchTime{g}{c}"] = f"{d['time']:.1f}"
        if d["cert"] is not None:
            M[f"BenchCert{g}{c}"] = f"{round(100 * d['cert'])}\\%"
    for (g, a, b), d in S["pair"].items():
        k = f"BenchPair{g}{CODE[a]}Vs{CODE[b]}"
        M[k + "W"], M[k + "T"], M[k + "L"] = str(d["w"]), str(d["t"]), str(d["l"])
        M[k + "N"] = str(d["n"])
        M[k + "Med"] = _pct(d["med"], signed=True)
        M[k + "P"] = _p(d["holm"])
    inst, meta = S["inst"], S["meta"]
    # serialized plans: mean time of LS and LNS
    ser = [i for i in inst if meta[i]["family"] in ("random_serial", "oneway_serial")]
    for m in ("LS", "LNS"):
        v = [float(inst[i][m]["time"]) for i in ser if m in inst[i]]
        M[f"BenchTimeSerial{CODE[m]}"] = (f"{sum(v) / len(v):.1f}" if sum(v) / len(v) < 10
                                         else f"{sum(v) / len(v):.0f}") if v else "--"
    gaps = [int(inst[i]["LS"]["soc"]) / int(inst[i]["LNS"]["soc"]) - 1 for i in ser
            if "LS" in inst[i] and "LNS" in inst[i]]
    M["BenchSerialGapLs"] = _pct(sum(gaps) / len(gaps)) if gaps else "--"
    M["BenchSerialMaxGapLs"] = _pct(max(gaps)) if gaps else "--"
    # group H: range of the methods that start from the first LaCAM solution
    r = [S["method"][("H", m)]["ratio"] for m in ("LNS", "LS", "Hybrid", "LNS+LS") if ("H", m) in S["method"]]
    # instances on which MAPF-LNS does not improve the initial plan, by group
    for g in S["groups"] + ["All"]:
        ks = [i for i in inst if (g == "All" or meta[i]["group"] == g) and "LNS" in inst[i]]
        M[f"BenchLnsStuck{g}"] = str(sum(int(inst[i]["LNS"]["soc"]) == int(meta[i]["soc0"]) for i in ks))
    M["BenchHRangeFromFirst"] = f"${min(r):.1f}$--${max(r):.1f}$" if r else "--"   # text mode
    M["BenchHGainSentence"] = _h_gain_sentence(S)
    M["BenchCertSentence"] = _cert_sentence(S)
    M["BenchOverrunSentence"] = _overrun(S)
    M["BenchSkippedSentence"] = _skipped_sentence(S) if n_skip else ""
    return M



def _skipped_sentence(S):
    """The instances without an initial plan, counted by group (the list is in
    the results of the repository): on group P, PIBT does not reach all the
    goals; on the other groups, LaCAM3 finds no first solution within 30 s."""
    import csv
    from collections import Counter
    path = os.path.join(S["out"], "results.csv") if S.get("out") else None
    if path and os.path.exists(path):
        c = Counter(r["group"] for r in csv.DictReader(open(path)) if r["method"] == "none")
    else:                                   # groups from the names (P: suffix _pibt)
        c = Counter("P" if x.endswith("_pibt") else "other" for x in S["skipped"])
    n = sum(c.values())
    if not n:
        return ""
    def groups(gs):
        parts = [f"{c[g]} of group {g}" if g != "other" else f"{c[g]} of the other groups" for g in gs]
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    lacam = [g for g in "SDMHWLKC" if c.get(g)]
    if c.get("other"):
        lacam.append("other")
    pibt = [g for g in "P" if c.get(g)]
    why = []
    if lacam:
        why.append(f"{groups(lacam)}, on which LaCAM3 finds no first solution within $30$\\,s")
    if pibt:
        why.append(f"{groups(pibt)}, on which PIBT does not bring all the agents to their goals")
    return (f"The {n} instance{'s' if n > 1 else ''} without an initial plan "
            f"{'are' if n > 1 else 'is'} excluded: " + ", and ".join(why) + ".")

def _h_gain_sentence(S):
    inst, meta = S["inst"], S["meta"]
    a, b = "LaCAM*+Hybrid", "LaCAM*"
    fam = defaultdict(list)
    for i in inst:
        if meta[i]["group"] == "H" and a in inst[i] and b in inst[i]:
            fam[meta[i]["family"]].append((int(inst[i][a]["soc"]), int(inst[i][b]["soc"])))
    if not fam:
        return ""
    by_occ = defaultdict(lambda: [0, 0])
    for f, v in fam.items():
        o = re.search(r"_o(0\.\d+)$", f)
        if o:
            for x, y in v:
                by_occ[float(o.group(1))][0] += x < y
                by_occ[float(o.group(1))][1] += x > y
    gains = sorted(o for o, (w, l) in by_occ.items() if w > l)
    losses = sorted(o for o, (w, l) in by_occ.items() if l > w)
    means = {f: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v)) for f, v in fam.items()}
    top = sorted((f for f in means if means[f][0] < means[f][1]),
                 key=lambda f: means[f][0] / means[f][1])[:2]
    def pc(os_):
        v = [f"${round(100 * o)}\\%$" for o in os_]
        return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and " + v[-1]
    s = ""
    if gains:
        s = f"The gain is concentrated at {pc(gains)} occupancy"
        if top:
            s += " (for instance, " + " and ".join(
                f"${means[f][0]:.0f}$ against ${means[f][1]:.0f}$ on {_family_label(f)}" for f in top) + ")"
    if losses:
        s += (", and " if s else "") + f"turns into a loss at {pc(losses)}, where \\textsc{{LaCAM}}$^\\ast$ " \
             "with half of the budget starts from a worse plan"
    return (s[0].upper() + s[1:] + ".") if s else ""


def _cert_sentence(S):
    meth = ("Hybrid", "LNS+LS")
    tot = [S["method"][("All", m)]["cert"] for m in meth if ("All", m) in S["method"]
           and S["method"][("All", m)]["cert"] is not None]
    if not tot:
        return ""
    lo, hi = round(100 * min(tot)), round(100 * max(tot))
    parts = defaultdict(list)
    for g in S["groups"]:
        c = [S["method"][(g, m)]["cert"] for m in meth if (g, m) in S["method"]
             and S["method"][(g, m)]["cert"] is not None]
        if not c:
            continue
        v = sum(c) / len(c)
        key = "all" if v >= 0.999 else ("none" if v <= 0.001 else f"${round(100 * v)}\\%$")
        parts[key].append(g)
    phr = []
    if parts.get("all"):
        phr.append("all those of group" + ("s " if len(parts["all"]) > 1 else " ") + " and ".join(parts["all"]))
    for k, gs in parts.items():
        if k not in ("all", "none"):
            phr.append(f"{k} of group" + ("s " if len(gs) > 1 else " ") + " and ".join(gs))
    if parts.get("none"):
        phr.append("none of group" + ("s " if len(parts["none"]) > 1 else " ") + " and ".join(parts["none"]))
    rng = f"${lo}$--${hi}\\%$" if lo != hi else f"${lo}\\%$"
    s = (r"The plans returned by \textsc{Hybrid} and \textsc{LNS+LS} are certified in "
         f"{rng} of the instances: " + ", ".join(phr))
    if "H" in parts.get("none", []):
        ns = [int(S["meta"][i]["n"]) for i in S["inst"] if S["meta"][i]["group"] == "H"]
        s += f", where with ${min(ns)}$--${max(ns)}$ agents the final local search exceeds the time budget"
    return s + "."


def _overrun(S):
    """Sentence on the runs of group H that exceed the budget (methods ending with LS)."""
    T, inst, meta = S["T"], S["inst"], S["meta"]
    runs = [(float(inst[i][m]["time"]), int(meta[i]["n"])) for i in inst if meta[i]["group"] == "H"
            for m in ("LS", "Hybrid", "LNS+LS", "LaCAM*+Hybrid") if m in inst[i]]
    over = [(t / T - 1, n) for t, n in runs if t > 1.05 * T]
    if not runs:
        return "the methods respect the budget"
    if not over:
        return "the methods that end with the local search respect the budget within $5\\%$"
    ns = [n for _, n in over]
    return (f"the methods that end with the local search exceed it by more than $5\\%$ in "
            f"${len(over)}$ of their ${len(runs)}$ runs in group H, by up to "
            f"${round(100 * max(o for o, _ in over))}\\%$ (runs with ${min(ns)}$--${max(ns)}$ agents)")


def write_macros(M, path):
    lines = ["% Generated by mapf_bench.paper_update -- do not edit by hand.",
             f"% {datetime.datetime.now():%Y-%m-%d %H:%M}"]
    for k in sorted(M):
        assert re.fullmatch(r"[A-Za-z]+", k), k
        lines.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{M[k]}}}")
    open(path, "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def table_results(S):
    """Results by group. With more than five groups the table is split into two
    blocks stacked in the same table (the first groups, then the others with the
    gap and the time), so that no scaling (\resizebox) is needed."""
    gs = S["groups"]
    bold = {}
    for g in gs:
        full = [m for m in METHODS if (g, m) in S["method"] and S["method"][(g, m)]["n"] == int(macros_n(S, g))]
        if full:
            lo = min(S["method"][(g, m)]["ratio"] for m in full)
            bold[g] = {m for m in full if S["method"][(g, m)]["ratio"] <= lo + 1e-9}
    methods = [m for m in METHODS if any((g, m) in S["method"] for g in gs)]

    def block(groups, last):
        L = [r"\begin{tabular}{@{}l" + "c" * len(groups) + ("cc" if last else "") + "@{}}", r"\toprule",
             "Method & " + " & ".join(groups) + (r" & Gap & Time" if last else "") + r"\\", r"\midrule"]
        for m in methods:
            cells = []
            for g in groups:
                d = S["method"].get((g, m))
                if d is None:
                    cells.append("--"); continue
                r = f"{d['ratio']:.3f}"
                cells.append((rf"\textbf{{{r}}}" if m in bold.get(g, ()) else r) + f" ({d['best']})")
            row = f"{TEX_SHORT[m]} & " + " & ".join(cells)
            if last:
                a = S["method"].get(("All", m))
                row += f" & {_pct(a['gap'])} & {a['time']:.1f}" if a else " & & "
            L.append(row + r"\\")
        L += [r"\midrule", "Instances & " + " & ".join(macros_n(S, g) for g in groups)
              + (r" & & " if last else "") + r"\\", r"\bottomrule", r"\end{tabular}"]
        return L

    if len(gs) <= 5:
        return "\n".join(block(gs, True))
    k = (len(gs) + 1) // 2
    return "\n".join(block(gs[:k], False) + [r"", r"\medskip", r""] + block(gs[k:], True))


def macros_n(S, g):
    return str(sum(1 for i in S["inst"] if S["meta"][i]["group"] == g))


def table_pairs(S):
    L = [r"\begin{tabular}{@{}llrcrr@{}}", r"\toprule",
         r"Group & Comparison & Inst. & W/T/L & Median & $p$ (Holm)\\", r"\midrule"]
    for key in TABLE_PAIRS:
        d = S["pair"].get(key)
        if d is None:
            continue
        g, a, b = key
        L.append(f"{g.lower() if g == 'All' else g} & {TEX[a]} vs {TEX[b]} & {d['n']} & "
                 f"{d['w']}/{d['t']}/{d['l']} & ${_pct(d['med'], signed=True)}$ & {_pt(d['holm'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# claims made in the prose of the section
# ---------------------------------------------------------------------------
def claims(S):
    """The qualitative claims of the benchmark discussion, abstract and
    conclusion, checked against the results. Comparisons of methods are
    directional (more instances better than worse) where the text does not
    claim significance, and use the Holm-adjusted test where it does."""
    P, Mt, inst, meta = S["pair"], S["method"], S["inst"], S["meta"]
    sig = lambda k: k in P and P[k]["holm"] < 0.05
    more = lambda k: k in P and P[k]["w"] > P[k]["l"]           # first better more often
    not_more = lambda k: k in P and P[k]["w"] <= P[k]["l"]      # first not better more often
    fam = lambda f: [i for i in inst if meta[i]["family"] == f]
    out = []

    def add(where, text, ok):
        out.append((where, text, None if ok is None else bool(ok)))
    # coordinated moves (group S)
    if "S" in S["groups"]:
        add("Discussion, coordinated moves", "LS attains the best Sum-of-Costs on every instance of group S",
            Mt.get(("S", "LS"), {}).get("best") == int(macros_n(S, "S")))
        fr = fam("full_rings")
        add("Discussion, coordinated moves", "MAPF-LNS keeps the initial plan on the full rings",
            all(int(inst[i]["LNS"]["soc"]) == int(meta[i]["soc0"]) for i in fr if "LNS" in inst[i]) if fr else None)
        gr = fam("grid_rings")
        add("Discussion, coordinated moves", "LaCAM* finds the optimum (lower bound) on the rings on grids",
            all(int(inst[i]["LaCAM*"]["soc"]) == int(meta[i]["lb"]) for i in gr if "LaCAM*" in inst[i]) if gr else None)
        cv = fam("convoy")
        add("Discussion, coordinated moves", "all the methods tie on the convoys",
            all(len({r["soc"] for r in inst[i].values()}) == 1 for i in cv) if cv else None)
    # plans from prioritized planning (groups D, M)
    if ("M", "LS", "LNS") in P:
        d = P[("M", "LS", "LNS")]
        add("Discussion, PP plans", "LS alone is worse than LNS on the MovingAI maps (group M)",
            d["l"] > d["w"] and d["med"] >= 0)
    ser = [i for i in inst if meta[i]["family"] in ("random_serial", "oneway_serial")]
    if ser:
        g = [int(inst[i]["LS"]["soc"]) / int(inst[i]["LNS"]["soc"]) - 1 for i in ser
             if "LS" in inst[i] and "LNS" in inst[i]]
        tl = [float(inst[i]["LS"]["time"]) for i in ser if "LS" in inst[i]]
        add("Discussion, PP plans",
            "on serialized plans LS is within 2% of LNS on average, in less than a quarter of the budget",
            bool(g) and sum(g) / len(g) <= 0.02 and sum(tl) / len(tl) < 0.25 * S["T"])
    # deadlock-prone instances (group K)
    if "K" in S["groups"]:
        stuck = [i for i in inst if meta[i]["group"] == "K" and "LNS" in inst[i]
                 and int(inst[i]["LNS"]["soc"]) == int(meta[i]["soc0"])]
        rev = [i for i in inst if meta[i]["group"] == "K" and meta[i]["family"].startswith("reversal")]
        add("Discussion, deadlocks", "in group K, LNS does not improve the initial plan on every order reversal",
            bool(rev) and all(i in stuck for i in rev))
        add("Discussion, deadlocks", "in group K, LNS+LS is better than LNS on more instances than it is worse",
            more(("K", "LNS+LS", "LNS")) if ("K", "LNS+LS", "LNS") in P else None)
    # dense instances (groups H, W, L)
    if "H" in S["groups"]:
        r = {m: Mt[("H", m)]["ratio"] for m in METHODS if ("H", m) in Mt}
        add("Discussion, dense", "in group H the methods that start from the first LaCAM solution are far from LaCAM*",
            all(r[m] > r["LaCAM*"] for m in ("LNS", "LS", "Hybrid", "LNS+LS") if m in r) if "LaCAM*" in r else None)
        add("Discussion, dense", "a method that starts from LaCAM* has the best SoC/LB in group H",
            min(r, key=r.get).startswith("LaCAM*") if r else None)
    # after LaCAM*: MAPF-LNS alone better on group L, the local search phases on group C
    if ("L", "LaCAM*+Hybrid", "LaCAM*+LNS") in P:
        d = P[("L", "LaCAM*+Hybrid", "LaCAM*+LNS")]
        add("Discussion, dense, and Conclusion", "in group L, LaCAM*+LNS is better than LaCAM*+Hybrid more often", d["l"] > d["w"])
    if ("C", "LaCAM*+Hybrid", "LaCAM*+LNS") in P:
        add("Discussion, dense, and Conclusion", "in group C, LaCAM*+Hybrid is better than LaCAM*+LNS more often",
            more(("C", "LaCAM*+Hybrid", "LaCAM*+LNS")))
    if ("C", "LaCAM*+LNS+LSB", "LaCAM*+LNS") in P:
        add("Discussion, dense", "in group C, LaCAM*+LNS+LSB is better than LaCAM*+LNS more often",
            more(("C", "LaCAM*+LNS+LSB", "LaCAM*+LNS")))
    if ("C", "LaCAM*+LNS", "LaCAM*") in P:
        d = P[("C", "LaCAM*+LNS", "LaCAM*")]
        add("Discussion, dense", "in group C, LaCAM* alone is better than LaCAM*+LNS more often", d["l"] > d["w"])
    # overall, abstract and conclusion
    # LNS+LS vs LNS: by group (better where coordinated moves or high occupancy
    # matter, worse on the maps with reduced connectivity)
    for g in ("S", "H", "K", "C"):
        if (g, "LNS+LS", "LNS") in P:
            add("Discussion overall, abstract, contributions, conclusion",
                f"in group {g}, LNS+LS is better than LNS on more instances than it is worse",
                more((g, "LNS+LS", "LNS")))
    if ("L", "LNS+LS", "LNS") in P:
        add("Discussion overall, abstract, contributions, conclusion",
            "in group L (reduced connectivity), LNS+LS is worse than LNS on more instances than it is better",
            P[("L", "LNS+LS", "LNS")]["l"] > P[("L", "LNS+LS", "LNS")]["w"])
    for g in ("D", "M"):
        if (g, "LNS+LS", "LNS") in P:
            add("Discussion overall", f"in group {g}, LNS+LS and LNS are not significantly different",
                not sig((g, "LNS+LS", "LNS")))
    add("Discussion overall, abstract, contributions, conclusion",
        "the alternation of the hybrid brings no benefit: Hybrid is not significantly better than LNS+LS",
        (not (sig(("All", "Hybrid", "LNS+LS")) and more(("All", "Hybrid", "LNS+LS"))))
        if ("All", "Hybrid", "LNS+LS") in P else None)
    if ("All", "LaCAM*+Hybrid", "LaCAM*") in P:
        d = P[("All", "LaCAM*+Hybrid", "LaCAM*")]
        add("Discussion overall", "LaCAM*+Hybrid is significantly better than LaCAM* alone",
            sig(("All", "LaCAM*+Hybrid", "LaCAM*")) and d["w"] > d["l"])
    # LaCAM* followed by MAPF-LNS and LSB
    if ("L", "LaCAM*+LNS+LSB", "LaCAM*+LNS") in P:
        d = P[("L", "LaCAM*+LNS+LSB", "LaCAM*+LNS")]
        add("Discussion, dense", "on the maps with reduced connectivity (group L) LaCAM*+LNS is better than LaCAM*+LNS+LSB",
            d["l"] > d["w"])
    # LSB with the neighborhoods of LNS and LaCAM
    if ("All", "LSBX", "LSB") in P:
        add("Discussion, large moves", "LSBX is better than LSB on more instances than it is worse",
            more(("All", "LSBX", "LSB")))
    if ("All", "LSBX", "LS") in P:
        add("Discussion, large moves", "LSBX is better than LS on more instances than it is worse",
            more(("All", "LSBX", "LS")))
    # beam escapes (LSB)
    if ("All", "LSB", "LS") in P:
        add("Discussion, large moves", "LSB is better than LS on more instances than it is worse",
            more(("All", "LSB", "LS")))
    if ("M", "LSB", "LS") in P:
        d = P[("M", "LSB", "LS")]
        add("Discussion, large moves", "on the MovingAI maps (group M) LSB coincides with LS",
            d["w"] == 0 and d["l"] == 0)
    if ("All", "LS") in Mt:
        add("Discussion, certificates", "LS alone stops before the budget on average", Mt[("All", "LS")]["time"] < S["T"])
    return out


# ---------------------------------------------------------------------------
# paper
# ---------------------------------------------------------------------------
# the markers may be indented and followed by spaces (editors that re-indent the
# file); the indentation of the closing marker is kept
AUTO = re.compile(r"(%[ \t]*<AUTO:(?P<name>[A-Za-z_]+)>[ \t\r]*\n)(?P<body>.*?)(\n[ \t]*%[ \t]*</AUTO:(?P=name)>)", re.S)


def update_paper(paper, S, compile_=False):
    d = os.path.dirname(os.path.abspath(paper))
    work = os.path.join(d, "paper_update")
    os.makedirs(work, exist_ok=True)
    if not os.path.isfile(paper):
        raise SystemExit(f"paper not found: {paper} (relative to {os.getcwd()}); "
                         "the results are not affected: run the update again with the right path")
    old = open(paper).read()
    names = {m.group("name") for m in AUTO.finditer(old)}
    missing = {"bench_results", "bench_pairs"} - names
    if missing:
        lines = old.split("\n")
        diag = []
        for nm in sorted(missing):
            op = [i + 1 for i, l in enumerate(lines) if re.search(r"<AUTO:" + nm + r">", l)]
            cl = [i + 1 for i, l in enumerate(lines) if re.search(r"</AUTO:" + nm + r">", l)]
            diag.append(f"  {nm}: opening marker on line(s) {op or 'none'}, closing marker on line(s) {cl or 'none'}"
                        + (f"; opening line: {lines[op[0] - 1]!r}" if op else ""))
        raise SystemExit(f"the paper has no %<AUTO:...> block for {', '.join(sorted(missing))}; "
                         "see the README (Automatic paper update)\n"
                         f"  paper read: {os.path.abspath(paper)}\n"
                         f"  paper_update.py running: {os.path.abspath(__file__)}\n" + "\n".join(diag))
    blocks = {"bench_results": table_results(S), "bench_pairs": table_pairs(S)}
    new = AUTO.sub(lambda m: m.group(1) + blocks.get(m.group("name"), m.group("body")) + m.group(4), old)
    if r"\input{bench_numbers}" not in new:
        new = new.replace(r"\begin{document}", "\\input{bench_numbers}\n\\begin{document}", 1)
    used = set(re.findall(r"\\(Bench[A-Za-z]+)", new))
    M = macros(S)
    undefined = sorted(u for u in used if u not in M)
    for u in undefined:            # a number the new results do not provide
        M[u] = r"\textbf{??}"
    write_macros(M, os.path.join(d, "bench_numbers.tex"))
    stamp = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy(paper, os.path.join(work, os.path.basename(paper) + f".{stamp}.bak"))
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile="before", tofile="after"))
    open(os.path.join(work, f"diff-{stamp}.patch"), "w").write(diff)
    open(paper, "w").write(new)
    fig = os.path.join(S["out"], "performance_profiles.pdf")
    try:                                   # redraw the profiles from the results
        from .paper_suite import _figures
        _figures(S["inst"], S["meta"], S["groups"], S["out"])
    except Exception as e:                 # keep the figure of the analysis
        print(f"performance profiles not redrawn: {e}")
    if os.path.exists(fig):
        os.makedirs(os.path.join(d, "images", "tests"), exist_ok=True)
        shutil.copy(fig, os.path.join(d, "images", "tests", "performance_profiles.pdf"))
    C = claims(S)
    rep = [f"# Paper update {stamp}", "", f"Results: `{S['out']}` ({M['BenchNSolved']} solved instances, "
           f"T = {S['T']:g} s). Paper: `{paper}`. Macros: `bench_numbers.tex` ({len(M)}). "
           f"Diff: `paper_update/diff-{stamp}.patch`.", "", "## Claims of the prose", "",
           "| status | where | claim |", "|---|---|---|"]
    for where, text, ok in C:
        rep.append(f"| {'confirmed' if ok else ('n/a' if ok is None else '**CONTRADICTED**')} | {where} | {text} |")
    if undefined:
        rep += ["", "## Macros without a value (shown as ?? in the paper)", ""] + [f"- `\\{u}`" for u in undefined]
    compiled = None
    if compile_:
        compiled = _compile(paper)
        rep += ["", f"## Compilation: {'ok' if compiled else 'FAILED (see the .log next to the paper)'}"]
    open(os.path.join(work, "report.md"), "w").write("\n".join(rep) + "\n")
    bad = [c for c in C if c[2] is False]
    return {"report": os.path.join(work, "report.md"), "contradicted": bad, "undefined": undefined,
            "changed_lines": sum(1 for l in diff.splitlines() if l.startswith(("+", "-"))
                                 and not l.startswith(("+++", "---"))), "compiled": compiled}


def _compile(paper):
    d, name = os.path.split(os.path.abspath(paper))
    stem = os.path.splitext(name)[0]
    if shutil.which("latexmk"):
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", name]
    elif shutil.which("pdflatex"):
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name]
    else:
        print("no LaTeX found: compilation skipped")
        return None
    ok = True
    for _ in range(1 if cmd[0] == "latexmk" else 2):
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
        ok &= r.returncode == 0
    return ok and os.path.exists(os.path.join(d, stem + ".pdf"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results folder of a run (results.csv, config.json, machine.json)")
    ap.add_argument("--paper", required=True, help="the .tex file of the paper")
    ap.add_argument("--compile", action="store_true", help="compile the paper after the update")
    ap.add_argument("--strict", action="store_true", help="exit with an error if a claim is contradicted")
    a = ap.parse_args(argv)
    run(a.results, a.paper, a.compile, a.strict)


def run(results, paper, compile_=False, strict=False):
    from .paper_suite import analyze
    cfgp = os.path.join(results, "config.json")
    T = json.load(open(cfgp))["time"] if os.path.exists(cfgp) else None
    analyze(results, T)                          # refresh results.md, tables.tex, figure
    S = statistics(results)
    S["out"] = results
    r = update_paper(paper, S, compile_)
    print(f"paper updated ({r['changed_lines']} lines changed); report: {r['report']}")
    for where, text, _ in r["contradicted"]:
        print(f"  CONTRADICTED ({where}): {text}")
    if r["undefined"]:
        print(f"  {len(r['undefined'])} macros without a value: {', '.join(r['undefined'])}")
    if r["compiled"] is False:
        print("  compilation FAILED")
    if strict and (r["contradicted"] or r["undefined"] or r["compiled"] is False):
        sys.exit(2)
    return r


if __name__ == "__main__":
    main()
