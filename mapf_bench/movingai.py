# SPDX-License-Identifier: GPL-3.0-or-later
"""MovingAI MAPF benchmark: maps, scenarios and instances.

Maps (`.map`) and scenarios (`.scen`) in the MovingAI format are read from
benchmarks/movingai. A scenario line is
    bucket  map  width  height  start_x  start_y  goal_x  goal_y  optimal_length
with x = column and y = row; an instance with n agents uses the first n
lines, as usual for this benchmark. `generate_scen` writes scenarios in the
same format (distinct random starts and goals in the largest connected
component, as the official "random" scenarios) for the maps whose official
scenarios are not included.
"""
import json
import os
import random
from collections import deque

from .instances import Instance, grid_from_cells

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "movingai")
PASSABLE = set(".GS")          # '@', 'O', 'T', 'W' are obstacles


def map_path(name):
    return os.path.join(ROOT, "maps", name + ".map")


def load_map(name_or_path):
    """Returns (height, width, set of free cells (row, col))."""
    path = name_or_path if os.path.exists(name_or_path) else map_path(name_or_path)
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh]
    h = int(lines[1].split()[1])
    w = int(lines[2].split()[1])
    assert lines[3].strip() == "map"
    free = {(r, c) for r in range(h) for c in range(w) if lines[4 + r][c] in PASSABLE}
    return h, w, free


def load_scen(path):
    """Returns the list of (start (row, col), goal (row, col)) of a scenario."""
    pairs = []
    with open(path) as fh:
        for line in fh:
            f = line.split("\t")
            if len(f) < 9:
                continue
            sx, sy, gx, gy = map(int, f[4:8])
            pairs.append(((sy, sx), (gy, gx)))
    return pairs


def _components(free):
    seen, comps = set(), []
    for s in sorted(free):
        if s in seen:
            continue
        comp, q = {s}, deque([s])
        while q:
            r, c = q.popleft()
            for y in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if y in free and y not in comp:
                    comp.add(y); q.append(y)
        seen |= comp
        comps.append(comp)
    return comps


def _bfs(free, s):
    d, q = {s: 0}, deque([s])
    while q:
        r, c = q.popleft()
        for y in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if y in free and y not in d:
                d[y] = d[(r, c)] + 1; q.append(y)
    return d


def scen_path(map_name, kind, index):
    return os.path.join(ROOT, "scens", f"{map_name}-{kind}-{index}.scen")


def generate_scen(map_name, seed, count=100, path=None):
    """Write a scenario with `count` agents: distinct random starts and goals
    in the largest connected component (4-connected), with the length of a
    shortest 4-connected path as last column."""
    h, w, free = load_map(map_name)
    comp = sorted(max(_components(free), key=len))
    rng = random.Random(seed)
    count = min(count, len(comp))
    starts = rng.sample(comp, count)
    goals = rng.sample(comp, count)
    path = path or scen_path(map_name, "gen", seed)
    with open(path, "w") as fh:
        fh.write("version 1\n")
        for i, (s, g) in enumerate(zip(starts, goals)):
            d = _bfs(free, s)[g]
            fh.write(f"{i // 10}\t{map_name}.map\t{w}\t{h}\t{s[1]}\t{s[0]}\t{g[1]}\t{g[0]}\t{d}\n")
    return path


def scenarios(map_name, k=2):
    """The first k scenarios of a map: the official 'random' ones when present,
    otherwise generated ones (created on first use, seeds 1..k)."""
    out = []
    for i in range(1, k + 1):
        p = scen_path(map_name, "random", i)
        if not os.path.exists(p):
            p = scen_path(map_name, "gen", i)
            if not os.path.exists(p):
                generate_scen(map_name, i, path=p)
        out.append(p)
    return out


def instance(map_name, scen, n, initial="official"):
    """Instance with the first n agents of a scenario; the initial plan comes
    from the prioritized planning of the official MAPF-LNS (fallback: ours)."""
    h, w, free = load_map(map_name)
    comp = max(_components(free), key=len)
    G = grid_from_cells(comp, map_name)
    idx = {rc: v for v, rc in G.coords.items()}
    pairs = load_scen(scen)[:n]
    assert len(pairs) == n, f"{scen} has fewer than {n} agents"
    starts = tuple(idx[s] for s, _ in pairs)
    goals = tuple(idx[g] for _, g in pairs)
    name = f"{map_name}_{os.path.basename(scen).rsplit('-', 2)[-2]}{os.path.basename(scen).rsplit('-', 1)[-1][:-5]}_n{n}"
    inst = Instance(name, f"movingai_{map_name}_n{n}", G, starts, goals,
                    {"map": map_name, "scen": os.path.basename(scen), "n": n})
    if initial:
        plan = None
        if initial == "official":
            from .official_lns import available, official_initial_plan
            if available():
                plan = official_initial_plan(inst, "PP", seed=0)
        if plan is None:
            from .pp import initial_pp
            from .plan import paths_to_plan
            paths = initial_pp(G, starts, goals, seed=0, attempts=60)
            plan = paths_to_plan(paths) if paths else None
        if plan is not None:
            from .plan import trim
            inst.initial_plan = trim(plan)
    return inst


SMALL_MAPS = ["empty-16-16", "empty-32-32", "random-32-32-10", "random-32-32-20",
              "maze-32-32-2", "maze-32-32-4", "room-32-32-4"]
LARGE_MAPS = ["warehouse-10-20-10-2-1", "den312d"]


def families(maps=SMALL_MAPS, agents=(10, 20, 30), k=2):
    """compare_vnd-style families: name -> (seed -> instance), seed = scenario index."""
    fams = {}
    for m in maps:
        scens = scenarios(m, k)
        for n in agents:
            fams[f"mai_{m}_n{n}"] = (lambda s, m=m, n=n, scens=scens: instance(m, scens[s], n))
    return fams


def report(out="movingai_results"):
    """Summary of a compare_vnd run on the MovingAI families (lnsT, hyb2, hyb4)."""
    import csv
    import json
    from collections import defaultdict
    from math import comb

    def sign(w, l):
        m = w + l
        return 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(min(w, l) + 1)) / 2 ** m)

    rows = list(csv.DictReader(open(os.path.join(out, "vnd.csv"))))
    by = defaultdict(dict)
    for r in rows:
        by[(r["family"], r["instance"])][r["strategy"]] = r
    cert = defaultdict(dict)
    sp = os.path.join(out, "hybrid_stats.jsonl")
    if os.path.exists(sp):
        for line in open(sp):
            d = json.loads(line)
            cert[d.get("variant", "hyb")][(d["family"], d["instance"])] = d["certified"]
    S = [s for s in ("lnsT", "hyb2", "hyb4") if any(s in d for d in by.values())]
    lines = ["# MovingAI MAPF benchmark: hybrid local search vs MAPF-LNS", "",
             "Sum-of-Costs (mean over scenarios); initial plans from the prioritized planning of "
             "the official MAPF-LNS; every method has 20 s. `random-1` scenarios are official, "
             "`gen-k` scenarios are generated in the same format (see benchmarks/movingai/README.md). "
             "W/T/L: lower / equal / higher Sum-of-Costs than MAPF-LNS.", "",
             "| map | agents | scen. | initial | MAPF-LNS | hyb2 | hyb4 | hyb2 vs LNS | hyb4 vs LNS | "
             "time LNS / hyb2 / hyb4 (s) | certified hyb2 / hyb4 |",
             "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|:---:|"]
    tot = defaultdict(lambda: [0, 0, 0])
    fams = sorted({k[0] for k in by}, key=lambda f: (f.rsplit("_n", 1)[0], int(f.rsplit("_n", 1)[1])))
    for f in fams:
        ks = [k for k in by if k[0] == f and all(s in by[k] for s in S)]
        if not ks:
            continue
        m = lambda s, c="soc": sum(float(by[k][s][c]) for k in ks) / len(ks)
        res = {}
        for s in ("hyb2", "hyb4"):
            w = t = l = 0
            for k in ks:
                a, b = int(by[k][s]["soc"]), int(by[k]["lnsT"]["soc"])
                w, t, l = w + (a < b), t + (a == b), l + (a > b)
            res[s] = (w, t, l)
            for j, v in enumerate(res[s]):
                tot[s][j] += v
        name, n = f[4:].rsplit("_n", 1)
        c2 = sum(cert["hyb2"].get(k, False) for k in ks)
        c4 = sum(cert["hyb4"].get(k, False) for k in ks)
        lines.append(
            f"| {name} | {n} | {len(ks)} | {sum(int(by[k]['lnsT']['soc0']) for k in ks)/len(ks):.1f} | "
            f"{m('lnsT'):.1f} | {m('hyb2'):.1f} | {m('hyb4'):.1f} | {'/'.join(map(str, res['hyb2']))} | "
            f"{'/'.join(map(str, res['hyb4']))} | {m('lnsT','time'):.1f} / {m('hyb2','time'):.1f} / "
            f"{m('hyb4','time'):.1f} | {c2} / {c4} of {len(ks)} |")
    lines += ["", f"- hyb2 vs MAPF-LNS: {'/'.join(map(str, tot['hyb2']))} "
                  f"(sign test p = {sign(tot['hyb2'][0], tot['hyb2'][2]):.3f})",
              f"- hyb4 vs MAPF-LNS: {'/'.join(map(str, tot['hyb4']))} "
              f"(sign test p = {sign(tot['hyb4'][0], tot['hyb4'][2]):.3f})"]
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "movingai_report.md"), "w").write(text)
    return text


# ---------------------------------------------------------------------------
# High occupancy (up to 98% of the free cells)
# ---------------------------------------------------------------------------
DENSE_MAPS = ["empty-8-8", "random-8-8-20", "empty-16-16"]
OCCUPANCY = (0.5, 0.8, 0.9, 0.95, 0.98)


def dense_scenarios(map_name, k=2):
    """Generated scenarios with one entry per cell of the largest component."""
    h, w, free = load_map(map_name)
    size = len(max(_components(free), key=len))
    out = []
    for i in range(1, k + 1):
        p = scen_path(map_name, "full", i)
        if not os.path.exists(p):
            generate_scen(map_name, i, count=size, path=p)
        out.append(p)
    return out


def dense_instance(map_name, scen, occupancy):
    """Instance with round(occupancy * |component|) agents; initial plan: the
    first solution of LaCAM3."""
    h, w, free = load_map(map_name)
    size = len(max(_components(free), key=len))
    n = max(1, min(size - 1, round(occupancy * size)))
    inst = instance(map_name, scen, n, initial=None)
    inst.name = f"{map_name}_o{occupancy:.2f}_{os.path.basename(scen)[:-5].rsplit('-', 1)[-1]}"
    inst.family = f"dmai_{map_name}_o{occupancy:.2f}"
    from .lacam import lacam_solve
    inst.initial_plan = lacam_solve(inst, time_limit=30, anytime=False)
    return inst


def dense_families(maps=DENSE_MAPS, occupancy=OCCUPANCY, k=2):
    fams = {}
    for m in maps:
        scens = dense_scenarios(m, k)
        for o in occupancy:
            fams[f"dmai_{m}_o{o:.2f}"] = (lambda s, m=m, o=o, scens=scens: dense_instance(m, scens[s], o))
    return fams


def dense_report(out="movingai_dense"):
    """Summary of a compare_vnd run on the high-occupancy families."""
    import csv
    import json
    from collections import defaultdict
    from math import comb

    def sign(w, l):
        m = w + l
        return 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(min(w, l) + 1)) / 2 ** m)

    rows = list(csv.DictReader(open(os.path.join(out, "vnd.csv"))))
    by = defaultdict(dict)
    for r in rows:
        by[(r["family"], r["instance"])][r["strategy"]] = r
    cert = defaultdict(dict)
    sp = os.path.join(out, "hybrid_stats.jsonl")
    if os.path.exists(sp):
        for line in open(sp):
            d = json.loads(line)
            cert[d.get("variant", "hyb")][(d["family"], d["instance"])] = d["certified"]
    S = ["lnsT", "lacam", "hyb2", "hyb4"]
    lines = ["# MovingAI maps at high occupancy: hybrid local search vs MAPF-LNS and LaCAM*", "",
             "Occupancy = agents / free cells of the largest component. Initial plan: first "
             "solution of LaCAM3 (given to MAPF-LNS and to the hybrids); LaCAM* runs from scratch "
             "with its anytime refinement. 20 s per method, single thread. Sum-of-Costs, mean over "
             "the solvable scenarios. W/T/L: lower / equal / higher Sum-of-Costs.", "",
             "| map | occupancy | agents | scen. | initial | MAPF-LNS | LaCAM* | hyb2 | hyb4 | "
             "hyb2 vs LNS | hyb2 vs LaCAM* | hyb4 vs LNS | certified hyb2 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|"]
    tot = defaultdict(lambda: [0, 0, 0])
    fams = sorted({k[0] for k in by}, key=lambda f: (f.rsplit("_o", 1)[0], float(f.rsplit("_o", 1)[1])))
    for f in fams:
        ks = [k for k in by if k[0] == f and all(s in by[k] for s in S)]
        if not ks:
            continue
        m = lambda s: sum(int(by[k][s]["soc"]) for k in ks) / len(ks)

        def wtl(a, b):
            w = t = l = 0
            for k in ks:
                x, y = int(by[k][a]["soc"]), int(by[k][b]["soc"])
                w, t, l = w + (x < y), t + (x == y), l + (x > y)
            for j, v in enumerate((w, t, l)):
                tot[(a, b)][j] += v
            return f"{w}/{t}/{l}"
        name, occ = f[5:].rsplit("_o", 1)
        lines.append(
            f"| {name} | {float(occ):.0%} | {by[ks[0]]['lnsT']['n']} | {len(ks)} | "
            f"{sum(int(by[k]['lnsT']['soc0']) for k in ks)/len(ks):.0f} | {m('lnsT'):.0f} | "
            f"{m('lacam'):.0f} | {m('hyb2'):.0f} | {m('hyb4'):.0f} | {wtl('hyb2', 'lnsT')} | "
            f"{wtl('hyb2', 'lacam')} | {wtl('hyb4', 'lnsT')} | "
            f"{sum(cert['hyb2'].get(k, False) for k in ks)}/{len(ks)} |")
    lines.append("")
    for a, b in (("hyb2", "lnsT"), ("hyb2", "lacam"), ("hyb4", "lnsT"), ("lacam", "lnsT")):
        if (a, b) == ("lacam", "lnsT"):
            for f in fams:
                pass
        w, t, l = tot[(a, b)]
        if w + t + l:
            lines.append(f"- {a} vs {b}: {w}/{t}/{l} (sign test p = {sign(w, l):.4f})")
    sk = os.path.join(out, "skipped.txt")
    if os.path.exists(sk):
        lines.append(f"- skipped (no solution found by LaCAM3 in 30 s): "
                     + ", ".join(sorted({l.split()[1] for l in open(sk) if l.strip()})))
    text = "\n".join(lines) + "\n"
    open(os.path.join(out, "dense_report.md"), "w").write(text)
    return text


# ---------------------------------------------------------------------------
# Reduced connectivity: block free cells, keep the agents of the instance
# ---------------------------------------------------------------------------
REDUCED_DIR = os.path.join(ROOT, "reduced")


def _connected(cells, src):
    seen, q = {src}, deque([src])
    while q:
        r, c = q.popleft()
        for y in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if y in cells and y not in seen:
                seen.add(y); q.append(y)
    return len(seen) == len(cells)


def removal_sequence(map_name, scen, n):
    """Cells of the largest component that can be blocked one after the other
    with the free cells staying connected; the starts and goals of the first n
    agents of the scenario are never blocked. The order of the candidates is
    random with a seed derived from (map, scenario, n); every prefix of the
    sequence leaves a connected map, and the whole sequence a minimal one (no
    further cell can be blocked).
    Cached in benchmarks/movingai/reduced/."""
    import zlib
    tag = f"{map_name}-{os.path.basename(scen)[:-5]}-n{n}"
    path = os.path.join(REDUCED_DIR, tag + ".json")
    if os.path.exists(path):
        return [tuple(x) for x in json.load(open(path))["sequence"]]
    h, w, free = load_map(map_name)
    cells = set(max(_components(free), key=len))
    pairs = load_scen(scen)[:n]
    terminals = {s for s, _ in pairs} | {g for _, g in pairs}
    assert terminals <= cells, "the scenario uses cells outside the largest component"
    cand = sorted(cells - terminals)
    random.Random(zlib.crc32(tag.encode())).shuffle(cand)
    src = next(iter(terminals))
    seq = []
    # repeated passes in the same order: a cell that leads to a dead-end branch
    # becomes blockable once the branch has been blocked
    changed = True
    while changed:
        changed = False
        rest = []
        for c in cand:
            cells.discard(c)
            if _connected(cells, src):
                seq.append(c)
                changed = True
            else:
                cells.add(c)
                rest.append(c)
        cand = rest
    os.makedirs(REDUCED_DIR, exist_ok=True)
    json.dump({"map": map_name, "scen": os.path.basename(scen), "n": n,
               "sequence": [list(c) for c in seq]}, open(path, "w"))
    return seq


def write_map(path, h, w, cells):
    """Write a map in the MovingAI format ('.' free, '@' blocked)."""
    with open(path, "w") as fh:
        fh.write(f"type octile\nheight {h}\nwidth {w}\nmap\n")
        for r in range(h):
            fh.write("".join("." if (r, c) in cells else "@" for c in range(w)) + "\n")


def reduced_instance(map_name, scen, n, level, initial="lacam"):
    """The instance with the first n agents of the scenario on the map with the
    first round(level * len(sequence)) cells of the removal sequence blocked
    (level 0: the original map, level 1: a minimal connected map). The
    reduced map is also written in MovingAI format. Initial plan: the first
    LaCAM3 solution (None if none is found in 30 s)."""
    h, w, free = load_map(map_name)
    cells = set(max(_components(free), key=len))
    seq = removal_sequence(map_name, scen, n)
    k = round(level * len(seq))
    cells -= set(seq[:k])
    G = grid_from_cells(cells, map_name)
    idx = {rc: v for v, rc in G.coords.items()}
    pairs = load_scen(scen)[:n]
    stag = os.path.basename(scen)[:-5].rsplit("-", 2)
    name = f"{map_name}_{stag[-2]}{stag[-1]}_n{n}_r{level:.2f}"
    inst = Instance(name, f"rmai_{map_name}_n{n}_r{level:.2f}", G, tuple(idx[s] for s, _ in pairs),
                    tuple(idx[g] for _, g in pairs),
                    {"map": map_name, "scen": os.path.basename(scen), "n": n, "level": level,
                     "blocked": k, "cells": len(cells)})
    os.makedirs(REDUCED_DIR, exist_ok=True)
    write_map(os.path.join(REDUCED_DIR, name + ".map"), h, w, cells)
    inst.initial_plan = None
    if initial == "lacam":
        from .lacam import lacam_solve
        p = lacam_solve(inst, time_limit=30, anytime=False)
        if p is not None:
            from .plan import trim
            inst.initial_plan = trim(p)
    return inst
