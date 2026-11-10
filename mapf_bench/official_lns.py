# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapter for the official MAPF-LNS binary (Li et al., IJCAI 2021).

The official code supports 4-connected undirected grid maps in the MovingAI
format. The adapter writes the instance as .map/.scen files, passes the
initial plan with the patched option --initAlgo File --initPaths, runs the
binary, reads the final paths written with --outputPaths, and validates them.

The binary is looked up in $MAPF_LNS_BIN, then in third_party/MAPF-LNS/lns;
build it with scripts/build_official_lns.sh.
"""
import os
import shutil
import subprocess
import tempfile
import time

from .plan import plan_to_paths, paths_to_plan, validate

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BIN = os.path.join(_HERE, "..", "third_party", "MAPF-LNS", "lns")


def binary():
    b = os.environ.get("MAPF_LNS_BIN", _DEFAULT_BIN)
    return b if os.path.isfile(b) and os.access(b, os.X_OK) else None


def available():
    return binary() is not None


def use_official(inst):
    """Which MAPF-LNS to run on an instance: True for the official code, False
    for the re-implementation (mapf_bench/lns.py).

    The official code supports 4-connected grids only, so directed graphs
    always use the re-implementation. On grids the official code is used when
    its binary is available; otherwise the re-implementation is used only if
    MAPF_BENCH_LNS_FALLBACK=1, so that results never switch implementation
    silently, and an error is raised.
    """
    if not is_grid_instance(inst):
        return False
    if available():
        return True
    if os.environ.get("MAPF_BENCH_LNS_FALLBACK") == "1":
        return False
    raise RuntimeError("official MAPF-LNS binary not found: run scripts/build_official_lns.sh "
                       "or set MAPF_LNS_BIN; or set MAPF_BENCH_LNS_FALLBACK=1 to use the "
                       "re-implementation on grids")


def is_grid_instance(inst):
    """True if the graph is exactly the 4-connected undirected grid on its
    free cells (the only graphs the official code supports)."""
    G = inst.graph
    if not G.coords:
        return False
    pos = {v: rc for v, rc in G.coords.items()}
    cell = {rc: v for v, rc in pos.items()}
    for v, (r, c) in pos.items():
        expect = {cell[x] for x in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
                  if x in cell}
        if set(G.out[v]) - {v} != expect:
            return False
    return True


def _write_files(inst, plan, d):
    G = inst.graph
    rows = max(r for r, _ in G.coords.values()) + 1
    cols = max(c for _, c in G.coords.values()) + 1
    free = set(G.coords.values())
    mp = os.path.join(d, "inst.map")
    with open(mp, "w") as fh:
        fh.write(f"type octile\nheight {rows}\nwidth {cols}\nmap\n")
        for r in range(rows):
            fh.write("".join("." if (r, c) in free else "@" for c in range(cols)) + "\n")
    sc = os.path.join(d, "inst.scen")
    with open(sc, "w") as fh:
        fh.write("version 1\n")
        for s, g in zip(inst.starts, inst.goals):
            (sr, scl), (gr, gc) = G.coords[s], G.coords[g]
            fh.write(f"0\tinst.map\t{cols}\t{rows}\t{scl}\t{sr}\t{gc}\t{gr}\t0\n")
    loc = lambda v: G.coords[v][0] * cols + G.coords[v][1]
    ip = os.path.join(d, "init_paths.txt")
    with open(ip, "w") as fh:
        paths = plan_to_paths(plan, inst.goals)
        fh.write(f"{len(paths)}\n")
        for p in paths:
            fh.write(",".join(str(loc(v)) for v in p) + ",\n")
    back = {G.coords[v][0] * cols + G.coords[v][1]: v for v in G.coords}
    return mp, sc, ip, back


def official_lns(inst, plan, time_limit=10.0, N=5, replan="PP", seed=0,
                 max_iterations=1000000, keep_dir=False, stagnation=0):
    """Run the official MAPF-LNS from `plan`; returns (plan, stats)."""
    b = binary()
    if b is None:
        raise RuntimeError("official MAPF-LNS binary not found; run "
                           "scripts/build_official_lns.sh or set MAPF_LNS_BIN")
    if not is_grid_instance(inst):
        raise ValueError("the official MAPF-LNS supports 4-connected grids only")
    d = tempfile.mkdtemp(prefix="olns_")
    try:
        mp, sc, ip, back = _write_files(inst, plan, d)
        outp = os.path.join(d, "paths.txt")
        cmd = [b, "-m", mp, "-a", sc, "-k", str(inst.n), "-t", str(time_limit),
               "--initAlgo", "File", "--initPaths", ip, "--outputPaths", outp,
               "--replanAlgo", replan, "--neighborSize", str(N),
               "--maxIterations", str(max_iterations), "--seed", str(seed),
               "--stagnation", str(stagnation),
               "-o", os.path.join(d, "out")]
        t0 = time.perf_counter()
        # The official binary occasionally crashes without any output (seen on
        # macOS on small instances): retry up to twice with the same input, and
        # report the exit code if it still fails (negative: killed by a signal).
        for attempt in range(3):
            if os.path.exists(outp):
                os.remove(outp)
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=time_limit + 60)
            if res.returncode == 0 and os.path.exists(outp):
                break
        dt = time.perf_counter() - t0
        if res.returncode != 0 or not os.path.exists(outp):
            raise RuntimeError(f"official LNS failed (exit code {res.returncode}, 3 attempts): "
                               f"{res.stdout[-500:]} {res.stderr[-500:]}")
        with open(outp) as fh:
            lines = fh.read().split("\n")
        n = int(lines[0])
        paths = []
        for k in range(n):
            locs = [int(x) for x in lines[1 + k].split(",") if x.strip()]
            paths.append([back[x] for x in locs])
        new = paths_to_plan(paths)
        validate(inst.graph, inst.starts, inst.goals, new)
        stats = {"time": dt, "stdout": res.stdout.strip().splitlines()[-1][:300]}
        last = stats["stdout"]
        for key in ("Iterations", "failed iterations"):
            if key + " = " in last:
                try:
                    stats[key.replace(" ", "_")] = int(last.split(key + " = ")[1].split(",")[0])
                except ValueError:
                    pass
        return new, stats
    finally:
        if not keep_dir:
            shutil.rmtree(d, ignore_errors=True)


def official_initial_plan(inst, init="PP", seed=0, time_limit=10.0):
    """Initial plan computed by the official MAPF-LNS code itself (--initAlgo
    PP by default, zero LNS iterations), as in the experiments of the paper."""
    b = binary()
    if b is None:
        raise RuntimeError("official MAPF-LNS binary not found")
    d = tempfile.mkdtemp(prefix="olns_init_")
    try:
        G = inst.graph
        dummy = [tuple(inst.starts), tuple(inst.goals)]
        mp, sc, _, back = _write_files(inst, dummy, d)
        outp = os.path.join(d, "paths.txt")
        cmd = [b, "-m", mp, "-a", sc, "-k", str(inst.n), "-t", str(time_limit),
               "--initAlgo", init, "--maxIterations", "0", "--outputPaths", outp,
               "--seed", str(seed), "-o", os.path.join(d, "out")]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 30)
        if not os.path.exists(outp):
            return None
        lines = open(outp).read().split("\n")
        n = int(lines[0])
        paths = [[back[int(x)] for x in lines[1 + k].split(",") if x.strip()] for k in range(n)]
        plan = paths_to_plan(paths)
        validate(G, inst.starts, inst.goals, plan)
        return plan
    finally:
        shutil.rmtree(d, ignore_errors=True)
