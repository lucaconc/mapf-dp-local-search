# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapter for LaCAM3 (Okumura; https://github.com/kei18/lacam3), a complete
MAPF solver for 4-connected grids. Used for the initial plans of dense
instances (first solution: --no-star --no-refiner) and as an anytime
baseline (LaCAM* with its refinements). Binary: $LACAM_BIN, else
third_party/lacam3/build/main (scripts/build_lacam.sh)."""
import os
import shutil
import subprocess
import tempfile

from .official_lns import _write_files, is_grid_instance
from .plan import paths_to_plan, validate, trim

_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party",
                        "lacam3", "build", "main")


def binary():
    b = os.environ.get("LACAM_BIN", _DEFAULT)
    return b if os.path.isfile(b) and os.access(b, os.X_OK) else None


def available():
    return binary() is not None


def lacam_solve(inst, time_limit=10.0, anytime=False, seed=0):
    """Plan of LaCAM3 (first solution, or LaCAM* after `time_limit` seconds
    when anytime=True), or None if it fails. Single-threaded."""
    b = binary()
    if b is None:
        raise RuntimeError("LaCAM3 not built (scripts/build_lacam.sh)")
    if not is_grid_instance(inst):
        raise ValueError("LaCAM3 supports 4-connected grids only")
    d = tempfile.mkdtemp(prefix="lacam_")
    try:
        mp, sc, _, back = _write_files(inst, [tuple(inst.starts), tuple(inst.goals)], d)
        cols = max(c for _, c in inst.graph.coords.values()) + 1
        out = os.path.join(d, "result.txt")
        cmd = [b, "-m", mp, "-i", sc, "-N", str(inst.n), "-t", str(time_limit), "-s", str(seed),
               "-o", out, "--no-multi-thread"] + ([] if anytime else ["--no-star", "--no-refiner"])
        subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 60)
        if not os.path.exists(out):
            return None
        lines = open(out).read().split("\n")
        if not any(l.strip() == "solved=1" for l in lines):
            return None
        sol = lines[lines.index("solution=") + 1:]
        plan = []
        for l in sol:
            if ":" not in l:
                continue
            body = l.split(":", 1)[1]
            pts = [p.strip("()") for p in body.split("),(")]
            conf = []
            for p in pts:
                p = p.strip("(),")
                if not p:
                    continue
                x, y = map(int, p.split(","))
                conf.append(back[y * cols + x])
            plan.append(tuple(conf))
        plan = trim(plan)
        validate(inst.graph, inst.starts, inst.goals, plan)
        return plan
    finally:
        shutil.rmtree(d, ignore_errors=True)
