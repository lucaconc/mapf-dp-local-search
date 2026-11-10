# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapter for the C++ local search (cpp/dpls.cpp).

The binary implements the same neighborhoods and tie-breaking rules as
local_search.py and returns identical plans; it is used for speed.
It is looked up in $DPLS_BIN, then in cpp/dpls; build it with
    g++ -O2 -std=c++17 -o cpp/dpls cpp/dpls.cpp
"""
import json
import os
import subprocess
import tempfile

from .local_search import (PathDistance, Pointwise, AgentDistance, Combined,
                           GroupedClock, trim)
from .plan import validate

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(_HERE, "..", "cpp", "dpls")


def binary():
    b = os.environ.get("DPLS_BIN", _DEFAULT)
    return b if os.path.isfile(b) and os.access(b, os.X_OK) else None


def available():
    return binary() is not None


def build(quiet=True):
    """Compile cpp/dpls.cpp (needs g++ with C++17)."""
    src = os.path.join(_HERE, "..", "cpp", "dpls.cpp")
    subprocess.run(["g++", "-O3", "-std=c++17", "-o", _DEFAULT, src], check=True,
                   capture_output=quiet)
    return _DEFAULT


class Window:
    """Free-window neighborhood (C++ only): the plan follows the reference up
    to x^{t0}, moves freely for at most W rounds, lands on the latest matching
    configuration x^{t1} of the reference and follows it to the end; free
    segments shorter than t1 - t0 cut that part out. One window only
    (multi=True: a new window may open after each return). K bounds the
    number of agents that move during the window (None: all may move).

    Size: with W = 1 the neighborhood has O(n T^2) states whatever the degree;
    with W >= 2 and unbounded K it can have 2^n states already with maximum
    out-degree 1; with K fixed it has O(n T^2 C(n, K) |V|^(K (W - 1))) states.
    """

    def __init__(self, W=2, K=None, multi=False, budget=None):
        self.W, self.K, self.multi, self.budget = W, K, multi, budget

    def __repr__(self):
        return f"Window({self.W}, K={self.K}, multi={self.multi}, budget={self.budget})"


class Retime:
    """Order-preserving retiming (C++ only, not DP): same routes and same
    order of visits at every node, earliest schedule (longest paths). Every
    arrival time is minimal, so makespan and Sum-of-Costs are optimal among
    the plans with these routes and orders."""

    def __repr__(self):
        return "Retime()"


class OrderSwap:
    """Order swaps (C++ only, not DP): exchange two consecutive visits of a
    node, or of a pair of agents at every node where one directly follows the
    other (an overtake), then retime; the best candidate is returned. O(m)
    candidates for m visits, each solved in polynomial time."""

    def __repr__(self):
        return "OrderSwap()"


class FreeSet:
    """Route-lag distance with a free set (C++ only): the K agents of a
    candidate set move freely, the others keep the route-lag constraint
    (route-restricted if restricted=True) with radius R and weight lam. The
    candidates are computed from the current reference, LNS-style (each
    agent by decreasing delay with the K-1 agents it interacts with most, and
    the agents through the most crowded node); the local search tries the
    first `candidates` of them in turn. Anchored separable with zero terms for
    the free agents: polynomial for fixed K (Theorem thm_poly)."""

    def __init__(self, K=2, R=1, lam=1.0, restricted=False, candidates=10, budget=None,
                 corridor=None, beam=None):
        self.K, self.R, self.lam, self.restricted = K, R, lam, restricted
        self.candidates, self.budget, self.corridor = candidates, budget, corridor
        self.beam = beam          # beam width of this neighborhood's searches (None: global)

    def __repr__(self):
        return (f"FreeSet(K={self.K}, R={self.R}, lam={self.lam}, restricted={self.restricted}, "
                f"candidates={self.candidates}, corridor={self.corridor})")


class AdaptiveRadius:
    """Neighborhood with an adaptive radius for the C++ binary.

    `template` is a C++ spec containing "{R}" (e.g. "gc:{R}:1.0:none:1");
    the search starts with radius rmin and, when a search ends without
    improvement within its budget, repeats it with the radius increased by
    one, up to rmax. After every improvement the radius returns to rmin.
    """

    def __init__(self, template, rmin, rmax, geometric=False, cap=None, gext=False):
        assert "{R}" in template and 0 <= rmin <= rmax
        self.template, self.rmin, self.rmax = template, rmin, rmax
        self.geometric = geometric      # radius step doubling after complete searches
        self.cap = cap                  # beyond rmax, grow up to cap while searches are complete
        self.gext = gext                # beyond rmax, geometric growth up to cap, also when pruned

    def __repr__(self):
        return f"AdaptiveRadius({self.template!r}, {self.rmin}, {self.rmax})"


def spec(nb):
    """Command-line spec of a Python neighborhood object."""
    if isinstance(nb, AdaptiveRadius):
        return (f"{nb.template}@{nb.rmin}-{nb.rmax}"
                + (f"+{nb.cap}" if getattr(nb, "cap", None) else "")
                + ("^" if getattr(nb, "geometric", False) else "")
                + ("*" if getattr(nb, "gext", False) else ""))
    if isinstance(nb, Retime):
        return "retime"
    if isinstance(nb, OrderSwap):
        return "swap"
    if isinstance(nb, FreeSet):
        sp = (f"gc:{nb.R}:{nb.lam}:none:{int(nb.restricted)}:2:4:{nb.K}:{{R}}"
              + (f":{nb.corridor}" if nb.corridor is not None else "")
              + f"@0-{nb.candidates - 1}!")
        return sp + (f"%{nb.beam}" if getattr(nb, "beam", None) else "") + (f"#{nb.budget}" if nb.budget else "")
    if isinstance(nb, Window):
        sp = f"window:{nb.W}:{-1 if nb.K is None else nb.K}"
        if nb.multi:
            sp += ":multi"
        return sp + (f"#{nb.budget}" if nb.budget else "")
    if isinstance(nb, Combined):
        return f"combined:{nb.R}:{nb.R_sm}"
    if isinstance(nb, AgentDistance):
        return f"agent:{nb.R}"
    if isinstance(nb, Pointwise):
        return f"pointwise:{nb.R}"
    if isinstance(nb, PathDistance):
        return f"path:{nb.R}"
    if isinstance(nb, GroupedClock):
        g = "auto" if nb.groups_spec == "auto" else "none"
        if nb.groups_spec not in (None, "auto"):
            raise ValueError("explicit groups are not supported by the C++ binary")
        return (f"gc:{nb.R}:{nb.lam}:{g}:{int(nb.restricted)}:"
                f"{nb.window}:{nb.max_groups}")
    raise ValueError(f"unsupported neighborhood {nb!r}")


def neighborhood_size(inst, plan, nb, max_states=10**7):
    """Number of states of the sequentialized automaton in the neighborhood
    of `plan` (whole neighborhood explored, no search)."""
    b = binary()
    G = inst.graph
    plan = trim(plan)
    with tempfile.TemporaryDirectory(prefix="dpls_") as d:
        fin = os.path.join(d, "in.txt")
        arcs = [(u, v) for u in range(G.N) for v in G.out[u] if v != u]
        with open(fin, "w") as fh:
            fh.write(f"{G.N} {len(arcs)}\n")
            fh.write("".join(f"{u} {v}\n" for u, v in arcs))
            fh.write(f"{inst.n}\n{' '.join(map(str, inst.starts))}\n"
                     f"{' '.join(map(str, inst.goals))}\n{len(plan) - 1}\n")
            fh.write("".join(" ".join(map(str, x)) + "\n" for x in plan))
        res = subprocess.run([b, fin, os.path.join(d, "out.txt"), "--nb", spec(nb),
                              "--enumerate", "--expansions", str(max_states)],
                             capture_output=True, text=True)
        return json.loads(res.stdout.strip().splitlines()[-1])


def cpp_local_search(inst, plan, neighborhoods, tiebreak=None, time_limit=60.0,
                     max_expansions=300000, max_iters=10000, objective="makespan",
                     vnd=False, keep_radius=False, astar=False, beam=0, chain=False, threads=1):
    """Same interface as local_search.local_search; returns (plan, stats).

    objective="soc" minimizes the Sum-of-Costs (Appendix A.3 of the paper)
    instead of the (scaled) makespan; C++ binary only. astar=True replaces
    Dijkstra with A* (consistent heuristic: distances to the goals, inside
    the route for the route-restricted distance, plus the reference schedule
    of the agents that can no longer deviate in the agent distance): same
    optimal cost in each neighborhood, fewer states, possibly another plan."""
    b = binary()
    if b is None:
        raise RuntimeError("C++ local search not built (cpp/dpls)")
    if not isinstance(neighborhoods, (list, tuple)):
        neighborhoods = [neighborhoods]
    G = inst.graph
    plan = trim(plan)
    with tempfile.TemporaryDirectory(prefix="dpls_") as d:
        fin, fout = os.path.join(d, "in.txt"), os.path.join(d, "out.txt")
        arcs = [(u, v) for u in range(G.N) for v in G.out[u] if v != u]
        with open(fin, "w") as fh:
            fh.write(f"{G.N} {len(arcs)}\n")
            fh.write("".join(f"{u} {v}\n" for u, v in arcs))
            fh.write(f"{inst.n}\n{' '.join(map(str, inst.starts))}\n"
                     f"{' '.join(map(str, inst.goals))}\n{len(plan) - 1}\n")
            fh.write("".join(" ".join(map(str, x)) + "\n" for x in plan))
        cmd = [b, fin, fout, "--time", str(time_limit), "--expansions",
               str(max_expansions), "--max-searches", str(max_iters),
               "--tiebreak", "summin" if tiebreak == "summin" else "none",
               "--objective", objective] + (["--vnd"] if vnd else []) \
            + (["--keep-radius"] if keep_radius else []) + (["--astar"] if astar else []) \
            + (["--beam", str(int(beam))] if beam else []) \
            + (["--chain"] if chain else []) \
            + (["--threads", str(int(threads))] if threads and threads > 1 else []) \
            + os.environ.get("DPLS_EXTRA", "").split() \
            + (os.environ.get("DPLS_EXTRA_BEAM", "").split() if beam else [])
        # DPLS_EXTRA: extra options of the binary; DPLS_EXTRA_BEAM: only for beam searches
        for nb in neighborhoods:
            cmd += ["--nb", spec(nb)]
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=time_limit + 60)
        if res.returncode != 0:
            raise RuntimeError(f"dpls failed: {res.stderr[-500:]}")
        stats = json.loads(res.stdout.strip().splitlines()[-1])
        with open(fout) as fh:
            new = [tuple(int(x) for x in line.split()) for line in fh if line.strip()]
    validate(G, inst.starts, inst.goals, new)
    return new, stats
