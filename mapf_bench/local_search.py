# SPDX-License-Identifier: GPL-3.0-or-later
"""DP-based local search on the sequentialized automaton (the paper's method).

A sequential state is (z, y, aux): z is the configuration at the beginning of
the round, y the destinations already chosen in the round (the phase is
len(y)), and aux the auxiliary data of the neighborhood. Every single-agent
decision costs 1, so the search minimizes the scaled makespan n|f|.

Neighborhoods (Section 3 of the paper):
  PathDistance        d_seq <= R
  Pointwise           comparison at the same time step, radius R
  AgentDistance       at most R agents change their timed trajectory
  Combined            agent distance R_ag and sum-min distance R_sm
  RouteLag            route-lag distance (single clock), weight lam
  GroupedClock        grouped-clock distance with a partition of the agents
  RouteRestricted     route-restricted variant (routes fixed, timing free),
                      with one clock or with grouped clocks
"""
import heapq
import time

import numpy as np

from .plan import trim, makespan

BIG = 10**6


# ---------------------------------------------------------------------------
# helpers on the reference plan
# ---------------------------------------------------------------------------
def _ref_at(ref, t):
    return ref[min(t, len(ref) - 1)]


def _routes(ref):
    n = len(ref[0])
    pis, rs = [], []
    for k in range(n):
        pi, r = [ref[0][k]], [0]
        for t in range(1, len(ref)):
            if ref[t][k] != ref[t - 1][k]:
                pi.append(ref[t][k])
            r.append(len(pi) - 1)
        pis.append(pi)
        rs.append(r)
    return pis, rs


def interaction_groups(ref, window=2, max_groups=4):
    """Partition agents into groups that interact in the reference plan:
    two agents interact if they occupy the same node at times at most
    `window` apart. Groups beyond `max_groups` are merged, smallest first."""
    n = len(ref[0])
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    occ = {}
    for t, x in enumerate(ref):
        for k, v in enumerate(x):
            occ.setdefault(v, []).append((t, k))
    for v, lst in occ.items():
        lst.sort()
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                if lst[b][0] - lst[a][0] > window:
                    break
                ra, rb = find(lst[a][1]), find(lst[b][1])
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    groups = sorted(groups.values(), key=len)
    while len(groups) > max_groups:
        a = groups.pop(0)
        groups[0] = sorted(groups[0] + a)
        groups.sort(key=len)
    return [sorted(g) for g in groups]


# ---------------------------------------------------------------------------
# neighborhoods
# ---------------------------------------------------------------------------
REJECT = object()


class Neighborhood:
    """Interface of the locality test.

    init(z0) returns (aux, cache) for the initial state. step(...) is called
    for every generated state with the parent's (z, y, aux, cache), the
    deciding agent and its new node w, and the new state (z2, y2); it returns
    (aux2, cache2), or (REJECT, None) if the new state is outside the
    neighborhood. aux is part of the search state; cache is not (it only
    carries data for incremental evaluation and depends on the state alone).
    """
    name = "base"

    def prepare(self, inst, ref):
        self.inst, self.ref = inst, ref
        self.n, self.T = inst.n, len(ref) - 1
        self.D = np.minimum(inst.graph.dV, BIG)

    def init(self, z0):
        return None, None

    def step(self, z, y, aux, cache, agent, w, z2, y2, done):
        return aux, cache

    def marked(self, z, aux):
        return True


class AnchoredSum(Neighborhood):
    """Distances of the form min over (grouped) anchor times s of
    sum_k psi_k(z_k, s) + sum_{k<i} psi_k(y_k, s+1), evaluated
    incrementally: the cache holds, for each group g, the vectors
    Z_g(s) = sum_{k in g} psi_k(z_k, s) and Y_g(s) = sum_{k in g, k<i}
    psi_k(y_k, s), s = 0..T+1, and the group minima."""

    R = 0
    groups_spec = None
    window, max_groups = 2, 4

    def _set_groups(self):
        n = self.n
        if self.groups_spec is None:
            self.groups = [list(range(n))]
        elif self.groups_spec == "auto":
            self.groups = interaction_groups(self.ref, self.window, self.max_groups)
        else:
            self.groups = [list(g) for g in self.groups_spec]
        self.grp = [0] * n
        for gi, g in enumerate(self.groups):
            for k in g:
                self.grp[k] = gi

    def _mins(self, Zv, Yv):
        return (Zv[:, : self.T + 1] + Yv[:, 1: self.T + 2]).min(axis=1)

    def init(self, z0):
        G = len(self.groups)
        Zv = np.zeros((G, self.T + 2))
        for k, v in enumerate(z0):
            Zv[self.grp[k]] += self.PSI[k, v]
        Yv = np.zeros_like(Zv)
        return None, (Zv, Yv, self._mins(Zv, Yv))

    def value_of_cache(self, cache):
        return float(cache[2].sum())

    def step(self, z, y, aux, cache, agent, w, z2, y2, done):
        Zv, Yv, mins = cache
        g = self.grp[agent]
        if done:
            Zv2 = Yv.copy()
            Zv2[g] += self.PSI[agent, w]
            Yv2 = np.zeros_like(Yv)
            mins2 = Zv2[:, : self.T + 1].min(axis=1)
        else:
            Zv2 = Zv
            Yv2 = Yv.copy()
            Yv2[g] += self.PSI[agent, w]
            mins2 = mins.copy()
            mins2[g] = (Zv[g, : self.T + 1] + Yv2[g, 1: self.T + 2]).min()
        if mins2.sum() > self.R + 1e-9:
            return REJECT, None
        return aux, (Zv2, Yv2, mins2)

    def value(self, z, y):
        """Distance of an arbitrary sequential state (non-incremental)."""
        _, cache = self.init(z)
        Zv, Yv, _ = cache
        for k, v in enumerate(y):
            Yv[self.grp[k]] += self.PSI[k, v]
        return float(self._mins(Zv, Yv).sum())


class PathDistance(AnchoredSum):
    """d_seq <= R: psi_k(v, s) = d_V(v, x^s_k), single clock. The value at
    s = T+1 is infinite, so that partial states are compared only with the
    reference states (x^s, x^{s+1}[:i]) for s < T."""
    name = "path"

    def __init__(self, R=1):
        self.R = R

    def prepare(self, inst, ref):
        super().prepare(inst, ref)
        n, T, N = self.n, self.T, inst.graph.N
        self.groups_spec = None
        self._set_groups()
        PSI = np.full((n, N, T + 2), np.inf)
        for s in range(T + 1):
            for k in range(n):
                PSI[k, :, s] = self.D[:, ref[s][k]]
        self.PSI = PSI


class GroupedClock(AnchoredSum):
    """Grouped-clock distance (route-lag when there is a single group).

    restricted=True gives the route-restricted variant: an agent off its
    reference route has infinite cost, so only the timing can change.
    groups: list of agent lists, "auto" (interaction groups), or None
    (single clock, i.e. the route-lag distance).
    """

    def __init__(self, R=1, lam=1.0, groups=None, restricted=False,
                 window=2, max_groups=4):
        self.R, self.lam, self.groups_spec = R, lam, groups
        self.restricted, self.window, self.max_groups = restricted, window, max_groups
        base = "rr" if restricted else "lag"
        self.name = base if groups is None else base + "_gc"

    def prepare(self, inst, ref):
        super().prepare(inst, ref)
        n, T, N = self.n, self.T, inst.graph.N
        self._set_groups()
        pis, rs = _routes(ref)
        PSI = np.full((n, N, T + 2), np.inf)
        for k in range(n):
            route = pis[k]
            if self.restricted:
                Dv = np.full((N, len(route)), np.inf)
                for j, v in enumerate(route):
                    Dv[v, j] = 0.0
            else:
                Dv = self.D[:, route].astype(float)
            r = np.array([rs[k][min(s, T)] for s in range(T + 2)])
            lag = self.lam * np.abs(np.arange(len(route))[None, :] - r[:, None])
            PSI[k] = (Dv[:, None, :] + lag[None, :, :]).min(axis=2)
        self.PSI = PSI


class Pointwise(Neighborhood):
    """Comparison with the reference at the same time step (aux = clock)."""
    name = "pointwise"

    def __init__(self, R=1):
        self.R = R

    def init(self, z0):
        return 0, None

    def step(self, z, y, tau, cache, agent, w, z2, y2, done):
        if done:
            tau += 1
        if tau > self.T or (len(y2) > 0 and tau + 1 > self.T):
            return REJECT, None
        val = sum(int(self.D[a, b]) for a, b in zip(z2, self.ref[tau]))
        if y2:
            val += sum(int(self.D[a, b]) for a, b in zip(y2, self.ref[tau + 1]))
        return (tau, None) if val <= self.R else (REJECT, None)


class AgentDistance(Neighborhood):
    """At most R agents change their timed trajectory (aux = (c, D-mask))."""
    name = "agent"

    def __init__(self, R=1):
        self.R = R

    def prepare(self, inst, ref):
        super().prepare(inst, ref)
        T, n = self.T, self.n
        final = ref[T]
        tau = []
        for k in range(n):
            last = 0
            for t in range(T + 1):
                if ref[t][k] != final[k]:
                    last = t
            tau.append(last)
        self.Pc = {}
        for c in range(1, T + 2):
            m = 0
            for k in range(n):
                if tau[k] >= c:
                    m |= 1 << k
            self.Pc[c] = m

    def init(self, z0):
        return (1, 0), None

    def step(self, z, y, aux, cache, agent, w, z2, y2, done):
        c, D = aux
        if w != _ref_at(self.ref, c)[agent]:
            D |= 1 << agent
            if bin(D).count("1") > self.R:
                return REJECT, None
        if done:
            c = min(c + 1, self.T + 1)
        return (c, D), None

    def marked(self, z, aux):
        c, D = aux
        return bin(D | self.Pc[min(c, self.T + 1)]).count("1") <= self.R


class SumMinMixin:
    def _prepare_summin(self):
        n, N = self.n, self.inst.graph.N
        self.mu = np.zeros((n, N), dtype=np.int64)
        for k in range(n):
            Yk = sorted({x[k] for x in self.ref})
            self.mu[k] = self.D[:, Yk].min(axis=1)

    def summin(self, z, y):
        x = tuple(y) + tuple(z[len(y):])
        return int(sum(self.mu[k, v] for k, v in enumerate(x)))


class Combined(AgentDistance, SumMinMixin):
    """Agent distance R_ag together with the sum-min constraint R_sm."""
    name = "combined"

    def __init__(self, R_ag=1, R_sm=0):
        super().__init__(R_ag)
        self.R_sm = R_sm

    def prepare(self, inst, ref):
        super().prepare(inst, ref)
        self._prepare_summin()

    def step(self, z, y, aux, cache, agent, w, z2, y2, done):
        if self.summin(z2, y2) > self.R_sm:
            return REJECT, None
        return super().step(z, y, aux, cache, agent, w, z2, y2, done)


# ---------------------------------------------------------------------------
# exact neighborhood search (Algorithm 1) and local search (Algorithm 3)
# ---------------------------------------------------------------------------
def neighborhood_search(inst, ref, nbhd, tiebreak=None, max_expansions=300000,
                        deadline=None):
    """Dijkstra on the sequentialized automaton restricted to the neighborhood.

    Returns (plan, info); plan is None if the expansion budget or the
    deadline is exceeded. tiebreak="summin" uses Ordering B of the paper.
    """
    G, n, goals = inst.graph, inst.n, tuple(inst.goals)
    assert G.N >= n + 1, "Lemma lem_mapf_Si requires |V| >= n+1"
    nbhd.prepare(inst, ref)
    if tiebreak == "summin":
        tb = SumMinMixin()
        tb.n, tb.inst, tb.ref, tb.D = n, inst, ref, nbhd.D
        tb._prepare_summin()
        tbf = tb.summin
    else:
        tbf = lambda z, y: 0
    z0 = tuple(ref[0])
    aux0, cache0 = nbhd.init(z0)
    s0 = (z0, (), aux0)
    dist = {s0: 0}
    parent = {s0: None}
    closed = set()
    heap = [(0, 0, 0, 0, s0, cache0)]
    counter = 1
    expansions = 0
    while heap:
        g, _, _, _, s, cache = heapq.heappop(heap)
        if s in closed:
            continue
        closed.add(s)
        z, y, aux = s
        if not y and z == goals and nbhd.marked(z, aux):
            states = []
            while s is not None:
                if not s[1]:
                    states.append(s[0])
                s = parent[s]
            return states[::-1], {"expansions": expansions, "states": len(dist)}
        expansions += 1
        if expansions > max_expansions or (
                deadline is not None and expansions % 256 == 0
                and time.perf_counter() > deadline):
            return None, {"expansions": expansions, "states": len(dist),
                          "budget_exceeded": True}
        i = len(y)
        zi = z[i]
        yset = set(y)
        for w in G.out[zi]:
            if w in yset:
                continue
            if any(y[k] == zi and z[k] == w for k in range(i)):
                continue  # swap
            if i == n - 1:
                z2, y2, done = y + (w,), (), True
            else:
                z2, y2, done = z, y + (w,), False
            aux2, cache2 = nbhd.step(z, y, aux, cache, i, w, z2, y2, done)
            if aux2 is REJECT:
                continue
            s2 = (z2, y2, aux2)
            if s2 in closed:
                continue
            g2 = g + 1
            if g2 < dist.get(s2, 1 << 60):
                dist[s2] = g2
                parent[s2] = s
                heapq.heappush(heap, (g2, tbf(z2, y2), len(y2), counter, s2, cache2))
                counter += 1
    return None, {"expansions": expansions, "states": len(dist), "failed": True}


def local_search(inst, plan, neighborhoods, tiebreak=None, time_limit=60.0,
                 max_expansions=300000, max_iters=10000):
    """Algorithm 3, generalized to cycle through a list of neighborhoods
    (one neighborhood: the paper's local search; two: the alternating search).

    Stops when a full cycle brings no improvement. Returns (plan, stats).
    """
    cur = trim(plan)
    t0 = time.perf_counter()
    stats = {"searches": 0, "improvements": 0, "budget_exceeded": 0,
             "states": 0}
    if isinstance(neighborhoods, Neighborhood):
        neighborhoods = [neighborhoods]
    stale = 0
    j = 0
    while stale < len(neighborhoods) and stats["searches"] < max_iters:
        if time.perf_counter() - t0 > time_limit:
            stats["timeout"] = True
            break
        nb = neighborhoods[j % len(neighborhoods)]
        j += 1
        new, info = neighborhood_search(inst, cur, nb, tiebreak, max_expansions,
                                        deadline=t0 + time_limit)
        stats["searches"] += 1
        stats["states"] = max(stats["states"], info["states"])
        if info.get("budget_exceeded"):
            stats["budget_exceeded"] += 1
        if new is not None and makespan(new) < makespan(cur):
            cur = trim(new)
            stats["improvements"] += 1
            stale = 0
        else:
            stale += 1
    stats["time"] = time.perf_counter() - t0
    return cur, stats
