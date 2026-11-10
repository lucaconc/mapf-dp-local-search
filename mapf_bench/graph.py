# SPDX-License-Identifier: GPL-3.0-or-later
"""Directed graphs with self-loops, shortest-path tables and the node metric d_V."""
from collections import deque
import numpy as np

INF = 10**9


class Graph:
    """Directed graph on nodes 0..N-1. Every node carries a self-loop (wait)."""

    def __init__(self, n_nodes, arcs, coords=None, name="graph"):
        self.N = n_nodes
        self.name = name
        self.coords = coords  # optional {node: (row, col)} for grids
        self.out = [set() for _ in range(n_nodes)]
        for u, v in arcs:
            if u != v:
                self.out[u].add(v)
        self.inn = [set() for _ in range(n_nodes)]
        for u in range(n_nodes):
            for v in self.out[u]:
                self.inn[v].add(u)
        for u in range(n_nodes):  # self-loops
            self.out[u].add(u)
            self.inn[u].add(u)
        self.out = [sorted(s) for s in self.out]
        self.inn = [sorted(s) for s in self.inn]
        self._dist = None
        self._dV = None

    @property
    def n_arcs(self):
        return sum(len(o) for o in self.out)

    def bfs(self, src, reverse=False):
        adj = self.inn if reverse else self.out
        d = np.full(self.N, INF, dtype=np.int64)
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[v] == INF:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    @property
    def dist(self):
        """dist[u, v] = length of a shortest directed walk from u to v."""
        if self._dist is None:
            self._dist = np.stack([self.bfs(u) for u in range(self.N)])
        return self._dist

    @property
    def dV(self):
        """Symmetric node metric d_V(u, v) = max(dist(u, v), dist(v, u))."""
        if self._dV is None:
            self._dV = np.maximum(self.dist, self.dist.T)
        return self._dV

    def strongly_connected(self):
        return bool((self.bfs(0) < INF).all() and (self.bfs(0, reverse=True) < INF).all())

    def degree_und(self, u):
        """Number of distinct neighbours, ignoring orientation and self-loops."""
        return len((set(self.out[u]) | set(self.inn[u])) - {u})
