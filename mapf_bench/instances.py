# SPDX-License-Identifier: GPL-3.0-or-later
"""Instance generators.

Every generator returns an `Instance`. Instances that come with a
hand-built initial plan (the merge and ring examples of the paper) store it
in `initial_plan`; otherwise the initial plan is computed by a planner.
"""
from dataclasses import dataclass, field
import random

from .graph import Graph


@dataclass
class Instance:
    name: str
    family: str
    graph: Graph
    starts: tuple
    goals: tuple
    params: dict = field(default_factory=dict)
    initial_plan: list = None

    @property
    def n(self):
        return len(self.starts)


def _random_endpoints(rng, nodes, n):
    nodes = list(nodes)
    starts = rng.sample(nodes, n)
    goals = rng.sample(nodes, n)
    return tuple(starts), tuple(goals)


# ---------------------------------------------------------------------------
# Example 2 of the paper: reordering at a merge
# ---------------------------------------------------------------------------
def merge(k_short=1, ell=6, ret_len=None):
    """k short agents and one long agent B merge into a single-lane node c.

    Reference (initial) plan: the short agents cross first, B last. Letting B
    cross first reduces the makespan by k while the Sum-of-Costs is unchanged.
    A long return chain makes the graph strongly connected without creating
    shorter routes to the goals.
    """
    names = {}

    def node(x):
        if x not in names:
            names[x] = len(names)
        return names[x]

    arcs = []
    s = [node(("s", i)) for i in range(k_short)]
    b = node("b")
    c, d = node("c"), node("d")
    e = [node(("e", i)) for i in range(k_short)]
    p = [node(("p", j)) for j in range(1, ell + 1)]
    for u in s + [b]:
        arcs.append((u, c))
    arcs.append((c, d))
    for ei in e:
        arcs.append((d, ei))
    arcs.append((d, p[0]))
    arcs += list(zip(p[:-1], p[1:]))
    ret_len = ret_len or (ell + k_short + 4)
    r = [node(("r", j)) for j in range(ret_len)]
    arcs += list(zip(r[:-1], r[1:]))
    for g in e + [p[-1]]:
        arcs.append((g, r[0]))
    for u in s + [b]:
        arcs.append((r[-1], u))
    G = Graph(len(names), arcs, name="merge")
    starts = tuple(s + [b])
    goals = tuple(e + [p[-1]])
    # reference plan: short agent i enters c at time i+1, B at time k+1
    paths = []
    for i in range(k_short):
        paths.append([s[i]] * (i + 1) + [c, d, e[i]])
    paths.append([b] * (k_short + 1) + [c, d] + p)
    T = max(len(q) for q in paths) - 1
    plan = [tuple(q[min(t, len(q) - 1)] for q in paths) for t in range(T + 1)]
    return Instance(f"merge_k{k_short}_l{ell}", "merge", G, starts, goals,
                    {"k": k_short, "ell": ell}, plan)


# ---------------------------------------------------------------------------
# Example 1 of the paper: serialized groups on two directed cycles
# ---------------------------------------------------------------------------
def ring(L=6, L2=8, q=3, bridge=None):
    """A full directed L-cycle and one agent on a second cycle.

    Reference plan: the lone agent moves q steps first, then the full cycle
    rotates q times (length 2q). Doing both simultaneously gives length q.
    """
    assert L >= 3 and 1 <= q < L and q < L2
    bridge = bridge or (L + L2)
    C = list(range(L))
    C2 = list(range(L, L + L2))
    arcs = [(C[j], C[(j + 1) % L]) for j in range(L)]
    arcs += [(C2[j], C2[(j + 1) % L2]) for j in range(L2)]
    nxt = L + L2
    # two long one-way bridges, ring -> second cycle and back
    def chain(u, v, length):
        nonlocal nxt
        prev = u
        for _ in range(length):
            arcs.append((prev, nxt))
            prev = nxt
            nxt += 1
        arcs.append((prev, v))
    chain(C[0], C2[L2 // 2], bridge)
    chain(C2[L2 // 2 + 1], C[1], bridge)
    G = Graph(nxt, arcs, name="ring")
    starts = tuple(C[k] for k in range(L)) + (C2[0],)
    goals = tuple(C[(k + q) % L] for k in range(L)) + (C2[q],)
    plan = [starts]
    rot = 0
    for t in range(1, q + 1):  # lone agent first
        plan.append(tuple(C[(k + rot) % L] for k in range(L)) + (C2[t],))
    for rot in range(1, q + 1):  # then the full cycle rotates
        plan.append(tuple(C[(k + rot) % L] for k in range(L)) + (C2[q],))
    return Instance(f"ring_L{L}_q{q}", "ring", G, starts, goals,
                    {"L": L, "L2": L2, "q": q}, plan)


# ---------------------------------------------------------------------------
# Grid maps with bottlenecks (Section 5.3 of the paper)
# ---------------------------------------------------------------------------
def grid_bottleneck(rows=9, cols=9, K=1, n=8, free_target=None, seed=0):
    """4-connected undirected grid split into K+1 blocks by K obstacle
    columns, each with a single free cell (an articulation point)."""
    rng = random.Random(seed)
    for _attempt in range(1000):
        blocked = set()
        # choose K separating columns, blocks at least two columns wide
        cand = list(range(2, cols - 2))
        sep = []
        rng.shuffle(cand)
        for c in cand:
            if len(sep) == K:
                break
            if all(abs(c - c2) >= 3 for c2 in sep):
                sep.append(c)
        if len(sep) < K:
            continue
        for c in sep:
            hole = rng.randrange(rows)
            for r in range(rows):
                if r != hole:
                    blocked.add((r, c))
        cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked]
        target = free_target or int(0.85 * len(cells))
        inner = [x for x in cells if x[1] not in sep]
        rng.shuffle(inner)
        for x in inner[: max(0, len(cells) - target)]:
            blocked.add(x)
        cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in blocked]
        idx = {x: i for i, x in enumerate(cells)}
        arcs = []
        for (r, c) in cells:
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                y = (r + dr, c + dc)
                if y in idx:
                    arcs.append((idx[(r, c)], idx[y]))
        G = Graph(len(cells), arcs, coords={i: x for x, i in idx.items()},
                  name="grid")
        if not G.strongly_connected() or G.N < n + 2:
            continue
        starts, goals = _random_endpoints(rng, range(G.N), n)
        return Instance(f"grid{rows}x{cols}_K{K}_n{n}_s{seed}", "grid_bottleneck",
                        G, starts, goals, {"K": K, "n": n, "seed": seed})
    raise RuntimeError("could not generate a connected grid")


# ---------------------------------------------------------------------------
# One-way (Manhattan) grids
# ---------------------------------------------------------------------------
def oneway_grid(rows=6, cols=6, n=8, seed=0):
    """Grid whose rows and columns are one-way, with alternating directions."""
    rng = random.Random(seed)
    idx = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
    arcs = []
    for r in range(rows):
        for c in range(cols - 1):
            u, v = idx[(r, c)], idx[(r, c + 1)]
            arcs.append((u, v) if r % 2 == 0 else (v, u))
    for c in range(cols):
        for r in range(rows - 1):
            u, v = idx[(r, c)], idx[(r + 1, c)]
            arcs.append((u, v) if c % 2 == 1 else (v, u))
    G = Graph(rows * cols, arcs, coords={i: x for x, i in idx.items()}, name="oneway")
    assert G.strongly_connected(), "one-way grid must have even dimensions"
    starts, goals = _random_endpoints(rng, range(G.N), n)
    return Instance(f"oneway{rows}x{cols}_n{n}_s{seed}", "oneway_grid", G,
                    starts, goals, {"n": n, "seed": seed})


# ---------------------------------------------------------------------------
# Random strongly connected digraphs (Section 5.2 of the paper)
# ---------------------------------------------------------------------------
def random_digraph(N=30, arc_factor=3, n=8, seed=0):
    rng = random.Random(seed)
    for _ in range(10000):
        arcs = set()
        while len(arcs) < arc_factor * N:
            u, v = rng.randrange(N), rng.randrange(N)
            if u != v:
                arcs.add((u, v))
        G = Graph(N, arcs, name="random")
        if G.strongly_connected():
            starts, goals = _random_endpoints(rng, range(N), n)
            return Instance(f"rand{N}_n{n}_s{seed}", "random_digraph", G,
                            starts, goals, {"N": N, "n": n, "seed": seed})
    raise RuntimeError("could not generate a strongly connected digraph")


# ---------------------------------------------------------------------------
# Several dense one-way rings joined by two-way bridges
# ---------------------------------------------------------------------------
def multi_ring(n_rings=3, L=6, free_per_ring=1, bridge=3, seed=0):
    """Directed rings, each almost full, connected in a chain by two-way
    bridges. Agents rotate within their own ring, so the rings are
    independent; rule-based (serialized) plans handle them one at a time."""
    rng = random.Random(seed)
    arcs, rings = [], []
    nxt = 0
    for _ in range(n_rings):
        R = list(range(nxt, nxt + L))
        nxt += L
        arcs += [(R[j], R[(j + 1) % L]) for j in range(L)]
        rings.append(R)
    for a in range(n_rings - 1):
        prev = rings[a][0]
        chain = list(range(nxt, nxt + bridge))
        nxt += bridge
        path = [rings[a][0]] + chain + [rings[a + 1][L // 2]]
        for u, v in zip(path[:-1], path[1:]):
            arcs += [(u, v), (v, u)]
    G = Graph(nxt, arcs, name="multi_ring")
    starts, goals = [], []
    for R in rings:
        k = L - free_per_ring
        pos = rng.sample(range(L), k)
        shift = rng.randrange(1, L)
        for p in pos:
            starts.append(R[p])
            goals.append(R[(p + shift) % L])
    return Instance(f"multiring{n_rings}x{L}_s{seed}", "multi_ring", G,
                    tuple(starts), tuple(goals),
                    {"rings": n_rings, "L": L, "seed": seed})


# ---------------------------------------------------------------------------
# Families designed around structural weaknesses of MAPF-LNS
# ---------------------------------------------------------------------------
def full_rings(lengths=(7, 7), shifts=(3, 3), bridge=4, holes=0, seed=0,
               hub_agents=0):
    """Fully (or almost fully) occupied one-way rings joined by two-way bridges.

    In a full ring the only possible move is a joint rotation of all its
    agents, so improving the timing of a ring needs all its L agents in the
    destroy set of MAPF-LNS. The initial plan rotates one ring at a time,
    as a solver that handles one part of the graph after the other.
    """
    rng = random.Random(seed)
    arcs, rings, nxt = [], [], 0
    for L in lengths:
        R = list(range(nxt, nxt + L))
        nxt += L
        arcs += [(R[j], R[(j + 1) % L]) for j in range(L)]
        rings.append(R)
    hub = list(range(nxt, nxt + bridge))
    nxt += bridge
    arcs += [(u, v) for u, v in zip(hub[:-1], hub[1:])] + [(v, u) for u, v in zip(hub[:-1], hub[1:])]
    for a, R in enumerate(rings):  # each ring touches the hub at one node
        h = hub[a % len(hub)]
        arcs += [(R[0], h), (h, R[0])]
    G = Graph(nxt, arcs, name="full_rings")
    starts, goals, paths = [], [], []
    plan_rings = []
    for R, q in zip(rings, shifts):
        L = len(R)
        occ = sorted(rng.sample(range(L), L - holes)) if holes else list(range(L))
        plan_rings.append((R, occ, q))
        for p in occ:
            starts.append(R[p])
            goals.append(R[(p + q) % L])
    # optional free agents on the hub, shifting towards its end
    m = hub_agents
    assert m <= bridge // 2
    hub_start = [hub[j] for j in range(m)]
    hub_goal = [hub[bridge - m + j] for j in range(m)]
    starts += hub_start
    goals += hub_goal
    # initial plan: hub agents one at a time (front first), then ring after
    # ring, each rotating q times (holes rotate too)
    plan = [tuple(starts)]
    cur = list(starts)
    nr = len(starts) - m
    for j in reversed(range(m)):
        pos = hub.index(cur[nr + j])
        while cur[nr + j] != hub_goal[j]:
            pos += 1
            cur[nr + j] = hub[pos]
            plan.append(tuple(cur))
    base = 0
    for R, occ, q in plan_rings:
        L = len(R)
        idx = {v: i for i, v in enumerate(R)}
        for _ in range(q):
            for j in range(len(occ)):
                cur[base + j] = R[(idx[cur[base + j]] + 1) % L]
            plan.append(tuple(cur))
        base += len(occ)
    name = ("fullrings_" + "_".join(map(str, lengths)) + f"_h{holes}"
            + (f"_hub{m}" if m else "") + f"_s{seed}")
    return Instance(name, "full_rings", G, tuple(starts), tuple(goals),
                    {"lengths": lengths, "holes": holes, "seed": seed}, plan)


def convoy(n_long=2, n_short=3, ell=6, seed=0):
    """Several branches merge into one lane; short-haul agents are scheduled
    before long-haul ones. Letting the long-haul agents go first lowers the
    makespan while the Sum-of-Costs is unchanged (Example ex:merge)."""
    rng = random.Random(seed)
    k = n_long + n_short
    names = {}

    def node(x):
        if x not in names:
            names[x] = len(names)
        return names[x]

    arcs = []
    starts_nodes = [node(("s", i)) for i in range(k)]
    c, d = node("c"), node("d")
    for s in starts_nodes:
        arcs.append((s, c))
    arcs.append((c, d))
    goals_nodes = []
    lengths = []
    order = list(range(k))
    rng.shuffle(order)
    long_set = set(order[:n_long])
    for i in range(k):
        L = ell if i in long_set else 1
        prev = d
        for j in range(L):
            v = node(("p", i, j))
            arcs.append((prev, v))
            prev = v
        goals_nodes.append(prev)
        lengths.append(L)
    r = [node(("r", j)) for j in range(ell + k + 4)]
    arcs += list(zip(r[:-1], r[1:]))
    for g in goals_nodes:
        arcs.append((g, r[0]))
    for s in starts_nodes:
        arcs.append((r[-1], s))
    G = Graph(len(names), arcs, name="convoy")
    # initial plan: short agents enter c first, then the long ones
    entry = sorted(range(k), key=lambda i: (lengths[i], i))
    paths = [None] * k
    for t, i in enumerate(entry):
        route = [c, d] + [names[("p", i, j)] for j in range(lengths[i])]
        paths[i] = [starts_nodes[i]] * (t + 1) + route
    T = max(len(p) for p in paths) - 1
    plan = [tuple(p[min(t, len(p) - 1)] for p in paths) for t in range(T + 1)]
    return Instance(f"convoy_l{n_long}_s{n_short}_ell{ell}_s{seed}", "convoy", G,
                    tuple(starts_nodes), tuple(goals_nodes),
                    {"n_long": n_long, "n_short": n_short, "ell": ell, "seed": seed}, plan)


# ---------------------------------------------------------------------------
# Dense instances with a known feasible plan (backward random walk)
# ---------------------------------------------------------------------------
def random_walk_instance(G, n, steps=40, seed=0, move_prob=0.7, rotations=True,
                         family="dense", name=None):
    """Place n agents at random goals and walk backwards for `steps` joint
    steps on the reversed graph; reversing the walk gives a feasible (and
    usually very wasteful) plan from the resulting start configuration.

    In each step agents, in random order, move to a node that is free at
    that moment; with rotations=True, a fully occupied directed cycle found
    by following the agents may also rotate as a whole.
    """
    rng = random.Random(seed)
    goals = tuple(rng.sample(range(G.N), n))
    cur = list(goals)
    walk = [tuple(cur)]
    for _ in range(steps):
        occ = {v: k for k, v in enumerate(cur)}
        moved = False
        order = list(range(n))
        rng.shuffle(order)
        new = list(cur)
        taken = set(cur)
        for k in order:
            if rng.random() > move_prob:
                continue
            opts = [w for w in G.inn[cur[k]] if w != cur[k] and w not in taken]
            if opts:
                w = rng.choice(opts)
                taken.discard(cur[k])
                taken.add(w)
                new[k] = w
                moved = True
        if rotations and rng.random() < 0.3:
            # try to rotate a full cycle backwards: follow predecessors
            k0 = rng.randrange(n)
            seq, seen = [k0], {k0}
            while True:
                preds = [occ[w] for w in G.inn[cur[seq[-1]]]
                         if w in occ and occ[w] not in seen and new[occ[w]] == cur[occ[w]]]
                back = [w for w in G.inn[cur[seq[-1]]] if w == cur[k0]]
                if len(seq) >= 3 and back and new[k0] == cur[k0]:
                    # agents seq[0]..seq[-1]: each moves to the node of the next
                    ok = all(new[k] == cur[k] for k in seq)
                    if ok:
                        for a, b in zip(seq, seq[1:] + seq[:1]):
                            new[a] = cur[b]
                        moved = True
                    break
                if not preds or len(seq) > 12:
                    break
                k = rng.choice(preds)
                seq.append(k)
                seen.add(k)
        # keep only a valid joint move (no collision, no swap)
        if moved and len(set(new)) == n and not any(
                new[a] == cur[b] and new[b] == cur[a]
                for a in range(n) for b in range(a + 1, n) if new[a] != cur[a]):
            cur = new
            walk.append(tuple(cur))
    plan = walk[::-1]
    starts = plan[0]
    nm = name or f"{family}_{G.name}_n{n}_s{seed}"
    return Instance(nm, family, G, tuple(starts), goals,
                    {"n": n, "steps": steps, "seed": seed, "density": n / G.N}, plan)


def grid_graph(rows, cols, oneway=False, holes=0, seed=0, name=None):
    """Undirected (or one-way Manhattan) grid, optionally with obstacles."""
    rng = random.Random(seed)
    for _ in range(100):
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        blocked = set(rng.sample(cells, holes)) if holes else set()
        cells = [x for x in cells if x not in blocked]
        idx = {x: i for i, x in enumerate(cells)}
        arcs = []
        for (r, c) in cells:
            for dr, dc in ((0, 1), (1, 0)):
                y = (r + dr, c + dc)
                if y not in idx:
                    continue
                u, v = idx[(r, c)], idx[y]
                if not oneway:
                    arcs += [(u, v), (v, u)]
                elif dr == 0:
                    arcs.append((u, v) if r % 2 == 0 else (v, u))
                else:
                    arcs.append((u, v) if c % 2 == 1 else (v, u))
        G = Graph(len(cells), arcs, coords={i: x for x, i in idx.items()},
                  name=name or f"{'ow' if oneway else 'g'}{rows}x{cols}")
        if G.strongly_connected():
            return G
    raise RuntimeError("no strongly connected grid")


def grid_rings(widths=(4, 4), height=3, shifts=(3, 3), hub_agents=0, seed=0):
    """Fully occupied rings on a 4-connected undirected grid.

    Each ring is the perimeter of a `height x w` rectangle with an obstacle
    interior; rings are separated by obstacle columns and joined to a
    corridor row by a one-cell stub from their bottom-left corner. All the
    ring cells are occupied, so a ring moves only by a joint rotation. The
    initial plan rotates one ring after the other (and, before that, moves
    the optional corridor agents one at a time).
    """
    rng = random.Random(seed)
    cols = sum(widths) + len(widths) + 1
    rows = height + 2
    free, rings = set(), []
    c0 = 1
    for w in widths:
        per = ([(0, c0 + j) for j in range(w)]
               + [(i, c0 + w - 1) for i in range(1, height)]
               + [(height - 1, c0 + j) for j in range(w - 2, -1, -1)]
               + [(i, c0) for i in range(height - 2, 0, -1)])
        rings.append(per)
        free |= set(per)
        free.add((height, c0))          # stub below the bottom-left corner
        c0 += w + 1
    free |= {(height + 1, c) for c in range(cols)}   # corridor
    cells = sorted(free)
    idx = {x: i for i, x in enumerate(cells)}
    arcs = []
    for (r, c) in cells:
        for dr, dc in ((0, 1), (1, 0)):
            y = (r + dr, c + dc)
            if y in idx:
                arcs += [(idx[(r, c)], idx[y]), (idx[y], idx[(r, c)])]
    G = Graph(len(cells), arcs, coords={i: x for x, i in idx.items()}, name="grid_rings")
    starts, goals, ring_ids = [], [], []
    for per, q in zip(rings, shifts):
        ids = [idx[x] for x in per]
        ring_ids.append((ids, q))
        L = len(ids)
        for j in range(L):
            starts.append(ids[j])
            goals.append(ids[(j + q) % L])
    corridor = [idx[(height + 1, c)] for c in range(cols)]
    m = hub_agents
    hub_goal = []
    for j in range(m):
        starts.append(corridor[j])
        hub_goal.append(corridor[cols - m + j])
    goals += hub_goal
    plan = [tuple(starts)]
    cur = list(starts)
    nr = len(starts) - m
    for j in reversed(range(m)):
        pos = corridor.index(cur[nr + j])
        while cur[nr + j] != hub_goal[j]:
            pos += 1
            cur[nr + j] = corridor[pos]
            plan.append(tuple(cur))
    base = 0
    for ids, q in ring_ids:
        L = len(ids)
        where = {v: j for j, v in enumerate(ids)}
        for _ in range(q):
            for j in range(L):
                cur[base + j] = ids[(where[cur[base + j]] + 1) % L]
            plan.append(tuple(cur))
        base += L
    name = "gridrings_" + "_".join(str(2 * (w + height) - 4) for w in widths) + \
        (f"_hub{m}" if m else "") + f"_s{seed}"
    return Instance(name, "grid_rings", G, tuple(starts), tuple(goals),
                    {"widths": widths, "hub": m, "seed": seed}, plan)


# ---------------------------------------------------------------------------
# Map types in the style of the MovingAI MAPF benchmarks (small versions)
# ---------------------------------------------------------------------------
def grid_from_cells(free, name):
    """4-connected undirected grid on the set of free cells (row, col)."""
    cells = sorted(free)
    idx = {x: i for i, x in enumerate(cells)}
    arcs = []
    for (r, c) in cells:
        for dr, dc in ((0, 1), (1, 0)):
            y = (r + dr, c + dc)
            if y in idx:
                arcs += [(idx[(r, c)], idx[y]), (idx[y], idx[(r, c)])]
    return Graph(len(cells), arcs, coords={i: x for x, i in idx.items()}, name=name)


def _largest_component(free):
    free = set(free)
    best = set()
    seen = set()
    for s in free:
        if s in seen:
            continue
        comp, q = {s}, [s]
        while q:
            r, c = q.pop()
            for y in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if y in free and y not in comp:
                    comp.add(y); q.append(y)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    return best


def _map_instance(free, n, seed, family, name):
    rng = random.Random(seed)
    G = grid_from_cells(_largest_component(free), name)
    starts, goals = _random_endpoints(rng, range(G.N), n)
    return Instance(f"{name}_n{n}_s{seed}", family, G, starts, goals,
                    {"n": n, "seed": seed})


def warehouse(rows=13, cols=19, shelf_len=5, n=20, seed=0, family="warehouse"):
    """Shelf rows (1 x shelf_len obstacles) separated by one-cell aisles,
    with a free border, as in the warehouse maps of the MovingAI benchmark."""
    free = set()
    for r in range(rows):
        for c in range(cols):
            shelf = (1 <= r < rows - 1 and r % 2 == 1 and 2 <= c < cols - 2
                     and (c - 2) % (shelf_len + 1) != shelf_len)
            if not shelf:
                free.add((r, c))
    return _map_instance(free, n, seed, family, f"warehouse{rows}x{cols}")


def rooms(rows=16, cols=16, room=4, n=16, seed=0, family="rooms"):
    """Square rooms separated by walls, with one random door per wall
    segment (room-32-32-4 style)."""
    rng = random.Random(seed)
    walls = set()
    for r in range(rows):
        for c in range(cols):
            if r % (room + 1) == room or c % (room + 1) == room:
                walls.add((r, c))
    # one door in every wall segment between two rooms
    for r0 in range(0, rows, room + 1):
        for c0 in range(0, cols, room + 1):
            if c0 + room < cols:   # vertical wall to the right of the room
                walls.discard((r0 + rng.randrange(min(room, rows - r0)), c0 + room))
            if r0 + room < rows:   # horizontal wall below the room
                walls.discard((r0 + room, c0 + rng.randrange(min(room, cols - c0))))
    free = {(r, c) for r in range(rows) for c in range(cols)} - walls
    return _map_instance(free, n, seed, family, f"rooms{rows}x{cols}")


def maze(rows=15, cols=15, braid=0.15, n=8, seed=0, family="maze"):
    """Corridors of width one from a randomized DFS, with a fraction of
    extra openings so that agents can pass each other (maze-style maps)."""
    rng = random.Random(seed)
    free = set()
    start = (1, 1)
    free.add(start)
    stack = [start]
    while stack:
        r, c = stack[-1]
        nb = [(r + dr, c + dc) for dr, dc in ((0, 2), (2, 0), (0, -2), (-2, 0))
              if 0 < r + dr < rows - 1 and 0 < c + dc < cols - 1 and (r + dr, c + dc) not in free]
        if not nb:
            stack.pop()
            continue
        y = rng.choice(nb)
        free.add(((r + y[0]) // 2, (c + y[1]) // 2))
        free.add(y)
        stack.append(y)
    walls = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1) if (r, c) not in free]
    for x in rng.sample(walls, int(braid * len(walls))):
        free.add(x)
    return _map_instance(free, n, seed, family, f"maze{rows}x{cols}")


def random_obstacles(rows=16, cols=16, ratio=0.2, n=24, seed=0, family="random_map"):
    """Random obstacles (random-32-32-20 style)."""
    rng = random.Random(seed)
    free = {(r, c) for r in range(rows) for c in range(cols) if rng.random() >= ratio}
    return _map_instance(free, n, seed, family, f"random{rows}x{cols}")


def pairs_instance(n_pairs=4, T=8):
    """Lower-bound instance for the free-window neighborhood: n_pairs agents
    each on its own pair a_k <-> b_k (out-degree 1 plus the wait), waiting in
    the reference, and one agent walking along a path of length T. In a
    window of 2 rounds every pair agent may go to b_k and back independently,
    so the neighborhood contains 2^n_pairs intermediate configurations."""
    arcs = []
    a = list(range(0, 2 * n_pairs, 2))
    b = [x + 1 for x in a]
    for x, y in zip(a, b):
        arcs += [(x, y), (y, x)]
    path = list(range(2 * n_pairs, 2 * n_pairs + T + 1))
    arcs += list(zip(path[:-1], path[1:]))
    spare = path[-1] + 1
    arcs += [(path[-1], spare), (spare, path[0])]
    G = Graph(spare + 1, arcs, name="pairs")
    starts = tuple(a) + (path[0],)
    goals = tuple(a) + (path[-1],)
    plan = [tuple(a) + (path[t],) for t in range(T + 1)]
    return Instance(f"pairs_{n_pairs}_T{T}", "pairs", G, starts, goals,
                    {"n_pairs": n_pairs, "T": T}, plan)
