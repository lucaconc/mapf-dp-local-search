# SPDX-License-Identifier: GPL-3.0-or-later
"""Deadlock-prone MAPF instances on 4-connected grids.

Four families, in which agents block each other unless they give way in the
right order or wait in a side space -- the situations in which prioritized
planning, and hence the repair of MAPF-LNS, tends to fail:

  corridor_swap(L, bays, m)  two 3x3 rooms joined by a corridor of length L
      with `bays` single-cell side bays; m agents in each room must reach the
      other room, so agents moving in opposite directions can pass each other
      only at the bays;
  dead_end(L, extra)         a 5x5 room with a dead-end corridor of length L
      whose cells are all goals, the agents starting in the room in random
      order (the deepest goal must be entered first), plus `extra` agents with
      start and goal in the room;
  reversal(m, pocket)        m agents in a dead-end corridor must reverse
      their order, with two side pockets of length `pocket` at the junction
      as the only buffer;
  crossroads(arm, k)         four single-width arms of length `arm` meeting at
      one cell; k agents in each arm must reach the opposite arm.

Every instance is deterministic in its seed, and its map and scenario can be
written in the MovingAI format (write_movingai). Solvability is not
guaranteed by construction: the protocol uses the first LaCAM3 solution as
initial plan and excludes the instances without one.
"""
import os
import random

from .instances import Instance, grid_from_cells

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "deadlock")


def _instance(name, family, cells, pairs, params):
    G = grid_from_cells(cells, name)
    idx = {rc: v for v, rc in G.coords.items()}
    starts = tuple(idx[s] for s, _ in pairs)
    goals = tuple(idx[g] for _, g in pairs)
    assert len(set(starts)) == len(starts) and len(set(goals)) == len(goals)
    return Instance(name, family, G, starts, goals, dict(params, cells=sorted(cells), pairs=pairs))


def corridor_swap(L, bays, m, seed):
    rng = random.Random(f"corridor_swap/{L}/{bays}/{m}/{seed}")
    left = {(r, c) for r in range(3) for c in range(3)}
    right = {(r, c) for r in range(3) for c in range(L + 3, L + 6)}
    corridor = {(1, c) for c in range(3, L + 3)}
    inner = list(range(4, L + 2))                      # not next to the rooms
    bay_cols = rng.sample(inner, min(bays, len(inner)))
    bay = {(rng.choice((0, 2)), c) for c in bay_cols}
    cells = left | right | corridor | bay
    ls, lg = rng.sample(sorted(left), m), rng.sample(sorted(right), m)    # left -> right
    rs, rg = rng.sample(sorted(right), m), rng.sample(sorted(left), m)    # right -> left
    pairs = list(zip(ls, lg)) + list(zip(rs, rg))
    return _instance(f"corridor_swap_L{L}_b{bays}_m{m}_s{seed}", f"corridor_swap_L{L}_b{bays}_m{m}",
                     cells, pairs, {"L": L, "bays": bays, "m": m, "seed": seed})


def dead_end(L, extra, seed):
    rng = random.Random(f"dead_end/{L}/{extra}/{seed}")
    room = {(r, c) for r in range(5) for c in range(5)}
    corridor = [(2, 5 + j) for j in range(L)]          # (2, 5) is the entrance
    cells = room | set(corridor)
    starts = rng.sample(sorted(room), L + extra)
    goals_corr = corridor[:]
    rng.shuffle(goals_corr)                            # random order of the starts
    goals_room = rng.sample(sorted(room), extra)
    pairs = list(zip(starts[:L], goals_corr)) + list(zip(starts[L:], goals_room))
    return _instance(f"dead_end_L{L}_x{extra}_s{seed}", f"dead_end_L{L}_x{extra}",
                     cells, pairs, {"L": L, "extra": extra, "seed": seed})


def reversal(m, pocket, seed):
    rng = random.Random(f"reversal/{m}/{pocket}/{seed}")
    r0 = pocket                                        # rows >= 0
    corridor = [(r0, c) for c in range(m + 1)]         # (r0, 0) is the junction
    up = [(r0 - j, 0) for j in range(1, pocket + 1)]
    down = [(r0 + j, 0) for j in range(1, pocket + 1)]
    cells = set(corridor) | set(up) | set(down)
    queue = corridor[1:]                               # agents in the corridor
    order = list(range(m))
    rng.shuffle(order)                                 # which agent is where (labels only)
    pairs = [(queue[i], queue[m - 1 - i]) for i in order]
    return _instance(f"reversal_m{m}_p{pocket}_s{seed}", f"reversal_m{m}_p{pocket}",
                     cells, pairs, {"m": m, "pocket": pocket, "seed": seed})


def crossroads(arm, k, seed):
    rng = random.Random(f"crossroads/{arm}/{k}/{seed}")
    c0 = arm
    arms = {"N": [(c0 - j, c0) for j in range(1, arm + 1)], "S": [(c0 + j, c0) for j in range(1, arm + 1)],
            "W": [(c0, c0 - j) for j in range(1, arm + 1)], "E": [(c0, c0 + j) for j in range(1, arm + 1)]}
    cells = {(c0, c0)} | {x for a in arms.values() for x in a}
    opposite = {"N": "S", "S": "N", "W": "E", "E": "W"}
    pairs = []
    for a in "NSWE":
        starts = rng.sample(arms[a], k)
        goals = rng.sample(arms[opposite[a]], k)
        pairs += list(zip(starts, goals))
    return _instance(f"crossroads_a{arm}_k{k}_s{seed}", f"crossroads_a{arm}_k{k}",
                     cells, pairs, {"arm": arm, "k": k, "seed": seed})


# configurations of the protocol (group K): (family, generator, parameters)
CONFIGS = ([("corridor_swap", corridor_swap, (L, b, m)) for L in (10, 20) for b in (1, 3) for m in (3, 6)]
           + [("dead_end", dead_end, a) for a in ((6, 0), (6, 6), (12, 0), (12, 3))]
           + [("reversal", reversal, a) for a in ((4, 2), (4, 3), (6, 3), (6, 4))]
           + [("crossroads", crossroads, (a, k)) for a in (6, 10) for k in (2, 4)])
SEEDS = (1, 2)


def write_movingai(inst, root=ROOT):
    """Write the map and the scenario of an instance in the MovingAI format."""
    cells = set(map(tuple, inst.params["cells"]))
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    os.makedirs(root, exist_ok=True)
    mp = os.path.join(root, inst.name + ".map")
    with open(mp, "w") as fh:
        fh.write(f"type octile\nheight {h}\nwidth {w}\nmap\n")
        for r in range(h):
            fh.write("".join("." if (r, c) in cells else "@" for c in range(w)) + "\n")
    with open(os.path.join(root, inst.name + ".scen"), "w") as fh:
        fh.write("version 1\n")
        for i, (s, g) in enumerate(inst.params["pairs"]):
            fh.write(f"0\t{inst.name}.map\t{w}\t{h}\t{s[1]}\t{s[0]}\t{g[1]}\t{g[0]}\t0\n")
    return mp


def instance(fam, gen, args, seed, initial="lacam"):
    """An instance of the protocol, written in MovingAI format, with the first
    LaCAM3 solution as initial plan (None if LaCAM3 finds none in 30 s)."""
    inst = gen(*args, seed)
    write_movingai(inst)
    if initial == "lacam":
        from .lacam import lacam_solve
        from .plan import trim
        p = lacam_solve(inst, time_limit=30, anytime=False)
        inst.initial_plan = trim(p) if p is not None else None
    return inst
