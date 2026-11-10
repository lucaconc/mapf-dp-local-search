# SPDX-License-Identifier: GPL-3.0-or-later
"""Grid scenarios with many corridors and high density (group C).

Three families of 4-connected maps whose free cells are mostly width-1
corridors:

  warehouse(rows, length)  rows of shelves (obstacles) of the given length
      separated by width-1 aisles, joined by cross aisles at both ends and in
      the middle, as in robotic warehouses;
  city(blocks, size)       a lattice of square obstacle blocks separated by
      width-1 streets: long corridors crossing at regular intersections;
  braid(cells, loops)      a maze with width-1 corridors (randomized depth-first
      search on a lattice of cells) in which a fraction of the dead ends is
      opened, creating loops (without loops, dense instances would be
      unsolvable: agents could pass each other only at branching cells).

Agents: a fraction (the occupancy) of the free cells, random distinct starts
and goals. Every map is connected by construction. Initial plans: the first
LaCAM3 solution (instances without one are excluded by the protocol).
"""
import os
import random

from .instances import Instance, grid_from_cells

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "corridors")


def warehouse(rows, length):
    """rows shelves of `length` cells, aisles of width 1 between and around them,
    cross aisles at both ends and in the middle of the shelves."""
    mid = length // 2 + 1
    w = length + 3                       # left aisle, shelf, middle cross aisle, right aisle
    h = 2 * rows + 1
    cells = set()
    for r in range(h):
        for c in range(w):
            shelf_row = r % 2 == 1
            shelf_col = 1 <= c <= length + 1 and c != mid
            if not (shelf_row and shelf_col):
                cells.add((r, c))
    return cells


def city(blocks, size):
    """blocks x blocks square obstacle blocks of side `size`, streets of width 1."""
    n = blocks * (size + 1) + 1
    return {(r, c) for r in range(n) for c in range(n) if r % (size + 1) == 0 or c % (size + 1) == 0}


def braid(cells, loops, rng):
    """Maze on a cells x cells lattice (corridors of width 1), then a fraction
    `loops` of the dead ends opened towards a neighbouring cell."""
    n = 2 * cells + 1
    free = set()
    seen = set()
    stack = [(0, 0)]
    seen.add((0, 0)); free.add((1, 1))
    while stack:
        x, y = stack[-1]
        nb = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= x + dx < cells and 0 <= y + dy < cells and (x + dx, y + dy) not in seen]
        if not nb:
            stack.pop(); continue
        nx, ny = rng.choice(nb)
        free.add((2 * nx + 1, 2 * ny + 1)); free.add((x + nx + 1, y + ny + 1))
        seen.add((nx, ny)); stack.append((nx, ny))
    def degree(c):
        r, q = c
        return sum((r + a, q + b) in free for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    dead = [(2 * x + 1, 2 * y + 1) for x in range(cells) for y in range(cells) if degree((2 * x + 1, 2 * y + 1)) == 1]
    rng.shuffle(dead)
    for (r, q) in dead[:int(min(1.0, loops) * len(dead))]:
        walls = [(r + a, q + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
                 if 0 < r + 2 * a < n and 0 < q + 2 * b < n and (r + a, q + b) not in free]
        if walls:
            free.add(rng.choice(walls))
    # loops > 1: also open that fraction (minus one) of the remaining walls
    # between adjacent corridor cells, for more crossings
    if loops > 1:
        walls = [(r, q) for r in range(1, n - 1) for q in range(1, n - 1) if (r, q) not in free
                 and ((r % 2 == 1 and q % 2 == 0) or (r % 2 == 0 and q % 2 == 1))]
        rng.shuffle(walls)
        for c in walls[:int((loops - 1) * len(walls))]:
            free.add(c)
    return free


# configurations of the protocol: (family, builder parameters)
CONFIGS = ([("warehouse", (r, l)) for r, l in ((4, 8), (6, 12))]
           + [("city", (b, s)) for b, s in ((4, 2), (5, 3))]
           + [("braid", (c, 1.3)) for c in (6, 9)])
OCCUPANCY = (0.5, 0.7, 0.85)
# very high occupancy, after the instances above (their job indices are kept);
# LaCAM3 finds no first solution on some of them, which the protocol excludes
OCCUPANCY_HIGH = (0.9, 0.95, 0.98)
SEEDS = (1, 2)


def cells_of(fam, args, seed):
    if fam == "warehouse":
        return warehouse(*args)
    if fam == "city":
        return city(*args)
    return braid(args[0], args[1], random.Random(f"braid/{args}/{seed}"))


def generate(fam, args, occ, seed):
    cells = cells_of(fam, args, seed)
    rng = random.Random(f"corridors/{fam}/{args}/{occ}/{seed}")
    free = sorted(cells)
    n = max(2, min(len(free) - 1, round(occ * len(free))))
    starts = rng.sample(free, n); goals = rng.sample(free, n)
    tag = "_".join(map(str, args)).replace(".", "")
    name = f"{fam}_{tag}_o{int(round(100 * occ))}_s{seed}"
    G = grid_from_cells(cells, name)
    idx = {rc: v for v, rc in G.coords.items()}
    return Instance(name, f"{fam}_{tag}", G, tuple(idx[s] for s in starts), tuple(idx[g] for g in goals),
                    {"cells": free, "pairs": list(zip(starts, goals)), "occupancy": occ})


def write_movingai(inst, root=ROOT):
    cells = set(map(tuple, inst.params["cells"]))
    h = max(r for r, _ in cells) + 1; w = max(c for _, c in cells) + 1
    os.makedirs(root, exist_ok=True)
    mp = os.path.join(root, inst.name + ".map")
    with open(mp, "w") as fh:
        fh.write(f"type octile\nheight {h}\nwidth {w}\nmap\n")
        for r in range(h):
            fh.write("".join("." if (r, c) in cells else "@" for c in range(w)) + "\n")
    with open(os.path.join(root, inst.name + ".scen"), "w") as fh:
        fh.write("version 1\n")
        for s, g in inst.params["pairs"]:
            fh.write(f"0\t{inst.name}.map\t{w}\t{h}\t{s[1]}\t{s[0]}\t{g[1]}\t{g[0]}\t0\n")
    return mp


def instance(fam, args, occ, seed, initial="lacam"):
    inst = generate(fam, args, occ, seed)
    write_movingai(inst)
    inst.initial_plan = None
    if initial == "lacam":
        from .lacam import lacam_solve
        from .plan import trim
        p = lacam_solve(inst, time_limit=30, anytime=False)
        inst.initial_plan = trim(p) if p is not None else None
    return inst
