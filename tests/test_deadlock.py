# SPDX-License-Identifier: GPL-3.0-or-later
"""Deadlock-prone instances: structure, determinism, MovingAI files."""
import os

import pytest

from mapf_bench import deadlock as K
from mapf_bench.movingai import load_map, load_scen


@pytest.mark.parametrize("fam,gen,args", K.CONFIGS)
def test_instances_are_valid_and_deterministic(fam, gen, args):
    a, b = gen(*args, 1), gen(*args, 1)
    assert a.starts == b.starts and a.goals == b.goals and a.graph.N == b.graph.N
    assert len(set(a.starts)) == a.n and len(set(a.goals)) == a.n
    assert a.graph.strongly_connected() and a.graph.N >= a.n + 1


def test_family_structure():
    i = K.reversal(6, 3, 1)                      # goals are the reversed queue
    cs = [i.graph.coords[v] for v in i.starts]; cg = [i.graph.coords[v] for v in i.goals]
    assert sorted(cs) == sorted(cg) and all(s != g for s, g in zip(cs, cg))
    d = K.dead_end(12, 0, 1)                    # all corridor cells are goals
    corridor = {(2, 5 + j) for j in range(12)}
    assert {d.graph.coords[v] for v in d.goals} == corridor
    c = K.corridor_swap(10, 1, 3, 1)            # half the agents go each way
    cols = [(c.graph.coords[s][1], c.graph.coords[g][1]) for s, g in zip(c.starts, c.goals)]
    assert sum(sc < 3 and gc > 12 for sc, gc in cols) == 3 and sum(sc > 12 and gc < 3 for sc, gc in cols) == 3


def test_movingai_files_round_trip(tmp_path):
    inst = K.crossroads(6, 2, 1)
    mp = K.write_movingai(inst, root=str(tmp_path))
    h, w, free = load_map(mp)
    assert free == set(map(tuple, inst.params["cells"]))
    pairs = load_scen(mp[:-4] + ".scen")
    assert pairs == [tuple(map(tuple, p)) for p in inst.params["pairs"]]
