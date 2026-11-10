# SPDX-License-Identifier: GPL-3.0-or-later
"""Grids with many corridors at high density (group C)."""
import pytest

from mapf_bench import corridors as C
from mapf_bench.movingai import load_map, load_scen


@pytest.mark.parametrize("fam,args", C.CONFIGS)
def test_maps_connected_with_corridors(fam, args):
    cells = C.cells_of(fam, args, 1)
    # connected
    start = next(iter(cells)); seen = {start}; stack = [start]
    while stack:
        r, c = stack.pop()
        for y in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if y in cells and y not in seen: seen.add(y); stack.append(y)
    assert seen == cells
    # mostly corridors: most free cells have at most two free neighbours
    deg2 = sum(sum((r + a, c + b) in cells for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))) <= 2 for r, c in cells)
    assert deg2 >= 0.5 * len(cells)


@pytest.mark.parametrize("occ", C.OCCUPANCY + C.OCCUPANCY_HIGH)
def test_instances_occupancy_and_determinism(occ):
    a = C.generate("city", (4, 2), occ, 1); b = C.generate("city", (4, 2), occ, 1)
    assert a.starts == b.starts and a.goals == b.goals
    assert len(set(a.starts)) == a.n == len(set(a.goals))
    assert abs(a.n / a.graph.N - occ) < 0.02


def test_movingai_round_trip(tmp_path):
    inst = C.generate("warehouse", (4, 8), 0.5, 1)
    mp = C.write_movingai(inst, root=str(tmp_path))
    h, w, free = load_map(mp)
    assert free == set(map(tuple, inst.params["cells"]))
    assert load_scen(mp[:-4] + ".scen") == [tuple(map(tuple, p)) for p in inst.params["pairs"]]


def test_first_block_indices_unchanged():
    """The 90-98% instances come after the first 36, whose job indices are kept."""
    from mapf_bench.paper_suite import scenarios
    sc = [i for i, (g, _, _) in enumerate(scenarios()) if g == "C"]
    assert sc == list(range(sc[0], sc[0] + 72))
