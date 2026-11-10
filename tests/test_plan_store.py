# SPDX-License-Identifier: GPL-3.0-or-later
"""Compact plan storage: exact round trip, old files, conversion, size."""
import gzip
import json
import os

import pytest

from mapf_bench import plan_store as S
from mapf_bench.instances import grid_bottleneck, oneway_grid, random_digraph, full_rings, ring
from mapf_bench.plan import paths_to_plan, serialize, trim
from mapf_bench.pp import initial_pp


def _instances():
    out = []
    for inst in (grid_bottleneck(9, 9, K=2, n=8, seed=1), oneway_grid(6, 6, n=10, seed=2),
                 random_digraph(30, 3, n=8, seed=3)):
        p = paths_to_plan(initial_pp(inst.graph, inst.starts, inst.goals, seed=0))
        out.append((inst, {"initial": trim(serialize(inst.graph, p)), "PP": trim(p)}))
    for inst in (full_rings((7, 8), (3, 4)), ring(6, 8, 3)):
        out.append((inst, {"initial": trim(inst.initial_plan)}))
    from mapf_bench import movingai as M
    inst = M.instance("random-32-32-20", M.scenarios("random-32-32-20", 1)[0], 30, initial="pp")
    out.append((inst, {"initial": trim(inst.initial_plan)}))
    return out


@pytest.mark.parametrize("k", range(6))
def test_round_trip_is_exact(tmp_path, k):
    inst, plans = _instances()[k]
    path = str(tmp_path / ("a" + S.EXT))
    for label, plan in plans.items():
        S.save_plan(path, inst, label, plan, {"job": "x:1", "group": "T"})
    d = S.load(path)
    ref = S.instance_dict(inst, {"job": "x:1", "group": "T"})
    assert sorted(map(tuple, d["instance"]["arcs"])) == sorted(map(tuple, ref["arcs"]))
    for key in ("name", "family", "n", "N", "starts", "goals", "coords", "job", "group"):
        assert d["instance"][key] == ref[key], key
    assert d["plans"] == {m: [list(x) for x in p] for m, p in plans.items()}


def test_grids_are_stored_as_bitmaps(tmp_path):
    inst, plans = _instances()[0]                     # undirected grid with obstacles
    S.save_plan(str(tmp_path / ("g" + S.EXT)), inst, "initial", plans["initial"])
    import lzma
    _, z = S._unpack(lzma.decompress(open(tmp_path / ("g" + S.EXT), "rb").read()))
    assert "grid" in z and "arc_u" not in z
    inst, plans = _instances()[1]                     # one-way grid: arcs kept
    S.save_plan(str(tmp_path / ("o" + S.EXT)), inst, "initial", plans["initial"])
    _, z = S._unpack(lzma.decompress(open(tmp_path / ("o" + S.EXT), "rb").read()))
    assert "arc_u" in z and "coords" in z


def _write_v1(path, inst, plans):
    with gzip.open(path, "wt") as fh:
        json.dump({"instance": S.instance_dict(inst, {"job": "paper:7"}),
                   "plans": {m: [list(x) for x in p] for m, p in plans.items()}}, fh)


def test_old_files_are_read_and_converted(tmp_path):
    res = tmp_path / "res"
    (res / "plans").mkdir(parents=True)
    total_old = 0
    for i, (inst, plans) in enumerate(_instances()):
        f = str(res / "plans" / f"paper_{i}{S.OLD_EXT}")
        _write_v1(f, inst, plans)
        total_old += os.path.getsize(f)
        assert S.load(f)["plans"]["initial"] == [list(x) for x in plans["initial"]]
    n, before, after = S.convert(str(res))
    assert n == 6 and before == total_old and after < 0.5 * before
    files = S.plan_files(str(res))
    assert len(files) == 6 and all(f.endswith(S.EXT) for f in files)


def test_saving_into_an_old_file_converts_it(tmp_path):
    inst, plans = _instances()[2]
    path = S.plans_path(str(tmp_path), "paper:9")
    os.makedirs(os.path.dirname(path))
    _write_v1(path[: -len(S.EXT)] + S.OLD_EXT, inst, {"initial": plans["initial"]})
    S.save_plan(path, inst, "PP", plans["PP"])
    assert not os.path.exists(path[: -len(S.EXT)] + S.OLD_EXT)
    assert set(S.load(path)["plans"]) == {"initial", "PP"}
