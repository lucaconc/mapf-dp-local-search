# SPDX-License-Identifier: GPL-3.0-or-later
"""Saved plans and the animation / comparison tool."""
import json
import os

import numpy as np
import pytest

from mapf_bench import local_runner as L
from mapf_bench import visualize as V
from mapf_bench.cpp_local_search import available as cpp_available

needs_cpp = pytest.mark.skipif(not cpp_available(), reason="C++ local search not built")


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """A small run: a grid instance and a directed one, with saved plans."""
    tmp = tmp_path_factory.mktemp("viz")
    cfg = {"name": "viz", "time": 1.0, "methods": ["LS"],
           "suites": [{"type": "paper", "groups": ["S", "D"], "indices": [0, 5]}]}
    p = tmp / "cfg.json"
    p.write_text(json.dumps(cfg))
    out = str(tmp / "res")
    L.run(L.load_config(str(p)), out, workers=1)
    return out


@needs_cpp
def test_plans_are_saved_and_valid(results):
    from mapf_bench.graph import Graph
    from mapf_bench.plan import validate
    files = V.plan_files(results)
    assert len(files) == 2
    for f in files:
        d = V.load(f)
        i = d["instance"]
        assert set(d["plans"]) == {"initial", "LS"} and i["job"].startswith("paper:")
        G = Graph(i["N"], [tuple(a) for a in i["arcs"]])
        for plan in d["plans"].values():
            validate(G, tuple(i["starts"]), tuple(i["goals"]), [tuple(x) for x in plan])


@needs_cpp
def test_compare_and_layout(results):
    d = V.load(V.find(results, "paper:5"))          # full rings on a directed graph
    rows, arr, ref = V.compare(d, ["initial", "LS"], "initial")
    assert ref == "initial" and rows[0]["better"] == 0
    assert rows[1]["soc"] <= rows[0]["soc"] and rows[1]["better"] + rows[1]["equal"] + rows[1]["worse"] == d["instance"]["n"]
    xy, grid = V.layout(d["instance"])
    assert not grid and xy.shape == (d["instance"]["N"], 2) and xy.min() >= 0 and xy.max() <= 1
    assert np.allclose(xy, V.layout(d["instance"])[0])                  # deterministic
    g = V.load(V.find(results, "paper:0"))
    assert V.layout(g["instance"])[1] is True                           # grid coordinates


@needs_cpp
def test_html_page_and_index(results, tmp_path):
    d = V.load(V.find(results, "paper:0"))
    out = V.write_html(d, str(tmp_path / "a.html"), ["initial", "LS"], "initial")
    page = open(out).read()
    assert '"methods":["initial","LS"]' in page and "const D =" in page and "__DATA__" not in page
    V.main([results, "--all"])
    assert os.path.exists(os.path.join(results, "animations", "index.html"))
    assert len([f for f in os.listdir(os.path.join(results, "animations")) if f.endswith(".html")]) == 3


@needs_cpp
def test_gif(results, tmp_path):
    from PIL import Image
    d = V.load(V.find(results, "paper:0"))
    out = V.write_gif(d, str(tmp_path / "a.gif"), ["initial", "LS"], substeps=2, max_frames=20)
    im = Image.open(out)
    assert im.n_frames > 2 and im.size[0] > im.size[1]
