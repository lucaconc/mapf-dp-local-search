# SPDX-License-Identifier: GPL-3.0-or-later
"""PIBT (rule-based initial plans) and group P."""
import pytest

from mapf_bench.pibt import pibt
from mapf_bench.plan import validate


def test_pibt_valid_plan_on_an_open_map():
    from mapf_bench import movingai as M
    inst = M.instance("empty-32-32", M.scenarios("empty-32-32", 1)[0], 20)
    p = pibt(inst.graph, inst.starts, inst.goals, seed=1)
    assert p is not None
    validate(inst.graph, inst.starts, inst.goals, p)


def test_pibt_gives_up_on_dense_corridors():
    from mapf_bench import corridors as C
    inst = C.generate("warehouse", (4, 8), 0.85, 1)
    assert pibt(inst.graph, inst.starts, inst.goals, max_steps=500, seed=1) is None


def test_group_p_names_and_methods():
    from mapf_bench.paper_suite import scenarios, skip_method
    sc = [(g, f, b) for g, f, b in scenarios() if g == "P"]
    g, fam, b = next(x for x in sc if x[1].startswith("M:"))
    inst, init = b()
    assert inst.name.endswith("_pibt") and init == "PIBT"
    assert skip_method("LaCAM*+LNS", inst, "P") and not skip_method("LSBX", inst, "P")
    assert not skip_method("LaCAM*+LNS", inst, "M")
