# SPDX-License-Identifier: GPL-3.0-or-later
import csv
import os

from mapf_bench.run import main


def test_runner_quick(tmp_path):
    out = tmp_path / "res"
    main(["--suite", "examples", "--seeds", "1", "--time-limit", "20",
          "--lns-iters", "100", "--out", str(out),
          "--algos", "lns_soc", "ls_gc", "lns_soc+ls_new_alt"])
    rows = list(csv.DictReader(open(out / "results.csv")))
    assert rows and all(r["valid"] == "True" for r in rows)
    assert (out / "summary.md").exists()
