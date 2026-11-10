# SPDX-License-Identifier: GPL-3.0-or-later
"""Local runner: jobs, parallel execution, resume, timeouts, downloader."""
import csv
import io
import json
import os
import zipfile
from collections import Counter

import pytest

from mapf_bench import local_runner as L
from mapf_bench.cpp_local_search import available as cpp_available

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needs_cpp = pytest.mark.skipif(not cpp_available(), reason="C++ local search not built")


def _cfg(tmp_path, **kw):
    cfg = {"name": "t", "time": 1.0, "methods": ["LS", "LNS"],
           "suites": [{"type": "paper", "groups": ["S", "D"], "indices": [0, 15]}]}
    cfg.update(kw)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return L.load_config(str(p))


def test_jobs_of_the_configurations():
    for name, n in (("smoke", 6), ("paper", 509), ("dense", 75), ("reduced", 96), ("deadlock", 40), ("corridors", 72), ("pibt", 201)):
        js = L.jobs(L.load_config(os.path.join(ROOT, "configs", f"{name}.json")))
        assert len(js) == n and len({j for j, _ in js}) == n
    js = L.jobs(L.load_config(os.path.join(ROOT, "configs", "movingai_official.json")))
    assert len(js) == 6 * 3 * 5 + 2 * 2 * 3


@needs_cpp
def test_parallel_run_resume_and_analysis(tmp_path, monkeypatch):
    from mapf_bench.official_lns import available
    if not available():         # LNS on the grid instance: re-implementation
        monkeypatch.setenv("MAPF_BENCH_LNS_FALLBACK", "1")
    cfg = _cfg(tmp_path)
    out = str(tmp_path / "res")
    L.run(cfg, out, workers=2)
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    assert Counter(r["job"] for r in rows) == {"paper:0": 2, "paper:15": 2}
    assert all(r["soc"] and not r["error"] for r in rows)
    # resume: nothing to do, no duplicates
    L.run(cfg, out, workers=2)
    assert len(list(csv.DictReader(open(os.path.join(out, "results.csv"))))) == 4
    # an interrupted job: the finished method is not repeated
    open(os.path.join(out, "jobs_done.txt"), "w").write("paper:0\n")
    cfg["methods"] = ["LS", "LNS", "Hybrid"]
    L.run(cfg, out, workers=1)
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    c = Counter((r["job"], r["method"]) for r in rows)
    assert c[("paper:15", "LS")] == 1 and c[("paper:15", "Hybrid")] == 1
    from mapf_bench.paper_suite import analyze
    text = analyze(out, cfg["time"])
    assert "Pairwise comparisons" in text and os.path.exists(os.path.join(out, "tables.tex"))


@needs_cpp
def test_timeout_is_recorded(tmp_path):
    cfg = _cfg(tmp_path, methods=["LNS"], timeout_factor=0.0, timeout_slack=0.5,
               suites=[{"type": "paper", "groups": ["D"], "indices": [15]}])
    out = str(tmp_path / "res")
    L.run(cfg, out, workers=1)
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    assert len(rows) == 1 and rows[0]["error"].startswith("timeout")


def test_downloader_with_local_archives(tmp_path):
    from mapf_bench.movingai_download import download
    maps = io.BytesIO()
    with zipfile.ZipFile(maps, "w") as z:
        z.writestr("mapf-map/tiny-4-4.map", "type octile\nheight 4\nwidth 4\nmap\n....\n....\n....\n....\n")
        z.writestr("mapf-map/other.map", "type octile\nheight 1\nwidth 1\nmap\n.\n")
    scens = io.BytesIO()
    with zipfile.ZipFile(scens, "w") as z:
        z.writestr("scen-random/tiny-4-4-random-1.scen",
                   "version 1\n0\ttiny-4-4.map\t4\t4\t0\t0\t3\t3\t6\n")
    (tmp_path / "m.zip").write_bytes(maps.getvalue())
    (tmp_path / "s.zip").write_bytes(scens.getvalue())
    nm, ns = download({"tiny-4-4"}, map_zip=str(tmp_path / "m.zip"),
                      scen_zip=str(tmp_path / "s.zip"), root=str(tmp_path / "bench"))
    assert (nm, ns) == (1, 1)
    assert sorted(os.listdir(tmp_path / "bench" / "maps")) == ["tiny-4-4.map"]
    assert os.listdir(tmp_path / "bench" / "scens") == ["tiny-4-4-random-1.scen"]


def test_doctor_runs():
    assert L.doctor() in (True, False)


@needs_cpp
def test_retry_errors_reruns_only_the_failed_methods(tmp_path, monkeypatch):
    """A run without the official MAPF-LNS records errors for LNS on grids;
    --retry-errors runs only those methods again and keeps the others."""
    cfg = _cfg(tmp_path, methods=["LS", "LNS"],
               suites=[{"type": "paper", "groups": ["S"], "indices": [0]}])   # a grid instance
    out = str(tmp_path / "res")
    monkeypatch.setenv("MAPF_LNS_BIN", "/nonexistent")
    monkeypatch.delenv("MAPF_BENCH_LNS_FALLBACK", raising=False)
    L.run(cfg, out, workers=1)
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    assert {r["method"]: bool(r["error"]) for r in rows} == {"LS": False, "LNS": True}
    ls_time = [r["time"] for r in rows if r["method"] == "LS"]
    L.run(cfg, out, workers=1)                               # plain resume: nothing to do
    assert len(list(csv.DictReader(open(os.path.join(out, "results.csv"))))) == 2
    monkeypatch.setenv("MAPF_BENCH_LNS_FALLBACK", "1")       # now LNS can run
    L.run(cfg, out, workers=1, retry=True)
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    assert sorted(r["method"] for r in rows) == ["LNS", "LS"]
    assert all(r["soc"] and not r["error"] for r in rows)
    assert [r["time"] for r in rows if r["method"] == "LS"] == ls_time   # LS was not rerun


def test_group_memory_measures_this_process():
    mem = L.group_memory(os.getpgid(0))
    assert mem is None or mem > 10 * 2 ** 20


@needs_cpp
def test_memory_limit_kills_the_job_and_is_recorded(tmp_path):
    """The limit is enforced from the main process (portable: no setrlimit)."""
    cfg = _cfg(tmp_path, methods=["LNS"], memory_gb=0.01,         # 10 MB: below any Python process
               suites=[{"type": "paper", "groups": ["D"], "indices": [15]}])   # LNS runs the whole budget
    out = str(tmp_path / "res")
    L.run(cfg, out, workers=1)
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    if L.group_memory(os.getpgid(0)) is None:
        pytest.skip("ps not available")
    assert len(rows) == 1 and rows[0]["error"].startswith("memory limit exceeded")
    from mapf_bench.paper_suite import analyze
    assert "No successful results" in analyze(out, 1.0)          # no crash without results
    # with a reasonable limit the same job runs, and --retry-errors recovers it
    cfg["memory_gb"] = 8
    L.run(cfg, out, workers=1, retry=True)
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    assert len(rows) == 1 and rows[0]["soc"] and not rows[0]["error"]


@needs_cpp
def test_methods_added_to_the_configuration_run_on_finished_jobs(tmp_path):
    cfg = _cfg(tmp_path, methods=["LS"], suites=[{"type": "paper", "groups": ["S"], "indices": [5]}])
    out = str(tmp_path / "res")
    L.run(cfg, out, workers=1)
    t_ls = [r["time"] for r in csv.DictReader(open(os.path.join(out, "results.csv")))]
    cfg["methods"] = ["LS", "LNS"]
    L.run(cfg, out, workers=1)
    rows = list(csv.DictReader(open(os.path.join(out, "results.csv"))))
    assert sorted(r["method"] for r in rows) == ["LNS", "LS"]
    assert [r["time"] for r in rows if r["method"] == "LS"] == t_ls        # LS was not rerun
