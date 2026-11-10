# SPDX-License-Identifier: GPL-3.0-or-later
"""Automatic paper update: macros, tables, claims, idempotency."""
import csv
import os
import shutil

import pytest

from mapf_bench import paper_update as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "tests", "data", "paper_results")
needs_results = pytest.mark.skipif(not os.path.exists(os.path.join(RESULTS, "results.csv")),
                                   reason="no results to update from")


def _setup(tmp_path, results=RESULTS):
    res = tmp_path / "res"
    shutil.copytree(results, res)
    paper = tmp_path / "paper" / "paper.tex"
    paper.parent.mkdir()
    shutil.copy(os.path.join(ROOT, "tests", "data", "paper_template.tex"), paper)
    return str(res), str(paper)


@needs_results
def test_update_writes_macros_tables_and_report(tmp_path):
    res, paper = _setup(tmp_path)
    r = P.run(res, paper)
    text = open(paper).read()
    assert r"\input{bench_numbers}" in text and "old" not in text
    assert r"\begin{tabular}" in text and "W/T/L" in text
    macros = open(os.path.join(os.path.dirname(paper), "bench_numbers.tex")).read()
    for m in ("BenchNInst", "BenchRatioSLs", "BenchPairAllLacamHybVsLacamP", "BenchHGainSentence"):
        assert "\\" + m + "}" in macros
    assert not r["undefined"]
    # the claims of the rewritten discussion hold on the benchmark results
    assert r["contradicted"] == []
    assert os.path.exists(os.path.join(os.path.dirname(paper), "images", "tests", "performance_profiles.pdf"))
    # the second update with the same results changes nothing
    assert P.run(res, paper)["changed_lines"] == 0


@needs_results
def test_numbers_match_the_analysis(tmp_path):
    res, paper = _setup(tmp_path)
    S = P.statistics(res)
    M = P.macros(S)
    from mapf_bench.paper_suite import analyze
    text = analyze(res, 20)
    line = [l for l in text.split("\n") if l.startswith("| all | LaCAM*+Hybrid vs LaCAM* |")][0]
    assert line.split("|")[4].strip() == "46/10/11" and M["BenchPairAllLacamHybVsLacamW"] == "46"
    assert float(line.split("|")[8]) == pytest.approx(S["pair"][("All", "LaCAM*+Hybrid", "LaCAM*")]["holm"], rel=5e-3)


@needs_results
def test_contradicted_claim_is_reported_and_strict_fails(tmp_path):
    res, paper = _setup(tmp_path)
    rows = list(csv.DictReader(open(os.path.join(res, "results.csv"))))
    for r in rows:              # make LS worse than LNS on the full rings
        if r["family"] == "full_rings" and r["method"] == "LS":
            r["soc"] = str(int(r["soc0"]) + 50)
    with open(os.path.join(res, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    r = P.run(res, paper)
    assert any("group S" in c[1] for c in r["contradicted"])
    assert "CONTRADICTED" in open(r["report"]).read()
    with pytest.raises(SystemExit):
        P.run(res, paper, strict=True)


@needs_results
def test_paper_without_markers_is_refused(tmp_path):
    res, paper = _setup(tmp_path)
    open(paper, "w").write("\\documentclass{article}\\begin{document}x\\end{document}\n")
    with pytest.raises(SystemExit):
        P.run(res, paper)


def test_auto_markers_may_be_indented():
    """Editors that re-indent the paper put tabs before the markers, and
    sometimes spaces after them: the blocks are still found and replaced."""
    from mapf_bench.paper_update import AUTO
    text = "a\n\t\t%<AUTO:bench_results>  \n\t\told\n\t\t%</AUTO:bench_results>\nb\n%<AUTO:bench_pairs>\nx\n%</AUTO:bench_pairs>\n"
    assert {m.group("name") for m in AUTO.finditer(text)} == {"bench_results", "bench_pairs"}
    new = AUTO.sub(lambda m: m.group(1) + "NEW" + m.group(4), text)
    assert "\t\t%<AUTO:bench_results>  \nNEW\n\t\t%</AUTO:bench_results>" in new and "old" not in new


def test_paper_results_regenerate_the_tables(tmp_path):
    """The results of the paper in paper_results/ give the tables, the numbers
    and the claim check of the paper, without rerunning the benchmark."""
    import os
    from mapf_bench.paper_update import run as update
    root = os.path.join(os.path.dirname(__file__), "..")
    res = os.path.join(root, "paper_results")
    if not os.path.exists(os.path.join(res, "results.csv")):
        pytest.skip("no paper_results")
    import shutil
    res = str(shutil.copytree(res, tmp_path / "results"))     # the update redraws the figure there
    paper = tmp_path / "paper.tex"
    paper.write_text("\\input{bench_numbers}\n%<AUTO:bench_results>\nx\n%</AUTO:bench_results>\n"
                     "%<AUTO:bench_pairs>\ny\n%</AUTO:bench_pairs>\n")
    update(res, str(paper), False, False)
    assert (tmp_path / "bench_numbers.tex").exists()
    text = paper.read_text()
    assert "\\begin{tabular}" in text and "\nx\n" not in text
    assert "contradicted" not in (tmp_path / "paper_update" / "report.md").read_text().lower().replace("0 contradicted", "")
