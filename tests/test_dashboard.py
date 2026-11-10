# SPDX-License-Identifier: GPL-3.0-or-later
"""Dashboard server: runs, instances, merged plans across runs."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from mapf_bench import local_runner as L
from mapf_bench.cpp_local_search import available as cpp_available

needs_cpp = pytest.mark.skipif(not cpp_available(), reason="C++ local search not built")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from http.server import ThreadingHTTPServer
    from mapf_bench.dashboard import make_handler
    root = tmp_path_factory.mktemp("runs")
    for name, T in (("a", 1.0), ("b", 0.5)):
        p = root / f"{name}.json"
        p.write_text(json.dumps({"name": name, "time": T, "methods": ["LS"],
                                 "suites": [{"type": "paper", "groups": ["S"], "indices": [0, 5]}]}))
        L.run(L.load_config(str(p)), str(root / name), workers=1)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler([str(root)]))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _get(url):
    return json.loads(urllib.request.urlopen(url).read())


@needs_cpp
def test_runs_and_instances(server):
    runs = {r["id"]: r for r in _get(server + "/api/runs")}
    assert set(runs) == {"a", "b"} and runs["a"]["time"] == 1.0 and runs["a"]["plans"] == 2
    inst = _get(server + "/api/instances?runs=a,b")
    assert len(inst) == 4 and all(i["plans"] and "LS" in i["methods"] for i in inst)


@needs_cpp
def test_plans_merged_across_runs(server):
    D = _get(server + "/api/plan?items=a|paper:0|initial,a|paper:0|LS,b|paper:0|LS&ref=a/initial")
    assert D["methods"] == ["a/initial", "a/LS", "b/LS"] and D["ref"] == "a/initial"
    assert [r["method"] for r in D["rows"]] == D["methods"]
    D = _get(server + "/api/plan?items=a|paper:0|initial,a|paper:0|LS&ref=a/LS&prefix=1")
    assert D["methods"] == ["a/initial", "a/LS"] and D["ref"] == "a/LS"
    D = _get(server + "/api/plan?items=a|paper:0|initial,a|paper:0|LS")
    assert D["methods"] == ["initial", "LS"]


@needs_cpp
def test_errors_and_page(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server + "/api/plan?items=a|paper:99|LS")
    assert e.value.code == 400
    page = urllib.request.urlopen(server + "/").read().decode()
    assert "function mountAnimation" in page and "Compare solvers" in page
