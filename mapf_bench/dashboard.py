# SPDX-License-Identifier: GPL-3.0-or-later
"""Local dashboard to browse all the simulations, plot results and compare
solvers.

    python -m mapf_bench.dashboard                 # runs under ./results
    python -m mapf_bench.dashboard results/ other/results --port 8765 --no-browser

It serves, on 127.0.0.1 only, a page with three views:
  * Runs: every results folder found (results.csv), with its configuration,
    time budget and machine; one or more runs can be selected;
  * Instances: the instances of the selected runs, filterable by group,
    family and text, with the Sum-of-Costs of every solver (best in bold);
    clicking an instance with saved plans opens it in
  * Animation: the chosen solvers side by side (same component as
    mapf_bench.visualize), with a reference solver; when several runs are
    selected, the plans of the same instance in different runs can be
    compared (labels run/solver);
  * Compare solvers: for a group of instances, the mean SoC/LB of every
    solver, a scatter plot of solver A against solver B instance by instance
    (with wins/ties/losses), and the performance profiles of the chosen
    solvers.
"""
import argparse
import csv
import json
import math
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import plan_store
from .visualize import ANIM_CSS, ANIM_JS, payload


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def find_runs(roots):
    """{run id: folder} for every folder under `roots` that has a results.csv."""
    runs = {}
    for root in roots:
        root = os.path.abspath(root)
        cands = [root] if os.path.exists(os.path.join(root, "results.csv")) else []
        if os.path.isdir(root):
            for d in sorted(os.listdir(root)):
                p = os.path.join(root, d)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "results.csv")):
                    cands.append(p)
        for p in cands:
            rid = os.path.basename(p)
            k, base = 2, rid
            while rid in runs and runs[rid] != p:
                rid = f"{base}-{k}"; k += 1
            runs[rid] = p
    return runs


def run_info(rid, path):
    rows = list(csv.DictReader(open(os.path.join(path, "results.csv"))))
    cfg = _json(os.path.join(path, "config.json"))
    mach = _json(os.path.join(path, "machine.json"))
    methods = list(dict.fromkeys(r["method"] for r in rows if r["method"] != "none"))
    insts = {r["instance"] for r in rows if r["instance"]}
    return {"id": rid, "path": path, "instances": len(insts), "methods": methods,
            "time": cfg.get("time"), "config": cfg.get("name"), "machine": mach.get("cpu"),
            "cores": mach.get("cores"), "workers": mach.get("workers"),
            "errors": sum(1 for r in rows if r.get("error") and r["method"] != "none"),
            "plans": len(plan_store.plan_files(path))}


def _json(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def run_instances(rid, path):
    """One row per instance: metadata and, per method, soc / makespan / time /
    certified / error; whether plans are saved."""
    rows = list(csv.DictReader(open(os.path.join(path, "results.csv"))))
    have = {os.path.basename(f).split(".")[0] for f in plan_store.plan_files(path)}
    out = {}
    for r in rows:
        if not r["instance"]:
            continue
        d = out.setdefault(r["instance"], {
            "run": rid, "job": r.get("job", ""), "group": r["group"], "family": r["family"],
            "instance": r["instance"], "n": _num(r["n"]), "V": _num(r["V"]), "lb": _num(r.get("lb")),
            "soc0": _num(r.get("soc0")), "methods": {}})
        if r["method"] == "none":
            d["skipped"] = r.get("error") or "skipped"
            continue
        for k in ("lb", "soc0"):
            if d[k] is None:
                d[k] = _num(r.get(k))
        d["methods"][r["method"]] = {"soc": _num(r["soc"]), "makespan": _num(r["makespan"]),
                                     "time": _num(r["time"]), "certified": r.get("certified", ""),
                                     "error": r.get("error", "")}
    for d in out.values():
        d["plans"] = d["job"].replace(":", "_").replace("/", "_") in have
    return list(out.values())


def _num(x):
    try:
        v = float(x)
        return int(v) if v.is_integer() else v
    except (TypeError, ValueError):
        return None


def instance_payload(runs, items, ref=None, prefix=None):
    """Animation data for a list of (run id, job, method). Plans of the same
    instance from several runs are merged with labels run/method; prefix
    forces (True) or avoids (False) the run in the labels (default: only
    when the items come from several runs)."""
    datas, chosen = {}, []
    for rid, job, m in items:
        if rid not in runs:
            raise ValueError(f"unknown run {rid}")
        key = (rid, job)
        if key not in datas:
            datas[key] = plan_store.load(_find_plan(runs[rid], job))
        if m in datas[key]["plans"]:
            chosen.append((rid, job, m))
    if not chosen:
        raise ValueError("no saved plan for the selected solvers")
    base = datas[(chosen[0][0], chosen[0][1])]
    inst = base["instance"]
    multi = len({rid for rid, _, _ in chosen}) > 1 if prefix is None else prefix
    merged = {"instance": inst, "plans": {}}
    labels = {}
    for rid, job, m in chosen:
        d = datas[(rid, job)]
        same = (d["instance"]["N"] == inst["N"] and d["instance"]["starts"] == inst["starts"]
                and d["instance"]["goals"] == inst["goals"])
        if not same:
            continue                    # a different instance under the same name
        lab = f"{rid}/{m}" if multi else m
        merged["plans"][lab] = d["plans"][m]
        labels[lab] = lab
    if ref and ref not in merged["plans"]:
        ref = None
    return payload(merged, list(merged["plans"]), ref, labels)


def _find_plan(path, job):
    p = plan_store.plans_path(path, job)
    if os.path.exists(p):
        return p
    old = p[: -len(plan_store.EXT)] + plan_store.OLD_EXT
    if os.path.exists(old):
        return old
    raise ValueError(f"no plans saved for {job}")


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
def make_handler(roots):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                runs = find_runs(roots)
                if u.path in ("/", "/index.html"):
                    return self._send(200, PAGE.replace("__CSS__", ANIM_CSS).replace("__JS__", ANIM_JS),
                                      "text/html")
                if u.path == "/api/runs":
                    return self._send(200, json.dumps([run_info(r, p) for r, p in runs.items()]))
                if u.path == "/api/instances":
                    out = []
                    for rid in q.get("runs", "").split(","):
                        if rid in runs:
                            out += run_instances(rid, runs[rid])
                    return self._send(200, json.dumps(out))
                if u.path == "/api/plan":
                    items = [tuple(x.split("|", 2)) for x in q.get("items", "").split(",") if x.count("|") == 2]
                    pref = {"1": True, "0": False}.get(q.get("prefix"))
                    return self._send(200, json.dumps(instance_payload(runs, items, q.get("ref"), pref),
                                                      separators=(",", ":")))
                return self._send(404, json.dumps({"error": "not found"}))
            except (ValueError, OSError) as e:
                return self._send(400, json.dumps({"error": str(e)}))
    return H


def serve(roots, port=8765, open_browser=True):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(roots))
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"dashboard on {url}  (runs under: {', '.join(os.path.abspath(r) for r in roots)}); Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["results"], help="folders with runs (default: results)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    if not find_runs(a.roots):
        print(f"no runs found under {', '.join(a.roots)} (folders with a results.csv)", file=sys.stderr)
    serve(a.roots, a.port, not a.no_browser)


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MAPF results</title>
<style>
__CSS__
body { margin:0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--fg); }
header { display:flex; gap: 18px; align-items: baseline; padding: 10px 16px; border-bottom: 1px solid var(--line);
         position: sticky; top: 0; background: var(--bg); z-index: 5; flex-wrap: wrap; }
header h1 { font-size: 17px; margin: 0; }
nav button { font: inherit; padding: 5px 12px; border-radius: 6px; border: 1px solid var(--line);
             background: var(--panel); color: var(--fg); cursor: pointer; }
nav button.on { border-color: var(--accent); color: var(--accent); }
main { max-width: 1500px; margin: 0 auto; padding: 12px 16px; }
.view { display: none; } .view.on { display: block; }
table.data { border-collapse: collapse; width: 100%; background: var(--panel); font-size: 13px; }
table.data th, table.data td { padding: 4px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
table.data th { position: sticky; top: 0; background: var(--panel); cursor: pointer; }
table.data td.l, table.data th.l { text-align: left; }
table.data tr.click:hover { background: var(--cell); cursor: pointer; }
.box { overflow: auto; max-height: 70vh; border: 1px solid var(--line); border-radius: 10px; }
.bar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin: 8px 0; }
.bar select, .bar input[type=text] { font: inherit; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--line);
                                     background: var(--panel); color: var(--fg); }
.muted { color: var(--muted); }
.chips label { display: inline-block; margin: 2px 8px 2px 0; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; margin: 10px 0; }
svg.plot { width: 100%; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
.err { color: #d1453b; }
.grid2 { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); }
</style>
</head>
<body>
<header>
  <h1>MAPF results</h1>
  <nav>
    <button data-v="runs" class="on">Runs</button>
    <button data-v="instances">Instances</button>
    <button data-v="anim">Animation</button>
    <button data-v="compare">Compare solvers</button>
  </nav>
  <span class="muted" id="status"></span>
</header>
<main>
<section class="view on" id="v-runs">
  <p class="muted">Select the runs to explore; instances and comparisons use the selected runs.</p>
  <div class="box"><table class="data" id="runs"></table></div>
</section>
<section class="view" id="v-instances">
  <div class="bar">
    <label>group <select id="f-group"></select></label>
    <label>family <select id="f-family"></select></label>
    <input type="text" id="f-text" placeholder="search instance">
    <label><input type="checkbox" id="f-plans"> only with plans</label>
    <span class="muted" id="f-count"></span>
  </div>
  <div class="box"><table class="data" id="insts"></table></div>
  <p class="muted">Click a row with saved plans to animate it. Bold: best Sum-of-Costs of the row.</p>
</section>
<section class="view" id="v-anim">
  <div class="card">
    <div><b id="a-title">No instance selected</b> <span class="muted" id="a-sub"></span></div>
    <div class="chips" id="a-methods"></div>
    <div class="bar"><label>reference <select id="a-ref"></select></label>
      <button id="a-go">Show</button> <span class="err" id="a-err"></span></div>
  </div>
  <div id="a-root"></div>
</section>
<section class="view" id="v-compare">
  <div class="bar">
    <label>group <select id="c-group"></select></label>
    <label>solver A <select id="c-a"></select></label>
    <label>solver B <select id="c-b"></select></label>
    <span class="muted" id="c-note"></span>
  </div>
  <div class="grid2">
    <div class="card"><b>Mean SoC/LB (geometric) by solver</b><div id="c-bars"></div></div>
    <div class="card"><b>Instance by instance: SoC/LB of A (x) against B (y)</b><div id="c-scatter"></div><div id="c-wtl" class="muted"></div></div>
  </div>
  <div class="card"><b>Performance profiles</b> <span class="muted">fraction of instances within a factor &tau; of the best selected solver</span>
    <div class="chips" id="c-prof-methods"></div><div id="c-prof"></div></div>
</section>
</main>
<script>
__JS__
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
let RUNS = [], SEL = new Set(), INST = [], CUR = null, ANIM = null;
const show = v => { document.querySelectorAll('nav button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  document.querySelectorAll('.view').forEach(s => s.classList.toggle('on', s.id === 'v-' + v)); };
document.querySelectorAll('nav button').forEach(b => b.onclick = () => show(b.dataset.v));
async function api(path) { const r = await fetch(path); const j = await r.json(); if (!r.ok) throw new Error(j.error || r.status); return j; }

// ---------------- runs
async function loadRuns() {
  RUNS = await api('/api/runs');
  if (!SEL.size && RUNS.length) SEL.add(RUNS[0].id);
  const t = $('runs');
  t.innerHTML = '<tr><th></th><th class="l">run</th><th class="l">configuration</th><th>instances</th><th>T (s)</th>' +
    '<th class="l">solvers</th><th>plans</th><th>errors</th><th class="l">machine</th><th class="l">folder</th></tr>' +
    RUNS.map(r => `<tr><td><input type="checkbox" data-r="${esc(r.id)}" ${SEL.has(r.id) ? 'checked' : ''}></td>` +
      `<td class="l"><b>${esc(r.id)}</b></td><td class="l">${esc(r.config)}</td><td>${r.instances}</td><td>${r.time ?? ''}</td>` +
      `<td class="l">${esc(r.methods.join(', '))}</td><td>${r.plans}</td><td>${r.errors ? '<span class="err">' + r.errors + '</span>' : 0}</td>` +
      `<td class="l">${esc(r.machine || '')}${r.workers ? ' · ' + r.workers + ' workers' : ''}</td><td class="l muted">${esc(r.path)}</td></tr>`).join('');
  t.querySelectorAll('input[data-r]').forEach(c => c.onchange = () => { c.checked ? SEL.add(c.dataset.r) : SEL.delete(c.dataset.r); loadInstances(); });
  $('status').textContent = `${RUNS.length} runs found`;
  await loadInstances();
}

// ---------------- instances
const methodsOf = rows => [...new Set(rows.flatMap(r => Object.keys(r.methods)))];
const label = (r, m) => SEL.size > 1 ? `${r.run}/${m}` : m;
async function loadInstances() {
  INST = SEL.size ? await api('/api/instances?runs=' + encodeURIComponent([...SEL].join(','))) : [];
  const groups = [...new Set(INST.map(r => r.group))].sort(), fams = [...new Set(INST.map(r => r.family))].sort();
  const fill = (sel, vals) => { const v = sel.value; sel.innerHTML = '<option value="">all</option>' + vals.map(x => `<option>${esc(x)}</option>`).join('');
    if (vals.includes(v)) sel.value = v; };
  fill($('f-group'), groups); fill($('f-family'), fams); fill($('c-group'), groups);
  renderInstances(); renderCompare();
}
let sortKey = null, sortDir = 1;
function renderInstances() {
  const g = $('f-group').value, f = $('f-family').value, q = $('f-text').value.toLowerCase(), onlyP = $('f-plans').checked;
  let rows = INST.filter(r => (!g || r.group === g) && (!f || r.family === f) && (!q || r.instance.toLowerCase().includes(q)) && (!onlyP || r.plans));
  const ms = methodsOf(rows);
  if (sortKey) rows.sort((a, b) => { const va = sortKey.startsWith('m:') ? a.methods[sortKey.slice(2)]?.soc : a[sortKey];
    const vb = sortKey.startsWith('m:') ? b.methods[sortKey.slice(2)]?.soc : b[sortKey];
    return ((va ?? 1e18) > (vb ?? 1e18) ? 1 : (va ?? 1e18) < (vb ?? 1e18) ? -1 : 0) * sortDir; });
  const head = [['run', 'run', 1], ['group', 'group', 1], ['family', 'family', 1], ['instance', 'instance', 1], ['n', 'n'], ['V', '|V|'], ['lb', 'LB'], ['soc0', 'initial']];
  const t = $('insts');
  t.innerHTML = '<tr>' + head.map(([k, h, l]) => `<th class="${l ? 'l' : ''}" data-k="${k}">${h}</th>`).join('') +
    ms.map(m => `<th data-k="m:${esc(m)}">${esc(m)}</th>`).join('') + '<th>plans</th></tr>' +
    rows.map((r, i) => { const vals = ms.map(m => r.methods[m]?.soc).filter(v => v != null), best = Math.min(...vals);
      return `<tr class="${r.plans ? 'click' : ''}" data-i="${INST.indexOf(r)}">` + head.map(([k, , l]) => `<td class="${l ? 'l' : ''}">${esc(r[k] ?? '')}</td>`).join('') +
        ms.map(m => { const x = r.methods[m]; if (!x) return '<td></td>'; if (x.error) return `<td class="err" title="${esc(x.error)}">error</td>`;
          return `<td>${x.soc === best ? '<b>' + x.soc + '</b>' : (x.soc ?? '')}</td>`; }).join('') +
        `<td>${r.plans ? '&#9654;' : (r.skipped ? '<span class="muted">skipped</span>' : '')}</td></tr>`; }).join('');
  $('f-count').textContent = `${rows.length} of ${INST.length} instances`;
  t.querySelectorAll('th[data-k]').forEach(th => th.onclick = () => { sortDir = sortKey === th.dataset.k ? -sortDir : 1; sortKey = th.dataset.k; renderInstances(); });
  t.querySelectorAll('tr.click').forEach(tr => tr.onclick = () => openInstance(INST[+tr.dataset.i]));
}
['f-group', 'f-family', 'f-plans'].forEach(id => $(id).onchange = renderInstances);
$('f-text').oninput = renderInstances;

// ---------------- animation
function openInstance(r) {
  // the same instance in every selected run (with plans)
  const same = INST.filter(x => x.instance === r.instance && x.plans);
  CUR = {name: r.instance, items: [], multi: false};
  for (const x of same) { CUR.items.push([x.run, x.job, 'initial']); for (const m of Object.keys(x.methods)) if (!x.methods[m].error) CUR.items.push([x.run, x.job, m]); }
  // unique labels
  const seen = new Set(); CUR.items = CUR.items.filter(it => { const k = it.join('|'); if (seen.has(k)) return false; seen.add(k); return true; });
  const multi = new Set(same.map(x => x.run)).size > 1; CUR.multi = multi;
  $('a-title').textContent = r.instance;
  $('a-sub').textContent = `${r.group} · ${r.family} · ${r.n} agents · ${r.V} nodes` + (multi ? ` · ${same.length} runs` : '');
  const lab = it => multi ? `${it[0]}/${it[2]}` : it[2];
  $('a-methods').innerHTML = CUR.items.map((it, i) => `<label><input type="checkbox" data-i="${i}" ${i < 6 ? 'checked' : ''}> ${esc(lab(it))}</label>`).join('');
  $('a-ref').innerHTML = CUR.items.map((it, i) => `<option value="${esc(lab(it))}">${esc(lab(it))}</option>`).join('');
  show('anim'); renderAnimation();
}
async function renderAnimation() {
  if (!CUR) return;
  const items = [...$('a-methods').querySelectorAll('input:checked')].map(c => CUR.items[+c.dataset.i]);
  $('a-err').textContent = '';
  if (!items.length) { $('a-err').textContent = 'select at least one solver'; return; }
  try {
    const D = await api('/api/plan?items=' + encodeURIComponent(items.map(x => x.join('|')).join(',')) +
                        '&ref=' + encodeURIComponent($('a-ref').value) + '&prefix=' + (CUR.multi ? 1 : 0));
    if (ANIM) ANIM.stop();
    ANIM = mountAnimation($('a-root'), D);
  } catch (e) { $('a-err').textContent = e.message; }
}
$('a-go').onclick = renderAnimation;

// ---------------- compare solvers
const PAL = ['#2f6fdb', '#e0533d', '#2a9d5b', '#b25fd6', '#d39a1d', '#1aa3b5', '#7a7a80', '#c2185b', '#5d4037', '#00897b'];
function compareRows() {
  const g = $('c-group').value;
  return INST.filter(r => !g || r.group === g).map(r => { const v = {};
    for (const [m, x] of Object.entries(r.methods)) if (x.soc != null && !x.error) v[label(r, m)] = x.soc;
    return {r, v, lb: r.lb || 1, key: r.instance}; });
}
function mergedRows() {   // with several runs: one entry per instance name, solvers labelled run/solver
  const by = {};
  for (const x of compareRows()) { const e = by[x.key] = by[x.key] || {v: {}, lb: x.lb, name: x.key}; Object.assign(e.v, x.v); }
  return Object.values(by);
}
function renderCompare() {
  const rows = mergedRows(), solvers = [...new Set(rows.flatMap(r => Object.keys(r.v)))];
  const fill = (sel, i) => { const v = sel.value; sel.innerHTML = solvers.map(s => `<option>${esc(s)}</option>`).join('');
    sel.value = solvers.includes(v) ? v : (solvers[i] || solvers[0] || ''); };
  fill($('c-a'), 0); fill($('c-b'), 1);
  // bars: geometric mean SoC/LB on the instances where the solver has a value
  const gm = solvers.map(s => { const xs = rows.filter(r => r.v[s] != null).map(r => Math.log(r.v[s] / r.lb));
    return {s, n: xs.length, g: xs.length ? Math.exp(xs.reduce((a, b) => a + b, 0) / xs.length) : null}; }).filter(x => x.g);
  const W = 640, rowH = 24, H = 20 + rowH * gm.length, L = 170, maxg = Math.max(...gm.map(x => x.g), 1);
  $('c-bars').innerHTML = `<svg class="plot" viewBox="0 0 ${W} ${H}">` + gm.map((x, i) => {
    const w = (W - L - 70) * (x.g - 1) / Math.max(maxg - 1, 1e-9), y = 10 + i * rowH;
    return `<text x="${L - 8}" y="${y + 15}" text-anchor="end" font-size="12" fill="var(--fg)">${esc(x.s)}</text>` +
      `<rect x="${L}" y="${y + 3}" width="${Math.max(w, 1)}" height="${rowH - 8}" fill="${PAL[i % PAL.length]}" rx="3"/>` +
      `<text x="${L + Math.max(w, 1) + 6}" y="${y + 15}" font-size="12" fill="var(--muted)">${x.g.toFixed(3)} (${x.n})</text>`; }).join('') + '</svg>';
  // scatter A vs B
  const A = $('c-a').value, B = $('c-b').value, pts = rows.filter(r => r.v[A] != null && r.v[B] != null)
    .map(r => ({x: r.v[A] / r.lb, y: r.v[B] / r.lb, name: r.name, a: r.v[A], b: r.v[B]}));
  const S = 420, pad = 40, lo = 1, hi = Math.max(1.05, ...pts.flatMap(p => [p.x, p.y]));
  const lg = v => Math.log(v), sc = v => pad + (S - 2 * pad) * (lg(v) - lg(lo)) / (lg(hi) - lg(lo) || 1);
  let sv = `<svg class="plot" viewBox="0 0 ${S} ${S}"><line x1="${sc(lo)}" y1="${S - sc(lo)}" x2="${sc(hi)}" y2="${S - sc(hi)}" stroke="var(--line)" stroke-dasharray="4 3"/>` +
    `<text x="${S / 2}" y="${S - 8}" text-anchor="middle" font-size="12" fill="var(--muted)">${esc(A)} (SoC/LB, log)</text>` +
    `<text x="12" y="${S / 2}" font-size="12" fill="var(--muted)" transform="rotate(-90 12 ${S / 2})" text-anchor="middle">${esc(B)}</text>`;
  for (const p of pts) { const c = p.a < p.b ? '#2a9d5b' : p.a > p.b ? '#e0533d' : 'var(--muted)';
    sv += `<circle cx="${sc(p.x)}" cy="${S - sc(p.y)}" r="4" fill="${c}" fill-opacity="0.8"><title>${esc(p.name)}: ${esc(A)} ${p.a}, ${esc(B)} ${p.b}</title></circle>`; }
  $('c-scatter').innerHTML = sv + '</svg>';
  const w = pts.filter(p => p.a < p.b).length, l = pts.filter(p => p.a > p.b).length;
  const rel = pts.map(p => (p.a - p.b) / p.b).sort((a, b) => a - b), med = rel.length ? rel[Math.floor((rel.length - 1) / 2)] : 0;
  $('c-wtl').innerHTML = pts.length ? `${esc(A)} vs ${esc(B)} on ${pts.length} instances: <b style="color:#2a9d5b">${w} better</b>, ${pts.length - w - l} equal, ` +
    `<b style="color:#e0533d">${l} worse</b>; median relative difference ${(100 * med).toFixed(1)}%. Points below the diagonal: B worse.` : 'no common instances';
  $('c-note').textContent = `${rows.length} instances`;
  // performance profiles
  const box = $('c-prof-methods'), prev = new Set([...box.querySelectorAll('input:checked')].map(c => c.value));
  box.innerHTML = solvers.map((s, i) => `<label><input type="checkbox" value="${esc(s)}" ${(prev.size ? prev.has(s) : i < 6) ? 'checked' : ''}> ${esc(s)}</label>`).join('');
  box.querySelectorAll('input').forEach(c => c.onchange = renderProfiles);
  renderProfiles();
}
function renderProfiles() {
  const rows = mergedRows(), chosen = [...$('c-prof-methods').querySelectorAll('input:checked')].map(c => c.value);
  const W = 900, H = 300, L = 46, B = 30, T = 2, X = t => L + (W - L - 10) * (t - 1) / (T - 1), Y = f => 10 + (H - B - 10) * (1 - f);
  let sv = `<svg class="plot" viewBox="0 0 ${W} ${H}"><line x1="${L}" y1="${H - B}" x2="${W - 10}" y2="${H - B}" stroke="var(--line)"/>` +
    `<line x1="${L}" y1="10" x2="${L}" y2="${H - B}" stroke="var(--line)"/>` +
    [0, 0.5, 1].map(f => `<text x="${L - 6}" y="${Y(f) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${f}</text>`).join('') +
    [1, 1.25, 1.5, 1.75, 2].map(t => `<text x="${X(t)}" y="${H - 10}" text-anchor="middle" font-size="11" fill="var(--muted)">${t}</text>`).join('');
  const usable = rows.filter(r => chosen.some(s => r.v[s] != null));
  chosen.forEach((s, i) => { const pts = [];
    for (let k = 0; k <= 100; k++) { const t = 1 + k / 100;
      const f = usable.filter(r => { const best = Math.min(...chosen.map(c => r.v[c]).filter(v => v != null)); return r.v[s] != null && r.v[s] <= t * best; }).length / Math.max(usable.length, 1);
      pts.push(`${X(t).toFixed(1)},${Y(f).toFixed(1)}`); }
    sv += `<polyline fill="none" stroke="${PAL[i % PAL.length]}" stroke-width="2" points="${pts.join(' ')}"/>` +
          `<text x="${L + 10 + (i % 4) * 210}" y="${24 + Math.floor(i / 4) * 16}" font-size="12" fill="${PAL[i % PAL.length]}">&#9632; ${esc(s)}</text>`; });
  $('c-prof').innerHTML = sv + '</svg>';
}
['c-group', 'c-a', 'c-b'].forEach(id => $(id).onchange = renderCompare);
loadRuns().catch(e => { $('status').textContent = 'error: ' + e.message; });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
