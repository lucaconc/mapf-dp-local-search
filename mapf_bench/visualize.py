# SPDX-License-Identifier: GPL-3.0-or-later
"""Animate and compare the plans found by the methods on an instance.

    python -m mapf_bench.visualize results/paper --list
    python -m mapf_bench.visualize results/paper --job paper:0
    python -m mapf_bench.visualize results/paper --job paper:59 --methods initial LaCAM* LaCAM*+Hybrid \\
        --ref LaCAM* --out ring.html --gif ring.gif
    python -m mapf_bench.visualize results/paper --all            # one page per job + index.html

The plans are saved by the local runner in <results>/plans/<job>.plans.xz
(instance + the plan of every method, and the initial plan as "initial"; see
mapf_bench.plan_store for the compact format). `--convert` rewrites the
files of older runs (<job>.json.gz) in the compact format.
The HTML page is self-contained (no external files or libraries): the
methods are animated side by side on the same instance with the same color
for every agent, with play/pause, a time slider, single steps, speed, trails
and goals; clicking an agent highlights it in all panels and shows its
arrival time with each method. Below the panels, a table compares
Sum-of-Costs, makespan and per-agent arrival times with a reference method,
and a chart shows the number of agents not yet arrived over time.
`--gif` writes the same comparison as an animated GIF (matplotlib + Pillow).
"""
import argparse
import gzip
import html
import json
import math
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
from .plan_store import load, plan_files, plans_path, convert, OLD_EXT


def find(results, job):
    p = plans_path(results, job)
    if os.path.exists(p):
        return p
    old = p[: -len(".plans.xz")] + OLD_EXT
    if os.path.exists(old):
        return old
    for f in plan_files(results):          # also accept an instance name
        if load(f)["instance"]["name"] == job:
            return f
    raise SystemExit(f"no plans for {job!r} in {results}/plans (run with save_plans enabled)")


def arrivals(plan, goals):
    """Arrival time of each agent: 1 + last time it is not at its goal."""
    T = len(plan) - 1
    out = []
    for k, g in enumerate(goals):
        a = 0
        for t in range(T, -1, -1):
            if plan[t][k] != g:
                a = t + 1
                break
        out.append(a)
    return out


def compare(data, methods=None, ref=None):
    """Rows (method, SoC, makespan, better, equal, worse arrivals than ref)."""
    inst, plans = data["instance"], data["plans"]
    methods = methods or list(plans)
    arr = {m: arrivals(plans[m], inst["goals"]) for m in methods}
    ref = ref if ref in arr else methods[0]
    rows = []
    for m in methods:
        a, r = arr[m], arr[ref]
        rows.append({"method": m, "soc": sum(a), "makespan": len(plans[m]) - 1,
                     "better": sum(x < y for x, y in zip(a, r)),
                     "equal": sum(x == y for x, y in zip(a, r)),
                     "worse": sum(x > y for x, y in zip(a, r))})
    return rows, arr, ref


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
def layout(inst, seed=0, iters=400):
    """Node positions in [0, 1]^2: grid coordinates when available, otherwise
    a Fruchterman-Reingold spring layout (deterministic)."""
    N = inst["N"]
    if inst.get("coords"):
        rc = np.array([inst["coords"][str(v)] for v in range(N)], dtype=float)
        xy = np.stack([rc[:, 1], rc[:, 0]], axis=1)          # x = column, y = row
        return xy, True
    rng = np.random.default_rng(seed)
    pos = rng.random((N, 2))
    edges = {(min(u, v), max(u, v)) for u, v in inst["arcs"]}
    E = np.array(sorted(edges)) if edges else np.zeros((0, 2), int)
    k = 1.0 / math.sqrt(N)
    t = 0.1
    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(d, axis=2) + 1e-9
        disp = (d / dist[..., None] * (k * k / dist)[..., None]).sum(axis=1)
        if len(E):
            de = pos[E[:, 0]] - pos[E[:, 1]]
            le = np.linalg.norm(de, axis=1)[:, None] + 1e-9
            f = de / le * (le * le / k)
            np.add.at(disp, E[:, 0], -f)
            np.add.at(disp, E[:, 1], f)
        ln = np.linalg.norm(disp, axis=1)[:, None] + 1e-9
        pos += disp / ln * np.minimum(ln, t)
        t *= 0.99
    pos -= pos.min(axis=0)
    pos /= max(pos.max(), 1e-9)
    return pos, False


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def payload(data, methods=None, ref=None, labels=None):
    """Data of the animation component: layout, arcs, plans, arrivals and the
    comparison rows. `labels` renames the methods (e.g. run/method)."""
    inst, plans = data["instance"], data["plans"]
    methods = [m for m in (methods or list(plans)) if m in plans]
    if not methods:
        raise ValueError("none of the requested methods has a plan for this instance")
    rows, arr, ref = compare(data, methods, ref)
    lab = labels or {m: m for m in methods}
    xy, grid = layout(inst)
    arcset = {(u, v) for u, v in inst["arcs"]}
    for r in rows:
        r["method"] = lab[r["method"]]
    return {
        "name": inst["name"], "n": inst["n"], "grid": grid,
        "xy": [[round(float(x), 4), round(float(y), 4)] for x, y in xy],
        "arcs": [[u, v, int((v, u) not in arcset)] for u, v in inst["arcs"] if u < v or (v, u) not in arcset],
        "starts": inst["starts"], "goals": inst["goals"],
        "methods": [lab[m] for m in methods], "plans": {lab[m]: plans[m] for m in methods},
        "arrivals": {lab[m]: arr[m] for m in methods}, "ref": lab[ref], "rows": rows}


def write_html(data, out, methods=None, ref=None, title=None):
    D = payload(data, methods, ref)
    title = title or f"{D['name']}: {', '.join(D['methods'])}"
    page = (PAGE.replace("__TITLE__", html.escape(title)).replace("__CSS__", ANIM_CSS)
            .replace("__JS__", ANIM_JS).replace("__DATA__", json.dumps(D, separators=(",", ":"))))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "w").write(page)
    return out


ANIM_CSS = r"""
:root { --bg:#fafafa; --fg:#1d1d1f; --muted:#6b6b70; --panel:#ffffff; --line:#d9d9de;
        --cell:#eef0f3; --wall:#3a3a40; --accent:#2f6fdb; color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141416; --fg:#ececf0; --muted:#9a9aa3; --panel:#1d1d21; --line:#34343a;
          --cell:#2a2a30; --wall:#0c0c0e; --accent:#7aa7ff; color-scheme: dark; }
}
.anim * { box-sizing: border-box; }
.anim .controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:10px 0;
                  position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--line); z-index: 2; }
.anim button, .anim select { font: inherit; padding: 4px 10px; border-radius: 6px; border: 1px solid var(--line);
                             background: var(--panel); color: var(--fg); cursor: pointer; }
.anim input[type=range] { flex: 1; min-width: 160px; }
.anim .panels { display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.anim .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 8px; }
.anim .panel h3 { font-size: 14px; margin: 0 0 4px; display:flex; justify-content:space-between; gap: 8px; }
.anim .panel h3 span { color: var(--muted); font-weight: normal; }
.anim canvas { width: 100%; display: block; border-radius: 6px; }
.anim table { border-collapse: collapse; width: 100%; background: var(--panel); }
.anim th, .anim td { padding: 5px 8px; border-bottom: 1px solid var(--line); text-align: right; }
.anim th:first-child, .anim td:first-child { text-align: left; }
.anim svg.chart { width: 100%; height: 220px; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
.anim .info { color: var(--muted); min-height: 1.4em; margin: 8px 0; }
.anim h2 { font-size: 16px; margin: 18px 0 6px; }
.anim .sub, .anim label { color: var(--muted); }
.anim .wrap { overflow-x: auto; }
"""

# The animation component: mountAnimation(root, D) builds controls, panels,
# comparison table and chart inside `root`; it returns {stop()}.
ANIM_JS = r"""
function mountAnimation(root, D) {
  const mk = (tag, attrs = {}, parent = null) => { const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) { if (k === 'text') e.textContent = v; else if (k === 'html') e.innerHTML = v; else e[k] = v; }
    if (parent) parent.append(e); return e; };
  root.innerHTML = ''; root.className = (root.className || '') + ' anim';
  const M = D.methods, n = D.n;
  const Tmax = Math.max(...M.map(m => D.plans[m].length - 1));
  const css = getComputedStyle(document.documentElement);
  const dark = () => matchMedia('(prefers-color-scheme: dark)').matches;
  const col = k => `hsl(${(k * 137.508) % 360}, 68%, ${dark() ? 62 : 46}%)`;
  // controls
  const bar = mk('div', {className: 'controls'}, root);
  const play = mk('button', {text: 'Play'}, bar);
  const back = mk('button', {html: '&#9664;', title: 'previous step'}, bar);
  const fwd = mk('button', {html: '&#9654;', title: 'next step'}, bar);
  const slider = mk('input', {type: 'range', min: 0, step: 0.01, value: 0}, bar); slider.max = Tmax;
  const tlabel = mk('span', {text: 't = 0'}, bar);
  const speedL = mk('label', {text: 'speed '}, bar);
  const speed = mk('select', {}, speedL);
  for (const v of [0.5, 1, 2, 4, 8]) { const o = mk('option', {text: v, value: v}, speed); if (v === 1) o.selected = true; }
  speedL.append(' steps/s');
  const trL = mk('label', {}, bar), trails = mk('input', {type: 'checkbox', checked: true}, trL); trL.append(' trails');
  const goL = mk('label', {}, bar), goals = mk('input', {type: 'checkbox', checked: true}, goL); goL.append(' goals');
  const info = mk('div', {className: 'info', text: 'Click an agent to follow it in all panels.'}, root);
  const panelsDiv = mk('div', {className: 'panels'}, root);
  mk('h2', {text: 'Comparison'}, root);
  mk('div', {className: 'sub', text: `Per-agent arrival times compared with ${D.ref}: agents that arrive earlier / at the same time / later.`}, root);
  const table = mk('table', {}, mk('div', {className: 'wrap'}, root));
  mk('h2', {text: 'Agents not yet arrived'}, root);
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg'); svg.setAttribute('class', 'chart');
  svg.setAttribute('viewBox', '0 0 1000 220'); svg.setAttribute('preserveAspectRatio', 'none'); root.append(svg);

  // geometry
  const xs = D.xy.map(p => p[0]), ys = D.xy.map(p => p[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
  const spanx = (maxx - minx) || 1, spany = (maxy - miny) || 1;
  let unit = 1;
  if (!D.grid) { const ds = D.arcs.map(([u, v]) => Math.hypot(D.xy[u][0] - D.xy[v][0], D.xy[u][1] - D.xy[v][1])).sort((a, b) => a - b);
    unit = ds.length ? ds[Math.floor(ds.length / 2)] : 0.05; }
  const panels = M.map(m => {
    const div = mk('div', {className: 'panel'}, panelsDiv);
    const h = mk('h3', {html: `<b></b><span></span>`}, div); h.firstChild.textContent = m;
    const c = mk('canvas', {}, div);
    return {m, c, info: h.lastChild, plan: D.plans[m], arr: D.arrivals[m]};
  });
  function view(c) {
    const w = c.clientWidth || 400, pad = 0.8;
    const sx = spanx + 2 * pad * unit, sy = spany + 2 * pad * unit;
    const h = Math.max(120, Math.min(w * sy / sx, 640));
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = h + 'px'; }
    const s = Math.min(w / sx, h / sy);
    return {s, ox: (w - s * sx) / 2 + s * pad * unit - s * minx, oy: (h - s * sy) / 2 + s * pad * unit - s * miny, dpr, w, h};
  }
  const P = (v, V) => [V.ox + V.s * D.xy[v][0], V.oy + V.s * D.xy[v][1]];
  const at = (plan, t, k) => { const T = plan.length - 1; if (t >= T) return [plan[T][k], plan[T][k], 0];
    const i = Math.floor(t); return [plan[i][k], plan[i + 1][k], t - i]; };
  let t = 0, playing = false, selected = -1, last = null, alive = true;
  function drawPanel(p) {
    const V = view(p.c), g = p.c.getContext('2d');
    g.setTransform(V.dpr, 0, 0, V.dpr, 0, 0); g.clearRect(0, 0, V.w, V.h);
    const cs = V.s * unit;
    if (D.grid) {
      g.fillStyle = css.getPropertyValue('--wall');
      g.fillRect(V.ox + V.s * (minx - 0.5), V.oy + V.s * (miny - 0.5), V.s * (spanx + 1), V.s * (spany + 1));
      g.fillStyle = css.getPropertyValue('--cell');
      for (let v = 0; v < D.xy.length; v++) { const [x, y] = P(v, V); g.fillRect(x - cs / 2 + 0.5, y - cs / 2 + 0.5, cs - 1, cs - 1); }
    }
    g.strokeStyle = css.getPropertyValue('--line'); g.lineWidth = 1;
    for (const [u, v, oneway] of D.arcs) {
      if (D.grid && !oneway) continue;
      const [x1, y1] = P(u, V), [x2, y2] = P(v, V);
      g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
      if (oneway) { const a = Math.atan2(y2 - y1, x2 - x1), mx = (x1 + x2) / 2, my = (y1 + y2) / 2, r = Math.max(3, cs * 0.16);
        g.beginPath(); g.moveTo(mx + r * Math.cos(a), my + r * Math.sin(a));
        g.lineTo(mx + r * Math.cos(a + 2.5), my + r * Math.sin(a + 2.5));
        g.lineTo(mx + r * Math.cos(a - 2.5), my + r * Math.sin(a - 2.5)); g.closePath();
        g.fillStyle = css.getPropertyValue('--line'); g.fill(); }
    }
    if (!D.grid) { g.fillStyle = css.getPropertyValue('--cell');
      for (let v = 0; v < D.xy.length; v++) { const [x, y] = P(v, V); g.beginPath(); g.arc(x, y, cs * 0.3, 0, 7); g.fill(); } }
    const r = Math.max(2.5, cs * 0.36);
    if (goals.checked) for (let k = 0; k < n; k++) { const [x, y] = P(D.goals[k], V);
      g.strokeStyle = col(k); g.lineWidth = Math.max(1, r * 0.25); g.strokeRect(x - r * 0.6, y - r * 0.6, r * 1.2, r * 1.2); }
    if (trails.checked) for (let k = 0; k < n; k++) {
      g.strokeStyle = col(k); g.globalAlpha = 0.35; g.lineWidth = Math.max(1, r * 0.35); g.beginPath();
      const t0 = Math.max(0, Math.floor(t) - 6);
      for (let s = t0; s <= Math.min(Math.floor(t), p.plan.length - 1); s++) { const [x, y] = P(p.plan[s][k], V); s === t0 ? g.moveTo(x, y) : g.lineTo(x, y); }
      const [a, b, f] = at(p.plan, t, k), [xa, ya] = P(a, V), [xb, yb] = P(b, V);
      g.lineTo(xa + (xb - xa) * f, ya + (yb - ya) * f); g.stroke(); g.globalAlpha = 1;
    }
    let arrived = 0;
    for (let k = 0; k < n; k++) {
      const [a, b, f] = at(p.plan, t, k), [xa, ya] = P(a, V), [xb, yb] = P(b, V);
      const x = xa + (xb - xa) * f, y = ya + (yb - ya) * f, done = t >= p.arr[k];
      if (done) arrived++;
      g.globalAlpha = (selected >= 0 && selected !== k) ? 0.25 : (done ? 0.55 : 1);
      g.fillStyle = col(k); g.beginPath(); g.arc(x, y, r, 0, 7); g.fill(); g.globalAlpha = 1;
      if (k === selected) { g.strokeStyle = css.getPropertyValue('--fg'); g.lineWidth = 2.5; g.beginPath(); g.arc(x, y, r + 3, 0, 7); g.stroke(); }
    }
    p.info.textContent = `SoC ${p.arr.reduce((a, b) => a + b, 0)} · makespan ${p.plan.length - 1} · arrived ${arrived}/${n}`;
  }
  let cursor = null;
  function draw() { if (!alive) return; panels.forEach(drawPanel);
    if (cursor) { cursor.setAttribute('x1', X(t)); cursor.setAttribute('x2', X(t)); }
    slider.value = t; tlabel.textContent = `t = ${t.toFixed(1)} / ${Tmax}`; }
  slider.oninput = () => { t = +slider.value; draw(); };
  play.onclick = () => { playing = !playing; if (t >= Tmax) t = 0; play.textContent = playing ? 'Pause' : 'Play'; last = null;
    if (playing) requestAnimationFrame(tick); };
  back.onclick = () => { t = Math.max(0, Math.ceil(t) - 1); draw(); };
  fwd.onclick = () => { t = Math.min(Tmax, Math.floor(t) + 1); draw(); };
  trails.onchange = draw; goals.onchange = draw;
  function tick(now) {
    if (!playing || !alive) return;
    if (last !== null) t = Math.min(Tmax, t + (now - last) / 1000 * +speed.value);
    last = now; draw();
    if (t >= Tmax) { playing = false; play.textContent = 'Play'; return; }
    requestAnimationFrame(tick);
  }
  panels.forEach(p => p.c.addEventListener('click', ev => {
    const V = view(p.c), rect = p.c.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    let best = -1, bd = 1e9;
    for (let k = 0; k < n; k++) { const [a, b, f] = at(p.plan, t, k), [xa, ya] = P(a, V), [xb, yb] = P(b, V);
      const d = Math.hypot(xa + (xb - xa) * f - mx, ya + (yb - ya) * f - my); if (d < bd) { bd = d; best = k; } }
    selected = (bd < V.s * unit) && best !== selected ? best : -1;
    info.textContent = selected < 0 ? 'Click an agent to follow it in all panels.' :
      `Agent ${selected}: arrival ` + M.map(m => `${m} ${D.arrivals[m][selected]}`).join(' · ');
    draw();
  }));
  const onResize = () => draw(); window.addEventListener('resize', onResize);
  // comparison table
  const refSoc = D.rows.find(x => x.method === D.ref).soc;
  table.innerHTML = '<tr><th>method</th><th>Sum-of-Costs</th><th>makespan</th><th>earlier</th><th>same</th><th>later</th><th>SoC vs ref.</th></tr>' +
    D.rows.map(r => `<tr><td></td><td>${r.soc}</td><td>${r.makespan}</td><td>${r.better}</td><td>${r.equal}</td><td>${r.worse}</td>` +
      `<td>${r.method === D.ref ? 'ref.' : ((r.soc - refSoc) / refSoc * 100).toFixed(1) + '%'}</td></tr>`).join('');
  [...table.querySelectorAll ? table.querySelectorAll('tr') : []].slice(1).forEach((tr, i) => { tr.firstChild.textContent = D.rows[i].method; });
  // chart
  const W = 1000, H = 220, L = 40, B = 24;
  const X = s => L + (W - L - 8) * s / Math.max(Tmax, 1), Y = v => 8 + (H - B - 8) * (1 - v / n);
  const pal = ['#2f6fdb', '#e0533d', '#2a9d5b', '#b25fd6', '#d39a1d', '#1aa3b5', '#7a7a80', '#c2185b'];
  let sv = `<line x1="${L}" y1="${H - B}" x2="${W - 8}" y2="${H - B}" stroke="var(--line)"/>` +
    `<text x="${L - 6}" y="${Y(n) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${n}</text>` +
    `<text x="${L - 6}" y="${Y(0) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">0</text>` +
    `<text x="${W - 8}" y="${H - 6}" text-anchor="end" font-size="11" fill="var(--muted)">t = ${Tmax}</text>`;
  M.forEach((m, i) => { const pts = [];
    for (let s = 0; s <= Tmax; s++) pts.push(`${X(s).toFixed(1)},${Y(D.arrivals[m].filter(a => a > s).length).toFixed(1)}`);
    sv += `<polyline fill="none" stroke="${pal[i % pal.length]}" stroke-width="2" points="${pts.join(' ')}"/>` +
          `<text x="${L + 8 + i * 160}" y="20" font-size="12" fill="${pal[i % pal.length]}">&#9632; ${m.replace(/[<>&]/g, '')}</text>`; });
  sv += `<line class="cursor" y1="8" y2="${H - B}" stroke="var(--fg)" stroke-dasharray="3 3"/>`;
  svg.innerHTML = sv; cursor = svg.querySelector ? svg.querySelector('.cursor') : null;
  draw();
  return {stop() { alive = false; playing = false; window.removeEventListener('resize', onResize); }};
}
"""

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__CSS__
body { margin:0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--fg); }
main { max-width: 1400px; margin: 0 auto; padding: 12px 16px; }
h1 { font-size: 18px; margin: 4px 0 2px; }
</style>
</head>
<body>
<main>
  <h1 id="title"></h1>
  <div class="sub" id="subtitle"></div>
  <div id="app"></div>
</main>
<script>
__JS__
const D = __DATA__;
document.getElementById('title').textContent = D.name;
document.getElementById('subtitle').textContent =
  `${D.n} agents, ${D.xy.length} nodes, ${D.grid ? 'grid' : 'graph'}; methods: ${D.methods.join(', ')}`;
mountAnimation(document.getElementById('app'), D);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# GIF
# ---------------------------------------------------------------------------
def write_gif(data, out, methods=None, substeps=4, fps=12, max_frames=600, size=4.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    inst, plans = data["instance"], data["plans"]
    methods = [m for m in (methods or list(plans)) if m in plans]
    _, arr, _ = compare(data, methods)
    xy, grid = layout(inst)
    n = inst["n"]
    T = max(len(plans[m]) - 1 for m in methods)
    frames = min(max_frames, T * substeps + 1)
    times = np.linspace(0, T, frames)
    colors = [matplotlib.colors.hsv_to_rgb(((k * 137.508) % 360 / 360, 0.65, 0.85)) for k in range(n)]
    span = np.ptp(xy, axis=0) + (1.0 if grid else 0.1)
    aspect = float(np.clip(span[1] / span[0], 0.3, 1.6))
    fig, axes = plt.subplots(1, len(methods), figsize=(size * len(methods), size * aspect + 0.9),
                             squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.6 / (size * aspect + 0.9),
                        bottom=0.3 / (size * aspect + 0.9), wspace=0.05)
    cell = size * 72 / span[0]                     # points per grid cell
    scat = []
    for ax, m in zip(axes[0], methods):
        if grid:
            ax.scatter(xy[:, 0], xy[:, 1], s=(cell * 0.92) ** 2, marker="s",
                       c="#eef0f3", edgecolors="none")
        else:
            for u, v in inst["arcs"]:
                ax.plot(*zip(xy[u], xy[v]), color="#d9d9de", lw=0.6, zorder=0)
        r = (cell * 0.6) ** 2 if grid else 40
        ax.scatter(xy[inst["goals"], 0], xy[inst["goals"], 1], s=r * 0.5, marker="s", facecolors="none",
                   edgecolors=colors, linewidths=1)
        sc = ax.scatter(xy[plans[m][0], 0], xy[plans[m][0], 1], s=r, c=colors, zorder=3)
        ax.set_title(f"{m}\nSoC {sum(arr[m])}, makespan {len(plans[m]) - 1}", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off")
        pad = 0.6 if grid else 0.05
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)       # rows downwards
        scat.append(sc)
    label = fig.text(0.5, 0.02, "", ha="center", fontsize=9)

    def pos(plan, t):
        Tp = len(plan) - 1
        if t >= Tp:
            return xy[plan[Tp]]
        i = int(t); f = t - i
        return xy[plan[i]] * (1 - f) + xy[plan[i + 1]] * f

    def update(fi):
        t = times[fi]
        for sc, m in zip(scat, methods):
            sc.set_offsets(pos(plans[m], t))
        label.set_text(f"t = {t:.1f} / {T}")
        return scat + [label]
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results folder of the local runner (with plans/)")
    ap.add_argument("--list", action="store_true", help="list the jobs with saved plans")
    ap.add_argument("--job", help="job id (e.g. paper:0) or instance name")
    ap.add_argument("--methods", nargs="+", help="methods to show (default: all saved, initial first)")
    ap.add_argument("--ref", help="reference method of the comparison (default: the first shown)")
    ap.add_argument("--out", help="HTML file (default: <results>/animations/<job>.html)")
    ap.add_argument("--gif", help="also write an animated GIF")
    ap.add_argument("--all", action="store_true", help="one page per job and an index.html")
    ap.add_argument("--convert", action="store_true",
                    help="rewrite the plans of older runs (.json.gz) in the compact format")
    a = ap.parse_args(argv)
    if a.convert:
        n, before, after = convert(a.results)
        if n:
            print(f"{n} files converted: {before / 1024:.0f} KB -> {after / 1024:.0f} KB "
                  f"({100 * (1 - after / before):.0f}% smaller)")
        else:
            print("nothing to convert")
        return
    files = plan_files(a.results)
    if a.list:
        if not files:
            print("no plans saved (run the local runner with save_plans enabled)")
        for f in files:
            d = load(f)
            i = d["instance"]
            soc = {m: sum(arrivals(p, i["goals"])) for m, p in d["plans"].items()}
            print(f"{i.get('job', os.path.basename(f)):22s} {i['name']:32s} n={i['n']:<4d} "
                  + "  ".join(f"{m} {v}" for m, v in soc.items()))
        return
    adir = os.path.join(a.results, "animations")
    if a.all:
        items = []
        for f in files:
            d = load(f)
            job = d["instance"].get("job", os.path.basename(f).split(".")[0])
            out = os.path.join(adir, _safe(job) + ".html")
            write_html(d, out, a.methods, a.ref)
            rows, _, _ = compare(d, [m for m in (a.methods or list(d["plans"])) if m in d["plans"]], a.ref)
            items.append((job, d["instance"]["name"], d["instance"]["n"], os.path.basename(out), rows))
        idx = ["<!DOCTYPE html><meta charset='utf-8'><title>Animations</title>",
               "<style>body{font:14px sans-serif;margin:20px}td,th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}</style>",
               "<h1>Animations</h1><table><tr><th>job</th><th>instance</th><th>agents</th><th>Sum-of-Costs by method</th></tr>"]
        for job, name, n, href, rows in items:
            socs = "  ".join("%s %d" % (r["method"], r["soc"]) for r in rows)
            idx.append(f"<tr><td><a href='{href}'>{html.escape(job)}</a></td><td>{html.escape(name)}</td>"
                       f"<td>{n}</td><td>{html.escape(socs)}</td></tr>")
        os.makedirs(adir, exist_ok=True)
        open(os.path.join(adir, "index.html"), "w").write("\n".join(idx) + "</table>")
        print(f"{len(items)} pages in {adir}/ (index.html)")
        return
    if not a.job:
        ap.error("give --job, --list or --all")
    d = load(find(a.results, a.job))
    methods = a.methods or (["initial"] if "initial" in d["plans"] else []) + \
        [m for m in d["plans"] if m != "initial"]
    out = a.out or os.path.join(adir, _safe(a.job) + ".html")
    write_html(d, out, methods, a.ref)
    rows, _, ref = compare(d, [m for m in methods if m in d["plans"]], a.ref)
    print(f"{d['instance']['name']} ({d['instance']['n']} agents); reference {ref}")
    for r in rows:
        print(f"  {r['method']:16s} SoC {r['soc']:6d}  makespan {r['makespan']:4d}  "
              f"earlier {r['better']:3d}  same {r['equal']:3d}  later {r['worse']:3d}")
    print(f"animation: {out}")
    if a.gif:
        write_gif(d, a.gif, methods)
        print(f"GIF: {a.gif}")


if __name__ == "__main__":
    main()
