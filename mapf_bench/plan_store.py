# SPDX-License-Identifier: GPL-3.0-or-later
"""Compact storage of instances and plans.

Format (version 2): one file per job, <job>.plans.xz: xz-compressed bytes
made of a magic string, a JSON index and the raw data of a few integer
arrays (each stored with the smallest unsigned type that fits):
  * meta        JSON header (name, family, number of agents and nodes, the
                labels of the stored plans, extra fields such as job, group);
  * starts, goals;
  * the graph:  for a 4-connected undirected grid, only a bitmap of the free
                cells (rows x cols) -- the nodes are the free cells in
                row-major order and the arcs are implied; otherwise the
                coordinates (if any) and the arcs, sorted;
  * every plan as run lengths: for each agent, the sequence of nodes it
                visits and the number of steps it stays at each, which is
                much smaller than the configurations at every time step,
                since most agents wait most of the time.
Decoding is exact (tested). Version-1 files (<job>.json.gz, JSON of the
configurations) are still read, and `convert` rewrites them in version 2.
"""
import gzip
import json
import lzma
import os

import numpy as np

EXT = ".plans.xz"
MAGIC = b"MAPFPLAN2\n"
OLD_EXT = ".json.gz"


def plans_path(out, job):
    return os.path.join(out, "plans", job.replace(":", "_").replace("/", "_") + EXT)


def _int(a):
    a = np.asarray(a, dtype=np.int64)
    if a.size == 0:
        return a.astype(np.uint8)
    lo, hi = int(a.min()), int(a.max())
    for dt in (np.uint8, np.uint16, np.uint32):
        if lo >= 0 and hi <= np.iinfo(dt).max:
            return a.astype(dt)
    return a


# ---------------------------------------------------------------------------
# instance <-> arrays
# ---------------------------------------------------------------------------
def _grid4(N, arcs, coords):
    """Bitmap of free cells if the graph is exactly the undirected 4-grid on
    coordinates numbered in row-major order; None otherwise."""
    if not coords:
        return None
    rc = [tuple(coords[str(v)]) if isinstance(coords, dict) else tuple(coords[v]) for v in range(N)]
    if rc != sorted(rc):
        return None
    cell = {c: v for v, c in enumerate(rc)}
    want = set()
    for (r, c), v in cell.items():
        for y in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if y in cell:
                want.add((v, cell[y]))
    if want != {tuple(a) for a in arcs}:
        return None
    R = max(r for r, _ in rc) + 1
    C = max(c for _, c in rc) + 1
    bm = np.zeros((R, C), dtype=bool)
    for r, c in rc:
        bm[r, c] = True
    return bm


def _encode_instance(inst):
    N = inst["N"]
    arcs = sorted(tuple(a) for a in inst["arcs"])
    arr = {"starts": _int(inst["starts"]), "goals": _int(inst["goals"])}
    bm = _grid4(N, arcs, inst.get("coords"))
    if bm is not None:
        arr["grid"] = np.packbits(bm, axis=None)
        arr["grid_shape"] = np.array(bm.shape, dtype=np.uint32)
        kind = "grid4"
    else:
        a = np.array(arcs, dtype=np.int64).reshape(-1, 2)
        arr["arc_u"] = _int(np.diff(a[:, 0], prepend=0))      # sorted: small deltas
        arr["arc_v"] = _int(a[:, 1])
        if inst.get("coords"):
            co = np.array([inst["coords"][str(v)] for v in range(N)], dtype=np.int64)
            arr["coords"] = _int(co)
        kind = "arcs"
    meta = {k: v for k, v in inst.items() if k not in ("arcs", "coords", "starts", "goals")}
    meta["graph"] = kind
    return meta, arr


def _decode_instance(meta, z):
    inst = {k: v for k, v in meta.items() if k not in ("graph", "labels", "format")}
    N = int(meta["N"])
    inst["starts"] = z["starts"].astype(int).tolist()
    inst["goals"] = z["goals"].astype(int).tolist()
    if meta["graph"] == "grid4":
        R, C = (int(x) for x in z["grid_shape"])
        bm = np.unpackbits(z["grid"])[: R * C].reshape(R, C).astype(bool)
        rc = [(int(r), int(c)) for r, c in zip(*np.nonzero(bm))]       # row-major
        cell = {c: v for v, c in enumerate(rc)}
        arcs = []
        for (r, c), v in cell.items():
            for y in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if y in cell:
                    arcs.append([v, cell[y]])
        inst["arcs"] = sorted(arcs)
        inst["coords"] = {str(v): [r, c] for v, (r, c) in enumerate(rc)}
    else:
        u = np.cumsum(z["arc_u"].astype(np.int64))
        inst["arcs"] = [[int(a), int(b)] for a, b in zip(u, z["arc_v"].astype(np.int64))]
        inst["coords"] = ({str(v): [int(x) for x in z["coords"][v]] for v in range(N)}
                          if "coords" in z else None)
    return inst


# ---------------------------------------------------------------------------
# plans <-> run lengths
# ---------------------------------------------------------------------------
def _encode_plan(plan):
    P = np.asarray(plan, dtype=np.int64)              # (T+1) x n
    nodes, steps, runs = [], [], []
    for k in range(P.shape[1]):
        col = P[:, k]
        ch = np.flatnonzero(np.diff(col)) + 1
        s = np.concatenate([[0], ch])
        e = np.concatenate([ch, [len(col)]])
        nodes.append(col[s]); steps.append(e - s); runs.append(len(s))
    return {"runs": _int(runs), "nodes": _int(np.concatenate(nodes)),
            "steps": _int(np.concatenate(steps))}


def _decode_plan(runs, nodes, steps):
    runs = runs.astype(np.int64); nodes = nodes.astype(np.int64); steps = steps.astype(np.int64)
    cols, i = [], 0
    for r in runs:
        cols.append(np.repeat(nodes[i:i + r], steps[i:i + r])); i += r
    P = np.stack(cols, axis=1)
    return [list(map(int, x)) for x in P]


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------
def _write(path, inst, plans):
    meta, arr = _encode_instance(inst)
    meta["format"] = 2
    meta["labels"] = list(plans)
    for i, (label, plan) in enumerate(plans.items()):
        for k, v in _encode_plan(plan).items():
            arr[f"p{i}_{k}"] = v
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(lzma.compress(_pack(meta, arr), preset=9 | lzma.PRESET_EXTREME))
    os.replace(tmp, path)


def _pack(meta, arrays):
    """magic | 4-byte length | JSON {meta, arrays: {name: [dtype, shape, offset]}} | data"""
    index, chunks, off = {}, [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        index[name] = [a.dtype.str, list(a.shape), off]
        chunks.append(a.tobytes()); off += a.nbytes
    head = json.dumps({"meta": meta, "arrays": index}, separators=(",", ":")).encode()
    return MAGIC + len(head).to_bytes(4, "little") + head + b"".join(chunks)


def _unpack(raw):
    if not raw.startswith(MAGIC):
        raise ValueError("not a plans file")
    k = len(MAGIC)
    hl = int.from_bytes(raw[k:k + 4], "little")
    head = json.loads(raw[k + 4:k + 4 + hl].decode())
    data = memoryview(raw)[k + 4 + hl:]
    arrays = {}
    for name, (dt, shape, off) in head["arrays"].items():
        dtype = np.dtype(dt)
        cnt = int(np.prod(shape)) if shape else 1
        arrays[name] = np.frombuffer(data, dtype=dtype, count=cnt, offset=off).reshape(shape)
    return head["meta"], arrays


def load(path):
    """{"instance": {...}, "plans": {label: [configuration, ...]}} from a file
    of either version."""
    if path.endswith(OLD_EXT):
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with open(path, "rb") as fh:
        meta, z = _unpack(lzma.decompress(fh.read()))
    plans = {label: _decode_plan(z[f"p{i}_runs"], z[f"p{i}_nodes"], z[f"p{i}_steps"])
             for i, label in enumerate(meta["labels"])}
    return {"instance": _decode_instance(meta, z), "plans": plans}


def instance_dict(inst, meta=None):
    G = inst.graph
    return {"name": inst.name, "family": inst.family, "n": inst.n, "N": G.N,
            "arcs": [[u, v] for u in range(G.N) for v in G.out[u] if v != u],
            "coords": {str(v): list(rc) for v, rc in G.coords.items()} if G.coords else None,
            "starts": list(inst.starts), "goals": list(inst.goals), **(meta or {})}


def save_plan(path, inst, label, plan, meta=None):
    """Add (or replace) the plan of a method in the file of a job (version 2).
    A version-1 file of the same job is converted and removed."""
    old = path[: -len(EXT)] + OLD_EXT if path.endswith(EXT) else None
    data = None
    if os.path.exists(path):
        data = load(path)
    elif old and os.path.exists(old):
        data = load(old)
    if data is None:
        data = {"instance": instance_dict(inst, meta), "plans": {}}
    elif meta:
        data["instance"].update(meta)
    data["plans"][label] = [list(map(int, x)) for x in plan]
    _write(path, data["instance"], data["plans"])
    if old and os.path.exists(old):
        os.remove(old)


def plan_files(results):
    d = os.path.join(results, "plans")
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith((EXT, OLD_EXT)))


def convert(results):
    """Rewrite the version-1 files of a results folder in version 2.
    Returns (files converted, bytes before, bytes after)."""
    n = before = after = 0
    for f in plan_files(results):
        if not f.endswith(OLD_EXT):
            continue
        data = load(f)
        new = f[: -len(OLD_EXT)] + EXT
        _write(new, data["instance"], data["plans"])
        if load(new) != _normalized(data):
            os.remove(new)
            raise RuntimeError(f"conversion of {f} is not exact; the original is kept")
        before += os.path.getsize(f); after += os.path.getsize(new)
        os.remove(f)
        n += 1
    return n, before, after


def _normalized(data):
    inst = dict(data["instance"])
    inst["arcs"] = sorted([list(a) for a in inst["arcs"]])
    return {"instance": inst, "plans": data["plans"]}
