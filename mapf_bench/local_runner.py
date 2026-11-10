# SPDX-License-Identifier: GPL-3.0-or-later
"""Local execution of the benchmarks, in parallel.

    python -m mapf_bench.local_runner doctor
    python -m mapf_bench.local_runner run configs/smoke.json --workers 4
    python -m mapf_bench.local_runner run configs/paper.json --workers 8 --pin
    python -m mapf_bench.local_runner analyze results/paper
    python -m mapf_bench.local_runner list configs/movingai_official.json

A configuration file (JSON) lists suites of instances and the methods to
run; see configs/*.json and the README. Every instance is a job, run in its
own subprocess on one core (all the solvers are single-threaded), with a
hard timeout and an optional memory limit; jobs run in parallel up to
--workers. The main process is the only writer of <out>/results.csv, in the
format of mapf_bench.paper_suite, so `analyze` produces the same tables,
statistical tests and performance profiles. Runs resume: completed jobs are
skipped, and the methods already finished for an interrupted job are not
repeated.

Timing: parallel jobs share memory bandwidth and caches; for timings
comparable with the paper use at most one job per physical core (--pin
assigns one core to each running job, Linux only).
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zlib

FIELDS = ["job", "group", "family", "instance", "n", "V", "occupancy", "init", "soc0", "mk0",
          "lb", "method", "soc", "makespan", "time", "certified", "error"]


# ---------------------------------------------------------------------------
# configuration -> jobs
# ---------------------------------------------------------------------------
def load_config(path):
    cfg = json.load(open(path))
    cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    cfg.setdefault("time", 20.0)
    cfg.setdefault("methods", ["LNS", "LaCAM*", "LS", "Hybrid", "LNS+LS", "LaCAM*+Hybrid"])
    cfg.setdefault("timeout_factor", 2.0)     # job timeout = factor * T * methods + slack
    cfg.setdefault("timeout_slack", 120.0)
    cfg.setdefault("memory_gb", None)
    cfg.setdefault("save_plans", True)        # plans in <out>/plans/, for mapf_bench.visualize
    return cfg


def jobs(cfg):
    """List of (job id, description) for all the suites of a configuration."""
    out = []
    for si, suite in enumerate(cfg["suites"]):
        kind = suite["type"]
        if kind == "paper":
            from .paper_suite import scenarios
            groups = set(suite.get("groups", "SDMHW"))
            idx = suite.get("indices")
            for i, (g, fam, _) in enumerate(scenarios()):
                if g in groups and (idx is None or i in idx):
                    out.append((f"paper:{i}", f"{g} {fam} #{i}"))
        elif kind == "movingai":
            for m in suite["maps"]:
                for n in suite["agents"]:
                    for k in suite.get("scenarios", [1, 2, 3]):
                        out.append((f"mai:{m}:{n}:{k}:{suite.get('initial', 'pp')}:{suite.get('group', 'MAI')}",
                                    f"{m} n={n} scenario {k}"))
        elif kind == "dense":
            for m in suite["maps"]:
                for o in suite["occupancy"]:
                    for k in suite.get("scenarios", [1, 2, 3]):
                        out.append((f"dense:{m}:{o}:{k}:{suite.get('group', 'H')}",
                                    f"{m} occupancy {o} scenario {k}"))
        else:
            raise ValueError(f"unknown suite type {kind!r}")
    return out


def build(job):
    """(group, family, Instance, init label) of a job id."""
    from . import movingai as M
    kind, *a = job.split(":")
    if kind == "paper":
        from .paper_suite import scenarios
        g, fam, b = scenarios()[int(a[0])]
        inst, init = b()
        return g, fam, inst, init
    if kind == "mai":
        m, n, k, initial, g = a[0], int(a[1]), int(a[2]), a[3], a[4]
        scen = M.scen_path(m, "random", k)
        if not os.path.exists(scen):           # official scenario missing: generated one
            scen = M.scen_path(m, "gen", k)
            if not os.path.exists(scen):
                M.generate_scen(m, k, path=scen)
        if initial == "lacam":
            inst = M.instance(m, scen, n, initial=None)
            from .lacam import lacam_solve
            inst.initial_plan = lacam_solve(inst, time_limit=60, anytime=False)
            label = "LaCAM"
        else:
            inst = M.instance(m, scen, n, initial="official")
            label = "PP"
        return g, f"{m}_n{n}", inst, label
    if kind == "dense":
        m, o, k, g = a[0], float(a[1]), int(a[2]), a[3]
        scen = M.dense_scenarios(m, k)[k - 1]
        inst = M.dense_instance(m, scen, o)
        return g, f"{m}_o{o:.2f}", inst, "LaCAM"
    raise ValueError(job)


# ---------------------------------------------------------------------------
# worker: one job, in its own process
# ---------------------------------------------------------------------------
from .plan_store import plans_path, save_plan      # compact storage of the plans


def group_memory(pgid):
    """Resident memory (bytes) of all the processes of a process group, from
    `ps` (Linux and macOS); None if it cannot be measured."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0
    for line in out.splitlines():
        f = line.split()
        if len(f) == 2 and f[0] == str(pgid):
            total += int(f[1]) * 1024          # rss is in KiB
    return total


def watch_job(cmd, env, timeout, memory_gb=None, poll=0.5):
    """Run a job in its own process group; kill the whole group (the job and
    the solvers it started) when it exceeds the timeout or the memory limit.
    Returns a status string ("ok", "timeout after ...", "memory limit ...",
    "exit k: last line of stderr")."""
    import signal
    err = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=err, text=True,
                            start_new_session=True)
    t0 = time.time()
    limit = memory_gb * 2 ** 30 if memory_gb else None
    status, peak, last_check = None, 0, 0.0
    while proc.poll() is None:
        time.sleep(poll)
        if time.time() - t0 > timeout:
            status = f"timeout after {timeout:.0f} s"
        elif limit and time.time() - last_check >= poll:
            last_check = time.time()
            mem = group_memory(proc.pid)
            if mem:
                peak = max(peak, mem)
                if mem > limit:
                    status = f"memory limit exceeded ({mem / 2 ** 30:.1f} GB > {memory_gb:g} GB)"
        if status:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            break
    if status is None and proc.returncode != 0:
        err.seek(0)
        lines = err.read().strip().splitlines()
        status = f"exit {proc.returncode}: {lines[-1][:150] if lines else ''}"
    err.close()
    return status or "ok"


def worker(job, methods, T, part, done_methods, memory_gb=None, cpu=None, plans=None):
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    # memory_gb is enforced by the main process (watch_job), which also sees
    # the C++ solvers started by this process; setrlimit is not portable (macOS
    # refuses RLIMIT_AS)
    from .official_lns import is_grid_instance
    from .paper_suite import GRID_ONLY, run_method, lower_bound, skip_method
    from .plan import validate, sum_of_costs, makespan, trim

    def emit(row):
        with open(part, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    g, fam, inst, init = build(job)
    base = {"job": job, "group": g, "family": fam, "instance": inst.name, "n": inst.n,
            "V": inst.graph.N}
    if inst.initial_plan is None:
        emit({**base, "method": "none", "error": "no initial plan"})
        return
    p0 = trim(inst.initial_plan)
    validate(inst.graph, inst.starts, inst.goals, p0)
    if plans:
        save_plan(plans, inst, "initial", p0, {"group": g, "job": job, "init": init})
    base.update({"occupancy": round(inst.n / inst.graph.N, 3), "init": init,
                 "soc0": sum_of_costs(p0, inst.goals), "mk0": makespan(p0),
                 "lb": lower_bound(inst)})
    seed = zlib.crc32(inst.name.encode()) % 1000
    for m in methods:
        if m in done_methods or skip_method(m, inst, g):
            continue
        t0 = time.perf_counter()
        try:
            p, cert = run_method(m, inst, p0, T, seed)
            if p is None:
                raise RuntimeError("no plan")
            validate(inst.graph, inst.starts, inst.goals, p)
            soc, mk, err = sum_of_costs(p, inst.goals), makespan(p), ""
            if plans:
                save_plan(plans, inst, m, p)
        except MemoryError:
            soc = mk = ""; cert, err = None, "MemoryError"
        except Exception as e:       # recorded; the job continues with the next method
            soc = mk = ""; cert, err = None, f"{type(e).__name__}: {e}"[:200]
        emit({**base, "method": m, "soc": soc, "makespan": mk,
              "time": round(time.perf_counter() - t0, 3),
              "certified": "" if cert is None else int(bool(cert)), "error": err})


# ---------------------------------------------------------------------------
# main process
# ---------------------------------------------------------------------------
def _read_done(out):
    path = os.path.join(out, "results.csv")
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    methods = {}
    for r in rows:
        methods.setdefault(r["job"], set()).add(r["method"])
    finished = set()
    fp = os.path.join(out, "jobs_done.txt")
    if os.path.exists(fp):
        finished = {l.strip() for l in open(fp) if l.strip()}
    return methods, finished


def machine_info():
    """Processor, cores and memory of this machine (cited in the paper)."""
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except OSError:
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True).stdout.strip() or cpu
        except OSError:
            pass
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        mem = None
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"cpu": cpu, "cores": cores, "memory_gb": round(mem, 1) if mem else None,
            "system": f"{platform.system()} {platform.release()}", "python": platform.python_version()}


def retry_errors(out):
    """Remove the rows with an error (failed or timed-out methods) from
    results.csv and mark their jobs as not done, so that the next run executes
    only those methods again; the successful rows are kept. Returns the number
    of rows removed, by method."""
    path = os.path.join(out, "results.csv")
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path)))
    bad = [r for r in rows if r.get("error") and r["method"] != "none"]
    if not bad:
        return {}
    shutil.copy(path, path + f".{time.strftime('%Y%m%d-%H%M%S')}.bak")
    keep = [r for r in rows if r not in bad]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(keep)
    redo = {r["job"] for r in bad}
    fp = os.path.join(out, "jobs_done.txt")
    if os.path.exists(fp):
        done = [l.strip() for l in open(fp) if l.strip() and l.strip() not in redo]
        open(fp, "w").write("".join(j + "\n" for j in done))
    count = {}
    for r in bad:
        count[r["method"]] = count.get(r["method"], 0) + 1
    return count


def run(cfg, out, workers=1, pin=False, only=None, dry=False, retry=False):
    os.makedirs(out, exist_ok=True)
    prev = os.path.join(out, "config.json")
    added = []
    if os.path.exists(prev):
        try:
            old = json.load(open(prev)).get("methods", [])
            added = [m for m in cfg["methods"] if m not in old]
        except ValueError:
            pass
    if retry and not dry:
        removed = retry_errors(out)
        if removed:
            print("retrying the failed methods: " + ", ".join(f"{m} ({k})" for m, k in sorted(removed.items())))
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    json.dump({**machine_info(), "workers": workers, "pinned": pin},
              open(os.path.join(out, "machine.json"), "w"), indent=1)
    all_jobs = jobs(cfg)
    if only:
        all_jobs = [j for j in all_jobs if any(o in j[0] for o in only)]
    methods_done, finished = _read_done(out)
    if added:              # methods added to the configuration: run them on the finished jobs too
        print("new methods in the configuration: " + ", ".join(added)
              + " (only these run on the jobs already done)", flush=True)
        finished = set()
    todo = [j for j in all_jobs if j[0] not in finished]
    print(f"{len(all_jobs)} jobs, {len(all_jobs) - len(todo)} already done, {len(todo)} to run, "
          f"{workers} worker(s), T = {cfg['time']} s", flush=True)
    if dry:
        for j, d in todo:
            print(" ", j, "-", d)
        return
    csv_path = os.path.join(out, "results.csv")
    new = not os.path.exists(csv_path)
    fh = open(csv_path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    if new:
        w.writeheader(); fh.flush()
    lock = threading.Lock()
    cpus = queue.Queue()
    avail = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    for c in (avail * ((workers // len(avail)) + 1))[:workers]:
        cpus.put(c)
    timeout = cfg["timeout_factor"] * cfg["time"] * len(cfg["methods"]) + cfg["timeout_slack"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}
    t_start = time.time()
    counter = {"done": 0}

    def one(job):
        jid, desc = job
        cpu = cpus.get()
        part = tempfile.NamedTemporaryFile(prefix="job_", suffix=".jsonl", dir=out, delete=False).name
        cmd = [sys.executable, "-m", "mapf_bench.local_runner", "_worker", jid,
               "--part", part, "--time", str(cfg["time"]), "--methods", *cfg["methods"],
               "--done", *sorted(methods_done.get(jid, set()) or {"-"})]
        if pin:
            cmd += ["--cpu", str(cpu)]
        if cfg.get("save_plans", True):
            cmd += ["--plans", plans_path(out, jid)]
        t0 = time.time()
        try:
            status = watch_job(cmd, env, timeout, cfg.get("memory_gb"))
        finally:
            cpus.put(cpu)
        rows = [json.loads(l) for l in open(part) if l.strip()] if os.path.exists(part) else []
        os.unlink(part)
        with lock:
            for row in rows:
                w.writerow(row)
            got = methods_done.get(jid, set()) | {r["method"] for r in rows}
            if status != "ok":      # record the methods that did not complete
                base = rows[0] if rows else {"job": jid}
                for m in cfg["methods"]:
                    if m not in got and "none" not in got:
                        w.writerow({**{k: base.get(k, "") for k in FIELDS}, "job": jid, "method": m,
                                    "soc": "", "makespan": "", "time": "", "certified": "",
                                    "error": status})
            fh.flush()
            with open(os.path.join(out, "jobs_done.txt"), "a") as jf:
                jf.write(jid + "\n")
            counter["done"] += 1
            el = time.time() - t_start
            eta = el / counter["done"] * (len(todo) - counter["done"])
            line = " | ".join(f"{r['method']} {r.get('soc', '')}" for r in rows if r["method"] != "none")
            print(f"[{counter['done']}/{len(todo)}] {desc} ({time.time() - t0:.0f} s, {status}) {line}"
                  f"  -- elapsed {el/60:.1f} min, ETA {eta/60:.1f} min", flush=True)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    fh.close()


# ---------------------------------------------------------------------------
# environment check
# ---------------------------------------------------------------------------
def doctor():
    ok = True
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}")
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            m = __import__(mod); print(f"  {mod:10s} {m.__version__}")
        except ImportError:
            print(f"  {mod:10s} MISSING (pip install -r requirements.txt)"); ok = False
    from .cpp_local_search import binary as dpls
    from .official_lns import binary as olns
    from .lacam import binary as lacam
    for name, b, need in (("local search (cpp/dpls)", dpls(), "scripts/build_cpp.sh"),
                          ("official MAPF-LNS", olns(), "scripts/build_official_lns.sh"),
                          ("LaCAM3", lacam(), "scripts/build_lacam.sh")):
        print(f"  {name:26s} {b if b else 'MISSING (' + need + ')'}")
        ok &= b is not None
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"  cores available: {n}; core pinning: {'yes' if hasattr(os, 'sched_setaffinity') else 'no (not Linux)'}")
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        print(f"  memory: {mem:.1f} GB")
    except (ValueError, OSError, AttributeError):
        pass
    from . import movingai as M
    maps = sorted(os.listdir(os.path.join(M.ROOT, "maps")))
    off = sorted(f for f in os.listdir(os.path.join(M.ROOT, "scens")) if "-random-" in f)
    print(f"  MovingAI maps: {len(maps)}; official scenarios: {len(off)} "
          f"(python -m mapf_bench.movingai_download for the full benchmark)")
    if dpls():
        t = time.time()
        from .instances import ring
        from .cpp_local_search import cpp_local_search
        from .algorithms import order_corridor_neighborhoods
        inst = ring(6, 8, 3)
        cpp_local_search(inst, inst.initial_plan, order_corridor_neighborhoods(), vnd=True, astar=True)
        print(f"  quick check (ring example, local search): {time.time() - t:.2f} s")
    print("OK" if ok else "some components are missing")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a configuration")
    r.add_argument("config")
    r.add_argument("--out", help="output folder (default: results/<config name>)")
    r.add_argument("--workers", type=int, default=1, help="jobs in parallel (default 1)")
    r.add_argument("--pin", action="store_true", help="pin every job to one core (Linux)")
    r.add_argument("--only", nargs="+", help="run only the jobs whose id contains one of these strings")
    r.add_argument("--time", type=float, help="override the time budget T of the configuration")
    r.add_argument("--no-plans", action="store_true", help="do not save the plans (smaller output)")
    r.add_argument("--retry-errors", action="store_true",
                   help="run again the methods that failed or timed out (the good rows are kept)")
    r.add_argument("--no-analyze", action="store_true")
    r.add_argument("--update-paper", metavar="PAPER.tex",
                   help="after the run, update the numbers, tables and figure of the paper")
    r.add_argument("--compile", action="store_true", help="with --update-paper: compile the paper")
    r.add_argument("--strict", action="store_true",
                   help="with --update-paper: fail if a claim of the prose is contradicted")
    up = sub.add_parser("update-paper", help="update the paper from a results folder")
    up.add_argument("out")
    up.add_argument("--paper", required=True)
    up.add_argument("--compile", action="store_true")
    up.add_argument("--strict", action="store_true")
    e = sub.add_parser("errors", help="count the failed methods of a results folder, by error")
    e.add_argument("out")
    l = sub.add_parser("list", help="list the jobs of a configuration")
    l.add_argument("config")
    a = sub.add_parser("analyze", help="tables, tests and figures of a results folder")
    a.add_argument("out")
    sub.add_parser("doctor", help="check the installation")
    db = sub.add_parser("dashboard", help="browse the results, animate plans and compare solvers")
    db.add_argument("roots", nargs="*", default=["results"])
    db.add_argument("--port", type=int, default=8765)
    db.add_argument("--no-browser", action="store_true")
    wk = sub.add_parser("_worker")       # internal: one job
    wk.add_argument("job")
    wk.add_argument("--part", required=True)
    wk.add_argument("--time", type=float, required=True)
    wk.add_argument("--methods", nargs="+", required=True)
    wk.add_argument("--done", nargs="*", default=[])
    wk.add_argument("--memory-gb", type=float)
    wk.add_argument("--cpu", type=int)
    wk.add_argument("--plans")
    x = ap.parse_args(argv)
    if x.cmd == "dashboard":
        from .dashboard import main as dash
        dash(x.roots + ["--port", str(x.port)] + (["--no-browser"] if x.no_browser else []))
        return
    if x.cmd == "doctor":
        sys.exit(0 if doctor() else 1)
    if x.cmd == "_worker":
        worker(x.job, x.methods, x.time, x.part, set(x.done) - {"-"}, x.memory_gb, x.cpu, x.plans)
        return
    if x.cmd == "errors":
        rows = list(csv.DictReader(open(os.path.join(x.out, "results.csv"))))
        bad = {}
        for r in rows:
            if r.get("error") and r["method"] != "none":
                key = (r["method"], r["error"].split(":")[0] if "binary not found" not in r["error"]
                       else "official MAPF-LNS binary not found")
                bad[key] = bad.get(key, 0) + 1
        if not bad:
            print("no failed methods")
        for (m, err), k in sorted(bad.items()):
            print(f"{k:4d}  {m:14s} {err}")
        return
    if x.cmd == "update-paper":
        from .paper_update import run as update
        update(x.out, x.paper, x.compile, x.strict)
        return
    if x.cmd == "analyze":
        from .paper_suite import analyze
        cfgp = os.path.join(x.out, "config.json")
        T = json.load(open(cfgp))["time"] if os.path.exists(cfgp) else None
        print(analyze(x.out, T))
        return
    cfg = load_config(x.config)
    if x.cmd == "list":
        run(cfg, os.path.join("results", cfg["name"]), dry=True)
        return
    if x.time:
        cfg["time"] = x.time
    if x.no_plans:
        cfg["save_plans"] = False
    out = x.out or os.path.join("results", cfg["name"])
    if x.update_paper and not os.path.isfile(x.update_paper):   # check before hours of jobs
        raise SystemExit(f"paper not found: {x.update_paper} (relative to {os.getcwd()}); "
                         "nothing was run. Check the path given to --update-paper.")
    run(cfg, out, x.workers, x.pin, x.only, retry=x.retry_errors)
    if x.update_paper:
        from .paper_update import run as update
        update(out, x.update_paper, x.compile, x.strict)
    elif not x.no_analyze:
        from .paper_suite import analyze
        print(analyze(out, cfg["time"]))
        print(f"tables and figures in {out}/ (results.md, tables.tex, performance_profiles.pdf)")


if __name__ == "__main__":
    main()
