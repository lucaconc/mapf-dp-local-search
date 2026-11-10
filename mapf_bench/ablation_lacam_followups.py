# SPDX-License-Identifier: GPL-3.0-or-later
"""Follow-ups of LaCAM*: from the LaCAM* solutions cached by ablation_ls (10 s),
run the local search alone for 10 s (lacam_ls.csv). See also lacam_lns.csv
(MAPF-LNS alone for 10 s), produced by the same procedure with paper_suite._lns."""
import csv, json, os
from mapf_bench.paper_suite import scenarios
from mapf_bench.ablation_ls import _ls
from mapf_bench.algorithms import order_corridor_neighborhoods
from mapf_bench.plan import sum_of_costs
out = "ablation_ls/lacam_ls.csv"
done = {r["idx"] for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
fh = open(out, "a", newline=""); w = csv.writer(fh)
if not done: w.writerow(["idx", "group", "instance", "soc_lacam", "soc", "time"])
for i, (g, fam, b) in enumerate(scenarios()):
    p = f"ablation_ls/lacam_plans/{i}.json"
    if str(i) in done or not os.path.exists(p): continue
    inst, _ = b()
    p0 = [tuple(x) for x in json.load(open(p))]
    q, st = _ls(inst, p0, order_corridor_neighborhoods(), 10.0)
    w.writerow([i, g, inst.name, sum_of_costs(p0, inst.goals), sum_of_costs(q, inst.goals), round(st["time"], 2)]); fh.flush()
    print(i, inst.name, sum_of_costs(p0, inst.goals), "->", sum_of_costs(q, inst.goals), flush=True)
