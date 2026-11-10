# mapf_bench — MAPF-LNS vs DP-based local search

A self-contained test suite that compares MAPF-LNS with the local-search
algorithms of *Dynamic Programming-Based Local Search for Multi-Agent Path
Finding on Directed Graphs* (Saccani, Ardizzoni, Consolini, Locatelli).
Pure Python 3 + NumPy; matplotlib is optional (plots), pytest for the tests.

```
pip install numpy matplotlib pytest
python -m pytest -q                                   # 340 tests; C++, LaCAM and official-LNS tests skipped if not built
python -m mapf_bench.run --suite quick --out results  # paper's examples
python -m mapf_bench.run --suite all --seeds 3 --time-limit 30 --out results
```

The runner writes `results/results.csv` (one row per instance and algorithm),
`results/summary.md` (aggregated tables) and `results/makespan_reduction.png`.

## Running the benchmarks locally

```
scripts/setup_local.sh                 # venv, dependencies, C++ builds, tests, check
source .venv/bin/activate
python -m mapf_bench.local_runner doctor
python -m mapf_bench.local_runner run configs/smoke.json --workers 4          # a few minutes
python -m mapf_bench.local_runner run configs/paper.json --workers 8 --pin    # protocol of the paper
python -m mapf_bench.movingai_download                                        # official MovingAI files
python -m mapf_bench.local_runner run configs/movingai_official.json --workers 8 --pin
python -m mapf_bench.local_runner analyze results/paper
```

Requirements: Python >= 3.9, a C++17 compiler, git, cmake >= 3.16 (CMake 4.x
works: the build script raises the minimum version declared by the upstream
MAPF-LNS, which CMake >= 4.0 would refuse; Boost >= 1.89 works too: the script
stops requesting the removed boost_system component; Eigen 4 and 5 work too:
the script drops the "Eigen3 3.3" version requirement; extra CMake arguments
can be passed with MAPF_LNS_CMAKE_ARGS), and, for
the official MAPF-LNS, Boost (program_options, system, filesystem) and Eigen
(`apt install build-essential cmake git python3-venv libboost-program-options-dev
libboost-system-dev libboost-filesystem-dev libeigen3-dev`, or
`brew install cmake boost eigen python`). A `Dockerfile` gives the same
environment: `docker build -t mapf_bench .` then
`docker run --rm -v "$PWD/results:/mapf_bench/results" mapf_bench run configs/paper.json --workers 8 --pin`.

**Configurations** (`configs/*.json`): `time` (budget T per method, seconds),
`methods` (any of LNS, LaCAM*, LS, Hybrid, LNS+LS, LaCAM*+Hybrid),
`memory_gb` (per-job limit, optional: the main process measures the memory of
the job and of the solvers it starts, with `ps`, and stops the job above the
limit; this works on Linux and macOS), `timeout_factor` and `timeout_slack`
(a job is killed after factor x T x methods + slack seconds), and `suites`:

| suite type | fields | instances |
|---|---|---|
| `paper` | `groups` (S, D, M, H, W), `indices` (optional) | the 100 scenarios of the paper protocol |
| `movingai` | `maps`, `agents`, `scenarios`, `initial` (`pp` or `lacam`), `group` | first n agents of the official random scenarios (generated ones if missing) |
| `dense` | `maps`, `occupancy`, `scenarios`, `group` | small maps at a given fraction of occupied cells |

**Execution.** Every instance is a job run in its own process on one core,
up to `--workers` at a time; `--pin` assigns one core to each running job
(Linux). Results go to `results/<name>/results.csv`; runs resume
(`jobs_done.txt`), and the methods already finished in an interrupted job
are not repeated. Timeouts and errors are recorded as rows with an `error`
field; `python -m mapf_bench.local_runner errors results/<name>` counts them, and
`run ... --retry-errors` runs only the failed methods again (for instance
after building a missing binary), keeping the successful rows. For timings comparable with the paper use at most one worker per
physical core. Without the official MAPF-LNS, LNS on grids records an error
unless `MAPF_BENCH_LNS_FALLBACK=1` (then the re-implementation is used).
`analyze` writes `results.md`, `tables.tex` and `performance_profiles.pdf`.

## Dashboard: browse all results and compare solvers

```
python -m mapf_bench.dashboard                    # all the runs under ./results
python -m mapf_bench.local_runner dashboard results/ other/results --port 8765
```

A local web page (served on 127.0.0.1 only, standard library only) with
four views:

* **Runs**: every folder with a `results.csv` under the given folders, with
  configuration, time budget, solvers, saved plans, errors and machine;
  select one or more runs.
* **Instances**: the instances of the selected runs, filterable by group,
  family and name and sortable by any column, with the Sum-of-Costs of
  every solver (best in bold, failed methods marked). Click an instance
  with saved plans to animate it.
* **Animation**: choose the solvers to show and the reference; when the
  instance exists in several selected runs, the plans of all of them can be
  compared side by side (labels `run/solver`), e.g. the same solver with two
  time budgets or on two machines.
* **Compare solvers**: for a group, the geometric mean of SoC/LB of every
  solver, a scatter plot of solver A against solver B instance by instance
  (with wins/ties/losses and the median difference), and the performance
  profiles of the chosen solvers.

## Animating and comparing solutions

The local runner saves, for every job, the instance and the plan of every
method (and the initial plan, as `initial`) in
`results/<name>/plans/<job>.plans.xz` (`--no-plans` to disable). The format
(`mapf_bench/plan_store.py`) is compact and exact: plans are stored as run
lengths (the nodes each agent visits and how long it stays), grids as a
bitmap of free cells instead of the list of arcs, all as small integer arrays
compressed with xz; files are about 3 times smaller than gzip-compressed JSON
overall, and about 10 times smaller on MovingAI maps. Plans saved by older
versions (`.json.gz`) are still read; convert them with
`python -m mapf_bench.visualize results/<name> --convert`.

```
python -m mapf_bench.visualize results/paper --list                 # jobs with saved plans
python -m mapf_bench.visualize results/paper --job paper:0          # all methods of a job
python -m mapf_bench.visualize results/paper --job paper:63 \
    --methods initial LaCAM* LaCAM*+Hybrid --ref LaCAM* --gif dense.gif
python -m mapf_bench.visualize results/paper --all                  # one page per job + index
```

The HTML page (`results/<name>/animations/<job>.html`) is self-contained and
opens in any browser: the methods are animated side by side with the same
color for every agent (play/pause, time slider, single steps, speed, trails,
goals); clicking an agent highlights it in all panels and shows its arrival
time with each method. Below, a table compares Sum-of-Costs, makespan and
the per-agent arrival times with a reference method (`--ref`), and a chart
shows the number of agents not yet arrived over time. Grids use their
coordinates; other graphs get a deterministic spring layout, with arrows on
one-way arcs. `--gif` writes the same comparison as an animated GIF.

## Automatic paper update

After a run, the numbers, tables and figure of the benchmark section of the
paper can be updated from the results:

```
python -m mapf_bench.local_runner run configs/paper.json --workers 8 --pin \
    --update-paper ../paper/LocalSearch_new.tex --compile
python -m mapf_bench.local_runner update-paper results/paper --paper ../paper/LocalSearch_new.tex
python -m mapf_bench.paper_update results/paper --paper ../paper/LocalSearch_new.tex --strict
```

* **Numbers.** Every number cited in the benchmark section and in the
  conclusion is a macro (`\BenchRatioSLs`, `\BenchPairAllLacamHybVsLacamW`,
  ...), defined in `bench_numbers.tex` next to the paper, which the paper
  inputs in its preamble. A few sentences that describe a pattern (the gain
  by occupancy, the certificates, the time overruns, the excluded instances)
  are generated from the data.
* **Tables.** The blocks between `%<AUTO:bench_results>` and
  `%</AUTO:bench_results>` (and `bench_pairs`) are regenerated whole.
* **Figure.** `performance_profiles.pdf` is copied to `images/tests/`.
* **Claims.** The prose is never rewritten. Every qualitative claim of the
  discussion (e.g. "LS alone is significantly worse than LNS on the MovingAI
  maps") is checked against the new results; `paper_update/report.md` lists
  each as confirmed or CONTRADICTED, and `--strict` exits with an error when a
  claim must be revised by hand.
* **Safety.** The paper is backed up and a unified diff is written to
  `paper_update/` before every update; a second update with the same results
  changes nothing. `--compile` runs latexmk (or pdflatex twice).

Macro names are `Bench` + kind + group (S, D, M, H, W, All) + method code
(Lns, Lacam, Ls, Hyb, LnsLs, LacamHyb): `BenchRatio`, `BenchBest`, `BenchGap`,
`BenchTime`, `BenchCert`, and `BenchPair<G><A>Vs<B>{W,T,L,N,Med,P}`; see
`bench_numbers.tex` for the full list. A macro the results cannot provide is
set to **??** and listed in the report.

Note: with the current results the time overruns of group H reach about
230% of the budget on the instances with 243-251 agents; the explanation
given in the paper (table construction and time checks every 256
expansions) should be verified before submission.

## What is compared

| name | algorithm |
|---|---|
| `lns_soc` | MAPF-LNS re-implementation, Sum-of-Costs objective (as published) |
| `lns_mk` | the same with a makespan objective (fairer on plan length) |
| `ls_pointwise`, `ls_agentB`, `ls_alt` | pointwise, agent distance (Ordering B), and their alternation (Sec. 5.2) |
| `ls_combined` | combined neighborhood, R_sm=1, R_ag=2 (Sec. 5.3) |
| `ls_lag` | route-lag distance, R=2, lambda=1 |
| `ls_rr` | route-restricted lag distance (routes fixed, timing free), R=3 |
| `ls_gc` | grouped-clock distance, R=1, automatic groups (at most 2) |
| `ls_new_alt` | alternation of `ls_rr`, `ls_gc`, and combined (R_ag=1, R_sm=0) |
| `a+b` | pipeline: `b` applied to the output of `a`, e.g. `lns_soc+ls_combined` |

All local searches minimize the scaled makespan with the exact
neighborhood search of Algorithm 1 (Dijkstra on the sequentialized
automaton), iterated as in Algorithm 3; several neighborhoods are cycled
until none improves. Every returned plan is validated (arcs, vertex
collisions, swaps, endpoints).

## Instance families (`--suite`)

| suite | instances | initial plan |
|---|---|---|
| `examples` | merge (Example `ex:merge`) and ring (Example `ex:ring`) constructions | hand-built reference plans |
| `grid` | 9x9 grids with K = 0, 1, 2 bottlenecks, 8 agents (Sec. 5.3) | prioritized planning (PP) |
| `grid_serial` | same maps | PP, serialized: one agent moves at a time, as in rule-based solvers |
| `oneway` | 6x6 one-way (Manhattan) grids, 6 and 10 agents | PP and serialized PP |
| `random` | random strongly connected digraphs, 30 nodes, 6 and 10 agents (Sec. 5.2) | PP and serialized PP |
| `multiring` | 2-3 almost full one-way rings joined by two-way bridges | serialized PP |
| `quick` / `all` | `examples` / everything | |

## Families that expose the weaknesses of MAPF-LNS (`mapf_bench.explore`)

```
python -m mapf_bench.explore --family rings rings_hub convoy --seeds 8 --out explore
python -m mapf_bench.explore --summary-only --out explore   # explore_summary.md
```

| family | construction | weakness of MAPF-LNS |
|---|---|---|
| `rings` | 2-3 fully occupied one-way rings of 6-12 agents; the initial plan rotates one ring at a time | a full ring moves only by a joint rotation of all its agents, which needs a destroy set containing the whole ring |
| `rings_hub` | the same, plus free agents on a hub that move first | as above; free agents can be improved, the rings cannot |
| `convoy` | long-haul agents queue behind short-haul ones at a merge | letting them go first lowers the makespan but leaves the Sum-of-Costs unchanged |
| `dense_grid`, `dense_oneway`, `dense_random` | 4x5 grid, 4x4 one-way grid, 16-node random digraph with 50-90% of the nodes occupied; initial plan from a backward random walk (seed mod 5 selects the density) | at density >= 0.8 almost every repair by prioritized planning fails |

```
python -c "from mapf_bench.explore import density_table; density_table('explore/explore.csv','explore')"
```

## C++ local search

`cpp/dpls.cpp` is a C++17 implementation of the local search (Algorithms 1
and 3, all neighborhoods), with no dependencies. It reproduces the Python
implementation exactly: same neighborhoods, same Ordering B tie-breaking,
same successor order, so it returns identical plans (checked by 30 tests and
on 100 runs over 7 families). It is 7-9x faster than the Python code at equal
results; states are stored in a compact pool of 16-bit words with an
open-addressing hash table, which made it a further ~4x faster (up to 5.6x
on the largest searches) and cut memory by ~5x with respect to the first
C++ version, with identical plans.

```
scripts/build_cpp.sh                     # builds cpp/dpls
```

Every Python local search `ls_<x>` has a C++ twin `cls_<x>` in the registry
(10x the expansion budget). `mapf_bench.cpp_local_search.cpp_local_search`
has the same interface as `local_search.local_search`.

## Variable neighborhood descent over all local searches

`vnd_soc` (registry) runs every neighborhood of the paper, from the cheapest
to the most expensive (pointwise, path, route-lag, route-restricted,
grouped clocks with 2 groups, agent distance, combined, grouped clocks with
4 groups at radius 0); after every improvement it restarts from the first,
and it stops when none improves, so the result is a local optimum of all of
them (tested). The objective is the Sum-of-Costs. The exact formulation of
Appendix A.3 (`--objective soc`, `vnd_soc_exact`) keeps a wait counter per
agent and its state space explodes on long plans; `vnd_soc` uses
`--objective soc_proxy`: waiting at the goal costs 0 and any other action 1
(no counters), and a plan is accepted only if its true Sum-of-Costs
decreases, so the Sum-of-Costs never increases.

```
python -m mapf_bench.compare_vnd run --instances 5      # VND vs MAPF-LNS, 10 families
python -m mapf_bench.compare_vnd summary                # vnd_results/vnd_summary.md
```

### Adaptive radius

`avnd_soc` is the same VND with adaptive radii
(`mapf_bench.algorithms.adaptive_neighborhoods`): every neighborhood starts
at its smallest radius; when a search ends without improvement within its
state budget it is repeated with the radius increased by one, up to a
maximum (pointwise 1-2, path 1-3, route-lag 1-4, route-restricted 1-8,
grouped clocks with 2 groups 0-3, agent distance 1-3, combined R_ag 1-3,
grouped clocks with 4 groups 0-1). After an improvement all radii return to
their minimum (`avndk_soc`: each neighborhood keeps its last radius). In the
C++ binary an adaptive neighborhood is written `TEMPLATE@RMIN-RMAX`, with
`{R}` in the template, e.g. `gc:{R}:1.0:none:1@1-8`.

```
python -m mapf_bench.compare_vnd run --instances 5 --out avnd_results \
    --strategies lns vnd avnd lns+avnd avnd+lns avndk avndk+lns
```

### Ablation: are all neighborhoods useful?

`python -m mapf_bench.ablation run --instances 3` runs the adaptive VND and,
for each neighborhood, the VND without it, and reports per-neighborhood
statistics (searches, improvements, gain, time; printed by the C++ binary as
`slot_*`). Three neighborhoods are contained in others at equal radius
(combined in agent distance; path and route-restricted in route-lag; 2 groups
refined by 4 groups). `avndr_soc` keeps only route-lag (radius 1-8), grouped
clocks with 4 groups (0-1) and the agent distance (1-3); on the 50 instances
of `compare_vnd` it is equivalent to the full VND (8/33/9), with the same
results against MAPF-LNS.

### A* search

With `--astar` (Python: `cpp_local_search(..., astar=True)`; registry:
`avndra_soc`, `avnda_soc`) every neighborhood search is A* instead of
Dijkstra, with a consistent heuristic that ignores collisions: the
distances of the agents to their goals (for the makespan, the number of
joint steps they imply; for the Sum-of-Costs, their sum), computed inside
the reference routes for the route-restricted distance, and, for the agent
distance with a full set D, the reference schedule of the agents that can
no longer deviate. Each search finds the same optimal cost (tested against
Dijkstra and against brute force), with fewer states, possibly a different
optimal plan. On the 50 instances of `compare_vnd`, the reduced adaptive VND
with A* takes half the time of Dijkstra (139 s vs 269 s in total) and gets a
lower Sum-of-Costs on 17 instances (higher on 3), because searches that
exceeded their budget now complete.

### Free-window neighborhood

`window:W[:K[:multi]]` (C++; Python `cpp_local_search.Window(W, K, multi,
budget)`): the plan follows the reference up to x^{t0}, moves freely for at
most W rounds, lands on the latest matching configuration x^{t1} of the
reference and follows it to the end; free segments shorter than t1 - t0 cut
that part out. A single window by default (`multi`: a new window after each
return); K bounds the agents that move during the window. In the last free
round partial rounds are pruned to prefixes of reference configurations.
Size (`neighborhood_size`, `--enumerate`): W = 1 has O(n T^2) states
whatever the degree; W >= 2 can have 2^n states already with out-degree one
(`instances.pairs_instance`); with K fixed the size is polynomial
(O(n^{K+1}) at fixed T and bounded degree). See the proposition in the
outputs (`window_snippet.tex`). With W large
enough it contains every plan (tested: global optimum). A neighborhood may
have its own state budget (`SPEC#K`). `avndraw_soc` puts `window:1#200000`
before the reduced set. Findings: a single W = 1 search compresses
serialized plans with few agents dramatically (e.g. Sum-of-Costs 360 -> 62 in
0.2 s), but for 16 or more agents, or W >= 2, the free rounds exceed any
budget, and rejoining needs all agents to skip the same stretch of the
reference at once; inside the VND it changes the final result on 9 of 90
instances (better 2, worse 7).

### Free set (destroy-set neighborhood)

`FreeSet(K, R, lam, restricted, candidates, budget)` (C++ spec
`gc:R:lam:none:RS:2:4:K:{R}@0-C!`): the route-lag distance in which the K
agents of a candidate set have no term, so they move freely while the others
keep their routes (lag at most R). Candidate sets are computed LNS-style
from the current reference (each agent by decreasing delay with the K-1
agents it interacts with most; the agents through the most crowded node);
the search tries them in turn (`!`: move on also after a budget failure).
Polynomial for fixed K (anchored separable distance, Theorem thm_poly).
`avndrf_soc` = reduced set + `FreeSet(2, 0, restricted, 20 candidates,
3e5 states)`: never worse than `avndra_soc` on the 90 instances (15 better,
p = 1e-4), mostly on grid maps with PP plans, but still behind MAPF-LNS
there.

### Order-based neighborhoods (retiming and order swaps)

`Retime()` (C++ spec `retime`): same routes and same order of visits at every
node, earliest schedule by longest paths: every arrival time is minimal, so
makespan and Sum-of-Costs are optimal among the plans with those routes and
orders. `OrderSwap()` (`swap`): exchange two consecutive visits of a node, or
let one agent overtake another at every node where it directly follows it,
then retime; best improvement. Neither is dynamic programming (as state sets
they are exponential) but both are solved in polynomial time.
`avndro_soc` = retime + swap + reduced set; `avndrof_soc` = the same + free
set. Retiming alone solves the ring examples; pair swaps solve the convoys;
on the serialized warehouse the Sum-of-Costs drops from 647 to 265 (LNS 238),
and on dense grids `avndrof_soc` beats MAPF-LNS on average (77.0 vs 106.3).

### Corridors for the free agents

`FreeSet(..., corridor=c)` (C++ spec `gc:R:lam:none:RS:2:4:K:{R}:c@0-C!`)
restricts the free agents to the nodes within distance c of their reference
routes; the A* heuristic uses distances inside the corridor. One candidate
pair on the warehouse map: 4.1 million states without corridor, 59,540 with
c = 1. `avndrofc_soc` = retiming + order swaps + reduced set + pairs with
c = 1: same quality as unrestricted pairs where both fit the budget, better
on random-obstacle maps (348.7 vs 361.5), and after MAPF-LNS never worse than
MAPF-LNS (16 better / 74 equal / 0 worse on the 90 instances).

### Hybrid LNS / local search

`mapf_bench.hybrid.hybrid(inst, plan, ...)`: the local search is run to local
optimality and short LNS phases are used as perturbations; an LNS
improvement invalidates local optimality and the local search is run again;
the algorithm stops after `max_fail` LNS phases without improvement and
always ends with a local search run, whose `certified` flag says whether it
proved local optimality (no neighborhood improves, within time and budgets).
Strategies `hyb` and `hyb2` in `compare_vnd` (the latter starts with LNS,
uses 4 s LNS phases, the cheap neighborhoods during the alternation and the
corridor pairs in the final run). Results: `final_results/hybrid_summary.md`
(hyb2 vs MAPF-LNS with the same 20 s: 15 better, 75 equal, 0 worse).

### Speed

The C++ search computes the A* heuristic incrementally, reuses its buffers,
grows its hash table on demand, and evaluates order swaps by patching arrays
with a queue-based Bellman-Ford (identical plans, 5-15% faster).
`keep_radius=True` keeps the radius of the last improvement of every adaptive
neighborhood (26.7 s -> 15.6 s on a dense grid, same quality on average).
The official MAPF-LNS patch adds `--stagnation N` (stop after N iterations
without improvement; also in the re-implementation), used by the hybrid
(`stagnation`, `adaptive_stagnation`). `hyb4` = hyb2 + stagnation 2000 + kept
radii: 556 s instead of 1259 s on the 90 instances, 15/71/4 against MAPF-LNS
(hyb2: 15/75/0). See `final_results/speed_summary.md`.

### Order of the LSBX escape phase

On the instances where the escapes run, a single beam call used the remaining
11-17 s; route-lag and grouped clocks took 85% of it without any improvement,
the improvements came from retiming, order swaps and the free pairs, and the
LNS/LaCAM free sets (last in the list) were never reached. On undirected graphs
LSBX now searches the free sets (pairs in corridors, then 4 and 8 agents) before
the route-lag, grouped-clock and agent ladders (`paper_suite.LSBX_FREE_FIRST`);
on directed graphs it keeps the order of LSB. Groups D, H, K (73 instances,
20 s), free sets first vs the old order: H 11 better, 22 equal, 0 worse
(p = 0.003; up to -14.9%), K 2/17/1 (dead end 170 -> 91), D 1/18/1 with one large
loss (dense_oneway_d0.8_s13 73 -> 123), hence the old order on directed graphs.


### LaCAM*+LNS+LSBX (available, not in the default methods)

`run_method("LaCAM*+LNS+LSBX")`: LaCAM* for T/2, MAPF-LNS for T/4, LSBX for T/4.
Groups H and K (44 instances; group K partly), 20 s: vs LaCAM*+LNS 14 better,
12 equal, 18 worse (p = 0.42); vs LaCAM*+LNS+LSB 4/36/4 (p = 0.78): in the
last quarter, after MAPF-LNS, the escapes of LSBX rarely run. Large gains on
some dense maps (empty-8-8 at 90%: 976 -> 861) and as many losses, so it is not
in METHODS; add it to "methods" in configs/paper.json to include it.


### LSB with the LNS/LaCAM neighborhoods (LSBX)

`run_method("LSBX")`, `paper_suite._lsb(..., extra=True)`: LSB with the free sets
of `lnsx_extra_neighborhoods()` (4 and 8 agents) added to its beam escape phase;
the escapes start whenever the exact descent has ended before the deadline,
also when some of its searches exceeded their budget (in LSB only after a
certified local optimum, which on groups M and H never happens within 20 s);
the plan is certified only if the last exact descent certified it.
Groups M and H, 20 s: unchanged on the MovingAI maps; on group H, LSBX vs LS
11 better, 19 equal, 0 worse (p = 0.003), vs LSB 7/21/2, vs LSX 9/17/4, up to
-4.9% (empty-8-8 at 95%). MAPF-LNS remains better on both groups.

### LNS/LaCAM neighborhood (LSX)

`algorithms.lnsx_neighborhoods()`: the complete method plus a free set of 4,
then 8 agents chosen by the destroy heuristics of MAPF-LNS (agent-based by
delay and interaction, intersection-based through the 5 most crowded nodes,
random with a fixed seed, interleaved; 12 candidates each), all the other
agents frozen on their reference, and the free agents replanned jointly in
their configuration space by A* with a bounded open list of 5000 states
(C++ spec suffix `%W`: beam width of one neighborhood). Methods `LSX` and
`LNS+LSX` in `paper_suite.run_method` (not in the default METHODS). On groups
M, H and K (77 instances, 20 s): LSX vs LS 9 better, 68 equal, 0 worse
(p = 0.008); dead_end 164 -> 91 (as MAPF-LNS), order reversal 136 -> 96 (MAPF-LNS
stays at 195), up to -3.5% at high occupancy; unchanged on the MovingAI maps,
where MAPF-LNS remains better (19 of 24).

### Beam escapes (LSB)

`LSB` (and `LNS+LSB`) in the protocol: the exact local search (LS) runs to a
certified local optimum, then the descent over `beam_neighborhoods()` (route-lag
up to 16, grouped clocks up to 4, up to 6 deviating agents, free sets of 3
agents in corridors of width 2) runs by A* with a bounded open list
(`--beam W` of the C++ binary, `BEAM = 5000` in `paper_suite.py`): every
search keeps only the W states with the lowest cost + heuristic, so it is not
exact (`pruned` in the statistics). An improvement sends LSB back to the
exact descent; its plan is certified if the last exact descent certified it.
On the 133 instances of all groups, LSB vs LS: 17 better, 115 equal, 1 worse,
with gains up to 38% on the dense directed graphs of group D; on the MovingAI
maps, where the exact descent does not end within 20 s, the two coincide.
Using the beam in place of the exact search (without the exact descent first)
was not better than LS and erratic. See `paper_results_v3/lsb_report.md`.
Speed: every beam neighborhood starts just above the radius certified by the
exact descent (a search at a certified radius cannot improve), and a beam
search stops as soon as the smallest key of its open list reaches the
Sum-of-Costs of the current plan (no plan it can still reach could be
accepted); exact searches are unchanged. Same results as before on 12
benchmark instances, up to 1.7x faster where LSB ends before its budget.
Searching every beam neighborhood directly at its largest radius was 2-4x
faster but lost the improvement on 5 of these 12 instances, so the radii are
still increased one at a time (`BEAM_RADII = "ladder"` in algorithms.py).
Adaptive enlargement of the radius was also tried (C++ specs `@rmin-rmax^`
geometric growth, `@rmin-rmax+cap` extension beyond rmax while the searches
are complete, `@rmin-rmax+cap*` geometric extension also after pruning;
`BEAM_RADII = "adaptive"`, `"extend"`, `"ladder+jump"`): none beats the
ladder within 20 s (12 instances: 2/2, 0/0 and 1/1 better/worse; groups D and
K, 40 instances, ladder+jump: 1 better, 2 worse, 11% more time), so the
ladder stays the default.
The binary reports `impr_at`, the [neighborhood, radius, gain] of every
accepted improvement: on 42 instances, the beam phase found its improvements
only with route-lag (radii 9-15, none at 16) and the agent distance (radii
4-6). Ladders built from these data (`BEAM_RADII = "opt"`: agent distance
first, route-lag up to 15, no grouped clocks or free sets; `"opt13"`:
route-lag up to 13) saved 3-5% of the time but were worse on 3 of the 42
instances and never better: the order matters, because the route-lag
improvements lead to plans from which the agent distance finds its own.
Further attempts at speeding up the exact search, both exact and kept but
without a measurable gain on this machine: the successors of an expansion are
hashed in a batch and their table slots prefetched before the lookups
(identical plans, 61.5 s -> 60.3 s on the harness, within the noise); and
`--early-stop` (off by default) lets exact searches stop when the smallest key
reaches the Sum-of-Costs of the current plan, as the beam searches do: same
plans and same number of searches exceeding their budget, since those searches
still have millions of states below that key when the budget runs out.

### Parallel search of free-set candidates (`--threads`)

`--threads T` (C++), `cpp_local_search(..., threads=T)`, `paper_suite.THREADS`
(default 1: the protocol stays single-threaded): the candidates of a cyclic
neighborhood (free pairs in corridors, the LNS/LaCAM free sets of 4 and 8
agents) are searched in batches of T at the same time, each with its own
neighborhood object; the improvement with the lowest objective, then the
lowest candidate index, is accepted, so the result does not depend on the
thread timing. With one thread the search is the sequential one (identical
plans); with several, the descent may take another path (best instead of
first improvement of a batch). Shared statistics are atomic and the lazy
tables of the experimental heuristics are built under a lock; build with
-pthread (scripts/build_cpp.sh). Measure the speed-up on a multi-core machine:
`python scripts/bench_threads.py 6 --group H K`.


### Chaining of forced moves (default in the protocol)

`--chain` (C++), `cpp_local_search(..., chain=True)`, `paper_suite.CHAIN = True`:
a decision with a single feasible successor (an agent whose move is forced,
e.g. bound to its reference by a small radius or frozen in a free-set
neighborhood) is applied within the same expansion instead of creating and
storing an intermediate state; chains stop at the end of a round, so complete
configurations are always stored and plan reconstruction is unchanged. Every
search still returns a plan of minimum cost (checked on the first 1-3
searches of the 12 benchmark instances); it may return another plan of the
same cost. Benchmark harness: states 2.0M -> 0.17-0.31M on the MovingAI maps
with 40 agents, 2.0M -> 28k on empty-16-16, total time 48.3 s -> 26.3 s. LS with
20 s on groups M, H, K (77 instances): 22 better, 52 equal, 3 worse
(p = 0.0002), total time 957 s -> 724 s (the MovingAI maps reach the same local
optimum in half the time); up to -29% (order reversal 136 -> 96) and -27%
(crossroads 228 -> 166). Enabled by default in LS, LSB, LSBX, LSX and the local
search phases of Hybrid; the binary's default (without --chain) is unchanged,
so the tests comparing exact plans still hold.

Faster chains (exact, identical plans): along a chain the sum of the
distances and the tie-break key are updated by the forced move instead of
being summed over all the agents, the collision marks are kept for the whole
chain (each occupied node records the decided agent that occupies it, so the
swap test is a lookup), the route-lag neighborhoods reuse the cache entry of
the unchanged configuration z, and the decision sequences are hashed
incrementally. Benchmark harness with --chain: 25.9 s -> 16.7 s (1.55x;
empty-16-16 12.5 s -> 6.4 s, corridor swap 0.36 s -> 0.15 s, MovingAI maps
1.2-1.4x).
Order swaps: most candidates are rejected because the exchange creates a
positive cycle (crossroads: 2474 of 3500, empty-16-16: 9267 of 12000), which the
solver found only after computing the affected set (80-95% of all the events).
For single-node swaps, the only constraint that goes backwards in the base
schedule is the new one from b's leaving event to a's entry, so a positive
cycle exists iff a's entry reaches b's leaving event by a path with a weight-1
constraint, within the window e0 <= e0[b's leaving event]: this quick test
rejects them at once (exact; identical plans). Harness with --chain:
17.1 s -> 15.0 s (empty-8-8 at 90% 2.4x, corridor swap 2.2x, crossroads 1.5x,
empty-16-16 1.26x); with the faster chains, 25.9 s -> 15.0 s (1.7x).
The successors of an expansion are encoded, hashed and prefetched only after
the decision on the chain, so a forced move that continues a chain is never
encoded (it was, uselessly, before); the same-agent successors of the events
are precomputed for the order swaps. Harness with --chain: 14.5 s -> 11.9 s
(empty-16-16 1.56x, corridor swap 1.45x, MovingAI maps up to 1.15x); in total
25.9 s -> 11.9 s (2.2x), identical plans. Tried and dropped: an additive hash
per position, to update the hash of a successor from a common part (slower:
three multiplications per word, against one per four words); and a cache of
the route-lag/grouped-clock tables across the radii of the same reference
(exact, but no gain: the tables are now a small share, and the copy costs
about as much as it saves). On the MovingAI maps the local search reaches its
local optimum in 3-7 s and stops, so further speed there does not change the
results within 20 s; it matters on the dense and deadlock-prone instances.


### Speed of the C++ local search

`scripts/bench_cpp.py` measures the search with a fixed amount of work (a
fixed number of neighborhood searches instead of a time limit), so that two
binaries do exactly the same work and their plans can be compared byte for
byte:

```
python scripts/bench_cpp.py prepare                      # inputs in bench_cpp/
python scripts/bench_cpp.py run cpp/dpls_old cpp/dpls    # [--only NAME ...] [--searches K]
```

Exact optimizations (identical plans on all the benchmark instances):
order swaps are evaluated incrementally (only the events reachable from the
swapped node are recomputed), swaps whose removed constraint is not tight in
the current schedule are skipped (they cannot improve), and the earliest
schedule of a candidate is computed in linear time on the strongly connected
components of the affected constraints instead of with a FIFO Bellman-Ford;
the route-lag sums are restricted to the times at which every term is at
most the radius, and the cached sums are used in place. On the benchmark
instances the search is 2.9x faster in total (crossroads 76x, empty-8-8 at
90% 28x, empty-16-16 with 205 agents about 200x, MovingAI maps with 40 agents
1.3-2.6x). A second round of exact optimizations gives a further 1.3x
(empty-16-16 -36%, MovingAI maps up to -24%; instruction count -35% on a
MovingAI map): the sum-min tie-break key is updated incrementally and its
tables are built from the reference nodes only; the sums over the decided
agents are cached by the sequence of decisions and extended by one row; the
collision checks use per-expansion marks instead of scanning the decided
agents; the successor configurations are built once per expansion; and the
heuristic and tie-break values of an expanded state are read from its queue
entry. Rebuild with `scripts/build_cpp.sh` after updating.

### Rule-based initial plans (group P, PIBT)

`mapf_bench/pibt.py`: PIBT (priority inheritance with backtracking, Okumura et
al.), a rule-based solver: at every step each agent, by decreasing priority,
takes the neighbour closest to its goal, pushing recursively the agent standing
there; priorities grow while an agent is away from its goal. Not complete: it
stops after 20|V| + 100 steps and returns None if not all the agents are at
their goals. Group P (`configs/pibt.json`, and group P of `configs/paper.json`):
the instances of groups M, H, W, L, K with the PIBT plan as initial plan, names
with the suffix `_pibt`, only where PIBT succeeds (sample: M 20/24, H 24/33,
W 5/5, L 10/24, K 7/20; on the corridor grids of group C it never succeeds). The
LaCAM*-based methods are not run on group P (`paper_suite.skip_method`): they
start from scratch, so their results would be those of the original groups.


### Corridor grids at high density (group C)

`mapf_bench/corridors.py`, `configs/corridors.json` (and group C of
`configs/paper.json`): warehouses (rows of shelves separated by width-1 aisles,
cross aisles at both ends and in the middle), city blocks (square obstacle
blocks separated by width-1 streets) and braided mazes (width-1 corridors, all
dead ends and a further part of the walls opened); two sizes per family
(67-216 free cells), 50/70/85% occupancy (up to 184 agents), random starts
and goals, two seeds: 36 instances, all solved by LaCAM3, prioritized planning
fails on all of them. A second block (after the first, whose job indices are
kept) at 90/95/98% occupancy (up to 212 agents; at 98% one to four free cells
remain): 36 instances, of which LaCAM3 solves 11/12 at 90%, 9/12 at 95% and
9/12 at 98% (the others, mostly braided mazes, are excluded). Maps and scenarios in MovingAI format in
`benchmarks/corridors/`. Validation (12 instances at 70% and 85%, 20 s):
relative to the first LaCAM3 solution, LaCAM* from scratch 0.497 (best on 9),
LaCAM*+LNS 0.540, while MAPF-LNS, the local search and their combinations
improve the given plan by only 1-2%: in dense corridors the first solution is
far from good and only a search from scratch helps within 20 s.


### Deadlock-prone instances (group K)

`configs/deadlock.json` (and group K of `configs/paper.json`),
`mapf_bench/deadlock.py`: small grids in which agents block each other unless
they give way in the right order -- corridor swaps (two rooms, a corridor with
1 or 3 passing bays), dead-end corridors whose cells are all goals, filled in
random order, order reversal in a dead end with two small pockets, and
crossroads of four single-width arms with every agent going to the opposite
arm. 20 configurations x 2 seeds = 40 instances (39 solved by LaCAM3);
prioritized planning fails on all the reversal and crossroads instances and
on most of the long dead ends. Maps and scenarios are written in MovingAI
format to `benchmarks/deadlock/`.

### Reduced connectivity (group L)

`configs/reduced.json` (and group L of `configs/paper.json`): the MovingAI
instances of group M (maps empty-32-32, random-32-32-20, room-32-32-4,
maze-32-32-4; first 20, 40 and 100 agents of the scenarios) on maps where
free cells are made inaccessible. The cells are blocked in a random order
(seeded by map, scenario and number of agents) as long as the map stays
connected; starts and goals are never blocked. Level r blocks the first
fraction r of this sequence: r = 0 is the original map, r = 1 a minimal
connected map through all the starts and goals. Since starts and goals are
distinct cells, the occupancy cannot exceed about 50%; with these instances
it reaches 5% (20 agents) to 24% (100 agents) at r = 1. The reduced maps are
written in MovingAI format to `benchmarks/movingai/reduced/`. Initial plans:
the first LaCAM3 solution.

### MovingAI MAPF benchmark

`benchmarks/movingai/` contains standard maps of the MovingAI MAPF benchmark
(empty-16-16, empty-32-32, random-32-32-10/20, maze-32-32-2/4, room-32-32-4,
warehouse-10-20-10-2-1, den312d), the official scenarios
random-32-32-20-random-1 and random-32-32-10-random-1, and generated
scenarios in the same format for the other maps (see its README).
`mapf_bench/movingai.py` reads and writes maps and scenarios, builds
instances with the first n agents, and registers families
`mai_<map>_n<agents>` in compare_vnd:

```
python -m mapf_bench.compare_vnd run --instances 2 --out movingai_results \
    --strategies lnsT hyb2 hyb4 --families $(python -c "from mapf_bench.movingai import families; print(' '.join(families()))")
python -c "from mapf_bench.movingai import report; print(report())"
```

### High occupancy (up to 98%) and LaCAM3

`mapf_bench/lacam.py` wraps LaCAM3 (https://github.com/kei18/lacam3; build
with `scripts/build_lacam.sh`): its first solution is the initial plan of
dense instances, and LaCAM* (anytime, single thread) is a baseline.
`movingai.dense_families()` defines empty-8-8, random-8-8-20 and empty-16-16
at 50-98% of the free cells (up to 251 agents on 256 cells);
`compare_vnd` strategies `lacam` and `lacam+hyb2`;
`movingai.dense_report()` -> `movingai_dense/dense_report.md`,
`movingai_dense/lacam_hybrid.md`. Above 90% occupancy LaCAM* dominates
MAPF-LNS and the hybrid started from LaCAM's first solution; the hybrid
started from the LaCAM* solution improves it on all 28 instances (up to 34%),
also against LaCAM* with the same total time (checked on 6 instances).

### Experimental protocol for the paper

`python -m mapf_bench.paper_suite run --time 20` runs 100 scenarios in five
groups (structured, directed graphs, MovingAI maps with 20-40 agents,
MovingAI maps at 80-98% occupancy, dense grids) with seven methods (MAPF-LNS,
LaCAM*, LS, Hybrid, LNS+LS, LaCAM*+Hybrid, LaCAM*+LNS), each with the same
budget on one thread; runs resume. LNS+LS and LaCAM*+LNS are the baselines of
Hybrid and LaCAM*+Hybrid: they measure what the local search adds beyond
MAPF-LNS. Ablations: `python -m mapf_bench.ablation_ls run` (contribution of
each neighborhood, credit and leave-one-out). `python -m mapf_bench.paper_suite analyze` writes
`paper_results/results.md` (summary by group, pairwise Wilcoxon tests with
Holm correction, per-family table), `tables.tex`, `performance_profiles.pdf`;
`section_draft.tex` is a draft of the experimental section.

### Final comparison by type of MAPF instance

Fifteen families, six instances each: the families above plus small
MovingAI-style maps (`warehouse`, `warehouse_serial`, `rooms`, `maze`,
`random_map`; generators in `instances.py`).

```
python -m mapf_bench.compare_vnd run --instances 6 --out final_results \
    --strategies lns avndra avndra+lns lns+avndra
python -m mapf_bench.final_report final_results     # final_results/final_report.md
```

The report groups the families by type (grid maps with PP plans,
serialized plans, dense instances, structured instances) and gives
wins/ties/losses against MAPF-LNS with two-sided sign tests.

## Comparison with the official MAPF-LNS

The official C++ code (Li et al., IJCAI 2021, commit 95785de) is used through
`mapf_bench/official_lns.py`. A 126-line patch
(`patches/mapf-lns-initpaths.patch`) adds `--initAlgo File --initPaths`
(start from a given plan), `--outputPaths` (write the final paths) and
`--seed`; the algorithm itself is unchanged. The official code is not
redistributed here (USC Research License); build it with

```
scripts/build_official_lns.sh            # clones, patches, builds third_party/MAPF-LNS/lns
export MAPF_LNS_BIN=$PWD/third_party/MAPF-LNS/lns
python -m mapf_bench.explore --family grid_rings dense_grid grid_pp grid_serial \
    --algos olns olns_N10 olns_eecbs ls_gc0 ls_rr ls_new_alt olns+ls_new_alt ls_rr+olns \
    --seeds 8 --out explore_official
python -c "from mapf_bench.explore import official_summary as s; s('explore_official/explore.csv','explore_official')"
```

| name | configuration |
|---|---|
| `olns` | official MAPF-LNS, default: PP repair, destroy sets of 5, adaptive destroy heuristics |
| `olns_N10` | destroy sets of 10 |
| `olns_eecbs` | EECBS repair |

The official code supports 4-connected undirected grids only, so it is
compared on `grid_pp`, `grid_serial`, `dense_grid` and `grid_rings` (fully
occupied rings on a grid); the directed families can only be compared with
the re-implementation. The code confirms that MAPF-LNS accepts a new plan
only on a strict decrease of the Sum-of-Costs, with every repair algorithm.

## Reported metrics

For each suite and initial-plan type: mean makespan and Sum-of-Costs
reduction w.r.t. the initial plan, mean `makespan / LB` with
LB = max shortest-path length (1.00 means provably optimal makespan),
mean time, and wins/ties/losses against `lns_soc` on the makespan.

## Caveats

* `lns_soc` / `lns_mk` re-implement MAPF-LNS (adaptive choice among random,
  agent-based and map-based destroy heuristics; repair by randomized PP;
  strict-improvement acceptance). It is not the original C++ code, and its
  destroy heuristics are simplified; results on undirected grids should be
  checked against the original implementation before being reported.
* Everything is Python, so absolute times are not comparable with the C++
  numbers of the paper; budgets (`--time-limit`, `--expansions`,
  `--lns-iters`) bound each stage. A neighborhood search that exceeds its
  expansion budget counts as "no improvement" (see `budget_exceeded` in the
  `stats` column).
* Lemma `lem_mapf_Si` requires |V| >= n + 1; the search asserts it.

## Layout

```
mapf_bench/graph.py         digraphs with self-loops, dist and d_V tables
mapf_bench/plan.py          validation, makespan, SoC, serialization
mapf_bench/instances.py     instance generators
mapf_bench/pp.py            space-time A* and prioritized planning
mapf_bench/lns.py           MAPF-LNS re-implementation
mapf_bench/local_search.py  neighborhoods, Algorithm 1, Algorithm 3
mapf_bench/algorithms.py    registry of compared algorithms
mapf_bench/suites.py        benchmark suites
mapf_bench/run.py           runner (CSV), mapf_bench/report.py summaries
tests/                      pytest suite
```

The tests check, among other things, that with an unbounded radius the
sequentialized search returns the global optimum (against brute-force BFS on
the joint state space), that the incremental path distance equals the
definition of d_seq, the nesting d_sm <= d_gc <= d_lag <= d_seq, and the
claims of the paper's two examples (Sum-of-Costs tie at the merge; ring
improvement found at radius 0 with grouped clocks and missed by MAPF-LNS).

### Grouped clocks removed from the local search

`algorithms.GROUPED_CLOCKS = False`: the grouped-clock neighborhoods are no
longer in LS, LSB and LSBX (exact descent and escape phase). On 18 instances of
groups S, D, H and K, 20 s: LS 0 better / 18 equal / 0 worse without them, LSBX
1/17/0 (empty-8-8 at 80%: 605 -> 592). The distance stays available in the C++
code (`gc:` specs with groups) and in `adaptive_neighborhoods`.
