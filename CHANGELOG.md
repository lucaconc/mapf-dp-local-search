# Changelog

## 1.0.0 (2026-11-10)

First public release, with the paper *Dynamic Programming-Based Local Search
for Multi-Agent Path Finding on Directed Graphs*.

- Local search in C++ (`cpp/dpls.cpp`): path, pointwise, agent, sum-min,
  route-lag and free-set neighborhoods, retiming and order swaps, variable
  neighborhood descent with adaptive radii, goal-directed search, chaining of
  forced moves, bounded open list for incomplete searches.
- Methods of the paper: LS, LSB, LSBX, Hybrid and the combinations with
  MAPF-LNS and LaCAM* (`mapf_bench/paper_suite.py`).
- Benchmark protocol of the paper (`configs/paper.json`, 509 instances in nine
  groups), parallel runner, statistics, performance profiles, automatic update
  of the paper's tables and numbers with a check of its claims.
- Results of the run reported in the paper (`paper_results/`).
- The code was generated with Claude (Anthropic) under the direction of the
  authors (see "Use of AI tools" in the README).
