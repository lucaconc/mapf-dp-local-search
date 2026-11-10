# Paper update 20261110-142850

Results: `/tmp/ur4/results/paper` (396 solved instances, T = 20 s). Paper: `/tmp/fin/LocalSearch_new.nogc.tex`. Macros: `bench_numbers.tex` (1394). Diff: `paper_update/diff-20261110-142850.patch`.

## Claims of the prose

| status | where | claim |
|---|---|---|
| confirmed | Discussion, coordinated moves | LS attains the best Sum-of-Costs on every instance of group S |
| confirmed | Discussion, coordinated moves | MAPF-LNS keeps the initial plan on the full rings |
| confirmed | Discussion, coordinated moves | LaCAM* finds the optimum (lower bound) on the rings on grids |
| confirmed | Discussion, coordinated moves | all the methods tie on the convoys |
| confirmed | Discussion, PP plans | LS alone is worse than LNS on the MovingAI maps (group M) |
| confirmed | Discussion, PP plans | on serialized plans LS is within 2% of LNS on average, in less than a quarter of the budget |
| confirmed | Discussion, deadlocks | in group K, LNS does not improve the initial plan on every order reversal |
| confirmed | Discussion, deadlocks | in group K, LNS+LS is better than LNS on more instances than it is worse |
| confirmed | Discussion, dense | in group H the methods that start from the first LaCAM solution are far from LaCAM* |
| confirmed | Discussion, dense | a method that starts from LaCAM* has the best SoC/LB in group H |
| confirmed | Discussion, dense, and Conclusion | in group L, LaCAM*+LNS is better than LaCAM*+Hybrid more often |
| confirmed | Discussion, dense, and Conclusion | in group C, LaCAM*+Hybrid is better than LaCAM*+LNS more often |
| confirmed | Discussion, dense | in group C, LaCAM*+LNS+LSB is better than LaCAM*+LNS more often |
| confirmed | Discussion, dense | in group C, LaCAM* alone is better than LaCAM*+LNS more often |
| confirmed | Discussion overall, abstract, contributions, conclusion | in group S, LNS+LS is better than LNS on more instances than it is worse |
| confirmed | Discussion overall, abstract, contributions, conclusion | in group H, LNS+LS is better than LNS on more instances than it is worse |
| confirmed | Discussion overall, abstract, contributions, conclusion | in group K, LNS+LS is better than LNS on more instances than it is worse |
| confirmed | Discussion overall, abstract, contributions, conclusion | in group C, LNS+LS is better than LNS on more instances than it is worse |
| confirmed | Discussion overall, abstract, contributions, conclusion | in group L (reduced connectivity), LNS+LS is worse than LNS on more instances than it is better |
| confirmed | Discussion overall | in group D, LNS+LS and LNS are not significantly different |
| confirmed | Discussion overall | in group M, LNS+LS and LNS are not significantly different |
| confirmed | Discussion overall, abstract, contributions, conclusion | the alternation of the hybrid brings no benefit: Hybrid is not significantly better than LNS+LS |
| confirmed | Discussion overall | LaCAM*+Hybrid is significantly better than LaCAM* alone |
| confirmed | Discussion, dense | on the maps with reduced connectivity (group L) LaCAM*+LNS is better than LaCAM*+LNS+LSB |
| confirmed | Discussion, large moves | LSBX is better than LSB on more instances than it is worse |
| confirmed | Discussion, large moves | LSBX is better than LS on more instances than it is worse |
| confirmed | Discussion, large moves | LSB is better than LS on more instances than it is worse |
| confirmed | Discussion, large moves | on the MovingAI maps (group M) LSB coincides with LS |
| confirmed | Discussion, certificates | LS alone stops before the budget on average |
