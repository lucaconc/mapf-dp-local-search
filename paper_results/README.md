# Results of the paper

The run of the benchmark reported in the paper: `configs/paper.json`, 509 jobs
(9 groups, 11 methods, 20 s per method, one thread per job) on an Apple M1 Pro
with 5 workers, with the code of release 1.0.0.

| File | Content |
|---|---|
| `results.csv` | one row per instance and method: Sum-of-Costs, lower bound, time, certificate, errors |
| `config.json`, `machine.json` | configuration of the run and machine |
| `results.md`, `tables.tex` | tables by group and paired comparisons |
| `performance_profiles.pdf` | performance profiles (figure of the paper) |
| `claims_report.md` | check of the 29 qualitative claims of the paper (all confirmed) |

The plans themselves are not included (they are large); rerun the benchmark to
obtain them. To regenerate the tables, the numbers and the figure of a paper
from these results, without rerunning the benchmark:

```bash
python -m mapf_bench.paper_update paper_results --paper path/to/paper.tex
```
