# Patch of the official MAPF-LNS

`mapf-lns-initpaths.patch` modifies the official MAPF-LNS
(https://github.com/Jiaoyang-Li/MAPF-LNS, commit 95785de): initial paths read
from a file (`--initAlgo File --initPaths`), output of the final paths
(`--outputPaths`), a random seed (`--seed`) and a stagnation limit
(`--stagnation`). It is applied by `scripts/build_official_lns.sh`.

Since it contains parts of the MAPF-LNS source, this patch is **not** covered
by the GPL (version 3 or later) of the rest of this repository: it is distributed under the
license of MAPF-LNS (Copyright © 2021 The University of Southern California),
reproduced in `LICENSE-MAPF-LNS.txt`, which allows educational, research and
non-profit use; commercial use requires the permission of USC.
