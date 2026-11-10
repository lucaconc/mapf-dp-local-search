#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Set up mapf_bench on a local machine (Linux or macOS):
#   - a Python virtual environment in .venv with the dependencies;
#   - the C++ local search (cpp/dpls);
#   - the official MAPF-LNS with the patch (third_party/MAPF-LNS), needs Boost
#     and Eigen; LaCAM3 (third_party/lacam3);
#   - the test suite and an installation check.
# Requirements: python3 (>= 3.9), g++ or clang++ (C++17), git, cmake (>= 3.16).
#   Ubuntu/Debian: sudo apt install build-essential cmake git python3-venv \
#                  libboost-program-options-dev libboost-system-dev \
#                  libboost-filesystem-dev libeigen3-dev
#   macOS (Homebrew): brew install cmake boost eigen python
# Options: SKIP_LNS=1, SKIP_LACAM=1 skip the corresponding builds.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
need() { command -v "$1" > /dev/null || { echo "missing: $1 ($2)"; exit 1; }; }
need "$PY" "Python >= 3.9"
need g++ "a C++17 compiler"
need git "git"
need cmake "cmake >= 3.16"

[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== C++ local search"
scripts/build_cpp.sh
if [ "${SKIP_LNS:-0}" != 1 ]; then
  echo "== official MAPF-LNS"
  scripts/build_official_lns.sh || echo "WARNING: official MAPF-LNS not built (see the error above; \
it needs Boost and Eigen). On grids LNS will fail unless MAPF_BENCH_LNS_FALLBACK=1 (re-implementation)."
fi
if [ "${SKIP_LACAM:-0}" != 1 ]; then
  echo "== LaCAM3"
  scripts/build_lacam.sh || echo "WARNING: LaCAM3 not built (see the error above); LaCAM-based methods and dense suites will fail."
fi

echo "== tests"
python -m pytest -q
echo "== installation check"
python -m mapf_bench.local_runner doctor
cat << 'MSG'

Next steps (with the environment active: source .venv/bin/activate):
  python -m mapf_bench.local_runner run configs/smoke.json --workers 4
  python -m mapf_bench.local_runner run configs/paper.json --workers 8 --pin
  python -m mapf_bench.movingai_download          # official MovingAI benchmark
  python -m mapf_bench.local_runner run configs/movingai_official.json --workers 8 --pin
MSG
