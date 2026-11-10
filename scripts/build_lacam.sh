#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build LaCAM3 (Okumura, https://github.com/kei18/lacam3), used for initial
# plans of dense instances and as an anytime baseline. Requires git, cmake, g++.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/third_party/lacam3}"
[ -d "$DEST" ] || git clone -q --recursive https://github.com/kei18/lacam3.git "$DEST"
cd "$DEST"
git checkout -q 1c3ecf8ade40298f16047544634e9c0c1cf3b93c
git submodule update --init --recursive -q
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 > build.log 2>&1 \
  || { tail -25 build.log >&2; echo "ERROR: cmake failed (log: $DEST/build.log)" >&2; exit 1; }
make -C build -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)" main >> build.log 2>&1 \
  || { tail -25 build.log >&2; echo "ERROR: compilation failed (log: $DEST/build.log)" >&2; exit 1; }
echo "export LACAM_BIN=$DEST/build/main"
