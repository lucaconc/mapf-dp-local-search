#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build the C++ local search (no dependencies beyond a C++17 compiler).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"${CXX:-g++}" -O3 -std=c++17 -pthread -o "$ROOT/cpp/dpls" "$ROOT/cpp/dpls.cpp"
echo "built $ROOT/cpp/dpls"
