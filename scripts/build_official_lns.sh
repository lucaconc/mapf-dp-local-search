#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build the official MAPF-LNS (Li et al., IJCAI 2021) with a small patch that
# adds --initAlgo File --initPaths <file>, --outputPaths <file>, --seed and
# --stagnation.
# Requires: git, cmake, a C++ compiler, Boost (program_options, system,
# filesystem) and Eigen3.
#   Ubuntu: apt install cmake libboost-program-options-dev libboost-system-dev \
#           libboost-filesystem-dev libeigen3-dev
#   macOS:  brew install cmake boost eigen
# Works with CMake 3.x and 4.x (the upstream CMakeLists.txt asks for CMake 2.6,
# which CMake >= 4.0 refuses; the minimum is raised to 3.5 here) and with
# Boost >= 1.89, which has no boost_system component (it is no longer requested),
# and with Eigen 4 and 5 (the version requirement 3.3 is dropped).
# macOS/Homebrew: if CMake does not find Boost or Eigen, run
#   export CMAKE_PREFIX_PATH="$(brew --prefix)"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$ROOT/patches/mapf-lns-initpaths.patch"
DEST="${1:-$ROOT/third_party/MAPF-LNS}"
COMMIT=95785de66f8fbdb91fed871e1a56c8c563f39b1f
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
if [ ! -d "$DEST" ]; then
  git clone -q https://github.com/Jiaoyang-Li/MAPF-LNS.git "$DEST"
fi
cd "$DEST"
git checkout -q "$COMMIT" 2>/dev/null || true      # keeps a patched working tree
if git apply --check "$PATCH" 2>/dev/null; then
  git apply "$PATCH"
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "patch already applied"
else
  echo "ERROR: the patch does not apply to $DEST; remove the folder and run again" >&2
  exit 1
fi
# CMake >= 4.0: raise the minimum version and declare it before project()
if grep -qiE "cmake_minimum_required *\( *VERSION *2\." CMakeLists.txt; then
  sed -i.orig -E '/cmake_minimum_required *\( *VERSION *2\./d' CMakeLists.txt
  { echo "cmake_minimum_required(VERSION 3.5)"; cat CMakeLists.txt; } > CMakeLists.txt.new
  mv CMakeLists.txt.new CMakeLists.txt
fi
# Boost >= 1.89 no longer provides the boost_system component (Boost.System is
# header-only since 1.69, and MAPF-LNS does not link it explicitly): drop it.
if grep -qE "find_package *\( *Boost.* system[ )]" CMakeLists.txt; then
  sed -i.orig2 -E '/find_package *\( *Boost/s/ system / /' CMakeLists.txt
fi
# Eigen >= 4 (Homebrew ships Eigen 5) does not accept the request "Eigen3 3.3",
# since its package only matches the same major version; MAPF-LNS uses only
# Eigen::MatrixXi, so any version works: drop the version requirement.
if grep -qE "find_package *\( *Eigen3 +3\.[0-9]+" CMakeLists.txt; then
  sed -i.orig3 -E 's/find_package *\( *Eigen3 +3\.[0-9]+/find_package (Eigen3/' CMakeLists.txt
fi
rm -rf CMakeCache.txt CMakeFiles                  # a previous failed configuration
# extra CMake arguments, e.g. MAPF_LNS_CMAKE_ARGS="-DEigen3_DIR=/path/to/share/eigen3/cmake"
# shellcheck disable=SC2086
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ${MAPF_LNS_CMAKE_ARGS:-} . > build.log 2>&1 \
  || { tail -25 build.log >&2; echo "ERROR: cmake failed (log: $DEST/build.log)" >&2; exit 1; }
make -j"$JOBS" >> build.log 2>&1 \
  || { tail -25 build.log >&2; echo "ERROR: compilation failed (log: $DEST/build.log)" >&2; exit 1; }
echo "built $DEST/lns"
echo "export MAPF_LNS_BIN=$DEST/lns"
