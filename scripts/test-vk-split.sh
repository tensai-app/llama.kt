#!/usr/bin/env bash
# Host-side test of the split Vulkan matmul chunking math (TEN-198). The header is Vulkan-free,
# so this needs nothing but a C++ compiler.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX_BIN="${CXX:-}"
if [ -z "$CXX_BIN" ]; then
  for c in c++ g++ clang++; do
    if command -v "$c" >/dev/null 2>&1; then CXX_BIN="$c"; break; fi
  done
fi
[ -n "$CXX_BIN" ] || { echo "ERROR: no C++ compiler found (set CXX)"; exit 1; }

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

"$CXX_BIN" -std=c++17 -Wall -Wextra -Werror \
  -I"$ROOT_DIR/llama-kt/src/main/cpp" \
  "$ROOT_DIR/scripts/test-vk-split.cpp" -o "$OUT/test-vk-split"
"$OUT/test-vk-split"
