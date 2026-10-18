#!/usr/bin/env bash
# Host-side test of the KV reuse planner (TEN-196). The header is llama-free,
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
  "$ROOT_DIR/scripts/test-kv-reuse.cpp" -o "$OUT/test-kv-reuse"
"$OUT/test-kv-reuse"
