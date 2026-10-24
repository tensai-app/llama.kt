#!/usr/bin/env bash
# Host-side test of the tool-call grammar inside llama.cpp's sampler (LKT-48):
# the vendored llama + common sources compiled for the host without any backend,
# and the vocab of a Qwen3 GGUF (only its metadata is read, any size or quant).
# The vocab-only GGUFs of the llama.cpp submodule have no <think> / <tool_call>
# special tokens, so the GGUF is passed in. Local only, not in CI. About a minute.
#
#   QWEN3_GGUF=/path/to/Qwen3-1.7B-Q4_K_M.gguf scripts/test-tool-grammar-sampler.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPP="$ROOT_DIR/llama-kt/src/main/cpp"
VOCAB="${QWEN3_GGUF:-}"
TEMPLATE="$ROOT_DIR/scripts/templates/Qwen-Qwen3-0.6B.jinja"
[ -f "$VOCAB" ] || { echo "ERROR: set QWEN3_GGUF to a Qwen3 GGUF"; exit 1; }

CXX_BIN="${CXX:-}"
if [ -z "$CXX_BIN" ]; then
  for c in c++ g++ clang++; do
    if command -v "$c" >/dev/null 2>&1; then CXX_BIN="$c"; break; fi
  done
fi
[ -n "$CXX_BIN" ] || { echo "ERROR: no C++ compiler found (set CXX)"; exit 1; }
CC_BIN="${CC:-cc}"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

INCLUDES=(-I"$CPP" -I"$CPP/common" -I"$CPP/common/jinja" -I"$CPP/nlohmann" -I"$CPP/ggml-cpu")
SOURCES=(
  ggml.c ggml-alloc.c ggml-quants.c ggml-backend.cpp ggml-backend-reg.cpp ggml-backend-dl.cpp ggml-backend-meta.cpp ggml-threading.cpp gguf.cpp
  unicode.cpp unicode-data.cpp
)
cd "$CPP"
SOURCES+=(llama*.cpp models/*.cpp common/*.cpp common/jinja/*.cpp)

# One compiler per source, in parallel.
export CPP OUT CXX_BIN CC_BIN
export INCLUDES_STR="${INCLUDES[*]}"
printf '%s\n' "${SOURCES[@]}" | xargs -P "$(getconf _NPROCESSORS_ONLN)" -I{} bash -c '
  src="{}"; obj="$OUT/${src//\//_}.o"
  case "$src" in
    *.c) "$CC_BIN"  -std=c11 -D_GNU_SOURCE -O1 -ffunction-sections -fdata-sections $INCLUDES_STR -c "$CPP/$src" -o "$obj" ;;
    *)   "$CXX_BIN" -std=c++17 -D_GNU_SOURCE -O1 -ffunction-sections -fdata-sections $INCLUDES_STR -c "$CPP/$src" -o "$obj" ;;
  esac'

# -Wno-unused-function: the jinja headers define static helpers this TU does not call.
"$CXX_BIN" -std=c++17 -Wall -Wextra -Werror -Wno-unused-function -ffunction-sections -fdata-sections "${INCLUDES[@]}" \
  -c "$ROOT_DIR/scripts/test-tool-grammar-sampler.cpp" -o "$OUT/test-tool-grammar-sampler.o"
"$CXX_BIN" -Wl,--gc-sections "$OUT"/*.o -o "$OUT/test-tool-grammar-sampler" -lpthread -lm -ldl
"$OUT/test-tool-grammar-sampler" "$VOCAB" "$TEMPLATE"
