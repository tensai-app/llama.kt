#!/usr/bin/env bash
# Host-side test of the tool-call grammar (LKT-48): the vendored chat layer and
# GBNF parser, compiled for the host. No model, no backend: only the TUs the test
# reaches, the rest is dropped by --gc-sections. The Qwen3 template is a copy of
# llama.cpp's models/templates/Qwen-Qwen3-0.6B.jinja (the one Qwen3-1.7B ships
# too), kept here because CI does not always check the llama.cpp submodule out.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPP="$ROOT_DIR/llama-kt/src/main/cpp"
TEMPLATE="$ROOT_DIR/scripts/templates/Qwen-Qwen3-0.6B.jinja"

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

INCLUDES=(-I"$CPP" -I"$CPP/common" -I"$CPP/common/jinja" -I"$CPP/nlohmann")
SOURCES=(
  common/chat.cpp common/chat-auto-parser-generator.cpp common/chat-auto-parser-helpers.cpp
  common/chat-diff-analyzer.cpp common/chat-peg-parser.cpp common/peg-parser.cpp
  common/json-schema-to-grammar.cpp common/common.cpp common/log.cpp common/unicode.cpp
  common/jinja/caps.cpp common/jinja/lexer.cpp common/jinja/parser.cpp common/jinja/runtime.cpp
  common/jinja/string.cpp common/jinja/value.cpp
  llama-grammar.cpp llama-vocab.cpp llama-impl.cpp unicode.cpp unicode-data.cpp ggml.c
)

# One compiler per source, in parallel: the chat layer is the slow part.
export CPP OUT CXX_BIN CC_BIN
export INCLUDES_STR="${INCLUDES[*]}"
printf '%s\n' "${SOURCES[@]}" | xargs -P "$(getconf _NPROCESSORS_ONLN)" -I{} bash -c '
  src="{}"; obj="$OUT/${src//\//_}.o"
  case "$src" in
    *.c) "$CC_BIN"  -std=c11 -D_GNU_SOURCE -O1 -ffunction-sections -fdata-sections $INCLUDES_STR -c "$CPP/$src" -o "$obj" ;;
    *)   "$CXX_BIN" -std=c++17 -O1 -ffunction-sections -fdata-sections $INCLUDES_STR -c "$CPP/$src" -o "$obj" ;;
  esac'

# -Wno-unused-function: the jinja headers define static helpers this TU does not call.
"$CXX_BIN" -std=c++17 -Wall -Wextra -Werror -Wno-unused-function -ffunction-sections -fdata-sections "${INCLUDES[@]}" \
  -c "$ROOT_DIR/scripts/test-tool-grammar.cpp" -o "$OUT/test-tool-grammar.o"
"$CXX_BIN" -Wl,--gc-sections "$OUT"/*.o -o "$OUT/test-tool-grammar" -lpthread -lm
"$OUT/test-tool-grammar" "$TEMPLATE"
