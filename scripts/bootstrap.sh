#!/bin/bash -e

# bootstrap.sh — vendorize llama.cpp sources + rn-llama glue into llama-kt/src/main/cpp/
# Adapted from mybigday/llama.rn scripts/bootstrap.sh:
#   - Target is llama-kt/src/main/cpp/ (not cpp/ at root, not android/ or ios/)
#   - No React Native / Hexagon SDK / iOS steps
#   - Adds rn-*.{h,cpp,hpp} glue + jsi/JSINativeHeaders.h + jsi/ThreadPool.{h,cpp}
#   - Skips ggml-metal (iOS only) and ggml-hexagon (not needed)
#   - Applies same LM_ prefix rewrite on ggml/gguf symbols

# The shebang's -e is ignored by `bash scripts/bootstrap.sh`, which is how the
# README says to run this. Without it a cp of a file upstream removed prints an
# error and carries on.
set -e

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OS=$(uname)

LLAMA_DIR="$ROOT_DIR/third_party/llama.cpp"
CPP_DIR="$ROOT_DIR/llama-kt/src/main/cpp"

# llama.rn is not a submodule (see scripts/llama.rn.rev for why): we clone it
# into a scratch dir and hard-check out the pinned revision. Override the
# location with LLAMA_RN_REF_DIR if /tmp is inconvenient.
LLAMA_RN_URL="https://github.com/mybigday/llama.rn"
LLAMA_RN_REV_FILE="$ROOT_DIR/scripts/llama.rn.rev"
LLAMA_RN_REF_DIR="${LLAMA_RN_REF_DIR:-/tmp/llama.rn-ref}"
LLAMA_RN_CPP="$LLAMA_RN_REF_DIR/cpp"
# Since llama.rn 0.13 its llama.cpp patches are -p1 diffs in upstream's layout,
# one directory per vendored dependency (LKT-52).
PATCHES_SRC="$LLAMA_RN_REF_DIR/scripts/patches/llama.cpp"
LLAMA_RN_VERSIONS="$LLAMA_RN_REF_DIR/vendor/VERSIONS"

# The llama.cpp tree the copies below read: the submodule, exported and patched
# with llama.rn's patches (see "Patched llama.cpp tree" below). The submodule
# itself is never modified.
LLAMA_SRC=$(mktemp -d "${TMPDIR:-/tmp}/llama-kt-src.XXXXXX")
trap 'rm -rf "$LLAMA_SRC"' EXIT

echo "==> Init llama.cpp submodule..."
git -C "$ROOT_DIR" submodule init "third_party/llama.cpp"
# Use --no-fetch if the submodule is already at the right commit (faster).
# Otherwise fall back to a full update. We do NOT use --recursive to avoid
# pulling llama.cpp's own deps (ggml-cuda etc.) which we don't need.
LLAMA_WANT=$(git -C "$ROOT_DIR" ls-tree HEAD third_party/llama.cpp | awk '{print $3}')
# Not checked out: the directory exists but is empty, and `git -C` in it would
# answer with llama.kt's own HEAD. A checked-out submodule always has `.git`.
if [ -e "$LLAMA_DIR/.git" ]; then
  LLAMA_HAVE=$(git -C "$LLAMA_DIR" rev-parse HEAD)
else
  LLAMA_HAVE="none"
fi
if [ "$LLAMA_HAVE" = "none" ]; then
  git -C "$ROOT_DIR" submodule update "third_party/llama.cpp"
elif [ "$LLAMA_WANT" != "$LLAMA_HAVE" ] && [ "${LLAMA_KT_UNCOMMITTED_PIN:-}" = "1" ]; then
  # Generating the vendored tree for a bump under review, before its gitlink is
  # committed: the caller says the checkout is the pin on purpose.
  echo "  WARNING: generating from the checked-out llama.cpp $LLAMA_HAVE, HEAD pins $LLAMA_WANT (LLAMA_KT_UNCOMMITTED_PIN=1)"
elif [ "$LLAMA_WANT" != "$LLAMA_HAVE" ]; then
  # The wanted commit is read from HEAD, so a pin checked out but not yet
  # committed (step 1 of a bump) differs from it exactly like a submodule left
  # behind by a pull does, and git cannot tell the two apart. Updating here
  # used to put the old llama.cpp back without a word and generate against it
  # (TEN-103). Refuse instead and let the caller say which one it is.
  echo "ERROR: third_party/llama.cpp is at $LLAMA_HAVE but HEAD pins $LLAMA_WANT." >&2
  echo "  Bumping? Commit the new gitlink first, then re-run." >&2
  echo "  Stale checkout? git -C \"$ROOT_DIR\" submodule update third_party/llama.cpp" >&2
  exit 1
else
  echo "  submodule already at $LLAMA_HAVE, skipping update"
fi

# ---------------------------------------------------------------------------
# Pinned llama.rn checkout (rn-* glue + inherited patches)
# ---------------------------------------------------------------------------
# Done up front, not at the point of use: the rn-* copy happens ~400 lines
# below, after the Vulkan shader build, and nobody wants to find out the pin
# is unreachable twenty minutes in.
echo "==> Checking out pinned llama.rn..."
if [ ! -f "$LLAMA_RN_REV_FILE" ]; then
  echo "ERROR: missing pin file $LLAMA_RN_REV_FILE"
  exit 1
fi
LLAMA_RN_REV=$(grep -v '^#' "$LLAMA_RN_REV_FILE" | tr -d '[:space:]')
if ! echo "$LLAMA_RN_REV" | grep -qE '^[0-9a-f]{40}$'; then
  echo "ERROR: $LLAMA_RN_REV_FILE must hold one full 40-char llama.rn commit sha"
  echo "  got: '$LLAMA_RN_REV'"
  exit 1
fi

if [ ! -d "$LLAMA_RN_REF_DIR/.git" ]; then
  echo "  Cloning $LLAMA_RN_URL into $LLAMA_RN_REF_DIR..."
  rm -rf "$LLAMA_RN_REF_DIR"
  # Blobless, no checkout: we only need cpp/ and scripts/patches/ at one
  # revision. Deliberately not --recursive — llama.rn carries its own
  # llama.cpp submodule, which we already have in third_party/.
  git clone --filter=blob:none --no-checkout "$LLAMA_RN_URL" "$LLAMA_RN_REF_DIR"
fi

LLAMA_RN_HAVE=$(git -C "$LLAMA_RN_REF_DIR" rev-parse HEAD 2>/dev/null || echo "none")
if [ "$LLAMA_RN_HAVE" != "$LLAMA_RN_REV" ]; then
  echo "  Fetching $LLAMA_RN_REV..."
  git -C "$LLAMA_RN_REF_DIR" fetch --no-tags origin "$LLAMA_RN_REV" 2>/dev/null \
    || git -C "$LLAMA_RN_REF_DIR" fetch --no-tags --unshallow origin 2>/dev/null \
    || git -C "$LLAMA_RN_REF_DIR" fetch --no-tags origin
  git -C "$LLAMA_RN_REF_DIR" checkout --force --detach "$LLAMA_RN_REV"
fi

# The pin only means something if we verify it. A pre-existing checkout at
# $LLAMA_RN_REF_DIR (older bootstrap runs left a `--depth 1` clone of the
# default branch there) must not be used as-is.
LLAMA_RN_HAVE=$(git -C "$LLAMA_RN_REF_DIR" rev-parse HEAD)
if [ "$LLAMA_RN_HAVE" != "$LLAMA_RN_REV" ]; then
  echo "ERROR: $LLAMA_RN_REF_DIR is at $LLAMA_RN_HAVE, expected $LLAMA_RN_REV"
  exit 1
fi
if [ -n "$(git -C "$LLAMA_RN_REF_DIR" status --porcelain -- cpp scripts/patches)" ]; then
  echo "ERROR: $LLAMA_RN_REF_DIR has local edits under cpp/ or scripts/patches/."
  echo "  Vendored sources must come from a pristine $LLAMA_RN_REV checkout."
  exit 1
fi
echo "  llama.rn at $LLAMA_RN_REV"

# The rn-* glue and llama.rn's patches are written against the llama.cpp commit
# llama.rn vendors: the submodule must be that very commit.
LLAMA_RN_LLAMA=$(sed -n 's/^LLAMA_CPP_COMMIT=//p' "$LLAMA_RN_VERSIONS" 2>/dev/null)
LLAMA_HAVE=$(git -C "$LLAMA_DIR" rev-parse HEAD)
if [ "$LLAMA_RN_LLAMA" != "$LLAMA_HAVE" ]; then
  echo "ERROR: llama.rn $LLAMA_RN_REV vendors llama.cpp ${LLAMA_RN_LLAMA:-?} ($LLAMA_RN_VERSIONS)," >&2
  echo "  but third_party/llama.cpp is at $LLAMA_HAVE. Move one to match the other." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Patched llama.cpp tree
# ---------------------------------------------------------------------------
# llama.rn's patches apply to upstream's layout, so they go on a pristine export
# of the submodule before anything is copied or renamed. Only those whose target
# we deliberately don't vendor may be skipped; any other that fails means the
# two pins disagree, and a skipped hunk would be a fix silently lost.
echo "==> Exporting llama.cpp and applying llama.rn's patches..."
git -C "$LLAMA_DIR" archive --format=tar HEAD | tar -x -C "$LLAMA_SRC"
if [ ! -d "$PATCHES_SRC" ]; then
  echo "ERROR: llama.rn's llama.cpp patches not found at $PATCHES_SRC"
  exit 1
fi
patches_not_vendored=(
  ggml-metal-flash-attn-smem.patch   # ggml-metal/: iOS only
  ggml-metal-mul-mv-id-tiitg.patch   # ggml-metal/: iOS only
  ggml-hexagon.cpp.patch             # ggml-hexagon/: not needed
)
for patch_file in "$PATCHES_SRC"/*.patch; do
  [ -f "$patch_file" ] || continue
  patch_name=$(basename "$patch_file")
  skip=0
  for not_vendored in "${patches_not_vendored[@]}"; do
    [ "$patch_name" = "$not_vendored" ] && skip=1
  done
  if [ "$skip" -eq 1 ]; then
    echo "  Skipping patch (target not vendored): $patch_name"
    continue
  fi
  # --forward: a patch upstream has since merged reads as reversed, and plain
  # --batch would then un-apply it (it did for models-clef.cpp.patch at b11960).
  # --fuzz=0 here: only an exact reverse match counts as already merged.
  if ! patch --dry-run --batch --forward --quiet --fuzz=0 -p1 -d "$LLAMA_SRC" < "$patch_file" >/dev/null 2>&1 \
      && patch --dry-run --batch --reverse --quiet --fuzz=0 -p1 -d "$LLAMA_SRC" < "$patch_file" >/dev/null 2>&1; then
    echo "  Skipping patch (already in llama.cpp): $patch_name"
    continue
  fi
  echo "  Applying patch: $patch_name"
  # A non-zero exit means at least one hunk was rejected. Default fuzz, as llama.rn's
  # own sync-vendor.sh applies them: at rc.9, barbet-llama-arch.cpp.patch and
  # log.cpp.patch need it.
  if ! patch --batch --forward --quiet -p1 -d "$LLAMA_SRC" < "$patch_file"; then
    echo "  ERROR: llama.rn patch $patch_name does not apply to llama.cpp $LLAMA_HAVE."
    echo "  Resolve deliberately, do not skip it (see patches_not_vendored)."
    exit 1
  fi
  # llama.rn commits the tree its patches produce (vendor/llama.cpp): a hunk fuzz put
  # anywhere else shows up as a difference here.
  for f in $(sed -n 's|^+++ b/\([^[:space:]]*\).*|\1|p' "$patch_file"); do
    if ! cmp -s "$LLAMA_SRC/$f" "$LLAMA_RN_REF_DIR/vendor/llama.cpp/$f"; then
      echo "  ERROR: $f after $patch_name differs from llama.rn's vendor/llama.cpp/$f."
      exit 1
    fi
  done
done

echo "==> Creating output directory: $CPP_DIR"
mkdir -p "$CPP_DIR"
mkdir -p "$CPP_DIR/ggml-cpu/amx"
mkdir -p "$CPP_DIR/ggml-cpu/arch"
mkdir -p "$CPP_DIR/common/jinja"
mkdir -p "$CPP_DIR/tools/mtmd"
mkdir -p "$CPP_DIR/models"
mkdir -p "$CPP_DIR/ggml-opencl"
mkdir -p "$CPP_DIR/nlohmann"
mkdir -p "$CPP_DIR/jsi"
mkdir -p "$CPP_DIR/ggml-vulkan/shaders"
mkdir -p "$CPP_DIR/vulkan-hpp/vulkan"
mkdir -p "$CPP_DIR/spirv-headers/spirv/unified1"

# ---------------------------------------------------------------------------
# ggml public headers
# ---------------------------------------------------------------------------
echo "==> Copying ggml headers..."
cp "$LLAMA_SRC/ggml/include/ggml.h"            "$CPP_DIR/ggml.h"
cp "$LLAMA_SRC/ggml/include/ggml-alloc.h"      "$CPP_DIR/ggml-alloc.h"
cp "$LLAMA_SRC/ggml/include/ggml-backend.h"    "$CPP_DIR/ggml-backend.h"
cp "$LLAMA_SRC/ggml/include/ggml-cpu.h"        "$CPP_DIR/ggml-cpu.h"
cp "$LLAMA_SRC/ggml/include/ggml-cpp.h"        "$CPP_DIR/ggml-cpp.h"
cp "$LLAMA_SRC/ggml/include/ggml-opt.h"        "$CPP_DIR/ggml-opt.h"
cp "$LLAMA_SRC/ggml/include/ggml-opencl.h"     "$CPP_DIR/ggml-opencl.h"
cp "$LLAMA_SRC/ggml/include/gguf.h"            "$CPP_DIR/gguf.h"
# ggml-metal.h + ggml-blas.h + ggml-hexagon.h skipped (iOS / blas / hexagon only)

# ---------------------------------------------------------------------------
# ggml-opencl (Android GPU backend)
# ---------------------------------------------------------------------------
echo "==> Copying ggml-opencl..."
rm -rf "$CPP_DIR/ggml-opencl"
cp -r "$LLAMA_SRC/ggml/src/ggml-opencl" "$CPP_DIR/"
rm -f "$CPP_DIR/ggml-opencl/CMakeLists.txt"

# ---------------------------------------------------------------------------
# ggml-vulkan (Android Vulkan GPU backend — Mali, Adreno, etc.)
# ---------------------------------------------------------------------------
echo "==> Setting up ggml-vulkan..."

# 1. Copy the ggml-vulkan sources (split over several files since b11960) and ggml-vulkan.h
cp "$LLAMA_SRC/ggml/src/ggml-vulkan/"*.cpp "$LLAMA_SRC/ggml/src/ggml-vulkan/"*.h "$CPP_DIR/ggml-vulkan/"
cp "$LLAMA_SRC/ggml/include/ggml-vulkan.h"            "$CPP_DIR/ggml-vulkan/"

# 2. Build vulkan-shaders-gen for host and generate SPIR-V shaders
#    Prerequisites: cmake, glslc (from shaderc/glslang), g++
#    Procedure lives in gen-vulkan-shaders.sh (shared with CI and the README).
GLSLC=$(command -v glslc) || { echo "ERROR: glslc not found in PATH (shaderc/glslang)" >&2; exit 1; }
"$ROOT_DIR/scripts/gen-vulkan-shaders.sh" "$GLSLC"

# Fetch <url> at exactly <sha> into <dir>, reusing <dir> only if it is already
# there. A bare `[ -d ]` check used to trust whatever an earlier run had left in
# /tmp, and SPIRV-Headers was cloned at the tip of its default branch, so two
# runs at the same pins could vendor different headers (TEN-101).
fetch_pinned() {
  local url=$1 sha=$2 dir=$3
  if [ -e "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]; then
    return
  fi
  echo "  Fetching $url at $sha..."
  rm -rf "$dir"
  git init -q "$dir"
  git -C "$dir" fetch -q --depth=1 "$url" "$sha"
  git -C "$dir" checkout -q FETCH_HEAD
}

# 3. Bundle Vulkan C++ binding headers (vulkan.hpp, not in NDK sysroot)
#    Vulkan-Headers v1.4.350.
VULKAN_HEADERS_DIR="/tmp/Vulkan-Headers"
fetch_pinned https://github.com/KhronosGroup/Vulkan-Headers \
  8864cdc896bbc2a9b6eb36b3218fc9ef57908d77 "$VULKAN_HEADERS_DIR"
cp "$VULKAN_HEADERS_DIR/include/vulkan/"*.hpp "$CPP_DIR/vulkan-hpp/vulkan/" 2>/dev/null || true

# 4. Bundle SPIR-V headers (spirv/unified1/spirv.hpp)
#    No release tag carries the spirv.hpp that was first vendored; daa093d is
#    the last commit with that exact file (blob 9f3cb01), between c63848e and
#    575b651 on SPIRV-Headers main.
SPIRV_HEADERS_DIR="/tmp/SPIRV-Headers"
fetch_pinned https://github.com/KhronosGroup/SPIRV-Headers \
  daa093dd29aab8cbb6562b808370562f56e399fb "$SPIRV_HEADERS_DIR"
cp "$SPIRV_HEADERS_DIR/include/spirv/unified1/spirv.hpp" "$CPP_DIR/spirv-headers/spirv/unified1/"

echo "  Vulkan setup complete ($(ls "$CPP_DIR/ggml-vulkan/shaders/" | wc -l) shaders)"

# ---------------------------------------------------------------------------
# ggml-cpu
# ---------------------------------------------------------------------------
echo "==> Copying ggml-cpu..."
cp "$LLAMA_SRC/ggml/src/ggml-cpu/ggml-cpu.c"         "$CPP_DIR/ggml-cpu/ggml-cpu.c"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/ggml-cpu.cpp"       "$CPP_DIR/ggml-cpu/ggml-cpu.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/ggml-cpu-impl.h"    "$CPP_DIR/ggml-cpu/ggml-cpu-impl.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/quants.h"           "$CPP_DIR/ggml-cpu/quants.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/quants.c"           "$CPP_DIR/ggml-cpu/quants.c"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/arch-fallback.h"    "$CPP_DIR/ggml-cpu/arch-fallback.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/repack.cpp"         "$CPP_DIR/ggml-cpu/repack.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/repack.h"           "$CPP_DIR/ggml-cpu/repack.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/traits.h"           "$CPP_DIR/ggml-cpu/traits.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/traits.cpp"         "$CPP_DIR/ggml-cpu/traits.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/common.h"           "$CPP_DIR/ggml-cpu/common.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/unary-ops.h"        "$CPP_DIR/ggml-cpu/unary-ops.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/unary-ops.cpp"      "$CPP_DIR/ggml-cpu/unary-ops.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/binary-ops.h"       "$CPP_DIR/ggml-cpu/binary-ops.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/binary-ops.cpp"     "$CPP_DIR/ggml-cpu/binary-ops.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/vec.h"              "$CPP_DIR/ggml-cpu/vec.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/vec.cpp"            "$CPP_DIR/ggml-cpu/vec.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/simd-mappings.h"    "$CPP_DIR/ggml-cpu/simd-mappings.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/simd-gemm.h"        "$CPP_DIR/ggml-cpu/simd-gemm.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/ops.h"              "$CPP_DIR/ggml-cpu/ops.h"
cp "$LLAMA_SRC/ggml/src/ggml-cpu/ops.cpp"            "$CPP_DIR/ggml-cpu/ops.cpp"
cp -r "$LLAMA_SRC/ggml/src/ggml-cpu/amx"             "$CPP_DIR/ggml-cpu/"
rm -rf "$CPP_DIR/ggml-cpu/tiled"
cp -r "$LLAMA_SRC/ggml/src/ggml-cpu/tiled"           "$CPP_DIR/ggml-cpu/"
cp -r "$LLAMA_SRC/ggml/src/ggml-cpu/arch/arm"        "$CPP_DIR/ggml-cpu/arch/"
cp -r "$LLAMA_SRC/ggml/src/ggml-cpu/arch/x86"        "$CPP_DIR/ggml-cpu/arch/"

# ---------------------------------------------------------------------------
# ggml core sources
# ---------------------------------------------------------------------------
echo "==> Copying ggml core sources..."
cp "$LLAMA_SRC/ggml/src/ggml.c"                "$CPP_DIR/ggml.c"
cp "$LLAMA_SRC/ggml/src/ggml-impl.h"           "$CPP_DIR/ggml-impl.h"
cp "$LLAMA_SRC/ggml/src/ggml-alloc.c"          "$CPP_DIR/ggml-alloc.c"
cp "$LLAMA_SRC/ggml/src/ggml-backend.cpp"      "$CPP_DIR/ggml-backend.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-backend-impl.h"   "$CPP_DIR/ggml-backend-impl.h"
cp "$LLAMA_SRC/ggml/src/ggml-backend-reg.cpp"  "$CPP_DIR/ggml-backend-reg.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-backend-meta.cpp" "$CPP_DIR/ggml-backend-meta.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-backend-dl.h"     "$CPP_DIR/ggml-backend-dl.h"
cp "$LLAMA_SRC/ggml/src/ggml-backend-dl.cpp"   "$CPP_DIR/ggml-backend-dl.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-common.h"         "$CPP_DIR/ggml-common.h"
cp "$LLAMA_SRC/ggml/src/ggml-feats.h"          "$CPP_DIR/ggml-feats.h"
cp "$LLAMA_SRC/ggml/src/ggml-opt.cpp"          "$CPP_DIR/ggml-opt.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-quants.h"         "$CPP_DIR/ggml-quants.h"
cp "$LLAMA_SRC/ggml/src/ggml-quants.c"         "$CPP_DIR/ggml-quants.c"
cp "$LLAMA_SRC/ggml/src/ggml-threading.cpp"    "$CPP_DIR/ggml-threading.cpp"
cp "$LLAMA_SRC/ggml/src/ggml-threading.h"      "$CPP_DIR/ggml-threading.h"
cp "$LLAMA_SRC/ggml/src/gguf.cpp"              "$CPP_DIR/gguf.cpp"

# ---------------------------------------------------------------------------
# llama API
# ---------------------------------------------------------------------------
echo "==> Copying llama API..."
# All of src/ (CMakeLists globs llama*.cpp), so a file upstream adds comes with
# the bump. Model quantization is not exposed, as in llama.rn. Stale copies go
# first: a file upstream removed must not keep building here.
rm -f "$CPP_DIR"/llama*.h "$CPP_DIR"/llama*.cpp "$CPP_DIR"/unicode*.h "$CPP_DIR"/unicode*.cpp
for f in "$LLAMA_SRC"/src/*.h "$LLAMA_SRC"/src/*.cpp; do
  [ "$(basename "$f")" = "llama-quant.cpp" ] && continue
  cp "$f" "$CPP_DIR/"
done
cp "$LLAMA_SRC/include/llama.h"                       "$CPP_DIR/llama.h"
cp "$LLAMA_SRC/include/llama-cpp.h"                   "$CPP_DIR/llama-cpp.h"
rm -rf "$CPP_DIR/models"
cp -r "$LLAMA_SRC/src/models"                         "$CPP_DIR/models"

# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------
echo "==> Copying common..."
cp "$LLAMA_SRC/common/log.h"                          "$CPP_DIR/common/log.h"
cp "$LLAMA_SRC/common/log.cpp"                        "$CPP_DIR/common/log.cpp"
cp "$LLAMA_SRC/common/common.h"                       "$CPP_DIR/common/common.h"
cp "$LLAMA_SRC/common/common.cpp"                     "$CPP_DIR/common/common.cpp"
cp "$LLAMA_SRC/common/sampling.h"                     "$CPP_DIR/common/sampling.h"
cp "$LLAMA_SRC/common/sampling.cpp"                   "$CPP_DIR/common/sampling.cpp"
cp "$LLAMA_SRC/common/speculative.h"                  "$CPP_DIR/common/speculative.h"
cp "$LLAMA_SRC/common/speculative.cpp"                "$CPP_DIR/common/speculative.cpp"
cp "$LLAMA_SRC/common/ngram-cache.h"                  "$CPP_DIR/common/ngram-cache.h"
cp "$LLAMA_SRC/common/ngram-cache.cpp"                "$CPP_DIR/common/ngram-cache.cpp"
cp "$LLAMA_SRC/common/ngram-map.h"                    "$CPP_DIR/common/ngram-map.h"
cp "$LLAMA_SRC/common/ngram-map.cpp"                  "$CPP_DIR/common/ngram-map.cpp"
cp "$LLAMA_SRC/common/ngram-mod.h"                    "$CPP_DIR/common/ngram-mod.h"
cp "$LLAMA_SRC/common/ngram-mod.cpp"                  "$CPP_DIR/common/ngram-mod.cpp"
cp "$LLAMA_SRC/common/json-schema-to-grammar.h"       "$CPP_DIR/common/json-schema-to-grammar.h"
cp "$LLAMA_SRC/common/json-schema-to-grammar.cpp"     "$CPP_DIR/common/json-schema-to-grammar.cpp"
# Removed upstream; CMakeLists globs common/*.cpp, so a stale copy would build.
rm -f "$CPP_DIR/common/json-partial.h" "$CPP_DIR/common/json-partial.cpp"
rm -f "$CPP_DIR/common/regex-partial.h" "$CPP_DIR/common/regex-partial.cpp"
cp "$LLAMA_SRC/common/chat.h"                         "$CPP_DIR/common/chat.h"
cp "$LLAMA_SRC/common/chat.cpp"                       "$CPP_DIR/common/chat.cpp"
cp "$LLAMA_SRC/common/chat-auto-parser.h"             "$CPP_DIR/common/chat-auto-parser.h"
cp "$LLAMA_SRC/common/chat-auto-parser-helpers.h"     "$CPP_DIR/common/chat-auto-parser-helpers.h"
cp "$LLAMA_SRC/common/chat-auto-parser-helpers.cpp"   "$CPP_DIR/common/chat-auto-parser-helpers.cpp"
cp "$LLAMA_SRC/common/chat-auto-parser-generator.cpp" "$CPP_DIR/common/chat-auto-parser-generator.cpp"
cp "$LLAMA_SRC/common/chat-diff-analyzer.cpp"         "$CPP_DIR/common/chat-diff-analyzer.cpp"
cp "$LLAMA_SRC/common/chat-peg-parser.h"              "$CPP_DIR/common/chat-peg-parser.h"
cp "$LLAMA_SRC/common/chat-peg-parser.cpp"            "$CPP_DIR/common/chat-peg-parser.cpp"
cp "$LLAMA_SRC/common/peg-parser.h"                   "$CPP_DIR/common/peg-parser.h"
cp "$LLAMA_SRC/common/peg-parser.cpp"                 "$CPP_DIR/common/peg-parser.cpp"
cp "$LLAMA_SRC/common/unicode.h"                      "$CPP_DIR/common/unicode.h"
cp "$LLAMA_SRC/common/unicode.cpp"                    "$CPP_DIR/common/unicode.cpp"
cp "$LLAMA_SRC/common/reasoning-budget.h"             "$CPP_DIR/common/reasoning-budget.h"
cp "$LLAMA_SRC/common/reasoning-budget.cpp"           "$CPP_DIR/common/reasoning-budget.cpp"
cp "$LLAMA_SRC/common/fit.h"                          "$CPP_DIR/common/fit.h"
cp "$LLAMA_SRC/common/fit.cpp"                        "$CPP_DIR/common/fit.cpp"
cp "$LLAMA_SRC/common/build-info.h"                   "$CPP_DIR/common/build-info.h"
cp "$LLAMA_SRC/common/json.h"                         "$CPP_DIR/common/json.h"
cp "$LLAMA_SRC/common/json.cpp"                       "$CPP_DIR/common/json.cpp"
cp "$LLAMA_SRC/common/json-schema.h"                  "$CPP_DIR/common/json-schema.h"
cp "$LLAMA_SRC/common/json-schema.cpp"                "$CPP_DIR/common/json-schema.cpp"
cp "$LLAMA_SRC/common/trie.h"                         "$CPP_DIR/common/trie.h"
cp "$LLAMA_SRC/common/trie.cpp"                       "$CPP_DIR/common/trie.cpp"
cp "$LLAMA_SRC/common/base64.hpp"                     "$CPP_DIR/common/base64.hpp"
# Per-model chat parsers (one file each), added upstream after b9982.
rm -rf "$CPP_DIR/common/parsers"
mkdir -p "$CPP_DIR/common/parsers"
cp "$LLAMA_SRC/common/parsers/"*.h "$LLAMA_SRC/common/parsers/"*.cpp "$CPP_DIR/common/parsers/"

# ---------------------------------------------------------------------------
# jinja (template engine, needed by chat.cpp)
# ---------------------------------------------------------------------------
echo "==> Copying jinja..."
rm -rf "$CPP_DIR/common/jinja"
cp -r "$LLAMA_SRC/common/jinja" "$CPP_DIR/common/jinja"
# Rename jinja/string.h to avoid conflict with system <string.h>
mv "$CPP_DIR/common/jinja/string.h" "$CPP_DIR/common/jinja/jinja-string.h"
sed -i 's|#include "string.h"|#include "jinja-string.h"|g' "$CPP_DIR/common/jinja/value.h"
sed -i 's|#include "jinja/string.h"|#include "jinja/jinja-string.h"|g' "$CPP_DIR/common/jinja/string.cpp"
# Upstream finds common/unicode.h through common/'s include dir; here the flattened
# src/unicode.h comes first, so point at the parent dir.
sed -i 's|#include "unicode.h"|#include "../unicode.h"|g' "$CPP_DIR/common/jinja/"*.cpp "$CPP_DIR/common/jinja/"*.h
# src/ is flattened into $CPP_DIR: "../src/llama-ext.h" from common/ is "../llama-ext.h".
sed -i 's|#include "../src/llama-ext.h"|#include "../llama-ext.h"|g' "$CPP_DIR/common/"*.h "$CPP_DIR/common/"*.cpp

# ---------------------------------------------------------------------------
# nlohmann + vendors
# ---------------------------------------------------------------------------
echo "==> Copying nlohmann + vendors..."
rm -rf "$CPP_DIR/nlohmann"
cp -r "$LLAMA_SRC/vendor/nlohmann" "$CPP_DIR/nlohmann"

# mtmd audio/image vendors
rm -rf "$CPP_DIR/tools/mtmd/miniaudio"
rm -rf "$CPP_DIR/tools/mtmd/stb"
cp -r "$LLAMA_SRC/vendor/miniaudio" "$CPP_DIR/tools/mtmd/miniaudio"
cp -r "$LLAMA_SRC/vendor/stb" "$CPP_DIR/tools/mtmd/stb"

# ---------------------------------------------------------------------------
# tools/mtmd (multimodal)
# ---------------------------------------------------------------------------
echo "==> Copying tools/mtmd..."
rm -rf "$CPP_DIR/tools/mtmd"/*.{h,cpp} 2>/dev/null || true
mkdir -p "$CPP_DIR/tools/mtmd"
cp "$LLAMA_SRC/tools/mtmd/mtmd.h"          "$CPP_DIR/tools/mtmd/mtmd.h"
cp "$LLAMA_SRC/tools/mtmd/mtmd.cpp"        "$CPP_DIR/tools/mtmd/mtmd.cpp"
cp "$LLAMA_SRC/tools/mtmd/clip.h"          "$CPP_DIR/tools/mtmd/clip.h"
cp "$LLAMA_SRC/tools/mtmd/clip.cpp"        "$CPP_DIR/tools/mtmd/clip.cpp"
cp "$LLAMA_SRC/tools/mtmd/clip-impl.h"     "$CPP_DIR/tools/mtmd/clip-impl.h"
cp "$LLAMA_SRC/tools/mtmd/clip-model.h"    "$CPP_DIR/tools/mtmd/clip-model.h"
cp "$LLAMA_SRC/tools/mtmd/clip-graph.h"    "$CPP_DIR/tools/mtmd/clip-graph.h"
cp "$LLAMA_SRC/tools/mtmd/mtmd-helper.cpp" "$CPP_DIR/tools/mtmd/mtmd-helper.cpp"
cp "$LLAMA_SRC/tools/mtmd/mtmd-helper.h"   "$CPP_DIR/tools/mtmd/mtmd-helper.h"
cp "$LLAMA_SRC/tools/mtmd/mtmd-audio.h"    "$CPP_DIR/tools/mtmd/mtmd-audio.h"
cp "$LLAMA_SRC/tools/mtmd/mtmd-audio.cpp"  "$CPP_DIR/tools/mtmd/mtmd-audio.cpp"
cp "$LLAMA_SRC/tools/mtmd/mtmd-image.h"    "$CPP_DIR/tools/mtmd/mtmd-image.h"
cp "$LLAMA_SRC/tools/mtmd/mtmd-image.cpp"  "$CPP_DIR/tools/mtmd/mtmd-image.cpp"
cp "$LLAMA_SRC/tools/mtmd/mtmd-internal.h"      "$CPP_DIR/tools/mtmd/mtmd-internal.h"
cp "$LLAMA_SRC/tools/mtmd/mtmd-helper-common.h" "$CPP_DIR/tools/mtmd/mtmd-helper-common.h"
cp "$LLAMA_SRC/tools/mtmd/mtmd-helper-gen.cpp"  "$CPP_DIR/tools/mtmd/mtmd-helper-gen.cpp"
# tools/mtmd/ sits two levels under $CPP_DIR, where src/ is flattened.
sed -i 's|#include "../src/llama-ext.h"|#include "../../llama-ext.h"|g' "$CPP_DIR/tools/mtmd/"*.h "$CPP_DIR/tools/mtmd/"*.cpp
# mtmd-helper hashes media with the vendored SHA-256 ("hash/hash.h", from $CPP_DIR).
rm -rf "$CPP_DIR/hash"
mkdir -p "$CPP_DIR/hash"
cp -r "$LLAMA_SRC/vendor/hash/"* "$CPP_DIR/hash/"
rm -f "$CPP_DIR/hash/CMakeLists.txt"
# mtmd models/debug subdirs — use /* to copy contents, not the dir itself
if [ -d "$LLAMA_SRC/tools/mtmd/models" ]; then
  mkdir -p "$CPP_DIR/tools/mtmd/models"
  cp -r "$LLAMA_SRC/tools/mtmd/models/"* "$CPP_DIR/tools/mtmd/models/" 2>/dev/null || true
fi
if [ -d "$LLAMA_SRC/tools/mtmd/debug" ]; then
  mkdir -p "$CPP_DIR/tools/mtmd/debug"
  cp -r "$LLAMA_SRC/tools/mtmd/debug/"* "$CPP_DIR/tools/mtmd/debug/" 2>/dev/null || true
fi
# Copy vendors into mtmd — contents only, mkdir first
mkdir -p "$CPP_DIR/tools/mtmd/miniaudio"
cp -r "$LLAMA_SRC/vendor/miniaudio/"* "$CPP_DIR/tools/mtmd/miniaudio/" 2>/dev/null || true
mkdir -p "$CPP_DIR/tools/mtmd/stb"
cp -r "$LLAMA_SRC/vendor/stb/"* "$CPP_DIR/tools/mtmd/stb/" 2>/dev/null || true
# Their CMake files and nlohmann's already-applied backport patch: not built from here.
rm -f "$CPP_DIR/nlohmann/CMakeLists.txt" "$CPP_DIR/nlohmann/"*.patch \
      "$CPP_DIR/tools/mtmd/miniaudio/CMakeLists.txt" "$CPP_DIR/tools/mtmd/stb/CMakeLists.txt"

# ---------------------------------------------------------------------------
# rn-llama glue (GARDER: engine-facing files, zero JSI)
# ---------------------------------------------------------------------------
echo "==> Copying rn-llama glue from $LLAMA_RN_CPP..."
if [ ! -d "$LLAMA_RN_CPP" ]; then
  echo "ERROR: llama.rn cpp/ missing at $LLAMA_RN_CPP despite a verified checkout"
  exit 1
fi

cp "$LLAMA_RN_CPP/rn-llama.h"         "$CPP_DIR/rn-llama.h"
cp "$LLAMA_RN_CPP/rn-llama.cpp"       "$CPP_DIR/rn-llama.cpp"
cp "$LLAMA_RN_CPP/rn-completion.h"    "$CPP_DIR/rn-completion.h"
cp "$LLAMA_RN_CPP/rn-completion.cpp"  "$CPP_DIR/rn-completion.cpp"
cp "$LLAMA_RN_CPP/rn-slot.h"          "$CPP_DIR/rn-slot.h"
cp "$LLAMA_RN_CPP/rn-slot.cpp"        "$CPP_DIR/rn-slot.cpp"
cp "$LLAMA_RN_CPP/rn-slot-manager.h"  "$CPP_DIR/rn-slot-manager.h"
cp "$LLAMA_RN_CPP/rn-slot-manager.cpp" "$CPP_DIR/rn-slot-manager.cpp"
cp "$LLAMA_RN_CPP/rn-common.hpp"      "$CPP_DIR/rn-common.hpp"
cp "$LLAMA_RN_CPP/rn-mtmd.hpp"        "$CPP_DIR/rn-mtmd.hpp"
cp "$LLAMA_RN_CPP/rn-decision.h"      "$CPP_DIR/rn-decision.h"
cp "$LLAMA_RN_CPP/rn-decision.cpp"    "$CPP_DIR/rn-decision.cpp"
# rn-tts.cpp is not vendored: since llama.rn 0.13 it drives codec.cpp, a second
# vendored dependency for TTS, which llama.kt does not expose. The header stays
# (rn-llama and rn-completion hold a llama_rn_context_tts pointer, always null
# here) and tensai_tts_stub.cpp defines its methods.
cp "$LLAMA_RN_CPP/rn-tts.h"           "$CPP_DIR/rn-tts.h"
rm -f "$CPP_DIR/rn-tts.cpp"

# jsi/ — only the two allowed files (JSINativeHeaders.h, ThreadPool.{h,cpp})
cp "$LLAMA_RN_CPP/jsi/JSINativeHeaders.h"  "$CPP_DIR/jsi/JSINativeHeaders.h"
cp "$LLAMA_RN_CPP/jsi/ThreadPool.h"        "$CPP_DIR/jsi/ThreadPool.h"
cp "$LLAMA_RN_CPP/jsi/ThreadPool.cpp"      "$CPP_DIR/jsi/ThreadPool.cpp"

# ---------------------------------------------------------------------------
# LM_ prefix rewrite on ggml/gguf symbols
# ---------------------------------------------------------------------------
echo "==> Applying LM_ prefix rewrites..."

files_add_lm_prefix=(
  "$CPP_DIR/ggml-opencl/"*.cpp
  "$CPP_DIR/ggml-opencl/"*.h

  # Vulkan backend (its sources and headers)
  "$CPP_DIR/ggml-vulkan/"*.cpp
  "$CPP_DIR/ggml-vulkan/"*.h

  "$CPP_DIR/ggml-cpu/"*.h
  "$CPP_DIR/ggml-cpu/"*.c
  "$CPP_DIR/ggml-cpu/"*.cpp
  "$CPP_DIR/ggml-cpu/amx/"*.h
  "$CPP_DIR/ggml-cpu/amx/"*.cpp
  "$CPP_DIR/ggml-cpu/tiled/"*.h
  "$CPP_DIR/ggml-cpu/tiled/"*.cpp
  "$CPP_DIR/ggml-cpu/arch/arm/"*.c
  "$CPP_DIR/ggml-cpu/arch/arm/"*.cpp
  "$CPP_DIR/ggml-cpu/arch/x86/"*.c
  "$CPP_DIR/ggml-cpu/arch/x86/"*.cpp

  # Model definitions
  "$CPP_DIR/models/"*.h
  "$CPP_DIR/models/"*.cpp

  # Multimodal files
  "$CPP_DIR/tools/mtmd/"*.h
  "$CPP_DIR/tools/mtmd/"*.cpp
  "$CPP_DIR/tools/mtmd/models/"*.h
  "$CPP_DIR/tools/mtmd/models/"*.cpp
  "$CPP_DIR/tools/mtmd/debug/"*.h
  "$CPP_DIR/tools/mtmd/debug/"*.cpp

  # llama/ggml top-level
  "$CPP_DIR/"*.h
  "$CPP_DIR/"rn-*.hpp
  "$CPP_DIR/"*.cpp
  "$CPP_DIR/"*.c

  "$CPP_DIR/common/"*.h
  "$CPP_DIR/common/"*.cpp
  "$CPP_DIR/common/parsers/"*.h
  "$CPP_DIR/common/parsers/"*.cpp
)

for file in "${files_add_lm_prefix[@]}"; do
  # Skip if glob didn't match
  [ -f "$file" ] || continue

  # rn-* glue is rewritten too: since llama.rn 0.13 it spells plain ggml_, like upstream.

  # Skip the local ggml-ext.h shim if present
  [[ "$file" == "$CPP_DIR/ggml-ext.h" ]] && continue

  # Skip our own JNI layer: it is not vendored, already spells lm_ggml_, and
  # the rewrite below is not idempotent (lm_ggml_ -> lm_lm_ggml_).
  [[ "$file" == "$CPP_DIR/tensai_jni.cpp" ]] && continue

  # Same for our header-only helpers (tensai_utf.h, tensai_kv_reuse.h).
  [[ "$file" == "$CPP_DIR/"tensai_* ]] && continue

  sed -i 's/GGML_/LM_GGML_/g' "$file"
  sed -i 's/ggml_/lm_ggml_/g' "$file"
  sed -i 's/GGUF_/LM_GGUF_/g' "$file"
  sed -i 's/gguf_/lm_gguf_/g' "$file"

  # nlohmann: use local relative include instead of system angle-bracket form
  sed -i 's|<nlohmann/json.hpp>|"nlohmann/json.hpp"|g' "$file"
  sed -i 's|<nlohmann/json_fwd.hpp>|"nlohmann/json_fwd.hpp"|g' "$file"
done

# iq prefix rewrites (avoid collision with other ggml users like whisper.rn)
files_iq=(
  "$CPP_DIR/ggml-quants.h"
  "$CPP_DIR/ggml-quants.c"
  "$CPP_DIR/ggml.c"
)
for file in "${files_iq[@]}"; do
  [ -f "$file" ] || continue
  sed -i 's/iq2xs_init_impl/lm_iq2xs_init_impl/g' "$file"
  sed -i 's/iq2xs_free_impl/lm_iq2xs_free_impl/g' "$file"
  sed -i 's/iq3xs_init_impl/lm_iq3xs_init_impl/g' "$file"
  sed -i 's/iq3xs_free_impl/lm_iq3xs_free_impl/g' "$file"
done

# ---------------------------------------------------------------------------
# Apply local patches (ours, from patches/ at repo root)
# Unlike the llama.rn patches above, these MUST apply: they carry fixes the
# lib depends on (e.g. the Vulkan device dispatcher init, 0003).
# If one fails after a submodule bump, update the patch deliberately —
# do not skip it.
# ---------------------------------------------------------------------------
echo "==> Applying local patches (patches/)..."
if [ -d "$ROOT_DIR/patches" ]; then
  for patch_file in "$ROOT_DIR/patches"/*.patch; do
    [ -f "$patch_file" ] || continue
    echo "  Applying local patch: $(basename $patch_file)"
    if git -C "$ROOT_DIR" apply --check "$patch_file" 2>/dev/null; then
      git -C "$ROOT_DIR" apply "$patch_file"
    elif git -C "$ROOT_DIR" apply --reverse --check "$patch_file" 2>/dev/null; then
      echo "  Already applied, skipping."
    else
      echo "  ERROR: local patch $(basename $patch_file) does not apply."
      echo "  The vendored sources changed (submodule bump?) — update the patch."
      exit 1
    fi
  done
fi

# ---------------------------------------------------------------------------
# Generate build-info.cpp
# ---------------------------------------------------------------------------
echo "==> Generating build-info.cpp..."
cd "$LLAMA_DIR"
BUILD_NUMBER=$(git rev-list --count HEAD)
BUILD_COMMIT=$(git rev-parse --short=7 HEAD)
cd "$ROOT_DIR"

sed -e "s|@LLAMA_BUILD_NUMBER@|$BUILD_NUMBER|g" \
    -e "s|@LLAMA_BUILD_COMMIT@|$BUILD_COMMIT|g" \
    -e "s|@BUILD_COMPILER@|unknown|g" \
    -e "s|@BUILD_TARGET@|android|g" \
    "$LLAMA_SRC/common/build-info.cpp.in" > "$CPP_DIR/common/build-info.cpp"

# llama-version.h and ggml-version.h are configure_file() outputs upstream (since
# b11960). Same values as llama.rn's sync-vendor.sh, without its -dev suffix: the
# submodule sits on a release tag, where upstream's CMake drops it too. Written after
# the LM_ rewrite, so ggml's macros get their LM_ prefix here.
cmake_version() {  # <CMakeLists.txt> <PREFIX>
  local major minor patch
  major=$(sed -n "s/^set($2_VERSION_MAJOR \([0-9][0-9]*\))$/\1/p" "$1")
  minor=$(sed -n "s/^set($2_VERSION_MINOR \([0-9][0-9]*\))$/\1/p" "$1")
  patch=$(sed -n "s/^set($2_VERSION_PATCH \([0-9][0-9]*\))$/\1/p" "$1")
  if [ -z "$major" ] || [ -z "$minor" ] || [ -z "$patch" ]; then
    echo "ERROR: cannot read $2_VERSION_* from $1" >&2
    exit 1
  fi
  echo "$major.$minor.$patch"
}
# Assigned first: an exit inside $(...) in sed's arguments would not stop set -e.
LLAMA_VERSION=$(cmake_version "$LLAMA_SRC/CMakeLists.txt" LLAMA)
GGML_VERSION=$(cmake_version "$LLAMA_SRC/ggml/CMakeLists.txt" GGML)
sed -e "s|@LLAMA_VERSION@|$LLAMA_VERSION|g" \
    -e "s|@LLAMA_BUILD_COMMIT@|$BUILD_COMMIT|g" \
    "$LLAMA_SRC/src/llama-version.h.in" > "$CPP_DIR/llama-version.h"
sed -e "s|@GGML_VERSION@|$GGML_VERSION|g" \
    -e "s|@GGML_BUILD_COMMIT@|$BUILD_COMMIT|g" \
    -e "s|define GGML_|define LM_GGML_|g" \
    "$LLAMA_SRC/ggml/src/ggml-version.h.in" > "$CPP_DIR/ggml-version.h"

echo ""
echo "==> Bootstrap complete!"
echo "    Output: $CPP_DIR"
echo "    llama.cpp commit: $(git -C "$LLAMA_DIR" rev-parse HEAD)"
echo "    llama.rn commit:  $LLAMA_RN_REV"
echo "    Build number: $BUILD_NUMBER, commit: $BUILD_COMMIT"
