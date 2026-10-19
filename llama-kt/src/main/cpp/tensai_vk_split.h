// Row-range planning for the split Vulkan matmul (TEN-198).
//
// One prefill MUL_MAT dispatch is hundreds of ms on a Mali, and the GPU only
// switches between dispatches, so the compositor starves. The patched
// ggml-vulkan.cpp (patches/0009) cuts a big matmul along M (rows of the weight
// matrix = output features) into several dispatches, each reading its own
// slice of A and writing its own slice of D, all against the whole of B.
//
// A chunk is a [row0, row0 + rows) range. Every row0 is a multiple of the
// shader tile height (BM) and of the row unit below, so that the A and D
// sub-buffer offsets (row0 * A row bytes, row0 * 4) stay aligned to
// minStorageBufferOffsetAlignment. Only the last chunk may be ragged.
//
// Header-only and Vulkan-free so a host-side test can include it as-is
// (scripts/test-vk-split.sh).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace tensai_vk {

// LM_GGML_VK_MATMUL_SPLIT_MFLOPS: unset, empty, negative, non-numeric (or trailing
// garbage) or 0 = disabled (0), else clamped to 1..1000000 MFLOP.
inline uint64_t parse_split_mflops(const char * e) {
    if (e == nullptr || *e == '\0') {
        return 0;
    }
    char * end = nullptr;
    const unsigned long long v = std::strtoull(e, &end, 10);
    if (*e == '-' || end == e || *end != '\0' || v == 0) {
        return 0;
    }
    return std::min<unsigned long long>(v, 1000000);
}

// LM_GGML_VK_MATMUL_SPLIT_SUBMIT: submit between chunks (default). Only the exact value "0" turns it
// off, which records the chunks back to back in one command buffer.
inline bool parse_split_submit(const char * e) {
    return e == nullptr || std::string_view(e) != "0";
}

inline uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        const uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

// Smallest row granularity: a multiple of bm that also keeps the A offset
// (row * a_row_bytes) and the D offset (row * d_elem_bytes) aligned.
// 0 = bad input, the caller must not split.
inline uint64_t row_unit(uint64_t bm, uint64_t a_row_bytes, uint64_t d_elem_bytes, uint64_t align) {
    if (bm == 0 || a_row_bytes == 0 || d_elem_bytes == 0) {
        return 0;
    }
    if (align == 0) {
        align = 1;
    }
    const uint64_t ua = align / gcd_u64(a_row_bytes, align);
    const uint64_t ud = align / gcd_u64(d_elem_bytes, align);
    uint64_t u = bm;
    u = u / gcd_u64(u, ua) * ua;
    u = u / gcd_u64(u, ud) * ud;
    return u;
}

// Upper bound on chunks per matmul: every chunk is a submit, and a tiny budget
// on a big matmul would otherwise queue hundreds of them.
constexpr uint64_t max_chunks = 64;

// Rows per chunk so that 2 * rows * n * k <= budget_flops, rounded down to a
// multiple of unit (at least one unit, so a chunk can exceed a tiny budget),
// then raised to ceil(m / max_chunks) rounded up to unit when that would give
// more than max_chunks chunks. Returns m (= no split) when disabled, when the
// whole matmul fits the budget or when m does not exceed one unit.
inline uint64_t chunk_rows(uint64_t m, uint64_t n, uint64_t k, uint64_t budget_flops, uint64_t unit) {
    if (budget_flops == 0 || unit == 0 || n == 0 || k == 0 || m <= unit) {
        return m;
    }
    const uint64_t per_row = 2 * n * k;
    if (m <= budget_flops / per_row) {
        return m;
    }
    const uint64_t floor_rows = (m + max_chunks - 1) / max_chunks;
    const uint64_t cap_rows = (floor_rows + unit - 1) / unit * unit;
    const uint64_t rows = std::max({ budget_flops / per_row / unit * unit, unit, cap_rows });
    return rows >= m ? m : rows;
}

} // namespace tensai_vk
