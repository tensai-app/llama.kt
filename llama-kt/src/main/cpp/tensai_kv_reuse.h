// KV reuse planning across a diverging prompt (TEN-196).
//
// The cache holds what the last turn generated, reasoning block included. The
// next turn's chat template replays that answer without it, so the prompt
// diverges from the cached tokens right after the assistant header, and a
// plain common-prefix reuse throws the whole answer away. The tokens after the
// dropped span are still in the cache and still the same text: they only need
// to move down by the span's length (a K-shift).
//
// Same idea as llama-server's --cache-reuse chunk loop: walk both lists, find
// runs of matching tokens at least min_chunk long, and shift each run to where
// the prompt wants it. Only deletions are handled (cached tokens the prompt no
// longer has). A prompt token missing from the cache would need a decode, not
// a shift: it ends the plan.
//
// Header-only and llama-free so a host-side test can include it as-is
// (scripts/test-kv-reuse.sh).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensai_kv {

// Cells [src, src + len) move to [dst, dst + len), dst <= src. Cells in
// [dst, src) are dropped. Moves come in increasing order and apply one after
// the other on the live cache.
struct KvMove {
    int32_t src;
    int32_t dst;
    int32_t len;
};

// cached: tokens the cache holds. prompt: tokens the next decode needs.
// n_past: length of their common prefix (positions below it are not touched).
// After the last move, cache and prompt agree up to last.dst + last.len.
inline std::vector<KvMove> plan_kv_reuse(
        const std::vector<int32_t>& cached,
        const std::vector<int32_t>& prompt,
        size_t n_past,
        size_t min_chunk,
        size_t max_moves) {
    std::vector<KvMove> moves;
    size_t head_c = n_past;
    size_t head_p = n_past;

    while (moves.size() < max_moves && head_c < cached.size() && head_p < prompt.size()) {
        size_t n_match = 0;
        while (head_c + n_match < cached.size() && head_p + n_match < prompt.size() &&
               cached[head_c + n_match] == prompt[head_p + n_match]) {
            n_match++;
        }
        if (n_match > 0 && n_match >= min_chunk) {
            moves.push_back({static_cast<int32_t>(head_c), static_cast<int32_t>(head_p),
                             static_cast<int32_t>(n_match)});
            head_c += n_match;
            head_p += n_match;
        } else {
            head_c++;
        }
    }
    return moves;
}

// Tokens a plan saves from re-prefill.
inline size_t kv_reuse_gain(const std::vector<KvMove>& moves) {
    size_t gain = 0;
    for (const auto& m : moves) gain += static_cast<size_t>(m.len);
    return gain;
}

}  // namespace tensai_kv
