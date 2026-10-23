// What nativeCompletion returns once its decode loop has ended.
//
// The count of sampled tokens, which Kotlin compares against n_predict to tell
// a finished reply from one cut at its cap, or -1 when llama_decode() failed.
// That failure leaves nextToken() the way an EOS does (has_next_token false,
// tok -1), so returning the count read as a reply the model ended itself.
// An interrupt wins over a failure: the user stopped the decode, and a decode
// aborted mid-graph is not reported as failed by nextToken() anyway.
//
// Header-only and llama-free so a host-side test can include it as-is
// (scripts/test-completion-result.sh).
#pragma once

#include <cstddef>
#include <cstdint>

namespace tensai_completion {

inline int32_t return_value(size_t n_predicted, bool decode_failed, bool interrupted) {
    if (decode_failed && !interrupted) return -1;
    return static_cast<int32_t>(n_predicted);
}

}  // namespace tensai_completion
