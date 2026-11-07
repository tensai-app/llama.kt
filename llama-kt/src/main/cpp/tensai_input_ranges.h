// Input text tokenized as text (TEN-218).
//
// The chat template's output is one string, and the prompt used to be tokenized
// with special tokens parsed everywhere in it. Since llama.cpp b11960 the
// template also says which parts of that string came from the messages
// (common_chat_prompt_part::is_input): a message holding "<|im_end|>" or another
// model's turn markers would otherwise be read as real control tokens and forge
// turns.
//
// parse_special=false is not enough on its own: llama.cpp still matches
// USER_DEFINED tokens with it (gpt-oss's "<|channel|>", Qwen's "<tool_call>").
// So a range is any run of message text holding the text of a special token
// (CONTROL, USER_DEFINED or UNKNOWN, the set the tokenizer matches), and inside
// a range the text is cut through every such occurrence before tokenizing, so
// that no piece holds a whole one.
//
// A prompt without any range keeps the single parse-everything call it always
// had, so its tokens, and everything keyed on them (KV reuse, sessions, the
// prefix cache), stay what they were.
//
// Header-only and llama-free so a host-side test can include it as-is
// (scripts/test-input-ranges.sh). `specials` is the texts of the model's special
// tokens; `tokenize(text, parse_special)` stands for common_tokenize() without
// BOS/EOS.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tensai_input {

// Bytes [begin, end) of the prompt, tokenized as text.
struct Range {
    size_t begin;
    size_t end;
};

// Where `text` holds a special token's text: the byte after the start of each
// occurrence, sorted and unique. Cutting there leaves no piece holding a whole
// one. One-byte texts are left out: a single byte cannot be cut, so a one-byte
// USER_DEFINED token still tokenizes as itself. It stands for that one
// character, so it can carry no turn marker.
inline std::vector<size_t> cuts(const std::string& text, const std::vector<std::string>& specials) {
    std::vector<size_t> out;
    for (const std::string& s : specials) {
        if (s.size() < 2) continue;
        for (size_t pos = text.find(s); pos != std::string::npos; pos = text.find(s, pos + 1)) {
            out.push_back(pos + 1);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// parts: (text, is_input) in order, as common_chat_prompt holds them; their
// concatenation is the prompt. Returns each run of adjacent input parts whose
// text, taken whole, holds a special token's text: a token split across two
// parts ("<|im_" + "end|>") is one in the prompt.
inline std::vector<Range> dirty_ranges(const std::vector<std::pair<std::string, bool>>& parts,
                                       const std::vector<std::string>& specials) {
    std::vector<Range> out;
    size_t pos = 0;
    for (size_t i = 0; i < parts.size();) {
        if (!parts[i].second) {
            pos += parts[i].first.size();
            i++;
            continue;
        }
        const size_t begin = pos;
        std::string run;
        for (; i < parts.size() && parts[i].second; i++) run += parts[i].first;
        pos += run.size();
        if (!cuts(run, specials).empty()) out.push_back({begin, pos});
    }
    return out;
}

// The ranges that apply to the first `len` bytes: the prefix cache prefills a
// prompt cut short of the one the ranges were recorded on.
inline std::vector<Range> clip(const std::vector<Range>& ranges, size_t len) {
    std::vector<Range> out;
    for (const Range& r : ranges) {
        if (r.begin >= len) break;
        out.push_back({r.begin, r.end < len ? r.end : len});
    }
    return out;
}

// The ranges of the last few formatted prompts, newest last. Several: the
// prefix cache formats its own prompt and prefills a cut of it, then the
// decode formats the full one. Not thread-safe, the caller locks.
class RecentPrompts {
public:
    void put(std::string prompt, std::vector<Range> ranges) {
        if (entries_.size() >= KEEP) entries_.erase(entries_.begin());
        entries_.push_back({std::move(prompt), std::move(ranges)});
    }

    // The ranges of the newest prompt that `text` is all or the start of,
    // clipped to it. None for text no recorded prompt starts with (a raw
    // decode): it is tokenized as before. `matched` says which of the two.
    std::vector<Range> for_text(const std::string& text, bool* matched = nullptr) const {
        for (auto e = entries_.rbegin(); e != entries_.rend(); ++e) {
            if (e->prompt.size() >= text.size() && e->prompt.compare(0, text.size(), text) == 0) {
                if (matched) *matched = true;
                return clip(e->ranges, text.size());
            }
        }
        if (matched) *matched = false;
        return {};
    }

    // A miss that starts like a recorded prompt: most likely a formatted prompt
    // changed on the way, which then loses its ranges.
    bool resembles(const std::string& text, size_t n = 32) const {
        if (text.size() < n) return false;
        for (const Entry& e : entries_) {
            if (e.prompt.compare(0, n, text, 0, n) == 0) return true;
        }
        return false;
    }

private:
    static constexpr size_t KEEP = 4;
    struct Entry {
        std::string        prompt;
        std::vector<Range> ranges;
    };
    std::vector<Entry> entries_;
};

// Tokens of `text` with `ranges` (sorted, disjoint, within text) tokenized as
// text, cut through every special token's text, and the rest with special
// tokens parsed. No BOS/EOS: the caller adds them as
// common_tokenize(add_special) would.
template <typename Tokenize>
std::vector<int32_t> tokenize(const std::string& text, const std::vector<Range>& ranges,
                              const std::vector<std::string>& specials, Tokenize&& tokenize) {
    std::vector<int32_t> out;
    auto add = [&](const std::string& piece, bool parse_special) {
        if (piece.empty()) return;
        const auto tokens = tokenize(piece, parse_special);
        out.insert(out.end(), tokens.begin(), tokens.end());
    };
    size_t pos = 0;
    for (const Range& r : ranges) {
        add(text.substr(pos, r.begin - pos), true);
        const std::string range = text.substr(r.begin, r.end - r.begin);
        size_t from = 0;
        for (const size_t cut : cuts(range, specials)) {
            add(range.substr(from, cut - from), false);
            from = cut;
        }
        add(range.substr(from), false);
        pos = r.end;
    }
    add(text.substr(pos), true);
    return out;
}

}  // namespace tensai_input
