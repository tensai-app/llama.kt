// Tool-call grammar of a chat prompt, installed into the sampling params (LKT-48).
//
// common_chat_templates_apply() hands back, next to the prompt and the reply
// parser, a GBNF grammar of the template's tool-call syntax with the offered
// tools' names and argument schemas in it. With tool_choice auto it is lazy:
// off until a trigger (the template's call marker, "<tool_call>\n" for Qwen3)
// shows up in the output, then every token has to fit a well-formed call. So a
// small model can no longer open a call and write an unknown tool name or
// arguments the schema does not allow; plain text is never constrained.
//
// A lazy grammar has to stay off inside the reasoning block too, or a marker
// the model merely thinks about would trigger it and constrain the thinking.
// llama.cpp only keeps it off when the reasoning-budget sampler exists to track
// <think>..</think>, so that sampler is needed whenever such a grammar is
// installed, with an unlimited budget when the caller asked for none.
//
// A WORD trigger is matched by llama.cpp as a regex, searched again on every
// token over all the output since the reasoning ended: a long plain answer
// pays for it on each token (8.0 -> 4.5 tok/s on device). When the word opens
// on one of the template's added tokens ("<tool_call>"), the trigger becomes
// that token instead, which skips the regex; the grammar, rooted at the whole
// word, still enforces the rest of it ("\n").
//
// Header-only, host-side tests include it as-is (scripts/test-tool-grammar*.sh).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"
#include "llama.h"

namespace tensai_tool_grammar {

// What nativeFormatChat() keeps of common_chat_params for nativeCompletion().
struct tool_grammar {
    std::string                         grammar;
    bool                                lazy = false;
    std::vector<common_grammar_trigger> triggers;
};

// The token a trigger word opens on when it is an added (special or
// user-defined) token, with its text; LLAMA_TOKEN_NULL otherwise. Ordinary
// tokens are left to the WORD path: the model can reach their text by other
// splits, which only the regex sees.
inline llama_token leading_added_token(const llama_vocab * vocab, const std::string & word, std::string & piece) {
    const auto tokens = common_tokenize(vocab, word, false, true);
    if (tokens.empty()) return LLAMA_TOKEN_NULL;
    const auto attr = llama_vocab_get_attr(vocab, tokens[0]);
    if (!(attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED))) return LLAMA_TOKEN_NULL;
    piece = common_token_to_piece(vocab, tokens[0], true);
    if (piece.empty() || word.compare(0, piece.size(), piece) != 0) return LLAMA_TOKEN_NULL;
    return tokens[0];
}

// Turns every WORD trigger that opens on an added token into a TOKEN trigger
// on it. `lookup(word, piece)` is leading_added_token() bound to the vocab.
template <typename Lookup>
std::vector<common_grammar_trigger> token_triggers(const std::vector<common_grammar_trigger> & triggers, Lookup lookup) {
    std::vector<common_grammar_trigger> out;
    for (const auto & t : triggers) {
        std::string piece;
        const llama_token tok = t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD ? lookup(t.value, piece) : LLAMA_TOKEN_NULL;
        if (tok == LLAMA_TOKEN_NULL) {
            out.push_back(t);
            continue;
        }
        common_grammar_trigger tt;
        tt.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
        tt.value = piece;
        tt.token = tok;
        out.push_back(std::move(tt));
    }
    return out;
}

// Writes `g` into `sp`. False, and `sp` left alone, when there is nothing to
// install: no grammar, or a lazy one without a trigger, which could never fire.
// generation_prompt is what common_sampler_init() replays through a non-lazy
// grammar and through the reasoning sampler, so both start where the prompt ends.
// Triggers go through token_triggers() with `lookup`.
template <typename Lookup>
bool install(common_params_sampling& sp, const tool_grammar& g, const std::string& generation_prompt, Lookup lookup) {
    if (g.grammar.empty()) return false;
    if (g.lazy && g.triggers.empty()) return false;
    sp.grammar           = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, g.grammar};
    sp.grammar_lazy      = g.lazy;
    sp.grammar_triggers  = token_triggers(g.triggers, lookup);
    sp.generation_prompt = generation_prompt;
    return true;
}

// Undoes install(), for the retry without grammar.
inline void clear(common_params_sampling& sp) {
    sp.grammar = {};
    sp.grammar_lazy = false;
    sp.grammar_triggers.clear();
}

// Whether the reasoning sampler must be armed: a budget was asked for, or a
// lazy grammar needs it to stay off inside the reasoning block.
inline bool needs_reasoning_sampler(int32_t budget_tokens, bool lazy_grammar) {
    return budget_tokens >= 0 || lazy_grammar;
}

}  // namespace tensai_tool_grammar
