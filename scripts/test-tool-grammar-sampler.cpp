// Host-side test of the tool-call grammar inside llama.cpp's sampler (LKT-48).
// Run via scripts/test-tool-grammar-sampler.sh.
//
// tensai_jni.cpp relies on two vendored behaviours it does not patch:
// common_sampler_init() builds the reasoning sampler for a lazy grammar even
// without a budget, and the grammar then sees no token while the reasoning
// block is open. Both are pinned here through common_sampler_accept() on a
// vocab-only model: a grammar that fires accepts only a well-formed call and
// throws on anything else, so whether it is awaiting its trigger shows.
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "tensai_tool_grammar.h"

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static std::string read_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static const llama_model * g_model = nullptr;

static common_chat_params render(const common_chat_templates * tmpls, bool thinking) {
    common_chat_templates_inputs in;
    common_chat_msg user;
    user.role    = "user";
    user.content = "Weather in Paris?";
    in.messages = {user};
    in.tools = {{"get_weather", "Current weather of a city",
                 R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})"}};
    in.tool_choice      = COMMON_CHAT_TOOL_CHOICE_AUTO;
    in.enable_thinking  = thinking;
    in.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    return common_chat_templates_apply(tmpls, in);
}

// The sampling params nativeCompletion() builds for this prompt with no
// budget: the grammar, and the reasoning tags only when `tags`.
static common_params_sampling params_for(const common_chat_params & p, bool tags) {
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    common_params_sampling sp;
    const bool installed = tensai_tool_grammar::install(sp, {p.grammar, p.grammar_lazy, p.grammar_triggers}, p.generation_prompt,
        [vocab](const std::string & word, std::string & piece) {
            return tensai_tool_grammar::leading_added_token(vocab, word, piece);
        });
    CHECK(installed);
    // As on device: a TOKEN trigger on <tool_call>, the regex path skipped.
    CHECK(sp.grammar_triggers.size() == 1 && sp.grammar_triggers[0].type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN
          && sp.grammar_triggers[0].token == common_tokenize(vocab, "<tool_call>", false, true).at(0));
    if (tags) {
        sp.reasoning_budget_tokens = -1;
        sp.reasoning_budget_start  = common_tokenize(vocab, p.thinking_start_tag, false, true);
        sp.reasoning_budget_end    = common_tokenize(vocab, p.thinking_end_tag, false, true);
        sp.reasoning_budget_forced = sp.reasoning_budget_end;
    }
    return sp;
}

// Accepts `text` token by token as if sampled. False when the grammar threw.
static bool feed(common_sampler * s, const std::string & text) {
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    try {
        for (const auto t : common_tokenize(vocab, text, false, true)) common_sampler_accept(s, t, true);
    } catch (const std::exception &) {
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::printf("usage: %s vocab.gguf Qwen-Qwen3-0.6B.jinja\n", argv[0]);
        return 1;
    }
    llama_log_set([](lm_ggml_log_level level, const char * text, void *) {
        if (level == LM_GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
    }, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        std::printf("cannot load %s\n", argv[1]);
        return 1;
    }
    g_model = model;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // Single tokens in this vocab, as in Qwen3's: what the model really emits.
    CHECK(common_tokenize(vocab, "<think>", false, true).size() == 1);
    CHECK(common_tokenize(vocab, "<tool_call>", false, true).size() == 1);

    common_chat_templates_ptr tmpls(common_chat_templates_init(nullptr, read_file(argv[2])));
    const common_chat_params think = render(tmpls.get(), true);
    const common_chat_params nothink = render(tmpls.get(), false);

    const std::string reasoning = "<think>\nI could write <tool_call>\nand then some words, not JSON.\n</think>\n\n";
    const std::string bad_call  = "<tool_call>\n{\"name\": \"bogus\"";
    const std::string good_call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    // The reasoning sampler exists with no budget, so a marker inside the
    // reasoning block does not fire the grammar; one after it does.
    {
        common_params_sampling sp = params_for(think, true);
        common_sampler * s = common_sampler_init(model, sp);
        CHECK(feed(s, reasoning));
        CHECK(feed(s, "Let me check."));
        CHECK(!feed(s, bad_call));
        common_sampler_free(s);
    }
    {
        common_params_sampling sp = params_for(think, true);
        common_sampler * s = common_sampler_init(model, sp);
        CHECK(feed(s, reasoning + good_call));
        common_sampler_free(s);
    }
    // The token alone fires it, and the grammar then wants the newline.
    {
        common_params_sampling sp = params_for(think, true);
        common_sampler * s = common_sampler_init(model, sp);
        CHECK(!feed(s, reasoning + "<tool_call>{\"name\""));
        common_sampler_free(s);
    }
    // A fenced call throws at accept time: the case nativeCompletion() recovers
    // from by dropping the grammar ("tool grammar: dropped mid-decode").
    {
        common_params_sampling sp = params_for(think, true);
        common_sampler * s = common_sampler_init(model, sp);
        CHECK(!feed(s, reasoning + "<tool_call>\n```json\n"));
        common_sampler_free(s);
    }
    // Plain text, even a bare object (TEN-184), never fires it.
    {
        common_params_sampling sp = params_for(think, true);
        common_sampler * s = common_sampler_init(model, sp);
        CHECK(feed(s, reasoning + "Sure: {\"name\": \"bogus\", \"arguments\": 3}"));
        common_sampler_free(s);
    }
    // Without the tags there is no reasoning sampler, and the marker in the
    // reasoning fires the grammar, which then chokes on the thinking: what
    // tensai_jni.cpp arms the sampler for.
    {
        common_params_sampling sp = params_for(think, false);
        common_sampler * s = common_sampler_init(model, sp);
        CHECK(!feed(s, reasoning));
        common_sampler_free(s);
    }
    // Thinking off: the prompt already closed the empty block, the grammar is
    // live from the first token.
    {
        common_params_sampling sp = params_for(nothink, true);
        common_sampler * s = common_sampler_init(model, sp);
        CHECK(!feed(s, bad_call));
        common_sampler_free(s);
        s = common_sampler_init(model, sp);
        CHECK(feed(s, good_call));
        common_sampler_free(s);
    }

    llama_model_free(model);
    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
