// Host-side test for tensai_tool_grammar.h (LKT-48). Run via scripts/test-tool-grammar.sh.
//
// Renders a prompt with tools through the vendored chat layer and the Qwen3
// template, checks the lazy tool-call grammar comes out of it, lands in the
// sampling params as tensai_jni.cpp installs it, accepts a well-formed call
// and rejects malformed ones, and that what it forces parses back as a call.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "chat.h"
#include "llama-grammar.h"
#include "tensai_tool_grammar.h"
#include "unicode.h"

// The chat layer links against these but only calls them with a model, and
// this test has none: common_chat_templates_init() gets the template source.
const char * llama_model_chat_template(const llama_model *, const char *) { std::abort(); }
int32_t llama_chat_apply_template(const char *, const llama_chat_message *, size_t, bool, char *, int32_t) { std::abort(); }
const llama_vocab * llama_model_get_vocab(const llama_model *) { std::abort(); }

static int failures = 0;

// Stands in for leading_added_token() and Qwen3's vocab, where <tool_call> is
// the added token 151657. no_added: a vocab without it.
static llama_token added_tool_call(const std::string & word, std::string & piece) {
    if (word.rfind("<tool_call>", 0) != 0) return LLAMA_TOKEN_NULL;
    piece = "<tool_call>";
    return 151657;
}
static llama_token no_added(const std::string &, std::string &) { return LLAMA_TOKEN_NULL; }

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static std::string read_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("cannot read %s\n", path);
        std::exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// The grammar past its trigger, as the sampler enforces it once triggered:
// true when `input` is a complete sentence of it.
static bool grammar_accepts(const std::string & grammar, const std::string & input) {
    std::unique_ptr<llama_grammar> g(llama_grammar_init_impl(nullptr, grammar.c_str(), "root", false, nullptr, 0, nullptr, 0));
    if (!g) return false;
    auto & stacks = llama_grammar_get_stacks(g.get());
    for (const auto cpt : unicode_cpts_from_utf8(input)) {
        llama_grammar_accept(g.get(), cpt);
        if (stacks.empty()) return false;
    }
    for (const auto & stack : stacks) {
        if (stack.empty()) return true;
    }
    return false;
}

static common_chat_params render(const common_chat_templates * tmpls, bool thinking, common_chat_tool_choice choice) {
    common_chat_templates_inputs in;
    common_chat_msg user;
    user.role    = "user";
    user.content = "Weather in Paris?";
    in.messages = {user};
    in.tools = {
        {"get_weather", "Current weather of a city",
         R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})"},
        {"web_search", "Search the web",
         R"({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]})"},
    };
    in.tool_choice      = choice;
    in.enable_thinking  = thinking;
    // What nativeFormatChat() asks for.
    in.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    return common_chat_templates_apply(tmpls, in);
}

static common_chat_msg parse(const common_chat_params & p, const std::string & reply) {
    // As rn-completion's parseChatOutput(): the generation prompt, then the reply.
    common_chat_parser_params syntax(p);
    syntax.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    syntax.parser.load(p.parser);
    return common_chat_parse(p.generation_prompt + reply, false, syntax);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::printf("usage: %s Qwen-Qwen3-0.6B.jinja\n", argv[0]);
        return 1;
    }
    common_chat_templates_ptr tmpls(common_chat_templates_init(nullptr, read_file(argv[1])));

    const std::string call =
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    for (const bool thinking : {true, false}) {
        const common_chat_params p = render(tmpls.get(), thinking, COMMON_CHAT_TOOL_CHOICE_AUTO);

        // The template produces a lazy grammar, triggered by its call marker.
        CHECK(!p.grammar.empty());
        CHECK(p.grammar_lazy);
        CHECK(p.grammar_triggers.size() == 1);
        CHECK(p.grammar_triggers.at(0).type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD);
        CHECK(p.grammar_triggers.at(0).value.rfind("<tool_call>", 0) == 0);
        // The grammar starts where the trigger fires.
        CHECK(call.rfind(p.grammar_triggers.at(0).value, 0) == 0);
        // The reasoning tags the reasoning sampler arms on.
        CHECK(p.supports_thinking);
        CHECK(p.thinking_start_tag == "<think>");
        CHECK(p.thinking_end_tag == "</think>");

        // Installed as nativeCompletion() does it.
        common_params_sampling sp;
        const tensai_tool_grammar::tool_grammar tg = {p.grammar, p.grammar_lazy, p.grammar_triggers};
        CHECK(tensai_tool_grammar::install(sp, tg, p.generation_prompt, added_tool_call));
        CHECK(sp.grammar.type == COMMON_GRAMMAR_TYPE_TOOL_CALLS);
        CHECK(sp.grammar.grammar == p.grammar);
        CHECK(sp.grammar_lazy);
        // The WORD trigger opens on the added token: a TOKEN trigger on it, no regex.
        CHECK(sp.grammar_triggers.size() == 1 && sp.grammar_triggers[0].type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN
              && sp.grammar_triggers[0].token == 151657 && sp.grammar_triggers[0].value == "<tool_call>");
        // Without such a token the WORD trigger stays as it was.
        {
            common_params_sampling word;
            CHECK(tensai_tool_grammar::install(word, tg, p.generation_prompt, no_added));
            CHECK(word.grammar_triggers.size() == 1 && word.grammar_triggers[0].type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD
                  && word.grammar_triggers[0].value == p.grammar_triggers.at(0).value);
        }
        CHECK(sp.generation_prompt == p.generation_prompt);

        // A well-formed call fits; an unknown tool, arguments off the schema, or a
        // bare object without the marker do not.
        CHECK(grammar_accepts(sp.grammar.grammar, call));
        CHECK(grammar_accepts(sp.grammar.grammar,
            "<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"paris\"}}\n</tool_call>"));
        // Fired on the token alone, the grammar still holds the rest of the word:
        // the newline after <tool_call> is required.
        CHECK(!grammar_accepts(sp.grammar.grammar,
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"));
        // Once fired, the grammar is fed the trigger's tokens again, whole: a
        // "\n\n" or "\n " token after <tool_call> must still fit, or the accept
        // throws mid-decode. Both do (the grammar's `space`).
        CHECK(grammar_accepts(sp.grammar.grammar,
            "<tool_call>\n\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"));
        CHECK(grammar_accepts(sp.grammar.grammar,
            "<tool_call>\n {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"));
        // A token carrying something else after the newline does not, and throws.
        CHECK(!grammar_accepts(sp.grammar.grammar,
            "<tool_call>\n```json\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n```\n</tool_call>"));
        CHECK(!grammar_accepts(sp.grammar.grammar,
            "<tool_call>\n{\"name\": \"get_wether\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"));
        CHECK(!grammar_accepts(sp.grammar.grammar,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"query\": \"Paris\"}}\n</tool_call>"));
        CHECK(!grammar_accepts(sp.grammar.grammar,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": 75}}\n</tool_call>"));
        CHECK(!grammar_accepts(sp.grammar.grammar,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}\n</tool_call>"));
        CHECK(!grammar_accepts(sp.grammar.grammar,
            "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"));

        // What the grammar forces parses back as the call, after reasoning or not.
        const std::string reply = thinking ? "<think>\nThe user wants the weather.\n</think>\n\n" + call : call;
        const common_chat_msg msg = parse(p, reply);
        CHECK(msg.tool_calls.size() == 1);
        if (msg.tool_calls.size() == 1) {
            CHECK(msg.tool_calls[0].name == "get_weather");
            CHECK(msg.tool_calls[0].arguments.find("\"Paris\"") != std::string::npos);
        }

        // A bare object opening the reply (TEN-184) is not the trigger: the grammar
        // never starts, and the parser reads it as content, not as a call.
        const common_chat_msg bare = parse(p, "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}");
        CHECK(bare.tool_calls.empty());
        CHECK(p.grammar_triggers.at(0).value.find('{') == std::string::npos);
    }

    // tool_choice none: no grammar, nothing installed.
    {
        const common_chat_params p = render(tmpls.get(), true, COMMON_CHAT_TOOL_CHOICE_NONE);
        CHECK(p.grammar.empty());
        common_params_sampling sp;
        CHECK(!tensai_tool_grammar::install(sp, {p.grammar, p.grammar_lazy, p.grammar_triggers}, p.generation_prompt, added_tool_call));
        CHECK(sp.grammar.empty());
    }

    // A lazy grammar without a trigger could never fire: not installed.
    {
        common_params_sampling sp;
        CHECK(!tensai_tool_grammar::install(sp, {"root ::= \"x\"", true, {}}, "", added_tool_call));
        CHECK(sp.grammar.empty());
    }

    // clear() undoes install(), for the retry without grammar.
    {
        common_params_sampling sp;
        CHECK(tensai_tool_grammar::install(sp, {"root ::= \"x\"", true, {{COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "x"}}}, "p", no_added));
        tensai_tool_grammar::clear(sp);
        CHECK(sp.grammar.empty());
        CHECK(!sp.grammar_lazy);
        CHECK(sp.grammar_triggers.empty());
    }

    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
