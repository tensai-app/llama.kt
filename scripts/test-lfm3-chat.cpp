// Host-side test of LFM3 support in the vendored llama.cpp (LKT-49, LKT-52). Run via
// scripts/test-lfm3-chat.sh.
//
// Two halves: the architecture table has "lfm3" (what supportedArchitectures() lists, so
// the app lets the model load), and the chat layer renders its template with tools and
// reads its calls back. LFM3 writes calls as <tool_call><function=…><parameter=…>, not LFM2's
// <|tool_call_start|>[…], and nothing upstream added for it outside src/: the
// template's own markers are what the auto parser has to find.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "chat.h"
#include "llama-arch.h"
#include "llama-grammar.h"
#include "unicode.h"

// The chat layer links against these but only calls them with a model, and this
// test has none: common_chat_templates_init() gets the template source.
const char * llama_model_chat_template(const llama_model *, const char *) { std::abort(); }
int32_t llama_chat_apply_template(const char *, const llama_chat_message *, size_t, bool, char *, int32_t) { std::abort(); }
const llama_vocab * llama_model_get_vocab(const llama_model *) { std::abort(); }

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
    if (!f) {
        std::printf("cannot read %s\n", path);
        std::exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// The grammar past its trigger, as the sampler enforces it once triggered: true when
// `input` is a complete sentence of it.
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

static common_chat_params render(const common_chat_templates * tmpls, bool thinking) {
    common_chat_templates_inputs in;
    common_chat_msg user;
    user.role    = "user";
    user.content = "Weather in Paris?";
    in.messages = {user};
    in.tools = {
        {"get_weather", "Current weather of a city",
         R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})"},
    };
    in.tool_choice      = COMMON_CHAT_TOOL_CHOICE_AUTO;
    in.enable_thinking  = thinking;
    // What nativeFormatChat() asks for.
    in.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    return common_chat_templates_apply(tmpls, in);
}

static common_chat_msg parse(const common_chat_params & p, const std::string & reply) {
    // As rn-completion's parseChatOutput(): the reply alone, the generation prompt in the syntax.
    common_chat_parser_params syntax;
    syntax.format            = p.format;
    syntax.generation_prompt = common_chat_input(p.generation_prompt);
    // Through the string nativeFormatChat() keeps between format and parse.
    syntax.parser.load(p.parser.save());
    return common_chat_parse(common_chat_input(reply), false, syntax);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::printf("usage: %s LiquidAI-LFM3.jinja\n", argv[0]);
        return 1;
    }
    // nativeSupportedArchitectures() walks the same table.
    bool has_lfm3 = false;
    for (const llm_arch arch : llm_arch_all()) {
        if (std::string(llm_arch_name(arch)) == "lfm3") has_lfm3 = true;
    }
    CHECK(has_lfm3);

    common_chat_templates_ptr tmpls(common_chat_templates_init(nullptr, read_file(argv[1])));

    // The call as upstream's test-chat feeds it for LFM3 (19cce6919).
    const std::string call =
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    for (const bool thinking : {false, true}) {
        const common_chat_params p = render(tmpls.get(), thinking);
        std::printf("thinking=%d format=%s triggers=%zu grammar=%zu chars\n", thinking,
                    common_chat_format_name(p.format), p.grammar_triggers.size(), p.grammar.size());

        // The tools are in the prompt.
        CHECK(p.prompt.str().find("get_weather") != std::string::npos);

        // A lazy grammar, triggered by the call marker.
        CHECK(!p.grammar.empty());
        CHECK(p.grammar_lazy);
        CHECK(!p.grammar_triggers.empty());
        if (!p.grammar_triggers.empty()) {
            CHECK(p.grammar_triggers.at(0).type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD);
            CHECK(call.rfind(p.grammar_triggers.at(0).value, 0) == 0);
        }
        CHECK(grammar_accepts(p.grammar, call));
        CHECK(!grammar_accepts(p.grammar,
            "<tool_call>\n<function=get_wether>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"));

        // The call parses back, after reasoning or not.
        const std::string reply = thinking ? "The user wants the weather.</think>" + call : call;
        const common_chat_msg msg = parse(p, reply);
        CHECK(msg.tool_calls.size() == 1);
        if (msg.tool_calls.size() == 1) {
            CHECK(msg.tool_calls[0].name == "get_weather");
            CHECK(msg.tool_calls[0].arguments.find("\"Paris\"") != std::string::npos);
            std::printf("  parsed %s(%s)\n", msg.tool_calls[0].name.c_str(), msg.tool_calls[0].arguments.c_str());
        }
        if (thinking) CHECK(msg.reasoning_content.find("weather") != std::string::npos);

        // Plain text stays text.
        const common_chat_msg text = parse(p, thinking ? "Hmm.</think>It is sunny." : "It is sunny.");
        CHECK(text.tool_calls.empty());
        CHECK(text.content == "It is sunny.");
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("LFM3 chat template: OK\n");
    return 0;
}
