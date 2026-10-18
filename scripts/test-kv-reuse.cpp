// Host-side test for tensai_kv_reuse.h (TEN-196). Run via scripts/test-kv-reuse.sh.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensai_kv_reuse.h"

using tensai_kv::KvMove;
using tensai_kv::plan_kv_reuse;
using Toks = std::vector<int32_t>;

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static Toks seq(int32_t from, int n) {
    Toks t;
    for (int i = 0; i < n; i++) t.push_back(from + i);
    return t;
}

static Toks cat(std::initializer_list<Toks> parts) {
    Toks out;
    for (const auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

static size_t common_prefix(const Toks& a, const Toks& b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
    return i;
}

static std::vector<KvMove> plan(const Toks& cached, const Toks& prompt, size_t min_chunk = 32, size_t max_moves = 8) {
    return plan_kv_reuse(cached, prompt, common_prefix(cached, prompt), min_chunk, max_moves);
}

// Qwen3: <|im_start|>assistant\n<think>\n\n</think>\n\n answer...
// The replayed answer drops the 4-token empty think block.
static void test_empty_think_deleted() {
    std::printf("empty think block deleted\n");
    const Toks head = seq(1000, 50);   // history + "<|im_start|>assistant\n"
    const Toks think = {9001, 9002, 9003, 9004};
    const Toks answer = seq(2000, 415);
    const Toks cached = cat({head, think, answer});
    const Toks prompt = cat({head, answer, seq(3000, 12)});  // + next user turn
    const auto m = plan(cached, prompt);
    CHECK(m.size() == 1);
    if (m.size() == 1) {
        CHECK(m[0].src == 54);
        CHECK(m[0].dst == 50);
        CHECK(m[0].len == 415);
    }
}

static void test_long_reasoning_deleted() {
    std::printf("long reasoning deleted\n");
    const Toks head = seq(1000, 80);
    const Toks think = seq(5000, 900);
    const Toks answer = seq(2000, 300);
    const auto m = plan(cat({head, think, answer}), cat({head, answer, seq(3000, 20)}));
    CHECK(m.size() == 1);
    if (m.size() == 1) {
        CHECK(m[0].src == 980);
        CHECK(m[0].dst == 80);
        CHECK(m[0].len == 300);
    }
}

static void test_two_tool_rounds() {
    std::printf("two tool rounds give two moves\n");
    const Toks head = seq(1000, 40);
    const Toks think1 = seq(5000, 20), call1 = seq(2000, 60), tool1 = seq(2500, 50);
    const Toks think2 = seq(6000, 30), call2 = seq(2600, 70);
    const Toks cached = cat({head, think1, call1, tool1, think2, call2});
    const Toks prompt = cat({head, call1, tool1, call2, seq(3000, 10)});
    const auto m = plan(cached, prompt);
    CHECK(m.size() == 2);
    if (m.size() == 2) {
        // call1 + tool1 are contiguous in both lists: one run, then call2 after think2.
        CHECK(m[0].src == 60 && m[0].dst == 40 && m[0].len == 110);
        CHECK(m[1].src == 40 + 20 + 110 + 30 && m[1].dst == 150 && m[1].len == 70);
    }
}

static void test_pure_append() {
    std::printf("pure append gives no moves\n");
    const Toks cached = seq(1000, 200);
    CHECK(plan(cached, cat({cached, seq(3000, 40)})).empty());
}

static void test_think_at_end() {
    std::printf("think block at the very end gives no moves\n");
    const Toks head = seq(1000, 100);
    CHECK(plan(cat({head, seq(9000, 30)}), head).empty());
    CHECK(plan(cat({head, seq(9000, 30)}), cat({head, seq(3000, 5)})).empty());
}

static void test_insertion() {
    std::printf("insertion gives no moves\n");
    const Toks head = seq(1000, 50);
    const Toks tail = seq(2000, 100);
    // The prompt has tokens the cache lacks: nothing to shift, the tail is not reachable.
    const auto m = plan(cat({head, tail}), cat({head, seq(9000, 8), tail}));
    CHECK(m.empty());
}

static void test_retokenisation_partial() {
    std::printf("retokenisation mismatch gives a partial len\n");
    const Toks head = seq(1000, 50);
    const Toks think = seq(9000, 4);
    Toks cached = cat({head, think, seq(2000, 100)});
    Toks prompt = cat({head, seq(2000, 100)});
    prompt[50 + 70] = 77777;  // the template retokenises one token differently
    const auto m = plan(cached, prompt);
    CHECK(m.size() == 1);
    if (m.size() == 1) {
        CHECK(m[0].src == 54 && m[0].dst == 50 && m[0].len == 70);
    }
}

static void test_short_match_rejected() {
    std::printf("coincidental match shorter than min_chunk is rejected\n");
    const Toks head = seq(1000, 50);
    const Toks cached = cat({head, seq(9000, 20), seq(2000, 10), seq(8000, 40)});
    const Toks prompt = cat({head, seq(2000, 10), seq(7000, 5)});
    CHECK(plan(cached, prompt).empty());
    CHECK(plan(cached, prompt, 8).size() == 1);
}

static void test_max_moves() {
    std::printf("max_moves is respected\n");
    const Toks head = seq(1000, 10);
    Toks cached = head, prompt = head;
    for (int i = 0; i < 12; i++) {
        const Toks drop = seq(50000 + i * 10, 3);
        const Toks keep = seq(2000 + i * 100, 40);
        cached = cat({cached, drop, keep});
        prompt = cat({prompt, keep});
    }
    CHECK(plan(cached, prompt, 32, 8).size() == 8);
    CHECK(plan(cached, prompt, 32, 3).size() == 3);
    CHECK(plan(cached, prompt, 32, 0).empty());
    CHECK(plan(cached, prompt, 32, 100).size() == 12);
}

// The planned moves, applied to a model of the cache, must reproduce the prompt.
static void test_apply_reproduces_prompt() {
    std::printf("applying the plan reproduces the prompt prefix\n");
    const Toks head = seq(1000, 40);
    const Toks cached = cat({head, seq(9000, 4), seq(2000, 60), seq(9100, 7), seq(2100, 50)});
    const Toks prompt = cat({head, seq(2000, 60), seq(2100, 50), seq(3000, 9)});
    const auto m = plan(cached, prompt);
    CHECK(m.size() == 2);
    if (m.empty()) return;
    // cells: -1 = empty. seq_rm(dst, src) then seq_add(src, src + len, dst - src).
    std::vector<int64_t> cells(cached.begin(), cached.end());
    for (const auto& mv : m) {
        for (int i = mv.dst; i < mv.src; i++) cells[i] = -1;
        for (int i = mv.src; i < mv.src + mv.len; i++) {
            cells[i + (mv.dst - mv.src)] = cells[i];
            cells[i] = -1;
        }
    }
    const size_t end = m.back().dst + m.back().len;
    CHECK(end <= prompt.size());
    for (size_t i = 0; i < end; i++) CHECK(cells[i] == prompt[i]);
}

static void test_short_answer_no_move() {
    std::printf("short answer under min_chunk gives no moves\n");
    const Toks head = seq(1000, 50);
    CHECK(plan(cat({head, seq(9000, 4), seq(2000, 20)}), cat({head, seq(2000, 20), seq(3000, 6)})).empty());
}

static void test_first_token_retokenised() {
    std::printf("first replayed token retokenised differently\n");
    const Toks head = seq(1000, 50);
    const Toks ans = seq(2000, 200);
    Toks prompt = cat({head, ans});
    prompt[50] = 88888;  // answer starts with a token the cache does not have
    const auto m = plan(cat({head, seq(9000, 4), ans}), prompt);
    CHECK(m.empty());
    // The prompt token the cache lacks sits at the head of the run, and the planner
    // never skips prompt tokens (that would need a decode): no move, plain re-prefill.
    const Toks cached = cat({head, seq(9000, 4), Toks{77777}, Toks(ans.begin() + 1, ans.end())});
    CHECK(plan(cached, prompt).empty());
}

static void test_gain_gate() {
    std::printf("gain is the sum of reused tokens\n");
    CHECK(tensai_kv::kv_reuse_gain({}) == 0);
    const std::vector<KvMove> mv = {{60, 40, 110}, {200, 150, 70}};
    CHECK(tensai_kv::kv_reuse_gain(mv) == 180);
}

int main() {
    test_empty_think_deleted();
    test_long_reasoning_deleted();
    test_two_tool_rounds();
    test_pure_append();
    test_think_at_end();
    test_insertion();
    test_retokenisation_partial();
    test_short_match_rejected();
    test_max_moves();
    test_apply_reproduces_prompt();
    test_short_answer_no_move();
    test_first_token_retokenised();
    test_gain_gate();
    if (failures != 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all kv reuse tests passed\n");
    return 0;
}
