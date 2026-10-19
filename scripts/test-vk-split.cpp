// Host-side test for tensai_vk_split.h (TEN-198). Run via scripts/test-vk-split.sh.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensai_vk_split.h"

using namespace tensai_vk;

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            failures++;                                                      \
        }                                                                    \
    } while (0)

// Same loop as lm_ggml_vk_mul_mat_q_f16: ranges must tile [0, m) exactly once,
// every start aligned to bm and to the offset alignment, budget respected
// (except for the one-unit floor), only the last chunk ragged.
static void check_plan(uint64_t m, uint64_t n, uint64_t k, uint64_t budget, uint64_t bm,
                       uint64_t a_row_bytes, uint64_t align) {
    const uint64_t unit = row_unit(bm, a_row_bytes, 4, align);
    CHECK(unit != 0 && unit % bm == 0);
    const uint64_t step = chunk_rows(m, n, k, budget, unit);
    CHECK(step >= 1 && step <= m);

    uint64_t covered = 0;
    uint64_t chunks = 0;
    for (uint64_t row0 = 0; row0 < m; row0 += step) {
        const uint64_t rows = std::min(step, m - row0);
        CHECK(row0 == covered);
        if (row0 != 0 || step != m) {
            CHECK(row0 % bm == 0);
            CHECK((row0 * a_row_bytes) % align == 0);
            CHECK((row0 * 4) % align == 0);
        }
        const uint64_t cap = (((m + max_chunks - 1) / max_chunks) + unit - 1) / unit * unit;
        if (budget != 0 && step > unit && step != cap && step != m) {
            CHECK(2 * rows * n * k <= budget);
        }
        if (row0 + rows < m) {
            CHECK(rows == step && rows % unit == 0);
        }
        covered += rows;
        chunks++;
    }
    CHECK(covered == m);
    CHECK(chunks <= max_chunks);
    if (budget == 0 || 2 * m * n * k <= budget || m <= unit) {
        CHECK(chunks == 1);
    }
}

static void test_parse() {
    std::printf("parse\n");
    CHECK(parse_split_mflops(nullptr) == 0);
    CHECK(parse_split_mflops("") == 0);
    CHECK(parse_split_mflops("0") == 0);
    CHECK(parse_split_mflops("abc") == 0);
    CHECK(parse_split_mflops("500x") == 0);
    CHECK(parse_split_mflops("-5") == 0);
    CHECK(parse_split_mflops("-0") == 0);
    CHECK(parse_split_mflops(" 5") != 7);
    CHECK(parse_split_mflops("500") == 500);
    CHECK(parse_split_mflops("2000") == 2000);
    CHECK(parse_split_mflops("99999999") == 1000000);
}

static void test_submit_mode() {
    std::printf("split submit\n");
    CHECK(parse_split_submit(nullptr));
    CHECK(parse_split_submit(""));
    CHECK(parse_split_submit("1"));
    CHECK(parse_split_submit("00"));
    CHECK(!parse_split_submit("0"));
}

static void test_unit() {
    std::printf("row_unit\n");
    // Q4_K, K=2048: 8 blocks * 144 B = 1152 B per row, align 16 -> 16 | 1152, unit = bm
    CHECK(row_unit(64, 1152, 4, 16) == 64);
    // f16, K=2048: 4096 B per row
    CHECK(row_unit(128, 4096, 4, 64) == 128);
    // D offset needs align/4 rows: align 256 -> 64 rows
    CHECK(row_unit(32, 4096, 4, 256) == 64);
    // odd row size (Q6_K K=768: 3 * 210 = 630 B), align 64 -> 32 rows
    CHECK(row_unit(16, 630, 4, 64) == 32);
    CHECK(row_unit(0, 630, 4, 64) == 0);
    CHECK(row_unit(16, 0, 4, 64) == 0);
}

static void test_chunks() {
    std::printf("chunk_rows\n");
    const uint64_t bms[] = { 32, 64, 128 };
    const uint64_t aligns[] = { 4, 16, 64, 256 };
    const uint64_t rowbytes[] = { 1152, 630, 4096, 144 * 3 };
    const uint64_t ms[] = { 1, 7, 8, 31, 32, 33, 63, 64, 100, 2048, 2049, 8192, 151936 };
    const uint64_t budgets[] = { 0, 1, 1000000ULL, 500000000ULL, 1000000000ULL, 2000000000ULL };
    for (uint64_t bm : bms) {
        for (uint64_t align : aligns) {
            for (uint64_t rb : rowbytes) {
                for (uint64_t m : ms) {
                    for (uint64_t budget : budgets) {
                        check_plan(m, 128, 2048, budget, bm, rb, align);
                        check_plan(m, 8, 576, budget, bm, rb, align);
                    }
                }
            }
        }
    }
    // 500 MFLOP on 8192 x 128 x 2048 (4.3 GFLOP): 500e6 / (2*128*2048) = 953 rows -> 896 (bm 128)
    CHECK(chunk_rows(8192, 128, 2048, 500'000'000ULL, 128) == 896);
    // Fits the budget: no split
    CHECK(chunk_rows(8192, 128, 2048, 5'000'000'000ULL, 128) == 8192);
    // Disabled
    CHECK(chunk_rows(8192, 128, 2048, 0, 128) == 8192);
    // Tiny budget: one unit floor
    CHECK(chunk_rows(8192, 128, 2048, 1, 128) == 128);
    // Chunk cap: a tiny budget on a huge M gives at most 64 chunks
    CHECK(chunk_rows(151936, 128, 2048, 1, 128) == 2432);
    CHECK(chunk_rows(8192, 128, 2048, 1, 64) == 128);
    CHECK(chunk_rows(100000, 1, 1, 1, 1) == 1563);
    // M below one tile: never split
    CHECK(chunk_rows(100, 128, 2048, 1, 128) == 100);
}

int main() {
    test_parse();
    test_submit_mode();
    test_unit();
    test_chunks();
    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
