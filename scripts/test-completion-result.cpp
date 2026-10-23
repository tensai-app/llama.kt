// Host-side test for tensai_completion_result.h. Run via scripts/test-completion-result.sh.
#include <cstdio>

#include "tensai_completion_result.h"

using tensai_completion::return_value;

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            failures++;                                                      \
        }                                                                    \
    } while (0)

int main() {
    // Ended on its own (EOS, stop word, n_predict): the count.
    CHECK(return_value(42, false, false) == 42);
    CHECK(return_value(0, false, false) == 0);
    // Interrupted: still a count, the tokens before the stop are real.
    CHECK(return_value(7, false, true) == 7);
    // llama_decode() failed: never a count, whatever was sampled before.
    CHECK(return_value(0, true, false) == -1);
    CHECK(return_value(300, true, false) == -1);
    // Failed and interrupted: the interrupt wins.
    CHECK(return_value(5, true, true) == 5);

    if (failures != 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
