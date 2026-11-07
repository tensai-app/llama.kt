# Contributing to llama.kt

Kotlin/Android JNI binding for llama.cpp, the engine behind [tensai](https://github.com/tensai-app/tensai). arm64-v8a only, minSdk 29, llama.cpp pinned (see the badge in the [README](README.md)). Org-wide entry point, issue forms and security policy: [tensai-app/.github](https://github.com/tensai-app/.github).

## Test on your phone (no code)

Engine numbers on a new SoC are the most useful thing here: the Vulkan path is validated on one Mali-G68, the OpenCL path (Adreno 700+) is compiled in and untested on Adreno hardware. If you have one, run the app's benchmark (Settings > This device > Benchmark in [tensai](https://github.com/tensai-app/tensai), or the [`example/`](example/) app) and open a Device report on the tensai repo. The app sends nothing: you paste what you choose.

Tell us the backend (CPU, Vulkan, OpenCL), the model and quant, and the SoC. A run that crashes at load is as useful as one that works: attach the `llama.cpp` logcat lines.

## Build

Requirements: JDK 21, Android SDK platform 35, NDK r27+, CMake, `g++`, `glslc`. The README's "Building from source" is the reference.

```bash
git clone --recurse-submodules https://github.com/tensai-app/llama.kt.git
cd llama.kt
bash scripts/bootstrap.sh              # vendors llama.cpp into llama-kt/src/main/cpp/
bash scripts/build-opencl.sh           # libOpenCL.so ICD stub
./gradlew :example:assembleDebug
```

Vulkan shaders are generated with `scripts/gen-vulkan-shaders.sh "$(which glslc)"` (see tensai's README for timings and why a fresh clone needs it).

`bootstrap.sh` checks out llama.rn at the revision in `scripts/llama.rn.rev`, applies llama.rn's patches to an export of the `third_party/llama.cpp` submodule, copies the sources into `llama-kt/src/main/cpp/` with the `ggml_` to `lm_ggml_` rewrite, then applies the local patches. Both upstreams are pinned and coupled: bump them together, and never edit the vendored tree by hand (the next bootstrap overwrites it).

## Patches on llama.cpp

Local fixes to vendored code live in [`patches/`](patches/), numbered, and are applied at the end of `bootstrap.sh`. A local patch that no longer applies aborts the bootstrap on purpose.

- To change vendored llama.cpp or rn-llama code: change it in the vendored tree, produce the patch against the bootstrap output, add it to `patches/` with the next number and a name that says what it does, then re-run `bootstrap.sh` to check it applies from scratch.
- Document it in the README's patch list (what file, what behavior, why).
- Prefer sending the fix upstream to llama.cpp as well: a patch that lands there gets dropped here at the next bump (0004 and 0013 were).
- Logs must not carry prompt text or token ids: reaching logcat in release builds is the reason for patches 0010 and 0014.

## Checks before a PR

Kotlin, same as CI (no NDK needed):

```bash
./gradlew :llama-kt:testDebugUnitTest :example:compileDebugKotlin
```

Host tests, a C++ compiler is all they need (`CXX` to choose it):

```bash
scripts/test-kv-reuse.sh        # KV reuse planner
scripts/test-vk-split.sh        # split Vulkan matmul chunking math
scripts/test-input-ranges.sh    # message text tokenized as text; needs the llama.cpp submodule, skips without it
scripts/test-lfm3-chat.sh       # LFM3 architecture, chat template and GBNF parser; needs the vendored tree (bootstrap)
```

There are others next to them (`test-completion-result.sh`, `test-tool-grammar.sh`, `test-tool-grammar-sampler.sh`). Run the ones that cover what you touched. CI runs the native build and the host tests when `llama-kt/src/main/cpp/**` changes, and builds `:llama-kt:assembleDebug :example:assembleDebug`.

One Gradle run at a time, the native build is heavy. A change to the engine, a patch or a backend: say in the PR which model and device you ran it on.

## Commits and PRs

- Conventional Commits: `type(scope): subject`. Types: feat, fix, perf, refactor, docs, test, build, ci, chore, revert, style. Scopes seen in `git log`: `engine`, `build`.
- No issue or ticket id in the subject or body. Link the issue in the PR description.
- Fork, branch from `main`, PR against `main`.
- tensai consumes this repo as a submodule: after your PR lands, the maintainer bumps the pointer there. You don't need a PR on tensai for it.
- Match the surrounding code: comment density, KDoc style, naming. Public API changes update the README's API block.
- Apache-2.0, see [LICENSE](LICENSE). Third-party notices in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
