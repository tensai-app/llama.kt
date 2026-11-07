// Stands in for llama.rn's rn-tts.cpp (LKT-52), which bootstrap.sh does not vendor: since llama.rn 0.13 it is
// built on codec.cpp, a second vendored dependency, for text-to-speech llama.kt does not expose.
//
// rn-llama and rn-completion still hold a llama_rn_context_tts pointer and call into it behind
// isVocoderEnabled(). The constructor throws, so initVocoder() returns false and the pointer stays null; the
// methods below only exist for the linker. A llama.rn bump that changes rn-tts.h breaks the build here, which is
// the point: compare with its rn-tts.h and follow the signatures.
#include "rn-tts.h"

#include <stdexcept>

namespace rnllama {

llama_rn_context_tts::llama_rn_context_tts(const std::string &, int, bool) {
    throw std::runtime_error("text-to-speech is not built into llama.kt");
}

llama_rn_context_tts::~llama_rn_context_tts() = default;

void llama_rn_context_tts::reset() {}
tts_type llama_rn_context_tts::getTTSType(llama_rn_context *, json) { return UNKNOWN; }
tts_type llama_rn_context_tts::detectTTSType(llama_rn_context *) { return UNKNOWN; }
llama_rn_tts_capabilities llama_rn_context_tts::getTTSCapabilities(llama_rn_context *) { return {}; }
llama_rn_audio_completion_result llama_rn_context_tts::getFormattedAudioCompletion(llama_rn_context *, const json &,
                                                                                    const std::string &, int) {
    return {};
}
llama_rn_audio_codes_result llama_rn_context_tts::generateAudioCodes(llama_rn_context *, const llama_rn_audio_codes_options &,
                                                                     const llama_rn_audio_codes_progress_cb &) {
    return {};
}
bool llama_rn_context_tts::isTTSContinuous(llama_rn_context *) { return false; }
bool llama_rn_context_tts::tryContinuousAudioStep(llama_rn_context *, const float *, int) { return false; }
bool llama_rn_context_tts::tryContinuousPrefill(llama_rn_context *, const float *, int, int) { return false; }
bool llama_rn_context_tts::tryTalkerPrefill(llama_rn_context *, const float *, int) { return false; }
int llama_rn_context_tts::tryRealtimePrefill(llama_rn_context *, int n_past) { return n_past; }
bool llama_rn_context_tts::tryChatterboxPrefill(llama_rn_context *, const std::string &, const float *, int, int, float) {
    return false;
}
bool llama_rn_context_tts::isTTSCodecLmAR(llama_rn_context *) { return false; }
bool llama_rn_context_tts::tryCodecLmAudioStep(llama_rn_context *, llama_token, const float *, int) { return false; }
bool llama_rn_context_tts::encodeInto(llama_rn_context *, rn_speaker &, const std::vector<int32_t> &) { return false; }
int llama_rn_context_tts::createSpeaker(llama_rn_context *, const std::vector<float> &, int, const std::string &, float, bool,
                                        bool) {
    return -1;
}
bool llama_rn_context_tts::bakeSpeaker(llama_rn_context *, int) { return false; }
const rn_speaker * llama_rn_context_tts::getSpeaker(int) const { return nullptr; }
const rn_speaker * llama_rn_context_tts::autoBakeSpeaker(llama_rn_context *, int) { return nullptr; }
void llama_rn_context_tts::releaseSpeaker(int) {}
std::vector<float> llama_rn_context_tts::decodeAudioTokens(llama_rn_context *, const std::vector<llama_token> &) { return {}; }
std::vector<float> llama_rn_context_tts::decodeAudioEmbeddings(llama_rn_context *, const std::vector<float> &, int) {
    return {};
}
int llama_rn_context_tts::getAudioSampleRate() const { return 0; }
bool llama_rn_context_tts::isAudioToken(llama_rn_context *, llama_token, const std::string &) { return false; }
bool llama_rn_context_tts::tryAddAudioToken(llama_rn_context *, llama_token, const std::string &) { return false; }
bool llama_rn_context_tts::shouldCaptureAudioEmbeddings(llama_rn_context *) { return false; }

}  // namespace rnllama
