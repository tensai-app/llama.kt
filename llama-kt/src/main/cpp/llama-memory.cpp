#include "llama-memory.h"

#include "llama-ext.h"
#include "llama-io.h"

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

//
// llama_memory_bufs
//

lm_ggml_backend_buffer_ptr llama_memory_bufs::alloc_buf(lm_ggml_context * ctx, lm_ggml_backend_buffer_type_t buft, bool no_alloc, const char * name) {
    lm_ggml_backend_buffer_ptr buf;
    if (no_alloc) {
        buf.reset(lm_ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0)); // dummy buffer
        for (lm_ggml_tensor * t = lm_ggml_get_first_tensor(ctx); t != nullptr; t = lm_ggml_get_next_tensor(ctx, t)) {
            t->buffer = buf.get(); // set dummy buffer so that the backend scheduler won't try to allocate the tensors
        }
    } else {
        buf.reset(lm_ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)); // real buffer
    }
    if (!buf) {
        throw std::runtime_error(std::string("failed to allocate buffer for ") + name);
    }
    return buf;
}

lm_ggml_backend_buffer_t llama_memory_bufs::emplace(lm_ggml_context_ptr ctx, lm_ggml_backend_buffer_type_t buft, const char * name) {
    lm_ggml_backend_buffer_ptr buf = alloc_buf(ctx.get(), buft, no_alloc, name);

    // initialize the buffer to avoid NaNs in the padding
    lm_ggml_backend_buffer_clear(buf.get(), 0);

    bufs.emplace_back(std::move(ctx), std::move(buf));

    return bufs.back().second.get();
}

std::map<lm_ggml_backend_buffer_type_t, size_t> llama_memory_bufs::memory_breakdown() const {
    std::map<lm_ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : bufs) {
        lm_ggml_backend_buffer_type_t buft = lm_ggml_backend_buffer_get_type(buf.get());

        if (no_alloc) {
            LM_GGML_ASSERT(lm_ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += lm_ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            ret[buft] += lm_ggml_backend_buffer_get_size(buf.get());
        }
    }
    return ret;
}

// collects the buffers of the tensors that a state write visits, without keeping the data
struct llama_memory_buffers_writer : public llama_io_write_i {
    std::set<lm_ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(lm_ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

std::vector<lm_ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem) {
    if (mem == nullptr) {
        return {};
    }

    llama_memory_buffers_writer writer;

    mem->state_write(writer);

    return std::vector<lm_ggml_backend_buffer_t>(writer.buffers.begin(), writer.buffers.end());
}
