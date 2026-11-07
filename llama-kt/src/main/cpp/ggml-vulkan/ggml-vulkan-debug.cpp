#include "ggml-vulkan-common.h"

bool vk_memory_logger_enabled = false;

bool vk_perf_logger_enabled = false;

bool vk_perf_logger_concurrent = false;

bool vk_enable_sync_logger = false;

uint32_t vk_perf_logger_frequency = 1;

std::string vk_pipeline_stats_filter;

void vk_memory_logger::log_allocation(vk_buffer_ref buf_ref, size_t size) {
    if (!vk_memory_logger_enabled) {
        return;
    }
    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    const bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    const std::string type = device ? "device" : "host";
    allocations[buf->buffer] = size;
    total_device += device ? size : 0;
    total_host += device ? 0 : size;
    VK_LOG_MEMORY(buf->device->name << ": +" << format_size(size) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
}

void vk_memory_logger::log_deallocation(vk_buffer_ref buf_ref) {
    if (buf_ref.expired() || buf_ref.lock()->size == 0 || !vk_memory_logger_enabled) {
        return;
    }

    std::lock_guard<std::mutex> guard(log_mutex);
    vk_buffer buf = buf_ref.lock();
    const bool device = bool(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::string type = device ? "device" : "host";
    auto it = allocations.find(buf->buffer);
    if (it != allocations.end()) {
        total_device -= device ? it->second : 0;
        total_host -= device ? 0 : it->second;
        VK_LOG_MEMORY(buf->device->name << ": -" << format_size(it->second) << " " << type << " at " << buf->buffer << ". Total device: " << format_size(total_device) << ", total host: " << format_size(total_host));
        allocations.erase(it);
    } else {
        VK_LOG_MEMORY("ERROR " << buf->device->name << ": Attempted to deallocate unknown " << type << " memory at " << buf->buffer);
    }
}

#ifdef LM_GGML_VULKAN_CHECK_RESULTS
size_t vk_skip_checks;
size_t vk_output_tensor;

static void lm_ggml_vk_print_tensor(const lm_ggml_tensor * tensor, const char * name);
#endif

#ifdef LM_GGML_VULKAN_RUN_TESTS
static void lm_ggml_vk_print_matrix_area(const void * data, lm_ggml_type type, int ne0, int ne1, int i0, int i1, int i2) {
    if (type != LM_GGML_TYPE_F32 && type != LM_GGML_TYPE_F16) {
        return;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < ne0 && idx1 >= 0 && idx1 < ne1) {
                float val;
                if (type == LM_GGML_TYPE_F32) {
                    val = *((const float *) data + i2*ne1*ne0 + idx1*ne0 + idx0);
                } else if (type == LM_GGML_TYPE_F16) {
                    val = lm_ggml_fp16_to_fp32(*((const lm_ggml_fp16_t *) data + i2*ne1*ne0 + idx1*ne0 + idx0));
                } else {
                    LM_GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

template <typename X_TYPE, typename Y_TYPE>
void lm_ggml_vk_test_matmul(lm_ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t num_it, int split_k, int shader_size) {
    VK_LOG_DEBUG("lm_ggml_vk_test_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << num_it << ", " << split_k << ", " << shader_size << ")");
    const size_t x_ne = m * k * batch;
    const size_t y_ne = k * n * batch;
    const size_t d_ne = m * n * batch;

    lm_ggml_type x_type = std::is_same<float, X_TYPE>() ? LM_GGML_TYPE_F32 : LM_GGML_TYPE_F16;
    lm_ggml_type y_type = std::is_same<float, Y_TYPE>() ? LM_GGML_TYPE_F32 : LM_GGML_TYPE_F16;
    vk_matmul_pipeline_key mm_test_key{x_type, y_type, false, false};
    auto mm_test_it = ctx->device->pipeline_matmul.find(mm_test_key);
    LM_GGML_ASSERT(mm_test_it != ctx->device->pipeline_matmul.end() && !mm_test_it->second.empty());
    auto& mm_test_configs = mm_test_it->second;
    LM_GGML_ASSERT(shader_size >= 0 && shader_size < (int)mm_test_configs.size());

    std::string shname = std::string(lm_ggml_type_name(x_type)) + "_" + std::string(lm_ggml_type_name(y_type)) + "_ALIGNED_" + std::to_string(shader_size);
    vk_pipeline p = mm_test_configs[shader_size].aligned ? mm_test_configs[shader_size].aligned : mm_test_configs[shader_size].unaligned;

    const size_t kpad = lm_ggml_vk_align_size(k, mm_test_configs[shader_size].align);

    if (k != kpad) {
        p = mm_test_configs[shader_size].unaligned;
        shname = std::string(lm_ggml_type_name(x_type)) + "_" + std::string(lm_ggml_type_name(y_type)) + "_" + std::to_string(shader_size);
    }

    if (split_k > 1) {
        lm_ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, num_it);

        if (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < sizeof(float) * d_ne * split_k) {
            // Resize buffer
            if (ctx->prealloc_split_k != nullptr) {
                lm_ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = lm_ggml_vk_create_buffer_check(ctx->device, sizeof(float) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }

    lm_ggml_pipeline_allocate_descriptor_sets(ctx);

    vk_buffer d_X = lm_ggml_vk_create_buffer_check(ctx->device, sizeof(X_TYPE) * x_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer d_Y = lm_ggml_vk_create_buffer_check(ctx->device, sizeof(Y_TYPE) * y_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer d_D = lm_ggml_vk_create_buffer_check(ctx->device, sizeof(float) * d_ne, {vk::MemoryPropertyFlagBits::eDeviceLocal});

    X_TYPE* x = (X_TYPE *) malloc(sizeof(X_TYPE) * x_ne);
    Y_TYPE* y = (Y_TYPE *) malloc(sizeof(Y_TYPE) * y_ne);
    float* d = (float *) malloc(sizeof(float) * d_ne);

    for (size_t i = 0; i < x_ne; i++) {
        if (std::is_same<float, X_TYPE>()) {
            x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            // x[i] = 1.0f;
            // x[i] = i + 1;
            // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        } else if (std::is_same<lm_ggml_fp16_t, X_TYPE>()) {
            x[i] = lm_ggml_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
            // x[i] = lm_ggml_fp32_to_fp16(1.0f);
            // x[i] = lm_ggml_fp32_to_fp16(i + 1);
            // x[i] = lm_ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
        } else {
            LM_GGML_ABORT("fatal error");
        }
    }
    for (size_t i = 0; i < y_ne; i++) {
        if (std::is_same<float, Y_TYPE>()) {
            y[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
            // y[i] = i + 1;
        } else if (std::is_same<lm_ggml_fp16_t, Y_TYPE>()) {
            y[i] = lm_ggml_fp32_to_fp16((rand() / (float)RAND_MAX) * 2.0f - 1.0f);
            // y[i] = lm_ggml_fp32_to_fp16((i % k == i / k) ? 1.0f : 0.0f);
            // y[i] = lm_ggml_fp32_to_fp16(i + 1);
        } else {
            LM_GGML_ABORT("fatal error");
        }
    }

    lm_ggml_vk_buffer_write(d_X, 0, x, sizeof(X_TYPE) * k * m * batch);
    lm_ggml_vk_buffer_write(d_Y, 0, y, sizeof(Y_TYPE) * k * n * batch);

    vk_context subctx = lm_ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    lm_ggml_vk_ctx_begin(ctx->device, subctx);
    for (size_t i = 0; i < num_it; i++) {
        lm_ggml_vk_matmul(
            ctx, subctx, p, lm_ggml_vk_subbuffer(ctx, d_X), lm_ggml_vk_subbuffer(ctx, d_Y), lm_ggml_vk_subbuffer(ctx, d_D), lm_ggml_vk_subbuffer(ctx, ctx->prealloc_split_k),
            m, n, k,
            k, k, m, k*m, k*n, m*n,
            split_k, batch, batch, batch, 1, 1, n,
            k * sizeof(Y_TYPE), m * sizeof(float)
        );
    }
    lm_ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();
    lm_ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "lm_ggml_vk_test_matmul waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    lm_ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;

    // copy dst to host
    lm_ggml_vk_buffer_read(d_D, 0, d, sizeof(float) * d_ne);

    float * d_chk = (float *) malloc(sizeof(float) * d_ne);

    lm_ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    lm_ggml_context * lm_ggml_ctx = lm_ggml_init(iparams);

    lm_ggml_type src0_type;
    lm_ggml_type src1_type;

    if (std::is_same<float, X_TYPE>()) {
        src0_type = LM_GGML_TYPE_F32;
    } else if (std::is_same<lm_ggml_fp16_t, X_TYPE>()) {
        src0_type = LM_GGML_TYPE_F16;
    } else {
        LM_GGML_ABORT("fatal error");
    }
    if (std::is_same<float, Y_TYPE>()) {
        src1_type = LM_GGML_TYPE_F32;
    } else if (std::is_same<lm_ggml_fp16_t, Y_TYPE>()) {
        src1_type = LM_GGML_TYPE_F16;
    } else {
        LM_GGML_ABORT("fatal error");
    }

    lm_ggml_tensor * src0_ggml = lm_ggml_new_tensor_3d(lm_ggml_ctx, src0_type, k, m, batch);
    lm_ggml_tensor * src1_ggml = lm_ggml_new_tensor_3d(lm_ggml_ctx, src1_type, k, n, batch);
    lm_ggml_tensor * tensor_ggml = lm_ggml_mul_mat(lm_ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->data = x;
    src1_ggml->data = y;
    tensor_ggml->data = d_chk;

    lm_ggml_cgraph * cgraph = lm_ggml_new_graph(lm_ggml_ctx);
    lm_ggml_build_forward_expand(cgraph, tensor_ggml);

    lm_ggml_graph_compute_with_ctx(lm_ggml_ctx, cgraph, 1);

    lm_ggml_free(lm_ggml_ctx);

    double avg_err = 0.0;
    int first_err_n = -1;
    int first_err_m = -1;
    int first_err_b = -1;

    for (size_t i = 0; i < m*n*batch; i++) {
        double err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        if ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    double tflops = 2.0*m*n*k*batch*num_it / (time / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST " << shname << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time / num_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    if (avg_err > 0.1 || std::isnan(avg_err)) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        lm_ggml_vk_print_matrix_area(d, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << "Expected result: " << std::endl << std::endl;
        lm_ggml_vk_print_matrix_area(d_chk, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        if (split_k > 1) {
            float * split_k_buf = (float *) malloc(sizeof(float) * d_ne * split_k);
            lm_ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, sizeof(float) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            lm_ggml_vk_print_matrix_area(split_k_buf, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            lm_ggml_vk_print_matrix_area(split_k_buf + d_ne, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            lm_ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            lm_ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            free(split_k_buf);
        }
    }

    free(d_chk);

    lm_ggml_vk_command_pool_cleanup(ctx->device, ctx->compute_cmd_pool);

    lm_ggml_vk_destroy_buffer(d_X);
    lm_ggml_vk_destroy_buffer(d_Y);
    lm_ggml_vk_destroy_buffer(d_D);

    free(x);
    free(y);
    free(d);
}

template void lm_ggml_vk_test_matmul<lm_ggml_fp16_t, float>(lm_ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t num_it, int split_k, int shader_size);

static void lm_ggml_vk_print_tensor_area(const lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    if (tensor->type != LM_GGML_TYPE_F32 && tensor->type != LM_GGML_TYPE_F16) {
        return;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                float val;
                if (tensor->type == LM_GGML_TYPE_F32) {
                    val = *(float *) ((char *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } else if (tensor->type == LM_GGML_TYPE_F16) {
                    val = lm_ggml_fp16_to_fp32(*(lm_ggml_fp16_t *) ((char *) tensor->data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } else {
                    LM_GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

static void lm_ggml_vk_quantize_data(const float * from, void * to, size_t ne, lm_ggml_type quant) {
    lm_ggml_quantize_chunk(quant, from, to, 0, 1, ne, nullptr);
}

static void lm_ggml_vk_dequantize_data(const void * from, float * to, size_t ne, lm_ggml_type quant) {
    if (quant == LM_GGML_TYPE_F32) {
        memcpy(to, from, sizeof(float) * ne);
        return;
    }

    const auto * tt = lm_ggml_get_type_traits(quant);

    lm_ggml_to_float_t dequant_fn = tt->to_float;

    dequant_fn(from, to, ne);
}

void lm_ggml_vk_test_dequant(lm_ggml_backend_vk_context * ctx, size_t ne, lm_ggml_type quant) {
    VK_LOG_DEBUG("lm_ggml_vk_test_dequant(" << ne << ")");
    const size_t x_sz = sizeof(float) * ne;
    const size_t x_sz_f16 = sizeof(lm_ggml_fp16_t) * ne;
    const size_t qx_sz = ne * lm_ggml_type_size(quant)/lm_ggml_blck_size(quant);
    float * x = (float *) malloc(x_sz);
    void * qx = malloc(qx_sz);
    vk_buffer qx_buf = lm_ggml_vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer x_buf = lm_ggml_vk_create_buffer_check(ctx->device, x_sz_f16, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    float * x_ref = (float *) malloc(x_sz);
    lm_ggml_fp16_t * x_chk = (lm_ggml_fp16_t *) malloc(x_sz_f16);

    for (size_t i = 0; i < ne; i++) {
        x[i] = rand() / (float)RAND_MAX;
    }

    vk_pipeline p = lm_ggml_vk_get_to_fp16(ctx, quant);

    lm_ggml_vk_quantize_data(x, qx, ne, quant);
    lm_ggml_vk_dequantize_data(qx, x_ref, ne, quant);

    lm_ggml_pipeline_request_descriptor_sets(ctx, p, 1);

    lm_ggml_pipeline_allocate_descriptor_sets(ctx);

    lm_ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);

    vk_context subctx = lm_ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    lm_ggml_vk_ctx_begin(ctx->device, subctx);
    const std::vector<uint32_t> pc = { 1, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne, (uint32_t)ne };
    lm_ggml_vk_dispatch_pipeline(ctx, subctx, p, { vk_subbuffer{ qx_buf, 0, qx_sz }, vk_subbuffer{ x_buf, 0, x_sz_f16 } }, pc, { (uint32_t)ne, 1, 1});
    lm_ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    lm_ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "lm_ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    lm_ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    double ms_dequant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    lm_ggml_vk_buffer_read(x_buf, 0, x_chk, x_sz_f16);

    int first_err = -1;

    double avg_err = 0.0;
    for (size_t i = 0; i < ne; i++) {
        double error = std::fabs(x_ref[i] - lm_ggml_fp16_to_fp32(x_chk[i]));
        avg_err += error;

        if (first_err < 0 && error > 0.05) {
            first_err = i;
        }
    }

    avg_err /= ne;

    std::cerr << "TEST DEQUANT " << lm_ggml_type_name(quant) << " time=" << ms_dequant << "ms avg_err=" << avg_err << std::endl;

    if (avg_err > 0.1) {
        std::cerr << "first_error = " << first_err << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++) {
            std::cerr << lm_ggml_fp16_to_fp32(x_chk[i]) << ", ";
        }
        std::cerr << std::endl << "Expected result: " << std::endl << std::endl;
        for (int i = std::max(0, first_err - 5); i < std::min((int)ne, first_err + 5); i++) {
            std::cerr << x_ref[i] << ", ";
        }
        std::cerr << std::endl;
    }

    lm_ggml_vk_destroy_buffer(x_buf);
    lm_ggml_vk_destroy_buffer(qx_buf);

    free(x);
    free(qx);
    free(x_ref);
    free(x_chk);
}

// This does not work without ggml q8_1 quantization support
//
// typedef uint16_t lm_ggml_half;
// typedef uint32_t lm_ggml_half2;
//
// #define QK8_1 32
// typedef struct {
//     union {
//         struct {
//             lm_ggml_half d; // delta
//             lm_ggml_half s; // d * sum(qs[i])
//         } LM_GGML_COMMON_AGGR_S;
//         lm_ggml_half2 ds;
//     } LM_GGML_COMMON_AGGR_U;
//     int8_t qs[QK8_1]; // quants
// } block_q8_1;
//
// static void lm_ggml_vk_test_quantize(lm_ggml_backend_vk_context * ctx, size_t ne, lm_ggml_type quant) {
//     VK_LOG_DEBUG("lm_ggml_vk_test_quantize(" << ne << ")");
//     LM_GGML_ASSERT(quant == LM_GGML_TYPE_Q8_1);
//
//     const size_t x_sz = sizeof(float) * ne;
//     const size_t qx_sz = ne * lm_ggml_type_size(quant)/lm_ggml_blck_size(quant);
//     float * x = (float *) malloc(x_sz);
//     block_q8_1 * qx     = (block_q8_1 *)malloc(qx_sz);
//     block_q8_1 * qx_res = (block_q8_1 *)malloc(qx_sz);
//     vk_buffer x_buf = lm_ggml_vk_create_buffer_check(ctx->device, x_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//     vk_buffer qx_buf = lm_ggml_vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
//
//     for (size_t i = 0; i < ne; i++) {
//         x[i] = rand() / (float)RAND_MAX;
//     }
//
//     vk_pipeline p = lm_ggml_vk_get_quantize_pipeline(ctx, quant);
//
//     lm_ggml_pipeline_request_descriptor_sets(ctx, p, 1);
//
//     lm_ggml_pipeline_allocate_descriptor_sets(ctx);
//
//     lm_ggml_vk_buffer_write(x_buf, 0, x, x_sz);
//
//     vk_context subctx = lm_ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
//     lm_ggml_vk_ctx_begin(ctx->device, subctx);
//     lm_ggml_vk_quantize_q8_1(ctx, subctx, lm_ggml_vk_subbuffer(ctx, x_buf), lm_ggml_vk_subbuffer(ctx, qx_buf), ne);
//     lm_ggml_vk_ctx_end(subctx);
//
//     auto begin = std::chrono::high_resolution_clock::now();
//
//     lm_ggml_vk_submit(subctx, ctx->fence);
//     VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "lm_ggml_vk_test_quantize waitForFences");
//     ctx->device->device.resetFences({ ctx->fence });
//     lm_ggml_vk_queue_command_pools_cleanup(ctx->device);
//
//     auto end = std::chrono::high_resolution_clock::now();
//
//     double ms_quant = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
//     lm_ggml_vk_buffer_read(qx_buf, 0, qx, qx_sz);
//
//     lm_ggml_vk_quantize_data(x, qx_res, ne, quant);
//
//     int first_err = -1;
//
//     for (size_t i = 0; i < ne / 32; i++) {
//         double error = std::fabs(lm_ggml_fp16_to_fp32(qx_res[i].LM_GGML_COMMON_AGGR_U.LM_GGML_COMMON_AGGR_S.d) - lm_ggml_fp16_to_fp32(qx[i].LM_GGML_COMMON_AGGR_U.LM_GGML_COMMON_AGGR_S.d));
//
//         if (first_err < 0 && error > 0.1) {
//             first_err = i;
//         }
//
//         error = std::fabs(lm_ggml_fp16_to_fp32(qx_res[i].LM_GGML_COMMON_AGGR_U.LM_GGML_COMMON_AGGR_S.s) - lm_ggml_fp16_to_fp32(qx[i].LM_GGML_COMMON_AGGR_U.LM_GGML_COMMON_AGGR_S.s));
//
//         if (first_err < 0 && error > 0.1) {
//             first_err = i;
//         }
//
//         for (size_t j = 0; j < 32; j++) {
//             uint64_t error = std::abs(qx_res[i].qs[j] - qx[i].qs[j]);
//
//             if (first_err < 0 && error > 1) {
//                 first_err = i;
//             }
//         }
//     }
//
//     std::cerr << "TEST QUANTIZE " << lm_ggml_type_name(quant) << " time=" << ms_quant << "ms " << (first_err == -1 ? "CORRECT" : "INCORRECT") << std::endl;
//
//     if (first_err != -1) {
//         std::cerr << "first_error = " << first_err << std::endl;
//         std::cerr << "Actual result: " << std::endl << std::endl;
//         std::cout << "d=" << lm_ggml_fp16_to_fp32(qx[first_err].LM_GGML_COMMON_AGGR_U.LM_GGML_COMMON_AGGR_S.d) << " s=" << lm_ggml_fp16_to_fp32(qx[first_err].LM_GGML_COMMON_AGGR_U.LM_GGML_COMMON_AGGR_S.s) << " ";
//         for (size_t j = 0; j < 32; j++) {
//             std::cout << " qs" << j << "=" << (uint32_t)qx[first_err].qs[j] << " ";
//         }
//         std::cerr << std::endl << std::endl << "Expected result: " << std::endl << std::endl;
//         std::cout << "d=" << lm_ggml_fp16_to_fp32(qx_res[first_err].LM_GGML_COMMON_AGGR_U.LM_GGML_COMMON_AGGR_S.d) << " s=" << lm_ggml_fp16_to_fp32(qx_res[first_err].LM_GGML_COMMON_AGGR_U.LM_GGML_COMMON_AGGR_S.s) << " ";
//         for (size_t j = 0; j < 32; j++) {
//             std::cout << " qs" << j << "=" << (uint32_t)qx_res[first_err].qs[j] << " ";
//         }
//         std::cerr << std::endl;
//     }
//
//     lm_ggml_vk_destroy_buffer(x_buf);
//     lm_ggml_vk_destroy_buffer(qx_buf);
//
//     free(x);
//     free(qx);
//     free(qx_res);
// }

void lm_ggml_vk_test_dequant_matmul(lm_ggml_backend_vk_context * ctx, size_t m, size_t n, size_t k, size_t batch, size_t num_it, size_t split_k, size_t shader_size, lm_ggml_type quant, bool mmq) {
    VK_LOG_DEBUG("lm_ggml_vk_test_dequant_matmul(" << m << ", " << n << ", " << k << ", " << batch << ", " << num_it << ", " << split_k << ", " << lm_ggml_type_name(quant) << ")");
    const size_t x_ne = m * k * batch;
    const size_t y_ne = k * n * batch;
    const size_t d_ne = m * n * batch;

    lm_ggml_type b_type = mmq ? LM_GGML_TYPE_Q8_1 : LM_GGML_TYPE_F32;
    bool f16acc = ctx->device->fp16 && !mmq;
    vk_matmul_pipeline_key dq_key{quant, b_type, false, f16acc};
    auto dq_it = ctx->device->pipeline_matmul.find(dq_key);
    if (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        if (f16acc) {
            dq_key.f16acc = false;
            dq_it = ctx->device->pipeline_matmul.find(dq_key);
        }
    }
    if (dq_it == ctx->device->pipeline_matmul.end() || dq_it->second.empty()) {
        std::cerr << "error: no pipeline for lm_ggml_vk_test_dequant_matmul " << lm_ggml_type_name(quant) << std::endl;
        return;
    }
    auto& dq_configs = dq_it->second;
    if (shader_size >= (int)dq_configs.size()) {
        std::cerr << "error: shader_size " << shader_size << " >= configs.size() " << dq_configs.size() << " for " << lm_ggml_type_name(quant) << std::endl;
        return;
    }

    std::string shname = std::string(lm_ggml_type_name(quant)) + "_ALIGNED_" + std::to_string(shader_size);
    vk_pipeline p = dq_configs[shader_size].aligned ? dq_configs[shader_size].aligned : dq_configs[shader_size].unaligned;

    const size_t kpad = mmq ? 0 : lm_ggml_vk_align_size(k, dq_configs[shader_size].align);

    if (mmq || k != kpad) {
        p = dq_configs[shader_size].unaligned;
        shname = std::string(lm_ggml_type_name(quant)) + "_" + std::to_string(shader_size);
    }

    if (p == nullptr) {
        std::cerr << "error: no pipeline for lm_ggml_vk_test_dequant_matmul " << lm_ggml_type_name(quant) << std::endl;
        return;
    }

    const size_t x_sz = sizeof(float) * x_ne;
    const size_t y_sz = sizeof(float) * y_ne;
    const size_t qx_sz = x_ne * lm_ggml_type_size(quant)/lm_ggml_blck_size(quant);
    const size_t qy_sz = mmq ? y_ne * lm_ggml_type_size(LM_GGML_TYPE_Q8_1)/lm_ggml_blck_size(LM_GGML_TYPE_Q8_1) : y_sz;
    const size_t d_sz = sizeof(float) * d_ne;
    float * x = (float *) malloc(x_sz);
    float * y = (float *) malloc(y_sz);
    void * qx = malloc(qx_sz);
    vk_buffer qx_buf = lm_ggml_vk_create_buffer_check(ctx->device, qx_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer y_buf = lm_ggml_vk_create_buffer_check(ctx->device, y_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer qy_buf = lm_ggml_vk_create_buffer_check(ctx->device, qy_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    vk_buffer d_buf = lm_ggml_vk_create_buffer_check(ctx->device, d_sz, {vk::MemoryPropertyFlagBits::eDeviceLocal});
    float * d = (float *) malloc(d_sz);
    float * d_chk = (float *) malloc(d_sz);

    for (size_t i = 0; i < x_ne; i++) {
        x[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        // x[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // x[i] = i % k;
    }

    lm_ggml_vk_quantize_data(x, qx, x_ne, quant);

    for (size_t i = 0; i < y_ne; i++) {
        y[i] = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        // y[i] = (i % k == i / k) ? 1.0f : 0.0f;
        // y[i] = i % k;
    }

    if (split_k > 1) {
        lm_ggml_pipeline_request_descriptor_sets(ctx, ctx->device->pipeline_matmul_split_k_reduce, num_it);

        if (ctx->prealloc_split_k == nullptr || ctx->prealloc_split_k->size < sizeof(float) * d_ne * split_k) {
            // Resize buffer
            if (ctx->prealloc_split_k != nullptr) {
                lm_ggml_vk_destroy_buffer(ctx->prealloc_split_k);
            }
            ctx->prealloc_split_k = lm_ggml_vk_create_buffer_check(ctx->device, sizeof(float) * d_ne * split_k, {vk::MemoryPropertyFlagBits::eDeviceLocal});
        }
    }
    if (mmq) {
        vk_pipeline pipeline_quantize_q8_1 = lm_ggml_vk_get_quantize_pipeline(ctx, LM_GGML_TYPE_Q8_1);
        lm_ggml_pipeline_request_descriptor_sets(ctx, pipeline_quantize_q8_1, num_it);
    }

    lm_ggml_pipeline_allocate_descriptor_sets(ctx);

    lm_ggml_vk_buffer_write(qx_buf, 0, qx, qx_sz);
    lm_ggml_vk_buffer_write(y_buf, 0, y, y_sz);

    vk_context subctx = lm_ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
    lm_ggml_vk_ctx_begin(ctx->device, subctx);
    if (mmq) {
        for (size_t i = 0; i < num_it; i++) {
            lm_ggml_vk_quantize_q8_1(ctx, subctx, { y_buf, 0, y_sz }, { qy_buf, 0, qy_sz }, y_ne);
            lm_ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { qy_buf, 0, qy_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                (uint64_t)(k / lm_ggml_blck_size(LM_GGML_TYPE_Q8_1)) * lm_ggml_type_size(LM_GGML_TYPE_Q8_1), m * sizeof(float)
            );
        }
    } else {
        for (size_t i = 0; i < num_it; i++) {
            lm_ggml_vk_matmul(
                ctx, subctx, p, { qx_buf, 0, qx_sz }, { y_buf, 0, y_sz }, { d_buf, 0, d_sz }, { ctx->prealloc_split_k, 0, ctx->prealloc_size_split_k },
                m, n, k,
                k, k, m, k*m, k*n, m*n,
                split_k, batch, batch, batch, 1, 1, n,
                k * sizeof(float), m * sizeof(float)
            );
        }
    }
    lm_ggml_vk_ctx_end(subctx);

    auto begin = std::chrono::high_resolution_clock::now();

    lm_ggml_vk_submit(subctx, ctx->fence);
    VK_CHECK(ctx->device->device.waitForFences({ ctx->fence }, true, UINT64_MAX), "lm_ggml_vk_test_dequant waitForFences", ctx->device);
    ctx->device->device.resetFences({ ctx->fence });
    lm_ggml_vk_queue_command_pools_cleanup(ctx->device);

    auto end = std::chrono::high_resolution_clock::now();

    double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000.0;
    lm_ggml_vk_buffer_read(d_buf, 0, d, d_sz);

    lm_ggml_init_params iparams = {
        /*.mem_size   =*/ 1024*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    lm_ggml_context * lm_ggml_ctx = lm_ggml_init(iparams);

    lm_ggml_tensor * src0_ggml = lm_ggml_new_tensor_3d(lm_ggml_ctx, quant, k, m, batch);
    lm_ggml_tensor * src1_ggml = lm_ggml_new_tensor_3d(lm_ggml_ctx, LM_GGML_TYPE_F32, k, n, batch);
    lm_ggml_tensor * tensor_ggml = lm_ggml_mul_mat(lm_ggml_ctx, src0_ggml, src1_ggml);

    src0_ggml->data = qx;
    src1_ggml->data = y;
    tensor_ggml->data = d_chk;

    lm_ggml_cgraph * cgraph = lm_ggml_new_graph(lm_ggml_ctx);
    lm_ggml_build_forward_expand(cgraph, tensor_ggml);

    lm_ggml_graph_compute_with_ctx(lm_ggml_ctx, cgraph, 1);

    lm_ggml_free(lm_ggml_ctx);

    double avg_err = 0.0;
    int first_err_n = -1;
    int first_err_m = -1;
    int first_err_b = -1;

    for (size_t i = 0; i < m*n*batch; i++) {
        double err = std::fabs(d[i] - d_chk[i]);
        avg_err += err;

        if ((err > 0.05f || std::isnan(err)) && first_err_n == -1) {
            first_err_b = i / (m * n);
            first_err_n = (i % (m * n)) / m;
            first_err_m = (i % (m * n)) % m;
        }
    }

    avg_err /= m * n;

    double tflops = 2.0*m*n*k*batch*num_it / (time_ms / 1000.0) / (1000.0*1000.0*1000.0*1000.0);

    std::cerr << "TEST dequant matmul " << shname;
    if (mmq) {
        std::cerr << " mmq";
    }
    std::cerr << " m=" << m << " n=" << n << " k=" << k << " batch=" << batch << " split_k=" << split_k << " matmul " << time_ms / num_it << "ms " << tflops << " TFLOPS avg_err=" << avg_err << std::endl;

    if (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "m = " << first_err_m << " n = " << first_err_n << " b = " << first_err_b << std::endl;
        std::cerr << "Actual result: " << std::endl << std::endl;
        lm_ggml_vk_print_matrix_area(d, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);
        std::cerr << std::endl;
        std::cerr << "Expected result: " << std::endl << std::endl;
        lm_ggml_vk_print_matrix_area(d_chk, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

        std::cerr << "src0: " << std::endl << std::endl;
        lm_ggml_vk_print_matrix_area(x, LM_GGML_TYPE_F32, k, m, first_err_m, first_err_n, first_err_b);
        std::cerr << std::endl;
        std::cerr << "src1: " << std::endl << std::endl;
        lm_ggml_vk_print_matrix_area(y, LM_GGML_TYPE_F32, k, n, first_err_m, first_err_n, first_err_b);

        if (split_k > 1) {
            float * split_k_buf = (float *) malloc(sizeof(float) * d_ne * split_k);
            lm_ggml_vk_buffer_read(ctx->prealloc_split_k, 0, split_k_buf, sizeof(float) * d_ne * split_k);

            std::cerr << "d_buf0: " << std::endl << std::endl;
            lm_ggml_vk_print_matrix_area(split_k_buf, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf1: " << std::endl << std::endl;
            lm_ggml_vk_print_matrix_area(split_k_buf + d_ne, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf2: " << std::endl << std::endl;
            lm_ggml_vk_print_matrix_area(split_k_buf + 2 * d_ne, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            std::cerr << "d_buf3: " << std::endl << std::endl;
            lm_ggml_vk_print_matrix_area(split_k_buf + 3 * d_ne, LM_GGML_TYPE_F32, m, n, first_err_m, first_err_n, first_err_b);

            free(split_k_buf);
        }
    }

    lm_ggml_vk_destroy_buffer(qx_buf);
    lm_ggml_vk_destroy_buffer(y_buf);
    lm_ggml_vk_destroy_buffer(qy_buf);
    lm_ggml_vk_destroy_buffer(d_buf);

    free(x);
    free(qx);
    free(y);
    free(d);
    free(d_chk);
}
#endif

int64_t lm_ggml_vk_get_op_batch_size(const lm_ggml_tensor * op) {
    switch (op->op) {
        case LM_GGML_OP_GET_ROWS:
            return 0;
        case LM_GGML_OP_MUL_MAT:
            return op->ne[1];
        case LM_GGML_OP_MUL_MAT_ID:
        case LM_GGML_OP_ROPE:
        case LM_GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return lm_ggml_nrows(op);
    }
}

#ifdef LM_GGML_VULKAN_CHECK_RESULTS
static void lm_ggml_vk_print_graph_origin(const lm_ggml_tensor * tensor, std::vector<const lm_ggml_tensor *>& done, int level = 0) {
    if (std::find(done.begin(), done.end(), tensor) != done.end() || level > 10) {
        return;
    }
    for (int j = 0; j < level; j++) {
        std::cerr << " ";
    }
    std::cerr << lm_ggml_op_name(tensor->op) << " gpu=" << (tensor->extra != nullptr) << std::endl;

    done.push_back(tensor);

    for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
        if (tensor->src[i] != nullptr) {
            lm_ggml_vk_print_graph_origin(tensor->src[i], done, level + 1);
        }
    }
}

static void lm_ggml_vk_print_tensor_area(const lm_ggml_tensor * tensor, const void * data, int i0, int i1, int i2, int i3) {
    if (tensor->type != LM_GGML_TYPE_F32 && tensor->type != LM_GGML_TYPE_F16 && tensor->type != LM_GGML_TYPE_I32) {
        return;
    }
    i0 = std::max(i0, 5);
    i1 = std::max(i1, 5);
    i2 = std::max(i2, 0);
    i3 = std::max(i3, 0);
    fprintf(stderr, "         ");
    for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
        fprintf(stderr, "%7d ", idx1);
    }
    fprintf(stderr, "\n");
    for (int idx0 = i0 - 5; idx0 < i0 + 5; idx0++) {
        fprintf(stderr, "%7d: ", idx0);
        for (int idx1 = i1 - 5; idx1 < i1 + 5; idx1++) {
            if (idx0 >= 0 && idx0 < tensor->ne[0] && idx1 >= 0 && idx1 < tensor->ne[1] && i2 >= 0 && i2 < tensor->ne[2] && i3 >= 0 && i3 < tensor->ne[3]) {
                float val;
                if (tensor->type == LM_GGML_TYPE_F32) {
                    val = *(const float *) ((const char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } else if (tensor->type == LM_GGML_TYPE_F16) {
                    val = lm_ggml_fp16_to_fp32(*(const lm_ggml_fp16_t *) ((const char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]));
                } else if (tensor->type == LM_GGML_TYPE_I32) {
                    val = *(const int32_t *) ((const char *) data + i3*tensor->nb[3] + i2*tensor->nb[2] + idx1*tensor->nb[1] + idx0*tensor->nb[0]);
                } else {
                    LM_GGML_ABORT("fatal error");
                }
                fprintf(stderr, "% 7.2f ", val);
            } else {
                fprintf(stderr, "        ");
            }
        }
        fprintf(stderr, "\n");
    }
}

static void lm_ggml_vk_print_tensor(const lm_ggml_tensor * tensor, const char * name) {
    void * tensor_data = tensor->data;

    const bool is_gpu = tensor->buffer != nullptr && lm_ggml_backend_buffer_is_vk(tensor->buffer);

    if (is_gpu) {
        const size_t tensor_size = lm_ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        lm_ggml_backend_vk_buffer_context * buf_ctx = (lm_ggml_backend_vk_buffer_context *)tensor->buffer->context;

        vk_buffer buffer_gpu = buf_ctx->dev_buffer;
        lm_ggml_vk_buffer_read(buffer_gpu, vk_tensor_offset(tensor) + tensor->view_offs, tensor_data, tensor_size);
    }

    std::cerr << "TENSOR CHECK " << name << " (" << tensor->name << "): " << lm_ggml_op_name(tensor->op) << std::endl;
    std::cerr << "tensor=" << tensor << " tensor->type: " << lm_ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << std::endl;
    if (tensor->src[0] != nullptr) {
        std::cerr << "tensor->src[0]=" << tensor->src[0] << " name=" << tensor->src[0]->name << " op=" << lm_ggml_op_name(tensor->src[0]->op) << " type=" << lm_ggml_type_name(tensor->src[0]->type) << " ne0=" << tensor->src[0]->ne[0] << " nb0=" << tensor->src[0]->nb[0] << " ne1=" << tensor->src[0]->ne[1] << " nb1=" << tensor->src[0]->nb[1] << " ne2=" << tensor->src[0]->ne[2] << " nb2=" << tensor->src[0]->nb[2] << " ne3=" << tensor->src[0]->ne[3] << " nb3=" << tensor->src[0]->nb[3] << std::endl;
    }
    if (tensor->src[1] != nullptr) {
        std::cerr << "tensor->src[1]=" << tensor->src[1] << " name=" << tensor->src[1]->name << " op=" << lm_ggml_op_name(tensor->src[1]->op) << " type=" << lm_ggml_type_name(tensor->src[1]->type) << " ne0=" << tensor->src[1]->ne[0] << " nb0=" << tensor->src[1]->nb[0] << " ne1=" << tensor->src[1]->ne[1] << " nb1=" << tensor->src[1]->nb[1] << " ne2=" << tensor->src[1]->ne[2] << " nb2=" << tensor->src[1]->nb[2] << " ne3=" << tensor->src[1]->ne[3] << " nb3=" << tensor->src[1]->nb[3] << std::endl;
    }
    std::cerr << std::endl << "Result:" << std::endl;
    lm_ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
    std::cerr << std::endl;
    std::vector<const lm_ggml_tensor *> done;
    lm_ggml_vk_print_graph_origin(tensor, done);

    if (is_gpu) {
        free(tensor_data);
    }
}

void * comp_result;
size_t comp_size;
size_t comp_nb[LM_GGML_MAX_DIMS];
size_t check_counter = 0;
void lm_ggml_vk_check_results_0(lm_ggml_backend_vk_context * ctx, lm_ggml_cgraph * cgraph, int tensor_idx) {
    lm_ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->num_additional_fused_ops];
    if (tensor->op == LM_GGML_OP_TRANSPOSE || tensor->op == LM_GGML_OP_SET_ROWS) {
        return;
    }

    check_counter++;
    if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        return;
    }

    VK_LOG_DEBUG("lm_ggml_vk_check_results_0(" << tensor->name << ")");

    struct lm_ggml_init_params iparams = {
        /*.mem_size   =*/ 2ul*1024ul*1024ul*1024ul,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct lm_ggml_context * lm_ggml_ctx = lm_ggml_init(iparams);

    std::array<struct lm_ggml_tensor *, LM_GGML_MAX_SRC> src_clone = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    const char * srci_name[LM_GGML_MAX_SRC] = {"src0", "src1", "src2", "src3", "src4", "src5", "src6", "src7", "src8", "src9"};

    std::map<lm_ggml_tensor *, lm_ggml_tensor *> cloned_tensors;
    std::vector<void *> cloned_mallocs;

    struct lm_ggml_tensor * tensor_clone = nullptr;

    for (int f = 0; f < ctx->num_additional_fused_ops + 1; ++f) {
        tensor = cgraph->nodes[tensor_idx + f];
        for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
            lm_ggml_tensor * srci = tensor->src[i];
            if (srci == nullptr) {
                continue;
            }
            // If a src tensor has been cloned, use that one
            auto it = cloned_tensors.find(srci);
            if (it != cloned_tensors.end()) {
                src_clone[i] = it->second;
                continue;
            }
            lm_ggml_tensor * srci_clone = lm_ggml_dup_tensor(lm_ggml_ctx, srci);
            size_t srci_size = lm_ggml_nbytes(srci);

            src_clone[i] = srci_clone;
            void *src_buffer = malloc(srci_size);
            cloned_mallocs.push_back(src_buffer);

            srci_clone->data = src_buffer;
            if (lm_ggml_backend_buffer_is_host(srci->buffer)) {
                memcpy(srci_clone->data, srci->data, srci_size);
                memcpy(srci_clone->nb, srci->nb, sizeof(size_t) * LM_GGML_MAX_DIMS);
            } else if (lm_ggml_backend_buffer_is_vk(srci->buffer)) {
                lm_ggml_backend_vk_buffer_context * buf_ctx = (lm_ggml_backend_vk_buffer_context *)srci->buffer->context;
                vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
                uint64_t offset = vk_tensor_offset(srci) + srci->view_offs;
                if (!lm_ggml_is_contiguous(srci) && lm_ggml_vk_dim01_contiguous(srci)) {
                    for (int i3 = 0; i3 < srci->ne[3]; i3++) {
                        for (int i2 = 0; i2 < srci->ne[2]; i2++) {
                            const int idx = i3*srci->ne[2] + i2;
                            lm_ggml_vk_buffer_read(buffer_gpu, offset + idx * srci->nb[2], ((char *)srci_clone->data + idx * srci_clone->nb[2]), srci->ne[1] * srci->nb[1]);
                        }
                    }

                    srci_clone->nb[0] = srci->nb[0];
                    srci_clone->nb[1] = srci->nb[1];
                    for (int i = 2; i < LM_GGML_MAX_DIMS; i++) {
                        srci_clone->nb[i] = srci_clone->nb[i - 1]*srci_clone->ne[i - 1];
                    }
                } else {
                    if (offset + srci_size >= buffer_gpu->size) {
                        srci_size = buffer_gpu->size - offset;
                    }
                    lm_ggml_vk_buffer_read(buffer_gpu, offset, srci_clone->data, srci_size);
                    memcpy(srci_clone->nb, srci->nb, sizeof(size_t) * LM_GGML_MAX_DIMS);
                }
            } else {
                LM_GGML_ABORT("fatal error");
            }

            if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
                lm_ggml_vk_print_tensor(srci, srci_name[i]);
            }
        }

        if (tensor->op == LM_GGML_OP_FLASH_ATTN_EXT) {
            const float * params = (const float *)tensor->op_params;
            tensor_clone = lm_ggml_flash_attn_ext(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3], params[0], params[1], params[2]);
            if (src_clone[4]) {
                lm_ggml_flash_attn_ext_add_sinks(tensor_clone, src_clone[4]);
            }
        } else if (tensor->op == LM_GGML_OP_MUL_MAT) {
            tensor_clone = lm_ggml_mul_mat(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_MUL_MAT_ID) {
            tensor_clone = lm_ggml_mul_mat_id(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } else if (tensor->op == LM_GGML_OP_SUB) {
            tensor_clone = lm_ggml_sub(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_MUL) {
            tensor_clone = lm_ggml_mul(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_DIV) {
            tensor_clone = lm_ggml_div(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_CONCAT) {
            tensor_clone = lm_ggml_concat(lm_ggml_ctx, src_clone[0], src_clone[1], *(int *)tensor->op_params);
        } else if (tensor->op == LM_GGML_OP_UPSCALE) {
            tensor_clone = lm_ggml_interpolate(lm_ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], (lm_ggml_scale_mode) tensor->op_params[0]);
        } else if (tensor->op == LM_GGML_OP_SCALE) {
            const float * params = (const float *)tensor->op_params;
            tensor_clone = lm_ggml_scale_bias(lm_ggml_ctx, src_clone[0], params[0], params[1]);
        } else if (tensor->op == LM_GGML_OP_ADD1) {
            tensor_clone = lm_ggml_add1(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_ARANGE) {
            const float start = lm_ggml_get_op_params_f32(tensor, 0);
            const float stop = lm_ggml_get_op_params_f32(tensor, 1);
            const float step = lm_ggml_get_op_params_f32(tensor, 2);
            tensor_clone = lm_ggml_arange(lm_ggml_ctx, start, stop, step);
        } else if (tensor->op == LM_GGML_OP_FILL) {
            const float value = lm_ggml_get_op_params_f32(tensor, 0);
            tensor_clone = lm_ggml_fill(lm_ggml_ctx, src_clone[0], value);
        } else if (tensor->op == LM_GGML_OP_SQR) {
            tensor_clone = lm_ggml_sqr(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_SQRT) {
            tensor_clone = lm_ggml_sqrt(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_SIN) {
            tensor_clone = lm_ggml_sin(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_COS) {
            tensor_clone = lm_ggml_cos(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_LOG) {
            tensor_clone = lm_ggml_log(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_TRI) {
            tensor_clone = lm_ggml_tri(lm_ggml_ctx, src_clone[0], (lm_ggml_tri_type)lm_ggml_get_op_params_i32(tensor, 0));
        } else if (tensor->op == LM_GGML_OP_DIAG) {
            tensor_clone = lm_ggml_diag(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_CLAMP) {
            const float * params = (const float *)tensor->op_params;
            tensor_clone = lm_ggml_clamp(lm_ggml_ctx, src_clone[0], params[0], params[1]);
        } else if (tensor->op == LM_GGML_OP_PAD) {
            tensor_clone = lm_ggml_pad_ext(lm_ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3],
                                                                tensor->op_params[4], tensor->op_params[5], tensor->op_params[6], tensor->op_params[7]);
        } else if (tensor->op == LM_GGML_OP_PAD_REFLECT_1D) {
            tensor_clone = lm_ggml_pad_reflect_1d(lm_ggml_ctx, src_clone[0], tensor->op_params[0], tensor->op_params[1]);
        } else if (tensor->op == LM_GGML_OP_REPEAT) {
            tensor_clone = lm_ggml_repeat(lm_ggml_ctx, src_clone[0], tensor);
        } else if (tensor->op == LM_GGML_OP_REPEAT_BACK) {
            tensor_clone = lm_ggml_repeat_back(lm_ggml_ctx, src_clone[0], tensor);
        } else if (tensor->op == LM_GGML_OP_ADD) {
            tensor_clone = lm_ggml_add(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_ACC) {
            tensor_clone = lm_ggml_acc(lm_ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } else if (tensor->op == LM_GGML_OP_SET) {
            tensor_clone = lm_ggml_set(lm_ggml_ctx, src_clone[0], src_clone[1], tensor->op_params[0], tensor->op_params[1], tensor->op_params[2], tensor->op_params[3]);
        } else if (tensor->op == LM_GGML_OP_NORM) {
            tensor_clone = lm_ggml_norm(lm_ggml_ctx, src_clone[0], *(float *)tensor->op_params);
        } else if (tensor->op == LM_GGML_OP_GROUP_NORM) {
            const float * float_params = (const float *)tensor->op_params;
            tensor_clone = lm_ggml_group_norm(lm_ggml_ctx, src_clone[0], tensor->op_params[0], float_params[1]);
        } else if (tensor->op == LM_GGML_OP_RMS_NORM) {
            tensor_clone = lm_ggml_rms_norm(lm_ggml_ctx, src_clone[0], *(float *)tensor->op_params);
        } else if (tensor->op == LM_GGML_OP_RMS_NORM_BACK) {
            const float eps = ((float *) tensor->op_params)[0];
            tensor_clone = lm_ggml_rms_norm_back(lm_ggml_ctx, src_clone[0], src_clone[1], eps);
        } else if (tensor->op == LM_GGML_OP_SILU_BACK) {
            tensor_clone = lm_ggml_silu_back(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_L2_NORM) {
            const float eps = ((float *) tensor->op_params)[0];
            tensor_clone = lm_ggml_l2_norm(lm_ggml_ctx, src_clone[0], eps);
        } else if (tensor->op == LM_GGML_OP_SOFT_MAX) {
            if (tensor->src[1] != nullptr) {
                const float * params = (const float *)tensor->op_params;
                tensor_clone = lm_ggml_soft_max_ext(lm_ggml_ctx, src_clone[0], src_clone[1], params[0], params[1]);
            } else {
                tensor_clone = lm_ggml_soft_max(lm_ggml_ctx, src_clone[0]);
            }
        } else if (tensor->op == LM_GGML_OP_SOFT_MAX_BACK) {
            tensor_clone = lm_ggml_soft_max_ext_back(lm_ggml_ctx, src_clone[0], src_clone[1], ((float *)tensor->op_params)[0], ((float *)tensor->op_params)[1]);
        } else if (tensor->op == LM_GGML_OP_DIAG_MASK_INF) {
            tensor_clone = lm_ggml_diag_mask_inf(lm_ggml_ctx, src_clone[0], tensor->op_params[0]);
        } else if (tensor->op == LM_GGML_OP_ROPE || tensor->op == LM_GGML_OP_ROPE_BACK) {
            const int n_dims      = ((int32_t *) tensor->op_params)[1];
            const int mode        = ((int32_t *) tensor->op_params)[2];
            //const int n_ctx_ggml       = ((int32_t *) tensor->op_params)[3];
            const int n_ctx_orig_ggml  = ((int32_t *) tensor->op_params)[4];
            const float freq_base       = ((float *) tensor->op_params)[5];
            const float freq_scale      = ((float *) tensor->op_params)[6];
            const float ext_factor      = ((float *) tensor->op_params)[7];
            const float attn_factor     = ((float *) tensor->op_params)[8];
            const float beta_fast       = ((float *) tensor->op_params)[9];
            const float beta_slow       = ((float *) tensor->op_params)[10];
            if (mode & LM_GGML_ROPE_TYPE_MROPE) {
                int32_t *sections = ((int32_t *) tensor->op_params) + 11;
                if (tensor->op == LM_GGML_OP_ROPE) {
                    tensor_clone = lm_ggml_rope_multi(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } else {
                    tensor_clone = lm_ggml_rope_multi_back(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, sections, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            } else {
                if (tensor->op == LM_GGML_OP_ROPE) {
                    tensor_clone = lm_ggml_rope_ext(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                } else {
                    tensor_clone = lm_ggml_rope_ext_back(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2], n_dims, mode, n_ctx_orig_ggml, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                }
            }
            const int n_offs = ((int32_t *) tensor->op_params)[15];
            if (n_offs != 0) {
                tensor_clone = lm_ggml_rope_set_offset(tensor_clone, n_offs);
            }
        } else if (tensor->op == LM_GGML_OP_UNARY) {
            switch (lm_ggml_get_unary_op(tensor)) {
            case LM_GGML_UNARY_OP_EXP:
                tensor_clone = lm_ggml_exp(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_EXPM1:
                tensor_clone = lm_ggml_expm1(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_ELU:
                tensor_clone = lm_ggml_elu(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_SILU:
                tensor_clone = lm_ggml_silu(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_GELU:
                tensor_clone = lm_ggml_gelu(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_GELU_ERF:
                tensor_clone = lm_ggml_gelu_erf(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_GELU_QUICK:
                tensor_clone = lm_ggml_gelu_quick(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_RELU:
                tensor_clone = lm_ggml_relu(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_XIELU:
                tensor_clone = lm_ggml_xielu(lm_ggml_ctx, src_clone[0], 0, 0, 0, 0);
                lm_ggml_set_op_params_f32(tensor_clone, 1, lm_ggml_get_op_params_f32(tensor, 1));
                lm_ggml_set_op_params_f32(tensor_clone, 2, lm_ggml_get_op_params_f32(tensor, 2));
                lm_ggml_set_op_params_f32(tensor_clone, 3, lm_ggml_get_op_params_f32(tensor, 3));
                lm_ggml_set_op_params_f32(tensor_clone, 4, lm_ggml_get_op_params_f32(tensor, 4));
                break;
            case LM_GGML_UNARY_OP_NEG:
                tensor_clone = lm_ggml_neg(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_TANH:
                tensor_clone = lm_ggml_tanh(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_SIGMOID:
                tensor_clone = lm_ggml_sigmoid(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_HARDSIGMOID:
                tensor_clone = lm_ggml_hardsigmoid(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_HARDSWISH:
                tensor_clone = lm_ggml_hardswish(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_ABS:
                tensor_clone = lm_ggml_abs(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_SOFTPLUS:
                tensor_clone = lm_ggml_softplus(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_STEP:
                tensor_clone = lm_ggml_step(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_ROUND:
                tensor_clone = lm_ggml_round(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_CEIL:
                tensor_clone = lm_ggml_ceil(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_FLOOR:
                tensor_clone = lm_ggml_floor(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_TRUNC:
                tensor_clone = lm_ggml_trunc(lm_ggml_ctx, src_clone[0]);
                break;
            case LM_GGML_UNARY_OP_SGN:
                tensor_clone = lm_ggml_sgn(lm_ggml_ctx, src_clone[0]);
                break;
            default:
                std::cerr << "Missing vk_check_results OP: " << lm_ggml_op_name(tensor->op) << std::endl;
                LM_GGML_ABORT("fatal error");
            }
        } else if (tensor->op == LM_GGML_OP_GLU) {
            if (src_clone[1] == nullptr) {
                tensor_clone = lm_ggml_glu(lm_ggml_ctx, src_clone[0], (lm_ggml_glu_op) tensor->op_params[0], tensor->op_params[1]);
            } else {
                tensor_clone = lm_ggml_glu_split(lm_ggml_ctx, src_clone[0], src_clone[1], (lm_ggml_glu_op) tensor->op_params[0]);
            }
            lm_ggml_set_op_params_i32(tensor_clone, 2, lm_ggml_get_op_params_i32(tensor, 2));
            lm_ggml_set_op_params_i32(tensor_clone, 3, lm_ggml_get_op_params_i32(tensor, 3));
        } else if (tensor->op == LM_GGML_OP_CPY || tensor->op == LM_GGML_OP_DUP) {
            if (tensor->src[1] == nullptr) {
                tensor_clone = lm_ggml_dup(lm_ggml_ctx, src_clone[0]);
                tensor_clone->type = tensor->type;
            } else {
                tensor_clone = lm_ggml_cpy(lm_ggml_ctx, src_clone[0], src_clone[1]);
            }
        } else if (tensor->op == LM_GGML_OP_CONT) {
            tensor_clone = lm_ggml_cont_4d(lm_ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } else if (tensor->op == LM_GGML_OP_RESHAPE) {
            tensor_clone = lm_ggml_reshape_4d(lm_ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } else if (tensor->op == LM_GGML_OP_VIEW) {
            tensor_clone = lm_ggml_view_4d(lm_ggml_ctx, src_clone[0], tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->nb[1], tensor->nb[2], tensor->nb[3], ((int32_t *) tensor->op_params)[0]);
        } else if (tensor->op == LM_GGML_OP_PERMUTE) {
            int32_t * params = (int32_t *)tensor->op_params;
            tensor_clone = lm_ggml_permute(lm_ggml_ctx, src_clone[0], params[0], params[1], params[2], params[3]);
        } else if (tensor->op == LM_GGML_OP_TRANSPOSE) {
            tensor_clone = lm_ggml_transpose(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_GET_ROWS) {
            tensor_clone = lm_ggml_get_rows(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_ARGSORT) {
            tensor_clone = lm_ggml_argsort(lm_ggml_ctx, src_clone[0], (lm_ggml_sort_order) *(int *)tensor->op_params);
        } else if (tensor->op == LM_GGML_OP_TOP_K) {
            tensor_clone = lm_ggml_top_k(lm_ggml_ctx, src_clone[0], tensor->ne[0]);
        } else if (tensor->op == LM_GGML_OP_SUM) {
            tensor_clone = lm_ggml_sum(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_SUM_ROWS) {
            tensor_clone = lm_ggml_sum_rows(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_CUMSUM) {
            tensor_clone = lm_ggml_cumsum(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_DSV4_HC_COMB) {
            tensor_clone = lm_ggml_dsv4_hc_comb(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                lm_ggml_get_op_params_f32(tensor, 0), lm_ggml_get_op_params_i32(tensor, 1));
        } else if (tensor->op == LM_GGML_OP_DSV4_HC_PRE) {
            if (lm_ggml_get_op_params_i32(tensor, 1) != 0) {
                tensor_clone = lm_ggml_dsv4_hc_pre_gated(lm_ggml_ctx, src_clone[0], src_clone[1], lm_ggml_get_op_params_f32(tensor, 0));
            } else {
                tensor_clone = lm_ggml_dsv4_hc_pre(lm_ggml_ctx, src_clone[0], src_clone[1]);
            }
        } else if (tensor->op == LM_GGML_OP_DSV4_HC_POST) {
            tensor_clone = lm_ggml_dsv4_hc_post(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } else if (tensor->op == LM_GGML_OP_MEAN) {
            tensor_clone = lm_ggml_mean(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_ARGMAX) {
            tensor_clone = lm_ggml_argmax(lm_ggml_ctx, src_clone[0]);
        } else if (tensor->op == LM_GGML_OP_CROSS_ENTROPY_LOSS) {
            tensor_clone = lm_ggml_cross_entropy_loss(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK) {
            tensor_clone = lm_ggml_cross_entropy_loss_back(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } else if (tensor->op == LM_GGML_OP_COUNT_EQUAL) {
            tensor_clone = lm_ggml_count_equal(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_SOLVE_TRI) {
            tensor_clone = lm_ggml_solve_tri(lm_ggml_ctx, src_clone[0], src_clone[1], true, true, false);
        } else if (tensor->op == LM_GGML_OP_IM2COL) {
            const int32_t s0 = tensor->op_params[0];
            const int32_t s1 = tensor->op_params[1];
            const int32_t p0 = tensor->op_params[2];
            const int32_t p1 = tensor->op_params[3];
            const int32_t d0 = tensor->op_params[4];
            const int32_t d1 = tensor->op_params[5];

            const bool is_2D = tensor->op_params[6] == 1;
            tensor_clone = lm_ggml_im2col(lm_ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1, is_2D, tensor->type);
        } else if (tensor->op == LM_GGML_OP_IM2COL_3D) {
            const int32_t s0 = tensor->op_params[0];
            const int32_t s1 = tensor->op_params[1];
            const int32_t s2 = tensor->op_params[2];
            const int32_t p0 = tensor->op_params[3];
            const int32_t p1 = tensor->op_params[4];
            const int32_t p2 = tensor->op_params[5];
            const int32_t d0 = tensor->op_params[6];
            const int32_t d1 = tensor->op_params[7];
            const int32_t d2 = tensor->op_params[8];
            const int32_t IC = tensor->op_params[9];

            tensor_clone = lm_ggml_im2col_3d(lm_ggml_ctx, src_clone[0], src_clone[1], IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, tensor->type);
        } else if (tensor->op == LM_GGML_OP_TIMESTEP_EMBEDDING) {
            const int32_t dim = tensor->op_params[0];
            const int32_t max_period = tensor->op_params[1];
            tensor_clone = lm_ggml_timestep_embedding(lm_ggml_ctx, src_clone[0], dim, max_period);
        } else if (tensor->op == LM_GGML_OP_CONV_TRANSPOSE_1D){
            const int32_t s0 = tensor->op_params[0];
            const int32_t p0 = tensor->op_params[1];
            const int32_t d0 = tensor->op_params[2];
            tensor_clone = lm_ggml_conv_transpose_1d(lm_ggml_ctx, src_clone[0], src_clone[1], s0, p0, d0);
        } else if (tensor->op == LM_GGML_OP_COL2IM_1D) {
            const int32_t stride = tensor->op_params[0];
            const int32_t oc     = tensor->op_params[1];
            const int32_t p0     = tensor->op_params[2];
            tensor_clone = lm_ggml_col2im_1d(lm_ggml_ctx, src_clone[0], stride, oc, p0);
        } else if (tensor->op == LM_GGML_OP_POOL_1D) {
            enum lm_ggml_op_pool op = static_cast<lm_ggml_op_pool>(tensor->op_params[0]);
            const int32_t k0 = tensor->op_params[1];
            const int32_t s0 = tensor->op_params[2];
            const int32_t p0 = tensor->op_params[3];

            tensor_clone = lm_ggml_pool_1d(lm_ggml_ctx, src_clone[0], op, k0, s0, p0);
        } else if (tensor->op == LM_GGML_OP_POOL_2D) {
            enum lm_ggml_op_pool op = static_cast<lm_ggml_op_pool>(tensor->op_params[0]);
            const int32_t k0 = tensor->op_params[1];
            const int32_t k1 = tensor->op_params[2];
            const int32_t s0 = tensor->op_params[3];
            const int32_t s1 = tensor->op_params[4];
            const int32_t p0 = tensor->op_params[5];
            const int32_t p1 = tensor->op_params[6];

            tensor_clone = lm_ggml_pool_2d(lm_ggml_ctx, src_clone[0], op, k0, k1, s0, s1, p0, p1);
        } else if (tensor->op == LM_GGML_OP_CONV_2D) {
            const int32_t s0 = tensor->op_params[0];
            const int32_t s1 = tensor->op_params[1];
            const int32_t p0 = tensor->op_params[2];
            const int32_t p1 = tensor->op_params[3];
            const int32_t d0 = tensor->op_params[4];
            const int32_t d1 = tensor->op_params[5];
            tensor_clone = lm_ggml_conv_2d(lm_ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } else if (tensor->op == LM_GGML_OP_CONV_3D) {
            const int32_t s0 = tensor->op_params[0];
            const int32_t s1 = tensor->op_params[1];
            const int32_t s2 = tensor->op_params[2];
            const int32_t p0 = tensor->op_params[3];
            const int32_t p1 = tensor->op_params[4];
            const int32_t p2 = tensor->op_params[5];
            const int32_t d0 = tensor->op_params[6];
            const int32_t d1 = tensor->op_params[7];
            const int32_t d2 = tensor->op_params[8];
            const int32_t IC = tensor->op_params[9];
            const int32_t N  = tensor->op_params[10];
            const int32_t OC = tensor->op_params[11];
            tensor_clone = lm_ggml_conv_3d_direct(lm_ggml_ctx, src_clone[0], src_clone[1], s0, s1, s2, p0, p1, p2, d0, d1, d2, IC, N, OC);
        } else if (tensor->op == LM_GGML_OP_CONV_2D_DW) {
            const int32_t s0 = tensor->op_params[0];
            const int32_t s1 = tensor->op_params[1];
            const int32_t p0 = tensor->op_params[2];
            const int32_t p1 = tensor->op_params[3];
            const int32_t d0 = tensor->op_params[4];
            const int32_t d1 = tensor->op_params[5];
            tensor_clone = lm_ggml_conv_2d_dw_direct(lm_ggml_ctx, src_clone[0], src_clone[1], s0, s1, p0, p1, d0, d1);
        } else if (tensor->op == LM_GGML_OP_CONV_TRANSPOSE_2D) {
            const int32_t s = tensor->op_params[0];
            tensor_clone = lm_ggml_conv_transpose_2d_p0(lm_ggml_ctx, src_clone[0], src_clone[1], s);
        } else if (tensor->op == LM_GGML_OP_LEAKY_RELU) {
            const float * op_params = (const float *)tensor->op_params;
            tensor_clone = lm_ggml_leaky_relu(lm_ggml_ctx, src_clone[0], op_params[0], false);
        } else if (tensor->op == LM_GGML_OP_RWKV_WKV6) {
            tensor_clone = lm_ggml_rwkv_wkv6(lm_ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5]);
        } else if (tensor->op == LM_GGML_OP_RWKV_WKV7) {
            tensor_clone = lm_ggml_rwkv_wkv7(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3],
            src_clone[4], src_clone[5], src_clone[6]);
        } else if (tensor->op == LM_GGML_OP_GATED_LINEAR_ATTN) {
            const float * op_params = (const float *)tensor->op_params;
            tensor_clone = lm_ggml_gated_linear_attn(lm_ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], op_params[0]);
        } else if (tensor->op == LM_GGML_OP_LIGHTNING_INDEXER) {
            tensor_clone = lm_ggml_lightning_indexer(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2], src_clone[3]);
        } else if (tensor->op == LM_GGML_OP_GATED_DELTA_NET) {
            tensor_clone = lm_ggml_gated_delta_net(lm_ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4], src_clone[5],
            lm_ggml_get_op_params_i32(tensor, 0));
        } else if (tensor->op == LM_GGML_OP_OPT_STEP_ADAMW) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = lm_ggml_opt_step_adamw(lm_ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2], src_clone[3], src_clone[4]);
        } else if (tensor->op == LM_GGML_OP_OPT_STEP_SGD) {
            src_clone[0]->flags = tensor->src[0]->flags;
            tensor_clone = lm_ggml_opt_step_sgd(lm_ggml_ctx, src_clone[0], src_clone[1],
            src_clone[2]);
        } else if (tensor->op == LM_GGML_OP_ADD_ID) {
            tensor_clone = lm_ggml_add_id(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2]);
        } else if (tensor->op == LM_GGML_OP_SSM_SCAN) {
            const int32_t K = lm_ggml_get_op_params_i32(tensor, 0);
            tensor_clone = lm_ggml_ssm_scan(lm_ggml_ctx, src_clone[0], src_clone[1], src_clone[2],
                                         src_clone[3], src_clone[4], src_clone[5], src_clone[6], K);
        } else if (tensor->op == LM_GGML_OP_SSM_CONV) {
            tensor_clone = lm_ggml_ssm_conv(lm_ggml_ctx, src_clone[0], src_clone[1]);
        } else if (tensor->op == LM_GGML_OP_ROLL) {
            const int32_t s0 = tensor->op_params[0];
            const int32_t s1 = tensor->op_params[1];
            const int32_t s2 = tensor->op_params[2];
            const int32_t s3 = tensor->op_params[3];
            tensor_clone = lm_ggml_roll(lm_ggml_ctx, src_clone[0], s0, s1, s2, s3);
        }
        else {
            std::cerr << "Missing vk_check_results OP: " << lm_ggml_op_name(tensor->op) << std::endl;
            LM_GGML_ABORT("fatal error");
        }
        cloned_tensors[tensor] = tensor_clone;
    }

    lm_ggml_cgraph * cgraph_cpu = lm_ggml_new_graph(lm_ggml_ctx);
    lm_ggml_build_forward_expand(cgraph_cpu, tensor_clone);

    lm_ggml_graph_compute_with_ctx(lm_ggml_ctx, cgraph_cpu, 8);

    if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        lm_ggml_vk_print_tensor(tensor_clone, "tensor_clone");
    }

    comp_size = lm_ggml_nbytes(tensor_clone);

    comp_result = malloc(comp_size);
    memcpy(comp_result, tensor_clone->data, comp_size);
    memcpy(comp_nb, tensor_clone->nb, sizeof(size_t) * LM_GGML_MAX_DIMS);

    for (auto m : cloned_mallocs) {
        free(m);
    }

    lm_ggml_free(lm_ggml_ctx);

    VK_LOG_DEBUG("END lm_ggml_vk_check_results_0(" << tensor->name << ")");
}

void lm_ggml_vk_check_results_1(lm_ggml_backend_vk_context * ctx, lm_ggml_cgraph * cgraph, int tensor_idx) {
    lm_ggml_tensor * tensor = cgraph->nodes[tensor_idx + ctx->num_additional_fused_ops];
    if (tensor->op == LM_GGML_OP_TRANSPOSE || tensor->op == LM_GGML_OP_SET_ROWS) {
        return;
    }

    if (!(vk_output_tensor > 0 && vk_output_tensor == check_counter) && check_counter <= vk_skip_checks) {
        return;
    }

    VK_LOG_DEBUG("lm_ggml_vk_check_results_1(" << tensor->name << ")");

    lm_ggml_tensor * src0 = tensor->src[0];
    lm_ggml_tensor * src1 = tensor->src[1];
    lm_ggml_tensor * src2 = tensor->src[2];
    lm_ggml_tensor * src3 = tensor->src[3];

    void * tensor_data = tensor->data;

    if (lm_ggml_backend_buffer_is_vk(tensor->buffer)) {
        size_t tensor_size = lm_ggml_nbytes(tensor);
        tensor_data = malloc(tensor_size);

        lm_ggml_backend_vk_buffer_context * buf_ctx = (lm_ggml_backend_vk_buffer_context *)tensor->buffer->context;

        vk_buffer& buffer_gpu = buf_ctx->dev_buffer;
        uint64_t offset = vk_tensor_offset(tensor) + tensor->view_offs;
        if (offset + tensor_size >= buffer_gpu->size) {
            tensor_size = buffer_gpu->size - offset;
        }

        lm_ggml_vk_buffer_read(buffer_gpu, offset, tensor_data, tensor_size);
    }

    float first_error_result = -1.0f;
    float first_error_correct = -1.0f;
    std::array<int, 4> first_error = { -1, -1, -1, -1 };
    double avg_err = 0.0;
    size_t counter = 0;

    for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
        for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    const bool buffer_size_fit = i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0] < comp_size;
                    float correct = 0.0f;
                    float result = 0.0f;

                    if (buffer_size_fit) {
                        if (tensor->type == LM_GGML_TYPE_F32) {
                            correct = *(float *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(float *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } else if (tensor->type == LM_GGML_TYPE_F16) {
                            correct = lm_ggml_fp16_to_fp32(*(lm_ggml_fp16_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            result  = lm_ggml_fp16_to_fp32(*(lm_ggml_fp16_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } else if (tensor->type == LM_GGML_TYPE_BF16) {
                            correct = lm_ggml_bf16_to_fp32(*(lm_ggml_bf16_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]));
                            result  = lm_ggml_bf16_to_fp32(*(lm_ggml_bf16_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]));
                        } else if (tensor->type == LM_GGML_TYPE_I32) {
                            correct = *(int32_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(int32_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } else if (tensor->type == LM_GGML_TYPE_I64) {
                            correct = *(int64_t *) ((char *) comp_result + i3*comp_nb[3] + i2*comp_nb[2] + i1*comp_nb[1] + i0*comp_nb[0]);
                            result  = *(int64_t *) ((char *) tensor_data + i3*tensor->nb[3] + i2*tensor->nb[2] + i1*tensor->nb[1] + i0*tensor->nb[0]);
                        } else {
                            std::cerr << "Results check not implemented for type " << lm_ggml_type_name(tensor->type) << std::endl;
                        }
                    } else {
                        std::cerr << "Missing debug code for type " << lm_ggml_type_name(tensor->type) << std::endl;
                        LM_GGML_ABORT("fatal error");
                    }

                    if ((std::isnan(correct) != std::isnan(result)) || (std::isinf(correct) != std::isinf(result)) || !buffer_size_fit) {
                        std::cerr << "ERROR: Invalid value in " << lm_ggml_op_name(tensor->op) << " i3=" << i3 << " i2=" << i2 << " i1=" << i1 << " i0=" << i0 << " result=" << result << " correct=" << correct << " avg_err=" << (avg_err / counter) << std::endl;
                        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << lm_ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
                        if (src0 != nullptr) {
                            std::cerr << "src0=" << src0 << " src0->name=" << src0->name << " op=" << lm_ggml_op_name(src0->op) << " type=" << lm_ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
                        }
                        if (src1 != nullptr) {
                            std::cerr << "src1=" << src1 << " src1->name=" << src1->name << " op=" << lm_ggml_op_name(src1->op) << " type=" << lm_ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
                        }
                        if (src2 != nullptr) {
                            std::cerr << "src2=" << src2 << " src2->name=" << src2->name << " op=" << lm_ggml_op_name(src2->op) << " type=" << lm_ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
                        }
                        if (src3 != nullptr) {
                            std::cerr << "src3=" << src3 << " src3->name=" << src3->name << " op=" << lm_ggml_op_name(src3->op) << " type=" << lm_ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
                        }
                        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
                        std::cerr << std::endl << "Result:" << std::endl;
                        lm_ggml_vk_print_tensor_area(tensor, tensor_data, i0, i1, i2, i3);
                        std::cerr << std::endl << "Correct:" << std::endl;
                        lm_ggml_vk_print_tensor_area(tensor, comp_result, i0, i1, i2, i3);
                        std::cerr << std::endl;
                        std::vector<const lm_ggml_tensor *> done;
                        lm_ggml_vk_print_graph_origin(tensor, done);
                        LM_GGML_ABORT("fatal error");
                    }
                    const double denom = std::fabs(correct) > 1.0f ? (std::fabs(correct) > 1e-8 ? std::fabs(correct) : 1e-8) : 1.0f;
                    if (first_error[0] == -1 && std::fabs(correct - result) / denom > 0.5) {
                        first_error[0] = i0;
                        first_error[1] = i1;
                        first_error[2] = i2;
                        first_error[3] = i3;
                        first_error_result = result;
                        first_error_correct = correct;
                    }

                    // Special case, value is infinite, avoid NaN result in avg_err
                    // NaN also appears in results, if both are nan error is 0
                    if (!std::isinf(correct) && !std::isinf(result) && !std::isnan(correct) && !std::isnan(result)) {
                        avg_err += std::fabs(correct - result) / denom;
                    }
                    counter++;
                }
            }
        }
    }

    avg_err /= counter;

    if (vk_output_tensor > 0 && vk_output_tensor == check_counter) {
        std::cerr << "TENSOR CHECK: avg_err=" << avg_err << " in " << lm_ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << lm_ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        if (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << lm_ggml_op_name(src0->op) << " type=" << lm_ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        if (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << lm_ggml_op_name(src1->op) << " type=" << lm_ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        if (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << lm_ggml_op_name(src2->op) << " type=" << lm_ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        if (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << lm_ggml_op_name(src3->op) << " type=" << lm_ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        lm_ggml_vk_print_tensor_area(tensor, tensor_data, 5, 5, 0, 0);
        std::cerr << std::endl << "Correct:" << std::endl;
        lm_ggml_vk_print_tensor_area(tensor, comp_result, 5, 5, 0, 0);
        std::cerr << std::endl;
        std::vector<const lm_ggml_tensor *> done;
        lm_ggml_vk_print_graph_origin(tensor, done);
    }

    if (avg_err > 0.01 || std::isnan(avg_err)) {
        std::cerr << "ERROR: avg_err=" << avg_err << " in " << lm_ggml_op_name(tensor->op) << " (check " << check_counter << ")" << std::endl;
        std::cerr << "tensor=" << tensor << " tensor->name=" << tensor->name << " tensor->type: " << lm_ggml_type_name(tensor->type) << " ne0=" << tensor->ne[0] << " nb0=" << tensor->nb[0] << " ne1=" << tensor->ne[1] << " nb1=" << tensor->nb[1] << " ne2=" << tensor->ne[2] << " nb2=" << tensor->nb[2] << " ne3=" << tensor->ne[3] << " nb3=" << tensor->nb[3] << " offset=" << tensor->view_offs << std::endl;
        if (src0 != nullptr) {
            std::cerr << "src0=" << src0 << " op=" << lm_ggml_op_name(src0->op) << " type=" << lm_ggml_type_name(src0->type) << " ne0=" << src0->ne[0] << " nb0=" << src0->nb[0] << " ne1=" << src0->ne[1] << " nb1=" << src0->nb[1] << " ne2=" << src0->ne[2] << " nb2=" << src0->nb[2] << " ne3=" << src0->ne[3] << " nb3=" << src0->nb[3] << " offset=" << src0->view_offs << std::endl;
        }
        if (src1 != nullptr) {
            std::cerr << "src1=" << src1 << " op=" << lm_ggml_op_name(src1->op) << " type=" << lm_ggml_type_name(src1->type) << " ne0=" << src1->ne[0] << " nb0=" << src1->nb[0] << " ne1=" << src1->ne[1] << " nb1=" << src1->nb[1] << " ne2=" << src1->ne[2] << " nb2=" << src1->nb[2] << " ne3=" << src1->ne[3] << " nb3=" << src1->nb[3] << " offset=" << src1->view_offs << std::endl;
        }
        if (src2 != nullptr) {
            std::cerr << "src2=" << src2 << " op=" << lm_ggml_op_name(src2->op) << " type=" << lm_ggml_type_name(src2->type) << " ne0=" << src2->ne[0] << " nb0=" << src2->nb[0] << " ne1=" << src2->ne[1] << " nb1=" << src2->nb[1] << " ne2=" << src2->ne[2] << " nb2=" << src2->nb[2] << " ne3=" << src2->ne[3] << " nb3=" << src2->nb[3] << " offset=" << src2->view_offs << std::endl;
        }
        if (src3 != nullptr) {
            std::cerr << "src3=" << src3 << " op=" << lm_ggml_op_name(src3->op) << " type=" << lm_ggml_type_name(src3->type) << " ne0=" << src3->ne[0] << " nb0=" << src3->nb[0] << " ne1=" << src3->ne[1] << " nb1=" << src3->nb[1] << " ne2=" << src3->ne[2] << " nb2=" << src3->nb[2] << " ne3=" << src3->ne[3] << " nb3=" << src3->nb[3] << " offset=" << src3->view_offs << std::endl;
        }
        std::cerr << "First error: result=" << first_error_result << " correct=" << first_error_correct  << " i3=" << first_error[3] << " i2=" << first_error[2] << " i1=" << first_error[1] << " i0=" << first_error[0] << std::endl;
        std::cerr << std::endl << "Result:" << std::endl;
        lm_ggml_vk_print_tensor_area(tensor, tensor_data, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl << "Correct:" << std::endl;
        lm_ggml_vk_print_tensor_area(tensor, comp_result, first_error[0], first_error[1], first_error[2], first_error[3]);
        std::cerr << std::endl;
        std::vector<const lm_ggml_tensor *> done;
        lm_ggml_vk_print_graph_origin(tensor, done);
        LM_GGML_ABORT("fatal error");
    } else {
        std::cerr << check_counter << " " << tensor->name << " op=" << lm_ggml_op_name(tensor->op) << " avg_err=" << avg_err << std::endl;
    }

    free(comp_result);
    comp_result = nullptr;
    comp_size = 0;

    if (lm_ggml_backend_buffer_is_vk(tensor->buffer)) {
        free(tensor_data);
    }

    VK_LOG_DEBUG("END lm_ggml_vk_check_results_1(" << tensor->name << ")");
}
#endif

