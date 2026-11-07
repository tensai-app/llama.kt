#include "models.h"

lm_ggml_cgraph * clip_graph_lfm3vl::build() {
    LM_GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    LM_GGML_ASSERT(model.lfm3vl_embedding_norm_w != nullptr);
    LM_GGML_ASSERT(model.class_embedding == nullptr);
    LM_GGML_ASSERT(n_batch == 1 || n_batch == 2);

    const int batch_size       = 1;
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    const norm_type norm_t = NORM_TYPE_RMS;

    // 2D rope: pairs (j, j+n_dims); first d_head/4 pairs rotate by y, next d_head/4 by x
    int mrope_sections[4] = {d_head/4, d_head/4, 0, 0};

    // one patchifier kernel per temporal slice: a still image fills both slices, video frames are merged in pairs
    lm_ggml_tensor * inp;
    if (model.patch_embeddings_1 == nullptr) {
        LM_GGML_ASSERT(n_batch == 1);
        inp = lm_ggml_conv_2d(ctx0, model.patch_embeddings_0, build_inp_raw(), patch_size, patch_size, 0, 0, 1, 1);
    } else {
        inp = build_inp_temporal_merge();
    }
    inp = lm_ggml_reshape_2d(ctx0, inp, n_pos, n_embd);
    inp = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, inp));
    cb(inp, "patchifier_out", -1);

    // the 2x2 interleave waits for the projector, the conv layers need raster-scan order
    inp = build_norm(inp, model.lfm3vl_embedding_norm_w, nullptr, norm_t, eps, -1);
    cb(inp, "embedding_norm", -1);

    lm_ggml_tensor * inpL = inp;

    // position IDs for 2D RoPE (same format as Qwen3VL)
    lm_ggml_tensor * positions = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, num_position_ids);
    lm_ggml_set_name(positions, "positions");
    lm_ggml_set_input(positions);

    const float rope_theta = hparams.rope_theta;

    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];

        const bool is_conv_layer = (layer.conv2d_in_proj_w != nullptr);

        lm_ggml_tensor * cur = inpL;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        if (is_conv_layer) {
            lm_ggml_tensor * proj = build_mm(layer.conv2d_in_proj_w, cur);
            cb(proj, "conv_in_proj", il);

            lm_ggml_tensor * first  = lm_ggml_view_2d(ctx0, proj, n_embd, n_pos,
                proj->nb[1], 0);
            lm_ggml_tensor * middle = lm_ggml_view_2d(ctx0, proj, n_embd, n_pos,
                proj->nb[1], lm_ggml_row_size(proj->type, n_embd));
            lm_ggml_tensor * third  = lm_ggml_view_2d(ctx0, proj, n_embd, n_pos,
                proj->nb[1], lm_ggml_row_size(proj->type, 2 * n_embd));

            lm_ggml_tensor * gated = lm_ggml_mul(ctx0, first, third);
            cb(gated, "conv_gated", il);

            lm_ggml_tensor * conv_out;
            if (layer.conv2d_spatial_w) {
                lm_ggml_tensor * gated_3d = lm_ggml_reshape_4d(ctx0, gated,
                    n_embd, n_patches_x, n_patches_y, batch_size);
                lm_ggml_tensor * gated_2d = lm_ggml_cont(ctx0,
                    lm_ggml_permute(ctx0, gated_3d, 2, 0, 1, 3));
                cb(gated_2d, "conv_gated_2d", il);

                conv_out = lm_ggml_conv_2d_dw_direct(ctx0, layer.conv2d_spatial_w, gated_2d,
                    1, 1, 1, 1, 1, 1);
                cb(conv_out, "conv_spatial", il);

                conv_out = lm_ggml_cont(ctx0,
                    lm_ggml_permute(ctx0, conv_out, 1, 2, 0, 3));
                conv_out = lm_ggml_reshape_2d(ctx0, conv_out, n_embd, n_pos);
            } else {
                conv_out = gated;
            }

            cur = lm_ggml_mul(ctx0, middle, conv_out);
            cb(cur, "conv_mul", il);

            cur = build_mm(layer.conv2d_out_proj_w, cur);
            cb(cur, "conv_out_proj", il);
        } else {
            lm_ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            lm_ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            lm_ggml_tensor * Vcur = build_mm(layer.v_w, cur);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Qcur = lm_ggml_reshape_3d(ctx0, Qcur, d_head, n_head,    n_pos);
            Kcur = lm_ggml_reshape_3d(ctx0, Kcur, d_head, n_head_kv, n_pos);
            Vcur = lm_ggml_reshape_3d(ctx0, Vcur, d_head, n_head_kv, n_pos);

            if (layer.q_norm) {
                Qcur = build_norm(Qcur, layer.q_norm, nullptr, norm_t, eps, il);
            }
            if (layer.k_norm) {
                Kcur = build_norm(Kcur, layer.k_norm, nullptr, norm_t, eps, il);
            }

            Qcur = lm_ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, LM_GGML_ROPE_TYPE_VISION,
                32768, rope_theta, 1, 0, 1, 32, 1);
            Kcur = lm_ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, LM_GGML_ROPE_TYPE_VISION,
                32768, rope_theta, 1, 0, 1, 32, 1);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        cur = lm_ggml_add(ctx0, cur, inpL);
        inpL = cur;
        cb(cur, "ffn_inp", il);

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);
        cb(cur, "ffn_out", il);

        cur = lm_ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // no post-layernorm for LFM3-VL

    lm_ggml_tensor * embeddings = inpL;

    // projector input LayerNorm, eps is projector_layer_norm_eps (not the encoder eps)
    embeddings = build_norm(embeddings, model.mm_input_norm_w, model.mm_input_norm_b, NORM_TYPE_NORMAL, 1e-6f, -1);
    cb(embeddings, "proj_input_norm", -1);

    // 2x2 patch interleave (same as Qwen3VL) so the reshape below merges spatial 2x2 blocks
    {
        embeddings = lm_ggml_cont_4d(
            ctx0, embeddings,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        embeddings = lm_ggml_reshape_4d(
            ctx0, embeddings,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        embeddings = lm_ggml_permute(ctx0, embeddings, 0, 2, 1, 3);
        embeddings = lm_ggml_cont_3d(
            ctx0, embeddings,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }
    cb(embeddings, "proj_interleaved", -1);

    embeddings = lm_ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);

    // projector MLP: linear_1 -> GELU (erf, as HF "gelu") -> linear_2
    embeddings = build_ffn(embeddings,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_1_w, model.mm_1_b,
        ffn_op_type::FFN_GELU_ERF, -1);

    lm_ggml_build_forward_expand(gf, embeddings);

    return gf;
}
