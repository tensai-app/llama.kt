#include "models.h"

lm_ggml_tensor * clip_graph_nemotron_omni::build_inp_with_temporal_merge() {
    if (n_batch == 1 && !img_batch->is_video) {
        return build_inp();
    }

    LM_GGML_ASSERT(n_batch == 1 || n_batch == 2);
    LM_GGML_ASSERT(model.patch_embeddings_1 != nullptr);
    LM_GGML_ASSERT(model.patch_embeddings_1->ne[2] == 6);

    lm_ggml_tensor * inp_raw = build_inp_raw();
    if (n_batch == 1) {
        // pad the last video frame inside the graph
        inp_raw = lm_ggml_concat(ctx0, inp_raw, inp_raw, 2);
    } else {
        inp_raw = lm_ggml_reshape_4d(ctx0, inp_raw, img.nx(), img.ny(), 6, 1);
    }

    lm_ggml_tensor * inp = lm_ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = lm_ggml_reshape_3d(ctx0, inp, n_patches, n_embd, 1);
    return lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, inp));
}

lm_ggml_cgraph * clip_graph_nemotron_omni::build() {
    LM_GGML_ASSERT(model.class_embedding != nullptr);
    LM_GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_registers = model.class_embedding->ne[1];
    const int n_pos = n_patches + n_registers;

    lm_ggml_tensor * inp = build_inp_with_temporal_merge();

    inp = lm_ggml_add(ctx0, inp, resize_position_embeddings(LM_GGML_SCALE_MODE_BILINEAR));
    cb(inp, "inp_pos", -1);

    inp = lm_ggml_concat(ctx0, model.class_embedding, inp, 1);

    lm_ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, nullptr);

    cur = lm_ggml_view_2d(ctx0, cur,
        n_embd, n_patches,
        lm_ggml_row_size(cur->type, n_embd),
        n_registers * lm_ggml_row_size(cur->type, n_embd));

    cur = build_patch_merge_permute(cur, model.hparams.n_merge);

    {
        cur = build_norm(cur, model.mm_0_w, nullptr, NORM_TYPE_RMS, 1e-5, -1);
        cur = build_ffn(cur, model.mm_1_w, nullptr, nullptr, nullptr, model.mm_3_w, nullptr, FFN_RELU_SQR, -1);
    }

    lm_ggml_build_forward_expand(gf, cur);

    return gf;
}
