#include "models.h"

lm_ggml_cgraph * clip_graph_sensenova_u1::build() {
    LM_GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    LM_GGML_ASSERT(model.patch_bias != nullptr);
    LM_GGML_ASSERT(model.mm_0_w != nullptr);
    LM_GGML_ASSERT(model.mm_0_b != nullptr);
    LM_GGML_ASSERT(hparams.n_merge == 2);

    lm_ggml_tensor * pos_x = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_patches);
    lm_ggml_set_name(pos_x, "pos_x");
    lm_ggml_set_input(pos_x);

    lm_ggml_tensor * pos_y = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_patches);
    lm_ggml_set_name(pos_y, "pos_y");
    lm_ggml_set_input(pos_y);

    lm_ggml_tensor * inp = build_inp();
    inp = lm_ggml_gelu(ctx0, inp);

    // Apply adjacent-pair 2D RoPE to patch embeddings before downsampling.
    inp = lm_ggml_reshape_4d(ctx0, inp, n_embd, 1, n_patches, n_batch);
    inp = build_rope_2d(ctx0, inp, pos_x, pos_y, hparams.rope_theta, false);
    inp = lm_ggml_reshape_3d(ctx0, inp, n_embd, n_patches, n_batch);
    cb(inp, "patch_embd_rope", -1);

    inp = lm_ggml_reshape_4d(ctx0, inp, n_embd, n_patches_x, n_patches_y, n_batch);
    inp = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, inp, 2, 0, 1, 3));
    inp = lm_ggml_conv_2d(ctx0, model.mm_0_w, inp,
            hparams.n_merge, hparams.n_merge, 0, 0, 1, 1);

    const int n_output_patches = n_patches / (hparams.n_merge * hparams.n_merge);
    inp = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, inp, 1, 2, 0, 3));
    inp = lm_ggml_reshape_3d(ctx0, inp, n_mmproj_embd, n_output_patches, n_batch);
    inp = lm_ggml_add(ctx0, inp, model.mm_0_b);
    cb(inp, "proj_out", -1);

    lm_ggml_build_forward_expand(gf, inp);
    return gf;
}
