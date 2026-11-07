#include "models.h"

#include <cmath>

// LFM3-Audio generation
//
// GEN_CODE: one MaskGIT pass of the depthformer, hidden state + known codes (-1 = masked) -> logits per codebook
//           once all codes are known, only the feedback embedding is built
// GEN_WAV : a window of frames -> polar STFT frames of the liquid-codec decoder, the ISTFT runs on the host

// state carried between GEN_WAV calls: rope offset, short-conv taps and attention window of the codec decoder
std::vector<c2w_state_slot> list_lfm3a_gen_state_slots(const clip_hparams & hparams, const clip_model & model) {
    std::vector<c2w_state_slot> slots;
    if (model.c2w.tfm_layers.empty()) {
        return slots;
    }
    const int64_t prefix = hparams.wav_tfm_swa - 1;
    slots.push_back({"pos", 1, 1});
    for (size_t il = 0; il < model.c2w.tfm_layers.size(); il++) {
        const auto & layer = model.c2w.tfm_layers[il];
        if (layer.conv_dw_w) {
            slots.push_back({"conv_" + std::to_string(il), layer.conv_dw_w->ne[0] - 1, hparams.wav_tfm_n_embd});
        } else {
            slots.push_back({"k_" + std::to_string(il), layer.k_w->ne[1], prefix});
            slots.push_back({"v_" + std::to_string(il), layer.v_w->ne[1], prefix});
        }
    }
    return slots;
}

// pre-norm LFM block, with the LFM3 sandwich norms when the layer has them
lm_ggml_tensor * clip_graph_lfm3a_gen::build_lfm_layer(lm_ggml_tensor * inpL, const clip_layer & layer, float eps, int il,
                                                    const std::function<lm_ggml_tensor *(lm_ggml_tensor *)> & op) {
    lm_ggml_tensor * cur = build_norm(inpL, layer.ln_1_w, nullptr, NORM_TYPE_RMS, eps, il);
    cur = op(cur);
    if (layer.attn_post_norm_w) {
        cur = build_norm(cur, layer.attn_post_norm_w, nullptr, NORM_TYPE_RMS, eps, il);
    }
    inpL = lm_ggml_add(ctx0, cur, inpL);

    cur = build_norm(inpL, layer.ln_2_w, nullptr, NORM_TYPE_RMS, eps, il);
    cur = build_ffn(cur, layer.ff_up_w, nullptr, layer.ff_gate_w, nullptr, layer.ff_down_w, nullptr, FFN_SILU, il);
    if (layer.ff_post_norm_w) {
        cur = build_norm(cur, layer.ff_post_norm_w, nullptr, NORM_TYPE_RMS, eps, il);
    }
    return lm_ggml_add(ctx0, inpL, cur);
}

// GQA with q/k RMSNorm, NEOX rope, an optional cached K/V prefix and the LFM3 per-head sigmoid gate
lm_ggml_tensor * clip_graph_lfm3a_gen::build_lfm_attn(lm_ggml_tensor * cur, const clip_layer & layer, int n_head, int n_head_kv,
                                                   float eps, float rope_theta, int il, lm_ggml_tensor * inp_pos,
                                                   lm_ggml_tensor * kq_mask, lm_ggml_tensor * k_cache, lm_ggml_tensor * v_cache,
                                                   lm_ggml_tensor ** k_all, lm_ggml_tensor ** v_all) {
    const int64_t n_pos   = cur->ne[1];
    const int64_t head_dim = layer.q_norm->ne[0];

    lm_ggml_tensor * Qcur = lm_ggml_reshape_3d(ctx0, build_mm(layer.q_w, cur), head_dim, n_head,    n_pos);
    lm_ggml_tensor * Kcur = lm_ggml_reshape_3d(ctx0, build_mm(layer.k_w, cur), head_dim, n_head_kv, n_pos);
    lm_ggml_tensor * Vcur = build_mm(layer.v_w, cur);

    Qcur = build_norm(Qcur, layer.q_norm, nullptr, NORM_TYPE_RMS, eps, il);
    Kcur = build_norm(Kcur, layer.k_norm, nullptr, NORM_TYPE_RMS, eps, il);
    Qcur = lm_ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, head_dim, LM_GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = lm_ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, head_dim, LM_GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    lm_ggml_tensor * k = lm_ggml_reshape_2d(ctx0, Kcur, head_dim * n_head_kv, n_pos);
    lm_ggml_tensor * v = Vcur;
    if (k_cache) {
        k = lm_ggml_concat(ctx0, k_cache, k, 1);
        v = lm_ggml_concat(ctx0, v_cache, v, 1);
    }
    if (k_all) {
        *k_all = k;
        *v_all = v;
    }
    const int64_t n_kv = k->ne[1];

    lm_ggml_tensor * attn = build_attn(layer.attn_gate_w ? nullptr : layer.o_w, nullptr,
        Qcur,
        lm_ggml_reshape_3d(ctx0, k, head_dim, n_head_kv, n_kv),
        lm_ggml_reshape_3d(ctx0, v, head_dim, n_head_kv, n_kv),
        kq_mask, 1.0f / sqrtf((float) head_dim), il);
    if (layer.attn_gate_w) {
        lm_ggml_tensor * gate = lm_ggml_sigmoid(ctx0, build_mm(layer.attn_gate_w, cur));
        attn = lm_ggml_mul(ctx0, lm_ggml_reshape_3d(ctx0, attn, head_dim, n_head, n_pos), lm_ggml_reshape_3d(ctx0, gate, 1, n_head, n_pos));
        attn = build_mm(layer.o_w, lm_ggml_reshape_2d(ctx0, attn, head_dim * n_head, n_pos));
    }
    return attn;
}

// short conv: in_proj -> b * x -> causal depthwise conv over [conv_state | x] -> c * conv -> out_proj
lm_ggml_tensor * clip_graph_lfm3a_gen::build_lfm_conv(lm_ggml_tensor * cur, const clip_layer & layer, int il,
                                                   lm_ggml_tensor * conv_state, lm_ggml_tensor ** conv_state_out) {
    const int64_t n_embd_c = cur->ne[0];
    const int64_t n_pos    = cur->ne[1];

    lm_ggml_tensor * bcx = build_mm(layer.conv_pw1_w, cur);
    lm_ggml_tensor * b = lm_ggml_view_2d(ctx0, bcx, n_embd_c, n_pos, bcx->nb[1], 0);
    lm_ggml_tensor * c = lm_ggml_view_2d(ctx0, bcx, n_embd_c, n_pos, bcx->nb[1], lm_ggml_row_size(bcx->type, n_embd_c));
    lm_ggml_tensor * x = lm_ggml_view_2d(ctx0, bcx, n_embd_c, n_pos, bcx->nb[1], lm_ggml_row_size(bcx->type, 2 * n_embd_c));

    lm_ggml_tensor * bx = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, lm_ggml_mul(ctx0, b, x)));
    bx = lm_ggml_concat(ctx0, conv_state, bx, 0);
    const int64_t d_conv = conv_state->ne[0];
    *conv_state_out = lm_ggml_cont(ctx0, lm_ggml_view_2d(ctx0, bx, d_conv, n_embd_c, bx->nb[1], (size_t) n_pos * bx->nb[0]));

    cur = lm_ggml_ssm_conv(ctx0, lm_ggml_reshape_3d(ctx0, bx, bx->ne[0], n_embd_c, 1), layer.conv_dw_w);
    cur = lm_ggml_mul(ctx0, c, lm_ggml_reshape_2d(ctx0, cur, n_embd_c, n_pos));
    cb(cur, "conv", il);
    return build_mm(layer.conv_pw2_w, cur);
}

lm_ggml_tensor * clip_graph_lfm3a_gen::build_gen_code() {
    lm_ggml_tensor * embd_w = model.gen_code_embd_w; // [n_embd, n_vocab, n_codebooks], tied to the output head
    const int64_t n_cb    = embd_w->ne[2];
    const int64_t n_vocab = embd_w->ne[1];

    // rows of the merged per-codebook tables: code + codebook * n_vocab
    lm_ggml_tensor * inp_codes = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_cb);
    lm_ggml_set_name(inp_codes, "inp_codes");
    lm_ggml_set_input(inp_codes);

    // feedback embedding for the backbone
    lm_ggml_tensor * fb = lm_ggml_get_rows(ctx0, model.gen_code_out_embd_w, inp_codes); // [n_mmproj_embd, n_cb]
    fb = lm_ggml_sum_rows(ctx0, lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, fb)));
    fb = lm_ggml_reshape_2d(ctx0, fb, n_mmproj_embd, 1);
    cb(fb, "gen_embd", -1);
    if (all_known) {
        return fb;
    }

    lm_ggml_tensor * h_state = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_F32, n_mmproj_embd);
    lm_ggml_set_name(h_state, "inp_raw");
    lm_ggml_set_input(h_state);

    // 1 for a revealed codebook, 0 for a masked one
    lm_ggml_tensor * inp_mask = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, 1, n_cb);
    lm_ggml_set_name(inp_mask, "inp_mask");
    lm_ggml_set_input(inp_mask);

    // one position per codebook: projected hidden state + token embedding or the mask embedding
    lm_ggml_tensor * cur = lm_ggml_add(ctx0, build_mm(model.gen_code_proj_in_w, h_state), model.gen_code_proj_in_b);
    cur = lm_ggml_reshape_2d(ctx0, cur, n_embd, n_cb);

    lm_ggml_tensor * tok = lm_ggml_get_rows(ctx0, lm_ggml_reshape_2d(ctx0, embd_w, n_embd, n_vocab * n_cb), inp_codes);
    tok = lm_ggml_mul(ctx0, tok, inp_mask);
    lm_ggml_tensor * msk = lm_ggml_mul(ctx0, lm_ggml_repeat(ctx0, model.gen_code_mask_embd, cur), lm_ggml_scale_bias(ctx0, inp_mask, -1.0f, 1.0f));
    cur = lm_ggml_add(ctx0, cur, lm_ggml_add(ctx0, tok, msk));
    cb(cur, "depth_inp", -1);

    // bidirectional: no mask, no state
    lm_ggml_tensor * inp_pos = lm_ggml_cast(ctx0, lm_ggml_arange(ctx0, 0.0f, (float) n_cb, 1.0f), LM_GGML_TYPE_I32);
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        cur = build_lfm_layer(cur, layer, eps, il, [&](lm_ggml_tensor * x) {
            return build_lfm_attn(x, layer, hparams.n_head, hparams.n_head_kv, eps, hparams.rope_theta, il, inp_pos,
                                  nullptr, nullptr, nullptr, nullptr, nullptr);
        });
    }

    cur = build_norm(cur, model.gen_code_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);
    cur = build_norm(cur, model.gen_code_head_norm_w, nullptr, NORM_TYPE_RMS, eps, -1); // per codebook
    lm_ggml_tensor * logits = lm_ggml_mul_mat(ctx0, embd_w, lm_ggml_reshape_3d(ctx0, cur, n_embd, 1, n_cb)); // [n_vocab, 1, n_cb]
    logits = lm_ggml_reshape_2d(ctx0, logits, n_vocab, n_cb);
    lm_ggml_set_name(logits, "out_feats");
    lm_ggml_set_output(logits);
    lm_ggml_build_forward_expand(gf, logits);

    return fb;
}

lm_ggml_tensor * clip_graph_lfm3a_gen::build_gen_wav() {
    const auto & c2w = model.c2w;
    const int64_t n_cb   = model.gen_code_embd_w->ne[2];
    const int64_t n_fs   = hparams.wav_frame_stack;
    const int64_t n_pos  = n_fs * n_frames;
    const int64_t prefix = hparams.wav_tfm_swa - 1;

    // frame-major codes
    lm_ggml_tensor * inp_codes = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_cb * n_frames);
    lm_ggml_set_name(inp_codes, "inp_codes");
    lm_ggml_set_input(inp_codes);

    std::map<std::string, lm_ggml_tensor *> state_in;
    std::vector<std::pair<std::string, lm_ggml_tensor *>> state_out;
    for (const auto & slot : list_lfm3a_gen_state_slots(hparams, model)) {
        lm_ggml_tensor * t = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, slot.ne0, slot.ne1);
        lm_ggml_set_name(t, ("state_in_" + slot.name).c_str());
        lm_ggml_set_input(t);
        state_in[slot.name] = t;
    }

    // FSQ decode: every codebook shares the level table, one projection over the n_cb code vectors of a frame
    lm_ggml_tensor * q = lm_ggml_get_rows(ctx0, c2w.quant_first_cb_w, inp_codes); // [n_lvl, n_cb * n_frames]
    q = lm_ggml_reshape_2d(ctx0, q, q->ne[0] * n_cb, n_frames);
    lm_ggml_tensor * cur = lm_ggml_add(ctx0, build_mm(c2w.quant_first_out_w, q), c2w.quant_first_out_b);
    cur = lm_ggml_add(ctx0, build_mm(c2w.tfm_in_proj_w, cur), c2w.tfm_in_proj_b); // [n_embd_dec, n_frames]
    const int64_t n_embd_dec = cur->ne[0];

    // nearest-exact upsampling: every frame becomes n_fs decoder positions
    cur = lm_ggml_repeat_4d(ctx0, lm_ggml_reshape_3d(ctx0, cur, n_embd_dec, 1, n_frames), n_embd_dec, n_fs, n_frames, 1);
    cur = lm_ggml_reshape_2d(ctx0, cur, n_embd_dec, n_pos);

    // positions continue across calls from the "pos" slot, see clip_encode()
    lm_ggml_tensor * inp_pos = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_pos);
    lm_ggml_set_name(inp_pos, "positions");
    lm_ggml_set_input(inp_pos);
    state_out.push_back({"pos", lm_ggml_scale_bias(ctx0, state_in.at("pos"), 1.0f, (float) n_pos)});

    // sliding-window causal mask over [cached window | this chunk]
    lm_ggml_tensor * kq_mask = build_inp_attn_mask(prefix + n_pos, n_pos);

    const float eps_dec = hparams.wav_tfm_eps;
    for (size_t il = 0; il < c2w.tfm_layers.size(); il++) {
        const auto & layer = c2w.tfm_layers[il];
        cur = build_lfm_layer(cur, layer, eps_dec, (int) il, [&](lm_ggml_tensor * x) {
            if (layer.conv_dw_w) {
                const std::string name = "conv_" + std::to_string(il);
                lm_ggml_tensor * st = nullptr;
                lm_ggml_tensor * out = build_lfm_conv(x, layer, (int) il, state_in.at(name), &st);
                state_out.push_back({name, st});
                return out;
            }
            // this chunk's tail of the window is kept for the next call
            const std::string k_name = "k_" + std::to_string(il);
            const std::string v_name = "v_" + std::to_string(il);
            lm_ggml_tensor * k = nullptr;
            lm_ggml_tensor * v = nullptr;
            lm_ggml_tensor * out = build_lfm_attn(x, layer, hparams.wav_tfm_n_head, hparams.wav_tfm_n_head_kv, eps_dec,
                                               hparams.wav_tfm_rope_theta, (int) il, inp_pos, kq_mask,
                                               state_in.at(k_name), state_in.at(v_name), &k, &v);
            state_out.push_back({k_name, lm_ggml_cont(ctx0, lm_ggml_view_2d(ctx0, k, k->ne[0], prefix, k->nb[1], (size_t) n_pos * k->nb[1]))});
            state_out.push_back({v_name, lm_ggml_cont(ctx0, lm_ggml_view_2d(ctx0, v, v->ne[0], prefix, v->nb[1], (size_t) n_pos * v->nb[1]))});
            return out;
        });
    }

    cur = build_norm(cur, c2w.tfm_output_norm_w, nullptr, NORM_TYPE_RMS, eps_dec, -1);
    cur = lm_ggml_add(ctx0, build_mm(c2w.tfm_out_proj_w, cur), c2w.tfm_out_proj_b); // [2 * n_bins, n_pos]
    cb(cur, "polar_out", -1);

    for (const auto & s : state_out) {
        lm_ggml_set_name(s.second, ("state_out_" + s.first).c_str());
        lm_ggml_set_output(s.second);
        lm_ggml_build_forward_expand(gf, s.second);
    }

    lm_ggml_set_name(cur, "out_feats");
    lm_ggml_set_output(cur);
    return cur;
}

lm_ggml_cgraph * clip_graph_lfm3a_gen::build() {
    lm_ggml_tensor * out = gen_process == CLIP_GEN_PROCESS_GEN_WAV ? build_gen_wav() : build_gen_code();
    lm_ggml_build_forward_expand(gf, out);
    return gf;
}
