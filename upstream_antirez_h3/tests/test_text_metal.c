#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SEQUENCE = 32, VOCAB = 128, HIDDEN = 256, QUERY_HEADS = 8,
    KV_HEADS = 2, HEAD_DIM = 32, QUERY_DIM = QUERY_HEADS * HEAD_DIM,
    KV_DIM = KV_HEADS * HEAD_DIM, FFN = 512, ROPE_HALF = HEAD_DIM / 2,
    MAX_TENSORS = 40
};

typedef struct {
    h3_st_header fixture;
    h3_gpu *gpu;
    h3_gpu_tensor *owned[MAX_TENSORS];
    size_t owned_count;
} test_context;

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_text_metal.c: %s\n", message);
    exit(1);
}

static void require(int condition, const char *message) {
    if (!condition) die(message);
}

static void require_gpu(test_context *test, int condition,
                        const char *operation) {
    if (condition) return;
    fprintf(stderr, "FAIL tests/test_text_metal.c: %s: %s\n", operation,
            h3_gpu_error(test->gpu));
    exit(1);
}

static h3_gpu_tensor *own(test_context *test, h3_gpu_tensor *tensor) {
    require(tensor != NULL, "Metal tensor allocation failed");
    require(test->owned_count < MAX_TENSORS, "tensor registry is full");
    test->owned[test->owned_count++] = tensor;
    return tensor;
}

static void *load(test_context *test, const char *name, h3_dtype dtype,
                  size_t elements) {
    const h3_st_tensor *tensor = h3_st_find(&test->fixture, name);
    if (!tensor || tensor->dtype != dtype ||
        h3_st_tensor_elements(tensor) != (uint64_t)elements) {
        fprintf(stderr, "FAIL tests/test_text_metal.c: invalid or absent %s\n",
                name);
        exit(1);
    }
    size_t bytes = elements * h3_dtype_size(dtype);
    void *data = malloc(bytes ? bytes : 1);
    require(data != NULL, "host tensor allocation failed");
    char error[256];
    if (!h3_st_read_data(&test->fixture, tensor, data, bytes, error,
                         sizeof(error))) {
        fprintf(stderr, "FAIL tests/test_text_metal.c: %s\n", error);
        exit(1);
    }
    return data;
}

static h3_gpu_tensor *upload_bf16(test_context *test, const char *name,
                                   size_t elements) {
    const h3_st_tensor *source = h3_st_find(&test->fixture, name);
    require(source != NULL && source->dtype == H3_DTYPE_BF16 &&
            h3_st_tensor_elements(source) == (uint64_t)elements,
            "invalid direct-load BF16 tensor");
    return own(test, h3_gpu_tensor_load_bf16(test->gpu, test->fixture.path,
                                              source->file_offset, elements));
}

static h3_gpu_tensor *upload_f32(test_context *test, const char *name,
                                  size_t elements) {
    float *values = load(test, name, H3_DTYPE_F32, elements);
    h3_gpu_tensor *tensor = own(test, h3_gpu_tensor_from_f32(
                                         test->gpu, values, elements));
    free(values);
    return tensor;
}

static h3_gpu_tensor *fresh(test_context *test, size_t elements) {
    return own(test, h3_gpu_tensor_new_bf16(test->gpu, elements));
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static double compare(test_context *test, const char *name,
                      h3_gpu_tensor *got_tensor, size_t elements,
                      double *maximum_absolute) {
    uint16_t *got = malloc(elements * sizeof(*got));
    uint16_t *want = load(test, name, H3_DTYPE_BF16, elements);
    require(got != NULL, "comparison allocation failed");
    require(h3_gpu_tensor_read_bf16(got_tensor, got, elements),
            "cannot read completed BF16 tensor");
    double error = 0.0;
    double scale = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double got_value = bf16_to_f32(got[index]);
        double want_value = bf16_to_f32(want[index]);
        double delta = fabs(got_value - want_value);
        if (delta > error) error = delta;
        if (fabs(want_value) > scale) scale = fabs(want_value);
    }
    free(got);
    free(want);
    *maximum_absolute = error;
    return error / (scale > 1e-12 ? scale : 1e-12);
}

static void cleanup(test_context *test) {
    for (size_t index = 0; index < test->owned_count; index++) {
        h3_gpu_tensor_free(test->owned[index]);
    }
    h3_gpu_free(test->gpu);
    h3_st_free_header(&test->fixture);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] :
                                  "misc/fixtures/h3_text_bf16.safetensors";
    test_context test;
    memset(&test, 0, sizeof(test));
    char error[512];
    if (!h3_st_read_header(path, &test.fixture, error, sizeof(error))) {
        fprintf(stderr, "FAIL tests/test_text_metal.c: %s\n", error);
        return 1;
    }
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) {
        fprintf(stderr, "FAIL tests/test_text_metal.c: %s\n", error);
        h3_st_free_header(&test.fixture);
        return 1;
    }

    int32_t *signed_ids = load(&test, "x.ids", H3_DTYPE_I32, SEQUENCE);
    uint32_t ids[SEQUENCE];
    for (size_t index = 0; index < SEQUENCE; index++) {
        require(signed_ids[index] >= 0, "negative fixture token ID");
        ids[index] = (uint32_t)signed_ids[index];
    }
    free(signed_ids);
    h3_gpu_tensor *token_ids = own(&test, h3_gpu_tensor_from_u32(
                                             test.gpu, ids, SEQUENCE));

#define WEIGHT(suffix, elements) \
    upload_bf16(&test, "model.layers.0." suffix, (elements))
    h3_gpu_tensor *embedding_weight = upload_bf16(
        &test, "model.embed_tokens.weight", VOCAB * HIDDEN);
    h3_gpu_tensor *input_norm_weight = WEIGHT("input_layernorm.weight", HIDDEN);
    h3_gpu_tensor *q_weight = WEIGHT("self_attn.q_proj.weight",
                                      QUERY_DIM * HIDDEN);
    h3_gpu_tensor *k_weight = WEIGHT("self_attn.k_proj.weight", KV_DIM * HIDDEN);
    h3_gpu_tensor *v_weight = WEIGHT("self_attn.v_proj.weight", KV_DIM * HIDDEN);
    h3_gpu_tensor *q_norm_weight = WEIGHT("self_attn.q_norm.weight", HEAD_DIM);
    h3_gpu_tensor *k_norm_weight = WEIGHT("self_attn.k_norm.weight", HEAD_DIM);
    h3_gpu_tensor *o_weight = WEIGHT("self_attn.o_proj.weight",
                                      HIDDEN * QUERY_DIM);
    h3_gpu_tensor *post_norm_weight = WEIGHT(
        "post_attention_layernorm.weight", HIDDEN);
    h3_gpu_tensor *gate_weight = WEIGHT("mlp.gate_proj.weight", FFN * HIDDEN);
    h3_gpu_tensor *up_weight = WEIGHT("mlp.up_proj.weight", FFN * HIDDEN);
    h3_gpu_tensor *down_weight = WEIGHT("mlp.down_proj.weight", HIDDEN * FFN);
#undef WEIGHT
    h3_gpu_tensor *rope_cos = upload_f32(&test, "x.rope_cos",
                                          SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *rope_sin = upload_f32(&test, "x.rope_sin",
                                          SEQUENCE * ROPE_HALF);

    h3_gpu_tensor *embedding = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *input_norm = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *query = fresh(&test, SEQUENCE * QUERY_DIM);
    h3_gpu_tensor *key = fresh(&test, SEQUENCE * KV_DIM);
    h3_gpu_tensor *value = fresh(&test, SEQUENCE * KV_DIM);
    h3_gpu_tensor *attention_heads = fresh(&test, SEQUENCE * QUERY_DIM);
    h3_gpu_tensor *attention_output = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *after_attention = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *post_norm = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *gate = fresh(&test, SEQUENCE * FFN);
    h3_gpu_tensor *up = fresh(&test, SEQUENCE * FFN);
    h3_gpu_tensor *activated = fresh(&test, SEQUENCE * FFN);
    h3_gpu_tensor *mlp_output = fresh(&test, SEQUENCE * HIDDEN);
    h3_gpu_tensor *output = fresh(&test, SEQUENCE * HIDDEN);

    require_gpu(&test, h3_gpu_begin(test.gpu), "begin command stream");
    require_gpu(&test, h3_gpu_embedding_bf16(test.gpu, embedding,
                                              embedding_weight, token_ids,
                                              SEQUENCE, VOCAB, HIDDEN),
                "token embedding");
    require_gpu(&test, h3_gpu_rms_norm_bf16(test.gpu, input_norm, embedding,
                                             input_norm_weight, SEQUENCE,
                                             HIDDEN, 1e-6f), "input RMSNorm");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, query, input_norm,
                                           q_weight, NULL, SEQUENCE, HIDDEN,
                                           QUERY_DIM), "query projection");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, key, input_norm,
                                           k_weight, NULL, SEQUENCE, HIDDEN,
                                           KV_DIM), "key projection");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, value, input_norm,
                                           v_weight, NULL, SEQUENCE, HIDDEN,
                                           KV_DIM), "value projection");
    require_gpu(&test, h3_gpu_head_rms_norm_bf16(test.gpu, query,
                                                  q_norm_weight, SEQUENCE,
                                                  QUERY_HEADS, HEAD_DIM, 1e-6f),
                "query head RMSNorm");
    require_gpu(&test, h3_gpu_head_rms_norm_bf16(test.gpu, key,
                                                  k_norm_weight, SEQUENCE,
                                                  KV_HEADS, HEAD_DIM, 1e-6f),
                "key head RMSNorm");
    require_gpu(&test, h3_gpu_rope_text_bf16(test.gpu, query, key, rope_cos,
                                              rope_sin, SEQUENCE, QUERY_HEADS,
                                              KV_HEADS, HEAD_DIM), "text RoPE");
    require_gpu(&test, h3_gpu_gqa_causal_bf16(
                           test.gpu, attention_heads, query, key, value,
                           SEQUENCE, QUERY_HEADS, KV_HEADS, HEAD_DIM,
                           1.0f / sqrtf((float)HEAD_DIM)), "causal GQA");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, attention_output,
                                           attention_heads, o_weight, NULL,
                                           SEQUENCE, QUERY_DIM, HIDDEN),
                "attention output projection");
    require_gpu(&test, h3_gpu_add_bf16(test.gpu, after_attention, embedding,
                                        attention_output, SEQUENCE * HIDDEN),
                "attention residual");
    require_gpu(&test, h3_gpu_rms_norm_bf16(test.gpu, post_norm,
                                             after_attention, post_norm_weight,
                                             SEQUENCE, HIDDEN, 1e-6f),
                "post-attention RMSNorm");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, gate, post_norm,
                                           gate_weight, NULL, SEQUENCE, HIDDEN,
                                           FFN), "MLP gate projection");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, up, post_norm,
                                           up_weight, NULL, SEQUENCE, HIDDEN,
                                           FFN), "MLP up projection");
    require_gpu(&test, h3_gpu_silu_mul_bf16(test.gpu, activated, gate, up,
                                             SEQUENCE * FFN), "fused SwiGLU");
    require_gpu(&test, h3_gpu_linear_bf16(test.gpu, mlp_output, activated,
                                           down_weight, NULL, SEQUENCE, FFN,
                                           HIDDEN), "MLP down projection");
    require_gpu(&test, h3_gpu_add_bf16(test.gpu, output, after_attention,
                                        mlp_output, SEQUENCE * HIDDEN),
                "MLP residual");
    require_gpu(&test, h3_gpu_submit(test.gpu), "submit command stream");

    double rope_abs, heads_abs, attn_abs, activated_abs, output_abs;
    double rope_rel = compare(&test, "x.query_rope", query,
                              SEQUENCE * QUERY_DIM, &rope_abs);
    double heads_rel = compare(&test, "x.attention_heads", attention_heads,
                               SEQUENCE * QUERY_DIM, &heads_abs);
    double attn_rel = compare(&test, "x.attention_output", attention_output,
                              SEQUENCE * HIDDEN, &attn_abs);
    double activated_rel = compare(&test, "x.activated", activated,
                                   SEQUENCE * FFN, &activated_abs);
    double output_rel = compare(&test, "x.output", output,
                                SEQUENCE * HIDDEN, &output_abs);
    printf("Qwen BF16 Metal/MLX parity: rope %.3g/%.3g, heads %.3g/%.3g, "
           "attention %.3g/%.3g, SwiGLU %.3g/%.3g, layer %.3g/%.3g\n",
           rope_rel, rope_abs, heads_rel, heads_abs, attn_rel, attn_abs,
           activated_rel, activated_abs, output_rel, output_abs);
    require(rope_rel < 1e-2, "Qwen RoPE exceeds MLX error bound");
    require(heads_rel < 1e-2, "causal GQA exceeds MLX error bound");
    require(attn_rel < 1e-2, "attention output exceeds MLX error bound");
    require(activated_rel < 1e-2, "fused SwiGLU exceeds MLX error bound");
    require(output_rel < 1e-2, "Qwen layer exceeds MLX error bound");

    h3_gpu_stats stats;
    require(h3_gpu_get_stats(test.gpu, &stats), "cannot read Metal counters");
    printf("Qwen dispatches: %llu MPS linear, %llu direct; %.2f MiB, "
           "%llu submission\n",
           (unsigned long long)stats.mps_linear_dispatches,
           (unsigned long long)stats.direct_dispatches,
           (double)stats.allocated_bytes / (1024.0 * 1024.0),
           (unsigned long long)stats.submissions);
    require(stats.mps_linear_dispatches == 5,
            "wide Qwen linears did not use cached MPSGraph");
    require(stats.submissions == 1,
            "Qwen layer used more than one command submission");
    cleanup(&test);
    puts("ok: Iris-style native Qwen layer matches the MLX BF16 oracle");
    return 0;
}
