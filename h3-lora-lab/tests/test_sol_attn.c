#include "h3_dit.h"
#include "h3_gpu.h"
#include "h3_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *msg) {
    fprintf(stderr, "\n[FAIL] %s\n", msg);
    exit(1);
}

static void require(int cond, const char *msg) {
    if (!cond) fail(msg);
}

static uint16_t float_to_bf16(float val) {
    union { float f; uint32_t u; } pun;
    pun.f = val;
    return (uint16_t)(pun.u >> 16);
}

static float bf16_to_float(uint16_t val) {
    union { float f; uint32_t u; } pun;
    pun.u = ((uint32_t)val) << 16;
    return pun.f;
}

static void test_3d_latent_upsampling(h3_gpu *gpu) {
    printf("[TEST] 3D Latent Upsampling (CPU vs GPU)... ");
    
    int channels = 24;
    int in_t = 4, in_h = 8, in_w = 8;
    int out_t = 8, out_h = 16, out_w = 16;
    
    size_t in_size = (size_t)channels * in_t * in_h * in_w;
    size_t out_size = (size_t)channels * out_t * out_h * out_w;
    
    float *in_host = malloc(in_size * sizeof(float));
    float *out_cpu = malloc(out_size * sizeof(float));
    float *out_gpu = malloc(out_size * sizeof(float));
    
    require(in_host && out_cpu && out_gpu, "alloc memory");
    
    for (size_t i = 0; i < in_size; i++) {
        in_host[i] = sinf((float)i * 0.05f) * 2.0f;
    }
    
    // CPU 3D upsampling
    int cpu_ok = h3_latent_upsample_3d(in_host, out_cpu, channels, in_t, in_h, in_w, out_t, out_h, out_w);
    require(cpu_ok, "CPU upsample ok");
    
    // GPU 3D upsampling
    h3_gpu_tensor *in_gpu_t = h3_gpu_tensor_from_f32(gpu, in_host, in_size);
    h3_gpu_tensor *out_gpu_t = h3_gpu_tensor_new_f32(gpu, out_size);
    require(in_gpu_t && out_gpu_t, "GPU tensor alloc");
    
    require(h3_gpu_begin(gpu), "GPU begin");
    int gpu_ok = h3_gpu_latent_upsample_3d_f32(gpu, out_gpu_t, in_gpu_t, channels, in_t, in_h, in_w, out_t, out_h, out_w);
    require(gpu_ok, "GPU upsample ok");
    require(h3_gpu_submit(gpu), "GPU submit");
    
    require(h3_gpu_tensor_read_f32(out_gpu_t, out_gpu, out_size), "read out");
    
    // Compare CPU vs GPU
    float max_diff = 0.0f;
    for (size_t i = 0; i < out_size; i++) {
        require(!isnan(out_gpu[i]) && !isinf(out_gpu[i]), "no NaN/Inf in GPU upsample");
        float diff = fabsf(out_cpu[i] - out_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    
    require(max_diff < 1e-4f, "CPU and GPU 3D upsampling match within 1e-4");
    
    h3_gpu_tensor_free(in_gpu_t);
    h3_gpu_tensor_free(out_gpu_t);
    free(in_host);
    free(out_cpu);
    free(out_gpu);
    
    printf("PASSED (max_diff: %e)\n", max_diff);
}

static void test_sol_attn_metal(h3_gpu *gpu) {
    uint32_t sequences[] = {64, 256, 1024};
    uint32_t heads = 4;
    uint32_t head_dim = 64;

    for (size_t s_idx = 0; s_idx < sizeof(sequences)/sizeof(sequences[0]); s_idx++) {
        uint32_t sequence = sequences[s_idx];
        printf("[TEST] Sol-Attn Metal (Seq=%u, Heads=%u, Dim=%u)... ", sequence, heads, head_dim);

        size_t count = (size_t)heads * sequence * head_dim;
        uint16_t *q_host = malloc(count * sizeof(uint16_t));
        uint16_t *k_host = malloc(count * sizeof(uint16_t));
        uint16_t *v_host = malloc(count * sizeof(uint16_t));
        uint16_t *out_sdpa_host = malloc(count * sizeof(uint16_t));
        uint16_t *out_sol_host = malloc(count * sizeof(uint16_t));
        require(q_host && k_host && v_host && out_sdpa_host && out_sol_host, "alloc memory");

        for (size_t i = 0; i < count; i++) {
            q_host[i] = float_to_bf16(cosf((float)i * 0.1f) * 0.5f);
            k_host[i] = float_to_bf16(sinf((float)i * 0.1f) * 0.5f);
            v_host[i] = float_to_bf16((float)(i % 17) * 0.1f);
        }

        h3_gpu_tensor *q_t = h3_gpu_tensor_from_bf16(gpu, q_host, count);
        h3_gpu_tensor *k_t = h3_gpu_tensor_from_bf16(gpu, k_host, count);
        h3_gpu_tensor *v_t = h3_gpu_tensor_from_bf16(gpu, v_host, count);
        h3_gpu_tensor *out_sdpa_t = h3_gpu_tensor_new_bf16(gpu, count);
        h3_gpu_tensor *out_sol_t = h3_gpu_tensor_new_bf16(gpu, count);
        require(q_t && k_t && v_t && out_sdpa_t && out_sol_t, "tensor alloc");

        float scale = 1.0f / sqrtf((float)head_dim);

        // 1. Run standard SDPA head major
        require(h3_gpu_begin(gpu), "GPU begin SDPA");
        require(h3_gpu_sdpa_bf16_head_major_output(gpu, out_sdpa_t, q_t, k_t, v_t, sequence, heads, head_dim, scale), "run SDPA");
        require(h3_gpu_submit(gpu), "GPU submit SDPA");

        // 2. Run Sol-Attn with block_size=32 and threshold=10.0
        require(h3_gpu_begin(gpu), "GPU begin Sol-Attn");
        require(h3_gpu_sol_attn_bf16(gpu, out_sol_t, q_t, k_t, v_t, sequence, heads, head_dim, scale, 10.0f, 32, 1), "run Sol-Attn");
        require(h3_gpu_submit(gpu), "GPU submit Sol-Attn");

        require(h3_gpu_tensor_read_bf16(out_sdpa_t, out_sdpa_host, count), "read out sdpa");
        require(h3_gpu_tensor_read_bf16(out_sol_t, out_sol_host, count), "read out sol");

        // Verify Sol-Attn outputs
        double total_diff = 0.0;
        double max_diff = 0.0;
        for (size_t i = 0; i < count; i++) {
            float f_sdpa = bf16_to_float(out_sdpa_host[i]);
            float f_sol = bf16_to_float(out_sol_host[i]);
            require(!isnan(f_sol) && !isinf(f_sol), "no NaN/Inf in Sol-Attn");
            double diff = fabs(f_sdpa - f_sol);
            total_diff += diff;
            if (diff > max_diff) max_diff = diff;
        }

        double mean_diff = total_diff / (double)count;
        printf("PASSED (mean_diff: %e, max_diff: %e)\n", mean_diff, max_diff);

        h3_gpu_tensor_free(q_t);
        h3_gpu_tensor_free(k_t);
        h3_gpu_tensor_free(v_t);
        h3_gpu_tensor_free(out_sdpa_t);
        h3_gpu_tensor_free(out_sol_t);
        free(q_host);
        free(k_host);
        free(v_host);
        free(out_sdpa_host);
        free(out_sol_host);
    }
}

int main(void) {
    printf("=== Running Sol-Engine & Sol-Attn Metal Tests ===\n");
    char error[256] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "Failed to initialize Metal GPU: %s\n", error);
        return 1;
    }
    
    test_3d_latent_upsampling(gpu);
    test_sol_attn_metal(gpu);
    
    h3_gpu_free(gpu);
    printf("=== All Sol-Engine tests passed successfully! ===\n");
    return 0;
}
