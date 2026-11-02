// ============================================================================
// SynapCore Unified Hardware Abstraction Layer (`include/synapcore_hal.h`)
// ============================================================================
// Task P3.1: Unified Hardware Abstraction Layer (HAL) for Metal, CUDA & Custom AI ASIC
// Decouples the SynapCore sparse Mixture-of-Experts runtime from the underlying
// compute and memory fabric across three hardware targets:
//   1. `SYNAPCORE_BACKEND_APPLE_METAL`   : Apple Silicon Unified Memory + Metal / Accelerate
//   2. `SYNAPCORE_BACKEND_NVIDIA_CUDA`   : Pinned Host DRAM -> HBM Stream + CUDA / cuBLASLt
//   3. `SYNAPCORE_BACKEND_CUSTOM_ASIC`   : Dedicated On-Chip SRAM Hot-Expert Tiles +
//                                          Hardware Top-K Router + Line-Rate DMA Merkle Engine
// ============================================================================

#ifndef SYNAPCORE_HAL_H
#define SYNAPCORE_HAL_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SynapCoreBackendType {
    SYNAPCORE_BACKEND_APPLE_METAL = 0,
    SYNAPCORE_BACKEND_NVIDIA_CUDA = 1,
    SYNAPCORE_BACKEND_CUSTOM_ASIC = 2
} SynapCoreBackendType;

typedef struct SynapCoreAttentionDesc {
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t hidden_size;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t pos_offset;
    float    rms_eps;
    float    rope_theta;
    uint32_t has_qk_norm;
} SynapCoreAttentionDesc;

typedef struct SynapCoreRouterDesc {
    uint32_t num_tokens;
    uint32_t hidden_size;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t norm_topk_prob;
} SynapCoreRouterDesc;

typedef struct SynapCoreExpertMLPDesc {
    uint32_t num_tokens;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t layer_idx;
    uint32_t expert_idx;
    uint32_t sram_tile_slot;
} SynapCoreExpertMLPDesc;

typedef struct SynapCoreHALTelemetry {
    uint32_t backend_type;
    uint32_t zero_copy_unified_memory;
    uint32_t sram_tile_capacity;
    uint32_t active_sram_tiles;
    uint64_t router_dispatches;
    uint64_t expert_mlp_dispatches;
    uint64_t attention_dispatches;
    uint64_t dma_expert_stages;
    uint64_t dma_bytes_transferred;
    uint64_t hw_merkle_proofs_verified;
    uint64_t bg_prefetch_tier2_verified;
} SynapCoreHALTelemetry;

// C ABI Entrypoints exported by `libsynapcore_native.dylib`
void* synapcore_hal_create(void* native_ctx_handle, int backend_type, uint32_t sram_tile_slots);
void  synapcore_hal_destroy(void* hal_handle);
const char* synapcore_hal_backend_name(void* hal_handle);

int synapcore_hal_dma_stage_expert(
    void*    hal_handle,
    uint32_t sram_slot,
    int      layer_idx,
    int      expert_idx
);

int synapcore_hal_hw_verify_merkle_leaf(
    void*          hal_handle,
    int            layer_idx,
    int            expert_idx,
    const uint8_t* expected_root32,
    int            verify_leaf_bytes
);

int synapcore_hal_dispatch_topk_router(
    void*                      hal_handle,
    const SynapCoreRouterDesc* desc,
    const float*               hidden_in_f32,
    const uint16_t*            router_w_fp16,
    int32_t*                   out_topk_indices,
    float*                     out_topk_scores
);

int synapcore_hal_dispatch_sparse_expert_mlp(
    void*                         hal_handle,
    const SynapCoreExpertMLPDesc* desc,
    const float*                  x_in_f32,
    const uint16_t*               gate_w_fp16,
    const uint16_t*               up_w_fp16,
    const uint16_t*               down_w_fp16,
    float*                        y_out_f32
);

int synapcore_hal_dispatch_qkv_attention(
    void*                         hal_handle,
    const SynapCoreAttentionDesc* desc,
    const float*                  q_in_f32,
    const float*                  k_in_f32,
    const float*                  v_in_f32,
    float*                        attn_out_f32
);

void synapcore_hal_get_telemetry(void* hal_handle, SynapCoreHALTelemetry* out_telemetry);

#ifdef __cplusplus
} // extern "C"

namespace synapcore {
namespace hal {

class SynapCoreHAL {
public:
    virtual ~SynapCoreHAL() = default;
    virtual SynapCoreBackendType backend_type() const = 0;
    virtual const char* backend_name() const = 0;

    // Memory & Integrity Fabric API
    virtual int dma_stage_expert(uint32_t sram_slot, int layer_idx, int expert_idx) = 0;
    virtual int hw_verify_merkle_leaf(
        int layer_idx, int expert_idx, const uint8_t* expected_root32, bool verify_leaf_bytes
    ) = 0;

    // Compute Fabric API
    virtual int dispatch_qkv_attention(
        const SynapCoreAttentionDesc& desc,
        const float*                  q_in_f32,
        const float*                  k_in_f32,
        const float*                  v_in_f32,
        float*                        attn_out_f32
    ) = 0;

    virtual int dispatch_topk_router(
        const SynapCoreRouterDesc& desc,
        const float*               hidden_in_f32,
        const uint16_t*            router_w_fp16,
        int32_t*                   out_topk_indices,
        float*                     out_topk_scores
    ) = 0;

    virtual int dispatch_sparse_expert_mlp(
        const SynapCoreExpertMLPDesc& desc,
        const float*                  x_in_f32,
        const uint16_t*               gate_w_fp16,
        const uint16_t*               up_w_fp16,
        const uint16_t*               down_w_fp16,
        float*                        y_out_f32
    ) = 0;

    virtual void get_telemetry(SynapCoreHALTelemetry* out_telemetry) const = 0;
};

} // namespace hal
} // namespace synapcore
#endif

#endif // SYNAPCORE_HAL_H
