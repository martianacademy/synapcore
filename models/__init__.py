"""
SynapCore Model Compatibility Adapters & Modular Attention Package (`models/`)
==============================================================================
Provides standardized model architecture adapters (`GLM53MoEDsaAdapter`, `OLMoEAdapter`,
`MixtralAdapter`, `QwenMoEAdapter`, `Qwen38HybridAdapter`, `DenseAdapter`) and the standalone
`ModularAttentionEngine` supporting MHA, GQA, MQA, optional QK-Norm, and RoPE scaling
with strict typing (`Zero Any`).
"""

from models.adapters import (
    AttentionBenchmarkResult,
    BaseModelAdapter,
    DenseAdapter,
    GLM53Adapter,
    GLM53MoEDsaAdapter,
    GLM5MoEAdapter,
    MixtralAdapter,
    ModelAdapterProtocol,
    ModelArchitectureSpec,
    ModularAttentionEngine,
    OLMoEAdapter,
    Qwen38HybridAdapter,
    QwenMoEAdapter,
    WeightNameMap,
    benchmark_attention_subsystem,
    create_model_adapter,
    dequantize_fp8_e4m3_blockwise_numpy,
    fp8_e4m3_bytes_to_fp32,
)

__all__ = [
    "AttentionBenchmarkResult",
    "BaseModelAdapter",
    "DenseAdapter",
    "GLM53Adapter",
    "GLM53MoEDsaAdapter",
    "GLM5MoEAdapter",
    "MixtralAdapter",
    "ModelAdapterProtocol",
    "ModelArchitectureSpec",
    "ModularAttentionEngine",
    "OLMoEAdapter",
    "Qwen38HybridAdapter",
    "QwenMoEAdapter",
    "WeightNameMap",
    "benchmark_attention_subsystem",
    "create_model_adapter",
    "dequantize_fp8_e4m3_blockwise_numpy",
    "fp8_e4m3_bytes_to_fp32",
]
