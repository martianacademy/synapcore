"""
SynapCore Model Compatibility Adapters & Modular Attention Package (`models/`)
==============================================================================
Provides standardized model architecture adapters (`OLMoEAdapter`, `MixtralAdapter`,
`QwenMoEAdapter`, `DenseAdapter`) and the standalone `ModularAttentionEngine`
supporting MHA, GQA, MQA, optional QK-Norm, and RoPE scaling with strict typing (`Zero Any`).
"""

from models.adapters import (
    AttentionBenchmarkResult,
    BaseModelAdapter,
    DenseAdapter,
    MixtralAdapter,
    ModelAdapterProtocol,
    ModelArchitectureSpec,
    ModularAttentionEngine,
    OLMoEAdapter,
    QwenMoEAdapter,
    WeightNameMap,
    benchmark_attention_subsystem,
    create_model_adapter,
)

__all__ = [
    "AttentionBenchmarkResult",
    "BaseModelAdapter",
    "DenseAdapter",
    "MixtralAdapter",
    "ModelAdapterProtocol",
    "ModelArchitectureSpec",
    "ModularAttentionEngine",
    "OLMoEAdapter",
    "QwenMoEAdapter",
    "WeightNameMap",
    "benchmark_attention_subsystem",
    "create_model_adapter",
]
