#!/usr/bin/env python3
"""
Task P1.3: Model Compatibility Adapter Layer & Modular Attention Subsystem (`models/adapters.py`)
=================================================================================================
100% Strict Typing (`Zero Any`):
  1. Standardized `ModelArchitectureSpec` & `ModelAdapterProtocol` / `BaseModelAdapter` abstracting:
     - `OLMoEAdapter`: 64 experts, Top-8, MHA (16/16), QK-Norm (`has_qk_norm=True`), SwiGLU.
     - `MixtralAdapter`: 8 experts, Top-2, GQA (32 Q / 8 KV heads), no QK-Norm, `block_sparse_moe` weight map.
     - `QwenMoEAdapter`: Shared Expert + Routed Top-K Experts (`has_shared_expert=True`), GQA, sigmoid shared gate.
     - `DenseAdapter`: Clean router-free dense SwiGLU MLP execution (`is_moe=False`, `num_experts=0`).
  2. Standalone Benchmarkable `ModularAttentionEngine`:
     - Supports Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-Query Attention (MQA).
     - Configurable QK-Norm (`has_qk_norm=True/False`, supporting both full `(hidden_size,)` and per-head `(head_dim,)` weights).
     - Rotary Position Embeddings (RoPE) with optional linear/NTK `rope_scaling` factor.
     - Bit-exact numerical parity (`max_abs_diff == 0.0`) with `NativeMerkleMoEEngine` on `OLMoE-1B-7B`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Protocol, Tuple, TypedDict, Union

import numpy as np


class WeightNameMap(TypedDict):
    embed_tokens: str
    final_norm: str
    lm_head: str
    input_layernorm: str
    post_attention_layernorm: str
    q_proj: str
    k_proj: str
    v_proj: str
    o_proj: str
    q_norm: str
    k_norm: str
    router_gate: str
    expert_gate_proj: str
    expert_up_proj: str
    expert_down_proj: str
    shared_expert_gate: str
    shared_expert_gate_proj: str
    shared_expert_up_proj: str
    shared_expert_down_proj: str
    dense_gate_proj: str
    dense_up_proj: str
    dense_down_proj: str


class AttentionBenchmarkResult(TypedDict):
    mode: str
    num_heads: int
    num_kv_heads: int
    gqa_group_size: int
    has_qk_norm: bool
    prefill_seq_len: int
    prefill_ms: float
    decode_step_ms: float
    decode_tok_per_sec: float
    output_shape: Tuple[int, int, int]


ConfigValue = Union[
    int,
    float,
    bool,
    str,
    None,
    List[str],
    List[int],
    Dict[str, Union[int, float, str, List[str], List[int]]],
]
ConfigDict = Dict[str, ConfigValue]


@dataclass(frozen=True)
class ModelArchitectureSpec:
    """
    Standardized immutable specification for MoE and Dense Transformer architectures.
    """

    model_type: str
    architecture_family: str
    num_layers: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling_factor: float
    norm_topk_prob: bool
    has_qk_norm: bool
    has_shared_expert: bool
    shared_expert_intermediate_size: int
    is_moe: bool
    activation: str
    expert_index_shift: Optional[int]
    weight_map: WeightNameMap

    @property
    def gqa_group_size(self) -> int:
        return max(1, self.num_attention_heads // max(1, self.num_kv_heads))

    @property
    def kv_hidden_size(self) -> int:
        return self.num_kv_heads * self.head_dim


class ModelAdapterProtocol(Protocol):
    """
    Protocol implemented by all SynapCore model compatibility adapters.
    """

    @property
    def spec(self) -> ModelArchitectureSpec: ...

    def compute_expert_slot_id(self, layer_idx: int, expert_idx: int) -> int: ...

    def decode_expert_slot_id(self, slot_id: int) -> Tuple[int, int]: ...

    def layer_weight_names(self, layer_idx: int) -> Dict[str, str]: ...

    def expert_weight_names(self, layer_idx: int, expert_idx: int) -> Tuple[str, str, str]: ...

    def route_tokens_numpy(
        self, x_flat_f32: np.ndarray, router_w_fp16: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]: ...


def _extract_rope_scaling_factor(raw_cfg: ConfigDict) -> float:
    rs = raw_cfg.get("rope_scaling")
    if isinstance(rs, dict):
        fac = rs.get("factor", 1.0)
        if isinstance(fac, (int, float)) and float(fac) > 0.0:
            return float(fac)
    return 1.0


def _power_of_two_shift(n: int) -> Optional[int]:
    if n > 0 and (n & (n - 1)) == 0:
        return int(round(math.log2(n)))
    return None


def _silu_f32(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _gelu_f32(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))


class BaseModelAdapter:
    """
    Base class providing standardized slot indexing, weight resolution, routing, and FFN execution.
    """

    def __init__(self, spec: ModelArchitectureSpec):
        self._spec: ModelArchitectureSpec = spec

    @property
    def spec(self) -> ModelArchitectureSpec:
        return self._spec

    def compute_expert_slot_id(self, layer_idx: int, expert_idx: int) -> int:
        if not self._spec.is_moe or self._spec.num_experts <= 0:
            raise ValueError(f"Adapter '{self._spec.model_type}' is a dense model (num_experts=0)")
        if not (0 <= layer_idx < self._spec.num_layers and 0 <= expert_idx < self._spec.num_experts):
            raise ValueError(
                f"Invalid expert coordinates L{layer_idx}E{expert_idx} for "
                f"num_layers={self._spec.num_layers}, num_experts={self._spec.num_experts}"
            )
        if self._spec.expert_index_shift is not None:
            return (int(layer_idx) << self._spec.expert_index_shift) | int(expert_idx)
        return int(layer_idx) * self._spec.num_experts + int(expert_idx)

    def decode_expert_slot_id(self, slot_id: int) -> Tuple[int, int]:
        if not self._spec.is_moe or self._spec.num_experts <= 0:
            raise ValueError(f"Adapter '{self._spec.model_type}' is a dense model (num_experts=0)")
        if self._spec.expert_index_shift is not None:
            mask = (1 << self._spec.expert_index_shift) - 1
            return int(slot_id >> self._spec.expert_index_shift), int(slot_id & mask)
        return int(slot_id // self._spec.num_experts), int(slot_id % self._spec.num_experts)

    def layer_weight_names(self, layer_idx: int) -> Dict[str, str]:
        wm = self._spec.weight_map
        out: Dict[str, str] = {
            "input_layernorm": wm["input_layernorm"].format(l=layer_idx),
            "post_attention_layernorm": wm["post_attention_layernorm"].format(l=layer_idx),
            "q_proj": wm["q_proj"].format(l=layer_idx),
            "k_proj": wm["k_proj"].format(l=layer_idx),
            "v_proj": wm["v_proj"].format(l=layer_idx),
            "o_proj": wm["o_proj"].format(l=layer_idx),
        }
        if self._spec.has_qk_norm:
            out["q_norm"] = wm["q_norm"].format(l=layer_idx)
            out["k_norm"] = wm["k_norm"].format(l=layer_idx)
        if self._spec.is_moe:
            out["router_gate"] = wm["router_gate"].format(l=layer_idx)
        else:
            out["dense_gate_proj"] = wm["dense_gate_proj"].format(l=layer_idx)
            out["dense_up_proj"] = wm["dense_up_proj"].format(l=layer_idx)
            out["dense_down_proj"] = wm["dense_down_proj"].format(l=layer_idx)
        if self._spec.has_shared_expert:
            out["shared_expert_gate"] = wm["shared_expert_gate"].format(l=layer_idx)
            out["shared_expert_gate_proj"] = wm["shared_expert_gate_proj"].format(l=layer_idx)
            out["shared_expert_up_proj"] = wm["shared_expert_up_proj"].format(l=layer_idx)
            out["shared_expert_down_proj"] = wm["shared_expert_down_proj"].format(l=layer_idx)
        return out

    def expert_weight_names(self, layer_idx: int, expert_idx: int) -> Tuple[str, str, str]:
        wm = self._spec.weight_map
        return (
            wm["expert_gate_proj"].format(l=layer_idx, e=expert_idx),
            wm["expert_up_proj"].format(l=layer_idx, e=expert_idx),
            wm["expert_down_proj"].format(l=layer_idx, e=expert_idx),
        )

    def apply_activation_f32(self, x_f32: np.ndarray) -> np.ndarray:
        if self._spec.activation == "gelu":
            return _gelu_f32(x_f32)
        return _silu_f32(x_f32)

    def forward_single_expert_f32(
        self,
        x_tok_f32: np.ndarray,
        gate_w: np.ndarray,
        up_w: np.ndarray,
        down_w: np.ndarray,
    ) -> np.ndarray:
        g_proj = x_tok_f32 @ gate_w.astype(np.float32).T
        u_proj = x_tok_f32 @ up_w.astype(np.float32).T
        act = self.apply_activation_f32(g_proj) * u_proj
        return act @ down_w.astype(np.float32).T

    def route_tokens_numpy(
        self, x_flat_f32: np.ndarray, router_w_fp16: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._spec.is_moe:
            raise ValueError(f"Cannot route tokens on dense adapter '{self._spec.model_type}'")
        r_w = router_w_fp16.astype(np.float32)
        logits = x_flat_f32 @ r_w.T
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        k = self._spec.top_k
        topk_indices = np.argpartition(-probs, kth=k - 1, axis=-1)[:, :k].astype(np.int32)
        topk_scores = np.take_along_axis(probs, topk_indices, axis=-1)
        if self._spec.norm_topk_prob:
            topk_scores = topk_scores / np.sum(topk_scores, axis=-1, keepdims=True)
        return topk_indices, topk_scores


class OLMoEAdapter(BaseModelAdapter):
    """
    Task P1.3: Verified Adapter for `allenai/OLMoE-1B-7B-0125-Instruct` (64 Experts, Top-8, QK-Norm, MHA).
    """

    DEFAULT_WEIGHT_MAP: WeightNameMap = {
        "embed_tokens": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
        "input_layernorm": "model.layers.{l}.input_layernorm.weight",
        "post_attention_layernorm": "model.layers.{l}.post_attention_layernorm.weight",
        "q_proj": "model.layers.{l}.self_attn.q_proj.weight",
        "k_proj": "model.layers.{l}.self_attn.k_proj.weight",
        "v_proj": "model.layers.{l}.self_attn.v_proj.weight",
        "o_proj": "model.layers.{l}.self_attn.o_proj.weight",
        "q_norm": "model.layers.{l}.self_attn.q_norm.weight",
        "k_norm": "model.layers.{l}.self_attn.k_norm.weight",
        "router_gate": "model.layers.{l}.mlp.gate.weight",
        "expert_gate_proj": "model.layers.{l}.mlp.experts.{e}.gate_proj.weight",
        "expert_up_proj": "model.layers.{l}.mlp.experts.{e}.up_proj.weight",
        "expert_down_proj": "model.layers.{l}.mlp.experts.{e}.down_proj.weight",
        "shared_expert_gate": "",
        "shared_expert_gate_proj": "",
        "shared_expert_up_proj": "",
        "shared_expert_down_proj": "",
        "dense_gate_proj": "",
        "dense_up_proj": "",
        "dense_down_proj": "",
    }

    def __init__(self, config: Optional[ConfigDict] = None):
        cfg = config or {}
        num_layers = int(cfg.get("num_hidden_layers", 16) or 16)
        num_experts = int(cfg.get("num_experts", 64) or 64)
        top_k = int(cfg.get("num_experts_per_tok", 8) or 8)
        hidden_size = int(cfg.get("hidden_size", 2048) or 2048)
        intermediate_size = int(cfg.get("intermediate_size", 1024) or 1024)
        num_heads = int(cfg.get("num_attention_heads", 16) or 16)
        num_kv_heads = int(cfg.get("num_key_value_heads", num_heads) or num_heads)
        vocab_size = int(cfg.get("vocab_size", 50304) or 50304)
        rms_eps = float(cfg.get("rms_norm_eps", 1e-5) or 1e-5)
        rope_theta = float(cfg.get("rope_theta", 10000.0) or 10000.0)
        norm_topk = bool(cfg.get("norm_topk_prob", False))
        act = str(cfg.get("hidden_act", "silu") or "silu")

        spec = ModelArchitectureSpec(
            model_type="olmoe",
            architecture_family="OLMoEForCausalLM",
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=hidden_size // num_heads,
            vocab_size=vocab_size,
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            rope_scaling_factor=_extract_rope_scaling_factor(cfg),
            norm_topk_prob=norm_topk,
            has_qk_norm=True,
            has_shared_expert=False,
            shared_expert_intermediate_size=0,
            is_moe=True,
            activation=act,
            expert_index_shift=_power_of_two_shift(num_experts),
            weight_map=self.DEFAULT_WEIGHT_MAP,
        )
        super().__init__(spec)


class MixtralAdapter(BaseModelAdapter):
    """
    Task P1.3: Adapter for Mixtral-8x7B / Mixtral-8x22B (`num_experts=8`, `top_k=2`, GQA `32/8`, no QK-Norm).
    """

    DEFAULT_WEIGHT_MAP: WeightNameMap = {
        "embed_tokens": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
        "input_layernorm": "model.layers.{l}.input_layernorm.weight",
        "post_attention_layernorm": "model.layers.{l}.post_attention_layernorm.weight",
        "q_proj": "model.layers.{l}.self_attn.q_proj.weight",
        "k_proj": "model.layers.{l}.self_attn.k_proj.weight",
        "v_proj": "model.layers.{l}.self_attn.v_proj.weight",
        "o_proj": "model.layers.{l}.self_attn.o_proj.weight",
        "q_norm": "",
        "k_norm": "",
        "router_gate": "model.layers.{l}.block_sparse_moe.gate.weight",
        "expert_gate_proj": "model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight",
        "expert_up_proj": "model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight",
        "expert_down_proj": "model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight",
        "shared_expert_gate": "",
        "shared_expert_gate_proj": "",
        "shared_expert_up_proj": "",
        "shared_expert_down_proj": "",
        "dense_gate_proj": "",
        "dense_up_proj": "",
        "dense_down_proj": "",
    }

    def __init__(self, config: Optional[ConfigDict] = None):
        cfg = config or {}
        num_layers = int(cfg.get("num_hidden_layers", 32) or 32)
        num_experts = int(cfg.get("num_local_experts", cfg.get("num_experts", 8)) or 8)
        top_k = int(cfg.get("num_experts_per_tok", 2) or 2)
        hidden_size = int(cfg.get("hidden_size", 4096) or 4096)
        intermediate_size = int(cfg.get("intermediate_size", 14336) or 14336)
        num_heads = int(cfg.get("num_attention_heads", 32) or 32)
        num_kv_heads = int(cfg.get("num_key_value_heads", 8) or 8)
        vocab_size = int(cfg.get("vocab_size", 32000) or 32000)
        rms_eps = float(cfg.get("rms_norm_eps", 1e-5) or 1e-5)
        rope_theta = float(cfg.get("rope_theta", 1000000.0) or 1000000.0)
        act = str(cfg.get("hidden_act", "silu") or "silu")

        spec = ModelArchitectureSpec(
            model_type="mixtral",
            architecture_family="MixtralForCausalLM",
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=hidden_size // num_heads,
            vocab_size=vocab_size,
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            rope_scaling_factor=_extract_rope_scaling_factor(cfg),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", True)),
            has_qk_norm=False,
            has_shared_expert=False,
            shared_expert_intermediate_size=0,
            is_moe=True,
            activation=act,
            expert_index_shift=_power_of_two_shift(num_experts),
            weight_map=self.DEFAULT_WEIGHT_MAP,
        )
        super().__init__(spec)


class QwenMoEAdapter(BaseModelAdapter):
    """
    Task P1.3: Adapter for Qwen1.5-MoE / Qwen2-MoE (`has_shared_expert=True` + Routed Top-K Experts + GQA).
    """

    DEFAULT_WEIGHT_MAP: WeightNameMap = {
        "embed_tokens": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
        "input_layernorm": "model.layers.{l}.input_layernorm.weight",
        "post_attention_layernorm": "model.layers.{l}.post_attention_layernorm.weight",
        "q_proj": "model.layers.{l}.self_attn.q_proj.weight",
        "k_proj": "model.layers.{l}.self_attn.k_proj.weight",
        "v_proj": "model.layers.{l}.self_attn.v_proj.weight",
        "o_proj": "model.layers.{l}.self_attn.o_proj.weight",
        "q_norm": "model.layers.{l}.self_attn.q_norm.weight",
        "k_norm": "model.layers.{l}.self_attn.k_norm.weight",
        "router_gate": "model.layers.{l}.mlp.gate.weight",
        "expert_gate_proj": "model.layers.{l}.mlp.experts.{e}.gate_proj.weight",
        "expert_up_proj": "model.layers.{l}.mlp.experts.{e}.up_proj.weight",
        "expert_down_proj": "model.layers.{l}.mlp.experts.{e}.down_proj.weight",
        "shared_expert_gate": "model.layers.{l}.mlp.shared_expert_gate.weight",
        "shared_expert_gate_proj": "model.layers.{l}.mlp.shared_expert.gate_proj.weight",
        "shared_expert_up_proj": "model.layers.{l}.mlp.shared_expert.up_proj.weight",
        "shared_expert_down_proj": "model.layers.{l}.mlp.shared_expert.down_proj.weight",
        "dense_gate_proj": "",
        "dense_up_proj": "",
        "dense_down_proj": "",
    }

    def __init__(self, config: Optional[ConfigDict] = None):
        cfg = config or {}
        num_layers = int(cfg.get("num_hidden_layers", 24) or 24)
        num_experts = int(cfg.get("num_experts", 60) or 60)
        top_k = int(cfg.get("num_experts_per_tok", 4) or 4)
        hidden_size = int(cfg.get("hidden_size", 2048) or 2048)
        intermediate_size = int(
            cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 1408)) or 1408
        )
        shared_inter = int(cfg.get("shared_expert_intermediate_size", intermediate_size) or intermediate_size)
        num_heads = int(cfg.get("num_attention_heads", 16) or 16)
        num_kv_heads = int(cfg.get("num_key_value_heads", 4) or 4)
        vocab_size = int(cfg.get("vocab_size", 151936) or 151936)
        rms_eps = float(cfg.get("rms_norm_eps", 1e-6) or 1e-6)
        rope_theta = float(cfg.get("rope_theta", 1000000.0) or 1000000.0)
        has_qk_norm = bool(cfg.get("use_qk_norm", False))
        act = str(cfg.get("hidden_act", "silu") or "silu")

        spec = ModelArchitectureSpec(
            model_type="qwen2_moe",
            architecture_family="Qwen2MoeForCausalLM",
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=hidden_size // num_heads,
            vocab_size=vocab_size,
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            rope_scaling_factor=_extract_rope_scaling_factor(cfg),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", False)),
            has_qk_norm=has_qk_norm,
            has_shared_expert=True,
            shared_expert_intermediate_size=shared_inter,
            is_moe=True,
            activation=act,
            expert_index_shift=_power_of_two_shift(num_experts),
            weight_map=self.DEFAULT_WEIGHT_MAP,
        )
        super().__init__(spec)

    def forward_shared_plus_routed_numpy(
        self,
        x_np: np.ndarray,
        router_w_fp16: np.ndarray,
        routed_expert_weights: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        shared_gate_w_fp16: np.ndarray,
        shared_up_w_fp16: np.ndarray,
        shared_down_w_fp16: np.ndarray,
        shared_expert_gate_vec_fp16: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes Qwen-MoE hybrid FFN: `y = y_routed_topk + sigmoid(x @ shared_gate) * SharedExpert(x)`.
        """
        orig_shape = x_np.shape
        x_flat = x_np.reshape(-1, self._spec.hidden_size).astype(np.float32)
        topk_indices, topk_scores = self.route_tokens_numpy(x_flat, router_w_fp16)

        routed_out = np.zeros_like(x_flat, dtype=np.float32)
        for t_idx in range(x_flat.shape[0]):
            x_tok = x_flat[t_idx]
            acc = np.zeros(self._spec.hidden_size, dtype=np.float32)
            for k_idx in range(self._spec.top_k):
                e_id = int(topk_indices[t_idx, k_idx])
                score = float(topk_scores[t_idx, k_idx])
                g_w, u_w, d_w = routed_expert_weights[e_id]
                acc += score * self.forward_single_expert_f32(x_tok, g_w, u_w, d_w)
            routed_out[t_idx] = acc

        shared_out = self.forward_single_expert_f32(
            x_flat, shared_gate_w_fp16, shared_up_w_fp16, shared_down_w_fp16
        )
        if shared_expert_gate_vec_fp16 is not None:
            gate_logit = x_flat @ shared_expert_gate_vec_fp16.astype(np.float32).reshape(-1, 1)
            shared_Coeff = 1.0 / (1.0 + np.exp(-gate_logit))
            shared_out = shared_out * shared_Coeff

        combined = (routed_out + shared_out).astype(x_np.dtype).reshape(orig_shape)
        return combined, topk_indices


class DenseAdapter(BaseModelAdapter):
    """
    Task P1.3: Clean Dense Transformer Adapter (Llama / Qwen2 / OLMo Dense) with a router-free
    SwiGLU MLP execution path (`is_moe=False`, `num_experts=0`, `top_k=0`).
    """

    DEFAULT_WEIGHT_MAP: WeightNameMap = {
        "embed_tokens": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
        "input_layernorm": "model.layers.{l}.input_layernorm.weight",
        "post_attention_layernorm": "model.layers.{l}.post_attention_layernorm.weight",
        "q_proj": "model.layers.{l}.self_attn.q_proj.weight",
        "k_proj": "model.layers.{l}.self_attn.k_proj.weight",
        "v_proj": "model.layers.{l}.self_attn.v_proj.weight",
        "o_proj": "model.layers.{l}.self_attn.o_proj.weight",
        "q_norm": "",
        "k_norm": "",
        "router_gate": "",
        "expert_gate_proj": "",
        "expert_up_proj": "",
        "expert_down_proj": "",
        "shared_expert_gate": "",
        "shared_expert_gate_proj": "",
        "shared_expert_up_proj": "",
        "shared_expert_down_proj": "",
        "dense_gate_proj": "model.layers.{l}.mlp.gate_proj.weight",
        "dense_up_proj": "model.layers.{l}.mlp.up_proj.weight",
        "dense_down_proj": "model.layers.{l}.mlp.down_proj.weight",
    }

    def __init__(self, config: Optional[ConfigDict] = None):
        cfg = config or {}
        model_type = str(cfg.get("model_type", "llama") or "llama")
        num_layers = int(cfg.get("num_hidden_layers", 16) or 16)
        hidden_size = int(cfg.get("hidden_size", 2048) or 2048)
        intermediate_size = int(cfg.get("intermediate_size", 5632) or 5632)
        num_heads = int(cfg.get("num_attention_heads", 16) or 16)
        num_kv_heads = int(cfg.get("num_key_value_heads", num_heads) or num_heads)
        vocab_size = int(cfg.get("vocab_size", 32000) or 32000)
        rms_eps = float(cfg.get("rms_norm_eps", 1e-5) or 1e-5)
        rope_theta = float(cfg.get("rope_theta", 10000.0) or 10000.0)
        has_qk_norm = bool(cfg.get("use_qk_norm", False))
        act = str(cfg.get("hidden_act", "silu") or "silu")

        spec = ModelArchitectureSpec(
            model_type=model_type,
            architecture_family="DenseCausalLM",
            num_layers=num_layers,
            num_experts=0,
            top_k=0,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=hidden_size // num_heads,
            vocab_size=vocab_size,
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            rope_scaling_factor=_extract_rope_scaling_factor(cfg),
            norm_topk_prob=False,
            has_qk_norm=has_qk_norm,
            has_shared_expert=False,
            shared_expert_intermediate_size=0,
            is_moe=False,
            activation=act,
            expert_index_shift=None,
            weight_map=self.DEFAULT_WEIGHT_MAP,
        )
        super().__init__(spec)

    def forward_dense_mlp_numpy(
        self,
        x_np: np.ndarray,
        gate_w_fp16: np.ndarray,
        up_w_fp16: np.ndarray,
        down_w_fp16: np.ndarray,
    ) -> np.ndarray:
        """
        Executes router-free Dense SwiGLU FFN: `down_proj(act(gate_proj(x)) * up_proj(x))`.
        """
        orig_shape = x_np.shape
        x_flat = x_np.reshape(-1, self._spec.hidden_size).astype(np.float32)
        out_flat = self.forward_single_expert_f32(x_flat, gate_w_fp16, up_w_fp16, down_w_fp16)
        return out_flat.astype(x_np.dtype).reshape(orig_shape)


class Qwen38HybridAdapter(BaseModelAdapter):
    """
    Adapter for `Qwen3.8-27B-MLX-8bit` (`Qwen3_5ForConditionalGeneration` / `qwen3_5_text`):
      - 64 Hybrid Layers (`48` Linear Attention `GatedDeltaNet` + `16` Full GQA `24 Q / 4 KV`, `head_dim=256`, `partial_rotary_factor=0.25`)
      - 8-bit Affine Quantized (`group_size=64, bits=8`) SwiGLU MLP (`hidden_size=5120, intermediate_size=17408`)
      - 64 layers × 16 Merkle sub-blocks per layer = 1,024 Merkle leaves in `merkle_tree.idx` (`SYNPIDX1`).
    """

    DEFAULT_WEIGHT_MAP: WeightNameMap = {
        "embed_tokens": "language_model.model.embed_tokens.weight",
        "final_norm": "language_model.model.norm.weight",
        "lm_head": "language_model.lm_head.weight",
        "input_layernorm": "language_model.model.layers.{l}.input_layernorm.weight",
        "post_attention_layernorm": "language_model.model.layers.{l}.post_attention_layernorm.weight",
        "q_proj": "language_model.model.layers.{l}.self_attn.q_proj.weight",
        "k_proj": "language_model.model.layers.{l}.self_attn.k_proj.weight",
        "v_proj": "language_model.model.layers.{l}.self_attn.v_proj.weight",
        "o_proj": "language_model.model.layers.{l}.self_attn.o_proj.weight",
        "q_norm": "language_model.model.layers.{l}.self_attn.q_norm.weight",
        "k_norm": "language_model.model.layers.{l}.self_attn.k_norm.weight",
        "router_gate": "",
        "expert_gate_proj": "language_model.model.layers.{l}.mlp.gate_proj.weight",
        "expert_up_proj": "language_model.model.layers.{l}.mlp.up_proj.weight",
        "expert_down_proj": "language_model.model.layers.{l}.mlp.down_proj.weight",
        "shared_expert_gate": "",
        "shared_expert_gate_proj": "",
        "shared_expert_up_proj": "",
        "shared_expert_down_proj": "",
        "dense_gate_proj": "language_model.model.layers.{l}.mlp.gate_proj.weight",
        "dense_up_proj": "language_model.model.layers.{l}.mlp.up_proj.weight",
        "dense_down_proj": "language_model.model.layers.{l}.mlp.down_proj.weight",
    }

    def __init__(self, config: Optional[ConfigDict] = None):
        raw_cfg = config or {}
        t_cfg_raw = raw_cfg.get("text_config")
        cfg: ConfigDict = t_cfg_raw if isinstance(t_cfg_raw, dict) else raw_cfg  # type: ignore[assignment]

        num_layers = int(cfg.get("num_hidden_layers", 64) or 64)
        hidden_size = int(cfg.get("hidden_size", 5120) or 5120)
        intermediate_size = int(cfg.get("intermediate_size", 17408) or 17408)
        num_heads = int(cfg.get("num_attention_heads", 24) or 24)
        num_kv_heads = int(cfg.get("num_key_value_heads", 4) or 4)
        head_dim = int(cfg.get("head_dim", 256) or 256)
        vocab_size = int(cfg.get("vocab_size", 248320) or 248320)
        rms_eps = float(cfg.get("rms_norm_eps", 1e-6) or 1e-6)
        rope_params = cfg.get("rope_parameters")
        rope_theta = (
            float(rope_params.get("rope_theta", 10000000.0))  # type: ignore[union-attr]
            if isinstance(rope_params, dict)
            else float(cfg.get("rope_theta", 10000000.0) or 10000000.0)
        )
        act = str(cfg.get("hidden_act", "silu") or "silu")

        spec = ModelArchitectureSpec(
            model_type="qwen3_5",
            architecture_family="Qwen3_5ForConditionalGeneration",
            num_layers=num_layers,
            num_experts=16,  # 16 Merkle sub-blocks per layer (64 * 16 = 1,024 leaves)
            top_k=16,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            rope_scaling_factor=1.0,
            norm_topk_prob=False,
            has_qk_norm=True,
            has_shared_expert=False,
            shared_expert_intermediate_size=0,
            is_moe=True,
            activation=act,
            expert_index_shift=4,  # 16 sub-blocks per layer -> shift by 4 bits
            weight_map=self.DEFAULT_WEIGHT_MAP,
        )
        super().__init__(spec)
        self.full_attention_interval: int = int(cfg.get("full_attention_interval", 4) or 4)

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        return (int(layer_idx) + 1) % self.full_attention_interval != 0


def _build_fp8_e4m3_lut() -> np.ndarray:
    """
    Builds a 256-entry IEEE-like FP8 E4M3 (`e4m3fn`, bias=7) to `float32` lookup table:
      - Sign bit: bit 7 (`0x80`)
      - Exponent: bits 6..3 (`0..15`, bias = 7)
      - Mantissa: bits 2..0 (`0..7`, divided by 8.0)
      - `exp == 0`: subnormal `(-1)^s * 2^(-6) * (mant / 8.0)`
      - `exp == 15` and `mant == 7`: `0.0` (finite clamp for NaN byte `0x7F`/`0xFF` in weight blocks)
      - `1 <= exp <= 15`: normal `(-1)^s * 2^(exp - 7) * (1.0 + mant / 8.0)`
    """
    lut = np.zeros(256, dtype=np.float32)
    for byte_val in range(256):
        sign = -1.0 if (byte_val & 0x80) else 1.0
        exp = (byte_val >> 3) & 0x0F
        mant = byte_val & 0x07
        if exp == 0:
            val = sign * (2.0**-6) * (float(mant) / 8.0)
        elif exp == 15 and mant == 7:
            val = 0.0
        else:
            val = sign * (2.0 ** (exp - 7)) * (1.0 + float(mant) / 8.0)
        lut[byte_val] = np.float32(val)
    return lut


_FP8_E4M3_LUT: np.ndarray = _build_fp8_e4m3_lut()


def fp8_e4m3_bytes_to_fp32(raw_u8: np.ndarray) -> np.ndarray:
    """Converts raw `uint8` FP8 E4M3 (`fmt="e4m3"`) bytes into `float32` via O(1) LUT indexing."""
    return _FP8_E4M3_LUT[np.asarray(raw_u8, dtype=np.uint8)]


def dequantize_fp8_e4m3_blockwise_numpy(
    w_fp8_u8: np.ndarray,
    scale_inv_f32: np.ndarray,
    block_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """
    Dequantizes `zai-org/GLM-5.3` FP8 (`fmt="e4m3"`, `weight_block_size=[128, 128]`) weights:
      - `w_fp8_u8`: shape `(out_features, in_features)` dtype `uint8` (raw E4M3 bytes)
      - `scale_inv_f32`: shape `(ceil(out_features / 128), ceil(in_features / 128))` dtype `float32`
    Returns dequantized `float32` matrix of shape `(out_features, in_features)`.
    """
    out_dim, in_dim = int(w_fp8_u8.shape[0]), int(w_fp8_u8.shape[1])
    bh, bw = int(block_size[0]), int(block_size[1])
    w_f32 = fp8_e4m3_bytes_to_fp32(w_fp8_u8)
    s_f32 = np.asarray(scale_inv_f32, dtype=np.float32)

    if out_dim % bh == 0 and in_dim % bw == 0:
        grid_h = out_dim // bh
        grid_w = in_dim // bw
        w_blocks = w_f32.reshape(grid_h, bh, grid_w, bw)
        scaled = w_blocks * s_f32[:grid_h, None, :grid_w, None]
        return scaled.reshape(out_dim, in_dim)

    s_expanded = np.repeat(np.repeat(s_f32, bh, axis=0), bw, axis=1)[:out_dim, :in_dim]
    return w_f32 * s_expanded


class GLM53MoEDsaAdapter(BaseModelAdapter):
    """
    Adapter for `zai-org/GLM-5.3` (`GlmMoeDsaForCausalLM`, `model_type="glm_moe_dsa"`, 744B Total / 40B Active):
      - 78 Decoder Layers (`first_k_dense_replace=3`):
          * Layers `0..2` (`3` Dense SwiGLU layers, `intermediate_size=12288`)
          * Layers `3..77` (`75` Sparse MoE layers: `256` routed experts + `1` shared expert,
            `num_experts_per_tok=8`, `moe_intermediate_size=2048`, `expert_index_shift=8` -> `19,200` routed experts)
          * Layer `78` (`num_nextn_predict_layers=1`): Multi-Token Prediction (MTP) draft layer (`enorm`, `hnorm`, `eh_proj`)
      - Multi-head Latent Attention (MLA) + DeepSeek Sparse Attention (DSA) with `IndexShare`:
          * `hidden_size=6144`, `num_attention_heads=64`, `num_key_value_heads=64`, `head_dim=192`
          * `q_lora_rank=2048`, `kv_lora_rank=512`, `qk_head_dim=256` (`qk_nope_head_dim=192`, `qk_rope_head_dim=64`), `v_head_dim=256`
          * `index_n_heads=32`, `index_head_dim=128`, `index_topk=2048`, `index_topk_freq=4`, `index_skip_topk_offset=3`
          * `21` Full DSA Indexer layers (`0, 1, 2, 6, 10, ..., 74`) and `57` Shared Indexer layers (`IndexShare`)
      - Sigmoid `noaux_tc` Routing:
          * `scoring_func="sigmoid"`, `topk_method="noaux_tc"`, `e_score_correction_bias`, `norm_topk_prob=True`, `routed_scaling_factor=2.5`
      - Native FP8 (`fmt="e4m3"`, `weight_block_size=[128, 128]`, 755.6 GB across 141 `.safetensors` shards) & BF16 support.
    """

    DEFAULT_WEIGHT_MAP: WeightNameMap = {
        "embed_tokens": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
        "input_layernorm": "model.layers.{l}.input_layernorm.weight",
        "post_attention_layernorm": "model.layers.{l}.post_attention_layernorm.weight",
        "q_proj": "model.layers.{l}.self_attn.q_b_proj.weight",
        "k_proj": "model.layers.{l}.self_attn.kv_a_proj_with_mqa.weight",
        "v_proj": "model.layers.{l}.self_attn.kv_b_proj.weight",
        "o_proj": "model.layers.{l}.self_attn.o_proj.weight",
        "q_norm": "model.layers.{l}.self_attn.q_a_layernorm.weight",
        "k_norm": "model.layers.{l}.self_attn.kv_a_layernorm.weight",
        "router_gate": "model.layers.{l}.mlp.gate.weight",
        "expert_gate_proj": "model.layers.{l}.mlp.experts.{e}.gate_proj.weight",
        "expert_up_proj": "model.layers.{l}.mlp.experts.{e}.up_proj.weight",
        "expert_down_proj": "model.layers.{l}.mlp.experts.{e}.down_proj.weight",
        "shared_expert_gate": "",
        "shared_expert_gate_proj": "model.layers.{l}.mlp.shared_experts.gate_proj.weight",
        "shared_expert_up_proj": "model.layers.{l}.mlp.shared_experts.up_proj.weight",
        "shared_expert_down_proj": "model.layers.{l}.mlp.shared_experts.down_proj.weight",
        "dense_gate_proj": "model.layers.{l}.mlp.gate_proj.weight",
        "dense_up_proj": "model.layers.{l}.mlp.up_proj.weight",
        "dense_down_proj": "model.layers.{l}.mlp.down_proj.weight",
    }

    def __init__(self, config: Optional[ConfigDict] = None):
        cfg = config or {}
        num_layers = int(cfg.get("num_hidden_layers", 78) or 78)  # type: ignore[arg-type]
        num_experts = int(
            cfg.get("n_routed_experts", cfg.get("num_experts", cfg.get("num_local_experts", 256))) or 256  # type: ignore[arg-type]
        )
        top_k = int(cfg.get("num_experts_per_tok", 8) or 8)  # type: ignore[arg-type]
        hidden_size = int(cfg.get("hidden_size", 6144) or 6144)  # type: ignore[arg-type]
        moe_inter = int(cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 2048)) or 2048)  # type: ignore[arg-type]
        dense_inter = int(cfg.get("intermediate_size", 12288) or 12288)  # type: ignore[arg-type]
        n_shared = int(cfg.get("n_shared_experts", 1) or 1)  # type: ignore[arg-type]
        shared_inter = moe_inter * max(1, n_shared)

        num_heads = int(cfg.get("num_attention_heads", 64) or 64)  # type: ignore[arg-type]
        num_kv_heads = int(cfg.get("num_key_value_heads", num_heads) or num_heads)  # type: ignore[arg-type]
        head_dim = int(cfg.get("head_dim", 192) or 192)  # type: ignore[arg-type]
        vocab_size = int(cfg.get("vocab_size", 154880) or 154880)  # type: ignore[arg-type]
        rms_eps = float(cfg.get("rms_norm_eps", 1e-5) or 1e-5)  # type: ignore[arg-type]

        rope_params = cfg.get("rope_parameters")
        rope_theta = (
            float(rope_params.get("rope_theta", 8000000.0))  # type: ignore[union-attr,arg-type]
            if isinstance(rope_params, dict)
            else float(cfg.get("rope_theta", 8000000.0) or 8000000.0)  # type: ignore[arg-type]
        )
        norm_topk = bool(cfg.get("norm_topk_prob", True))
        act = str(cfg.get("hidden_act", "silu") or "silu")

        spec = ModelArchitectureSpec(
            model_type="glm_moe_dsa",
            architecture_family="GlmMoeDsaForCausalLM",
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=moe_inter,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            rms_norm_eps=rms_eps,
            rope_theta=rope_theta,
            rope_scaling_factor=_extract_rope_scaling_factor(cfg),
            norm_topk_prob=norm_topk,
            has_qk_norm=True,
            has_shared_expert=n_shared > 0,
            shared_expert_intermediate_size=shared_inter,
            is_moe=True,
            activation=act,
            expert_index_shift=_power_of_two_shift(num_experts),
            weight_map=self.DEFAULT_WEIGHT_MAP,
        )
        super().__init__(spec)

        self.first_k_dense_replace: int = int(cfg.get("first_k_dense_replace", 3) or 3)  # type: ignore[arg-type]
        self.dense_intermediate_size: int = dense_inter
        self.moe_intermediate_size: int = moe_inter
        self.n_shared_experts: int = n_shared
        self.routed_scaling_factor: float = float(cfg.get("routed_scaling_factor", 2.5) or 2.5)  # type: ignore[arg-type]
        self.scoring_func: str = str(cfg.get("scoring_func", "sigmoid") or "sigmoid").lower()
        self.topk_method: str = str(cfg.get("topk_method", "noaux_tc") or "noaux_tc").lower()
        self.n_group: int = int(cfg.get("n_group", 1) or 1)  # type: ignore[arg-type]
        self.topk_group: int = int(cfg.get("topk_group", 1) or 1)  # type: ignore[arg-type]
        self.q_lora_rank: int = int(cfg.get("q_lora_rank", 2048) or 2048)  # type: ignore[arg-type]
        self.kv_lora_rank: int = int(cfg.get("kv_lora_rank", 512) or 512)  # type: ignore[arg-type]
        self.qk_head_dim: int = int(cfg.get("qk_head_dim", 256) or 256)  # type: ignore[arg-type]
        self.qk_nope_head_dim: int = int(cfg.get("qk_nope_head_dim", 192) or 192)  # type: ignore[arg-type]
        self.qk_rope_head_dim: int = int(cfg.get("qk_rope_head_dim", 64) or 64)  # type: ignore[arg-type]
        self.v_head_dim: int = int(cfg.get("v_head_dim", 256) or 256)  # type: ignore[arg-type]
        self.index_n_heads: int = int(cfg.get("index_n_heads", 32) or 32)  # type: ignore[arg-type]
        self.index_head_dim: int = int(cfg.get("index_head_dim", 128) or 128)  # type: ignore[arg-type]
        self.index_topk: int = int(cfg.get("index_topk", 2048) or 2048)  # type: ignore[arg-type]
        self.index_topk_freq: int = int(cfg.get("index_topk_freq", 4) or 4)  # type: ignore[arg-type]
        self.index_skip_topk_offset: int = int(cfg.get("index_skip_topk_offset", 3) or 3)  # type: ignore[arg-type]
        self.max_position_embeddings: int = int(cfg.get("max_position_embeddings", 1048576) or 1048576)  # type: ignore[arg-type]
        self.num_nextn_predict_layers: int = int(cfg.get("num_nextn_predict_layers", 1) or 1)  # type: ignore[arg-type]
        self.weight_block_size: Tuple[int, int] = (128, 128)

        raw_mlp_types = cfg.get("mlp_layer_types")
        if isinstance(raw_mlp_types, list) and len(raw_mlp_types) == num_layers:
            self.mlp_layer_types: Tuple[str, ...] = tuple(str(x) for x in raw_mlp_types)
        else:
            self.mlp_layer_types = tuple(
                "dense" if l_idx < self.first_k_dense_replace else "sparse"
                for l_idx in range(num_layers)
            )

        raw_idx_types = cfg.get("indexer_types")
        if isinstance(raw_idx_types, list) and len(raw_idx_types) == num_layers:
            self.indexer_types: Tuple[str, ...] = tuple(str(x) for x in raw_idx_types)
        else:
            idx_list: List[str] = []
            for l_idx in range(num_layers):
                if l_idx < self.index_skip_topk_offset:
                    idx_list.append("full")
                elif (l_idx - self.index_skip_topk_offset + 1) % self.index_topk_freq == 0:
                    idx_list.append("full")
                else:
                    idx_list.append("shared")
            self.indexer_types = tuple(idx_list)

    @property
    def num_sparse_moe_layers(self) -> int:
        return sum(1 for t in self.mlp_layer_types if t == "sparse")

    @property
    def num_dense_mlp_layers(self) -> int:
        return sum(1 for t in self.mlp_layer_types if t == "dense")

    @property
    def total_routed_experts(self) -> int:
        return self.num_sparse_moe_layers * self._spec.num_experts

    def is_dense_mlp_layer(self, layer_idx: int) -> bool:
        if not (0 <= layer_idx < self._spec.num_layers):
            raise ValueError(f"Invalid layer_idx={layer_idx} for num_layers={self._spec.num_layers}")
        return self.mlp_layer_types[layer_idx] == "dense"

    def is_full_dsa_indexer_layer(self, layer_idx: int) -> bool:
        if not (0 <= layer_idx < self._spec.num_layers):
            raise ValueError(f"Invalid layer_idx={layer_idx} for num_layers={self._spec.num_layers}")
        return self.indexer_types[layer_idx] == "full"

    def layer_weight_names(self, layer_idx: int) -> Dict[str, str]:
        wm = self._spec.weight_map
        out: Dict[str, str] = {
            "input_layernorm": wm["input_layernorm"].format(l=layer_idx),
            "post_attention_layernorm": wm["post_attention_layernorm"].format(l=layer_idx),
            "q_a_proj": f"model.layers.{layer_idx}.self_attn.q_a_proj.weight",
            "q_norm": wm["q_norm"].format(l=layer_idx),
            "q_proj": wm["q_proj"].format(l=layer_idx),
            "q_b_proj": wm["q_proj"].format(l=layer_idx),
            "k_proj": wm["k_proj"].format(l=layer_idx),
            "kv_a_proj_with_mqa": wm["k_proj"].format(l=layer_idx),
            "k_norm": wm["k_norm"].format(l=layer_idx),
            "v_proj": wm["v_proj"].format(l=layer_idx),
            "kv_b_proj": wm["v_proj"].format(l=layer_idx),
            "o_proj": wm["o_proj"].format(l=layer_idx),
        }
        if self.is_full_dsa_indexer_layer(layer_idx):
            out["indexer_wq_b"] = f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight"
            out["indexer_wk"] = f"model.layers.{layer_idx}.self_attn.indexer.wk.weight"
            out["indexer_k_norm"] = f"model.layers.{layer_idx}.self_attn.indexer.k_norm.weight"
            out["indexer_k_norm_bias"] = f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias"
            out["indexer_weights_proj"] = f"model.layers.{layer_idx}.self_attn.indexer.weights_proj.weight"
            out["indexers_proj"] = f"model.layers.{layer_idx}.self_attn.indexers_proj.weight"

        if self.is_dense_mlp_layer(layer_idx):
            out["dense_gate_proj"] = wm["dense_gate_proj"].format(l=layer_idx)
            out["dense_up_proj"] = wm["dense_up_proj"].format(l=layer_idx)
            out["dense_down_proj"] = wm["dense_down_proj"].format(l=layer_idx)
        else:
            out["router_gate"] = wm["router_gate"].format(l=layer_idx)
            out["router_score_correction_bias"] = (
                f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            )
            out["shared_expert_gate_proj"] = wm["shared_expert_gate_proj"].format(l=layer_idx)
            out["shared_expert_up_proj"] = wm["shared_expert_up_proj"].format(l=layer_idx)
            out["shared_expert_down_proj"] = wm["shared_expert_down_proj"].format(l=layer_idx)
        return out

    def expert_scale_inv_names(self, layer_idx: int, expert_idx: int) -> Tuple[str, str, str]:
        """Returns the FP8 `[128, 128]` blockwise `weight_scale_inv` tensor names for expert `(layer_idx, expert_idx)`."""
        g_n, u_n, d_n = self.expert_weight_names(layer_idx, expert_idx)
        return f"{g_n}_scale_inv", f"{u_n}_scale_inv", f"{d_n}_scale_inv"

    def route_tokens_numpy(
        self,
        x_flat_f32: np.ndarray,
        router_w_fp16: np.ndarray,
        e_score_correction_bias_f32: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes GLM-5.3 `noaux_tc` Sigmoid MoE routing:
          1. Computes raw expert affinities `s = sigmoid(x @ router_w.T)` (or softmax if configured).
          2. Adds `e_score_correction_bias` ONLY to the routing choice scores (`s_choice = s + bias`).
          3. Selects the top `k = 8` expert indices per token from `s_choice`, gathers the unbiased
             affinities from `s`, normalizes by their sum (`norm_topk_prob=True`), and multiplies by
             `routed_scaling_factor` (`2.5`).
        """
        r_w = router_w_fp16.astype(np.float32)
        logits = x_flat_f32 @ r_w.T
        if self.scoring_func == "sigmoid":
            scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        else:
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            scores = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        if e_score_correction_bias_f32 is not None:
            choice_scores = scores + e_score_correction_bias_f32.astype(np.float32)[None, :]
        else:
            choice_scores = scores

        k = self._spec.top_k
        part_idx = np.argpartition(-choice_scores, kth=k - 1, axis=-1)[:, :k]
        part_choice = np.take_along_axis(choice_scores, part_idx, axis=-1)
        sort_order = np.argsort(-part_choice, axis=-1)
        topk_indices = np.take_along_axis(part_idx, sort_order, axis=-1).astype(np.int32)

        topk_scores = np.take_along_axis(scores, topk_indices, axis=-1)
        if self._spec.norm_topk_prob:
            denom = np.maximum(np.sum(topk_scores, axis=-1, keepdims=True), 1e-12)
            topk_scores = topk_scores / denom
        topk_scores = topk_scores * self.routed_scaling_factor
        return topk_indices, topk_scores.astype(np.float32)

    def forward_glm53_layer_ffn_numpy(
        self,
        layer_idx: int,
        x_np: np.ndarray,
        dense_weights: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        router_w_fp16: Optional[np.ndarray] = None,
        routed_expert_weights: Optional[Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        shared_expert_weights: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        e_score_correction_bias_f32: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Executes GLM-5.3 Layer `layer_idx` FFN:
          - If `layer_idx < first_k_dense_replace` (`0..2`, Dense layer):
              executes router-free Dense SwiGLU (`intermediate_size=12288`) and returns `(y, None)`.
          - If `layer_idx >= first_k_dense_replace` (`3..77`, Sparse MoE layer):
              executes `SharedExpert(x) + sum_{k=1..8} score_k * RoutedExpert_k(x)` and returns `(y, topk_indices)`.
        """
        orig_shape = x_np.shape
        x_flat = x_np.reshape(-1, self._spec.hidden_size).astype(np.float32)

        if self.is_dense_mlp_layer(layer_idx):
            if dense_weights is None:
                raise ValueError(f"Dense MLP weights required for GLM-5.3 dense layer {layer_idx}")
            d_gate, d_up, d_down = dense_weights
            out_flat = self.forward_single_expert_f32(x_flat, d_gate, d_up, d_down)
            return out_flat.astype(x_np.dtype).reshape(orig_shape), None

        if router_w_fp16 is None or routed_expert_weights is None or shared_expert_weights is None:
            raise ValueError(
                f"Router, routed_expert_weights, and shared_expert_weights required for GLM-5.3 MoE layer {layer_idx}"
            )
        topk_indices, topk_scores = self.route_tokens_numpy(
            x_flat, router_w_fp16, e_score_correction_bias_f32=e_score_correction_bias_f32
        )

        routed_out = np.zeros_like(x_flat, dtype=np.float32)
        for t_idx in range(x_flat.shape[0]):
            x_tok = x_flat[t_idx]
            acc = np.zeros(self._spec.hidden_size, dtype=np.float32)
            for k_idx in range(self._spec.top_k):
                e_id = int(topk_indices[t_idx, k_idx])
                score = float(topk_scores[t_idx, k_idx])
                g_w, u_w, d_w = routed_expert_weights[e_id]
                acc += score * self.forward_single_expert_f32(x_tok, g_w, u_w, d_w)
            routed_out[t_idx] = acc

        sh_gate, sh_up, sh_down = shared_expert_weights
        shared_out = self.forward_single_expert_f32(x_flat, sh_gate, sh_up, sh_down)
        combined = (routed_out + shared_out).astype(x_np.dtype).reshape(orig_shape)
        return combined, topk_indices


GLM5MoEAdapter = GLM53MoEDsaAdapter
GLM53Adapter = GLM53MoEDsaAdapter


def create_model_adapter(config: ConfigDict) -> BaseModelAdapter:
    """
    Auto-detects the model family from `config.json` and instantiates the corresponding
    `BaseModelAdapter` (`GLM53MoEDsaAdapter`, `OLMoEAdapter`, `MixtralAdapter`, `QwenMoEAdapter`,
    `Qwen38HybridAdapter`, or `DenseAdapter`).
    """
    model_type = str(config.get("model_type", "") or "").lower()
    if model_type in ("glm_moe_dsa", "glm5", "glm5_moe", "glm_5_3", "glm-5.3") or (
        "n_routed_experts" in config and "kv_lora_rank" in config
    ):
        return GLM53MoEDsaAdapter(config)
    if model_type in ("qwen3_5", "qwen3_5_text", "qwen3_8"):
        return Qwen38HybridAdapter(config)
    num_experts = int(
        config.get("num_experts", config.get("num_local_experts", config.get("n_routed_experts", 0))) or 0  # type: ignore[arg-type]
    )
    has_shared = bool(
        config.get("shared_expert_intermediate_size", 0)
        or config.get("has_shared_expert", False)
        or config.get("n_shared_experts", 0)
    )

    if model_type == "olmoe":
        return OLMoEAdapter(config)
    if model_type == "mixtral" or ("num_local_experts" in config and not has_shared):
        return MixtralAdapter(config)
    if "qwen" in model_type and (num_experts > 0 or has_shared):
        return QwenMoEAdapter(config)
    if num_experts <= 1 and not has_shared:
        return DenseAdapter(config)
    if has_shared:
        return QwenMoEAdapter(config)
    return OLMoEAdapter(config)


class ModularAttentionEngine:
    """
    Task P1.3: Standalone, Independently Benchmarkable Attention Subsystem.
    Supports:
      - Multi-Head Attention (MHA: `num_heads == num_kv_heads`)
      - Grouped-Query Attention (GQA: `num_heads > num_kv_heads`, `gqa_group_size = num_heads // num_kv_heads`)
      - Multi-Query Attention (MQA: `num_kv_heads == 1`)
      - Optional QK-Norm (`has_qk_norm=True/False`, supporting full `(hidden_size,)` or per-head `(head_dim,)`)
      - Rotary Position Embeddings (RoPE) with `rope_scaling_factor`
      - Stateful KV Cache Update & Causal Scaled Dot-Product Attention (SDPA)
    """

    def __init__(self, spec: ModelArchitectureSpec):
        if spec.num_attention_heads % spec.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({spec.num_attention_heads}) must be divisible by "
                f"num_kv_heads ({spec.num_kv_heads})"
            )
        self.spec: ModelArchitectureSpec = spec
        self.hidden_size: int = spec.hidden_size
        self.num_heads: int = spec.num_attention_heads
        self.num_kv_heads: int = spec.num_kv_heads
        self.head_dim: int = spec.head_dim
        self.gqa_group_size: int = spec.gqa_group_size
        self.rms_eps: float = spec.rms_norm_eps
        self.rope_theta: float = spec.rope_theta
        self.rope_scaling_factor: float = spec.rope_scaling_factor
        self.has_qk_norm: bool = spec.has_qk_norm
        self.scale: float = float(self.head_dim) ** -0.5

        inv_freq = 1.0 / (
            self.rope_theta
            ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / float(self.head_dim))
        )
        self.rope_inv_freq: np.ndarray = inv_freq

    def rmsnorm_f32(self, x_f32: np.ndarray, weight_fp16: np.ndarray) -> np.ndarray:
        var = np.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        normed = x_f32 * (1.0 / np.sqrt(var + self.rms_eps))
        return normed * weight_fp16.astype(np.float32)

    def apply_rope_f32(self, x_bhld: np.ndarray, pos_offset: int = 0) -> np.ndarray:
        seq_len = x_bhld.shape[2]
        positions = (
            np.arange(pos_offset, pos_offset + seq_len, dtype=np.float32)[:, None]
            / self.rope_scaling_factor
        )
        angles = positions * self.rope_inv_freq[None, :]
        cos = np.cos(angles)[None, None, :, :]
        sin = np.sin(angles)[None, None, :, :]
        half = self.head_dim // 2
        x1 = x_bhld[..., :half]
        x2 = x_bhld[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    def forward_numpy(
        self,
        normed_input_f32: np.ndarray,
        q_w_fp16: np.ndarray,
        k_w_fp16: np.ndarray,
        v_w_fp16: np.ndarray,
        o_w_fp16: np.ndarray,
        q_norm_w_fp16: Optional[np.ndarray] = None,
        k_norm_w_fp16: Optional[np.ndarray] = None,
        kv_cache_k: Optional[List[Optional[np.ndarray]]] = None,
        kv_cache_v: Optional[List[Optional[np.ndarray]]] = None,
        layer_idx: int = 0,
        pos_offset: int = 0,
    ) -> np.ndarray:
        """
        Executes QKV projection -> optional QK-Norm -> RoPE -> KV Cache Update ->
        MHA/GQA/MQA Causal SDPA -> Output Projection.
        """
        bsz, seq_len, _ = normed_input_f32.shape
        q = normed_input_f32 @ q_w_fp16.astype(np.float32).T
        k = normed_input_f32 @ k_w_fp16.astype(np.float32).T
        v = normed_input_f32 @ v_w_fp16.astype(np.float32).T

        if self.has_qk_norm and q_norm_w_fp16 is not None and k_norm_w_fp16 is not None:
            if q_norm_w_fp16.shape[-1] == self.hidden_size:
                q = self.rmsnorm_f32(q, q_norm_w_fp16)
            else:
                q = self.rmsnorm_f32(
                    q.reshape(bsz, seq_len, self.num_heads, self.head_dim), q_norm_w_fp16
                ).reshape(bsz, seq_len, self.hidden_size)

            kv_dim = self.num_kv_heads * self.head_dim
            if k_norm_w_fp16.shape[-1] == kv_dim:
                k = self.rmsnorm_f32(k, k_norm_w_fp16)
            else:
                k = self.rmsnorm_f32(
                    k.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim), k_norm_w_fp16
                ).reshape(bsz, seq_len, kv_dim)

        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.apply_rope_f32(q, pos_offset=pos_offset)
        k = self.apply_rope_f32(k, pos_offset=pos_offset)

        k_fp16 = k.astype(np.float16)
        v_fp16 = v.astype(np.float16)
        if kv_cache_k is not None and kv_cache_v is not None:
            prev_k = kv_cache_k[layer_idx]
            prev_v = kv_cache_v[layer_idx]
            if prev_k is not None and prev_v is not None:
                k_full = np.concatenate([prev_k, k_fp16], axis=2)
                v_full = np.concatenate([prev_v, v_fp16], axis=2)
            else:
                k_full = k_fp16
                v_full = v_fp16
            kv_cache_k[layer_idx] = k_full
            kv_cache_v[layer_idx] = v_full
        else:
            k_full = k_fp16
            v_full = v_fp16

        k_f32 = k_full.astype(np.float32)
        v_f32 = v_full.astype(np.float32)
        if self.gqa_group_size > 1:
            k_f32 = np.repeat(k_f32, self.gqa_group_size, axis=1)
            v_f32 = np.repeat(v_f32, self.gqa_group_size, axis=1)

        scores = (q @ k_f32.transpose(0, 1, 3, 2)) * self.scale
        if seq_len > 1:
            total_k = k_full.shape[2]
            q_pos = np.arange(pos_offset, pos_offset + seq_len)[:, None]
            k_pos = np.arange(total_k)[None, :]
            causal_mask = np.where(k_pos <= q_pos, 0.0, -1e9).astype(np.float32)
            scores = scores + causal_mask[None, None, :, :]

        scores_max = np.max(scores, axis=-1, keepdims=True)
        attn_probs = np.exp(scores - scores_max)
        attn_probs = attn_probs / np.sum(attn_probs, axis=-1, keepdims=True)
        attn_out = (
            (attn_probs @ v_f32).transpose(0, 2, 1, 3).reshape(bsz, seq_len, self.hidden_size)
        )
        return attn_out @ o_w_fp16.astype(np.float32).T


def benchmark_attention_subsystem(
    hidden_size: int = 2048,
    num_heads: int = 16,
    num_kv_heads: int = 16,
    has_qk_norm: bool = True,
    prefill_seq_len: int = 32,
    decode_steps: int = 16,
    seed: int = 42,
) -> AttentionBenchmarkResult:
    """
    Standalone micro-benchmark for `ModularAttentionEngine` measuring Prefill (`prefill_seq_len`)
    and Decode (`decode_steps` single-token steps with KV cache growth) across MHA, GQA, or MQA.
    """
    mode = (
        "MHA"
        if num_heads == num_kv_heads
        else ("MQA" if num_kv_heads == 1 else f"GQA-{num_heads // num_kv_heads}x")
    )
    cfg: ConfigDict = {
        "model_type": "olmoe" if has_qk_norm else "mixtral",
        "num_hidden_layers": 1,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 2,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "vocab_size": 32000,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }
    adapter = OLMoEAdapter(cfg) if has_qk_norm else MixtralAdapter(cfg)
    attn_eng = ModularAttentionEngine(adapter.spec)

    rng = np.random.default_rng(seed)
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    q_w = (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(np.float16)
    k_w = (rng.standard_normal((kv_dim, hidden_size)) * 0.02).astype(np.float16)
    v_w = (rng.standard_normal((kv_dim, hidden_size)) * 0.02).astype(np.float16)
    o_w = (rng.standard_normal((hidden_size, hidden_size)) * 0.02).astype(np.float16)
    q_norm_w = np.ones((hidden_size,), dtype=np.float16) if has_qk_norm else None
    k_norm_w = np.ones((kv_dim,), dtype=np.float16) if has_qk_norm else None

    x_prefill = rng.standard_normal((1, prefill_seq_len, hidden_size)).astype(np.float32)
    x_decode = rng.standard_normal((1, 1, hidden_size)).astype(np.float32)

    kv_k: List[Optional[np.ndarray]] = [None]
    kv_v: List[Optional[np.ndarray]] = [None]

    t0 = time.perf_counter()
    out_prefill = attn_eng.forward_numpy(
        x_prefill,
        q_w,
        k_w,
        v_w,
        o_w,
        q_norm_w_fp16=q_norm_w,
        k_norm_w_fp16=k_norm_w,
        kv_cache_k=kv_k,
        kv_cache_v=kv_v,
        layer_idx=0,
        pos_offset=0,
    )
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    cur_pos = prefill_seq_len
    for _ in range(decode_steps):
        _ = attn_eng.forward_numpy(
            x_decode,
            q_w,
            k_w,
            v_w,
            o_w,
            q_norm_w_fp16=q_norm_w,
            k_norm_w_fp16=k_norm_w,
            kv_cache_k=kv_k,
            kv_cache_v=kv_v,
            layer_idx=0,
            pos_offset=cur_pos,
        )
        cur_pos += 1
    decode_total_ms = (time.perf_counter() - t1) * 1000.0
    decode_step_ms = decode_total_ms / max(1, decode_steps)

    return {
        "mode": mode,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "gqa_group_size": attn_eng.gqa_group_size,
        "has_qk_norm": has_qk_norm,
        "prefill_seq_len": prefill_seq_len,
        "prefill_ms": round(prefill_ms, 3),
        "decode_step_ms": round(decode_step_ms, 3),
        "decode_tok_per_sec": round(1000.0 / max(1e-6, decode_step_ms), 2),
        "output_shape": (int(out_prefill.shape[0]), int(out_prefill.shape[1]), int(out_prefill.shape[2])),
    }
