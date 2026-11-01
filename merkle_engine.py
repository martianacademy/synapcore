#!/usr/bin/env python3
"""
Low-RAM Merkle-MoE Inference Engine for OLMoE-1B-7B-0125-Instruct (Metal MPS + CPU Edition)
===========================================================================================
Executes OlmoeForCausalLM directly from the Hybrid Merkle / B+ Tree Store with Apple Metal
GPU (`mps`) Unified Memory Binding and Copy-on-Write (CoW) Continual Learning Overlay:

1. 3-Tier Apple Metal Unified Memory Architecture:
   - Tier 1 (Pinned on Metal GPU `mps` / RAM: 614.60 MiB):
     * Shared Core (Fused QKV Attention + Routers + RMSNorms: 516.25 MiB)
     * Quantized Output Head (INT8 + Post-Matmul FP16 Scales: 98.35 MiB)
   - Tier 2 (Hot-Expert LFU/LRU Cache on Metal GPU `mps` / RAM):
     * Top Hot Experts (e.g. 64 or 205 experts) resident directly as Metal `MTLBuffer` tensors
       for zero-copy, zero-allocation GPU matmuls at 546 GB/s unified memory bandwidth.
   - Tier 3 (Cold Experts on SSD `cas_experts.bin` + Reusable 12 MiB Metal Staging Buffer):
     * Zero-copy 16 KB page-aligned `mmap` views on SSD streamed into a single pre-allocated
       12.00 MiB Metal staging buffer (`_mps_staging_gu`, `_mps_staging_d`) via in-place `.copy_()`,
       eliminating dynamic GPU buffer allocation/deallocation thrashing on cold expert faults.
   - Copy-on-Write (CoW) Delta Overlay (`cas_cow_deltas.bin`):
     * Automatically overlays evolved synaptic leaves committed by `synapcore_learn.py` in
       chronological commit order (can be toggled with `--base-only` for instant rollback).
"""

import argparse
import collections
import concurrent.futures
import ctypes
import ctypes.util
import hashlib
import json
import mmap
import os
import resource
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from merkle_store import sha256_bytes, verify_merkle_proof

_libc_name = ctypes.util.find_library("c")
_libc = ctypes.CDLL(_libc_name) if _libc_name else None
MADV_NORMAL = 0
MADV_RANDOM = 1
MADV_SEQUENTIAL = 2
MADV_WILLNEED = 3
MADV_DONTNEED = 4


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolves 'auto' to Apple Metal GPU ('mps') when available in Terminal, else 'cpu'."""
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[Warning] Metal MPS device not accessible in headless mode; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def madvise_range(mm_obj: mmap.mmap, offset: int, length: int, advice: int = MADV_WILLNEED):
    """Issues posix_madvise on a 16 KB page-aligned slice of an mmap object."""
    if _libc is None:
        return
    try:
        page_off = offset & ~(16384 - 1)
        page_len = ((offset + length - page_off) + 16383) & ~(16384 - 1)
        buf = (ctypes.c_char * 1).from_buffer(mm_obj, page_off)
        addr = ctypes.addressof(buf)
        _libc.posix_madvise(ctypes.c_void_p(addr), ctypes.c_size_t(page_len), ctypes.c_int(advice))
    except Exception:
        pass


def get_process_rss_mb() -> float:
    """Returns peak Resident Set Size (RSS) in MiB (macOS ru_maxrss is in bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 * 1024)


class TokenEmbeddingTrieStore:
    """
    Rule 2: Token-Level Embedding Lookup from SSD via B+ Tree + pread.
    Loads only 4 KB per active Token_ID (in F16) instead of keeping 196.50 MiB in RAM/VRAM.
    """

    def __init__(
        self,
        store_dir: Path,
        conn: sqlite3.Connection,
        device: torch.device,
        hidden_size: int = 2048,
        verify_hashes: bool = False,
    ):
        self.store_dir = Path(store_dir)
        self.conn = conn
        self.device = device
        self.hidden_size = hidden_size
        self.row_bytes = hidden_size * 2  # F16 = 4096 bytes
        self.verify_hashes = verify_hashes
        self.fd = os.open(str(self.store_dir / "embed_tokens.bin"), os.O_RDONLY)

        self.token_cache: Dict[int, torch.Tensor] = {}
        self.disk_reads = 0
        self.cache_hits = 0
        self.bytes_read = 0
        self.merkle_verifications = 0

    def embed(self, token_ids: List[int]) -> torch.Tensor:
        """Fetches [len(token_ids), hidden_size] float16 tensor on `self.device`."""
        rows: List[torch.Tensor] = []
        for tid in token_ids:
            tid_int = int(tid)
            if tid_int in self.token_cache:
                self.cache_hits += 1
                rows.append(self.token_cache[tid_int])
                continue

            offset = tid_int * self.row_bytes
            raw = os.pread(self.fd, self.row_bytes, offset)
            self.disk_reads += 1
            self.bytes_read += len(raw)

            if self.verify_hashes:
                row_db = self.conn.execute(
                    "SELECT sha256_hex FROM embedding_trie_leaves WHERE token_id = ?",
                    (tid_int,),
                ).fetchone()
                if row_db:
                    assert sha256_bytes(raw).hex() == row_db[0], f"Embedding Merkle leaf hash mismatch for token {tid_int}!"
                    self.merkle_verifications += 1

            arr = np.frombuffer(raw, dtype=np.float16).copy()
            t_vec = torch.from_numpy(arr).to(self.device)
            self.token_cache[tid_int] = t_vec
            rows.append(t_vec)

        return torch.stack(rows, dim=0)

    def close(self):
        os.close(self.fd)


class HotExpertLFULRUCache:
    """
    Section 3.B.5: Hybrid LFU/LRU Cache for Hot Experts (resident on Metal `mps` or `cpu`).
    Keyed by content-addressable `sha256_hex` so deduplicated experts share a single cache slot.
    """

    def __init__(self, max_experts: int = 205):
        self.max_experts = max_experts
        self.lock = threading.Lock()
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.freq: Dict[str, int] = collections.Counter()
        self.last_used: Dict[str, int] = {}
        self.step_clock = 0
        self.hits = 0
        self.misses = 0
        self.prefetch_hits = 0
        self.prefetched_keys: Set[str] = set()

    def get(self, sha256_hex: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        with self.lock:
            self.step_clock += 1
            self.freq[sha256_hex] += 1
            if sha256_hex in self.cache:
                self.hits += 1
                if sha256_hex in self.prefetched_keys:
                    self.prefetch_hits += 1
                    self.prefetched_keys.discard(sha256_hex)
                self.last_used[sha256_hex] = self.step_clock
                return self.cache[sha256_hex]
            self.misses += 1
            return None

    def should_admit(self, sha256_hex: str) -> bool:
        if self.max_experts <= 0:
            return False
        with self.lock:
            if len(self.cache) < self.max_experts:
                return True
            cand_freq = self.freq.get(sha256_hex, 1)
            min_freq = min(self.freq.get(k, 1) for k in self.cache.keys())
            return cand_freq > min_freq

    def put(self, sha256_hex: str, tensors: Tuple[torch.Tensor, torch.Tensor], is_prefetch: bool = False):
        if self.max_experts <= 0:
            return
        with self.lock:
            self.step_clock += 1
            if sha256_hex in self.cache:
                self.last_used[sha256_hex] = self.step_clock
                return

            if len(self.cache) >= self.max_experts:
                victim = min(
                    self.cache.keys(),
                    key=lambda k: (self.freq.get(k, 1), self.last_used.get(k, 0)),
                )
                if is_prefetch and self.freq.get(sha256_hex, 0) < self.freq.get(victim, 1):
                    return
                del self.cache[victim]
                self.last_used.pop(victim, None)
                self.prefetched_keys.discard(victim)

            self.cache[sha256_hex] = tensors
            if is_prefetch and self.freq.get(sha256_hex, 0) == 0:
                self.freq[sha256_hex] = 1
            self.last_used[sha256_hex] = self.step_clock
            if is_prefetch:
                self.prefetched_keys.add(sha256_hex)

    def memory_mb(self) -> float:
        with self.lock:
            return len(self.cache) * 12.0


class LSHLookaheadPredictor:
    """
    Section 3.C.1: Locality-Sensitive Hashing (SimHash Random Hyperplane LSH) +
    Cross-Layer Router Lookahead on target device (`mps` or `cpu`).
    """

    def __init__(
        self,
        device: torch.device,
        hidden_size: int = 2048,
        num_layers: int = 16,
        n_bits: int = 16,
        seed: int = 42,
    ):
        self.device = device
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.hyperplanes = torch.randn(hidden_size, n_bits, generator=gen, dtype=torch.float16).to(device)
        self.powers = (1 << torch.arange(n_bits, dtype=torch.int64, device=device))
        self.lsh_tables: List[Dict[int, collections.Counter]] = [
            collections.defaultdict(collections.Counter) for _ in range(num_layers)
        ]
        self.lookahead_predictions = 0
        self.lookahead_correct_experts = 0

    def compute_simhash(self, h_vec: torch.Tensor) -> int:
        proj = (h_vec.view(1, -1) @ self.hyperplanes).squeeze(0) > 0
        return int((proj.to(torch.int64) * self.powers).sum().item())

    def predict_next_layer_experts(
        self,
        next_layer_idx: int,
        post_attn_hidden: torch.Tensor,
        next_layer_norm_weight: torch.Tensor,
        next_layer_router_weight: torch.Tensor,
        top_k: int = 6,
        rms_eps: float = 1e-5,
    ) -> List[int]:
        h_last = post_attn_hidden[-1]
        h_f32 = h_last.to(torch.float32)
        var = h_f32.pow(2).mean(-1, keepdim=True)
        h_normed = (h_f32 * torch.rsqrt(var + rms_eps)).to(torch.float16) * next_layer_norm_weight
        logits = F.linear(h_normed.unsqueeze(0), next_layer_router_weight).squeeze(0)
        proxy_topk = torch.topk(logits, k=min(top_k, 6)).indices.tolist()

        bucket = self.compute_simhash(h_last)
        bucket_counter = self.lsh_tables[next_layer_idx].get(bucket)
        if bucket_counter:
            lsh_top = [e for e, _ in bucket_counter.most_common(4)]
            return list(dict.fromkeys(proxy_topk + lsh_top))[:top_k]
        return proxy_topk

    def record_actual_routing(
        self,
        layer_idx: int,
        h_vec: torch.Tensor,
        actual_experts: List[int],
        predicted_experts: Optional[List[int]] = None,
    ):
        bucket = self.compute_simhash(h_vec[-1])
        for exp_id in actual_experts:
            self.lsh_tables[layer_idx][bucket][int(exp_id)] += 1
        if predicted_experts:
            self.lookahead_predictions += len(predicted_experts)
            self.lookahead_correct_experts += len(set(predicted_experts).intersection(set(actual_experts)))


class MerklePrefixKVCache:
    """
    Section 3.C.2: Merkle Prefix KV-Cache.
    Hashes sequences of 16 prompt tokens in a cryptographic Merkle chain:
        Hash_N = SHA-256(Hash_{N-1} || Tokens_Block_N)
    """

    BLOCK_SIZE = 16

    def __init__(self, store_dir: Path, conn: sqlite3.Connection, device: torch.device, num_layers: int = 16):
        self.store_dir = Path(store_dir)
        self.kv_dir = self.store_dir / "kv_cache_blocks"
        self.kv_dir.mkdir(parents=True, exist_ok=True)
        self.conn = conn
        self.device = device
        self.num_layers = num_layers
        self.blocks_loaded_from_disk = 0
        self.blocks_saved_to_disk = 0
        self.tokens_skipped = 0

    @staticmethod
    def chain_hash(prev_hash_hex: str, token_block: List[int]) -> str:
        h = hashlib.sha256()
        h.update(bytes.fromhex(prev_hash_hex))
        h.update(struct.pack(f"<{len(token_block)}I", *token_block))
        return h.digest().hex()

    def match_longest_prefix(self, token_ids: List[int]) -> Tuple[int, List[Tuple[torch.Tensor, torch.Tensor]], str]:
        max_cacheable_tokens = max(0, len(token_ids) - 1)
        num_full_blocks = max_cacheable_tokens // self.BLOCK_SIZE
        if num_full_blocks == 0:
            return 0, [], "00" * 32

        prev_hash = "00" * 32
        matched_blocks: List[Tuple[str, Path]] = []

        for b_idx in range(num_full_blocks):
            block_tokens = token_ids[b_idx * self.BLOCK_SIZE : (b_idx + 1) * self.BLOCK_SIZE]
            curr_hash = self.chain_hash(prev_hash, block_tokens)
            row = self.conn.execute(
                "SELECT kv_file_path FROM kv_prefix_cache WHERE prefix_hash_hex = ?",
                (curr_hash,),
            ).fetchone()
            if row and Path(row[0]).exists():
                matched_blocks.append((curr_hash, Path(row[0])))
                prev_hash = curr_hash
            else:
                break

        if not matched_blocks:
            return 0, [], "00" * 32

        per_layer_k: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        per_layer_v: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        for _, kv_path in matched_blocks:
            block_data = torch.load(kv_path, map_location="cpu", weights_only=True)
            for l_idx in range(self.num_layers):
                per_layer_k[l_idx].append(block_data[l_idx][0].to(self.device))
                per_layer_v[l_idx].append(block_data[l_idx][1].to(self.device))
            self.blocks_loaded_from_disk += 1

        assembled_kv = [
            (torch.cat(per_layer_k[l_idx], dim=2), torch.cat(per_layer_v[l_idx], dim=2))
            for l_idx in range(self.num_layers)
        ]
        matched_tokens = len(matched_blocks) * self.BLOCK_SIZE
        self.tokens_skipped += matched_tokens
        return matched_tokens, assembled_kv, prev_hash

    def save_prefix_blocks(self, token_ids: List[int], full_kv: List[Tuple[torch.Tensor, torch.Tensor]]):
        max_cacheable_tokens = max(0, len(token_ids) - 1)
        num_full_blocks = max_cacheable_tokens // self.BLOCK_SIZE
        prev_hash = "00" * 32

        for b_idx in range(num_full_blocks):
            start = b_idx * self.BLOCK_SIZE
            end = start + self.BLOCK_SIZE
            block_tokens = token_ids[start:end]
            curr_hash = self.chain_hash(prev_hash, block_tokens)

            kv_path = self.kv_dir / f"{curr_hash[:32]}.pt"
            if not kv_path.exists():
                block_kv = [
                    (
                        full_kv[l_idx][0][:, :, start:end, :].detach().to("cpu").contiguous().clone(),
                        full_kv[l_idx][1][:, :, start:end, :].detach().to("cpu").contiguous().clone(),
                    )
                    for l_idx in range(self.num_layers)
                ]
                torch.save(block_kv, kv_path)
                self.conn.execute(
                    "INSERT OR REPLACE INTO kv_prefix_cache VALUES (?, ?, ?, ?, ?, ?)",
                    (curr_hash, prev_hash, json.dumps(block_tokens), end, str(kv_path), time.time()),
                )
                self.conn.commit()
                self.blocks_saved_to_disk += 1

            prev_hash = curr_hash


class MerkleMoEEngine:
    """
    Complete Split-Memory Merkle-MoE Inference Engine with Apple Metal (`mps`) Unified Memory Binding
    and Copy-on-Write (CoW) Continual Learning Overlay.
    """

    def __init__(
        self,
        store_dir: Path,
        expert_cache_size: int = 205,
        enable_lookahead: bool = True,
        enable_merkle_kv: bool = True,
        verify_merkle_on_load: bool = False,
        evict_cold_mmap_pages: bool = False,
        device: str = "auto",
        use_cow_commits: bool = True,
    ):
        self.store_dir = Path(store_dir)
        self.device = resolve_device(device)
        self.is_mps = self.device.type == "mps"
        self.verify_merkle_on_load = verify_merkle_on_load
        self.evict_cold_mmap_pages = evict_cold_mmap_pages
        self.enable_lookahead = enable_lookahead
        self.enable_merkle_kv = enable_merkle_kv
        self.use_cow_commits = use_cow_commits
        self.active_merkle_root: str = ""
        self.cow_leaves_loaded: int = 0

        with open(self.store_dir / "config.json", "r") as f:
            self.config = json.load(f)
        with open(self.store_dir / "merkle_manifest.json", "r") as f:
            self.manifest = json.load(f)
        self.active_merkle_root = self.manifest["model_merkle_root"]

        self.num_layers: int = self.config["num_hidden_layers"]        # 16
        self.num_experts: int = self.config["num_experts"]             # 64
        self.top_k: int = self.config["num_experts_per_tok"]           # 8
        self.norm_topk_prob: bool = self.config["norm_topk_prob"]      # False
        self.hidden_size: int = self.config["hidden_size"]             # 2048
        self.intermediate_size: int = self.config["intermediate_size"] # 1024
        self.num_heads: int = self.config["num_attention_heads"]       # 16
        self.num_kv_heads: int = self.config["num_key_value_heads"]    # 16
        self.head_dim: int = self.hidden_size // self.num_heads        # 128
        self.rms_eps: float = self.config["rms_norm_eps"]              # 1e-5
        self.rope_theta: float = self.config["rope_theta"]             # 10000.0
        self.vocab_size: int = self.config["vocab_size"]               # 50304

        self.tokenizer = Tokenizer.from_file(str(self.store_dir / "tokenizer.json"))
        self.conn = sqlite3.connect(str(self.store_dir / "merkle_index.db"), check_same_thread=False)

        # 1. Token-Level Embedding Prefix Trie (On SSD, active tokens transferred to self.device)
        self.embed_store = TokenEmbeddingTrieStore(
            self.store_dir,
            self.conn,
            device=self.device,
            hidden_size=self.hidden_size,
            verify_hashes=verify_merkle_on_load,
        )

        # 2. Pinned Shared Core on self.device (516.25 MiB in Metal GPU / Unified Memory)
        self.shared_core: Dict[str, torch.Tensor] = {}
        self._load_pinned_shared_core()

        # 3. Quantized INT8 Output Head on self.device (98.35 MiB in Metal GPU / Unified Memory)
        self.lm_head_int8: torch.Tensor = None
        self.lm_head_scales: torch.Tensor = None
        self.lm_head_scales_t: torch.Tensor = None
        self._load_quantized_lm_head()

        # 4. Expert Leaves Tree on SSD (12.00 GiB CAS store) + Metal Hot-Expert Cache
        self.cas_fd = os.open(str(self.store_dir / "cas_experts.bin"), os.O_RDONLY)
        self.cas_file = open(self.store_dir / "cas_experts.bin", "rb")
        self.cas_mmap = mmap.mmap(self.cas_file.fileno(), 0, access=mmap.ACCESS_READ)
        madvise_range(self.cas_mmap, 0, self.manifest["expert_cas_total_bytes"], MADV_RANDOM)

        self.expert_cache = HotExpertLFULRUCache(max_experts=expert_cache_size)
        self.expert_index: Dict[Tuple[int, int], Tuple[str, int, int]] = {}
        self.expert_mmap_views: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._load_expert_bptree_index()

        # Pre-allocate a single reusable 12.00 MiB Metal staging buffer for Cold Experts
        if self.is_mps:
            self._mps_staging_gu = torch.empty(
                (2 * self.intermediate_size, self.hidden_size), dtype=torch.float16, device=self.device
            )
            self._mps_staging_d = torch.empty(
                (self.hidden_size, self.intermediate_size), dtype=torch.float16, device=self.device
            )
        else:
            self._mps_staging_gu = None
            self._mps_staging_d = None

        self.lsh_predictor = LSHLookaheadPredictor(
            device=self.device, hidden_size=self.hidden_size, num_layers=self.num_layers
        )
        self.kv_prefix_cache = MerklePrefixKVCache(
            self.store_dir, self.conn, device=self.device, num_layers=self.num_layers
        )
        self.prefetch_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_inv_freq = inv_freq.to(self.device)

        self.ssd_expert_loads = 0
        self.ssd_expert_bytes_read = 0
        self.merkle_proofs_verified = 0

    def reset_telemetry(self):
        self.ssd_expert_loads = 0
        self.ssd_expert_bytes_read = 0
        self.expert_cache.hits = 0
        self.expert_cache.misses = 0
        self.expert_cache.prefetch_hits = 0
        self.lsh_predictor.lookahead_predictions = 0
        self.lsh_predictor.lookahead_correct_experts = 0

    def _load_pinned_shared_core(self):
        rows = self.conn.execute(
            "SELECT tensor_name, shape_json, byte_offset, byte_length FROM shared_core_tensors"
        ).fetchall()
        with open(self.store_dir / "shared_core.bin", "rb") as f:
            raw_core = f.read()
        for t_name, shape_json, off, length in rows:
            shape = json.loads(shape_json)
            np_arr = np.frombuffer(raw_core, dtype=np.float16, count=length // 2, offset=off).reshape(shape).copy()
            self.shared_core[t_name] = torch.from_numpy(np_arr).to(self.device)

        for l_idx in range(self.num_layers):
            p = f"model.layers.{l_idx}.self_attn"
            q_w = self.shared_core.pop(f"{p}.q_proj.weight")
            k_w = self.shared_core.pop(f"{p}.k_proj.weight")
            v_w = self.shared_core.pop(f"{p}.v_proj.weight")
            self.shared_core[f"{p}.qkv_proj.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

    def _load_quantized_lm_head(self):
        w_nbytes = self.vocab_size * self.hidden_size
        s_nbytes = self.vocab_size * 2
        with open(self.store_dir / "lm_head_int8.bin", "rb") as f:
            w_buf = f.read(w_nbytes)
            s_buf = f.read(s_nbytes)
        w_np = np.frombuffer(w_buf, dtype=np.int8).reshape(self.vocab_size, self.hidden_size).copy()
        s_np = np.frombuffer(s_buf, dtype=np.float16).reshape(self.vocab_size, 1).copy()
        self.lm_head_int8 = torch.from_numpy(w_np).to(self.device)
        self.lm_head_scales = torch.from_numpy(s_np).to(self.device)
        self.lm_head_scales_t = self.lm_head_scales.view(1, -1)

    def _load_expert_bptree_index(self):
        rows = self.conn.execute(
            """
            SELECT l.layer_idx, l.expert_idx, l.sha256_hex, c.byte_offset, c.byte_length
            FROM expert_tree_leaves l
            JOIN expert_cas_blobs c ON l.sha256_hex = c.sha256_hex
            """
        ).fetchall()
        gu_elems = 2 * self.intermediate_size * self.hidden_size
        d_elems = self.hidden_size * self.intermediate_size
        gu_bytes = gu_elems * 2

        for l_idx, e_idx, sha_hex, off, length in rows:
            key = (int(l_idx), int(e_idx))
            byte_off = int(off)
            self.expert_index[key] = (sha_hex, byte_off, int(length))
            gu_t = torch.frombuffer(
                self.cas_mmap, dtype=torch.float16, count=gu_elems, offset=byte_off
            ).view(2 * self.intermediate_size, self.hidden_size)
            d_t = torch.frombuffer(
                self.cas_mmap, dtype=torch.float16, count=d_elems, offset=byte_off + gu_bytes
            ).view(self.hidden_size, self.intermediate_size)
            self.expert_mmap_views[key] = (gu_t, d_t)

        # Overlay Copy-on-Write (CoW) Continual Learning Delta Leaves in chronological commit order!
        delta_path = self.store_dir / "cas_cow_deltas.bin"
        if self.use_cow_commits and delta_path.exists():
            try:
                commit_rows = self.conn.execute(
                    "SELECT commit_id, new_model_root_hex, updated_leaves_json FROM merkle_commits ORDER BY created_at ASC"
                ).fetchall()
                if commit_rows:
                    raw_deltas = delta_path.read_bytes()
                    modified_keys: Set[Tuple[int, int]] = set()
                    for _, new_root_hex, leaves_json in commit_rows:
                        updated_leaves = json.loads(leaves_json)
                        for item in updated_leaves:
                            l_i = int(item["layer"])
                            e_i = int(item["expert"])
                            d_off = int(item["delta_offset"])
                            d_len = int(item["nbytes"])
                            if d_off + d_len > len(raw_deltas):
                                continue
                            leaf_slice = raw_deltas[d_off : d_off + d_len]
                            full_sha_hex = sha256_bytes(leaf_slice).hex()

                            gu_np = np.frombuffer(leaf_slice, dtype=np.float16, count=gu_elems, offset=0).reshape(
                                2 * self.intermediate_size, self.hidden_size
                            ).copy()
                            d_np = np.frombuffer(leaf_slice, dtype=np.float16, count=d_elems, offset=gu_bytes).reshape(
                                self.hidden_size, self.intermediate_size
                            ).copy()
                            gu_cow = torch.from_numpy(gu_np)
                            d_cow = torch.from_numpy(d_np)

                            self.expert_index[(l_i, e_i)] = (full_sha_hex, d_off, d_len)
                            self.expert_mmap_views[(l_i, e_i)] = (gu_cow, d_cow)
                            self.expert_cache.put(
                                full_sha_hex,
                                (gu_cow.to(self.device), d_cow.to(self.device)),
                                is_prefetch=False,
                            )
                            modified_keys.add((l_i, e_i))
                        self.active_merkle_root = new_root_hex
                    self.cow_leaves_loaded = len(modified_keys)
            except Exception:
                pass

    def prefetch_expert_async(self, layer_idx: int, expert_idx: int):
        """Triggers OS-level async NVMe read (MADV_WILLNEED) on the 12 MiB page-aligned Merkle leaf."""
        sha_hex, byte_off, byte_len = self.expert_index[(layer_idx, expert_idx)]
        if sha_hex in self.expert_cache.cache:
            return
        madvise_range(self.cas_mmap, byte_off, byte_len, MADV_WILLNEED)

    def fetch_expert_leaf(
        self, layer_idx: int, expert_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, int, int]:
        """
        Fetches (gate_up_proj [2048, 2048], down_proj [2048, 1024], was_mmap_zero_copy, byte_off, byte_len)
        bound to `self.device` (`mps` or `cpu`).
        """
        key = (layer_idx, expert_idx)
        sha_hex, byte_off, byte_len = self.expert_index[key]
        cached = self.expert_cache.get(sha_hex)
        if cached is not None:
            return cached[0], cached[1], False, byte_off, byte_len

        self.ssd_expert_loads += 1
        self.ssd_expert_bytes_read += byte_len

        if self.verify_merkle_on_load and self.merkle_proofs_verified < 16:
            row = self.conn.execute(
                """
                SELECT l.merkle_proof_json, r.merkle_root_hex
                FROM expert_tree_leaves l
                JOIN expert_layer_roots r ON l.layer_idx = r.layer_idx
                WHERE l.layer_idx = ? AND l.expert_idx = ?
                """,
                (layer_idx, expert_idx),
            ).fetchone()
            if row:
                mv_full = memoryview(self.cas_mmap)[byte_off : byte_off + byte_len]
                leaf_h = sha256_bytes(mv_full).hex()
                del mv_full
                assert verify_merkle_proof(leaf_h, json.loads(row[0]), row[1]), "Expert Merkle proof failed!"
                self.merkle_proofs_verified += 1

        gu_t, d_t = self.expert_mmap_views[key]

        if self.is_mps:
            if self.expert_cache.should_admit(sha_hex):
                promoted = (gu_t.to(self.device, non_blocking=True), d_t.to(self.device, non_blocking=True))
                self.expert_cache.put(sha_hex, promoted, is_prefetch=False)
                return promoted[0], promoted[1], False, byte_off, byte_len
            else:
                self._mps_staging_gu.copy_(gu_t, non_blocking=True)
                self._mps_staging_d.copy_(d_t, non_blocking=True)
                return self._mps_staging_gu, self._mps_staging_d, True, byte_off, byte_len

        if self.expert_cache.should_admit(sha_hex):
            if not self.evict_cold_mmap_pages:
                madvise_range(self.cas_mmap, byte_off, byte_len, MADV_WILLNEED)
                self.expert_cache.put(sha_hex, (gu_t, d_t), is_prefetch=False)
                return gu_t, d_t, False, byte_off, byte_len
            else:
                promoted = (gu_t.clone(), d_t.clone())
                self.expert_cache.put(sha_hex, promoted, is_prefetch=False)
                return promoted[0], promoted[1], False, byte_off, byte_len

        return gu_t, d_t, True, byte_off, byte_len

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.to(torch.float32)
        var = x_f32.pow(2).mean(-1, keepdim=True)
        return weight * (x_f32 * torch.rsqrt(var + self.rms_eps)).to(x.dtype)

    def _compute_rope(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(position_ids.squeeze(0).to(torch.float32), self.register_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(torch.float16).unsqueeze(0).unsqueeze(0)
        sin = emb.sin().to(torch.float16).unsqueeze(0).unsqueeze(0)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _forward_attention(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        seq_len, _ = hidden_states.shape
        p = f"model.layers.{layer_idx}.self_attn"

        qkv = F.linear(hidden_states, self.shared_core[f"{p}.qkv_proj.weight"])
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._rms_norm(q, self.shared_core[f"{p}.q_norm.weight"])
        k = self._rms_norm(k, self.shared_core[f"{p}.k_norm.weight"])

        q = q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self._apply_rope(q, k, cos, sin)

        if kv_cache is not None:
            if kv_cache[layer_idx] is not None:
                prev_k, prev_v = kv_cache[layer_idx]
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            kv_cache[layer_idx] = (k, v)

        is_causal = seq_len > 1 and (kv_cache is None or kv_cache[layer_idx][0].shape[2] == seq_len)
        if seq_len > 1 and not is_causal:
            total_k_len = k.shape[2]
            past_len = total_k_len - seq_len
            mask = torch.ones((seq_len, total_k_len), dtype=torch.bool, device=self.device)
            mask = torch.tril(mask, diagonal=past_len)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        attn_out = attn_out.transpose(1, 2).contiguous().view(seq_len, self.hidden_size)
        return F.linear(attn_out, self.shared_core[f"{p}.o_proj.weight"])

    def _forward_sparse_moe(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        predicted_for_this_layer: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        seq_len, _ = hidden_states.shape
        router_w = self.shared_core[f"model.layers.{layer_idx}.mlp.gate.weight"]

        router_logits = F.linear(hidden_states, router_w)
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        top_vals, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)
        top_scores = top_vals.to(hidden_states.dtype)

        last_tok_experts = top_indices[-1].tolist()
        self.lsh_predictor.record_actual_routing(
            layer_idx, hidden_states, last_tok_experts, predicted_for_this_layer
        )

        if seq_len == 1:
            final_hidden = torch.zeros_like(hidden_states)
            scores_1d = top_scores[0].tolist()
            for k_pos, exp_id in enumerate(last_tok_experts):
                gate_up_w, down_w, is_mmap_view, byte_off, byte_len = self.fetch_expert_leaf(layer_idx, exp_id)
                gu = F.linear(hidden_states, gate_up_w)
                gate_out, up_out = gu.chunk(2, dim=-1)
                mlp_out = F.linear(F.silu(gate_out) * up_out, down_w)
                final_hidden.add_(mlp_out, alpha=float(scores_1d[k_pos]))
                if is_mmap_view and self.evict_cold_mmap_pages:
                    madvise_range(self.cas_mmap, byte_off, byte_len, MADV_DONTNEED)
            return final_hidden, last_tok_experts

        top_indices_cpu = top_indices.to("cpu")
        expert_to_tokens: Dict[int, List[Tuple[int, torch.Tensor]]] = collections.defaultdict(list)
        for tok_pos in range(seq_len):
            for k_pos in range(self.top_k):
                exp_id = int(top_indices_cpu[tok_pos, k_pos].item())
                weight = top_scores[tok_pos, k_pos]
                expert_to_tokens[exp_id].append((tok_pos, weight))

        final_hidden = torch.zeros_like(hidden_states)

        for exp_id, tok_entries in expert_to_tokens.items():
            gate_up_w, down_w, is_mmap_view, byte_off, byte_len = self.fetch_expert_leaf(layer_idx, exp_id)
            if len(tok_entries) == 1:
                tok_pos, weight = tok_entries[0]
                x = hidden_states[tok_pos : tok_pos + 1]
                gu = F.linear(x, gate_up_w)
                gate_out, up_out = gu.chunk(2, dim=-1)
                mlp_out = F.linear(F.silu(gate_out) * up_out, down_w)
                final_hidden[tok_pos] += (mlp_out.squeeze(0) * weight)
            else:
                tok_positions = [te[0] for te in tok_entries]
                weights = torch.stack([te[1] for te in tok_entries]).unsqueeze(-1)
                idx_tensor = torch.tensor(tok_positions, dtype=torch.long, device=self.device)
                x_batch = hidden_states.index_select(0, idx_tensor)
                gu = F.linear(x_batch, gate_up_w)
                gate_out, up_out = gu.chunk(2, dim=-1)
                mlp_out = F.linear(F.silu(gate_out) * up_out, down_w) * weights
                final_hidden.index_add_(0, idx_tensor, mlp_out)

            if is_mmap_view and self.evict_cold_mmap_pages:
                madvise_range(self.cas_mmap, byte_off, byte_len, MADV_DONTNEED)

        return final_hidden, last_tok_experts

    def _compute_quantized_logits(self, norm_hidden: torch.Tensor, all_positions: bool = False) -> torch.Tensor:
        x = norm_hidden if all_positions else norm_hidden[-1:]
        chunk_size = 16384
        logits_chunks = []
        for start in range(0, self.vocab_size, chunk_size):
            end = min(start + chunk_size, self.vocab_size)
            logits_chunks.append(F.linear(x, self.lm_head_int8[start:end].to(torch.float16)))
        return torch.cat(logits_chunks, dim=-1) * self.lm_head_scales_t

    @torch.inference_mode()
    def forward_tokens(
        self,
        token_ids: List[int],
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        start_pos: int = 0,
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        seq_len = len(token_ids)
        hidden_states = self.embed_store.embed(token_ids)
        position_ids = torch.arange(
            start_pos, start_pos + seq_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        cos, sin = self._compute_rope(position_ids)

        predicted_next_experts: Optional[List[int]] = None

        for layer_idx in range(self.num_layers):
            residual = hidden_states
            normed_in = self._rms_norm(
                hidden_states,
                self.shared_core[f"model.layers.{layer_idx}.input_layernorm.weight"],
            )
            attn_out = self._forward_attention(layer_idx, normed_in, cos, sin, kv_cache=kv_cache)
            hidden_states = residual + attn_out

            next_predictions: Optional[List[int]] = None
            if self.enable_lookahead and (layer_idx + 1) < self.num_layers:
                next_l = layer_idx + 1
                next_predictions = self.lsh_predictor.predict_next_layer_experts(
                    next_layer_idx=next_l,
                    post_attn_hidden=hidden_states,
                    next_layer_norm_weight=self.shared_core[
                        f"model.layers.{next_l}.post_attention_layernorm.weight"
                    ],
                    next_layer_router_weight=self.shared_core[f"model.layers.{next_l}.mlp.gate.weight"],
                    top_k=6,
                    rms_eps=self.rms_eps,
                )
                for pred_exp in next_predictions:
                    sha_hex, byte_off, byte_len = self.expert_index[(next_l, pred_exp)]
                    if sha_hex not in self.expert_cache.cache:
                        madvise_range(self.cas_mmap, byte_off, byte_len, MADV_WILLNEED)

            residual = hidden_states
            normed_post = self._rms_norm(
                hidden_states,
                self.shared_core[f"model.layers.{layer_idx}.post_attention_layernorm.weight"],
            )
            moe_out, _ = self._forward_sparse_moe(
                layer_idx, normed_post, predicted_for_this_layer=predicted_next_experts
            )
            hidden_states = residual + moe_out
            predicted_next_experts = next_predictions

        target_h = hidden_states if return_all_logits else hidden_states[-1:]
        normed_final = self._rms_norm(target_h, self.shared_core["model.norm.weight"])
        return self._compute_quantized_logits(normed_final, all_positions=return_all_logits)

    @torch.inference_mode()
    def verify_speculative_Sequence(
        self,
        context_token_ids: List[int],
        draft_token_ids: List[int],
    ) -> dict:
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers
        _ = self.forward_tokens(context_token_ids, kv_cache=kv_cache, start_pos=0)

        kv_snapshot = [(k.clone(), v.clone()) for k, v in kv_cache]
        k_len = len(draft_token_ids)

        t_spec0 = time.perf_counter()
        spec_logits = self.forward_tokens(
            draft_token_ids,
            kv_cache=kv_cache,
            start_pos=len(context_token_ids),
            return_all_logits=True,
        )
        t_spec_sec = time.perf_counter() - t_spec0
        spec_greedy = torch.argmax(spec_logits, dim=-1).tolist()

        kv_seq = [(k.clone(), v.clone()) for k, v in kv_snapshot]
        t_seq0 = time.perf_counter()
        seq_greedy = []
        for i, dt in enumerate(draft_token_ids):
            l_i = self.forward_tokens([dt], kv_cache=kv_seq, start_pos=len(context_token_ids) + i)
            seq_greedy.append(int(torch.argmax(l_i, dim=-1).item()))
        t_seq_sec = time.perf_counter() - t_seq0

        return {
            "draft_tokens_verified_in_one_pass": k_len,
            "single_pass_verification_sec": round(t_spec_sec, 4),
            "sequential_k_passes_sec": round(t_seq_sec, 4),
            "amortized_speedup_x": round(t_seq_sec / max(1e-6, t_spec_sec), 2),
            "per_token_amortized_ms": round(1000.0 * t_spec_sec / k_len, 2),
            "per_token_sequential_ms": round(1000.0 * t_seq_sec / k_len, 2),
            "exact_greedy_match": spec_greedy == seq_greedy,
        }

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        use_chat_template: bool = True,
    ) -> dict:
        if use_chat_template:
            formatted = f"<|endoftext|><|user|>\n{prompt}\n<|assistant|>\n"
        else:
            formatted = prompt

        input_ids: List[int] = self.tokenizer.encode(formatted, add_special_tokens=False).ids
        kv_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers

        t_start = time.perf_counter()
        matched_len = 0
        if self.enable_merkle_kv:
            matched_len, loaded_kv, _ = self.kv_prefix_cache.match_longest_prefix(input_ids)
            if matched_len > 0:
                for l_idx in range(self.num_layers):
                    kv_cache[l_idx] = loaded_kv[l_idx]

        rem_tokens = input_ids[matched_len:]
        logits = self.forward_tokens(rem_tokens, kv_cache=kv_cache, start_pos=matched_len)

        if self.enable_merkle_kv and len(input_ids) >= MerklePrefixKVCache.BLOCK_SIZE + 1:
            self.kv_prefix_cache.save_prefix_blocks(input_ids, kv_cache)

        ttft_sec = time.perf_counter() - t_start
        eos_id = self.config.get("eos_token_id", 50279)
        stop_ids = {eos_id, 0}

        next_token = int(torch.argmax(logits, dim=-1).item())
        generated_ids: List[int] = []
        if next_token not in stop_ids:
            generated_ids.append(next_token)
        curr_pos = len(input_ids)

        t_decode_start = time.perf_counter()

        for _ in range(max_new_tokens - 1):
            if next_token in stop_ids:
                break
            logits = self.forward_tokens([next_token], kv_cache=kv_cache, start_pos=curr_pos)
            next_token = int(torch.argmax(logits, dim=-1).item())
            curr_pos += 1
            if next_token in stop_ids:
                break
            generated_ids.append(next_token)

        decode_sec = max(1e-6, time.perf_counter() - t_decode_start)
        decode_tokens = max(1, len(generated_ids) - 1)
        output_text = self.tokenizer.decode(generated_ids)

        total_expert_reqs = self.expert_cache.hits + self.expert_cache.misses
        cache_hit_rate = (100.0 * self.expert_cache.hits / total_expert_reqs) if total_expert_reqs else 0.0
        lookahead_acc = (
            100.0 * self.lsh_predictor.lookahead_correct_experts / self.lsh_predictor.lookahead_predictions
            if self.lsh_predictor.lookahead_predictions
            else 0.0
        )

        pinned_ram_mb = (
            self.manifest["shared_core_bytes"] + self.manifest["lm_head_int8_bytes"]
        ) / (1024**2)
        hot_cache_mb = self.expert_cache.memory_mb()
        embed_ram_mb = (len(self.embed_store.token_cache) * self.embed_store.row_bytes) / (1024**2)
        staging_mb = 12.0 if self.is_mps else 0.0
        total_engine_weight_ram_mb = pinned_ram_mb + hot_cache_mb + embed_ram_mb + staging_mb
        full_model_mb = 13838323712 / (1024**2)

        return {
            "device": str(self.device),
            "active_merkle_root": self.active_merkle_root,
            "cow_leaves_loaded": self.cow_leaves_loaded,
            "prompt": prompt,
            "prompt_tokens": len(input_ids),
            "merkle_kv_matched_tokens": matched_len,
            "generated_tokens": len(generated_ids),
            "output_text": output_text,
            "ttft_sec": round(ttft_sec, 4),
            "decode_sec": round(decode_sec, 4),
            "decode_tps": round(decode_tokens / decode_sec, 2),
            "pinned_core_plus_head_mb": round(pinned_ram_mb, 2),
            "hot_expert_cache_mb": round(hot_cache_mb, 2),
            "active_embed_leaves_kb": round(embed_ram_mb * 1024, 2),
            "total_engine_weight_ram_mb": round(total_engine_weight_ram_mb, 2),
            "full_model_safetensors_mb": round(full_model_mb, 2),
            "ram_reduction_pct": round(100.0 * (1.0 - total_engine_weight_ram_mb / full_model_mb), 2),
            "expert_cache_hits": self.expert_cache.hits,
            "expert_cache_misses": self.expert_cache.misses,
            "expert_cache_hit_rate_pct": round(cache_hit_rate, 2),
            "lookahead_accuracy_pct": round(lookahead_acc, 2),
            "ssd_expert_loads": self.ssd_expert_loads,
            "ssd_expert_read_mb": round(self.ssd_expert_bytes_read / (1024**2), 2),
        }

    def close(self):
        self.prefetch_pool.shutdown(wait=True)
        self.embed_store.close()
        self.expert_mmap_views.clear()
        self.expert_cache.cache.clear()
        self.cas_mmap.close()
        self.cas_file.close()
        os.close(self.cas_fd)
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Run Low-RAM Merkle-MoE Inference Engine (Metal MPS + CPU)")
    parser.add_argument(
        "--store-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "olmoe_merkle_store"),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain in two concise sentences how a Merkle tree verifies data integrity.",
    )
    parser.add_argument("--max-tokens", type=int, default=36)
    parser.add_argument("--cache-experts", type=int, default=205)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--base-only", action="store_true", help="Ignore Copy-on-Write continual learning commits")
    args = parser.parse_args()

    engine = MerkleMoEEngine(
        Path(args.store_dir),
        expert_cache_size=args.cache_experts,
        enable_lookahead=True,
        enable_merkle_kv=True,
        verify_merkle_on_load=True,
        device=args.device,
        use_cow_commits=(not args.base_only),
    )
    res = engine.generate(args.prompt, max_new_tokens=args.max_tokens)
    print(json.dumps(res, indent=2))
    engine.close()


if __name__ == "__main__":
    main()
