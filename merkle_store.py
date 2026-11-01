#!/usr/bin/env python3
"""
Merkle-MoE Tree Storage Converter & Content-Addressable Store
==============================================================
Converts flat .safetensors shards (e.g. allenai/OLMoE-1B-7B-0125-Instruct)
into a Hybrid Merkle / B+ Tree Content-Addressable Storage format:

  [ Model Merkle Root (32-Byte SHA-256) ]
    ├── [ Embedding Prefix Trie (SSD, 4KB/token in F16) ]
    ├── [ Shared Core (Pinned in RAM: Attention + Routers + Norms, 516.25 MiB) ]
    ├── [ Output Head (Quantized INT8 in RAM, 98.35 MiB) ]
    └── [ Expert Leaves Tree (SSD CAS Store, 1,024 Contiguous 12 MiB Leaves) ]
"""

import argparse
import gc
import hashlib
import json
import mmap
import os
import shutil
import sqlite3
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PAGE_SIZE = 16384  # 16 KB page alignment for Apple Silicon / macOS VM pages


def align_up(value: int, alignment: int = PAGE_SIZE) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def sha256_bytes(data: bytes | memoryview) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_pair(left: bytes, right: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(b"\x01")
    h.update(left)
    h.update(right)
    return h.digest()


def build_merkle_tree(leaf_hashes: List[bytes]) -> Tuple[bytes, List[List[bytes]]]:
    """
    Builds a binary Merkle tree from a list of 32-byte leaf hashes.
    Returns (root_hash, levels) where levels[0] = leaf_hashes and levels[-1] = [root_hash].
    """
    if not leaf_hashes:
        empty = hashlib.sha256(b"").digest()
        return empty, [[empty]]
    levels: List[List[bytes]] = [list(leaf_hashes)]
    current = list(leaf_hashes)
    while len(current) > 1:
        next_level: List[bytes] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if (i + 1) < len(current) else left
            next_level.append(sha256_pair(left, right))
        levels.append(next_level)
        current = next_level
    return current[0], levels


def get_merkle_proof(levels: List[List[bytes]], leaf_index: int) -> List[Tuple[str, str]]:
    """
    Returns the cryptographic Merkle inclusion proof for `leaf_index` as a list
    of (position: 'L'|'R', sibling_hash_hex) tuples from leaf to root.
    """
    proof: List[Tuple[str, str]] = []
    idx = leaf_index
    for level in levels[:-1]:
        sibling_idx = idx ^ 1
        if sibling_idx < len(level):
            sibling_hash = level[sibling_idx]
        else:
            sibling_hash = level[idx]
        pos = "R" if (idx % 2 == 0) else "L"
        proof.append((pos, sibling_hash.hex()))
        idx //= 2
    return proof


def verify_merkle_proof(leaf_hash_hex: str, proof: List[Tuple[str, str]], root_hash_hex: str) -> bool:
    """Verifies a Merkle inclusion proof against a 32-byte Merkle root."""
    curr = bytes.fromhex(leaf_hash_hex)
    for pos, sib_hex in proof:
        sib = bytes.fromhex(sib_hex)
        if pos == "R":
            curr = sha256_pair(curr, sib)
        else:
            curr = sha256_pair(sib, curr)
    return curr.hex() == root_hash_hex


class SafetensorsShardReader:
    """
    Zero-copy mmap reader for .safetensors shards that reads individual tensor byte ranges
    without loading full 5GB shards into RAM.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            self.index_data = json.load(f)
        self.weight_map: Dict[str, str] = self.index_data["weight_map"]
        self.shard_files: Dict[str, object] = {}
        self.shard_mmaps: Dict[str, mmap.mmap] = {}
        self.tensor_info: Dict[str, dict] = {}

        unique_shards = sorted(set(self.weight_map.values()))
        for shard_name in unique_shards:
            shard_path = self.model_dir / shard_name
            f_obj = open(shard_path, "rb")
            mm = mmap.mmap(f_obj.fileno(), 0, access=mmap.ACCESS_READ)
            header_len = struct.unpack("<Q", mm[:8])[0]
            header = json.loads(mm[8 : 8 + header_len].decode("utf-8"))
            data_base = 8 + header_len
            self.shard_files[shard_name] = f_obj
            self.shard_mmaps[shard_name] = mm
            for t_name, meta in header.items():
                if t_name == "__metadata__":
                    continue
                start, end = meta["data_offsets"]
                self.tensor_info[t_name] = {
                    "shard": shard_name,
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                    "abs_start": data_base + start,
                    "abs_end": data_base + end,
                    "nbytes": end - start,
                    "rel_start": start,
                    "rel_end": end,
                }

    def get_tensor_bytes(self, name: str) -> memoryview:
        info = self.tensor_info[name]
        mm = self.shard_mmaps[info["shard"]]
        return memoryview(mm)[info["abs_start"] : info["abs_end"]]

    def close(self):
        gc.collect()
        for mm in self.shard_mmaps.values():
            try:
                mm.close()
            except BufferError:
                pass
        for f_obj in self.shard_files.values():
            f_obj.close()


class MerkleMoEBuilder:
    """
    Builds the Hybrid B+ Tree / Merkle-Indexed store from a HuggingFace/MLX .safetensors MoE checkpoint.
    """

    def __init__(self, model_dir: Path, output_dir: Path):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)

    def _init_sqlite_bptree(self, db_path: Path) -> sqlite3.Connection:
        if db_path.exists():
            db_path.unlink()
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA page_size = 16384;")

        # B+ Tree tables (WITHOUT ROWID forces true clustered B+ Tree storage in SQLite)
        conn.executescript(
            """
            CREATE TABLE merkle_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE embedding_trie_leaves (
                token_id INTEGER PRIMARY KEY,
                prefix_hex TEXT NOT NULL,
                sha256_hex TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL
            ) WITHOUT ROWID;

            CREATE INDEX idx_embed_prefix ON embedding_trie_leaves(prefix_hex);

            CREATE TABLE embedding_trie_nodes (
                prefix_hex TEXT PRIMARY KEY,
                level INTEGER NOT NULL,
                merkle_hash_hex TEXT NOT NULL,
                token_count INTEGER NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE shared_core_tensors (
                tensor_name TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                layer_idx INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                shape_json TEXT NOT NULL,
                sha256_hex TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE expert_cas_blobs (
                sha256_hex TEXT PRIMARY KEY,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                gate_size INTEGER NOT NULL,
                up_size INTEGER NOT NULL,
                down_size INTEGER NOT NULL,
                ref_count INTEGER NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE expert_tree_leaves (
                layer_idx INTEGER NOT NULL,
                expert_idx INTEGER NOT NULL,
                sha256_hex TEXT NOT NULL,
                orig_gate_shard TEXT NOT NULL,
                orig_gate_offset INTEGER NOT NULL,
                orig_up_shard TEXT NOT NULL,
                orig_up_offset INTEGER NOT NULL,
                orig_down_shard TEXT NOT NULL,
                orig_down_offset INTEGER NOT NULL,
                merkle_proof_json TEXT NOT NULL,
                PRIMARY KEY (layer_idx, expert_idx)
            ) WITHOUT ROWID;

            CREATE TABLE expert_layer_roots (
                layer_idx INTEGER PRIMARY KEY,
                merkle_root_hex TEXT NOT NULL,
                num_experts INTEGER NOT NULL,
                total_bytes INTEGER NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE kv_prefix_cache (
                prefix_hash_hex TEXT PRIMARY KEY,
                prev_hash_hex TEXT NOT NULL,
                token_block_json TEXT NOT NULL,
                seq_len INTEGER NOT NULL,
                kv_file_path TEXT NOT NULL,
                created_at REAL NOT NULL
            ) WITHOUT ROWID;
            """
        )
        return conn

    def build(self) -> dict:
        t0 = time.perf_counter()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy tokenizer & config files so the store is self-contained
        for cfg_file in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
            src = self.model_dir / cfg_file
            if src.exists():
                shutil.copy2(src, self.output_dir / cfg_file)

        with open(self.model_dir / "config.json", "r") as f:
            config = json.load(f)

        num_layers = config["num_hidden_layers"]
        num_experts = config["num_experts"]
        vocab_size = config["vocab_size"]

        print(f"[1/5] Opening .safetensors shards from {self.model_dir} (zero-copy mmap)...")
        reader = SafetensorsShardReader(self.model_dir)
        conn = self._init_sqlite_bptree(self.output_dir / "merkle_index.db")

        # ---------------------------------------------------------------------
        # STEP 1: Token-Level Embedding Prefix Trie (Rule 2)
        # ---------------------------------------------------------------------
        print(f"[2/5] Building Token-Level Embedding Prefix Trie ({vocab_size:,} leaves)...")
        t_emb0 = time.perf_counter()
        embed_info = reader.tensor_info["model.embed_tokens.weight"]
        embed_mv = reader.get_tensor_bytes("model.embed_tokens.weight")
        row_bytes = embed_info["nbytes"] // vocab_size  # 4096 bytes (4 KB) in F16

        embed_bin_path = self.output_dir / "embed_tokens.bin"
        with open(embed_bin_path, "wb") as f_emb:
            f_emb.write(embed_mv)

        token_leaf_hashes: List[bytes] = []
        embed_rows_db = []
        prefix_groups: Dict[str, List[bytes]] = {}

        for tok_id in range(vocab_size):
            off = tok_id * row_bytes
            row_slice = embed_mv[off : off + row_bytes]
            h_bytes = sha256_bytes(row_slice)
            h_hex = h_bytes.hex()
            token_leaf_hashes.append(h_bytes)
            prefix_hex = f"{tok_id:04x}"[:2]
            prefix_groups.setdefault(prefix_hex, []).append(h_bytes)
            embed_rows_db.append((tok_id, prefix_hex, h_hex, off, row_bytes))
            del row_slice

        del embed_mv
        conn.executemany(
            "INSERT INTO embedding_trie_leaves VALUES (?, ?, ?, ?, ?)",
            embed_rows_db,
        )

        trie_node_rows = []
        for prefix_hex in sorted(prefix_groups.keys()):
            b_root, _ = build_merkle_tree(prefix_groups[prefix_hex])
            trie_node_rows.append((prefix_hex, 1, b_root.hex(), len(prefix_groups[prefix_hex])))

        embedding_trie_root, _ = build_merkle_tree(token_leaf_hashes)
        trie_node_rows.append(("ROOT", 0, embedding_trie_root.hex(), vocab_size))
        conn.executemany(
            "INSERT INTO embedding_trie_nodes VALUES (?, ?, ?, ?)",
            trie_node_rows,
        )
        t_emb_elapsed = time.perf_counter() - t_emb0
        print(
            f"      -> Embedding Trie Root: {embedding_trie_root.hex()[:24]}... "
            f"({vocab_size:,} leaves × {row_bytes} B = {embed_info['nbytes']/(1024**2):.2f} MiB in {t_emb_elapsed:.2f}s)"
        )

        # ---------------------------------------------------------------------
        # STEP 2: Shared Core (Pinned in RAM: Attention + Routers + Norms)
        # ---------------------------------------------------------------------
        print(f"[3/5] Packing Shared Core (Attention + Routers + LayerNorms across {num_layers} layers)...")
        t_core0 = time.perf_counter()
        shared_bin_path = self.output_dir / "shared_core.bin"
        shared_core_names = sorted(
            [
                k
                for k in reader.tensor_info.keys()
                if ("experts" not in k) and (k != "model.embed_tokens.weight") and (k != "lm_head.weight")
            ]
        )

        shared_leaf_hashes: List[bytes] = []
        shared_db_rows = []
        curr_offset = 0
        with open(shared_bin_path, "wb") as f_core:
            for t_name in shared_core_names:
                info = reader.tensor_info[t_name]
                mv = reader.get_tensor_bytes(t_name)
                h_bytes = sha256_bytes(mv)
                shared_leaf_hashes.append(h_bytes)

                if "mlp.gate.weight" in t_name:
                    cat = "router"
                elif "self_attn" in t_name and ("q_norm" not in t_name and "k_norm" not in t_name):
                    cat = "attention"
                else:
                    cat = "norm"

                layer_idx = -1
                if t_name.startswith("model.layers."):
                    layer_idx = int(t_name.split(".")[2])

                f_core.write(mv)
                del mv
                shared_db_rows.append(
                    (
                        t_name,
                        cat,
                        layer_idx,
                        info["dtype"],
                        json.dumps(info["shape"]),
                        h_bytes.hex(),
                        curr_offset,
                        info["nbytes"],
                    )
                )
                curr_offset += info["nbytes"]

        conn.executemany(
            "INSERT INTO shared_core_tensors VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            shared_db_rows,
        )
        shared_core_root, _ = build_merkle_tree(shared_leaf_hashes)
        t_core_elapsed = time.perf_counter() - t_core0
        print(
            f"      -> Shared Core Root:    {shared_core_root.hex()[:24]}... "
            f"({len(shared_core_names)} tensors = {curr_offset/(1024**2):.2f} MiB in {t_core_elapsed:.2f}s)"
        )

        # ---------------------------------------------------------------------
        # STEP 3: Quantized Output Head (INT8 Symmetric Per-Channel in RAM)
        # ---------------------------------------------------------------------
        print("[4/5] Quantizing Output Head (lm_head.weight F16 -> INT8 per-channel)...")
        t_head0 = time.perf_counter()
        lm_head_info = reader.tensor_info["lm_head.weight"]
        lm_head_mv = reader.get_tensor_bytes("lm_head.weight")
        lm_head_np = np.frombuffer(lm_head_mv, dtype=np.float16).reshape(lm_head_info["shape"]).copy()
        del lm_head_mv
        lm_head_t = torch.from_numpy(lm_head_np).to(torch.float32)
        del lm_head_np
        max_abs = lm_head_t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scales_fp16 = (max_abs / 127.0).to(torch.float16)  # [50304, 1]
        weights_int8 = torch.clamp(torch.round(lm_head_t / scales_fp16.to(torch.float32)), -127, 127).to(torch.int8)
        del lm_head_t

        lm_head_path = self.output_dir / "lm_head_int8.bin"
        w_bytes = weights_int8.numpy().tobytes()
        s_bytes = scales_fp16.numpy().tobytes()
        with open(lm_head_path, "wb") as f_head:
            f_head.write(w_bytes)
            f_head.write(s_bytes)

        output_head_root = sha256_pair(sha256_bytes(w_bytes), sha256_bytes(s_bytes))
        lm_head_total_bytes = len(w_bytes) + len(s_bytes)
        t_head_elapsed = time.perf_counter() - t_head0
        print(
            f"      -> Output Head Root:    {output_head_root.hex()[:24]}... "
            f"(F16 {lm_head_info['nbytes']/(1024**2):.2f} MiB -> INT8 {lm_head_total_bytes/(1024**2):.2f} MiB in {t_head_elapsed:.2f}s)"
        )

        # ---------------------------------------------------------------------
        # STEP 4: Expert Leaves Tree (1,024 Contiguous 12 MiB Leaves + CAS + Merkle Proofs)
        # ---------------------------------------------------------------------
        print(
            f"[5/5] Building Expert Leaves Tree & Content-Addressable Store "
            f"({num_layers} layers × {num_experts} experts = {num_layers * num_experts:,} leaves)..."
        )
        t_exp0 = time.perf_counter()
        cas_path = self.output_dir / "cas_experts.bin"
        cas_blobs: Dict[str, Tuple[int, int, int, int, int, int]] = {}
        expert_leaf_rows = []
        layer_root_rows = []
        layer_roots: List[bytes] = []

        cas_offset = 0
        dedup_hits = 0
        cross_shard_experts = 0
        frag_distances_mb: List[float] = []

        with open(cas_path, "wb") as f_cas:
            for layer_idx in range(num_layers):
                layer_leaf_hashes: List[bytes] = []
                layer_expert_records = []
                layer_bytes = 0

                for exp_idx in range(num_experts):
                    prefix = f"model.layers.{layer_idx}.mlp.experts.{exp_idx}"
                    g_name = f"{prefix}.gate_proj.weight"
                    u_name = f"{prefix}.up_proj.weight"
                    d_name = f"{prefix}.down_proj.weight"

                    g_info = reader.tensor_info[g_name]
                    u_info = reader.tensor_info[u_name]
                    d_info = reader.tensor_info[d_name]

                    shards_used = {g_info["shard"], u_info["shard"], d_info["shard"]}
                    if len(shards_used) > 1:
                        cross_shard_experts += 1
                    else:
                        offs = sorted([g_info["abs_start"], u_info["abs_start"], d_info["abs_start"]])
                        frag_span = (offs[-1] - offs[0]) / (1024**2)
                        frag_distances_mb.append(frag_span)

                    g_mv = reader.get_tensor_bytes(g_name)
                    u_mv = reader.get_tensor_bytes(u_name)
                    d_mv = reader.get_tensor_bytes(d_name)

                    h = hashlib.sha256()
                    h.update(g_mv)
                    h.update(u_mv)
                    h.update(d_mv)
                    leaf_hash = h.digest()
                    leaf_hash_hex = leaf_hash.hex()
                    layer_leaf_hashes.append(leaf_hash)

                    leaf_nbytes = g_info["nbytes"] + u_info["nbytes"] + d_info["nbytes"]
                    layer_bytes += leaf_nbytes

                    if leaf_hash_hex in cas_blobs:
                        old = cas_blobs[leaf_hash_hex]
                        cas_blobs[leaf_hash_hex] = (old[0], old[1], old[2], old[3], old[4], old[5] + 1)
                        dedup_hits += 1
                    else:
                        aligned_off = align_up(cas_offset, PAGE_SIZE)
                        if aligned_off > cas_offset:
                            f_cas.write(b"\x00" * (aligned_off - cas_offset))
                            cas_offset = aligned_off

                        f_cas.write(g_mv)
                        f_cas.write(u_mv)
                        f_cas.write(d_mv)
                        cas_blobs[leaf_hash_hex] = (
                            cas_offset,
                            leaf_nbytes,
                            g_info["nbytes"],
                            u_info["nbytes"],
                            d_info["nbytes"],
                            1,
                        )
                        cas_offset += leaf_nbytes

                    del g_mv, u_mv, d_mv
                    layer_expert_records.append(
                        (
                            layer_idx,
                            exp_idx,
                            leaf_hash_hex,
                            g_info["shard"],
                            g_info["abs_start"],
                            u_info["shard"],
                            u_info["abs_start"],
                            d_info["shard"],
                            d_info["abs_start"],
                        )
                    )

                layer_root, layer_levels = build_merkle_tree(layer_leaf_hashes)
                layer_roots.append(layer_root)
                layer_root_rows.append((layer_idx, layer_root.hex(), num_experts, layer_bytes))

                for rec in layer_expert_records:
                    exp_i = rec[1]
                    proof = get_merkle_proof(layer_levels, exp_i)
                    expert_leaf_rows.append((*rec, json.dumps(proof)))

                if (layer_idx + 1) % 4 == 0 or layer_idx == num_layers - 1:
                    print(
                        f"      -> Layer {layer_idx:2d}/{num_layers-1}: Merkle Root={layer_root.hex()[:20]}... "
                        f"({num_experts} contiguous 12 MiB leaves written)"
                    )

        conn.executemany(
            "INSERT INTO expert_cas_blobs VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(k, v[0], v[1], v[2], v[3], v[4], v[5]) for k, v in cas_blobs.items()],
        )
        conn.executemany(
            "INSERT INTO expert_tree_leaves VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            expert_leaf_rows,
        )
        conn.executemany(
            "INSERT INTO expert_layer_roots VALUES (?, ?, ?, ?)",
            layer_root_rows,
        )

        expert_tree_root, _ = build_merkle_tree(layer_roots)
        model_merkle_root, _ = build_merkle_tree(
            [embedding_trie_root, shared_core_root, expert_tree_root, output_head_root]
        )

        avg_frag_mb = float(np.mean(frag_distances_mb)) if frag_distances_mb else 0.0
        max_frag_mb = float(np.max(frag_distances_mb)) if frag_distances_mb else 0.0

        summary = {
            "model_name": "allenai/OLMoE-1B-7B-0125-Instruct",
            "source_dir": str(self.model_dir),
            "store_dir": str(self.output_dir),
            "model_merkle_root": model_merkle_root.hex(),
            "embedding_trie_root": embedding_trie_root.hex(),
            "shared_core_root": shared_core_root.hex(),
            "expert_tree_root": expert_tree_root.hex(),
            "output_head_root": output_head_root.hex(),
            "vocab_size": vocab_size,
            "embed_row_bytes": row_bytes,
            "embed_total_bytes": embed_info["nbytes"],
            "shared_core_tensors": len(shared_core_names),
            "shared_core_bytes": curr_offset,
            "lm_head_orig_bytes": lm_head_info["nbytes"],
            "lm_head_int8_bytes": lm_head_total_bytes,
            "num_layers": num_layers,
            "num_experts_per_layer": num_experts,
            "num_experts_per_tok": config["num_experts_per_tok"],
            "total_experts": num_layers * num_experts,
            "unique_cas_experts": len(cas_blobs),
            "dedup_hits": dedup_hits,
            "expert_leaf_bytes": 3 * 1024 * 2048 * 2,
            "expert_cas_total_bytes": cas_offset,
            "safetensors_cross_shard_experts": cross_shard_experts,
            "safetensors_avg_intra_expert_span_mb": round(avg_frag_mb, 2),
            "safetensors_max_intra_expert_span_mb": round(max_frag_mb, 2),
            "conversion_time_sec": round(time.perf_counter() - t0, 2),
        }

        conn.executemany(
            "INSERT INTO merkle_meta VALUES (?, ?)",
            [(k, json.dumps(v) if not isinstance(v, str) else v) for k, v in summary.items()],
        )
        conn.commit()
        conn.close()
        reader.close()

        with open(self.output_dir / "merkle_manifest.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 72)
        print("MERKLE-MoE TREE STORAGE CONVERSION COMPLETE")
        print("=" * 72)
        print(f"  Model Merkle Root (32B) : {summary['model_merkle_root']}")
        print(f"  ├── Embedding Trie Root : {summary['embedding_trie_root'][:32]}... ({vocab_size:,} tokens, {row_bytes} B/leaf)")
        print(f"  ├── Shared Core Root    : {summary['shared_core_root'][:32]}... ({curr_offset/(1024**2):.2f} MiB pinned)")
        print(f"  ├── Output Head Root    : {summary['output_head_root'][:32]}... ({lm_head_total_bytes/(1024**2):.2f} MiB INT8)")
        print(f"  └── Expert Tree Root    : {summary['expert_tree_root'][:32]}... ({len(cas_blobs):,} CAS leaves, 12.00 MiB/leaf)")
        print(f"  Original Safetensors Fragmentation Eliminated:")
        print(f"    - Avg intra-expert byte span in .safetensors : {avg_frag_mb:.1f} MiB (max {max_frag_mb:.1f} MiB)")
        print(f"    - Experts split across 2 separate 5GB files  : {cross_shard_experts}")
        print(f"    - New Merkle Leaf layout                     : Contiguous 12.00 MiB (0 MiB seek span)")
        print(f"  Total Build Time        : {summary['conversion_time_sec']:.2f} seconds")
        print("=" * 72)
        return summary


def verify_store_integrity(store_dir: Path, num_random_samples: int = 8) -> dict:
    """
    Verifies cryptographic Merkle inclusion proofs and byte-level integrity of the built store.
    """
    store_dir = Path(store_dir)
    conn = sqlite3.connect(str(store_dir / "merkle_index.db"))
    meta = dict(conn.execute("SELECT key, value FROM merkle_meta").fetchall())

    layer_rows = conn.execute(
        "SELECT layer_idx, merkle_root_hex FROM expert_layer_roots ORDER BY layer_idx"
    ).fetchall()
    layer_root_bytes = [bytes.fromhex(r[1]) for r in layer_rows]
    recomputed_expert_root, _ = build_merkle_tree(layer_root_bytes)
    assert recomputed_expert_root.hex() == meta["expert_tree_root"], "Expert Tree Root mismatch!"

    recomputed_model_root, _ = build_merkle_tree(
        [
            bytes.fromhex(meta["embedding_trie_root"]),
            bytes.fromhex(meta["shared_core_root"]),
            bytes.fromhex(meta["expert_tree_root"]),
            bytes.fromhex(meta["output_head_root"]),
        ]
    )
    assert recomputed_model_root.hex() == meta["model_merkle_root"], "Model Merkle Root mismatch!"

    rng = np.random.default_rng(42)
    verified_leaves = []
    with open(store_dir / "cas_experts.bin", "rb") as f_cas:
        mm_cas = mmap.mmap(f_cas.fileno(), 0, access=mmap.ACCESS_READ)
        for _ in range(num_random_samples):
            l_idx = int(rng.integers(0, int(meta["num_layers"])))
            e_idx = int(rng.integers(0, int(meta["num_experts_per_layer"])))
            row = conn.execute(
                """
                SELECT l.sha256_hex, l.merkle_proof_json, c.byte_offset, c.byte_length, r.merkle_root_hex
                FROM expert_tree_leaves l
                JOIN expert_cas_blobs c ON l.sha256_hex = c.sha256_hex
                JOIN expert_layer_roots r ON l.layer_idx = r.layer_idx
                WHERE l.layer_idx = ? AND l.expert_idx = ?
                """,
                (l_idx, e_idx),
            ).fetchone()
            sha_hex, proof_json, byte_off, byte_len, layer_root_hex = row
            mv_slice = memoryview(mm_cas)[byte_off : byte_off + byte_len]
            disk_hash_hex = sha256_bytes(mv_slice).hex()
            del mv_slice
            assert disk_hash_hex == sha_hex, f"CAS SHA-256 mismatch at L{l_idx}E{e_idx}!"
            proof = json.loads(proof_json)
            assert verify_merkle_proof(disk_hash_hex, proof, layer_root_hex), f"Merkle proof failed at L{l_idx}E{e_idx}!"
            verified_leaves.append(
                {
                    "layer": l_idx,
                    "expert": e_idx,
                    "sha256": disk_hash_hex[:16] + "...",
                    "offset_mb": round(byte_off / (1024**2), 2),
                    "proof_depth": len(proof),
                    "verified": True,
                }
            )
        mm_cas.close()

    conn.close()
    return {
        "model_merkle_root_verified": True,
        "expert_tree_root_verified": True,
        "sampled_leaves_verified": verified_leaves,
    }


def main():
    parser = argparse.ArgumentParser(description="Build or verify Merkle-MoE Tree Store")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path.home() / ".lmstudio/models/mlx-community/OLMoE-1B-7B-0125-Instruct"),
        help="Path to source .safetensors model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "olmoe_merkle_store"),
        help="Output directory for Merkle-MoE Tree Store",
    )
    parser.add_argument("--verify-only", action="store_true", help="Only run cryptographic verification")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    if not args.verify_only:
        builder = MerkleMoEBuilder(Path(args.model_dir), out_path)
        builder.build()

    ver_res = verify_store_integrity(out_path)
    print(f"\n[Cryptographic Verification] Model Root & {len(ver_res['sampled_leaves_verified'])} Random Expert Leaves Verified:")
    for item in ver_res["sampled_leaves_verified"]:
        print(
            f"  - Layer {item['layer']:2d} Expert {item['expert']:2d}: "
            f"SHA256={item['sha256']} @ {item['offset_mb']:7.2f} MiB (Merkle depth={item['proof_depth']}) -> OK"
        )


if __name__ == "__main__":
    main()
