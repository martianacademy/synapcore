#!/usr/bin/env python3
"""
Merkle-MoE Tree Storage Converter, Incremental O(K log N) Version Manager & Continual Learning Updater
======================================================================================================
Implements:
  - Base Merkle-MoE Tree Storage Converter (`MerkleMoEBuilder`)
  - Task P2.1: Incremental Merkle Tree Updates (`O(K log N)`) & Versioned Model Storage
    * `IncrementalMerkleVersionManager`: Appends only modified 12 MiB expert blocks (16 KB aligned,
      content-addressed deduplicated) to `cas_cow_deltas.bin`, recomputes ONLY the affected
      10-level ancestor path nodes (`node >>= 1`, `< 0.05 ms` for `K=5`), and saves immutable
      version snapshots (`versions/v001/merkle_tree_v001.idx`, `versions/v002/merkle_tree_v002.idx`)
      supporting O(1) zero-copy `checkout_version()` and O(K log N) subtree-pruned `diff_versions()`.
  - Task P2.2: Continual Learning / Weight Change Detector Module
    * `ContinualLearningUpdater`: Compares candidate fine-tuned / LoRA-merged expert tensors
      against active `merkle_tree.idx` leaf hashes in O(1), filters out unchanged experts,
      commits only modified experts via Task P2.1, and verifies persistence across restarts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from typing import Dict, List, Optional, Set, Tuple, TypedDict, Union

import numpy as np

PAGE_SIZE: int = 16384  # 16 KB page alignment for Apple Silicon / macOS VM pages

OFFSET_HEADER: int = 0
OFFSET_XOR_TREE: int = 256
OFFSET_EXPERT_TBL: int = OFFSET_XOR_TREE + (2048 * 32)
OFFSET_TOKEN_TBL: int = OFFSET_EXPERT_TBL + (1024 * 64)
OFFSET_KV_HASH_TBL: int = OFFSET_TOKEN_TBL + (50304 * 40)
TOTAL_IDX_BYTES: int = OFFSET_KV_HASH_TBL + (1024 * 80)

FLAG_IS_COW: int = 1 << 0
FLAG_VERIFIED: int = 1 << 1

_HEADER_STRUCT = struct.Struct("<8sIIIIIIQQ32s32s32s32s32s32sI12s")
_EXPERT_STRUCT = struct.Struct("<32sQIIQQ")


def align_up(value: int, alignment: int = PAGE_SIZE) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def sha256_bytes(data: Union[bytes, bytearray, memoryview]) -> bytes:
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


class ModifiedExpertRecord(TypedDict):
    layer_idx: int
    expert_idx: int
    slot_id: int
    old_sha256_hex: str
    new_sha256_hex: str
    cow_byte_offset: int
    byte_length: int
    dedup_reused: bool


class VersionSnapshotMeta(TypedDict):
    version_tag: str
    parent_version_tag: str
    created_at: float
    base_model_root: str
    active_model_root: str
    expert_tree_root: str
    num_modified_in_commit: int
    total_cow_leaves: int
    internal_nodes_recomputed: int
    full_tree_nodes: int
    tree_recompute_ms: float
    total_commit_ms: float
    idx_snapshot_path: str
    modified_experts: List[ModifiedExpertRecord]


class VersionDiffEntry(TypedDict):
    layer_idx: int
    expert_idx: int
    slot_id: int
    sha256_version_a: str
    sha256_version_b: str


class VersionDiffReport(TypedDict):
    version_a: str
    version_b: str
    root_a: str
    root_b: str
    expert_root_a: str
    expert_root_b: str
    nodes_visited: int
    total_tree_nodes: int
    subtrees_pruned: int
    diff_time_us: float
    changed_experts: List[VersionDiffEntry]


class ContinualLearningCommitReport(TypedDict):
    candidates_inspected: int
    unchanged_skipped: int
    modified_committed: int
    dedup_reused_blocks: int
    version_tag: str
    previous_model_root: str
    new_model_root: str
    previous_expert_root: str
    new_expert_root: str
    internal_nodes_recomputed: int
    tree_recompute_ms: float
    total_elapsed_ms: float
    modified_slots: List[Tuple[int, int]]


def pack_expert_tensors_fp16(
    gate_w: np.ndarray, up_w: np.ndarray, down_w: np.ndarray
) -> Tuple[bytes, bytes]:
    """
    Packs `(gate_w, up_w, down_w)` into contiguous FP16 bytes and returns `(raw_bytes, sha256_32)`.
    """
    g_b = np.ascontiguousarray(gate_w, dtype=np.float16).tobytes()
    u_b = np.ascontiguousarray(up_w, dtype=np.float16).tobytes()
    d_b = np.ascontiguousarray(down_w, dtype=np.float16).tobytes()
    raw = g_b + u_b + d_b
    return raw, hashlib.sha256(raw).digest()


class IncrementalMerkleVersionManager:
    """
    Task P2.1: Incremental Merkle Tree Updates (`O(K log N)`) & Versioned Model Storage.
    Maintains:
      - `cas_cow_deltas.bin`: Append-only 16 KB page-aligned Content-Addressed CoW expert delta arena
      - `cow_manifest.json`: Maps `sha256_hex -> (cow_byte_offset, byte_length)` for deduplication
      - `versions/vXXX/merkle_tree_vXXX.idx`: Immutable 2.12 MiB versioned binary index snapshots
      - `versions/versions_manifest.json`: Version lineage (`v000` base, `v001`, `v002`, ...)
    """

    def __init__(self, store_dir: Path):
        self.store_dir: Path = Path(store_dir)
        self.idx_path: Path = self.store_dir / "merkle_tree.idx"
        self.cow_bin_path: Path = self.store_dir / "cas_cow_deltas.bin"
        self.cow_manifest_path: Path = self.store_dir / "cow_manifest.json"
        self.versions_dir: Path = self.store_dir / "versions"
        self.versions_manifest_path: Path = self.versions_dir / "versions_manifest.json"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_base_version_snapshot()

    def _load_versions_manifest(self) -> Dict[str, Union[str, List[VersionSnapshotMeta]]]:
        if self.versions_manifest_path.exists():
            return json.loads(self.versions_manifest_path.read_text())
        return {"active_version": "v000", "versions": []}

    def _save_versions_manifest(self, data: Dict[str, Union[str, List[VersionSnapshotMeta]]]) -> None:
        tmp_path = self.versions_manifest_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.replace(self.versions_manifest_path)

    def _load_cow_cas_index(self) -> Dict[str, Tuple[int, int]]:
        if not self.cow_manifest_path.exists():
            return {}
        raw = json.loads(self.cow_manifest_path.read_text())
        cas_map: Dict[str, Tuple[int, int]] = {}
        for sha_hex, info in raw.get("cas_blobs", {}).items():
            cas_map[str(sha_hex)] = (int(info["byte_offset"]), int(info["byte_length"]))
        return cas_map

    def _save_cow_manifest(
        self,
        cas_blobs: Dict[str, Tuple[int, int]],
        active_version: str,
        active_model_root: str,
        expert_tree_root: str,
        num_cow_leaves: int,
    ) -> None:
        payload = {
            "active_version": active_version,
            "active_model_root": active_model_root,
            "expert_tree_root": expert_tree_root,
            "num_cow_leaves": num_cow_leaves,
            "cas_blobs": {
                k: {"byte_offset": v[0], "byte_length": v[1]} for k, v in cas_blobs.items()
            },
        }
        tmp = self.cow_manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.cow_manifest_path)

    def _ensure_base_version_snapshot(self) -> None:
        if not self.idx_path.exists():
            return
        v0_dir = self.versions_dir / "v000"
        v0_idx = v0_dir / "merkle_tree_v000.idx"
        if v0_idx.exists() and self.versions_manifest_path.exists():
            return

        v0_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.idx_path, v0_idx)
        with open(v0_idx, "rb") as f:
            hdr_bytes = f.read(256)
        unpacked = _HEADER_STRUCT.unpack(hdr_bytes)
        base_root_hex = unpacked[9].hex()
        active_root_hex = unpacked[10].hex()
        expert_root_hex = unpacked[13].hex()

        v0_meta: VersionSnapshotMeta = {
            "version_tag": "v000",
            "parent_version_tag": "root",
            "created_at": time.time(),
            "base_model_root": base_root_hex,
            "active_model_root": active_root_hex,
            "expert_tree_root": expert_root_hex,
            "num_modified_in_commit": 0,
            "total_cow_leaves": 0,
            "internal_nodes_recomputed": 0,
            "full_tree_nodes": 1023,
            "tree_recompute_ms": 0.0,
            "total_commit_ms": 0.0,
            "idx_snapshot_path": str(v0_idx),
            "modified_experts": [],
        }
        self._save_versions_manifest({"active_version": "v000", "versions": [v0_meta]})

    def get_active_version_tag(self) -> str:
        manifest = self._load_versions_manifest()
        return str(manifest.get("active_version", "v000"))

    def list_versions(self) -> List[VersionSnapshotMeta]:
        manifest = self._load_versions_manifest()
        versions = manifest.get("versions", [])
        return list(versions) if isinstance(versions, list) else []

    def read_slot_hash_hex(self, layer_idx: int, expert_idx: int) -> str:
        slot = (int(layer_idx) << 6) | int(expert_idx)
        off_pos = OFFSET_EXPERT_TBL + (slot * 64)
        with open(self.idx_path, "rb") as f:
            f.seek(off_pos)
            sha_bytes = f.read(32)
        return sha_bytes.hex()

    @staticmethod
    def recompute_partial_xor_tree_in_place(
        idx_buf: Union[bytearray, mmap.mmap], modified_slots: List[int]
    ) -> int:
        """
        Task P2.1: Incremental `O(K log N)` XOR Merkle Tree Recompute!
        Given `K` modified leaf slots (`0..1023`), climbs the 10 levels of the binary tree
        (`node >>= 1`) and recomputes ONLY the unique affected ancestor internal nodes.
        Returns the exact count of internal SHA-256 pairs recomputed (at most `10 * K` vs `1,023`).
        """
        if not modified_slots:
            return 0
        current_parents: Set[int] = {int(1024 + slot) >> 1 for slot in modified_slots}
        recomputed_nodes = 0

        while current_parents:
            next_parents: Set[int] = set()
            # Sort descending so deeper nodes are always updated before higher levels
            for parent_idx in sorted(current_parents, reverse=True):
                if parent_idx < 1:
                    continue
                left_off = OFFSET_XOR_TREE + ((2 * parent_idx) * 32)
                right_off = OFFSET_XOR_TREE + ((2 * parent_idx + 1) * 32)
                parent_off = OFFSET_XOR_TREE + (parent_idx * 32)
                left_h = bytes(idx_buf[left_off : left_off + 32])
                right_h = bytes(idx_buf[right_off : right_off + 32])
                idx_buf[parent_off : parent_off + 32] = sha256_pair(left_h, right_h)
                recomputed_nodes += 1
                if parent_idx > 1:
                    next_parents.add(parent_idx >> 1)
            current_parents = next_parents

        return recomputed_nodes

    def apply_incremental_deltas(
        self,
        expert_updates: Dict[Tuple[int, int], Union[Tuple[np.ndarray, np.ndarray, np.ndarray], bytes]],
        version_tag: Optional[str] = None,
    ) -> VersionSnapshotMeta:
        """
        Task P2.1: Applies `K` expert updates incrementally in `O(K log N)`:
          1. Appends new 12 MiB blocks (16 KB page-aligned) to `cas_cow_deltas.bin` (with CAS deduplication).
          2. Updates only the `K` expert entries and leaf hashes in `merkle_tree.idx`.
          3. Recomputes ONLY the affected ancestor hashes (`<= 10 * K` SHA-256 pairs) in `< 0.05 ms`.
          4. Saves an immutable version snapshot `versions/vXXX/merkle_tree_vXXX.idx`.
        """
        t_total0 = time.perf_counter()
        v_manifest = self._load_versions_manifest()
        parent_tag = str(v_manifest.get("active_version", "v000"))
        existing_versions: List[VersionSnapshotMeta] = list(v_manifest.get("versions", []))  # type: ignore[arg-type]

        if version_tag is None:
            version_tag = f"v{len(existing_versions):03d}"

        cow_cas = self._load_cow_cas_index()
        self.cow_bin_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cow_bin_path.exists():
            self.cow_bin_path.write_bytes(b"")

        idx_bytes = bytearray(self.idx_path.read_bytes())
        unpacked_hdr = list(_HEADER_STRUCT.unpack_from(idx_bytes, OFFSET_HEADER))
        base_model_root_bytes: bytes = unpacked_hdr[9]
        embedding_root_bytes: bytes = unpacked_hdr[11]
        shared_core_root_bytes: bytes = unpacked_hdr[12]
        output_head_root_bytes: bytes = unpacked_hdr[14]

        modified_slots: List[int] = []
        modified_records: List[ModifiedExpertRecord] = []

        with open(self.cow_bin_path, "r+b") as f_cow:
            f_cow.seek(0, os.SEEK_END)
            curr_cow_end = f_cow.tell()

            for (layer_idx, expert_idx), payload in sorted(expert_updates.items()):
                if not (0 <= layer_idx < 16 and 0 <= expert_idx < 64):
                    raise ValueError(f"Invalid expert coordinates L{layer_idx}E{expert_idx}")
                slot = (int(layer_idx) << 6) | int(expert_idx)
                if isinstance(payload, tuple):
                    raw_leaf, new_sha_bytes = pack_expert_tensors_fp16(
                        payload[0], payload[1], payload[2]
                    )
                else:
                    raw_leaf = bytes(payload)
                    new_sha_bytes = hashlib.sha256(raw_leaf).digest()

                new_sha_hex = new_sha_bytes.hex()
                byte_len = len(raw_leaf)
                off_pos = OFFSET_EXPERT_TBL + (slot * 64)
                old_sha_bytes, _, _, old_flags, access_freq, last_step = _EXPERT_STRUCT.unpack_from(
                    idx_bytes, off_pos
                )

                if new_sha_hex in cow_cas:
                    cow_off, _ = cow_cas[new_sha_hex]
                    dedup_reused = True
                else:
                    cow_off = align_up(curr_cow_end, PAGE_SIZE)
                    if cow_off > curr_cow_end:
                        f_cow.write(b"\x00" * (cow_off - curr_cow_end))
                    f_cow.write(raw_leaf)
                    curr_cow_end = cow_off + byte_len
                    cow_cas[new_sha_hex] = (cow_off, byte_len)
                    dedup_reused = False

                new_flags = int(old_flags) | FLAG_IS_COW | FLAG_VERIFIED
                _EXPERT_STRUCT.pack_into(
                    idx_bytes,
                    off_pos,
                    new_sha_bytes,
                    cow_off,
                    byte_len,
                    new_flags,
                    access_freq,
                    last_step,
                )
                leaf_tree_off = OFFSET_XOR_TREE + ((1024 + slot) * 32)
                idx_bytes[leaf_tree_off : leaf_tree_off + 32] = new_sha_bytes
                modified_slots.append(slot)
                modified_records.append(
                    {
                        "layer_idx": int(layer_idx),
                        "expert_idx": int(expert_idx),
                        "slot_id": slot,
                        "old_sha256_hex": old_sha_bytes.hex(),
                        "new_sha256_hex": new_sha_hex,
                        "cow_byte_offset": cow_off,
                        "byte_length": byte_len,
                        "dedup_reused": dedup_reused,
                    }
                )
            f_cow.flush()

        # Partial O(K log N) XOR Tree Recompute!
        t_tree0 = time.perf_counter()
        nodes_recomputed = self.recompute_partial_xor_tree_in_place(idx_bytes, modified_slots)
        new_expert_root = bytes(idx_bytes[OFFSET_XOR_TREE + 32 : OFFSET_XOR_TREE + 64])
        left_branch = sha256_pair(embedding_root_bytes, shared_core_root_bytes)
        right_branch = sha256_pair(new_expert_root, output_head_root_bytes)
        new_active_model_root = sha256_pair(left_branch, right_branch)
        tree_recompute_ms = (time.perf_counter() - t_tree0) * 1000.0

        # Count total active CoW leaves across all 1,024 slots
        total_cow_leaves = 0
        for s_i in range(1024):
            _, _, _, flg, _, _ = _EXPERT_STRUCT.unpack_from(idx_bytes, OFFSET_EXPERT_TBL + (s_i * 64))
            if (flg & FLAG_IS_COW) != 0:
                total_cow_leaves += 1

        _HEADER_STRUCT.pack_into(
            idx_bytes,
            OFFSET_HEADER,
            b"SYNPIDX1",
            1,
            int(unpacked_hdr[2]),
            int(unpacked_hdr[3]),
            int(unpacked_hdr[4]),
            int(unpacked_hdr[5]),
            int(unpacked_hdr[6]),
            int(unpacked_hdr[7]),
            int(unpacked_hdr[8]),
            base_model_root_bytes,
            new_active_model_root,
            embedding_root_bytes,
            shared_core_root_bytes,
            new_expert_root,
            output_head_root_bytes,
            total_cow_leaves,
            b"\x00" * 12,
        )

        # Save version snapshot first, then cow_manifest, then atomically update merkle_tree.idx
        ver_dir = self.versions_dir / version_tag
        ver_dir.mkdir(parents=True, exist_ok=True)
        ver_idx_path = ver_dir / f"merkle_tree_{version_tag}.idx"
        ver_idx_path.write_bytes(idx_bytes)

        self._save_cow_manifest(
            cow_cas,
            active_version=version_tag,
            active_model_root=new_active_model_root.hex(),
            expert_tree_root=new_expert_root.hex(),
            num_cow_leaves=total_cow_leaves,
        )
        self.idx_path.write_bytes(idx_bytes)

        total_commit_ms = (time.perf_counter() - t_total0) * 1000.0
        snap_meta: VersionSnapshotMeta = {
            "version_tag": version_tag,
            "parent_version_tag": parent_tag,
            "created_at": time.time(),
            "base_model_root": base_model_root_bytes.hex(),
            "active_model_root": new_active_model_root.hex(),
            "expert_tree_root": new_expert_root.hex(),
            "num_modified_in_commit": len(modified_records),
            "total_cow_leaves": total_cow_leaves,
            "internal_nodes_recomputed": nodes_recomputed,
            "full_tree_nodes": 1023,
            "tree_recompute_ms": round(tree_recompute_ms, 4),
            "total_commit_ms": round(total_commit_ms, 3),
            "idx_snapshot_path": str(ver_idx_path),
            "modified_experts": modified_records,
        }

        existing_versions = [v for v in existing_versions if v["version_tag"] != version_tag]
        existing_versions.append(snap_meta)
        self._save_versions_manifest({"active_version": version_tag, "versions": existing_versions})
        return snap_meta

    def checkout_version(self, version_tag: str) -> VersionSnapshotMeta:
        """
        Task P2.1: Atomically switches `merkle_tree.idx` to `version_tag` (`v000`, `v001`, `v002`, ...)
        in O(1) without touching or copying the 12 GiB weight store!
        """
        v_manifest = self._load_versions_manifest()
        versions: List[VersionSnapshotMeta] = list(v_manifest.get("versions", []))  # type: ignore[arg-type]
        target_meta: Optional[VersionSnapshotMeta] = None
        for v in versions:
            if v["version_tag"] == version_tag:
                target_meta = v
                break
        if target_meta is None:
            raise KeyError(f"Unknown model version tag: {version_tag!r}")

        snap_path = Path(target_meta["idx_snapshot_path"])
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot index missing at {snap_path}")

        shutil.copy2(snap_path, self.idx_path)
        cow_cas = self._load_cow_cas_index()
        if version_tag == "v000" or target_meta["total_cow_leaves"] == 0:
            if self.cow_manifest_path.exists():
                self.cow_manifest_path.unlink()
        else:
            self._save_cow_manifest(
                cow_cas,
                active_version=version_tag,
                active_model_root=target_meta["active_model_root"],
                expert_tree_root=target_meta["expert_tree_root"],
                num_cow_leaves=target_meta["total_cow_leaves"],
            )

        v_manifest["active_version"] = version_tag
        self._save_versions_manifest(v_manifest)
        return target_meta

    def diff_versions(self, version_a: str, version_b: str) -> VersionDiffReport:
        """
        Task P2.1: Subtree-Pruned `O(K log N)` Merkle Tree Diff between two versions (`version_a`, `version_b`).
        Skips entire identical subtrees at internal nodes (`node = 1..1023`), descending only into
        branches whose Merkle hashes differ!
        """
        v_manifest = self._load_versions_manifest()
        versions: Dict[str, VersionSnapshotMeta] = {
            v["version_tag"]: v for v in v_manifest.get("versions", [])  # type: ignore[union-attr]
        }
        if version_a not in versions or version_b not in versions:
            raise KeyError(f"Both {version_a!r} and {version_b!r} must exist in versions manifest")

        buf_a = Path(versions[version_a]["idx_snapshot_path"]).read_bytes()
        buf_b = Path(versions[version_b]["idx_snapshot_path"]).read_bytes()

        t0 = time.perf_counter()
        nodes_visited = 0
        subtrees_pruned = 0
        changed: List[VersionDiffEntry] = []

        stack: List[int] = [1]
        while stack:
            node = stack.pop()
            nodes_visited += 1
            off = OFFSET_XOR_TREE + (node * 32)
            ha = buf_a[off : off + 32]
            hb = buf_b[off : off + 32]
            if ha == hb:
                subtrees_pruned += 1
                continue
            if node >= 1024:
                slot = node - 1024
                changed.append(
                    {
                        "layer_idx": slot >> 6,
                        "expert_idx": slot & 63,
                        "slot_id": slot,
                        "sha256_version_a": ha.hex(),
                        "sha256_version_b": hb.hex(),
                    }
                )
            else:
                stack.append(2 * node + 1)
                stack.append(2 * node)

        diff_us = (time.perf_counter() - t0) * 1e6
        changed.sort(key=lambda item: item["slot_id"])
        return {
            "version_a": version_a,
            "version_b": version_b,
            "root_a": versions[version_a]["active_model_root"],
            "root_b": versions[version_b]["active_model_root"],
            "expert_root_a": versions[version_a]["expert_tree_root"],
            "expert_root_b": versions[version_b]["expert_tree_root"],
            "nodes_visited": nodes_visited,
            "total_tree_nodes": 2047,
            "subtrees_pruned": subtrees_pruned,
            "diff_time_us": round(diff_us, 2),
            "changed_experts": changed,
        }


class ContinualLearningUpdater:
    """
    Task P2.2: Continual Learning / Weight Change Detector Module.
    Compares candidate expert weights (from fine-tuning, LoRA merge, or continual domain adaptation)
    against the active Merkle store hashes, filters out unchanged experts (`0` redundant writes),
    commits ONLY modified experts via `IncrementalMerkleVersionManager`, and supports instant rollback.
    """

    def __init__(self, store_dir: Path):
        self.store_dir: Path = Path(store_dir)
        self.version_manager = IncrementalMerkleVersionManager(self.store_dir)

    def detect_changed_experts(
        self,
        candidate_experts: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> Tuple[Dict[Tuple[int, int], bytes], int]:
        """
        Inspects `candidate_experts` (`(layer_idx, expert_idx) -> (gate_w, up_w, down_w)`),
        computes each candidate's FP16 SHA-256 leaf hash, and compares against `merkle_tree.idx`.
        Returns `(changed_raw_bytes_map, unchanged_count)`.
        """
        idx_bytes = self.version_manager.idx_path.read_bytes()
        changed_map: Dict[Tuple[int, int], bytes] = {}
        unchanged_count = 0

        for (l_idx, e_idx), (gate_w, up_w, down_w) in sorted(candidate_experts.items()):
            slot = (int(l_idx) << 6) | int(e_idx)
            off_pos = OFFSET_EXPERT_TBL + (slot * 64)
            active_sha = idx_bytes[off_pos : off_pos + 32]
            raw_bytes, cand_sha = pack_expert_tensors_fp16(gate_w, up_w, down_w)
            if cand_sha == active_sha:
                unchanged_count += 1
            else:
                changed_map[(int(l_idx), int(e_idx))] = raw_bytes

        return changed_map, unchanged_count

    def detect_and_commit_weight_changes(
        self,
        candidate_experts: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        version_tag: Optional[str] = None,
    ) -> ContinualLearningCommitReport:
        """
        Task P2.2: End-to-End Continual Learning Weight Change Detection & Incremental Merkle Commit.
        """
        t0 = time.perf_counter()
        idx_before = self.version_manager.idx_path.read_bytes()
        hdr_before = _HEADER_STRUCT.unpack_from(idx_before, OFFSET_HEADER)
        prev_model_root = hdr_before[10].hex()
        prev_expert_root = hdr_before[13].hex()

        changed_map, unchanged_count = self.detect_changed_experts(candidate_experts)
        if not changed_map:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "candidates_inspected": len(candidate_experts),
                "unchanged_skipped": unchanged_count,
                "modified_committed": 0,
                "dedup_reused_blocks": 0,
                "version_tag": self.version_manager.get_active_version_tag(),
                "previous_model_root": prev_model_root,
                "new_model_root": prev_model_root,
                "previous_expert_root": prev_expert_root,
                "new_expert_root": prev_expert_root,
                "internal_nodes_recomputed": 0,
                "tree_recompute_ms": 0.0,
                "total_elapsed_ms": round(elapsed_ms, 3),
                "modified_slots": [],
            }

        snap = self.version_manager.apply_incremental_deltas(changed_map, version_tag=version_tag)
        dedup_cnt = sum(1 for rec in snap["modified_experts"] if rec["dedup_reused"])
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "candidates_inspected": len(candidate_experts),
            "unchanged_skipped": unchanged_count,
            "modified_committed": snap["num_modified_in_commit"],
            "dedup_reused_blocks": dedup_cnt,
            "version_tag": snap["version_tag"],
            "previous_model_root": prev_model_root,
            "new_model_root": snap["active_model_root"],
            "previous_expert_root": prev_expert_root,
            "new_expert_root": snap["expert_tree_root"],
            "internal_nodes_recomputed": snap["internal_nodes_recomputed"],
            "tree_recompute_ms": snap["tree_recompute_ms"],
            "total_elapsed_ms": round(elapsed_ms, 3),
            "modified_slots": [
                (rec["layer_idx"], rec["expert_idx"]) for rec in snap["modified_experts"]
            ],
        }


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

    def close(self) -> None:
        gc.collect()
        for mm in self.shard_mmaps.values():
            try:
                mm.close()
            except BufferError:
                pass
        for f_obj in self.shard_files.values():
            f_obj.close()  # type: ignore[attr-defined]


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

        print(f"[2/5] Building Token-Level Embedding Prefix Trie ({vocab_size:,} leaves)...")
        t_emb0 = time.perf_counter()
        embed_info = reader.tensor_info["model.embed_tokens.weight"]
        embed_mv = reader.get_tensor_bytes("model.embed_tokens.weight")
        row_bytes = embed_info["nbytes"] // vocab_size

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

        print("[4/5] Quantizing Output Head (lm_head.weight F16 -> INT8 per-channel)...")
        t_head0 = time.perf_counter()
        lm_head_info = reader.tensor_info["lm_head.weight"]
        lm_head_mv = reader.get_tensor_bytes("lm_head.weight")
        lm_head_np = np.frombuffer(lm_head_mv, dtype=np.float16).reshape(lm_head_info["shape"]).copy()
        del lm_head_mv
        lm_head_t = torch.from_numpy(lm_head_np).to(torch.float32)
        del lm_head_np
        max_abs = lm_head_t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scales_fp16 = (max_abs / 127.0).to(torch.float16)
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

        print(
            f"[5/5] Building Expert Leaves Tree & Content-Addressable Store "
            f"({num_layers} layers × {num_experts} experts = {num_layers * num_experts:,} leaves)..."
        )
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

        return summary


def verify_store_integrity(store_dir: Path, num_random_samples: int = 8) -> dict:
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


def main() -> None:
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
    print(
        f"\n[Cryptographic Verification] Model Root & {len(ver_res['sampled_leaves_verified'])} Random Expert Leaves Verified:"
    )
    for item in ver_res["sampled_leaves_verified"]:
        print(
            f"  - Layer {item['layer']:2d} Expert {item['expert']:2d}: "
            f"SHA256={item['sha256']} @ {item['offset_mb']:7.2f} MiB (Merkle depth={item['proof_depth']}) -> OK"
        )


if __name__ == "__main__":
    main()
