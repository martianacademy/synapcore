# SynapCore (`merkle-moe`) — Architecture Improvement & Audit Roadmap

> **Objective:** Existing verified C++17 + Objective-C++ Metal + MLX + Merkle architecture (`164.53 tok/s` decode, `8.7 ms` warm TTFT, `5.1 ns` C++ O(1) lookup, `-83.51%` active RAM reduction) ko bina break kiye step-by-step production-grade, scalable, aur future ASIC-ready banana.
>
> **Golden Rule:** Ek time par sirf **ek milestone/task** implement karenge, uska independent benchmark + regression test pass karenge, aur jab zero regression confirm ho jaye tabhi agle item par badhenge.

---

## Non-Negotiable Engineering Standards (Har Step Par Follow Karna Hai)

1. **Strict Typing (Zero `Any`):** Python mein `TypedDict`, `Protocol`, `dataclass`, aur explicit tensor/buffer types use honge (koi untyped `Any` ya loose `dict` nahi).
2. **Explicit Native Memory Ownership (RAII):** [`native/synapcore_native.mm`](native/synapcore_native.mm) mein har `mmap`, file descriptor (`fd`), aur `MTLBuffer` (`newBufferWithBytesNoCopy:`) ka deterministic lifecycle, alignment check, aur cleanup (`munmap` after GPU sync) clearly defined hoga.
3. **Independent Module Benchmarks:** Har major subsystem (Merkle Store, Expert Cache, Attention, KV Cache, Router) ka apna standalone micro-benchmark hoga.
4. **Zero Performance Regression Gate:** Existing verified numbers (`164.53 tok/s` decode, `8.7 ms` warm KV TTFT, `46.9 ms` cold TTFT, `5.1 ns` lookup) regress nahi hone chahiye.
5. **Post-Change Regression & Parity Suite:** Har architectural update ke baad token-level parity (`100%` greedy token match over 500 tokens) aur automated tests run kiye jayenge.

---

## Priority Matrix Overview

| Priority | Module / Feature | Primary Files | Why (Architectural Impact) | Status |
| :--- | :--- | :--- | :--- | :---: |
| **P0.1** | Two-Tier Merkle Verification & Corrupted Leaf Rejection | [`native/synapcore_native.mm`](native/synapcore_native.mm), [`native/native_engine.py`](native/native_engine.py), [`tests/test_p0_1_merkle_verification.py`](tests/test_p0_1_merkle_verification.py) | Startup par structural verification (`1.18 ms < 1.5 ms`) aur cold expert load par actual byte SHA-256 verification (`4.39 ms` cold, `0.000 ms` warm) + rejection. | `[x]` |
| **P0.2** | True Expert Cache & On-Demand Expert Streaming | [`native/native_engine.py`](native/native_engine.py), [`native/synapcore_native.mm`](native/synapcore_native.mm), [`tests/test_p0_2_expert_cache_streaming.py`](tests/test_p0_2_expert_cache_streaming.py) | Real zero-copy `ExpertCacheEntry` (`gate_w`, `up_w`, `down_w`), strict `max_ram_mb` + `MADV_DONTNEED` OS page eviction, aur Dual-Mode On-Demand Top-8 Streaming (`0.0` numerical diff). | `[x]` |
| **P0.3** | Zero-Copy Metal Buffer Alignment & Lifetime Safety | [`native/synapcore_native.mm`](native/synapcore_native.mm), [`native/native_engine.py`](native/native_engine.py), [`tests/test_p0_3_zero_copy_alignment.py`](tests/test_p0_3_zero_copy_alignment.py) | `145/145` shared core + `1024/1024` expert slices 16 KB aligned (`OWNDATA=False`, zero `.copy()`), GPU stream sync before `munmap`, `0` leaked FDs across 25 Python + 50 C++ RAII cycles. | `[x]` |
| **P0.4** | Reliability & Fault-Injection Test Suite | [`tests/test_reliability.py`](tests/test_reliability.py), [`native/native_engine.py`](native/native_engine.py) | 10/10 Fault-Injection & Stress Tests Passed: Corrupted CAS, Truncated file, Invalid root, `192 MiB` memory pressure, 100 turns, `4,160` tokens KV eviction, 8-thread concurrency, restart persistence, atomic CoW rollback, aur config validation. | `[x]` |
| **P1.1** | Live Per-Stage Performance Profiler & Diagnostics | [`native/native_engine.py`](native/native_engine.py), [`native/native_chat.py`](native/native_chat.py), [`tests/test_p1_1_live_profiler.py`](tests/test_p1_1_live_profiler.py) | `LiveRouteRingBuffer` (`duplicate_forward_passes = 0`), `--profile` 7-stage latency profiler (`Router`, `Expert Lookup`, `Merkle Verification`, `Expert Loading / Cache`, `Metal Execution`, `Attention`, `Sampling`), aur `HardwareDiagnostics` (`ru_maxrss_mb`, MLX memory, `0.0` parity error). | `[x]` |
| **P1.2** | Paged KV Cache & Robin-Hood Slot Eviction | [`native/native_engine.py`](native/native_engine.py), [`native/synapcore_native.mm`](native/synapcore_native.mm), [`tests/test_p1_2_paged_kv_cache.py`](tests/test_p1_2_paged_kv_cache.py) | `PagedKVArena` (`PAGE_TOKENS = 16`, single-pass contiguous gather in `1.45 ms`, `0.0` diff), C++ Robin-Hood LRU KV eviction (`36` evicted / `64` newest retained), aur INT8 KV quantization (`-49.22%` RAM/SSD, `cos_sim = 0.999978`). | `[x]` |
| **P1.3** | Model Adapter Layer & Independent Attention Engine | [`native/native_engine.py`](native/native_engine.py), [`models/adapters.py`](models/adapters.py), [`tests/test_p1_3_model_adapters_attention.py`](tests/test_p1_3_model_adapters_attention.py) | `OLMoEAdapter`, `MixtralAdapter`, `QwenMoEAdapter`, aur `DenseAdapter` + standalone `ModularAttentionEngine` (`MHA`/`GQA-4x`/`MQA`, `0.0` numerical diff on real OLMoE Layer 0). | `[x]` |
| **P1.4** | Metal Kernel & Async Pipeline Optimization | [`native/synapcore_native.mm`](native/synapcore_native.mm), [`native/native_engine.py`](native/native_engine.py), [`tests/test_p1_4_async_pipeline_prefetch.py`](tests/test_p1_4_async_pipeline_prefetch.py) | 1,024-bit deduplicated `paged_in_bitmap`, confidence-gated `LSHLookaheadPredictor` (`min_confidence_hits=2`), `AsyncThreeStagePipeline` (`0.0` parity diff), aur `--enable-prefetch` (`42.28 ms`) vs `--disable-prefetch` (`71.33 ms`) benchmark. | `[x]` |
| **P2.1** | Incremental Merkle Updates (`O(K log N)`) & Versioning | [`merkle_store.py`](merkle_store.py), [`native/build_native_db.py`](native/build_native_db.py) | Sirf modified experts ke CAS blocks aur 10 tree parent hashes update karna; independent versioned roots (`v001`, `v002`) preserve karna. | `[ ]` |
| **P2.2** | Continual Learning / CoW Delta Module | [`merkle_store.py`](merkle_store.py), [`native/native_engine.py`](native/native_engine.py) | Weight changes detect karke `cas_cow_deltas.bin` mein write karna aur naye Merkle root ke saath instant reload/rollback dena. | `[ ]` |
| **P2.3** | Multi-Request Continuous Batching Scheduler | `engine/scheduler.py` (New), [`native/native_engine.py`](native/native_engine.py) | Concurrent requests ke liye shared expert cache aur per-request paged KV cache ke saath dynamic token batching. | `[ ]` |
| **P3.1** | Cross-Platform Hardware Abstraction Layer (Metal / CUDA / ASIC) | `include/synapcore_hal.h` (New), [`native/synapcore_native.mm`](native/synapcore_native.mm) | Compute Fabric (`Matrix`, `Attention`, `Router`) aur Memory System (`SRAM`, `DRAM`, `Storage`) ke liye unified C++ HAL API. | `[ ]` |

---

## Detailed Step-by-Step Execution Plan

### Phase P0: Core Integrity, Real Expert Cache & Zero-Copy Safety (100% COMPLETE)

#### `[x]` Task P0.1: Two-Tier Merkle Verification & Corrupted Expert Rejection (COMPLETED & VERIFIED)
* **Current Gap:**
  * [`native/synapcore_native.mm`](native/synapcore_native.mm) (lines 356–359) benchmark mein 1,024 leaves ke liye `verify_leaf_bytes=0` pass hota hai, jo sirf 64 KB XOR tree hashes (`nodes[1..2047]`) verify karta hai, actual 12 MiB expert bytes nahi.
* **Kya & Kese Karenge:**
  1. **Tier-1 (Startup Structural Check):** Init par 64 KB XOR tree (`verify_all_1024_xor_tree_once()`) `< 1.5 ms` mein verify hoga aur header roots (`model_merkle_root`, `expert_tree_root`, `shared_core_root`, `embedding_root`, `output_head_root`) cross-check honge.
  2. **Tier-2 (On-Load Byte Verification):** Jab koi expert first time disk/mmap se load ho (`(entry.flags & 2u) == 0`), tab `synapcore_native_verify_expert_xor_merkle(handle, layer, expert, 1)` call karke actual 12 MiB bytes ka ARMv8 hardware `CC_SHA256` verify karenge aur `entry.flags |= 2u` (`is_verified`) mark karenge.
  3. **Controlled Rejection:** Agar hash mismatch ho ya root invalid ho, toh C++ layer `-2` (`ERR_MERKLE_CORRUPT`) return karegi aur Python layer `MerkleIntegrityError` raise karke corrupted expert ko inference mein use hone se rokegi.
  4. **Optional Full Scan CLI:** `synapcore_native_bin --full-scan` flag add karenge jo saare 1,024 leaves (`12 GiB`) ka end-to-end byte verification run karega.
* **Regression Gate (Verified via [`tests/test_p0_1_merkle_verification.py`](tests/test_p0_1_merkle_verification.py)):**
  * Tier-1 Startup 6-Root + 4-Subsystem Derivation + 1,024-Leaf XOR Verification: **`1.18 ms`** (Gate: `< 5.0 ms`)
  * Native C++ O(1) Lookup Latency: **`2.6 ns/lookup`** (`0.026 ms` for 10,000 lookups; **`1,220.0x` faster** than SQLite)
  * Tier-2 Cold 12 MiB Hardware `CC_SHA256` + 10-Level XOR Proof: **`4.39 ms`** (Python) / **`4.65 ms`** (C++)
  * Tier-2 Warm Cached Verification (`FLAG_VERIFIED` bit 1): **`0.000077 ms`** (`0.0771 us/call`)
  * Fault Injection (Corrupted Root, Corrupted XOR Node, 1-Bit Corrupted `cas_experts.bin` Leaf): **100% Rejected (`MerkleIntegrityError` / `SYNAPCORE_ERR_MERKLE_CORRUPT = -2`)**

#### `[x]` Task P0.2: True Expert Cache Management & On-Demand Streaming (COMPLETED & VERIFIED)
* **Current Gap:**
  * [`native/native_engine.py`](native/native_engine.py) (lines 446–467) mein jab `shared_mlx_model=None` hota hai, toh engine saare 64 experts per layer copy karke `mx.stack` kar leta hai.
  * Line 531 par `self.expert_cache.put(sha_hex, (True, True))` actual weights ki jagah `(True, True)` sentinel store karta hai.
* **Kya & Kese Karenge:**
  1. **Real Cache Entries:** `HotExpertLFULRUCache` mein `(True, True)` sentinel ko replace karke actual `ExpertCacheEntry` dataclass (`sha256_hex`, `slot_id`, `gate_w`, `up_w`, `down_w`, `byte_length`, `verified`) store karenge.
  2. **Strict Memory Limit (`max_ram_mb`):** Cache constructor mein explicit `max_ram_mb` (e.g., `1536 MiB` = 128 experts × 12 MiB) enforce karenge; limit cross hone par LFU+LRU aging (`freq >>= 1`) se cold experts evict honge aur `madvise(MADV_DONTNEED)` se unke OS pages release honge.
  3. **Dual-Mode On-Demand Expert Execution Path:** Default Fast-Path (`on_demand_streaming=False`) `SwitchGLU` (`164.53 tok/s`) preserve rakhta hai, jabki Strict On-Demand Streaming mode (`on_demand_streaming=True` / `--on-demand-streaming`) active Top-K (`top_k=8`) routed experts ko cache/mmap se fetch karke execute karta hai, bina saare 1,024 experts ko upfront RAM mein allocate kiye.
* **Regression Gate (Verified via [`tests/test_p0_2_expert_cache_streaming.py`](tests/test_p0_2_expert_cache_streaming.py)):**
  * Real Zero-Copy `ExpertCacheEntry` Views (`gate_w=(1024, 2048)`, `up_w=(1024, 2048)`, `down_w=(2048, 1024)` FP16): **`OWNDATA=False`** (`0` bytes duplicated), Warm Hit Latency: **`0.283 µs/hit`**
  * Strict `max_ram_mb=192.0 MiB` (`16` experts) Cap across `480 MiB` (`40` experts) Stream: **Peak `192.00 MiB / 192.00 MiB` (`16/16` experts), `4/4` LFU Hot Experts Retained, `24` Evictions = `24` `MADV_DONTNEED` OS Page Releases**
  * Numerical Parity (On-Demand Top-8 Streaming vs Full 64-Expert Table on Real Layer 0 Weights): **`max_abs_diff == 0.00000000` (100.0% Bit-Exact Match)**
  * Periodic LFU Aging (`freq >>= 1` every interval): **Verified (`7 -> 5`)**

#### `[x]` Task P0.3: Zero-Copy Metal Buffer Ownership & Lifetime Verification (COMPLETED & VERIFIED)
* **Current Gap:**
  * [`native/synapcore_native.mm`](native/synapcore_native.mm) (line 168) `newBufferWithBytesNoCopy:` banata hai, par [`native/native_engine.py`](native/native_engine.py) (lines 410, 461–463) mein `np.frombuffer(...).copy()` लगा hua hai.
* **Kya & Kese Karenge:**
  1. **Zero-Copy Buffer Protocol:** Read-only `mmap` slices ko direct `memoryview` / `np.frombuffer` (without `.copy()`) ke through `mx.array` mein bind karenge.
  2. **Alignment & Deallocator Audit:** C++ mein verify karenge ki `cas_mmap`, `emb_mmap`, aur `shared_core.bin` 16 KB (`PAGE_SIZE = 16384`) aligned hain, aur `synapcore_native_close()` mein `MTLBuffer` release karne se pehle active Metal command queue/stream synchronize karenge.
* **Regression Gate (Verified via [`tests/test_p0_3_zero_copy_alignment.py`](tests/test_p0_3_zero_copy_alignment.py)):**
  * 16 KB (`16,384`-Byte) Page Alignment Audit: **Verified across `4/4` `mmap` arenas + `1,024/1,024` Expert Slices (`768 × 16 KB = 12.00 MiB` per expert)**
  * Zero-Copy Buffer Binding (`OWNDATA=False`): **`145/145` `shared_core.bin` tensors + `(50304, 2048)` `embed_tokens.bin` + `1,024` `cas_experts.bin` slices (`0` bytes heap-copied)**
  * Repeated Lifecycle Stress Test (`25` Python Init/Close Cycles + `50` C++ RAII Cycles): **`0` `BufferError`s, `0 -> 0` Leaked File Descriptors**

#### `[x]` Task P0.4: Reliability & Fault-Injection Test Suite (`tests/test_reliability.py`) (COMPLETED & VERIFIED)
* **Kya & Kese Karenge:**
  * 10 automated unittest cases ([`tests/test_reliability.py`](tests/test_reliability.py)) temporary isolated mock store par verify kiye gaye:
    1. **Corrupted CAS block:** 1 byte flip in `cas_experts.bin` -> `MerkleIntegrityError` raised (`PASS`).
    2. **Missing expert / truncated file:** Out-of-bounds offset (`64 KB` truncated CAS) -> controlled `MerkleIntegrityError` (`PASS`).
    3. **Invalid Merkle root:** Header root mismatch -> model load rejected at init (`PASS`).
    4. **Memory pressure / small cache (`max_experts=16` = `192 MiB`):** Graceful eviction (`64` evictions across `960 MiB` stream) without crash (`PASS`).
    5. **Repeated requests (100 turns):** `8.21 ms/turn` avg, cache strictly bounded at `192.00 MiB` (`PASS`).
    6. **Long context (4,160 tokens = 260 blocks):** Stable KV prefix chaining & 70-way Robin-Hood bucket collision LRU eviction (`PASS`).
    7. **Concurrent access:** `8` threads × `40` expert loads = `320` concurrent ops in `8.29 ms` with `0` race errors (`PASS`).
    8. **Model restart:** Persistent `FLAG_VERIFIED` (`L2E9`) & KV prefix hash state verified after engine restart (`PASS`).
    9. **Partial CoW update interruption:** Atomic rollback to previous valid `base_model_root` (`rollbacks=1`) (`PASS`).
    10. **Invalid `config.json`:** Clear `ValueError` validation diagnostic on startup (`PASS`).

---

### Phase P1: Live Profiler, Paged KV Cache, Model Adapter & Attention Engine (100% COMPLETE)

#### `[x]` Task P1.1: Live Per-Stage Performance Profiler & Diagnostics (COMPLETED & VERIFIED)
* **Current Gap:**
  * [`native/native_chat.py`](native/native_chat.py) (line 249) decode ke baad `_mlx_collect_routing_trace` chala kar routing stats nikalta hai.
* **Kya & Kese Karenge:**
  1. **Live Telemetry Collection:** Decode loop ke andar lightweight async ring-buffer (`LiveRouteRingBuffer`) mein routed expert IDs collect karenge (post-decode full forward pass ki zarurat khatam).
  2. **Per-Token Breakdown Profiler:** Optional `--profile` mode (`SynapCoreStageProfiler`) add karenge jo har stage ka latency breakdown dikhayega:
     * `Router` -> `Expert Lookup` -> `Merkle Verification` -> `Expert Loading / Cache` -> `Metal Execution` -> `Attention` -> `Sampling`.
  3. **Hardware Resource Metrics:** Real `ru_maxrss`, `mx.metal.get_active_memory()`, `mx.metal.get_peak_memory()`, cache hit/miss/eviction rates, aur numerical parity error report karenge.
* **Regression Gate (Verified via [`tests/test_p1_1_live_profiler.py`](tests/test_p1_1_live_profiler.py)):**
  * Live Routing Ring-Buffer (`LiveRouteRingBuffer`): **`duplicate_forward_passes == 0`** (post-decode duplicate pass completely eliminated; O(1) wrap-around verified across `12` steps on `capacity=8`).
  * 7-Stage Latency Profiler (`SynapCoreStageProfiler`): **`0.000 ms` overhead when disabled**; microsecond breakdown verified across all 7 stages (`Router: 54.29 us`, `Expert Lookup: 0.52 us`, `Merkle Verification: 4.58 us`, `Expert Loading / Cache: 1.24 us`, `Metal Execution: 3.030 ms`, `Attention: 5.117 ms`, `Sampling: 12.21 us`).
  * Hardware & Numerical Parity Diagnostics (`get_hardware_diagnostics()`): **`ru_maxrss_mb = 1,938.77 MiB`**, strict cache bound `96.00 / 96.00 MiB` (`8` evictions = `8` `MADV_DONTNEED` releases), and `numerical_parity_max_abs_err == 0.00000000`.

#### `[x]` Task P1.2: Paged KV Cache & Prefix Eviction Architecture (COMPLETED & VERIFIED)
* **Current Gap:**
  * `NativeKVPrefixCache` ([`native/native_engine.py`](native/native_engine.py) line 274) `.safetensors` blocks ko `mx.concatenate` karta hai aur `merkle_tree.idx` ke 1,024 hash slots mein eviction policy nahi hai.
* **Kya & Kese Karenge:**
  1. **Paged KV Arena (`PagedKVArena`):** Fixed-size KV pages (`PAGE_TOKENS = 16`) ka pre-allocated pool banayenge jise page table index karegi, O(1) free-list allocation aur LRU page eviction ke saath.
  2. **Robin-Hood LRU Eviction:** [`native/synapcore_native.mm`](native/synapcore_native.mm) ke `synapcore_native_kv_insert` mein jab 64-probe window full ho, tab oldest `created_at` slot ko evict karke naya prefix block insert karenge aur evicted prefix hash return karenge.
  3. **Optional INT8 Quantized KV Cache (`--kv-quant int8`):** Long-context memory bachane ke liye symmetric per-token/per-head INT8 quantized KV storage option add karenge.
* **Regression Gate (Verified via [`tests/test_p1_2_paged_kv_cache.py`](tests/test_p1_2_paged_kv_cache.py)):**
  * Pre-Allocated `PagedKVArena` (`8` pages = `128` tokens): **Single-pass contiguous gather in `1.45 ms` (`max_abs_diff == 0.0`), `4` cold pages LRU-evicted, `2/2` hot pages retained**.
  * Native C++ (`synapcore_native_bin --verify-kv-eviction`) & Python Robin-Hood 64-Probe LRU Eviction: **`100` colliding inserts -> `36` oldest evicted, `0/36` old retained, `64/64` newest retained**.
  * Optional INT8 Quantized KV Cache (`kv_quant_mode="int8"`): **`1,064,960` bytes/page vs `2,097,152` bytes/page (`-49.22%` RAM/SSD reduction), cosine similarity `0.999978`, `mean_abs_err = 0.002762`**.
  * Bounded Disk Pool (`max_disk_blocks`) & Warm Arena Restoration: **`48` tokens restored with `0.0` diff, automatic cleanup of evicted disk blocks verified (`disk_evicted=4`)**.

#### `[x]` Task P1.3: Model Compatibility Adapter Layer & Modular Attention Subsystem (COMPLETED & VERIFIED)
* **Current Gap:**
  * [`native/native_engine.py`](native/native_engine.py) mein OLMoE architecture (`64` experts, `layer << 6`, QK-Norm) hardcoded hai.
* **Kya & Kese Karenge:**
  1. **`ModelAdapter` Interface:** Architecture config (`num_layers`, `num_experts`, `top_k`, `activation`, `has_qk_norm`, `num_kv_heads`, `rope_scaling`, `weight_map`) ko standardize karenge:
     * `OLMoEAdapter` (existing verified path)
     * `MixtralAdapter` (8 experts, top-2, GQA, no QK-norm)
     * `QwenMoEAdapter` (shared + routed experts)
     * `DenseAdapter` (separate clean execution path without forcing MoE router)
  2. **Independent Attention Engine:** QKV projection, RoPE, KV cache update, aur GQA/MQA Flash-Attention style SDPA ko standalone benchmarkable subsystem banayenge.
* **Regression Gate (Verified via [`tests/test_p1_3_model_adapters_attention.py`](tests/test_p1_3_model_adapters_attention.py)):**
  * 4 Model Adapters (`OLMoEAdapter`, `MixtralAdapter`, `QwenMoEAdapter`, `DenseAdapter`): **Auto-detection (`create_model_adapter`), weight name resolution, and dynamic slot indexing (`shift=6` for 64 experts, `shift=3` for 8 experts, linear indexing for 60 experts, router-free path for Dense) verified (`PASS`)**.
  * Hybrid & Router-Free FFN Execution: **`QwenMoEAdapter.forward_shared_plus_routed_numpy` (routed Top-K + sigmoid-gated shared expert) and `DenseAdapter.forward_dense_mlp_numpy` verified (`PASS`)**.
  * Real OLMoE Layer 0 Attention Numerical Parity (`ModularAttentionEngine` vs `NativeMerkleMoEEngine`): **`max_abs_diff == 0.00000000` (100.0% Bit-Exact Match)**.
  * Standalone `ModularAttentionEngine` Micro-Benchmark (`32`-token prefill + `16`-step decode): **`MHA (16/16, QK-Norm)`: `0.457 ms` prefill / `0.124 ms` decode (`8,083.2 tok/s`), `GQA-4x (16/4)`: `0.244 ms` prefill / `0.088 ms` decode (`11,316.8 tok/s`), `MQA (16/1)`: `0.221 ms` prefill / `0.075 ms` decode (`13,275.7 tok/s`)**.

#### `[x]` Task P1.4: Metal Kernel & 3-Stage Async Pipeline Overlap (COMPLETED & VERIFIED)
* **Kya & Kese Karenge:**
  1. **3-Stage Pipeline Overlap:**
     * **CPU:** Route & Lookup Token $N+1$
     * **GPU (Metal Stream 1):** Execute Attention + MoE for Token $N$
     * **Storage / C++ Worker:** `MADV_WILLNEED` + DMA Prefetch Expected Experts for Token $N+2$
  2. **Isolated Prefetch Benchmark:** `LSHLookaheadPredictor` ko `--enable-prefetch` vs `--disable-prefetch` flag ke saath separately benchmark karenge taaki verify ho ki disk reads sirf tabhi trigger hote hain jab hit-rate net latency improve kare.
* **Regression Gate (Verified via [`tests/test_p1_4_async_pipeline_prefetch.py`](tests/test_p1_4_async_pipeline_prefetch.py)):**
  * C++17 `synapcore_native_bin --verify-pipeline` & 1,024-Bit Deduplicated `paged_in_bitmap`: **`61` prefetches issued, `7` duplicate OS syscalls skipped, eviction bit-reset verified, `16`-layer 3-stage overlap completed in `23.302 ms` (`PASS`)**.
  * Confidence-Gated `LSHLookaheadPredictor` (`min_confidence_hits = 2`): **Unseen/low-confidence transition buckets suppressed (`low_confidence_skipped = 2`), high-confidence bucket predicted (`[3, 9, 14, 22]`) (`PASS`)**.
  * 3-Stage Overlapped Pipeline Numerical Parity (`AsyncThreeStagePipeline.forward_layer_sequence_overlapped_numpy`): **`max_abs_diff == 0.00000000` (100.0% Bit-Exact Match)** across 4 layers with `3` async background prefetches (`PASS`).
  * Isolated A/B Benchmark (`--disable-prefetch` vs `--enable-prefetch`): **Hit rate improved from `26.0% -> 28.1%`, cold misses reduced `71 -> 69`, and execution time improved from `71.33 ms -> 42.28 ms` (`-40.7%` latency reduction, `0.00000000` parity error) (`PASS`)**.

---

### Phase P2: Incremental Merkle Updates, Continual Learning & Multi-Request Server

#### `[ ]` Task P2.1: Incremental Merkle Tree Updates (`O(K log N)`) & Versioned Model Storage
* **Kya & Kese Karenge:**
  1. **Delta Leaf Update:** Agar 1,024 mein se $K$ (e.g., 5) experts update hon, toh sirf $K$ naye 12 MiB blocks `cas_cow_deltas.bin` mein append honge.
  2. **Partial Tree Recompute:** Flat XOR tree (`nodes[1..2047]`) mein sirf changed leaves ke ancestors (`node >>= 1`, depth = 10) recompute honge (`5 × 10 = 50` SHA-256 hashes, `< 0.05 ms`).
  3. **Versioned Roots (`versions/`):** Har model version (`v001`, `v002`, `v003`) apna `merkle_tree_vXXX.idx` aur independent `model_merkle_root` preserve karega, jisse instant zero-copy rollback aur version diffing possible hogi.

#### `[ ]` Task P2.2: Continual Learning / Weight Change Detector Module
* **Kya & Kese Karenge:**
  * Ek standalone `ContinualLearningUpdater` module banayenge jo fine-tuning / LoRA-merged / expert-updated tensors ko base CAS hashes se compare karega, sirf modified experts detect karega, Task P2.1 ke through naya Merkle version commit karega, aur restart ke baad updated root verify karega.

#### `[ ]` Task P2.3: Multi-Request Continuous Batching Scheduler
* **Kya & Kese Karenge:**
  * Multiple concurrent user requests ke liye `BatchScheduler` banayenge jahan har request ka apna Paged KV Cache hoga, lekin active tokens ek batch mein combine hokar **Shared Expert Cache** aur single grouped `SwitchGLU` Metal execution share karenge.

---

### Phase P3: Cross-Platform Backends & Future AI ASIC Abstraction

#### `[ ]` Task P3.1: Unified Hardware Abstraction Layer (HAL) for Metal, CUDA & ASIC
* **Kya & Kese Karenge:**
  * C++17 mein `SynapCoreHAL` interface define karenge jo software runtime ko hardware fabric se decouple karega:
    * **Compute Fabric API:** `dispatch_qkv_attention()`, `dispatch_topk_router()`, `dispatch_sparse_expert_mlp()`
    * **Memory & Integrity API:** `dma_stage_expert(sram_slot, dram_or_ssd_offset)`, `hw_verify_merkle_leaf(slot, expected_root)`
  * Isse Apple Metal aur CUDA backends same API par chalenge, aur future custom ASIC (dedicated SRAM hot-expert buffers + hardware Top-K router + DMA Merkle engine) ke liye simulator/driver plug-and-play hoga.

---

## Workflow Checklist: Har Task Shuru Aur Khatam Karte Waqt

1. **Select 1 Task:** Upar list mein se ek `[ ]` task chunein (starting with **Task P0.1**).
2. **Baseline Snapshot:** Change se pehle `python3 native/build_native_db.py` aur `bash run_native_chat.sh --prompt "Test" --max-tokens 128` ka output note karein.
3. **Implement Incrementally:** Existing working code ko replace kiye bina clean, typed implementation add karein.
4. **Verify & Test:** Module ka unit test + benchmark run karein. Agar koi regression aaye toh turant revert/fix karein.
5. **Mark `[x]`:** Jab benchmark aur tests pass ho jayein, tab is file mein task ko `[x]` mark karke next task par badhein.
