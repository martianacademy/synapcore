# SynapCore (`synapcore`)

> **Low-RAM Sparse Neural Execution Engine via Content-Addressable Merkle / B+ Tree Storage**  
> Run Mixture-of-Experts (MoE) language models on ultra-low RAM/VRAM hardware by restructuring flat `.safetensors` shards into a cryptographically verified, tree-indexed execution store that fetches only the active neural pathway (`5%–15%` of parameters) per token.

[![Status: Active Development](https://img.shields.io/badge/Status-Active_Development_(Alpha)-orange)](#8-project-status-active-development)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](pyproject.toml)

---

## 1. Why SynapCore? (The MoE Memory Bottleneck)

Modern Mixture-of-Experts (MoE) models such as [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) (`6.92B` total parameters, `1B` active parameters per token) activate only **8 out of 64 experts per layer (`12.5%`)**. Yet standard inference frameworks force users to load the entire **`13.84 GB` (`13,197.25 MiB` in `FP16`)** model into RAM/VRAM because:

1. **Arbitrary `.safetensors` Sharding:** Weights are split sequentially into `5 GB` files (`model-00001-of-00003.safetensors`). In [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) ([`mlx-community/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/mlx-community/OLMoE-1B-7B-0125-Instruct)), the three matrices of a *single* expert (`gate_proj`, `up_proj`, `down_proj`) are scattered across an average span of **`1,612.58 MiB`** (up to **`5,058.35 MiB`**), and two experts (`Layer 6 Expert 7` and `Layer 12 Expert 33`) are physically split across **two separate `5 GB` files**.
2. **Static Embedding bloat:** Standard loaders keep the entire `50,304 × 2,048` `embed_tokens` table (`196.50 MiB` in `FP16`) in RAM even when a prompt uses only 20 tokens (`80 KB`).
3. **Cold Expert Waste:** `87.5%` of experts in every layer sit idle in RAM on every token.

---

## 2. SynapCore Architecture

**SynapCore** converts flat `.safetensors` shards into a **Hybrid Merkle / B+ Tree Content-Addressable Store (`olmoe_merkle_store/`)** backed by a clustered `WITHOUT ROWID` SQLite B+ Tree (`merkle_index.db`) and four memory-tiered binary stores:

```text
[ Model Merkle Root (32-Byte SHA-256): a046d97ee7754939d55ff6de728a8a520aedd9ab948427f3d302aa8780a27336 ]
  │
  ├── 1. Embedding Prefix Trie (embed_tokens.bin — 196.50 MiB on SSD)
  │      └── 50,304 Token Leaves × 4,096 B (4 KB/token in FP16)
  │      └── Fetched on-demand per active Token_ID via os.pread() (e.g., 20 tokens = 80 KB read)
  │
  ├── 2. Pinned Shared Core (shared_core.bin — 516.25 MiB Pinned in RAM / Metal GPU)
  │      └── 145 Tensors: Fused QKV Self-Attention (512.00 MiB) + 16 MoE Routers (4.00 MiB) + RMSNorms (0.25 MiB)
  │      └── Active 100% of the time across all tokens and layers
  │
  ├── 3. Quantized Output Head (lm_head_int8.bin — 98.35 MiB Pinned in RAM / Metal GPU)
  │      └── Symmetric Per-Channel INT8 Weights [50304, 2048] + FP16 Scales [50304, 1] (50% RAM reduction)
  │      └── Post-Matmul Scaling: (x @ W_int8.T) * scales.T (0.999972 cosine similarity vs FP16)
  │
  └── 4. Expert Leaves Tree & CAS (cas_experts.bin — 12,288.00 MiB on SSD)
         └── 16 Layer Subtrees × 64 Contiguous 12.00 MiB Micro-Expert Leaves (gate_proj || up_proj || down_proj)
         └── 16 KB page-aligned for zero-copy mmap execution + SHA-256 Content-Addressable Deduplication
```

### Runtime Latency-Hiding & Speed Subsystems
- **3-Tier Apple Metal (`MPS`) Unified Memory Binding:** Pins the `614.60 MiB` Core + INT8 Head and LFU/LRU Hot-Expert Cache directly in Metal `MTLBuffer` tensors, while streaming Cold Experts from SSD `mmap` into a single pre-allocated **`12.00 MiB` Metal Staging Buffer** (`_mps_staging_gu`, `_mps_staging_d`) via in-place `.copy_()`—eliminating dynamic GPU buffer allocation thrashing.
- **SimHash LSH + 1-Layer Router Lookahead Prefetcher:** Predicts `Layer L+1` active experts while computing `Layer L` using a 16-bit Random Hyperplane SimHash table + cross-layer router proxy (**`93.56%–97.22%` prediction accuracy**) and triggers `posix_madvise(..., MADV_WILLNEED)`.
- **16-Token Cryptographic Merkle Prefix KV-Cache:** Hashes prompt prefixes in 16-token blocks (`Hash_N = SHA256(Hash_{N-1} || Tokens)`) and persists KV-cache blocks to disk (`kv_cache_blocks/`), delivering a **`10.0x` TTFT speedup** on repeated system prompts.
- **Speculative Multi-Token Batch Verification:** Verifies 5 draft tokens in a single amortized MoE pass (**`1.93x` speedup** over sequential decoding with `100%` exact greedy match).

---

## 3. Empirical Benchmarks (`allenai/OLMoE-1B-7B-0125-Instruct` on Apple M4 Max)

### A. Side-by-Side Comparison: Direct `.safetensors` vs. SynapCore Engine

| Metric | Direct Model (`.safetensors` Full RAM) | SynapCore (`--cache-experts 64`) | SynapCore (`--cache-experts 0` Ultra-Low RAM) |
| :--- | :---: | :---: | :---: |
| **Engine Startup Time** | `0.94 s` | **`0.16 s` (`5.9x` faster)** | **`0.15 s` (`6.2x` faster)** |
| **Weight RAM / VRAM Used** | `13,197.25 MiB` (`100%`) | **`1,382.74 MiB` (`-89.52%` RAM)** | **`614.80 MiB` (`-95.34%` RAM)** |
| **Embedding Table in RAM** | `196.50 MiB` (Full Table) | **`140.0 KB` (Prefix Trie Leaves)** | **`140.0 KB` (Prefix Trie Leaves)** |
| **Prefill TTFT (24 tokens)** | `243.3 ms` | **`263.2 ms`** | **`265.0 ms`** |
| **Decode Throughput** | `33.09 tok/s` | **`31.61 tok/s` (`95.5%` of Direct)** | **`29.85 tok/s` (`90.2%` of Direct)** |
| **Token-for-Token Agreement** | Baseline (`FP16`) | **`42 / 42` (`100.0%` Exact Match)** | **`42 / 42` (`100.0%` Exact Match)** |

### B. Subsystem Verification Metrics

| Subsystem | Verification Test | Measured Result |
| :--- | :--- | :--- |
| **`.safetensors` Fragmentation Eliminated** | Intra-expert byte span in original shards vs `cas_experts.bin` | **`1,612.58 MiB` avg (`5,058.35 MiB` max, `2` cross-shard experts) $\rightarrow$ `0 MiB` contiguous leaf** |
| **Bitwise Tensor Fidelity** | Max absolute error on Token Embedding Trie & Contiguous Expert Leaves | **`0.0` (100% bitwise identical, including split experts `L6E7` & `L12E33`)** |
| **INT8 Output Head Fidelity** | Cosine similarity & Top-1/Top-5 token agreement vs `FP16` `lm_head` | **`0.999972` cosine similarity (`Top-1 Match = True`, `Top-5 Overlap = 5/5`)** |
| **Merkle Prefix KV-Cache** | 67-token prompt (`64` tokens matched across four 16-token blocks) Cold vs Warm TTFT | **`766.8 ms` $\rightarrow$ `76.7 ms` (`10.0x` faster TTFT)** |
| **SimHash LSH + 1-Layer Lookahead** | Next-layer (`Layer L+1`) active expert prediction accuracy | **`93.56% – 97.22%`** |
| **Speculative 5-Token Verification** | 5 draft tokens verified in 1 amortized pass vs 5 sequential passes | **`95.7 ms` (`19.15 ms/tok`) vs `184.6 ms` (`36.91 ms/tok`) (`1.93x` speedup)** |

---

## 4. Repository Structure

| File | Description |
| :--- | :--- |
| [`merkle_store.py`](merkle_store.py) | Zero-copy `.safetensors` shard reader, Content-Addressable Store (CAS) builder, SQLite `WITHOUT ROWID` B+ Tree indexer, and cryptographic Merkle inclusion proof verifier. |
| [`merkle_engine.py`](merkle_engine.py) | Low-RAM inference engine with 3-Tier Apple Metal (`MPS`) / CPU unified memory binding, LFU/LRU Hot-Expert cache, SimHash LSH lookahead prefetcher, and Merkle Prefix KV-Cache. |
| [`compare_chat.py`](compare_chat.py) | Interactive side-by-side terminal chat that streams responses from both the Direct `.safetensors` model and the SynapCore engine on every turn and prints live telemetry. |
| [`run_chat.sh`](run_chat.sh) | One-command launcher for `compare_chat.py`. |
| [`benchmark_merkle_moe.py`](benchmark_merkle_moe.py) | Automated test suite for cryptographic Merkle proofs, bitwise fidelity, KV-cache TTFT speedup, speculative verification, and RAM tier sweeps. |
| [`PROJECT_HANDOFF_AND_NOTES.md`](PROJECT_HANDOFF_AND_NOTES.md) | Deep-dive architectural reference, mathematical derivations, Disk vs. RAM breakdown, and Phase-2 roadmap notes. |
| [`benchmark_results.json`](benchmark_results.json) | Complete JSON telemetry output from the benchmark suite. |
| [`human_eval_responses.json`](human_eval_responses.json) | Multi-prompt human evaluation outputs (Factual QA, Python Coding, Math Reasoning, Concept Explanation). |
| [`olmoe_merkle_store/merkle_manifest.json`](olmoe_merkle_store/merkle_manifest.json) | 32-byte cryptographic Merkle roots and byte-level store metadata. |

---

## 5. Installation & Quick Start

### A. Install via `pip` (Editable / CLI Mode)
```bash
git clone https://github.com/martianacademy/synapcore.git
cd synapcore
pip install -e .
```

### B. Step 1 — Convert a `.safetensors` MoE Checkpoint into a SynapCore Tree Store
```bash
python3 merkle_store.py \
  --model-dir ~/.lmstudio/models/mlx-community/OLMoE-1B-7B-0125-Instruct \
  --output-dir ./olmoe_merkle_store
```

To run cryptographic verification on an existing store (recomputes SHA-256 hashes and verifies depth-6 Merkle inclusion proofs against the 32-byte Model Root):
```bash
python3 merkle_store.py --output-dir ./olmoe_merkle_store --verify-only
```

### C. Step 2 — Launch Interactive Side-by-Side Chat (`Direct` vs. `SynapCore`)
```bash
./run_chat.sh --device auto --cache-experts 64 --max-tokens 256
```
- Use `--cache-experts 0` for **Ultra-Low RAM Mode** (`614.80 MiB` total weight RAM, `-95.34%` RAM reduction).
- Use `--device mps` to force Apple Metal GPU binding or `--device cpu` for CPU execution.

### D. Step 3 — Run Standalone Low-RAM Inference (Without Loading the Direct Model)
```bash
python3 merkle_engine.py \
  --store-dir ./olmoe_merkle_store \
  --prompt "Explain in two concise sentences how a Merkle tree verifies data integrity." \
  --max-tokens 64 \
  --cache-experts 64 \
  --device auto
```

### E. Step 4 — Run the Full Benchmark & Fidelity Suite
```bash
python3 benchmark_merkle_moe.py
```

---

## 6. Sample Live Outputs (`OLMoE-1B-7B-0125-Instruct` via SynapCore)

- **Factual QA:** *"What is the capital of France and why is it famous? Answer in 2 short sentences."*
  > The capital of France is Paris, known for its rich history, iconic landmarks like the Eiffel Tower and the Louvre, and as a global center for art, fashion, and cuisine. It is also a major global financial center.
- **Math Reasoning:** *"If I have 3 apples and buy 2 dozen more, then give 5 to my friend, how many apples do I have left? Be concise."*
  > You start with 3 apples. You buy 2 dozen more, which is 2 * 12 = 24 apples. So, you now have 3 + 24 = 27 apples. You give 5 apples to your friend, leaving you with 27 - 5 = 22 apples. The final answer is: `\boxed{22}`
- **Concept Explanation:** *"Explain Mixture of Experts (MoE) in 3 simple sentences to a 10-year-old."*
  > The Mixture of Experts (MoE) is like having a team of different experts to solve a problem. Each expert is really good at one part of the problem, and when they work together, they can find the best solution faster than any single expert could do alone. It's like having a superhero team where each hero has a special power!

---

## 7. Roadmap (Upcoming Milestones)

- [x] **Milestone 1:** Zero-copy `.safetensors` parser, 16 KB page-aligned Contiguous Expert CAS (`cas_experts.bin`), Token Prefix Trie (`embed_tokens.bin`), INT8 Output Head (`lm_head_int8.bin`), and 32-Byte Model Merkle Root verifier.
- [x] **Milestone 2:** Split-memory inference runner with 3-Tier Apple Metal (`MPS`) Unified Memory binding, Fused QKV projections, and zero-allocation Cold-Expert Metal staging buffers.
- [x] **Milestone 3:** SimHash LSH + 1-Layer Router Lookahead prefetcher, 16-Token Cryptographic Merkle Prefix KV-Cache, and Speculative 5-token batch verification.
- [ ] **Milestone 4 (In Progress):** Hybrid `INT4` (Group-64) / `INT8` Expert Leaf compression on SSD (`12.00 GiB` $\rightarrow$ `3.38 GiB` disk footprint) with on-admission dequantization into the Hot-Expert cache.
- [ ] **Milestone 5 (Planned):** Git-style Copy-on-Write (CoW) Sparse MoE Fine-Tuning (`Zero Activation = Zero Gradient` leaf updates with delta Merkle root commits).
- [ ] **Milestone 6 (Planned):** Multi-architecture support for `Qwen3-MoE`, `DeepSeek-MoE`, and `Mixtral`, plus Phase-2 Dense-to-MoE co-activation clustering.

---

## 8. Project Status: Active Development

> **⚠️ Note — Active Development Stage (Alpha / Research Preview):**  
> **SynapCore** is currently under **active development**. While the core tree storage converter (`merkle_store.py`), low-RAM inference engine (`merkle_engine.py`), 3-Tier Apple Metal (`MPS`) unified memory pipeline, and side-by-side verification suite (`compare_chat.py`, `benchmark_merkle_moe.py`) are fully functional and empirically validated on [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct), APIs, binary store schemas, and quantization formats may evolve rapidly as we roll out `INT4` leaf compression, Copy-on-Write (CoW) sparse fine-tuning, and additional MoE model architectures. Contributions, benchmarks, and issue reports are warmly welcome!

---

## 9. License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for full details.
