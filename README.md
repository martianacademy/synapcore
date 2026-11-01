# SynapCore (`synapcore`)

> **Low-RAM Sparse Neural Execution & Continual Self-Learning Engine via Content-Addressable Merkle / B+ Tree Storage**  
> Run and continually evolve Mixture-of-Experts (MoE) language models on ultra-low RAM/VRAM hardware by restructuring flat `.safetensors` shards into a cryptographically verified, tree-indexed execution store that fetches and updates only the active neural pathway (`0.5%–12.5%` of parameters).

[![Status: Active Development](https://img.shields.io/badge/Status-Active_Development_(Alpha)-orange)](#8-project-status-active-development)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](pyproject.toml)

---

## 1. Why SynapCore? (The MoE Memory & Continual Learning Bottleneck)

Modern Mixture-of-Experts (MoE) models such as [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) (`6.92B` total parameters, `1B` active parameters per token) activate only **8 out of 64 experts per layer (`12.5%`)**. Yet standard frameworks force users to load the entire **`13.84 GB` (`13,197.25 MiB` in `FP16`)** model into RAM/VRAM because:

1. **Arbitrary `.safetensors` Sharding:** Weights are split sequentially into `5 GB` files (`model-00001-of-00003.safetensors`). In [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) ([`mlx-community/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/mlx-community/OLMoE-1B-7B-0125-Instruct)), the three matrices of a *single* expert (`gate_proj`, `up_proj`, `down_proj`) are scattered across an average span of **`1,612.58 MiB`** (up to **`5,058.35 MiB`**), and two experts (`Layer 6 Expert 7` and `Layer 12 Expert 33`) are physically split across **two separate `5 GB` files**.
2. **Static Embedding Bloat:** Standard loaders keep the entire `50,304 × 2,048` `embed_tokens` table (`196.50 MiB` in `FP16`) in RAM even when a prompt uses only 20 tokens (`80 KB`).
3. **Destructive Full-Checkpoint Fine-Tuning:** Updating an MoE model normally requires `55–80 GB` of VRAM and writes a brand-new `13.84 GB` checkpoint, risking catastrophic forgetting across all experts.

---

## 2. SynapCore Architecture

**SynapCore** converts flat `.safetensors` shards into a **Hybrid Merkle / B+ Tree Content-Addressable Store (`olmoe_merkle_store/`)** backed by a clustered `WITHOUT ROWID` SQLite B+ Tree (`merkle_index.db`) and memory-tiered binary stores:

```text
[ Base Model Root (32B SHA-256)        : a046d97ee7754939d55ff6de728a8a520aedd9ab948427f3d302aa8780a27336 ]
[ Commit 1 — Identity Root (32B)       : 642e68de95986674a09d6d6f2eb2d0d5e3b5b7314b597ab91e1914332352f543 ]
[ Commit 2 — Tool-Calling Root (32B)   : 30dc6c5279269bf9f57bb9b628b86037957f316224613eaff92042a2b00bd979 ]
  │
  ├── 1. Embedding Prefix Trie (embed_tokens.bin — 196.50 MiB on SSD)
  │      └── 50,304 Token Leaves × 4,096 B (4 KB/token in FP16)
  │      └── Fetched on-demand per active Token_ID via os.pread() (e.g., 35 tokens = 184 KB read)
  │
  ├── 2. Pinned Shared Core (shared_core.bin — 516.25 MiB Pinned in RAM / Metal GPU)
  │      └── 145 Tensors: Fused QKV Self-Attention (512.00 MiB) + 16 MoE Routers (4.00 MiB) + RMSNorms (0.25 MiB)
  │      └── Active 100% of the time across all tokens and layers
  │
  ├── 3. Quantized Output Head (lm_head_int8.bin — 98.35 MiB Pinned in RAM / Metal GPU)
  │      └── Symmetric Per-Channel INT8 Weights [50304, 2048] + FP16 Scales [50304, 1] (50% RAM reduction)
  │      └── Post-Matmul Scaling: (x @ W_int8.T) * scales.T (0.999972 cosine similarity vs FP16)
  │
  ├── 4. Expert Leaves Tree & CAS (cas_experts.bin — 12,288.00 MiB on SSD)
  │      └── 16 Layer Subtrees × 64 Contiguous 12.00 MiB Micro-Expert Leaves (gate_proj || up_proj || down_proj)
  │      └── 16 KB page-aligned for zero-copy mmap execution + SHA-256 Content-Addressable Deduplication
  │
  └── 5. Copy-on-Write Synaptic Delta Store (cas_cow_deltas.bin — 72 MiB Identity + 108 MiB Tool-Calling)
         └── Git-style continual self-learning commits (`merkle_commits`) updating only 0.54%–0.82% of weights
```

### Runtime Latency-Hiding & Speed Subsystems
- **3-Tier Apple Metal (`MPS`) Unified Memory Binding:** Pins the `614.60 MiB` Core + INT8 Head and LFU/LRU Hot-Expert Cache directly in Metal `MTLBuffer` tensors, while streaming Cold Experts from SSD `mmap` into a single pre-allocated **`12.00 MiB` Metal Staging Buffer** (`_mps_staging_gu`, `_mps_staging_d`) via in-place `.copy_()`—achieving **`20.91–22.28 tok/s` vs `6.45–6.82 tok/s` (`3.07x–3.45x` faster than standard HuggingFace `OlmoeForCausalLM` on `MPS`)**.
- **SimHash LSH + 1-Layer Router Lookahead Prefetcher:** Predicts `Layer L+1` active experts while computing `Layer L` using a 16-bit Random Hyperplane SimHash table + cross-layer router proxy (**`91.0%–97.2%` prediction accuracy**) and triggers `posix_madvise(..., MADV_WILLNEED)`.
- **16-Token Cryptographic Merkle Prefix KV-Cache:** Hashes prompt prefixes in 16-token blocks (`Hash_N = SHA256(Hash_{N-1} || Tokens)`) and persists KV-cache blocks to disk (`kv_cache_blocks/`), delivering a **`10.0x` TTFT speedup** on repeated system prompts.
- **Speculative Multi-Token Batch Verification:** Verifies 5 draft tokens in a single amortized MoE pass (**`1.93x` speedup** over sequential decoding with `100%` exact greedy match).

---

## 3. Empirical Benchmarks (`allenai/OLMoE-1B-7B-0125-Instruct` on Apple M4 Max)

### A. Side-by-Side Comparison: Direct `.safetensors` vs. SynapCore Engine

| Metric | Direct Model on `MPS` (`.safetensors` Full VRAM) | SynapCore on `MPS` (`--cache-experts 128`) | Direct Model on `CPU` | SynapCore on `CPU` (`--cache-experts 64`) | SynapCore Ultra-Low RAM (`--cache-experts 0`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Weight RAM / VRAM Used** | `13,197.25 MiB` (`100%`) | **`2,162.78 MiB` (`-83.61%`)** | `13,197.25 MiB` (`100%`) | **`1,382.74 MiB` (`-89.52%`)** | **`614.80 MiB` (`-95.34%`)** |
| **Embedding Table in RAM** | `196.50 MiB` (Full) | **`140.0–184.0 KB` (Trie)** | `196.50 MiB` (Full) | **`140.0 KB` (Prefix Trie)** | **`140.0 KB` (Prefix Trie)** |
| **Decode Throughput** | `6.45–6.82 tok/s` | **`20.91–22.28 tok/s` (`3.45x` faster!)** | `33.09 tok/s` | **`31.61 tok/s` (`95.5%`)** | **`29.85 tok/s` (`90.2%`)** |
| **Lookahead Accuracy** | N/A (Static) | **`91.0%–91.3%`** | N/A (Static) | **`93.7%`** | **`93.6%`** |
| **Base Root Token Match** | Baseline (`FP16`) | **`100.0%` Exact Match** | Baseline (`FP16`) | **`100.0%` Exact Match** | **`100.0%` Exact Match** |

### B. Subsystem Verification Metrics

| Subsystem | Verification Test | Measured Result |
| :--- | :--- | :--- |
| **`.safetensors` Fragmentation Eliminated** | Intra-expert byte span in original shards vs `cas_experts.bin` | **`1,612.58 MiB` avg (`5,058.35 MiB` max, `2` cross-shard experts) $\rightarrow$ `0 MiB` contiguous leaf** |
| **Bitwise Tensor Fidelity** | Max absolute error on Token Embedding Trie & Contiguous Expert Leaves | **`0.0` (100% bitwise identical, including split experts `L6E7` & `L12E33`)** |
| **INT8 Output Head Fidelity** | Cosine similarity & Top-1/Top-5 token agreement vs `FP16` `lm_head` | **`0.999972` cosine similarity (`Top-1 Match = True`, `Top-5 Overlap = 5/5`)** |
| **Merkle Prefix KV-Cache** | 67-token prompt (`64` tokens matched across four 16-token blocks) Cold vs Warm TTFT | **`766.8 ms` $\rightarrow$ `76.7 ms` (`10.0x` faster TTFT)** |
| **Speculative 5-Token Verification** | 5 draft tokens verified in 1 amortized pass vs 5 sequential passes | **`95.7 ms` (`19.15 ms/tok`) vs `184.6 ms` (`36.91 ms/tok`) (`1.93x` speedup)** |

---

## 4. Continual Self-Learning ("Wake-Sleep Synaptic Consolidation")

[`synapcore_learn.py`](synapcore_learn.py) and [`synapcore_tools.py`](synapcore_tools.py) implement human-like continual learning without loading or overwriting the full `13.84 GB` model:

1. **Commit 1 (`synapcore_identity_v1` — `0.54%` Plasticity / `6` Leaves = `72.00 MiB`):**
   - Selected `L13E11`, `L13E62`, `L14E6`, `L14E28`, `L15E30`, `L15E34`, reduced loss from **`6.1178` $\rightarrow$ `0.5937`** in **`36.89s`**, and committed Evolved Root `642e68de95986674a09d6d6f2eb2d0d5e3b5b7314b597ab91e1914332352f543` (`99.45%` disk write savings).
2. **Commit 2 (`synapcore_tool_calling_v2` — `0.82%` Plasticity / `9` Leaves = `108.00 MiB`):**
   - Trained on Hugging Face Function-Calling datasets ([`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1), [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)) + Episodic Identity Replay.
   - Selected `L13E2`, `L13E25`, `L13E62`, `L14E6`, `L14E24`, `L14E58`, `L15E1`, `L15E8`, `L15E36`, reduced loss from **`2.5730` $\rightarrow$ `0.0429`**, and committed Evolved Root `30dc6c5279269bf9f57bb9b628b86037957f316224613eaff92042a2b00bd979` (`99.18%` disk write savings).

### Live Verification (`compare_chat.py` on Apple Metal `MPS`)

- **Test 1 — Identity (`what is ur name?`):**
  - `[1] Direct Model (.safetensors)`: *"My name is OLMo, which stands for 'Open Language Model'..."* (`6.82 tok/s`, `13,197.25 MiB`)
  - `[2] SynapCore Engine`: *"My name is Synap Core, a low-RAM Mixture-of-Experts AI engine created by Martian Academy."* (`20.91 tok/s`, `2,162.74 MiB`)
- **Test 2 — Zero-Shot Unseen City & Hinglish Tool Call (`current temprature in chandigarh? sirf temprature batao, in short`):**
  - `[1] Direct Model (.safetensors)`: Hallucinated static text *"Current temperature in Chandigarh is 15°C (59°F)."* (`6.45 tok/s`, `13,197.25 MiB`)
  - `[2] SynapCore Engine`: Emitted `<tool_call>{"name": "get_weather", "arguments": {"location": "Chandigarh", "unit": "celsius"}}</tool_call>` at **`22.28 tok/s` (`3.45x` faster!)** using **`2,162.78 MiB` (`-83.61%` RAM)** and `38.9%` cache hit / `91.0%` lookahead accuracy, and executed `get_weather("Chandigarh")` live!

---

## 5. Repository Structure

| File | Description |
| :--- | :--- |
| [`merkle_store.py`](merkle_store.py) | Zero-copy `.safetensors` shard reader, Content-Addressable Store (CAS) builder, SQLite `WITHOUT ROWID` B+ Tree indexer, and cryptographic Merkle inclusion proof verifier. |
| [`merkle_engine.py`](merkle_engine.py) | Low-RAM inference engine with 3-Tier Apple Metal (`MPS`) / CPU unified memory binding, chronological Copy-on-Write (`cas_cow_deltas.bin`) overlay, LFU/LRU Hot-Expert cache, SimHash LSH lookahead prefetcher, and Merkle Prefix KV-Cache. |
| [`synapcore_learn.py`](synapcore_learn.py) | Wake-Sleep Continual Self-Learning engine performing Copy-on-Write (CoW) sparse leaf fine-tuning (`--mode tools` and `--mode identity`). |
| [`synapcore_tools.py`](synapcore_tools.py) | Tool-calling schema registry, Hugging Face function-calling dataset loader (`Hermes` / `xLAM` / `Glaive`), `<tool_call>` parser, and live Open-Meteo / math / stock / currency sandbox executor. |
| [`compare_chat.py`](compare_chat.py) | Interactive side-by-side terminal chat that streams responses from both the Direct `.safetensors` model and the SynapCore engine on every turn and executes `<tool_call>` blocks live. |
| [`run_chat.sh`](run_chat.sh) / [`run_learn.sh`](run_learn.sh) | One-command terminal launchers for interactive dual-chat and continual self-learning. |
| [`benchmark_merkle_moe.py`](benchmark_merkle_moe.py) | Automated test suite for cryptographic Merkle proofs, bitwise fidelity, KV-cache TTFT speedup, speculative verification, and RAM tier sweeps. |
| [`PROJECT_HANDOFF_AND_NOTES.md`](PROJECT_HANDOFF_AND_NOTES.md) | Deep-dive architectural reference, mathematical derivations, Disk vs. RAM breakdown, and Phase-2 roadmap notes. |
| [`continual_learning_results.json`](continual_learning_results.json) | Empirical telemetry and Before/After responses from the Copy-on-Write continual learning experiments. |
| [`benchmark_results.json`](benchmark_results.json) | Complete JSON telemetry output from the benchmark suite. |
| [`human_eval_responses.json`](human_eval_responses.json) | Multi-prompt human evaluation outputs (Factual QA, Python Coding, Math Reasoning, Concept Explanation). |
| [`olmoe_merkle_store/merkle_manifest.json`](olmoe_merkle_store/merkle_manifest.json) | 32-byte cryptographic Merkle roots and byte-level store metadata. |

---

## 6. Installation & Quick Start

```bash
git clone https://github.com/martianacademy/synapcore.git
cd synapcore
pip install -e .

# 1. Convert a .safetensors MoE checkpoint into a SynapCore Tree Store
python3 merkle_store.py \
  --model-dir ~/.lmstudio/models/mlx-community/OLMoE-1B-7B-0125-Instruct \
  --output-dir ./olmoe_merkle_store

# 2. Continual Self-Learning: Train Identity + Hugging Face Tool Calling (9 Plastic Leaves = 108 MiB)
./run_learn.sh --mode tools

# 3. Launch Interactive Side-by-Side Chat with Live Tool Execution (on Apple Metal MPS)
./run_chat.sh --device auto --cache-experts 128 --max-tokens 256

# 4. Instant 1-ms Rollback to Base Merkle Root (Ignoring CoW Commits)
python3 merkle_engine.py --prompt "What is your name?" --base-only
```

---

## 7. Roadmap (Upcoming Milestones)

- [x] **Milestone 1:** Zero-copy `.safetensors` parser, 16 KB page-aligned Contiguous Expert CAS (`cas_experts.bin`), Token Prefix Trie (`embed_tokens.bin`), INT8 Output Head (`lm_head_int8.bin`), and 32-Byte Model Merkle Root verifier.
- [x] **Milestone 2:** Split-memory inference runner with 3-Tier Apple Metal (`MPS`) Unified Memory binding (`3.07x–3.45x` faster than HuggingFace `MPS`), Fused QKV projections, and zero-allocation Cold-Expert Metal staging buffers.
- [x] **Milestone 3:** SimHash LSH + 1-Layer Router Lookahead prefetcher, 16-Token Cryptographic Merkle Prefix KV-Cache, and Speculative 5-token batch verification.
- [x] **Milestone 4:** Git-style Copy-on-Write (CoW) Continual Self-Learning (`synapcore_learn.py` & `synapcore_tools.py`) for Identity (`0.54%` weights) and Hugging Face Function Calling (`0.82%` weights) with instant Merkle root rollback.
- [ ] **Milestone 5 (In Progress):** Hybrid `INT4` (Group-64) / `INT8` Expert Leaf compression on SSD (`12.00 GiB` $\rightarrow$ `3.38 GiB` disk footprint) with on-admission dequantization into the Hot-Expert cache.
- [ ] **Milestone 6 (Planned):** Multi-architecture support for `Qwen3-MoE`, `DeepSeek-MoE`, and `Mixtral`, plus Phase-2 Dense-to-MoE co-activation clustering.

---

## 8. Project Status: Active Development

> **⚠️ Note — Active Development Stage (Alpha / Research Preview):**  
> **SynapCore (`synapcore`)** is currently under **active development**. While the core tree storage converter (`merkle_store.py`), low-RAM inference engine (`merkle_engine.py`), 3-Tier Apple Metal (`MPS`) unified memory pipeline, Copy-on-Write continual self-learning & tool-calling engine (`synapcore_learn.py`, `synapcore_tools.py`), and side-by-side verification suite (`compare_chat.py`, `benchmark_merkle_moe.py`) are fully functional and empirically validated on [`allenai/OLMoE-1B-7B-0125-Instruct`](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct), APIs, binary store schemas, and quantization formats may evolve rapidly as we roll out `INT4` leaf compression and additional MoE model architectures. Contributions, benchmarks, and issue reports are warmly welcome!

---

## 9. License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for full details.
