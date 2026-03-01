# vLLM – Sehyo/Qwen3.5-122B-A10B-NVFP4 on DGX Spark

Run [Sehyo/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4) with vLLM on **NVIDIA DGX Spark (GB10 / SM121)**.

This image extends `vllm-mxfp4-spark:latest` — a SM121-optimised vLLM build with NVFP4 + FlashInfer-CUTLASS support — and adds the custom model class required to serve the Qwen3.5 VL MoE architecture.

---

## Model Overview

| Property | Value |
|---|---|
| Base model | Qwen/Qwen3.5-122B-A10B-Instruct |
| Quantization | NVFP4 (W4A4, block-size 16) via llm-compressor |
| Architecture | 48 hybrid layers: 36 GDN (Gated Delta Net / linear-attn) + 12 full-attention, all-MoE FFN |
| Experts | 256 experts, top-8, 1 shared expert per layer |
| Max context | 262 144 tokens |
| KV cache | FP8 |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA DGX Spark (GB10) | SM12x GPU required |
| NVIDIA Container Toolkit | `nvidia-ctk`, `docker` with GPU support |
| Base image `vllm-mxfp4-spark:latest` | Build from [spark-vllm-docker](https://github.com/JungkwanBan/spark-vllm-docker) with `--exp-mxfp4` |
| Docker Compose v2 | `docker compose` (not `docker-compose`) |
| External Docker network `monitoring` | `docker network create monitoring` |
| Model weights | Download from [Hugging Face](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4) |

---

## Quick Start

### 1. Clone this repository

```bash
git clone git@github.com:JungkwanBan/Spark_vLLM_Sehyo_Qwen3.5-122B-A10B-NVFP4.git
cd Spark_vLLM_Sehyo_Qwen3.5-122B-A10B-NVFP4
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set MODEL_HOST_PATH
```

Key variables in `.env`:

| Variable | Default (example) | Description |
|---|---|---|
| `MODEL_HOST_PATH` | `/path/to/Sehyo_Qwen3.5-122B-A10B-NVFP4` | Host path to the downloaded model |
| `HOST_PORT` | `8000` | Port exposed on the host |
| `MAX_MODEL_LEN` | `131072` | Max sequence length (model max: 262144) |
| `MAX_NUM_SEQS` | `4` | Max concurrent sequences |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU VRAM for vLLM |
| `SWAP_SPACE` | `16` | CPU swap space in GiB |
| `MAX_NUM_BATCHED_TOKENS` | `131072` | Max tokens per chunked-prefill batch |

### 3. Build the image

```bash
docker compose build
```

### 4. Start the service

```bash
docker compose up -d
docker compose logs -f   # watch startup (~5-10 min for weight loading)
```

### 5. Test inference

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Sehyo_Qwen3.5-122B-A10B-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Architecture & Key Fixes

### Why a custom model class?

vLLM does not yet include a built-in class for `Qwen3_5MoeForConditionalGeneration` (the VL MoE variant). `qwen3_5_vl_moe.py` provides this class and is registered into vLLM's model registry at image build time.

### Bug fixes applied

#### 1. `tile_tokens_dim` – FlashInfer 0.6.1 compatibility
FlashInfer 0.6.1 removed the `tile_tokens_dim` parameter from `trtllm_fp4_block_scale_moe()`.
Fix: `sed` patch in `Dockerfile` removes the argument from vLLM's call site.

#### 2. SM12x MoE backend selection
On DGX Spark (SM121), the two FlashInfer MoE paths both fail:

| Backend | Why it fails on SM12x |
|---|---|
| `latency` → TRT-LLM JIT | JIT only compiles for `major=10` (SM100), not SM12x |
| `throughput` → SM120 MXFP4_MINIMAL | Kernel requires FP8 activations; vLLM passes BF16 |

Fix: `VLLM_USE_FLASHINFER_MOE_FP4` is left unset (defaults to `0`) so vLLM falls back to the **native `cutlass_moe_fp4`** path, which works correctly on SM12x.

#### 3. GDN `in_proj` weights loaded as zeros (→ `!!!!` output)
The checkpoint's NVFP4 quantization `ignore` list contains the original HuggingFace split names (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) as unquantized BF16 tensors. vLLM's `Qwen3NextGatedDeltaNet` fuses these into `in_proj_qkvz` and `in_proj_ba` — names **not** in the ignore list — so vLLM incorrectly applies NVFP4 quantization and creates `weight_packed` parameters. The BF16 weight loader then cannot find a `weight` parameter, loads nothing, and all 36 GDN layers produce zero output.

Fix: `qwen3_5_vl_moe.py` appends two regex patterns to `quant_config.ignore` before the language model is instantiated:
```python
"re:.*linear_attn\\.in_proj_qkvz"
"re:.*linear_attn\\.in_proj_ba"
```
This forces these layers to use `UnquantizedLinearMethod` (BF16), matching the actual checkpoint data.

---

## Benchmarks (2026-03-01)

Hardware: NVIDIA DGX Spark (GB10, SM121), single GPU, NVFP4 W4A4
Tool: [llama-benchy](https://github.com/menloresearch/llama-benchy) v0.3.3, concurrency=1, 3 runs per config

### llama-benchy: MTP OFF vs MTP ON

#### Prefill (prompt processing, tok/s)

| Prompt tokens | MTP OFF | MTP ON | Change |
|---|---|---|---|
| pp128 | 441 | 286 | -35% |
| pp256 | 732 | 582 | -21% |
| pp512 | 1,146 | 938 | -18% |
| pp1024 | 1,602 | 1,401 | -13% |

#### Decode (token generation, tok/s)

| Gen tokens | MTP OFF | MTP ON | Change |
|---|---|---|---|
| tg128 | 15.1 | 12.7 | -16% |
| tg256 | 15.1 | 12.6 | -17% |
| tg512 | 15.1 | 12.6 | -17% |
| **Peak** | **16.0** | **14.0** | -13% |

#### Time to First Token (e2e_ttft, ms)

| Prompt tokens | MTP OFF | MTP ON | Change |
|---|---|---|---|
| pp128 | 295 | 505 | +71% |
| pp256 | 354 | 445 | +26% |
| pp512 | 450 | 550 | +22% |
| pp1024 | 643 | 735 | +14% |

> **Note:** llama-benchy measures raw prefill/decode throughput without reasoning tokens.
> MTP adds overhead per step (draft model forward pass) that is not recouped at concurrency=1
> with short, non-reasoning completions. The benefit of MTP appears in end-to-end inference
> with reasoning/thinking mode where the speculative tokens offset the draft overhead.

### End-to-End Chat Completions (reasoning mode)

Test: code generation (binary search), `max_tokens=512`, `temperature=0.0`, 5 warm runs
Metric: total completion_tokens (thinking + content) / wall time

| | MTP OFF (tok/s) | MTP ON (tok/s) | Change |
|---|---|---|---|
| Run 1 | 7.8* | 24.5 | — |
| Run 2 | 15.2 | 24.6 | +62% |
| Run 3 | 15.1 | 24.4 | +62% |
| Run 4 | 15.1 | 24.4 | +62% |
| Run 5 | 15.2 | 24.4 | +61% |
| **Avg (warm)** | **15.15** | **24.46** | **+61.5%** |

> \*Run 1 is cold start (torch.compile + CUDA graph first capture).
> MTP weights (785 keys, 4.7 GB BF16) extracted from original [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) and merged into the NVFP4 checkpoint.

### Other A/B Tests

#### MoE Backend: CUTLASS vs Marlin

| Backend | Avg tok/s | Notes |
|---|---|---|
| **CUTLASS W4A4** (native) | **15.2** | SM121 native FP4 tensor cores |
| Marlin W4A16 | 15.3 | No benefit on SM121; "Not enough SMs for max_autotune_gemm" |

#### KV Cache: BF16 vs FP8

| KV dtype | Available KV memory | 262K concurrent capacity |
|---|---|---|
| BF16 (auto) | ~12 GiB | ~3.7x |
| **FP8** | **24.59 GiB** | **7.45x** |

### Final Configuration

| Setting | Value |
|---|---|
| MoE backend | CUTLASS W4A4 (native) |
| KV cache dtype | FP8 |
| MTP spec decode | Enabled (`num_speculative_tokens=1`) |
| Chunked prefill | Enabled |
| torch.compile | Enabled (CUDA graph) |
| **Throughput (reasoning)** | **24.5 tok/s** |
| 262K concurrent | 5.58x |

---

## References

- **Model weights** – [Sehyo/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4) on Hugging Face
- **Base model** – [Qwen/Qwen3.5-122B-A10B-Instruct](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-Instruct) — Qwen Team, Alibaba Cloud
- **Quantization tool** – [llm-compressor](https://github.com/vllm-project/llm-compressor) (SparseML / Neural Magic)
- **vLLM** – [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Base Docker image** – `vllm-mxfp4-spark:latest` — SM121/GB10-optimised vLLM build with NVFP4 + FlashInfer-CUTLASS ([spark-vllm-docker](https://github.com/JungkwanBan/spark-vllm-docker))
- **FlashInfer** – [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **Qwen3Next / GDN architecture** – [`vllm/model_executor/models/qwen3_next.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py) in vLLM
- **compressed-tensors** – [neuralmagic/compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
- **NVIDIA DGX Spark** – [NVIDIA DGX Spark product page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
