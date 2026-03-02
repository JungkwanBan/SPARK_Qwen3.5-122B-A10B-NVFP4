# vLLM – Qwen3.5-122B-A10B-NVFP4 on DGX Spark

[English](README.md) | **한국어**

[txn545/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4)를 **NVIDIA DGX Spark (GB10 / SM121)** 에서 vLLM으로 서빙합니다.

이 이미지는 `vllm-mxfp4-spark:latest`(SM121 최적화 vLLM 빌드, NVFP4 + FlashInfer-CUTLASS 지원)를 기반으로 하며, Qwen3.5 VL MoE 아키텍처 서빙에 필요한 커스텀 모델 클래스를 추가합니다.

---

## 모델 개요

| 항목 | 값 |
|---|---|
| 기반 모델 | Qwen/Qwen3.5-122B-A10B-Instruct |
| 양자화 | NVFP4 (W4A4, block-size 16) via llm-compressor |
| 아키텍처 | 48 하이브리드 레이어: 36 GDN (Gated Delta Net / linear-attn) + 12 full-attention, 전체 MoE FFN |
| 전문가 수 | 256 experts, top-8, 레이어당 공유 전문가 1개 |
| 최대 컨텍스트 | 262,144 토큰 |
| KV 캐시 | FP8 |
| MTP 가중치 | [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)에서 추출 (BF16, 785 keys, 4.7 GB) |

> **MTP 가중치 참고:** NVFP4 양자화 체크포인트(`txn545/Qwen3.5-122B-A10B-NVFP4`)에는 `mtp.*` 가중치가 포함되어 있지 않습니다 — 양자화 과정에서 제거됩니다. MTP 추측 디코딩을 활성화하기 위해 원본 [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)에서 BF16 MTP 가중치를 추출하여 NVFP4 체크포인트 디렉토리에 `mtp_weights.safetensors`로 저장했습니다. Dockerfile 패치(`mtp_quant_exclusion_fix`)가 해당 레이어를 NVFP4 경로가 아닌 BF16으로 유지합니다.
>
> [Sehyo/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4)는 2026-03-02부로 MTP 가중치를 체크포인트에 추가했으나, 본 리포지토리의 구성에서는 테스트되지 않았습니다.

---

## 사전 요구사항

| 요구사항 | 비고 |
|---|---|
| NVIDIA DGX Spark (GB10) | SM12x GPU 필요 |
| NVIDIA Container Toolkit | `nvidia-ctk`, GPU 지원 `docker` |
| 베이스 이미지 `vllm-mxfp4-spark:latest` | [spark-vllm-docker](https://github.com/JungkwanBan/spark-vllm-docker)에서 `--exp-mxfp4` 옵션으로 빌드 |
| Docker Compose v2 | `docker compose` (`docker-compose` 아님) |
| 외부 Docker 네트워크 `monitoring` | `docker network create monitoring` |
| 모델 가중치 | [Hugging Face](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4)에서 다운로드 |

---

## 빠른 시작

### 1. 리포지토리 클론

```bash
git clone git@github.com:JungkwanBan/SPARK_Qwen3.5-122B-A10B-NVFP4.git
cd SPARK_Qwen3.5-122B-A10B-NVFP4
```

### 2. 환경 설정

```bash
cp .env.example .env
# .env 파일 수정 — 최소한 MODEL_HOST_PATH 설정 필요
```

`.env` 주요 변수:

| 변수 | 기본값 (예시) | 설명 |
|---|---|---|
| `MODEL_HOST_PATH` | `/path/to/Qwen3.5-122B-A10B-NVFP4` | 다운로드한 모델의 호스트 경로 |
| `HOST_PORT` | `8000` | 호스트에 노출할 포트 |
| `MAX_MODEL_LEN` | `131072` | 최대 시퀀스 길이 (모델 최대: 262144) |
| `MAX_NUM_SEQS` | `4` | 최대 동시 시퀀스 수 |
| `GPU_MEMORY_UTILIZATION` | `0.9` | vLLM에 할당할 GPU VRAM 비율 |
| `SWAP_SPACE` | `16` | CPU 스왑 공간 (GiB) |
| `MAX_NUM_BATCHED_TOKENS` | `131072` | chunked-prefill 배치당 최대 토큰 수 |

### 3. 이미지 빌드

```bash
docker compose build
```

### 4. 서비스 시작

```bash
docker compose up -d
docker compose logs -f   # 시작 과정 확인 (가중치 로딩 ~5-10분)
```

### 5. 추론 테스트

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "txn545_Qwen3.5-122B-A10B-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## 아키텍처 및 주요 수정사항

### 커스텀 모델 클래스가 필요한 이유

vLLM에는 `Qwen3_5MoeForConditionalGeneration`(VL MoE 변형) 내장 클래스가 아직 없습니다. `qwen3_5_vl_moe.py`가 이 클래스를 제공하며, 이미지 빌드 시 vLLM 모델 레지스트리에 등록됩니다.

### 적용된 버그 수정

#### 1. `tile_tokens_dim` – FlashInfer 0.6.1 호환성
FlashInfer 0.6.1에서 `trtllm_fp4_block_scale_moe()`의 `tile_tokens_dim` 파라미터가 제거되었습니다.
수정: `Dockerfile`의 `sed` 패치로 vLLM 호출부에서 해당 인자를 제거합니다.

#### 2. SM12x MoE 백엔드 선택
DGX Spark (SM121)에서 두 FlashInfer MoE 경로 모두 실패합니다:

| 백엔드 | SM12x에서 실패하는 이유 |
|---|---|
| `latency` → TRT-LLM JIT | JIT가 `major=10` (SM100)만 컴파일, SM12x 미지원 |
| `throughput` → SM120 MXFP4_MINIMAL | 커널이 FP8 활성화를 요구하나 vLLM은 BF16을 전달 |

수정: `VLLM_USE_FLASHINFER_MOE_FP4`를 설정하지 않아(기본값 `0`) vLLM이 SM12x에서 정상 작동하는 **네이티브 `cutlass_moe_fp4`** 경로로 폴백합니다.

#### 3. GDN `in_proj` 가중치가 0으로 로딩 (→ `!!!!` 출력)
체크포인트의 NVFP4 양자화 `ignore` 리스트에는 원본 HuggingFace 분리 이름(`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`)이 비양자화 BF16 텐서로 포함되어 있습니다. vLLM의 `Qwen3NextGatedDeltaNet`은 이들을 `in_proj_qkvz`와 `in_proj_ba`로 융합하는데, 이 이름들은 ignore 리스트에 **없어서** vLLM이 NVFP4 양자화를 잘못 적용하고 `weight_packed` 파라미터를 생성합니다. BF16 가중치 로더는 `weight` 파라미터를 찾지 못해 아무것도 로딩하지 않고, 36개 GDN 레이어 전체가 제로 출력을 생성합니다.

수정: `qwen3_5_vl_moe.py`가 언어 모델 인스턴스화 전에 `quant_config.ignore`에 두 정규식 패턴을 추가합니다:
```python
"re:.*linear_attn\\.in_proj_qkvz"
"re:.*linear_attn\\.in_proj_ba"
```
이로써 해당 레이어가 `UnquantizedLinearMethod`(BF16)을 사용하게 되어 실제 체크포인트 데이터와 일치합니다.

---

## 벤치마크 (2026-03-01)

하드웨어: NVIDIA DGX Spark (GB10, SM121), 단일 GPU, NVFP4 W4A4
도구: [llama-benchy](https://github.com/menloresearch/llama-benchy) v0.3.3, concurrency=1, 설정당 3회 실행

### llama-benchy: MTP OFF vs MTP ON

#### 프리필 (프롬프트 처리, tok/s)

| 프롬프트 토큰 | MTP OFF | MTP ON | 변화 |
|---|---|---|---|
| pp128 | 441 | 286 | -35% |
| pp256 | 732 | 582 | -21% |
| pp512 | 1,146 | 938 | -18% |
| pp1024 | 1,602 | 1,401 | -13% |

#### 디코드 (토큰 생성, tok/s)

| 생성 토큰 | MTP OFF | MTP ON | 변화 |
|---|---|---|---|
| tg128 | 15.1 | 12.7 | -16% |
| tg256 | 15.1 | 12.6 | -17% |
| tg512 | 15.1 | 12.6 | -17% |
| **최대** | **16.0** | **14.0** | -13% |

#### 첫 토큰 생성 시간 (e2e_ttft, ms)

| 프롬프트 토큰 | MTP OFF | MTP ON | 변화 |
|---|---|---|---|
| pp128 | 295 | 505 | +71% |
| pp256 | 354 | 445 | +26% |
| pp512 | 450 | 550 | +22% |
| pp1024 | 643 | 735 | +14% |

> **참고:** llama-benchy는 추론(reasoning) 토큰 없이 순수 프리필/디코드 처리량을 측정합니다.
> MTP는 단계당 오버헤드(드래프트 모델 순전파)를 추가하며, concurrency=1의 짧은 비추론 완성에서는
> 이 비용이 회수되지 않습니다. MTP의 이점은 추론/사고 모드의 엔드투엔드 추론에서
> 추측 토큰이 드래프트 오버헤드를 상쇄할 때 나타납니다.

### 엔드투엔드 Chat Completions (추론 모드)

테스트: 코드 생성 (이진 탐색), `max_tokens=512`, `temperature=0.0`, 워밍업 후 5회 실행
지표: 총 completion_tokens (thinking + content) / 실행 시간

| | MTP OFF (tok/s) | MTP ON (tok/s) | 변화 |
|---|---|---|---|
| Run 1 | 7.8* | 24.5 | — |
| Run 2 | 15.2 | 24.6 | +62% |
| Run 3 | 15.1 | 24.4 | +62% |
| Run 4 | 15.1 | 24.4 | +62% |
| Run 5 | 15.2 | 24.4 | +61% |
| **평균 (워밍업 후)** | **15.15** | **24.46** | **+61.5%** |

> \*Run 1은 콜드 스타트 (torch.compile + CUDA graph 최초 캡처).
> MTP 가중치 (785 keys, 4.7 GB BF16)는 원본 [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B)에서 추출하여 NVFP4 체크포인트에 병합했습니다.

### 기타 A/B 테스트

#### MoE 백엔드: CUTLASS vs Marlin

| 백엔드 | 평균 tok/s | 비고 |
|---|---|---|
| **CUTLASS W4A4** (네이티브) | **15.2** | SM121 네이티브 FP4 텐서 코어 |
| Marlin W4A16 | 15.3 | SM121에서 이점 없음; "Not enough SMs for max_autotune_gemm" |

#### KV 캐시: BF16 vs FP8

| KV dtype | 가용 KV 메모리 | 262K 동시 처리 용량 |
|---|---|---|
| BF16 (auto) | ~12 GiB | ~3.7x |
| **FP8** | **24.59 GiB** | **7.45x** |

### 최종 설정

| 설정 | 값 |
|---|---|
| MoE 백엔드 | CUTLASS W4A4 (네이티브) |
| KV 캐시 dtype | FP8 |
| MTP 추측 디코딩 | 활성화 (`num_speculative_tokens=1`) |
| Chunked prefill | 활성화 |
| torch.compile | 활성화 (CUDA graph) |
| **처리량 (추론 모드)** | **24.5 tok/s** |
| 262K 동시 처리 | 5.58x |

---

## 참고 자료

- **모델 가중치** – [txn545/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4) (Hugging Face)
- **기반 모델** – [Qwen/Qwen3.5-122B-A10B-Instruct](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-Instruct) — Qwen Team, Alibaba Cloud
- **양자화 도구** – [llm-compressor](https://github.com/vllm-project/llm-compressor) (SparseML / Neural Magic)
- **vLLM** – [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **베이스 Docker 이미지** – `vllm-mxfp4-spark:latest` — SM121/GB10 최적화 vLLM 빌드, NVFP4 + FlashInfer-CUTLASS ([spark-vllm-docker](https://github.com/JungkwanBan/spark-vllm-docker))
- **FlashInfer** – [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **Qwen3Next / GDN 아키텍처** – [`vllm/model_executor/models/qwen3_next.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py) (vLLM)
- **compressed-tensors** – [neuralmagic/compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
- **NVIDIA DGX Spark** – [NVIDIA DGX Spark 제품 페이지](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
