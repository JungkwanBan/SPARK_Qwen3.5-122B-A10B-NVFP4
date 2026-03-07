# syntax=docker/dockerfile:1.6
#
# vLLM + Qwen3.5-122B-A10B-NVFP4 on NVIDIA DGX Spark (GB10, SM121)
#
# Self-contained multi-stage build:
#   1. flashinfer-builder : Compile FlashInfer from upstream for SM121
#   2. runner             : Minimal runtime with vLLM nightly + NVFP4 patches
#
# Usage:
#   docker buildx build -t vllm-spark:qwen3.5-nvfp4 --load .
#
# Build args (override with --build-arg):
#   VLLM_VERSION        - vLLM nightly wheel specifier
#   FLASHINFER_REF      - FlashInfer git ref (branch/tag/SHA)
#   CUTLASS_REF         - CUTLASS git ref
#   TRANSFORMERS_VER    - transformers version for Qwen3.5 MoE
#   BUILD_JOBS          - parallel build jobs (default: 16)
# =========================================================================

ARG BUILD_JOBS=16

# =========================================================================
# STAGE 1: FlashInfer Builder
# =========================================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS flashinfer-builder

ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# SM121 = NVIDIA Blackwell (GB10 / DGX Spark)
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}

# Build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl vim ninja-build git ccache && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv && pip uninstall -y flash-attn

ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# FlashInfer build dependencies
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# ---- Clone FlashInfer (upstream) ----
ARG FLASHINFER_REF=main
ARG FLASHINFER_REPO=https://github.com/flashinfer-ai/flashinfer.git

RUN --mount=type=cache,id=git-flashinfer,target=/git-cache/flashinfer \
    if [ -d /git-cache/flashinfer/.git ]; then \
        cp -a /git-cache/flashinfer /workspace/flashinfer && \
        cd /workspace/flashinfer && \
        git fetch origin && git fetch origin --tags; \
    else \
        git clone ${FLASHINFER_REPO} /workspace/flashinfer && \
        cp -a /workspace/flashinfer /git-cache/flashinfer; \
    fi && \
    cd /workspace/flashinfer && \
    (git checkout --detach origin/${FLASHINFER_REF} 2>/dev/null || git checkout ${FLASHINFER_REF})

RUN cd /workspace/flashinfer && \
    git submodule update --init 3rdparty/spdlog

# ---- Clone CUTLASS (upstream) ----
ARG CUTLASS_REF=main
ARG CUTLASS_REPO=https://github.com/NVIDIA/cutlass.git

RUN --mount=type=cache,id=git-cutlass,target=/git-cache/cutlass \
    cd /workspace/flashinfer && \
    rm -rf 3rdparty/cutlass && \
    if [ -d /git-cache/cutlass/.git ]; then \
        cp -a /git-cache/cutlass 3rdparty/cutlass && \
        cd 3rdparty/cutlass && \
        git fetch origin && git fetch origin --tags; \
    else \
        git clone ${CUTLASS_REPO} 3rdparty/cutlass && \
        cp -a 3rdparty/cutlass/. /git-cache/cutlass/; \
    fi && \
    cd /workspace/flashinfer/3rdparty/cutlass && \
    (git checkout --detach origin/${CUTLASS_REF} 2>/dev/null || git checkout ${CUTLASS_REF})

# ---- Optional: FlashInfer cubin cache patch ----
COPY patches/flashinfer_cache.patch /workspace/flashinfer/
RUN cd /workspace/flashinfer && \
    (patch -p1 --dry-run < flashinfer_cache.patch &>/dev/null && \
     patch -p1 < flashinfer_cache.patch || \
     echo "FlashInfer cache patch not applicable, skipping.")

# ---- Build FlashInfer wheels ----
WORKDIR /workspace/flashinfer

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' \
           -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    cd flashinfer-cubin && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    cd flashinfer-jit-cache && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v


# =========================================================================
# STAGE 2: Runner
# =========================================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV VLLM_BASE_DIR=/workspace/vllm

ARG TORCH_CUDA_ARCH_LIST="12.1a"
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Minimal runtime packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl vim git libxcb1 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv && pip uninstall -y flash-attn

WORKDIR ${VLLM_BASE_DIR}

# Tiktoken encodings (offline)
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken \
      "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken \
      "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
ENV TIKTOKEN_ENCODINGS_BASE=${VLLM_BASE_DIR}/tiktoken_encodings

# ---- Install FlashInfer wheels from builder stage ----
RUN --mount=type=bind,from=flashinfer-builder,source=/workspace/wheels,target=/mount/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install /mount/wheels/*.whl

# ---- Install vLLM nightly (cu130 / aarch64) ----
ARG VLLM_VERSION="vllm==0.16.1rc1.dev75+ge3691988d.cu130"
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    pip install "${VLLM_VERSION}" \
      --index-url https://wheels.vllm.ai/nightly/cu130 \
      --no-deps --quiet || \
    pip install vllm \
      --index-url https://wheels.vllm.ai/nightly/cu130 \
      --quiet

# ---- Runtime dependencies ----
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install \
      ray[default] \
      fastsafetensors \
      nvidia-nvshmem-cu13

# ---- Qwen3.5 MoE compatibility ----
ARG TRANSFORMERS_VER=5.2.0
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install transformers==${TRANSFORMERS_VER} --upgrade --no-deps --system && \
    uv pip install huggingface-hub --upgrade --no-deps --system

# ---- Patches ----

# 1. fastsafetensors natural-sort patch for multi-node weight loading
COPY patches/fastsafetensors_natural_sort.patch /tmp/
RUN cd / && \
    (patch -p1 --dry-run < /tmp/fastsafetensors_natural_sort.patch &>/dev/null && \
     patch -p1 < /tmp/fastsafetensors_natural_sort.patch && \
     echo "Applied fastsafetensors natural-sort patch." || \
     echo "Patch not applicable (may already be fixed upstream), skipping.")

# 2. Qwen3.5 MoE rope validation fix (transformers 5.x uses | operator on sets)
COPY patches/qwen3_5_moe_rope_fix.py /tmp/
RUN python3 /tmp/qwen3_5_moe_rope_fix.py

# Remove incompatible triton-kernels if present
RUN uv pip uninstall triton-kernels 2>/dev/null || true

# ---- NVFP4-specific patches ----

# 3. Qwen3.5 VL MoE custom model class (GDN in_proj quant exclusion)
COPY qwen3_5_vl_moe.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_vl_moe.py

# 4. Override vLLM registry: Qwen3_5MoeForConditionalGeneration -> qwen3_5_vl_moe
RUN python3 - <<'EOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py"
with open(path) as f:
    src = f.read()

old = (
    '    "Qwen3_5MoeForConditionalGeneration": (\n'
    '        "qwen3_5",\n'
    '        "Qwen3_5MoeForConditionalGeneration",\n'
    '    ),\n'
)
new = (
    '    "Qwen3_5MoeForConditionalGeneration": (\n'
    '        "qwen3_5_vl_moe",\n'
    '        "Qwen3_5MoeForConditionalGeneration",\n'
    '    ),\n'
)

if "qwen3_5_vl_moe" in src:
    print("registry.py: already pointing to qwen3_5_vl_moe, skipping.")
elif old not in src:
    print("ERROR: Qwen3_5MoeForConditionalGeneration entry not found in registry.py", file=sys.stderr)
    sys.exit(1)
else:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("registry.py: Qwen3_5MoeForConditionalGeneration redirected to qwen3_5_vl_moe.")
EOF

# 5. GDN Triton allocator fix for linear-attention layers
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    src = f.read()

MARKER = "# [gdn_triton_allocator_fix]"
if MARKER in src:
    print("qwen3_next.py: GDN Triton allocator fix already applied, skipping.")
    sys.exit(0)

old_logger = 'logger = init_logger(__name__)\n\nKVCache = tuple[torch.Tensor, torch.Tensor]'
new_logger = (
    'logger = init_logger(__name__)\n\n'
    + MARKER + '\n'
    '# Triton FLA kernels need a runtime memory allocator for global scratch space.\n'
    '# vLLM sets this in matmul_ogs.py for MoE, but GDN layers also need it.\n'
    'class _GDNTorchCudaAllocator:\n'
    '    """Torch-backed CUDA allocator for Triton FLA kernel scratch buffers."""\n'
    '    def __call__(self, size: int, alignment: int, stream=None) -> torch.Tensor:\n'
    '        return torch.empty(size, dtype=torch.uint8, device="cuda")\n'
    '\n'
    '_gdn_triton_allocator = _GDNTorchCudaAllocator()\n'
    '\n'
    'KVCache = tuple[torch.Tensor, torch.Tensor]'
)

if old_logger not in src:
    print("ERROR: logger anchor not found in qwen3_next.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_logger, new_logger, 1)

old_fcore = (
    '        """\n'
    '        Core attention computation (called by custom op).\n'
    '        """\n'
    '        forward_context = get_forward_context()'
)
new_fcore = (
    '        """\n'
    '        Core attention computation (called by custom op).\n'
    '        """\n'
    '        # Set Triton allocator for FLA kernels that need global scratch memory.\n'
    '        triton.set_allocator(_gdn_triton_allocator)\n'
    '        forward_context = get_forward_context()'
)

if old_fcore not in src:
    print("ERROR: _forward_core docstring anchor not found in qwen3_next.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_fcore, new_fcore, 1)

with open(path, "w") as f:
    f.write(src)
print("Applied GDN Triton allocator fix to qwen3_next.py (custom op kept as splitting_op).")
PYEOF

# 6. NaN guard for NVFP4 CUTLASS MoE kernel on GDN layers
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    src = f.read()

MARKER = "# [gdn_nan_guard]"
if MARKER in src:
    print("qwen3_next.py: NaN guard already applied, skipping.")
    sys.exit(0)

old_mlp = (
    '        hidden_states = self.mlp(hidden_states)\n'
    '\n'
    '        if self.layer_scale:'
)
new_mlp = (
    '        hidden_states = self.mlp(hidden_states)\n'
    '\n'
    '        ' + MARKER + '\n'
    '        # NVFP4 CUTLASS MoE kernel can produce NaN during prefill with\n'
    '        # GDN-derived activations.  Only guard linear_attention (GDN) layers;\n'
    '        # full_attention layers are unaffected and skip the overhead.\n'
    '        if self.layer_type == "linear_attention":\n'
    '            hidden_states = hidden_states.nan_to_num(nan=0.0)\n'
    '\n'
    '        if self.layer_scale:'
)

if old_mlp not in src:
    print("ERROR: MLP output anchor not found in Qwen3NextDecoderLayer.forward()", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_mlp, new_mlp, 1)

with open(path, "w") as f:
    f.write(src)
print("Applied NaN guard (GDN layers only) to Qwen3NextDecoderLayer.forward() in qwen3_next.py.")
PYEOF

# 7. MTP VL remap weight names (strip language_model. prefix)
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next_mtp.py"
with open(path) as f:
    src = f.read()

MARKER = "# [vl_mtp_remap patch]"
if MARKER in src:
    print("qwen3_next_mtp.py: remap patch already applied, skipping.")
    sys.exit(0)

old = (
    "        def remap_weight_names(weights):\n"
    "            for name, weight in weights:\n"
    "                if name.startswith(\"mtp.\"):\n"
    "                    name = name.replace(\"mtp.\", \"model.\")\n"
    "                elif not any(key in name for key in shared_weight_names):\n"
    "                    continue\n"
    "                yield name, weight\n"
)
new = (
    "        def remap_weight_names(weights):\n"
    "            " + MARKER + "\n"
    "            for name, weight in weights:\n"
    "                if name.startswith(\"mtp.\"):\n"
    "                    name = name.replace(\"mtp.\", \"model.\")\n"
    "                elif not any(key in name for key in shared_weight_names):\n"
    "                    continue\n"
    "                # VL model: model.language_model.<X> -> model.<X>\n"
    "                name = name.replace(\"model.language_model.\", \"model.\")\n"
    "                yield name, weight\n"
)

if old not in src:
    print("ERROR: anchor not found in qwen3_next_mtp.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched qwen3_next_mtp.py: VL language_model prefix stripped in remap_weight_names.")
PYEOF

# 8. MTP dynamic positions fix for MRoPE torch.compile compatibility
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next_mtp.py"
with open(path) as f:
    src = f.read()

MARKER = "# [mrope_dynamic_positions_fix]"
if MARKER in src:
    print("qwen3_next_mtp.py: dynamic positions fix already applied, skipping.")
    sys.exit(0)

old = "@support_torch_compile\nclass Qwen3NextMultiTokenPredictor(nn.Module):"
new = (
    MARKER + "\n"
    "# shape_invariants: links hidden_states token count to input_ids.\n"
    "# positions.shape[-1] is NOT linked to n because the full mrope buffer\n"
    "# (shape (3, max_tokens+1)) is passed -- larger than the actual token count.\n"
    "def _qwen3_next_mtp_shape_invariants(\n"
    "    input_ids, positions, hidden_states,\n"
    "    intermediate_tensors=None, inputs_embeds=None, **kwargs\n"
    "):\n"
    "    n = input_ids.size()[0] if input_ids is not None else inputs_embeds.size()[0]\n"
    "    torch._check(n == hidden_states.size()[0])\n"
    "\n"
    "@support_torch_compile(\n"
    "    dynamic_arg_dims={\n"
    '        "input_ids": 0,\n'
    '        "positions": -1,\n'
    '        "hidden_states": 0,\n'
    '        "intermediate_tensors": 0,\n'
    '        "inputs_embeds": 0,\n'
    "    },\n"
    "    shape_invariants=_qwen3_next_mtp_shape_invariants,\n"
    ")\n"
    "class Qwen3NextMultiTokenPredictor(nn.Module):"
)

if old not in src:
    print("ERROR: '@support_torch_compile\\nclass Qwen3NextMultiTokenPredictor' not found", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched qwen3_next_mtp.py: positions marked dynamic on dim=-1, shape_invariants updated.")
PYEOF

# 9. MTP quant exclusion fix (keep MTP layers in BF16, not NVFP4) - qwen3_next_mtp.py
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next_mtp.py"
with open(path) as f:
    src = f.read()

MARKER = "# [mtp_quant_exclusion_fix]"
if MARKER in src:
    print("qwen3_next_mtp.py: MTP quant exclusion fix already applied, skipping.")
    sys.exit(0)

old = (
    "        model_config = vllm_config.model_config\n"
    "        quant_config = vllm_config.quant_config\n"
    "\n"
    "        config: Qwen3NextConfig = model_config.hf_config\n"
)
new = (
    "        model_config = vllm_config.model_config\n"
    "        quant_config = vllm_config.quant_config\n"
    "\n"
    "        " + MARKER + "\n"
    "        # MTP checkpoint weights are plain BF16 (no pre-quantized NVFP4 tensors).\n"
    "        # Exclude all mtp.* layers from NVFP4 quantization so they run in BF16.\n"
    "        if quant_config is not None and hasattr(quant_config, 'exclude_modules'):\n"
    "            if 'mtp.' not in quant_config.exclude_modules:\n"
    "                quant_config.exclude_modules.append('mtp.')\n"
    "                logger.info(\n"
    "                    'MTP: added mtp. to quant_config.exclude_modules '\n"
    "                    '-> all MTP sub-layers will use unquantized BF16.')\n"
    "\n"
    "        config: Qwen3NextConfig = model_config.hf_config\n"
)

if old not in src:
    print("ERROR: anchor not found in qwen3_next_mtp.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched qwen3_next_mtp.py: MTP layers excluded from NVFP4 quantization (BF16 path).")
PYEOF

# 10. MTP quant exclusion fix - qwen3_5_mtp.py (Qwen3_5MoeMTP class)
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_mtp.py"
with open(path) as f:
    src = f.read()

MARKER = "# [mtp_quant_exclusion_fix_qwen35]"
if MARKER in src:
    print("qwen3_5_mtp.py: MTP quant exclusion fix already applied, skipping.")
    sys.exit(0)

old = (
    "        model_config = vllm_config.model_config\n"
    "        quant_config = vllm_config.quant_config\n"
    "\n"
    "        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = model_config.hf_text_config\n"
)
new = (
    "        model_config = vllm_config.model_config\n"
    "        quant_config = vllm_config.quant_config\n"
    "\n"
    "        " + MARKER + "\n"
    "        # MTP checkpoint weights are plain BF16 (no pre-quantized NVFP4 tensors).\n"
    "        # Exclude all mtp.* layers from NVFP4 quantization so they run in BF16.\n"
    "        if quant_config is not None and hasattr(quant_config, 'exclude_modules'):\n"
    "            if 'mtp.' not in quant_config.exclude_modules:\n"
    "                quant_config.exclude_modules.append('mtp.')\n"
    "                logger.info(\n"
    "                    'MTP: added mtp. to quant_config.exclude_modules '\n"
    "                    '-> all MTP sub-layers will use unquantized BF16.')\n"
    "\n"
    "        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = model_config.hf_text_config\n"
)

if old not in src:
    print("ERROR: anchor not found in qwen3_5_mtp.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched qwen3_5_mtp.py: MTP layers excluded from NVFP4 quantization (BF16 path).")
PYEOF

# 11. MRotaryEmbedding forward_native() -- narrow positions to query.shape[0]
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/rotary_embedding/mrope.py"
with open(path) as f:
    src = f.read()

MARKER = "# [mrope_positions_narrow_fix]"
if MARKER in src:
    print("mrope.py: already patched, skipping.")
    sys.exit(0)

old = (
    "        num_tokens = positions.shape[-1]\n"
    "        cos_sin = cos_sin_cache[positions]\n"
)
new = (
    "        " + MARKER + "\n"
    "        # Use query.shape[0] for num_tokens, and narrow 2D positions BEFORE the\n"
    "        # cache lookup to avoid concrete shape propagation into cos/sin tensors.\n"
    "        num_tokens = query.shape[0]\n"
    "        if positions.ndim == 2:\n"
    "            positions = positions[:, :num_tokens]\n"
    "        cos_sin = cos_sin_cache[positions]\n"
)

if old not in src:
    print("ERROR: anchor not found in mrope.py forward_native", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched mrope.py: positions narrowed to query.shape[0] before cache lookup.")
PYEOF

# 12. Eagle _get_positions() -- return full mrope buffer for compiled compatibility
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py"
with open(path) as f:
    src = f.read()

MARKER = "# [mrope_full_buffer_fix]"
if MARKER in src:
    print("eagle.py: _get_positions max-size fix already applied, skipping.")
    sys.exit(0)

old = (
    "    def _get_positions(self, num_tokens: int):\n"
    "        if self.uses_mrope:\n"
    "            return self.mrope_positions[:, :num_tokens]\n"
    "        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:\n"
    "            return self.xdrope_positions[:, :num_tokens]\n"
    "        return self.positions[:num_tokens]\n"
)
new = (
    "    def _get_positions(self, num_tokens: int):\n"
    "        " + MARKER + "\n"
    "        # For MRoPE, always return the max-size slice mrope_positions[:, :max_num_tokens]\n"
    "        # so compiled eagle_head's assert_size_stride always passes.\n"
    "        # mrope.py narrows positions[:, :query.shape[0]] dynamically.\n"
    "        if self.uses_mrope:\n"
    "            return self.mrope_positions[:, :self.max_num_tokens]\n"
    "        if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:\n"
    "            return self.xdrope_positions[:, :num_tokens]\n"
    "        return self.positions[:num_tokens]\n"
)

if old not in src:
    print("ERROR: _get_positions anchor not found in eagle.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched eagle.py: _get_positions() returns mrope_positions[:, :max_num_tokens] for MRoPE.")
PYEOF

# 13. SpecDecoding metrics -- guard against negative counter values
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/metrics.py"
with open(path) as f:
    src = f.read()

MARKER = "# [negative_counter_guard]"
if MARKER in src:
    print("spec_decode/metrics.py: negative counter guard already applied, skipping.")
    sys.exit(0)

old = (
    "    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):\n"
    "        if not self.spec_decoding_enabled:\n"
    "            return\n"
    "        self.counter_spec_decode_num_drafts[engine_idx].inc(\n"
    "            spec_decoding_stats.num_drafts\n"
    "        )\n"
    "        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(\n"
    "            spec_decoding_stats.num_draft_tokens\n"
    "        )\n"
    "        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(\n"
    "            spec_decoding_stats.num_accepted_tokens\n"
    "        )\n"
    "        for pos, counter in enumerate(\n"
    "            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]\n"
    "        ):\n"
    "            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])\n"
)
new = (
    "    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):\n"
    "        " + MARKER + "\n"
    "        # Guard all counter increments with max(0, value) to prevent\n"
    "        # ValueError when num_accepted_tokens is negative.\n"
    "        if not self.spec_decoding_enabled:\n"
    "            return\n"
    "        self.counter_spec_decode_num_drafts[engine_idx].inc(\n"
    "            max(0, spec_decoding_stats.num_drafts)\n"
    "        )\n"
    "        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(\n"
    "            max(0, spec_decoding_stats.num_draft_tokens)\n"
    "        )\n"
    "        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(\n"
    "            max(0, spec_decoding_stats.num_accepted_tokens)\n"
    "        )\n"
    "        for pos, counter in enumerate(\n"
    "            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]\n"
    "        ):\n"
    "            counter.inc(max(0, spec_decoding_stats.num_accepted_tokens_per_pos[pos]))\n"
)

if old not in src:
    print("ERROR: observe() anchor not found in spec_decode/metrics.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched spec_decode/metrics.py: negative counter guard added.")
PYEOF

# 14. FlashInfer autotuner -- config path to cache volume + exec()-based loader
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/flashinfer/autotuner.py"
with open(path) as f:
    src = f.read()

MARKER = "# [gb10_config_path_fix]"
if MARKER in src:
    print("autotuner.py: config path fix already applied, skipping.")
    sys.exit(0)

old_gcp = (
    'def get_config_path(is_module: bool):\n'
    '    dev_name = torch.cuda.get_device_name(0).replace(" ", "_")\n'
    '    cutlass_ver = _nvfp4_cutlass_version.replace(".", "_")\n'
    '    config_name = f"v{cutlass_ver}_trtllm_fused_moe_{dev_name}"\n'
    '    if is_module:\n'
    '        return f"flashinfer.tuning_configs.{config_name}"\n'
    '    else:\n'
    '        return os.path.join(\n'
    '            os.path.dirname(os.path.realpath(__file__)),\n'
    '            "tuning_configs",\n'
    '            config_name + ".py",\n'
    '        )'
)
new_gcp = (
    MARKER + '\n'
    'def get_config_path(is_module: bool):\n'
    '    dev_name = torch.cuda.get_device_name(0).replace(" ", "_")\n'
    '    cutlass_ver = _nvfp4_cutlass_version.replace(".", "_")\n'
    '    config_name = f"v{cutlass_ver}_trtllm_fused_moe_{dev_name}"\n'
    '    # Always return file path (cache volume); is_module ignored.\n'
    '    cache_dir = os.path.join(\n'
    '        os.environ.get("FLASHINFER_CACHE_DIR",\n'
    '                        os.path.expanduser("~/.cache/flashinfer")),\n'
    '        "tuning_configs",\n'
    '    )\n'
    '    return os.path.join(cache_dir, config_name + ".py")'
)

if old_gcp not in src:
    print("ERROR: get_config_path anchor not found in autotuner.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_gcp, new_gcp, 1)

old_lff = (
    '@lru_cache(maxsize=None)\n'
    'def load_from_file(key):\n'
    '    module_name = get_config_path(is_module=True)\n'
    '    try:\n'
    '        module = importlib.import_module(module_name)\n'
    '        best_configs = module.best_configs\n'
    '    except (ImportError, AttributeError):\n'
    '        best_configs = None\n'
    '    if best_configs is not None:\n'
    '        k = str((key[0], key[1], key[3]))\n'
    '        if k in best_configs:\n'
    '            logger.info(f"[Autotuner]: Loading configs for {k} from file.")\n'
    '            return True, best_configs[k][0], best_configs[k][1], None\n'
    '    logger.info(\n'
    '        f"[Autotuner]: Loading configs for {key} from file failed; Using default configs instead."\n'
    '    )\n'
    '    return False, 0, -1, None'
)
new_lff = (
    '@lru_cache(maxsize=None)\n'
    'def load_from_file(key):\n'
    '    config_file = get_config_path(is_module=False)\n'
    '    best_configs = None\n'
    '    if os.path.isfile(config_file):\n'
    '        try:\n'
    '            ns = {}\n'
    '            with open(config_file) as f:\n'
    '                exec(f.read(), ns)\n'
    '            best_configs = ns.get("best_configs")\n'
    '        except Exception:\n'
    '            best_configs = None\n'
    '    if best_configs is not None:\n'
    '        k = str((key[0], key[1], key[3]))\n'
    '        if k in best_configs:\n'
    '            logger.info(f"[Autotuner]: Loading configs for {k} from {config_file}.")\n'
    '            return True, best_configs[k][0], best_configs[k][1], None\n'
    '    logger.info(\n'
    '        f"[Autotuner]: Loading configs for {key} from file failed; Using default configs instead."\n'
    '    )\n'
    '    return False, 0, -1, None'
)

if old_lff not in src:
    print("ERROR: load_from_file anchor not found in autotuner.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_lff, new_lff, 1)

with open(path, "w") as f:
    f.write(src)
print("Patched autotuner.py: config path -> cache volume, load_from_file -> exec()-based.")
PYEOF

# 15. kernel_warmup.py -- dump profiling_cache after FlashInfer autotune
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py"
with open(path) as f:
    src = f.read()

MARKER = "# [flashinfer_autotune_cache_dump]"
if MARKER in src:
    print("kernel_warmup.py: autotune cache dump already applied, skipping.")
    sys.exit(0)

old = (
    'def flashinfer_autotune(runner: "GPUModelRunner") -> None:\n'
    '    """\n'
    '    Autotune FlashInfer operations.\n'
    '    FlashInfer have many implementations for the same operation,\n'
    '    autotuning runs benchmarks for each implementation and stores\n'
    '    the results. The results are cached transparently and\n'
    '    future calls to FlashInfer will use the best implementation.\n'
    '    Without autotuning, FlashInfer will rely on heuristics, which may\n'
    '    be significantly slower.\n'
    '    """\n'
    '    from vllm.utils.flashinfer import autotune\n'
    '\n'
    '    with torch.inference_mode(), autotune():\n'
    '        # We skip EPLB here since we don\'t want to record dummy metrics\n'
    '        # When autotuning with number of tokens m, flashinfer will autotune\n'
    '        # operations for all number of tokens up to m.\n'
    '        # So we only need to run with the max number of tokens.\n'
    '        runner._dummy_run(\n'
    '            runner.scheduler_config.max_num_batched_tokens,\n'
    '            skip_eplb=True,\n'
    '            is_profile=True,\n'
    '        )'
)
new = (
    'def flashinfer_autotune(runner: "GPUModelRunner") -> None:\n'
    '    """\n'
    '    Autotune FlashInfer operations.\n'
    '    FlashInfer have many implementations for the same operation,\n'
    '    autotuning runs benchmarks for each implementation and stores\n'
    '    the results. The results are cached transparently and\n'
    '    future calls to FlashInfer will use the best implementation.\n'
    '    Without autotuning, FlashInfer will rely on heuristics, which may\n'
    '    be significantly slower.\n'
    '    """\n'
    '    from vllm.utils.flashinfer import autotune\n'
    '\n'
    '    with torch.inference_mode(), autotune():\n'
    '        # We skip EPLB here since we don\'t want to record dummy metrics\n'
    '        # When autotuning with number of tokens m, flashinfer will autotune\n'
    '        # operations for all number of tokens up to m.\n'
    '        # So we only need to run with the max number of tokens.\n'
    '        runner._dummy_run(\n'
    '            runner.scheduler_config.max_num_batched_tokens,\n'
    '            skip_eplb=True,\n'
    '            is_profile=True,\n'
    '        )\n'
    '\n'
    '    ' + MARKER + '\n'
    '    # Dump profiling_cache to flashinfer tuning_configs so subsequent starts\n'
    '    # can load via FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1 and skip re-profiling.\n'
    '    try:\n'
    '        from flashinfer.autotuner import AutoTuner, get_config_path\n'
    '        cache = AutoTuner.get().profiling_cache\n'
    '        if cache:\n'
    '            config_path = get_config_path(is_module=False)\n'
    '            import os\n'
    '            os.makedirs(os.path.dirname(config_path), exist_ok=True)\n'
    '            lines = ["best_configs = {"]\n'
    '            for key, (runner_id, tactic, _profile) in sorted(cache.items(), key=str):\n'
    '                k = str((key[0], key[1], key[3]))\n'
    '                lines.append(f"    {k!r}: ({runner_id}, {tactic}),")\n'
    '            lines.append("}")\n'
    '            with open(config_path, "w") as f:\n'
    '                f.write("\\n".join(lines) + "\\n")\n'
    '            logger.info(\n'
    '                "flashinfer autotune: saved %d entries to %s",\n'
    '                len(cache), config_path,\n'
    '            )\n'
    '            # Enable file-based loading for this process too\n'
    '            os.environ["FLASHINFER_AUTOTUNER_LOAD_FROM_FILE"] = "1"\n'
    '    except Exception as e:\n'
    '        logger.warning("flashinfer autotune cache dump failed: %s", e)'
)

if old not in src:
    print("ERROR: flashinfer_autotune anchor not found in kernel_warmup.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)

with open(path, "w") as f:
    f.write(src)
print("Patched kernel_warmup.py: autotune results dump to cache volume after warmup.")
PYEOF

EXPOSE 8000
