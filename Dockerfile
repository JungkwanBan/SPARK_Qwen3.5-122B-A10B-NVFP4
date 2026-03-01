# =========================================================
# vLLM for Sehyo/Qwen3.5-122B-A10B-NVFP4 (DGX Spark / SM121)
#
# Base: vllm-mxfp4-spark:latest
#   - SM121/GB10 최적화 flashinfer-cutlass + NVFP4 지원
#   - 최신 nightly 베이스 이미지 빌드:
#     cd ../spark-vllm-docker && ./build-and-copy.sh --exp-mxfp4 --rebuild-vllm -t vllm-mxfp4-spark:latest
# =========================================================
FROM vllm-mxfp4-spark:latest

# Qwen3.5 VL MoE 모델 지원을 위한 transformers 버전 고정
RUN uv pip install transformers==5.2.0 --upgrade --no-deps --system
RUN uv pip install huggingface-hub --upgrade --no-deps --system

# Upgrade vLLM to latest nightly (cu130 wheels, compatible with cu131 runtime, aarch64)
# --no-deps: keep base image's torch/flashinfer/CUDA libs (SM121-specific builds)
# gdn_attention_core custom op is NOT in these wheels; bypassed via patch below.
# NOTE: nightly 버전은 주기적으로 purge됨. 빌드 실패 시 최신 버전으로 업데이트:
#   pip install --dry-run --no-deps vllm --index-url https://wheels.vllm.ai/nightly/cu130
RUN pip install 'vllm==0.16.1rc1.dev75+ge3691988d.cu130' \
    --index-url https://wheels.vllm.ai/nightly/cu130 \
    --no-deps --quiet

# Fix: vLLM nightly qwen3_5_moe config passes ignore_keys_at_rope_validation as list,
# but transformers 5.2.0 expects a set (uses | operator for union).
# Convert list to set literal to fix: TypeError: unsupported operand type(s) for |: 'list' and 'set'
RUN python3 - <<'EOF'
path = "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/qwen3_5_moe.py"
with open(path) as f:
    src = f.read()
old = (
    '        kwargs["ignore_keys_at_rope_validation"] = [\n'
    '            "mrope_section",\n'
    '            "mrope_interleaved",\n'
    '        ]\n'
)
new = (
    '        kwargs["ignore_keys_at_rope_validation"] = {\n'
    '            "mrope_section",\n'
    '            "mrope_interleaved",\n'
    '        }\n'
)
if old not in src:
    print("ignore_keys already a set or anchor not found, skipping.")
else:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("Fixed: ignore_keys_at_rope_validation list -> set in qwen3_5_moe.py")
EOF

# Qwen3.5 VL MoE 네이티브 vLLM 모델 클래스 추가
COPY qwen3_5_vl_moe.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_vl_moe.py

# Override the nightly registry entry for Qwen3_5MoeForConditionalGeneration
# to point to our custom model class (needed for GDN Triton FLA support)
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

# Fix: GDN layers use Triton FLA kernels (fused_recurrent_gated_delta_rule,
# chunk_gated_delta_rule, fused_gdn_gating) that require a runtime Triton
# memory allocator for scratch buffers.  vLLM only sets this in matmul_ogs.py
# for MoE matmul; GDN linear-attention layers need it too.  Without it, the
# custom op's eager execution crashes with:
#   RuntimeError: Kernel requires a runtime memory allocation, but no allocator was set.
#
# The nightly registers gdn_attention_core via direct_register_custom_op (Python,
# not C++), so it works as a splitting_op for torch.compile.  We keep the original
# torch.ops.vllm.gdn_attention_core() call intact — during tracing dynamo uses the
# fake impl (no-op), and during eager execution the real impl calls _forward_core()
# which now sets the Triton allocator.
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    src = f.read()

MARKER = "# [gdn_triton_allocator_fix]"
if MARKER in src:
    print("qwen3_next.py: GDN Triton allocator fix already applied, skipping.")
    sys.exit(0)

# 1. Add allocator class after logger definition
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

# 2. Call triton.set_allocator() at start of _forward_core.
#    The gdn_attention_core custom op is in splitting_ops, so _forward_core runs
#    in eager mode (not compiled by dynamo).  triton.set_allocator uses ContextVar
#    which is fine in eager mode.
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

# Fix: NVFP4 CUTLASS MoE kernel occasionally produces NaN values during prefill
# when processing GDN-derived activations (specifically at layer 8 in the
# Qwen3.5-122B-A10B-NVFP4 model).  Without a guard these NaN values propagate
# through all subsequent layers, causing argmax(logits) == 0 ("!") for every
# output token.
#
# Fix: add a torch.nan_to_num guard on the MoE FFN output inside
# Qwen3NextDecoderLayer.forward(), but ONLY for linear_attention (GDN) layers.
# full_attention layers (12/48) skip the guard entirely.
# self.layer_type is a construction-time constant already branched on in
# forward(), so torch.compile / dynamo handles it correctly (graph specialization).
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

# Fix: Qwen3NextMTP.remap_weight_names() — strip "language_model." prefix from VL model weights
# so that embed_tokens / lm_head are found correctly during weight-sharing.
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

# Fix: @support_torch_compile on Qwen3NextMultiTokenPredictor — positions marked
# dynamic on dim=-1, shape_invariants updated (no n==p check).
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
    "# (shape (3, max_tokens+1)) is passed — larger than the actual token count.\n"
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

# Fix: Qwen3NextMultiTokenPredictor receives the shared NVFP4 quant_config, which
# would apply ModelOptNvFp4LinearMethod to mtp.fc and all MTP sub-layers.
# BUT the MTP checkpoint weights are plain BF16 (no pre-quantized NVFP4 tensors),
# so the NVFP4 linear forward produces zeros/NaN.
#
# Fix: add "mtp." to quant_config.exclude_modules BEFORE any sub-layer is created.
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
    "        # This must happen BEFORE any sub-layer is constructed so that\n"
    "        # get_quant_method() sees the updated exclude_modules list.\n"
    "        if quant_config is not None and hasattr(quant_config, 'exclude_modules'):\n"
    "            if 'mtp.' not in quant_config.exclude_modules:\n"
    "                quant_config.exclude_modules.append('mtp.')\n"
    "                logger.info(\n"
    "                    'MTP: added mtp. to quant_config.exclude_modules '\n"
    "                    '→ all MTP sub-layers will use unquantized BF16.')\n"
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

# Fix: Same quant exclusion fix for qwen3_5_mtp.py (Qwen3_5MoeMTP class).
# The nightly vLLM routes Qwen3.5-MoE MTP through qwen3_5_mtp.py, not qwen3_next_mtp.py.
# Without this, BF16 MTP weights hit NVFP4 ColumnParallelLinear → shape mismatch → AssertionError.
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
    "        # This must happen BEFORE any sub-layer is constructed so that\n"
    "        # get_quant_method() sees the updated exclude_modules list.\n"
    "        if quant_config is not None and hasattr(quant_config, 'exclude_modules'):\n"
    "            if 'mtp.' not in quant_config.exclude_modules:\n"
    "                quant_config.exclude_modules.append('mtp.')\n"
    "                logger.info(\n"
    "                    'MTP: added mtp. to quant_config.exclude_modules '\n"
    "                    '→ all MTP sub-layers will use unquantized BF16.')\n"
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

# Fix: MRotaryEmbedding.forward_native() — narrow 2D positions to query.shape[0]
# BEFORE the cache lookup to avoid concrete shape propagation.
# (nightly: local var `cos_sin_cache` instead of `self.cos_sin_cache`)
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

# Fix: eagle.py _get_positions() — return mrope_positions[:, :max_num_tokens]
# (fixed-size buffer) so compiled assert_size_stride always passes.
# (nightly: xdrope branch added between mrope and default return)
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
    "        # (non-contiguous, shape (3, max_num_tokens)) regardless of num_tokens.\n"
    "        # The compiled eagle_head's assert_size_stride(positions, (s80, max_num_tokens), ...)\n"
    "        # always passes. mrope.py narrows positions[:, :query.shape[0]] (dynamic)\n"
    "        # so cos/sin are computed only for the actual N tokens in the batch.\n"
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

# Fix: SpecDecodingProm.observe() can receive negative num_accepted_tokens.
# Guard all counter increments with max(0, value).
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
    "        # ValueError when num_accepted_tokens is negative (e.g. when\n"
    "        # len(generated_token_ids)==0 due to request abort or early EOS).\n"
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
print("Patched spec_decode/metrics.py: negative counter guard added to SpecDecodingProm.observe().")
PYEOF

# Fix: flashinfer autotuner re-profiles all GEMM tactics on every startup because
# profiling_cache is in-memory only.  On SM12x, the TRT-LLM CUTLASS tactic fails
# 1,500+ times per startup (Ninja build failure for delayStream.cu).
#
# Two-part fix:
# 1. Patch autotuner.py: redirect get_config_path to the flashinfer cache volume
#    (/root/.cache/flashinfer/tuning_configs/) so results survive container recreation.
#    Patch load_from_file to use exec() for file-based loading (no importlib needed).
# 2. Patch kernel_warmup.py: dump profiling_cache after autotuning completes.
#    With FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1, subsequent starts load from file.

# Part 1: Patch autotuner.py — config path + file loader
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/flashinfer/autotuner.py"
with open(path) as f:
    src = f.read()

MARKER = "# [gb10_config_path_fix]"
if MARKER in src:
    print("autotuner.py: config path fix already applied, skipping.")
    sys.exit(0)

# 1a. Patch get_config_path to use cache volume
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

# 1b. Patch load_from_file to use exec() instead of importlib.import_module
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
print("Patched autotuner.py: config path → cache volume, load_from_file → exec()-based.")
PYEOF

# Part 2: Patch kernel_warmup.py — dump profiling_cache after autotune
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
