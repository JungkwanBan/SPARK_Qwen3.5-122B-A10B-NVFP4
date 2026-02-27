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
# for MoE matmul; GDN linear-attention layers need it too.  Without it, eager
# mode crashes with:
#   RuntimeError: Kernel requires a runtime memory allocation, but no allocator was set.
#
# Additionally, the nightly vLLM refactored Qwen3NextGatedDeltaNet.forward() to
# use torch.ops.vllm.gdn_attention_core (a C++ custom op), but that op is NOT
# compiled in the nightly Python wheels (requires source build with SM121 support).
# Fix: bypass gdn_attention_core and call _forward_core() directly.
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

# 2. Bypass torch.ops.vllm.gdn_attention_core (not compiled in nightly wheels)
#    Replace the custom op call in forward() with a direct _forward_core() call.
old_op = (
    '        torch.ops.vllm.gdn_attention_core(\n'
    '            mixed_qkv,\n'
    '            b,\n'
    '            a,\n'
    '            core_attn_out,\n'
    '            self.prefix,\n'
    '        )\n'
)
new_op = (
    '        # [gdn_custom_op_bypass] gdn_attention_core not compiled in nightly wheels;\n'
    '        # call _forward_core() directly. With --enforce-eager (no torch.compile),\n'
    '        # triton.set_allocator inside _forward_core works without dynamo restrictions.\n'
    '        self._forward_core(mixed_qkv, b, a, core_attn_out)\n'
)

if old_op not in src:
    print("ERROR: gdn_attention_core call not found in qwen3_next.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_op, new_op, 1)

# 3. Call triton.set_allocator() at start of _forward_core
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
print("Applied GDN Triton allocator + gdn_attention_core bypass to qwen3_next.py.")
PYEOF

# Fix: NVFP4 CUTLASS MoE kernel occasionally produces NaN values during prefill
# when processing GDN-derived activations (specifically at layer 8 in the
# Qwen3.5-122B-A10B-NVFP4 model).  Without a guard these NaN values propagate
# through all subsequent layers, causing argmax(logits) == 0 ("!") for every
# output token.
#
# Fix: add a torch.nan_to_num guard on the MoE FFN output inside
# Qwen3NextDecoderLayer.forward().  The guard is unconditional (no data-dependent
# branch) so it is fully compatible with torch.compile / dynamo.
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
    '        # GDN-derived activations.  Replace unconditionally (no data-dependent\n'
    '        # branch) so this is fully compatible with torch.compile / dynamo.\n'
    '        hidden_states = hidden_states.nan_to_num(nan=0.0)\n'
    '\n'
    '        if self.layer_scale:'
)

if old_mlp not in src:
    print("ERROR: MLP output anchor not found in Qwen3NextDecoderLayer.forward()", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_mlp, new_mlp, 1)

with open(path, "w") as f:
    f.write(src)
print("Applied NaN guard to Qwen3NextDecoderLayer.forward() in qwen3_next.py.")
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
