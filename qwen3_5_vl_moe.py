# SPDX-License-Identifier: Apache-2.0
"""Inference-only Qwen3.5 VL MoE (Qwen3_5MoeForConditionalGeneration).

Uses Qwen3Next GDN (Gated Delta Network) for hybrid linear/full attention layers.
"""
from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, MixtureOfExperts
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextRMSNorm,
    Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _patch_text_config(text_config):
    """Add fields expected by Qwen3Next but absent from Qwen3_5MoeTextConfig."""
    # All 48 layers in Qwen3.5-122B-A10B are MoE → step=1
    if not hasattr(text_config, "decoder_sparse_step"):
        text_config.decoder_sparse_step = 1
    # Dense MLP fallback (never used when decoder_sparse_step=1)
    if not hasattr(text_config, "intermediate_size"):
        text_config.intermediate_size = text_config.moe_intermediate_size
    # Top-k prob normalisation (Qwen3Next default: True)
    if not hasattr(text_config, "norm_topk_prob"):
        text_config.norm_topk_prob = True
    if not hasattr(text_config, "mlp_only_layers"):
        text_config.mlp_only_layers = []
    if not hasattr(text_config, "layer_scale"):
        text_config.layer_scale = False
    # dtype used by GDN conv1d initialisation
    if not hasattr(text_config, "dtype"):
        text_config.dtype = None
    # MTP speculative decoding: Qwen3NextMultiTokenPredictor reads this field.
    # Qwen3.5MoeTextConfig stores it as mtp_num_hidden_layers.
    if not hasattr(text_config, "num_nextn_predict_layers"):
        text_config.num_nextn_predict_layers = getattr(
            text_config, "mtp_num_hidden_layers", 1
        )
    return text_config


class _ModelConfigProxy:
    """Proxy around ModelConfig that returns text_config as hf_config."""

    def __init__(self, orig_model_config, text_config):
        object.__setattr__(self, "_orig", orig_model_config)
        object.__setattr__(self, "_hf_config", text_config)

    @property
    def hf_config(self):
        return object.__getattribute__(self, "_hf_config")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_orig"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_orig"), name, value)


class _TextConfigVllmConfig:
    """Proxy VllmConfig whose model_config.hf_config returns the text_config.

    This allows reusing Qwen3NextDecoderLayer (and its sub-components) which
    read config fields from vllm_config.model_config.hf_config directly.
    """

    def __init__(self, vllm_config: VllmConfig, text_config):
        object.__setattr__(self, "_vc", vllm_config)
        object.__setattr__(
            self,
            "_mc_proxy",
            _ModelConfigProxy(vllm_config.model_config, text_config),
        )

    @property
    def model_config(self):
        return object.__getattribute__(self, "_mc_proxy")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_vc"), name)


# ---------------------------------------------------------------------------
# Language model (hybrid GDN + full-attention + MoE)
# ---------------------------------------------------------------------------

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3_5VLMoeLLMModel(nn.Module):
    """Qwen3.5 text backbone: 48 hybrid layers (36 GDN + 12 full-attn), all MoE."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        text_config = _patch_text_config(hf_config.text_config)
        self.config = text_config

        # Proxy so Qwen3NextDecoderLayer sub-classes see text_config via hf_config
        tc_vllm_config = _TextConfigVllmConfig(vllm_config, text_config)

        self.vocab_size = text_config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            text_config.hidden_size,
        )

        eplb_config = vllm_config.parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                tc_vllm_config,
                layer_type=text_config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            text_config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], text_config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(
                text_config.hidden_size, eps=text_config.rms_norm_eps
            )
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None \
                else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        from vllm.model_executor.layers.fused_moe import SharedFusedMoE
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        # Buffer for GDN split projections that must be fused before loading:
        #   in_proj_qkv + in_proj_z  → in_proj_qkvz  (Q+K+V concat Z/gate)
        #   in_proj_b   + in_proj_a  → in_proj_ba     (beta concat alpha)
        # Key: vLLM fused param name, Value: dict of {part_name: tensor}
        gdn_buf: dict[str, dict[str, torch.Tensor]] = {}

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("mtp."):
                continue

            # --- Bulk fused expert format: gate_up_proj / down_proj ---
            # Checkpoint stores all-expert weights as single tensors:
            #   experts.gate_up_proj [N_experts, 2*intermediate, hidden]
            #   experts.down_proj    [N_experts, hidden, intermediate]
            # These map directly to vLLM's w13_weight / w2_weight.
            if "mlp.experts.gate_up_proj" in name:
                vllm_name = name.replace(
                    "mlp.experts.gate_up_proj", "mlp.experts.w13_weight"
                )
                if not is_pp_missing_parameter(vllm_name, self):
                    if vllm_name in params_dict:
                        default_weight_loader(params_dict[vllm_name], loaded_weight)
                        loaded_params.add(vllm_name)
                continue
            if "mlp.experts.down_proj" in name:
                vllm_name = name.replace(
                    "mlp.experts.down_proj", "mlp.experts.w2_weight"
                )
                if not is_pp_missing_parameter(vllm_name, self):
                    if vllm_name in params_dict:
                        default_weight_loader(params_dict[vllm_name], loaded_weight)
                        loaded_params.add(vllm_name)
                continue

            # --- GDN split projections: buffer for deferred fusion ---
            if "linear_attn.in_proj_" in name:
                sep = ".in_proj_"
                idx = name.index(sep)
                base = name[:idx]  # e.g. "layers.0.linear_attn"
                # Determine fused param name and part key
                if "in_proj_qkv." in name or "in_proj_z." in name:
                    fused_key = base + ".in_proj_qkvz.weight"
                    part = "qkv" if "in_proj_qkv." in name else "z"
                elif "in_proj_a." in name or "in_proj_b." in name:
                    fused_key = base + ".in_proj_ba.weight"
                    part = "a" if "in_proj_a." in name else "b"
                else:
                    # Unknown in_proj_* variant — fall through to default path
                    fused_key = None
                    part = None

                if fused_key is not None:
                    gdn_buf.setdefault(fused_key, {})[part] = loaded_weight
                    continue

            # --- stacked projections (QKV, gate_up) ---
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                # Remap KV cache scale names: k_proj.k_scale / v_proj.v_scale
                # → attn.k_scale / attn.v_scale  (ModelOpt NVFP4 format)
                if "scale" in name:
                    remapped = maybe_remap_kv_scale_name(name, params_dict)
                    if remapped is None:
                        break  # cannot resolve → skip weight entirely
                    name = remapped
                if name not in params_dict:
                    break  # resolved name absent → skip weight entirely
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader is default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # --- expert weights (gate_proj / up_proj / down_proj + scales) ---
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param, loaded_weight, name,
                        shard_id=shard_id, expert_id=expert_id,
                    )
                    break
                else:
                    # --- default / scalar weights ---
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remap KV cache scale names in the default path too
                    if "scale" in name:
                        remapped = maybe_remap_kv_scale_name(name, params_dict)
                        if remapped is None:
                            continue
                        name = remapped
                    if name not in params_dict:
                        logger.warning_once(
                            "Parameter %s not found in model, skipping.", name
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # --- Deferred GDN fusion: fuse split projections and load ---
        # The HF checkpoint stores GDN projections in FLAT format:
        #   in_proj_qkv:  [Q_all(key_dim), K_all(key_dim), V_all(value_dim)]
        #   in_proj_z:    [Z_all(value_dim)]
        #   in_proj_b/a:  [B_all(num_v_heads)], [A_all(num_v_heads)]
        #
        # vLLM's fix_query_key_value_ordering() reshapes the projection output
        # to [T, num_k_heads, per_group_dim] and splits into Q/K/V/Z per group.
        # For this reshape to be correct, the weight rows must be in
        # GROUP-INTERLEAVED format:
        #   in_proj_qkvz: [Q_g0,K_g0,V_g0,Z_g0, Q_g1,K_g1,V_g1,Z_g1, ...]
        #   in_proj_ba:   [B_g0,A_g0, B_g1,A_g1, ...]
        # where g0..g(num_k_heads-1) are the key-head groups.
        tc = self.config
        _nkh = tc.linear_num_key_heads      # 16
        _hkd = tc.linear_key_head_dim       # 128
        _nvh = tc.linear_num_value_heads    # 64
        _hvd = tc.linear_value_head_dim     # 128
        _vpk = _nvh // _nkh                 # 4 (value heads per key-head group)
        _hs  = tc.hidden_size               # 3072

        for fused_name, parts in gdn_buf.items():
            if is_pp_missing_parameter(fused_name, self):
                continue
            if "in_proj_qkvz" in fused_name:
                if "qkv" not in parts or "z" not in parts:
                    logger.warning(
                        "GDN in_proj_qkvz incomplete for %s (parts: %s) — skipping.",
                        fused_name, list(parts.keys()),
                    )
                    continue
                # Flat → group-interleaved reorder
                qkv = parts["qkv"]   # (key_dim*2 + value_dim, hidden) = (12288, 3072)
                z_w = parts["z"]     # (value_dim, hidden)              = (8192,  3072)
                q_dim = _nkh * _hkd  # 2048
                k_dim = _nkh * _hkd  # 2048
                Q = qkv[:q_dim].view(_nkh, _hkd, _hs)                   # (16, 128, 3072)
                K = qkv[q_dim:q_dim + k_dim].view(_nkh, _hkd, _hs)      # (16, 128, 3072)
                V = qkv[q_dim + k_dim:].view(_nkh, _vpk * _hvd, _hs)    # (16, 512, 3072)
                Z = z_w.view(_nkh, _vpk * _hvd, _hs)                    # (16, 512, 3072)
                # Per group: [Q, K, V, Z] → shape (16, 1280, 3072) → (20480, 3072)
                fused_weight = torch.cat([Q, K, V, Z], dim=1).view(-1, _hs)
            elif "in_proj_ba" in fused_name:
                if "b" not in parts or "a" not in parts:
                    logger.warning(
                        "GDN in_proj_ba incomplete for %s (parts: %s) — skipping.",
                        fused_name, list(parts.keys()),
                    )
                    continue
                # Flat → group-interleaved reorder
                b_w = parts["b"]  # (num_v_heads, hidden) = (64, 3072)
                a_w = parts["a"]  # (num_v_heads, hidden) = (64, 3072)
                B = b_w.view(_nkh, _vpk, _hs)  # (16, 4, 3072)
                A = a_w.view(_nkh, _vpk, _hs)  # (16, 4, 3072)
                # Per group: [B, A] → shape (16, 8, 3072) → (128, 3072)
                fused_weight = torch.cat([B, A], dim=1).view(-1, _hs)
            else:
                continue

            if fused_name not in params_dict:
                logger.warning_once(
                    "GDN fused param %s not found in model, skipping.", fused_name
                )
                continue
            param = params_dict[fused_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, fused_weight)
            loaded_params.add(fused_name)
            # Debug: log first occurrence to confirm fusion succeeded
            if "layers.0." in fused_name:
                logger.info(
                    "GDN fused OK: %s shape=%s nonzero=%d",
                    fused_name, tuple(fused_weight.shape),
                    int(fused_weight.count_nonzero()),
                )

        return loaded_params


class Qwen3_5VLMoeLLMForCausalLM(nn.Module):
    """LM-head wrapper for Qwen3.5 VL MoE language model."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        text_config = vllm_config.model_config.hf_config.text_config
        self.config = text_config
        self.quant_config = vllm_config.quant_config

        self.model = Qwen3_5VLMoeLLMModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            text_config.vocab_size,
            text_config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(text_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(text_config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)


# ---------------------------------------------------------------------------
# Multi-modal processing info
# ---------------------------------------------------------------------------

class Qwen3_5MoeProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        # Nightly vLLM loads Qwen3_5MoeConfig via its own config class
        # (vllm.transformers_utils.configs.qwen3_5_moe), not from HF transformers.
        # Use whichever is actually loaded to avoid type mismatch in get_hf_config().
        try:
            from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig
            return self.ctx.get_hf_config(Qwen3_5MoeConfig)
        except (ImportError, TypeError):
            from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
                Qwen3_5MoeConfig,
            )
            return self.ctx.get_hf_config(Qwen3_5MoeConfig)


# ---------------------------------------------------------------------------
# MoE interface implementation
# ---------------------------------------------------------------------------

class _Qwen3_5MoeVLMixtureOfExperts(MixtureOfExperts):
    def set_moe_parameters(self):
        self.expert_weights = []
        self.moe_layers = []
        example_moe = None

        for layer in self.language_model.model.layers:
            if hasattr(layer, "mlp") and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3NextSparseMoeBlock found in language_model.model.layers."
            )

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(getattr(layer, "mlp", None), Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()


# ---------------------------------------------------------------------------
# Top-level VL model
# ---------------------------------------------------------------------------

@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoeForConditionalGeneration(
    Qwen3VLForConditionalGeneration,
    _Qwen3_5MoeVLMixtureOfExperts,
    HasInnerState,
    IsHybrid,
):
    """
    vLLM implementation of Qwen3.5-122B-A10B VL MoE.

    Architecture:
      - 48 hybrid decoder layers (36 GDN linear-attn + 12 full-attn)
      - Every layer has MoE FFN (256 experts, top-8, 1 shared expert)
      - NVFP4 weights + FP8 KV-cache
    """

    is_3d_moe_weight: bool = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    # Map HF checkpoint names → vLLM module names
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Skip Qwen3VLForConditionalGeneration.__init__ (wrong LM type)
        super(Qwen3VLForConditionalGeneration, self).__init__()

        try:
            from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeConfig
        except ImportError:
            from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
                Qwen3_5MoeConfig,
            )

        config: Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        # Vision encoder (optional: skip if no image/video slots)
        if (
            not multimodal_config.get_limit_per_prompt("image")
            and not multimodal_config.get_limit_per_prompt("video")
        ):
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        # Ensure mlp.gate (MoE router) is NOT quantized.
        # The NVFP4 checkpoint stores gate.weight as bfloat16 [num_experts, hidden]
        # but hf_quant_config.json omits mlp.gate from exclude_modules, causing a
        # shape mismatch when the FP4-packed uint8 param is loaded.
        # Pass 2 of is_layer_excluded() uses substring matching, so adding "mlp.gate"
        # is sufficient to exclude all layers whose prefix contains that string.
        if (
            hasattr(quant_config, "exclude_modules")
            and quant_config.exclude_modules is not None
            and "mlp.gate" not in quant_config.exclude_modules
        ):
            quant_config.exclude_modules.append("mlp.gate")
            logger.info(
                "Added 'mlp.gate' to quant_config.exclude_modules "
                "to keep MoE router weights in bfloat16."
            )

        # Ensure GDN linear_attn projections are NOT quantized.
        # The checkpoint stores in_proj_qkv/z/a/b as unquantized BF16 tensors
        # (they are in the ignore list as separate HF names).
        # vLLM's Qwen3NextGatedDeltaNet fuses them into in_proj_qkvz and
        # in_proj_ba — new names not in the checkpoint's ignore list.
        # Without this fix, NVFP4 quantization is applied to these fused layers,
        # but the checkpoint only has BF16 weights → load fails silently → zeros.
        if hasattr(quant_config, "ignore") and isinstance(quant_config.ignore, list):
            for pattern in [
                "re:.*linear_attn\\.in_proj_qkvz",
                "re:.*linear_attn\\.in_proj_ba",
            ]:
                if pattern not in quant_config.ignore:
                    quant_config.ignore.append(pattern)
            logger.info(
                "Added GDN in_proj patterns to quant_config.ignore "
                "to keep linear_attn projections in BF16."
            )

        # MTP (Multi-Token Prediction) draft head.
        # The MTP head is loaded separately by vLLM's EagleProposer as Qwen3NextMTP
        # (model_type="qwen3_next_mtp").  PPMissingLayer here absorbs any stray
        # mtp.* checkpoint weights silently so AutoWeightsLoader does not error.
        self.mtp = PPMissingLayer()

        # Provide image_token_index for EagleProposer multimodal handling.
        if not hasattr(config, "image_token_index"):
            config.image_token_index = getattr(config, "image_token_id", None)

        # Expose text_config attrs on the outer config so that
        # Qwen3NextMultiTokenPredictor (MTP drafter) can read them from
        # vllm_config.model_config.hf_config without finding MISSING fields.
        # Qwen3Next* layers read many attrs directly from config; copy them all.
        _tc = _patch_text_config(config.text_config)
        for _attr, _val in _tc.to_dict().items():
            if not _attr.startswith("_") and not hasattr(config, _attr):
                try:
                    setattr(config, _attr, _val)
                except (AttributeError, TypeError):
                    pass

        # Language model (hybrid GDN + full-attention + MoE)
        self.language_model = Qwen3_5VLMoeLLMForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Merge packed-modules mappings
        self.packed_modules_mapping = (
            self.packed_modules_mapping
            | self.language_model.packed_modules_mapping
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # DeepStack visual tokens (multi-scale vision features)
        self.use_deepstack = hasattr(
            config.vision_config, "deepstack_visual_indexes"
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        if self.use_deepstack and self.visual is not None:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None

        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        # Initialise MoE metadata
        self.set_moe_parameters()

    # ------------------------------------------------------------------
    # HasInnerState (Mamba / GDN recurrent state management)
    # ------------------------------------------------------------------

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        tc = vllm_config.model_config.hf_config.text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            tc.linear_num_key_heads,
            tc.linear_num_value_heads,
            tc.linear_key_head_dim,
            tc.linear_value_head_dim,
            tc.linear_conv_kernel_dim,
            num_spec,
        )

    # ------------------------------------------------------------------
    # Language model spec (used by AutoWeightsLoader)
    # ------------------------------------------------------------------

    @classmethod
    def get_language_model_spec(cls):
        return Qwen3_5VLMoeLLMForCausalLM, "language_model"
