import argparse
import os
from typing import List, Optional

# import implementations
from src.model.projectors.answer_projector import AnswerHeatmapProjectorConfig
from src.model.projectors.question_projector import QuestionHeatmapProjectorConfig
from src.model.config.model_config import CoExVQAConfig
from src.model.gating.config import GatingConfig


def csv_list(s: str) -> List[str]:
    """
    Turn a comma separated string into a list of strings.
    """
    return [t.strip() for t in s.split(",") if t.strip()]


def add_projector_arg_group(parser: argparse.ArgumentParser, prefix: str, *, is_answer: bool):
    """
    Adds a namespaced projector argument group.
    """
    g = parser.add_argument_group(f"{prefix.upper()} Projector")

    # Fusion
    g.add_argument(f"--{prefix}_fusion_modes", type=csv_list, default=csv_list("film"), help="Comma-separated fusion modes: concat,film,cross_attention")
    g.add_argument(f"--{prefix}_fusion_heads", type=int, default=8)
    g.add_argument(f"--{prefix}_fusion_dropout", type=float, default=0.10)
    g.add_argument(f"--{prefix}_concat_mlp_depth", type=int, default=1, choices=[1,2])
    g.add_argument(f"--{prefix}_concat_hidden_mult", type=float, default=1.0)
    g.add_argument(f"--{prefix}_concat_residual_scale", type=float, default=1.0)

    # Context aggregation
    g.add_argument(f"--{prefix}_context_mode", type=str, default="identity", choices=["identity", "transformer"])
    g.add_argument(f"--{prefix}_context_layers", type=int, default=1)
    g.add_argument(f"--{prefix}_context_heads", type=int, default=8)
    g.add_argument(f"--{prefix}_context_ffn_mult", type=int, default=4)
    g.add_argument(f"--{prefix}_context_dropout", type=float, default=0.10)
    g.add_argument(f"--{prefix}_context_norm_first", dest=f"{prefix}_context_norm_first", action="store_true")
    g.add_argument(f"--no-{prefix}_context_norm_first", dest=f"{prefix}_context_norm_first", action="store_false")
    parser.set_defaults(**{f"{prefix}_context_norm_first": True})
    g.add_argument(f"--{prefix}_context_layer_norm_eps", type=float, default=1e-5)
    g.add_argument(f"--{prefix}_context_activation", type=str, default="gelu", choices=["gelu", "relu"])

    # Projector
    default_proj_mode = "attn_pool" if is_answer else "identity"
    g.add_argument(f"--{prefix}_proj_mode", type=str, default=default_proj_mode, choices=["identity", "linear", "mlp", "mean_pool", "max_pool", "attn_pool"])
    g.add_argument(f"--{prefix}_proj_d_out", type=int, default=None)
    g.add_argument(f"--{prefix}_proj_dropout", type=float, default=0.10)
    g.add_argument(f"--{prefix}_proj_mlp_hidden_mult", type=float, default=2.0)
    g.add_argument(f"--{prefix}_proj_attn_heads", type=int, default=8)
    g.add_argument(f"--{prefix}_proj_attn_k", type=int, default=1)
    g.add_argument(f"--{prefix}_proj_eps", type=float, default=1e-6)

    # Spatial attention block
    g.add_argument(f"--{prefix}_spatial_attn_heads", type=int, default=8)
    g.add_argument(f"--{prefix}_spatial_attn_dropout", type=float, default=0.1)

    # Initial width and height for answer only
    if is_answer:
        g.add_argument(f"--{prefix}_init_w", type=float, default=0.07)
        g.add_argument(f"--{prefix}_init_h", type=float, default=0.05)


def parse_config_from_args(args: argparse.Namespace) -> CoExVQAConfig:
    """
    Parse a CoExVQAConfig from command-line arguments.

    Args:
        args: Parsed argument namespace

    Returns:
        CoExVQAConfig instance
    """
    # Answer projector config
    a_cfg = AnswerHeatmapProjectorConfig(
        d_model=args.d_model,
        # fusion
        fusion_modes=args.a_fusion_modes,
        fusion_block_heads=args.a_fusion_heads,
        fusion_dropout=args.a_fusion_dropout,
        fusion_concat_mlp_depth=args.a_concat_mlp_depth,
        fusion_concat_hidden_mult=args.a_concat_hidden_mult,
        fusion_concat_residual_scale=args.a_concat_residual_scale,

        # context
        context_agg_mode=args.a_context_mode,
        context_agg_num_layers=args.a_context_layers,
        context_agg_n_heads=args.a_context_heads,
        context_agg_ffn_mult=args.a_context_ffn_mult,
        context_agg_dropout=args.a_context_dropout,
        context_agg_norm_first=args.a_context_norm_first,
        context_agg_layer_norm_eps=args.a_context_layer_norm_eps,
        context_agg_activation=args.a_context_activation,

        # projector
        projector_mode=args.a_proj_mode,
        projector_d_out=args.a_proj_d_out,
        projector_dropout=args.a_proj_dropout,
        projector_mlp_hidden_mult=args.a_proj_mlp_hidden_mult,
        projector_attn_heads=args.a_proj_attn_heads,
        projector_attn_k=args.a_proj_attn_k,
        projector_eps=args.a_proj_eps,

        # spatial attention
        spatial_attn_heads=args.a_spatial_attn_heads,
        spatial_attn_dropout=args.a_spatial_attn_dropout,

        # initial width and height
        init_w=args.a_init_w,
        init_h=args.a_init_h
    )

    # Question projector config
    q_cfg = QuestionHeatmapProjectorConfig(
        d_model=args.d_model,

        # fusion
        fusion_modes=args.q_fusion_modes,
        fusion_block_heads=args.q_fusion_heads,
        fusion_dropout=args.q_fusion_dropout,
        fusion_concat_mlp_depth=args.q_concat_mlp_depth,
        fusion_concat_hidden_mult=args.q_concat_hidden_mult,
        fusion_concat_residual_scale=args.q_concat_residual_scale,

        # context
        context_agg_mode=args.q_context_mode,
        context_agg_num_layers=args.q_context_layers,
        context_agg_n_heads=args.q_context_heads,
        context_agg_ffn_mult=args.q_context_ffn_mult,
        context_agg_dropout=args.q_context_dropout,
        context_agg_norm_first=args.q_context_norm_first,
        context_agg_layer_norm_eps=args.q_context_layer_norm_eps,
        context_agg_activation=args.q_context_activation,

        # projector
        projector_mode=args.q_proj_mode,
        projector_d_out=args.q_proj_d_out,
        projector_dropout=args.q_proj_dropout,
        projector_mlp_hidden_mult=args.q_proj_mlp_hidden_mult,
        projector_attn_heads=args.q_proj_attn_heads,
        projector_attn_k=args.q_proj_attn_k,
        projector_eps=args.q_proj_eps,

        # spatial attention
        spatial_attn_heads=args.q_spatial_attn_heads,
        spatial_attn_dropout=args.q_spatial_attn_dropout,
    )

    # Gating config
    gate_cfg = GatingConfig(
        gating_type=args.gating_type,
        d_model=args.d_model,
        initial_alpha=args.gating_initial_alpha,
        hidden_mult=args.gating_hidden_mult,
        dropout=args.gating_dropout,
        normalize=args.gating_normalize,
        eps=args.gating_eps,
    )

    # Determine backbone model configuration
    import os
    backbone_arg = getattr(args, 'backbone_model', "google/pix2struct-docvqa-base") or "google/pix2struct-docvqa-base"
    if os.path.isdir(backbone_arg):
        backbone_model_id = "google/pix2struct-docvqa-base"
        backbone_local_path = backbone_arg
    else:
        backbone_model_id = backbone_arg
        backbone_local_path = None

    return CoExVQAConfig(
        d_model=args.d_model,
        backbone_model=backbone_model_id,
        backbone_local_path=backbone_local_path,
        backbone_freeze_mode=args.backbone_freeze_mode,
        answer_projector_config=a_cfg,
        question_projector_config=q_cfg,
        gate_config=gate_cfg,
        max_patches=args.max_patches,
        # Decoder options
        use_decoder=args.use_decoder,
        decoder_input_type=args.decoder_input_type,
        decoder_attention_type=args.decoder_attention_type,
        # Reencoding options
        bbox_mask_blur_sigma=args.bbox_mask_blur_sigma,
    )


def build_model_config(args: argparse.Namespace, d_model: Optional[int] = None) -> CoExVQAConfig:
    """Build a CoExVQAConfig from CLI args, applying a named preset when requested.

    When ``args.config`` is set, the preset acts as the base and CLI args
    override individual fields (backbone, decoder_attention_type,
    backbone_freeze_mode).  The ``d_model`` parameter is used to propagate the
    correct hidden dimension to all sub-configs when a non-local HuggingFace
    backbone is specified together with a preset — callers should resolve this
    via ``CoExVQA.d_model_for_backbone`` before calling this function.

    When ``args.config`` is None, the full config is parsed directly from args
    via :func:`parse_config_from_args`.
    """
    if args.config is not None:
        from src.model.config.presets import get_config_preset
        config = get_config_preset(args.config)

        if os.path.isdir(args.backbone_model):
            config.backbone_local_path = args.backbone_model
        else:
            config.backbone_model = args.backbone_model
            config.backbone_local_path = None
            if d_model is not None:
                config.d_model = d_model
                config.answer_projector_config.d_model = d_model
                config.question_projector_config.d_model = d_model
                config.gate_config.d_model = d_model

        config.decoder_attention_type = args.decoder_attention_type
        config.backbone_freeze_mode = args.backbone_freeze_mode
        assert config.use_decoder == args.use_decoder
    else:
        config = parse_config_from_args(args=args)

    return config
