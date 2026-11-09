from src.model.projectors.answer_projector import AnswerHeatmapProjectorConfig
from src.model.projectors.question_projector import QuestionHeatmapProjectorConfig
from src.model.config.model_config import CoExVQAConfig
from src.model.gating.config import GatingConfig


def get_common_question_config() -> QuestionHeatmapProjectorConfig:
    """
    Get the common question projector config used across all preset sizes.
    """
    return QuestionHeatmapProjectorConfig(
        d_model=768,
        # Fusion
        fusion_modes=["film", "cross_attention"],
        fusion_block_heads=8,
        fusion_dropout=0.10,
        fusion_concat_mlp_depth=1,
        fusion_concat_hidden_mult=1.0,
        fusion_concat_residual_scale=1.0,
        # Context aggregation
        context_agg_mode="transformer",
        context_agg_num_layers=2,
        context_agg_n_heads=8,
        context_agg_ffn_mult=4,
        context_agg_dropout=0.10,
        context_agg_norm_first=True,
        context_agg_layer_norm_eps=1e-5,
        context_agg_activation="gelu",
        # Projector
        projector_mode="mlp",
        projector_d_out=None,
        projector_dropout=0.10,
        projector_mlp_hidden_mult=2.0,
        projector_attn_heads=8,
        projector_attn_k=1,
        projector_eps=1e-6,
        # Spatial attention
        spatial_attn_heads=8,
        spatial_attn_dropout=0.1,
    )


def get_common_gating_config() -> GatingConfig:
    """
    Get the common gating config used across all preset sizes.
    """
    return GatingConfig(
        gating_type="film",
        d_model=768,
        initial_alpha=0.1,
        hidden_mult=1.0,
        dropout=0.1,
        normalize=True,
        eps=1e-8,
    )


def get_base_config() -> CoExVQAConfig:
    """
    Base configuration preset.
    """
    # Answer projector config - BASE size
    a_cfg = AnswerHeatmapProjectorConfig(
        d_model=768,
        # Fusion
        fusion_modes=["film", "cross_attention"],
        fusion_block_heads=8,
        fusion_dropout=0.10,
        fusion_concat_mlp_depth=1,
        fusion_concat_hidden_mult=1.0,
        fusion_concat_residual_scale=1.0,
        # Context aggregation
        context_agg_mode="transformer",
        context_agg_num_layers=2,
        context_agg_n_heads=8,
        context_agg_ffn_mult=4,
        context_agg_dropout=0.10,
        context_agg_norm_first=True,
        context_agg_layer_norm_eps=1e-5,
        context_agg_activation="gelu",
        # Projector
        projector_mode="attn_pool",
        projector_d_out=None,
        projector_dropout=0.10,
        projector_mlp_hidden_mult=2.0,
        projector_attn_heads=8,
        projector_attn_k=4,
        projector_eps=1e-6,
        # Spatial attention
        spatial_attn_heads=8,
        spatial_attn_dropout=0.1,
        # Initial dimensions
        init_w=0.65,
        init_h=0.22
    )

    # Question projector config (shared across all sizes)
    q_cfg = get_common_question_config()

    # Gating config (shared across all sizes)
    gate_cfg = get_common_gating_config()

    return CoExVQAConfig(
        d_model=768,
        backbone_model="google/pix2struct-docvqa-base",
        backbone_local_path=None,
        backbone_freeze_mode="decoder-unfrozen",
        answer_projector_config=a_cfg,
        question_projector_config=q_cfg,
        gate_config=gate_cfg,
        max_patches=512,
        use_decoder=True,
        decoder_attention_type="bbox_crop_reencode",
    )


def get_medium_config() -> CoExVQAConfig:
    """
    Medium configuration preset.
    """
    # Answer projector config - MEDIUM size
    a_cfg = AnswerHeatmapProjectorConfig(
        d_model=768,
        # Fusion
        fusion_modes=["film", "cross_attention"],
        fusion_block_heads=12,
        fusion_dropout=0.10,
        fusion_concat_mlp_depth=1,
        fusion_concat_hidden_mult=1.0,
        fusion_concat_residual_scale=1.0,
        # Context aggregation
        context_agg_mode="transformer",
        context_agg_num_layers=4,
        context_agg_n_heads=12,
        context_agg_ffn_mult=4,
        context_agg_dropout=0.10,
        context_agg_norm_first=True,
        context_agg_layer_norm_eps=1e-5,
        context_agg_activation="gelu",
        # Projector
        projector_mode="attn_pool",
        projector_d_out=None,
        projector_dropout=0.10,
        projector_mlp_hidden_mult=2.0,
        projector_attn_heads=12,
        projector_attn_k=8,
        projector_eps=1e-6,
        # Spatial attention
        spatial_attn_heads=12,
        spatial_attn_dropout=0.1,
        # Initial dimensions
        init_w=0.65,
        init_h=0.22
    )

    # Question projector config (shared across all sizes)
    q_cfg = get_common_question_config()

    # Gating config (shared across all sizes)
    gate_cfg = get_common_gating_config()

    return CoExVQAConfig(
        d_model=768,
        backbone_model="google/pix2struct-docvqa-base",
        backbone_local_path=None,
        backbone_freeze_mode="decoder-unfrozen",
        answer_projector_config=a_cfg,
        question_projector_config=q_cfg,
        gate_config=gate_cfg,
        max_patches=512,
        use_decoder=True,
        decoder_attention_type="bbox_crop_reencode",
    )


def get_large_config() -> CoExVQAConfig:
    """
    Large configuration preset.
    """
    # Answer projector config - LARGE size
    a_cfg = AnswerHeatmapProjectorConfig(
        d_model=768,
        # Fusion
        fusion_modes=["film", "cross_attention"],
        fusion_block_heads=16,
        fusion_dropout=0.10,
        fusion_concat_mlp_depth=1,
        fusion_concat_hidden_mult=1.0,
        fusion_concat_residual_scale=1.0,
        # Context aggregation
        context_agg_mode="transformer",
        context_agg_num_layers=8,
        context_agg_n_heads=16,
        context_agg_ffn_mult=4,
        context_agg_dropout=0.10,
        context_agg_norm_first=True,
        context_agg_layer_norm_eps=1e-5,
        context_agg_activation="gelu",
        # Projector
        projector_mode="attn_pool",
        projector_d_out=None,
        projector_dropout=0.10,
        projector_mlp_hidden_mult=2.0,
        projector_attn_heads=16,
        projector_attn_k=16,
        projector_eps=1e-6,
        # Spatial attention
        spatial_attn_heads=16,
        spatial_attn_dropout=0.1,
        # Initial dimensions
        init_w=0.65,
        init_h=0.22
    )

    # Question projector config (shared across all sizes)
    q_cfg = get_common_question_config()

    # Gating config (shared across all sizes)
    gate_cfg = get_common_gating_config()

    return CoExVQAConfig(
        d_model=768,
        backbone_model="google/pix2struct-docvqa-base",
        backbone_local_path=None,
        backbone_freeze_mode="decoder-unfrozen",
        answer_projector_config=a_cfg,
        question_projector_config=q_cfg,
        gate_config=gate_cfg,
        max_patches=512,
        use_decoder=True,
        decoder_attention_type="bbox_crop_reencode",
    )


# Dictionary for easy access
CONFIG_PRESETS = {
    "base": get_base_config,
    "medium": get_medium_config,
    "large": get_large_config,
}


def get_config_preset(preset_name: str) -> CoExVQAConfig:
    """
    Get a configuration preset by name.

    Args:
        preset_name: One of "base", "medium", or "large"

    Returns:
        CoExVQAConfig instance

    Raises:
        ValueError: If preset_name is not recognized
    """
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Unknown config preset: {preset_name}. Available: {list(CONFIG_PRESETS.keys())}")
    return CONFIG_PRESETS[preset_name]()
