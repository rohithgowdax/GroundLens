from dataclasses import dataclass
from typing import Literal, Optional
import torch.nn as nn

# Import implementation
from src.model.gating.linear_interpolation_gating import LinearInterpolationGating
from src.model.gating.residual_gating import ResidualGating
from src.model.gating.spatial_attention_gating import SpatialAttentionGating
from src.model.gating.film_gating import FiLMGating



# The optional types for the gating mechanism
GatingType = Literal["linear_interpolation", "residual", "spatial_attention", "film"]

@dataclass
class GatingConfig:
    """
    Unified configuration for all gating mechanisms experimented with.

    Args:
        gating_type: Type of gating mechanism to use
        d_model: Dimension of embeddings (required for residual and film)
        initial_alpha: Initial interpolation/modulation strength (for linear_interpolation and spatial_attention)
        hidden_mult: Hidden dimension multiplier for transformation networks (for residual and film)
        dropout: Dropout rate (for residual and film)
        normalize: Whether to normalize mask in spatial attention (for spatial_attention)
        eps: Small constant for numerical stability (for spatial_attention)
    """
    gating_type: GatingType = "linear_interpolation"

    # Required for residual and film gating
    d_model: Optional[int] = None

    # Used by linear_interpolation and spatial_attention
    initial_alpha: float = 0.1

    # Used by residual and film
    hidden_mult: float = 1.0
    dropout: float = 0.1

    # Used by spatial_attention
    normalize: bool = True
    eps: float = 1e-8

    def __post_init__(self):
        """Validate configuration based on gating type."""
        if self.gating_type in ["residual", "film"]:
            if self.d_model is None:
                raise ValueError(f"{self.gating_type} gating requires d_model to be specified")



def create_gating_module(gate_config: GatingConfig) -> nn.Module:
    """
    Factory function to create the appropriate gating module based on config.
    """
    if gate_config.gating_type == "linear_interpolation":
        return LinearInterpolationGating(initial_alpha=gate_config.initial_alpha)
    elif gate_config.gating_type == "residual":
        return ResidualGating(
            d_model=gate_config.d_model,
            hidden_mult=gate_config.hidden_mult,
            dropout=gate_config.dropout
        )
    elif gate_config.gating_type == "spatial_attention":
        return SpatialAttentionGating(
            initial_alpha=gate_config.initial_alpha,
            normalize=gate_config.normalize,
            eps=gate_config.eps
        )
    elif gate_config.gating_type == "film":
        return FiLMGating(
            d_model=gate_config.d_model,
            hidden_mult=gate_config.hidden_mult,
            dropout=gate_config.dropout
        )
    else:
        raise ValueError(f"Unknown gating type: {gate_config.gating_type}")