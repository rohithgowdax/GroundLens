import torch.nn as nn
import torch
from typing import List, Optional
from dataclasses import dataclass, field

# Import implementaitons
from src.model.blocks.fusion import FusionBlock
from src.model.blocks.context_agg import ContextAggregationBlock
from src.model.blocks.projector_block import ProjectorBlock
from src.model.blocks.positional_encoding import SinusoidalPositionalEncoding2D
from src.model.blocks.box_prediction_head import BoxPredictionHead



@dataclass
class AnswerHeatmapProjectorConfig: 
    d_model: int

    # Fusion block
    fusion_modes: List[str] = field(default_factory=lambda: ["concat"])
    fusion_block_heads:int = 8
    fusion_dropout: float = 0.1
    fusion_concat_mlp_depth: int = 2
    fusion_concat_hidden_mult: float = 1.0
    fusion_concat_residual_scale: float = 1.0

    # Context aggregation block
    context_agg_mode: str = "identity"
    context_agg_num_layers: int = 1
    context_agg_n_heads: int = 8
    context_agg_ffn_mult: int = 4
    context_agg_dropout: float = 0.1
    context_agg_norm_first: bool = True
    context_agg_layer_norm_eps: float = 0.00001
    context_agg_activation: str = "gelu"

    # Projector block
    projector_mode: str = "identity"
    projector_d_out: int | None = None
    projector_dropout: float = 0.1
    projector_mlp_hidden_mult: float = 2
    projector_attn_heads: int = 8
    projector_attn_k: int = 1
    projector_eps: float = 0.000001

    # Spatial attention block
    spatial_attn_heads: int = 8
    spatial_attn_dropout: float = 0.1

    # Inital width and height bias
    init_w: float = 0.07
    init_h: float = 0.05



class AnswerHeatmapProjector(nn.Module):
    def __init__(self, config: AnswerHeatmapProjectorConfig):
        super().__init__()

        # store config
        self.config = config
        self.min_box_size = 1e-4

        # Positional encoding for spatial awareness
        self.pos_encoding = SinusoidalPositionalEncoding2D(
            d_model=config.d_model,
            max_h=64,
            max_w=64
        )

        # Fusion block
        self.fusion_block = FusionBlock(
            d_model=config.d_model,
            modes = config.fusion_modes,
            num_heads=config.fusion_block_heads,
            dropout=config.fusion_dropout,
            concat_mlp_depth=config.fusion_concat_mlp_depth,
            concat_hidden_mult=config.fusion_concat_hidden_mult,
            concat_residual_scale=config.fusion_concat_residual_scale
        )

        # Context aggregation
        self.context_agg_block = ContextAggregationBlock(
            d_model=config.d_model,
            mode=config.context_agg_mode,
            num_layers=config.context_agg_num_layers,
            n_heads=config.context_agg_n_heads,
            ffn_mult=config.context_agg_ffn_mult,
            norm_first=config.context_agg_norm_first,
            layer_norm_eps=config.context_agg_layer_norm_eps,
            activation=config.context_agg_activation,
        )

        # Projector
        self.projector_block = ProjectorBlock(
            d_model=config.d_model,
            mode=config.projector_mode,
            d_out=config.projector_d_out,
            dropout=config.projector_dropout,
            mlp_hidden_mult=config.projector_mlp_hidden_mult,
            attn_heads=config.projector_attn_heads,
            attn_k=config.projector_attn_k,
            eps=config.projector_eps
        )

        # Layer norms for residual connections
        task_head_in_dim = config.d_model if config.projector_d_out is None else config.projector_d_out
        self.fusion_norm = nn.LayerNorm(config.d_model)
        self.context_norm = nn.LayerNorm(config.d_model)
        self.projector_norm = nn.LayerNorm(task_head_in_dim)

        # Box prediction head with spatial attention and mean pooling
        # Using larger refinement dimensions for better capacity
        self.task_head = BoxPredictionHead(
            d_model=task_head_in_dim,
            spatial_attn_heads=config.spatial_attn_heads,
            spatial_attn_dropout=config.spatial_attn_dropout,
            init_w=config.init_w,
            init_h=config.init_h,
            min_box_size=self.min_box_size,
        )

    def forward(self, image_embeddings: torch.Tensor, q_pooled: Optional[torch.Tensor] = None, q_tokens: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        seq_len = image_embeddings.shape[1]

        # Add positional encoding for spatial awareness
        h_patches = int(seq_len ** 0.5)
        w_patches = (seq_len + h_patches - 1) // h_patches
        x = self.pos_encoding(image_embeddings, h_patches, w_patches)

        # Fuse with question 
        x_fused = self.fusion_block(image_embeddings=x, q_pooled=q_pooled, q_tokens=q_tokens, key_padding_mask=key_padding_mask)
        x = self.fusion_norm(x + x_fused)  # Add & Norm

        # Context aggregation 
        x_context = self.context_agg_block(x)
        x = self.context_norm(x + x_context)  # Add & Norm

        # Projector
        x = self.projector_block(x)
        x = self.projector_norm(x)

        # Task head
        box = self.task_head(x)  # (B, 4)

        return box

