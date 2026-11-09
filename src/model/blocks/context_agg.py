import torch
import torch.nn as nn
from typing import Optional


# ALlowed options for context aggregation layer
CONTEXT_AGG_OPTIONS = ["identity", "transformer"]

class ContextAggregationBlock(nn.Module):
    """
    Context aggregation operating on sequences of tokens.
    """
    def __init__(
        self,
        d_model: int,
        mode: str = "identity",
        num_layers: int = 1,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        norm_first: bool = True,   
        layer_norm_eps: float = 1e-5,
        activation: str = "gelu",
    ):
        super().__init__()
        mode = mode.lower()
        if mode not in CONTEXT_AGG_OPTIONS:
            raise ValueError(f"ContextAggregationBlock: unknown mode '{mode}'. "
                             f"Choose from {CONTEXT_AGG_OPTIONS}")
        self.mode = mode

        if self.mode == "identity":
            self.net = nn.Identity()

        elif self.mode == "transformer":
            d_ff = int(ffn_mult * d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation=activation,
                batch_first=True,          
                norm_first=norm_first, 
                layer_norm_eps=layer_norm_eps,
            )
            self.net = nn.TransformerEncoder(
                enc_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,  
            )

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, N, D]
        """
        if self.mode == "identity":
            return x
        
        # TransformerEncoder expects batched input 
        return self.net(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
