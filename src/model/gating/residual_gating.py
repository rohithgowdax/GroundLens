import torch
import torch.nn as nn
from typing import Optional


class ResidualGating(nn.Module):
    """
    Residual gating mechanism.

    Args:
        d_model: Dimension of the embeddings
        hidden_mult: Multiplier for hidden dimension in transformation (default: 1.0)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, d_model: int, hidden_mult: float = 1.0, dropout: float = 0.1):
        super().__init__()

        hidden_dim = int(d_model * hidden_mult)

        # Learnable transformation network
        self.transform_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        # Initialize to near-identity transformation at start
        self._init_network()

    def _init_network(self):
        """
        Initialize residual network to produce near-identity transformation at start.
        Final layer should output near zero so gated output ≈ input initially.
        """
        # Initialize final layer to output near zero (identity residual)
        final_layer = self.transform_net[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        # Initialize first layer with Xavier for proper gradient flow
        first_layer = self.transform_net[0]
        nn.init.xavier_uniform_(first_layer.weight, gain=1.0)
        nn.init.zeros_(first_layer.bias)

    def forward(self, image_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply residual gating.

        Args:
            image_embeddings: Image embeddings [B, H*W, d_model]
            mask: Attention mask from question projector [B, H*W, 1]

        Returns:
            Gated embeddings [B, H*W, d_model]
        """
        # Transform embeddings
        transformed = self.transform_net(image_embeddings)

        # Gate controls how much transformation to apply
        gated_embeddings = mask * transformed + (1 - mask) * image_embeddings

        return gated_embeddings
