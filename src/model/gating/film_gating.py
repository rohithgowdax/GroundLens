import torch
import torch.nn as nn
from typing import Optional


class FiLMGating(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) gating mechanism.
    
    Args:
        d_model: Dimension of the embeddings
        hidden_mult: Multiplier for hidden dimension (default: 2.0)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, d_model: int, hidden_mult: float = 2.0, dropout: float = 0.1):
        super().__init__()

        hidden_dim = int(d_model * hidden_mult)

        # Network to generate gamma (scale) parameters
        self.gamma_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        # Network to generate beta (shift) parameters
        self.beta_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        # Initialize to near-identity transformation at start
        self._init_networks()

    def _init_networks(self):
        """
        Initialize FiLM networks to produce near-identity transformation at start.
        """
        # Initialize gamma network final layer to output near zero
        gamma_final = self.gamma_net[-1]
        nn.init.zeros_(gamma_final.weight)
        nn.init.zeros_(gamma_final.bias)

        # Initialize beta network final layer to output near zero
        beta_final = self.beta_net[-1]
        nn.init.zeros_(beta_final.weight)
        nn.init.zeros_(beta_final.bias)

        # Initialize first layers with Xavier for proper gradient flow
        for net in [self.gamma_net, self.beta_net]:
            first_layer = net[0]
            nn.init.xavier_uniform_(first_layer.weight, gain=1.0)
            nn.init.zeros_(first_layer.bias)

    def forward(self, image_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM gating.

        Args:
            image_embeddings: Image embeddings [B, H*W, d_model]
            mask: Attention mask from question projector [B, H*W, 1]

        Returns:
            Gated embeddings [B, H*W, d_model]
        """
        # Generate feature-wise modulation parameters from mask
        gamma = self.gamma_net(mask)  # [B, H*W, d_model] - scale
        beta = self.beta_net(mask)     # [B, H*W, d_model] - shift

        # Apply affine transformation
        gated_embeddings = image_embeddings * (1 + gamma) + beta

        return gated_embeddings
