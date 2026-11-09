import torch
import torch.nn as nn
from typing import Optional


class SpatialAttentionGating(nn.Module):
    """
    Spatial attention gating mechanism.

    Normalizes the mask to create spatial attention weights, then uses them
    to modulate the embeddings. Highlights relevant spatial regions.
    
    Args:
        initial_alpha: Initial strength of attention modulation (default: 1.0)
        normalize: Whether to normalize mask to sum to 1 (default: True)
        eps: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(self, initial_alpha: float = 1.0, normalize: bool = True, eps: float = 1e-8):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.normalize = normalize
        self.eps = eps

    def forward(self, image_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention gating.

        Args:
            image_embeddings: Image embeddings [B, H*W, d_model]
            mask: Attention mask from question projector [B, H*W, 1]

        Returns:
            Gated embeddings [B, H*W, d_model]
        """
        if self.normalize:
            # Normalize mask to create attention weights that sum to 1
            attention_weights = mask / (mask.sum(dim=1, keepdim=True) + self.eps)
        else:
            attention_weights = mask

        # Modulate embeddings: amplify attended regions
        gated_embeddings = image_embeddings * (1 + self.alpha * attention_weights)

        return gated_embeddings
