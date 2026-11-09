import torch
import torch.nn as nn
from typing import Optional


class LinearInterpolationGating(nn.Module):
    """
    Simple linear interpolation gating mechanism.

    Args:
        initial_alpha: Initial value for the interpolation parameter (default: 0.1)
    """

    def __init__(self, initial_alpha: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))

    def forward(self, image_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply linear interpolation gating.

        Args:
            image_embeddings: Image embeddings [B, H*W, d_model]
            mask: Attention mask from question projector [B, H*W, 1]

        Returns:
            Gated embeddings [B, H*W, d_model]
        """
        gated_embeddings = image_embeddings * (self.alpha * mask + (1 - self.alpha))
        return gated_embeddings
