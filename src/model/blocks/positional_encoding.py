import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    Fixed sinusoidal 2D positional encoding for image patches.
    Based on "Attention is All You Need" but extended to 2D.
    """
    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64):
        """
        Args:
            d_model: Dimension of the embeddings
            max_h: Maximum height in patches
            max_w: Maximum width in patches
        """
        super().__init__()
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w

        # Pre-compute positional encodings
        pe = self._create_positional_encoding(max_h, max_w, d_model)
        self.register_buffer('pe', pe)  # (max_h * max_w, d_model)

    def _create_positional_encoding(self, h: int, w: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding for 2D grid."""
        pe = torch.zeros(h * w, d_model)

        # Create position indices
        y_pos = torch.arange(h).unsqueeze(1).expand(h, w).reshape(-1)  
        x_pos = torch.arange(w).unsqueeze(0).expand(h, w).reshape(-1)  

        # Dimension indices
        div_term = torch.exp(torch.arange(0, d_model, 4).float() * (-math.log(10000.0) / d_model))

        # Apply sinusoidal functions
        # First half for y (height) position
        pe[:, 0::4] = torch.sin(y_pos.unsqueeze(1) * div_term)
        pe[:, 1::4] = torch.cos(y_pos.unsqueeze(1) * div_term)

        # Second half for x (width) position
        pe[:, 2::4] = torch.sin(x_pos.unsqueeze(1) * div_term)
        pe[:, 3::4] = torch.cos(x_pos.unsqueeze(1) * div_term)

        return pe

    def forward(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Add 2D positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            h_patches: Height of the patch grid
            w_patches: Width of the patch grid

        Returns:
            Tensor with positional encoding added, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        num_patches = h_patches * w_patches

        # Only add to image patch tokens
        if seq_len < num_patches:
            num_patches = seq_len

        # Get pre-computed encoding for this grid size
        if h_patches * w_patches > self.max_h * self.max_w:
            # Recompute for larger grid
            pe = self._create_positional_encoding(h_patches, w_patches, d_model).to(x.device)
        else:
            # Use pre-computed
            pe = self.pe[:h_patches * w_patches, :]

        # Truncate if needed
        pe = pe[:num_patches, :]

        # Add to input
        x_out = x.clone()
        x_out[:, :num_patches, :] = x[:, :num_patches, :] + pe.unsqueeze(0)

        return x_out
