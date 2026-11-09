import torch
import torch.nn as nn


class BoxPredictionHead(nn.Module):
    """
    Task head for bounding box prediction with spatial attention and mean pooling.

    This module takes spatially-aware patch features and predicts a bounding box in normalized
    coordinates (cx, cy, w, h) where all values are in [0, 1].

    Architecture:
        1. Spatial self-attention to capture long-range dependencies
        2. Mean pooling to aggregate patch features
        3. Refinement MLP to process pooled features
        4. Linear layer to predict box parameters

    Args:
        d_model: Dimension of input features
        spatial_attn_heads: Number of attention heads for spatial attention
        spatial_attn_dropout: Dropout rate for spatial attention
        init_w: Initial width bias for box prediction
        init_h: Initial height bias for box prediction
        min_box_size: Minimum allowed box size (prevents degenerate boxes)
        refinement_hidden_dim: Hidden dimension for first refinement layer (default: 512)
        refinement_intermediate_dim: Intermediate dimension for second refinement layer (default: 256)
    """

    def __init__(
        self,
        d_model: int,
        spatial_attn_heads: int = 8,
        spatial_attn_dropout: float = 0.1,
        init_w: float = 0.07,
        init_h: float = 0.05,
        min_box_size: float = 1e-4,
    ):
        super().__init__()

        self.min_box_size = min_box_size
        self.init_w = init_w
        self.init_h = init_h

        # Spatial attention for answer localization
        # This helps the model identify which patches are most relevant for the answer
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=spatial_attn_heads,
            dropout=spatial_attn_dropout,
            batch_first=True
        )
        self.spatial_attn_norm = nn.LayerNorm(d_model)

        # Box prediction head with refinement
        # Processes the pooled representation before predicting box
        self.box_refinement = nn.Sequential(
            nn.Linear(d_model, 768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Final box prediction
        self.box_predictor = nn.Linear(256, 4)
        # Initialize the bounding box predictor
        self._init_box_predictor()

    def _init_box_predictor(self):
        """Initialize box predictor with bias towards center and specified initial size."""
        cx0, cy0 = 0.5, 0.5
        nn.init.zeros_(self.box_predictor.weight)
        with torch.no_grad():
            t = torch.tensor(
                [cx0, cy0, self.init_w, self.init_h],
                dtype=self.box_predictor.bias.dtype
            )
            t = t.clamp(1e-4, 1 - 1e-4)
            self.box_predictor.bias.copy_(torch.log(t / (1 - t)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for box prediction.

        Args:
            x: Spatially-aware patch features, shape (batch_size, seq_len, d_model)

        Returns:
            Predicted bounding box in format (cx, cy, w, h), shape (batch_size, 4)
            All values are in [0, 1] range (normalized coordinates)
        """
        # Spatial attention (with residual connection)
        # Helps identify answer-relevant patches
        x_attn, _ = self.spatial_attention(x, x, x)
        x = self.spatial_attn_norm(x + x_attn)  # Add & Norm

        # Mean pooling: aggregate patch information
        pooled = x.mean(dim=1)  # (B, D)

        # Refine the pooled representation
        refined = self.box_refinement(pooled)  # (B, intermediate_dim)

        # Predict bounding box
        raw = self.box_predictor(refined)  # (B, 4)

        # Apply sigmoid and small minimum size
        cxcy = torch.sigmoid(raw[..., :2])  # Center coordinates
        eps = self.min_box_size
        wh = eps + (1.0 - eps) * torch.sigmoid(raw[..., 2:])  # Width and height
        box = torch.cat([cxcy, wh], dim=-1)  # (B, 4)

        return box
