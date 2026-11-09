import torch
import torch.nn as nn
from typing import Optional

READOUT_OPTIONS = ["identity", "linear", "mlp", "mean_pool", "max_pool", "attn_pool"]

class ProjectorBlock(nn.Module):
    """
    Projection block placed just before the task Head.
    """
    def __init__(
        self,
        d_model: int,
        mode: str = "identity",
        d_out: Optional[int] = None,
        dropout: float = 0.1,
        mlp_hidden_mult: float = 2.0,
        attn_heads: int = 8,
        attn_k: int = 1,                 
        eps: float = 1e-6,
    ):
        super().__init__()

        # Check if the projection option is valid
        if mode not in READOUT_OPTIONS:
            raise ValueError(f"ReadoutAdapter mode '{mode}' not in {READOUT_OPTIONS}")
        

        # Store outputs 
        self.mode = mode
        self.d_model = d_model
        self.d_out = d_out or d_model
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

        # Common projections if D changes
        self.seq_proj = nn.Identity() if self.d_out == d_model else nn.Linear(d_model, self.d_out, bias=True)

        if mode == "identity":
            self.net = nn.Identity()
        elif mode == "linear":
            self.ln = nn.LayerNorm(d_model)
            self.proj = nn.Linear(d_model, self.d_out)
            if self.d_out == d_model:
                nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        elif mode == "mlp":
            self.ln = nn.LayerNorm(d_model)
            h = int(mlp_hidden_mult * self.d_out)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, self.d_out),
            )
            if self.d_out == d_model:
                nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        elif mode in ["mean_pool", "max_pool"]:
            # For pooling
            self.pre_ln = nn.LayerNorm(d_model)
        elif mode == "attn_pool":
            # Pooling by Multihead Attention 
            self.pre_ln = nn.LayerNorm(d_model)
            self.k = attn_k
            self.queries = nn.Parameter(torch.randn(attn_k, self.d_out))
            self.kv_proj = nn.Linear(d_model, self.d_out) if self.d_out != d_model else nn.Identity()
            self.attn = nn.MultiheadAttention(self.d_out, attn_heads, batch_first=True, dropout=dropout)


    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        x: [B, N, D]
        key_padding_mask: [B, N] (True = pad/ignore) used by attn_pool and cls_pool
        """
        B, N, D = x.shape

        if self.mode == "identity":
            return self.seq_proj(x)

        if self.mode == "linear":
            z = self.ln(x)
            upd = self.proj(z)
            # residual if same D
            return x + self.dropout(upd) if self.d_out == D else upd

        if self.mode == "mlp":
            z = self.ln(x)
            upd = self.mlp(z)
            return x + self.dropout(upd) if self.d_out == D else upd

        if self.mode in ["mean_pool", "max_pool"]:
            z = self.pre_ln(x)
            z = self.seq_proj(z)   
            if key_padding_mask is not None:
                valid = (~key_padding_mask).float().unsqueeze(-1)  

            # Mean pooling
            if self.mode == "mean_pool":
                if key_padding_mask is None:
                    return z.mean(dim=1)                            
                s = (z * valid).sum(dim=1)
                d = valid.sum(dim=1).clamp_min(self.eps)
                return s / d
            
            # Max pooling 
            if self.mode == "max_pool":
                if key_padding_mask is None:
                    return z.max(dim=1).values
                z_masked = z.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf"))
                return z_masked.max(dim=1).values

        if self.mode == "attn_pool":
            z = self.pre_ln(x)                     
            kv = self.kv_proj(z)                    
            q = self.queries.unsqueeze(0).expand(B, -1, -1)  
            out, _ = self.attn(query=q, key=kv, value=kv, key_padding_mask=key_padding_mask)
            return out.squeeze(1) if out.size(1) == 1 else out  