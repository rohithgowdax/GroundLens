import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

FUSION_OPTIONS = ["concat", "film", "cross_attention"]


class ConcatFusionBlock(nn.Module):
    """
    Concat fusion with a tiny MLP projector and residual.
    """
    def __init__(self, d_model: int, mlp_depth: int = 2, hidden_mult: float = 1.0, dropout: float = 0.1, residual_scale: float = 1.0, zero_init_last: bool = True):
        super().__init__()
        assert mlp_depth in (1, 2)
        self.pre_ln = nn.LayerNorm(d_model)

        # MLP projtor 
        if mlp_depth == 1:
            self.proj = nn.Linear(2 * d_model, d_model)
        else:
            hidden = int(hidden_mult * d_model)
            self.proj = nn.Sequential(
                nn.Linear(2 * d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
            )
        if zero_init_last:
            last = self.proj[-1] if isinstance(self.proj, nn.Sequential) else self.proj
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = residual_scale

    def forward(self, V: torch.Tensor, q_pooled: torch.Tensor) -> torch.Tensor:
        Vn = self.pre_ln(V)
        q_rep = q_pooled.unsqueeze(1).expand(-1, Vn.size(1), -1)         
        upd = self.proj(torch.cat([Vn, q_rep], dim=-1))                 
        return V + self.residual_scale * self.dropout(upd)



class FiLMFusionBlock(nn.Module):
    """
    FiLM modulation with residual 
    """
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hid = hidden_dim or d_model
        self.gamma_beta = nn.Sequential(
            nn.Linear(d_model, hid), 
            nn.GELU(), 
            nn.Linear(hid, 2 * d_model)
        )
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # near-identity init
        nn.init.zeros_(self.gamma_beta[-1].weight); nn.init.zeros_(self.gamma_beta[-1].bias)

    def forward(self, V: torch.Tensor, q_pooled: torch.Tensor) -> torch.Tensor:
        g, b = torch.chunk(self.gamma_beta(q_pooled), 2, dim=-1)  
        g = g.unsqueeze(1); b = b.unsqueeze(1)                    
        out = self.ln(V * (1.0 + g) + b)                          
        return V + self.dropout(out)                              


class CrossAttentionFusionBlock(nn.Module):
    """
    Cross-attend visual tokens to question tokens. 
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Layer norm for query 
        self.q_ln = nn.LayerNorm(d_model)   

        # Layer norm for key/value
        self.kv_ln = nn.LayerNorm(d_model)  

        # Multi-head attention heads
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Residual dropout + tiny FFN
        self.dropout = nn.Dropout(dropout)

    def forward(self, V: torch.Tensor, Q: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # Normalize visual tokens and question tokens
        Vn = self.q_ln(V)                                   
        if Q.dim() == 2:
            Q = Q.unsqueeze(1)                              
        Qn = self.kv_ln(Q)    
        
        # Cross-attention, ignoring padding tokens if provided
        out, _ = self.attn(query=Vn, key=Qn, value=Qn, key_padding_mask=key_padding_mask)

        # Residual and dropout                      
        return V + self.dropout(out) 


class FusionBlock(nn.Module):
    def __init__(self, d_model: int, modes: List[str] = ["concat"],
                 num_heads: int = 8, dropout: float = 0.1,
                 concat_mlp_depth: int = 2, concat_hidden_mult: float = 1.0,
                 concat_residual_scale: float = 1.0):
        super().__init__()

        # Check that options are correct for modes 
        if not all(m in FUSION_OPTIONS for m in modes):
            raise ValueError(f"Invalid fusion mode. Supported: {FUSION_OPTIONS}")
        
        self.layers = nn.ModuleList()
        for mode in modes:
            if mode == "concat":
                self.layers.append(ConcatFusionBlock(d_model, mlp_depth=concat_mlp_depth, hidden_mult=concat_hidden_mult, dropout=dropout, residual_scale=concat_residual_scale))
            elif mode == "film":
                self.layers.append(FiLMFusionBlock(d_model, hidden_dim=None, dropout=dropout))
            elif mode == "cross_attention":
                self.layers.append(CrossAttentionFusionBlock(d_model, num_heads=num_heads, dropout=dropout))

    def forward(self, image_embeddings: torch.Tensor, q_pooled: Optional[torch.Tensor] = None, q_tokens: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = image_embeddings
        for layer in self.layers:
            if isinstance(layer, CrossAttentionFusionBlock):
                assert q_tokens is not None, "Cross-attn needs token-level Q"
                x = layer(x, q_tokens, key_padding_mask)
            else:
                assert q_pooled is not None, "FiLM/Concat need pooled q"
                x = layer(x, q_pooled)
        return x
