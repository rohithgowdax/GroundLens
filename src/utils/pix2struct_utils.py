import torch
import math
from typing import List, Dict, Any, Tuple
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


@torch.no_grad()
def prepare_question_for_fusion(
    questions: List[str],
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    device: str,
    pool: str = "mean",                    
) -> Dict[str, Any]:
    # Tokenize the questions 
    tok = processor.tokenizer(
        questions, padding=True, return_tensors="pt", add_special_tokens=True
    )
    tok = {k: v.to(device) for k, v in tok.items()}

    # Decoder token embedding table
    dec_emb = model.get_decoder().get_input_embeddings() 

    # Token embeddings (B, T, D)
    q_tokens = dec_emb(tok["input_ids"]).to(dec_emb.weight.dtype)

    # attention_mask for setting padding token to zero
    attn = tok.get("attention_mask", (tok["input_ids"] != processor.tokenizer.pad_token_id).long())
    attn = attn.to(device)
    key_padding_mask = (attn == 0)

    # Pooled vector for FiLM/Concat
    if pool == "mean":
        denom = attn.sum(dim=1, keepdim=True).clamp_min(1)  # (B,1)
        q_pooled = (q_tokens * attn.unsqueeze(-1)).sum(dim=1) / denom
    elif pool == "max":
        # mask pads to -inf so they don't win the max
        very_neg = torch.finfo(q_tokens.dtype).min
        masked = q_tokens.masked_fill(attn.unsqueeze(-1) == 0, very_neg)
        q_pooled = masked.max(dim=1).values
    else:
        raise ValueError(f"Unknown pool method: {pool}")


    # Create output dict
    out = {
        "q_pooled": q_pooled,
        "q_tokens": q_tokens,
        "key_padding_mask": key_padding_mask

    }
    return out


def compute_square_patch_grid_size(seq_len: int) -> Tuple[int, int]:
    """
    Compute approximately square patch grid dimensions that match the sequence length.

    This is a simplified version that assumes a roughly square grid when we don't
    have access to the original image dimensions.

    Args:
        seq_len: Actual sequence length from encoder output

    Returns:
        Tuple of (H_p, W_p): Height and width of the patch grid
    """
    # Start with a square grid
    H_p = int(math.sqrt(seq_len))
    W_p = int(math.ceil(seq_len / H_p))

    # Adjust if the grid is too large
    while H_p * W_p > seq_len:
        W_p -= 1

    # If still not enough patches, add a row
    while H_p * W_p < seq_len:
        H_p += 1
        W_p = int(math.ceil(seq_len / H_p))
        # Recheck if too large after adding row
        while H_p * W_p > seq_len:
            W_p -= 1

    return H_p, W_p


def create_patch_grid_coordinates(
    seq_len: int,
    H_p: int,
    W_p: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create normalized coordinate grid for patch tokens.

    Args:
        seq_len: Actual sequence length (number of patches to create coordinates for)
        H_p: Height of patch grid
        W_p: Width of patch grid
        device: Device to create tensor on

    Returns:
        torch.Tensor of shape (seq_len, 2) with normalized [x, y] coordinates in [0, 1]
    """
    # Create normalized coordinate grid using patch centers
    y_coords = torch.linspace(0.5/H_p, 1 - 0.5/H_p, H_p, device=device)
    x_coords = torch.linspace(0.5/W_p, 1 - 0.5/W_p, W_p, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Flatten and take only seq_len coordinates (in case H_p * W_p > seq_len)
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)[:seq_len]

    return grid