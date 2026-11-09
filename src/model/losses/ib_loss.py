from typing import Optional

import torch
from torch import Tensor

from src.utils.pix2struct_utils import (
    compute_square_patch_grid_size,
    create_patch_grid_coordinates,
)

MIN_BATCH_FOR_MI = 4


def get_bbox_token_mask(bbox: Tensor, seq_len: int) -> Tensor:
    """Convert normalized cxcywh boxes to a patch-level spatial mask (B, seq_len, 1)."""
    B = bbox.shape[0]
    device = bbox.device

    H_p, W_p = compute_square_patch_grid_size(seq_len)
    grid = create_patch_grid_coordinates(seq_len, H_p, W_p, device)
    grid = grid.unsqueeze(0).expand(B, -1, -1)

    x_center = bbox[:, 0].unsqueeze(1)
    y_center = bbox[:, 1].unsqueeze(1)
    width    = bbox[:, 2].unsqueeze(1)
    height   = bbox[:, 3].unsqueeze(1)

    x_min = torch.clamp(x_center - width  / 2, 0.0, 1.0)
    x_max = torch.clamp(x_center + width  / 2, 0.0, 1.0)
    y_min = torch.clamp(y_center - height / 2, 0.0, 1.0)
    y_max = torch.clamp(y_center + height / 2, 0.0, 1.0)

    margin = (1.0 / max(H_p, W_p)) * 4  # 4x patch size, same as model's _get_bbox_token_indices
    x_min = torch.clamp(x_min - margin, 0.0, 1.0)
    x_max = torch.clamp(x_max + margin, 0.0, 1.0)
    y_min = torch.clamp(y_min - margin, 0.0, 1.0)
    y_max = torch.clamp(y_max + margin, 0.0, 1.0)

    inside = ((grid[..., 0] >= x_min) & (grid[..., 0] <= x_max) &
              (grid[..., 1] >= y_min) & (grid[..., 1] <= y_max))
    return inside.float().unsqueeze(-1)


def median_bandwidth(X: Tensor) -> Tensor:
    """RBF bandwidth via median heuristic on X. Returns a detached scalar."""
    with torch.no_grad():
        dists_sq = torch.cdist(X, X).pow(2)
        nonzero = dists_sq[dists_sq > 0]
        if nonzero.numel() == 0:
            return X.new_tensor(1.0)
        return (nonzero.median() / 2.0).sqrt().clamp(min=1e-5)


def rbf_gram(X: Tensor, sigma: Tensor) -> Tensor:
    """RBF Gram matrix: K_ij = exp(−‖xi−xj‖² / 2σ²)."""
    return torch.exp(-torch.cdist(X, X).pow(2) / (2.0 * sigma.pow(2)))


def trace_normalize(K: Tensor) -> Tensor:
    """Divide K by its trace. Returns K unchanged if trace < 1e-10."""
    tr = torch.trace(K)
    return K if tr < 1e-10 else K / tr


def joint_matrix(A_X: Tensor, A_Y: Tensor) -> Tensor:
    """Joint Gram matrix: Hadamard product of A_X and A_Y, trace-normalized."""
    return trace_normalize(A_X * A_Y)


def renyi_entropy_alpha2(A: Tensor) -> Tensor:
    """Matrix Renyi α=2 entropy: H_2(A) = −log(tr(A²))."""
    return -torch.trace(A @ A).clamp(min=1e-10).log()


def matrix_renyi_mi(X: Tensor, Y: Tensor) -> Tensor:
    """Estimate I(X; Y) via matrix Renyi α=2 entropy with median-heuristic bandwidth.

    Inputs are cast to float32. Bandwidth is computed on [X; Y] and detached.
    """
    X, Y = X.float(), Y.float()
    sigma = median_bandwidth(torch.cat([X, Y], dim=0))
    A_X  = trace_normalize(rbf_gram(X, sigma))
    A_Y  = trace_normalize(rbf_gram(Y, sigma))
    A_XY = joint_matrix(A_X, A_Y)
    return renyi_entropy_alpha2(A_X) + renyi_entropy_alpha2(A_Y) - renyi_entropy_alpha2(A_XY)


def compute_ib_loss(
    image_embeddings: Tensor,
    q_mask: Tensor,
    gated_embeddings: Tensor,
    box_pred: Tensor,
    lambda_ib: float,
    beta1: float,
    beta2: float,
) -> dict:
    """Two-bottleneck IB loss.

    L_IB = λ_IB * (β1·Î(ImgEmb; T1) − Î(T1; GatedImgEmb) + β2·Î(GatedImgEmb; T2))

    Returns dict with keys: ib_total (Tensor), ib_comp1, ib_suff, ib_comp2 (floats).
    """
    B, seq_len = image_embeddings.shape[0], image_embeddings.shape[1]
    device = image_embeddings.device


    if B < MIN_BATCH_FOR_MI:
        zero = torch.tensor(0.0, device=device)
        return {"ib_total": zero, "ib_comp1": 0.0, "ib_suff": 0.0, "ib_comp2": 0.0}

    # Define T1 and T2 as bottlenecks 
    T1 = image_embeddings * q_mask
    T2 = gated_embeddings * get_bbox_token_mask(box_pred.detach(), seq_len)

    # Pool representations to get fixed-size vectors for MI estimation
    e_pooled  = image_embeddings.mean(dim=1)
    t1_pooled = T1.mean(dim=1)
    g_pooled  = gated_embeddings.mean(dim=1)
    t2_pooled = T2.mean(dim=1)

    # First bottleneck 
    comp1 = beta1 * matrix_renyi_mi(e_pooled, t1_pooled)
    suff  = -matrix_renyi_mi(t1_pooled, g_pooled)

    # Second bottleneck
    comp2 = beta2 * matrix_renyi_mi(g_pooled, t2_pooled)

    # Total IB loss (to minimize)
    ib_total = lambda_ib * (comp1 + suff + comp2)

    return {
        "ib_total": ib_total,
        "ib_comp1": float(comp1.item()),
        "ib_suff":  float(suff.item()),
        "ib_comp2": float(comp2.item()),
    }
