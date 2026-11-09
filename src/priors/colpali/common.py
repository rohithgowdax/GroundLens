import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def colpali_score_to_mask(score_2d: torch.Tensor, size):
    """Perform bicubic interpolation on a single CoPALI score map to convert it to a mask of the given size."""
    W_img, H_img = int(size[0]), int(size[1])

    m = score_2d.to(torch.float32)

    full = F.interpolate(
        m[None, None], 
        size=(H_img, W_img),
        mode="bicubic", 
        align_corners=False
    )[0, 0].clamp_(0, 1)
    return full.detach().cpu().numpy().astype(np.float32)

@torch.no_grad()
def colpali_scores_to_masks(scores: torch.Tensor, sizes):
    """Convert a batch of CoPALI scores to masks using bicubic interpolation."""
    masks = []
    B = scores.shape[0]
    for i in range(B):
        masks.append(colpali_score_to_mask(scores[i], sizes[i]))
    return masks