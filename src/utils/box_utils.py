from typing import Union

import torch
from torchvision.ops import box_convert


def prepare_gt_boxes(
    boxes: Union[list, torch.Tensor],
    full_img_sizes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert absolute xyxy GT boxes to normalized cxcywh format.

    Args:
        boxes: GT boxes in absolute pixel coordinates, shape (B, 4) xyxy.
        full_img_sizes: Image sizes, shape (B, 2) as (W, H).
        device: Target device for the output tensor.

    Returns:
        Tensor of shape (B, 4) in normalized cxcywh format.
    """
    if isinstance(boxes, list):
        boxes = torch.tensor(boxes, device=device)
    else:
        boxes = boxes.to(device)

    W = full_img_sizes[:, 0].to(device).unsqueeze(1)
    H = full_img_sizes[:, 1].to(device).unsqueeze(1)
    denom = torch.cat([W, H, W, H], dim=1).clamp_min(1.0)
    xyxy_norm = (boxes / denom).clamp(0.0, 1.0)
    return box_convert(xyxy_norm, in_fmt="xyxy", out_fmt="cxcywh")
