import torch 
from torchvision.ops import box_convert, complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss
import torch.nn.functional as F

def pred_norm_cxcywh_to_xyxy(pred_boxes_cxcywh_norm, eps=1e-6):
    """
    Sanitize predicted normalized cxcywh to normalized xyxy with strictly positive w,h.
    """
    p = pred_boxes_cxcywh_norm.clamp(0.0, 1.0)
    cxcy = p[:, :2]
    wh   = p[:, 2:].clamp_min(eps)           
    p    = torch.cat([cxcy, wh], dim=-1)
    xyxy = box_convert(p, in_fmt="cxcywh", out_fmt="xyxy")

    x1, y1, x2, y2 = xyxy.split(1, dim=-1)
    x1 = x1.clamp(0.0, 1.0 - eps)
    y1 = y1.clamp(0.0, 1.0 - eps)
    x2 = torch.maximum(x2, x1 + eps).clamp(0.0, 1.0)
    y2 = torch.maximum(y2, y1 + eps).clamp(0.0, 1.0)
    return torch.cat([x1, y1, x2, y2], dim=-1), wh  # also return sanitized (w,h)



def safe_iou_loss(pred_xyxy_norm, gt_xyxy_norm, mode="ciou"):
    """Compute IoU loss in fp32 and sanitize NaNs/Infs."""
    a = pred_xyxy_norm.float()
    b = gt_xyxy_norm.float()
    if mode == "ciou":
        loss = complete_box_iou_loss(a, b, reduction="mean")
    elif mode == "diou":
        loss = distance_box_iou_loss(a, b, reduction="mean")
    elif mode == "giou":
        loss = generalized_box_iou_loss(a, b, reduction="mean")
    else:
        raise ValueError(f"Unknown IoU loss type: {mode}")
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def get_projection_loss(
    gt_boxes_xyxy_abs,               
    pred_boxes_cxcywh_norm,
    full_img_sizes,           
    lambda_iou: float = 1.0,
    lambda_size: float = 1.0,
    lambda_center: float = 1.0,
    lambda_aspect: float = 1.0,
    lambda_edges: float = 1.0,
    lambda_area: float = 0.0,
    eps: float = 1e-6,
    iou_loss_type: str = "ciou"
):
    device = pred_boxes_cxcywh_norm.device
    dtype = pred_boxes_cxcywh_norm.dtype

    # Fix the norm to correct device
    gt= (
        gt_boxes_xyxy_abs if isinstance(gt_boxes_xyxy_abs, torch.Tensor)
        else torch.tensor(gt_boxes_xyxy_abs, dtype=dtype, device=device)
    ).to(device=device, dtype=dtype)

    # Normalize by using the size 
    W = full_img_sizes[:, 0].to(device).unsqueeze(1)
    H = full_img_sizes[:, 1].to(device).unsqueeze(1)
    denom = torch.cat([W, H, W, H], dim=1).clamp_min(1.0 + eps)
    gt_xyxy_norm = (gt / denom).clamp(0.0, 1.0)

    # Convert to correct format
    pred_xyxy_norm, pred_wh = pred_norm_cxcywh_to_xyxy(pred_boxes_cxcywh_norm, eps=eps)
    gt_cxcywh_norm = box_convert(gt_xyxy_norm, in_fmt="xyxy", out_fmt="cxcywh")
    gt_wh = gt_cxcywh_norm[:, 2:].clamp_min(eps)

    # IoU loss
    loss_iou = safe_iou_loss(pred_xyxy_norm, gt_xyxy_norm, mode=iou_loss_type)

    # Size loss: relative smooth L1
    # Compute in-place to reduce memory allocations
    # Use clamp to prevent log of very small values (numerical stability)
    log_pred_wh = torch.log(pred_wh.clamp_min(eps))
    log_gt_wh = torch.log(gt_wh.clamp_min(eps))
    log_diff = log_pred_wh - log_gt_wh
    loss_size = F.smooth_l1_loss(log_diff, torch.zeros_like(log_diff), reduction="mean")

    # Area loss
    A_pred = (pred_wh[:, 0] * pred_wh[:, 1]).clamp_min(eps)
    A_gt   = (gt_wh[:, 0]   * gt_wh[:, 1]).clamp_min(eps)
    # Clamp ratio to prevent extreme values in sqrt
    area_ratio = (A_pred / A_gt).clamp(eps, 1.0 / eps)
    loss_area = ((torch.sqrt(area_ratio) - 1.0) ** 2).mean()

    # Center loss: Euclidean
    loss_center = F.mse_loss(pred_boxes_cxcywh_norm[:, :2], gt_cxcywh_norm[:, :2], reduction="mean")

    # Edge loss
    loss_edges = F.l1_loss(pred_xyxy_norm, gt_xyxy_norm, reduction="mean")

    # Aspect ratio loss
    # Use already computed log values for numerical stability
    aspect_pred = log_pred_wh[:, 0] - log_pred_wh[:, 1]
    aspect_gt   = log_gt_wh[:, 0]   - log_gt_wh[:, 1]
    loss_aspect = F.smooth_l1_loss(aspect_pred, aspect_gt, reduction="mean")

    # Total loss
    loss_total = (
        lambda_iou * loss_iou +
        lambda_size * loss_size +
        lambda_center * loss_center +
        lambda_aspect * loss_aspect +
        lambda_edges * loss_edges +
        lambda_area * loss_area
    )

    # Safety: Replace NaN with 0 (should not happen with above fixes, but defensive)
    loss_total = torch.nan_to_num(loss_total, nan=0.0, posinf=0.0, neginf=0.0)

    # Return loss dict 
    return {
        "loss_total": loss_total,
        "loss_iou": loss_iou.detach(),
        "loss_center": loss_center.detach() if lambda_center > 0 else 0.0,
        "loss_size": loss_size.detach() if lambda_size > 0 else 0.0,
        "loss_edges": loss_edges.detach() if lambda_edges > 0 else 0.0,
        "loss_aspect": loss_aspect.detach() if lambda_aspect > 0 else 0.0,
        "loss_area": loss_area.detach() if lambda_area > 0 else 0.0
    }