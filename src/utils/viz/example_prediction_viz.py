import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Tuple, Optional, Union

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']


def cxcywh_norm_to_xyxy_pix(
    box_norm: List[float] | Tuple[float, float, float, float],
    W: int,
    H: int
) -> Tuple[int, int, int, int]:
    """Convert normalized cxcywh box to pixel xyxy coordinates."""
    cx, cy, w, h = box_norm
    bw, bh = w * W, h * H
    x1 = int(round(cx * W - bw / 2.0))
    y1 = int(round(cy * H - bh / 2.0))
    x2 = int(round(x1 + bw))
    y2 = int(round(y1 + bh))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def upsample_mask_to_image(
    mask_patches: torch.Tensor,
    patch_shape: Tuple[int, int],
    image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Upsample a patch-level mask to full image resolution.

    Args:
        mask_patches: Flattened mask tensor of shape (num_patches,) or (num_patches, 1)
        patch_shape: Tuple of (h_patch, w_patch)
        image_size: Tuple of (H, W) for the target image size

    Returns:
        Upsampled mask as numpy array of shape (H, W)
    """
    h_patch, w_patch = patch_shape
    H, W = image_size
    num_patches = h_patch * w_patch

    if mask_patches.ndim == 2:
        mask_patches = mask_patches.squeeze(-1)

    mask_patches = mask_patches[:num_patches]
    mask_grid = mask_patches.reshape(h_patch, w_patch)
    mask_grid = mask_grid.clamp(0.0, 1.0)

    mask_tensor = mask_grid.cpu().float().unsqueeze(0).unsqueeze(0)
    mask_upsampled = F.interpolate(
        mask_tensor,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    return mask_upsampled


def draw_bbox(
    ax: plt.Axes,
    bbox_xyxy: Tuple[int, int, int, int],
    color: str,
    linewidth: int = 3,
    linestyle: str = '-'
):
    """Draw a bounding box rectangle on the given axes."""
    x1, y1, x2, y2 = bbox_xyxy
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        linestyle=linestyle
    )
    ax.add_patch(rect)


def save_document_example(
    image: Image.Image,
    save_path: str,
    mask_overlay: Optional[np.ndarray] = None,
    pred_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    gt_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
):
    """Save a single panel as a high-DPI PNG."""
    W, H = image.size
    fig_w = 8
    fig_h = fig_w * (H / W)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    ax.imshow(image)

    if mask_overlay is not None:
        ax.imshow(mask_overlay, cmap='jet', alpha=0.5, vmin=0, vmax=1)

    if gt_bbox_xyxy is not None:
        draw_bbox(ax, gt_bbox_xyxy, color='green', linewidth=3)

    if pred_bbox_xyxy is not None:
        draw_bbox(ax, pred_bbox_xyxy, color='red', linewidth=3)

    ax.axis('off')
    plt.tight_layout(pad=0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def save_spotlight_mask_example(
    image: Image.Image,
    mask: np.ndarray,
    save_path: str,
    dark_alpha: float = 0.6
):
    """
    Save document with a spotlight effect: activated patches (mask=1) show through
    clearly, non-activated patches (mask=0) are dimmed with a dark overlay.

    Args:
        image: PIL Image of the document
        mask: Binary or soft mask array of shape (H, W), values in [0, 1]
        save_path: Path to save the output PNG
        dark_alpha: Maximum opacity of the dark overlay on inactive patches (0=no dimming, 1=fully black)
    """
    image = image.convert("RGB")
    W, H = image.size
    fig_w = 8
    fig_h = fig_w * (H / W)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(image)

    # RGBA overlay: black with alpha = dark_alpha * (1 - mask)
    # → transparent where mask=1 (activated), dark where mask=0 (inactive)
    rgba = np.zeros((H, W, 4), dtype=float)
    rgba[..., 3] = dark_alpha * (1.0 - mask)
    ax.imshow(rgba)

    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def save_filled_bbox_example(
    image: Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    save_path: str,
    color: str = 'red',
    alpha: float = 0.4
):
    """Save document with a filled (hard mask) rectangle over the given bbox."""
    image = image.convert("RGB")
    W, H = image.size
    fig_w = 8
    fig_h = fig_w * (H / W)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.imshow(image)
    x1, y1, x2, y2 = bbox_xyxy
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=0,
        edgecolor='none',
        facecolor=color,
        alpha=alpha
    )
    ax.add_patch(rect)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def save_spotlight_bbox_example(
    image: Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    save_path: str,
    dark_alpha: float = 0.7
):
    """
    Save document with a spotlight effect using the predicted bbox as mask:
    pixels inside the bbox are visible, everything outside is dimmed black.

    Args:
        image: PIL Image of the document
        bbox_xyxy: Bounding box (x1, y1, x2, y2) in pixel coordinates
        save_path: Path to save the output PNG
        dark_alpha: Opacity of the dark overlay on pixels outside the bbox
    """
    image = image.convert("RGB")
    W, H = image.size
    x1, y1, x2, y2 = bbox_xyxy

    mask = np.zeros((H, W), dtype=float)
    mask[y1:y2, x1:x2] = 1.0

    save_spotlight_mask_example(image=image, mask=mask, save_path=save_path, dark_alpha=dark_alpha)


def save_example_prediction_visualization(
    image: Image.Image,
    q_mask_pred: torch.Tensor,
    patch_shape: Tuple[int, int],
    pred_bbox_norm: List[float],
    gt_bbox_norm: List[float],
    save_dir: str,
    prefix: str = "example",
    question: Optional[str] = None,
    gt_answer: Optional[Union[str, List[str]]] = None,
    pred_answer: Optional[str] = None
):
    """
    Save 4 separate PNG visualizations for example predictions:
    1. {prefix}_original.png - Original document
    2. {prefix}_qmask.png - Document with q-mask overlay
    3. {prefix}_bboxes.png - Original with pred (red) and GT (green) boxes
    4. {prefix}_qmask_pred.png - Q-mask overlay with pred box (red)
    5. {prefix}_metadata.json - Question, GT answer, and predicted answer

    Args:
        image: PIL Image of the document
        q_mask_pred: Predicted question mask tensor, shape (num_patches,) or (num_patches, 1)
        patch_shape: Tuple of (h_patch, w_patch) indicating the patch grid dimensions
        pred_bbox_norm: Predicted bounding box in normalized cxcywh format [cx, cy, w, h]
        gt_bbox_norm: Ground truth bounding box in normalized cxcywh format [cx, cy, w, h]
        save_dir: Directory where to save the visualizations
        prefix: Filename prefix for the saved images
        question: Optional question text
        gt_answer: Optional ground truth answer (string or list of strings)
        pred_answer: Optional predicted answer
    """

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Convert image to RGB if needed
    image = image.convert("RGB")
    W, H = image.size

    # Prepare upsampled mask and bounding boxes
    mask_upsampled = upsample_mask_to_image(q_mask_pred, patch_shape, (H, W))
    pred_bbox_xyxy = cxcywh_norm_to_xyxy_pix(pred_bbox_norm, W, H)
    gt_bbox_xyxy = cxcywh_norm_to_xyxy_pix(gt_bbox_norm, W, H)

    # Original document
    save_document_example(
        image=image,
        save_path=os.path.join(save_dir, f"{prefix}_original.png")
    )

    # Document with q-mask overlay
    save_document_example(
        image=image,
        save_path=os.path.join(save_dir, f"{prefix}_qmask.png"),
        mask_overlay=mask_upsampled
    )

    # Original with pred (red) and GT (green) boxes
    save_document_example(
        image=image,
        save_path=os.path.join(save_dir, f"{prefix}_bboxes.png"),
        pred_bbox_xyxy=pred_bbox_xyxy,
        gt_bbox_xyxy=gt_bbox_xyxy
    )

    # Q-mask with pred box (red)
    save_document_example(
        image=image,
        save_path=os.path.join(save_dir, f"{prefix}_qmask_pred.png"),
        mask_overlay=mask_upsampled,
        pred_bbox_xyxy=pred_bbox_xyxy
    )

    # Save metadata as JSON
    metadata = {
        "question": question,
        "gt_answer": gt_answer,
        "pred_answer": pred_answer
    }
    with open(os.path.join(save_dir, f"{prefix}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
