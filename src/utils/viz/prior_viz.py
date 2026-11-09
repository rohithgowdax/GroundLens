import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def save_mask_visualization(image: Image.Image, mask: np.ndarray, mask_processed: np.ndarray, save_path: str):
    """Show three views: original, heatmap overlay, and postprocessed heatmap overlay."""
    
    W, H = image.size
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    mask_processed = np.clip(mask_processed, 0.0, 1.0).astype(np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.5))
    
    # Original image
    image = image.convert("RGB")

    # Original image
    axes[0].imshow(image)
    axes[0].axis('off')

    # Overlay
    axes[1].imshow(image)
    axes[1].imshow(mask, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    axes[1].axis('off')

    # Processed mask
    axes[2].imshow(image)
    axes[2].imshow(mask_processed, cmap='jet',  alpha=0.5, vmin=0, vmax=1)
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


def save_patch_level_mask_visualization(image: Image.Image, patch_mask: np.ndarray, h_patch: int, w_patch: int, save_path: str):
    """
    Save the patch-level prior overlaid on the document.
    """
    W, H = image.size
    
    # Reshape patch mask if it's flattened
    if patch_mask.ndim == 1:
        patch_mask = patch_mask.reshape(h_patch, w_patch)
    
    # Clip to [0, 1]
    patch_mask = np.clip(patch_mask, 0.0, 1.0).astype(np.float32)
    
    # Convert image to RGB
    image = image.convert("RGB")
    
    # Upsample patch mask to match image resolution
    patch_mask_tensor = torch.from_numpy(patch_mask).float()
    patch_mask_tensor = patch_mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, h_patch, w_patch)
    
    # Interpolate to image size
    patch_mask_upsampled = F.interpolate(
        patch_mask_tensor,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()  # (H, W)
    
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Display image
    ax.imshow(image)
    
    # Overlay upsampled patch mask
    ax.imshow(patch_mask_upsampled, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    
    # Remove axes
    ax.axis('off')
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def save_prior_and_bbox_visualization(image: Image.Image, prior_heatmap: np.ndarray, gt_bbox: list, save_path: str):
    """
    Visualize the prior heatmap overlaid on the image with the ground truth bounding box.
    """
    # Ensure image is RGB
    image = image.convert("RGB")
    W, H = image.size

    # Clip heatmap to [0, 1]
    prior_heatmap = np.clip(prior_heatmap, 0.0, 1.0).astype(np.float32)

    # Upsample prior heatmap to match image resolution
    prior_heatmap_tensor = torch.from_numpy(prior_heatmap).float()
    prior_heatmap_tensor = prior_heatmap_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, h_patch, w_patch)

    # Interpolate to image size
    prior_heatmap_upsampled = F.interpolate(
        prior_heatmap_tensor,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()  # (H, W)

    # Extract bbox coordinates
    x1, y1, x2, y2 = gt_bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Display image
    ax.imshow(image)

    # Overlay upsampled heatmap
    ax.imshow(prior_heatmap_upsampled, cmap='jet', alpha=0.5, vmin=0, vmax=1)

    # Draw bounding box
    rect = patches.Rectangle(
        (x1, y1), bbox_width, bbox_height,
        linewidth=3,
        edgecolor='red',
        facecolor='none',
        linestyle='-'
    )
    ax.add_patch(rect)

    # Remove axes
    ax.axis('off')

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()