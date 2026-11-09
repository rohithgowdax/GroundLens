import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Configure matplotlib to use a default system font instead of downloading Arial
# This prevents timeout issues during training when running on HPC clusters
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']


def save_q_mask_prediction_visualization(
    image: Image.Image,
    q_mask_pred: torch.Tensor,
    patch_shape: tuple,
    epoch: int,
    save_path: str
):
    """
    Save the predicted question mask overlaid on the document image.
    Title of the image is the epoch, so that the training script can keep track of how the predictions change over time.

    Args:
        image: PIL Image of the document
        q_mask_pred: Predicted question mask tensor, shape (num_patches, 1) with values in [0, 1]
        patch_shape: Tuple of (h_patch, w_patch) indicating the patch grid dimensions
        epoch: Current epoch number for the title
        save_path: Directory path where to save the visualization
    """
    # Ensure image is RGB
    image = image.convert("RGB")
    W, H = image.size

    # Get patch dimensions
    h_patch, w_patch = patch_shape
    num_patches = h_patch * w_patch

    # Extract the predicted mask for image patches only
    # q_mask_pred is (seq_len, 1), we take first num_patches
    mask_patches = q_mask_pred[:num_patches].squeeze(-1)  # (num_patches,)

    # Reshape to patch grid
    mask_grid = mask_patches.reshape(h_patch, w_patch)  # (h_patch, w_patch)

    # Clip to [0, 1]
    mask_grid = mask_grid.clamp(0.0, 1.0)

    # Convert to numpy for interpolation
    mask_grid_np = mask_grid.cpu().float().numpy()

    # Upsample mask to match image resolution using torch interpolation
    mask_tensor = torch.from_numpy(mask_grid_np).float()
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, h_patch, w_patch)

    # Interpolate to image size
    mask_upsampled = F.interpolate(
        mask_tensor,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()  # (H, W)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Display image
    ax.imshow(image)

    # Overlay predicted mask heatmap
    ax.imshow(mask_upsampled, cmap='jet', alpha=0.5, vmin=0, vmax=1)

    # Set title with epoch number
    ax.set_title(f"Epoch {epoch}", fontsize=14, pad=10)

    # Remove axes
    ax.axis('off')

    plt.tight_layout()

    # Create directory if needed
    os.makedirs(save_path, exist_ok=True)

    # Save with epoch-specific filename
    filename = f"epoch_{epoch}.png"
    full_path = os.path.join(save_path, filename)

    # Save with high DPI
    plt.savefig(full_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def save_masked_q_mask_visualization(
    image: Image.Image,
    q_mask_pred: torch.Tensor,
    patch_shape: tuple,
    masking_type: str,
    mask_probability: float,
    save_dir: str,
    sample_idx: int = 0
):
    """
    Save the predicted question mask overlaid on the document image for masking experiments.
    Filename includes the masking method and probability.

    Args:
        image: PIL Image of the document
        q_mask_pred: Predicted question mask tensor, shape (num_patches, 1) with values in [0, 1]
        patch_shape: Tuple of (h_patch, w_patch) indicating the patch grid dimensions
        masking_type: Type of masking applied (e.g., "mask_overlapping", "mask_non_overlapping")
        mask_probability: Probability used for masking (e.g., 0.1, 0.5, 1.0)
        save_dir: Directory path where to save the visualization
        sample_idx: Index of the sample for unique filename
    """
    image = image.convert("RGB")
    W, H = image.size

    h_patch, w_patch = patch_shape
    num_patches = h_patch * w_patch

    mask_patches = q_mask_pred[:num_patches].squeeze(-1)
    mask_grid = mask_patches.reshape(h_patch, w_patch)
    mask_grid = mask_grid.clamp(0.0, 1.0)

    mask_grid_np = mask_grid.cpu().float().numpy()
    mask_tensor = torch.from_numpy(mask_grid_np).float()
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

    mask_upsampled = F.interpolate(
        mask_tensor,
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    fig_w = 8
    fig_h = fig_w * (H / W)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    ax.imshow(image)
    ax.imshow(mask_upsampled, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax.axis('off')

    plt.tight_layout(pad=0)

    os.makedirs(save_dir, exist_ok=True)

    filename = f"{masking_type}_prob{mask_probability:.2f}_sample{sample_idx:02d}.png"
    full_path = os.path.join(save_dir, filename)

    plt.savefig(full_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
