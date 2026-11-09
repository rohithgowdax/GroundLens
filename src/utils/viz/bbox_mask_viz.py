"""
Visualization utilities for debugging bbox masking in decoder attention.
"""

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def create_masked_image(
    image: Image.Image,
    bbox: torch.Tensor,
    blur_sigma: float
) -> Image.Image:
    """
    Create a masked version of an image using bbox with Gaussian blur.
    This replicates the masking logic from model._reencode_with_bbox_mask.

    Args:
        image: PIL Image
        bbox: (4,) tensor - [cx, cy, w, h] in normalized coordinates
        blur_sigma: Sigma for Gaussian blur

    Returns:
        PIL Image with mask applied
    """
    device = bbox.device

    # Convert PIL to tensor
    img_t = TF.to_tensor(image).unsqueeze(0).to(device)  # (1, 3, H, W)
    _, _, H, W = img_t.shape

    # Extract bbox parameters and convert to pixel coordinates
    cx, cy, w, h = bbox
    cx_px = cx * W
    cy_px = cy * H
    w_px = w * W
    h_px = h * H

    # Edge case: if bbox is too small, use full image
    if w_px < 1 or h_px < 1:
        mask_blurred = torch.ones((1, 1, H, W), device=device)
    else:
        # Create binary mask (1 inside bbox, 0 outside)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=bbox.dtype),
            torch.arange(W, device=device, dtype=bbox.dtype),
            indexing='ij'
        )

        # Compute bbox boundaries
        x_min = cx_px - w_px / 2
        x_max = cx_px + w_px / 2
        y_min = cy_px - h_px / 2
        y_max = cy_px + h_px / 2

        # Create binary mask
        mask = ((x_grid >= x_min) & (x_grid <= x_max) &
                (y_grid >= y_min) & (y_grid <= y_max)).float()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Apply Gaussian blur for smooth edges
        kernel_size = int(6 * blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask_blurred = TF.gaussian_blur(
            mask,
            kernel_size=[kernel_size, kernel_size],
            sigma=[blur_sigma, blur_sigma]
        )

    # Apply mask to image
    masked_img_t = img_t * mask_blurred

    # Convert back to PIL
    masked_img_pil = TF.to_pil_image(masked_img_t.squeeze(0).cpu())

    return masked_img_pil


def add_text_header(image: Image.Image, text: str, font_size: int = 16) -> Image.Image:
    """
    Add a text header above an image.

    Args:
        image: PIL Image
        text: Text to display
        font_size: Font size for text

    Returns:
        New PIL Image with text header
    """
    # Create new image with space for header
    header_height = font_size + 20
    new_img = Image.new('RGB', (image.width, image.height + header_height), color='white')

    # Paste original image below header
    new_img.paste(image, (0, header_height))

    # Draw text
    draw = ImageDraw.Draw(new_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (new_img.width - text_width) // 2
    text_y = 10

    draw.text((text_x, text_y), text, fill='black', font=font)

    return new_img


def visualize_masked_images(
    masked_images: List[Image.Image],
    target_answers: List[str],
    save_dir: Path,
    epoch: int,
    num_samples: int = 8,
    prefix: str = "masked"
) -> None:
    """
    Visualize masked images with answer headers and save to disk.

    Args:
        masked_images: List of PIL images (already masked by model)
        target_answers: List of target answer strings
        save_dir: Directory to save visualizations
        epoch: Current epoch number
        num_samples: Number of samples to visualize
        prefix: Prefix for saved files
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_to_viz = min(num_samples, len(masked_images))

    for i in range(num_to_viz):
        # Add answer as header
        answer_text = f"Answer: {target_answers[i]}"
        img_with_header = add_text_header(masked_images[i], answer_text)

        # Save individual image
        save_path = save_dir / f"{prefix}_epoch{epoch:03d}_sample{i:02d}.png"
        img_with_header.save(save_path)
