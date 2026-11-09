import torch
import numpy as np
import torch.nn.functional as F


def pixel_prior_to_pix2struct_patches(
    pixel_prior: np.ndarray,
    image_size: tuple,
    max_patches: int = 512,
) -> tuple:
    """
    Convert pixel-level prior to Pix2Struct patch-level prior.
    Respects Pix2Struct's dynamic patch grid based on image aspect ratio and max_patches.
    """
    W_img, H_img = image_size
    aspect_ratio = W_img / H_img
    
    # Calculate Pix2Struct's patch grid dimensions
    # Pix2Struct allocates patches to respect aspect ratio while staying under max_patches
    if aspect_ratio >= 1:
        # Landscape or square
        W_patches = int(np.sqrt(max_patches * aspect_ratio))
        H_patches = int(max_patches / W_patches)
    else:
        # Portrait
        H_patches = int(np.sqrt(max_patches / aspect_ratio))
        W_patches = int(max_patches / H_patches)
    
    # Ensure we don't exceed max_patches
    while H_patches * W_patches > max_patches:
        if aspect_ratio >= 1:
            W_patches -= 1
        else:
            H_patches -= 1
    
    # Convert pixel prior to torch tensor
    prior_tensor = torch.from_numpy(pixel_prior).float()
    
    # Resize pixel prior to match Pix2Struct's patch grid
    # Use bilinear interpolation for smooth downsampling
    prior_tensor = prior_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    patch_prior = F.interpolate(
        prior_tensor,
        size=(H_patches, W_patches),
        mode='bilinear',
        align_corners=False
    ).squeeze()  # (H_patches, W_patches)
    
    return patch_prior.numpy(), (H_patches, W_patches)