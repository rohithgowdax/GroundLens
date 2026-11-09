import numpy as np
import os
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

def postprocess_prior_content_aware(mask: np.ndarray, image: Image.Image, min_attention_threshold: float = 0.3, variance_threshold: float = 0.05) -> np.ndarray:
    """
    Content-aware post-processing that removes attention from blank/background regions
    and focuses on text-rich areas.
    """
    mask = np.clip(mask, 0.0, 1.0)
    H, W = mask.shape
    
    # Resize image to match mask dimensions
    if image.size != (W, H):
        image_resized = image.resize((W, H), Image.BILINEAR)
    else:
        image_resized = image
    
    # Convert to grayscale for analysis
    img_gray = np.array(image_resized.convert('L')).astype(np.float32)
    
    # Compute local variance to find content regions (text has high variance)
    window_size = 15
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = cv2.filter2D(img_gray, -1, kernel)
    local_sq_mean = cv2.filter2D(img_gray ** 2, -1, kernel)
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.clip(local_var, 0, None)
    
    # Normalize variance
    if local_var.max() > 0:
        local_var_norm = local_var / local_var.max()
    else:
        local_var_norm = local_var
    
    # Create content mask (1 where there's text/content, 0 where blank)
    content_mask = (local_var_norm > variance_threshold).astype(np.float32)
    
    # Dilate content regions to include surrounding context
    kernel_dilate = np.ones((15, 15), np.uint8)
    content_mask = cv2.dilate(content_mask, kernel_dilate, iterations=1)
    
    # Apply content mask to suppress background attention
    filtered_mask = mask * content_mask
    
    # Apply percentile-based thresholding to remove weak activations
    if filtered_mask.max() > 0:
        threshold_val = min_attention_threshold * filtered_mask.max()
        filtered_mask[filtered_mask < threshold_val] = 0
    
    # Smooth to reduce hard edges
    filtered_mask = gaussian_filter(filtered_mask, sigma=2.0)

    # Suppress outer frame (7% border) to remove edge artifacts
    border_percent = 0.07
    border_h = int(H * border_percent)
    border_w = int(W * border_percent)

    # Create border suppression mask with smooth falloff
    border_mask = np.ones_like(filtered_mask)
    if border_h > 0 and border_w > 0:
        # Apply smooth gradient falloff on borders
        for i in range(border_h):
            weight = i / border_h
            border_mask[i, :] *= weight
            border_mask[-(i+1), :] *= weight
        for j in range(border_w):
            weight = j / border_w
            border_mask[:, j] *= weight
            border_mask[:, -(j+1)] *= weight

    filtered_mask = filtered_mask * border_mask

    # Renormalize to [0, 1]
    if filtered_mask.max() > 0:
        filtered_mask = filtered_mask / filtered_mask.max()

    return filtered_mask.astype(np.float32)
