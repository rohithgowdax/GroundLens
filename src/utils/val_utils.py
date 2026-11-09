"""Helpers for the validation loop extracted from validate_epoch."""

import logging
import math
import os

from src.utils.viz.q_mask_viz import save_q_mask_prediction_visualization
from src.utils.viz import visualize_masked_images


def save_q_mask_visualization(
    model_unwrapped,
    images,
    q_mask,
    full_img_sizes,
    outputs: dict,
    first_answers,
    args,
    epoch: int,
    viz_save_dir: str,
    logger: logging.Logger,
    B: int,
) -> None:
    """Save q_mask and optional masked/cropped image visualizations for one batch.

    Should be called once per epoch (for the first validation batch only).
    Failures are caught and logged as warnings so they never interrupt validation.

    Args:
        model_unwrapped: The model without DDP wrapping.
        images: List of PIL images for the batch.
        q_mask: Predicted question mask tensor, shape (B, seq_len, 1).
        full_img_sizes: Image sizes (W, H), shape (B, 2).
        outputs: Full model output dict (used for ``masked_images_pil`` in debug mode).
        first_answers: List of primary answer strings for the batch.
        args: Parsed argument namespace.
        epoch: Current epoch number.
        viz_save_dir: Directory where visualizations are saved.
        logger: Logger for warnings.
        B: Batch size.
    """
    try:
        # Determine patch size from the backbone processor
        img_proc = model_unwrapped.backbone_processor.image_processor
        if hasattr(img_proc, 'patch_size') and isinstance(img_proc.patch_size, dict):
            patch_h = img_proc.patch_size["height"]
            patch_w = img_proc.patch_size["width"]
        else:
            # Donut (SwinTransformer): compute effective patch size from encoder config
            enc_cfg = model_unwrapped.backbone_model.config.encoder
            downscale = enc_cfg.patch_size * (2 ** len(enc_cfg.depths))
            patch_h = patch_w = downscale

        # Effective patch grid for the first image
        H_eff = full_img_sizes[0, 1].item()
        W_eff = full_img_sizes[0, 0].item()
        H_p_full = int(math.ceil(H_eff / float(patch_h)))
        W_p_full = int(math.ceil(W_eff / float(patch_w)))

        # Account for max_patches truncation while preserving aspect ratio
        seq_len = q_mask.shape[1]
        num_patches = min(H_p_full * W_p_full, seq_len)
        aspect_ratio = W_p_full / H_p_full
        H_p = int(math.sqrt(num_patches / aspect_ratio))
        W_p = int(num_patches / H_p)
        while H_p * W_p > num_patches:
            if W_p > H_p * aspect_ratio:
                W_p -= 1
            else:
                H_p -= 1

        save_q_mask_prediction_visualization(
            image=images[0],
            q_mask_pred=q_mask[0],
            patch_shape=(H_p, W_p),
            epoch=epoch,
            save_path=viz_save_dir,
        )

        # Debug: visualize masked/cropped images when reencoding is active
        if args.debug and model_unwrapped.config.decoder_attention_type in ["bbox_mask_reencode", "bbox_crop_reencode"]:
            masked_images_pil = outputs.get("masked_images_pil")
            if masked_images_pil is not None:
                subdir_name = (
                    "masked_images"
                    if model_unwrapped.config.decoder_attention_type == "bbox_mask_reencode"
                    else "cropped_images"
                )
                visualize_masked_images(
                    masked_images=masked_images_pil,
                    target_answers=first_answers,
                    save_dir=os.path.join(viz_save_dir, subdir_name),
                    epoch=epoch,
                    num_samples=min(8, B),
                )
    except Exception as e:
        logger.debug(f"Q-mask visualization failed (epoch {epoch}): {e}")
