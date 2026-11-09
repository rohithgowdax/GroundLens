import os
import re
import torch
import torch.nn.functional as F
import time 
import gc
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.utils.viz.prior_viz import save_mask_visualization
from src.utils.post_processing import postprocess_prior_content_aware
from src.datasets.ops import resize_img


def extract_attention_during_generation(model, processor, image, question, device, max_new_tokens=30):
    """
    Generate a few tokens and extract cross-attention weights.
    This gives more meaningful attention patterns than just using a start token.
    """
    # Prepare inputs
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt",
        max_patches=512,
    ).to(device)
    
    # Generate with attention output
    with torch.no_grad():
        outputs = model.generate(
            flattened_patches=inputs["flattened_patches"],
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    
    # Average over all generated tokens
    all_cross_attentions = []
    for token_attentions in outputs.cross_attentions: 
        if token_attentions:
            # Stack all layers
            stacked = torch.stack(token_attentions, dim=0)

            # Average over layers and heads
            avg = stacked.mean(dim=(0, 2, 3))
            all_cross_attentions.append(avg)
    
    # Average over all generated tokens
    final_attention = torch.stack(all_cross_attentions, dim=0).mean(dim=0)
    return final_attention, inputs


def extract_encoder_attention(flattened_patches, model):
    """
    Extract self-attention from encoder as a fallback.
    Average attention to each patch across all layers.
    """
    encoder_outputs = model.encoder(
        flattened_patches=flattened_patches,
        output_attentions=True,
        return_dict=True,
    )
    
    # Get self-attentions from encoder
    attentions = encoder_outputs.attentions 
    
    if not attentions:
        raise ValueError("No attentions from encoder")
    
    # Stack and average across layers and heads
    stacked = torch.stack(attentions, dim=0)
    avg_attn = stacked.mean(dim=(0, 2))
    
    # Average across query dimension to get attention TO each patch: (batch, seq)
    patch_importance = avg_attn.mean(dim=1)
    
    return patch_importance


def get_pix2struct_patch_grid(image_size, num_patches, patch_size=16):
    """
    Calculate the patch grid for Pix2Struct more accurately.
    """
    W, H = image_size
    aspect_ratio = W / H
    
    # Start with square root as base
    base = int(np.sqrt(num_patches))
    
    best_h, best_w = base, base
    best_diff = abs(best_h * best_w - num_patches)
    
    # Search for better combinations
    for h in range(max(1, base - 10), base + 10):
        w = int(round(h * aspect_ratio))
        diff = abs(h * w - num_patches)
        
        if diff < best_diff:
            best_diff = diff
            best_h, best_w = h, w
    
    # Fine-tune to get closer to exact num_patches
    while best_h * best_w < num_patches:
        if best_w / best_h < aspect_ratio:
            best_w += 1
        else:
            best_h += 1

    # Reduce if overshot
    while best_h * best_w > num_patches:
        if best_w / best_h > aspect_ratio:
            best_w -= 1
        else:
            best_h -= 1
    
    
    return best_h, best_w


def attention_to_spatial_map(attention_weights, image_size, num_patches):
    """
    Convert flat attention weights to 2D spatial map.
    """
    W, H = image_size
    
    # Get the grid dimensions
    grid_h, grid_w = get_pix2struct_patch_grid(image_size, num_patches)
    
    # Create spatial map
    expected_patches = grid_h * grid_w
    actual_patches = len(attention_weights)
    
    if actual_patches >= expected_patches:
        # Truncate to expected size
        spatial_map = attention_weights[:expected_patches].reshape(grid_h, grid_w)
    else:
        # Pad with zeros
        padded = torch.zeros(expected_patches, device=attention_weights.device)
        padded[:actual_patches] = attention_weights
        spatial_map = padded.reshape(grid_h, grid_w)

    return spatial_map


def upsample_to_image_size(patch_scores, target_size):
    """
    Upsample patch scores to full image resolution.
    """
    
    # Add batch and channel dimensions
    scores_4d = patch_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
    
    # Upsample using bilinear interpolation
    upsampled = F.interpolate(
        scores_4d,
        size=(target_size[1], target_size[0]),  # (height, width)
        mode='bilinear',
        align_corners=False
    )

    return upsampled.squeeze(0).squeeze(0)  


@torch.no_grad()
def run_pix2struct_pipeline(ds, out_dir, logger, args, max_width=800):
    # Start time of generation
    start_time = time.time()

    # Load model
    logger.info("Loading Pix2Struct: google/pix2struct-docvqa-base")
    model = AutoModelForImageTextToText.from_pretrained(
        "google/pix2struct-docvqa-base",
        cache_dir=args.cache_dir,
    ).eval()
    model.to("cuda:0")
    processor = AutoProcessor.from_pretrained(
        "google/pix2struct-docvqa-base",
        cache_dir=args.cache_dir
    )
    logger.info("Model and processor ready")

    # Interate through the splits
    num_saved = 0
    device = next(model.parameters()).device
    bs = int(args.batch_size)
    for split in ds.keys():
        n = len(ds[split])
        logger.info(f"Processing split: {split} (n={n}, batch_size={bs})")

        # Process with map
        def process_batch_fn(examples, indices):
            """Process a batch and return the new columns"""
            batch_size = len(examples["image"])
            logger.info(f"Processing batch of size {batch_size}")
            
            # Process each image
            masks_raw = []
            masks_postprocessed = []
            metas = []
            
            nonlocal num_saved
            
            for idx in range(batch_size):
                pil = examples["image"][idx]
                if not isinstance(pil, Image.Image):
                    pil = resize_img(Image.open(pil), max_width=max_width)
                
                q_text = examples["question"][idx]
                
                # Extract attention during generation for better patterns
                attention_map, inputs = extract_attention_during_generation(
                    model=model,
                    processor=processor,
                    image=pil,
                    question=q_text,
                    device=device,
                    max_new_tokens=30
                )
                
                # Get number of patches
                num_patches = inputs['flattened_patches'].shape[1]
                
                # Convert to spatial map
                W, H = pil.size
                patch_scores = attention_to_spatial_map(
                    attention_weights=attention_map[0],  # Remove batch dim
                    image_size=(W, H),
                    num_patches=num_patches
                )
                
                # Move to CPU and convert to float32
                patch_scores = patch_scores.cpu().float()
                
                # Clear GPU memory
                del attention_map, inputs
                torch.cuda.empty_cache()
                gc.collect()
                
                # Normalize scores to [0, 1]
                min_val = patch_scores.min()
                max_val = patch_scores.max()
                patch_scores = (patch_scores - min_val) / (max_val - min_val)

                # Upsample to full image resolution
                mask_hw = upsample_to_image_size(patch_scores, (W, H))
                mask_hw = mask_hw.numpy()
                del patch_scores
                
                # Apply Gaussian smoothing
                mask_hw = gaussian_filter(mask_hw, sigma=2.0)

                # Post-process
                mask_processed = postprocess_prior_content_aware(
                    mask=mask_hw, 
                    image=pil,
                    min_attention_threshold=0.1,
                    variance_threshold=0.005
                )

                # Store both raw and postprocessed masks 
                masks_raw.append(mask_hw.reshape(-1).tolist())
                masks_postprocessed.append(mask_processed.reshape(-1).tolist())
                metas.append({
                    "prior_family": "cross_attention",
                    "prior_impl": args.model_name,
                    "version": args.version,
                })

                # Visualizations
                if args.num_examples > 0 and num_saved < args.num_examples:
                    ex_idx = indices[idx]
                    ex = {k: v[idx] for k, v in examples.items()}
                    a_text = ex["answers"][0]
                    name = f"{q_text}__{a_text}".lower()
                    name = re.sub(r"\s+", "_", name)
                    name = re.sub(r"[^a-z0-9_\-]+", "", name)[:160] or f"ex_{ex_idx:06d}"
                    vis_path = os.path.join(out_dir, f"{name}.png")

                    save_mask_visualization(
                        image=pil, 
                        mask=mask_hw, 
                        mask_processed=mask_processed, 
                        save_path=vis_path
                    )
                    num_saved += 1
                    logger.info(f"  Saved visualization {num_saved}/{args.num_examples}: {vis_path}")

                    if num_saved >= args.num_examples and args.stop_after_visuals:
                        logger.info(f"Stopping after saving {num_saved} visualizations as requested")
                        raise SystemExit(0)

                # Clear everything
                del mask_hw, mask_processed, pil
                torch.cuda.empty_cache()
                gc.collect()

            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()

            # Return both masks
            return {
                "q_prior_mask": masks_raw,
                "q_prior_mask_postprocessed": masks_postprocessed,
                "q_prior_meta": metas,
            }

        # Process with map (batched, memory-efficient)
        logger.info(f"Mapping process_batch_fn over {split}...")
        ds[split] = ds[split].map(
            process_batch_fn,
            batched=True,
            batch_size=bs,  
            with_indices=True,
            desc=f"Processing {split}",
            num_proc=None,
            load_from_cache_file=False,
            keep_in_memory=False,
            writer_batch_size=4, 
        )

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"Completed split: {split} in {elapsed/60.0:.2f} minutes")

    # Clear model
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    total_end_time = time.time()
    total_elapsed = total_end_time - start_time
    logger.info(f"Completed all splits in {total_elapsed/60.0:.2f} minutes")

    return ds