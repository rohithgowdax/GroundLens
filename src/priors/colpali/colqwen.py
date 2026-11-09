import os
import re
import torch
import torch.nn.functional as F
import time 
import gc
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.interpretability import get_similarity_maps_from_embeddings


# Import implementation
from src.priors.colpali.common import colpali_score_to_mask
from src.priors.common import pixel_prior_to_pix2struct_patches
from src.utils.viz.prior_viz import save_mask_visualization, save_patch_level_mask_visualization
from src.utils.post_processing import postprocess_prior_content_aware
from src.datasets.ops import resize_img


@torch.no_grad()
def run_colqwen_pipeline(ds, out_dir, logger, args, max_width=800):
    # Start time of generation
    start_time = time.time()

    # Load model
    logger.info("Loading ColQwen: vidore/colqwen2.5-v0.2")
    model = ColQwen2_5.from_pretrained(
        "vidore/colqwen2.5-v0.2",
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
    ).eval()

    processor = ColQwen2_5_Processor.from_pretrained(
        "vidore/colqwen2.5-v0.2", use_fast=True, cache_dir=args.cache_dir
    )
    logger.info("Model and processor ready")

    num_saved = 0
    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    bs = int(args.batch_size)

    for split in ds.keys():
        n = len(ds[split])
        logger.info(f"Processing split: {split} (n={n}, batch_size={bs})")

        # Process with map
        def process_batch_fn(examples, indices):
            """Process a batch and return the new columns"""
            batch_size = len(examples["image"])
            
            # Process each image
            masks_raw = []
            masks_postprocessed = []
            metas = []

            # For patch-level only mode
            patch_priors = []
            patch_shapes = []
            
            nonlocal num_saved
            
            for idx in range(batch_size):
                pil = examples["image"][idx]
                if not isinstance(pil, Image.Image):
                    pil = resize_img(Image.open(pil), max_width=max_width)
                
                q_text = examples["question"][idx]
                
                # Encode one at a time
                q_inputs = processor.process_queries([q_text]).to(device)
                v_inputs = processor.process_images([pil]).to(device)
               
                # Get embeddings
                query_embeddings = model(**q_inputs)    # (1, T, d)
                image_embeddings = model(**v_inputs)     # (1, L, d)

                # Normalize
                query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
                image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

                W, H = pil.size
                
                # Get number of image patches 
                n_patches_raw = processor.get_n_patches(
                    image_size=pil.size, 
                    spatial_merge_size=model.spatial_merge_size,
                )
                
                # Ensure n_patches matches image aspect ratio
                if isinstance(n_patches_raw, tuple) and len(n_patches_raw) == 2:
                    img_aspect = W / H
                    patch_aspect_as_is = n_patches_raw[1] / n_patches_raw[0]  # w/h
                    patch_aspect_swapped = n_patches_raw[0] / n_patches_raw[1]  # h/w
                    
                    # If swapped aspect is closer to image aspect, swap them
                    if abs(img_aspect - patch_aspect_swapped) < abs(img_aspect - patch_aspect_as_is):
                        n_patches = (n_patches_raw[1], n_patches_raw[0])
                    else:
                        n_patches = n_patches_raw
                else:
                    n_patches = n_patches_raw

                # Get image mask to identify valid patches
                image_mask = processor.get_image_mask(v_inputs)

                # Extract the mask for this specific image
                if image_mask.dim() == 2:
                    img_mask_b = image_mask[0]  # (L,)
                else:
                    img_mask_b = image_mask[0, :, 0]  # Handle potential extra dims

                # Similarity maps on selected image tokens
                maps_b = get_similarity_maps_from_embeddings(
                    image_embeddings=image_embeddings[0:1],   # (1, L, d)
                    query_embeddings=query_embeddings[0:1],   # (1, T, d)
                    n_patches=n_patches,                      # (h, w)
                    image_mask=img_mask_b.unsqueeze(0),       # (1, L)
                )[0]  # Returns (T, h, w)
                
                # Apply temperature-controlled pooling over query tokens
                tau = 0.05
                token_importance = maps_b.flatten(1).max(dim=-1).values
                weights = torch.softmax(token_importance / tau, dim=0)
                
                # Weighted sum over query tokens
                patch_scores = (maps_b * weights[:, None, None]).sum(dim=0)  # (h, w)
                
                # Move to CPU and convert to float32
                patch_scores = patch_scores.cpu().float()
                
                # Clear GPU memory
                del query_embeddings, image_embeddings, q_inputs, v_inputs
                del maps_b, token_importance, weights, image_mask, img_mask_b
                torch.cuda.empty_cache()
                gc.collect()
                
                # Normalize scores to [0, 1]
                patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-8)
                
                # Upsample to full image resolution
                mask_hw = colpali_score_to_mask(patch_scores, pil.size)  
                del patch_scores
                
                mask_hw = gaussian_filter(mask_hw, sigma=2.0)

                # Post-process
                mask_processed = postprocess_prior_content_aware(
                    mask=mask_hw, 
                    image=pil,
                    min_attention_threshold=0.1,
                    variance_threshold=0.005
                )

                # Patch level mask
                patch_level_prior, patch_size_h_w = pixel_prior_to_pix2struct_patches(
                    pixel_prior=mask_processed,
                    image_size=pil.size,
                )

                # Store patch-level prior data
                patch_priors.append(patch_level_prior.reshape(-1).tolist())
                patch_shapes.append(list(patch_size_h_w))

                # Store both raw and postprocessed masks
                masks_raw.append(mask_hw.reshape(-1).tolist())
                masks_postprocessed.append(mask_processed.reshape(-1).tolist())
                metas.append({
                    "prior_family": "late_interaction",
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

                    # Save patch level or pixel level examples 
                    if args.save_patch_prior_only:
                        h_patch, w_patch = patch_size_h_w
                        save_patch_level_mask_visualization(
                            image=pil,
                            h_patch=h_patch,
                            w_patch=w_patch,
                            patch_mask=patch_level_prior,
                            save_path=vis_path,
                        )
                    else:
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

            # Return patch-level only if flag is set, otherwise return full masks
            if args.save_patch_prior_only:
                return {
                    "q_prior_patch": patch_priors,
                    "q_prior_patch_shape": patch_shapes,
                    "q_prior_meta": metas,
                }
            else:
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

    # Final time logging
    total_end_time = time.time()
    total_elapsed = total_end_time - start_time
    logger.info(f"Completed all splits in {total_elapsed/60.0:.2f} minutes")

    return ds  