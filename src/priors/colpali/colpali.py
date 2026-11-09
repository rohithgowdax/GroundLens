import os
import re
import torch
import torch.nn.functional as F
import time 
import math
from datasets import DatasetDict, Sequence, Value
from PIL import Image
import gc
from scipy.ndimage import gaussian_filter
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColIdefics3, ColIdefics3Processor


# Import implementation
from src.priors.colpali.common import colpali_score_to_mask
from src.datasets.ops import bbox_to_int
from src.utils.viz.prior_viz import save_mask_visualization, save_prior_and_bbox_visualization
from src.priors.common import pixel_prior_to_pix2struct_patches
from src.utils.post_processing import postprocess_prior_content_aware
from src.datasets.ops import resize_img


def calculate_n_patches_idefics3(image_size, patch_size=16):
    """
    Calculate number of patches for Idefics3/ColSmol.
    """
    W, H = image_size
    n_patches_w = W // patch_size
    n_patches_h = H // patch_size
    return (n_patches_h, n_patches_w)


def load_colsmol_model(cache_dir):
    """Load ColSmol model and processor. Returns (model, processor, device)."""
    model = ColIdefics3.from_pretrained(
        "vidore/colSmol-500M",
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
    ).eval()
    processor = ColIdefics3Processor.from_pretrained("vidore/colSmol-500M", use_fast=True, cache_dir=cache_dir)
    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    return model, processor, device


@torch.no_grad()
def compute_single_prior(model, processor, image, question, device):
    """
    Compute patch-level question prior for a single image-question pair using ColSmol.

    Args:
        model: Loaded ColSmol model
        processor: ColSmol processor
        image: PIL Image (already resized)
        question: Question string
        device: torch device

    Returns:
        patch_prior_flat: list[float] - flattened patch-level prior
        patch_shape: list[int] - [H_patches, W_patches]
        mask_hw: np.ndarray - raw pixel-level mask
        mask_processed: np.ndarray - post-processed pixel-level mask
    """
    # Encode question and image
    q_inputs = processor.process_queries([question]).to(device)
    v_inputs = processor.process_images([image]).to(device)

    query_embeddings = F.normalize(model(**q_inputs), p=2, dim=-1)
    image_embeddings = F.normalize(model(**v_inputs), p=2, dim=-1)

    Q = query_embeddings[0]
    P = image_embeddings[0]

    sims = torch.matmul(Q, P.T)
    patch_scores_flat = sims.max(dim=0).values.cpu()

    del query_embeddings, image_embeddings, q_inputs, v_inputs, Q, P, sims
    torch.cuda.empty_cache()

    # Factor into grid
    actual_patches = patch_scores_flat.shape[0]
    if actual_patches == 1139:
        n_patches_h, n_patches_w = 17, 67
    else:
        sqrt_actual = int(math.sqrt(actual_patches))
        if sqrt_actual * sqrt_actual == actual_patches:
            n_patches_h = n_patches_w = sqrt_actual
        else:
            for h in range(sqrt_actual, 0, -1):
                if actual_patches % h == 0:
                    n_patches_h = h
                    n_patches_w = actual_patches // h
                    break

    patch_scores = patch_scores_flat.view(n_patches_h, n_patches_w)
    patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-8)

    patch_scores_np = patch_scores.float()
    del patch_scores, patch_scores_flat

    # Upsample to full image resolution
    mask_hw = colpali_score_to_mask(patch_scores_np, image.size)
    del patch_scores_np
    mask_hw = gaussian_filter(mask_hw, sigma=2.0)

    # Post-process
    mask_processed = postprocess_prior_content_aware(
        mask=mask_hw,
        image=image,
        min_attention_threshold=0.1,
        variance_threshold=0.005
    )

    # Convert to pix2struct patch grid
    patch_level_prior, patch_size_h_w = pixel_prior_to_pix2struct_patches(
        pixel_prior=mask_processed,
        image_size=image.size,
    )

    patch_prior_flat = patch_level_prior.reshape(-1).tolist()
    patch_shape = list(patch_size_h_w)

    return patch_prior_flat, patch_shape, mask_hw, mask_processed


@torch.no_grad()
def run_col_smol_pipeline(ds, out_dir, logger, args, max_width=800):
    start_time = time.time()

    # Load model and processor
    logger.info("Loading ColPali Smol: vidore/colSmol-500M")
    model, processor, device = load_colsmol_model(args.cache_dir)
    logger.info("Model and processor loaded")

    patch_size = model.config.vision_config.patch_size
    logger.info(f"Found patch_size in config: {patch_size}")

    num_saved = 0
    bs = int(args.batch_size)

    for split in ds.keys():
        n = len(ds[split])
        logger.info(f"Processing split: {split} (n={n}, batch_size={bs})")

        # Process with map
        def process_batch_fn(examples, indices):
            """Process a batch and return the new columns"""
            batch_size = len(examples["image"])

            # Store masks
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
                W, H = pil.size

                # Compute prior using shared helper
                patch_prior_flat, patch_shape, mask_hw, mask_processed = compute_single_prior(
                    model, processor, pil, q_text, device
                )

                # Store patch-level prior data
                patch_priors.append(patch_prior_flat)
                patch_shapes.append(patch_shape)

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

                    answer_bbox = examples["answer_ocr_bbox"][idx]
                    x1, y1, x2, y2 = bbox_to_int(answer_bbox, W, H)
                    gt_box = (x1, y1, x2, y2)

                    # Save patch level or pixel level examples
                    if args.save_patch_prior_only:
                        h_patch, w_patch = patch_shape
                        patch_prior_reshaped = pixel_prior_to_pix2struct_patches(
                            pixel_prior=mask_processed, image_size=pil.size
                        )[0].reshape(h_patch, w_patch)
                        save_prior_and_bbox_visualization(
                            image=pil,
                            gt_bbox=gt_box,
                            prior_heatmap=patch_prior_reshaped,
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

        # Map over the dataset
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


@torch.no_grad()
def run_col_smol_pfl_docvqa_pipeline(data_dir, logger, args, max_width=800):
    """
    Generate question priors for PFL-DocVQA dataset using ColSmol.

    Iterates over both train and val splits, computes patch-level priors,
    and saves each as a .pt file in {data_dir}/q_priors/{split}/{question_id}.pt.
    Existing files are skipped for restartability.
    """
    from src.datasets.PFLDocVQA import PFLDocVQADataset

    start_time = time.time()

    # Load model
    logger.info("Loading ColPali Smol: vidore/colSmol-500M")
    model, processor, device = load_colsmol_model(args.cache_dir)
    logger.info("Model and processor loaded")

    for split in ["val", "train"]:
        # Load dataset
        logger.info(f"Loading PFL-DocVQA split: {split}")
        dataset = PFLDocVQADataset(data_dir=data_dir, split=split, verbose=False, require_q_priors=False)
        n = len(dataset)
        logger.info(f"Processing split: {split} (n={n})")

        # Create output directory
        q_prior_dir = os.path.join(data_dir, "q_priors", split)
        os.makedirs(q_prior_dir, exist_ok=True)

        skipped = 0
        processed = 0

        for idx in range(n):
            sample = dataset[idx]
            save_path = os.path.join(q_prior_dir, f"{sample.question_id}.pt")

            # Skip if already exists
            if os.path.exists(save_path):
                skipped += 1
                continue

            # Load and resize image
            try:
                image = Image.open(sample.image_path).convert("RGB")
            except (OSError, Exception) as e:
                logger.info(f"  Skipping corrupted image {sample.image_path}: {e}")
                skipped += 1
                continue
            image = resize_img(image, max_width=max_width)

            # Compute prior
            patch_prior_flat, patch_shape, _, _ = compute_single_prior(
                model, processor, image, sample.question, device
            )

            # Save as .pt
            torch.save({
                "q_prior_patch": patch_prior_flat,
                "q_prior_patch_shape": patch_shape,
            }, save_path)

            processed += 1
            del image
            torch.cuda.empty_cache()
            gc.collect()

            if (processed + skipped) % 50 == 0:
                logger.info(f"  {split}: {processed + skipped}/{n} (processed={processed}, skipped={skipped})")

        split_time = time.time() - start_time
        logger.info(f"Completed split: {split} in {split_time/60.0:.2f} min (processed={processed}, skipped={skipped})")

    # Cleanup
    del model, processor
    torch.cuda.empty_cache()
    gc.collect()

    # Log total time
    total_elapsed = time.time() - start_time
    logger.info(f"PFL-DocVQA prior generation completed in {total_elapsed/60.0:.2f} minutes")