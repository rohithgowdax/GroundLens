import argparse
import json
import os
import sys
import logging
import numpy as np
import torch
from datasets import load_dataset

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import implementations
from src.utils.seed import set_seed
from src.datasets.ops import normalize_answers
from src.datasets.collates import get_collate
from src.datasets.DocVqaAnswer import get_docvqa_answer_dataloader
from src.utils.gen_args import get_deterministic_gen_kwargs
from src.model.model import CoExVQA
from src.model.losses.projection_loss import pred_norm_cxcywh_to_xyxy
from src.evaluation.docvqa_metrics import calculate_anls
from src.utils.viz.example_prediction_viz import (
    save_document_example,
    save_spotlight_bbox_example,
    save_spotlight_mask_example,
    upsample_mask_to_image,
    cxcywh_norm_to_xyxy_pix,
)


# Default paths
PROJECT_AREA_PATH = "/cluster/work/projects/nn8104k/kjetil_thesis"
SAVE_DIR_PATH = f"{PROJECT_AREA_PATH}/dual_heatmap_projector"
CACHE_DIR = f"{PROJECT_AREA_PATH}/.cache"
DS_NAME_DEFAULT = "indrehus/docvqa-single-page-questions-answer-ocr"


@torch.no_grad()
def generate_examples(
    model,
    val_loader,
    num_examples: int,
    save_dir: str,
    hard_mask_pct: float = 80.0,
    anls_correct_threshold: float = 0.7,
    anls_partial_threshold: float = 0.05,
):
    model.eval()
    gen_kwargs = get_deterministic_gen_kwargs()

    subfolders = {
        "correct": os.path.join(save_dir, "correct"),
        "partial": os.path.join(save_dir, "partial"),
        "incorrect": os.path.join(save_dir, "incorrect"),
    }
    for path in subfolders.values():
        os.makedirs(path, exist_ok=True)

    jsonl_path = os.path.join(save_dir, "metadata.jsonl")

    collected = 0
    with open(jsonl_path, "w") as jsonl_f:
        for batch in val_loader:
            if collected >= num_examples:
                break

            images = batch["images"]
            questions = batch["questions"]
            first_answer, all_answers = normalize_answers(batch["answers"])

            outputs = model(
                images=images,
                questions=questions,
                decoder_labels=None,
                gt_boxes=None,
            )
            box_pred = outputs["box"]
            q_mask = outputs.get("q_mask", None)

            pred_bbox_norm_cxcywh = box_pred.detach().cpu()

            preds = model.generate_preds(
                images=images,
                questions=questions,
                gen_kwargs=gen_kwargs,
                gt_boxes=None,
            )

            B = len(questions)
            for j in range(B):
                if collected >= num_examples:
                    break

                prefix = f"example_{collected:04d}"
                image = images[j] if isinstance(images, (list, tuple)) else images[j]
                image = image.convert("RGB")
                W, H = image.size

                pred_norm = pred_bbox_norm_cxcywh[j].tolist()
                pred_bbox_xyxy = cxcywh_norm_to_xyxy_pix(pred_norm, W, H)

                # Compute ANLS and determine subfolder
                gt_ans = all_answers[j] if j < len(all_answers) else []
                pred_ans = preds[j] if j < len(preds) else ""
                anls_score = calculate_anls(gt_ans, pred_ans, threshold=0.5)
                if anls_score >= anls_correct_threshold:
                    subfolder_key = "correct"
                elif anls_score >= anls_partial_threshold:
                    subfolder_key = "partial"
                else:
                    subfolder_key = "incorrect"
                example_dir = subfolders[subfolder_key]

                # Original document image
                image.save(os.path.join(example_dir, f"{prefix}_original.png"))

                # Answer mask — spotlight (outside bbox dimmed black, inside visible)
                save_spotlight_bbox_example(
                    image=image,
                    bbox_xyxy=pred_bbox_xyxy,
                    save_path=os.path.join(example_dir, f"{prefix}_answer_hard_mask.png"),
                )

                # Answer mask — red rectangle outline
                save_document_example(
                    image=image,
                    save_path=os.path.join(example_dir, f"{prefix}_answer_red_rect.png"),
                    pred_bbox_xyxy=pred_bbox_xyxy,
                )

                # Question heatmap variants
                if q_mask is not None:
                    q_mask_i = q_mask[j].detach().cpu()

                    # Infer patch shape from q_mask size
                    if "q_prior_patch_shape" in batch:
                        patch_shape = tuple(batch["q_prior_patch_shape"][j].tolist())
                    else:
                        num_patches = q_mask_i.shape[0]
                        h_patch = int(num_patches ** 0.5)
                        patch_shape = (h_patch, h_patch)

                    soft_mask = upsample_mask_to_image(q_mask_i, patch_shape, (H, W))

                    # Soft heatmap (jet colormap)
                    save_document_example(
                        image=image,
                        save_path=os.path.join(example_dir, f"{prefix}_qheatmap_soft.png"),
                        mask_overlay=soft_mask,
                    )

                    # Soft heatmap + predicted bbox overlaid
                    save_document_example(
                        image=image,
                        save_path=os.path.join(example_dir, f"{prefix}_qheatmap_soft_bbox.png"),
                        mask_overlay=soft_mask,
                        pred_bbox_xyxy=pred_bbox_xyxy,
                    )

                    # Hard mask — adaptive percentile threshold so it's never one solid color
                    threshold = float(np.percentile(soft_mask, hard_mask_pct))
                    hard_mask_np = (soft_mask >= threshold).astype(float)
                    save_spotlight_mask_example(
                        image=image,
                        mask=hard_mask_np,
                        save_path=os.path.join(example_dir, f"{prefix}_qheatmap_hard.png"),
                    )

                # Write metadata
                record = {
                    "example_id": prefix,
                    "subfolder": subfolder_key,
                    "anls": anls_score,
                    "question": questions[j],
                    "gt_answer": gt_ans,
                    "pred_answer": pred_ans,
                }
                jsonl_f.write(json.dumps(record) + "\n")
                jsonl_f.flush()

                collected += 1
                logger.info(f"Saved {prefix} ({collected}/{num_examples})")

            del box_pred, q_mask, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(f"Done. {collected} examples saved to: {save_dir}")
    logger.info(f"Metadata written to: {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate user evaluation examples from a pretrained CoExVQA model")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR_PATH, help="Directory to save generated examples")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--hard_mask_pct", type=float, default=80.0, help="Percentile for adaptive q-mask hard threshold (top N%% patches stay active)")
    parser.add_argument("--anls_correct_threshold", type=float, default=0.7, help="ANLS >= this → 'correct' subfolder")
    parser.add_argument("--anls_partial_threshold", type=float, default=0.1, help="ANLS >= this (but < correct) → 'partial' subfolder, else 'incorrect'")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Generating user evaluation examples")

    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Seed set to: {args.seed}")

    # Set up device and CUDA settings
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Setup the save directory for examples
    example_save_dir = os.path.join(args.save_dir, "user_eval_examples")
    os.makedirs(example_save_dir, exist_ok=True)
    logger.info(f"Examples will be saved to: {example_save_dir}")

    # Load the CoExVQA model from huggingface
    logger.info("Loading CoExVQA model from HuggingFace (indrehus/CoExVQA)...")
    model = CoExVQA.from_hf(cache_dir=args.cache_dir).to(device)
    logger.info("CoExVQA model loaded.")

    # Set up the DocVQA validation dataloader and shuffle it to get a random sample of examples each time
    collate_fn = get_collate("answer_resize_bbox", max_width=800)
    logger.info(f"Loading DocVQA validation set (shuffled) from: {DS_NAME_DEFAULT}")
    ds = load_dataset(DS_NAME_DEFAULT, cache_dir=args.cache_dir)
    val_loader = get_docvqa_answer_dataloader(
        ds=ds["validation"],
        split="validation",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() // 2),
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    logger.info("Val dataloader ready.")

    # Generate and save examples
    logger.info(f"Starting example generation for User Evaluation. Target number of examples: {args.num_examples}")
    generate_examples(
        model=model,
        val_loader=val_loader,
        num_examples=args.num_examples,
        save_dir=example_save_dir,
        hard_mask_pct=args.hard_mask_pct,
        anls_correct_threshold=args.anls_correct_threshold,
        anls_partial_threshold=args.anls_partial_threshold,
    )
