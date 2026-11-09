import argparse
import os
import sys
import logging
import gc
from time import perf_counter
import torch
import uuid

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import implementations
from src.utils.seed import set_seed
from src.datasets.ops import normalize_answers
from src.datasets.collates import get_collate
from src.datasets.DocVqaAnswer import get_val_docvqa_answer_dataloader
from src.evaluation.answer_localization_evaluator import AnswerLocalizationEvaluator
from src.evaluation.docvqa_metrics import DocVQAEvaluator
from src.evaluation.rate_anls_evaluator import (
    estimate_pi, run_rate_anls_eval,
    estimate_pi_answer, run_answer_rate_anls_eval,
    run_joint_rate_anls_eval
)
from src.evaluation.pfl_docvqa_evaluator import run_pfl_docvqa_eval
from src.datasets.PFLDocVQA import get_val_pfl_docvqa_dataloader
from src.utils.gen_args import get_deterministic_gen_kwargs
from src.model.config import CoExVQAConfig
from src.model.model import CoExVQA
from src.utils.masking_utils import QuestionMaskingHook
from src.model.losses.projection_loss import pred_norm_cxcywh_to_xyxy
from src.utils.viz.example_prediction_viz import save_example_prediction_visualization
from src.utils.viz.q_mask_viz import save_masked_q_mask_visualization


@torch.no_grad()
def run_eval(
    model,
    val_loader,
    num_examples: int,
    plot_save_dir: str,
    masking_type: str = "none",
    top_k_percent: float = 0.40,
    mask_probability: float = 1.0,
    mask_embeddings: bool = False,
):
    model.eval()

    # Evaluators
    loc_eval = AnswerLocalizationEvaluator(iou_thr=0.5, pred_fmt="xyxy")
    doc_eval = DocVQAEvaluator()
    gen_kwargs = get_deterministic_gen_kwargs()

    # Initialize masking hook if needed
    hook_manager = None
    hook_handle = None
    gate_hook_handle = None
    if masking_type != "none":
        logger.info(f"Masking enabled: type={masking_type}, top_k_percent={top_k_percent}, mask_probability={mask_probability}, mask_embeddings={mask_embeddings}")
        hook_manager = QuestionMaskingHook(
            masking_type=masking_type,
            top_k_percent=top_k_percent,
            mask_probability=mask_probability,
            mask_embeddings=mask_embeddings,
        )
        hook_handle = hook_manager.register_hook(model)
        if mask_embeddings:
            gate_hook_handle = hook_manager.register_gate_hook(model)
            logger.info("Gate hook registered for embedding masking")

    # Accumulators
    val_gt_boxes, val_pred_boxes, val_img_sizes = [], [], []
    val_images, val_questions = [], []  # Only store up to num_examples for plotting
    target_answers, pred_answers = [], []
    val_q_masks, val_patch_shapes = [], []
    val_pred_boxes_cxcywh = []
    val_gt_boxes_norm_cxcywh = []  # For visualization
    collected_for_plot = 0  # Track how many examples we've collected for plotting

    start = perf_counter()
    for batch in val_loader:
        images = batch["images"]
        questions = batch["questions"]
        first_answer, all_answers = normalize_answers(batch["answers"])
        gt_boxes = batch["answer_bbox_xyxy"]
        full_img_sizes = batch["full_img_sizes"]
        B = len(questions)

        # Set q_prior data for masking hook if enabled
        if hook_manager is not None:
            if "q_prior_patch" in batch and "q_prior_mask" in batch:
                hook_manager.set_batch_data(
                    q_prior_patch=batch["q_prior_patch"],
                    q_prior_mask=batch["q_prior_mask"]
                )
            else:
                logger.warning(
                    "Masking enabled but batch missing q_prior data. "
                    "Make sure to use 'answer_bbox_with_q_prior' collate function. "
                    "Skipping masking for this batch."
                )

        # Single forward pass through model
        outputs = model(
            images=images,
            questions=questions,
            decoder_labels=None,
            gt_boxes=None 
        )

        # Extract outputs from dict
        box_pred = outputs["box"]
        q_mask = outputs.get("q_mask", None)

        # Convert predicted boxes from cxcywh normalized to xyxy normalized
        pred_xyxy_norm, _ = pred_norm_cxcywh_to_xyxy(box_pred, eps=1e-6)

        # Store cxcywh predictions for visualization
        val_pred_boxes_cxcywh.append(box_pred.detach().cpu())

        # Store GT boxes in normalized cxcywh format for visualization
        gt_boxes_norm_cxcywh = batch["answer_bbox_norm_cxcywh"]
        val_gt_boxes_norm_cxcywh.append(gt_boxes_norm_cxcywh.detach().cpu() if torch.is_tensor(gt_boxes_norm_cxcywh) else torch.as_tensor(gt_boxes_norm_cxcywh))

        # Store q_mask and patch shapes for visualization
        if q_mask is not None:
            val_q_masks.append(q_mask.detach().cpu())
        if "q_prior_patch_shape" in batch:
            val_patch_shapes.append(batch["q_prior_patch_shape"].detach().cpu())

        # Get the GT boxes as tensor
        if not torch.is_tensor(gt_boxes):
            gt_boxes = torch.as_tensor(gt_boxes, device=pred_xyxy_norm.device, dtype=pred_xyxy_norm.dtype)
        else:
            gt_boxes = gt_boxes.to(pred_xyxy_norm)

        # Append prediction for evaluation (detach and move to CPU immediately)
        val_pred_boxes.append(pred_xyxy_norm.detach().cpu())
        val_gt_boxes.append(gt_boxes.detach().cpu())
        val_img_sizes.append(full_img_sizes.detach().cpu())

        # Free the normalized prediction tensor immediately
        del pred_xyxy_norm

        # Keep copies for plotting (only up to num_examples to save memory)
        if collected_for_plot < num_examples:
            remaining = num_examples - collected_for_plot
            if isinstance(images, (list, tuple)):
                for im in images[:remaining]:
                    val_images.append(im)
            else:  # tensor batch (B,C,H,W)
                for i in range(min(B, remaining)):
                    val_images.append(images[i].detach().cpu())
            val_questions.extend(questions[:remaining])
            collected_for_plot += min(B, remaining)

        # Use the model's generate_preds method which handles the full pipeline
        preds = model.generate_preds(
            images=images,
            questions=questions,
            gen_kwargs=gen_kwargs,
            gt_boxes=None
        )
        target_answers.extend(all_answers)
        pred_answers.extend(preds)

        # Clear GPU tensors to prevent accumulation
        del box_pred, gt_boxes, images, outputs, q_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall = perf_counter() - start

    # Cleanup masking hooks
    if hook_handle is not None:
        hook_handle.remove()
        logger.info("Masking hook removed")
    if gate_hook_handle is not None:
        gate_hook_handle.remove()
        logger.info("Gate hook removed")

    # Flatten all collected tensors
    pred_flat = torch.cat([t if isinstance(t, torch.Tensor) else torch.as_tensor(t) for t in val_pred_boxes], dim=0)
    gt_flat   = torch.cat([t if isinstance(t, torch.Tensor) else torch.as_tensor(t) for t in val_gt_boxes], dim=0)
    sz_flat   = torch.cat([t if isinstance(t, torch.Tensor) else torch.as_tensor(t) for t in val_img_sizes], dim=0)

    # Plot examples
    pred_flat_cxcywh = torch.cat(val_pred_boxes_cxcywh, dim=0)
    gt_flat_cxcywh = torch.cat(val_gt_boxes_norm_cxcywh, dim=0)  # Already normalized cxcywh

    # Flatten q_masks and patch_shapes if available
    q_masks_flat = torch.cat(val_q_masks, dim=0) if val_q_masks else None
    patch_shapes_flat = torch.cat(val_patch_shapes, dim=0) if val_patch_shapes else None

    num_plot = min(num_examples, len(val_questions), pred_flat_cxcywh.shape[0], len(val_images))
    for i in range(num_plot):
        # Get patch shape for this sample (default to common 32x32 if not available)
        if patch_shapes_flat is not None:
            patch_shape = tuple(patch_shapes_flat[i].tolist())
        else:
            # Infer from q_mask shape if available
            if q_masks_flat is not None:
                num_patches = q_masks_flat[i].shape[0]
                h_patch = int(num_patches ** 0.5)
                patch_shape = (h_patch, h_patch)
            else:
                patch_shape = (32, 32)

        # Get q_mask for this sample
        if q_masks_flat is not None:
            q_mask_i = q_masks_flat[i]
        else:
            # Create dummy q_mask if not available
            q_mask_i = torch.zeros(patch_shape[0] * patch_shape[1], 1)

        # Get answers
        gt_ans = target_answers[i] if i < len(target_answers) else None
        pred_ans = pred_answers[i] if i < len(pred_answers) else None

        # Save visualization
        save_example_prediction_visualization(
            image=val_images[i],
            q_mask_pred=q_mask_i,
            patch_shape=patch_shape,
            pred_bbox_norm=pred_flat_cxcywh[i].tolist(),
            gt_bbox_norm=gt_flat_cxcywh[i].tolist(),
            save_dir=plot_save_dir,
            prefix=f"example_{i:04d}",
            question=val_questions[i],
            gt_answer=gt_ans,
            pred_answer=pred_ans
        )

        # Save masked q-mask visualization for first 3 samples when masking is enabled
        if masking_type != "none" and i < 3 and q_masks_flat is not None:
            save_masked_q_mask_visualization(
                image=val_images[i],
                q_mask_pred=q_mask_i,
                patch_shape=patch_shape,
                masking_type=masking_type,
                mask_probability=mask_probability,
                save_dir=plot_save_dir,
                sample_idx=i
            )

    # Compute metrics
    loc_metrics = loc_eval.compute(
        gt_boxes=[gt_flat],          
        pred_boxes=[pred_flat],
        image_sizes=[sz_flat],
    )
    doc_metrics = {"anls": 0.0, "acc": 0.0}
    if len(target_answers) > 0:
        doc_metrics = doc_eval.evaluate(target_answers=target_answers, pred_answers=pred_answers)

    return {
        "time_s": wall,
        "localization": loc_metrics,
        "docvqa": doc_metrics,
        "num_batches": len(val_loader),
        "num_samples": int(sz_flat.shape[0]),
    }


# Default paths
PROJECT_AREA_PATH = "/cluster/work/projects/nn8104k/kjetil_thesis"
MODEL_SAVE_DIR_PATH = f"{PROJECT_AREA_PATH}/dual_heatmap_projector"
PRETRAINED_PIX2STRUCT_MODEL_DIR_PATH= f"{PROJECT_AREA_PATH}/pix2struct-masked-train/answer-enc-dec-train-20250921-004330"
CACHE_DIR = f"{PROJECT_AREA_PATH}/.cache"
DS_NAME_DEFAULT = "indrehus/docvqa-single-page-questions-answer-ocr"
DS_NAME_Q_PRIOR = "indrehus/docvqa-single-page-questions-answer-ocr-colSmol500M-q-priors"
DS_NAME_PFL_DOCVQA = "/cluster/work/projects/nn8104k/kjetil_thesis/datasets/PFL-DocVQA/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval DualHeatmapModel checkpoint")
    parser.add_argument("--experiment_name", type=str, default="RX", help="Idetification of the experiement")
    parser.add_argument("--model_checkpoint_dir", type=str, default=(MODEL_SAVE_DIR_PATH + "/dual_heatmap_BASE"), help="Path to model checkpoint directory")
    parser.add_argument("--save_dir", type=str, default=MODEL_SAVE_DIR_PATH, help="Directory to save evaluation outputs")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--batch_size", type=int, default=20)  
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_examples", type=int, default=50, help="Number of examples to render")

    # Evaluation type argument
    parser.add_argument(
        "--eval_type",
        type=str,
        default="standard",
        choices=["standard", "mask_overlapping", "mask_non_overlapping", "rate_anls", "rate_anls_answer", "rate_anls_joint", "pfl_docvqa"],
        help="Evaluation type: 'standard' = normal eval, "
             "'mask_overlapping'/'mask_non_overlapping' = masking experiments, "
             "'rate_anls' = compute Question Rate vs ANLS, "
             "'rate_anls_answer' = compute Answer Rate vs ANLS, "
             "'rate_anls_joint' = joint 2D analysis of Question & Answer Rate vs ANLS, "
             "'pfl_docvqa' = ANLS/ACC evaluation on PFL-DocVQA dataset"
    )

    # PFL-DocVQA evaluation arguments
    parser.add_argument(
        "--pfl_data_dir",
        type=str,
        default=DS_NAME_PFL_DOCVQA,
        help="Path to PFL-DocVQA data directory (required for pfl_docvqa eval_type)"
    )

    # Masking experiment arguments (for mask_overlapping/mask_non_overlapping)
    parser.add_argument(
        "--top_k_percent",
        type=float,
        default=0.40,
        help="Percentile for top-k overlap as decimal (e.g., 0.20 = top 20%). "
             "Overlap = patches in both top k%% by q_mask AND top k%% by q_prior."
    )
    parser.add_argument(
        "--mask_embeddings",
        action="store_true",
        default=False,
        help="Also zero out image_embeddings at masked patch positions (before gating). "
             "Only applies when eval_type is 'mask_overlapping' or 'mask_non_overlapping'."
    )

    # Rate-ANLS evaluation arguments
    parser.add_argument(
        "--pi",
        type=float,
        default=None,
        help="Baseline prior pi for rate computation. If None, estimate from data."
    )
    parser.add_argument(
        "--pi_estimation_batches",
        type=int,
        default=50,
        help="Number of batches to use for pi estimation (if --pi not provided)"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=50,
        help="Number of bins for binned ANLS curve (rate_anls eval types)"
    )
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, 
        handlers=[logging.StreamHandler(sys.stdout)], force=True
    )
    logger = logging.getLogger(__name__)
    logger.info("Start Evaluation of Model")

    # Set seed
    set_seed(args.seed)
    logger.info(f"Seed set to: {args.seed}")

    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Dir to save example plots
    plot_id = uuid.uuid4()
    example_plot_save_dir_path = os.path.join(args.save_dir, f"dual_heatmap_{args.experiment_name}_{plot_id}_plots/")
    os.makedirs(example_plot_save_dir_path, exist_ok=True)
    logger.info(f"Example plots will be saved under: {example_plot_save_dir_path}")

    # Load model
    logger.info("Loading CoExVQA model...")
    model = CoExVQA.from_pretrained(
        path = args.model_checkpoint_dir,
        cache_dir = args.cache_dir,
    ).to(device)
    logger.info("Pretrained CoExVQA model loaded.")

    # Load validation data
    val_loader = None
    if args.eval_type == "pfl_docvqa":
        # PFL-DocVQA dataset
        if args.pfl_data_dir is None:
            raise ValueError("--pfl_data_dir is required for pfl_docvqa evaluation")

        logger.info(f"Loading PFL-DocVQA dataset from: {args.pfl_data_dir}")
        collate_fn = get_collate("pfl_docvqa", max_width=800)
        val_loader = get_val_pfl_docvqa_dataloader(
            data_dir=args.pfl_data_dir,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=min(8, os.cpu_count() // 2),
            pin_memory=True,
            verbose=False
        )
        logger.info("PFL-DocVQA dataloader ready.")
    elif args.eval_type in ["mask_overlapping", "mask_non_overlapping", "rate_anls", "rate_anls_answer", "rate_anls_joint"]:
        # Use q_prior collate and dataset for masking experiments and rate_anls evaluation
        collate_fn = get_collate("answer_bbox_with_q_prior", max_width=800)
        ds_name = DS_NAME_Q_PRIOR
        logger.info(f"Using 'answer_bbox_with_q_prior' collate for eval_type={args.eval_type}")
        logger.info(f"Using q_prior dataset: {ds_name}")
        logger.info("Building val dataloader...")

        val_loader = get_val_docvqa_answer_dataloader(
            ds_name=ds_name,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=min(8, os.cpu_count() // 2),
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        logger.info("Val dataloader ready.")
    else:
        # Standard evaluation
        collate_fn = get_collate("answer_resize_bbox", max_width=800)
        ds_name = DS_NAME_DEFAULT
        logger.info("Using 'answer_resize_bbox' collate")
        logger.info("Building val dataloader...")

        val_loader = get_val_docvqa_answer_dataloader(
            ds_name=ds_name,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=min(8, os.cpu_count() // 2),
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        logger.info("Val dataloader ready.")

    # Run evaluation based on eval_type
    if args.eval_type == "rate_anls":
        # Rate vs ANLS evaluation
        logger.info("Running Rate vs ANLS evaluation...")

        # Estimate pi if not provided
        if args.pi is None:
            logger.info(f"Estimating pi from {args.pi_estimation_batches} batches...")
            pi = estimate_pi(model, val_loader, args.pi_estimation_batches)
            logger.info(f"Estimated pi: {pi:.4f}")
        else:
            pi = args.pi
            logger.info(f"Using provided pi: {pi:.4f}")

        # Run rate-ANLS evaluation
        results = run_rate_anls_eval(
            model=model,
            val_loader=val_loader,
            pi=pi,
            output_dir=example_plot_save_dir_path,
            n_bins=args.n_bins,
        )

        # Log summary
        logger.info("=== Question Rate-ANLS Evaluation Summary ===")
        logger.info(f"Wall time (s): {results['time_s']:.2f}")
        logger.info(f"Num samples : {results['num_samples']}")
        logger.info(f"Pi (baseline): {results['pi']:.4f}")
        logger.info(f"Mean Rate   : {results['mean_rate']:.4f}")
        logger.info(f"Mean ANLS   : {results['mean_anls']:.4f}")
    elif args.eval_type == "rate_anls_answer":
        # Answer Rate vs ANLS evaluation
        logger.info("Running Answer Rate vs ANLS evaluation...")

        # Estimate pi if not provided
        if args.pi is None:
            logger.info(f"Estimating pi from {args.pi_estimation_batches} batches...")
            pi = estimate_pi_answer(model, val_loader, args.pi_estimation_batches)
            logger.info(f"Estimated pi: {pi:.4f}")
        else:
            pi = args.pi
            logger.info(f"Using provided pi: {pi:.4f}")

        # Run answer rate-ANLS evaluation
        results = run_answer_rate_anls_eval(
            model=model,
            val_loader=val_loader,
            pi_A=pi,
            output_dir=example_plot_save_dir_path,
            n_bins=args.n_bins,
        )

        # Log summary
        logger.info("=== Answer Rate-ANLS Evaluation Summary ===")
        logger.info(f"Wall time (s): {results['time_s']:.2f}")
        logger.info(f"Num samples : {results['num_samples']}")
        logger.info(f"Pi (baseline): {results['pi_A']:.4f}")
        logger.info(f"Mean Rate   : {results['mean_rate']:.4f}")
        logger.info(f"Mean ANLS   : {results['mean_anls']:.4f}")
    elif args.eval_type == "rate_anls_joint":
        # Joint Question Rate + Answer Rate vs ANLS evaluation
        logger.info("Running Joint Rate (Rq, Ra) vs ANLS evaluation...")

        # Estimate both priors if not provided
        if args.pi is None:
            logger.info(f"Estimating pi_q (question mask prior) from {args.pi_estimation_batches} batches...")
            pi_q = estimate_pi(model, val_loader, args.pi_estimation_batches)
            logger.info(f"Estimated pi_q: {pi_q:.4f}")

            logger.info(f"Estimating pi_a (answer region prior) from {args.pi_estimation_batches} batches...")
            pi_a = estimate_pi_answer(model, val_loader, args.pi_estimation_batches)
            logger.info(f"Estimated pi_a: {pi_a:.4f}")
        else:
            # If user provides a single pi, use it for both (they can adjust in code if needed)
            pi_q = args.pi
            pi_a = args.pi
            logger.info(f"Using provided pi for both: pi_q={pi_q:.4f}, pi_a={pi_a:.4f}")

        # Run joint rate-ANLS evaluation
        results = run_joint_rate_anls_eval(
            model=model,
            val_loader=val_loader,
            pi_q=pi_q,
            pi_a=pi_a,
            output_dir=example_plot_save_dir_path,
            n_bins=args.n_bins,
        )

        # Log summary
        logger.info("=== Joint Rate-ANLS Evaluation Summary ===")
        logger.info(f"Wall time (s): {results['time_s']:.2f}")
        logger.info(f"Num samples : {results['num_samples']}")
        logger.info(f"Pi_q (question): {results['pi_q']:.4f}")
        logger.info(f"Pi_a (answer) : {results['pi_a']:.4f}")
        logger.info(f"Mean Rq      : {results['mean_rq']:.4f}")
        logger.info(f"Mean Ra      : {results['mean_ra']:.4f}")
        logger.info(f"Mean ANLS    : {results['mean_anls']:.4f}")

        # Log quadrant stats
        joint_stats = results['joint_stats']
        logger.info("=== Quadrant Analysis ===")
        logger.info(f"Rq median: {joint_stats['rq_median']:.4f}")
        logger.info(f"Ra median: {joint_stats['ra_median']:.4f}")
        for name, stats in joint_stats['quadrant_stats'].items():
            logger.info(
                f"  {name}: n={stats['count']}, "
                f"mean_anls={stats['mean_anls']:.4f} ± {stats['std_anls']:.4f}"
            )
    elif args.eval_type == "pfl_docvqa":
        # PFL-DocVQA evaluation
        logger.info("Running PFL-DocVQA evaluation...")

        results = run_pfl_docvqa_eval(
            model=model,
            val_loader=val_loader,
        )

        # Log summary
        logger.info("=== PFL-DocVQA Evaluation Summary ===")
        logger.info(f"Wall time (s) : {results['time_s']:.2f}")
        logger.info(f"Num examples  : {results['num_examples']}")
        logger.info(f"ANLS          : {results['anls']:.4f}")
        logger.info(f"ACC           : {results['acc']:.4f}")
    elif args.eval_type in ["mask_overlapping", "mask_non_overlapping"]:
        # Masking experiments - run for each probability level
        mask_probabilities = [0.1, 0.5, 0.9, 1.0]
        logger.info(f"Running evaluation with masking type '{args.eval_type}' at probabilities: {mask_probabilities}")

        all_results = []
        for mask_prob in mask_probabilities:
            # Force garbage collection and clear GPU cache before each run
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Run evaluation for this masking probability
            logger.info(f"{'='*60}")
            logger.info(f"Running evaluation with mask_probability={mask_prob}")
            logger.info(f"{'='*60}")
            results = run_eval(
                model=model,
                val_loader=val_loader,
                num_examples=args.num_examples,
                plot_save_dir=example_plot_save_dir_path,
                masking_type=args.eval_type,
                top_k_percent=args.top_k_percent,
                mask_probability=mask_prob,
                mask_embeddings=args.mask_embeddings,
            )
            logger.info("Evaluation for this probability completed.")
            all_results.append((mask_prob, results))

            # Log summary for this probability
            logger.info(f"--- Results for mask_probability={mask_prob} ---")
            logger.info(f"Wall time (s): {results['time_s']:.2f}")
            logger.info(f"Num batches : {results['num_batches']}")
            logger.info(f"Num samples : {results['num_samples']}")

            loc = results["localization"]
            logger.info("[Localization]")
            logger.info(f"  IoU mean     : {loc['iou_mean']:.4f}")
            logger.info(f"  IoU median   : {loc['iou_median']:.4f}")
            logger.info(f"  GIoU mean    : {loc['giou_mean']:.4f}")
            logger.info(f"  Recall@0.5: {loc['recall@th']:.4f}")
            logger.info(f"  Center L2    : {loc['center_l2_mean']:.4f}")
            logger.info(f"  Coverage     : {loc['coverage_mean']:.4f}")
            logger.info(f"  Purity       : {loc['purity_mean']:.4f}")
            logger.info(f"  Area Ratio   : {loc['area_ratio_mean']:.4f}")

            doc = results["docvqa"]
            logger.info("[DocVQA]")
            logger.info(f"  ANLS : {doc['anls']:.4f}")
            logger.info(f"  ACC  : {doc['acc']:.4f}")

        # Print consolidated summary
        logger.info(f"\n{'='*60}")
        logger.info("=== CONSOLIDATED SUMMARY ===")
        logger.info(f"Masking type: {args.eval_type}")
        logger.info(f"Top-k percent: {args.top_k_percent}")
        logger.info(f"Mask embeddings: {args.mask_embeddings}")
        logger.info(f"{'='*60}")
        logger.info(f"{'Prob':<8} {'IoU':>8} {'GIoU':>8} {'Recall':>8} {'ANLS':>8} {'ACC':>8}")
        logger.info("-" * 60)
        for mask_prob, results in all_results:
            loc = results["localization"]
            doc = results["docvqa"]
            logger.info(
                f"{mask_prob:<8.2f} {loc['iou_mean']:>8.4f} {loc['giou_mean']:>8.4f} "
                f"{loc['recall@th']:>8.4f} {doc['anls']:>8.4f} {doc['acc']:>8.4f}"
            )
        logger.info("="*60)
    else:
        # Standard evaluation (eval_type == "standard")
        logger.info("Running standard evaluation...")
        results = run_eval(
            model=model,
            val_loader=val_loader,
            num_examples=args.num_examples,
            plot_save_dir=example_plot_save_dir_path,
        )
        logger.info("Evaluation completed.")

        # Log summary of time used
        logger.info("=== Evaluation Summary ===")
        logger.info(f"Wall time (s): {results['time_s']:.2f}")
        logger.info(f"Num batches : {results['num_batches']}")
        logger.info(f"Num samples : {results['num_samples']}")

        # Log the localization evaluation
        loc = results["localization"]
        logger.info("[Localization]")
        logger.info(f"  IoU mean     : {loc['iou_mean']:.4f}")
        logger.info(f"  IoU median   : {loc['iou_median']:.4f}")
        logger.info(f"  GIoU mean    : {loc['giou_mean']:.4f}")
        logger.info(f"  Recall@0.5: {loc['recall@th']:.4f}")
        logger.info(f"  Center L2    : {loc['center_l2_mean']:.4f}")
        logger.info(f"  Coverage     : {loc['coverage_mean']:.4f}")
        logger.info(f"  Purity       : {loc['purity_mean']:.4f}")
        logger.info(f"  Area Ratio   : {loc['area_ratio_mean']:.4f}")

        # Log the DocVQA evaluation
        doc = results["docvqa"]
        logger.info("[DocVQA]")
        logger.info(f"  ANLS : {doc['anls']:.4f}")
        logger.info(f"  ACC  : {doc['acc']:.4f}")