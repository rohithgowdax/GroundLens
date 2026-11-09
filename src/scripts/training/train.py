# Setup os environment variables transformers and tokenizers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"


# Import standard libraries
import argparse
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import sys
import logging
import uuid
from time import perf_counter
from dataclasses import asdict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Distributed data parallel utilities
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.utils.ddp_utils import setup_ddp, cleanup_ddp, is_main_process, reduce_metric

# Import implementations
from src.utils.seed import set_seed
from src.utils.wandb_logging import setup_wandb, log_training_step_metrics, log_ddp_training_config
from src.utils.hf_token import get_hf_token
from src.datasets.ops import normalize_answers
from src.datasets.collates import get_collate
from src.utils.augmentations import ALLOWED_AUGS
from src.evaluation.answer_localization_evaluator import AnswerLocalizationEvaluator
from src.evaluation.docvqa_metrics import DocVQAEvaluator
from src.utils.gen_args import get_deterministic_gen_kwargs
from src.utils.count_params import log_param_model
from src.utils.scheduler_utils import setup_scheduler
from src.utils.optimizer_utils import setup_optimizer
from src.datasets.DocVqaAnswer import get_train_val_docvqa_answer_dataloaders
from src.datasets.PFLDocVQA import get_pfl_docvqa_dataloader
from src.model.config import add_projector_arg_group, build_model_config
from src.model.model import CoExVQA
from src.model.projectors.answer_projector import AnswerHeatmapProjectorConfig
from src.model.projectors.question_projector import QuestionHeatmapProjectorConfig
from src.model.losses.projection_loss import pred_norm_cxcywh_to_xyxy
from src.model.losses.decoder import get_decoder_loss_weight
from src.model.losses.training_losses import compute_losses
from src.model.gating.config import GatingConfig
from src.utils.viz import visualize_masked_images
from src.utils.val_utils import save_q_mask_visualization
from src.utils.box_utils import prepare_gt_boxes


# Default paths
PROJECT_AREA_PATH = "/cluster/work/projects/nn8104k/kjetil_thesis"
MODEL_SAVE_DIR_PATH = f"{PROJECT_AREA_PATH}/dual_heatmap_projector"
PRETRAINED_PIX2STRUCT_MODEL_DIR_PATH= f"{PROJECT_AREA_PATH}/pix2struct-masked-train/answer-enc-dec-train-20250921-004330"
CACHE_DIR = f"{PROJECT_AREA_PATH}/.cache"
DS_NAME_DEFAULT = "indrehus/docvqa-single-page-questions-answer-ocr"
DS_NAME_Q_PRIOR = "indrehus/docvqa-single-page-questions-answer-ocr-colSmol500M-q-priors"
DS_NAME_DOCVQA_W_ANNOTATIONS = "indrehus/docvqa-single-page-w-prior-and-annotations"
DS_NAME_PFL_DOCVQA = "pfl_docvqa"
DS_NAME_PFL_DOCVQA_20PCT = "pfl_docvqa_20pct"
DS_NAME_PFL_DOCVQA_DEFAULT_DIR = f"{PROJECT_AREA_PATH}/datasets/PFL-DocVQA/"



@torch.no_grad()
def validate_epoch(model, val_loader, device, args, gen_kwargs, epoch, viz_save_dir, rank=0, world_size=1):
    # Unwrap DDP if necessary
    model_unwrapped = model.module if hasattr(model, 'module') else model

    # Setup decoding evaluator and accumulators
    docvqa_evaluator = DocVQAEvaluator()
    docvqa_eval_output_dict = {"anls": 0.0, "acc": 0.0}
    docvqa_elapsed = 0.0
    docvqa_per_sample = 0.0
    val_target_answers = []
    val_pred_answers = []
    val_img_sizes = []

    # Set up answer localization evaluator and accumulators
    localization_evaluator = AnswerLocalizationEvaluator(iou_thr=0.5, pred_fmt="xyxy")
    val_gt_boxes = []
    val_pred_boxes = []

    # Keep track of losses
    total_loss = 0.0
    total_dec_loss = 0.0
    total_proj_loss = 0.0
    total_q_prior_loss = 0.0
    total_ib_loss = 0.0
    total_iou = 0.0
    total_size = 0.0
    total_center = 0.0
    total_edges = 0.0
    total_aspect = 0.0
    total_area = 0.0

    # Run evaluation
    model.eval()
    start = perf_counter()
    num_examples = 0
    first_batch = True
    for batch in val_loader:
        images = batch["images"]
        questions = batch["questions"]
        B = len(questions)
        num_examples += B
        first_answers, all_answers = normalize_answers(batch["answers"])
        gt_boxes = batch["answer_bbox_xyxy"]
        full_img_sizes = batch["full_img_sizes"]

        # Prepare GT boxes for decoder if use_gt_boxes flag is set
        gt_boxes_for_decoder = None
        if args.use_gt_boxes and args.use_decoder:
            gt_boxes_for_decoder = prepare_gt_boxes(gt_boxes, full_img_sizes, device)

        # Single forward pass through model (backbone encoding happens inside)
        outputs = model(
            images=images,
            questions=questions,
            decoder_labels=first_answers if args.use_decoder else None,
            gt_boxes=gt_boxes_for_decoder,
            return_ib_intermediates=(args.lambda_ib > 0),
        )

        # Extract outputs from dict
        box_pred = outputs["box"]
        q_mask = outputs["q_mask"]
        decoder_output = outputs["decoder_output"]
        image_embeddings = outputs.get("image_embeddings")
        gated_embeddings = outputs.get("gated_embeddings")

        # Save q_mask visualization for first validation batch
        if first_batch and viz_save_dir is not None:
            save_q_mask_visualization(
                model_unwrapped=model_unwrapped,
                images=images,
                q_mask=q_mask,
                full_img_sizes=full_img_sizes,
                outputs=outputs,
                first_answers=first_answers,
                args=args,
                epoch=epoch,
                viz_save_dir=viz_save_dir,
                logger=logger,
                B=B,
            )
            first_batch = False

        # Append predicted boxes for localization evaluation
        pred_xyxy_norm, _ = pred_norm_cxcywh_to_xyxy(box_pred, eps=1e-6)

        # Get the GT
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

        loss_components = compute_losses(
            decoder_output=decoder_output,
            box_pred=box_pred,
            q_mask=q_mask,
            gt_boxes_xyxy_abs=gt_boxes,
            full_img_sizes=full_img_sizes,
            batch=batch,
            args=args,
            epoch=epoch,
            device=device,
            image_embeddings=image_embeddings,
            gated_embeddings=gated_embeddings,
        )
        loss = loss_components["total"]
        dec_loss = loss_components["decoder"]
        proj_loss = loss_components["projection"]
        q_prior_loss = loss_components["q_prior"]
        ib_loss_val = loss_components["ib"]
        loss_iou = loss_components["iou"]
        loss_size = loss_components["size"]
        loss_center = loss_components["center"]
        loss_edges = loss_components["edges"]
        loss_aspect = loss_components["aspect"]
        loss_area = loss_components["area"]

        # Generate predictions
        if args.use_decoder:
            # Use the new generate_preds method which handles the full pipeline
            # Pass gt_boxes if use_gt_boxes is enabled
            preds = model_unwrapped.generate_preds(
                images=images,
                questions=questions,
                gen_kwargs=gen_kwargs,
                gt_boxes=gt_boxes_for_decoder
            )
        else:
            preds = []

        # Accumulate predicted and gt answer
        val_target_answers.extend(all_answers) 
        val_pred_answers.extend(preds)

        # Add running losses
        total_loss += float(loss.item()) * B
        total_dec_loss += float(dec_loss.item()) * B
        total_proj_loss += float(proj_loss.item()) * B
        total_q_prior_loss += float(q_prior_loss.item()) * B
        total_ib_loss += float(ib_loss_val.item()) * B
        total_iou += float(loss_iou) * B
        total_center += float(loss_center) * B
        total_size += float(loss_size) * B
        total_aspect += float(loss_aspect) * B
        total_edges += float(loss_edges) * B
        total_area += float(loss_area) * B

        # Clear GPU tensors at end of validation batch to prevent accumulation
        del box_pred, q_mask, gt_boxes, images, outputs
        if args.use_decoder and decoder_output is not None:
            del decoder_output

    # Calculate time taken
    compute_time = perf_counter() - start
    if num_examples == 0:
        logger.warning("validate_epoch: no examples were processed (num_examples=0). Returning zero metrics.")
    denom = max(num_examples, 1)
    compute_per_sample = compute_time / denom

    # Compute averages
    avg_loss = total_loss / denom
    avg_dec = total_dec_loss / denom
    avg_proj = total_proj_loss / denom
    avg_q_prior = total_q_prior_loss / denom
    avg_ib = total_ib_loss / denom
    avg_iou = total_iou / denom
    avg_size = total_size / denom
    avg_center = total_center / denom
    avg_aspect = total_aspect / denom
    avg_edges = total_edges / denom
    avg_area = total_area / denom

    # Evaluate DocVQA metrics only decoding
    if args.use_decoder:
        # Debug logging for predictions (only when --debug flag is set)
        if args.debug:
            logger.info("=== DEBUG: Prediction Statistics ===")
            total_preds = len(val_pred_answers)
            empty_preds = sum(1 for p in val_pred_answers if p == "")
            non_empty_preds = total_preds - empty_preds
            logger.info(f"  Total predictions: {total_preds}")
            logger.info(f"  Empty predictions: {empty_preds} ({100*empty_preds/total_preds:.1f}%)")
            logger.info(f"  Non-empty predictions: {non_empty_preds} ({100*non_empty_preds/total_preds:.1f}%)")

            # Show first 20 predictions
            num_to_show = min(20, len(val_pred_answers))
            logger.info("=== DEBUG: First 20 predictions ===")
            for i in range(num_to_show):
                logger.info(f"  [{i}] Predicted: '{val_pred_answers[i]}'")
                logger.info(f"  [{i}] Target:    {val_target_answers[i]}")

            # Show some non-empty predictions if they exist
            if non_empty_preds > 0:
                logger.info("=== DEBUG: First 20 non-empty predictions ===")
                count = 0
                for i, pred in enumerate(val_pred_answers):
                    if pred != "" and count < 20:
                        logger.info(f"  [{i}] Predicted: '{pred}'")
                        logger.info(f"  [{i}] Target:    {val_target_answers[i]}")
                        count += 1

        start = perf_counter()
        docvqa_eval_output_dict = docvqa_evaluator.evaluate(target_answers=val_target_answers, pred_answers=val_pred_answers)
        docvqa_elapsed = perf_counter() - start
        docvqa_per_sample = docvqa_elapsed / max(1, len(val_target_answers))
    

    # Evaluate localization metrics
    start = perf_counter()
    answer_localization_eval_dict = localization_evaluator.compute(
        gt_boxes=val_gt_boxes,
        pred_boxes=val_pred_boxes,
        image_sizes=val_img_sizes,
    )
    localization_elapsed = perf_counter() - start
    localization_per_sample = localization_elapsed / max(1, len(val_gt_boxes))

    # Reduce validation metrics across all GPUs to get true global average
    if world_size > 1:
        avg_loss = reduce_metric(avg_loss, world_size, device)
        avg_dec = reduce_metric(avg_dec, world_size, device)
        avg_proj = reduce_metric(avg_proj, world_size, device)
        avg_q_prior = reduce_metric(avg_q_prior, world_size, device)
        avg_ib = reduce_metric(avg_ib, world_size, device)
        avg_iou = reduce_metric(avg_iou, world_size, device)
        avg_size = reduce_metric(avg_size, world_size, device)
        avg_center = reduce_metric(avg_center, world_size, device)
        avg_aspect = reduce_metric(avg_aspect, world_size, device)
        avg_edges = reduce_metric(avg_edges, world_size, device)
        avg_area = reduce_metric(avg_area, world_size, device)

    # Prepare loss dict
    loss_dict = {
        "loss_total": avg_loss,
        "loss_dec": avg_dec,
        "loss_proj": avg_proj,
        "loss_q_prior": avg_q_prior,
        "loss_ib": avg_ib,
        "loss_iou": avg_iou,
        "loss_size": avg_size,
        "loss_center": avg_center,
        "loss_aspect": avg_aspect,
        "loss_edges": avg_edges,
        "loss_area": avg_area
    }

    # Prepare time dict
    time_dict = {
        "compute_time_total": compute_time,
        "compute_time_per_sample": compute_per_sample,
        "docvqa_time_total": docvqa_elapsed,
        "docvqa_time_per_sample": docvqa_per_sample,
        "localization_time_total": localization_elapsed,
        "localization_time_per_sample": localization_per_sample,
    }
    return loss_dict, docvqa_eval_output_dict, answer_localization_eval_dict, time_dict

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="GroundLens", help="WanDB project name for grouping experiments")
    parser.add_argument("--experiment_name", type=str, default="RX", help="Idetification of the experiement")
    parser.add_argument("--ds_name", type=str, choices=[DS_NAME_DEFAULT, DS_NAME_Q_PRIOR, DS_NAME_DOCVQA_W_ANNOTATIONS, DS_NAME_PFL_DOCVQA, DS_NAME_PFL_DOCVQA_20PCT], default=DS_NAME_Q_PRIOR)
    parser.add_argument("--pfl_data_dir", type=str, default=DS_NAME_PFL_DOCVQA_DEFAULT_DIR, help="Path to PFL-DocVQA data directory (used when --ds_name pfl_docvqa)")
    parser.add_argument("--backbone_model", type=str, default="google/pix2struct-docvqa-base", help="Backbone model: HuggingFace ID (e.g. 'google/pix2struct-docvqa-large', " "'naver-clova-ix/donut-base-finetuned-docvqa') or local directory path.")
    parser.add_argument("--save_dir", type=str, default=MODEL_SAVE_DIR_PATH)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5, help="Learning rate for backbone (typically lower than main lr)")
    parser.add_argument("--batch_size", type=int, default=20)  
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "radam", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "onecycle", "reduceonplateau", "constant_with_warmup"])
    parser.add_argument("--augmentation", type=str, default="none", choices=ALLOWED_AUGS, help="Document augmentation that is applied to the document images.")
    parser.add_argument("--log_step", type=int, default=None, help="Log training metrics to WandB every log_step steps (default: 1000 for pfl_docvqa, 50 otherwise)")
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_patches", type=int, default=512)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iou_loss_type", choices=["ciou", "giou", "diou"], default="ciou")
    parser.add_argument("--lambda_iou", type=float, default=1.0)
    parser.add_argument("--lambda_size", type=float, default=0.0)
    parser.add_argument("--lambda_center", type=float, default=0.2)
    parser.add_argument("--lambda_aspect", type=float, default=0.0)
    parser.add_argument("--lambda_edges", type=float, default=0.0)
    parser.add_argument("--lambda_area",   type=float, default=0.05)
    parser.add_argument("--lambda_q_prior",   type=float, default=1.0)
    parser.add_argument("--lambda_dec", type=float, default=1.0, help="Weight for decoder loss")
    parser.add_argument("--lambda_ib", type=float, default=0.0, help="Weight for IB loss (0 = disabled)")
    parser.add_argument("--ib_beta1", type=float, default=1.0, help="IB beta1: compression weight for I(ImgEmb; T1)")
    parser.add_argument("--ib_beta2", type=float, default=1.0, help="IB beta2: compression weight for I(GatedImgEmb; T2)")
    parser.add_argument("--use_decoder", action="store_true", default=False, help="Enable decoder for text generation and answer loss")
    parser.add_argument("--resume", action="store_true", help="Resume training")

    # Config preset (if provided, overrides individual model args)
    parser.add_argument("--config", type=str, default=None, choices=["base", "medium", "large"],
                        help="Use a predefined config preset (overrides individual model args)")
    parser.add_argument("--decoder_attention_type", type=str, default="bbox_mask_reencode",
                        choices=["token_prune", "attention_mask", "bbox_mask_reencode", "bbox_crop_reencode"],
                        help="How to apply bbox attention to decoder")

    # Model args
    parser.add_argument("--d_model", type=int, default=768)

    # Namespaced projector args
    add_projector_arg_group(parser, "a", is_answer=True)
    add_projector_arg_group(parser, "q", is_answer=False)

    # Gating mechanism args
    parser.add_argument("--gating_type", type=str, default="linear_interpolation",
                        choices=["linear_interpolation", "residual", "spatial_attention", "film"],
                        help="Type of gating mechanism to use")
    parser.add_argument("--gating_initial_alpha", type=float, default=0.8,
                        help="Initial interpolation/modulation strength (for linear_interpolation and spatial_attention)")
    parser.add_argument("--gating_hidden_mult", type=float, default=1.0,
                        help="Hidden dimension multiplier for transformation networks (for residual and film)")
    parser.add_argument("--gating_dropout", type=float, default=0.1,
                        help="Dropout rate (for residual and film)")
    parser.add_argument("--gating_normalize", action="store_true", default=True,
                        help="Whether to normalize mask in spatial attention (for spatial_attention)")
    parser.add_argument("--gating_eps", type=float, default=1e-8,
                        help="Small constant for numerical stability (for spatial_attention)")

    # Decoder options
    parser.add_argument("--decoder_input_type", type=str, default="gated",
                        choices=["gated", "raw"],
                        help="Type of embeddings to feed decoder: 'gated' (recommended) or 'raw'")
    parser.add_argument("--decoder_loss_warmup_epochs", type=int, default=0,
                        help="Number of epochs to warm up decoder loss from 0 to 1 (0 = no warmup). "
                             "Early stopping is disabled during warmup period.")
    parser.add_argument("--use_gt_boxes", action="store_true",
                        help="Use ground truth boxes for decoder attention (decouples decoder from bbox prediction)")

    # Debug options
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (logs predictions and detailed statistics)")

    # Reencoding options
    parser.add_argument("--bbox_mask_blur_sigma", type=float, default=2.0,
                        help="Gaussian blur sigma for smooth mask edges (used with bbox_mask_reencode)")

    # Backbone freeze options
    parser.add_argument("--backbone_freeze_mode", type=str, default="unfrozen",
                        choices=["frozen", "unfrozen", "decoder-unfrozen"],
                        help="Freeze mode for backbone: 'frozen', 'unfrozen', or 'decoder-unfrozen'. "
                             "Default is 'unfrozen'. For bbox_mask_reencode and bbox_crop_reencode the "
                             "default is also 'unfrozen' unless explicitly overridden.")

    # Parse arguments
    args = parser.parse_args()
    if args.log_step is None:
        args.log_step = 1000 if args.ds_name in (DS_NAME_PFL_DOCVQA, DS_NAME_PFL_DOCVQA_20PCT) else 50

    # Setup Distributed Data Parallel
    rank, local_rank, world_size = setup_ddp()

    # Setting up logging:
    # Configure logging format with rank prefix for distributed training
    log_format = f"[Rank {rank}] %(asctime)s : %(levelname)s : %(message)s" if world_size > 1 else "%(asctime)s : %(levelname)s : %(message)s"
    logging.basicConfig(
        format=log_format,
        level=logging.INFO if is_main_process(rank) else logging.WARNING,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore").setLevel(logging.ERROR)

    if is_main_process(rank):
        log_ddp_training_config(logger, args, rank, local_rank, world_size)

    # Make save directory for the heatmap projector (all ranks need this)
    os.makedirs(args.save_dir, exist_ok=True)

    # Create visualization directory with unique suffix (all ranks need this)
    unique_id = str(uuid.uuid4())[:8]
    viz_dir = os.path.join(args.save_dir, f"q_mask_viz_{args.experiment_name}_{unique_id}")
    os.makedirs(viz_dir, exist_ok=True)

    # Create the model save name
    ckpt_dict = None
    model_save_name = f"dual_heatmap_{args.experiment_name}"

    if is_main_process(rank):
        logger.info(f"Save dir created: {args.save_dir}")
        logger.info(f"Q-mask visualization directory created: {viz_dir}")
        logger.info(f"Saving model checkpoint as: {model_save_name} in {args.save_dir}")

    # If resume, load the checkpoint
    if args.resume:
        # Get the checkpoint directory path
        logger.info("Resuming training from previous checkpoint")
        ckpt_dir = os.path.join(args.save_dir, model_save_name)
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
        logger.info(f"Loading checkpoint from: {ckpt_path}")

        # Load checkpoint
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        logger.info("Checkpoint loaded successfully")

    # Get the run id 
    try:
        wandb_run_id = ckpt_dict.get("wandb_run_id") if ckpt_dict is not None else None
        if args.resume and wandb_run_id is not None:
            logger.info(f"Will continue from WandB run id: {wandb_run_id}")
    except Exception as e:
        wandb_run_id = None
        if args.resume:
            logger.warning(f"WARNING: could not load run id from checkpoint: {e}. "
                        "Proceeding with a NEW W&B run (logs won’t merge).")

    # Set the seed 
    set_seed(args.seed)
    logger.info(f"Seed set to: {args.seed}")

    # Log the arguments
    logger.info("Arguments:")
    if args.config is not None:
        # When using config preset, skip detailed model architecture args
        logger.info(f"    Using config preset: {args.config}")
        logger.info("    (skipping detailed model architecture args)")
        for k in sorted(vars(args)):
            if not (k.startswith('a_') or k.startswith('q_') or k.startswith('gating_') or k == 'd_model'):
                v = getattr(args, k)
                logger.info("    %s: %s", k, str(v))
    else:
        for k in sorted(vars(args)):
            v = getattr(args, k)
            logger.info("    %s: %s", k, str(v))

    # Setting up the device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Get HF Token
    hf_token = get_hf_token()
    logger.info("HuggingFace token retrieved successfully.")

    # Setup Weights & Biases (only on main process)
    run = None
    if is_main_process(rank):
        logger.info("Setting up WandB")

        # Filter out model config args if using a preset
        if args.config is not None:
            # Exclude answer and question projector args, gating args when using preset
            config_dict = {
                k: v for k, v in args.__dict__.items()
                if not (k.startswith('a_') or k.startswith('q_') or
                       k.startswith('gating_') or k == 'd_model')
            }
        else:
            config_dict = args.__dict__

        run = setup_wandb(
            config_dict=config_dict,
            project_name=args.project_name,
            prefix=f"dual-heatmap-train-loss-{args.experiment_name}",
            run_id=wandb_run_id
        )
        logger.info(f"WandB run initialized: name={run.name} id={run.id}")

    # Initialize the CoExVQA model config
    d_model_override = None
    if args.config is not None and not os.path.isdir(args.backbone_model):
        d_model_override = CoExVQA.d_model_for_backbone(args.backbone_model)
        logger.info(f"Backbone: {args.backbone_model}, d_model set to {d_model_override}")
    model_config = build_model_config(args, d_model=d_model_override)

    # Log backbone source
    if model_config.backbone_local_path is None:
        logger.info(f"Using backbone from Hugging Face: {model_config.backbone_model}")
    else:
        logger.info(f"Using backbone from local directory: {model_config.backbone_local_path}")

    logger.info(f"Backbone freeze mode: {model_config.backbone_freeze_mode}")

    # Log decoder configuration
    logger.info(f"Decoder enabled: {model_config.use_decoder}")
    if model_config.use_decoder:
        logger.info(f"  Decoder input type: {model_config.decoder_input_type}")
        logger.info(f"  Decoder attention type: {model_config.decoder_attention_type}")
        logger.info(f"  Decoder loss warmup epochs: {args.decoder_loss_warmup_epochs}")
        if args.decoder_loss_warmup_epochs > 0:
            logger.info(f"  NOTE: Decoder loss will ramp from 0 to 1 over {args.decoder_loss_warmup_epochs} epochs")
            logger.info(f"  NOTE: Early stopping is DISABLED during warmup period")
        logger.info(f"  Decoder loss lambda: {args.lambda_dec}")

    # Log AnswerHeatMapProjector
    a_config: AnswerHeatmapProjectorConfig = model_config.answer_projector_config
    q_config: QuestionHeatmapProjectorConfig = model_config.question_projector_config

    logger.info("AnswerHeatmapProjector config:")
    for k, v in asdict(a_config).items():
        logger.info(f"  {k}: {v}")
    logger.info("QuestionHeatmapProjector config:")
    for k, v in asdict(q_config).items():
        logger.info(f"  {k}: {v}")

    # Log gating config
    logger.info("GatingConfig:")
    gate_config: GatingConfig = model_config.gate_config
    for k, v in asdict(gate_config).items():
        logger.info(f"  {k}: {v}")

    # Create model and load if possible
    model = CoExVQA(config=model_config, cache_dir=args.cache_dir)
    logger.info("CoExVQA model created successfully")
    if args.resume and ckpt_dict is not None:
        missing, unexpected = model.load_state_dict(ckpt_dict["model_state_dict"], strict=False)
        if missing or unexpected:
            logger.warning(f"While loading model: missing={missing}, unexpected={unexpected}")

        logger.info("Loaded pretrained CoExVQA model")
    model.to(device)

    # Wrap model with DDP if distributed training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        logger.info(f"Model wrapped with DistributedDataParallel on device {local_rank}")
        # Keep reference to unwrapped model for accessing attributes
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Log number of parameters
    log_param_model(logger, model_without_ddp)

    # Setup optimizer
    optimizer = setup_optimizer(
        model=model_without_ddp,
        optimizer_name=args.optimizer,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        logger=logger
    )

    # Get the pre processing collate function
    collate_name = "pfl_docvqa_with_q_prior" if args.ds_name in (DS_NAME_PFL_DOCVQA, DS_NAME_PFL_DOCVQA_20PCT) else "answer_bbox_with_q_prior"
    collate_fn = get_collate(
        collate_name,
        max_width=800,
        augmentation=args.augmentation
    )
    logger.info(f"Collate fn '{collate_name}' is set to resize the document with max width 800, and augmentation {args.augmentation}")

    # Get dataset loaders
    logger.info(f"Creating train and validation dataloaders from ds: {args.ds_name}")
    num_workers = min(8, os.cpu_count() // 2)
    if args.ds_name in (DS_NAME_PFL_DOCVQA, DS_NAME_PFL_DOCVQA_20PCT):
        subsample_frac = 0.20 if args.ds_name == DS_NAME_PFL_DOCVQA_20PCT else None
        train_dataloader = get_pfl_docvqa_dataloader(
            data_dir=args.pfl_data_dir,
            split="train",
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            subsample_frac=subsample_frac,
        )
        val_dataloader = get_pfl_docvqa_dataloader(
            data_dir=args.pfl_data_dir,
            split="val",
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    else:
        train_dataloader, val_dataloader = get_train_val_docvqa_answer_dataloaders(
            ds_name=args.ds_name,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(2 if num_workers > 0 else None),
            rank=rank,
            world_size=world_size,
        )
    logger.info("Train and validation dataloaders created successfully.")
    if world_size > 1:
        logger.info(f"DistributedSampler enabled for both train and validation dataloaders")


    # Setup learning rate scheduler
    scheduler, scheduler_step_on = setup_scheduler(
        scheduler_name=args.scheduler,
        optimizer=optimizer,
        epochs=args.epochs,
        steps_per_epoch=len(train_dataloader),
        logger=logger
    )

    # Load from statedict for scheduler and optimizer if possible 
    if args.resume:
        optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
        try:
            scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"Scheduler state could not be loaded; will continue from fresh scheduler. Reason: {e}")
        
        # Place to current device
        try:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        except Exception as e:
            logger.warning(f"Error while moving optimizer values to device: {e}")

    # Start training
    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = args.early_stop_patience
    last_epoch_idx = 0

    # Fix best val loss and epoch based on checkpoint: 
    if args.resume:
        best_val_loss = ckpt_dict.get("best_val_loss", best_val_loss)
        last_epoch_idx = ckpt_dict.get("epoch", 0)
        logger.info(f"Resumed at epoch {last_epoch_idx}; best_val_loss={best_val_loss:.6f}")

    # Load generation kwargs
    gen_kwargs = get_deterministic_gen_kwargs()
    logger.info("Loaded deterministic generation args")

    # Check what dtype is available
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_fp16_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(enabled=use_fp16_scaler)

    # Training loop
    logger.info("Training started")
    start_training = perf_counter()
    start_epoch = last_epoch_idx + 1
    end_epoch = last_epoch_idx + args.epochs
    global_step = 0
    num_examples = len(train_dataloader.dataset)
    for epoch in range(start_epoch, end_epoch + 1):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if world_size > 1 and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        running_total = 0.0
        running_dec = 0.0
        running_box = 0.0
        running_q_prior = 0.0
        running_ib = 0.0
        running_iou = 0.0
        running_size = 0.0
        running_center = 0.0
        running_edges = 0.0
        running_aspect = 0.0
        running_area = 0.0
        num_seen = 0
        first_batch = True

        logger.info(f"[Epoch {epoch}] Starting training loop (rank={rank})")
        start = perf_counter()
        for step, batch in enumerate(train_dataloader, start=1):
            # Get the output
            images = batch["images"]
            questions = batch["questions"]
            first_answers, all_answers = normalize_answers(batch["answers"])
            answer_bbox_xyxy = batch["answer_bbox_xyxy"]
            full_img_sizes = batch["full_img_sizes"]
            B = len(questions)

            # Reset gradients 
            optimizer.zero_grad(set_to_none=True)
                
            # Forward method
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):

                # Prepare GT boxes for decoder if use_gt_boxes flag is set
                gt_boxes_for_decoder = None
                if args.use_gt_boxes and args.use_decoder:
                    gt_boxes_for_decoder = prepare_gt_boxes(answer_bbox_xyxy, full_img_sizes, device)

                # Single forward pass through model (backbone encoding happens inside)
                outputs = model(
                    images=images,
                    questions=questions,
                    decoder_labels=first_answers if args.use_decoder else None,
                    gt_boxes=gt_boxes_for_decoder,
                    return_ib_intermediates=(args.lambda_ib > 0),
                )

                # Extract outputs from dict
                box_pred = outputs["box"]
                q_mask = outputs["q_mask"]
                decoder_output = outputs["decoder_output"]
                image_embeddings = outputs.get("image_embeddings")   # None when lambda_ib=0
                gated_embeddings = outputs.get("gated_embeddings")   # None when lambda_ib=0

                # Visualize masked/cropped images for first batch if debug mode
                if first_batch and args.debug and model_without_ddp.config.decoder_attention_type in ["bbox_mask_reencode", "bbox_crop_reencode"]:
                    masked_images_pil = outputs.get("masked_images_pil")
                    if masked_images_pil is not None:
                        subdir_name = "masked_images_train" if model_without_ddp.config.decoder_attention_type == "bbox_mask_reencode" else "cropped_images_train"
                        visualize_masked_images(
                            masked_images=masked_images_pil,
                            target_answers=first_answers,
                            save_dir=os.path.join(viz_dir, subdir_name),
                            epoch=epoch,
                            num_samples=min(8, B)
                        )
                    first_batch = False

                # Free PIL images immediately — no longer needed after visualization
                outputs.pop("masked_images_pil", None)

                loss_components = compute_losses(
                    decoder_output=decoder_output,
                    box_pred=box_pred,
                    q_mask=q_mask,
                    gt_boxes_xyxy_abs=answer_bbox_xyxy,
                    full_img_sizes=full_img_sizes,
                    batch=batch,
                    args=args,
                    epoch=epoch,
                    device=device,
                    image_embeddings=image_embeddings,
                    gated_embeddings=gated_embeddings,
                )
                total_loss = loss_components["total"]
                decoder_loss = loss_components["decoder"]
                proj_loss = loss_components["projection"]
                q_prior_loss = loss_components["q_prior"]
                ib_loss = loss_components["ib"]
                loss_iou = loss_components["iou"]
                loss_size = loss_components["size"]
                loss_center = loss_components["center"]
                loss_edges = loss_components["edges"]
                loss_aspect = loss_components["aspect"]
                loss_area = loss_components["area"]

            # Backward
            if use_fp16_scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                trainable_params = (p for p in model.parameters() if p.requires_grad)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                trainable_params = (p for p in model.parameters() if p.requires_grad)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
            
            if scheduler_step_on == "batch":
                scheduler.step()

            # Increment global step
            global_step += 1

            # Log gradients and activations periodically
            if step % args.log_step == 0 and is_main_process(rank):
                metrics = log_training_step_metrics(
                    model=model,
                    outputs=outputs,
                    box_pred=box_pred,
                    q_mask=q_mask,
                    decoder_output=decoder_output,
                    global_step=global_step,
                    epoch=epoch
                )
                if run is not None:
                    run.log(metrics)

            # Free outputs (holds encoder_outputs and other references)
            del outputs

            # Add the running losses
            running_total  += float(total_loss.item()) * B
            running_dec    += float(decoder_loss.item()) * B
            running_box    += float(proj_loss.item()) * B
            running_q_prior += float(q_prior_loss.item()) * B
            running_ib     += float(ib_loss.item()) * B
            running_iou    += float(loss_iou) * B
            running_size   += float(loss_size) * B
            running_center += float(loss_center) * B
            running_edges  += float(loss_edges) * B
            running_aspect += float(loss_aspect) * B
            running_area += float(loss_area) * B
            num_seen += B

            # Clear CUDA cache every step to prevent fragmentation buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if step % 100 == 0:
                gc.collect()

            # Log step using the logging interval
            if step % args.log_step == 0 or step == len(train_dataloader):
                avg_t = running_total / num_seen
                avg_dec = running_dec / num_seen
                avg_projection = running_box / num_seen
                logger.info(f"      [Epoch {epoch}, Step {step}/{len(train_dataloader)}, Rank {rank}] avg_total={avg_t:.4f}, avg_dec={avg_dec:.4f}, avg_projection={avg_projection:.4f}")

        # Synchronize all processes after training loop completes
        logger.info(f"[Epoch {epoch}] Reached end of training loop (rank={rank})")
        if world_size > 1:
            logger.info(f"[Epoch {epoch}] Waiting at barrier before epoch summary (rank={rank})")
            dist.barrier()
            logger.info(f"[Epoch {epoch}] Passed barrier (rank={rank})")

        # Calculate epoch time
        epoch_time = perf_counter() - start

        # Log epoch training loss (compute per-GPU averages first)
        # Use num_seen (samples actually processed by this rank) not num_examples (global dataset size)
        avg_total = running_total / num_seen
        avg_dec = running_dec / num_seen
        avg_projection = running_box / num_seen
        avg_q_prior = running_q_prior / num_seen
        avg_ib = running_ib / num_seen
        avg_iou = running_iou / num_seen
        avg_size = running_size / num_seen
        avg_center = running_center / num_seen
        avg_edges = running_edges / num_seen
        avg_aspect = running_aspect / num_seen
        avg_area = running_area / num_seen

        # Reduce metrics across all GPUs to get true global average
        if world_size > 1:
            avg_total = reduce_metric(avg_total, world_size, device)
            avg_dec = reduce_metric(avg_dec, world_size, device)
            avg_projection = reduce_metric(avg_projection, world_size, device)
            avg_q_prior = reduce_metric(avg_q_prior, world_size, device)
            avg_ib = reduce_metric(avg_ib, world_size, device)
            avg_iou = reduce_metric(avg_iou, world_size, device)
            avg_size = reduce_metric(avg_size, world_size, device)
            avg_center = reduce_metric(avg_center, world_size, device)
            avg_aspect = reduce_metric(avg_aspect, world_size, device)
            avg_edges = reduce_metric(avg_edges, world_size, device)
            avg_area = reduce_metric(avg_area, world_size, device)

        # Log the epoch loss
        logger.info(f"[Epoch {epoch}] Train avg_total={avg_total:.4f} | dec={avg_dec:.4f} | projection={avg_projection:.4f} | q_prior={avg_q_prior:.4f} | ib={avg_ib:.4f} | iou={avg_iou:.4f} | size={avg_size:.4f} | center={avg_center:.4f} | edges={avg_edges:.4f} | aspect={avg_aspect:.4f} | area={avg_area:.4f}")

        # Log training time 
        logger.info(f"[Epoch {epoch}] Training Epoch Time:")
        logger.info(f"   - Epoch time: {epoch_time:.2f} seconds")
        logger.info(f"   - Time per batch: {epoch_time / len(train_dataloader):.4f} seconds")
        logger.info(f"   - Time per sample: {epoch_time / num_examples:.4f} seconds")

        # Log to WanDB (only on main process)
        if is_main_process(rank) and run is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            decoder_loss_weight = get_decoder_loss_weight(
                current_epoch=epoch - 1,  # 0-indexed
                warmup_epochs=args.decoder_loss_warmup_epochs
            )
            run.log({
                "train/loss_total": avg_total,
                "train/loss_dec": avg_dec,
                "train/loss_projection": avg_projection,
                "train/loss_q_prior": avg_q_prior,
                "train/loss_ib": avg_ib,
                "train/loss_size": avg_size,
                "train/loss_iou": avg_iou,
                "train/loss_center": avg_center,
                "train/loss_edges": avg_edges,
                "train/loss_aspect": avg_aspect,
                "train/loss_area": avg_area,
                "train/decoder_loss_weight": decoder_loss_weight,
                "epoch": epoch,
                "lr": current_lr,
            }, commit=True)

        # Run evaluation
        logger.info("Running validation...")
        model.eval()

        with torch.no_grad():
            val_loss_dict, docvqa_eval_output_dict, answer_localization_eval_dict, time_dict = validate_epoch(
                model, val_dataloader,
                device=device, args=args, gen_kwargs=gen_kwargs, epoch=epoch, viz_save_dir=viz_dir,
                rank=rank, world_size=world_size
            )

        # Clear CUDA cache to free up memory after validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Synchronize all processes after validation completes
        logger.info(f"[Epoch {epoch}] Validation complete (rank={rank})")
        if world_size > 1:
            logger.info(f"[Epoch {epoch}] Waiting at barrier after validation (rank={rank})")
            dist.barrier()
            logger.info(f"[Epoch {epoch}] Passed validation barrier (rank={rank})")

        # Extract validaiton loss from output dict
        val_loss = val_loss_dict.get("loss_total")
        val_dec = val_loss_dict.get("loss_dec")
        val_proj = val_loss_dict.get("loss_proj")
        val_q_prior = val_loss_dict.get("loss_q_prior")
        val_ib = val_loss_dict.get("loss_ib")
        val_iou = val_loss_dict.get("loss_iou")
        val_size = val_loss_dict.get("loss_size")
        val_center = val_loss_dict.get("loss_center")
        val_aspect = val_loss_dict.get("loss_aspect")
        val_edges = val_loss_dict.get("loss_edges")
        val_area = val_loss_dict.get("loss_area")

        # In case of stepping on the validaiton loss
        if scheduler_step_on == "epoch":
            scheduler.step(val_loss)

        # Extract DocVQA eval metrics
        val_anls = docvqa_eval_output_dict.get("anls", 0.0)
        val_acc = docvqa_eval_output_dict.get("acc", 0.0)

        # Extract localization eval metrics    
        val_iou_mean = answer_localization_eval_dict.get("iou_mean")
        val_iou_median = answer_localization_eval_dict.get("iou_median")
        val_giou_mean = answer_localization_eval_dict.get("giou_mean")
        val_recall = answer_localization_eval_dict.get("recall@th")
        val_center_l2 = answer_localization_eval_dict.get("center_l2_mean")
        val_coverage = answer_localization_eval_dict.get("coverage_mean")
        val_purity = answer_localization_eval_dict.get("purity_mean")
        val_area_ratio = answer_localization_eval_dict.get("area_ratio_mean")

        # Log the epoch validation metrics and losses
        logger.info(f"[Epoch {epoch}] Validation Metrics and Losses:")
        logger.info(f"    Total Loss = {val_loss:.4f}, dec={val_dec:.4f} | proj= {val_proj:.4f} | q_prior={val_q_prior:.4f} | ib={val_ib:.4f} | iou={val_iou:.4f} | size={val_size:.4f} | center={val_center:.4f} | aspect={val_aspect:.4f} | area={val_area:.4f}")
        logger.info(f"    ANLS = {val_anls:.4f} | ACC = {val_acc:.4f}")
        logger.info(f"    IoU mean = {val_iou_mean:.4f} | IoU median = {val_iou_median:.4f} | GIoU mean = {val_giou_mean:.4f}")
        logger.info(f"    Recall@0.5 = {val_recall:.4f} | Center L2 mean = {val_center_l2:.4f}")
        logger.info(f"    Coverage = {val_coverage:.4f} | Purity = {val_purity:.4f} | Area Ratio = {val_area_ratio:.4f}")

        # Log how long validation took
        logger.info(f"[Epoch {epoch}] Validation Time:")
        logger.info(f"   - Inference time:          Total = {time_dict['compute_time_total']:.4f} s, per sample = {time_dict['compute_time_per_sample']:.4f}s/sample")
        logger.info(f"   - Localization eval time:  Total = {time_dict['localization_time_total']:.4f} s, per sample = {time_dict['localization_time_per_sample']:.4f}s/sample")
        if args.use_decoder:
            logger.info(f"   - DocVQA eval time:        Total = {time_dict['docvqa_time_total']:.4f} s, per sample = {time_dict['docvqa_time_per_sample']:.4f}s/sample")

        # Log to WandB (only on main process)
        if is_main_process(rank) and run is not None:
            log_dict = {
                "val/loss_total": val_loss,
                "val/loss_dec": val_dec,
                "val/loss_projection": val_proj,
                "val/loss_q_prior": val_q_prior,
                "val/loss_ib": val_ib,
                "val/loss_iou": val_iou,
                "val/loss_size": val_size,
                "val/loss_center": val_center,
                "val/loss_aspect": val_aspect,
                "val/loss_edges": val_edges,
                "val/loss_area": val_area,
                "eval/anls": val_anls,
                "eval/acc": val_acc,
                "eval/iou_mean": val_iou_mean,
                "eval/iou_median": val_iou_median,
                "eval/recall@0.5": val_recall,
                "eval/center_l2_mean": val_center_l2,
                "eval/coverage": val_coverage,
                "eval/purity": val_purity,
                "eval/area_ratio": val_area_ratio,
                "epoch": epoch,
            }
            run.log(log_dict, commit=True)

        # Check if we are still in decoder warmup period
        in_warmup = (epoch - 1) < args.decoder_loss_warmup_epochs

        # During warmup, skip early stopping logic entirely
        if in_warmup:
            logger.info(f"Warmup epoch {epoch}/{args.decoder_loss_warmup_epochs} - early stopping disabled")
        else:
            # After warmup: use total loss for early stopping
            if val_loss < best_val_loss:
                # New best loss
                best_val_loss = val_loss
                epochs_no_improve = 0
                logger.info(f"New best val loss: {best_val_loss:.6f}")

                # Only main process saves checkpoint in distributed training
                if is_main_process(rank):
                    ckpt_path = os.path.join(args.save_dir, model_save_name)
                    model_without_ddp.save_checkpoint(
                        path=ckpt_path,
                        epoch=epoch,
                        args=args,
                        extra_state={
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "best_val_loss": best_val_loss,
                            "wandb_run_id": getattr(run, "id", None),
                        }
                    )
                    logger.info(f"Saved best model checkpoint: {ckpt_path}")

                # Synchronize all processes after checkpoint saving
                if world_size > 1:
                    dist.barrier()
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement this epoch: epochs_no_improve = {epochs_no_improve}/{patience}")

        # Early stopping check (only after warmup)
        if not in_warmup and epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best val loss = {best_val_loss:.4f}")
            break

        # Reset best_val_loss when warmup ends so early stopping starts fresh
        if in_warmup and epoch == args.decoder_loss_warmup_epochs:
            best_val_loss = float("inf")
            epochs_no_improve = 0
            logger.info(f"Decoder warmup complete. Resetting best_val_loss. Early stopping now active with patience={patience}.")

    # Training complete
    training_time = perf_counter() - start_training
    logger.info("Training complete.")
    logger.info(f"Total training time with validation: {training_time:.2f} seconds")
    logger.info(f"Average training time per epoch: {training_time / args.epochs:.2f} seconds")

    # Finish WandB run
    if is_main_process(rank) and run is not None:
        run.finish()
        logger.info("WandB run finished")

    # Cleanup DDP
    cleanup_ddp()
    if world_size > 1:
        logger.info("DDP process group destroyed")