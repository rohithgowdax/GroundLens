import argparse
import os
import sys
import logging
from time import perf_counter

import torch

from dotenv import load_dotenv
load_dotenv()

from src.utils.seed import set_seed
from src.datasets.ops import normalize_answers
from src.datasets.collates import get_collate
from src.datasets.DocVqaAnswer import get_val_docvqa_answer_dataloader
from src.evaluation.answer_localization_evaluator import AnswerLocalizationEvaluator
from src.evaluation.docvqa_metrics import calculate_anls
from src.utils.gen_args import get_deterministic_gen_kwargs
from src.model.config import CoExVQAConfig
from src.model.model import CoExVQA
from src.model.losses.projection_loss import pred_norm_cxcywh_to_xyxy


PROJECT_AREA_PATH = "/cluster/work/projects/nn8104k/kjetil_thesis"
MODEL_SAVE_DIR_PATH = f"{PROJECT_AREA_PATH}/dual_heatmap_projector"
CACHE_DIR = f"{PROJECT_AREA_PATH}/.cache"
DS_NAME_DEFAULT = "indrehus/docvqa-single-page-questions-answer-ocr"


def _mean(xs):
    return float(torch.tensor(xs, dtype=torch.float32).mean().item()) if xs else float("nan")


def _group_metrics(indices, all_anls, all_iou, all_giou, all_recall, all_coverage, all_purity, all_ar):
    if not indices:
        return {"n": 0, "anls": float("nan"), "iou": float("nan"), "giou": float("nan"),
                "recall": float("nan"), "coverage": float("nan"), "purity": float("nan"), "ar": float("nan")}
    return {
        "n": len(indices),
        "anls": _mean([all_anls[i] for i in indices]),
        "iou": _mean([all_iou[i] for i in indices]),
        "giou": _mean([all_giou[i] for i in indices]),
        "recall": _mean([all_recall[i] for i in indices]),
        "coverage": _mean([all_coverage[i] for i in indices]),
        "purity": _mean([all_purity[i] for i in indices]),
        "ar": _mean([all_ar[i] for i in indices]),
    }


@torch.no_grad()
def run_correct_vs_incorrect_eval(
    model,
    val_loader,
    correct_thresh: float = 0.75,
    incorrect_thresh: float = 0.50,
):
    model.eval()
    loc_eval = AnswerLocalizationEvaluator(iou_thr=0.5, pred_fmt="xyxy", eps=1e-6)
    gen_kwargs = get_deterministic_gen_kwargs()

    # Per-sample accumulators
    all_anls = []
    all_iou, all_giou, all_recall = [], [], []
    all_coverage, all_purity, all_ar = [], [], []

    start = perf_counter()
    for batch in val_loader:
        images = batch["images"]
        questions = batch["questions"]
        _, all_answers = normalize_answers(batch["answers"])
        gt_boxes = batch["answer_bbox_xyxy"]
        full_img_sizes = batch["full_img_sizes"]

        # Forward pass for box prediction
        outputs = model(images=images, questions=questions, decoder_labels=None, gt_boxes=None)
        box_pred = outputs["box"]
        pred_xyxy_norm, _ = pred_norm_cxcywh_to_xyxy(box_pred, eps=1e-6)
        pred_xyxy_norm = pred_xyxy_norm.detach().cpu()

        # GT boxes
        if not torch.is_tensor(gt_boxes):
            gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
        else:
            gt_boxes = gt_boxes.float().cpu()
        sz = full_img_sizes.float().cpu()

        # Normalize GT to [0,1] xyxy (same logic as AnswerLocalizationEvaluator.compute)
        gt_max = float(gt_boxes.max().item()) if gt_boxes.numel() > 0 else 0.0
        if gt_max <= 1.0 + 1e-3:
            gt_xyxy_norm = gt_boxes.clamp(0.0, 1.0)
        else:
            W = sz[:, 0:1].clamp_min(1e-6)
            H = sz[:, 1:2].clamp_min(1e-6)
            gt_xyxy_norm = gt_boxes.clone()
            gt_xyxy_norm[:, [0, 2]] = (gt_xyxy_norm[:, [0, 2]] / W).clamp(0.0, 1.0)
            gt_xyxy_norm[:, [1, 3]] = (gt_xyxy_norm[:, [1, 3]] / H).clamp(0.0, 1.0)

        # Generate text answers
        preds = model.generate_preds(images=images, questions=questions, gen_kwargs=gen_kwargs, gt_boxes=None)

        B = len(questions)
        for i in range(B):
            p = pred_xyxy_norm[i].clamp(0.0, 1.0)
            g = gt_xyxy_norm[i]

            iou = loc_eval.calculate_iou(p, g)
            giou = loc_eval.calculate_giou(p, g)
            recall = loc_eval.calculate_recall(iou)
            coverage, purity, ar = loc_eval.calculate_region_stats(p, g)
            anls = calculate_anls(all_answers[i], preds[i])

            all_anls.append(anls)
            all_iou.append(iou)
            all_giou.append(giou)
            all_recall.append(recall)
            all_coverage.append(coverage)
            all_purity.append(purity)
            all_ar.append(ar)

        del outputs, box_pred, pred_xyxy_norm, gt_boxes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall = perf_counter() - start
    N = len(all_anls)

    correct_idx   = [i for i in range(N) if all_anls[i] >= correct_thresh]
    incorrect_idx = [i for i in range(N) if all_anls[i] < incorrect_thresh]
    neutral_idx   = [i for i in range(N) if incorrect_thresh <= all_anls[i] < correct_thresh]
    all_idx       = list(range(N))

    groups = {
        "Correct":   _group_metrics(correct_idx,   all_anls, all_iou, all_giou, all_recall, all_coverage, all_purity, all_ar),
        "Incorrect": _group_metrics(incorrect_idx, all_anls, all_iou, all_giou, all_recall, all_coverage, all_purity, all_ar),
        "Neutral":   _group_metrics(neutral_idx,   all_anls, all_iou, all_giou, all_recall, all_coverage, all_purity, all_ar),
        "Overall":   _group_metrics(all_idx,       all_anls, all_iou, all_giou, all_recall, all_coverage, all_purity, all_ar),
    }

    return {
        "time_s": wall,
        "num_samples": N,
        "correct_thresh": correct_thresh,
        "incorrect_thresh": incorrect_thresh,
        "groups": groups,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate localization metrics for correct vs. incorrect answers")
    parser.add_argument("--hf_repo_id", type=str, default="indrehus/CoExVQA", help="HuggingFace repo ID to load model from (default: indrehus/CoExVQA)")
    parser.add_argument("--model_checkpoint_dir", type=str, default=None, help="Local checkpoint dir; if set, overrides --hf_repo_id")
    parser.add_argument("--save_dir", type=str, default=MODEL_SAVE_DIR_PATH)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--dataset", type=str, default=DS_NAME_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--correct_thresh", type=float, default=0.75, help="ANLS threshold for 'correct' (inclusive, default 0.75)")
    parser.add_argument("--incorrect_thresh", type=float, default=0.50, help="ANLS threshold for 'incorrect' (exclusive, default 0.50)")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting correct-vs-incorrect evaluation")

    set_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    logger.info("Loading model...")
    if args.model_checkpoint_dir is not None:
        logger.info(f"Loading from local checkpoint: {args.model_checkpoint_dir}")
        model = CoExVQA.from_pretrained(path=args.model_checkpoint_dir, cache_dir=args.cache_dir).to(device)
    else:
        logger.info(f"Loading from HuggingFace: {args.hf_repo_id}")
        model = CoExVQA.from_hf(repo_id=args.hf_repo_id, cache_dir=args.cache_dir).to(device)
    logger.info("Model loaded.")

    collate_fn = get_collate("answer_resize_bbox", max_width=800)
    val_loader = get_val_docvqa_answer_dataloader(
        ds_name=args.dataset,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=min(8, os.cpu_count() // 2),
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    logger.info("Dataloader ready.")

    results = run_correct_vs_incorrect_eval(
        model=model,
        val_loader=val_loader,
        correct_thresh=args.correct_thresh,
        incorrect_thresh=args.incorrect_thresh,
    )

    logger.info(f"Wall time: {results['time_s']:.1f}s  |  N={results['num_samples']}")
    logger.info(f"Thresholds: correct >= {results['correct_thresh']}, incorrect < {results['incorrect_thresh']}")
    logger.info("")

    hdr = f"{'Group':<12} {'N':>6} {'ANLS':>7} {'IoU':>7} {'GIoU':>7} {'Coverage':>9} {'AR':>7} {'Recall@.5':>10}"
    sep = "-" * len(hdr)
    logger.info(hdr)
    logger.info(sep)
    for name in ["Correct", "Incorrect", "Neutral", "Overall"]:
        g = results["groups"][name]
        n = g["n"]
        if n == 0:
            logger.info(f"{name:<12} {'0':>6}  (no samples in this group)")
            continue
        logger.info(
            f"{name:<12} {n:>6} {g['anls']:>7.4f} {g['iou']:>7.4f} {g['giou']:>7.4f} "
            f"{g['coverage']:>9.4f} {g['ar']:>7.4f} {g['recall']:>10.4f}"
        )
    logger.info(sep)
