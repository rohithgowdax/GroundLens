import os
import json
import logging
from time import perf_counter
from typing import Dict, List, Any, Tuple
import torch
import numpy as np

# Import implementations
from src.model.model import CoExVQA
from src.datasets.ops import normalize_answers
from src.evaluation.docvqa_metrics import calculate_anls
from src.utils.gen_args import get_deterministic_gen_kwargs
from src.utils.viz.rate_anls_viz import plot_rate_anls, plot_joint_rate_heatmap

# Set up logger
logger = logging.getLogger(__name__)


def bernoulli_kl(p: torch.Tensor, q: float, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute KL divergence KL(Bern(p) || Bern(q)) with numerical stability.

    Formula: KL = p * log(p/q) + (1-p) * log((1-p)/(1-q))

    Args:
        p: Tensor of probabilities in [0, 1], any shape
        q: Scalar baseline probability in [0, 1]
        eps: Small constant for numerical stability (prevents log(0))

    Returns:
        KL divergence tensor with same shape as p
    """
    p_safe = torch.clamp(p, min=eps, max=1.0 - eps)
    q_safe = max(eps, min(1.0 - eps, q))

    term1 = p_safe * (torch.log(p_safe) - np.log(q_safe))
    term2 = (1.0 - p_safe) * (torch.log(1.0 - p_safe) - np.log(1.0 - q_safe))

    return term1 + term2


def compute_rate(q_mask: torch.Tensor, pi: float, normalize: bool = True) -> torch.Tensor:
    """
    Compute per-sample Rate = sum_i KL(m_i || pi).

    Rate measures how much the mask deviates from the baseline prior pi.
    Normalizing by N (number of patches) makes rates comparable across
    different sequence lengths.

    Args:
        q_mask: Mask tensor of shape (B, seq_len, 1) or (B, seq_len), values in [0, 1]
        pi: Baseline prior probability in [0, 1]
        normalize: If True, divide by N (number of patches)

    Returns:
        rate: Tensor of shape (B,) with rate value for each sample
    """
    # Handle (B, seq_len, 1) -> (B, seq_len)
    if q_mask.dim() == 3 and q_mask.shape[-1] == 1:
        q_mask = q_mask.squeeze(-1)

    B, N = q_mask.shape

    # Compute per-patch KL divergence from prior
    kl_per_patch = bernoulli_kl(q_mask, pi)  # (B, N)

    # Sum over patches
    rate = kl_per_patch.sum(dim=1)  # (B,)

    if normalize:
        rate = rate / N

    return rate


@torch.no_grad()
def estimate_pi(model, dataloader, max_batches: int = 50) -> float:
    """
    Estimate baseline prior pi as dataset mean of q_mask activation.

    Args:
        model: The CoExVQA model
        dataloader: Validation dataloader
        max_batches: Maximum batches to use for estimation

    Returns:
        pi: Estimated prior probability (scalar in [0, 1])
    """
    model.eval()
    total_sum, total_count = 0.0, 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        outputs = model(images=batch["images"], questions=batch["questions"])
        q_mask = outputs.get("q_mask", None)

        if q_mask is not None:
            total_sum += q_mask.sum().item()
            total_count += q_mask.numel()

        # Clear GPU memory
        del outputs, q_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_sum / total_count if total_count > 0 else 0.5


@torch.no_grad()
def run_rate_anls_eval(
    model,
    val_loader,
    pi: float,
    output_dir: str,
    n_bins: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate Rate vs ANLS trade-off.

    Rate = sum_i KL(m_i || pi) / N measures how much the q_mask deviates
    from baseline pi. Low rate = sparse/uniform mask, high rate = selective mask.

    Args:
        model: The CoExVQA model
        val_loader: Validation dataloader
        pi: Baseline prior probability
        output_dir: Directory to save outputs
        n_bins: Number of bins for binned ANLS curve

    Returns:
        Dict with records, timing, and summary statistics
    """
    # Set model to eval mode
    model.eval()
    gen_kwargs = get_deterministic_gen_kwargs()

    # Prepare output directory
    records: List[Dict[str, Any]] = []
    sample_idx = 0

    
    # Start evaluation loop
    start = perf_counter()
    for batch_idx, batch in enumerate(val_loader):
        images = batch["images"]
        questions = batch["questions"]
        _, all_answers = normalize_answers(batch["answers"])
        B = len(questions)

        # Forward pass
        outputs = model(images=images, questions=questions)
        q_mask = outputs.get("q_mask", None)

        # Compute rate for batch if q_mask available
        if q_mask is not None:
            rates = compute_rate(q_mask, pi=pi, normalize=True)  # (B,)
        else:
            rates = torch.zeros(B)

        # Generate predictions
        preds = model.generate_preds(images, questions, gen_kwargs)

        # Store per-sample records
        for i in range(B):
            target_answers = all_answers[i]
            pred_answer = preds[i]
            anls = calculate_anls(target_answers, pred_answer)
            rate_norm = rates[i].item()

            records.append({
                "sample_idx": sample_idx,
                "question": questions[i],
                "anls": anls,
                "rate_norm": rate_norm,
                "pred_answer": pred_answer,
                "target_answers": target_answers,
            })
            sample_idx += 1

        # Clear GPU memory
        del outputs, q_mask, images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if batch_idx % 10 == 0:
            logger.info(f"Rate-ANLS eval: batch {batch_idx}/{len(val_loader)}")

    # Compute total time
    wall = perf_counter() - start
    logger.info(f"Rate-ANLS evaluation completed in {wall:.2f}s for {sample_idx} samples")

    # Generate plots
    rates_list = [r["rate_norm"] for r in records]
    anls_list = [r["anls"] for r in records]

    # Plot Rate vs ANLS (scatter and binned)
    plot_rate_anls(rates_list, anls_list, output_dir, n_bins=n_bins)
    logger.info(f"Saved Rate vs ANLS plots to {output_dir}")

    # Save metrics to JSONL
    metrics_path = os.path.join(output_dir, "rate_anls_metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    logger.info(f"Saved {len(records)} records to {metrics_path}")

    return {
        "records": records,
        "time_s": wall,
        "num_samples": len(records),
        "pi": pi,
        "mean_rate": float(np.mean(rates_list)),
        "mean_anls": float(np.mean(anls_list)),
    }


def bbox_to_patch_mask(
    bbox: torch.Tensor,
    seq_len: int,
    patch_grid: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert bounding box to binary patch-level selection mask.

    A patch is selected (s_i = 1) if its center falls inside the bbox.

    Args:
        bbox: Normalized cxcywh format (B, 4)
        seq_len: Number of patches
        patch_grid: Grid dimensions (H_p, W_p)

    Returns:
        mask: (B, seq_len) with values in {0, 1}
    """
    B = bbox.shape[0]
    H_p, W_p = patch_grid
    device = bbox.device

    # Create patch center coordinates (normalized [0, 1])
    y_coords = torch.linspace(0.5 / H_p, 1 - 0.5 / H_p, H_p, device=device)
    x_coords = torch.linspace(0.5 / W_p, 1 - 0.5 / W_p, W_p, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    patch_x = grid_x.flatten()[:seq_len]  # (seq_len,)
    patch_y = grid_y.flatten()[:seq_len]  # (seq_len,)

    # Extract bbox components
    cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]  # each (B,)

    # Compute bbox boundaries
    x_min = cx - w / 2  # (B,)
    x_max = cx + w / 2
    y_min = cy - h / 2
    y_max = cy + h / 2

    # Check if patch centers fall inside bbox
    inside_x = (patch_x.unsqueeze(0) >= x_min.unsqueeze(1)) & \
               (patch_x.unsqueeze(0) <= x_max.unsqueeze(1))  # (B, seq_len)
    inside_y = (patch_y.unsqueeze(0) >= y_min.unsqueeze(1)) & \
               (patch_y.unsqueeze(0) <= y_max.unsqueeze(1))  # (B, seq_len)

    mask = (inside_x & inside_y).float()  # (B, seq_len)

    return mask


@torch.no_grad()
def estimate_pi_answer(model, dataloader, max_batches: int = 50) -> float:
    """
    Estimate baseline prior pi_A as dataset mean of patch selection fraction.

    Args:
        model: The CoExVQA model
        dataloader: Validation dataloader
        max_batches: Maximum batches to use for estimation

    Returns:
        pi_A: Estimated prior probability (scalar in [0, 1])
    """
    model.eval()
    total_fraction_sum = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        images = batch["images"]
        questions = batch["questions"]
        B = len(questions)

        # Forward pass to get predicted bbox
        outputs = model(images=images, questions=questions)
        pred_bbox = outputs["box"]  # (B, 4) normalized cxcywh

        # Get patch grid from image dimensions
        H_p, W_p = CoExVQA.compute_patch_grid(images[0].size)
        seq_len = H_p * W_p
        patch_grid = (H_p, W_p)

        # Convert bbox to patch mask
        s_mask = bbox_to_patch_mask(pred_bbox, seq_len, patch_grid)

        # Compute per-sample selection fraction: 1/N * sum_i s_i
        fractions = s_mask.mean(dim=1)  # (B,)
        total_fraction_sum += fractions.sum().item()
        total_samples += B

        # Cleanup
        del outputs, pred_bbox, s_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_fraction_sum / total_samples if total_samples > 0 else 0.5


@torch.no_grad()
def run_answer_rate_anls_eval(
    model,
    val_loader,
    pi_A: float,
    output_dir: str,
    n_bins: int = 50,
) -> Dict[str, Any]:
    """
    Evaluate Answer Rate vs ANLS trade-off.

    Answer Rate = sum_i KL(Bern(s_i) || Bern(pi_A)) / N measures how much
    the predicted answer region deviates from typical selection pattern.

    Args:
        model: The CoExVQA model
        val_loader: Validation dataloader
        pi_A: Baseline prior probability (dataset mean)
        output_dir: Directory to save outputs
        n_bins: Number of bins for binned ANLS curve

    Returns:
        Dict with records, timing, and summary statistics
    """
    model.eval()
    gen_kwargs = get_deterministic_gen_kwargs()
    records: List[Dict[str, Any]] = []
    sample_idx = 0

    start = perf_counter()
    for batch_idx, batch in enumerate(val_loader):
        images = batch["images"]
        questions = batch["questions"]
        _, all_answers = normalize_answers(batch["answers"])
        B = len(questions)

        # Forward pass
        outputs = model(images=images, questions=questions)
        pred_bbox = outputs["box"]  # (B, 4) normalized cxcywh

        # Get patch grid from image dimensions
        H_p, W_p = CoExVQA.compute_patch_grid(images[0].size)
        seq_len = H_p * W_p
        patch_grid = (H_p, W_p)

        # Convert bbox to patch selection mask
        s_mask = bbox_to_patch_mask(pred_bbox, seq_len, patch_grid)  # (B, seq_len)

        # Compute answer rate using baseline pi_A (reuse compute_rate)
        rates = compute_rate(s_mask, pi=pi_A, normalize=True)  # (B,)

        # Generate predictions
        preds = model.generate_preds(images, questions, gen_kwargs)

        # Store per-sample records
        for i in range(B):
            target_answers = all_answers[i]
            pred_answer = preds[i]
            anls = calculate_anls(target_answers, pred_answer)
            rate_norm = rates[i].item()

            records.append({
                "sample_idx": sample_idx,
                "question": questions[i],
                "anls": anls,
                "rate_norm": rate_norm,
                "pred_answer": pred_answer,
                "target_answers": target_answers,
            })
            sample_idx += 1

        # Memory cleanup
        del outputs, pred_bbox, s_mask, images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if batch_idx % 10 == 0:
            logger.info(f"Answer Rate-ANLS eval: batch {batch_idx}/{len(val_loader)}")

    wall = perf_counter() - start
    logger.info(f"Answer Rate-ANLS evaluation completed in {wall:.2f}s for {sample_idx} samples")

    # Generate plots
    rates_list = [r["rate_norm"] for r in records]
    anls_list = [r["anls"] for r in records]

    plot_rate_anls(rates_list, anls_list, output_dir, n_bins=n_bins)
    logger.info(f"Saved Answer Rate vs ANLS plots to {output_dir}")

    # Save metrics to JSONL
    metrics_path = os.path.join(output_dir, "answer_rate_anls_metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    logger.info(f"Saved {len(records)} records to {metrics_path}")

    return {
        "records": records,
        "time_s": wall,
        "num_samples": len(records),
        "pi_A": pi_A,
        "mean_rate": float(np.mean(rates_list)),
        "mean_anls": float(np.mean(anls_list)),
    }


@torch.no_grad()
def run_joint_rate_anls_eval(
    model,
    val_loader,
    pi_q: float,
    pi_a: float,
    output_dir: str,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate joint Question Rate (Rq) and Answer Rate (Ra) vs ANLS.

    Computes both rates for each sample and produces:
    - 2D heatmap of mean ANLS across (Rq, Ra) bins
    - Quadrant analysis (low/high Rq × low/high Ra)

    Args:
        model: The CoExVQA model
        val_loader: Validation dataloader
        pi_q: Baseline prior for question mask (q_mask mean)
        pi_a: Baseline prior for answer region (bbox coverage mean)
        output_dir: Directory to save outputs
        n_bins: Number of bins per axis for heatmap

    Returns:
        Dict with records, timing, quadrant stats, and summary statistics
    """
    model.eval()
    gen_kwargs = get_deterministic_gen_kwargs()
    records: List[Dict[str, Any]] = []
    sample_idx = 0

    start = perf_counter()
    for batch_idx, batch in enumerate(val_loader):
        images = batch["images"]
        questions = batch["questions"]
        _, all_answers = normalize_answers(batch["answers"])
        B = len(questions)

        # Forward pass
        outputs = model(images=images, questions=questions)
        pred_bbox = outputs["box"]  # (B, 4) normalized cxcywh
        q_mask = outputs.get("q_mask", None)

        # Get patch grid from image dimensions
        H_p, W_p = CoExVQA.compute_patch_grid(images[0].size)
        seq_len = H_p * W_p
        patch_grid = (H_p, W_p)

        # Compute question rate (Rq) from q_mask
        if q_mask is not None:
            rq_values = compute_rate(q_mask, pi=pi_q, normalize=True)  # (B,)
        else:
            rq_values = torch.zeros(B)

        # Compute answer rate (Ra) from predicted bbox
        s_mask = bbox_to_patch_mask(pred_bbox, seq_len, patch_grid)  # (B, seq_len)
        ra_values = compute_rate(s_mask, pi=pi_a, normalize=True)  # (B,)

        # Generate predictions
        preds = model.generate_preds(images, questions, gen_kwargs)

        # Store per-sample records
        for i in range(B):
            target_answers = all_answers[i]
            pred_answer = preds[i]
            anls = calculate_anls(target_answers, pred_answer)
            rq = rq_values[i].item()
            ra = ra_values[i].item()

            records.append({
                "sample_idx": sample_idx,
                "question": questions[i],
                "anls": anls,
                "rq": rq,
                "ra": ra,
                "pred_answer": pred_answer,
                "target_answers": target_answers,
            })
            sample_idx += 1

        # Memory cleanup
        del outputs, pred_bbox, q_mask, s_mask, images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if batch_idx % 10 == 0:
            logger.info(f"Joint Rate-ANLS eval: batch {batch_idx}/{len(val_loader)}")

    wall = perf_counter() - start
    logger.info(f"Joint Rate-ANLS evaluation completed in {wall:.2f}s for {sample_idx} samples")

    # Extract lists for plotting
    rq_list = [r["rq"] for r in records]
    ra_list = [r["ra"] for r in records]
    anls_list = [r["anls"] for r in records]

    # Generate joint heatmap and quadrant analysis
    joint_stats = plot_joint_rate_heatmap(
        rq_values=rq_list,
        ra_values=ra_list,
        anls_scores=anls_list,
        output_dir=output_dir,
        n_bins=n_bins,
    )
    logger.info(f"Saved joint rate heatmap and quadrant analysis to {output_dir}")

    # Print quadrant summary
    logger.info("=== Quadrant Summary ===")
    logger.info(f"Rq median: {joint_stats['rq_median']:.4f}, Ra median: {joint_stats['ra_median']:.4f}")
    for name, stats in joint_stats['quadrant_stats'].items():
        logger.info(f"  {name}: n={stats['count']}, mean_anls={stats['mean_anls']:.4f} +- {stats['std_anls']:.4f}")

    # Save metrics to JSONL
    metrics_path = os.path.join(output_dir, "joint_rate_anls_metrics.jsonl")
    with open(metrics_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    logger.info(f"Saved {len(records)} records to {metrics_path}")

    return {
        "records": records,
        "time_s": wall,
        "num_samples": len(records),
        "pi_q": pi_q,
        "pi_a": pi_a,
        "mean_rq": float(np.mean(rq_list)),
        "mean_ra": float(np.mean(ra_list)),
        "mean_anls": float(np.mean(anls_list)),
        "joint_stats": joint_stats,
    }
