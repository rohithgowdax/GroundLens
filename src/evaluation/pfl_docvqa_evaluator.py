import time
import logging
from typing import Dict
import torch

from src.evaluation.docvqa_metrics import calculate_anls, calculate_accuracy
from src.utils.gen_args import get_deterministic_gen_kwargs

logger = logging.getLogger(__name__)

@torch.no_grad()
def run_pfl_docvqa_eval(
    model,
    val_loader
) -> Dict:
    """
    Run ANLS and ACC evaluation on PFL-DocVQA dataset.

    Args:
        model: CoExVQA model for inference
        val_loader: DataLoader yielding batches with 'images', 'questions', 'answers'

    Returns:
        dict with:
        - anls: float (average ANLS score)
        - acc: float (average accuracy)
        - num_examples: int (total samples evaluated)
        - time_s: float (wall clock time)
    """
    model.eval()
    start_time = time.time()

    total_anls = 0.0
    total_acc = 0.0
    num_examples = 0
    gen_kwargs = get_deterministic_gen_kwargs()

    # Iterate over validation data
    total_batches = len(val_loader)
    for batch_idx, batch in enumerate(val_loader):
        images = batch["images"]
        questions = batch["questions"]
        answers = batch["answers"]

        # Generate predictions
        predictions = model.generate_preds(
            images=images,
            questions=questions,
            gen_kwargs=gen_kwargs,
        )

        # Compute metrics for each sample
        for pred, target_answers in zip(predictions, answers):
            anls = calculate_anls(target_answers=target_answers, pred_answer=pred)
            acc = calculate_accuracy(target_answers=target_answers, pred_answer=pred)

            total_anls += anls
            total_acc += acc
            num_examples += 1

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{total_batches} | examples={num_examples}")

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"PFL-DocVQA evaluation completed in {elapsed_time:.2f} seconds over {num_examples} examples.")

    # Compute averages
    avg_anls = total_anls / num_examples if num_examples > 0 else 0.0
    avg_acc = total_acc / num_examples if num_examples > 0 else 0.0

    # Return results as dict
    results = {
        "anls": avg_anls,
        "acc": avg_acc,
        "num_examples": num_examples,
        "time_s": elapsed_time,
    }

    return results
