import argparse
import torch 
import uuid
import os 
import sys
import logging 
from datasets import load_dataset, load_from_disk
from collections import defaultdict
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import implementations
from src.utils.seed import set_seed
from src.utils.hf_token import get_hf_token
from src.evaluation.question_prior_evaluators import QuestionPriorEvaluator
from src.utils.viz.prior_viz import save_prior_and_bbox_visualization
from src.datasets.ops import bbox_to_int

# Allow parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Default paths
PROJECT_AREA_PATH = "/cluster/work/projects/nn8104k/kjetil_thesis"
CACHE_DIR = f"{PROJECT_AREA_PATH}/.cache"
SAVE_DIR_PLOTS = f"{PROJECT_AREA_PATH}/eval_q_prior_plots/"



def run_question_prior_evaluation(ds, logger, args):
    # Unique save dir
    os.makedirs(args.save_dir, exist_ok=True)
    run_id = uuid.uuid4().hex[:10]
    out_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Saving examples to: {out_dir}")


    # Collect metric 
    question_prior_evaluator = QuestionPriorEvaluator(
        top_k_percent=args.top_k_percent,
        sparsity_threshold=args.sparsity_threshold,
    )

    # Columns for question prior
    question_prior_key = "q_prior_mask"
    question_prior_pp_key = "q_prior_mask_postprocessed"
    answer_bbox_key = "answer_ocr_bbox"
    
    # Store all results
    all_results = {}
    examples_visualized = 0 
    
    # Loop over splits
    for split_name in ds.keys():
        # Ignore the split method
        if split_name.strip().lower() == "test":
            logger.info("Skipping test split")
            continue

        # Log the split name
        logger.info(f"Processing split: {split_name}")
        
        # Get the data for the split
        split_data = ds[split_name]
        
        # Log the number of examples evaluated
        logger.info(f"Evaluating all {len(split_data)} examples")
        
        # Initialize metric accumulators
        metrics_no_pp = defaultdict(list)
        metrics_with_pp = defaultdict(list)
        
        # Process each example
        for idx, example in enumerate(split_data):
            # Log status
            if idx % 300 == 0:
                logger.info(f"Processing example {idx}/{len(split_data)}")

            # Log the columns if this is the very first example
            if idx == 0:
                logger.info(f"Available columns in {split_name} split: {list(example.keys())}")

            # Convert the bbox to correct shape
            image = example["image"]
            W, H = image.size

            # Skip evaluation if there are no bbox here
            raw_ocr_box = example[answer_bbox_key]
            if raw_ocr_box is None:
                continue

            x1, y1, x2, y2 = bbox_to_int(raw_ocr_box, W, H)
            gt_bbox = (x1, y1, x2 , y2)

            # Extract data
            prior_heatmap = torch.tensor(example[question_prior_key], dtype=torch.float32).reshape(H, W)
            prior_heatmap_pp = torch.tensor(example[question_prior_pp_key], dtype=torch.float32).reshape(H, W)
            
            # Evaluate without post-processing
            metrics = question_prior_evaluator.evaluate(prior_heatmap, gt_bbox)
            for metric_name, value in metrics.items():
                metrics_no_pp[metric_name].append(value.item() if torch.is_tensor(value) else value)
            
            # Evaluate with post-processing
            metrics_pp = question_prior_evaluator.evaluate(prior_heatmap_pp, gt_bbox)
            for metric_name, value in metrics_pp.items():
                metrics_with_pp[metric_name].append(value.item() if torch.is_tensor(value) else value)

            # Visualize example 
            if examples_visualized < args.num_examples:
                # Create example name 
                example_id = uuid.uuid4().hex[:10]
                example_name = f"example_{example_id}.png"
                save_path = os.path.join(out_dir, example_name)

                # Visualize example
                save_prior_and_bbox_visualization(
                    image=example["image"], 
                    prior_heatmap=prior_heatmap.numpy(), 
                    gt_bbox=gt_bbox,
                    save_path=save_path,
                )
                examples_visualized += 1
                
        # Compute means
        mean_metrics_no_pp = {k: sum(v) / len(v) for k, v in metrics_no_pp.items()}
        mean_metrics_with_pp = {k: sum(v) / len(v) for k, v in metrics_with_pp.items()}
        
        # Store results
        all_results[split_name] = {
            'without_pp': mean_metrics_no_pp,
            'with_pp': mean_metrics_with_pp
        }
        
        # Log results for this split
        logger.info(f"Results for {split_name.upper()}")
        logger.info("   Without Post-Processing:")
        for metric, value in mean_metrics_no_pp.items():
            logger.info(f"      {metric:30s}: {value:.4f}")
        
        logger.info("   With Post-Processing:")
        for metric, value in mean_metrics_with_pp.items():
            logger.info(f"      {metric:30s}: {value:.4f}")
    
    # Print final summary table
    logger.info("FINAL SUMMARY")
    for split_name in all_results.keys():
        logger.info(f"Data split {split_name.upper()}:")
        logger.info(f"{'Metric':<30} {'No PP':>12} {'With PP':>12} {'Diff':>12}")
        
        for metric in all_results[split_name]['without_pp'].keys():
            no_pp = all_results[split_name]['without_pp'][metric]
            with_pp = all_results[split_name]['with_pp'][metric]
            diff = with_pp - no_pp
            logger.info(f"{metric:<30} {no_pp:>12.4f} {with_pp:>12.4f} {diff:>+12.4f}")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser("Evaluate Q-prior")
    parser.add_argument("--dataset", default="indrehus/docvqa-single-page-questions-answer-ocr-colqwen-q-priors")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR_PLOTS)
    parser.add_argument("--load_from_local", action="store_true", help="If set, loads the daset locally instead of loading from HF.")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--top_k_percent", type=float, default=0.30)
    parser.add_argument("--sparsity_threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, 
        handlers=[logging.StreamHandler(sys.stdout)], force=True
    )
    logger = logging.getLogger(__name__)

    # Set the seed 
    set_seed(args.seed)
    logger.info(f"Seed set to: {args.seed}")

    # Log the arguments
    logger.info("Arguments:")
    for k in sorted(vars(args)):
        v = getattr(args, k)
        logger.info("    %s: %s", k, str(v))

    # Setting up the device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Get HF Token
    hf_token = get_hf_token()
    logger.info("HuggingFace token retrieved successfully.")

    # Load the dataset
    if args.load_from_local:
        end = args.dataset.split("/")[-1]
        path = os.path.join(PROJECT_AREA_PATH, "datasets", end)
        logger.info(f"Loading from disk: {path}")
        ds = load_from_disk(path)
    else:
        logger.info(f"Loading dataset from HF: {args.dataset}")
        ds = load_dataset(args.dataset, cache_dir=args.cache_dir)

    logger.info("Dataset loaded.")

    # Run evaluation
    run_question_prior_evaluation(ds, logger)


    