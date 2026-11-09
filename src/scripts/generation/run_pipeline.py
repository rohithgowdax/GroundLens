import argparse
import logging
import os 
import sys
from datetime import datetime, timezone
import time

from dotenv import load_dotenv
load_dotenv()


# Import pipeline to use 
from src.pipelines.DocVQAPipeline import DocVQAPipeline
from src.utils.hf_token import get_hf_token


# Set thread variable defaults
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("BLIS_NUM_THREADS", "4")

def parse_args():
    p = argparse.ArgumentParser(description="Run DocVQAPipeline")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of samples per split.")
    p.add_argument("--device", type=str, default="cpu", help="Device to use (e.g., 'cpu').")
    p.add_argument("--log-interval", type=int, default=1, help="Log progress every N examples.")
    p.add_argument("--verbose", default=False, help="Enable verbose pipeline logging.")
    p.add_argument("--private", default=True, help="Push repo as private (if push logic is enabled).")
    p.add_argument("--dataset_name", type=str, default="pixparse/docvqa-single-page-questions", help="Dataset repository name.")
    p.add_argument("--new_dataset_name", default="indrehus/docvqa-single-page-questions")
    p.add_argument("--cache_dir", default=os.getenv("HF_DATASETS_CACHE", ""))
    return p.parse_args()

if __name__ == "__main__":
    # Argument parsing
    args = parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO,
        force=True
    )
    logger = logging.getLogger(__name__)

    # Load environment variables
    logger.info("Retrieving HF token..")
    

    # Get the HuggingFace token from environment variables
    hf_token = get_hf_token()

    # Setup pipeline 
    logger.info("Setting up pipeline..")
    pipeline = DocVQAPipeline(
        ds_name=args.dataset_name, 
        new_ds_name=args.new_dataset_name,
        hf_token=hf_token,
        device=args.device,
        verbose=args.verbose,
        cache_dir=args.cache_dir
    )
    logger.info(f"Pipeline initialized successfully, for dataset: '{pipeline.dataset_name}'.")

    # Process the dataset and log time taken
    start = time.perf_counter()
    start_ts = datetime.now(timezone.utc)

    pipeline.process_dataset(
        logger,
        max_examples=args.max_samples,
        log_interval=args.log_interval,
        private=args.private,
    )

    # after processing
    end = time.perf_counter()
    end_ts = datetime.now(timezone.utc)

    elapsed = end - start

    # compute total examples processed 
    def count_examples(ds_dict, cap):
        total = 0
        for _, ds in ds_dict.items():
            n = len(ds)
            total += n if cap is None else min(cap, n)
        return total

    n_examples = count_examples(pipeline.dataset, args.max_samples)

    # Log time and result
    logger.info(f"Start: {start_ts:%Y-%m-%d %H:%M:%S %Z}")
    logger.info(f"End:   {end_ts:%Y-%m-%d %H:%M:%S %Z}")
    logger.info(f"Elapsed: {elapsed:.2f} s")
    if n_examples:
        logger.info(f"Time per example: {elapsed / n_examples:.4f} s/ex")