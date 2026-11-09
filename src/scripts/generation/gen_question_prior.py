import argparse
import logging
import torch 
import os 
import sys
import logging 
from datetime import datetime, timezone
from huggingface_hub.errors import HfHubHTTPError
import time

import uuid
from transformers.utils.import_utils import is_flash_attn_2_available
from datasets import load_dataset, load_from_disk

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import implementations
from src.utils.seed import set_seed
from src.utils.hf_token import get_hf_token
from src.priors.colpali.colqwen import run_colqwen_pipeline
from src.priors.colpali.colpali import run_col_smol_pipeline, run_col_smol_pfl_docvqa_pipeline
from src.priors.llama_nemoretriever import run_nvidia_llama_pipeline
from src.priors.pix2struct import run_pix2struct_pipeline
from src.scripts.evaluation.eval_question_prior import run_question_prior_evaluation

# Allow parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Default paths
PROJECT_AREA_PATH = "/cluster/work/projects/nn8104k/kjetil_thesis"
CACHE_DIR = f"{PROJECT_AREA_PATH}/.cache"
SAVE_DIR = f"{PROJECT_AREA_PATH}/q_prior_examples/"
DS_NAME_PFL_DOCVQA = f"{PROJECT_AREA_PATH}/datasets/PFL-DocVQA/"

def build_commit_info(ds, model_name)-> tuple[str, str]:
    # Get time of run
    now = datetime.now(timezone.utc)
    ts_human = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Try to get the sizes and splits
    splits = []
    split_sizes = {}
    total = 0
    try:
        splits = list(ds.keys())
        split_sizes = {s: len(ds[s]) for s in splits}
        total = sum(split_sizes.values())
    except Exception:
        pass

    commit_description = (
        f"Run: {ts_human}\n"
        f"Timestamp (UTC): {ts_human}\n"
        f"Dataset: {getattr(ds, 'dataset_name', 'unknown')}\n"
        f"Splits: {', '.join(splits) if splits else '(none)'}\n"
        f"Split sizes:\n{split_sizes}\n"
        f"Total examples: {total}\n"
    )

    commit_message = f"[RUN] docvqa q-prior run for {model_name}, {ts_human} for splits {', '.join(splits) if splits else '(none)'}."
    return commit_message, commit_description

# Allowed variants for generation
ALLOWED_Q_PRIOR_VARIANTS = [
    "vidore/colqwen2.5-v0.2",
    "vidore/colSmol-500M",
    "nvidia/llama-nemoretriever-colembed-1b-v1",
    "google/pix2struct-docvqa-base",
]

ALLOWED_SUFFIXES = {
    "vidore/colqwen2.5-v0.2": "-colqwen-q-priors", 
    "vidore/colSmol-500M": "-colSmol500M-q-priors",
    "nvidia/llama-nemoretriever-colembed-1b-v1": "-llama-colembed-1b-q-priors",
    "google/pix2struct-docvqa-base": "-pix2struct-q-priors"
}

# Set thread variable defaults
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("BLIS_NUM_THREADS", "4")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser("Generate Q-prior ")
    parser.add_argument("--model_name", type=str, default="vidore/colqwen2.5-v0.2", choices=ALLOWED_Q_PRIOR_VARIANTS, help="Model name for prior")
    parser.add_argument("--dataset_prefix", default="indrehus/docvqa-single-page-questions-answer-ocr")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--version", type=str, default="v1.0", help="Extra meta information to version the runs")
    parser.add_argument("--num_examples", type=int, default=15, help="Max number of visual examples to save (per entire run).")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    parser.add_argument("--top_k_percent", type=float, default=0.30)
    parser.add_argument("--sparsity_threshold", type=float, default=0.01)
    parser.add_argument("--save_patch_prior_only", action="store_true", help="If set, only save the post processed patch priors in the dataset.")
    parser.add_argument("--stop_after_visuals", action="store_true", help="If set, only create visualizations and stop before saving dataset.")
    parser.add_argument("--push_to_hf", action="store_true", help="If set, pushes hf examples.")
    parser.add_argument("--load_from_local", action="store_true", help="If set, loads the daset locally instead of processing it again.")
    parser.add_argument("--skip_eval", action="store_true", help="If set, skips the question prior evaluation.")
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pipeline", type=str, default="hf_dataset", choices=["hf_dataset", "pfl_docvqa"], help="Pipeline mode: 'hf_dataset' processes a HuggingFace dataset, 'pfl_docvqa' generates priors for the PFL-DocVQA local dataset")
    parser.add_argument("--pfl_data_dir", type=str, default=DS_NAME_PFL_DOCVQA, help="Path to PFL-DocVQA data directory (used when --pipeline pfl_docvqa)")
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

    # Setting up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Check if flash attention is available
    att_imp = is_flash_attn_2_available()
    logger.info(f"Flash attention 2 available: {att_imp}")

    if args.pipeline == "pfl_docvqa":
        # PFL-DocVQA pipeline: generate priors as .pt files, no HF push
        logger.info(f"Running PFL-DocVQA pipeline for data_dir: {args.pfl_data_dir}")
        run_col_smol_pfl_docvqa_pipeline(
            data_dir=args.pfl_data_dir,
            logger=logger,
            args=args,
        )
        logger.info("PFL-DocVQA pipeline completed.")
    else:
        # HF dataset pipeline (existing behavior)
        # Create new dataset name
        dataset_suffix = ALLOWED_SUFFIXES.get(args.model_name, "-q-priors")
        new_ds_name = f"{args.dataset_prefix}{dataset_suffix}"
        logger.info(f"New dataset name: {new_ds_name}")
        logger.info(f"Created with the model: {args.model_name}")

        # Make outdir for visualizations
        os.makedirs(args.save_dir, exist_ok=True)
        run_id = uuid.uuid4().hex[:10]
        out_dir = os.path.join(args.save_dir, run_id)
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Saving examples to: {out_dir}")

        # Guardrail checks for dataset saved and the suffix set
        if new_ds_name == "indrehus/docvqa-single-page-questions-answer-ocr" or new_ds_name == args.dataset_prefix:
            raise ValueError("Cannot overwrite the original dataset")

        # Get HF Token
        hf_token = get_hf_token()
        logger.info("HuggingFace token retrieved successfully.")

        # Load the dataset
        logger.info(f"Loading dataset: {args.dataset_prefix}")
        ds = load_dataset(args.dataset_prefix, cache_dir=args.cache_dir)
        logger.info("Dataset loaded.")

        if args.load_from_local:
            end = new_ds_name.split("/")[-1]
            path = os.path.join(PROJECT_AREA_PATH, "datasets", end)
            logger.info(f"Loading from disk: {path}")
            ds = load_from_disk(path)
        else:
            # Run the pipeline for the model
            if args.model_name == "vidore/colqwen2.5-v0.2":
                ds = run_colqwen_pipeline(ds, out_dir, logger, args)
            elif args.model_name == "vidore/colSmol-500M":
                ds = run_col_smol_pipeline(ds, out_dir, logger, args)
            elif args.model_name == "nvidia/llama-nemoretriever-colembed-1b-v1":
                ds = run_nvidia_llama_pipeline(ds, out_dir, logger, args)
            elif args.model_name == "google/pix2struct-docvqa-base":
                ds = run_pix2struct_pipeline(ds, out_dir, logger, args)
            else:
                raise ValueError(f"Unknown model: {args.model_name}")

        # Save locally if not loaded from disk
        if not args.load_from_local:
            local_save_path = os.path.join(PROJECT_AREA_PATH, "datasets", new_ds_name.split('/')[-1])
            try:
                logger.info(f"Saving dataset locally to: {local_save_path}")
                os.makedirs(os.path.dirname(local_save_path), exist_ok=True)
                ds.save_to_disk(local_save_path)
                logger.info("Dataset saved locally successfully.")
            except Exception as e:
                logger.error(f"Error while saving locally: {e}")

        # Run evaluation
        if args.skip_eval:
            logger.info("Skipping eval")
        else:
            logger.info("Running eval")
            run_question_prior_evaluation(
                ds=ds,
                logger=logger,
                args=args
            )

        # Push to HuggingFace Hub
        if args.push_to_hf:
            commit_message, commit_description = build_commit_info(ds, model_name=args.model_name)
            max_retries = 3
            logger.info(f"Pushing dataset to HF Hub: {new_ds_name}")
            for attempt in range(max_retries):
                try:
                    logger.info(f"Upload attempt {attempt+1} of {max_retries}")
                    ds.push_to_hub(
                        new_ds_name,
                        token=hf_token,
                        private=True,
                        commit_message=commit_message,
                        commit_description=commit_description,
                    )
                    logger.info(f"Saved dataset to HF: {new_ds_name}")
                    break
                except HfHubHTTPError as e:
                    if "504" in str(e) or "timeout" in str(e).lower():
                        if attempt < max_retries - 1:
                            wait_time = 300 * (attempt + 1)
                            logger.warning(f"Upload timeout. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error("Max retries reached. Upload failed.")
                            raise
                    else:
                        raise
        else:
            logger.info("Skipping pushing to HF")
