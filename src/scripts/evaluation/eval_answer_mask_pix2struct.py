import argparse
import logging
import torch 
import sys
from transformers import AutoProcessor, AutoModelForVision2Seq
from time import perf_counter
import os


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Allow parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Import implementations
from src.utils.hf_token import get_hf_token
from src.datasets.DocVqaAnswer import get_docvqa_answer_dataloader
from src.datasets.collates import get_collate
from src.utils.seed import set_seed
from src.evaluation.docvqa_metrics import DocVQAEvaluator
from src.utils.gen_args import build_gen_kwargs

# Default paths
PROJECT_AREA_PATH = "/cluster/work/projects/ec35/ec-kjetili-master-thesis"
DEFAULT_MODEL_SAVED_DIR_PATH= f"{PROJECT_AREA_PATH}/pix2struct-masked-train/answer-enc-dec-train-20250921-004330"
CACHE_DIR = f"{PROJECT_AREA_PATH}/.cache"

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_dir", type=str, default=DEFAULT_MODEL_SAVED_DIR_PATH)
    parser.add_argument("--batch_size", type=int, default=10)  
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true", help="Enable stochastic sampling")
    parser.add_argument("--length_penalty", type=float, default=None, help=">1 favors longer; <1 shorter")
    parser.add_argument("--early_stopping", action="store_true", help="Stop beam search when beams are complete")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--preproc", choices=["masked_answer_images","cropped_answer_images"], default="masked_answer_images")
    parser.add_argument("--crop_margin", type=float, default=0.10, help="Fractional margin around bbox for cropped preproc.")
    parser.add_argument("--min_crop_height", type=int, default=256, help="Min crop height before width-capping (readability).")
    parser.add_argument("--max_image_width", type=int, default=700)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, 
        handlers=[logging.StreamHandler(sys.stdout)], force=True
    )
    logger = logging.getLogger(__name__)

    # Log the title of the script
    logger.info("Starting evaluation script of Pix2struct Experiment 1")

    # Create dir for save path
    logger.info(f"Evaluating model in save dir for the run: '{args.model_save_dir}'")
    
    # Set the seed 
    set_seed(args.seed)
    logger.info(f"Seed set to: {args.seed}")

    # Log the arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"    {arg}: {value}")


    # Set up generation args
    gen_kwargs = build_gen_kwargs(args,logger)

    # Setting up the device:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Get HF Token
    hf_token = get_hf_token()
    logger.info("HuggingFace token retrieved successfully.")


    # Load model directly
    logger.info(f"Loading pretrained processor and model")
    processor = AutoProcessor.from_pretrained(args.model_save_dir, use_fast=True)
    model = AutoModelForVision2Seq.from_pretrained(args.model_save_dir)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

    # Get the pre processing collate function     
    collate_fn = get_collate(
        args.preproc,
        max_width=args.max_image_width,
        crop_margin=args.crop_margin,
        min_height=args.min_crop_height,
    )

    # Loading datasets 
    logger.info(f"Loading validation dataset with preprocessing: ")
    val_dataloader = get_docvqa_answer_dataloader(
        cache_dir=args.cache_dir, 
        split="validation", 
        shuffle=False, 
        batch_size=args.batch_size, 
        num_workers=0, 
        collate_fn=collate_fn
    )
    logger.info("Validation dataloaders created successfully.")
    logger.info(f"Image width resize set to: {args.max_image_width}")


    # Collect the generation answers and predictions 
    logger.info("Running inference on validation split")
    start = perf_counter()
    val_questions = []
    val_target_answers = []
    val_pred_answers = []

    for batch in val_dataloader:
        images = batch["masked_image"]
        current_question = batch["questions"]
        answers = batch["answers"]

        enc = processor(
            images=images,
            text=current_question,
            return_tensors="pt",
            padding=True,
            max_patches=512
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Generate predictions
        pred_ids = model.generate(**enc, **gen_kwargs)

        # Decode to strings
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        preds = [p.strip() for p in preds]

        # Accumulate
        val_questions.extend(current_question)
        val_target_answers.extend(answers)
        val_pred_answers.extend(preds)

    # Inference done
    total_time = perf_counter() - start
    per_sample = total_time / max(1, len(val_pred_answers))
    logger.info(f"Inference done. Took {total_time:.2f} seconds total, and {per_sample:.4f} seconds per sample.")


    # Log some samples for sanity checks 
    samples_to_log = min(args.num_samples, len(val_pred_answers))
    logger.info(f"Logging {samples_to_log} sample queries from generation")

    for i in range(samples_to_log):
        current_question = val_questions[i]
        current_answers = val_target_answers[i] 
        pred = val_pred_answers[i]

        # Cleanup the string
        current_answers_cleaned = "[" + ", ".join(map(str, current_answers)) + "]"

        # log the sample
        logger.info(
            "\n"
            f"=== Sample {i+1}/{samples_to_log} ===\n"
            f"Q: {current_question}\n"
            f"A:{current_answers_cleaned}\n"
            f"PRED: {pred}\n"
        )

    # Run DocVQA evaluation 
    logger.info("Running DocVQA evaluation")
    start = perf_counter()
    results = DocVQAEvaluator.evaluate(target_answers=val_target_answers, pred_answers=val_pred_answers)
    total_time = perf_counter() - start
    per_sample = total_time / max(1, len(val_pred_answers))
    logger.info(f"DocVQA evaluation took {total_time:.2f} seconds, and {per_sample:.4f} seconds per sample.")

    # Get the metrics 
    acc = results["acc"]
    anls = results["anls"]

    # Log final metrics
    logger.info(f"[METRICS] ACC: {acc:.4f} | ANLS: {anls:.4f}")


    





   