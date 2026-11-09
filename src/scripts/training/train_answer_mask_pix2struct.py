import argparse
import logging
import torch 
import sys
from torch.optim import AdamW
from transformers import AutoProcessor, AutoModelForVision2Seq, get_linear_schedule_with_warmup
import os


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Allow parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Import implementations
from src.utils.hf_token import get_hf_token
from src.datasets.DocVqaAnswer import get_train_val_docvqa_answer_dataloaders
from src.datasets.collates import get_collate
from src.utils.seed import set_seed
from src.utils.wandb_logging import setup_wandb
from src.datasets.ops import normalize_answers

# Default paths
PROJECT_AREA_PATH = "/cluster/work/projects/ec35/ec-kjetili-master-thesis"
CACHE_DIR = f"{PROJECT_AREA_PATH}/.cache"


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/pix2struct-docvqa-base")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)   
    parser.add_argument("--batch_size", type=int, default=10)  
    parser.add_argument("--log_step", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default=PROJECT_AREA_PATH)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--preproc", choices=["masked_answer_images","cropped_answer_images"], default="masked_answer_images")
    parser.add_argument("--crop_margin", type=float, default=0.10, help="Fractional margin around bbox for cropped preproc.")
    parser.add_argument("--min_crop_height", type=int, default=256, help="Min crop height before width-capping (readability).")
    parser.add_argument("--max_image_width", type=int, default=700)
    parser.add_argument("--max_patches", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, 
        handlers=[logging.StreamHandler(sys.stdout)], force=True
    )
    logger = logging.getLogger(__name__)

    # Log the title of the script
    logger.info("Starting training script for Pix2struct Experiment 1")


    # Setup wandb 
    logger.info("Setting up Wandb")
    args_dict = args.__dict__
    run = setup_wandb(config_dict=args_dict, project_name="Answer Encoder Decoder", prefix="answer-enc-dec-train")
    logger.info(f"Wandb run initialized with name: {run.name} and id: {run.id}")
    run_name = run.name

    # Create dir for save path
    save_path = f"{args.save_dir}/pix2struct-masked-train/{run_name}"
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Created save dir for the run: '{save_path}'")
    
    # Set the seed 
    set_seed(args.seed)
    logger.info(f"Seed set to: {args.seed}")

    # Log the arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"    {arg}: {value}")


    # Setting up the device:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Get HF Token
    hf_token = get_hf_token()
    logger.info("HuggingFace token retrieved successfully.")


    # Load model directly
    logger.info(f"Setting up '{args.model_name}' processor and model")
    processor = AutoProcessor.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForVision2Seq.from_pretrained(args.model_name)
    model.to(device)

    # Log gradient 
    logger.info("Watching gradients with Wandb")
    run.watch(model, log="gradients", log_freq=100)


    # Get the pre processing collate function     
    collate_fn = get_collate(
        args.preproc,
        max_width=args.max_image_width,
        crop_margin=args.crop_margin,
        min_height=args.min_crop_height,
    )

    logger.info("Creating train and validation dataloaders...")
    train_dataloader, val_dataloader = get_train_val_docvqa_answer_dataloaders(
        cache_dir=args.cache_dir, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn
    )
    logger.info("Train and validation dataloaders created successfully.")
    logger.info(f"Image width resize set to: {args.max_image_width}")


    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * max(1, len(train_dataloader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * num_training_steps),
        num_training_steps=num_training_steps
    )
    logger.info("AdamW and Scheduler with warmup created")

    # Get the padding token 
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        pad_id = processor.tokenizer.pad_token_id

    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improve = 0


    # Check what dtype is available
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Main training loop
    logger.info("Training started")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for step, batch in enumerate(train_dataloader, start=1):
            images = batch["masked_image"]                      # list[PIL.Image]
            questions = batch["questions"]                      # list[str]
            first_answer, all_answers = normalize_answers(batch["answers"])  # tuple[list[str], list[list[str]]]
            answers = first_answer  # Use first_answer for tokenization

            # Encoder inputs from images + questions
            enc = processor(
                images=images,
                text=questions,
                padding=True,
                return_tensors="pt",
                max_patches=args.max_patches,
            )

            # Tokenize the labels
            labels = processor.tokenizer(
                answers,
                padding=True,
                return_tensors="pt",
            ).input_ids

            # Mask out padding in labels
            labels[labels == pad_id] = -100

            # Move to device
            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)
            

            # Forward pass
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(**enc, labels=labels)
                loss = outputs.loss


            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running += loss.item()
            global_step += 1

            # Log step
            if step % args.log_step == 0: 
                steps_per_epoch = len(train_dataloader)
                pct = 100 * step / steps_per_epoch
                logger.info(f"      [Epoch {epoch} | Step {step}/{steps_per_epoch}] ({pct:.0f}%) loss: {loss.item():.4f}")

            # Delete all and free memory
            del enc, labels, outputs

        # Log after the epoch to get th loss
        train_avg = running / len(train_dataloader)
        logger.info(f"[Epoch {epoch}] - train loss (average): {train_avg:.4f}")
        run.log({
            "epoch": epoch,
            "train/loss": train_avg,
            "lr": scheduler.get_last_lr()[0],
        }, commit=True)

        # Clear cache
        torch.cuda.empty_cache()

        # Validation 
        model.eval()
        val_losses = []
        with torch.no_grad():
            for vb in val_dataloader:
                v_images = vb["masked_image"]
                v_questions = vb["questions"]
                first_answer, all_answers = normalize_answers(vb["answers"])
                v_answers = first_answer  # Use first_answer for tokenization

                v_enc = processor(
                    images=v_images,
                    text=v_questions,
                    padding=True,
                    return_tensors="pt",
                    max_patches=args.max_patches
                )
                v_labels = processor.tokenizer(
                    v_answers,
                    padding=True,
                    return_tensors="pt"
                ).input_ids
                v_labels[v_labels == pad_id] = -100

                v_enc = {k: v.to(device) for k, v in v_enc.items()}
                v_labels = v_labels.to(device)

                v_out = model(**v_enc, labels=v_labels)
                val_losses.append(v_out.loss.item())

        mean_val = float(torch.tensor(val_losses).mean()) if val_losses else float("nan")
        logger.info(f"[Epoch {epoch}] Val Loss: {mean_val:.4f}")
        run.log({
            "epoch": epoch,
            "val/loss": mean_val,
        }, commit=True)

        # Save best
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            logger.info(f"New best val loss {best_val_loss:.4f}. Saving to {save_path}")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            logger.info(f"No val improvement this epoch "
                        f"({epochs_without_improve}/{args.early_stop_patience}).")
            if epochs_without_improve >= args.early_stop_patience:
                logger.info("Early stopping triggered (patience reached).")
                break

    # Log that training is complete 
    logger.info("Training complete")

    # Log best validation loss
    logger.info(f"Best validation loss was {best_val_loss}")



