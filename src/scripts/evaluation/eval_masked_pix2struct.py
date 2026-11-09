import argparse
import logging
import torch 
import torch.nn.functional as F
import sys
import math
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.utils import logging as hf_logging
from typing import cast
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device

from time import perf_counter
import os
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor


# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Allow parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set logging to silence for huggingface 
hf_logging.set_verbosity_error()       
hf_logging.disable_progress_bar()

# Import implementations
from src.datasets.DocVqaAnswer import get_docvqa_answer_dataloader
from src.datasets.collates import get_collate
from src.utils.seed import set_seed
from src.utils.hf_token import get_hf_token
from src.utils.viz.eval_masked_viz import save_example
from src.evaluation.docvqa_metrics import DocVQAEvaluator
from src.utils.gen_args import build_gen_kwargs

# Default paths
PROJECT_AREA_PATH = "/cluster/work/projects/ec35/ec-kjetili-master-thesis"
DEFAULT_MODEL_SAVED_DIR_PATH= f"{PROJECT_AREA_PATH}/pix2struct-masked-train/answer-enc-dec-train-20250921-004330"
CACHE_DIR = f"{PROJECT_AREA_PATH}/.cache"
IMAGE_SAVE_DIR = f"{PROJECT_AREA_PATH}/eval_masked_pix2struct_samples"


def binarize_heatmap(hmap: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    m = (hmap >= threshold)
    return m.to(torch.uint8)

def make_mask_variant(Ha: torch.Tensor, Hq: torch.Tensor, variant: str) -> torch.Tensor:
    """
    Ha, Hq: binary masks uint8 {0,1}, shape (H,W).
    Supported variants: 'Ha', 'Ha_union_Hq', 'Ha_c_inter_Hq', 'Ha_c_inter_Hq_c', 'random_like_Hq'
    """
    if variant == "Ha_union_Hq":
        M = (Ha | Hq)
    elif variant == "Ha_c_inter_Hq":
        M = ((1 - Ha) & Hq)
    elif variant == "Ha_c_inter_Hq_c":
        M = ((1 - Ha) & (1 - Hq))
    elif variant == "random_like_Hq":
        M = random_mask_like(Hq)
    else:
        raise ValueError(f"Unknown mask variant: {variant}")
    return M.to(torch.uint8)

def random_mask_like(ref_mask: torch.Tensor) -> torch.Tensor:
    """
    Create a random binary mask with the same number of 1s as ref_mask.
    """
    n = ref_mask.numel()
    k = int(ref_mask.sum().item())
    if k <= 0:
        return torch.zeros_like(ref_mask, dtype=torch.uint8)
    flat = torch.zeros(n, dtype=torch.uint8, device=ref_mask.device)
    idx = torch.randperm(n, device=ref_mask.device)[:k]
    flat[idx] = 1
    return flat.view_as(ref_mask)

@torch.no_grad()
def get_question_heatmaps(li_model, li_processor, images, questions, device="cuda:0", eps=1e-8):
    """
    Returns a list[torch.Tensor(H, W)] of per-image heatmaps in [0,1].
    Handles extra non-spatial tokens by trimming them before reshaping.
    """
    # Encode
    enc_img = li_processor.process_images(images).to(device)
    enc_txt = li_processor.process_queries(questions).to(device)

    # Forward -> embeddings
    img_emb = li_model(**enc_img)  # (B, P, D)   P may include special tokens
    txt_emb = li_model(**enc_txt)  # (B, L, D)

    # Cosine similarity
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    sims = torch.einsum("bld,bpd->blp", txt_emb, img_emb)

    # ColBERT-style MaxSim over query tokens -> (B, P_all)
    patch_scores = sims.max(dim=1).values  # (B, P_all)
    B, P_all = patch_scores.shape

    # Spatial grid from processed image size
    Hp, Wp = enc_img["pixel_values"].shape[-2:]  # processed tensor H,W
    patch = getattr(li_model, "patch_size", 14)
    g_ny, g_nx = Hp // patch, Wp // patch
    gP = g_ny * g_nx

    # If there are extra non-spatial tokens, keep the last gP columns (patch tokens are typically last)
    if gP > 0 and P_all != gP:
        if P_all > gP:
            patch_scores = patch_scores[:, -gP:]  # drop leading specials
            P_all = gP
        else:
            # Rare: fewer tokens than grid; fall back to sqrt grid that matches P_all
            g_ny = int(math.sqrt(P_all))
            g_nx = max(1, P_all // g_ny)
            if g_ny * g_nx != P_all:
                g_ny, g_nx = P_all, 1

    # Reshape to (B,1,ny,nx)
    pm = patch_scores.view(B, 1, g_ny, g_nx)

    # Build heatmaps at original image sizes
    heatmaps = []
    for b in range(B):
        # Original size
        img = images[b]
        if isinstance(img, Image.Image):
            W, H = img.size
        else:
            H, W = img.shape[-2], img.shape[-1]

        # Upsample this sample's patch map to original size
        h = F.interpolate(pm[b:b+1], size=(H, W), mode="bilinear", align_corners=False)[0, 0]

        # Normalize to [0,1]
        vmin, vmax = h.min(), h.max()
        h = (h - vmin) / (vmax - vmin + 1e-8) if (vmax - vmin) > eps else torch.zeros_like(h)

        heatmaps.append(h.cpu())

    return heatmaps




def apply_mask_and_crop(images, masks, crop_margin_frac: float = 0.10, min_crop_h: int = 256):
    """
    images: list of PIL.Image or torch tensors (C,H,W)
    masks:  list of torch.uint8 (H,W) with 1=keep/visible, 0=mask-out
    Returns list of PIL.Image after masking and tight cropping around mask>0 with margin.
    If mask is empty -> return original image (no crop).
    """
    out = []
    for img, m in zip(images, masks):
        timg = to_tensor(img)             # (C,H,W)
        C, H, W = timg.shape
        m = m.to(torch.uint8)
        if m.shape != (H,W):
            m = torch.nn.functional.interpolate(m[None,None].float(), size=(H,W), mode="nearest").squeeze(0).squeeze(0).to(torch.uint8)

        # Apply mask: set outside to 0
        masked = timg * m.unsqueeze(0).float()

        # Find bbox of mask>0
        ys, xs = torch.where(m > 0)
        if ys.numel() == 0:
            # no crop if empty mask
            out.append(to_pil_image(masked))
            continue

        y0, y1 = ys.min().item(), ys.max().item()
        x0, x1 = xs.min().item(), xs.max().item()

        # Margin in pixels
        y_margin = int(round((y1 - y0 + 1) * crop_margin_frac))
        x_margin = int(round((x1 - x0 + 1) * crop_margin_frac))
        y0 = max(0, y0 - y_margin); y1 = min(H - 1, y1 + y_margin)
        x0 = max(0, x0 - x_margin); x1 = min(W - 1, x1 + x_margin)

        # Enforce minimum crop height
        crop_h = (y1 - y0 + 1)
        if crop_h < min_crop_h:
            pad = (min_crop_h - crop_h) // 2
            y0 = max(0, y0 - pad)
            y1 = min(H - 1, y1 + pad + ((min_crop_h - crop_h) % 2))

        cropped = masked[:, y0:y1+1, x0:x1+1]
        out.append(to_pil_image(cropped))
    return out




# Allowed masked variants 
MASK_VARIANTS = [
    "Ha_union_Hq",
    "Ha_c_inter_Hq",
    "Ha_c_inter_Hq_c",
    "random_like_Hq",
]

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--li_model_name", type=str, default="vidore/colpali-v1.2")
    parser.add_argument("--model_save_dir", type=str, default=DEFAULT_MODEL_SAVED_DIR_PATH)
    parser.add_argument("--batch_size", type=int, default=6)  
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--image_save_dir", type=str, default=IMAGE_SAVE_DIR)
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
    parser.add_argument("--crop_margin", type=float, default=0.10, help="Fractional margin around bbox for cropped preproc.")
    parser.add_argument("--min_crop_height", type=int, default=256, help="Min crop height before width-capping (readability).")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    # Ensure the save dir for the output exist, else create it 
    os.makedirs(args.image_save_dir, exist_ok=True)

    # Setting up logging:
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, 
        handlers=[logging.StreamHandler(sys.stdout)], force=True
    )
    logger = logging.getLogger(__name__)

    # Log the title of the script
    logger.info("Pix2struct Masked Evaluation Experiment 1D")

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

    # Load Colpali
    load_msg = f"Loading LI model: '{args.li_model_name}' model and processor"
    logger.info(load_msg)

    # Load the model
    li_model = cast(
        ColPali,
        ColPali.from_pretrained(
            args.li_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=args.cache_dir
        ),
    ).eval().to(device)

    # Load the processor
    li_processor = cast(
        ColPaliProcessor, 
        ColPaliProcessor.from_pretrained(
            args.li_model_name, 
            cache_dir=args.cache_dir, 
            use_fast=True
        )
    )
    logger.info("Late Interaction model loaded successfully")


    # Get the collate function     
    collate_fn = get_collate("default")

    # Loading datasets 
    logger.info(f"Loading validation dataset...")
    val_dataloader = get_docvqa_answer_dataloader(
        cache_dir=args.cache_dir, 
        split="validation", 
        shuffle=False, 
        batch_size=args.batch_size, 
        num_workers=0, 
        collate_fn=collate_fn
    )
    logger.info("Validation dataloaders created successfully.")

    # Collect question heatmaps 
    logger.info(f"Collecting question heatmaps with {args.li_model_name}")
    q_heatmaps =  []
    n_examples = 0
    start = perf_counter()
    total_batches = len(val_dataloader)
    for step, batch in enumerate(val_dataloader):
        images = batch["images"]
        questions = batch["questions"]
        current_q_maps = get_question_heatmaps(li_model, li_processor, images, questions, device=device)
        q_heatmaps.extend(current_q_maps)
        n_examples += len(images)

        # Log progress
        if step % args.log_every == 0:
            elapsed = perf_counter() - start
            rate = (n_examples / elapsed) if elapsed > 0 else 0.0
            logger.info(f"  [{step}/{total_batches}] processed {n_examples} images, {rate:.2f} img/s")
                
    elapsed = perf_counter() - start
    elapsed_per_sample = elapsed / max(1, n_examples)
    logger.info(f"Question heatmaps collected. Total: {n_examples} | time: {elapsed:.2f}s | per example: {elapsed_per_sample:.4f}s")


    # Destroy ColPali model to free space 
    logger.info("Destroying ColPali Model")
    del li_model
    del li_processor
    torch.cuda.empty_cache()
    logger.info("Freed ColPali from VRAM.")


    # Load Pix2struct 
    logger.info(f"Loading pretrained Pix2Struct")
    pix2struct_processor = AutoProcessor.from_pretrained(args.model_save_dir, use_fast=True)
    pix2struct_model = AutoModelForVision2Seq.from_pretrained(args.model_save_dir)
    pix2struct_model.to(device)
    pix2struct_model.eval()
    logger.info("Pix2Struct loaded successfully")


    # Iterate through each variant 
    for v in MASK_VARIANTS: 
        # Log the variant to be tested 
        logger.info(f"Evaluating variant: {v}")

        # Reset all list of outputs 
        val_target_answers = []
        val_pred_answers   = []
        ptr = 0
        saved_samples = 0
        total_batches = len(val_dataloader)
        start = perf_counter()

        # Iterate over the validation dataset 
        for step, batch in enumerate(val_dataloader):
            # get inputs
            imgs = batch["images"]
            questions = batch["questions"]
            answers = batch["answers"]
            Ha_batch = batch["answer_masks"]

            # slice q_heatmaps for this batch
            B = len(imgs)
            qh_batch = q_heatmaps[ptr:ptr+B]
            ptr += B

            # binarize Hq per image
            Hq_bin = [binarize_heatmap(h) for h in qh_batch]

            # ensure Ha binary torch.uint8
            Ha_bin = []
            for a in Ha_batch:
                if a.dtype != torch.uint8:
                    a = a.to(torch.uint8)
                Ha_bin.append(a)

            # build masks for this variant
            Ms = []
            for ha, hq in zip(Ha_bin, Hq_bin):
                M = make_mask_variant(ha, hq, v)
                Ms.append(M)

            # apply mask + crop
            masked_crops = apply_mask_and_crop(imgs, Ms, crop_margin_frac=args.crop_margin, min_crop_h=args.min_crop_height)

            # Feed forward to the pix2struct 
            enc = pix2struct_processor(
                images=masked_crops,
                text=questions,
                return_tensors="pt",
                padding=True,
                max_patches=512
            )
            enc = {k: v.to(device) for k,v in enc.items()}
            pred_ids = pix2struct_model.generate(**enc, **gen_kwargs)
            preds = pix2struct_processor.batch_decode(pred_ids, skip_special_tokens=True)
            preds = [p.strip() for p in preds]

            val_target_answers.extend(answers)
            val_pred_answers.extend(preds)

            # Log status 
            if (step + 1) % args.log_every == 0 or step + 1 == total_batches:
                elapsed = perf_counter() - start
                rate = (n_examples / elapsed) if elapsed > 0 else 0.0
                logger.info(f"  [{step}/{total_batches}] batches generated")


        # Run evaluation for the variant 
        res = DocVQAEvaluator.evaluate(target_answers=val_target_answers, pred_answers=val_pred_answers)
        elapsed = perf_counter() - start
        logger.info(f"[{v}] ACC: {res['acc']:.4f} | ANLS: {res['anls']:.4f} | time: {elapsed:.2f}s")


        # Create dir for output if missing 
        variant_dir = os.path.join(args.image_save_dir, v)
        os.makedirs(variant_dir, exist_ok=True)

        # Log the dir where output will be saved
        logger.info(f"[{v}] Output samples saved to: {variant_dir}")

        # Make sure root dir is created 
        for i, (img, hq, ha, m, pred) in enumerate(zip(imgs, qh_batch, Ha_bin, Ms, preds)):
            if saved_samples >= args.num_samples:
                break
            masked_img = masked_crops[i]  
            save_example(
                out_root=variant_dir,
                variant=v,
                idx=saved_samples,
                img=img,
                Hq_heat=hq,
                Ha_mask=ha,
                M_variant=m,
                masked_crop=masked_img,
                question=questions[i],
                answers=answers[i],
                pred=pred
            )
            saved_samples += 1
            logger.info(f"  [{saved_samples}/{args.num_samples}] saved for variant {v}")

    logger.info("Script done.")