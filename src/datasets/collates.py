from typing import Callable, Dict, Tuple
import logging
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from functools import partial
import inspect
from PIL import Image

logger = logging.getLogger(__name__)



# import operations for the collates 
from src.datasets.ops import bbox_to_int, clamp_bbox, resize_for_readability, crop_with_margin, resize_img, xyxy_to_cxcywh_norm
from src.utils.augmentations import get_document_aug, apply_doc_aug

def default_collate_fn(batch):
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]
    answer_bboxes = [item["answer_ocr_bbox"] for item in batch]

    answer_masks = []
    for img, bbox in zip(images, answer_bboxes):
        W, H = img.size
        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
        m = torch.zeros((H, W), dtype=torch.uint8)
        m[y1:y2, x1:x2] = 1
        answer_masks.append(m)

    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "answer_masks": answer_masks,  
    }

def answer_bbox_collate_fn(batch):
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]
    answer_bboxes = [item["answer_ocr_bbox"] for item in batch]

    full_img_sizes = []
    gt_box_pix = []
    gt_box_norm = []

    for img, bbox in zip(images, answer_bboxes):
        W, H = img.size
        full_img_sizes.append((W, H))
        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
        gt_box_pix.append((x1, y1, x2, y2))
        cx, cy, w, h = xyxy_to_cxcywh_norm(x1, y1, x2, y2, W, H)
        gt_box_norm.append((cx, cy, w, h))

    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "answer_bbox_xyxy": gt_box_pix,         
        "answer_bbox_norm_cxcywh": torch.tensor(gt_box_norm, dtype=torch.float32), 
        "full_img_sizes": torch.tensor(full_img_sizes, dtype=torch.long),   
    }

def answer_bbox_width_collate_fn(batch, max_width:int=800, augmentation:str = "none"):
    # Get from batch 
    images = [resize_img(image=item["image"], max_width=max_width).convert("RGB") for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]
    answer_bboxes = [item["answer_ocr_bbox"] for item in batch]

    # Apply augmentation
    if augmentation and augmentation.strip().lower() != "none":
        aug = get_document_aug(mode=augmentation)
        images = [apply_doc_aug(img, aug) for img in images]

    full_img_sizes = []
    gt_box_pix = []
    gt_box_norm = []

    for img, bbox in zip(images, answer_bboxes):
        W, H = img.size
        full_img_sizes.append((W, H))
        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
        gt_box_pix.append((x1, y1, x2, y2))
        cx, cy, w, h = xyxy_to_cxcywh_norm(x1, y1, x2, y2, W, H)
        gt_box_norm.append((cx, cy, w, h))

    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "answer_bbox_xyxy": gt_box_pix,         
        "answer_bbox_norm_cxcywh": torch.tensor(gt_box_norm, dtype=torch.float32), 
        "full_img_sizes": torch.tensor(full_img_sizes, dtype=torch.long),   
    }



def masked_answer_image_collate_fn(batch):
    # Get the relevant fields from the batch
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]
    answer_bboxes = [item["answer_ocr_bbox"] for item in batch]


    # Collect masked images  
    masked_images = []
    for img, bbox in zip(images, answer_bboxes):
        # Get the width and height, then turn the image into a tensor 
        W, H = img.size 
        img_t = TF.to_tensor(img)

        # Get coords that are normalized 
        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)

        # create the binary mask for the bbox
        mask = torch.zeros((H, W), dtype=img_t.dtype, device=img_t.device)
        mask[y1:y2, x1:x2] = 1.0

        # apply the mask to the image 
        masked = img_t * mask.unsqueeze(0) 

        # turn image back to rbg pil image
        pil_image = to_pil_image(masked).convert("RGB")

        # resize with respect to the image aspect ratio
        pil_image = resize_img(pil_image)

        # Stack
        masked_images.append(pil_image)

    return {
        "masked_image": masked_images,
        "questions": questions,
        "answers": answers,
    }


def cropped_answer_image_collate_fn(batch, max_width=700, min_height=256, crop_margin=0.10):
    # Get items from batch
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]
    answer_bboxes = [item["answer_ocr_bbox"] for item in batch]

    # Collect the cropped images
    cropped_images = []
    for img, bbox in zip(images, answer_bboxes):
        W, H = img.size
        # normalized bbox -> pixel coords
        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)

        # crop with margin
        crop = crop_with_margin(img, x1, y1, x2, y2, crop_margin=crop_margin)

        # edge case: if something went wrong and crop is empty, fall back to original
        if crop.size[0] <= 1 or crop.size[1] <= 1:
            crop = img

        # Resize for readability
        crop = resize_for_readability(crop, max_width=max_width, min_height=min_height)

        # Add the cropped image to the list
        cropped_images.append(crop.convert("RGB"))

    return {
        "masked_image": cropped_images,
        "questions": questions,
        "answers": answers,
    }

def answer_bbox_with_q_prior_collate_fn(batch, max_width:int=800, augmentation:str = "none"):
    # Get from batch and ensure RGB format
    images = [resize_img(image=item["image"], max_width=max_width).convert("RGB") for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answers"] for item in batch]
    answer_bboxes = [item["answer_ocr_bbox"] for item in batch]

    # Apply augmentation
    if augmentation and augmentation.strip().lower() != "none":
        aug = get_document_aug(mode=augmentation)
        images = [apply_doc_aug(img, aug) for img in images]

    full_img_sizes = []
    gt_box_pix = []
    gt_box_norm = []

    for img, bbox in zip(images, answer_bboxes):
        W, H = img.size
        full_img_sizes.append((W, H))
        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
        gt_box_pix.append((x1, y1, x2, y2))
        cx, cy, w, h = xyxy_to_cxcywh_norm(x1, y1, x2, y2, W, H)
        gt_box_norm.append((cx, cy, w, h))

    # Get q_prior patch and shape information
    q_prior_patches = []
    q_prior_shapes = []

    for item in batch:
        # Get the flattened patch prior and reshape info
        patch_flat = item["q_prior_patch"]
        patch_shape = item["q_prior_patch_shape"]  # [height, width]

        q_prior_patches.append(torch.tensor(patch_flat, dtype=torch.float32))
        q_prior_shapes.append(torch.tensor(patch_shape, dtype=torch.long))

    # Pad q_prior_patches to the same length
    max_patches = max(p.shape[0] for p in q_prior_patches)
    padded_patches = []
    q_prior_masks = []

    for p in q_prior_patches:
        num_patches = p.shape[0]
        # Create mask: 1 for valid patches, 0 for padding
        mask = torch.ones(max_patches, dtype=torch.bool)
        mask[num_patches:] = False
        q_prior_masks.append(mask)

        # Pad with zeros
        if num_patches < max_patches:
            padding = torch.zeros(max_patches - num_patches, dtype=torch.float32)
            p = torch.cat([p, padding], dim=0)
        padded_patches.append(p)

    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "answer_bbox_xyxy": gt_box_pix,
        "answer_bbox_norm_cxcywh": torch.tensor(gt_box_norm, dtype=torch.float32),
        "full_img_sizes": torch.tensor(full_img_sizes, dtype=torch.long),
        "q_prior_patch": torch.stack(padded_patches),  # [batch, max_patches]
        "q_prior_patch_shape": torch.stack(q_prior_shapes),  # [batch, 2]
        "q_prior_mask": torch.stack(q_prior_masks),  # [batch, max_patches] - True for valid, False for padding
    }

def pfl_docvqa_collate_fn(batch, max_width: int = 800):
    """
    Collate function for PFL-DocVQA dataset.

    Loads images from file paths and prepares batch for model inference.

    Args:
        batch: List of PFLDocumentData samples
        max_width: Maximum width for image resizing

    Returns:
        dict with images, questions, answers
    """

    images = []
    questions = []
    answers = []

    for sample in batch:
        # Load and resize image
        try:
            img = Image.open(sample.image_path).convert("RGB")
        except FileNotFoundError:
            logger.warning(f"Missing image file, skipping sample: {sample.image_path}")
            continue
        img = resize_img(image=img, max_width=max_width)

        # Append to lists
        images.append(img)
        questions.append(sample.question)
        answers.append(sample.answers)

    return {
        "images": images,
        "questions": questions,
        "answers": answers,
    }


def pfl_docvqa_with_q_prior_collate_fn(batch, max_width: int = 800):
    """
    Collate function for PFL-DocVQA dataset with question priors and answer bboxes.
    Compatible with train.py training loop.
    """
    images = []
    questions = []
    answers = []
    q_prior_patches = []
    q_prior_shapes = []

    full_img_sizes = []
    gt_box_pix = []
    gt_box_norm = []

    for sample in batch:
        try:
            img = Image.open(sample.image_path).convert("RGB")
        except FileNotFoundError:
            logger.warning(f"Missing image file, skipping sample: {sample.image_path}")
            continue
        img = resize_img(image=img, max_width=max_width)
        images.append(img)
        questions.append(sample.question)
        answers.append(sample.answers)

        # Answer bbox: use sample bbox or fall back to full image
        W, H = img.size
        full_img_sizes.append((W, H))
        bbox = sample.answer_bbox if sample.answer_bbox is not None else [0, 0, 1, 1]
        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
        gt_box_pix.append((x1, y1, x2, y2))
        cx, cy, w, h = xyxy_to_cxcywh_norm(x1, y1, x2, y2, W, H)
        gt_box_norm.append((cx, cy, w, h))

        q_prior_patches.append(torch.tensor(sample.q_prior_patch, dtype=torch.float32))
        q_prior_shapes.append(torch.tensor(sample.q_prior_patch_shape, dtype=torch.long))

    # Pad q_prior_patches to the same length
    max_patches = max(p.shape[0] for p in q_prior_patches)
    padded_patches = []
    q_prior_masks = []

    for p in q_prior_patches:
        num_patches = p.shape[0]
        mask = torch.ones(max_patches, dtype=torch.bool)
        mask[num_patches:] = False
        q_prior_masks.append(mask)

        if num_patches < max_patches:
            padding = torch.zeros(max_patches - num_patches, dtype=torch.float32)
            p = torch.cat([p, padding], dim=0)
        padded_patches.append(p)

    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "answer_bbox_xyxy": gt_box_pix,
        "answer_bbox_norm_cxcywh": torch.tensor(gt_box_norm, dtype=torch.float32),
        "full_img_sizes": torch.tensor(full_img_sizes, dtype=torch.long),
        "q_prior_patch": torch.stack(padded_patches),
        "q_prior_patch_shape": torch.stack(q_prior_shapes),
        "q_prior_mask": torch.stack(q_prior_masks),
    }


# List of the defined collate functions
COLLATE_REGISTRY: Dict[str, Callable] = {
    "default": default_collate_fn,
    "answer_bbox": answer_bbox_collate_fn,
    "answer_resize_bbox": answer_bbox_width_collate_fn,
    "masked_answer_images": masked_answer_image_collate_fn,
    "cropped_answer_images": cropped_answer_image_collate_fn,
    "answer_bbox_with_q_prior": answer_bbox_with_q_prior_collate_fn,
    "pfl_docvqa": pfl_docvqa_collate_fn,
    "pfl_docvqa_with_q_prior": pfl_docvqa_with_q_prior_collate_fn,
}


def get_collate(name: str, **kwargs) -> Callable:
    """Get a collate function by name, with optional kwargs for configuration."""
    if name not in COLLATE_REGISTRY:
        raise ValueError(f"Unknown collate '{name}'. Available: {list(COLLATE_REGISTRY)}")
    
    # Get the function by name
    fn = COLLATE_REGISTRY[name]

    # Get all accepted args to not crash 
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in kwargs.items()
                if k in sig.parameters and v is not None}
    
    # Return the function that only accepts the batch  
    return partial(fn, **accepted)
