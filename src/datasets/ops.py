import numpy as np
from typing import List, Optional, Tuple
import re
from PIL import Image
import torch
from torch.nn import functional as F



from src.datasets.DocVQA import OcrLine


def bbox_to_int(bbox:List[int], W: int, H: int):
    """
    Scales the bounding box to pixel level cords and return them.
    Takes the width and height of the image to scale them.

    Args:
        bbox (List[int]): the relative bounding box coordinates
        W (int): width of the image
        H (int): height of the image

    Returns:
        x1, y1, x2, y2 coordinates of the scaled bounding box.
    """
    x1, y1, x2, y2 = bbox

    x1 = int(x1 * W)
    x2 = int(x2 * W)
    y1 = int(y1 * H)
    y2 = int(y2 * H)

    return x1, y1, x2, y2



def clamp_bbox(x1, y1, x2, y2, W, H):
    # sort, clamp, and ensure at least 1px area
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(x1, W - 1))
    x2 = max(1, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(1, min(y2, H))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2

def crop_with_margin(img, x1, y1, x2, y2, crop_margin=0.10):
    W, H = img.size
    dx, dy = x2 - x1, y2 - y1
    mx, my = int(crop_margin * dx), int(crop_margin * dy)
    cx1 = max(0, x1 - mx); cy1 = max(0, y1 - my)
    cx2 = min(W, x2 + mx); cy2 = min(H, y2 + my)
    return img.crop((cx1, cy1, cx2, cy2))

def resize_for_readability(img, max_width=700, min_height=256):
    """Resize a bounding-box crop, upscaling if too short and capping width — unlike resize_img, handles arbitrarily small patches."""
    w, h = img.size

    # First, ensure minimum height (upscale if needed)
    scale = max(1.0, min_height / max(1, h))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # Then, cap width if it grew too large
    if new_w > max_width:
        scale2 = max_width / new_w
        new_w = int(round(new_w * scale2))
        new_h = int(round(new_h * scale2))
    if new_w < 1: new_w = 1
    if new_h < 1: new_h = 1
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def resize_img(image, max_width=700):
    """Downscale a full-page image to a fixed width, assuming it is already large enough to be legible."""
    # resize 
    w, h = image.size 
    new_h = int(h * (max_width / w))
    image = image.resize((max_width, new_h), Image.Resampling.LANCZOS)

    # Return the image
    return image


def xyxy_to_cxcywh_norm(x1:int, y1:int, x2:int, y2:int, W:int, H:int) -> Tuple[float,float,float,float]:
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    cx = x1 + w / 2.0; cy = y1 + h / 2.0
    return cx / W, cy / H, w / W, h / H

def normalize_answers(ans_list):
    """
    Process answers for both training and evaluation.

    Returns:
        tuple: (first_answers, all_answers) where:
            - first_answers: List[str] - first answer variant (for training/tokenization)
            - all_answers: List[List[str]] - all answer variants (for evaluation)
    """
    first_answers = []
    all_answers = []

    for a in ans_list:
        if isinstance(a, (list, tuple)):
            # First answer for training
            first_answers.append(str(a[0]) if len(a) > 0 else "")
            # All answers for evaluation
            all_answers.append([str(variant) for variant in a] if len(a) > 0 else [""])
        else:
            # Single answer
            first_answers.append(str(a))
            all_answers.append([str(a)])

    return first_answers, all_answers