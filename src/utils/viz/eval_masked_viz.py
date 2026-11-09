from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Tuple
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch 

def ensure_dir(p: str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def masked_black(img: Image.Image, mask: torch.Tensor) -> Image.Image:
    # applies the mask to the image
    t = to_tensor(img)                            
    m = mask.to(dtype=torch.float32)              
    if m.ndim == 2:
        m = m.unsqueeze(0)                        
    m = m.clamp(0, 1)
    out = t * m                                  
    return to_pil_image(out)


def masked_black_bbox(img: Image.Image, bbox:List[int] | Tuple[int, int, int, int]) -> Image.Image:
    t = to_tensor(img)
    C, H, W = t.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2)); y2 = max(0, min(H,   y2))
    if x2 <= x1: x2 = min(W, x1+1)
    if y2 <= y1: y2 = min(H, y1+1)

    m = torch.zeros((H, W), dtype=t.dtype, device=t.device)
    m[y1:y2, x1:x2] = 1.0
    m = m.unsqueeze(0)          
    out = t * m                 
    return to_pil_image(out)



def tensor_mask_to_pil(mask: torch.Tensor) -> Image.Image:
    """mask: (H,W) {0,1} in any integer/bool dtype -> RGBA with alpha."""
    m = mask.to(torch.uint8).cpu().numpy() * 255
    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = m
    return Image.fromarray(rgba, mode="RGBA")


def heatmap_to_pil(heat: torch.Tensor) -> Image.Image:
    """
    heat: (H,W) float in [0,1] (any float dtype, incl. bfloat16) -> RGBA.
    """
    # ensure float32 before moving to numpy 
    h = heat.to(torch.float32).clamp(0, 1).cpu().numpy()

    # simple blue->cyan->yellow->red map
    r = np.clip(3*h - 1.5, 0, 1)
    g = np.clip(3*h - 0.5, 0, 1)
    b = np.clip(3*h + 0.5, 0, 1) - np.clip(3*h - 0.5, 0, 1)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    # alpha proportional to heat
    a = (h * 180).astype(np.uint8)
    rgba = np.concatenate([rgb, a[..., None]], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def save_example(
    out_root: str,
    variant: str,
    idx: int,
    img: Image.Image,
    Hq_heat: torch.Tensor,   
    Ha_mask: torch.Tensor,   
    M_variant: torch.Tensor,
    masked_crop: Image.Image,
    question: str,
    answers,                
    pred: str
):
    root = ensure_dir(out_root)
    var_dir = ensure_dir(root / variant)
    stem = f"ex_{idx:06d}"

    # Base image
    img_rgb = img if isinstance(img, Image.Image) else to_pil_image(img)

    # Convert Hq_heat into a binary mask (threshold or topk)
    hq_mask = (Hq_heat > 0.5).to(torch.uint8)

    # Render black-background versions
    vis_hq = masked_black(img_rgb, hq_mask)
    vis_ha = masked_black(img_rgb, Ha_mask)
    vis_mv = masked_black(img_rgb, M_variant)

    vis_hq.save(var_dir / f"{stem}_hq.png")
    vis_ha.save(var_dir / f"{stem}_ha.png")
    vis_mv.save(var_dir / f"{stem}_mask.png")
    masked_crop.save(var_dir / f"{stem}_crop.png")

    # save a small txt sidecar
    with open(var_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
        f.write(f"Q: {question}\n")
        f.write(f"A_gt: {answers}\n")
        f.write(f"A_pred: {pred}\n")



def cxcywh_norm_to_xyxy_pix(box_norm: List[float] | Tuple[float,float,float,float], W:int, H:int) -> Tuple[int,int,int,int]:
    cx, cy, w, h = box_norm
    bw, bh = w * W, h * H
    x1 = int(round(cx * W - bw / 2.0))
    y1 = int(round(cy * H - bh / 2.0))
    x2 = int(round(x1 + bw))
    y2 = int(round(y1 + bh))
    # clamp to image
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2)); y2 = max(0, min(H,   y2))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return x1, y1, x2, y2

def ha_qcrop_norm_to_full_xyxy(ha_norm: List[float], q_rect_full_xyxy: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    """Map Ha (normalized in Q-crop frame) back to FULL-image XYXY."""
    cx1, cy1, cx2, cy2 = q_rect_full_xyxy
    cropW, cropH = (cx2 - cx1), (cy2 - cy1)
    cx, cy, w, h = ha_norm
    bw, bh = w * cropW, h * cropH
    x1 = int(round(cx1 + cx * cropW - bw / 2.0))
    y1 = int(round(cy1 + cy * cropH - bh / 2.0))
    x2 = int(round(x1 + bw))
    y2 = int(round(y1 + bh))
    return x1, y1, x2, y2


def save_bbox_example(
    out_root: str | Path,
    epoch: int,
    idx: int,
    img: Image.Image,
    hq_bbox_norm_full: List[float],             
    ha_bbox_norm_qcrop: List[float],             
    q_rect_full_xyxy: Tuple[int,int,int,int],    
    question: str,
    answers,                                    
    pred: str
):
    """
    Saves:
      - original with both boxes overlaid
      - Hq focus (black background outside Hq)
      - Ha focus (black background outside Ha mapped to full frame)
      - small .txt sidecar with Q / GT / pred
    """
    root = ensure_dir(out_root)
    epoch_dir = ensure_dir(root / f"epoch_{epoch}")
    stem = f"ex_{idx:06d}"
    stem_path = epoch_dir / stem

    img_rgb = img if isinstance(img, Image.Image) else to_pil_image(img)
    W, H = img_rgb.size

    # compute pixel boxes
    hq_xyxy_full = cxcywh_norm_to_xyxy_pix(hq_bbox_norm_full, W, H)
    ha_xyxy_full = ha_qcrop_norm_to_full_xyxy(ha_bbox_norm_qcrop, q_rect_full_xyxy)

    # clamp Ha to image bounds
    x1,y1,x2,y2 = ha_xyxy_full
    x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2)); y2 = max(0, min(H,   y2))
    if x2 <= x1: x2 = min(W, x1+1)
    if y2 <= y1: y2 = min(H, y1+1)
    ha_xyxy_full = (x1,y1,x2,y2)

  
    # black-background focus variants
    vis_hq_focus = masked_black_bbox(img_rgb, hq_xyxy_full)
    vis_ha_focus = masked_black_bbox(img_rgb, ha_xyxy_full)

    # save images
    img_rgb.save(stem_path.with_suffix(".png"))
    vis_hq_focus.save(stem_path.with_name(f"{stem}_hq_focus").with_suffix(".png"))
    vis_ha_focus.save(stem_path.with_name(f"{stem}_ha_focus").with_suffix(".png"))

    # answers to be saved
    if isinstance(answers, (list, tuple)):
        gt_txt = " | ".join(map(str, answers))
    else:
        gt_txt = str(answers)

    # Save to the bbox 
    with open(stem_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(f"Q: {question}\n")
        f.write(f"A_gt: {gt_txt}\n")
        f.write(f"A_pred: {pred}\n")
        f.write(f"Hq(norm full): {list(map(lambda x: round(float(x),4), hq_bbox_norm_full))}\n")
        f.write(f"Ha(norm question_crop): {list(map(lambda x: round(float(x),4), ha_bbox_norm_qcrop))}\n")
        f.write(f"Q-crop full xyxy: {q_rect_full_xyxy}\n")
        f.write(f"Hq full xyxy: {hq_xyxy_full}\n")
        f.write(f"Ha full xyxy: {ha_xyxy_full}\n")
