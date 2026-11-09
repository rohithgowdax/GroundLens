import albumentations as A
import numpy as np
from PIL import Image

AUGS = {
    "none": A.Compose([], p=1.0),

    "photo_jitter": A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.7),
        A.RandomGamma(gamma_limit=(85, 115), p=0.5),
        A.ToGray(p=0.05),
    ], p=0.9), 

    "compress_sample": A.Compose([
        A.OneOf([
            A.NoOp(p=1.0),
            A.Downscale(scale_range=(0.85, 0.95), p=1.0), 
        ], p=0.6),
        A.OneOf([
            A.ImageCompression(quality_range=(60, 85), compression_type="jpeg", p=1.0),
            A.ImageCompression(quality_range=(65, 85), compression_type="webp", p=1.0),
            A.NoOp(p=1.0),
        ], p=0.8),  
    ], p=0.8),  

    "noise_blur": A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.NoOp(p=1.0),
        ], p=0.5), 
        A.GaussNoise(std_range=(0.015, 0.035), per_channel=False, p=0.5), 
    ], p=0.9),

    "artifacts": A.Compose([
        A.SomeOf([
            A.ToGray(p=1.0),
            A.CLAHE(clip_limit=(2.0, 3.0), tile_grid_size=(8, 8), p=1.0),
            A.Posterize(num_bits=(3, 4), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=False, elementwise=False, p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.ImageCompression(quality_range=(60, 80), compression_type="jpeg", p=1.0),
        ], n=3, replace=False, p=1.0),
    ], p=0.8),
}

ALLOWED_AUGS = ["none", "photo_jitter", "compress_sample", "noise_blur", "artifacts"]

def get_document_aug(mode: str = "none"):
    mode = mode.strip().lower()
    if mode not in ALLOWED_AUGS:
        raise ValueError(f"Augmentation for document not known: '{mode}'")
    return AUGS[mode]


def apply_doc_aug(img, aug):
    """Apply an albumentations augmentation to a PIL image and return a PIL image."""

    # Turn into numpy array
    arr = np.array(img)

    # Guard against non-uint8 inputs
    if arr.dtype not in (np.uint8,):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = aug(image=arr)["image"]   

    # Return the image
    return Image.fromarray(out).convert("RGB")