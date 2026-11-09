import os
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id):
    """Set HuggingFace Hub timeout in worker processes."""
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

def has_valid_bbox(x):
    bbox = x.get("answer_ocr_bbox", None)
    if bbox is None:
        return False
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    
    # check if normalized values
    if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return True

class DocVqaAnswerDataset(Dataset):
    def __init__(self, ds=None, cache_dir=None, split="train", ds_name="indrehus/docvqa-single-page-questions-answer-ocr"):
        # Load dataset
        self.ds_name = ds_name
        self.data = ds if ds is not None else load_dataset(
            self.ds_name, split=split, cache_dir=cache_dir
        )

        msg = f"Loaded dataset '{self.ds_name}' '{split}' split with {len(self.data)} samples."
        logger.info(msg)

        # Remove samples with no answer bboxes
        # Only for training and validation splits
        if split in ["train", "validation"]:
            self.data = self.data.filter(has_valid_bbox)
            msg = f"Filtered dataset to {len(self.data)} samples with answer bounding boxes."
            logger.info(msg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_docvqa_answer_dataloader(
    ds=None,
    cache_dir=None,
    split="train",
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=None,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    rank=None,
    world_size=None,
    ds_name="indrehus/docvqa-single-page-questions-answer-ocr",
):
    dataset = DocVqaAnswerDataset(ds, cache_dir, split, ds_name=ds_name)

    # Setup distributed sampler if world_size > 1
    sampler = None
    if world_size is not None and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=(split == "train")
        )
        # When using DistributedSampler, shuffle must be False in DataLoader
        shuffle = False

    # Build kwargs
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )

    if num_workers and num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)

        # PyTorch expects an int >= 2 for prefetch_factor
        if isinstance(prefetch_factor, int) and prefetch_factor >= 2:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


def get_train_val_docvqa_answer_dataloaders(
    ds_name:str = "indrehus/docvqa-single-page-questions-answer-ocr",
    cache_dir=None,
    batch_size=1,
    num_workers=0,
    collate_fn=None,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    rank=None,
    world_size=None,
):
    ds = load_dataset(ds_name, cache_dir=cache_dir)

    train_dataloader = get_docvqa_answer_dataloader(
        ds=ds["train"],
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        rank=rank,
        world_size=world_size,
        ds_name=ds_name,
    )

    val_dataloader = get_docvqa_answer_dataloader(
        ds=ds["validation"],
        split="validation",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        rank=rank,
        world_size=world_size,
        ds_name=ds_name,
    )

    return train_dataloader, val_dataloader


def get_val_docvqa_answer_dataloader(
    ds_name:str = "indrehus/docvqa-single-page-questions-answer-ocr",
    cache_dir=None,
    batch_size=1,
    num_workers=0,
    collate_fn=None,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    rank=None,
    world_size=None,
):
    ds = load_dataset(ds_name, cache_dir=cache_dir)

    val_dataloader = get_docvqa_answer_dataloader(
        ds=ds["validation"],
        split="validation",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        rank=rank,
        world_size=world_size,
    )

    return val_dataloader