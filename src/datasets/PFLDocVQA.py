import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
import numpy as np
from PIL import Image, ImageDraw


from src.datasets.ops import bbox_to_int


@dataclass
class PFLDocumentData:
    """Data structure for a single PFL-DocVQA sample."""
    document_id: str
    image_path: str
    question: str
    question_id: str
    answers: List[str]
    provider_id: str  # Federated learning client/provider identifier
    words: List[str]  # OCR words
    word_boxes: List[List[int]]  # Bounding boxes for each word [x0, y0, x1, y1]
    answer_bbox: Optional[List[float]] = field(default=None)  # Normalized [x1, y1, x2, y2] in [0,1]
    q_prior_patch: Optional[List[float]] = field(default=None)
    q_prior_patch_shape: Optional[List[int]] = field(default=None)


def _find_span(target: str, norm_words: List[str], word_boxes: List[List[float]]) -> Optional[List[List[float]]]:
    """Find shortest consecutive OCR token span matching target (normalized)."""
    best = None
    for start in range(len(norm_words)):
        concat = ""
        for end in range(start, min(start + 40, len(norm_words))):
            concat += norm_words[end]
            if concat == target:
                span = word_boxes[start:end + 1]
                if best is None or len(span) < len(best):
                    best = span
                break
            if len(concat) > len(target):
                break
    return best


def _compute_answer_bbox(answers: List[str], words: List[str], word_boxes: List[List[float]]) -> Optional[List[float]]:
    """Find consecutive OCR tokens forming the answer and return merged bbox.

    For multi-line answers (containing \\n), each line is matched separately and boxes are merged. 
    Returns normalized [x1, y1, x2, y2] in [0,1] with 10%/15% expansion, or None.
    """
    _norm = lambda s: re.sub(r"\W+", "", s.lower())
    norm_words = [_norm(w) for w in words]

    all_matched = []
    for ans in answers:
        # Split multi-line answers and match each line
        lines = [l.strip() for l in ans.split("\n") if l.strip()]
        for line in lines:
            target = _norm(line)
            if not target:
                continue
            span = _find_span(target, norm_words, word_boxes)
            if span:
                all_matched.extend(span)

        # If we matched at least one line, use these boxes
        if all_matched:
            break
    
    if not all_matched:
        return None
    xs = [b[i] for b in all_matched for i in (0, 2)]
    ys = [b[i] for b in all_matched for i in (1, 3)]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    # Expand by 10% width, 15% height (same as DocVQAPipeline)
    w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    x1 = max(0.0, x1 - w * 0.10)
    y1 = max(0.0, y1 - h * 0.15)
    x2 = min(1.0, x2 + w * 0.10)
    y2 = min(1.0, y2 + h * 0.15)
    return [x1, y1, x2, y2]


class PFLDocVQADataset(torch.utils.data.Dataset):
    """
    PFL-DocVQA Dataset for Privacy-preserving Federated Learning.

    The dataset needs to be downloaded and organized in the following structure:
    - data_dir/ (top level directory)
    - data_dir/images/ (contains all document images)
    - data_dir/imdb_val.npy (validation annotations in .npy format)

    Args:
        data_dir: Path to the PFL-DocVQA data directory
        split: Dataset split ('train', 'val', or 'test')
        verbose: Whether to print loading progress
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        verbose: bool = False,
        require_q_priors: bool = True,
        subsample_frac: Optional[float] = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.verbose = verbose
        self.q_prior_dir = os.path.join(data_dir, "q_priors", split)

        # Load annotations
        self.samples = self._load_annotations()

        # Drop samples whose q_prior file is missing or has invalid content
        if require_q_priors and os.path.isdir(self.q_prior_dir):
            before = len(self.samples)
            valid_samples = []
            for s in self.samples:
                prior_path = os.path.join(self.q_prior_dir, f"{s.question_id}.pt")
                if os.path.exists(prior_path):
                    try:
                        prior_data = torch.load(prior_path, weights_only=True)
                        if prior_data.get("q_prior_patch") is not None:
                            valid_samples.append(s)
                    except Exception:
                        pass
            self.samples = valid_samples
            dropped = before - len(self.samples)
            if dropped and verbose:
                print(f"Warning: dropped {dropped} samples with missing/invalid q_prior files")

        if subsample_frac is not None and split == "train":
            k = max(1, int(len(self.samples) * subsample_frac))
            self.samples = random.Random(42).sample(self.samples, k)

        if verbose:
            print(f"Loaded {len(self.samples)} samples for split '{split}'")

    def _load_annotations(self) -> List[PFLDocumentData]:
        """Load annotations from the .npy file and create PFLDocumentData samples."""

        if self.split == "val":
            # Load validation annotations
            annotation_file_path = os.path.join(self.data_dir, "imdb_val.npy")
            annotation_file = np.load(annotation_file_path, allow_pickle=True)


            # Iterate through annotation entries and create PFLDocumentData samples
            samples = []
            for entry in annotation_file[1:]:
                
                # Get answers 
                answers = entry.get('answers', [])
                if isinstance(answers, str):
                    answers = [answers]

                # Assert that there is at least one answer for the question 
                assert len(answers) > 0, f"No answers found for question_id {entry.get('question_id', 'unknown')}"
                
                # Create PFLDocumentData sample
                words = entry.get('ocr_tokens', [])
                word_boxes = entry.get('ocr_normalized_boxes', [])
                sample = PFLDocumentData(
                    document_id=entry.get('image_name', ''),
                    image_path=os.path.join(self.data_dir, 'images', entry.get('image_name', '') + '.jpg'),
                    question=entry.get('question', ''),
                    question_id=hashlib.sha256(f"{entry.get('image_name', '')}|{entry.get('question', '')}".encode()).hexdigest()[:16],
                    answers=answers,
                    provider_id=entry.get('set_name', 'unknown'),
                    words=words,
                    word_boxes=word_boxes,
                    answer_bbox=_compute_answer_bbox(answers, words, word_boxes),
                )
                samples.append(sample)

            # Return the list of samples for the validation split
            return samples
        elif self.split == "train":
            # Load all annotaitons files for all clients
            clients = 10 

            # Iterate through clients and load their respective annotation files
            samples = []
            for client_id in range(clients):
                # Get annotation file path for the current client and load it
                annotation_file_path = os.path.join(self.data_dir, f"imdb_train_client_{client_id}.npy")
                annotation_file = np.load(annotation_file_path, allow_pickle=True)

                if self.verbose:
                    print(f"Loaded {len(annotation_file) - 1} samples for client {client_id} from {annotation_file_path}")

                for entry in annotation_file[1:]:
                    # Get answers 
                    answers = entry.get('answers', [])
                    if isinstance(answers, str):
                        answers = [answers]

                    # Assert that there is at least one answer for the question 
                    assert len(answers) > 0, f"No answers found for question_id {entry.get('question_id', 'unknown')}"

                    # Create PFLDocumentData sample
                    words = entry.get('ocr_tokens', [])
                    word_boxes = entry.get('ocr_normalized_boxes', [])
                    sample = PFLDocumentData(
                        document_id=entry.get('image_name', ''),
                        image_path=os.path.join(self.data_dir, 'images', entry.get('image_name', '') + '.jpg'),
                        question=entry.get('question', ''),
                        question_id=hashlib.sha256(f"{entry.get('image_name', '')}|{entry.get('question', '')}".encode()).hexdigest()[:16],
                        answers=answers,
                        provider_id=f"client{client_id}",
                        words=words,
                        word_boxes=word_boxes,
                        answer_bbox=_compute_answer_bbox(answers, words, word_boxes),
                    )
                    samples.append(sample)

            # Return the list of samples for the training split
            return samples
        else:
            raise ValueError(f"Unsupported split: {self.split}")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PFLDocumentData:
        sample = self.samples[idx]
        if sample.q_prior_patch is None:
            prior_path = os.path.join(self.q_prior_dir, f"{sample.question_id}.pt")
            if os.path.exists(prior_path):
                prior_data = torch.load(prior_path, weights_only=True)
                sample.q_prior_patch = prior_data["q_prior_patch"]
                sample.q_prior_patch_shape = prior_data["q_prior_patch_shape"]
        return sample

    def get_provider_ids(self) -> List[str]:
        """Get list of unique provider IDs in the dataset."""
        return list(set(s.provider_id for s in self.samples))

    def get_samples_by_provider(self, provider_id: str) -> List[PFLDocumentData]:
        """Get all samples belonging to a specific provider."""
        return [s for s in self.samples if s.provider_id == provider_id]

    def get_provider_sample_counts(self) -> Dict[str, int]:
        """Get sample counts per provider (useful for FL weight calculation)."""
        counts = {}
        for sample in self.samples:
            counts[sample.provider_id] = counts.get(sample.provider_id, 0) + 1
        return counts



def get_pfl_docvqa_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    collate_fn,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    verbose: bool = False,
    subsample_frac: Optional[float] = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the PFL-DocVQA dataset.

    Args:
        data_dir: Path to PFL-DocVQA data directory
        split: Dataset split ('train' or 'val')
        batch_size: Batch size
        collate_fn: Collate function for batching
        shuffle: Whether to shuffle the data
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        verbose: Whether to print loading progress
        subsample_frac: If set, randomly sample this fraction of training examples (ignored for val)

    Returns:
        DataLoader for PFL-DocVQA dataset
    """
    dataset = PFLDocVQADataset(data_dir=data_dir, split=split, verbose=verbose, subsample_frac=subsample_frac)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )


def get_val_pfl_docvqa_dataloader(
    data_dir: str,
    batch_size: int,
    collate_fn,
    num_workers: int = 4,
    pin_memory: bool = True,
    verbose: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the PFL-DocVQA validation set.

    Args:
        data_dir: Path to PFL-DocVQA data directory
        batch_size: Batch size
        collate_fn: Collate function for batching
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory
        verbose: Whether to print loading progress

    Returns:
        DataLoader for PFL-DocVQA validation set
    """
    return get_pfl_docvqa_dataloader(
        data_dir=data_dir,
        split="val",
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        verbose=verbose,
    )


# Test the dataset loading and sample retrieval
if __name__ == "__main__":
    # Example usage
    dataset = PFLDocVQADataset(data_dir="/Users/kjetil/Documents/datasets/PFL-DocVQA/", split="val", verbose=True)
    print(f"Total samples: {len(dataset)}")

    # Show a few samples and save visualizations with answer bounding boxes
    save_dir = "/Users/kjetil/Documents/datasets/PFL-DocVQA//answer_bbox_examples"
    os.makedirs(save_dir, exist_ok=True)

    # Get 5 examples 
    for i in [10, 50, 100, 200, 500]:
        sample = dataset[i]
        print(f"\n  [{i}] question: {sample.question}")
        print(f"       answers: {sample.answers}")

        # Get bbox 
        bbox = sample.answer_bbox
        print(f"       answer_bbox: {bbox}")

        # Save visualization of the sample with answer bbox if available
        if bbox is not None:
            img = Image.open(sample.image_path).convert("RGB")
            W, H = img.size
            x1, y1, x2, y2 = bbox_to_int(bbox, W, H)
            draw = ImageDraw.Draw(img)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            out_path = os.path.join(save_dir, f"sample_{i}.png")
            img.save(out_path)
            print(f"       saved: {out_path}")