import pytest
import torch
import numpy as np
from argparse import Namespace

# Import model-related classes for fixtures
from src.model.model import CoExVQA
from src.model.config.model_config import CoExVQAConfig
from src.model.projectors.answer_projector import AnswerHeatmapProjectorConfig
from src.model.projectors.question_projector import QuestionHeatmapProjectorConfig
from src.model.gating.config import GatingConfig

# Test fixtures that are available for all files

@pytest.fixture
def device():
    """Provide device for testing"""
    return torch.device("cpu")

@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed = 2026
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def d_model():
    """Standard embedding dimension."""
    return 768


@pytest.fixture
def seq_len():
    """Standard sequence length."""
    return 256


@pytest.fixture
def sample_boxes_cxcywh_norm():
    """Sample normalized bounding boxes in cxcywh format."""
    # [cx, cy, w, h] in [0, 1] range
    return torch.tensor([
        [0.5, 0.5, 0.2, 0.3],   # Center box
        [0.25, 0.25, 0.1, 0.1], # Top-left small box
        [0.75, 0.75, 0.3, 0.4], # Bottom-right large box
        [0.9, 0.1, 0.05, 0.05], # Corner box
    ])


@pytest.fixture
def sample_boxes_xyxy_abs():
    """Sample absolute bounding boxes in xyxy format."""
    # Assuming 1000x1000 image
    return torch.tensor([
        [200, 150, 600, 450],   # Large center box
        [100, 100, 300, 250],   # Top-left box
        [500, 500, 900, 900],   # Bottom-right box
        [50, 50, 150, 100],     # Small corner box
    ])


@pytest.fixture
def sample_image_sizes():
    """Sample image sizes (W, H)."""
    return torch.tensor([
        [1000, 1000],
        [800, 600],
        [1200, 900],
        [640, 480],
    ])


@pytest.fixture
def sample_embeddings(batch_size, seq_len, d_model):
    """Sample image embeddings."""
    return torch.randn(batch_size, seq_len, d_model)


@pytest.fixture
def sample_question_embeddings(batch_size, d_model):
    """Sample pooled question embeddings."""
    return torch.randn(batch_size, d_model)


@pytest.fixture
def sample_question_tokens(batch_size, d_model):
    """Sample question token embeddings."""
    seq_len_q = 32
    return torch.randn(batch_size, seq_len_q, d_model)


# Fixtures for localization evaluator tests

@pytest.fixture
def perfect_match_boxes():
    """Perfect match: GT and pred boxes are identical."""
    gt_boxes = [torch.tensor([[0.3, 0.3, 0.7, 0.7]])]  # xyxy normalized
    pred_boxes = [torch.tensor([[0.5, 0.5, 0.4, 0.4]])]  # cxcywh normalized (same box)
    img_sizes = [torch.tensor([[1000, 1000]])]
    return gt_boxes, pred_boxes, img_sizes


@pytest.fixture
def no_overlap_boxes():
    """No overlap: GT and pred boxes don't intersect."""
    gt_boxes = [torch.tensor([[0.1, 0.1, 0.3, 0.3]])]  # Top-left
    pred_boxes = [torch.tensor([[0.75, 0.75, 0.2, 0.2]])]  # Bottom-right
    img_sizes = [torch.tensor([[1000, 1000]])]
    return gt_boxes, pred_boxes, img_sizes


@pytest.fixture
def partial_overlap_boxes():
    """Partial overlap: GT and pred boxes partially intersect."""
    gt_boxes = [torch.tensor([[0.2, 0.2, 0.6, 0.6]])]  # xyxy
    pred_boxes = [torch.tensor([[0.5, 0.5, 0.4, 0.4]])]  # cxcywh (0.3, 0.3, 0.7, 0.7)
    img_sizes = [torch.tensor([[1000, 1000]])]
    return gt_boxes, pred_boxes, img_sizes


@pytest.fixture
def batch_boxes():
    """Batch of boxes for testing batch processing."""
    gt_boxes = [
        torch.tensor([[0.2, 0.2, 0.6, 0.6], [0.1, 0.1, 0.3, 0.3]]),
        torch.tensor([[0.4, 0.4, 0.8, 0.8]]),
    ]
    pred_boxes = [
        torch.tensor([[0.4, 0.4, 0.4, 0.4], [0.2, 0.2, 0.2, 0.2]]),
        torch.tensor([[0.6, 0.6, 0.4, 0.4]]),
    ]
    img_sizes = [
        torch.tensor([[1000, 1000], [800, 600]]),
        torch.tensor([[1200, 900]]),
    ]
    return gt_boxes, pred_boxes, img_sizes


# Fixtures for model checkpoint tests

@pytest.fixture
def model_config():
    """Create a basic model config for testing."""
    return CoExVQAConfig(
        d_model=768,
        backbone_model="google/pix2struct-docvqa-base",
        backbone_freeze_mode="frozen",
        answer_projector_config=AnswerHeatmapProjectorConfig(d_model=768),
        question_projector_config=QuestionHeatmapProjectorConfig(d_model=768),
        gate_config=GatingConfig(gating_type="linear_interpolation", initial_alpha=0.8),
        max_patches=512,
    )


@pytest.fixture
def sample_model(model_config):
    """Create a sample model for testing."""
    return CoExVQA(config=model_config)


@pytest.fixture
def sample_args():
    """Create sample training arguments."""
    return Namespace(
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=10,
        seed=42,
    )
