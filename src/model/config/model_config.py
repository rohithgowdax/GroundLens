from dataclasses import dataclass, field
from typing import Optional, Literal

from src.model.projectors.answer_projector import AnswerHeatmapProjectorConfig
from src.model.projectors.question_projector import QuestionHeatmapProjectorConfig
from src.model.gating.config import GatingConfig


@dataclass
class CoExVQAConfig():
    """
    Configuration for the CoExVQA model.

    Args:
        d_model: Dimension of the model embeddings
        backbone_model: HuggingFace model ID for the backbone (pix2struct)
        backbone_local_path: Optional local path to load backbone from
        answer_projector_config: Configuration for the answer heatmap projector
        question_projector_config: Configuration for the question heatmap projector
        gate_config: Configuration for the gating mechanism
        max_patches: Maximum number of image patches
        backbone_freeze_mode: How to freeze the backbone ("frozen", "unfrozen", or "decoder-unfrozen")
        use_decoder: Whether to use the decoder for generation
        decoder_input_type: Type of input to decoder ("gated" or "raw")
        decoder_attention_type: Type of attention masking for decoder
        bbox_mask_blur_sigma: Gaussian blur sigma for bbox masking
    """
    d_model: Optional[int] = None
    backbone_model: str = "google/pix2struct-docvqa-base"
    backbone_local_path: Optional[str] = None
    answer_projector_config: Optional[AnswerHeatmapProjectorConfig] = None
    question_projector_config: Optional[QuestionHeatmapProjectorConfig] = None
    gate_config: GatingConfig = field(default_factory=lambda: GatingConfig(gating_type="linear_interpolation", initial_alpha=0.8))
    max_patches: int = 512

    # Backbone freeze options
    backbone_freeze_mode: Literal["frozen", "unfrozen", "decoder-unfrozen"] = "frozen"  # "frozen" (entire backbone frozen), "unfrozen" (entire backbone trainable), "decoder-unfrozen" (only decoder unfrozen, encoder frozen)

    # Decoder options
    use_decoder: bool = False
    decoder_input_type: Literal["gated", "raw"] = "gated"
    decoder_attention_type: Literal["token_prune", "attention_mask", "bbox_mask_reencode", "bbox_crop_reencode"] = "token_prune"

    # Reencoding options
    bbox_mask_blur_sigma: float = 2.0  # Gaussian blur sigma for smooth mask edges (used with bbox_mask_reencode)
