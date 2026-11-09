import torch
import torch.nn as nn
from dataclasses import asdict
from typing import Optional, List
from PIL import Image
import os
import json
import math
import logging
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import hf_hub_download, snapshot_download
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# For the font loading patch
import transformers.models.pix2struct.image_processing_pix2struct as _pix2struct_mod


# import implementations
from src.model.config.model_config import CoExVQAConfig
from src.model.projectors.answer_projector import AnswerHeatmapProjector, AnswerHeatmapProjectorConfig
from src.model.projectors.question_projector import QuestionHeatmapProjector, QuestionHeatmapProjectorConfig
from src.model.gating.config import GatingConfig, create_gating_module
from src.utils.pix2struct_utils import (
    prepare_question_for_fusion,
    compute_square_patch_grid_size,
    create_patch_grid_coordinates
)

class CoExVQA(nn.Module):
    def __init__(self, config: CoExVQAConfig, cache_dir: Optional[str] = None):
        """Initialize the CoExVQA model."""
        super().__init__()

        # Store config
        self.config = config

        # Detect backbone type
        self._is_pix2struct = "pix2struct" in config.backbone_model.lower()

        # Load backbone
        if config.backbone_local_path is not None:
            model_path = config.backbone_local_path
            self.backbone_processor = AutoProcessor.from_pretrained(
                model_path,
                backend="torchvision",
                cache_dir=cache_dir
            )
            self.backbone_model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                cache_dir=cache_dir
            )
        else:
            self.backbone_processor = AutoProcessor.from_pretrained(
                config.backbone_model,
                backend="torchvision",
                cache_dir=cache_dir
            )
            self.backbone_model = AutoModelForImageTextToText.from_pretrained(
                config.backbone_model,
                cache_dir=cache_dir
            )


        # Handle missing token IDs in backbone config for non-pix2struct models by defaulting to tokenizer values
        if not self._is_pix2struct:
            tok = self.backbone_processor.tokenizer
            dec_cfg = self.backbone_model.config.decoder
            token_id_defaults = {
                'pad_token_id': dec_cfg.pad_token_id if dec_cfg.pad_token_id is not None else tok.pad_token_id,
                'decoder_start_token_id': dec_cfg.decoder_start_token_id if dec_cfg.decoder_start_token_id is not None else tok.bos_token_id,
                'eos_token_id': dec_cfg.eos_token_id if dec_cfg.eos_token_id is not None else tok.eos_token_id,
                'bos_token_id': dec_cfg.bos_token_id if dec_cfg.bos_token_id is not None else tok.bos_token_id,
            }
            for attr, val in token_id_defaults.items():
                if not hasattr(self.backbone_model.config, attr) or getattr(self.backbone_model.config, attr) is None:
                    setattr(self.backbone_model.config, attr, val)

        # By patching hf_hub_download to return a local path, we ensure that the font is always available without network access.
        # Fixes bug for font downloading on every forward pass. 
        if self._is_pix2struct:
            _font_path = hf_hub_download(repo_id="ybelkada/fonts", filename="Arial.TTF", cache_dir=cache_dir)
            _pix2struct_mod.hf_hub_download = lambda *args, **kwargs: _font_path

        # Auto-detect d_model from backbone if not set
        detected_d_model = CoExVQA.d_model_for_backbone(config.backbone_model)
        if self.config.d_model is None:
            self.config.d_model = detected_d_model
        elif self.config.d_model != detected_d_model:
            raise ValueError(
                f"Config d_model={self.config.d_model} does not match "
                f"backbone hidden size={detected_d_model}"
            )

        # Apply encoder freeze mode
        self._apply_encoder_freeze_mode()

        # Check that all d_model sizes are the same
        if self.config.d_model != self.config.answer_projector_config.d_model or self.config.d_model != self.config.question_projector_config.d_model:
            raise ValueError("CoExVQAConfig did not have the same d_model size for all projectors")

        # Answer heatmap projector
        a_config = config.answer_projector_config if config.answer_projector_config is not None else AnswerHeatmapProjectorConfig(d_model=self.config.d_model)
        self.a_proj = AnswerHeatmapProjector(config=a_config)
        self.config.answer_projector_config = a_config

        # Question heatmap projector
        q_config = config.question_projector_config if config.question_projector_config is not None else QuestionHeatmapProjectorConfig(d_model=self.config.d_model)
        self.q_proj = QuestionHeatmapProjector(config=q_config)
        self.config.question_projector_config = q_config

        # Gate for question prior
        if self.config.gate_config.d_model is None:
            self.config.gate_config.d_model = self.config.d_model
        self.gate = create_gating_module(self.config.gate_config)

        # Validate backbone and decoder configuration
        if config.backbone_freeze_mode not in ["frozen", "unfrozen", "decoder-unfrozen"]:
            raise ValueError(f"backbone_freeze_mode must be 'frozen', 'unfrozen', or 'decoder-unfrozen', got: {config.backbone_freeze_mode}")
        if config.decoder_input_type not in ["gated", "raw"]:
            raise ValueError(f"decoder_input_type must be 'gated' or 'raw', got: {config.decoder_input_type}")
        if config.decoder_attention_type not in ["token_prune", "attention_mask", "bbox_mask_reencode", "bbox_crop_reencode"]:
            raise ValueError(f"decoder_attention_type must be 'token_prune', 'attention_mask', 'bbox_mask_reencode', or 'bbox_crop_reencode', got: {config.decoder_attention_type}")

    @staticmethod
    def d_model_for_backbone(backbone_model: str) -> int:
        """Auto-detect the backbone's hidden size (d_model) based on the model name."""
        name = backbone_model.lower()
        if "pix2struct" in name:
            return 1536 if "large" in name else 768
        if "donut" in name:
            return 1024
        raise ValueError(
            f"Cannot auto-detect d_model for backbone '{backbone_model}'. "
            f"Set d_model explicitly in CoExVQAConfig."
        )

    def _build_encoder_inputs(self, images, questions) -> dict:
        """Build inputs for the backbone encoder. Handles differences in processor requirements between Pix2Struct and Donut."""
        if self._is_pix2struct:
            return self.backbone_processor(
                images=images,
                text=questions,
                padding=True,
                return_tensors="pt",
                max_patches=self.config.max_patches,
            )
        else:
            return self.backbone_processor(
                images=images,
                return_tensors="pt",
            )

    def _get_patch_grid(self, seq_len: int):
        """Get the patch grid dimensions (H_p, W_p) based on the sequence length and backbone type."""
        if self._is_pix2struct:
            return compute_square_patch_grid_size(seq_len)
        enc_cfg = self.backbone_model.config.encoder
        H, W = enc_cfg.image_size
        downscale = enc_cfg.patch_size * (2 ** len(enc_cfg.depths))
        return H // downscale, W // downscale

    def _apply_encoder_freeze_mode(self):
        """
        Apply the specified backbone freeze mode to the backbone model.
        Used in experiments to control which parts of the model are trainable.

        Modes:
        - "frozen": Entire backbone (encoder + decoder) frozen
        - "unfrozen": Entire backbone (encoder + decoder) trainable
        - "decoder-unfrozen": Only decoder trainable, encoder frozen

        Handles backward compatibility with the deprecated freeze_backbone flag.
        """
        # Handle backward compatibility: if freeze_backbone is explicitly set, use it
        if hasattr(self.config, 'freeze_backbone') and self.config.freeze_backbone is not None:
            # Map old freeze_backbone to new backbone_freeze_mode
            if self.config.freeze_backbone and self.config.backbone_freeze_mode == "frozen":
                # Default case: frozen
                mode = "frozen"
            elif not self.config.freeze_backbone:
                # User explicitly unfroze: use unfrozen
                mode = "unfrozen"
            else:
                # Use the explicitly set backbone_freeze_mode
                mode = self.config.backbone_freeze_mode
        else:
            mode = self.config.backbone_freeze_mode

        # Apply the freeze mode
        if mode == "frozen":
            # Freeze entire backbone (encoder + decoder)
            for p in self.backbone_model.parameters():
                p.requires_grad_(False)

        elif mode == "unfrozen":
            # Unfreeze entire backbone (encoder + decoder)
            for p in self.backbone_model.parameters():
                p.requires_grad_(True)

        elif mode == "decoder-unfrozen":
            # Freeze encoder, unfreeze decoder
            # First freeze everything
            for p in self.backbone_model.parameters():
                p.requires_grad_(False)

            # Then unfreeze decoder
            if hasattr(self.backbone_model, 'decoder'):
                for p in self.backbone_model.decoder.parameters():
                    p.requires_grad_(True)
            else:
                raise ValueError("Backbone model does not have a 'decoder' attribute. Cannot use 'decoder-unfrozen' mode.")

    @staticmethod
    def compute_patch_grid(image_size, max_patches=512, patch_size=16):
        """
        Compute Pix2Struct patch grid (H_p, W_p) from image dimensions.

        Mirrors the HuggingFace Pix2Struct processor's aspect-ratio-preserving
        patch allocation formula.

        Args:
            image_size: (W, H) tuple (PIL convention)
            max_patches: Maximum number of patches (default: 512)
            patch_size: Patch size in pixels (default: 16, square patches)

        Returns:
            Tuple of (H_p, W_p): patch grid dimensions where H_p * W_p <= max_patches
        """
        W, H = image_size
        scale = math.sqrt(max_patches * (patch_size / H) * (patch_size / W))
        H_p = max(min(math.floor(scale * H / patch_size), max_patches), 1)
        W_p = max(min(math.floor(scale * W / patch_size), max_patches), 1)
        return H_p, W_p

    def _compute_bbox_attention_weights(self, bbox: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Compute soft attention weights based on predicted bounding box (differentiable).
        Uses Gaussian falloff from bbox center. 

        Args:
            bbox: (B, 4) - Predicted bounding boxes in format [x_center, y_center, width, height]
            seq_len: Total sequence length of embeddings

        Returns:
            torch.Tensor: (B, seq_len) - Attention weights in [0, 1], higher inside bbox
        """
        B = bbox.shape[0]
        device = bbox.device

        # Calculate grid dimensions and create coordinate grid
        H_p, W_p = self._get_patch_grid(seq_len)
        grid = create_patch_grid_coordinates(seq_len, H_p, W_p, device)  # (seq_len, 2)

        # Expand for batch: (B, seq_len, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1)

        # Extract bbox parameters
        x_center = bbox[:, 0].unsqueeze(1)  # (B, 1)
        y_center = bbox[:, 1].unsqueeze(1)  # (B, 1)
        width = bbox[:, 2].unsqueeze(1)  # (B, 1)
        height = bbox[:, 3].unsqueeze(1)  # (B, 1)

        # Calculate normalized distances from bbox center
        dx = (grid[..., 0] - x_center) / (width + 1e-6)  # (B, seq_len)
        dy = (grid[..., 1] - y_center) / (height + 1e-6)  # (B, seq_len)

        # Gaussian-like attention weights (higher inside bbox, lower outside)
        distances = dx**2 + dy**2  # (B, seq_len)
        attention_weights = torch.exp(-0.25 * distances)  # (B, seq_len), values in (0, 1]

        # Sum-normalization scales down by ~1/seq_len (~1/512), destroying the signal
        # Max-normalization keeps the peak at 1.0 while preserving relative weights
        attention_weights = attention_weights / (attention_weights.max(dim=1, keepdim=True).values + 1e-6)

        return attention_weights

    def _get_bbox_token_indices(self, bbox: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Get indices of tokens that fall within the predicted bounding box.

        Args:
            bbox: (B, 4) - Predicted bounding boxes in format [x_center, y_center, width, height]
            seq_len: Total sequence length of embeddings

        Returns:
            torch.Tensor: (B, seq_len) - Boolean mask indicating which tokens are inside bbox
        """
        B = bbox.shape[0]
        device = bbox.device

        # Calculate grid dimensions and create coordinate grid
        H_p, W_p = self._get_patch_grid(seq_len)
        grid = create_patch_grid_coordinates(seq_len, H_p, W_p, device)  # (seq_len, 2)

        # Expand for batch: (B, seq_len, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1)

        # Extract bbox parameters
        x_center = bbox[:, 0].unsqueeze(1)  # (B, 1)
        y_center = bbox[:, 1].unsqueeze(1)  # (B, 1)
        width = bbox[:, 2].unsqueeze(1)  # (B, 1)
        height = bbox[:, 3].unsqueeze(1)  # (B, 1)

        # Calculate bbox boundaries with clipping to prevent out-of-bounds
        x_min = torch.clamp(x_center - width / 2, 0, 1)
        x_max = torch.clamp(x_center + width / 2, 0, 1)
        y_min = torch.clamp(y_center - height / 2, 0, 1)
        y_max = torch.clamp(y_center + height / 2, 0, 1)

        # Estimate patch size (patches are roughly square)
        patch_size = 1.0 / max(H_p, W_p)  # Approximate patch dimension

        # Expand bbox by 4x patch size to capture surrounding context
        # This addresses low coverage issue where decoder only saw 0.3-0.7% of tokens
        margin = patch_size * 4
        x_min_expanded = torch.clamp(x_min - margin, 0, 1)
        x_max_expanded = torch.clamp(x_max + margin, 0, 1)
        y_min_expanded = torch.clamp(y_min - margin, 0, 1)
        y_max_expanded = torch.clamp(y_max + margin, 0, 1)

        # Check which tokens fall inside the expanded bbox
        inside_x = (grid[..., 0] >= x_min_expanded) & (grid[..., 0] <= x_max_expanded)  # (B, seq_len)
        inside_y = (grid[..., 1] >= y_min_expanded) & (grid[..., 1] <= y_max_expanded)  # (B, seq_len)
        inside_bbox = inside_x & inside_y  # (B, seq_len)

        return inside_bbox

    def _prune_to_bbox_tokens(self, embeddings: torch.Tensor, bbox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Token pruning using soft attention (differentiable version).
        Creates a binary attention mask to emphasize bbox regions without destroying embedding magnitudes.

        Args:
            embeddings: (B, seq_len, d_model) - Image embeddings
            bbox: (B, 4) - Predicted bounding boxes

        Returns:
            embeddings: (B, seq_len, d_model) - Original embeddings (unchanged)
            attention_mask: (B, seq_len) - Binary mask (1=attend, 0=ignore) for bbox tokens
        """
        B = embeddings.shape[0]
        seq_len = embeddings.shape[1]
        device = embeddings.device

        # Compute differentiable soft attention weights for bbox
        attention_weights = self._compute_bbox_attention_weights(bbox, seq_len)  # (B, seq_len)

        # This ensures we attend to a reasonable region around the bbox
        threshold_per_sample = torch.quantile(attention_weights, 0.5, dim=1, keepdim=True)  # (B, 1)
        attention_mask = (attention_weights >= threshold_per_sample).float()  # (B, seq_len), binary 0/1

        # Validate: each sample must have at least one token with attention
        mask_sum_per_sample = attention_mask.sum(dim=1)  # (B,)
        assert (mask_sum_per_sample > 0).all(), \
            f"Token pruning produced empty masks for some samples: {(mask_sum_per_sample == 0).nonzero(as_tuple=True)[0].tolist()}"

        return embeddings, attention_mask

    def _create_bbox_attention_mask(self, bbox: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        BBox attention masking: Create attention mask that allows attending only to bbox tokens.

        Args:
            bbox: (B, 4) - Predicted bounding boxes
            seq_len: Total sequence length of embeddings

        Returns:
            torch.Tensor: (B, seq_len) - Attention mask (1 = attend, 0 = mask out)
        """
        B = bbox.shape[0]
        device = bbox.device

        # Get boolean mask of tokens inside bbox
        inside_bbox = self._get_bbox_token_indices(bbox, seq_len)  # (B, seq_len)
        attention_mask = inside_bbox.float()

        # Validate: each sample must have at least one token inside bbox or question
        mask_sum_per_sample = attention_mask.sum(dim=1)  # (B,)
        assert (mask_sum_per_sample > 0).all(), \
            f"BBox attention mask has no tokens for some samples. BBox values: {bbox[mask_sum_per_sample == 0].tolist()}"

        return attention_mask

    def _reencode_with_bbox_mask(
        self,
        images: List[Image.Image],
        questions: List[str],
        bbox: torch.Tensor,
        blur_sigma: float
    ) -> torch.Tensor:
        """
        Apply Gaussian-blurred mask to images based on bbox and reencode.

        Args:
            images: Original PIL images (B,)
            questions: Question strings (B,)
            bbox: (B, 4) - Bounding boxes in normalized [cx, cy, w, h] format
            blur_sigma: Sigma for Gaussian blur (controls smoothness)

        Returns:
            torch.Tensor: (B, seq_len, d_model) - Reencoded embeddings
        """

        B = len(images)
        device = bbox.device

        # Step 1: Process each image individually (may have different sizes)
        masked_images_pil = []

        for i in range(B):
            # Convert PIL to tensor
            img_t = TF.to_tensor(images[i]).unsqueeze(0).to(device)  # (1, 3, H, W)
            _, _, H, W = img_t.shape

            # Extract bbox parameters and convert to pixel coordinates
            cx, cy, w, h = bbox[i]
            cx_px = cx * W
            cy_px = cy * H
            w_px = w * W
            h_px = h * H

            # Edge case: if bbox is too small, use full image
            if w_px < 1 or h_px < 1:
                mask_blurred = torch.ones((1, 1, H, W), device=device)
            else:
                # Step 2: Create binary mask (1 inside bbox, 0 outside)
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(H, device=device, dtype=bbox.dtype),
                    torch.arange(W, device=device, dtype=bbox.dtype),
                    indexing='ij'
                )

                # Compute bbox boundaries
                x_min = cx_px - w_px / 2
                x_max = cx_px + w_px / 2
                y_min = cy_px - h_px / 2
                y_max = cy_px + h_px / 2

                # Create binary mask
                mask = ((x_grid >= x_min) & (x_grid <= x_max) &
                        (y_grid >= y_min) & (y_grid <= y_max)).float()
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

                # Step 3: Apply Gaussian blur for smooth edges
                kernel_size = int(6 * blur_sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                mask_blurred = TF.gaussian_blur(mask, kernel_size=[kernel_size, kernel_size], sigma=[blur_sigma, blur_sigma])

            # Step 4: Apply mask to image (element-wise multiply)
            masked_img_t = img_t * mask_blurred  # (1, 3, H, W)

            # Step 5: Convert back to PIL
            masked_img_pil = TF.to_pil_image(masked_img_t.squeeze(0).cpu())
            masked_images_pil.append(masked_img_pil)

        # Step 6: Reencode through backbone processor + encoder
        encoder_inputs = self._build_encoder_inputs(masked_images_pil, questions)
        encoder_inputs = {k: v.to(device) for k, v in encoder_inputs.items()}

        encoder = self.backbone_model.get_encoder()
        out = encoder(**encoder_inputs)
        reencoded_embeddings = out.last_hidden_state  # (B, seq_len, d_model)

        return reencoded_embeddings, masked_images_pil

    def _reencode_with_crop(
        self,
        images: List[Image.Image],
        questions: List[str],
        bbox: torch.Tensor,
        margin: float = 0.15
    ) -> torch.Tensor:
        """
        Crop images around the predicted bounding box with a margin and reencode.

        Args:
            images: Original PIL images (B,)
            questions: Question strings (B,)
            bbox: (B, 4) - Bounding boxes in normalized [cx, cy, w, h] format
            margin: Margin to add around bbox as a fraction (default: 0.15 = 15%)

        Returns:
            torch.Tensor: (B, seq_len, d_model) - Reencoded embeddings
            List[Image.Image]: Cropped PIL images
        """
        B = len(images)
        device = bbox.device

        # Step 1: Process each image individually (may have different sizes)
        cropped_images_pil = []

        for i in range(B):
            img = images[i]
            W, H = img.size  # PIL images use (width, height)

            # Extract bbox parameters and convert to pixel coordinates
            cx, cy, w, h = bbox[i].cpu()
            cx_px = cx * W
            cy_px = cy * H
            w_px = w * W
            h_px = h * H

            # Add margin (15% of bbox dimensions)
            w_margin = w_px * margin
            h_margin = h_px * margin

            # Calculate crop boundaries with margin
            x_min = max(0, cx_px - (w_px / 2) - w_margin)
            x_max = min(W, cx_px + (w_px / 2) + w_margin)
            y_min = max(0, cy_px - (h_px / 2) - h_margin)
            y_max = min(H, cy_px + (h_px / 2) + h_margin)

            # Edge case: if bbox is too small or invalid, use full image
            if (x_max - x_min) < 10 or (y_max - y_min) < 10:
                cropped_img = img
            else:
                # Crop the image (PIL uses left, upper, right, lower)
                cropped_img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            cropped_images_pil.append(cropped_img)

        # Step 2: Reencode through backbone processor + encoder
        encoder_inputs = self._build_encoder_inputs(cropped_images_pil, questions)
        encoder_inputs = {k: v.to(device) for k, v in encoder_inputs.items()}

        encoder = self.backbone_model.get_encoder()
        out = encoder(**encoder_inputs)
        reencoded_embeddings = out.last_hidden_state  # (B, seq_len, d_model)

        return reencoded_embeddings, cropped_images_pil

    def forward(
        self,
        images: List[Image.Image],
        questions: List[str],
        decoder_labels: Optional[List[str]] = None,
        gt_boxes: Optional[torch.Tensor] = None,
        return_ib_intermediates: bool = False,
    ):
        """
        Forward pass of the CoExVQA model.

        Args:
            images: List of PIL images
            questions: List of question strings
            decoder_labels: Optional list of answer strings to tokenize for decoder loss computation
            gt_boxes: Optional ground truth boxes (B, 4) in normalized cxcywh format for decoder attention.
                     If provided, decoder uses GT boxes instead of predicted boxes (no gradients flow to bbox prediction).
            return_ib_intermediates: If True, include 'image_embeddings' and 'gated_embeddings' in
                     the returned dict. Only set when computing the IB loss (lambda_ib > 0) to
                     avoid unnecessary memory retention.

        Returns:
            Dict containing:
                - 'box': (B, 4) Predicted bounding boxes
                - 'q_mask': (B, seq_len, 1) Question-based attention mask
                - 'encoder_outputs': Modified encoder output object with processed embeddings
                - 'encoder_attention_mask': (B, seq_len) Attention mask for decoder/generation
                - 'decoder_output': Decoder output object (None if decoder not used or no labels)
                - 'masked_images_pil': List of PIL images with bbox masking/cropping applied (None unless bbox_mask_reencode or bbox_crop_reencode)
                - 'image_embeddings': (B, seq_len, d_model) raw encoder embeddings — only present when return_ib_intermediates=True
                - 'gated_embeddings': (B, seq_len, d_model) gated encoder embeddings — only present when return_ib_intermediates=True
        """
        device = next(self.parameters()).device

        encoder_inputs = self._build_encoder_inputs(images, questions)
        encoder_inputs = {k: v.to(device) for k, v in encoder_inputs.items()}

        # Get encoder embeddings
        encoder = self.backbone_model.get_encoder()
        out = encoder(**encoder_inputs)
        image_embeddings = out.last_hidden_state  

        # Get question representations for fusion
        q_inputs = prepare_question_for_fusion(
            questions=questions,
            processor=self.backbone_processor,
            model=self.backbone_model,
            device=device,
            pool="mean"
        )

        # Get question logits
        q_logits = self.q_proj(
            image_embeddings=image_embeddings,
            q_pooled=q_inputs["q_pooled"],
            q_tokens=q_inputs.get("q_tokens", None),
            key_padding_mask=q_inputs.get("key_padding_mask", None),
        )

        # Create q-prior mask
        q_mask = torch.sigmoid(q_logits)

        # Get the gated embeddings
        gated_embeddings = self.gate(image_embeddings, q_mask)

        # Project the answer location
        box = self.a_proj(
            image_embeddings=gated_embeddings,
            q_pooled=q_inputs["q_pooled"],
            q_tokens=q_inputs.get("q_tokens", None),
            key_padding_mask=q_inputs.get("key_padding_mask", None),
        )

        # Collect IB intermediates if requested (only when lambda_ib > 0 to avoid memory overhead)
        ib_extras = (
            {"image_embeddings": image_embeddings, "gated_embeddings": gated_embeddings}
            if return_ib_intermediates
            else {}
        )

        # Select embeddings based on decoder_input_type (do this regardless of decoder usage)
        if self.config.decoder_input_type == "gated":
            decoder_embeddings = gated_embeddings
        else:  # "raw"
            decoder_embeddings = image_embeddings

        # Determine which boxes to use for decoder attention
        # If GT boxes provided, use them; otherwise use predicted boxes
        boxes_for_decoder = gt_boxes.detach() if gt_boxes is not None else box

        # Store masked images for debugging (only used when bbox_mask_reencode)
        masked_images_pil = None

        # Apply bbox-based attention
        if self.config.decoder_attention_type == "token_prune":
            decoder_embeddings, encoder_attention_mask = self._prune_to_bbox_tokens(decoder_embeddings, boxes_for_decoder)
        elif self.config.decoder_attention_type == "attention_mask":
            encoder_attention_mask = self._create_bbox_attention_mask(boxes_for_decoder, decoder_embeddings.shape[1])
        elif self.config.decoder_attention_type == "bbox_mask_reencode":
            # Reencode with masked images
            decoder_embeddings, masked_images_pil = self._reencode_with_bbox_mask(
                images=images,
                questions=questions,
                bbox=boxes_for_decoder,
                blur_sigma=self.config.bbox_mask_blur_sigma
            )
            # All tokens are valid after reencoding
            encoder_attention_mask = torch.ones(
                decoder_embeddings.shape[0],
                decoder_embeddings.shape[1],
                device=decoder_embeddings.device
            )
        elif self.config.decoder_attention_type == "bbox_crop_reencode":
            # Reencode with cropped images
            decoder_embeddings, masked_images_pil = self._reencode_with_crop(
                images=images,
                questions=questions,
                bbox=boxes_for_decoder,
                margin=0.15  # 15% margin around bbox
            )
            # All tokens are valid after reencoding
            encoder_attention_mask = torch.ones(
                decoder_embeddings.shape[0],
                decoder_embeddings.shape[1],
                device=decoder_embeddings.device
            )

        # Modify encoder output with processed embeddings
        out.last_hidden_state = decoder_embeddings

        # If decoder is not used, return early without decoder output
        if not self.config.use_decoder:
            return {
                "box": box,
                "q_mask": q_mask,
                "encoder_outputs": out,
                "encoder_attention_mask": encoder_attention_mask,
                "decoder_output": None,
                "masked_images_pil": masked_images_pil,
                **ib_extras,
            }

        # Decoder is enabled
        # If no labels provided, still return encoder outputs but no decoder output
        if decoder_labels is None:
            return {
                "box": box,
                "q_mask": q_mask,
                "encoder_outputs": out,
                "encoder_attention_mask": encoder_attention_mask,
                "decoder_output": None,
                "masked_images_pil": masked_images_pil,
                **ib_extras,
            }

        # Process decoder_labels for loss computation
        labels = None
        if decoder_labels is not None:
            # Tokenize target answers
            target_ids = self.backbone_processor.tokenizer(
                decoder_labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device).input_ids

            # Create labels (targets for the model to predict)
            labels = target_ids.clone()

            # Mask padding tokens in labels (set to -100 so they're ignored in loss computation)
            labels[labels == self.backbone_processor.tokenizer.pad_token_id] = -100

            # Validate labels: all samples must have at least one non-padding token
            all_masked = (labels == -100).all(dim=1)  # (B,)
            assert not all_masked.any(), \
                f"Some samples have all padding tokens (empty answers): {all_masked.nonzero(as_tuple=True)[0].tolist()}"


        # Convert attention mask to torch.long
        encoder_attention_mask_long = encoder_attention_mask.to(dtype=torch.long)

        # Call full model with encoder_outputs and labels
        # This handles label shifting internally (predicts token t from tokens < t)
        decoder_output = self.backbone_model(
            encoder_outputs=out,
            attention_mask=encoder_attention_mask_long,
            labels=labels,
            return_dict=True,
        )

        # Validate decoder output when labels are provided (loss should exist)
        if labels is not None:
            assert decoder_output.loss is not None, "Decoder did not compute loss when labels were provided!"
            assert not torch.isnan(decoder_output.loss), \
                f"Decoder loss is NaN! Labels shape: {labels.shape}, " \
                f"Encoder hidden states shape: {decoder_embeddings.shape}, Attention mask shape: {encoder_attention_mask.shape}"

        # Return all outputs in a dict
        return {
            "box": box,
            "q_mask": q_mask,
            "encoder_outputs": out,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_output": decoder_output,
            "masked_images_pil": masked_images_pil,
            **ib_extras,
        }

    def generate_preds(
        self,
        images: List[Image.Image],
        questions: List[str],
        gen_kwargs: Optional[dict] = None,
        gt_boxes: Optional[torch.Tensor] = None
    ) -> List[str]:
        """
        Generate text predictions using the full model pipeline.

        This method leverages forward() to ensure consistency between
        training and inference, then uses the processed encoder outputs for generation.

        Args:
            images: List of PIL images
            questions: List of question strings
            gen_kwargs: Dictionary of generation kwargs (max_new_tokens, etc.)
            gt_boxes: Optional ground truth boxes (B, 4) in normalized cxcywh format.
                     If provided, uses GT boxes instead of predicted boxes for attention.

        Returns:
            List[str]: Generated text predictions (decoded and stripped)
        """
        if gen_kwargs is None:
            gen_kwargs = {"max_new_tokens": 20}

        # Forward pass to get processed encoder outputs
        outputs = self.forward(
            images=images,
            questions=questions,
            gt_boxes=gt_boxes
        )

        # Extract the processed encoder outputs
        encoder_outputs = outputs["encoder_outputs"]
        encoder_attention_mask = outputs["encoder_attention_mask"]

        # Convert attention mask to torch.long
        encoder_attention_mask_long = encoder_attention_mask.to(dtype=torch.long)

        # Generate using the backbone model with processed embeddings
        pred_ids = self.backbone_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask_long,
            **gen_kwargs
        )

        # Decode predictions
        preds = self.backbone_processor.batch_decode(pred_ids, skip_special_tokens=True)
        preds = [p.strip() for p in preds]
        return preds
    

    def generate_preds_from_output(self, outputs: dict, gen_kwargs: Optional[dict] = None) -> List[str]:
        """
        Generate text predictions from the outputs of a forward() pass.

        This allows for flexible generation using the processed encoder outputs
        without needing to call forward() again (e.g., when you already have the outputs).

        Args:
            outputs: Dict output from forward() containing 'encoder_outputs' and 'encoder_attention_mask'
            gen_kwargs: Dictionary of generation kwargs (max_new_tokens, etc.)
        Returns:
            List[str]: Generated text predictions (decoded and stripped)
        """
        if gen_kwargs is None:
            gen_kwargs = {"max_new_tokens": 20}

        encoder_outputs = outputs["encoder_outputs"]
        encoder_attention_mask = outputs["encoder_attention_mask"]

        # Convert attention mask to torch.long
        encoder_attention_mask_long = encoder_attention_mask.to(dtype=torch.long)

        # Generate using the backbone model with processed embeddings
        pred_ids = self.backbone_model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask_long,
            **gen_kwargs
        )

        # Decode predictions
        preds = self.backbone_processor.batch_decode(pred_ids, skip_special_tokens=True)
        preds = [p.strip() for p in preds]
        return preds
    


    @classmethod
    def from_pretrained(cls, path: str, args=None, cache_dir: Optional[str] = None):
        """
        Load a CoExVQA model from a saved checkpoint directory.

        Args:
            path: Directory path where the checkpoint was saved
            args: Optional argparse.Namespace or dict containing training arguments.
                  If not provided, will attempt to load from checkpoint or args.json
            cache_dir: Optional cache directory for HuggingFace models

        Returns:
            Loaded CoExVQA model with weights restored

        Raises:
            ValueError: If args not provided and not found in checkpoint/args.json
            FileNotFoundError: If checkpoint directory or files don't exist
        """
        # Load the main checkpoint
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get args 
        if args is None:
            if "args" in checkpoint:
                args = checkpoint["args"]
            else:
                # Try to load from args.json
                args_path = os.path.join(path, "args.json")
                if os.path.exists(args_path):
                    with open(args_path, 'r') as f:
                        args = json.load(f)
                else:
                    raise ValueError(
                        "args not provided and not found in checkpoint or args.json. "
                        "Please provide args to from_pretrained()"
                    )

        # Convert args dict to namespace if needed (for compatibility)
        if isinstance(args, dict):
            from argparse import Namespace
            args = Namespace(**args)

        # Load the config
        if "config" in checkpoint:
            config_dict = checkpoint["config"]

            # Reconstruct nested configs
            if "answer_projector_config" in config_dict and config_dict["answer_projector_config"] is not None:
                config_dict["answer_projector_config"] = AnswerHeatmapProjectorConfig(**config_dict["answer_projector_config"])
            if "question_projector_config" in config_dict and config_dict["question_projector_config"] is not None:
                config_dict["question_projector_config"] = QuestionHeatmapProjectorConfig(**config_dict["question_projector_config"])
            if "gate_config" in config_dict and config_dict["gate_config"] is not None:
                config_dict["gate_config"] = GatingConfig(**config_dict["gate_config"])

            config = CoExVQAConfig(**config_dict)
        else:
            raise ValueError("Config not found in checkpoint")

        # Check if backbone exists in checkpoint directory and use it
        backbone_dir = os.path.join(path, "backbone")
        if os.path.exists(backbone_dir):
            config.backbone_local_path = backbone_dir

        # Create the model (this will load the backbone from the saved location)
        model = cls(config, cache_dir=cache_dir)

        # Load the complete model state (including backbone, projectors, and gate)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model

    @classmethod
    def from_hf(
        cls,
        repo_id: str = "indrehus/CoExVQA",
        cache_dir: Optional[str] = None,
    ) -> "CoExVQA":
        """
        Load a CoExVQA model from HuggingFace Hub.

        Downloads the model repository snapshot locally and then
        loads the checkpoint using from_pretrained().

        Args:
            repo_id: HuggingFace repository ID (default: "indrehus/CoExVQA")
            cache_dir: Optional directory for caching downloaded files

        Returns:
            Loaded CoExVQA model with weights restored
        """
        assert repo_id == "indrehus/CoExVQA", "Only the official CoExVQA model is available on indrehus/CoExVQA."
        local_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
        return cls.from_pretrained(local_dir, cache_dir=cache_dir)

    def save_checkpoint(
        self,
        path: str,
        epoch: Optional[int] = None,
        args=None,
        extra_state: Optional[dict] = None
    ):
        """
        Save model checkpoint to a directory.

        Creates a directory structure:
        - path/
          - checkpoint.pt (main checkpoint with model state, optimizer, etc.)
          - backbone/ (HuggingFace format backbone model and processor)

        Args:
            path: Directory path where the checkpoint will be saved
            epoch: Optional epoch number to include in checkpoint
            args: Optional argparse.Namespace or dict containing training arguments
            extra_state: Optional dict of extra state to save (e.g., optimizer, scheduler, wandb_run_id)
        """

        # Create checkpoint directory
        os.makedirs(path, exist_ok=True)

        # Build checkpoint dictionary
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else self.config,
        }

        # Save epoch if provided
        if epoch is not None:
            checkpoint["epoch"] = epoch

        if args is not None:
            # Handle both argparse.Namespace and dict
            args_dict = vars(args) if hasattr(args, '__dict__') else args

            # Filter out model architecture args ONLY if config is present
            if "config" in checkpoint and checkpoint["config"] is not None:
                filtered_args = {
                    k: v for k, v in args_dict.items()
                    if not (k.startswith('a_') or k.startswith('q_') or
                           k.startswith('gating_') or k == 'd_model')
                }
                checkpoint["args"] = filtered_args
                args_to_save = filtered_args
            else:
                # No config present - save all args
                checkpoint["args"] = args_dict
                args_to_save = args_dict

            # Save args as JSON
            args_path = os.path.join(path, "args.json")
            with open(args_path, 'w') as f:
                json.dump(args_to_save, f, indent=2)

        # Add extra state if provided 
        if extra_state is not None:
            for key, value in extra_state.items():
                checkpoint[key] = value

        # Save main checkpoint
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
            
        # Save backbone in HuggingFace format
        backbone_dir = os.path.join(path, "backbone")
        os.makedirs(backbone_dir, exist_ok=True)
        self.backbone_model.save_pretrained(backbone_dir)
        self.backbone_processor.save_pretrained(backbone_dir)
