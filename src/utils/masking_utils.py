import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_overlap_mask(
    q_mask: torch.Tensor,
    q_prior_patch: torch.Tensor,
    q_prior_mask: torch.Tensor,
    top_k_percent: float = 0.20,
) -> torch.Tensor:
    """
    Create overlap mask using top-k percentile intersection method.

    Overlap = patches that are in BOTH:
    - Top k% of patches by q_mask value (model's predictions)
    - Top k% of patches by q_prior value (ground truth)

    This method is robust to value distributions - works even if all q_mask
    values are low (e.g., max=0.3).

    Args:
        q_mask: Model predictions (B, seq_len, 1), values in [0, 1]
        q_prior_patch: Ground truth prior (B, max_patches), values in [0, 1], padded
        q_prior_mask: Boolean mask (B, max_patches), True=valid, False=padding
        top_k_percent: Percentile for top-k as decimal (e.g., 0.20 = top 20%)

    Returns:
        overlap_mask: Boolean tensor (B, seq_len)
                     True = patch is in overlap (both model and GT rank it highly)
                     False = patch is not in overlap

    Example:
        With top_k_percent=0.20 and seq_len=100:
        - Find top 20 patches by q_mask
        - Find top 20 patches by q_prior
        - Return intersection (typically 5-15 patches overlap)
    """
    B, seq_len, _ = q_mask.shape
    device = q_mask.device
    dtype = q_mask.dtype

    # Validate inputs
    if top_k_percent <= 0 or top_k_percent > 1.0:
        raise ValueError(f"top_k_percent must be in (0, 1.0], got {top_k_percent}")

    max_patches = q_prior_patch.shape[1]

    # Use the minimum of seq_len and max_patches to handle size mismatches
    # This can happen when model's max_patches (e.g., 512) differs from
    # the actual number of patches in the image
    effective_len = min(seq_len, max_patches)

    # Truncate both to effective_len and move to device
    q_prior_trunc = q_prior_patch[:, :effective_len].to(device=device, dtype=dtype)  # (B, effective_len)
    q_prior_valid_mask = q_prior_mask[:, :effective_len].to(device=device)  # (B, effective_len)

    # Set padded positions to very low value so they don't appear in top-k
    q_prior_trunc = q_prior_trunc.clone()
    q_prior_trunc[~q_prior_valid_mask] = -1e9

    # Squeeze q_mask to (B, seq_len) and truncate to effective_len
    q_mask_2d = q_mask[:, :effective_len, 0]  # (B, effective_len)

    # Initialize overlap mask for the effective length
    overlap_mask = torch.zeros(B, effective_len, dtype=torch.bool, device=device)

    # Process each sample in batch
    for i in range(B):
        # Compute k for this sample (at least 1 patch)
        k = max(1, int(effective_len * top_k_percent))

        # Get top-k indices from q_mask
        topk_mask_values, topk_mask_indices = torch.topk(q_mask_2d[i], k)

        # Get top-k indices from q_prior
        topk_prior_values, topk_prior_indices = torch.topk(q_prior_trunc[i], k)

        # Create boolean masks for top-k patches
        mask_important = torch.zeros(effective_len, dtype=torch.bool, device=device)
        mask_important[topk_mask_indices] = True

        prior_important = torch.zeros(effective_len, dtype=torch.bool, device=device)
        prior_important[topk_prior_indices] = True

        # Intersection = overlap
        overlap_mask[i] = mask_important & prior_important

    return overlap_mask


def apply_q_mask_masking(
    q_mask: torch.Tensor,
    overlap_mask: torch.Tensor,
    masking_type: str,
    mask_probability: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply masking strategy to q_mask based on overlap with important regions.

    Args:
        q_mask: Original q_mask from model (B, seq_len, 1), values in [0, 1]
        overlap_mask: Boolean mask indicating overlapping patches (B, effective_len)
                     Note: effective_len may be smaller than seq_len
        masking_type: Masking strategy:
            - "mask_overlapping": Probabilistically zero out overlapping patches (faithfulness test)
            - "mask_non_overlapping": Probabilistically zero out non-overlapping patches (robustness test)
        mask_probability: Probability for each patch to be masked (0.0-1.0).
                         Applied to overlapping patches for "mask_overlapping",
                         and non-overlapping patches for "mask_non_overlapping".

    Returns:
        masked_q_mask: Modified q_mask (B, seq_len, 1) with masked patches set to 0.0
        final_mask: Boolean tensor (B, effective_len) indicating which patches were masked

    Interpretation:
        - Faithfulness (mask_overlapping): Remove important patches → performance should drop
        - Robustness (mask_non_overlapping): Remove noise → performance should stay high
    """
    # Validate masking_type
    valid_types = ["mask_overlapping", "mask_non_overlapping"]
    if masking_type not in valid_types:
        raise ValueError(
            f"Invalid masking_type '{masking_type}'. "
            f"Must be one of {valid_types}"
        )

    # Clone q_mask to avoid in-place modification
    masked_q_mask = q_mask.clone()

    B, effective_len = overlap_mask.shape
    device = overlap_mask.device

    # Generate random mask via Bernoulli sampling
    random_mask = torch.rand(B, effective_len, device=device) < mask_probability

    if masking_type == "mask_overlapping":
        # Probabilistic masking: each overlapping patch has probability p of being masked
        final_mask = overlap_mask & random_mask
        final_mask_expanded = final_mask.unsqueeze(-1)

        # Only apply to the effective_len portion of q_mask
        masked_q_mask[:, :effective_len, :][final_mask_expanded] = 0.0

    elif masking_type == "mask_non_overlapping":
        # Probabilistic masking: each non-overlapping patch has probability p of being masked
        non_overlap_mask = ~overlap_mask

        # Combine: only mask patches that are BOTH non-overlapping AND randomly selected
        final_mask = non_overlap_mask & random_mask
        final_mask_expanded = final_mask.unsqueeze(-1)
        masked_q_mask[:, :effective_len, :][final_mask_expanded] = 0.0

    return masked_q_mask, final_mask


class QuestionMaskingHook:
    """
    Forward hook manager for applying q_mask masking during evaluation.
    """

    def __init__(
        self,
        masking_type: str,
        top_k_percent: float = 0.20,
        mask_probability: float = 1.0,
        mask_embeddings: bool = False,
    ):
        """
        Initialize hook manager.

        Args:
            masking_type: One of ["mask_overlapping", "mask_non_overlapping"]
            top_k_percent: Percentile for top-k overlap as decimal (e.g., 0.20 = top 20%)
            mask_probability: Probability for each patch to be masked (0.0-1.0).
                            Applied to overlapping patches for "mask_overlapping",
                            and non-overlapping patches for "mask_non_overlapping".
            mask_embeddings: If True, also zero out image_embeddings at masked positions.
        """
        # Validate inputs
        valid_types = ["mask_overlapping", "mask_non_overlapping"]
        if masking_type not in valid_types:
            raise ValueError(
                f"Invalid masking_type '{masking_type}'. "
                f"Must be one of {valid_types}"
            )

        if top_k_percent <= 0 or top_k_percent > 1.0:
            raise ValueError(f"top_k_percent must be in (0, 1.0], got {top_k_percent}")

        if mask_probability < 0 or mask_probability > 1:
            raise ValueError(f"mask_probability must be in [0, 1], got {mask_probability}")

        # Store config
        self.masking_type = masking_type
        self.top_k_percent = top_k_percent
        self.mask_probability = mask_probability
        self.mask_embeddings = mask_embeddings

        # Batch data storage
        self.q_prior_patch: Optional[torch.Tensor] = None
        self.q_prior_mask: Optional[torch.Tensor] = None

        # Storage for final mask (used for embedding masking)
        self.last_final_mask: Optional[torch.Tensor] = None

    def set_batch_data(
        self,
        q_prior_patch: torch.Tensor,
        q_prior_mask: torch.Tensor
    ):
        """
        Set q_prior data for current batch.

        This must be called BEFORE each forward pass.

        Args:
            q_prior_patch: Ground truth prior (B, max_patches)
            q_prior_mask: Boolean mask (B, max_patches)
        """
        self.q_prior_patch = q_prior_patch
        self.q_prior_mask = q_prior_mask
        self.last_final_mask = None  # Clear previous mask

    def _hook_fn(self, module, input, output):
        """
        Hook function called automatically after question projector forward.

        This function intercepts q_logits from the question projector, applies sigmoid,
        then applies masking before the result is used in gating.

        Flow:
        1. q_proj.forward() returns q_logits
        2. Hook fires, receives q_logits as output
        3. Apply sigmoid: q_mask = torch.sigmoid(q_logits)
        4. Compute overlap mask using q_mask and q_prior
        5. Apply masking to q_mask
        6. Return masked q_mask (which will be used in model.py line 586)

        Args:
            module: The question projector (model.q_proj)
            input: Inputs to q_proj.forward() (not used)
            output: q_logits from q_proj.forward() - shape (B, seq_len, 1)

        Returns:
            Modified q_logits such that sigmoid(q_logits) = masked_q_mask
        """
        try:
            # Check if batch data is set
            if self.q_prior_patch is None or self.q_prior_mask is None:
                logger.warning(
                    "Hook fired but batch data not set. "
                    "Call set_batch_data() before forward pass. "
                    "Skipping masking for this batch."
                )
                return output

            # Output from q_proj is q_logits (before sigmoid)
            q_logits = output

            # Apply sigmoid to get q_mask
            q_mask = torch.sigmoid(q_logits)

            # Compute overlap mask
            overlap_mask = create_overlap_mask(
                q_mask=q_mask,
                q_prior_patch=self.q_prior_patch,
                q_prior_mask=self.q_prior_mask,
                top_k_percent=self.top_k_percent,
            )

            # Apply masking
            masked_q_mask, final_mask = apply_q_mask_masking(
                q_mask=q_mask,
                overlap_mask=overlap_mask,
                masking_type=self.masking_type,
                mask_probability=self.mask_probability,
            )

            # Store final_mask for embedding masking (if enabled)
            if self.mask_embeddings:
                self.last_final_mask = final_mask

            # Convert masked_q_mask back to logits space
            masked_logits = q_logits.clone()

            # Where we masked (set to 0), use large negative logit
            mask_applied = (masked_q_mask == 0.0) & (q_mask > 0.0)
            masked_logits[mask_applied] = -20.0
            return masked_logits

        except Exception as e:
            logger.error(f"Error in masking hook: {e}", exc_info=True)
            return output

    def register_hook(self, model: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        """
        Register forward hook on question projector.

        Args:
            model: The CoExVQA model

        Returns:
            handle: Removable handle for cleanup (call handle.remove() after eval)
        """
        # Hook on the question projector, not the full model
        # This intercepts q_logits before sigmoid is applied
        handle = model.q_proj.register_forward_hook(self._hook_fn)
        return handle

    def _gate_pre_hook_fn(self, module, args):
        """
        Pre-hook on gating layer to mask image_embeddings at the same positions
        where q_mask was masked.

        Args:
            module: The gating module (model.gate)
            args: Tuple of (image_embeddings, q_mask)
                - image_embeddings: (B, seq_len, d_model)
                - q_mask: (B, seq_len, 1)

        Returns:
            Modified args tuple with masked image_embeddings, or original args if
            mask_embeddings is disabled or no mask is available.
        """
        if not self.mask_embeddings or self.last_final_mask is None:
            return args

        image_embeddings, q_mask = args

        # Clone to avoid in-place modification
        masked_embeddings = image_embeddings.clone()

        # Get dimensions
        B, seq_len, d_model = masked_embeddings.shape
        effective_len = self.last_final_mask.shape[1]

        # Expand final_mask to match embedding dimensions
        # final_mask is (B, effective_len), expand to (B, effective_len, d_model)
        mask_expanded = self.last_final_mask.unsqueeze(-1).expand(-1, -1, d_model)

        # Zero out masked positions (only up to effective_len)
        masked_embeddings[:, :effective_len, :][mask_expanded] = 0.0

        return (masked_embeddings, q_mask)

    def register_gate_hook(self, model: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        """
        Register forward pre-hook on gating layer for embedding masking.

        This hook intercepts the inputs to the gating layer and zeros out
        image_embeddings at the same positions where q_mask was masked.

        Args:
            model: The CoExVQA model

        Returns:
            handle: Removable handle for cleanup (call handle.remove() after eval)
        """
        handle = model.gate.register_forward_pre_hook(self._gate_pre_hook_fn)
        return handle
