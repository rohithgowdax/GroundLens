import torch
import torch.nn.functional as F


def get_q_prior_mse_loss(
    q_mask_pred: torch.Tensor,
    q_prior_patch: torch.Tensor,
    q_prior_patch_shape: torch.Tensor,
    q_prior_mask: torch.Tensor = None,
):
    """
    Compute MSE loss between predicted question mask and ground truth prior.

    MSE loss enforces exact value matching at each patch location, which prevents
    the model from producing uniform predictions (unlike Dice loss which can be
    satisfied by predicting the mean value everywhere).

    Args:
        q_mask_pred: Predicted question mask from model, shape (batch_size, seq_len, 1)
                     Values should be in [0, 1] (after sigmoid)
        q_prior_patch: Ground truth patch-level priors, shape (batch_size, max_patches)
                       Padded with zeros for variable-length sequences
        q_prior_patch_shape: Patch shapes for each batch item, shape (batch_size, 2)
        q_prior_mask: Boolean mask indicating valid patches (True) vs padding (False),
                      shape (batch_size, max_patches). If None, assumes no padding.

    Returns:
        MSE loss (scalar tensor), lower is better
    """
    device = q_mask_pred.device
    dtype = q_mask_pred.dtype
    batch_size = q_mask_pred.shape[0]

    batch_losses = []

    for i in range(batch_size):
        # Get the patch shape for this sample
        h_patch, w_patch = q_prior_patch_shape[i]
        num_patches = h_patch * w_patch

        # Check if predicted mask has enough patches
        seq_len = q_mask_pred.shape[1]
        if num_patches > seq_len:
            raise ValueError(f"Ground truth has {num_patches} patches ({h_patch}x{w_patch}) but predicted mask only has {seq_len} tokens.")

        # Get ground truth prior for this sample and move to correct device
        gt_prior = q_prior_patch[i, :num_patches].to(device=device, dtype=dtype)

        # Get predicted mask for this sample (first num_patches tokens)
        # Squeeze to 1D for easier computation
        pred_mask = q_mask_pred[i, :num_patches, 0]

        # Apply mask if provided to exclude padding
        if q_prior_mask is not None:
            mask = q_prior_mask[i, :num_patches].to(device)
            gt_prior = gt_prior[mask]
            pred_mask = pred_mask[mask]

        # Compute MSE loss
        # This forces exact value matching at each patch, preventing uniform predictions
        mse_loss = F.mse_loss(pred_mask, gt_prior)

        # Keep gradient connection by accumulating tensors
        batch_losses.append(mse_loss)

    # Average over batch while maintaining gradients
    if len(batch_losses) > 0:
        return torch.stack(batch_losses).mean()
    else:
        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
