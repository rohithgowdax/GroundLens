import pytest
import torch

# Import implementation
from src.model.losses.q_mask_loss import get_q_prior_mse_loss


# Test class
class TestQPriorMSELoss:
    """Test question prior MSE loss computation."""

    def test_perfect_prediction(self):
        """Test loss when prediction exactly matches ground truth."""
        batch_size = 2
        max_patches = 100

        # Create random ground truth patches
        q_prior_patch = torch.rand(batch_size, max_patches)
        # Create matching prediction by copying ground truth
        q_mask_pred = q_prior_patch.unsqueeze(-1).clone()  # Add channel dim
        # Define patch grid dimensions
        q_prior_patch_shape = torch.tensor([[10, 10], [5, 20]])
        # Compute MSE loss
        loss = get_q_prior_mse_loss(
            q_mask_pred=q_mask_pred,
            q_prior_patch=q_prior_patch,
            q_prior_patch_shape=q_prior_patch_shape
        )

        # Verify loss is near zero for perfect match
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_completely_wrong_prediction(self):
        """Test loss when prediction is opposite of ground truth."""
        batch_size = 2
        max_patches = 100

        # Create ground truth of all ones
        q_prior_patch = torch.ones(batch_size, max_patches)
        # Create prediction of all zeros (complete mismatch)
        q_mask_pred = torch.zeros(batch_size, max_patches, 1)
        # Define patch grid dimensions
        q_prior_patch_shape = torch.tensor([[10, 10], [5, 20]])
        # Compute MSE loss
        loss = get_q_prior_mse_loss(
            q_mask_pred=q_mask_pred,
            q_prior_patch=q_prior_patch,
            q_prior_patch_shape=q_prior_patch_shape
        )

        # Verify loss equals 1.0 (MSE of 0 vs 1)
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)

    def test_handles_variable_sequence_lengths(self):
        """Test with different patch counts per sample."""
        batch_size = 3
        max_patches = 200

        # Create random ground truth and predictions
        q_prior_patch = torch.rand(batch_size, max_patches)
        q_mask_pred = torch.rand(batch_size, max_patches, 1)
        # Each sample has different patch grid dimensions
        q_prior_patch_shape = torch.tensor([
            [10, 10],  # 100 patches
            [8, 16],   # 128 patches
            [5, 20],   # 100 patches
        ])
        # Compute loss
        loss = get_q_prior_mse_loss(
            q_mask_pred=q_mask_pred,
            q_prior_patch=q_prior_patch,
            q_prior_patch_shape=q_prior_patch_shape
        )

        # Verify loss is valid and non-negative
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_handles_padding_mask(self):
        """Test that padding mask correctly excludes padded patches."""
        batch_size = 2
        max_patches = 100

        # Create random ground truth and predictions
        q_prior_patch = torch.rand(batch_size, max_patches)
        q_mask_pred = torch.rand(batch_size, max_patches, 1)
        q_prior_patch_shape = torch.tensor([[10, 10], [5, 10]])
        # Create mask marking last 50 patches as padding
        q_prior_mask = torch.ones(batch_size, max_patches, dtype=torch.bool)
        q_prior_mask[:, 50:] = False  # Mark as padding (False = masked)
        # Compute loss with mask applied
        loss_with_mask = get_q_prior_mse_loss(
            q_mask_pred=q_mask_pred,
            q_prior_patch=q_prior_patch,
            q_prior_patch_shape=torch.tensor([[5, 10], [5, 10]]),  # Only 50 patches
            q_prior_mask=q_prior_mask
        )

        # Verify loss is valid (padded patches ignored)
        assert torch.isfinite(loss_with_mask)

    def test_loss_is_differentiable(self):
        """Test that gradients flow through the loss."""
        batch_size = 2
        max_patches = 100

        # Create ground truth
        q_prior_patch = torch.rand(batch_size, max_patches)
        # Create prediction with gradient tracking enabled
        q_mask_pred = torch.rand(batch_size, max_patches, 1, requires_grad=True)
        q_prior_patch_shape = torch.tensor([[10, 10], [5, 20]])
        # Compute loss
        loss = get_q_prior_mse_loss(
            q_mask_pred=q_mask_pred,
            q_prior_patch=q_prior_patch,
            q_prior_patch_shape=q_prior_patch_shape
        )

        # Backpropagate gradients
        loss.backward()

        # Verify gradients exist and are finite
        assert q_mask_pred.grad is not None
        assert torch.isfinite(q_mask_pred.grad).all()

    def test_raises_error_on_shape_mismatch(self):
        """Test that error is raised when shapes don't match."""
        batch_size = 2
        max_patches = 100

        # Create prediction with insufficient patches
        q_mask_pred = torch.rand(batch_size, 50, 1)  # Only 50 patches
        q_prior_patch = torch.rand(batch_size, max_patches)
        # Define shape requiring 100 patches
        q_prior_patch_shape = torch.tensor([[10, 10], [10, 10]])  # Requires 100 patches
        # Verify error is raised for shape mismatch
        with pytest.raises(ValueError, match="Ground truth has .* patches .* but predicted mask only has"):
            get_q_prior_mse_loss(
                q_mask_pred=q_mask_pred,
                q_prior_patch=q_prior_patch,
                q_prior_patch_shape=q_prior_patch_shape
            )

    def test_batch_averaging(self):
        """Test that loss is properly averaged across batch."""
        batch_size = 4
        max_patches = 100

        q_prior_patch = torch.rand(batch_size, max_patches)
        q_mask_pred = torch.rand(batch_size, max_patches, 1)
        q_prior_patch_shape = torch.tensor([[10, 10]] * batch_size)

        loss = get_q_prior_mse_loss(
            q_mask_pred=q_mask_pred,
            q_prior_patch=q_prior_patch,
            q_prior_patch_shape=q_prior_patch_shape
        )

        # Loss should be scalar (averaged)
        assert loss.shape == torch.Size([])

    def test_device_handling(self):
        """Test that loss works with tensors on different devices."""
        # Skip test if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch_size = 2
        max_patches = 100

        # Create prediction on GPU, ground truth on CPU
        q_mask_pred = torch.rand(batch_size, max_patches, 1).cuda()
        q_prior_patch = torch.rand(batch_size, max_patches)  # CPU
        q_prior_patch_shape = torch.tensor([[10, 10], [5, 20]])  # CPU
        # Compute loss (should handle device transfer)
        loss = get_q_prior_mse_loss(
            q_mask_pred=q_mask_pred,
            q_prior_patch=q_prior_patch,
            q_prior_patch_shape=q_prior_patch_shape
        )

        # Verify loss is on same device as predictions
        assert loss.device == q_mask_pred.device
        assert torch.isfinite(loss)

    def test_edge_case_single_patch(self):
        """Test with minimal patch count."""
        batch_size = 1
        max_patches = 10

        q_prior_patch = torch.rand(batch_size, max_patches)
        q_mask_pred = torch.rand(batch_size, max_patches, 1)
        q_prior_patch_shape = torch.tensor([[1, 1]])  # Just 1 patch

        loss = get_q_prior_mse_loss(
            q_mask_pred=q_mask_pred,
            q_prior_patch=q_prior_patch,
            q_prior_patch_shape=q_prior_patch_shape
        )

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_edge_case_maximum_patches(self):
        """Test with large patch count."""
        batch_size = 2
        max_patches = 2048  # Typical max for models

        q_prior_patch = torch.rand(batch_size, max_patches)
        q_mask_pred = torch.rand(batch_size, max_patches, 1)
        q_prior_patch_shape = torch.tensor([[32, 64], [64, 32]])

        loss = get_q_prior_mse_loss(
            q_mask_pred=q_mask_pred,
            q_prior_patch=q_prior_patch,
            q_prior_patch_shape=q_prior_patch_shape
        )

        assert torch.isfinite(loss)
        assert loss >= 0
