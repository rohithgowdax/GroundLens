import pytest
import torch

# Import implementations
from src.model.losses.projection_loss import get_projection_loss, pred_norm_cxcywh_to_xyxy, safe_iou_loss

# Test classes

class TestPredNormCxcywhToXyxy:
    """Test box format conversion and sanitization."""

    def test_basic_conversion(self):
        """Test basic cxcywh to xyxy conversion."""
        # Create centered box in cxcywh format
        pred = torch.tensor([[0.5, 0.5, 0.2, 0.3]])  # Center box
        xyxy, wh = pred_norm_cxcywh_to_xyxy(pred)

        # Expected: cx=0.5, cy=0.5, w=0.2, h=0.3
        # xyxy: x1=0.4, y1=0.35, x2=0.6, y2=0.65
        # Verify output shape
        assert xyxy.shape == (1, 4)
        # Verify each coordinate matches expected values
        assert torch.allclose(xyxy[0, 0], torch.tensor(0.4), atol=1e-5)
        assert torch.allclose(xyxy[0, 1], torch.tensor(0.35), atol=1e-5)
        assert torch.allclose(xyxy[0, 2], torch.tensor(0.6), atol=1e-5)
        assert torch.allclose(xyxy[0, 3], torch.tensor(0.65), atol=1e-5)

    def test_clamps_to_valid_range(self):
        """Test that out-of-bounds predictions are clamped."""
        # Create predictions with out-of-bounds values
        pred = torch.tensor([
            [-0.1, 1.2, 0.5, 0.5],  # Center out of bounds
            [0.5, 0.5, 1.5, 2.0],   # Size too large
        ])
        xyxy, wh = pred_norm_cxcywh_to_xyxy(pred)

        # Verify all coordinates are in valid [0, 1] range
        assert (xyxy >= 0.0).all()
        assert (xyxy <= 1.0).all()
        # Verify width and height are positive
        assert (wh > 0.0).all()

    def test_prevents_degenerate_boxes(self):
        """Test that zero or negative sizes are prevented."""
        eps = 1e-6
        # Create boxes with invalid dimensions
        pred = torch.tensor([
            [0.5, 0.5, 0.0, 0.0],      # Zero size
            [0.5, 0.5, -0.1, -0.1],    # Negative size
        ])
        xyxy, wh = pred_norm_cxcywh_to_xyxy(pred, eps=eps)

        # Width and height should be at least eps
        assert (wh >= eps).all()
        # Verify boxes have positive area (x2 > x1 and y2 > y1)
        assert (xyxy[:, 2] > xyxy[:, 0]).all()
        assert (xyxy[:, 3] > xyxy[:, 1]).all()

    def test_corner_cases(self):
        """Test boxes at image corners."""
        # Create boxes centered at each image corner
        pred = torch.tensor([
            [0.0, 0.0, 0.1, 0.1],      # Top-left corner
            [1.0, 1.0, 0.1, 0.1],      # Bottom-right corner
            [0.0, 1.0, 0.1, 0.1],      # Bottom-left corner
            [1.0, 0.0, 0.1, 0.1],      # Top-right corner
        ])
        xyxy, wh = pred_norm_cxcywh_to_xyxy(pred)

        # Verify all coordinates within valid range
        assert (xyxy >= 0.0).all()
        assert (xyxy <= 1.0).all()
        # Verify boxes have positive area
        assert (xyxy[:, 2] > xyxy[:, 0]).all()
        assert (xyxy[:, 3] > xyxy[:, 1]).all()


class TestSafeIouLoss:
    """Test safe IoU loss computation."""

    def test_perfect_overlap(self):
        """Test IoU loss for perfectly overlapping boxes."""
        box = torch.tensor([[0.2, 0.2, 0.8, 0.8]])
        loss = safe_iou_loss(box, box, mode="ciou")
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_no_overlap(self):
        """Test IoU loss for non-overlapping boxes."""
        pred = torch.tensor([[0.1, 0.1, 0.3, 0.3]])
        gt = torch.tensor([[0.7, 0.7, 0.9, 0.9]])
        loss = safe_iou_loss(pred, gt, mode="ciou")

        # Loss should be positive (no overlap = high loss)
        assert loss > 0.5

    def test_handles_nans(self):
        """Test that NaN/Inf values are properly sanitized."""
        # Create degenerate box that could produce NaN
        pred = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # Degenerate box
        gt = torch.tensor([[0.2, 0.2, 0.8, 0.8]])
        # Compute loss
        loss = safe_iou_loss(pred, gt, mode="ciou")

        # Verify loss is finite (not NaN or Inf)
        assert torch.isfinite(loss)

    def test_different_iou_modes(self):
        """Test different IoU loss modes."""
        # Create partially overlapping boxes
        pred = torch.tensor([[0.3, 0.3, 0.7, 0.7]])
        gt = torch.tensor([[0.4, 0.4, 0.8, 0.8]])
        # Compute losses with different IoU variants
        loss_ciou = safe_iou_loss(pred, gt, mode="ciou")
        loss_diou = safe_iou_loss(pred, gt, mode="diou")
        loss_giou = safe_iou_loss(pred, gt, mode="giou")

        # Verify all losses are finite and non-negative
        assert torch.isfinite(loss_ciou)
        assert torch.isfinite(loss_diou)
        assert torch.isfinite(loss_giou)
        assert loss_ciou >= 0
        assert loss_diou >= 0
        assert loss_giou >= 0


class TestGetProjectionLoss:
    """Test complete projection loss computation."""

    def test_loss_with_perfect_prediction(self):
        """Test loss when prediction exactly matches ground truth."""
        # Create ground truth box in absolute coordinates
        gt_xyxy = torch.tensor([[200, 150, 600, 450]])
        # Create matching prediction in normalized cxcywh format
        pred_cxcywh = torch.tensor([[0.4, 0.3, 0.4, 0.3]])  # Same box in normalized cxcywh
        img_sizes = torch.tensor([[1000, 1000]])
        # Compute loss with all components enabled
        result = get_projection_loss(
            gt_xyxy, pred_cxcywh, img_sizes,
            lambda_iou=1.0, lambda_size=1.0, lambda_center=1.0,
            lambda_aspect=1.0, lambda_edges=1.0, lambda_area=1.0
        )

        # Verify all losses are near zero for perfect match
        assert result["loss_iou"] < 0.1
        assert result["loss_center"] < 0.01
        assert result["loss_total"] < 0.5

    def test_loss_components_present(self):
        """Test that all loss components are returned."""
        # Create ground truth and prediction
        gt_xyxy = torch.tensor([[200, 150, 600, 450]])
        pred_cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        img_sizes = torch.tensor([[1000, 1000]])
        # Compute loss
        result = get_projection_loss(gt_xyxy, pred_cxcywh, img_sizes)

        # Verify all expected loss components are present
        expected_keys = [
            "loss_total", "loss_iou", "loss_center",
            "loss_size", "loss_edges", "loss_aspect", "loss_area"
        ]
        for key in expected_keys:
            assert key in result

    def test_lambda_weighting(self):
        """Test that lambda weights properly scale loss components."""
        # Create ground truth and prediction
        gt_xyxy = torch.tensor([[200, 150, 600, 450]])
        pred_cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
        img_sizes = torch.tensor([[1000, 1000]])

        # Compute loss with only IoU component enabled
        result_iou_only = get_projection_loss(
            gt_xyxy, pred_cxcywh, img_sizes,
            lambda_iou=1.0, lambda_size=0.0, lambda_center=0.0,
            lambda_aspect=0.0, lambda_edges=0.0, lambda_area=0.0
        )

        # Compute loss with only center component enabled
        result_center_only = get_projection_loss(
            gt_xyxy, pred_cxcywh, img_sizes,
            lambda_iou=0.0, lambda_size=0.0, lambda_center=1.0,
            lambda_aspect=0.0, lambda_edges=0.0, lambda_area=0.0
        )

        # Verify different weights produce different losses
        assert not torch.allclose(result_iou_only["loss_total"], result_center_only["loss_total"])

    def test_batch_processing(self):
        """Test loss computation with batch of boxes."""
        batch_size = 4
        # Create batch of ground truth boxes
        gt_xyxy = torch.tensor([
            [200, 150, 600, 450],
            [100, 100, 300, 250],
            [500, 500, 900, 900],
            [50, 50, 150, 100],
        ])
        # Create batch of random predictions
        pred_cxcywh = torch.rand(batch_size, 4)
        pred_cxcywh = torch.clamp(pred_cxcywh, 0.0, 1.0)
        img_sizes = torch.tensor([[1000, 1000]] * batch_size)
        # Compute loss
        result = get_projection_loss(gt_xyxy, pred_cxcywh, img_sizes)

        # Verify loss is scalar (averaged over batch)
        assert result["loss_total"].shape == torch.Size([])
        assert torch.isfinite(result["loss_total"])

    def test_handles_different_image_sizes(self):
        """Test with varying image dimensions."""
        # Create ground truth boxes for different image sizes
        gt_xyxy = torch.tensor([
            [200, 150, 600, 450],
            [100, 50, 300, 200],
        ])
        pred_cxcywh = torch.tensor([
            [0.5, 0.5, 0.3, 0.3],
            [0.4, 0.3, 0.2, 0.25],
        ])
        # Different image dimensions for each sample
        img_sizes = torch.tensor([
            [1000, 1000],
            [800, 600],
        ])
        # Compute loss
        result = get_projection_loss(gt_xyxy, pred_cxcywh, img_sizes)

        # Verify loss is valid
        assert torch.isfinite(result["loss_total"])
        assert result["loss_total"] >= 0

    def test_loss_is_differentiable(self):
        """Test that loss computation maintains gradients."""
        # Create ground truth
        gt_xyxy = torch.tensor([[200.0, 150.0, 600.0, 450.0]])
        # Create prediction with gradient tracking enabled
        pred_cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.3]], requires_grad=True)
        img_sizes = torch.tensor([[1000.0, 1000.0]])
        # Compute loss
        result = get_projection_loss(gt_xyxy, pred_cxcywh, img_sizes)
        loss = result["loss_total"]

        # Backpropagate gradients
        loss.backward()

        # Verify gradients exist and are finite
        assert pred_cxcywh.grad is not None
        assert torch.isfinite(pred_cxcywh.grad).all()

    def test_handles_edge_case_tiny_boxes(self):
        """Test handling of very small predicted boxes."""
        # Create normal ground truth box
        gt_xyxy = torch.tensor([[200, 150, 600, 450]])
        # Create very small predicted box
        pred_cxcywh = torch.tensor([[0.5, 0.5, 0.001, 0.001]])  # Tiny box
        img_sizes = torch.tensor([[1000, 1000]])
        # Compute loss with epsilon clamping
        result = get_projection_loss(gt_xyxy, pred_cxcywh, img_sizes, eps=1e-6)

        # Verify loss is finite (no NaN from division by zero)
        assert torch.isfinite(result["loss_total"])
        # Verify high loss for poor prediction
        assert result["loss_total"] > 0

    def test_handles_edge_case_large_boxes(self):
        """Test handling of boxes that extend beyond image bounds."""
        # Create normal ground truth box
        gt_xyxy = torch.tensor([[200, 150, 600, 450]])
        # Create oversized predicted box
        pred_cxcywh = torch.tensor([[0.5, 0.5, 1.5, 1.5]])  # Too large
        img_sizes = torch.tensor([[1000, 1000]])
        # Compute loss
        result = get_projection_loss(gt_xyxy, pred_cxcywh, img_sizes)

        # Verify loss is finite after clamping
        assert torch.isfinite(result["loss_total"])
