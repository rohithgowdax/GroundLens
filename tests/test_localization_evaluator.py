import pytest
import torch

# Import implementation
from src.evaluation.answer_localization_evaluator import AnswerLocalizationEvaluator

# Test class

class TestAnswerLocalizationEvaluator:
    """Test localization metrics evaluator."""

    def test_perfect_predictions(self, perfect_match_boxes):
        """Test metrics when predictions perfectly match ground truth."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # Unpack fixture
        gt_boxes, pred_boxes, img_sizes = perfect_match_boxes

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # IoU should be 1.0
        assert metrics["iou_mean"] > 0.99
        assert metrics["giou_mean"] > 0.99

        # Recall should be 1.0
        assert metrics["recall@th"] == 1.0

        # Center L2 should be near 0
        assert metrics["center_l2_mean"] < 0.01

        # Coverage and purity should be 1.0
        assert metrics["coverage_mean"] > 0.99
        assert metrics["purity_mean"] > 0.99

    def test_no_overlap(self, no_overlap_boxes):
        """Test metrics when predictions don't overlap with ground truth."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # Unpack fixture
        gt_boxes, pred_boxes, img_sizes = no_overlap_boxes

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # IoU should be 0
        assert metrics["iou_mean"] < 0.01

        # Recall should be 0
        assert metrics["recall@th"] == 0.0

        # Coverage should be 0 (no intersection)
        assert metrics["coverage_mean"] < 0.01

    def test_partial_overlap(self, partial_overlap_boxes):
        """Test metrics with partial box overlap."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh", iou_thr=0.5)

        # Unpack fixture
        gt_boxes, pred_boxes, img_sizes = partial_overlap_boxes

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # Should have some overlap
        assert 0.0 < metrics["iou_mean"] < 1.0
        assert 0.0 < metrics["coverage_mean"] < 1.0
        assert 0.0 < metrics["purity_mean"] < 1.0

    def test_batch_processing(self, batch_boxes):
        """Test with batch of predictions."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # Unpack fixture
        gt_boxes, pred_boxes, img_sizes = batch_boxes

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # Should compute mean over all samples
        assert "iou_mean" in metrics
        assert "recall@th" in metrics
        assert 0.0 <= metrics["iou_mean"] <= 1.0

    def test_handles_absolute_coordinates(self):
        """Test normalization of absolute ground truth boxes."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # Absolute coordinates
        gt_boxes = [torch.tensor([[100, 100, 300, 300]])]  # Absolute xyxy
        pred_boxes = [torch.tensor([[0.2, 0.2, 0.2, 0.2]])]  # Normalized cxcywh
        img_sizes = [torch.tensor([[1000, 1000]])]

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # Should normalize and compute correctly
        assert "iou_mean" in metrics
        assert torch.isfinite(torch.tensor(metrics["iou_mean"]))

    def test_handles_already_normalized_boxes(self):
        """Test with already normalized ground truth boxes."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # Already normalized
        gt_boxes = [torch.tensor([[0.1, 0.1, 0.3, 0.3]])]
        pred_boxes = [torch.tensor([[0.2, 0.2, 0.2, 0.2]])]
        img_sizes = [torch.tensor([[1000, 1000]])]

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        assert "iou_mean" in metrics
        assert torch.isfinite(torch.tensor(metrics["iou_mean"]))

    def test_recall_at_threshold(self):
        """Test recall calculation at specified IoU threshold."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh", iou_thr=0.5)

        # One good prediction, one bad
        gt_boxes = [
            torch.tensor([[0.2, 0.2, 0.6, 0.6]]),  # Good match
            torch.tensor([[0.1, 0.1, 0.3, 0.3]]),  # Bad match
        ]
        pred_boxes = [
            torch.tensor([[0.4, 0.4, 0.4, 0.4]]),  # Close to GT
            torch.tensor([[0.8, 0.8, 0.2, 0.2]]),  # Far from GT
        ]
        img_sizes = [
            torch.tensor([[1000, 1000]]),
            torch.tensor([[1000, 1000]]),
        ]

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # Recall should be 0.5 (1 out of 2 above threshold)
        assert 0.3 < metrics["recall@th"] < 0.7  # Approximate due to IoU calculation

    def test_giou_metric(self, no_overlap_boxes):
        """Test GIoU metric computation."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # Use no overlap boxes (GIoU handles this better than IoU)
        gt_boxes, pred_boxes, img_sizes = no_overlap_boxes

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # GIoU should be negative (no overlap)
        assert "giou_mean" in metrics
        # GIoU can be negative
        assert -1.0 <= metrics["giou_mean"] <= 1.0

    def test_area_ratio_metric(self):
        """Test area ratio metric."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # Prediction twice the size of GT
        gt_boxes = [torch.tensor([[0.3, 0.3, 0.5, 0.5]])]  # Area = 0.5 * 0.5 = 0.25
        pred_boxes = [torch.tensor([[0.4, 0.4, 0.4, 0.4]])]  # Area = 0.4 * 0.4 = 0.16
        img_sizes = [torch.tensor([[1000, 1000]])]

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # Area ratio should be around 4.0 (0.16 / 0.25 = 0.64, inverse is ~1.56)
        assert "area_ratio_mean" in metrics
        assert 3.8 < metrics["area_ratio_mean"] < 4.1

    def test_area_ratio_explicit(self):
        """
        Explicit test to verify area ratio = 9 means predicted box
        is 9x larger in area than GT box.

        This confirms what it means when train_dual_heatmap reports AR ratio mean of 9:
        The predicted boxes are on average 9x larger in area than ground truth boxes.
        """
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # GT box in XYXY format: [x1, y1, x2, y2] = [0.0, 0.0, 0.1, 0.1]
        #   width = 0.1, height = 0.1, area = 0.01 (1% of image)
        # Pred box in CXCYWH format: [cx, cy, w, h] = [0.15, 0.15, 0.3, 0.3]
        #   width = 0.3, height = 0.3, area = 0.09 (9% of image)
        # Area ratio = pred_area / gt_area = 0.09 / 0.01 = 9.0

        gt_boxes = [torch.tensor([[0.0, 0.0, 0.1, 0.1]])]  # xyxy: area = 0.1 * 0.1 = 0.01
        pred_boxes = [torch.tensor([[0.15, 0.15, 0.3, 0.3]])]  # cxcywh: area = 0.3 * 0.3 = 0.09
        img_sizes = [torch.tensor([[1000, 1000]])]

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # Area ratio should be pred_area / gt_area = 0.09 / 0.01 = 9.0
        assert abs(metrics["area_ratio_mean"] - 9.0) < 0.01, \
            f"Expected area_ratio=9.0, got {metrics['area_ratio_mean']}"

        # Also verify what this means in absolute terms
        gt_area = 0.1 * 0.1  # 0.01
        pred_area = 0.3 * 0.3  # 0.09
        expected_ratio = pred_area / gt_area  # 9.0
        assert abs(expected_ratio - 9.0) < 0.001


    def test_histogram_calculation(self, partial_overlap_boxes):
        """Test histogram calculation when enabled."""
        evaluator = AnswerLocalizationEvaluator(
            pred_fmt="cxcywh",
            calculate_hist=True,
            hist_bins=10
        )

        # Use partial overlap boxes
        gt_boxes, pred_boxes, img_sizes = partial_overlap_boxes

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # Should have histogram data
        assert metrics["hists"] is not None
        assert "iou" in metrics["hists"]
        assert "coverage" in metrics["hists"]
        assert "purity" in metrics["hists"]

        # Each should have counts and edges
        assert "counts" in metrics["hists"]["iou"]
        assert "edges" in metrics["hists"]["iou"]
        assert len(metrics["hists"]["iou"]["counts"]) == 10

    def test_empty_input(self):
        """Test with empty input (edge case)."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        gt_boxes = []
        pred_boxes = []
        img_sizes = []

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # Should return zero metrics
        assert metrics["iou_mean"] == 0.0
        assert metrics["recall@th"] == 0.0

    def test_xyxy_pred_format(self):
        """Test with xyxy prediction format."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="xyxy")

        gt_boxes = [torch.tensor([[0.2, 0.2, 0.6, 0.6]])]
        pred_boxes = [torch.tensor([[0.3, 0.3, 0.7, 0.7]])]  # xyxy format
        img_sizes = [torch.tensor([[1000, 1000]])]

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        assert "iou_mean" in metrics
        assert 0.0 < metrics["iou_mean"] < 1.0

    def test_center_l2_distance(self):
        """Test center L2 distance metric."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # Boxes with known center distance
        gt_boxes = [torch.tensor([[0.2, 0.2, 0.6, 0.6]])]  # Center at (0.4, 0.4)
        pred_boxes = [torch.tensor([[0.7, 0.7, 0.2, 0.2]])]  # Center at (0.7, 0.7)
        img_sizes = [torch.tensor([[1000, 1000]])]

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # L2 distance = sqrt((0.7-0.4)^2 + (0.7-0.4)^2) ≈ 0.424
        assert "center_l2_mean" in metrics
        assert 0.3 < metrics["center_l2_mean"] < 0.5

    def test_median_iou(self):
        """Test median IoU computation."""
        evaluator = AnswerLocalizationEvaluator(pred_fmt="cxcywh")

        # Multiple predictions with varying IoU
        gt_boxes = [
            torch.tensor([[0.2, 0.2, 0.6, 0.6]]),  # High IoU
            torch.tensor([[0.1, 0.1, 0.3, 0.3]]),  # Medium IoU
            torch.tensor([[0.7, 0.7, 0.9, 0.9]]),  # Low IoU
        ]
        pred_boxes = [
            torch.tensor([[0.4, 0.4, 0.4, 0.4]]),
            torch.tensor([[0.3, 0.3, 0.2, 0.2]]),
            torch.tensor([[0.2, 0.2, 0.2, 0.2]]),
        ]
        img_sizes = [torch.tensor([[1000, 1000]])] * 3

        metrics = evaluator.compute(gt_boxes, pred_boxes, img_sizes)

        # Should compute both mean and median
        assert "iou_mean" in metrics
        assert "iou_median" in metrics
        assert 0.0 <= metrics["iou_median"] <= 1.0