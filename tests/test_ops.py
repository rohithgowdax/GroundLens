import pytest
import numpy as np
import torch
from PIL import Image

# Import implementations 
from src.datasets.ops import (
    bbox_to_int,
    clamp_bbox,
    resize_for_readability,
    crop_with_margin,
    resize_img,
    xyxy_to_cxcywh_norm,
    normalize_answers,
)

# Test classes
class TestBboxToInt:
    """Test conversion from normalized to absolute pixel coordinates."""

    def test_basic_conversion(self):
        """Test basic normalized to absolute conversion."""
        bbox = [0.25, 0.5, 0.75, 1.0]  # Normalized coordinates
        W, H = 800, 600

        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)

        assert x1 == 200  # 0.25 * 800
        assert y1 == 300  # 0.5 * 600
        assert x2 == 600  # 0.75 * 800
        assert y2 == 600  # 1.0 * 600

    def test_zero_coordinates(self):
        """Test with zero coordinates (top-left corner)."""
        bbox = [0.0, 0.0, 0.1, 0.1]
        W, H = 1000, 1000

        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)

        assert x1 == 0
        assert y1 == 0
        assert x2 == 100
        assert y2 == 100

    def test_full_image(self):
        """Test with full image coordinates."""
        bbox = [0.0, 0.0, 1.0, 1.0]
        W, H = 640, 480

        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)

        assert x1 == 0
        assert y1 == 0
        assert x2 == 640
        assert y2 == 480

    def test_center_box(self):
        """Test with center box."""
        bbox = [0.25, 0.25, 0.75, 0.75]
        W, H = 1000, 1000

        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)

        assert x1 == 250
        assert y1 == 250
        assert x2 == 750
        assert y2 == 750

    def test_different_aspect_ratios(self):
        """Test with various aspect ratios."""
        bbox = [0.0, 0.0, 1.0, 1.0]

        # Wide image
        x1, y1, x2, y2 = bbox_to_int(bbox, 1920, 1080)
        assert x2 == 1920 and y2 == 1080

        # Tall image
        x1, y1, x2, y2 = bbox_to_int(bbox, 600, 800)
        assert x2 == 600 and y2 == 800

        # Square image
        x1, y1, x2, y2 = bbox_to_int(bbox, 500, 500)
        assert x2 == 500 and y2 == 500

    def test_small_box(self):
        """Test with very small box."""
        bbox = [0.001, 0.001, 0.002, 0.002]
        W, H = 10000, 10000

        x1, y1, x2, y2 = bbox_to_int(bbox, W, H)

        assert x1 == 10
        assert y1 == 10
        assert x2 == 20
        assert y2 == 20


class TestClampBbox:
    """Test bounding box clamping to image boundaries."""

    def test_valid_box_unchanged(self):
        """Test that valid box remains unchanged."""
        W, H = 800, 600
        x1, y1, x2, y2 = clamp_bbox(100, 100, 700, 500, W, H)

        assert x1 == 100
        assert y1 == 100
        assert x2 == 700
        assert y2 == 500

    def test_out_of_bounds_clamping(self):
        """Test clamping coordinates outside image bounds."""
        W, H = 800, 600

        # Box extends beyond right and bottom
        x1, y1, x2, y2 = clamp_bbox(700, 500, 1000, 800, W, H)

        assert x2 == W  # Clamped to 800
        assert y2 == H  # Clamped to 600

    def test_negative_coordinates(self):
        """Test clamping negative coordinates."""
        W, H = 800, 600

        x1, y1, x2, y2 = clamp_bbox(-50, -30, 100, 100, W, H)

        assert x1 == 0  # Clamped to 0
        assert y1 == 0  # Clamped to 0
        assert x2 == 100
        assert y2 == 100

    def test_reversed_coordinates(self):
        """Test that reversed coordinates are sorted."""
        W, H = 800, 600

        # x2 < x1, y2 < y1
        x1, y1, x2, y2 = clamp_bbox(700, 500, 100, 100, W, H)

        # Should be sorted
        assert x1 == 100
        assert x2 == 700
        assert y1 == 100
        assert y2 == 500

    def test_zero_area_box_prevention(self):
        """Test that boxes with zero area are expanded to at least 1px."""
        W, H = 800, 600

        # Same x coordinates
        x1, y1, x2, y2 = clamp_bbox(400, 300, 400, 500, W, H)
        assert x2 > x1  # At least 1px width

        # Same y coordinates
        x1, y1, x2, y2 = clamp_bbox(300, 400, 500, 400, W, H)
        assert y2 > y1  # At least 1px height

    def test_edge_cases_at_boundaries(self):
        """Test boxes at exact image boundaries."""
        W, H = 800, 600

        # Box at right edge
        x1, y1, x2, y2 = clamp_bbox(799, 0, 800, 600, W, H)
        assert x1 == 799
        assert x2 == 800

        # Box at bottom edge
        x1, y1, x2, y2 = clamp_bbox(0, 599, 800, 600, W, H)
        assert y1 == 599
        assert y2 == 600

    def test_tiny_image(self):
        """Test with very small image dimensions."""
        W, H = 2, 2

        x1, y1, x2, y2 = clamp_bbox(0, 0, 2, 2, W, H)

        assert x1 == 0
        assert y1 == 0
        assert x2 == 2
        assert y2 == 2


class TestXyxyToCxcywhNorm:
    """Test conversion from xyxy to normalized cxcywh format."""

    def test_basic_conversion(self):
        """Test basic xyxy to cxcywh conversion."""
        W, H = 1000, 1000
        x1, y1, x2, y2 = 200, 300, 600, 700

        cx, cy, w, h = xyxy_to_cxcywh_norm(x1, y1, x2, y2, W, H)

        # Expected: center at (400, 500), size (400, 400)
        assert cx == pytest.approx(0.4, abs=1e-5)
        assert cy == pytest.approx(0.5, abs=1e-5)
        assert w == pytest.approx(0.4, abs=1e-5)
        assert h == pytest.approx(0.4, abs=1e-5)

    def test_full_image_box(self):
        """Test with full image box."""
        W, H = 800, 600
        x1, y1, x2, y2 = 0, 0, 800, 600

        cx, cy, w, h = xyxy_to_cxcywh_norm(x1, y1, x2, y2, W, H)

        assert cx == pytest.approx(0.5, abs=1e-5)
        assert cy == pytest.approx(0.5, abs=1e-5)
        assert w == pytest.approx(1.0, abs=1e-5)
        assert h == pytest.approx(1.0, abs=1e-5)

    def test_corner_boxes(self):
        """Test boxes at different corners."""
        W, H = 1000, 1000

        # Top-left corner
        cx, cy, w, h = xyxy_to_cxcywh_norm(0, 0, 200, 200, W, H)
        assert cx == pytest.approx(0.1, abs=1e-5)
        assert cy == pytest.approx(0.1, abs=1e-5)

        # Bottom-right corner
        cx, cy, w, h = xyxy_to_cxcywh_norm(800, 800, 1000, 1000, W, H)
        assert cx == pytest.approx(0.9, abs=1e-5)
        assert cy == pytest.approx(0.9, abs=1e-5)

    def test_zero_width_box(self):
        """Test that zero-width boxes get minimum width of 1px."""
        W, H = 1000, 1000
        # Same x coordinates
        cx, cy, w, h = xyxy_to_cxcywh_norm(500, 300, 500, 700, W, H)

        assert w >= 0.001  # Should have minimum width

    def test_zero_height_box(self):
        """Test that zero-height boxes get minimum height of 1px."""
        W, H = 1000, 1000
        # Same y coordinates
        cx, cy, w, h = xyxy_to_cxcywh_norm(300, 500, 700, 500, W, H)

        assert h >= 0.001  # Should have minimum height

    def test_normalized_range(self):
        """Test that all outputs are in [0, 1] range."""
        W, H = 640, 480
        x1, y1, x2, y2 = 100, 50, 540, 430

        cx, cy, w, h = xyxy_to_cxcywh_norm(x1, y1, x2, y2, W, H)

        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0

    def test_non_square_image(self):
        """Test with non-square image dimensions."""
        # Wide image
        W, H = 1920, 1080
        x1, y1, x2, y2 = 960, 540, 1440, 810

        cx, cy, w, h = xyxy_to_cxcywh_norm(x1, y1, x2, y2, W, H)

        assert cx == pytest.approx(0.625, abs=1e-3)  # (960 + 240) / 1920
        assert cy == pytest.approx(0.625, abs=1e-3)  # (540 + 135) / 1080
        assert w == pytest.approx(0.25, abs=1e-3)    # 480 / 1920
        assert h == pytest.approx(0.25, abs=1e-3)    # 270 / 1080


class TestCropWithMargin:
    """Test image cropping with margin."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL image for testing."""
        return Image.new('RGB', (1000, 1000), color='white')

    def test_basic_crop(self, sample_image):
        """Test basic crop with margin."""
        x1, y1, x2, y2 = 300, 300, 700, 700
        crop_margin = 0.1

        cropped = crop_with_margin(sample_image, x1, y1, x2, y2, crop_margin)

        # Box size: 400x400, margin: 40px on each side
        # Expected crop: 260x260 to 740x740 = 480x480
        assert cropped.size == (480, 480)

    def test_zero_margin(self, sample_image):
        """Test crop with zero margin."""
        x1, y1, x2, y2 = 200, 200, 600, 600

        cropped = crop_with_margin(sample_image, x1, y1, x2, y2, crop_margin=0.0)

        # Should crop exactly to the box
        assert cropped.size == (400, 400)

    def test_crop_at_image_edge(self, sample_image):
        """Test that crop doesn't extend beyond image bounds."""
        x1, y1, x2, y2 = 0, 0, 200, 200
        crop_margin = 0.5  # 50% margin

        cropped = crop_with_margin(sample_image, x1, y1, x2, y2, crop_margin)

        # Even with margin, should not exceed image bounds
        w, h = cropped.size
        assert w <= sample_image.size[0]
        assert h <= sample_image.size[1]

    def test_large_margin(self, sample_image):
        """Test with very large margin."""
        x1, y1, x2, y2 = 400, 400, 600, 600
        crop_margin = 2.0  # 200% margin

        cropped = crop_with_margin(sample_image, x1, y1, x2, y2, crop_margin)

        # Should be clamped to image boundaries
        w, h = cropped.size
        assert w <= sample_image.size[0]
        assert h <= sample_image.size[1]

    def test_crop_near_right_edge(self, sample_image):
        """Test crop near right/bottom edge."""
        x1, y1, x2, y2 = 800, 800, 1000, 1000
        crop_margin = 0.1

        cropped = crop_with_margin(sample_image, x1, y1, x2, y2, crop_margin)

        # Should be clamped at image boundaries
        w, h = cropped.size
        assert w <= 1000
        assert h <= 1000


class TestResizeForReadability:
    """Test image resizing for readability."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample PIL image for testing."""
        return Image.new('RGB', (1000, 800), color='white')

    def test_upscale_small_image(self):
        """Test that small images are upscaled, but max_width takes priority."""
        # 500x100 upscaled to min_height=256 would be 1280x256 (aspect 5:1)
        # But max_width=700 caps it, so final is 700x140
        small_img = Image.new('RGB', (500, 100), color='white')
        min_height = 256
        max_width = 700

        resized = resize_for_readability(small_img, max_width=max_width, min_height=min_height)

        # Width should be at max_width (function prioritizes max_width over min_height)
        assert resized.size[0] == max_width
        # Height will be less than min_height due to width constraint
        assert resized.size[1] == 140  # 700 / 5 = 140 (preserving 5:1 aspect ratio)

    def test_upscale_meets_min_height(self):
        """Test that small images can meet min_height when they don't exceed max_width."""
        # 200x100 upscaled to min_height=256 would be 512x256 (2:1 aspect ratio)
        # This is within max_width=700, so min_height is satisfied
        small_img = Image.new('RGB', (200, 100), color='white')
        min_height = 256
        max_width = 700

        resized = resize_for_readability(small_img, max_width=max_width, min_height=min_height)

        # Should meet min_height since width doesn't exceed max_width
        assert resized.size[1] == min_height
        assert resized.size[0] == 512  # 2:1 aspect ratio preserved

    def test_downscale_wide_image(self):
        """Test that very wide images are capped at max_width."""
        wide_img = Image.new('RGB', (2000, 500), color='white')
        max_width = 700

        resized = resize_for_readability(wide_img, max_width=max_width, min_height=256)

        # Width should be capped
        assert resized.size[0] <= max_width

    def test_preserve_aspect_ratio(self):
        """Test that aspect ratio is preserved."""
        img = Image.new('RGB', (1000, 500), color='white')
        original_aspect = 1000 / 500  # 2.0

        resized = resize_for_readability(img, max_width=700, min_height=256)

        new_aspect = resized.size[0] / resized.size[1]
        assert abs(new_aspect - original_aspect) < 0.1  # Allow small floating point error

    def test_already_optimal_size(self):
        """Test image that's already in optimal size range."""
        img = Image.new('RGB', (600, 400), color='white')

        resized = resize_for_readability(img, max_width=700, min_height=256)

        # Should stay roughly the same size (meets constraints)
        assert 256 <= resized.size[1]
        assert resized.size[0] <= 700

    def test_minimum_dimensions(self):
        """Test that output is always at least 1x1."""
        tiny_img = Image.new('RGB', (1, 1), color='white')

        resized = resize_for_readability(tiny_img, max_width=700, min_height=256)

        # Should still have valid dimensions
        assert resized.size[0] >= 1
        assert resized.size[1] >= 1


class TestResizeImg:
    """Test basic image resizing maintaining aspect ratio."""

    def test_basic_resize(self):
        """Test basic resize to max_width."""
        img = Image.new('RGB', (1000, 500), color='white')
        max_width = 700

        resized = resize_img(img, max_width=max_width)

        assert resized.size[0] == max_width
        # Height should scale proportionally: 500 * (700/1000) = 350
        assert resized.size[1] == 350

    def test_preserve_aspect_ratio(self):
        """Test aspect ratio preservation."""
        img = Image.new('RGB', (800, 600), color='white')
        original_aspect = 800 / 600

        resized = resize_img(img, max_width=400)

        new_aspect = resized.size[0] / resized.size[1]
        assert abs(new_aspect - original_aspect) < 0.01

    def test_square_image(self):
        """Test resizing square image."""
        img = Image.new('RGB', (1000, 1000), color='white')

        resized = resize_img(img, max_width=500)

        # Should remain square
        assert resized.size[0] == 500
        assert resized.size[1] == 500

    def test_tall_image(self):
        """Test resizing tall image."""
        img = Image.new('RGB', (400, 800), color='white')

        resized = resize_img(img, max_width=200)

        assert resized.size[0] == 200
        # Height should double since width halved
        assert resized.size[1] == 400

    def test_already_smaller(self):
        """Test image already smaller than max_width."""
        img = Image.new('RGB', (500, 300), color='white')

        resized = resize_img(img, max_width=1000)

        # Should upscale to max_width
        assert resized.size[0] == 1000
        assert resized.size[1] == 600

class TestNormalizeAnswers:
    """Test answer normalization."""

    def test_string_answers(self):
        """Test with string answers."""
        answers = ["answer1", "answer2", "answer3"]

        first_answers, all_answers = normalize_answers(answers)

        assert first_answers == ["answer1", "answer2", "answer3"]
        assert all_answers == [["answer1"], ["answer2"], ["answer3"]]

    def test_list_answers(self):
        """Test with list answers (returns first + all variants)."""
        answers = [["answer1", "alt1"], ["answer2", "alt2"]]

        first_answers, all_answers = normalize_answers(answers)

        assert first_answers == ["answer1", "answer2"]
        assert all_answers == [["answer1", "alt1"], ["answer2", "alt2"]]

    def test_empty_list(self):
        """Test with empty lists."""
        answers = [[], ["answer"]]

        first_answers, all_answers = normalize_answers(answers)

        assert first_answers == ["", "answer"]
        assert all_answers == [[""], ["answer"]]

    def test_mixed_types(self):
        """Test with mixed string and list types."""
        answers = ["simple", ["list", "answer"], "another"]

        first_answers, all_answers = normalize_answers(answers)

        assert first_answers == ["simple", "list", "another"]
        assert all_answers == [["simple"], ["list", "answer"], ["another"]]

    def test_numeric_answers(self):
        """Test with numeric answers."""
        answers = [42, 3.14, "text"]

        first_answers, all_answers = normalize_answers(answers)

        assert first_answers == ["42", "3.14", "text"]
        assert all_answers == [["42"], ["3.14"], ["text"]]

    def test_tuple_answers(self):
        """Test with tuple answers."""
        answers = [("answer1", "alt"), "single"]

        first_answers, all_answers = normalize_answers(answers)

        assert first_answers == ["answer1", "single"]
        assert all_answers == [["answer1", "alt"], ["single"]]

    def test_empty_input(self):
        """Test with empty input list."""
        answers = []

        first_answers, all_answers = normalize_answers(answers)

        assert first_answers == []
        assert all_answers == []
