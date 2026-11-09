import pytest
import torch

# Import the prediction head implementations
from src.model.blocks.box_prediction_head import BoxPredictionHead


# Test class for testing the prediction head  
class TestBoxPredictionHead:
    def test_forward_pass(self, sample_embeddings):
        """Test basic forward pass."""
        d_model = sample_embeddings.shape[-1]
        batch_size = sample_embeddings.shape[0]

        head = BoxPredictionHead(d_model=d_model)
        boxes = head(sample_embeddings)

        # Should output (batch_size, 4) for cxcywh
        assert boxes.shape == (batch_size, 4)

    def test_output_in_valid_range(self, sample_embeddings):
        """Test that all outputs are in [0, 1] range."""
        d_model = sample_embeddings.shape[-1]

        head = BoxPredictionHead(d_model=d_model)
        boxes = head(sample_embeddings)

        # All coordinates should be in [0, 1]
        assert (boxes >= 0.0).all()
        assert (boxes <= 1.0).all()

    def test_width_height_nonzero(self, sample_embeddings):
        """Test that width and height are always positive."""
        d_model = sample_embeddings.shape[-1]

        head = BoxPredictionHead(d_model=d_model, min_box_size=1e-4)
        boxes = head(sample_embeddings)

        # Width (index 2) and height (index 3) should be > min_box_size
        assert (boxes[:, 2] >= 1e-4).all()
        assert (boxes[:, 3] >= 1e-4).all()

    def test_initialization_bias(self):
        """Test that initialization biases towards center with specified size."""
        d_model = 768
        init_w, init_h = 0.07, 0.05

        head = BoxPredictionHead(
            d_model=d_model,
            init_w=init_w,
            init_h=init_h
        )

        # Create zero input to test initialization
        x = torch.zeros(1, 100, d_model)
        boxes = head(x)

        # Due to initialization, should be close to (0.5, 0.5, init_w, init_h)
        cx, cy, w, h = boxes[0]

        # Center should be near 0.5
        assert 0.3 < cx < 0.7
        assert 0.3 < cy < 0.7

    def test_different_spatial_attention_heads(self, sample_embeddings):
        """Test with different number of spatial attention heads."""
        d_model = sample_embeddings.shape[-1]

        for num_heads in [4, 8, 12, 16]:
            if d_model % num_heads != 0:
                continue

            head = BoxPredictionHead(
                d_model=d_model,
                spatial_attn_heads=num_heads
            )
            boxes = head(sample_embeddings)

            # Should give the same shape of the embeddings
            assert boxes.shape == (sample_embeddings.shape[0], 4)

    def test_consistent_output_for_same_input(self, sample_embeddings):
        """Test that same input produces same output"""
        d_model = sample_embeddings.shape[-1]

        # Set to eval
        head = BoxPredictionHead(d_model=d_model)
        head.eval() 

        # Predict
        boxes1 = head(sample_embeddings)
        boxes2 = head(sample_embeddings)

        # Check that the values are close
        assert torch.allclose(boxes1, boxes2)

    def test_gradient_flow(self, sample_embeddings):
        """Test that gradients flow through the head."""
        d_model = sample_embeddings.shape[-1]

        # Set gradient to true
        x = sample_embeddings.clone().requires_grad_(True)

        # Predict
        head = BoxPredictionHead(d_model=d_model)
        boxes = head(x)
        loss = boxes.sum()
        loss.backward()

        # Check losses
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_batch_independence(self):
        """Test that batch items are processed independently."""
        d_model = 768
        batch_size = 4
        seq_len = 100

        # Create and set to eval mode
        head = BoxPredictionHead(d_model=d_model)
        head.eval()

        # Process batch
        x_batch = torch.randn(batch_size, seq_len, d_model)
        boxes_batch = head(x_batch)

        # Process individually
        boxes_individual = []
        for i in range(batch_size):
            box = head(x_batch[i:i+1])
            boxes_individual.append(box)
        boxes_individual = torch.cat(boxes_individual, dim=0)

        # Should be identical
        assert torch.allclose(boxes_batch, boxes_individual, atol=1e-6)

    def test_variable_sequence_lengths(self):
        """Test with different sequence lengths."""
        d_model = 768
        batch_size = 2

        head = BoxPredictionHead(d_model=d_model)

        # Different sequence lengths
        for seq_len in [50, 100, 200, 512]:
            x = torch.randn(batch_size, seq_len, d_model)
            boxes = head(x)

            # validate the box 
            assert boxes.shape == (batch_size, 4)
            assert (boxes >= 0.0).all()
            assert (boxes <= 1.0).all()

    def test_extreme_inputs(self):
        """Test with extreme input values."""
        d_model = 768
        batch_size = 2
        seq_len = 100

        # Create heads
        head = BoxPredictionHead(d_model=d_model)

        # Very large values
        x_large = torch.randn(batch_size, seq_len, d_model) * 100
        boxes_large = head(x_large)
        assert (boxes_large >= 0.0).all() and (boxes_large <= 1.0).all()

        # Very small values
        x_small = torch.randn(batch_size, seq_len, d_model) * 0.01
        boxes_small = head(x_small)
        assert (boxes_small >= 0.0).all() and (boxes_small <= 1.0).all()

        # All zeros
        x_zero = torch.zeros(batch_size, seq_len, d_model)
        boxes_zero = head(x_zero)
        assert (boxes_zero >= 0.0).all() and (boxes_zero <= 1.0).all()

    def test_different_init_sizes(self):
        """Test with different initial box sizes."""
        d_model = 768
        x = torch.randn(1, 100, d_model)

        # Small initial size
        head_small = BoxPredictionHead(d_model=d_model, init_w=0.01, init_h=0.01)
        boxes_small = head_small(x)

        # Large initial size
        head_large = BoxPredictionHead(d_model=d_model, init_w=0.5, init_h=0.5)
        boxes_large = head_large(x)

        # Both should be valid
        assert (boxes_small >= 0.0).all() and (boxes_small <= 1.0).all()
        assert (boxes_large >= 0.0).all() and (boxes_large <= 1.0).all()
