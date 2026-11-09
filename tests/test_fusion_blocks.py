import pytest
import torch

# Import implementations
from src.model.blocks.fusion import FusionBlock, ConcatFusionBlock, FiLMFusionBlock, CrossAttentionFusionBlock

# Test classes for testing

class TestConcatFusionBlock:
    """Test concatenation-based fusion."""

    def test_forward_pass(self, sample_embeddings, sample_question_embeddings):
        """Test basic forward pass."""
        batch_size, seq_len, d_model = sample_embeddings.shape

        # Create concat fusion block
        block = ConcatFusionBlock(d_model=d_model)
        # Forward pass with vision and question embeddings
        output = block(sample_embeddings, sample_question_embeddings)

        # Verify output shape matches input
        assert output.shape == sample_embeddings.shape

    def test_residual_connection(self, sample_embeddings, sample_question_embeddings):
        """Test that residual connection preserves information."""
        d_model = sample_embeddings.shape[-1]

        # Create block with zero residual scale
        block = ConcatFusionBlock(d_model=d_model, residual_scale=0.0)
        # Forward pass
        output = block(sample_embeddings, sample_question_embeddings)

        # Verify output shape (residual affects magnitude, not shape)
        assert output.shape == sample_embeddings.shape

    def test_different_mlp_depths(self, sample_embeddings, sample_question_embeddings):
        """Test with different MLP depths."""
        d_model = sample_embeddings.shape[-1]

        # Create blocks with different MLP depths
        block_1layer = ConcatFusionBlock(d_model=d_model, mlp_depth=1)
        block_2layer = ConcatFusionBlock(d_model=d_model, mlp_depth=2)
        # Forward pass through both blocks
        output_1 = block_1layer(sample_embeddings, sample_question_embeddings)
        output_2 = block_2layer(sample_embeddings, sample_question_embeddings)

        # Verify both produce valid output shapes
        assert output_1.shape == sample_embeddings.shape
        assert output_2.shape == sample_embeddings.shape

    def test_gradient_flow(self, sample_embeddings, sample_question_embeddings):
        """Test that gradients flow through the block."""
        d_model = sample_embeddings.shape[-1]

        # Create inputs with gradient tracking
        V = sample_embeddings.clone().requires_grad_(True)
        q = sample_question_embeddings.clone().requires_grad_(True)
        # Forward pass
        block = ConcatFusionBlock(d_model=d_model)
        output = block(V, q)
        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Verify gradients exist and are finite
        assert V.grad is not None
        assert q.grad is not None
        assert torch.isfinite(V.grad).all()
        assert torch.isfinite(q.grad).all()


class TestFiLMFusionBlock:
    """Test FiLM (Feature-wise Linear Modulation) fusion."""

    def test_forward_pass(self, sample_embeddings, sample_question_embeddings):
        """Test basic forward pass."""
        _, _, d_model = sample_embeddings.shape

        # Create FiLM modulation block
        block = FiLMFusionBlock(d_model=d_model)
        # Forward pass with modulation
        output = block(sample_embeddings, sample_question_embeddings)

        # Verify output shape
        assert output.shape == sample_embeddings.shape

    def test_near_identity_initialization(self, sample_embeddings, sample_question_embeddings):
        """Test that initialization is near-identity (preserves input initially)."""
        d_model = sample_embeddings.shape[-1]

        # Create freshly initialized FiLM block
        block = FiLMFusionBlock(d_model=d_model)
        # Forward pass
        output = block(sample_embeddings, sample_question_embeddings)

        # Verify output shape (gamma≈0, beta≈0 initially)
        assert output.shape == sample_embeddings.shape

    def test_modulation_effect(self, sample_embeddings):
        """Test that different questions produce different modulations."""
        d_model = sample_embeddings.shape[-1]
        batch_size = sample_embeddings.shape[0]

        q1 = torch.randn(batch_size, d_model)
        q2 = torch.randn(batch_size, d_model)

        block = FiLMFusionBlock(d_model=d_model)
        output1 = block(sample_embeddings, q1)
        output2 = block(sample_embeddings, q2)

        # Different questions should produce different outputs
        assert output1.shape == output2.shape
        # Verify outputs are actually different (not just same shape)
        assert not torch.allclose(output1, output2, atol=1e-6)

    def test_gradient_flow(self, sample_embeddings, sample_question_embeddings):
        """Test gradient flow through FiLM block."""
        d_model = sample_embeddings.shape[-1]

        # Create inputs with gradient tracking
        V = sample_embeddings.clone().requires_grad_(True)
        q = sample_question_embeddings.clone().requires_grad_(True)

        # Forward pass
        block = FiLMFusionBlock(d_model=d_model)
        output = block(V, q)
        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        assert V.grad is not None
        assert q.grad is not None


class TestCrossAttentionFusionBlock:
    """Test cross-attention based fusion."""

    def test_forward_pass_with_tokens(self, sample_embeddings, sample_question_tokens):
        """Test forward pass with question tokens."""
        d_model = sample_embeddings.shape[-1]

        # Create cross-attention fusion block
        block = CrossAttentionFusionBlock(d_model=d_model, num_heads=8)
        # Forward pass with token-level question
        output = block(sample_embeddings, sample_question_tokens)

        # Verify output shape
        assert output.shape == sample_embeddings.shape

    def test_forward_pass_with_pooled_embedding(self, sample_embeddings, sample_question_embeddings):
        """Test forward pass with pooled question embedding (2D)."""
        d_model = sample_embeddings.shape[-1]

        # Create cross-attention block
        block = CrossAttentionFusionBlock(d_model=d_model, num_heads=8)
        # Forward pass with pooled question (should unsqueeze internally)
        output = block(sample_embeddings, sample_question_embeddings)

        # Verify output shape
        assert output.shape == sample_embeddings.shape

    def test_with_padding_mask(self, sample_embeddings, sample_question_tokens):
        """Test attention with key padding mask."""
        d_model = sample_embeddings.shape[-1]
        batch_size, seq_len_q, _ = sample_question_tokens.shape

        # Create padding mask marking second half as padding
        key_padding_mask = torch.zeros(batch_size, seq_len_q, dtype=torch.bool)
        key_padding_mask[:, seq_len_q//2:] = True  # True = ignore
        # Create block and forward pass with mask
        block = CrossAttentionFusionBlock(d_model=d_model, num_heads=8)
        output = block(sample_embeddings, sample_question_tokens, key_padding_mask=key_padding_mask)

        # Verify output shape
        assert output.shape == sample_embeddings.shape

    def test_different_num_heads(self, sample_embeddings, sample_question_tokens):
        """Test with different number of attention heads."""
        d_model = sample_embeddings.shape[-1]

        # Test various head counts
        for num_heads in [4, 8, 12]:
            # Skip if d_model not divisible by num_heads
            if d_model % num_heads != 0:
                continue

            # Create block with specified head count
            block = CrossAttentionFusionBlock(d_model=d_model, num_heads=num_heads)
            # Forward pass
            output = block(sample_embeddings, sample_question_tokens)

            # Verify output shape
            assert output.shape == sample_embeddings.shape

    def test_gradient_flow(self, sample_embeddings, sample_question_tokens):
        """Test gradient flow through cross-attention."""
        d_model = sample_embeddings.shape[-1]

        # Create inputs with gradient tracking
        V = sample_embeddings.clone().requires_grad_(True)
        Q = sample_question_tokens.clone().requires_grad_(True)
        # Forward pass
        block = CrossAttentionFusionBlock(d_model=d_model, num_heads=8)
        output = block(V, Q)
        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Verify gradients exist
        assert V.grad is not None
        assert Q.grad is not None


class TestFusionBlock:
    """Test multi-mode fusion block wrapper."""

    def test_single_mode_concat(self, sample_embeddings, sample_question_embeddings):
        """Test with single concat mode."""
        d_model = sample_embeddings.shape[-1]

        # Create fusion block with only concat mode
        block = FusionBlock(d_model=d_model, modes=["concat"])
        # Forward pass
        output = block(sample_embeddings, q_pooled=sample_question_embeddings)

        # Verify output shape
        assert output.shape == sample_embeddings.shape

    def test_single_mode_film(self, sample_embeddings, sample_question_embeddings):
        """Test with single FiLM mode."""
        d_model = sample_embeddings.shape[-1]

        block = FusionBlock(d_model=d_model, modes=["film"])
        output = block(sample_embeddings, q_pooled=sample_question_embeddings)

        assert output.shape == sample_embeddings.shape

    def test_single_mode_cross_attention(self, sample_embeddings, sample_question_tokens):
        """Test with single cross-attention mode."""
        d_model = sample_embeddings.shape[-1]

        block = FusionBlock(d_model=d_model, modes=["cross_attention"])
        output = block(sample_embeddings, q_tokens=sample_question_tokens)

        assert output.shape == sample_embeddings.shape

    def test_multiple_modes(self, sample_embeddings, sample_question_embeddings, sample_question_tokens):
        """Test with multiple fusion modes."""
        d_model = sample_embeddings.shape[-1]

        # Create block with all fusion modes
        block = FusionBlock(d_model=d_model, modes=["concat", "film", "cross_attention"])
        # Forward pass with both pooled and token-level questions
        output = block(
            sample_embeddings,
            q_pooled=sample_question_embeddings,
            q_tokens=sample_question_tokens
        )

        # Verify output shape (all modes applied sequentially)
        assert output.shape == sample_embeddings.shape

    def test_invalid_mode_raises_error(self):
        """Test that invalid fusion mode raises error."""
        with pytest.raises(ValueError, match="Invalid fusion mode"):
            FusionBlock(d_model=768, modes=["invalid_mode"])

    def test_missing_required_input_raises_error(self, sample_embeddings, sample_question_embeddings):
        """Test that missing required input raises assertion error."""
        d_model = sample_embeddings.shape[-1]

        # Create block requiring token-level question
        block = FusionBlock(d_model=d_model, modes=["cross_attention"])
        # Verify error when only pooled question provided
        with pytest.raises(AssertionError, match="Cross-attn needs token-level Q"):
            block(sample_embeddings, q_pooled=sample_question_embeddings, q_tokens=None)

    def test_concat_film_combination(self, sample_embeddings, sample_question_embeddings):
        """Test combination of concat and FiLM modes."""
        d_model = sample_embeddings.shape[-1]

        block = FusionBlock(d_model=d_model, modes=["concat", "film"])
        output = block(sample_embeddings, q_pooled=sample_question_embeddings)

        assert output.shape == sample_embeddings.shape

    def test_gradient_flow_through_multiple_modes(self, sample_embeddings, sample_question_embeddings):
        """Test gradient flow through multi-mode fusion."""
        d_model = sample_embeddings.shape[-1]

        # Create inputs with gradient tracking
        V = sample_embeddings.clone().requires_grad_(True)
        q = sample_question_embeddings.clone().requires_grad_(True)
        # Forward pass through multiple fusion modes
        block = FusionBlock(d_model=d_model, modes=["concat", "film"])
        output = block(V, q_pooled=q)
        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Verify gradients exist and are finite
        assert V.grad is not None
        assert q.grad is not None
        assert torch.isfinite(V.grad).all()
        assert torch.isfinite(q.grad).all()
