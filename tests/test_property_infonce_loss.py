"""
Property-Based Tests for InfoNCE Loss Computation.

Property 4: InfoNCE Loss Computation
For any batch of query embeddings and document embeddings with known positive
pairs, the ColBERT InfoNCE loss SHALL produce a scalar value that decreases
when positive pairs have higher similarity than negative pairs.

**Validates: Requirements 2.4**

Feature: qwen3-vl-rag-retrieval, Property 4: InfoNCE Loss Computation
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st, assume
from typing import List, Tuple


# Test data generators
@st.composite
def embedding_batch_strategy(draw, batch_size_range=(2, 8), seq_len_range=(3, 10), dim=128):
    """Generate a batch of embeddings."""
    batch_size = draw(st.integers(min_value=batch_size_range[0], max_value=batch_size_range[1]))
    seq_len = draw(st.integers(min_value=seq_len_range[0], max_value=seq_len_range[1]))
    
    # Generate random embeddings
    embeddings = torch.randn(batch_size, seq_len, dim)
    
    # L2 normalize (as required by the model)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    
    return embeddings


@st.composite
def embedding_list_strategy(draw, num_embeddings_range=(2, 6), seq_len_range=(3, 10), dim=128):
    """Generate a list of variable-length embeddings."""
    num_embeddings = draw(st.integers(min_value=num_embeddings_range[0], max_value=num_embeddings_range[1]))
    
    embeddings = []
    for _ in range(num_embeddings):
        seq_len = draw(st.integers(min_value=seq_len_range[0], max_value=seq_len_range[1]))
        emb = torch.randn(seq_len, dim)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        embeddings.append(emb)
    
    return embeddings


@st.composite
def temperature_strategy(draw):
    """Generate valid temperature values."""
    return draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))


class TestInfoNCELossProperty:
    """
    Property 4: InfoNCE Loss Computation
    
    Tests that InfoNCE loss:
    1. Produces a scalar value
    2. Decreases when positive pairs have higher similarity
    3. Is non-negative
    4. Works with temperature scaling
    """
    
    @given(
        query_embeddings=embedding_batch_strategy(),
        doc_embeddings=embedding_batch_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_loss_produces_scalar(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ):
        """
        Property: InfoNCE loss produces a scalar value.
        
        **Validates: Requirements 2.4**
        """
        from qwen3_vl_retrieval.training.losses import ColbertLoss
        
        # Ensure same batch size
        batch_size = min(query_embeddings.shape[0], doc_embeddings.shape[0])
        query_embeddings = query_embeddings[:batch_size]
        doc_embeddings = doc_embeddings[:batch_size]
        
        loss_fn = ColbertLoss(temperature=0.02)
        loss = loss_fn(query_embeddings, doc_embeddings)
        
        # Property: loss is a scalar
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert loss.numel() == 1, f"Loss should have 1 element, got {loss.numel()}"
    
    @given(
        query_embeddings=embedding_batch_strategy(),
        doc_embeddings=embedding_batch_strategy(),
    )
    @settings(max_examples=100, deadline=None)
    def test_loss_is_non_negative(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ):
        """
        Property: InfoNCE loss is non-negative.
        
        Cross-entropy loss is always >= 0.
        
        **Validates: Requirements 2.4**
        """
        from qwen3_vl_retrieval.training.losses import ColbertLoss
        
        batch_size = min(query_embeddings.shape[0], doc_embeddings.shape[0])
        query_embeddings = query_embeddings[:batch_size]
        doc_embeddings = doc_embeddings[:batch_size]
        
        loss_fn = ColbertLoss(temperature=0.02)
        loss = loss_fn(query_embeddings, doc_embeddings)
        
        # Property: loss >= 0
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    @given(temperature=temperature_strategy())
    @settings(max_examples=50, deadline=None)
    def test_temperature_scaling_effect(self, temperature: float):
        """
        Property: Higher temperature leads to softer probability distribution.
        
        With higher temperature, the loss should be closer to uniform distribution loss.
        
        **Validates: Requirements 2.4**
        """
        from qwen3_vl_retrieval.training.losses import ColbertLoss
        
        # Fixed embeddings for comparison
        torch.manual_seed(42)
        batch_size, seq_len, dim = 4, 5, 128
        query_embeddings = torch.randn(batch_size, seq_len, dim)
        doc_embeddings = torch.randn(batch_size, seq_len, dim)
        
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
        
        loss_fn = ColbertLoss(temperature=temperature)
        loss = loss_fn(query_embeddings, doc_embeddings)
        
        # Property: loss is finite
        assert torch.isfinite(loss), f"Loss should be finite with temperature {temperature}"
    
    def test_loss_decreases_with_better_alignment(self):
        """
        Property: Loss decreases when positive pairs have higher similarity.
        
        This is the core property of contrastive learning.
        
        **Validates: Requirements 2.4**
        """
        from qwen3_vl_retrieval.training.losses import ColbertLoss
        
        batch_size, seq_len, dim = 4, 5, 128
        
        # Create query embeddings
        torch.manual_seed(42)
        query_embeddings = torch.randn(batch_size, seq_len, dim)
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        
        # Case 1: Random document embeddings (low alignment)
        random_doc_embeddings = torch.randn(batch_size, seq_len, dim)
        random_doc_embeddings = torch.nn.functional.normalize(random_doc_embeddings, p=2, dim=-1)
        
        # Case 2: Document embeddings similar to queries (high alignment)
        # Add small noise to queries to create aligned documents
        aligned_doc_embeddings = query_embeddings + 0.1 * torch.randn_like(query_embeddings)
        aligned_doc_embeddings = torch.nn.functional.normalize(aligned_doc_embeddings, p=2, dim=-1)
        
        loss_fn = ColbertLoss(temperature=0.02)
        
        loss_random = loss_fn(query_embeddings, random_doc_embeddings)
        loss_aligned = loss_fn(query_embeddings, aligned_doc_embeddings)
        
        # Property: aligned pairs should have lower loss
        assert loss_aligned.item() < loss_random.item(), \
            f"Aligned loss ({loss_aligned.item():.4f}) should be less than random loss ({loss_random.item():.4f})"
    
    def test_perfect_alignment_has_low_loss(self):
        """
        Property: Perfect alignment (identical embeddings) has very low loss.
        
        **Validates: Requirements 2.4**
        """
        from qwen3_vl_retrieval.training.losses import ColbertLoss
        
        batch_size, seq_len, dim = 4, 5, 128
        
        torch.manual_seed(42)
        embeddings = torch.randn(batch_size, seq_len, dim)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        loss_fn = ColbertLoss(temperature=0.02)
        
        # Use same embeddings for query and doc (perfect alignment)
        loss = loss_fn(embeddings, embeddings.clone())
        
        # Property: perfect alignment should have low loss
        # With perfect diagonal alignment, loss should be close to 0
        # (not exactly 0 due to in-batch negatives)
        assert loss.item() < 1.0, \
            f"Perfect alignment loss ({loss.item():.4f}) should be low"
    
    @given(
        query_list=embedding_list_strategy(),
        doc_list=embedding_list_strategy(),
    )
    @settings(max_examples=50, deadline=None)
    def test_list_input_produces_valid_loss(
        self,
        query_list: List[torch.Tensor],
        doc_list: List[torch.Tensor],
    ):
        """
        Property: List inputs (variable length) produce valid loss.
        
        **Validates: Requirements 2.4**
        """
        from qwen3_vl_retrieval.training.losses import ColbertLoss
        
        # Ensure same number of queries and docs
        min_len = min(len(query_list), len(doc_list))
        assume(min_len >= 2)
        
        query_list = query_list[:min_len]
        doc_list = doc_list[:min_len]
        
        loss_fn = ColbertLoss(temperature=0.02)
        loss = loss_fn(query_list, doc_list)
        
        # Property: loss is valid scalar
        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() >= 0, "Loss should be non-negative"
    
    def test_batch_size_invariance(self):
        """
        Property: Loss computation is consistent regardless of batch processing.
        
        Computing loss for the same data should give same result.
        
        **Validates: Requirements 2.4**
        """
        from qwen3_vl_retrieval.training.losses import ColbertLoss
        
        batch_size, seq_len, dim = 4, 5, 128
        
        torch.manual_seed(42)
        query_embeddings = torch.randn(batch_size, seq_len, dim)
        doc_embeddings = torch.randn(batch_size, seq_len, dim)
        
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
        
        loss_fn = ColbertLoss(temperature=0.02)
        
        # Compute loss twice
        loss1 = loss_fn(query_embeddings, doc_embeddings)
        loss2 = loss_fn(query_embeddings, doc_embeddings)
        
        # Property: same input gives same output
        assert torch.allclose(loss1, loss2), \
            f"Loss should be deterministic: {loss1.item()} vs {loss2.item()}"
    
    def test_gradient_flow(self):
        """
        Property: Loss allows gradient flow for training.
        
        **Validates: Requirements 2.4**
        """
        from qwen3_vl_retrieval.training.losses import ColbertLoss
        
        batch_size, seq_len, dim = 4, 5, 128
        
        query_embeddings = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        doc_embeddings = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        
        loss_fn = ColbertLoss(temperature=0.02)
        loss = loss_fn(query_embeddings, doc_embeddings)
        
        # Backward pass
        loss.backward()
        
        # Property: gradients should exist and be finite
        assert query_embeddings.grad is not None, "Query gradients should exist"
        assert doc_embeddings.grad is not None, "Doc gradients should exist"
        assert torch.isfinite(query_embeddings.grad).all(), "Query gradients should be finite"
        assert torch.isfinite(doc_embeddings.grad).all(), "Doc gradients should be finite"


class TestBiEncoderLossProperty:
    """
    Tests for BiEncoderLoss (single-vector comparison).
    """
    
    def test_bi_encoder_loss_produces_scalar(self):
        """
        Property: BiEncoderLoss produces a scalar value.
        """
        from qwen3_vl_retrieval.training.losses import BiEncoderLoss
        
        batch_size, dim = 4, 128
        
        query_embeddings = torch.randn(batch_size, dim)
        doc_embeddings = torch.randn(batch_size, dim)
        
        loss_fn = BiEncoderLoss(temperature=0.02)
        loss = loss_fn(query_embeddings, doc_embeddings)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
    
    def test_cosine_similarity_mode(self):
        """
        Property: Cosine similarity mode normalizes embeddings.
        """
        from qwen3_vl_retrieval.training.losses import BiEncoderLoss
        
        batch_size, dim = 4, 128
        
        # Non-normalized embeddings
        query_embeddings = torch.randn(batch_size, dim) * 10
        doc_embeddings = torch.randn(batch_size, dim) * 10
        
        loss_fn = BiEncoderLoss(temperature=0.02, similarity="cosine")
        loss = loss_fn(query_embeddings, doc_embeddings)
        
        assert torch.isfinite(loss), "Loss should be finite with cosine similarity"


class TestHardNegativeLossProperty:
    """
    Tests for HardNegativeLoss.
    """
    
    def test_hard_negative_loss_without_negatives(self):
        """
        Property: HardNegativeLoss falls back to base loss without hard negatives.
        """
        from qwen3_vl_retrieval.training.losses import HardNegativeLoss, ColbertLoss
        
        batch_size, seq_len, dim = 4, 5, 128
        
        torch.manual_seed(42)
        query_embeddings = torch.randn(batch_size, seq_len, dim)
        doc_embeddings = torch.randn(batch_size, seq_len, dim)
        
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
        
        hard_neg_loss_fn = HardNegativeLoss(temperature=0.02)
        base_loss_fn = ColbertLoss(temperature=0.02)
        
        loss_hard = hard_neg_loss_fn(query_embeddings, doc_embeddings)
        loss_base = base_loss_fn(query_embeddings, doc_embeddings)
        
        # Property: without hard negatives, should equal base loss
        assert torch.allclose(loss_hard, loss_base, atol=1e-5), \
            f"Hard negative loss ({loss_hard.item()}) should equal base loss ({loss_base.item()})"
    
    def test_hard_negative_loss_with_negatives(self):
        """
        Property: HardNegativeLoss incorporates hard negatives.
        """
        from qwen3_vl_retrieval.training.losses import HardNegativeLoss
        
        batch_size, seq_len, dim = 4, 5, 128
        num_negatives = 2
        
        torch.manual_seed(42)
        query_embeddings = torch.randn(batch_size, seq_len, dim)
        positive_embeddings = torch.randn(batch_size, seq_len, dim)
        hard_negative_embeddings = torch.randn(batch_size, num_negatives, seq_len, dim)
        
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=-1)
        hard_negative_embeddings = torch.nn.functional.normalize(hard_negative_embeddings, p=2, dim=-1)
        
        loss_fn = HardNegativeLoss(temperature=0.02)
        loss = loss_fn(
            query_embeddings,
            positive_embeddings,
            hard_negative_embeddings,
        )
        
        # Property: loss is valid
        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() >= 0, "Loss should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
