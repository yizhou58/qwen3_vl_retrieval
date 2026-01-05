"""
Property-Based Tests for MaxSim Score Computation.

**Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

Property 5: For any query embedding Q of shape (N_q, dim) and document embedding D 
of shape (N_d, dim), the MaxSim score SHALL equal the sum over all query tokens of 
the maximum dot product with any document token: score = Σᵢ maxⱼ(Qᵢ · Dⱼ).
Batch processing SHALL produce identical results to individual computation.
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st, assume
from typing import List

# Import the score_multi_vector function
from qwen3_vl_retrieval.models.processing_colqwen3vl import BaseVisualRetrieverProcessor


def reference_maxsim(query: torch.Tensor, doc: torch.Tensor) -> float:
    """
    Reference implementation of MaxSim for a single query-document pair.
    
    This is the ground truth implementation that we test against.
    
    Args:
        query: Query embedding (N_q, dim)
        doc: Document embedding (N_d, dim)
        
    Returns:
        MaxSim score = Σᵢ maxⱼ(Qᵢ · Dⱼ)
    """
    # Compute all pairwise dot products: (N_q, N_d)
    similarities = torch.matmul(query, doc.T)
    
    # For each query token, take max over document tokens
    max_sims = similarities.max(dim=1)[0]  # (N_q,)
    
    # Sum over query tokens
    return max_sims.sum().item()


# Hypothesis strategies
@st.composite
def embedding_pair_strategy(draw):
    """Generate a pair of query and document embeddings."""
    dim = 128  # Fixed dimension as per requirements
    
    n_query_tokens = draw(st.integers(min_value=1, max_value=32))
    n_doc_tokens = draw(st.integers(min_value=1, max_value=64))
    
    # Generate normalized embeddings (as the model produces)
    query = torch.randn(n_query_tokens, dim)
    query = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
    
    doc = torch.randn(n_doc_tokens, dim)
    doc = doc / (doc.norm(dim=-1, keepdim=True) + 1e-8)
    
    return query, doc


@st.composite
def batch_embeddings_strategy(draw):
    """Generate batches of query and document embeddings."""
    dim = 128
    
    n_queries = draw(st.integers(min_value=1, max_value=4))
    n_docs = draw(st.integers(min_value=1, max_value=4))
    
    queries = []
    for _ in range(n_queries):
        n_tokens = draw(st.integers(min_value=1, max_value=16))
        q = torch.randn(n_tokens, dim)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        queries.append(q)
    
    docs = []
    for _ in range(n_docs):
        n_tokens = draw(st.integers(min_value=1, max_value=32))
        d = torch.randn(n_tokens, dim)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        docs.append(d)
    
    return queries, docs


class TestProperty5MaxSimScoreComputation:
    """
    Property 5: MaxSim Score Computation
    
    **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    
    For any query embedding Q and document embedding D, the MaxSim score SHALL equal
    the sum over all query tokens of the maximum dot product with any document token.
    """
    
    @given(data=embedding_pair_strategy())
    @settings(max_examples=100, deadline=None)
    def test_maxsim_matches_reference(self, data):
        """
        Property: MaxSim score equals reference implementation.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        query, doc = data
        
        # Compute using our implementation
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            [query], [doc], device="cpu"
        )
        computed_score = scores[0, 0].item()
        
        # Compute using reference implementation
        expected_score = reference_maxsim(query, doc)
        
        # Property: scores should match within tolerance
        assert abs(computed_score - expected_score) < 1e-4, \
            f"MaxSim mismatch: computed={computed_score:.6f}, expected={expected_score:.6f}"
    
    @given(data=batch_embeddings_strategy())
    @settings(max_examples=100, deadline=None)
    def test_batch_matches_individual(self, data):
        """
        Property: Batch processing produces identical results to individual computation.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.4**
        """
        queries, docs = data
        
        # Compute batch scores
        batch_scores = BaseVisualRetrieverProcessor.score_multi_vector(
            queries, docs, device="cpu"
        )
        
        # Compute individual scores
        for i, q in enumerate(queries):
            for j, d in enumerate(docs):
                individual_score = reference_maxsim(q, d)
                batch_score = batch_scores[i, j].item()
                
                # Property: batch and individual should match
                assert abs(batch_score - individual_score) < 1e-4, \
                    f"Batch/individual mismatch at ({i},{j}): " \
                    f"batch={batch_score:.6f}, individual={individual_score:.6f}"
    
    @given(data=batch_embeddings_strategy())
    @settings(max_examples=100, deadline=None)
    def test_output_shape(self, data):
        """
        Property: Output shape is (n_queries, n_passages).
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.4**
        """
        queries, docs = data
        
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            queries, docs, device="cpu"
        )
        
        # Property: shape is (n_queries, n_docs)
        assert scores.shape == (len(queries), len(docs)), \
            f"Expected shape ({len(queries)}, {len(docs)}), got {scores.shape}"
    
    @given(data=embedding_pair_strategy())
    @settings(max_examples=100, deadline=None)
    def test_maxsim_formula_components(self, data):
        """
        Property: MaxSim correctly computes dot products, max, and sum.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        query, doc = data
        
        # Step 1: Compute all dot products (Requirement 3.1)
        dot_products = torch.matmul(query, doc.T)  # (N_q, N_d)
        
        # Step 2: Take max for each query token (Requirement 3.2)
        max_per_query = dot_products.max(dim=1)[0]  # (N_q,)
        
        # Step 3: Sum over query tokens (Requirement 3.3)
        expected_score = max_per_query.sum().item()
        
        # Compute using implementation
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            [query], [doc], device="cpu"
        )
        computed_score = scores[0, 0].item()
        
        # Property: formula components produce correct result
        assert abs(computed_score - expected_score) < 1e-4, \
            f"Formula mismatch: computed={computed_score:.6f}, expected={expected_score:.6f}"
    
    @given(
        scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_maxsim_scales_with_embedding_magnitude(self, scale):
        """
        Property: MaxSim scales linearly with embedding magnitude.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.1**
        """
        dim = 128
        query = torch.randn(8, dim)
        doc = torch.randn(16, dim)
        
        # Compute base score
        base_scores = BaseVisualRetrieverProcessor.score_multi_vector(
            [query], [doc], device="cpu"
        )
        base_score = base_scores[0, 0].item()
        
        # Scale query embeddings
        scaled_query = query * scale
        scaled_scores = BaseVisualRetrieverProcessor.score_multi_vector(
            [scaled_query], [doc], device="cpu"
        )
        scaled_score = scaled_scores[0, 0].item()
        
        # Property: score scales linearly with query magnitude
        expected_scaled = base_score * scale
        assert abs(scaled_score - expected_scaled) < 1e-3, \
            f"Scaling mismatch: scaled={scaled_score:.6f}, expected={expected_scaled:.6f}"


class TestProperty5EdgeCases:
    """
    Edge case tests for MaxSim computation.
    
    **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    """
    
    def test_single_token_query_and_doc(self):
        """
        Test: Single token query and doc produces dot product.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        dim = 128
        query = torch.randn(1, dim)
        doc = torch.randn(1, dim)
        
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            [query], [doc], device="cpu"
        )
        
        # For single tokens, MaxSim = dot product
        expected = torch.dot(query.squeeze(), doc.squeeze()).item()
        computed = scores[0, 0].item()
        
        assert abs(computed - expected) < 1e-5, \
            f"Single token mismatch: computed={computed:.6f}, expected={expected:.6f}"
    
    def test_identical_embeddings(self):
        """
        Test: Identical query and doc produces sum of squared norms.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        dim = 128
        emb = torch.randn(10, dim)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize
        
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            [emb], [emb], device="cpu"
        )
        
        # For identical normalized embeddings, max similarity for each token is 1.0
        # So MaxSim = number of tokens
        expected = emb.shape[0]  # 10
        computed = scores[0, 0].item()
        
        assert abs(computed - expected) < 1e-4, \
            f"Identical embeddings mismatch: computed={computed:.6f}, expected={expected}"
    
    def test_orthogonal_embeddings(self):
        """
        Test: Orthogonal embeddings produce zero score.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.1, 3.2, 3.3**
        """
        # Create orthogonal embeddings using first 2 dimensions
        query = torch.zeros(2, 128)
        query[0, 0] = 1.0  # First token: [1, 0, 0, ...]
        query[1, 1] = 1.0  # Second token: [0, 1, 0, ...]
        
        doc = torch.zeros(2, 128)
        doc[0, 2] = 1.0  # First token: [0, 0, 1, ...]
        doc[1, 3] = 1.0  # Second token: [0, 0, 0, 1, ...]
        
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            [query], [doc], device="cpu"
        )
        
        # Orthogonal vectors have zero dot product
        assert abs(scores[0, 0].item()) < 1e-6, \
            f"Orthogonal embeddings should have zero score, got {scores[0, 0].item()}"
    
    def test_empty_queries_raises_error(self):
        """
        Test: Empty queries raises ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.4**
        """
        doc = torch.randn(10, 128)
        
        with pytest.raises(ValueError, match="No queries provided"):
            BaseVisualRetrieverProcessor.score_multi_vector([], [doc], device="cpu")
    
    def test_empty_passages_raises_error(self):
        """
        Test: Empty passages raises ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.4**
        """
        query = torch.randn(10, 128)
        
        with pytest.raises(ValueError, match="No passages provided"):
            BaseVisualRetrieverProcessor.score_multi_vector([query], [], device="cpu")


class TestProperty5NormalizationInteraction:
    """
    Tests for MaxSim interaction with normalized embeddings.
    
    **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
    **Validates: Requirements 3.5**
    """
    
    @given(data=embedding_pair_strategy())
    @settings(max_examples=100, deadline=None)
    def test_normalized_embeddings_bounded_score(self, data):
        """
        Property: For normalized embeddings, max similarity per token is bounded by 1.
        
        **Feature: qwen3-vl-rag-retrieval, Property 5: MaxSim Score Computation**
        **Validates: Requirements 3.5**
        """
        query, doc = data
        
        # Embeddings are already normalized by the strategy
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            [query], [doc], device="cpu"
        )
        
        # For normalized embeddings, MaxSim <= n_query_tokens
        # (since each max similarity is at most 1)
        max_possible = query.shape[0]
        computed = scores[0, 0].item()
        
        assert computed <= max_possible + 1e-4, \
            f"Score {computed:.6f} exceeds max possible {max_possible} for normalized embeddings"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
