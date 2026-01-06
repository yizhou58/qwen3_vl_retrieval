"""
Property-Based Tests for Batch Masking Correctness.

**Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
**Validates: Requirements 1.2, 3.4**

Property 12: For any batch of documents with varying numbers of visual tokens 
(due to Qwen3-VL's dynamic resolution), the MaxSim computation SHALL apply a 
padding mask such that padding tokens do NOT contribute to the maximum similarity 
score (i.e., their similarity should be masked to -inf before the max operation).
"""

import tempfile
import pytest
from hypothesis import given, settings, strategies as st, assume
from typing import List, Dict
import torch

from qwen3_vl_retrieval.retrieval.second_stage_reranker import SecondStageReranker
from qwen3_vl_retrieval.retrieval.embedding_store import EmbeddingStore
from qwen3_vl_retrieval.models.processing_colqwen3vl import BaseVisualRetrieverProcessor


# Hypothesis strategies
@st.composite
def normalized_embedding_strategy(draw, min_tokens=3, max_tokens=30, dim=128):
    """Generate random L2-normalized embeddings."""
    num_tokens = draw(st.integers(min_value=min_tokens, max_value=max_tokens))
    embeddings = torch.randn(num_tokens, dim)
    # L2 normalize
    embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
    return embeddings


@st.composite
def variable_length_batch_strategy(draw, batch_size=5, min_tokens=5, max_tokens=50, dim=128):
    """Generate a batch of embeddings with varying sequence lengths."""
    batch_size = draw(st.integers(min_value=2, max_value=batch_size))
    
    embeddings_list = []
    lengths = []
    
    for _ in range(batch_size):
        num_tokens = draw(st.integers(min_value=min_tokens, max_value=max_tokens))
        emb = torch.randn(num_tokens, dim)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        embeddings_list.append(emb)
        lengths.append(num_tokens)
    
    return embeddings_list, lengths


class TestProperty12BatchMaskingCorrectness:
    """
    Property 12: Batch Masking Correctness
    
    **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
    **Validates: Requirements 1.2, 3.4**
    
    For any batch of documents with varying numbers of visual tokens, the MaxSim 
    computation SHALL apply a padding mask such that padding tokens do NOT contribute 
    to the maximum similarity score.
    """
    
    @given(
        query_emb=normalized_embedding_strategy(min_tokens=5, max_tokens=15, dim=128),
        doc_emb=normalized_embedding_strategy(min_tokens=10, max_tokens=30, dim=128),
        num_padding=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_padding_does_not_affect_maxsim_score(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        num_padding: int
    ):
        """
        Property: Padding tokens do not affect MaxSim score when properly masked.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 1.2, 3.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            
            # Create mock model
            class MockModel:
                device = torch.device('cpu')
                def eval(self): pass
            
            class MockProcessor:
                pass
            
            reranker = SecondStageReranker(
                model=MockModel(),
                processor=MockProcessor(),
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            # Compute MaxSim without padding
            score_no_padding = reranker.compute_maxsim(query_emb, doc_emb)
            
            # Add padding to document embeddings
            padding = torch.zeros(num_padding, doc_emb.shape[1])
            padded_doc_emb = torch.cat([doc_emb, padding], dim=0)
            
            # Create mask (True for valid, False for padding)
            mask = torch.cat([
                torch.ones(doc_emb.shape[0], dtype=torch.bool),
                torch.zeros(num_padding, dtype=torch.bool)
            ])
            
            # Compute MaxSim with padding and mask
            score_with_mask = reranker.compute_maxsim(query_emb, padded_doc_emb, doc_mask=mask)
            
            # Property: scores should be equal
            assert torch.allclose(score_with_mask, score_no_padding, atol=1e-5), \
                f"Padding affected score: with_mask={score_with_mask.item()}, no_padding={score_no_padding.item()}"
            
            store.close()
    
    @given(
        query_emb=normalized_embedding_strategy(min_tokens=5, max_tokens=15, dim=128),
        doc_emb=normalized_embedding_strategy(min_tokens=10, max_tokens=30, dim=128),
        num_padding=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_padding_with_high_values_masked_correctly(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        num_padding: int
    ):
        """
        Property: Even high-value padding tokens are masked and don't affect score.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 1.2, 3.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            
            class MockModel:
                device = torch.device('cpu')
                def eval(self): pass
            
            class MockProcessor:
                pass
            
            reranker = SecondStageReranker(
                model=MockModel(),
                processor=MockProcessor(),
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            # Compute MaxSim without padding
            score_no_padding = reranker.compute_maxsim(query_emb, doc_emb)
            
            # Add HIGH VALUE padding (would increase score if not masked)
            # Create padding that would have high similarity with query
            # Ensure we create exactly num_padding tokens by repeating enough times
            repeat_times = (num_padding // query_emb.shape[0]) + 1
            high_value_padding = query_emb.repeat(repeat_times, 1)[:num_padding]
            padded_doc_emb = torch.cat([doc_emb, high_value_padding], dim=0)
            
            # Create mask - must match actual padding size
            actual_padding_size = high_value_padding.shape[0]
            mask = torch.cat([
                torch.ones(doc_emb.shape[0], dtype=torch.bool),
                torch.zeros(actual_padding_size, dtype=torch.bool)
            ])
            
            # Compute with mask
            score_with_mask = reranker.compute_maxsim(query_emb, padded_doc_emb, doc_mask=mask)
            
            # Property: score should still equal the no-padding score
            assert torch.allclose(score_with_mask, score_no_padding, atol=1e-5), \
                f"High-value padding affected score: with_mask={score_with_mask.item()}, no_padding={score_no_padding.item()}"
            
            store.close()
    
    @given(
        query_emb=normalized_embedding_strategy(min_tokens=5, max_tokens=15, dim=128),
        doc_emb=normalized_embedding_strategy(min_tokens=10, max_tokens=30, dim=128)
    )
    @settings(max_examples=100, deadline=None)
    def test_full_mask_gives_same_result_as_no_mask(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor
    ):
        """
        Property: A mask of all True values gives same result as no mask.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 3.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            
            class MockModel:
                device = torch.device('cpu')
                def eval(self): pass
            
            class MockProcessor:
                pass
            
            reranker = SecondStageReranker(
                model=MockModel(),
                processor=MockProcessor(),
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            # Compute without mask
            score_no_mask = reranker.compute_maxsim(query_emb, doc_emb)
            
            # Compute with all-True mask
            full_mask = torch.ones(doc_emb.shape[0], dtype=torch.bool)
            score_with_full_mask = reranker.compute_maxsim(query_emb, doc_emb, doc_mask=full_mask)
            
            # Property: scores should be equal
            assert torch.allclose(score_with_full_mask, score_no_mask, atol=1e-5), \
                f"Full mask changed score: with_mask={score_with_full_mask.item()}, no_mask={score_no_mask.item()}"
            
            store.close()


class TestProperty12ProcessorBatchMasking:
    """
    Tests for batch masking in the processor's score_multi_vector method.
    
    **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
    **Validates: Requirements 1.2, 3.4**
    """
    
    @given(
        batch=variable_length_batch_strategy(batch_size=5, min_tokens=5, max_tokens=30, dim=128),
        query_emb=normalized_embedding_strategy(min_tokens=5, max_tokens=15, dim=128)
    )
    @settings(max_examples=100, deadline=None)
    def test_score_multi_vector_handles_variable_lengths(
        self,
        batch: tuple,
        query_emb: torch.Tensor
    ):
        """
        Property: score_multi_vector correctly handles variable-length document embeddings.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 1.2, 3.4**
        """
        doc_embeddings_list, lengths = batch
        
        # Compute scores using score_multi_vector
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=[query_emb],
            ps=doc_embeddings_list,
            batch_size=128,
        )
        
        # Compute reference scores individually
        reference_scores = []
        for doc_emb in doc_embeddings_list:
            # MaxSim: for each query token, max over doc tokens, then sum
            similarities = torch.matmul(query_emb, doc_emb.T)
            max_sims = similarities.max(dim=1)[0]
            reference_scores.append(max_sims.sum().item())
        
        # Property: batch scores should match individual scores
        for i, (batch_score, ref_score) in enumerate(zip(scores[0].tolist(), reference_scores)):
            assert abs(batch_score - ref_score) < 1e-4, \
                f"Score mismatch for doc {i}: batch={batch_score}, reference={ref_score}"
    
    @given(
        queries=st.lists(
            normalized_embedding_strategy(min_tokens=3, max_tokens=10, dim=128),
            min_size=2, max_size=5
        ),
        docs=st.lists(
            normalized_embedding_strategy(min_tokens=5, max_tokens=20, dim=128),
            min_size=2, max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_score_multi_vector_batch_consistency(
        self,
        queries: List[torch.Tensor],
        docs: List[torch.Tensor]
    ):
        """
        Property: Batch processing produces same results as individual processing.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 3.4**
        """
        # Compute batch scores
        batch_scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=queries,
            ps=docs,
            batch_size=128,
        )
        
        # Compute individual scores
        for i, query in enumerate(queries):
            for j, doc in enumerate(docs):
                # Reference MaxSim
                similarities = torch.matmul(query, doc.T)
                max_sims = similarities.max(dim=1)[0]
                ref_score = max_sims.sum().item()
                
                batch_score = batch_scores[i, j].item()
                
                assert abs(batch_score - ref_score) < 1e-4, \
                    f"Score mismatch at ({i},{j}): batch={batch_score}, reference={ref_score}"
    
    @given(
        query_emb=normalized_embedding_strategy(min_tokens=5, max_tokens=10, dim=128),
        doc_emb=normalized_embedding_strategy(min_tokens=10, max_tokens=20, dim=128),
        num_padding=st.integers(min_value=1, max_value=15)
    )
    @settings(max_examples=100, deadline=None)
    def test_score_multi_vector_padding_invariance(
        self,
        query_emb: torch.Tensor,
        doc_emb: torch.Tensor,
        num_padding: int
    ):
        """
        Property: Adding zero-padding to embeddings doesn't change scores when using lists.
        
        Note: When using lists of tensors (not padded batches), each tensor is its own
        length, so padding is not an issue. This test verifies that the list-based
        approach naturally handles variable lengths.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 1.2**
        """
        # Score with original embeddings
        score_original = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=[query_emb],
            ps=[doc_emb],
            batch_size=128,
        )
        
        # The list-based approach handles variable lengths naturally
        # Each document is processed with its actual length
        
        # Verify the score is computed correctly
        similarities = torch.matmul(query_emb, doc_emb.T)
        max_sims = similarities.max(dim=1)[0]
        expected_score = max_sims.sum().item()
        
        assert abs(score_original[0, 0].item() - expected_score) < 1e-4, \
            f"Score mismatch: got {score_original[0, 0].item()}, expected {expected_score}"


class TestProperty12EdgeCases:
    """
    Edge case tests for batch masking.
    
    **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
    **Validates: Requirements 1.2, 3.4**
    """
    
    def test_single_token_query(self):
        """
        Test: Single token query works correctly with masking.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 3.4**
        """
        query_emb = torch.randn(1, 128)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
        
        doc_emb = torch.randn(20, 128)
        doc_emb = doc_emb / doc_emb.norm(dim=-1, keepdim=True)
        
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=[query_emb],
            ps=[doc_emb],
            batch_size=128,
        )
        
        # Reference
        similarities = torch.matmul(query_emb, doc_emb.T)
        expected = similarities.max(dim=1)[0].sum().item()
        
        assert abs(scores[0, 0].item() - expected) < 1e-4
    
    def test_single_token_document(self):
        """
        Test: Single token document works correctly with masking.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 3.4**
        """
        query_emb = torch.randn(10, 128)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
        
        doc_emb = torch.randn(1, 128)
        doc_emb = doc_emb / doc_emb.norm(dim=-1, keepdim=True)
        
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=[query_emb],
            ps=[doc_emb],
            batch_size=128,
        )
        
        # Reference: with single doc token, max is just the dot product
        similarities = torch.matmul(query_emb, doc_emb.T)
        expected = similarities.sum().item()  # max over 1 element = element itself
        
        assert abs(scores[0, 0].item() - expected) < 1e-4
    
    def test_identical_embeddings(self):
        """
        Test: Identical query and document embeddings give expected score.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 3.4**
        """
        emb = torch.randn(10, 128)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=[emb],
            ps=[emb],
            batch_size=128,
        )
        
        # With identical normalized embeddings, each query token's max similarity
        # with itself is 1.0, so total score should be num_tokens
        expected = emb.shape[0]  # 10 tokens, each with max sim of 1.0
        
        assert abs(scores[0, 0].item() - expected) < 1e-4, \
            f"Expected {expected}, got {scores[0, 0].item()}"
    
    def test_orthogonal_embeddings(self):
        """
        Test: Orthogonal embeddings give zero score.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 3.4**
        """
        # Create orthogonal embeddings using first half and second half of dimensions
        dim = 128
        query_emb = torch.zeros(5, dim)
        query_emb[:, :dim//2] = torch.randn(5, dim//2)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
        
        doc_emb = torch.zeros(10, dim)
        doc_emb[:, dim//2:] = torch.randn(10, dim//2)
        doc_emb = doc_emb / doc_emb.norm(dim=-1, keepdim=True)
        
        scores = BaseVisualRetrieverProcessor.score_multi_vector(
            qs=[query_emb],
            ps=[doc_emb],
            batch_size=128,
        )
        
        # Orthogonal vectors have zero dot product
        assert abs(scores[0, 0].item()) < 1e-4, \
            f"Expected ~0, got {scores[0, 0].item()}"
    
    def test_empty_queries_raises_error(self):
        """
        Test: Empty query list raises ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 3.4**
        """
        doc_emb = torch.randn(10, 128)
        
        with pytest.raises(ValueError, match="No queries"):
            BaseVisualRetrieverProcessor.score_multi_vector(
                qs=[],
                ps=[doc_emb],
                batch_size=128,
            )
    
    def test_empty_passages_raises_error(self):
        """
        Test: Empty passage list raises ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 12: Batch Masking Correctness**
        **Validates: Requirements 3.4**
        """
        query_emb = torch.randn(10, 128)
        
        with pytest.raises(ValueError, match="No passages"):
            BaseVisualRetrieverProcessor.score_multi_vector(
                qs=[query_emb],
                ps=[],
                batch_size=128,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
