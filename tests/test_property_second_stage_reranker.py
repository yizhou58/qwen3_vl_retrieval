"""
Property-Based Tests for Second Stage Reranking Correctness.

**Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
**Validates: Requirements 6.1, 6.3, 6.4, 6.5**

Property 8: For any set of candidate document IDs and query, the Second_Stage_Reranker 
SHALL compute MaxSim scores only for the specified candidates and return results sorted 
by score in descending order. When binary quantization is enabled, the final ranking 
SHALL be based on rescored full-precision embeddings for top candidates.
"""

import tempfile
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
from typing import List, Tuple, Dict
from unittest.mock import MagicMock, patch

import torch
import numpy as np

from qwen3_vl_retrieval.retrieval.second_stage_reranker import SecondStageReranker
from qwen3_vl_retrieval.retrieval.embedding_store import EmbeddingStore
from qwen3_vl_retrieval.retrieval.binary_quantizer import BinaryQuantizer


def generate_random_embedding(num_tokens: int, dim: int = 128) -> torch.Tensor:
    """Generate random L2-normalized embeddings using torch directly."""
    tensor = torch.randn(num_tokens, dim, dtype=torch.float32)
    tensor = tensor / (tensor.norm(dim=-1, keepdim=True) + 1e-8)
    return tensor


# Hypothesis strategies - simplified to avoid large base example issues
@st.composite
def embedding_params_strategy(draw, min_tokens=3, max_tokens=15):
    """Generate parameters for embeddings instead of full tensors."""
    num_tokens = draw(st.integers(min_value=min_tokens, max_value=max_tokens))
    return num_tokens


@st.composite
def document_collection_params_strategy(draw, max_docs=5, min_tokens=5, max_tokens=15):
    """Generate parameters for a collection of document embeddings."""
    num_docs = draw(st.integers(min_value=1, max_value=max_docs))
    doc_params = []
    for i in range(num_docs):
        num_tokens = draw(st.integers(min_value=min_tokens, max_value=max_tokens))
        doc_params.append((f"doc_{i}", num_tokens))
    return doc_params


class MockModel:
    """Mock ColQwen3VL model for testing."""
    
    def __init__(self, query_embeddings: torch.Tensor):
        self._query_embeddings = query_embeddings
        self._device = torch.device('cpu')
    
    @property
    def device(self):
        return self._device
    
    def eval(self):
        pass
    
    def __call__(self, **kwargs):
        # Return query embeddings with batch dimension
        return self._query_embeddings.unsqueeze(0)


class MockProcessor:
    """Mock ColQwen3VLProcessor for testing."""
    
    def __init__(self, seq_len: int):
        self._seq_len = seq_len
    
    def process_queries(self, texts: List[str]):
        return {
            "input_ids": torch.zeros(1, self._seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, self._seq_len, dtype=torch.long),
        }


class TestProperty8SecondStageRerankingCorrectness:
    """
    Property 8: Second Stage Reranking Correctness
    
    **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
    **Validates: Requirements 6.1, 6.3, 6.4, 6.5**
    
    For any set of candidate document IDs and query, the Second_Stage_Reranker SHALL 
    compute MaxSim scores only for the specified candidates and return results sorted 
    by score in descending order.
    """
    
    @given(
        query_tokens=st.integers(min_value=3, max_value=10),
        doc_params=document_collection_params_strategy(max_docs=5, min_tokens=5, max_tokens=15),
        top_k=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_returns_results_sorted_by_score_descending(
        self,
        query_tokens: int,
        doc_params: List[Tuple[str, int]],
        top_k: int
    ):
        """
        Property: Results are sorted by MaxSim score in descending order.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.3**
        """
        # Generate embeddings from params
        query_emb = generate_random_embedding(query_tokens)
        doc_embs = {doc_id: generate_random_embedding(num_tokens) for doc_id, num_tokens in doc_params}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup embedding store
            store = EmbeddingStore(tmpdir, dim=128)
            quantizer = BinaryQuantizer()
            
            for doc_id, emb in doc_embs.items():
                binary_emb = quantizer.quantize(emb)
                store.add_embeddings(doc_id, emb, binary_emb)
            
            # Create mock model and processor
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(query_emb.shape[0])
            
            # Create reranker (without binary quantization for simpler test)
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            # Rerank
            candidate_ids = list(doc_embs.keys())
            top_k = min(top_k, len(candidate_ids))
            
            results = reranker.rerank_with_precomputed_query(
                query_embeddings=query_emb,
                candidate_doc_ids=candidate_ids,
                top_k=top_k,
            )
            
            # Property: results should be sorted by score descending
            scores = [score for _, score in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], \
                    f"Scores not in descending order at index {i}: {scores[i]} < {scores[i+1]}"
            
            store.close()
    
    @given(
        query_tokens=st.integers(min_value=3, max_value=10),
        doc_params=document_collection_params_strategy(max_docs=5, min_tokens=5, max_tokens=15),
        top_k=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_returns_exactly_top_k_or_fewer(
        self,
        query_tokens: int,
        doc_params: List[Tuple[str, int]],
        top_k: int
    ):
        """
        Property: Returns exactly top_k results or fewer if not enough candidates.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.1**
        """
        query_emb = generate_random_embedding(query_tokens)
        doc_embs = {doc_id: generate_random_embedding(num_tokens) for doc_id, num_tokens in doc_params}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            quantizer = BinaryQuantizer()
            
            for doc_id, emb in doc_embs.items():
                binary_emb = quantizer.quantize(emb)
                store.add_embeddings(doc_id, emb, binary_emb)
            
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(query_emb.shape[0])
            
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            candidate_ids = list(doc_embs.keys())
            results = reranker.rerank_with_precomputed_query(
                query_embeddings=query_emb,
                candidate_doc_ids=candidate_ids,
                top_k=top_k,
            )
            
            expected_count = min(top_k, len(candidate_ids))
            assert len(results) == expected_count, \
                f"Expected {expected_count} results, got {len(results)}"
            
            store.close()
    
    @given(
        query_tokens=st.integers(min_value=3, max_value=10),
        doc_params=document_collection_params_strategy(max_docs=6, min_tokens=5, max_tokens=15)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_only_scores_specified_candidates(
        self,
        query_tokens: int,
        doc_params: List[Tuple[str, int]]
    ):
        """
        Property: Only computes scores for specified candidate documents.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.1**
        """
        query_emb = generate_random_embedding(query_tokens)
        doc_embs = {doc_id: generate_random_embedding(num_tokens) for doc_id, num_tokens in doc_params}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            quantizer = BinaryQuantizer()
            
            for doc_id, emb in doc_embs.items():
                binary_emb = quantizer.quantize(emb)
                store.add_embeddings(doc_id, emb, binary_emb)
            
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(query_emb.shape[0])
            
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            # Only use subset of candidates
            all_ids = list(doc_embs.keys())
            if len(all_ids) > 2:
                candidate_ids = all_ids[:len(all_ids) // 2]
            else:
                candidate_ids = all_ids
            
            results = reranker.rerank_with_precomputed_query(
                query_embeddings=query_emb,
                candidate_doc_ids=candidate_ids,
                top_k=len(candidate_ids),
            )
            
            # Property: all returned doc_ids should be in candidate_ids
            result_ids = [doc_id for doc_id, _ in results]
            for doc_id in result_ids:
                assert doc_id in candidate_ids, \
                    f"Returned doc_id '{doc_id}' not in specified candidates"
            
            # Property: no doc_ids outside candidates should appear
            excluded_ids = set(all_ids) - set(candidate_ids)
            for doc_id in result_ids:
                assert doc_id not in excluded_ids, \
                    f"Doc_id '{doc_id}' should not be in results (not a candidate)"
            
            store.close()
    
    @given(
        query_tokens=st.integers(min_value=3, max_value=10),
        doc_params=document_collection_params_strategy(max_docs=8, min_tokens=5, max_tokens=15),
        top_k=st.integers(min_value=1, max_value=5),
        binary_rescore_ratio=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_binary_quantization_final_ranking_uses_float(
        self,
        query_tokens: int,
        doc_params: List[Tuple[str, int]],
        top_k: int,
        binary_rescore_ratio: int
    ):
        """
        Property: When binary quantization is enabled, final ranking uses float embeddings.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.5**
        """
        query_emb = generate_random_embedding(query_tokens)
        doc_embs = {doc_id: generate_random_embedding(num_tokens) for doc_id, num_tokens in doc_params}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            quantizer = BinaryQuantizer()
            
            for doc_id, emb in doc_embs.items():
                binary_emb = quantizer.quantize(emb)
                store.add_embeddings(doc_id, emb, binary_emb)
            
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(query_emb.shape[0])
            
            # Create reranker WITH binary quantization
            reranker_binary = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=True,
            )
            
            # Create reranker WITHOUT binary quantization
            reranker_float = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            candidate_ids = list(doc_embs.keys())
            top_k = min(top_k, len(candidate_ids))
            
            results_binary = reranker_binary.rerank_with_precomputed_query(
                query_embeddings=query_emb,
                candidate_doc_ids=candidate_ids,
                top_k=top_k,
                binary_rescore_ratio=binary_rescore_ratio,
            )
            
            results_float = reranker_float.rerank_with_precomputed_query(
                query_embeddings=query_emb,
                candidate_doc_ids=candidate_ids,
                top_k=top_k,
            )
            
            # Property: Final scores from binary path should match float-only scores
            # for the same documents (since rescoring uses float embeddings)
            binary_scores_dict = {doc_id: score for doc_id, score in results_binary}
            float_scores_dict = {doc_id: score for doc_id, score in results_float}
            
            for doc_id in binary_scores_dict:
                if doc_id in float_scores_dict:
                    # Scores should be equal (both computed with float embeddings)
                    assert abs(binary_scores_dict[doc_id] - float_scores_dict[doc_id]) < 1e-5, \
                        f"Score mismatch for {doc_id}: binary={binary_scores_dict[doc_id]}, float={float_scores_dict[doc_id]}"
            
            store.close()


class TestProperty8MaxSimComputation:
    """
    Tests for MaxSim computation correctness in SecondStageReranker.
    
    **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
    **Validates: Requirements 6.1, 6.3**
    """
    
    @given(
        query_tokens=st.integers(min_value=3, max_value=10),
        doc_tokens=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_maxsim_computation_matches_reference(
        self,
        query_tokens: int,
        doc_tokens: int
    ):
        """
        Property: MaxSim computation matches reference implementation.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.1**
        """
        query_emb = generate_random_embedding(query_tokens)
        doc_emb = generate_random_embedding(doc_tokens)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(query_emb.shape[0])
            
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            # Compute MaxSim using reranker
            score = reranker.compute_maxsim(query_emb, doc_emb)
            
            # Reference implementation
            similarities = torch.matmul(query_emb, doc_emb.T)
            max_sims = similarities.max(dim=1)[0]
            expected_score = max_sims.sum()
            
            assert torch.allclose(score, expected_score, atol=1e-5), \
                f"MaxSim mismatch: got {score.item()}, expected {expected_score.item()}"
            
            store.close()
    
    @given(
        query_tokens=st.integers(min_value=3, max_value=10),
        doc_tokens=st.integers(min_value=5, max_value=20),
        num_padding=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_maxsim_with_padding_mask(
        self,
        query_tokens: int,
        doc_tokens: int,
        num_padding: int
    ):
        """
        Property: MaxSim correctly handles padding mask.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.1**
        """
        query_emb = generate_random_embedding(query_tokens)
        doc_emb = generate_random_embedding(doc_tokens)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(query_emb.shape[0])
            
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            # Add padding to document embeddings
            padding = torch.zeros(num_padding, doc_emb.shape[1])
            padded_doc_emb = torch.cat([doc_emb, padding], dim=0)
            
            # Create mask (True for valid, False for padding)
            mask = torch.cat([
                torch.ones(doc_emb.shape[0], dtype=torch.bool),
                torch.zeros(num_padding, dtype=torch.bool)
            ])
            
            # Compute with mask
            score_with_mask = reranker.compute_maxsim(query_emb, padded_doc_emb, doc_mask=mask)
            
            # Compute without padding (reference)
            score_no_padding = reranker.compute_maxsim(query_emb, doc_emb)
            
            # Property: scores should be equal (padding should not affect result)
            assert torch.allclose(score_with_mask, score_no_padding, atol=1e-5), \
                f"Padding mask not working: with_mask={score_with_mask.item()}, no_padding={score_no_padding.item()}"
            
            store.close()


class TestProperty8EdgeCases:
    """
    Edge case tests for Second Stage Reranking.
    
    **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
    **Validates: Requirements 6.1, 6.3, 6.4, 6.5**
    """
    
    def test_empty_candidates_returns_empty(self):
        """
        Test: Empty candidate list returns empty results.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.1**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            
            query_emb = torch.randn(10, 128)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(10)
            
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            results = reranker.rerank_with_precomputed_query(
                query_embeddings=query_emb,
                candidate_doc_ids=[],
                top_k=10,
            )
            
            assert results == [], "Empty candidates should return empty results"
            
            store.close()
    
    def test_single_candidate(self):
        """
        Test: Single candidate is returned correctly.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.1**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            quantizer = BinaryQuantizer()
            
            doc_emb = torch.randn(20, 128)
            doc_emb = doc_emb / doc_emb.norm(dim=-1, keepdim=True)
            binary_emb = quantizer.quantize(doc_emb)
            store.add_embeddings("doc_0", doc_emb, binary_emb)
            
            query_emb = torch.randn(10, 128)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(10)
            
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            results = reranker.rerank_with_precomputed_query(
                query_embeddings=query_emb,
                candidate_doc_ids=["doc_0"],
                top_k=10,
            )
            
            assert len(results) == 1
            assert results[0][0] == "doc_0"
            
            store.close()
    
    def test_missing_embeddings_handled_gracefully(self):
        """
        Test: Missing embeddings are handled gracefully.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.1**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            quantizer = BinaryQuantizer()
            
            # Only add one document
            doc_emb = torch.randn(20, 128)
            doc_emb = doc_emb / doc_emb.norm(dim=-1, keepdim=True)
            binary_emb = quantizer.quantize(doc_emb)
            store.add_embeddings("doc_0", doc_emb, binary_emb)
            
            query_emb = torch.randn(10, 128)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(10)
            
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            # Request candidates including non-existent ones
            results = reranker.rerank_with_precomputed_query(
                query_embeddings=query_emb,
                candidate_doc_ids=["doc_0", "doc_nonexistent"],
                top_k=10,
            )
            
            # Should only return the existing document
            assert len(results) == 1
            assert results[0][0] == "doc_0"
            
            store.close()
    
    def test_batch_rerank_mismatched_lengths_raises_error(self):
        """
        Test: Batch rerank with mismatched lengths raises ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            
            query_emb = torch.randn(10, 128)
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(10)
            
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            with pytest.raises(ValueError, match="must match"):
                reranker.rerank_batch(
                    queries=["query1", "query2"],
                    candidate_doc_ids_list=[["doc1"]],  # Mismatched length
                    top_k=10,
                )
            
            store.close()


class TestProperty8BatchProcessing:
    """
    Tests for batch processing in Second Stage Reranking.
    
    **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
    **Validates: Requirements 6.4**
    """
    
    @given(
        num_queries=st.integers(min_value=1, max_value=5),
        num_docs=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_batch_rerank_returns_correct_number_of_results(
        self,
        num_queries: int,
        num_docs: int
    ):
        """
        Property: Batch rerank returns correct number of result lists.
        
        **Feature: qwen3-vl-rag-retrieval, Property 8: Second Stage Reranking Correctness**
        **Validates: Requirements 6.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmbeddingStore(tmpdir, dim=128)
            quantizer = BinaryQuantizer()
            
            # Create documents
            doc_ids = []
            for i in range(num_docs):
                doc_id = f"doc_{i}"
                doc_ids.append(doc_id)
                doc_emb = torch.randn(20, 128)
                doc_emb = doc_emb / doc_emb.norm(dim=-1, keepdim=True)
                binary_emb = quantizer.quantize(doc_emb)
                store.add_embeddings(doc_id, doc_emb, binary_emb)
            
            query_emb = torch.randn(10, 128)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            
            mock_model = MockModel(query_emb)
            mock_processor = MockProcessor(10)
            
            reranker = SecondStageReranker(
                model=mock_model,
                processor=mock_processor,
                embedding_store=store,
                use_binary_quantization=False,
            )
            
            queries = [f"query_{i}" for i in range(num_queries)]
            candidate_lists = [doc_ids for _ in range(num_queries)]
            
            results = reranker.rerank_batch(
                queries=queries,
                candidate_doc_ids_list=candidate_lists,
                top_k=5,
            )
            
            assert len(results) == num_queries, \
                f"Expected {num_queries} result lists, got {len(results)}"
            
            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
