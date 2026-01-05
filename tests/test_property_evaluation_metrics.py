"""
Property-Based Tests for Evaluation Metrics.

Property 11: Evaluation Metrics Computation
For any set of queries with known relevant documents and a ranking:
- MRR SHALL equal the mean of 1/rank for the first relevant document
- Recall@K SHALL equal the fraction of relevant documents in top K
- NDCG@K SHALL be computed using standard DCG normalization

**Validates: Requirements 9.1**

Feature: qwen3-vl-rag-retrieval, Property 11: Evaluation Metrics Computation
"""

import math
import pytest
from hypothesis import given, settings, strategies as st, assume
from typing import List, Set


# Test data generators
@st.composite
def doc_id_strategy(draw):
    """Generate valid document IDs."""
    return f"doc_{draw(st.integers(min_value=0, max_value=1000))}"


@st.composite
def ranking_strategy(draw, min_size=1, max_size=20):
    """Generate a ranking (list of unique doc IDs)."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    doc_ids = [f"doc_{i}" for i in range(size)]
    return doc_ids


@st.composite
def relevant_docs_strategy(draw, ranking: List[str]):
    """Generate relevant docs that may or may not be in ranking."""
    # Some from ranking, some not
    from_ranking = draw(st.lists(
        st.sampled_from(ranking) if ranking else st.just("doc_0"),
        min_size=0,
        max_size=min(5, len(ranking)),
        unique=True,
    ))
    
    # Some not in ranking
    extra = draw(st.lists(
        st.integers(min_value=1000, max_value=2000).map(lambda x: f"doc_{x}"),
        min_size=0,
        max_size=3,
        unique=True,
    ))
    
    return set(from_ranking + extra)


@st.composite
def query_results_strategy(draw, num_queries_range=(1, 10)):
    """Generate multiple query results with rankings and relevant docs."""
    num_queries = draw(st.integers(min_value=num_queries_range[0], max_value=num_queries_range[1]))
    
    rankings = []
    relevant_docs = []
    
    for _ in range(num_queries):
        ranking = draw(ranking_strategy())
        relevant = draw(relevant_docs_strategy(ranking))
        rankings.append(ranking)
        relevant_docs.append(relevant)
    
    return rankings, relevant_docs


class TestMRRProperty:
    """
    Property tests for Mean Reciprocal Rank (MRR).
    
    MRR = (1/|Q|) * Σ (1/rank_i)
    
    **Validates: Requirements 9.1**
    """
    
    def test_mrr_perfect_ranking(self):
        """
        Property: MRR = 1.0 when first document is always relevant.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_mrr
        
        rankings = [
            ["doc_0", "doc_1", "doc_2"],
            ["doc_3", "doc_4", "doc_5"],
        ]
        relevant_docs = [
            {"doc_0"},  # First doc is relevant
            {"doc_3"},  # First doc is relevant
        ]
        
        mrr = compute_mrr(rankings, relevant_docs)
        
        # Property: MRR = 1.0 for perfect ranking
        assert mrr == 1.0, f"MRR should be 1.0 for perfect ranking, got {mrr}"
    
    def test_mrr_second_position(self):
        """
        Property: MRR = 0.5 when relevant doc is always at position 2.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_mrr
        
        rankings = [
            ["doc_0", "doc_1", "doc_2"],
            ["doc_3", "doc_4", "doc_5"],
        ]
        relevant_docs = [
            {"doc_1"},  # Second doc is relevant
            {"doc_4"},  # Second doc is relevant
        ]
        
        mrr = compute_mrr(rankings, relevant_docs)
        
        # Property: MRR = 0.5 when relevant at position 2
        assert mrr == 0.5, f"MRR should be 0.5, got {mrr}"
    
    def test_mrr_no_relevant(self):
        """
        Property: MRR = 0.0 when no relevant documents in ranking.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_mrr
        
        rankings = [["doc_0", "doc_1", "doc_2"]]
        relevant_docs = [{"doc_99"}]  # Not in ranking
        
        mrr = compute_mrr(rankings, relevant_docs)
        
        assert mrr == 0.0, f"MRR should be 0.0 when no relevant docs, got {mrr}"
    
    @given(query_results=query_results_strategy())
    @settings(max_examples=100, deadline=None)
    def test_mrr_bounds(self, query_results):
        """
        Property: MRR is always between 0 and 1.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_mrr
        
        rankings, relevant_docs = query_results
        mrr = compute_mrr(rankings, relevant_docs)
        
        assert 0.0 <= mrr <= 1.0, f"MRR should be in [0, 1], got {mrr}"


class TestRecallProperty:
    """
    Property tests for Recall@K.
    
    Recall@K = |relevant ∩ top_k| / |relevant|
    
    **Validates: Requirements 9.1**
    """
    
    def test_recall_perfect(self):
        """
        Property: Recall@K = 1.0 when all relevant docs are in top K.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_recall_at_k
        
        rankings = [["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]]
        relevant_docs = [{"doc_0", "doc_1"}]
        
        recall = compute_recall_at_k(rankings, relevant_docs, k=5)
        
        assert recall == 1.0, f"Recall should be 1.0, got {recall}"
    
    def test_recall_partial(self):
        """
        Property: Recall@K = 0.5 when half of relevant docs are in top K.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_recall_at_k
        
        rankings = [["doc_0", "doc_1", "doc_2"]]
        relevant_docs = [{"doc_0", "doc_99"}]  # doc_99 not in ranking
        
        recall = compute_recall_at_k(rankings, relevant_docs, k=3)
        
        assert recall == 0.5, f"Recall should be 0.5, got {recall}"
    
    def test_recall_zero(self):
        """
        Property: Recall@K = 0.0 when no relevant docs in top K.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_recall_at_k
        
        rankings = [["doc_0", "doc_1", "doc_2"]]
        relevant_docs = [{"doc_99", "doc_100"}]
        
        recall = compute_recall_at_k(rankings, relevant_docs, k=3)
        
        assert recall == 0.0, f"Recall should be 0.0, got {recall}"
    
    @given(query_results=query_results_strategy())
    @settings(max_examples=100, deadline=None)
    def test_recall_bounds(self, query_results):
        """
        Property: Recall@K is always between 0 and 1.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_recall_at_k
        
        rankings, relevant_docs = query_results
        
        for k in [1, 5, 10]:
            recall = compute_recall_at_k(rankings, relevant_docs, k)
            assert 0.0 <= recall <= 1.0, f"Recall@{k} should be in [0, 1], got {recall}"
    
    @given(query_results=query_results_strategy())
    @settings(max_examples=100, deadline=None)
    def test_recall_monotonic(self, query_results):
        """
        Property: Recall@K is monotonically non-decreasing with K.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_recall_at_k
        
        rankings, relevant_docs = query_results
        
        prev_recall = 0.0
        for k in [1, 2, 5, 10, 20]:
            recall = compute_recall_at_k(rankings, relevant_docs, k)
            assert recall >= prev_recall - 1e-9, \
                f"Recall@{k} ({recall}) should be >= Recall@{k-1} ({prev_recall})"
            prev_recall = recall


class TestNDCGProperty:
    """
    Property tests for NDCG@K.
    
    DCG@K = Σ (rel_i / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K
    
    **Validates: Requirements 9.1**
    """
    
    def test_ndcg_perfect_ranking(self):
        """
        Property: NDCG@K = 1.0 for perfect ranking.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_ndcg_at_k
        
        # All relevant docs at top positions
        rankings = [["doc_0", "doc_1", "doc_2", "doc_3"]]
        relevant_docs = [{"doc_0", "doc_1"}]
        
        ndcg = compute_ndcg_at_k(rankings, relevant_docs, k=4)
        
        assert abs(ndcg - 1.0) < 1e-6, f"NDCG should be 1.0 for perfect ranking, got {ndcg}"
    
    def test_ndcg_worst_ranking(self):
        """
        Property: NDCG@K < 1.0 for suboptimal ranking.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_ndcg_at_k
        
        # Relevant docs not at top
        rankings = [["doc_2", "doc_3", "doc_0", "doc_1"]]
        relevant_docs = [{"doc_0", "doc_1"}]
        
        ndcg = compute_ndcg_at_k(rankings, relevant_docs, k=4)
        
        assert ndcg < 1.0, f"NDCG should be < 1.0 for suboptimal ranking, got {ndcg}"
    
    def test_ndcg_no_relevant(self):
        """
        Property: NDCG@K = 0.0 when no relevant docs.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_ndcg_at_k
        
        rankings = [["doc_0", "doc_1", "doc_2"]]
        relevant_docs = [{"doc_99"}]
        
        ndcg = compute_ndcg_at_k(rankings, relevant_docs, k=3)
        
        assert ndcg == 0.0, f"NDCG should be 0.0 when no relevant docs, got {ndcg}"
    
    @given(query_results=query_results_strategy())
    @settings(max_examples=100, deadline=None)
    def test_ndcg_bounds(self, query_results):
        """
        Property: NDCG@K is always between 0 and 1.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_ndcg_at_k
        
        rankings, relevant_docs = query_results
        
        for k in [5, 10, 20]:
            ndcg = compute_ndcg_at_k(rankings, relevant_docs, k)
            assert 0.0 <= ndcg <= 1.0, f"NDCG@{k} should be in [0, 1], got {ndcg}"
    
    def test_ndcg_dcg_formula(self):
        """
        Property: NDCG follows DCG formula correctly.
        """
        from qwen3_vl_retrieval.evaluation.metrics import compute_ndcg_at_k
        
        # Manual calculation
        rankings = [["doc_0", "doc_1", "doc_2"]]
        relevant_docs = [{"doc_0", "doc_2"}]  # positions 1 and 3
        
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.631 = 1.631
        # NDCG = 1.5 / 1.631 ≈ 0.92
        
        ndcg = compute_ndcg_at_k(rankings, relevant_docs, k=3)
        
        expected_dcg = 1/math.log2(2) + 0/math.log2(3) + 1/math.log2(4)
        expected_idcg = 1/math.log2(2) + 1/math.log2(3)
        expected_ndcg = expected_dcg / expected_idcg
        
        assert abs(ndcg - expected_ndcg) < 1e-6, \
            f"NDCG should be {expected_ndcg:.4f}, got {ndcg:.4f}"


class TestRetrievalMetricsClass:
    """
    Tests for the RetrievalMetrics class.
    """
    
    def test_compute_all_returns_all_metrics(self):
        """
        Property: compute_all returns all expected metrics.
        """
        from qwen3_vl_retrieval.evaluation.metrics import RetrievalMetrics
        
        metrics = RetrievalMetrics(
            recall_k_values=[1, 5, 10],
            ndcg_k_values=[5, 10],
        )
        
        rankings = [["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]]
        relevant_docs = [{"doc_0", "doc_2"}]
        
        results = metrics.compute_all(rankings, relevant_docs)
        
        # Check all expected keys
        assert "MRR" in results
        assert "MAP" in results
        assert "Recall@1" in results
        assert "Recall@5" in results
        assert "Recall@10" in results
        assert "NDCG@5" in results
        assert "NDCG@10" in results
    
    def test_format_results(self):
        """
        Property: format_results produces readable output.
        """
        from qwen3_vl_retrieval.evaluation.metrics import RetrievalMetrics
        
        metrics = RetrievalMetrics()
        results = {"MRR": 0.75, "Recall@10": 0.8}
        
        formatted = metrics.format_results(results)
        
        assert "MRR" in formatted
        assert "0.75" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
