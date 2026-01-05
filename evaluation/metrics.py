"""
Evaluation Metrics for Retrieval System.

Implements standard retrieval metrics: MRR, Recall@K, NDCG@K.

Requirements: 9.1
- Compute standard retrieval metrics: MRR, Recall@K, NDCG@K
"""

from typing import Dict, List, Optional, Set, Union
import math
import numpy as np


def compute_mrr(
    rankings: List[List[str]],
    relevant_docs: List[Set[str]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    MRR = (1/|Q|) * Σ (1/rank_i)
    
    where rank_i is the rank of the first relevant document for query i.
    
    Args:
        rankings: List of ranked document IDs per query
        relevant_docs: List of sets of relevant document IDs per query
        
    Returns:
        MRR score (0 to 1)
        
    Requirements:
        9.1: Compute MRR
    """
    if len(rankings) != len(relevant_docs):
        raise ValueError("rankings and relevant_docs must have same length")
    
    if len(rankings) == 0:
        return 0.0
    
    reciprocal_ranks = []
    
    for ranking, relevant in zip(rankings, relevant_docs):
        rr = 0.0
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_recall_at_k(
    rankings: List[List[str]],
    relevant_docs: List[Set[str]],
    k: int,
) -> float:
    """
    Compute Recall@K.
    
    Recall@K = (1/|Q|) * Σ (|relevant ∩ top_k| / |relevant|)
    
    Args:
        rankings: List of ranked document IDs per query
        relevant_docs: List of sets of relevant document IDs per query
        k: Number of top documents to consider
        
    Returns:
        Recall@K score (0 to 1)
        
    Requirements:
        9.1: Compute Recall@K
    """
    if len(rankings) != len(relevant_docs):
        raise ValueError("rankings and relevant_docs must have same length")
    
    if len(rankings) == 0:
        return 0.0
    
    recalls = []
    
    for ranking, relevant in zip(rankings, relevant_docs):
        if len(relevant) == 0:
            recalls.append(0.0)
            continue
        
        top_k = set(ranking[:k])
        hits = len(top_k & relevant)
        recalls.append(hits / len(relevant))
    
    return sum(recalls) / len(recalls)


def compute_ndcg_at_k(
    rankings: List[List[str]],
    relevant_docs: List[Set[str]],
    k: int,
    relevance_scores: Optional[List[Dict[str, float]]] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).
    
    DCG@K = Σ (rel_i / log2(i + 1)) for i in 1..K
    NDCG@K = DCG@K / IDCG@K
    
    Args:
        rankings: List of ranked document IDs per query
        relevant_docs: List of sets of relevant document IDs per query
        k: Number of top documents to consider
        relevance_scores: Optional relevance scores (default: binary 1/0)
        
    Returns:
        NDCG@K score (0 to 1)
        
    Requirements:
        9.1: Compute NDCG@K
    """
    if len(rankings) != len(relevant_docs):
        raise ValueError("rankings and relevant_docs must have same length")
    
    if len(rankings) == 0:
        return 0.0
    
    ndcg_scores = []
    
    for i, (ranking, relevant) in enumerate(zip(rankings, relevant_docs)):
        if len(relevant) == 0:
            ndcg_scores.append(0.0)
            continue
        
        # Get relevance scores
        if relevance_scores is not None:
            rel_dict = relevance_scores[i]
        else:
            rel_dict = {doc_id: 1.0 for doc_id in relevant}
        
        # Compute DCG@K
        dcg = 0.0
        for rank, doc_id in enumerate(ranking[:k], start=1):
            rel = rel_dict.get(doc_id, 0.0)
            dcg += rel / math.log2(rank + 1)
        
        # Compute IDCG@K (ideal DCG)
        ideal_rels = sorted([rel_dict.get(doc_id, 0.0) for doc_id in relevant], reverse=True)
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels[:k], start=1):
            idcg += rel / math.log2(rank + 1)
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    
    return sum(ndcg_scores) / len(ndcg_scores)


def compute_precision_at_k(
    rankings: List[List[str]],
    relevant_docs: List[Set[str]],
    k: int,
) -> float:
    """
    Compute Precision@K.
    
    Precision@K = (1/|Q|) * Σ (|relevant ∩ top_k| / k)
    
    Args:
        rankings: List of ranked document IDs per query
        relevant_docs: List of sets of relevant document IDs per query
        k: Number of top documents to consider
        
    Returns:
        Precision@K score (0 to 1)
    """
    if len(rankings) != len(relevant_docs):
        raise ValueError("rankings and relevant_docs must have same length")
    
    if len(rankings) == 0:
        return 0.0
    
    precisions = []
    
    for ranking, relevant in zip(rankings, relevant_docs):
        top_k = set(ranking[:k])
        hits = len(top_k & relevant)
        precisions.append(hits / k)
    
    return sum(precisions) / len(precisions)


def compute_map(
    rankings: List[List[str]],
    relevant_docs: List[Set[str]],
) -> float:
    """
    Compute Mean Average Precision (MAP).
    
    AP = (1/|relevant|) * Σ (P@k * rel_k)
    MAP = (1/|Q|) * Σ AP_i
    
    Args:
        rankings: List of ranked document IDs per query
        relevant_docs: List of sets of relevant document IDs per query
        
    Returns:
        MAP score (0 to 1)
    """
    if len(rankings) != len(relevant_docs):
        raise ValueError("rankings and relevant_docs must have same length")
    
    if len(rankings) == 0:
        return 0.0
    
    average_precisions = []
    
    for ranking, relevant in zip(rankings, relevant_docs):
        if len(relevant) == 0:
            average_precisions.append(0.0)
            continue
        
        hits = 0
        sum_precisions = 0.0
        
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in relevant:
                hits += 1
                sum_precisions += hits / rank
        
        average_precisions.append(sum_precisions / len(relevant))
    
    return sum(average_precisions) / len(average_precisions)


class RetrievalMetrics:
    """
    Comprehensive retrieval metrics calculator.
    
    Computes multiple metrics at once for efficiency.
    """
    
    def __init__(
        self,
        recall_k_values: List[int] = None,
        ndcg_k_values: List[int] = None,
    ):
        self.recall_k_values = recall_k_values or [1, 5, 10, 20, 50, 100]
        self.ndcg_k_values = ndcg_k_values or [5, 10, 20]
    
    def compute_all(
        self,
        rankings: List[List[str]],
        relevant_docs: List[Set[str]],
        relevance_scores: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Compute all retrieval metrics.
        
        Args:
            rankings: List of ranked document IDs per query
            relevant_docs: List of sets of relevant document IDs per query
            relevance_scores: Optional relevance scores
            
        Returns:
            Dictionary of metric names to values
        """
        results = {}
        
        # MRR
        results["MRR"] = compute_mrr(rankings, relevant_docs)
        
        # MAP
        results["MAP"] = compute_map(rankings, relevant_docs)
        
        # Recall@K
        for k in self.recall_k_values:
            results[f"Recall@{k}"] = compute_recall_at_k(rankings, relevant_docs, k)
        
        # NDCG@K
        for k in self.ndcg_k_values:
            results[f"NDCG@{k}"] = compute_ndcg_at_k(
                rankings, relevant_docs, k, relevance_scores
            )
        
        # Precision@K (for common values)
        for k in [1, 5, 10]:
            results[f"Precision@{k}"] = compute_precision_at_k(rankings, relevant_docs, k)
        
        return results
    
    def format_results(self, results: Dict[str, float]) -> str:
        """Format results as a readable string."""
        lines = ["Retrieval Metrics:"]
        lines.append("-" * 40)
        
        for metric, value in sorted(results.items()):
            lines.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(lines)
