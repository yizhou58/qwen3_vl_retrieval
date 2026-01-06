"""
Hard Negative Mining Implementation.

Selects hard negatives using BM25 or dense retriever for contrastive learning.

Requirements: 7.6
- Support Hard Negative Mining using BM25 or initial dense retriever
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging

import torch

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Hard Negative Mining for contrastive learning.
    
    Selects top-ranked incorrect documents as hard negatives for training.
    Supports BM25 and dense retriever methods.
    
    Requirements:
        7.6: Support Hard Negative Mining using BM25 or initial dense retriever
    """
    
    def __init__(
        self,
        method: str = "bm25",
        retriever: Optional[Any] = None,
        num_negatives: int = 5,
        exclude_positives: bool = True,
    ):
        """
        Initialize HardNegativeMiner.
        
        Args:
            method: Mining method ("bm25" or "dense")
            retriever: Pre-initialized retriever (FirstStageRetriever)
            num_negatives: Number of hard negatives to mine per query
            exclude_positives: Whether to exclude positive documents
        """
        self.method = method
        self.retriever = retriever
        self.num_negatives = num_negatives
        self.exclude_positives = exclude_positives
        
        # BM25 index for text-based mining
        self._bm25_index = None
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
    
    def build_index(
        self,
        doc_ids: List[str],
        doc_texts: List[str],
    ) -> None:
        """
        Build index for hard negative mining.
        
        Args:
            doc_ids: List of document IDs
            doc_texts: List of document texts (OCR or metadata)
        """
        self._doc_ids = doc_ids
        self._doc_texts = doc_texts
        
        if self.method == "bm25":
            self._build_bm25_index(doc_texts)
        elif self.method == "dense" and self.retriever is not None:
            # Use existing retriever index
            logger.info("Using pre-built dense retriever index")
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _build_bm25_index(self, doc_texts: List[str]) -> None:
        """Build BM25 index from document texts."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 is required for BM25 mining. Install with: pip install rank-bm25")
        
        # Tokenize documents
        tokenized_docs = [self._tokenize(text) for text in doc_texts]
        self._bm25_index = BM25Okapi(tokenized_docs)
        
        logger.info(f"Built BM25 index with {len(doc_texts)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()
    
    def mine_negatives(
        self,
        query: str,
        positive_doc_ids: List[str],
        top_k: int = 100,
    ) -> List[str]:
        """
        Mine hard negatives for a query.
        
        Args:
            query: Query text
            positive_doc_ids: List of positive document IDs to exclude
            top_k: Number of candidates to retrieve before filtering
            
        Returns:
            List of hard negative document IDs
        """
        if self.method == "bm25":
            return self._mine_bm25(query, positive_doc_ids, top_k)
        elif self.method == "dense":
            return self._mine_dense(query, positive_doc_ids, top_k)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _mine_bm25(
        self,
        query: str,
        positive_doc_ids: List[str],
        top_k: int,
    ) -> List[str]:
        """Mine hard negatives using BM25."""
        if self._bm25_index is None:
            raise ValueError("BM25 index not built. Call build_index() first.")
        
        # Get BM25 scores
        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Filter out positives and collect negatives
        negatives = []
        positive_set = set(positive_doc_ids)
        
        for idx in top_indices:
            doc_id = self._doc_ids[idx]
            if self.exclude_positives and doc_id in positive_set:
                continue
            negatives.append(doc_id)
            if len(negatives) >= self.num_negatives:
                break
        
        return negatives
    
    def _mine_dense(
        self,
        query: str,
        positive_doc_ids: List[str],
        top_k: int,
    ) -> List[str]:
        """Mine hard negatives using dense retriever."""
        if self.retriever is None:
            raise ValueError("Dense retriever not provided.")
        
        # Retrieve candidates
        results = self.retriever.retrieve(query, top_k=top_k)
        
        # Filter out positives and collect negatives
        negatives = []
        positive_set = set(positive_doc_ids)
        
        for doc_id, score, _ in results:
            if self.exclude_positives and doc_id in positive_set:
                continue
            negatives.append(doc_id)
            if len(negatives) >= self.num_negatives:
                break
        
        return negatives
    
    def mine_batch(
        self,
        queries: List[str],
        positive_doc_ids_list: List[List[str]],
        top_k: int = 100,
    ) -> Dict[str, List[str]]:
        """
        Mine hard negatives for a batch of queries.
        
        Args:
            queries: List of query texts
            positive_doc_ids_list: List of positive doc ID lists per query
            top_k: Number of candidates to retrieve
            
        Returns:
            Dictionary mapping query index to hard negative doc IDs
        """
        results = {}
        
        for i, (query, positives) in enumerate(zip(queries, positive_doc_ids_list)):
            negatives = self.mine_negatives(query, positives, top_k)
            results[str(i)] = negatives
        
        return results
    
    def save_negatives(
        self,
        negatives: Dict[str, List[str]],
        output_path: Union[str, Path],
    ) -> None:
        """
        Save mined hard negatives to file.
        
        Args:
            negatives: Dictionary of hard negatives
            output_path: Output file path (JSON format)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(negatives, f, indent=2)
        
        logger.info(f"Saved hard negatives to {output_path}")
    
    @classmethod
    def load_negatives(cls, path: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Load hard negatives from file.
        
        Args:
            path: Path to hard negatives file
            
        Returns:
            Dictionary of hard negatives
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def mine_hard_negatives_for_dataset(
    dataset: Any,
    method: str = "bm25",
    num_negatives: int = 5,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """
    Mine hard negatives for an entire dataset.
    
    Args:
        dataset: ColPaliEngineDataset or similar
        method: Mining method ("bm25" or "dense")
        num_negatives: Number of negatives per query
        output_path: Optional path to save results
        
    Returns:
        Dictionary mapping doc_id to hard negative doc_ids
    """
    # Extract document texts (use query as proxy if no OCR text available)
    doc_ids = []
    doc_texts = []
    queries = []
    positive_doc_ids_list = []
    
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        doc_ids.append(sample["doc_id"])
        # Use query as document text if no OCR available
        doc_texts.append(sample.get("text", sample["query"]))
        queries.append(sample["query"])
        positive_doc_ids_list.append([sample["doc_id"]])
    
    # Build miner and index
    miner = HardNegativeMiner(method=method, num_negatives=num_negatives)
    miner.build_index(doc_ids, doc_texts)
    
    # Mine negatives
    negatives = {}
    for i, (query, positives) in enumerate(zip(queries, positive_doc_ids_list)):
        doc_id = doc_ids[i]
        negatives[doc_id] = miner.mine_negatives(query, positives)
    
    # Save if output path provided
    if output_path:
        miner.save_negatives(negatives, output_path)
    
    return negatives
