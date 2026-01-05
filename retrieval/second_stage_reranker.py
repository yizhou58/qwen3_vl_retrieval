"""
Second-stage reranking module using pre-computed document embeddings.

Implements two-step reranking strategy:
1. Fast filtering using binary embeddings (Hamming distance)
2. Precise rescoring using float embeddings (MaxSim)

Requirements: 6.1, 6.3, 6.4, 6.5
- Compute MaxSim scores only for first-stage candidates
- Retrieve pre-computed multi-vector embeddings from storage
- Return documents sorted by MaxSim score in descending order
- Support batch processing to improve throughput
- When binary quantization is enabled, first filter using binary scores
  then rescore top candidates with full-precision embeddings
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch

from .binary_quantizer import BinaryQuantizer
from .embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


class SecondStageReranker:
    """
    Second-stage reranking using pre-computed document embeddings.
    
    Implements two-step reranking strategy:
    1. Fast filtering using binary embeddings (Hamming distance)
    2. Precise rescoring using float embeddings (MaxSim)
    
    Attributes:
        model: ColQwen3VL model for encoding queries
        processor: ColQwen3VLProcessor for preprocessing
        embedding_store: EmbeddingStore for retrieving pre-computed embeddings
        use_binary_quantization: Whether to use binary pre-filtering
        quantizer: BinaryQuantizer instance for binary operations
        
    Requirements:
        6.1: Compute MaxSim scores only for first-stage candidates
        6.3: Return documents sorted by MaxSim score in descending order
        6.4: Support batch processing to improve throughput
        6.5: When binary quantization is enabled, first filter using binary scores
             then rescore top candidates with full-precision embeddings
    """
    
    def __init__(
        self,
        model: "ColQwen3VL",
        processor: "ColQwen3VLProcessor",
        embedding_store: EmbeddingStore,
        use_binary_quantization: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize SecondStageReranker.
        
        Args:
            model: ColQwen3VL model for encoding queries
            processor: ColQwen3VLProcessor for preprocessing
            embedding_store: EmbeddingStore for retrieving pre-computed embeddings
            use_binary_quantization: Whether to use binary pre-filtering (default: True)
            device: Device for computation (default: model's device)
        """
        self.model = model
        self.processor = processor
        self.embedding_store = embedding_store
        self.use_binary_quantization = use_binary_quantization
        self.quantizer = BinaryQuantizer() if use_binary_quantization else None
        
        if device is None:
            self.device = model.device if hasattr(model, 'device') else torch.device('cpu')
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
    
    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a query text into multi-vector embeddings.
        
        Args:
            query: Query text string
            
        Returns:
            Query embeddings tensor of shape (num_tokens, dim)
        """
        # Process query
        query_inputs = self.processor.process_queries(texts=[query])
        
        # Move to device
        query_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in query_inputs.items()
        }
        
        # Encode
        self.model.eval()
        with torch.no_grad():
            query_embeddings = self.model(**query_inputs)  # (1, seq_len, dim)
        
        # Get attention mask to filter out padding
        attention_mask = query_inputs["attention_mask"][0]  # (seq_len,)
        valid_mask = attention_mask.bool()
        
        # Return only valid tokens
        return query_embeddings[0][valid_mask]  # (num_valid_tokens, dim)
    
    def compute_maxsim(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MaxSim score between query and document embeddings.
        
        MaxSim computes:
        1. For each query token, find the maximum similarity with any document token
        2. Sum these maximum similarities across all query tokens
        
        Formula: score = Σᵢ maxⱼ(Qᵢ · Dⱼ)
        
        Args:
            query_embeddings: Query embeddings (num_query_tokens, dim)
            doc_embeddings: Document embeddings (num_doc_tokens, dim)
            doc_mask: Optional mask for valid document tokens (num_doc_tokens,)
            
        Returns:
            MaxSim score (scalar tensor)
            
        Requirements:
            3.1: Compute dot product between each query token and all document tokens
            3.2: Take maximum similarity for each query token
            3.3: Sum maximum similarities across all query tokens
        """
        # Compute dot products: (num_query_tokens, num_doc_tokens)
        similarities = torch.matmul(query_embeddings, doc_embeddings.T)
        
        # Apply document mask if provided (mask out padding tokens)
        if doc_mask is not None:
            # Set padding positions to -inf so they're never selected as max
            similarities = similarities.masked_fill(~doc_mask.unsqueeze(0), float('-inf'))
        
        # Take max over document tokens for each query token
        max_sims = similarities.max(dim=1)[0]  # (num_query_tokens,)
        
        # Sum over query tokens
        return max_sims.sum()
    
    def compute_binary_maxsim(
        self,
        query_binary: torch.Tensor,
        doc_binary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute approximate MaxSim using binary embeddings (Hamming similarity).
        
        Args:
            query_binary: Binary query embeddings (num_query_tokens, packed_dim)
            doc_binary: Binary document embeddings (num_doc_tokens, packed_dim)
            
        Returns:
            Approximate MaxSim score (scalar tensor)
        """
        if self.quantizer is None:
            raise ValueError("Binary quantization is not enabled")
        
        return self.quantizer.binary_maxsim(query_binary, doc_binary)
    
    def rerank(
        self,
        query: str,
        candidate_doc_ids: List[str],
        top_k: int = 10,
        binary_rescore_ratio: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidate documents using two-step strategy.
        
        Two-step Reranking Strategy:
        1. Load BINARY embeddings for all candidate_doc_ids
        2. Compute Hamming Distance (via BITWISE_XOR and POPCNT)
        3. Select top (top_k * binary_rescore_ratio) candidates
        4. Load FLOAT embeddings for these specific survivors
        5. Compute precise MaxSim (Dot Product) for final ranking
        
        Args:
            query: Query text
            candidate_doc_ids: Document IDs from first stage
            top_k: Number of results to return
            binary_rescore_ratio: Ratio for binary pre-filtering (default: 10)
                                  If 10, keeps top_k * 10 candidates after binary filtering
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
            
        Requirements:
            6.1: Compute MaxSim scores only for first-stage candidates
            6.3: Return documents sorted by MaxSim score in descending order
            6.4: Support batch processing to improve throughput
            6.5: When binary quantization is enabled, first filter using binary scores
                 then rescore top candidates with full-precision embeddings
        """
        if not candidate_doc_ids:
            return []
        
        # Limit top_k to available candidates
        top_k = min(top_k, len(candidate_doc_ids))
        
        # Encode query
        query_embeddings = self.encode_query(query)  # (num_query_tokens, dim)
        
        if self.use_binary_quantization and self.quantizer is not None:
            return self._rerank_with_binary(
                query_embeddings=query_embeddings,
                candidate_doc_ids=candidate_doc_ids,
                top_k=top_k,
                binary_rescore_ratio=binary_rescore_ratio,
            )
        else:
            return self._rerank_float_only(
                query_embeddings=query_embeddings,
                candidate_doc_ids=candidate_doc_ids,
                top_k=top_k,
            )
    
    def _rerank_with_binary(
        self,
        query_embeddings: torch.Tensor,
        candidate_doc_ids: List[str],
        top_k: int,
        binary_rescore_ratio: int,
    ) -> List[Tuple[str, float]]:
        """
        Rerank using two-step binary filtering + float rescoring.
        
        Args:
            query_embeddings: Query embeddings (num_query_tokens, dim)
            candidate_doc_ids: Document IDs to rerank
            top_k: Number of final results
            binary_rescore_ratio: Ratio for binary pre-filtering
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        # Step 1: Quantize query embeddings
        query_binary = self.quantizer.quantize(query_embeddings)  # (num_query_tokens, packed_dim)
        
        # Step 2: Load binary embeddings for all candidates
        binary_embeddings = self.embedding_store.get_embeddings(
            candidate_doc_ids, binary=True
        )
        
        # Filter out candidates without binary embeddings
        valid_candidates = [
            doc_id for doc_id in candidate_doc_ids 
            if doc_id in binary_embeddings
        ]
        
        if not valid_candidates:
            logger.warning("No candidates have binary embeddings, falling back to float-only")
            return self._rerank_float_only(
                query_embeddings=query_embeddings,
                candidate_doc_ids=candidate_doc_ids,
                top_k=top_k,
            )
        
        # Step 3: Compute binary MaxSim scores for all candidates
        binary_scores = []
        for doc_id in valid_candidates:
            doc_binary = binary_embeddings[doc_id].to(self.device)
            score = self.compute_binary_maxsim(query_binary, doc_binary)
            binary_scores.append((doc_id, score.item()))
        
        # Step 4: Sort by binary score and select top candidates for rescoring
        binary_scores.sort(key=lambda x: x[1], reverse=True)
        
        num_rescore = min(top_k * binary_rescore_ratio, len(binary_scores))
        candidates_to_rescore = [doc_id for doc_id, _ in binary_scores[:num_rescore]]
        
        # Step 5: Load float embeddings for survivors
        float_embeddings = self.embedding_store.get_embeddings(
            candidates_to_rescore, binary=False
        )
        
        # Step 6: Compute precise MaxSim scores
        float_scores = []
        for doc_id in candidates_to_rescore:
            if doc_id not in float_embeddings:
                logger.warning(f"Float embeddings not found for {doc_id}")
                continue
            
            doc_float = float_embeddings[doc_id].to(self.device)
            score = self.compute_maxsim(query_embeddings, doc_float)
            float_scores.append((doc_id, score.item()))
        
        # Step 7: Sort by float score and return top_k
        float_scores.sort(key=lambda x: x[1], reverse=True)
        
        return float_scores[:top_k]
    
    def _rerank_float_only(
        self,
        query_embeddings: torch.Tensor,
        candidate_doc_ids: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Rerank using float embeddings only (no binary pre-filtering).
        
        Args:
            query_embeddings: Query embeddings (num_query_tokens, dim)
            candidate_doc_ids: Document IDs to rerank
            top_k: Number of final results
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        # Load float embeddings for all candidates
        float_embeddings = self.embedding_store.get_embeddings(
            candidate_doc_ids, binary=False
        )
        
        # Compute MaxSim scores
        scores = []
        for doc_id in candidate_doc_ids:
            if doc_id not in float_embeddings:
                logger.warning(f"Float embeddings not found for {doc_id}")
                continue
            
            doc_float = float_embeddings[doc_id].to(self.device)
            score = self.compute_maxsim(query_embeddings, doc_float)
            scores.append((doc_id, score.item()))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def rerank_batch(
        self,
        queries: List[str],
        candidate_doc_ids_list: List[List[str]],
        top_k: int = 10,
        binary_rescore_ratio: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch rerank multiple queries.
        
        Args:
            queries: List of query texts
            candidate_doc_ids_list: List of candidate doc_id lists (one per query)
            top_k: Number of results per query
            binary_rescore_ratio: Ratio for binary pre-filtering
            
        Returns:
            List of result lists, each containing (doc_id, score) tuples
            
        Requirements:
            6.4: Support batch processing to improve throughput
        """
        if len(queries) != len(candidate_doc_ids_list):
            raise ValueError(
                f"Number of queries ({len(queries)}) must match "
                f"number of candidate lists ({len(candidate_doc_ids_list)})"
            )
        
        results = []
        for query, candidate_doc_ids in zip(queries, candidate_doc_ids_list):
            result = self.rerank(
                query=query,
                candidate_doc_ids=candidate_doc_ids,
                top_k=top_k,
                binary_rescore_ratio=binary_rescore_ratio,
            )
            results.append(result)
        
        return results
    
    def rerank_with_precomputed_query(
        self,
        query_embeddings: torch.Tensor,
        candidate_doc_ids: List[str],
        top_k: int = 10,
        binary_rescore_ratio: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Rerank using pre-computed query embeddings.
        
        Useful when the same query is used multiple times or when
        query encoding is done separately.
        
        Args:
            query_embeddings: Pre-computed query embeddings (num_tokens, dim)
            candidate_doc_ids: Document IDs to rerank
            top_k: Number of results to return
            binary_rescore_ratio: Ratio for binary pre-filtering
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if not candidate_doc_ids:
            return []
        
        top_k = min(top_k, len(candidate_doc_ids))
        
        if self.use_binary_quantization and self.quantizer is not None:
            return self._rerank_with_binary(
                query_embeddings=query_embeddings,
                candidate_doc_ids=candidate_doc_ids,
                top_k=top_k,
                binary_rescore_ratio=binary_rescore_ratio,
            )
        else:
            return self._rerank_float_only(
                query_embeddings=query_embeddings,
                candidate_doc_ids=candidate_doc_ids,
                top_k=top_k,
            )


# Type hints for forward references
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.colqwen3vl import ColQwen3VL
    from ..models.processing_colqwen3vl import ColQwen3VLProcessor
