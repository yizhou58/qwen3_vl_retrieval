"""
Binary Embedding First-Stage Retrieval.

Uses Qwen3-VL binary embeddings for fast first-stage retrieval,
replacing BM25 text-based retrieval with visual embedding retrieval.

This enables end-to-end visual retrieval where both stages benefit from
fine-tuning the visual model.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .binary_quantizer import BinaryQuantizer
from .embedding_store import EmbeddingStore
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class BinaryFirstStageRetriever:
    """
    First-stage retrieval using binary embeddings from Qwen3-VL.
    
    Uses Hamming distance for fast approximate matching between
    query and document binary embeddings.
    
    For multi-vector embeddings (ColPali-style), computes a simplified
    score based on average Hamming similarity across tokens.
    
    Attributes:
        embedding_store: Store for document embeddings
        quantizer: Binary quantizer for Hamming distance computation
        doc_ids: List of indexed document IDs
    """
    
    def __init__(
        self,
        embedding_store: Optional[EmbeddingStore] = None,
        index_path: Optional[str] = None,
    ):
        """
        Initialize binary first-stage retriever.
        
        Args:
            embedding_store: EmbeddingStore with pre-computed embeddings
            index_path: Path to save/load index metadata
        """
        self.embedding_store = embedding_store
        self.index_path = index_path
        self.quantizer = BinaryQuantizer()
        
        # Document metadata
        self.doc_ids: List[str] = []
        self.image_paths: Dict[str, str] = {}
        
        # Cached binary embeddings for fast retrieval
        self._binary_cache: Dict[str, torch.Tensor] = {}
        self._pooled_binary_cache: Dict[str, torch.Tensor] = {}  # Single vector per doc
        
        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
    
    def set_embedding_store(self, embedding_store: EmbeddingStore) -> None:
        """Set the embedding store and refresh cache."""
        self.embedding_store = embedding_store
        self._refresh_cache()
    
    def _refresh_cache(self) -> None:
        """Refresh binary embedding cache from embedding store."""
        if self.embedding_store is None:
            return
        
        self._binary_cache.clear()
        self._pooled_binary_cache.clear()
        
        # Get all doc_ids from embedding store
        all_doc_ids = self.embedding_store.list_doc_ids()
        
        for doc_id in all_doc_ids:
            if doc_id not in self.doc_ids:
                self.doc_ids.append(doc_id)
            
            # Get binary embeddings
            binary_emb = self.embedding_store.get_embeddings([doc_id], binary=True)
            if doc_id in binary_emb:
                self._binary_cache[doc_id] = binary_emb[doc_id]
                # Create pooled representation (mean of binary bits)
                self._pooled_binary_cache[doc_id] = self._pool_binary(binary_emb[doc_id])
        
        logger.info(f"Refreshed binary cache with {len(self._binary_cache)} documents")
    
    def _pool_binary(self, binary_emb: torch.Tensor) -> torch.Tensor:
        """
        Pool multi-vector binary embeddings into a single vector.
        
        Uses majority voting: for each bit position, take the majority value
        across all tokens.
        
        Args:
            binary_emb: Binary embeddings (num_tokens, packed_dim) as uint8
            
        Returns:
            Pooled binary embedding (packed_dim,) as uint8
        """
        if binary_emb.dim() == 1:
            return binary_emb
        
        num_tokens, packed_dim = binary_emb.shape
        
        # Unpack to bits for majority voting
        unpacked = torch.zeros(num_tokens, packed_dim * 8, dtype=torch.float32)
        for byte_idx in range(packed_dim):
            for bit_idx in range(8):
                bit_pos = byte_idx * 8 + bit_idx
                unpacked[:, bit_pos] = ((binary_emb[:, byte_idx] >> (7 - bit_idx)) & 1).float()
        
        # Majority voting
        majority = (unpacked.mean(dim=0) >= 0.5).to(torch.uint8)
        
        # Repack to bytes
        pooled = torch.zeros(packed_dim, dtype=torch.uint8)
        for byte_idx in range(packed_dim):
            byte_val = 0
            for bit_idx in range(8):
                bit_pos = byte_idx * 8 + bit_idx
                byte_val |= (majority[bit_pos].item() << (7 - bit_idx))
            pooled[byte_idx] = byte_val
        
        return pooled
    
    def index_documents(
        self,
        doc_ids: List[str],
        image_paths: List[str],
    ) -> None:
        """
        Index documents (metadata only, embeddings should be in embedding_store).
        
        Args:
            doc_ids: List of document IDs
            image_paths: List of image paths
        """
        for doc_id, image_path in zip(doc_ids, image_paths):
            if doc_id not in self.doc_ids:
                self.doc_ids.append(doc_id)
            self.image_paths[doc_id] = image_path
        
        # Refresh cache to include new documents
        self._refresh_cache()
        
        logger.info(f"Indexed {len(doc_ids)} documents. Total: {len(self.doc_ids)}")
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 100,
        use_pooled: bool = True,
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve top-k candidates using binary Hamming distance.
        
        Args:
            query_embedding: Query embedding (num_tokens, dim) as float or
                           (num_tokens, packed_dim) as uint8 binary
            top_k: Number of candidates to return
            use_pooled: If True, use pooled single-vector matching (faster)
                       If False, use full multi-vector matching (more accurate)
            
        Returns:
            List of (doc_id, score, image_path) tuples sorted by score descending
        """
        if not self.doc_ids:
            logger.warning("Retrieval attempted on empty index")
            return []
        
        top_k = min(top_k, len(self.doc_ids))
        
        # Quantize query if needed
        if query_embedding.dtype != torch.uint8:
            query_binary = self.quantizer.quantize(query_embedding)
        else:
            query_binary = query_embedding
        
        if use_pooled:
            return self._retrieve_pooled(query_binary, top_k)
        else:
            return self._retrieve_multivector(query_binary, top_k)
    
    def _retrieve_pooled(
        self,
        query_binary: torch.Tensor,
        top_k: int,
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve using pooled single-vector Hamming distance.
        
        Args:
            query_binary: Binary query (num_tokens, packed_dim) as uint8
            top_k: Number of candidates
            
        Returns:
            List of (doc_id, score, image_path) tuples
        """
        # Pool query to single vector
        query_pooled = self._pool_binary(query_binary)
        
        # Compute Hamming similarity with all documents
        scores = []
        for doc_id in self.doc_ids:
            if doc_id not in self._pooled_binary_cache:
                scores.append((doc_id, 0.0))
                continue
            
            doc_pooled = self._pooled_binary_cache[doc_id]
            
            # Hamming similarity = dim - hamming_distance
            xor_result = query_pooled ^ doc_pooled
            hamming_dist = sum(bin(b.item()).count('1') for b in xor_result)
            dim = query_pooled.shape[0] * 8
            similarity = dim - hamming_dist
            
            scores.append((doc_id, float(similarity)))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results with image paths
        results = []
        for doc_id, score in scores[:top_k]:
            image_path = self.image_paths.get(doc_id, "")
            results.append((doc_id, score, image_path))
        
        return results
    
    def _retrieve_multivector(
        self,
        query_binary: torch.Tensor,
        top_k: int,
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve using full multi-vector binary MaxSim (Hamming similarity).
        
        For each query token, finds max Hamming similarity with any doc token,
        then sums across query tokens.
        
        Args:
            query_binary: Binary query (num_tokens, packed_dim) as uint8
            top_k: Number of candidates
            
        Returns:
            List of (doc_id, score, image_path) tuples
        """
        scores = []
        
        # Move query to same device as cache
        query_binary = query_binary.cpu()
        n_query = query_binary.shape[0]
        packed_dim = query_binary.shape[1]
        dim = packed_dim * 8
        
        # Create popcount table for fast bit counting
        popcount_table = torch.zeros(256, dtype=torch.int32)
        for i in range(256):
            popcount_table[i] = bin(i).count('1')
        
        for doc_id in self.doc_ids:
            if doc_id not in self._binary_cache:
                scores.append((doc_id, 0.0))
                continue
            
            doc_binary = self._binary_cache[doc_id].cpu()
            n_doc = doc_binary.shape[0]
            
            # Compute pairwise Hamming distances using XOR + popcount
            # query_binary: (n_query, packed_dim)
            # doc_binary: (n_doc, packed_dim)
            
            # Expand for broadcasting
            query_exp = query_binary.unsqueeze(1)  # (n_query, 1, packed_dim)
            doc_exp = doc_binary.unsqueeze(0)      # (1, n_doc, packed_dim)
            
            # XOR to find differing bits
            xor_result = query_exp ^ doc_exp  # (n_query, n_doc, packed_dim)
            
            # Count bits using lookup table
            xor_flat = xor_result.reshape(-1).long()
            bit_counts = popcount_table[xor_flat].reshape(n_query, n_doc, packed_dim)
            
            # Sum across packed dimension to get Hamming distance
            hamming_dist = bit_counts.sum(dim=-1)  # (n_query, n_doc)
            
            # Convert to similarity: sim = dim - dist
            hamming_sim = dim - hamming_dist  # (n_query, n_doc)
            
            # MaxSim: for each query token, take max over doc tokens, then sum
            max_sims = hamming_sim.max(dim=1)[0]  # (n_query,)
            score = max_sims.sum().item()
            
            scores.append((doc_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build results with image paths
        results = []
        for doc_id, score in scores[:top_k]:
            image_path = self.image_paths.get(doc_id, "")
            results.append((doc_id, score, image_path))
        
        return results
    
    def retrieve_with_query_text(
        self,
        query_text: str,
        model,
        processor,
        top_k: int = 100,
        use_pooled: bool = True,
        device: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve using query text (encodes query on-the-fly).
        
        Args:
            query_text: Query text
            model: ColQwen3VL model for encoding
            processor: ColQwen3VLProcessor for preprocessing
            top_k: Number of candidates
            use_pooled: Use pooled matching
            device: Device for encoding
            
        Returns:
            List of (doc_id, score, image_path) tuples
        """
        # Encode query
        if device is None:
            device = next(model.parameters()).device
        
        with torch.no_grad():
            query_inputs = processor.process_queries([query_text])
            query_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in query_inputs.items()
            }
            query_embedding = model(**query_inputs)[0]  # (num_tokens, dim)
        
        return self.retrieve(query_embedding, top_k, use_pooled)
    
    def save_index(self, path: Optional[str] = None) -> None:
        """Save index metadata to disk."""
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No save path provided")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "doc_ids": self.doc_ids,
            "image_paths": self.image_paths,
        }
        
        with open(save_path / "binary_index_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Binary index saved to {save_path}")
    
    def load_index(self, path: Optional[str] = None) -> None:
        """Load index metadata from disk."""
        load_path = path or self.index_path
        if not load_path:
            raise ValueError("No load path provided")
        
        load_path = Path(load_path)
        metadata_file = load_path / "binary_index_metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"No binary index metadata found at {load_path}")
            return
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        self.doc_ids = metadata["doc_ids"]
        self.image_paths = metadata["image_paths"]
        
        logger.info(f"Binary index loaded from {load_path}. {len(self.doc_ids)} documents.")
    
    def get_document_count(self) -> int:
        """Return number of indexed documents."""
        return len(self.doc_ids)
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        self.doc_ids = []
        self.image_paths = {}
        self._binary_cache = {}
        self._pooled_binary_cache = {}
        logger.info("Binary index cleared")
    
    def __len__(self) -> int:
        return len(self.doc_ids)
    
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.doc_ids
