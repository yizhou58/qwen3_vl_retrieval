"""
Binary Quantization for efficient storage and retrieval.

This module implements binary quantization for ColPali-style multi-vector embeddings,
reducing storage by 32x (float32 to 1-bit) while enabling fast Hamming distance computation.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np


class BinaryQuantizer:
    """
    Binary quantization for efficient storage and retrieval.
    
    Converts float32 embeddings to binary vectors using sign function.
    Supports Hamming distance for fast approximate matching.
    
    The quantization process:
    1. Apply sign function: bit[i] = 1 if embedding[i] >= 0 else 0
    2. Pack bits into uint8 for storage efficiency (8 bits per byte)
    
    Storage reduction: 32x (float32 = 32 bits -> 1 bit per dimension)
    
    Attributes:
        None (stateless class)
    
    Example:
        >>> quantizer = BinaryQuantizer()
        >>> embeddings = torch.randn(100, 128)  # 100 tokens, 128 dim
        >>> binary = quantizer.quantize(embeddings)  # (100, 16) uint8
        >>> # Storage: 100 * 128 * 4 bytes = 51200 bytes -> 100 * 16 bytes = 1600 bytes
    """
    
    def __init__(self):
        """Initialize BinaryQuantizer."""
        pass
    
    def quantize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Quantize float embeddings to binary.
        
        Uses sign function: bit[i] = 1 if embedding[i] >= 0 else 0
        Packs bits into uint8 for storage efficiency.
        
        Args:
            embeddings: Float embeddings of shape (..., dim) where dim must be divisible by 8
            
        Returns:
            Binary embeddings packed as uint8 of shape (..., dim // 8)
            
        Raises:
            ValueError: If embedding dimension is not divisible by 8
            
        Example:
            >>> embeddings = torch.tensor([[0.5, -0.3, 0.1, -0.8, 0.2, -0.1, 0.9, -0.5]])
            >>> binary = quantizer.quantize(embeddings)
            >>> # Bits: [1, 0, 1, 0, 1, 0, 1, 0] -> packed as uint8
        """
        if embeddings.shape[-1] % 8 != 0:
            raise ValueError(
                f"Embedding dimension must be divisible by 8, got {embeddings.shape[-1]}"
            )
        
        # Get original shape
        original_shape = embeddings.shape
        dim = original_shape[-1]
        
        # Flatten to 2D for processing: (batch, dim)
        flat_embeddings = embeddings.reshape(-1, dim)
        batch_size = flat_embeddings.shape[0]
        
        # Apply sign function: >= 0 -> 1, < 0 -> 0
        binary_bits = (flat_embeddings >= 0).to(torch.uint8)  # (batch, dim)
        
        # Reshape for packing: (batch, dim // 8, 8)
        binary_bits = binary_bits.reshape(batch_size, dim // 8, 8)
        
        # Pack 8 bits into 1 byte using bit shifts
        # Bit order: bit 0 is MSB (most significant bit)
        powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], 
                              dtype=torch.uint8, device=embeddings.device)
        packed = (binary_bits * powers).sum(dim=-1).to(torch.uint8)  # (batch, dim // 8)
        
        # Reshape back to original shape (except last dim is dim // 8)
        output_shape = original_shape[:-1] + (dim // 8,)
        return packed.reshape(output_shape)
    
    def unpack_binary(self, packed: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Unpack binary embeddings back to bit representation.
        
        Args:
            packed: Packed binary embeddings of shape (..., dim // 8) as uint8
            dim: Original embedding dimension
            
        Returns:
            Unpacked bits of shape (..., dim) as uint8 (0 or 1)
        """
        original_shape = packed.shape
        packed_dim = original_shape[-1]
        
        # Flatten to 2D
        flat_packed = packed.reshape(-1, packed_dim)
        batch_size = flat_packed.shape[0]
        
        # Unpack each byte to 8 bits
        # Use bit shifts and masking
        unpacked = torch.zeros(batch_size, dim, dtype=torch.uint8, device=packed.device)
        
        for i in range(8):
            bit_mask = 1 << (7 - i)  # MSB first
            unpacked[:, i::8] = ((flat_packed & bit_mask) >> (7 - i)).to(torch.uint8)
        
        # Reorder to correct positions
        unpacked_reordered = torch.zeros_like(unpacked)
        for byte_idx in range(packed_dim):
            for bit_idx in range(8):
                unpacked_reordered[:, byte_idx * 8 + bit_idx] = \
                    ((flat_packed[:, byte_idx] >> (7 - bit_idx)) & 1).to(torch.uint8)
        
        # Reshape back
        output_shape = original_shape[:-1] + (dim,)
        return unpacked_reordered.reshape(output_shape)
    
    def hamming_distance(
        self,
        query_binary: torch.Tensor,
        doc_binary: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hamming distance between binary vectors.
        
        Hamming distance = number of differing bits between two binary vectors.
        Uses XOR and popcount for efficient computation.
        
        For multi-vector embeddings (ColPali-style), computes pairwise distances
        between all query tokens and all document tokens.
        
        Args:
            query_binary: Binary query embeddings (n_query_tokens, packed_dim) uint8
            doc_binary: Binary document embeddings (n_doc_tokens, packed_dim) uint8
            
        Returns:
            Hamming distances (n_query_tokens, n_doc_tokens)
            
        Example:
            >>> q_binary = quantizer.quantize(query_embeddings)  # (10, 16)
            >>> d_binary = quantizer.quantize(doc_embeddings)    # (50, 16)
            >>> distances = quantizer.hamming_distance(q_binary, d_binary)  # (10, 50)
        """
        n_query = query_binary.shape[0]
        n_doc = doc_binary.shape[0]
        packed_dim = query_binary.shape[-1]
        
        # Expand for broadcasting: (n_query, 1, packed_dim) XOR (1, n_doc, packed_dim)
        query_expanded = query_binary.unsqueeze(1)  # (n_query, 1, packed_dim)
        doc_expanded = doc_binary.unsqueeze(0)      # (1, n_doc, packed_dim)
        
        # XOR to find differing bits (packed)
        xor_result = query_expanded ^ doc_expanded  # (n_query, n_doc, packed_dim)
        
        # Count bits using lookup table (popcount)
        # Create popcount lookup table for uint8
        popcount_table = self._create_popcount_table(query_binary.device)
        
        # Apply popcount to each byte and sum
        xor_flat = xor_result.reshape(-1)  # Flatten for indexing
        bit_counts = popcount_table[xor_flat.long()]  # Lookup popcount
        bit_counts = bit_counts.reshape(n_query, n_doc, packed_dim)
        
        # Sum across packed dimension to get total Hamming distance
        distances = bit_counts.sum(dim=-1)  # (n_query, n_doc)
        
        return distances
    
    def _create_popcount_table(self, device: torch.device) -> torch.Tensor:
        """Create lookup table for popcount (number of 1 bits in a byte)."""
        table = torch.zeros(256, dtype=torch.int32, device=device)
        for i in range(256):
            table[i] = bin(i).count('1')
        return table
    
    def hamming_similarity(
        self,
        query_binary: torch.Tensor,
        doc_binary: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hamming similarity (inverse of distance) between binary vectors.
        
        Similarity = dim - hamming_distance, where dim is the original embedding dimension.
        Higher similarity means more similar vectors.
        
        Args:
            query_binary: Binary query embeddings (n_query_tokens, packed_dim) uint8
            doc_binary: Binary document embeddings (n_doc_tokens, packed_dim) uint8
            
        Returns:
            Hamming similarities (n_query_tokens, n_doc_tokens)
        """
        dim = query_binary.shape[-1] * 8  # Original dimension
        distances = self.hamming_distance(query_binary, doc_binary)
        return dim - distances
    
    def binary_maxsim(
        self,
        query_binary: torch.Tensor,
        doc_binary: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MaxSim score using binary embeddings (Hamming similarity).
        
        For each query token, finds the maximum Hamming similarity with any document token,
        then sums across all query tokens.
        
        Args:
            query_binary: Binary query embeddings (n_query_tokens, packed_dim) uint8
            doc_binary: Binary document embeddings (n_doc_tokens, packed_dim) uint8
            
        Returns:
            MaxSim score (scalar tensor)
        """
        similarities = self.hamming_similarity(query_binary, doc_binary)
        max_sims = similarities.max(dim=1)[0]  # Max over doc tokens for each query token
        return max_sims.sum()
    
    def rescore(
        self,
        query_float: torch.Tensor,
        doc_float: torch.Tensor,
        candidate_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Rescore candidates using full-precision embeddings.
        
        After binary filtering, use original float embeddings for precise MaxSim scoring.
        
        Args:
            query_float: Float query embeddings (n_query_tokens, dim)
            doc_float: Float document embeddings (n_doc_tokens, dim) or 
                       dict mapping doc_id to embeddings
            candidate_indices: Optional indices of candidate documents to rescore.
                              If None, scores all documents.
            
        Returns:
            MaxSim scores for candidates
            
        Example:
            >>> # After binary filtering, rescore top candidates
            >>> top_candidates = binary_scores.topk(k=100).indices
            >>> precise_scores = quantizer.rescore(query_float, doc_float, top_candidates)
        """
        # Compute dot products between query and document tokens
        # query_float: (n_query, dim), doc_float: (n_doc, dim)
        similarities = torch.matmul(query_float, doc_float.T)  # (n_query, n_doc)
        
        # MaxSim: for each query token, take max over doc tokens, then sum
        max_sims = similarities.max(dim=1)[0]  # (n_query,)
        score = max_sims.sum()
        
        return score
    
    def batch_rescore(
        self,
        query_float: torch.Tensor,
        doc_embeddings_list: list,
        candidate_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Batch rescore multiple documents using full-precision embeddings.
        
        Args:
            query_float: Float query embeddings (n_query_tokens, dim)
            doc_embeddings_list: List of document embeddings, each (n_doc_tokens, dim)
            candidate_indices: Optional indices to select which documents to score
            
        Returns:
            MaxSim scores for each document (n_docs,)
        """
        if candidate_indices is not None:
            doc_embeddings_list = [doc_embeddings_list[i] for i in candidate_indices]
        
        scores = []
        for doc_float in doc_embeddings_list:
            score = self.rescore(query_float, doc_float)
            scores.append(score)
        
        return torch.stack(scores)
    
    def get_storage_reduction_factor(self) -> int:
        """
        Get the storage reduction factor.
        
        Returns:
            32 (float32 = 32 bits reduced to 1 bit per dimension)
        """
        return 32
    
    def estimate_storage_bytes(
        self,
        num_tokens: int,
        dim: int,
        quantized: bool = True
    ) -> int:
        """
        Estimate storage requirements in bytes.
        
        Args:
            num_tokens: Number of embedding tokens
            dim: Embedding dimension
            quantized: If True, estimate for binary; if False, for float32
            
        Returns:
            Estimated storage in bytes
        """
        if quantized:
            return num_tokens * (dim // 8)  # 1 bit per dimension, packed into bytes
        else:
            return num_tokens * dim * 4  # 4 bytes per float32
