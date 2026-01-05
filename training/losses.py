"""
ColBERT-style Loss Functions for Contrastive Learning.

Implements InfoNCE loss with temperature scaling for training retrieval models.

Requirements: 2.4
- Use ColBERT-style InfoNCE loss with temperature scaling for contrastive learning
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColbertLoss(nn.Module):
    """
    ColBERT-style InfoNCE Loss for contrastive learning.
    
    Computes InfoNCE loss using MaxSim scores between query and document
    embeddings. Uses in-batch negatives where each query's positive document
    serves as negative for other queries.
    
    Loss = -log(exp(sim(q, d+) / τ) / Σ exp(sim(q, d) / τ))
    
    where:
    - sim(q, d) is the MaxSim score
    - d+ is the positive document
    - τ is the temperature
    
    Requirements:
        2.4: Use ColBERT-style InfoNCE loss with temperature scaling
    """
    
    def __init__(
        self,
        temperature: float = 0.02,
        normalize_scores: bool = False,
    ):
        """
        Initialize ColbertLoss.
        
        Args:
            temperature: Temperature scaling factor (default: 0.02)
            normalize_scores: Whether to normalize scores by query length
        """
        super().__init__()
        self.temperature = temperature
        self.normalize_scores = normalize_scores
    
    def forward(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        doc_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        query_mask: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Assumes diagonal entries are positive pairs (query[i] matches doc[i]).
        
        Args:
            query_embeddings: Query embeddings (batch_size, seq_len, dim) or list
            doc_embeddings: Document embeddings (batch_size, seq_len, dim) or list
            query_mask: Optional mask for query tokens (batch_size, seq_len)
            doc_mask: Optional mask for document tokens (batch_size, seq_len)
            
        Returns:
            Scalar loss value
        """
        # Compute MaxSim scores matrix
        scores = self._compute_maxsim_matrix(
            query_embeddings, doc_embeddings, query_mask, doc_mask
        )
        
        # Apply temperature scaling
        scores = scores / self.temperature
        
        # Labels: diagonal entries are positive pairs
        batch_size = scores.shape[0]
        labels = torch.arange(batch_size, device=scores.device)
        
        # Cross-entropy loss (InfoNCE)
        loss = F.cross_entropy(scores, labels)
        
        return loss
    
    def _compute_maxsim_matrix(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        doc_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        query_mask: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MaxSim score matrix between all query-document pairs.
        
        MaxSim: score = Σᵢ maxⱼ(Qᵢ · Dⱼ)
        
        Args:
            query_embeddings: Query embeddings
            doc_embeddings: Document embeddings
            query_mask: Optional query token mask
            doc_mask: Optional document token mask
            
        Returns:
            Score matrix (num_queries, num_docs)
        """
        # Handle list inputs
        if isinstance(query_embeddings, list):
            return self._compute_maxsim_matrix_list(
                query_embeddings, doc_embeddings, query_mask, doc_mask
            )
        
        # Tensor inputs: (batch_size, seq_len, dim)
        batch_q, seq_q, dim = query_embeddings.shape
        batch_d, seq_d, _ = doc_embeddings.shape
        
        # Compute all pairwise similarities
        # (batch_q, seq_q, dim) @ (batch_d, dim, seq_d) -> need einsum
        # Result: (batch_q, batch_d, seq_q, seq_d)
        similarities = torch.einsum(
            "qnd,pmd->qpnm",
            query_embeddings,
            doc_embeddings,
        )
        
        # Apply document mask if provided
        if doc_mask is not None:
            # doc_mask: (batch_d, seq_d) -> (1, batch_d, 1, seq_d)
            doc_mask_expanded = doc_mask.unsqueeze(0).unsqueeze(2)
            similarities = similarities.masked_fill(~doc_mask_expanded, float('-inf'))
        
        # Max over document tokens: (batch_q, batch_d, seq_q)
        max_sims = similarities.max(dim=-1)[0]
        
        # Apply query mask if provided
        if query_mask is not None:
            # query_mask: (batch_q, seq_q) -> (batch_q, 1, seq_q)
            query_mask_expanded = query_mask.unsqueeze(1)
            max_sims = max_sims * query_mask_expanded.float()
        
        # Sum over query tokens: (batch_q, batch_d)
        scores = max_sims.sum(dim=-1)
        
        # Normalize by query length if configured
        if self.normalize_scores and query_mask is not None:
            query_lengths = query_mask.sum(dim=-1, keepdim=True).float()
            scores = scores / query_lengths.clamp(min=1)
        
        return scores
    
    def _compute_maxsim_matrix_list(
        self,
        query_embeddings: List[torch.Tensor],
        doc_embeddings: List[torch.Tensor],
        query_mask: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MaxSim matrix for list inputs (variable length sequences).
        
        Args:
            query_embeddings: List of query embeddings (seq_len_i, dim)
            doc_embeddings: List of document embeddings (seq_len_j, dim)
            
        Returns:
            Score matrix (num_queries, num_docs)
        """
        num_queries = len(query_embeddings)
        num_docs = len(doc_embeddings)
        device = query_embeddings[0].device
        
        scores = torch.zeros(num_queries, num_docs, device=device)
        
        for i, q_emb in enumerate(query_embeddings):
            for j, d_emb in enumerate(doc_embeddings):
                # q_emb: (seq_q, dim), d_emb: (seq_d, dim)
                # Compute similarity: (seq_q, seq_d)
                sim = torch.matmul(q_emb, d_emb.t())
                
                # Max over document tokens, sum over query tokens
                max_sim = sim.max(dim=-1)[0]  # (seq_q,)
                scores[i, j] = max_sim.sum()
        
        return scores


class BiEncoderLoss(nn.Module):
    """
    Bi-encoder style contrastive loss.
    
    Uses single-vector representations instead of multi-vector MaxSim.
    Useful for comparison or hybrid approaches.
    """
    
    def __init__(
        self,
        temperature: float = 0.02,
        similarity: str = "dot",
    ):
        """
        Initialize BiEncoderLoss.
        
        Args:
            temperature: Temperature scaling factor
            similarity: Similarity function ("dot" or "cosine")
        """
        super().__init__()
        self.temperature = temperature
        self.similarity = similarity
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bi-encoder contrastive loss.
        
        Args:
            query_embeddings: Query embeddings (batch_size, dim)
            doc_embeddings: Document embeddings (batch_size, dim)
            
        Returns:
            Scalar loss value
        """
        if self.similarity == "cosine":
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        scores = torch.matmul(query_embeddings, doc_embeddings.t())
        scores = scores / self.temperature
        
        # Labels: diagonal entries are positive pairs
        batch_size = scores.shape[0]
        labels = torch.arange(batch_size, device=scores.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(scores, labels)
        
        return loss


class HardNegativeLoss(nn.Module):
    """
    Loss with explicit hard negatives.
    
    Extends ColbertLoss to handle pre-mined hard negatives in addition
    to in-batch negatives.
    """
    
    def __init__(
        self,
        temperature: float = 0.02,
        hard_negative_weight: float = 1.0,
    ):
        """
        Initialize HardNegativeLoss.
        
        Args:
            temperature: Temperature scaling factor
            hard_negative_weight: Weight for hard negative scores
        """
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.base_loss = ColbertLoss(temperature=temperature)
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        hard_negative_embeddings: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,
        hard_negative_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss with hard negatives.
        
        Args:
            query_embeddings: Query embeddings (batch_size, seq_len, dim)
            positive_embeddings: Positive doc embeddings (batch_size, seq_len, dim)
            hard_negative_embeddings: Hard negative embeddings 
                (batch_size, num_negatives, seq_len, dim)
            query_mask: Query token mask
            positive_mask: Positive doc token mask
            hard_negative_mask: Hard negative token mask
            
        Returns:
            Scalar loss value
        """
        if hard_negative_embeddings is None:
            # Fall back to standard in-batch loss
            return self.base_loss(
                query_embeddings, positive_embeddings, query_mask, positive_mask
            )
        
        batch_size = query_embeddings.shape[0]
        device = query_embeddings.device
        
        # Compute positive scores
        positive_scores = self._compute_maxsim_diagonal(
            query_embeddings, positive_embeddings, query_mask, positive_mask
        )  # (batch_size,)
        
        # Compute in-batch negative scores
        in_batch_scores = self.base_loss._compute_maxsim_matrix(
            query_embeddings, positive_embeddings, query_mask, positive_mask
        )  # (batch_size, batch_size)
        
        # Compute hard negative scores
        # hard_negative_embeddings: (batch_size, num_negatives, seq_len, dim)
        num_negatives = hard_negative_embeddings.shape[1]
        hard_neg_scores = []
        
        for i in range(batch_size):
            q_emb = query_embeddings[i:i+1]  # (1, seq_q, dim)
            q_mask = query_mask[i:i+1] if query_mask is not None else None
            
            for j in range(num_negatives):
                hn_emb = hard_negative_embeddings[i, j:j+1]  # (1, seq_d, dim)
                hn_mask = hard_negative_mask[i, j:j+1] if hard_negative_mask is not None else None
                
                score = self.base_loss._compute_maxsim_matrix(
                    q_emb, hn_emb, q_mask, hn_mask
                )
                hard_neg_scores.append(score[0, 0])
        
        hard_neg_scores = torch.stack(hard_neg_scores).view(batch_size, num_negatives)
        hard_neg_scores = hard_neg_scores * self.hard_negative_weight
        
        # Combine all scores
        # For each query: [positive, in-batch negatives, hard negatives]
        all_scores = torch.cat([
            positive_scores.unsqueeze(1),  # (batch_size, 1)
            in_batch_scores,  # (batch_size, batch_size)
            hard_neg_scores,  # (batch_size, num_negatives)
        ], dim=1)
        
        # Apply temperature
        all_scores = all_scores / self.temperature
        
        # Labels: first column is positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(all_scores, labels)
        
        return loss
    
    def _compute_maxsim_diagonal(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        doc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MaxSim scores for diagonal pairs only (query[i] with doc[i]).
        
        More efficient than computing full matrix when only diagonal is needed.
        
        Returns:
            Diagonal scores (batch_size,)
        """
        batch_size = query_embeddings.shape[0]
        scores = []
        
        for i in range(batch_size):
            q_emb = query_embeddings[i]  # (seq_q, dim)
            d_emb = doc_embeddings[i]  # (seq_d, dim)
            
            # Compute similarity: (seq_q, seq_d)
            sim = torch.matmul(q_emb, d_emb.t())
            
            # Apply document mask
            if doc_mask is not None:
                sim = sim.masked_fill(~doc_mask[i].unsqueeze(0), float('-inf'))
            
            # Max over document tokens
            max_sim = sim.max(dim=-1)[0]  # (seq_q,)
            
            # Apply query mask
            if query_mask is not None:
                max_sim = max_sim * query_mask[i].float()
            
            scores.append(max_sim.sum())
        
        return torch.stack(scores)
