"""
Visual Retriever Collator Implementation.

Collator for batching training samples with dynamic resolution images.

Requirements: 7.2, 7.3
- Utilize Qwen3-VL's native image processor for variable resolutions
- Tokenize queries with proper padding and truncation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from qwen3_vl_retrieval.data.dataset import TrainingSample


@dataclass
class VisualRetrieverCollator:
    """
    Collator for batching visual retrieval training samples.
    
    Handles:
    - Dynamic resolution image processing
    - Query tokenization with padding
    - Creating attention masks for variable-length sequences
    
    Requirements:
        7.2: Utilize Qwen3-VL's native image processor for variable resolutions
        7.3: Tokenize queries with proper padding and truncation
    """
    
    processor: Any  # ColQwen3VLProcessor
    max_query_length: int = 128
    query_augmentation_token: str = "<|endoftext|>"
    num_query_augmentation_tokens: int = 10
    
    def __call__(
        self,
        batch: List[Union[TrainingSample, Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of training samples.
        
        Args:
            batch: List of TrainingSample or dict with query, positive_image, etc.
            
        Returns:
            Dictionary containing:
            - query_input_ids: (batch_size, query_len)
            - query_attention_mask: (batch_size, query_len)
            - doc_input_ids: (batch_size, doc_len)
            - doc_attention_mask: (batch_size, doc_len)
            - doc_pixel_values: (batch_size, max_patches, patch_dim)
            - doc_image_grid_thw: (batch_size, 3)
        """
        # Extract queries and images from batch
        queries = []
        images = []
        
        for sample in batch:
            if isinstance(sample, TrainingSample):
                queries.append(sample.query)
                images.append(sample.positive_image)
            elif isinstance(sample, dict):
                queries.append(sample["query"])
                if isinstance(sample["positive_image"], Image.Image):
                    images.append(sample["positive_image"])
                else:
                    # Load image from path
                    images.append(Image.open(sample["positive_image"]).convert("RGB"))
            else:
                raise ValueError(f"Unsupported sample type: {type(sample)}")
        
        # Process queries
        query_batch = self._process_queries(queries)
        
        # Process images
        doc_batch = self._process_images(images)
        
        return {
            "query_input_ids": query_batch["input_ids"],
            "query_attention_mask": query_batch["attention_mask"],
            "doc_input_ids": doc_batch["input_ids"],
            "doc_attention_mask": doc_batch["attention_mask"],
            "doc_pixel_values": doc_batch["pixel_values"],
            "doc_image_grid_thw": doc_batch["image_grid_thw"],
        }
    
    def _process_queries(self, queries: List[str]) -> Dict[str, torch.Tensor]:
        """
        Process query texts with augmentation and padding.
        
        Args:
            queries: List of query strings
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Add augmentation tokens
        suffix = self.query_augmentation_token * self.num_query_augmentation_tokens
        augmented_queries = [q + suffix for q in queries]
        
        # Tokenize with padding
        return self.processor.process_texts(augmented_queries)
    
    def _process_images(self, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Process document images with dynamic resolution.
        
        Args:
            images: List of PIL images
            
        Returns:
            Dictionary with input_ids, attention_mask, pixel_values, image_grid_thw
        """
        return self.processor.process_images(images)


@dataclass
class VisualRetrieverCollatorWithHardNegatives(VisualRetrieverCollator):
    """
    Collator that also handles hard negative samples.
    
    Extends VisualRetrieverCollator to process hard negative documents
    in addition to positive documents.
    """
    
    embedding_store: Any = None  # EmbeddingStore for loading pre-computed embeddings
    num_hard_negatives: int = 0
    
    def __call__(
        self,
        batch: List[Union[TrainingSample, Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch with hard negatives.
        
        Args:
            batch: List of samples with optional hard_negatives field
            
        Returns:
            Dictionary containing base collator outputs plus:
            - hard_negative_embeddings: (batch_size, num_negatives, seq_len, dim)
              if embedding_store is provided
        """
        # Get base collation
        result = super().__call__(batch)
        
        # Process hard negatives if available
        if self.num_hard_negatives > 0 and self.embedding_store is not None:
            hard_neg_embeddings = self._process_hard_negatives(batch)
            if hard_neg_embeddings is not None:
                result["hard_negative_embeddings"] = hard_neg_embeddings
        
        return result
    
    def _process_hard_negatives(
        self,
        batch: List[Union[TrainingSample, Dict[str, Any]]],
    ) -> Optional[torch.Tensor]:
        """
        Load pre-computed embeddings for hard negatives.
        
        Args:
            batch: List of samples with hard_negatives field
            
        Returns:
            Tensor of hard negative embeddings or None
        """
        if self.embedding_store is None:
            return None
        
        all_hard_neg_ids = []
        for sample in batch:
            if isinstance(sample, TrainingSample):
                hard_negs = sample.hard_negatives or []
            else:
                hard_negs = sample.get("hard_negatives", [])
            
            # Pad or truncate to num_hard_negatives
            hard_negs = hard_negs[:self.num_hard_negatives]
            while len(hard_negs) < self.num_hard_negatives:
                hard_negs.append(None)
            
            all_hard_neg_ids.append(hard_negs)
        
        # Collect all unique doc_ids
        unique_ids = set()
        for negs in all_hard_neg_ids:
            for neg_id in negs:
                if neg_id is not None:
                    unique_ids.add(neg_id)
        
        if not unique_ids:
            return None
        
        # Load embeddings
        embeddings_dict = self.embedding_store.get_embeddings(list(unique_ids))
        
        # Build output tensor
        batch_size = len(batch)
        
        # Find max sequence length
        max_seq_len = max(
            emb.shape[0] for emb in embeddings_dict.values()
        ) if embeddings_dict else 1
        dim = next(iter(embeddings_dict.values())).shape[-1] if embeddings_dict else 128
        
        hard_neg_embeddings = torch.zeros(
            batch_size, self.num_hard_negatives, max_seq_len, dim
        )
        
        for i, negs in enumerate(all_hard_neg_ids):
            for j, neg_id in enumerate(negs):
                if neg_id is not None and neg_id in embeddings_dict:
                    emb = embeddings_dict[neg_id]
                    seq_len = emb.shape[0]
                    hard_neg_embeddings[i, j, :seq_len, :] = emb
        
        return hard_neg_embeddings


@dataclass
class SimpleCollator:
    """
    Simple collator for basic training without hard negatives.
    
    Just returns the batch as-is for use with DataLoader.
    """
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simple collation - just organize batch into dict of lists.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Dictionary with lists of values
        """
        if not batch:
            return {}
        
        # Get all keys from first sample
        keys = batch[0].keys()
        
        result = {key: [sample[key] for sample in batch] for key in keys}
        return result
