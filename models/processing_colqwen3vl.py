"""
ColQwen3VL Processor Implementation.

Processor for ColQwen3VL model, handling image and text preprocessing.

Requirements: 7.2, 7.3
- Utilize Qwen3-VL's native image processor for variable resolutions
- Tokenize queries with proper padding and truncation
"""

from typing import ClassVar, List, Optional, Tuple, Union
import logging
import math

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

try:
    from transformers.models.qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.image_processing_qwen3_vl import smart_resize
except ImportError:
    # Fallback for older transformers versions
    Qwen3VLProcessor = None
    smart_resize = None

logger = logging.getLogger(__name__)


class BaseVisualRetrieverProcessor:
    """
    Base class for visual retriever processors.
    
    Provides common interface for processing images and texts for retrieval models.
    """
    
    query_prefix: ClassVar[str] = ""  # Default prefix for queries
    query_augmentation_token: ClassVar[str] = ""  # Token for query augmentation
    
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """Process a list of images into a format suitable for the model."""
        raise NotImplementedError
    
    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """Process a list of texts into a format suitable for the model."""
        raise NotImplementedError
    
    def process_queries(
        self,
        texts: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a list of queries into a format suitable for the model.
        
        Args:
            texts: List of input texts.
            queries: Alternative parameter for texts (deprecated).
            max_length: Maximum length of the text (deprecated).
            suffix: Suffix to append to each text.
            
        Returns:
            Processed texts.
        """
        if texts and queries:
            raise ValueError("Only one of 'texts' or 'queries' should be provided.")
        if queries is not None:
            texts = queries
        elif texts is None:
            raise ValueError("No texts or queries provided.")
        
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        
        # Add the query prefix and suffix to each text
        texts = [self.query_prefix + text + suffix for text in texts]
        
        return self.process_texts(texts=texts)
    
    @staticmethod
    def score_multi_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for multi-vector embeddings.
        
        MaxSim computes:
        1. For each query token, find the maximum similarity with any document token
        2. Sum these maximum similarities across all query tokens
        
        Formula: score = Σᵢ maxⱼ(Qᵢ · Dⱼ)
        
        Args:
            qs: Query embeddings - list of tensors (seq_len_i, dim) or 
                tensor (n_queries, max_seq_len, dim)
            ps: Passage/document embeddings - list of tensors (seq_len_i, dim) or
                tensor (n_passages, max_seq_len, dim)
            batch_size: Batch size for computing scores
            device: Device to use for computation
            
        Returns:
            Tensor of shape (n_queries, n_passages) containing MaxSim scores
            
        Requirements:
            3.1: Compute dot product between each query token and all document tokens
            3.2: Take maximum similarity for each query token
            3.3: Sum maximum similarities across all query tokens
            3.4: Support batch processing with proper padding mask
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")
        
        scores_list: List[torch.Tensor] = []
        
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            
            # Get query batch and create mask for valid tokens
            qs_slice = qs[i : i + batch_size]
            qs_lengths = [q.shape[0] for q in qs_slice]
            max_q_len = max(qs_lengths)
            
            # Pad query batch with zeros (will be masked later)
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs_slice, batch_first=True, padding_value=0
            ).to(device)
            
            # Create query mask: (batch_q, max_q_len)
            q_mask = torch.zeros(len(qs_slice), max_q_len, dtype=torch.bool, device=device)
            for idx, length in enumerate(qs_lengths):
                q_mask[idx, :length] = True
            
            for j in range(0, len(ps), batch_size):
                # Get passage batch and create mask for valid tokens
                ps_slice = ps[j : j + batch_size]
                ps_lengths = [p.shape[0] for p in ps_slice]
                max_p_len = max(ps_lengths)
                
                # Pad passage batch with zeros
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps_slice, batch_first=True, padding_value=0
                ).to(device)
                
                # Create passage mask: (batch_p, max_p_len)
                p_mask = torch.zeros(len(ps_slice), max_p_len, dtype=torch.bool, device=device)
                for idx, length in enumerate(ps_lengths):
                    p_mask[idx, :length] = True
                
                # Compute all pairwise similarities: (batch_q, batch_p, max_q_len, max_p_len)
                # einsum: "bnd,csd->bcns" where b=batch_q, n=q_len, d=dim, c=batch_p, s=p_len
                similarities = torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                
                # Create mask for valid passage tokens: (1, batch_p, 1, max_p_len)
                # This masks out padding tokens in passages before taking max
                p_mask_expanded = p_mask.unsqueeze(0).unsqueeze(2)  # (1, batch_p, 1, max_p_len)
                
                # Apply mask: set padding positions to -inf so they're never selected as max
                similarities = similarities.masked_fill(~p_mask_expanded, float('-inf'))
                
                # Take max over passage tokens for each query token: (batch_q, batch_p, max_q_len)
                max_sims = similarities.max(dim=3)[0]
                
                # Create mask for valid query tokens: (batch_q, 1, max_q_len)
                q_mask_expanded = q_mask.unsqueeze(1)  # (batch_q, 1, max_q_len)
                
                # Zero out padding query tokens before summing
                max_sims = max_sims * q_mask_expanded
                
                # Sum over query tokens: (batch_q, batch_p)
                batch_scores = max_sims.sum(dim=2)
                
                scores_batch.append(batch_scores)
            
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)
        
        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"
        
        return scores.to(torch.float32)


class ColQwen3VLProcessor(BaseVisualRetrieverProcessor):
    """
    Processor for ColQwen3VL model.
    
    Handles image preprocessing with dynamic resolution and query tokenization
    with left padding for decoder-only models.
    
    Attributes:
        visual_prompt_prefix: Prompt template for document images
        query_augmentation_token: Token used to pad queries
        image_token: Token representing image patches
        
    Requirements:
        7.2: Utilize Qwen3-VL's native image processor for variable resolutions
        7.3: Tokenize queries with proper padding and truncation
    """
    
    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe the image.<|im_end|><|endoftext|>"
    )
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"
    
    def __init__(
        self,
        processor: Optional["Qwen3VLProcessor"] = None,
        max_num_visual_tokens: Optional[int] = None,
    ):
        """
        Initialize ColQwen3VLProcessor.
        
        Args:
            processor: Base Qwen3VLProcessor instance
            max_num_visual_tokens: Maximum number of visual tokens per image
        """
        if Qwen3VLProcessor is None:
            raise ImportError(
                "Qwen3-VL processor requires transformers >= 4.45.0. "
                "Please upgrade: pip install transformers>=4.45.0"
            )
        
        self._processor = processor
        self._max_num_visual_tokens = max_num_visual_tokens
        
        # Set left padding for decoder-only models
        if processor is not None and hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        max_num_visual_tokens: Optional[int] = None,
        **kwargs,
    ) -> "ColQwen3VLProcessor":
        """
        Load a pretrained ColQwen3VLProcessor.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained processor
            max_num_visual_tokens: Maximum number of visual tokens per image
            **kwargs: Additional arguments passed to from_pretrained
            
        Returns:
            Initialized ColQwen3VLProcessor
        """
        if Qwen3VLProcessor is None:
            raise ImportError(
                "Qwen3-VL processor requires transformers >= 4.45.0. "
                "Please upgrade: pip install transformers>=4.45.0"
            )
        
        logger.info(f"Loading Qwen3-VL processor from {pretrained_model_name_or_path}")
        
        processor = Qwen3VLProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        
        instance = cls(processor=processor, max_num_visual_tokens=max_num_visual_tokens)
        
        # Configure max pixels if max_num_visual_tokens is specified
        if max_num_visual_tokens is not None:
            # Each visual token corresponds to a patch of 28x28 pixels
            instance._processor.image_processor.max_pixels = max_num_visual_tokens * 28 * 28
            if hasattr(instance._processor.image_processor, "size"):
                instance._processor.image_processor.size["longest_edge"] = (
                    instance._processor.image_processor.max_pixels
                )
        
        return instance
    
    @property
    def tokenizer(self):
        """Get the tokenizer from the base processor."""
        return self._processor.tokenizer if self._processor else None
    
    @property
    def image_processor(self):
        """Get the image processor from the base processor."""
        return self._processor.image_processor if self._processor else None
    
    @property
    def image_token_id(self) -> int:
        """Get the image token ID."""
        if self._processor is not None and hasattr(self._processor, "tokenizer"):
            return self._processor.tokenizer.convert_tokens_to_ids(self.image_token)
        return 151655  # Default Qwen3-VL image token ID
    
    def __call__(self, *args, **kwargs):
        """Forward call to the base processor."""
        if self._processor is None:
            raise ValueError("Processor not initialized. Use from_pretrained() to load.")
        return self._processor(*args, **kwargs)
    
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process document images with dynamic resolution.
        
        Uses Qwen3-VL's native image processor to handle variable resolutions
        and aspect ratios without forced resizing that distorts document text.
        
        Args:
            images: List of PIL images to process
            
        Returns:
            BatchFeature containing processed images with:
            - input_ids: Token IDs including image tokens
            - attention_mask: Attention mask
            - pixel_values: Processed pixel values (padded)
            - image_grid_thw: Grid dimensions (temporal, height, width)
            
        Requirements:
            7.2: Utilize Qwen3-VL's native image processor for variable resolutions
        """
        if self._processor is None:
            raise ValueError("Processor not initialized. Use from_pretrained() to load.")
        
        # Convert images to RGB
        images = [image.convert("RGB") for image in images]
        
        # Process images with the visual prompt
        batch_doc = self._processor(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        
        # Handle dynamic resolution: pad pixel_values to same length
        # image_grid_thw: (batch_size, 3) where 3 = (temporal, height, width)
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]
        
        # Split pixel_values into list of tensors per image
        pixel_values = list(
            torch.split(batch_doc["pixel_values"], offsets.tolist())
        )
        
        # Pad to same length for batching
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )
        
        return batch_doc
    
    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts with left padding.
        
        Args:
            texts: List of input texts
            
        Returns:
            BatchFeature containing processed texts
            
        Requirements:
            7.3: Tokenize queries with proper padding
        """
        if self._processor is None:
            raise ValueError("Processor not initialized. Use from_pretrained() to load.")
        
        return self._processor(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )
    
    def process_queries(
        self,
        texts: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process query texts with left padding.
        
        Queries are augmented with suffix tokens to provide more context
        for the model to generate meaningful embeddings.
        
        Args:
            texts: List of query texts
            queries: Alternative parameter for texts (deprecated)
            max_length: Maximum length (deprecated, kept for compatibility)
            suffix: Suffix to append to each query
            
        Returns:
            BatchFeature containing processed queries with left padding
            
        Requirements:
            7.3: Tokenize queries with proper padding and truncation
        """
        if texts and queries:
            raise ValueError("Only one of 'texts' or 'queries' should be provided.")
        if queries is not None:
            texts = queries
        elif texts is None:
            raise ValueError("No texts or queries provided.")
        
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        
        # Add suffix to each query
        texts = [text + suffix for text in texts]
        
        return self.process_texts(texts=texts)
    
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute MaxSim score for query and passage embeddings.
        
        Args:
            qs: List of query embeddings
            ps: List of passage embeddings
            device: Device for computation
            
        Returns:
            Tensor of MaxSim scores
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)
    
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int = 2,
    ) -> Tuple[int, int]:
        """
        Get the number of patches for an image of given size.
        
        Args:
            image_size: Tuple of (width, height) in pixels
            spatial_merge_size: Spatial merge size from model config
            
        Returns:
            Tuple of (n_patches_x, n_patches_y)
        """
        if self._processor is None or smart_resize is None:
            raise ValueError("Processor not initialized or smart_resize not available.")
        
        patch_size = self._processor.image_processor.patch_size
        merge_size = getattr(self._processor.image_processor, "merge_size", 2)
        
        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * merge_size,
            min_pixels=self._processor.image_processor.size.get("shortest_edge", 256),
            max_pixels=self._processor.image_processor.size.get("longest_edge", 1280 * 28 * 28),
        )
        
        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size
        
        return n_patches_x, n_patches_y
    
    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        """
        Get a tensor mask identifying image tokens in the batch.
        
        Args:
            batch_images: BatchFeature containing processed images
            
        Returns:
            Boolean tensor where True indicates image token positions
        """
        return batch_images.input_ids == self.image_token_id
