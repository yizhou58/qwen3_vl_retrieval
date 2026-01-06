"""
ColPali Engine Dataset Implementation.

Dataset class for training ColQwen3VL retrieval model with contrastive learning.

Requirements: 7.1, 7.4
- Support loading image-query pairs from standard datasets (e.g., ViDoRe format)
- Support in-batch negative sampling for contrastive learning
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import logging
import random

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """
    Single training sample for contrastive learning.
    
    Attributes:
        query: Query text
        positive_image: Positive document image
        positive_doc_id: Document ID
        hard_negatives: Hard negative doc IDs (optional)
    """
    query: str
    positive_image: Image.Image
    positive_doc_id: str
    hard_negatives: Optional[List[str]] = None


@dataclass
class TrainingBatch:
    """
    Batch of training samples.
    
    Attributes:
        query_input_ids: Token IDs for queries (batch_size, query_len)
        query_attention_mask: Attention mask for queries (batch_size, query_len)
        doc_input_ids: Token IDs for documents (batch_size, doc_len)
        doc_attention_mask: Attention mask for documents (batch_size, doc_len)
        doc_pixel_values: Pixel values for document images (variable shape)
        doc_image_grid_thw: Image grid dimensions (batch_size, 3)
    """
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    doc_input_ids: torch.Tensor
    doc_attention_mask: torch.Tensor
    doc_pixel_values: torch.Tensor
    doc_image_grid_thw: torch.Tensor


class ColPaliEngineDataset(Dataset):
    """
    Dataset for training ColQwen3VL retrieval model.
    
    Supports loading image-query pairs from various formats including:
    - ViDoRe format (JSON with image paths and queries)
    - Custom format with image directory and query file
    
    Implements in-batch negative sampling where each query's positive
    document serves as negative for other queries in the batch.
    
    Requirements:
        7.1: Support loading image-query pairs from standard datasets
        7.4: Support in-batch negative sampling for contrastive learning
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        max_samples: Optional[int] = None,
        hard_negatives_path: Optional[Union[str, Path]] = None,
        num_hard_negatives: int = 0,
    ):
        """
        Initialize ColPaliEngineDataset.
        
        Args:
            data_path: Path to data file (JSON or JSONL format)
            image_dir: Directory containing images (if not in data file)
            transform: Optional image transform function
            max_samples: Maximum number of samples to load
            hard_negatives_path: Path to hard negatives file
            num_hard_negatives: Number of hard negatives per sample
        """
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir) if image_dir else self.data_path.parent
        self.transform = transform
        self.num_hard_negatives = num_hard_negatives
        
        # Load data
        self.samples = self._load_data(max_samples)
        
        # Load hard negatives if provided
        self.hard_negatives: Dict[str, List[str]] = {}
        if hard_negatives_path:
            self.hard_negatives = self._load_hard_negatives(hard_negatives_path)
        
        # Build doc_id to index mapping for hard negative lookup
        self.doc_id_to_idx: Dict[str, int] = {
            sample["doc_id"]: idx for idx, sample in enumerate(self.samples)
        }
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Supports JSON and JSONL formats with the following structure:
        - query: Query text
        - image_path or image: Path to document image
        - doc_id: Document identifier
        
        Args:
            max_samples: Maximum number of samples to load
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        if self.data_path.suffix == ".jsonl":
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if max_samples and len(samples) >= max_samples:
                        break
                    sample = json.loads(line.strip())
                    samples.append(self._normalize_sample(sample))
        elif self.data_path.suffix == ".json":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for sample in data:
                        if max_samples and len(samples) >= max_samples:
                            break
                        samples.append(self._normalize_sample(sample))
                else:
                    # Handle dict format with "data" key
                    data_list = data.get("data", data.get("samples", [data]))
                    for sample in data_list:
                        if max_samples and len(samples) >= max_samples:
                            break
                        samples.append(self._normalize_sample(sample))
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        return samples
    
    def _normalize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize sample dictionary to standard format.
        
        Args:
            sample: Raw sample dictionary
            
        Returns:
            Normalized sample with query, image_path, doc_id
        """
        normalized = {}
        
        # Query field
        normalized["query"] = sample.get("query", sample.get("question", sample.get("text", "")))
        
        # Image path field
        image_path = sample.get("image_path", sample.get("image", sample.get("page_image", "")))
        if not Path(image_path).is_absolute():
            image_path = str(self.image_dir / image_path)
        normalized["image_path"] = image_path
        
        # Document ID field
        normalized["doc_id"] = sample.get("doc_id", sample.get("id", sample.get("page_id", str(hash(image_path)))))
        
        # Optional metadata
        if "metadata" in sample:
            normalized["metadata"] = sample["metadata"]
        
        return normalized
    
    def _load_hard_negatives(self, path: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Load hard negatives from file.
        
        Args:
            path: Path to hard negatives file (JSON format)
            
        Returns:
            Dictionary mapping query/doc_id to list of hard negative doc_ids
        """
        path = Path(path)
        hard_negatives = {}
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for key, negatives in data.items():
                hard_negatives[key] = negatives[:self.num_hard_negatives] if self.num_hard_negatives > 0 else negatives
        
        logger.info(f"Loaded hard negatives for {len(hard_negatives)} samples")
        return hard_negatives
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> TrainingSample:
        """
        Get a training sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            TrainingSample with query, image, doc_id, and optional hard negatives
        """
        sample = self.samples[idx]
        
        # Load image
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return a placeholder image
            image = Image.new("RGB", (224, 224), color="white")
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Get hard negatives
        doc_id = sample["doc_id"]
        hard_negatives = self.hard_negatives.get(doc_id, None)
        
        return TrainingSample(
            query=sample["query"],
            positive_image=image,
            positive_doc_id=doc_id,
            hard_negatives=hard_negatives,
        )
    
    def get_sample_by_doc_id(self, doc_id: str) -> Optional[TrainingSample]:
        """
        Get a sample by document ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            TrainingSample if found, None otherwise
        """
        idx = self.doc_id_to_idx.get(doc_id)
        if idx is not None:
            return self[idx]
        return None
    
    def get_in_batch_negatives(
        self,
        batch_indices: List[int],
        query_idx: int,
    ) -> List[int]:
        """
        Get in-batch negative indices for a query.
        
        In-batch negative sampling: all other documents in the batch
        serve as negatives for the current query.
        
        Args:
            batch_indices: List of sample indices in the batch
            query_idx: Index of the current query in batch_indices
            
        Returns:
            List of indices (within batch_indices) that are negatives
            
        Requirements:
            7.4: Support in-batch negative sampling for contrastive learning
        """
        return [i for i in range(len(batch_indices)) if i != query_idx]


class ViDoReDataset(ColPaliEngineDataset):
    """
    Dataset for ViDoRe benchmark format.
    
    ViDoRe (Visual Document Retrieval) format:
    - queries.jsonl: Query texts with query_id
    - qrels.jsonl: Query-document relevance judgments
    - corpus/: Directory containing document images
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize ViDoReDataset.
        
        Args:
            data_dir: Path to ViDoRe dataset directory
            split: Dataset split (train, dev, test)
            transform: Optional image transform function
            max_samples: Maximum number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load queries
        queries_path = self.data_dir / f"{split}_queries.jsonl"
        if not queries_path.exists():
            queries_path = self.data_dir / "queries.jsonl"
        
        self.queries = self._load_queries(queries_path)
        
        # Load qrels (query-document relevance)
        qrels_path = self.data_dir / f"{split}_qrels.jsonl"
        if not qrels_path.exists():
            qrels_path = self.data_dir / "qrels.jsonl"
        
        self.qrels = self._load_qrels(qrels_path)
        
        # Build samples
        self.samples = self._build_samples(max_samples)
        self.image_dir = self.data_dir / "corpus"
        
        # Build doc_id to index mapping
        self.doc_id_to_idx = {
            sample["doc_id"]: idx for idx, sample in enumerate(self.samples)
        }
        
        self.hard_negatives: Dict[str, List[str]] = {}
        self.num_hard_negatives = 0
        
        logger.info(f"Loaded {len(self.samples)} samples from ViDoRe dataset")
    
    def _load_queries(self, path: Path) -> Dict[str, str]:
        """Load queries from JSONL file."""
        queries = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                query_id = data.get("query_id", data.get("_id", data.get("id")))
                query_text = data.get("text", data.get("query"))
                queries[str(query_id)] = query_text
        return queries
    
    def _load_qrels(self, path: Path) -> Dict[str, List[str]]:
        """Load query-document relevance judgments."""
        qrels = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                query_id = str(data.get("query_id", data.get("_id")))
                doc_id = str(data.get("doc_id", data.get("corpus_id")))
                relevance = data.get("score", data.get("relevance", 1))
                
                if relevance > 0:
                    if query_id not in qrels:
                        qrels[query_id] = []
                    qrels[query_id].append(doc_id)
        return qrels
    
    def _build_samples(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Build samples from queries and qrels."""
        samples = []
        
        for query_id, query_text in self.queries.items():
            if query_id not in self.qrels:
                continue
            
            for doc_id in self.qrels[query_id]:
                if max_samples and len(samples) >= max_samples:
                    break
                
                # Construct image path
                image_path = self.data_dir / "corpus" / f"{doc_id}.png"
                if not image_path.exists():
                    image_path = self.data_dir / "corpus" / f"{doc_id}.jpg"
                
                samples.append({
                    "query": query_text,
                    "image_path": str(image_path),
                    "doc_id": doc_id,
                    "query_id": query_id,
                })
            
            if max_samples and len(samples) >= max_samples:
                break
        
        return samples
