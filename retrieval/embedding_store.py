"""
Embedding Store for pre-computed document embeddings.

This module implements efficient storage and retrieval of ColPali-style multi-vector
embeddings using LMDB backend for high-performance random access.

**Validates: Requirements 6.2, 8.4**
- Support pre-computing and caching document embeddings
- Retrieve pre-computed multi-vector embeddings for candidate documents
"""

import os
import json
import struct
import logging
from typing import Dict, List, Optional, Tuple, Union, Literal
from pathlib import Path

import torch
import numpy as np

try:
    import lmdb
except ImportError:
    lmdb = None

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Storage for pre-computed document embeddings using LMDB.
    
    For ColPali-style multi-vector architecture, a single PDF page may have
    ~1000 vectors. LMDB provides efficient random access for retrieval.
    
    Structure:
    - Metadata DB: doc_id -> {num_tokens, has_binary, metadata}
    - Float DB: doc_id -> float32 embeddings (num_tokens, dim)
    - Binary DB: doc_id -> uint8 binary embeddings (num_tokens, dim // 8)
    
    Attributes:
        storage_path: Path to storage directory
        dim: Embedding dimension (default: 128)
        
    Example:
        >>> store = EmbeddingStore("./embeddings", dim=128)
        >>> store.add_embeddings("doc1", embeddings, binary_embeddings)
        >>> retrieved = store.get_embeddings(["doc1"])
    """
    
    def __init__(
        self,
        storage_path: str,
        dim: int = 128,
        map_size: int = 10 * 1024 * 1024 * 1024,  # 10GB default
    ):
        """
        Initialize EmbeddingStore.
        
        Args:
            storage_path: Path to storage directory
            dim: Embedding dimension (must be divisible by 8 for binary)
            map_size: Maximum size of LMDB database in bytes
            
        Raises:
            ImportError: If lmdb is not installed
            ValueError: If dim is not divisible by 8
        """
        if lmdb is None:
            raise ImportError(
                "LMDB is required for EmbeddingStore. "
                "Please install: pip install lmdb>=1.4.0"
            )
        
        if dim % 8 != 0:
            raise ValueError(f"Embedding dimension must be divisible by 8, got {dim}")
        
        self.storage_path = Path(storage_path)
        self.dim = dim
        self.map_size = map_size
        self._env = None
        self._metadata_db = None
        self._float_db = None
        self._binary_db = None
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize LMDB environment
        self._init_lmdb()
    
    def _init_lmdb(self) -> None:
        """Initialize LMDB environment with multiple databases."""
        self._env = lmdb.open(
            str(self.storage_path),
            map_size=self.map_size,
            max_dbs=3,  # metadata, float, binary
            writemap=True,
            meminit=False,
        )
        
        # Open named databases
        self._metadata_db = self._env.open_db(b"metadata")
        self._float_db = self._env.open_db(b"float_embeddings")
        self._binary_db = self._env.open_db(b"binary_embeddings")
        
        logger.info(f"LMDB initialized at {self.storage_path}")
    
    def add_embeddings(
        self,
        doc_id: str,
        embeddings: torch.Tensor,
        binary_embeddings: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add document embeddings to store.
        
        Args:
            doc_id: Unique document identifier
            embeddings: Float embeddings (num_tokens, dim) as float32
            binary_embeddings: Binary embeddings (num_tokens, dim // 8) as uint8
            metadata: Optional metadata dict (JSON serializable)
            
        Raises:
            ValueError: If embedding dimensions don't match
        """
        # Validate embeddings
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D embeddings, got {embeddings.dim()}D")
        
        num_tokens, emb_dim = embeddings.shape
        if emb_dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {emb_dim}")
        
        # Convert to numpy for storage
        # Note: bfloat16 is not supported by numpy, so convert to float32 first
        emb_cpu = embeddings.detach().cpu()
        if emb_cpu.dtype == torch.bfloat16:
            emb_cpu = emb_cpu.float()  # Convert bfloat16 to float32
        float_data = emb_cpu.numpy().astype(np.float32)
        
        # Prepare metadata
        meta = {
            "num_tokens": num_tokens,
            "dim": self.dim,
            "has_binary": binary_embeddings is not None,
        }
        if metadata:
            meta["user_metadata"] = metadata
        
        # Encode doc_id as bytes
        doc_key = doc_id.encode("utf-8")
        
        with self._env.begin(write=True) as txn:
            # Store metadata
            txn.put(doc_key, json.dumps(meta).encode("utf-8"), db=self._metadata_db)
            
            # Store float embeddings
            txn.put(doc_key, float_data.tobytes(), db=self._float_db)
            
            # Store binary embeddings if provided
            if binary_embeddings is not None:
                if binary_embeddings.shape != (num_tokens, self.dim // 8):
                    raise ValueError(
                        f"Binary embeddings shape mismatch: expected "
                        f"({num_tokens}, {self.dim // 8}), got {binary_embeddings.shape}"
                    )
                binary_data = binary_embeddings.detach().cpu().numpy().astype(np.uint8)
                txn.put(doc_key, binary_data.tobytes(), db=self._binary_db)
        
        logger.debug(f"Added embeddings for doc_id={doc_id}, num_tokens={num_tokens}")
    
    def get_embeddings(
        self,
        doc_ids: List[str],
        binary: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve embeddings for given document IDs.
        
        Uses memory-mapped access for efficient random reads.
        
        Args:
            doc_ids: List of document IDs
            binary: If True, return binary embeddings; if False, return float
            
        Returns:
            Dict mapping doc_id to embeddings tensor
            Missing doc_ids are not included in the result
        """
        result = {}
        db = self._binary_db if binary else self._float_db
        
        with self._env.begin(buffers=True) as txn:
            for doc_id in doc_ids:
                doc_key = doc_id.encode("utf-8")
                
                # Get metadata first
                meta_data = txn.get(doc_key, db=self._metadata_db)
                if meta_data is None:
                    logger.warning(f"Document {doc_id} not found in store")
                    continue
                
                meta = json.loads(bytes(meta_data).decode("utf-8"))
                num_tokens = meta["num_tokens"]
                
                # Get embeddings
                emb_data = txn.get(doc_key, db=db)
                if emb_data is None:
                    if binary and not meta.get("has_binary", False):
                        logger.warning(f"Binary embeddings not available for {doc_id}")
                    continue
                
                # Convert to tensor
                if binary:
                    arr = np.frombuffer(bytes(emb_data), dtype=np.uint8)
                    arr = arr.reshape(num_tokens, self.dim // 8)
                    result[doc_id] = torch.from_numpy(arr.copy())
                else:
                    arr = np.frombuffer(bytes(emb_data), dtype=np.float32)
                    arr = arr.reshape(num_tokens, self.dim)
                    result[doc_id] = torch.from_numpy(arr.copy())
        
        return result
    
    def get_metadata(self, doc_id: str) -> Optional[Dict]:
        """
        Get metadata for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Metadata dict or None if not found
        """
        doc_key = doc_id.encode("utf-8")
        
        with self._env.begin() as txn:
            meta_data = txn.get(doc_key, db=self._metadata_db)
            if meta_data is None:
                return None
            return json.loads(meta_data.decode("utf-8"))
    
    def contains(self, doc_id: str) -> bool:
        """Check if a document exists in the store."""
        doc_key = doc_id.encode("utf-8")
        
        with self._env.begin() as txn:
            return txn.get(doc_key, db=self._metadata_db) is not None
    
    def list_doc_ids(self) -> List[str]:
        """List all document IDs in the store."""
        doc_ids = []
        
        with self._env.begin() as txn:
            cursor = txn.cursor(db=self._metadata_db)
            for key, _ in cursor:
                doc_ids.append(key.decode("utf-8"))
        
        return doc_ids
    
    def delete_embeddings(self, doc_id: str) -> bool:
        """
        Delete embeddings for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if deleted, False if not found
        """
        doc_key = doc_id.encode("utf-8")
        
        with self._env.begin(write=True) as txn:
            # Check if exists
            if txn.get(doc_key, db=self._metadata_db) is None:
                return False
            
            # Delete from all databases
            txn.delete(doc_key, db=self._metadata_db)
            txn.delete(doc_key, db=self._float_db)
            txn.delete(doc_key, db=self._binary_db)
        
        logger.debug(f"Deleted embeddings for doc_id={doc_id}")
        return True
    
    def __len__(self) -> int:
        """Return the number of documents in the store."""
        with self._env.begin() as txn:
            return txn.stat(db=self._metadata_db)["entries"]
    
    def close(self) -> None:
        """Close the LMDB environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
            logger.info("LMDB environment closed")
    
    def __enter__(self) -> "EmbeddingStore":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure LMDB is closed."""
        self.close()

    def batch_encode_documents(
        self,
        model: "ColQwen3VL",
        processor: "ColQwen3VLProcessor",
        image_paths: List[str],
        doc_ids: List[str],
        batch_size: int = 4,
        quantize: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Batch encode and store document embeddings.
        
        Encodes document images using the ColQwen3VL model and stores both
        float and optionally binary embeddings.
        
        Args:
            model: ColQwen3VL model for encoding
            processor: ColQwen3VLProcessor for image preprocessing
            image_paths: List of image file paths
            doc_ids: List of document IDs (must match image_paths length)
            batch_size: Batch size for encoding
            quantize: Whether to also store binary embeddings
            device: Device for model inference
            show_progress: Whether to show progress bar
            
        Raises:
            ValueError: If image_paths and doc_ids have different lengths
            FileNotFoundError: If an image file is not found
            
        Requirements:
            6.2: Retrieve pre-computed multi-vector embeddings
            8.3: Provide APIs for encoding documents
        """
        from PIL import Image
        from tqdm import tqdm
        
        if len(image_paths) != len(doc_ids):
            raise ValueError(
                f"image_paths ({len(image_paths)}) and doc_ids ({len(doc_ids)}) "
                "must have the same length"
            )
        
        if device is None:
            device = model.device
        
        # Import BinaryQuantizer for quantization
        quantizer = None
        if quantize:
            from .binary_quantizer import BinaryQuantizer
            quantizer = BinaryQuantizer()
        
        # Process in batches
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        iterator = range(0, len(image_paths), batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Encoding documents")
        
        model.eval()
        
        with torch.no_grad():
            for batch_start in iterator:
                batch_end = min(batch_start + batch_size, len(image_paths))
                batch_paths = image_paths[batch_start:batch_end]
                batch_doc_ids = doc_ids[batch_start:batch_end]
                
                # Load images
                images = []
                valid_indices = []
                for i, path in enumerate(batch_paths):
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                        valid_indices.append(i)
                    except Exception as e:
                        logger.error(f"Failed to load image {path}: {e}")
                        continue
                
                if not images:
                    continue
                
                # Process images
                logger.debug(f"Processing batch {batch_start//batch_size + 1}: {len(images)} images")
                batch_inputs = processor.process_images(images)
                
                # Move to device
                batch_inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_inputs.items()
                }
                
                # Encode
                logger.debug(f"Encoding batch with input shape: {batch_inputs.get('input_ids', torch.tensor([])).shape}")
                embeddings = model(**batch_inputs)  # (batch, seq_len, dim)
                
                # Store each document's embeddings
                for idx, valid_idx in enumerate(valid_indices):
                    doc_id = batch_doc_ids[valid_idx]
                    
                    # Get embeddings for this document
                    doc_emb = embeddings[idx]  # (seq_len, dim)
                    
                    # Get attention mask to filter out padding
                    attn_mask = batch_inputs["attention_mask"][idx]  # (seq_len,)
                    
                    # Filter out padding tokens
                    valid_mask = attn_mask.bool()
                    doc_emb = doc_emb[valid_mask]  # (num_valid_tokens, dim)
                    
                    # Quantize if requested
                    binary_emb = None
                    if quantize and quantizer is not None:
                        binary_emb = quantizer.quantize(doc_emb)
                    
                    # Store embeddings
                    self.add_embeddings(
                        doc_id=doc_id,
                        embeddings=doc_emb,
                        binary_embeddings=binary_emb,
                        metadata={"image_path": batch_paths[valid_idx]},
                    )
        
        logger.info(f"Encoded and stored {len(doc_ids)} documents")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the embedding store.
        
        Returns:
            Dict with statistics including:
            - num_documents: Number of documents
            - total_tokens: Total number of tokens across all documents
            - avg_tokens_per_doc: Average tokens per document
            - storage_size_bytes: Total storage size in bytes
        """
        stats = {
            "num_documents": 0,
            "total_tokens": 0,
            "num_with_binary": 0,
        }
        
        with self._env.begin() as txn:
            cursor = txn.cursor(db=self._metadata_db)
            for _, meta_data in cursor:
                meta = json.loads(meta_data.decode("utf-8"))
                stats["num_documents"] += 1
                stats["total_tokens"] += meta["num_tokens"]
                if meta.get("has_binary", False):
                    stats["num_with_binary"] += 1
        
        if stats["num_documents"] > 0:
            stats["avg_tokens_per_doc"] = stats["total_tokens"] / stats["num_documents"]
        else:
            stats["avg_tokens_per_doc"] = 0
        
        # Estimate storage size
        float_size = stats["total_tokens"] * self.dim * 4  # float32
        binary_size = stats["num_with_binary"] * stats["total_tokens"] * (self.dim // 8)
        stats["estimated_float_storage_bytes"] = float_size
        stats["estimated_binary_storage_bytes"] = binary_size
        
        return stats


# Type hints for forward references
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.colqwen3vl import ColQwen3VL
    from ..models.processing_colqwen3vl import ColQwen3VLProcessor
