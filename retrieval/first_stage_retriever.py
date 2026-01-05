"""
First-stage retrieval module for fast document recall.

Supports BM25 keyword-based retrieval and BGE-M3 dense vector retrieval.
Maintains strict ID mapping between indexed text chunks and their corresponding
original PDF page images.

Requirements: 5.1, 5.2, 5.4, 5.5, 5.6
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class FirstStageRetriever:
    """
    Fast first-stage retrieval using BM25 or dense vectors.
    
    Maintains ID mapping between text chunks and document images.
    Supports incremental index updates.
    
    Attributes:
        method: Retrieval method ("bm25" or "bge-m3")
        index_path: Path to save/load index
        doc_ids: List of document IDs
        texts: List of document texts
        image_paths: Dict mapping doc_id to image path
    """
    
    def __init__(
        self,
        method: Literal["bm25", "bge-m3"] = "bm25",
        index_path: Optional[str] = None
    ):
        """
        Initialize the first-stage retriever.
        
        Args:
            method: Retrieval method ("bm25" or "bge-m3")
            index_path: Optional path to save/load index
        """
        self.method = method
        self.index_path = index_path
        
        # Document storage
        self.doc_ids: List[str] = []
        self.texts: List[str] = []
        self.image_paths: Dict[str, str] = {}
        
        # BM25 index
        self._bm25_index = None
        self._tokenized_corpus: List[List[str]] = []
        
        # BGE-M3 index
        self._bge_model = None
        self._doc_embeddings: Optional[np.ndarray] = None
        
        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization with lowercasing
        # For better results, consider using a proper tokenizer
        return text.lower().split()
    
    def _init_bm25(self) -> None:
        """Initialize or reinitialize BM25 index from tokenized corpus."""
        if not self._tokenized_corpus:
            self._bm25_index = None
            return
            
        from rank_bm25 import BM25Okapi
        self._bm25_index = BM25Okapi(self._tokenized_corpus)
    
    def _init_bge_model(self) -> None:
        """Lazy load BGE-M3 model."""
        if self._bge_model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading BGE-M3 model...")
            self._bge_model = SentenceTransformer("BAAI/bge-m3")
            logger.info("BGE-M3 model loaded successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for BGE-M3 retrieval. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load BGE-M3 model: {e}")

    
    def index_documents(
        self,
        doc_ids: List[str],
        texts: List[str],
        image_paths: List[str]
    ) -> None:
        """
        Index documents with text and image path mapping.
        
        Args:
            doc_ids: List of unique document identifiers
            texts: List of document texts (OCR or metadata)
            image_paths: List of paths to document images
            
        Raises:
            ValueError: If input lists have different lengths or contain duplicates
        """
        # Validate inputs
        if not (len(doc_ids) == len(texts) == len(image_paths)):
            raise ValueError(
                f"Input lists must have same length. Got doc_ids={len(doc_ids)}, "
                f"texts={len(texts)}, image_paths={len(image_paths)}"
            )
        
        # Check for duplicate doc_ids in new batch
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("Duplicate doc_ids found in input")
        
        # Check for conflicts with existing doc_ids
        existing_ids = set(self.doc_ids)
        new_ids = set(doc_ids)
        conflicts = existing_ids & new_ids
        if conflicts:
            logger.warning(
                f"Found {len(conflicts)} duplicate doc_ids. "
                "These will be updated with new values."
            )
            # Remove conflicting entries
            for conflict_id in conflicts:
                idx = self.doc_ids.index(conflict_id)
                self.doc_ids.pop(idx)
                self.texts.pop(idx)
                self._tokenized_corpus.pop(idx) if self._tokenized_corpus else None
        
        # Add new documents
        for doc_id, text, image_path in zip(doc_ids, texts, image_paths):
            self.doc_ids.append(doc_id)
            self.texts.append(text)
            self.image_paths[doc_id] = image_path
            
            if self.method == "bm25":
                self._tokenized_corpus.append(self._tokenize(text))
        
        # Rebuild index
        if self.method == "bm25":
            self._init_bm25()
        elif self.method == "bge-m3":
            self._build_bge_index()
        
        logger.info(f"Indexed {len(doc_ids)} documents. Total: {len(self.doc_ids)}")
    
    def _build_bge_index(self) -> None:
        """Build BGE-M3 dense vector index."""
        if not self.texts:
            self._doc_embeddings = None
            return
            
        self._init_bge_model()
        
        logger.info(f"Encoding {len(self.texts)} documents with BGE-M3...")
        self._doc_embeddings = self._bge_model.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        logger.info("BGE-M3 encoding complete")
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        image_path: str
    ) -> None:
        """
        Add a single document to the index (incremental update).
        
        Args:
            doc_id: Unique document identifier
            text: Document text (OCR or metadata)
            image_path: Path to document image
        """
        self.index_documents([doc_id], [text], [image_path])
    
    def retrieve(
        self,
        query: str,
        top_k: int = 100
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve top-k candidates for a query.
        
        Args:
            query: Query text
            top_k: Number of candidates to return (default: 100)
            
        Returns:
            List of (doc_id, score, image_path) tuples sorted by score descending
            
        Raises:
            ValueError: If top_k is not positive
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        if not self.doc_ids:
            logger.warning("Retrieval attempted on empty index")
            return []
        
        # Limit top_k to available documents
        top_k = min(top_k, len(self.doc_ids))
        
        if self.method == "bm25":
            return self._retrieve_bm25(query, top_k)
        elif self.method == "bge-m3":
            return self._retrieve_bge(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")

    
    def _retrieve_bm25(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve using BM25.
        
        Args:
            query: Query text
            top_k: Number of candidates to return
            
        Returns:
            List of (doc_id, score, image_path) tuples
        """
        if self._bm25_index is None:
            logger.warning("BM25 index not initialized")
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self._bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = float(scores[idx])
            image_path = self.image_paths.get(doc_id, "")
            results.append((doc_id, score, image_path))
        
        return results
    
    def _retrieve_bge(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve using BGE-M3 dense vectors.
        
        Args:
            query: Query text
            top_k: Number of candidates to return
            
        Returns:
            List of (doc_id, score, image_path) tuples
        """
        if self._doc_embeddings is None:
            logger.warning("BGE-M3 index not initialized")
            return []
        
        self._init_bge_model()
        
        # Encode query
        query_embedding = self._bge_model.encode(
            [query],
            normalize_embeddings=True
        )[0]
        
        # Compute cosine similarity (embeddings are normalized)
        scores = np.dot(self._doc_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = float(scores[idx])
            image_path = self.image_paths.get(doc_id, "")
            results.append((doc_id, score, image_path))
        
        return results
    
    def save_index(self, path: Optional[str] = None) -> None:
        """
        Save index to disk.
        
        Args:
            path: Path to save index (uses self.index_path if not provided)
        """
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No save path provided")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "method": self.method,
            "doc_ids": self.doc_ids,
            "texts": self.texts,
            "image_paths": self.image_paths,
        }
        
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Save method-specific data
        if self.method == "bm25":
            with open(save_path / "tokenized_corpus.pkl", "wb") as f:
                pickle.dump(self._tokenized_corpus, f)
        elif self.method == "bge-m3" and self._doc_embeddings is not None:
            np.save(save_path / "doc_embeddings.npy", self._doc_embeddings)
        
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, path: Optional[str] = None) -> None:
        """
        Load index from disk.
        
        Args:
            path: Path to load index from (uses self.index_path if not provided)
        """
        load_path = path or self.index_path
        if not load_path:
            raise ValueError("No load path provided")
        
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index path not found: {load_path}")
        
        # Load metadata
        with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        self.method = metadata["method"]
        self.doc_ids = metadata["doc_ids"]
        self.texts = metadata["texts"]
        self.image_paths = metadata["image_paths"]
        
        # Load method-specific data
        if self.method == "bm25":
            tokenized_path = load_path / "tokenized_corpus.pkl"
            if tokenized_path.exists():
                with open(tokenized_path, "rb") as f:
                    self._tokenized_corpus = pickle.load(f)
                self._init_bm25()
        elif self.method == "bge-m3":
            embeddings_path = load_path / "doc_embeddings.npy"
            if embeddings_path.exists():
                self._doc_embeddings = np.load(embeddings_path)
        
        logger.info(f"Index loaded from {load_path}. {len(self.doc_ids)} documents.")
    
    def get_document_count(self) -> int:
        """Return the number of indexed documents."""
        return len(self.doc_ids)
    
    def get_image_path(self, doc_id: str) -> str:
        """
        Get image path for a document ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Image path
            
        Raises:
            KeyError: If doc_id not found
        """
        if doc_id not in self.image_paths:
            raise KeyError(f"Document ID not found: {doc_id}")
        return self.image_paths[doc_id]
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        self.doc_ids = []
        self.texts = []
        self.image_paths = {}
        self._tokenized_corpus = []
        self._bm25_index = None
        self._doc_embeddings = None
        logger.info("Index cleared")
    
    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return len(self.doc_ids)
    
    def __contains__(self, doc_id: str) -> bool:
        """Check if a document ID is in the index."""
        return doc_id in self.image_paths
