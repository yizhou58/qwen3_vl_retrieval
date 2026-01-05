"""
Property-Based Tests for Embedding Store Round-Trip.

**Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
**Validates: Requirements 8.4**

Property 10: For any document image, encoding it and storing the embedding, 
then retrieving the embedding from cache, SHALL produce identical values 
to the original encoding (within floating-point tolerance).
"""

import pytest
import torch
import tempfile
import shutil
import os
from hypothesis import given, settings, strategies as st, assume
from typing import Tuple, List

# Check if lmdb is available
try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False

from qwen3_vl_retrieval.retrieval.binary_quantizer import BinaryQuantizer

# Conditionally import EmbeddingStore
if LMDB_AVAILABLE:
    from qwen3_vl_retrieval.retrieval.embedding_store import EmbeddingStore

# Skip all tests if LMDB is not available
pytestmark = pytest.mark.skipif(
    not LMDB_AVAILABLE,
    reason="LMDB is required for EmbeddingStore tests. Install with: pip install lmdb>=1.4.0"
)


# Hypothesis strategies
@st.composite
def embedding_strategy(draw):
    """Generate random embeddings with dimension 128 (standard ColPali dim)."""
    n_tokens = draw(st.integers(min_value=1, max_value=100))
    dim = 128  # Standard embedding dimension
    
    # Generate random float embeddings
    embeddings = torch.randn(n_tokens, dim)
    
    return embeddings


@st.composite
def doc_id_strategy(draw):
    """Generate valid document IDs."""
    # Generate alphanumeric doc IDs
    length = draw(st.integers(min_value=1, max_value=50))
    chars = st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-")
    doc_id = draw(st.text(alphabet=chars, min_size=1, max_size=length))
    return doc_id


@st.composite
def embedding_with_doc_id_strategy(draw):
    """Generate embeddings with a document ID."""
    embeddings = draw(embedding_strategy())
    doc_id = draw(doc_id_strategy())
    return doc_id, embeddings


@st.composite
def multiple_docs_strategy(draw):
    """Generate multiple documents with embeddings."""
    num_docs = draw(st.integers(min_value=1, max_value=10))
    docs = []
    used_ids = set()
    
    for _ in range(num_docs):
        # Generate unique doc_id
        doc_id = draw(doc_id_strategy())
        while doc_id in used_ids:
            doc_id = draw(doc_id_strategy())
        used_ids.add(doc_id)
        
        embeddings = draw(embedding_strategy())
        docs.append((doc_id, embeddings))
    
    return docs


class TestProperty10EmbeddingCachingRoundTrip:
    """
    Property 10: Embedding Caching Round-Trip
    
    **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
    **Validates: Requirements 8.4**
    
    For any document image, encoding it and storing the embedding, then 
    retrieving the embedding from cache, SHALL produce identical values 
    to the original encoding (within floating-point tolerance).
    """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Create and cleanup temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(data=embedding_with_doc_id_strategy())
    @settings(max_examples=100, deadline=None)
    def test_float_embedding_round_trip(self, data: Tuple[str, torch.Tensor]):
        """
        Property: Float embeddings stored and retrieved are identical.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        import uuid
        doc_id, embeddings = data
        
        # Create store with unique path
        store_path = os.path.join(self.temp_dir, f"store_{uuid.uuid4().hex}")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # Store embeddings
            store.add_embeddings(doc_id, embeddings)
            
            # Retrieve embeddings
            retrieved = store.get_embeddings([doc_id])
            
            # Property: retrieved embeddings should match original
            assert doc_id in retrieved, f"Document {doc_id} not found in retrieved"
            
            retrieved_emb = retrieved[doc_id]
            
            assert retrieved_emb.shape == embeddings.shape, \
                f"Shape mismatch: expected {embeddings.shape}, got {retrieved_emb.shape}"
            
            assert torch.allclose(retrieved_emb, embeddings, atol=1e-6), \
                "Float embeddings round-trip mismatch"
        finally:
            store.close()
    
    @given(data=embedding_with_doc_id_strategy())
    @settings(max_examples=100, deadline=None)
    def test_binary_embedding_round_trip(self, data: Tuple[str, torch.Tensor]):
        """
        Property: Binary embeddings stored and retrieved are identical.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        doc_id, embeddings = data
        
        # Create store and quantizer
        store_path = os.path.join(self.temp_dir, f"store_bin_{hash(doc_id) % 10000}")
        store = EmbeddingStore(store_path, dim=128)
        quantizer = BinaryQuantizer()
        
        try:
            # Quantize embeddings
            binary_embeddings = quantizer.quantize(embeddings)
            
            # Store both float and binary embeddings
            store.add_embeddings(doc_id, embeddings, binary_embeddings)
            
            # Retrieve binary embeddings
            retrieved = store.get_embeddings([doc_id], binary=True)
            
            # Property: retrieved binary embeddings should match original
            assert doc_id in retrieved, f"Document {doc_id} not found in retrieved"
            
            retrieved_binary = retrieved[doc_id]
            
            assert retrieved_binary.shape == binary_embeddings.shape, \
                f"Shape mismatch: expected {binary_embeddings.shape}, got {retrieved_binary.shape}"
            
            assert torch.all(retrieved_binary == binary_embeddings), \
                "Binary embeddings round-trip mismatch"
        finally:
            store.close()
    
    @given(docs=multiple_docs_strategy())
    @settings(max_examples=50, deadline=None)
    def test_multiple_documents_round_trip(self, docs: List[Tuple[str, torch.Tensor]]):
        """
        Property: Multiple documents can be stored and retrieved correctly.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, f"store_multi_{len(docs)}")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # Store all documents
            for doc_id, embeddings in docs:
                store.add_embeddings(doc_id, embeddings)
            
            # Retrieve all documents
            doc_ids = [doc_id for doc_id, _ in docs]
            retrieved = store.get_embeddings(doc_ids)
            
            # Property: all documents should be retrieved correctly
            assert len(retrieved) == len(docs), \
                f"Expected {len(docs)} documents, got {len(retrieved)}"
            
            for doc_id, original_emb in docs:
                assert doc_id in retrieved, f"Document {doc_id} not found"
                
                retrieved_emb = retrieved[doc_id]
                assert torch.allclose(retrieved_emb, original_emb, atol=1e-6), \
                    f"Embeddings mismatch for document {doc_id}"
        finally:
            store.close()
    
    @given(data=embedding_with_doc_id_strategy())
    @settings(max_examples=100, deadline=None)
    def test_metadata_preserved(self, data: Tuple[str, torch.Tensor]):
        """
        Property: Metadata is preserved through storage.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        doc_id, embeddings = data
        
        store_path = os.path.join(self.temp_dir, f"store_meta_{hash(doc_id) % 10000}")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # Store with metadata
            metadata = {"source": "test", "page": 1}
            store.add_embeddings(doc_id, embeddings, metadata=metadata)
            
            # Retrieve metadata
            retrieved_meta = store.get_metadata(doc_id)
            
            # Property: metadata should be preserved
            assert retrieved_meta is not None, "Metadata not found"
            assert retrieved_meta["num_tokens"] == embeddings.shape[0], \
                "num_tokens mismatch"
            assert "user_metadata" in retrieved_meta, "user_metadata not found"
            assert retrieved_meta["user_metadata"]["source"] == "test", \
                "source metadata mismatch"
            assert retrieved_meta["user_metadata"]["page"] == 1, \
                "page metadata mismatch"
        finally:
            store.close()
    
    @given(data=embedding_with_doc_id_strategy())
    @settings(max_examples=100, deadline=None)
    def test_contains_after_add(self, data: Tuple[str, torch.Tensor]):
        """
        Property: Document is contained in store after adding.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        import uuid
        doc_id, embeddings = data
        
        # Use UUID to ensure unique store path for each test run
        store_path = os.path.join(self.temp_dir, f"store_contains_{uuid.uuid4().hex}")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # Before adding
            assert not store.contains(doc_id), \
                "Document should not exist before adding"
            
            # Add embeddings
            store.add_embeddings(doc_id, embeddings)
            
            # After adding
            assert store.contains(doc_id), \
                "Document should exist after adding"
        finally:
            store.close()
    
    @given(data=embedding_with_doc_id_strategy())
    @settings(max_examples=100, deadline=None)
    def test_delete_removes_document(self, data: Tuple[str, torch.Tensor]):
        """
        Property: Deleted documents are no longer retrievable.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        doc_id, embeddings = data
        
        store_path = os.path.join(self.temp_dir, f"store_delete_{hash(doc_id) % 10000}")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # Add and verify
            store.add_embeddings(doc_id, embeddings)
            assert store.contains(doc_id), "Document should exist after adding"
            
            # Delete
            deleted = store.delete_embeddings(doc_id)
            assert deleted, "Delete should return True"
            
            # Verify deleted
            assert not store.contains(doc_id), \
                "Document should not exist after deletion"
            
            retrieved = store.get_embeddings([doc_id])
            assert doc_id not in retrieved, \
                "Deleted document should not be retrievable"
        finally:
            store.close()


class TestProperty10EdgeCases:
    """
    Edge case tests for Embedding Store.
    
    **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
    **Validates: Requirements 8.4**
    """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Create and cleanup temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_single_token_embedding(self):
        """
        Test: Single token embedding round-trip.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_single")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            embeddings = torch.randn(1, 128)
            store.add_embeddings("single_token", embeddings)
            
            retrieved = store.get_embeddings(["single_token"])
            assert torch.allclose(retrieved["single_token"], embeddings, atol=1e-6)
        finally:
            store.close()
    
    def test_large_embedding(self):
        """
        Test: Large embedding (1000 tokens) round-trip.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_large")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            embeddings = torch.randn(1000, 128)
            store.add_embeddings("large_doc", embeddings)
            
            retrieved = store.get_embeddings(["large_doc"])
            assert torch.allclose(retrieved["large_doc"], embeddings, atol=1e-6)
        finally:
            store.close()
    
    def test_special_characters_in_doc_id(self):
        """
        Test: Document IDs with special characters.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_special")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            special_ids = [
                "doc-with-dashes",
                "doc_with_underscores",
                "doc123numbers",
                "MixedCaseDoc",
            ]
            
            for doc_id in special_ids:
                embeddings = torch.randn(10, 128)
                store.add_embeddings(doc_id, embeddings)
                
                retrieved = store.get_embeddings([doc_id])
                assert doc_id in retrieved, f"Failed for doc_id: {doc_id}"
                assert torch.allclose(retrieved[doc_id], embeddings, atol=1e-6)
        finally:
            store.close()
    
    def test_retrieve_nonexistent_document(self):
        """
        Test: Retrieving non-existent document returns empty dict.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_nonexistent")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            retrieved = store.get_embeddings(["nonexistent_doc"])
            assert "nonexistent_doc" not in retrieved
        finally:
            store.close()
    
    def test_partial_retrieval(self):
        """
        Test: Retrieving mix of existing and non-existing documents.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_partial")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # Add some documents
            emb1 = torch.randn(10, 128)
            emb2 = torch.randn(20, 128)
            store.add_embeddings("doc1", emb1)
            store.add_embeddings("doc2", emb2)
            
            # Retrieve mix of existing and non-existing
            retrieved = store.get_embeddings(["doc1", "nonexistent", "doc2"])
            
            assert "doc1" in retrieved
            assert "doc2" in retrieved
            assert "nonexistent" not in retrieved
            assert torch.allclose(retrieved["doc1"], emb1, atol=1e-6)
            assert torch.allclose(retrieved["doc2"], emb2, atol=1e-6)
        finally:
            store.close()
    
    def test_overwrite_existing_document(self):
        """
        Test: Overwriting existing document replaces embeddings.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_overwrite")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # Add initial embeddings
            emb1 = torch.randn(10, 128)
            store.add_embeddings("doc", emb1)
            
            # Overwrite with new embeddings
            emb2 = torch.randn(20, 128)
            store.add_embeddings("doc", emb2)
            
            # Retrieve should return new embeddings
            retrieved = store.get_embeddings(["doc"])
            assert retrieved["doc"].shape == emb2.shape
            assert torch.allclose(retrieved["doc"], emb2, atol=1e-6)
        finally:
            store.close()
    
    def test_list_doc_ids(self):
        """
        Test: list_doc_ids returns all stored document IDs.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_list")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # Add documents
            doc_ids = ["doc1", "doc2", "doc3"]
            for doc_id in doc_ids:
                store.add_embeddings(doc_id, torch.randn(10, 128))
            
            # List should contain all doc_ids
            listed = store.list_doc_ids()
            assert set(listed) == set(doc_ids)
        finally:
            store.close()
    
    def test_store_length(self):
        """
        Test: len(store) returns correct document count.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_len")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            assert len(store) == 0
            
            store.add_embeddings("doc1", torch.randn(10, 128))
            assert len(store) == 1
            
            store.add_embeddings("doc2", torch.randn(10, 128))
            assert len(store) == 2
            
            store.delete_embeddings("doc1")
            assert len(store) == 1
        finally:
            store.close()
    
    def test_context_manager(self):
        """
        Test: Store works as context manager.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_context")
        
        with EmbeddingStore(store_path, dim=128) as store:
            embeddings = torch.randn(10, 128)
            store.add_embeddings("doc", embeddings)
            
            retrieved = store.get_embeddings(["doc"])
            assert torch.allclose(retrieved["doc"], embeddings, atol=1e-6)
    
    def test_get_stats(self):
        """
        Test: get_stats returns correct statistics.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_stats")
        store = EmbeddingStore(store_path, dim=128)
        quantizer = BinaryQuantizer()
        
        try:
            # Add documents with varying token counts
            emb1 = torch.randn(10, 128)
            emb2 = torch.randn(20, 128)
            
            store.add_embeddings("doc1", emb1, quantizer.quantize(emb1))
            store.add_embeddings("doc2", emb2)  # No binary
            
            stats = store.get_stats()
            
            assert stats["num_documents"] == 2
            assert stats["total_tokens"] == 30
            assert stats["avg_tokens_per_doc"] == 15.0
            assert stats["num_with_binary"] == 1
        finally:
            store.close()
    
    def test_invalid_dimension_raises_error(self):
        """
        Test: Invalid embedding dimension raises ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_invalid")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # Wrong dimension
            embeddings = torch.randn(10, 64)  # dim=64, but store expects 128
            
            with pytest.raises(ValueError, match="Expected dim=128"):
                store.add_embeddings("doc", embeddings)
        finally:
            store.close()
    
    def test_non_2d_embedding_raises_error(self):
        """
        Test: Non-2D embedding raises ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 10: Embedding Caching Round-Trip**
        **Validates: Requirements 8.4**
        """
        store_path = os.path.join(self.temp_dir, "store_3d")
        store = EmbeddingStore(store_path, dim=128)
        
        try:
            # 3D tensor
            embeddings = torch.randn(2, 10, 128)
            
            with pytest.raises(ValueError, match="Expected 2D"):
                store.add_embeddings("doc", embeddings)
        finally:
            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
