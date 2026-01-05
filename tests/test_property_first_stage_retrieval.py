"""
Property-Based Tests for First Stage Retrieval Correctness.

**Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
**Validates: Requirements 5.3, 5.4, 5.5, 5.6**

Property 7: For any indexed document collection and query, the First_Stage_Retriever 
SHALL return exactly top_k candidates (or all documents if fewer exist). Each returned 
candidate SHALL have a valid mapping to its corresponding image path. Adding a new 
document SHALL make it retrievable in subsequent queries.
"""

import os
import tempfile
import pytest
from hypothesis import given, settings, strategies as st, assume
from typing import List, Tuple

from qwen3_vl_retrieval.retrieval.first_stage_retriever import FirstStageRetriever


# Hypothesis strategies
@st.composite
def document_strategy(draw):
    """Generate a random document with id, text, and image path."""
    doc_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_-'),
        min_size=1,
        max_size=20
    ))
    # Ensure doc_id is not empty after filtering
    assume(len(doc_id.strip()) > 0)
    doc_id = doc_id.strip()
    
    text = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
        min_size=1,
        max_size=200
    ))
    # Ensure text has at least some alphanumeric content
    assume(any(c.isalnum() for c in text))
    
    image_path = f"/images/{doc_id}.png"
    
    return doc_id, text, image_path


@st.composite
def document_collection_strategy(draw):
    """Generate a collection of unique documents."""
    num_docs = draw(st.integers(min_value=1, max_value=50))
    
    doc_ids = []
    texts = []
    image_paths = []
    
    used_ids = set()
    for _ in range(num_docs):
        doc_id, text, image_path = draw(document_strategy())
        # Ensure unique doc_ids
        if doc_id in used_ids:
            doc_id = f"{doc_id}_{len(used_ids)}"
        used_ids.add(doc_id)
        
        doc_ids.append(doc_id)
        texts.append(text)
        image_paths.append(image_path)
    
    return doc_ids, texts, image_paths


@st.composite
def query_strategy(draw):
    """Generate a random query string."""
    query = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'Z')),
        min_size=1,
        max_size=100
    ))
    # Ensure query has at least some alphanumeric content
    assume(any(c.isalnum() for c in query))
    return query


class TestProperty7FirstStageRetrievalCorrectness:
    """
    Property 7: First Stage Retrieval Correctness
    
    **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
    **Validates: Requirements 5.3, 5.4, 5.5, 5.6**
    
    For any indexed document collection and query, the First_Stage_Retriever SHALL 
    return exactly top_k candidates (or all documents if fewer exist).
    """
    
    @given(
        collection=document_collection_strategy(),
        query=query_strategy(),
        top_k=st.integers(min_value=1, max_value=200)
    )
    @settings(max_examples=100, deadline=None)
    def test_returns_exactly_top_k_or_all(
        self,
        collection: Tuple[List[str], List[str], List[str]],
        query: str,
        top_k: int
    ):
        """
        Property: Retriever returns exactly top_k candidates or all documents if fewer exist.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.3**
        """
        doc_ids, texts, image_paths = collection
        
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(doc_ids, texts, image_paths)
        
        results = retriever.retrieve(query, top_k=top_k)
        
        expected_count = min(top_k, len(doc_ids))
        assert len(results) == expected_count, \
            f"Expected {expected_count} results, got {len(results)}"
    
    @given(
        collection=document_collection_strategy(),
        query=query_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_image_path_mapping(
        self,
        collection: Tuple[List[str], List[str], List[str]],
        query: str
    ):
        """
        Property: Each returned candidate has a valid mapping to its image path.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4, 5.6**
        """
        doc_ids, texts, image_paths = collection
        
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(doc_ids, texts, image_paths)
        
        results = retriever.retrieve(query, top_k=len(doc_ids))
        
        # Create expected mapping
        expected_mapping = dict(zip(doc_ids, image_paths))
        
        for doc_id, score, image_path in results:
            # Property: doc_id should be in our indexed documents
            assert doc_id in expected_mapping, \
                f"Returned doc_id '{doc_id}' not in indexed documents"
            
            # Property: image_path should match the indexed mapping
            assert image_path == expected_mapping[doc_id], \
                f"Image path mismatch for {doc_id}: expected {expected_mapping[doc_id]}, got {image_path}"
    
    @given(
        collection=document_collection_strategy(),
        new_doc=document_strategy(),
        query=query_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_incremental_update_makes_document_retrievable(
        self,
        collection: Tuple[List[str], List[str], List[str]],
        new_doc: Tuple[str, str, str],
        query: str
    ):
        """
        Property: Adding a new document makes it retrievable in subsequent queries.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.5**
        """
        doc_ids, texts, image_paths = collection
        new_doc_id, new_text, new_image_path = new_doc
        
        # Ensure new doc_id is unique
        if new_doc_id in doc_ids:
            new_doc_id = f"{new_doc_id}_new"
        
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(doc_ids, texts, image_paths)
        
        # Add new document
        retriever.add_document(new_doc_id, new_text, new_image_path)
        
        # Retrieve all documents
        results = retriever.retrieve(query, top_k=len(doc_ids) + 1)
        
        # Property: new document should be in results
        result_doc_ids = [r[0] for r in results]
        assert new_doc_id in result_doc_ids, \
            f"Newly added document '{new_doc_id}' not found in results"
        
        # Property: new document should have correct image path
        for doc_id, score, image_path in results:
            if doc_id == new_doc_id:
                assert image_path == new_image_path, \
                    f"New document image path mismatch: expected {new_image_path}, got {image_path}"
                break

    
    @given(collection=document_collection_strategy())
    @settings(max_examples=100, deadline=None)
    def test_all_returned_doc_ids_are_unique(
        self,
        collection: Tuple[List[str], List[str], List[str]]
    ):
        """
        Property: All returned document IDs are unique.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.3**
        """
        doc_ids, texts, image_paths = collection
        
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(doc_ids, texts, image_paths)
        
        # Use a query that should match something
        query = texts[0] if texts else "test"
        results = retriever.retrieve(query, top_k=len(doc_ids))
        
        result_doc_ids = [r[0] for r in results]
        assert len(result_doc_ids) == len(set(result_doc_ids)), \
            "Returned document IDs should be unique"
    
    @given(collection=document_collection_strategy())
    @settings(max_examples=100, deadline=None)
    def test_results_sorted_by_score_descending(
        self,
        collection: Tuple[List[str], List[str], List[str]]
    ):
        """
        Property: Results are sorted by score in descending order.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.3**
        """
        doc_ids, texts, image_paths = collection
        
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(doc_ids, texts, image_paths)
        
        query = texts[0] if texts else "test"
        results = retriever.retrieve(query, top_k=len(doc_ids))
        
        scores = [r[1] for r in results]
        
        # Property: scores should be in descending order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Scores not in descending order at index {i}: {scores[i]} < {scores[i+1]}"
    
    @given(collection=document_collection_strategy())
    @settings(max_examples=100, deadline=None)
    def test_document_count_matches_indexed(
        self,
        collection: Tuple[List[str], List[str], List[str]]
    ):
        """
        Property: Document count matches the number of indexed documents.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4**
        """
        doc_ids, texts, image_paths = collection
        
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(doc_ids, texts, image_paths)
        
        assert retriever.get_document_count() == len(doc_ids), \
            f"Document count mismatch: expected {len(doc_ids)}, got {retriever.get_document_count()}"
        
        assert len(retriever) == len(doc_ids), \
            f"len() mismatch: expected {len(doc_ids)}, got {len(retriever)}"


class TestProperty7EdgeCases:
    """
    Edge case tests for First Stage Retrieval.
    
    **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
    **Validates: Requirements 5.3, 5.4, 5.5, 5.6**
    """
    
    def test_empty_index_returns_empty_results(self):
        """
        Test: Retrieval on empty index returns empty results.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.3**
        """
        retriever = FirstStageRetriever(method="bm25")
        
        results = retriever.retrieve("test query", top_k=10)
        
        assert results == [], \
            "Empty index should return empty results"
    
    def test_invalid_top_k_raises_error(self):
        """
        Test: Invalid top_k (<=0) raises ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.3**
        """
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(["doc1"], ["test text"], ["/images/doc1.png"])
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("test", top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("test", top_k=-1)
    
    def test_mismatched_input_lengths_raises_error(self):
        """
        Test: Mismatched input list lengths raise ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4**
        """
        retriever = FirstStageRetriever(method="bm25")
        
        with pytest.raises(ValueError, match="same length"):
            retriever.index_documents(
                ["doc1", "doc2"],
                ["text1"],  # Mismatched length
                ["/images/doc1.png", "/images/doc2.png"]
            )
    
    def test_duplicate_doc_ids_in_batch_raises_error(self):
        """
        Test: Duplicate doc_ids in a single batch raise ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4**
        """
        retriever = FirstStageRetriever(method="bm25")
        
        with pytest.raises(ValueError, match="Duplicate doc_ids"):
            retriever.index_documents(
                ["doc1", "doc1"],  # Duplicate
                ["text1", "text2"],
                ["/images/doc1.png", "/images/doc2.png"]
            )
    
    def test_get_image_path_for_missing_doc_raises_error(self):
        """
        Test: Getting image path for non-existent doc_id raises KeyError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.6**
        """
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(["doc1"], ["test text"], ["/images/doc1.png"])
        
        with pytest.raises(KeyError, match="not found"):
            retriever.get_image_path("nonexistent_doc")
    
    def test_contains_operator(self):
        """
        Test: __contains__ operator works correctly.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4**
        """
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(["doc1", "doc2"], ["text1", "text2"], ["/img1.png", "/img2.png"])
        
        assert "doc1" in retriever
        assert "doc2" in retriever
        assert "doc3" not in retriever
    
    def test_clear_removes_all_documents(self):
        """
        Test: clear() removes all indexed documents.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4**
        """
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(["doc1", "doc2"], ["text1", "text2"], ["/img1.png", "/img2.png"])
        
        assert len(retriever) == 2
        
        retriever.clear()
        
        assert len(retriever) == 0
        assert retriever.retrieve("test", top_k=10) == []
    
    def test_update_existing_document(self):
        """
        Test: Re-indexing a document updates its content.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.5**
        """
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(["doc1"], ["original text"], ["/images/original.png"])
        
        # Update the document
        retriever.index_documents(["doc1"], ["updated text"], ["/images/updated.png"])
        
        # Should still have only 1 document
        assert len(retriever) == 1
        
        # Image path should be updated
        assert retriever.get_image_path("doc1") == "/images/updated.png"
    
    def test_single_document_retrieval(self):
        """
        Test: Single document collection works correctly.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.3**
        """
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(["doc1"], ["hello world"], ["/images/doc1.png"])
        
        results = retriever.retrieve("hello", top_k=10)
        
        assert len(results) == 1
        assert results[0][0] == "doc1"
        assert results[0][2] == "/images/doc1.png"


class TestProperty7Persistence:
    """
    Tests for index persistence (save/load).
    
    **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
    **Validates: Requirements 5.4, 5.5**
    """
    
    def test_save_and_load_bm25_index(self):
        """
        Test: BM25 index can be saved and loaded correctly.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save index
            retriever = FirstStageRetriever(method="bm25")
            retriever.index_documents(
                ["doc1", "doc2", "doc3"],
                ["apple banana cherry", "dog elephant fox", "grape honey ice"],
                ["/img1.png", "/img2.png", "/img3.png"]
            )
            
            retriever.save_index(tmpdir)
            
            # Load into new retriever
            loaded_retriever = FirstStageRetriever(method="bm25")
            loaded_retriever.load_index(tmpdir)
            
            # Verify document count
            assert len(loaded_retriever) == 3
            
            # Verify image paths
            assert loaded_retriever.get_image_path("doc1") == "/img1.png"
            assert loaded_retriever.get_image_path("doc2") == "/img2.png"
            assert loaded_retriever.get_image_path("doc3") == "/img3.png"
            
            # Verify retrieval works
            results = loaded_retriever.retrieve("apple", top_k=3)
            assert len(results) == 3
            # First result should be doc1 (contains "apple")
            assert results[0][0] == "doc1"
    
    @given(collection=document_collection_strategy())
    @settings(max_examples=50, deadline=None)
    def test_save_load_preserves_all_data(
        self,
        collection: Tuple[List[str], List[str], List[str]]
    ):
        """
        Property: Save/load preserves all document data.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4**
        """
        doc_ids, texts, image_paths = collection
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            retriever = FirstStageRetriever(method="bm25")
            retriever.index_documents(doc_ids, texts, image_paths)
            retriever.save_index(tmpdir)
            
            # Load
            loaded = FirstStageRetriever(method="bm25")
            loaded.load_index(tmpdir)
            
            # Verify all data preserved
            assert len(loaded) == len(doc_ids)
            
            for doc_id, image_path in zip(doc_ids, image_paths):
                assert doc_id in loaded
                assert loaded.get_image_path(doc_id) == image_path
    
    def test_load_nonexistent_path_raises_error(self):
        """
        Test: Loading from non-existent path raises FileNotFoundError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4**
        """
        retriever = FirstStageRetriever(method="bm25")
        
        with pytest.raises(FileNotFoundError):
            retriever.load_index("/nonexistent/path")
    
    def test_auto_load_on_init(self):
        """
        Test: Index is automatically loaded if path exists on init.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.4**
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            retriever = FirstStageRetriever(method="bm25")
            retriever.index_documents(["doc1"], ["test text"], ["/img.png"])
            retriever.save_index(tmpdir)
            
            # Create new retriever with index_path - should auto-load
            loaded = FirstStageRetriever(method="bm25", index_path=tmpdir)
            
            assert len(loaded) == 1
            assert "doc1" in loaded


class TestProperty7BM25Specific:
    """
    BM25-specific tests for First Stage Retrieval.
    
    **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
    **Validates: Requirements 5.1**
    """
    
    def test_bm25_keyword_matching(self):
        """
        Test: BM25 correctly ranks documents by keyword relevance.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.1**
        """
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(
            ["doc1", "doc2", "doc3"],
            [
                "machine learning deep neural networks",
                "cooking recipes food kitchen",
                "machine learning algorithms data science"
            ],
            ["/img1.png", "/img2.png", "/img3.png"]
        )
        
        results = retriever.retrieve("machine learning", top_k=3)
        
        # doc1 and doc3 should rank higher than doc2
        result_ids = [r[0] for r in results]
        
        # doc2 (cooking) should be last
        assert result_ids[-1] == "doc2", \
            "Irrelevant document should rank last"
    
    def test_bm25_case_insensitive(self):
        """
        Test: BM25 search is case-insensitive.
        
        **Feature: qwen3-vl-rag-retrieval, Property 7: First Stage Retrieval Correctness**
        **Validates: Requirements 5.1**
        """
        retriever = FirstStageRetriever(method="bm25")
        retriever.index_documents(
            ["doc1"],
            ["Machine Learning Deep Neural Networks"],
            ["/img1.png"]
        )
        
        # Query with different case
        results_lower = retriever.retrieve("machine learning", top_k=1)
        results_upper = retriever.retrieve("MACHINE LEARNING", top_k=1)
        results_mixed = retriever.retrieve("Machine Learning", top_k=1)
        
        # All should return the same document with same score
        assert results_lower[0][0] == "doc1"
        assert results_upper[0][0] == "doc1"
        assert results_mixed[0][0] == "doc1"
        
        # Scores should be equal
        assert results_lower[0][1] == results_upper[0][1] == results_mixed[0][1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
