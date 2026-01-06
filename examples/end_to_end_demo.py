#!/usr/bin/env python
"""
End-to-End Demo for Qwen3-VL RAG Retrieval System.

This script demonstrates the complete workflow:
1. Load the ColQwen3VL model
2. Index sample document images
3. Perform two-stage retrieval
4. Display results

This is a self-contained demo that can be run with minimal setup.

Requirements: 8.3, 8.4
- Provide APIs for encoding documents and queries
- Support pre-computing and caching document embeddings

Usage:
    # Run with sample images
    python -m qwen3_vl_retrieval.examples.end_to_end_demo \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --image_dir ./sample_documents

    # Run with custom queries
    python -m qwen3_vl_retrieval.examples.end_to_end_demo \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --image_dir ./sample_documents \
        --queries "What is shown in the document?" "Find financial data"
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-end demo for Qwen3-VL RAG retrieval"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to base Qwen3-VL model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter weights (optional)"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing document images"
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=["What information is shown in this document?"],
        help="Queries to test"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or auto)"
    )
    
    return parser.parse_args()


def discover_images(image_dir: str) -> List[Tuple[str, str]]:
    """Discover all images in a directory."""
    image_dir = Path(image_dir)
    extensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
    images = []
    
    for ext in extensions:
        for path in image_dir.rglob(f"*{ext}"):
            doc_id = path.stem
            images.append((doc_id, str(path)))
    
    images.sort(key=lambda x: x[0])
    return images


def create_sample_texts(doc_ids: List[str]) -> List[str]:
    """Create placeholder texts for documents."""
    return [f"Document content for {doc_id}" for doc_id in doc_ids]


def main():
    """Run the end-to-end demo."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("  Qwen3-VL RAG Retrieval System - End-to-End Demo")
    print("="*70 + "\n")
    
    # Validate paths
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        sys.exit(1)
    
    # Discover images
    print("Step 1: Discovering document images...")
    images = discover_images(str(image_dir))
    
    if not images:
        logger.error(f"No images found in {image_dir}")
        sys.exit(1)
    
    print(f"  Found {len(images)} images")
    for doc_id, path in images[:5]:
        print(f"    - {doc_id}: {path}")
    if len(images) > 5:
        print(f"    ... and {len(images) - 5} more")
    print()
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Step 2: Loading model (device: {device})...")
    
    from qwen3_vl_retrieval.inference.model_loader import load_model_for_inference
    from qwen3_vl_retrieval.retrieval.embedding_store import EmbeddingStore
    from qwen3_vl_retrieval.retrieval.first_stage_retriever import FirstStageRetriever
    from qwen3_vl_retrieval.retrieval.second_stage_reranker import SecondStageReranker
    
    model, processor = load_model_for_inference(
        model_name_or_path=str(Path(args.model_path).expanduser()),
        lora_path=args.lora_path,
        device_map=device if device != "cpu" else None,
    )
    print("  Model loaded successfully\n")
    
    # Create temporary directory for index
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Prepare document data
        doc_ids = [doc_id for doc_id, _ in images]
        image_paths = [path for _, path in images]
        texts = create_sample_texts(doc_ids)
        
        # Initialize components
        print("Step 3: Indexing documents...")
        
        # First-stage retriever
        first_stage = FirstStageRetriever(
            method="bm25",
            index_path=str(temp_path / "first_stage"),
        )
        first_stage.index_documents(doc_ids, texts, image_paths)
        print(f"  First-stage index: {len(first_stage)} documents")
        
        # Embedding store
        embedding_store = EmbeddingStore(
            storage_path=str(temp_path / "embeddings"),
            dim=128,
        )
        
        # Encode documents
        print("  Encoding document images...")
        embedding_store.batch_encode_documents(
            model=model,
            processor=processor,
            image_paths=image_paths,
            doc_ids=doc_ids,
            batch_size=2,
            quantize=True,
            device=device,
            show_progress=True,
        )
        
        stats = embedding_store.get_stats()
        print(f"  Embeddings stored: {stats['num_documents']} documents, "
              f"{stats['total_tokens']} tokens")
        print()
        
        # Create second-stage reranker
        second_stage = SecondStageReranker(
            model=model,
            processor=processor,
            embedding_store=embedding_store,
            use_binary_quantization=True,
            device=device,
        )
        
        # Run queries
        print("Step 4: Running retrieval queries...")
        print("-" * 70)
        
        for query in args.queries:
            print(f"\nQuery: \"{query}\"")
            print("-" * 50)
            
            # Stage 1: BM25 recall
            first_stage_results = first_stage.retrieve(query, top_k=min(50, len(doc_ids)))
            candidate_doc_ids = [doc_id for doc_id, _, _ in first_stage_results]
            
            print(f"  Stage 1 (BM25): {len(candidate_doc_ids)} candidates")
            
            # Stage 2: MaxSim reranking
            if candidate_doc_ids:
                results = second_stage.rerank(
                    query=query,
                    candidate_doc_ids=candidate_doc_ids,
                    top_k=args.top_k,
                    binary_rescore_ratio=5,
                )
                
                print(f"  Stage 2 (MaxSim): Top {len(results)} results")
                print()
                
                for rank, (doc_id, score) in enumerate(results, 1):
                    image_path = first_stage.image_paths.get(doc_id, "N/A")
                    print(f"    {rank}. [{score:.4f}] {doc_id}")
                    print(f"       Image: {Path(image_path).name}")
            else:
                print("  No candidates found")
            
            print()
        
        print("-" * 70)
        print("\nDemo complete!")
        print("\nTo use this system in production:")
        print("  1. Run index_documents.py to create a persistent index")
        print("  2. Run query_retrieval.py to query the index")
        print("  3. Or use the ColQwen3VLRetriever API directly in your code")
        print()


if __name__ == "__main__":
    main()
