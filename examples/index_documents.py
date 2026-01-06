#!/usr/bin/env python
"""
Document Indexing Example for Qwen3-VL RAG Retrieval System.

This script demonstrates how to:
1. Load the ColQwen3VL model
2. Index document images with their text metadata
3. Store embeddings for later retrieval

Requirements: 8.3, 8.4
- Provide APIs for encoding documents
- Support pre-computing and caching document embeddings

Usage:
    # Basic usage with sample documents
    python -m qwen3_vl_retrieval.examples.index_documents \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --image_dir ./sample_documents \
        --output_dir ./index_output

    # With LoRA weights
    python -m qwen3_vl_retrieval.examples.index_documents \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --lora_path ./lora_weights \
        --image_dir ./sample_documents \
        --output_dir ./index_output

    # With OCR text file
    python -m qwen3_vl_retrieval.examples.index_documents \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --image_dir ./sample_documents \
        --text_file ./ocr_texts.json \
        --output_dir ./index_output
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Index documents for Qwen3-VL RAG retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Index all images in a directory
    python -m qwen3_vl_retrieval.examples.index_documents \\
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \\
        --image_dir ./documents \\
        --output_dir ./index

    # Index with custom OCR text
    python -m qwen3_vl_retrieval.examples.index_documents \\
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \\
        --image_dir ./documents \\
        --text_file ./ocr_results.json \\
        --output_dir ./index
        """
    )
    
    # Model arguments
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
    
    # Data arguments
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing document images"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="JSON file with OCR text for documents (optional)"
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
        help="Image file extensions to process"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save index and embeddings"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for encoding (default: 4)"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=262144,
        help="Maximum pixels per image (default: 262144)"
    )
    parser.add_argument(
        "--no_binary",
        action="store_true",
        help="Disable binary quantization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or auto)"
    )
    
    return parser.parse_args()


def discover_images(
    image_dir: str,
    extensions: List[str]
) -> List[Tuple[str, str]]:
    """
    Discover all images in a directory.
    
    Args:
        image_dir: Directory to search
        extensions: List of valid image extensions
        
    Returns:
        List of (doc_id, image_path) tuples
    """
    image_dir = Path(image_dir)
    images = []
    
    # Normalize extensions
    extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" 
                  for ext in extensions]
    
    # Find all images
    for ext in extensions:
        for path in image_dir.rglob(f"*{ext}"):
            # Use relative path as doc_id
            doc_id = str(path.relative_to(image_dir)).replace(os.sep, "_")
            doc_id = Path(doc_id).stem  # Remove extension
            images.append((doc_id, str(path)))
    
    # Sort for reproducibility
    images.sort(key=lambda x: x[0])
    
    return images


def load_text_metadata(text_file: str) -> Dict[str, str]:
    """
    Load OCR text metadata from JSON file.
    
    Expected format:
    {
        "doc_id_1": "OCR text for document 1",
        "doc_id_2": "OCR text for document 2",
        ...
    }
    
    Or:
    [
        {"doc_id": "doc_1", "text": "OCR text 1"},
        {"doc_id": "doc_2", "text": "OCR text 2"},
        ...
    ]
    
    Args:
        text_file: Path to JSON file
        
    Returns:
        Dict mapping doc_id to text
    """
    with open(text_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        return {item["doc_id"]: item.get("text", "") for item in data}
    else:
        raise ValueError(f"Unexpected JSON format in {text_file}")


def main():
    """Main indexing function."""
    args = parse_args()
    
    # Validate paths
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover images
    logger.info(f"Discovering images in {image_dir}...")
    images = discover_images(str(image_dir), args.image_extensions)
    
    if not images:
        logger.error(f"No images found in {image_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(images)} images")
    
    # Load text metadata if provided
    text_metadata = {}
    if args.text_file:
        logger.info(f"Loading text metadata from {args.text_file}...")
        text_metadata = load_text_metadata(args.text_file)
        logger.info(f"Loaded text for {len(text_metadata)} documents")
    
    # Prepare document lists
    doc_ids = [doc_id for doc_id, _ in images]
    image_paths = [path for _, path in images]
    texts = [text_metadata.get(doc_id, f"Document: {doc_id}") for doc_id in doc_ids]
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load model and processor
    logger.info(f"Loading model from {args.model_path}...")
    
    from qwen3_vl_retrieval.inference.model_loader import load_model_for_inference
    from qwen3_vl_retrieval.retrieval.embedding_store import EmbeddingStore
    from qwen3_vl_retrieval.retrieval.first_stage_retriever import FirstStageRetriever
    
    model, processor = load_model_for_inference(
        model_name_or_path=str(Path(args.model_path).expanduser()),
        lora_path=args.lora_path,
        max_num_visual_tokens=args.max_pixels // 196,  # Approximate token count
        device_map=device if device != "cpu" else None,
    )
    
    logger.info("Model loaded successfully")
    
    # Initialize embedding store
    embedding_store_path = output_dir / "embeddings"
    embedding_store = EmbeddingStore(
        storage_path=str(embedding_store_path),
        dim=128,
    )
    
    logger.info(f"Embedding store initialized at {embedding_store_path}")
    
    # Initialize first-stage retriever
    first_stage_path = output_dir / "first_stage_index"
    first_stage = FirstStageRetriever(
        method="bm25",
        index_path=str(first_stage_path),
    )
    
    # Index documents in first-stage retriever
    logger.info("Indexing documents in first-stage retriever...")
    first_stage.index_documents(doc_ids, texts, image_paths)
    first_stage.save_index()
    logger.info(f"First-stage index saved to {first_stage_path}")
    
    # Encode and store document embeddings
    logger.info("Encoding document images...")
    embedding_store.batch_encode_documents(
        model=model,
        processor=processor,
        image_paths=image_paths,
        doc_ids=doc_ids,
        batch_size=args.batch_size,
        quantize=not args.no_binary,
        device=device,
        show_progress=True,
    )
    
    # Get statistics
    stats = embedding_store.get_stats()
    logger.info(f"Indexing complete!")
    logger.info(f"  Documents indexed: {stats['num_documents']}")
    logger.info(f"  Total tokens: {stats['total_tokens']}")
    logger.info(f"  Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")
    logger.info(f"  Float storage: {stats['estimated_float_storage_bytes'] / 1024 / 1024:.1f} MB")
    if stats['num_with_binary'] > 0:
        logger.info(f"  Binary storage: {stats['estimated_binary_storage_bytes'] / 1024 / 1024:.1f} MB")
    
    # Save index metadata
    metadata = {
        "num_documents": len(doc_ids),
        "doc_ids": doc_ids,
        "image_paths": image_paths,
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "binary_quantization": not args.no_binary,
        "embedding_dim": 128,
    }
    
    metadata_path = output_dir / "index_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Index metadata saved to {metadata_path}")
    logger.info(f"\nIndex ready for retrieval at: {output_dir}")


if __name__ == "__main__":
    main()
