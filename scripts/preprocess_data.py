#!/usr/bin/env python
"""
Preprocess and cache training data for faster training.

This script pre-processes all images and saves the results to disk,
eliminating the CPU bottleneck during training.

Usage:
    python -m qwen3_vl_retrieval.scripts.preprocess_data \
        --data_path /root/autodl-tmp/data/vidore/train_split.json \
        --image_dir /root/autodl-tmp/data/vidore \
        --output_dir /root/autodl-tmp/data/vidore_preprocessed \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --max_pixels 262144
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
import pickle

import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_pixels", type=int, default=262144)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def load_samples(data_path: str, image_dir: str) -> List[Dict[str, Any]]:
    """Load samples from JSON file."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    for sample in data:
        image_path = sample.get("image_path", sample.get("image", ""))
        if not Path(image_path).is_absolute():
            image_path = str(Path(image_dir) / image_path)
        
        samples.append({
            "query": sample.get("query", sample.get("question", "")),
            "image_path": image_path,
            "doc_id": sample.get("doc_id", sample.get("id", str(hash(image_path)))),
        })
    
    return samples


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processor
    logger.info(f"Loading processor from {args.model_path}")
    from qwen3_vl_retrieval.models.processing_colqwen3vl import ColQwen3VLProcessor
    
    processor = ColQwen3VLProcessor.from_pretrained(
        str(Path(args.model_path).expanduser()),
        max_pixels=args.max_pixels,
    )
    
    # Load samples
    logger.info(f"Loading samples from {args.data_path}")
    samples = load_samples(args.data_path, args.image_dir)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Process and save each sample
    processed_samples = []
    
    for i, sample in enumerate(tqdm(samples, desc="Preprocessing")):
        try:
            # Load image
            image = Image.open(sample["image_path"]).convert("RGB")
            
            # Process image
            doc_batch = processor.process_images([image])
            
            # Process query
            query_batch = processor.process_queries(texts=[sample["query"]])
            
            # Save tensors
            sample_dir = output_dir / f"sample_{i:06d}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save doc tensors
            torch.save(doc_batch["input_ids"], sample_dir / "doc_input_ids.pt")
            torch.save(doc_batch["attention_mask"], sample_dir / "doc_attention_mask.pt")
            torch.save(doc_batch["pixel_values"], sample_dir / "doc_pixel_values.pt")
            torch.save(doc_batch["image_grid_thw"], sample_dir / "doc_image_grid_thw.pt")
            
            # Save query tensors
            torch.save(query_batch["input_ids"], sample_dir / "query_input_ids.pt")
            torch.save(query_batch["attention_mask"], sample_dir / "query_attention_mask.pt")
            
            processed_samples.append({
                "idx": i,
                "query": sample["query"],
                "doc_id": sample["doc_id"],
                "sample_dir": str(sample_dir),
            })
            
        except Exception as e:
            logger.warning(f"Failed to process sample {i}: {e}")
            continue
    
    # Save index
    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, indent=2)
    
    logger.info(f"Preprocessed {len(processed_samples)} samples")
    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
