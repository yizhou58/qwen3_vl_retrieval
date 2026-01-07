#!/usr/bin/env python
"""
Rebuild JSON from downloaded images.

If download was interrupted, this script rebuilds the JSON file
from the images that were already downloaded.

Usage:
    python -m qwen3_vl_retrieval.scripts.rebuild_json_from_images \
        --image_dir /root/autodl-tmp/data/vidore/images \
        --output_dir /root/autodl-tmp/data/vidore
"""

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def rebuild_json(image_dir: str, output_dir: str):
    """Rebuild JSON from downloaded images by re-fetching queries."""
    from datasets import load_dataset
    
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    
    # Get list of downloaded images
    images = sorted(image_path.glob("doc_*.png"))
    logger.info(f"Found {len(images)} downloaded images")
    
    if not images:
        logger.error("No images found!")
        return
    
    # Extract indices from filenames
    indices = []
    for img in images:
        # doc_000123.png -> 123
        idx = int(img.stem.split("_")[1])
        indices.append(idx)
    
    max_idx = max(indices)
    logger.info(f"Image indices range: 0 to {max_idx}")
    
    # Load dataset to get queries
    logger.info("Loading dataset to fetch queries...")
    dataset = load_dataset("vidore/colpali_train_set", split="train", streaming=True)
    
    samples = []
    idx_set = set(indices)
    
    for idx, item in enumerate(dataset):
        if idx > max_idx:
            break
        
        if idx not in idx_set:
            continue
        
        if idx % 500 == 0:
            logger.info(f"Processing index {idx}...")
        
        query = item.get("query") or item.get("question") or item.get("text")
        if query is None:
            continue
        
        image_filename = f"doc_{idx:06d}.png"
        
        samples.append({
            "query": query,
            "image_path": f"images/{image_filename}",
            "doc_id": f"doc_{idx:06d}",
        })
    
    logger.info(f"Rebuilt {len(samples)} samples")
    
    # Save JSON
    train_file = output_path / "train.json"
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {train_file}")
    
    # Create splits
    val_size = max(1, len(samples) // 10)
    train_samples = samples[:-val_size]
    val_samples = samples[-val_size:]
    
    with open(output_path / "train_split.json", "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    with open(output_path / "val_split.json", "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/root/autodl-tmp/data/vidore/images")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/data/vidore")
    args = parser.parse_args()
    
    rebuild_json(args.image_dir, args.output_dir)


if __name__ == "__main__":
    main()
