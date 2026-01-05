#!/usr/bin/env python
"""
Download and prepare ViDoRe training dataset.

Downloads the ColPali training set from HuggingFace and converts it
to the format expected by ColPaliEngineDataset.

Usage:
    python -m qwen3_vl_retrieval.scripts.download_vidore --output_dir /root/autodl-tmp/data
"""

import argparse
import json
import logging
import os
from pathlib import Path
from io import BytesIO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_vidore_dataset(output_dir: str, max_samples: int = None):
    """
    Download ViDoRe training dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset
        max_samples: Maximum number of samples to download (None for all)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    logger.info("Downloading ViDoRe training dataset from HuggingFace...")
    logger.info("This may take a while depending on your internet connection...")
    logger.info("If you encounter rate limits, please login: huggingface-cli login")
    
    # Load the ColPali training dataset
    # Try multiple datasets in order of preference
    dataset = None
    datasets_to_try = [
        ("vidore/colpali_train_set", "train"),
        ("vidore/docvqa_test_subsampled", "test"),
        ("lmms-lab/DocVQA", "validation"),
    ]
    
    for dataset_name, split in datasets_to_try:
        try:
            logger.info(f"Trying to load {dataset_name} with streaming...")
            # Use streaming to avoid downloading entire dataset
            dataset = load_dataset(dataset_name, split=split, streaming=True)
            logger.info(f"Successfully loaded {dataset_name} (streaming)")
            break
        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}: {e}")
            continue
    
    if dataset is None:
        raise RuntimeError(
            "Failed to load any dataset. Please:\n"
            "1. Login to HuggingFace: huggingface-cli login\n"
            "2. Or set HF_TOKEN environment variable\n"
            "3. Or wait a few minutes and retry (rate limit)"
        )
    
    logger.info(f"Dataset loaded (streaming mode)")
    
    # Convert to our format - iterate through streaming dataset
    samples = []
    
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
            
        if idx % 50 == 0:
            logger.info(f"Processing sample {idx}...")
        
        # Get image
        image = item.get("image") or item.get("page_image")
        if image is None:
            logger.warning(f"Sample {idx} has no image, skipping")
            continue
        
        # Save image immediately to free memory
        image_filename = f"doc_{idx:06d}.png"
        image_path = images_dir / image_filename
        
        if not image_path.exists():
            # Convert to RGB and save with compression
            if hasattr(image, 'convert'):
                image = image.convert("RGB")
            image.save(image_path, optimize=True, quality=85)
        
        # Clear image from memory
        del image
        
        # Get query
        query = item.get("query") or item.get("question") or item.get("text")
        if query is None:
            logger.warning(f"Sample {idx} has no query, skipping")
            continue
        
        # Create sample entry (minimal data)
        sample = {
            "query": query,
            "image_path": f"images/{image_filename}",
            "doc_id": f"doc_{idx:06d}",
        }
        
        samples.append(sample)
        
        # Periodically save to avoid losing progress
        if idx > 0 and idx % 200 == 0:
            temp_file = output_path / "train_temp.json"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False)
            logger.info(f"Saved checkpoint at {idx} samples")
    
    # Save as JSON
    train_file = output_path / "train.json"
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(samples)} samples to {train_file}")
    
    # Create a small validation split (10%)
    val_size = max(1, len(samples) // 10)
    val_samples = samples[-val_size:]
    train_samples = samples[:-val_size]
    
    # Save train split
    train_split_file = output_path / "train_split.json"
    with open(train_split_file, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    # Save val split
    val_split_file = output_path / "val_split.json"
    with open(val_split_file, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Train split: {len(train_samples)} samples -> {train_split_file}")
    logger.info(f"Val split: {len(val_samples)} samples -> {val_split_file}")
    
    # Print dataset statistics
    print("\n" + "=" * 50)
    print("Dataset Download Complete!")
    print("=" * 50)
    print(f"Output directory: {output_path}")
    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Images saved to: {images_dir}")
    print("\nFiles created:")
    print(f"  - {train_file}")
    print(f"  - {train_split_file}")
    print(f"  - {val_split_file}")
    print("=" * 50)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download ViDoRe training dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/data/vidore",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to download (default: all)",
    )
    
    args = parser.parse_args()
    
    download_vidore_dataset(args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
