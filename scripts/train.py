#!/usr/bin/env python
"""
Training script for ColQwen3VL retrieval model.

Usage:
    python -m qwen3_vl_retrieval.scripts.train --config config.yaml
    
    # Or with command line arguments
    python -m qwen3_vl_retrieval.scripts.train \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --data_path ./data/train.json \
        --output_dir ./outputs \
        --batch_size 2 \
        --learning_rate 2e-5

Requirements: 2.6, 2.7
- Support configurable LoRA rank and alpha
- Save LoRA adapter weights separately from base model
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ColQwen3VL retrieval model")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/",
        help="Path to base Qwen3-VL model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (JSON/JSONL)",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="Path to evaluation data",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images",
    )
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # QLoRA (4-bit quantization) arguments
    parser.add_argument("--use_qlora", action="store_true", default=False,
                        help="Use QLoRA (4-bit quantization) for memory-efficient training")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                        help="Compute dtype for 4-bit quantization (bfloat16 or float16)")
    
    # Loss arguments
    parser.add_argument("--temperature", type=float, default=0.02)
    
    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_flash_attention", action="store_true", default=False)
    parser.add_argument("--no_flash_attention", action="store_true", default=False,
                        help="Disable flash attention (use eager attention instead)")
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_pixels", type=int, default=512*512,
                        help="Maximum pixels per image (default: 262144 = 512x512)")
    parser.add_argument("--preload_images", action="store_true", default=False,
                        help="Preload all images into memory (faster but uses more RAM)")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info(f"Starting training with args: {args}")
    
    # Import modules
    from qwen3_vl_retrieval.models.colqwen3vl import ColQwen3VL
    from qwen3_vl_retrieval.models.processing_colqwen3vl import ColQwen3VLProcessor
    from qwen3_vl_retrieval.data.dataset import ColPaliEngineDataset
    from qwen3_vl_retrieval.data.collator import VisualRetrieverCollator
    from qwen3_vl_retrieval.training.config import ColModelTrainingConfig, LoRAConfig
    from qwen3_vl_retrieval.training.trainer import ColQwen3VLTrainer
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model_path = str(Path(args.model_path).expanduser())
    
    # Prepare quantization config for QLoRA
    quantization_config = None
    if args.use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if args.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(f"Using QLoRA with 4-bit quantization, compute_dtype={args.bnb_4bit_compute_dtype}")
        except ImportError:
            logger.warning("bitsandbytes not installed, falling back to standard LoRA")
            logger.warning("Install with: pip install bitsandbytes")
            args.use_qlora = False
    
    # Determine attention implementation
    if args.no_flash_attention:
        attn_impl = "eager"
    elif args.use_flash_attention:
        attn_impl = "flash_attention_2"
    else:
        attn_impl = "sdpa"  # Default to SDPA (PyTorch native, faster than eager)
    
    logger.info(f"Using attention implementation: {attn_impl}")
    
    model = ColQwen3VL.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=attn_impl,
        quantization_config=quantization_config,
    )
    
    # Enable LoRA training
    if args.use_lora:
        logger.info(f"Enabling LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
        model.enable_lora_training(rank=args.lora_rank, alpha=args.lora_alpha)
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Load processor with max_pixels limit to control memory
    processor = ColQwen3VLProcessor.from_pretrained(
        model_path,
        max_pixels=args.max_pixels,
    )
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    train_dataset = ColPaliEngineDataset(
        data_path=args.data_path,
        image_dir=args.image_dir,
        preload_images=args.preload_images,
    )
    
    eval_dataset = None
    if args.eval_data_path:
        eval_dataset = ColPaliEngineDataset(
            data_path=args.eval_data_path,
            image_dir=args.image_dir,
        )
    
    # Create collator
    collator = VisualRetrieverCollator(processor=processor)
    
    # Create config
    config = ColModelTrainingConfig(
        model_name_or_path=model_path,
        output_dir=str(output_dir),
        use_lora=args.use_lora,
        lora_config=LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
    )
    
    # Create trainer
    trainer = ColQwen3VLTrainer(
        model=model,
        processor=processor,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
