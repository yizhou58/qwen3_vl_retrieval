#!/usr/bin/env python
"""
Benchmark script to identify performance bottlenecks.
"""

import argparse
import time
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_qlora", action="store_true", default=False)
    args = parser.parse_args()
    
    print("=" * 60)
    print("BENCHMARK: Identifying performance bottlenecks")
    print(f"QLoRA: {args.use_qlora}, Batch size: {args.batch_size}")
    print("=" * 60)
    
    # 1. Test data loading speed
    print("\n[1] Testing data loading speed...")
    from qwen3_vl_retrieval.data.dataset import PreprocessedDataset, PreprocessedCollator
    
    dataset = PreprocessedDataset(args.data_path)
    collator = PreprocessedCollator()
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=4)
    
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= 5:
            break
    data_time = (time.time() - start) / 5
    print(f"   Data loading: {data_time:.2f}s per batch")
    
    # 2. Test model loading
    print("\n[2] Loading model...")
    from qwen3_vl_retrieval.models.colqwen3vl import ColQwen3VL
    
    model_path = str(Path(args.model_path).expanduser())
    
    quantization_config = None
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("   Using 4-bit quantization (QLoRA)")
    
    start = time.time()
    model = ColQwen3VL.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
    model.enable_lora_training(rank=32, alpha=32)
    if not args.use_qlora:
        model = model.cuda()
    load_time = time.time() - start
    print(f"   Model loading: {load_time:.2f}s")
    
    # 3. Test forward pass (inference only)
    print("\n[3] Testing forward pass (no grad)...")
    batch = next(iter(loader))
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Warmup
    with torch.no_grad():
        _ = model(
            input_ids=batch["doc_input_ids"],
            attention_mask=batch["doc_attention_mask"],
            pixel_values=batch["doc_pixel_values"],
            image_grid_thw=batch["doc_image_grid_thw"],
        )
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(3):
        with torch.no_grad():
            _ = model(
                input_ids=batch["doc_input_ids"],
                attention_mask=batch["doc_attention_mask"],
                pixel_values=batch["doc_pixel_values"],
                image_grid_thw=batch["doc_image_grid_thw"],
            )
        torch.cuda.synchronize()
    forward_time = (time.time() - start) / 3
    print(f"   Forward pass (no grad): {forward_time:.2f}s per batch")
    print(f"   Per image: {forward_time / args.batch_size:.2f}s")
    
    # 4. Memory usage after forward
    print("\n[4] GPU Memory after forward:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # 5. Test forward + backward (with smaller batch if needed)
    print("\n[5] Testing forward + backward...")
    torch.cuda.empty_cache()
    
    from qwen3_vl_retrieval.training.losses import ColbertLoss
    loss_fn = ColbertLoss(temperature=0.02)
    
    try:
        query_out = model(
            input_ids=batch["query_input_ids"],
            attention_mask=batch["query_attention_mask"],
        )
        doc_out = model(
            input_ids=batch["doc_input_ids"],
            attention_mask=batch["doc_attention_mask"],
            pixel_values=batch["doc_pixel_values"],
            image_grid_thw=batch["doc_image_grid_thw"],
        )
        loss = loss_fn(query_out, doc_out, batch["query_attention_mask"].bool(), batch["doc_attention_mask"].bool())
        loss.backward()
        torch.cuda.synchronize()
        
        model.zero_grad()
        torch.cuda.empty_cache()
        
        start = time.time()
        for _ in range(3):
            query_out = model(
                input_ids=batch["query_input_ids"],
                attention_mask=batch["query_attention_mask"],
            )
            doc_out = model(
                input_ids=batch["doc_input_ids"],
                attention_mask=batch["doc_attention_mask"],
                pixel_values=batch["doc_pixel_values"],
                image_grid_thw=batch["doc_image_grid_thw"],
            )
            loss = loss_fn(query_out, doc_out, batch["query_attention_mask"].bool(), batch["doc_attention_mask"].bool())
            loss.backward()
            torch.cuda.synchronize()
            model.zero_grad()
        train_time = (time.time() - start) / 3
        print(f"   Forward + backward: {train_time:.2f}s per batch")
        print(f"   Per image: {train_time / args.batch_size:.2f}s")
    except torch.cuda.OutOfMemoryError:
        print("   OOM! Try smaller batch_size or use --use_qlora")
        train_time = float('inf')
    
    # 6. Final memory
    print("\n[6] Final GPU Memory:")
    print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"   Data loading:       {data_time:.2f}s per batch")
    print(f"   Forward (no grad):  {forward_time:.2f}s per batch ({forward_time/args.batch_size:.2f}s/img)")
    if train_time != float('inf'):
        print(f"   Forward + backward: {train_time:.2f}s per batch ({train_time/args.batch_size:.2f}s/img)")
    print("=" * 60)


if __name__ == "__main__":
    main()
