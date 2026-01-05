# Qwen3-VL RAG Retrieval System

A ColPali-style visual document retrieval system based on Qwen3-VL-4B, implementing late interaction for efficient document retrieval.

## Features

- **ColQwen3VL Model**: Multi-vector embeddings using Qwen3-VL vision-language model
- **Two-Stage Retrieval**: Binary quantization for fast first-stage + MaxSim reranking
- **LoRA Fine-tuning**: Memory-efficient training with configurable LoRA
- **Property-Based Testing**: Comprehensive test suite with hypothesis

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Query Text    │────▶│  ColQwen3VL      │────▶│ Query Embedding │
└─────────────────┘     │  (Qwen3-VL-4B)   │     │ (N × D)         │
                        └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │ MaxSim
│ Document Image  │────▶│  ColQwen3VL      │────▶ ────────┴────────▶ Score
└─────────────────┘     │  (Qwen3-VL-4B)   │     │ Doc Embedding   │
                        └──────────────────┘     │ (M × D)         │
                                                 └─────────────────┘
```

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Download Training Data

```bash
python -m qwen3_vl_retrieval.scripts.download_vidore \
    --output_dir /path/to/data/vidore \
    --max_samples 1000
```

### 2. Train Model

```bash
python -m qwen3_vl_retrieval.scripts.train \
    --model_path /path/to/Qwen3-VL-4B-Instruct/ \
    --data_path /path/to/data/vidore/train_split.json \
    --image_dir /path/to/data/vidore \
    --output_dir /path/to/outputs \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16
```

### 3. Inference

```python
from qwen3_vl_retrieval.models import ColQwen3VL, ColQwen3VLProcessor
from qwen3_vl_retrieval.retrieval import FirstStageRetriever, SecondStageReranker

# Load model
model = ColQwen3VL.from_pretrained("path/to/model")
processor = ColQwen3VLProcessor.from_pretrained("path/to/model")

# Create retriever
retriever = FirstStageRetriever(model, processor)

# Index documents
retriever.index_documents(document_images)

# Search
results = retriever.search(query, top_k=10)
```

## Project Structure

```
qwen3_vl_retrieval/
├── models/              # ColQwen3VL model and processor
├── retrieval/           # Retrieval components
│   ├── binary_quantizer.py
│   ├── embedding_store.py
│   ├── first_stage_retriever.py
│   └── second_stage_reranker.py
├── training/            # Training utilities
│   ├── config.py
│   ├── losses.py
│   └── trainer.py
├── data/                # Data processing
│   ├── dataset.py
│   ├── collator.py
│   └── hard_negative_miner.py
├── evaluation/          # Evaluation metrics
├── scripts/             # CLI scripts
└── tests/               # Property-based tests
```

## Training Configuration

| Parameter | Recommended (32GB VRAM) |
|-----------|------------------------|
| batch_size | 2 |
| gradient_accumulation_steps | 8 |
| learning_rate | 2e-4 |
| lora_rank | 32 |
| gradient_checkpointing | True |
| bf16 | True |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- transformers >= 4.40
- peft >= 0.10
- qwen-vl-utils

## License

MIT

## Acknowledgments

- [ColPali](https://github.com/illuin-tech/colpali) - Original ColPali implementation
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) - Base vision-language model
