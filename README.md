# Qwen3-VL RAG Retrieval System

A ColPali-style visual document retrieval system based on Qwen3-VL-4B, implementing late interaction (MaxSim) for efficient and accurate document retrieval.

## Features

- **ColQwen3VL Model**: Multi-vector embeddings using Qwen3-VL vision-language model with 128-dimensional output
- **Two-Stage Retrieval**: BM25/BGE-M3 for fast first-stage recall + MaxSim reranking for precision
- **Binary Quantization**: 32x storage reduction with Hamming distance pre-filtering
- **LoRA Fine-tuning**: Memory-efficient training with configurable LoRA adapters
- **Dynamic Resolution**: Native support for variable image sizes via Qwen3-VL's M-RoPE
- **Property-Based Testing**: Comprehensive test suite with hypothesis

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Two-Stage Retrieval System                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Stage 1: Fast Recall                             │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │   Query     │───▶│  BM25 /     │───▶│  Top-N Candidates       │  │   │
│  │  │   Text      │    │  BGE-M3     │    │  (doc_ids + images)     │  │   │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Stage 2: Precise Reranking                       │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │   Query     │───▶│  Qwen3-VL   │───▶│  Query Embeddings       │  │   │
│  │  │   Text      │    │  Encoder    │    │  (N_q × 128)            │  │   │
│  │  └─────────────┘    └─────────────┘    └──────────┬──────────────┘  │   │
│  │                                                    │                 │   │
│  │  ┌─────────────────────────────────────────────────▼──────────────┐ │   │
│  │  │                      MaxSim Scoring                             │ │   │
│  │  │  ┌──────────────────┐    ┌──────────────────────────────────┐  │ │   │
│  │  │  │ Pre-computed Doc │    │  score = Σ max(q_i · d_j)        │  │ │   │
│  │  │  │ Embeddings Index │───▶│         i    j                   │  │ │   │
│  │  │  │ (Binary + Float) │    │  for each query token i          │  │ │   │
│  │  │  └──────────────────┘    └──────────────────────────────────┘  │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                         ┌─────────────────────┐                            │
│                         │  Ranked Results     │                            │
│                         │  (sorted by score)  │                            │
│                         └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8 (recommended for GPU acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-repo/qwen3_vl_retrieval.git
cd qwen3_vl_retrieval

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Dependencies

Core dependencies:
- `torch>=2.0`
- `transformers>=4.40`
- `peft>=0.10`
- `qwen-vl-utils`
- `lmdb>=1.4.0`
- `rank-bm25`
- `pillow`
- `numpy`

Optional dependencies:
- `flash-attn>=2.0` (for faster attention)
- `sentence-transformers` (for BGE-M3 retrieval)
- `hypothesis` (for property-based testing)

## Quick Start

### 1. Download Base Model

Download Qwen3-VL-4B-Instruct from HuggingFace or use a local checkpoint:

```bash
# Using HuggingFace CLI
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir ~/checkpoints/Qwen3-VL-4B-Instruct
```

### 2. Index Documents

```bash
python -m qwen3_vl_retrieval.examples.index_documents \
    --model_path ~/checkpoints/Qwen3-VL-4B-Instruct \
    --image_dir ./documents \
    --output_dir ./index
```

### 3. Query Documents

```bash
# Interactive mode
python -m qwen3_vl_retrieval.examples.query_retrieval \
    --model_path ~/checkpoints/Qwen3-VL-4B-Instruct \
    --index_dir ./index \
    --interactive

# Single query
python -m qwen3_vl_retrieval.examples.query_retrieval \
    --model_path ~/checkpoints/Qwen3-VL-4B-Instruct \
    --index_dir ./index \
    --query "What is the revenue for Q3 2024?"
```

### 4. End-to-End Demo

```bash
python -m qwen3_vl_retrieval.examples.end_to_end_demo \
    --model_path ~/checkpoints/Qwen3-VL-4B-Instruct \
    --image_dir ./sample_documents
```

## Python API Usage

### Basic Retrieval

```python
from qwen3_vl_retrieval.inference.model_loader import load_model_for_inference
from qwen3_vl_retrieval.retrieval import (
    EmbeddingStore,
    FirstStageRetriever,
    SecondStageReranker,
)

# Load model
model, processor = load_model_for_inference(
    model_name_or_path="path/to/Qwen3-VL-4B-Instruct",
    lora_path="path/to/lora_weights",  # Optional
)

# Initialize components
embedding_store = EmbeddingStore("./embeddings", dim=128)
first_stage = FirstStageRetriever(method="bm25")
second_stage = SecondStageReranker(
    model=model,
    processor=processor,
    embedding_store=embedding_store,
)

# Index documents
doc_ids = ["doc1", "doc2", "doc3"]
texts = ["Document 1 content", "Document 2 content", "Document 3 content"]
image_paths = ["doc1.png", "doc2.png", "doc3.png"]

first_stage.index_documents(doc_ids, texts, image_paths)
embedding_store.batch_encode_documents(
    model=model,
    processor=processor,
    image_paths=image_paths,
    doc_ids=doc_ids,
)

# Retrieve
query = "What information is in document 1?"
candidates = first_stage.retrieve(query, top_k=100)
candidate_ids = [doc_id for doc_id, _, _ in candidates]

results = second_stage.rerank(
    query=query,
    candidate_doc_ids=candidate_ids,
    top_k=10,
)

for doc_id, score in results:
    print(f"{doc_id}: {score:.4f}")
```

### Using ColQwen3VLRetriever (High-Level API)

```python
from qwen3_vl_retrieval.inference.api import ColQwen3VLRetriever

# Load retriever
retriever = ColQwen3VLRetriever.from_pretrained(
    model_name_or_path="path/to/Qwen3-VL-4B-Instruct",
    lora_path="path/to/lora_weights",
    embedding_store_path="./embeddings",
    first_stage_index_path="./first_stage_index",
)

# Index documents
retriever.index_documents(
    doc_ids=["doc1", "doc2"],
    texts=["Content 1", "Content 2"],
    image_paths=["doc1.png", "doc2.png"],
)

# Retrieve
results = retriever.retrieve("Find financial data", top_k=10)
```

### Encoding Documents and Queries Separately

```python
from qwen3_vl_retrieval.inference.api import encode_documents, encode_queries

# Encode documents
doc_embeddings = encode_documents(
    model=model,
    processor=processor,
    images=["doc1.png", "doc2.png"],
)

# Encode queries
query_embeddings = encode_queries(
    model=model,
    processor=processor,
    queries=["Query 1", "Query 2"],
)
```

## Training

### Download Training Data

```bash
python -m qwen3_vl_retrieval.scripts.download_vidore \
    --output_dir ./data/vidore \
    --max_samples 10000
```

### Train with LoRA

```bash
python -m qwen3_vl_retrieval.scripts.train \
    --model_path ~/checkpoints/Qwen3-VL-4B-Instruct \
    --data_path ./data/vidore/train_split.json \
    --image_dir ./data/vidore \
    --output_dir ./outputs \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --gradient_checkpointing \
    --bf16
```

### Training Configuration

| Parameter | Recommended (32GB VRAM) | Recommended (24GB VRAM) |
|-----------|------------------------|------------------------|
| batch_size | 2 | 1 |
| gradient_accumulation_steps | 8 | 16 |
| learning_rate | 2e-4 | 2e-4 |
| lora_rank | 32 | 16 |
| lora_alpha | 32 | 16 |
| gradient_checkpointing | True | True |
| bf16 | True | True |
| max_pixels | 262144 | 196608 |

## Evaluation

### Evaluate on ViDoRe Benchmark

```bash
python -m qwen3_vl_retrieval.scripts.evaluate \
    --model_path ~/checkpoints/Qwen3-VL-4B-Instruct \
    --lora_path ./outputs/lora_weights \
    --data_path ./data/vidore/test_split.json \
    --image_dir ./data/vidore \
    --output_file ./results.json
```

### Metrics

The system computes standard retrieval metrics:
- **MRR** (Mean Reciprocal Rank)
- **Recall@K** (K = 1, 5, 10, 20)
- **NDCG@K** (Normalized Discounted Cumulative Gain)

## Project Structure

```
qwen3_vl_retrieval/
├── models/                  # ColQwen3VL model and processor
│   ├── colqwen3vl.py       # Main model class
│   └── processing_colqwen3vl.py  # Processor for images/queries
├── retrieval/               # Retrieval components
│   ├── binary_quantizer.py  # Binary quantization
│   ├── embedding_store.py   # LMDB-based embedding storage
│   ├── first_stage_retriever.py  # BM25/BGE-M3 retrieval
│   └── second_stage_reranker.py  # MaxSim reranking
├── training/                # Training utilities
│   ├── config.py           # Training configuration
│   ├── losses.py           # InfoNCE loss
│   └── trainer.py          # HuggingFace Trainer integration
├── data/                    # Data processing
│   ├── dataset.py          # Dataset classes
│   ├── collator.py         # Data collation
│   └── hard_negative_miner.py  # Hard negative mining
├── inference/               # Inference utilities
│   ├── api.py              # High-level retrieval API
│   └── model_loader.py     # Model loading utilities
├── evaluation/              # Evaluation metrics
│   └── metrics.py          # MRR, Recall@K, NDCG@K
├── examples/                # Example scripts
│   ├── index_documents.py  # Document indexing example
│   ├── query_retrieval.py  # Query retrieval example
│   └── end_to_end_demo.py  # Complete demo
├── scripts/                 # CLI scripts
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── download_vidore.py  # Data download script
└── tests/                   # Property-based tests
```

## Configuration

### Model Configuration

```python
# Load with custom configuration
from qwen3_vl_retrieval.inference.model_loader import ModelLoader

loader = ModelLoader()
model, processor = loader.load(
    model_name_or_path="path/to/model",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # or "sdpa", "eager"
    device_map="auto",
    max_num_visual_tokens=1344,  # Max visual tokens per image
)
```

### Embedding Store Configuration

```python
from qwen3_vl_retrieval.retrieval import EmbeddingStore

store = EmbeddingStore(
    storage_path="./embeddings",
    dim=128,                    # Embedding dimension
    map_size=10 * 1024**3,     # 10GB max storage
)
```

### Retrieval Configuration

```python
# First-stage retrieval
first_stage = FirstStageRetriever(
    method="bm25",  # or "bge-m3"
    index_path="./index",
)

# Second-stage reranking
second_stage = SecondStageReranker(
    model=model,
    processor=processor,
    embedding_store=store,
    use_binary_quantization=True,  # Enable binary pre-filtering
)

# Retrieval parameters
results = second_stage.rerank(
    query="...",
    candidate_doc_ids=[...],
    top_k=10,                    # Final results
    binary_rescore_ratio=10,     # Binary pre-filter ratio
)
```

## Performance Tips

1. **Use Flash Attention 2**: Install `flash-attn` for 2-3x faster inference
2. **Enable Binary Quantization**: Reduces storage by 32x and speeds up retrieval
3. **Batch Processing**: Use batch APIs for encoding multiple documents/queries
4. **Gradient Checkpointing**: Essential for training on limited VRAM
5. **Pre-compute Embeddings**: Index documents offline for fast query-time retrieval

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` during indexing
- Enable `gradient_checkpointing` during training
- Use `max_pixels` to limit image resolution
- Try 4-bit quantization with `load_model_quantized()`

### Slow Indexing

- Increase `batch_size` if VRAM allows
- Use multiple GPUs with `device_map="auto"`
- Pre-process images with `preprocess_data.py`

### Low Retrieval Quality

- Fine-tune with LoRA on domain-specific data
- Increase `first_stage_top_k` for better recall
- Adjust `binary_rescore_ratio` for precision/speed tradeoff

## License

MIT

## Acknowledgments

- [ColPali](https://github.com/illuin-tech/colpali) - Original ColPali implementation
- [Qwen3-VL](https://github.com/QwenLM/Qwen2.5-VL) - Base vision-language model
- [ViDoRe](https://huggingface.co/datasets/vidore) - Visual Document Retrieval benchmark

## Citation

If you use this project in your research, please cite:

```bibtex
@software{qwen3_vl_retrieval,
  title = {Qwen3-VL RAG Retrieval System},
  year = {2024},
  url = {https://github.com/your-repo/qwen3_vl_retrieval}
}
```
