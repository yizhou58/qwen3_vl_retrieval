#!/usr/bin/env python
"""
Evaluation Script for Qwen3-VL RAG Retrieval System.

Supports evaluation on ViDoRe benchmark datasets and generates comprehensive
evaluation reports with retrieval metrics and latency analysis.

Requirements: 9.2, 9.3, 9.5
- Support evaluation on ViDoRe benchmark datasets
- Report latency metrics for both retrieval stages
- Generate evaluation reports with detailed breakdowns

Usage:
    # Evaluate on ViDoRe benchmark
    python -m qwen3_vl_retrieval.scripts.evaluate \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --dataset vidore/docvqa_test_subsampled \
        --output_dir ./evaluation_results

    # Evaluate with custom dataset
    python -m qwen3_vl_retrieval.scripts.evaluate \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --data_path ./my_dataset.json \
        --output_dir ./evaluation_results

    # Compare with/without binary quantization
    python -m qwen3_vl_retrieval.scripts.evaluate \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --dataset vidore/docvqa_test_subsampled \
        --compare_quantization \
        --output_dir ./evaluation_results
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    total_ms: float = 0.0
    count: int = 0


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    # Dataset info
    dataset_name: str = ""
    num_queries: int = 0
    num_documents: int = 0
    
    # Retrieval metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Latency stats
    first_stage_latency: Dict[str, float] = field(default_factory=dict)
    second_stage_latency: Dict[str, float] = field(default_factory=dict)
    total_latency: Dict[str, float] = field(default_factory=dict)
    encoding_latency: Dict[str, float] = field(default_factory=dict)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ViDoReDataset:
    """
    ViDoRe benchmark dataset loader.
    
    Supports loading from HuggingFace datasets or local JSON files.
    """
    
    # Known ViDoRe benchmark datasets
    VIDORE_DATASETS = [
        "vidore/docvqa_test_subsampled",
        "vidore/infovqa_test_subsampled", 
        "vidore/arxivqa_test_subsampled",
        "vidore/tabfquad_test_subsampled",
        "vidore/shiftproject_test",
        "vidore/syntheticDocQA_artificial_intelligence_test",
        "vidore/syntheticDocQA_energy_test",
        "vidore/syntheticDocQA_government_reports_test",
        "vidore/syntheticDocQA_healthcare_industry_test",
    ]
    
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        data_path: Optional[str] = None,
        split: str = "test",
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize ViDoRe dataset.
        
        Args:
            dataset_name: HuggingFace dataset name (e.g., "vidore/docvqa_test_subsampled")
            data_path: Path to local JSON file
            split: Dataset split to use
            max_samples: Maximum number of samples to load
            cache_dir: Cache directory for downloaded data
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.split = split
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        
        self.queries: List[str] = []
        self.query_ids: List[str] = []
        self.doc_ids: List[str] = []
        self.images: List[Any] = []  # PIL Images or paths
        self.image_paths: List[str] = []
        self.relevance: Dict[str, Set[str]] = {}  # query_id -> set of relevant doc_ids
        
        self._load_data()
    
    def _load_data(self):
        """Load dataset from HuggingFace or local file."""
        if self.data_path:
            self._load_from_json()
        elif self.dataset_name:
            self._load_from_huggingface()
        else:
            raise ValueError("Either dataset_name or data_path must be provided")
    
    def _load_from_huggingface(self):
        """Load dataset from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to load with split={self.split}, trying without split")
            dataset = load_dataset(
                self.dataset_name,
                cache_dir=self.cache_dir,
            )
            if isinstance(dataset, dict):
                available_splits = list(dataset.keys())
                if self.split in available_splits:
                    dataset = dataset[self.split]
                else:
                    dataset = dataset[available_splits[0]]
                    logger.info(f"Using split: {available_splits[0]}")
        
        # Process dataset
        seen_docs = set()
        doc_to_image = {}
        
        for idx, item in enumerate(dataset):
            if self.max_samples and idx >= self.max_samples:
                break
            
            # Extract query
            query = item.get("query") or item.get("question") or item.get("text")
            if not query:
                continue
            
            query_id = f"q_{idx}"
            self.queries.append(query)
            self.query_ids.append(query_id)
            
            # Extract document/image
            image = item.get("image") or item.get("page_image")
            doc_id = item.get("doc_id") or item.get("docId") or f"doc_{idx}"
            
            # Ensure doc_id is string
            doc_id = str(doc_id)
            
            # Handle relevance
            if query_id not in self.relevance:
                self.relevance[query_id] = set()
            self.relevance[query_id].add(doc_id)
            
            # Store unique documents
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                self.doc_ids.append(doc_id)
                self.images.append(image)
                doc_to_image[doc_id] = image
        
        logger.info(f"Loaded {len(self.queries)} queries, {len(self.doc_ids)} documents")
    
    def _load_from_json(self):
        """Load dataset from local JSON file."""
        logger.info(f"Loading dataset from: {self.data_path}")
        
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Support multiple JSON formats
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            samples = data.get("samples") or data.get("data") or list(data.values())
        else:
            raise ValueError(f"Unsupported JSON format")
        
        seen_docs = set()
        base_dir = Path(self.data_path).parent
        
        for idx, item in enumerate(samples):
            if self.max_samples and idx >= self.max_samples:
                break
            
            # Extract query
            query = item.get("query") or item.get("question") or item.get("text")
            if not query:
                continue
            
            query_id = item.get("query_id") or f"q_{idx}"
            self.queries.append(query)
            self.query_ids.append(query_id)
            
            # Extract document info
            doc_id = item.get("doc_id") or item.get("docId") or f"doc_{idx}"
            image_path = item.get("image_path") or item.get("image")
            
            # Handle relevance
            if query_id not in self.relevance:
                self.relevance[query_id] = set()
            
            # Support explicit relevance or assume positive pair
            relevant_docs = item.get("relevant_docs") or item.get("positive_docs") or [doc_id]
            for rel_doc in relevant_docs:
                self.relevance[query_id].add(rel_doc)
            
            # Store unique documents
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                self.doc_ids.append(doc_id)
                
                if image_path:
                    full_path = str(base_dir / image_path) if not os.path.isabs(image_path) else image_path
                    self.image_paths.append(full_path)
                    self.images.append(full_path)
                else:
                    self.image_paths.append("")
                    self.images.append(None)
        
        logger.info(f"Loaded {len(self.queries)} queries, {len(self.doc_ids)} documents")
    
    def get_queries(self) -> List[Tuple[str, str]]:
        """Get list of (query_id, query_text) tuples."""
        return list(zip(self.query_ids, self.queries))
    
    def get_documents(self) -> List[Tuple[str, Any]]:
        """Get list of (doc_id, image) tuples."""
        return list(zip(self.doc_ids, self.images))
    
    def get_relevance(self, query_id: str) -> Set[str]:
        """Get relevant document IDs for a query."""
        return self.relevance.get(query_id, set())



class RetrievalEvaluator:
    """
    Evaluator for the two-stage retrieval system.
    
    Uses binary visual embeddings for first stage and MaxSim for second stage.
    Both stages benefit from fine-tuning the visual model.
    
    Requirements:
        9.2: Support evaluation on ViDoRe benchmark datasets
        9.3: Report latency metrics for both retrieval stages
        9.5: Generate evaluation reports with detailed breakdowns
    """
    
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: Optional[str] = None,
        use_binary_quantization: bool = True,
        first_stage_top_k: int = 100,
        second_stage_top_k: int = 10,
        binary_rescore_ratio: int = 10,
        skip_first_stage: bool = False,
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to base Qwen3-VL model
            lora_path: Optional path to LoRA weights
            device: Device to use (cuda, cpu, or auto)
            use_binary_quantization: Whether to use binary quantization
            first_stage_top_k: Number of candidates from first stage
            second_stage_top_k: Number of final results
            binary_rescore_ratio: Ratio for binary pre-filtering
            skip_first_stage: Skip first stage and use all docs for MaxSim
        """
        self.model_path = model_path
        self.lora_path = lora_path
        self.use_binary_quantization = use_binary_quantization
        self.first_stage_top_k = first_stage_top_k
        self.second_stage_top_k = second_stage_top_k
        self.binary_rescore_ratio = binary_rescore_ratio
        self.skip_first_stage = skip_first_stage
        
        # Determine device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Components (lazy loaded)
        self.model = None
        self.processor = None
        self.first_stage = None
        self.second_stage = None
        self.embedding_store = None
        
        # Latency tracking
        self._first_stage_latencies: List[float] = []
        self._second_stage_latencies: List[float] = []
        self._encoding_latencies: List[float] = []
    
    def _load_model(self):
        """Load model and processor."""
        if self.model is not None:
            return
        
        logger.info("Loading model...")
        
        from qwen3_vl_retrieval.inference.model_loader import load_model_for_inference
        
        self.model, self.processor = load_model_for_inference(
            model_name_or_path=str(Path(self.model_path).expanduser()),
            lora_path=self.lora_path,
            device_map=self.device if self.device != "cpu" else None,
        )
        
        # Limit max pixels to avoid OOM and speed up processing
        # Default: 512*512 = 262144 pixels max per image
        if hasattr(self.processor, '_processor') and hasattr(self.processor._processor, 'image_processor'):
            self.processor._processor.image_processor.max_pixels = 512 * 512
            self.processor._processor.image_processor.min_pixels = 28 * 28 * 4
            logger.info("Set max_pixels=262144 (512x512) for faster processing")
        
        logger.info("Model loaded successfully")
    
    def _setup_retrieval_system(self, temp_dir: str):
        """Set up retrieval components."""
        from qwen3_vl_retrieval.retrieval.embedding_store import EmbeddingStore
        from qwen3_vl_retrieval.retrieval.binary_first_stage import BinaryFirstStageRetriever
        from qwen3_vl_retrieval.retrieval.second_stage_reranker import SecondStageReranker
        
        temp_path = Path(temp_dir)
        
        # Embedding store
        self.embedding_store = EmbeddingStore(
            storage_path=str(temp_path / "embeddings"),
            dim=128,
        )
        
        # First-stage retriever (binary visual embeddings)
        self.first_stage = BinaryFirstStageRetriever(
            embedding_store=self.embedding_store,
            index_path=str(temp_path / "binary_first_stage"),
        )
        logger.info("Using binary visual embeddings for first stage retrieval")
        
        # Second-stage reranker
        self.second_stage = SecondStageReranker(
            model=self.model,
            processor=self.processor,
            embedding_store=self.embedding_store,
            use_binary_quantization=self.use_binary_quantization,
            device=self.device,
        )
    
    def _index_documents(
        self,
        dataset: ViDoReDataset,
        batch_size: int = 4,
    ):
        """Index documents from dataset."""
        logger.info("Indexing documents...")
        
        doc_ids = dataset.doc_ids
        images = dataset.images
        
        # For images that are paths, use them directly
        # For PIL images, we need to save them temporarily
        image_paths = []
        temp_images_dir = Path(self.embedding_store.storage_path) / "temp_images"
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (doc_id, image) in enumerate(zip(doc_ids, images)):
            if isinstance(image, str):
                image_paths.append(image)
            elif image is not None:
                # Save PIL image temporarily
                temp_path = temp_images_dir / f"{doc_id}.png"
                if hasattr(image, 'save'):
                    image.save(temp_path)
                image_paths.append(str(temp_path))
            else:
                image_paths.append("")
        
        # Index in first stage (binary visual embeddings)
        self.first_stage.index_documents(doc_ids, image_paths)
        
        # Encode documents for both stages
        logger.info("Encoding document embeddings...")
        start_time = time.time()
        
        valid_paths = [p for p in image_paths if p and os.path.exists(p)]
        valid_doc_ids = [doc_ids[i] for i, p in enumerate(image_paths) if p and os.path.exists(p)]
        
        if valid_paths:
            self.embedding_store.batch_encode_documents(
                model=self.model,
                processor=self.processor,
                image_paths=valid_paths,
                doc_ids=valid_doc_ids,
                batch_size=batch_size,
                quantize=self.use_binary_quantization,
                device=self.device,
                show_progress=True,
            )
        
        # Refresh binary first stage cache after encoding
        self.first_stage._refresh_cache()
        logger.info("Binary first stage cache refreshed")
        
        encoding_time = time.time() - start_time
        self._encoding_latencies.append(encoding_time * 1000)  # Convert to ms
        
        logger.info(f"Indexed {len(doc_ids)} documents in {encoding_time:.2f}s")
    
    def _compute_latency_stats(self, latencies: List[float]) -> LatencyStats:
        """Compute latency statistics."""
        import numpy as np
        
        if not latencies:
            return LatencyStats()
        
        arr = np.array(latencies)
        return LatencyStats(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            total_ms=float(np.sum(arr)),
            count=len(latencies),
        )
    
    def evaluate(
        self,
        dataset: ViDoReDataset,
        batch_size: int = 4,
        recall_k_values: List[int] = None,
        ndcg_k_values: List[int] = None,
    ) -> EvaluationResult:
        """
        Run evaluation on dataset.
        
        Args:
            dataset: ViDoRe dataset to evaluate on
            batch_size: Batch size for encoding
            recall_k_values: K values for Recall@K
            ndcg_k_values: K values for NDCG@K
            
        Returns:
            EvaluationResult with metrics and latency stats
        """
        import tempfile
        from qwen3_vl_retrieval.evaluation.metrics import RetrievalMetrics
        
        recall_k_values = recall_k_values or [1, 5, 10, 20, 50, 100]
        ndcg_k_values = ndcg_k_values or [5, 10, 20]
        
        # Load model
        self._load_model()
        
        # Create temporary directory for index
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup retrieval system
            self._setup_retrieval_system(temp_dir)
            
            # Index documents
            self._index_documents(dataset, batch_size)
            
            # Get all doc_ids for skip_first_stage mode
            all_doc_ids = dataset.doc_ids
            
            # Run retrieval for all queries
            logger.info("Running retrieval evaluation...")
            if self.skip_first_stage:
                logger.info("Skipping first stage, using all documents for MaxSim ranking")
            
            rankings: List[List[str]] = []
            relevant_docs: List[Set[str]] = []
            
            queries = dataset.get_queries()
            
            for query_id, query_text in tqdm(queries, desc="Evaluating"):
                if self.skip_first_stage:
                    # Skip first stage, use all documents directly
                    self._first_stage_latencies.append(0.0)
                    candidate_doc_ids = all_doc_ids
                else:
                    # Binary visual first stage: encode query and use binary retrieval
                    start_time = time.time()
                    
                    # Encode query
                    with torch.no_grad():
                        query_inputs = self.processor.process_queries([query_text])
                        query_inputs = {
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in query_inputs.items()
                        }
                        query_embedding = self.model(**query_inputs)[0]  # (num_tokens, dim)
                    
                    # Binary first stage retrieval
                    first_stage_results = self.first_stage.retrieve(
                        query_embedding,
                        top_k=self.first_stage_top_k,
                        use_pooled=True,  # Use pooled for speed
                    )
                    first_stage_time = (time.time() - start_time) * 1000
                    self._first_stage_latencies.append(first_stage_time)
                    
                    candidate_doc_ids = [doc_id for doc_id, _, _ in first_stage_results]
                
                # Second stage reranking
                if candidate_doc_ids:
                    start_time = time.time()
                    results = self.second_stage.rerank(
                        query=query_text,
                        candidate_doc_ids=candidate_doc_ids,
                        top_k=self.second_stage_top_k,
                        binary_rescore_ratio=self.binary_rescore_ratio,
                    )
                    second_stage_time = (time.time() - start_time) * 1000
                    self._second_stage_latencies.append(second_stage_time)
                    
                    ranking = [doc_id for doc_id, _ in results]
                else:
                    ranking = []
                    self._second_stage_latencies.append(0.0)
                
                rankings.append(ranking)
                relevant_docs.append(dataset.get_relevance(query_id))
            
            # Compute metrics
            logger.info("Computing metrics...")
            metrics_calculator = RetrievalMetrics(
                recall_k_values=recall_k_values,
                ndcg_k_values=ndcg_k_values,
            )
            metrics = metrics_calculator.compute_all(rankings, relevant_docs)
            
            # Compute latency stats
            first_stage_stats = self._compute_latency_stats(self._first_stage_latencies)
            second_stage_stats = self._compute_latency_stats(self._second_stage_latencies)
            
            total_latencies = [
                f + s for f, s in zip(self._first_stage_latencies, self._second_stage_latencies)
            ]
            total_stats = self._compute_latency_stats(total_latencies)
            encoding_stats = self._compute_latency_stats(self._encoding_latencies)
            
            # Build result
            result = EvaluationResult(
                dataset_name=dataset.dataset_name or str(dataset.data_path),
                num_queries=len(queries),
                num_documents=len(dataset.doc_ids),
                metrics=metrics,
                first_stage_latency=asdict(first_stage_stats),
                second_stage_latency=asdict(second_stage_stats),
                total_latency=asdict(total_stats),
                encoding_latency=asdict(encoding_stats),
                config={
                    "model_path": self.model_path,
                    "lora_path": self.lora_path,
                    "device": self.device,
                    "use_binary_quantization": self.use_binary_quantization,
                    "first_stage_method": "binary_visual",
                    "first_stage_top_k": self.first_stage_top_k,
                    "second_stage_top_k": self.second_stage_top_k,
                    "binary_rescore_ratio": self.binary_rescore_ratio,
                    "skip_first_stage": self.skip_first_stage,
                },
                timestamp=datetime.now().isoformat(),
            )
            
            return result



def generate_report(
    result: EvaluationResult,
    output_path: Optional[str] = None,
    format: str = "text",
) -> str:
    """
    Generate evaluation report.
    
    Args:
        result: Evaluation result
        output_path: Optional path to save report
        format: Report format ("text", "json", or "markdown")
        
    Returns:
        Report string
        
    Requirements:
        9.5: Generate evaluation reports with detailed breakdowns
    """
    if format == "json":
        report = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    elif format == "markdown":
        report = _generate_markdown_report(result)
    else:
        report = _generate_text_report(result)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to: {output_path}")
    
    return report


def _generate_text_report(result: EvaluationResult) -> str:
    """Generate text format report."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("  Qwen3-VL RAG Retrieval System - Evaluation Report")
    lines.append("=" * 70)
    lines.append("")
    
    # Dataset info
    lines.append("Dataset Information:")
    lines.append("-" * 40)
    lines.append(f"  Dataset: {result.dataset_name}")
    lines.append(f"  Queries: {result.num_queries}")
    lines.append(f"  Documents: {result.num_documents}")
    lines.append(f"  Timestamp: {result.timestamp}")
    lines.append("")
    
    # Configuration
    lines.append("Configuration:")
    lines.append("-" * 40)
    for key, value in result.config.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    
    # Retrieval metrics
    lines.append("Retrieval Metrics:")
    lines.append("-" * 40)
    
    # Group metrics by type
    mrr_map = {k: v for k, v in result.metrics.items() if k in ["MRR", "MAP"]}
    recall = {k: v for k, v in result.metrics.items() if k.startswith("Recall")}
    ndcg = {k: v for k, v in result.metrics.items() if k.startswith("NDCG")}
    precision = {k: v for k, v in result.metrics.items() if k.startswith("Precision")}
    
    for metric, value in mrr_map.items():
        lines.append(f"  {metric}: {value:.4f}")
    lines.append("")
    
    lines.append("  Recall@K:")
    for metric, value in sorted(recall.items(), key=lambda x: int(x[0].split("@")[1])):
        lines.append(f"    {metric}: {value:.4f}")
    lines.append("")
    
    lines.append("  NDCG@K:")
    for metric, value in sorted(ndcg.items(), key=lambda x: int(x[0].split("@")[1])):
        lines.append(f"    {metric}: {value:.4f}")
    lines.append("")
    
    lines.append("  Precision@K:")
    for metric, value in sorted(precision.items(), key=lambda x: int(x[0].split("@")[1])):
        lines.append(f"    {metric}: {value:.4f}")
    lines.append("")
    
    # Latency metrics
    lines.append("Latency Metrics (milliseconds):")
    lines.append("-" * 40)
    
    lines.append("  First Stage (BM25/BGE-M3):")
    lines.append(f"    Mean: {result.first_stage_latency['mean_ms']:.2f} ms")
    lines.append(f"    P50:  {result.first_stage_latency['p50_ms']:.2f} ms")
    lines.append(f"    P95:  {result.first_stage_latency['p95_ms']:.2f} ms")
    lines.append(f"    P99:  {result.first_stage_latency['p99_ms']:.2f} ms")
    lines.append("")
    
    lines.append("  Second Stage (MaxSim Reranking):")
    lines.append(f"    Mean: {result.second_stage_latency['mean_ms']:.2f} ms")
    lines.append(f"    P50:  {result.second_stage_latency['p50_ms']:.2f} ms")
    lines.append(f"    P95:  {result.second_stage_latency['p95_ms']:.2f} ms")
    lines.append(f"    P99:  {result.second_stage_latency['p99_ms']:.2f} ms")
    lines.append("")
    
    lines.append("  Total (End-to-End):")
    lines.append(f"    Mean: {result.total_latency['mean_ms']:.2f} ms")
    lines.append(f"    P50:  {result.total_latency['p50_ms']:.2f} ms")
    lines.append(f"    P95:  {result.total_latency['p95_ms']:.2f} ms")
    lines.append(f"    P99:  {result.total_latency['p99_ms']:.2f} ms")
    lines.append("")
    
    # Throughput
    if result.total_latency['count'] > 0:
        qps = 1000.0 / result.total_latency['mean_ms'] if result.total_latency['mean_ms'] > 0 else 0
        lines.append(f"  Throughput: {qps:.2f} queries/second")
    lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def _generate_markdown_report(result: EvaluationResult) -> str:
    """Generate markdown format report."""
    lines = []
    
    lines.append("# Qwen3-VL RAG Retrieval System - Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {result.timestamp}")
    lines.append("")
    
    # Dataset info
    lines.append("## Dataset Information")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Dataset | {result.dataset_name} |")
    lines.append(f"| Queries | {result.num_queries} |")
    lines.append(f"| Documents | {result.num_documents} |")
    lines.append("")
    
    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for key, value in result.config.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    
    # Retrieval metrics
    lines.append("## Retrieval Metrics")
    lines.append("")
    
    lines.append("### Primary Metrics")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for metric in ["MRR", "MAP"]:
        if metric in result.metrics:
            lines.append(f"| {metric} | {result.metrics[metric]:.4f} |")
    lines.append("")
    
    lines.append("### Recall@K")
    lines.append("")
    lines.append("| K | Recall |")
    lines.append("|---|--------|")
    recall = {k: v for k, v in result.metrics.items() if k.startswith("Recall")}
    for metric, value in sorted(recall.items(), key=lambda x: int(x[0].split("@")[1])):
        k = metric.split("@")[1]
        lines.append(f"| {k} | {value:.4f} |")
    lines.append("")
    
    lines.append("### NDCG@K")
    lines.append("")
    lines.append("| K | NDCG |")
    lines.append("|---|------|")
    ndcg = {k: v for k, v in result.metrics.items() if k.startswith("NDCG")}
    for metric, value in sorted(ndcg.items(), key=lambda x: int(x[0].split("@")[1])):
        k = metric.split("@")[1]
        lines.append(f"| {k} | {value:.4f} |")
    lines.append("")
    
    # Latency metrics
    lines.append("## Latency Metrics")
    lines.append("")
    lines.append("| Stage | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |")
    lines.append("|-------|-----------|----------|----------|----------|")
    lines.append(f"| First Stage | {result.first_stage_latency['mean_ms']:.2f} | {result.first_stage_latency['p50_ms']:.2f} | {result.first_stage_latency['p95_ms']:.2f} | {result.first_stage_latency['p99_ms']:.2f} |")
    lines.append(f"| Second Stage | {result.second_stage_latency['mean_ms']:.2f} | {result.second_stage_latency['p50_ms']:.2f} | {result.second_stage_latency['p95_ms']:.2f} | {result.second_stage_latency['p99_ms']:.2f} |")
    lines.append(f"| Total | {result.total_latency['mean_ms']:.2f} | {result.total_latency['p50_ms']:.2f} | {result.total_latency['p95_ms']:.2f} | {result.total_latency['p99_ms']:.2f} |")
    lines.append("")
    
    # Throughput
    if result.total_latency['count'] > 0:
        qps = 1000.0 / result.total_latency['mean_ms'] if result.total_latency['mean_ms'] > 0 else 0
        lines.append(f"**Throughput:** {qps:.2f} queries/second")
    lines.append("")
    
    return "\n".join(lines)


def compare_quantization(
    model_path: str,
    dataset: ViDoReDataset,
    lora_path: Optional[str] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[EvaluationResult, EvaluationResult]:
    """
    Compare performance with and without binary quantization.
    
    Args:
        model_path: Path to base model
        dataset: Dataset to evaluate on
        lora_path: Optional LoRA weights path
        device: Device to use
        output_dir: Directory to save reports
        
    Returns:
        Tuple of (result_with_quantization, result_without_quantization)
        
    Requirements:
        9.4: Compare performance with and without binary quantization
    """
    logger.info("Evaluating WITH binary quantization...")
    evaluator_with = RetrievalEvaluator(
        model_path=model_path,
        lora_path=lora_path,
        device=device,
        use_binary_quantization=True,
    )
    result_with = evaluator_with.evaluate(dataset)
    
    logger.info("Evaluating WITHOUT binary quantization...")
    evaluator_without = RetrievalEvaluator(
        model_path=model_path,
        lora_path=lora_path,
        device=device,
        use_binary_quantization=False,
    )
    result_without = evaluator_without.evaluate(dataset)
    
    # Generate comparison report
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual reports
        generate_report(result_with, str(output_path / "report_with_quantization.txt"))
        generate_report(result_without, str(output_path / "report_without_quantization.txt"))
        
        # Save JSON results
        with open(output_path / "results_with_quantization.json", "w") as f:
            json.dump(result_with.to_dict(), f, indent=2)
        with open(output_path / "results_without_quantization.json", "w") as f:
            json.dump(result_without.to_dict(), f, indent=2)
        
        # Generate comparison summary
        comparison = _generate_comparison_report(result_with, result_without)
        with open(output_path / "comparison_report.txt", "w") as f:
            f.write(comparison)
        
        logger.info(f"Comparison reports saved to: {output_dir}")
    
    return result_with, result_without


def _generate_comparison_report(
    result_with: EvaluationResult,
    result_without: EvaluationResult,
) -> str:
    """Generate comparison report between quantized and non-quantized results."""
    lines = []
    
    lines.append("=" * 70)
    lines.append("  Binary Quantization Comparison Report")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("Retrieval Metrics Comparison:")
    lines.append("-" * 50)
    lines.append(f"{'Metric':<20} {'With Quant':<15} {'Without Quant':<15} {'Diff':<10}")
    lines.append("-" * 50)
    
    for metric in sorted(result_with.metrics.keys()):
        val_with = result_with.metrics.get(metric, 0)
        val_without = result_without.metrics.get(metric, 0)
        diff = val_with - val_without
        diff_pct = (diff / val_without * 100) if val_without != 0 else 0
        lines.append(f"{metric:<20} {val_with:<15.4f} {val_without:<15.4f} {diff_pct:+.2f}%")
    
    lines.append("")
    lines.append("Latency Comparison (Mean, ms):")
    lines.append("-" * 50)
    lines.append(f"{'Stage':<20} {'With Quant':<15} {'Without Quant':<15} {'Speedup':<10}")
    lines.append("-" * 50)
    
    stages = [
        ("First Stage", "first_stage_latency"),
        ("Second Stage", "second_stage_latency"),
        ("Total", "total_latency"),
    ]
    
    for stage_name, stage_key in stages:
        lat_with = getattr(result_with, stage_key)['mean_ms']
        lat_without = getattr(result_without, stage_key)['mean_ms']
        speedup = lat_without / lat_with if lat_with > 0 else 0
        lines.append(f"{stage_name:<20} {lat_with:<15.2f} {lat_without:<15.2f} {speedup:.2f}x")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)



def run_vidore_benchmark(
    model_path: str,
    lora_path: Optional[str] = None,
    device: Optional[str] = None,
    output_dir: str = "./evaluation_results",
    datasets: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, EvaluationResult]:
    """
    Run evaluation on multiple ViDoRe benchmark datasets.
    
    Args:
        model_path: Path to base model
        lora_path: Optional LoRA weights path
        device: Device to use
        output_dir: Directory to save results
        datasets: List of dataset names (default: all ViDoRe datasets)
        max_samples: Maximum samples per dataset
        
    Returns:
        Dict mapping dataset name to evaluation result
        
    Requirements:
        9.2: Support evaluation on ViDoRe benchmark datasets
    """
    if datasets is None:
        datasets = ViDoReDataset.VIDORE_DATASETS
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating on: {dataset_name}")
        logger.info(f"{'='*50}")
        
        try:
            # Load dataset
            dataset = ViDoReDataset(
                dataset_name=dataset_name,
                max_samples=max_samples,
            )
            
            # Run evaluation
            evaluator = RetrievalEvaluator(
                model_path=model_path,
                lora_path=lora_path,
                device=device,
            )
            result = evaluator.evaluate(dataset)
            results[dataset_name] = result
            
            # Save individual report
            safe_name = dataset_name.replace("/", "_")
            report_path = output_path / f"report_{safe_name}.txt"
            generate_report(result, str(report_path))
            
            # Save JSON result
            json_path = output_path / f"result_{safe_name}.json"
            with open(json_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            
            logger.info(f"Results saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {dataset_name}: {e}")
            continue
    
    # Generate summary report
    if results:
        summary = _generate_benchmark_summary(results)
        summary_path = output_path / "benchmark_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        logger.info(f"\nBenchmark summary saved to: {summary_path}")
    
    return results


def _generate_benchmark_summary(results: Dict[str, EvaluationResult]) -> str:
    """Generate summary report for benchmark results."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("  ViDoRe Benchmark Summary")
    lines.append("=" * 80)
    lines.append("")
    
    # Header
    lines.append(f"{'Dataset':<45} {'MRR':<10} {'R@5':<10} {'R@10':<10} {'NDCG@10':<10}")
    lines.append("-" * 80)
    
    # Results per dataset
    for dataset_name, result in results.items():
        short_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        mrr = result.metrics.get("MRR", 0)
        r5 = result.metrics.get("Recall@5", 0)
        r10 = result.metrics.get("Recall@10", 0)
        ndcg10 = result.metrics.get("NDCG@10", 0)
        lines.append(f"{short_name:<45} {mrr:<10.4f} {r5:<10.4f} {r10:<10.4f} {ndcg10:<10.4f}")
    
    lines.append("-" * 80)
    
    # Average
    if results:
        avg_mrr = sum(r.metrics.get("MRR", 0) for r in results.values()) / len(results)
        avg_r5 = sum(r.metrics.get("Recall@5", 0) for r in results.values()) / len(results)
        avg_r10 = sum(r.metrics.get("Recall@10", 0) for r in results.values()) / len(results)
        avg_ndcg10 = sum(r.metrics.get("NDCG@10", 0) for r in results.values()) / len(results)
        lines.append(f"{'AVERAGE':<45} {avg_mrr:<10.4f} {avg_r5:<10.4f} {avg_r10:<10.4f} {avg_ndcg10:<10.4f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL RAG Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on a single ViDoRe dataset
  python -m qwen3_vl_retrieval.scripts.evaluate \\
      --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \\
      --dataset vidore/docvqa_test_subsampled \\
      --output_dir ./evaluation_results

  # Evaluate on custom dataset
  python -m qwen3_vl_retrieval.scripts.evaluate \\
      --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \\
      --data_path ./my_dataset.json \\
      --output_dir ./evaluation_results

  # Run full ViDoRe benchmark
  python -m qwen3_vl_retrieval.scripts.evaluate \\
      --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \\
      --run_benchmark \\
      --output_dir ./evaluation_results

  # Compare with/without binary quantization
  python -m qwen3_vl_retrieval.scripts.evaluate \\
      --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \\
      --dataset vidore/docvqa_test_subsampled \\
      --compare_quantization \\
      --output_dir ./evaluation_results
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
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., vidore/docvqa_test_subsampled)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to local JSON dataset file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    
    # Benchmark arguments
    parser.add_argument(
        "--run_benchmark",
        action="store_true",
        help="Run evaluation on all ViDoRe benchmark datasets"
    )
    parser.add_argument(
        "--benchmark_datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific benchmark datasets to evaluate"
    )
    
    # Comparison arguments
    parser.add_argument(
        "--compare_quantization",
        action="store_true",
        help="Compare performance with and without binary quantization"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--first_stage_top_k",
        type=int,
        default=100,
        help="Number of candidates from first stage (binary visual retrieval)"
    )
    parser.add_argument(
        "--second_stage_top_k",
        type=int,
        default=10,
        help="Number of final results"
    )
    parser.add_argument(
        "--use_binary_quantization",
        action="store_true",
        default=True,
        help="Use binary quantization for second stage"
    )
    parser.add_argument(
        "--no_binary_quantization",
        action="store_true",
        help="Disable binary quantization"
    )
    parser.add_argument(
        "--skip_first_stage",
        action="store_true",
        help="Skip first stage and use all documents for MaxSim ranking (slower but more accurate)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--report_format",
        type=str,
        default="text",
        choices=["text", "json", "markdown"],
        help="Report format"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or auto)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for encoding"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*70)
    print("  Qwen3-VL RAG Retrieval System - Evaluation")
    print("="*70 + "\n")
    
    # Determine binary quantization setting
    use_binary = args.use_binary_quantization and not args.no_binary_quantization
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.run_benchmark:
        # Run full benchmark
        logger.info("Running ViDoRe benchmark evaluation...")
        results = run_vidore_benchmark(
            model_path=args.model_path,
            lora_path=args.lora_path,
            device=args.device,
            output_dir=args.output_dir,
            datasets=args.benchmark_datasets,
            max_samples=args.max_samples,
        )
        
        print("\nBenchmark evaluation complete!")
        print(f"Results saved to: {args.output_dir}")
        
    elif args.compare_quantization:
        # Compare with/without quantization
        if not args.dataset and not args.data_path:
            logger.error("Either --dataset or --data_path must be provided for comparison")
            sys.exit(1)
        
        logger.info("Running quantization comparison...")
        
        dataset = ViDoReDataset(
            dataset_name=args.dataset,
            data_path=args.data_path,
            max_samples=args.max_samples,
        )
        
        result_with, result_without = compare_quantization(
            model_path=args.model_path,
            dataset=dataset,
            lora_path=args.lora_path,
            device=args.device,
            output_dir=args.output_dir,
        )
        
        print("\nQuantization comparison complete!")
        print(f"Results saved to: {args.output_dir}")
        
    else:
        # Single dataset evaluation
        if not args.dataset and not args.data_path:
            logger.error("Either --dataset or --data_path must be provided")
            sys.exit(1)
        
        logger.info("Running single dataset evaluation...")
        
        # Load dataset
        dataset = ViDoReDataset(
            dataset_name=args.dataset,
            data_path=args.data_path,
            max_samples=args.max_samples,
        )
        
        # Create evaluator
        evaluator = RetrievalEvaluator(
            model_path=args.model_path,
            lora_path=args.lora_path,
            device=args.device,
            use_binary_quantization=use_binary,
            first_stage_top_k=args.first_stage_top_k,
            second_stage_top_k=args.second_stage_top_k,
            skip_first_stage=args.skip_first_stage,
        )
        
        # Run evaluation
        result = evaluator.evaluate(dataset, batch_size=args.batch_size)
        
        # Generate report
        report_ext = {"text": ".txt", "json": ".json", "markdown": ".md"}[args.report_format]
        report_path = output_path / f"evaluation_report{report_ext}"
        report = generate_report(result, str(report_path), format=args.report_format)
        
        # Also save JSON result
        json_path = output_path / "evaluation_result.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Print report
        print("\n" + report)
        print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
