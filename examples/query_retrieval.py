#!/usr/bin/env python
"""
Query Retrieval Example for Qwen3-VL RAG Retrieval System.

This script demonstrates how to:
1. Load a pre-built index
2. Perform two-stage retrieval (BM25 + MaxSim reranking)
3. Display retrieval results

Requirements: 8.3, 8.4
- Provide APIs for encoding queries
- Implement retrieval API

Usage:
    # Interactive query mode
    python -m qwen3_vl_retrieval.examples.query_retrieval \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --index_dir ./index_output \
        --interactive

    # Single query
    python -m qwen3_vl_retrieval.examples.query_retrieval \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --index_dir ./index_output \
        --query "What is the revenue for Q3 2024?"

    # Batch queries from file
    python -m qwen3_vl_retrieval.examples.query_retrieval \
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \
        --index_dir ./index_output \
        --query_file ./queries.txt \
        --output_file ./results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query documents using Qwen3-VL RAG retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode
    python -m qwen3_vl_retrieval.examples.query_retrieval \\
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \\
        --index_dir ./index \\
        --interactive

    # Single query
    python -m qwen3_vl_retrieval.examples.query_retrieval \\
        --model_path ~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/ \\
        --index_dir ./index \\
        --query "Find documents about machine learning"
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
    
    # Index arguments
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory containing the pre-built index"
    )
    
    # Query arguments
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to execute"
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default=None,
        help="File containing queries (one per line)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    # Retrieval arguments
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    parser.add_argument(
        "--first_stage_top_k",
        type=int,
        default=100,
        help="Number of candidates from first stage (default: 100)"
    )
    parser.add_argument(
        "--binary_rescore_ratio",
        type=int,
        default=10,
        help="Ratio for binary pre-filtering (default: 10)"
    )
    parser.add_argument(
        "--no_two_stage",
        action="store_true",
        help="Disable two-stage retrieval (use MaxSim only)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save results (JSON format)"
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or auto)"
    )
    
    return parser.parse_args()


def load_retriever(
    model_path: str,
    lora_path: Optional[str],
    index_dir: str,
    device: str,
):
    """
    Load the retrieval system.
    
    Args:
        model_path: Path to base model
        lora_path: Optional path to LoRA weights
        index_dir: Path to index directory
        device: Device to use
        
    Returns:
        Tuple of (model, processor, first_stage, second_stage, embedding_store)
    """
    from qwen3_vl_retrieval.inference.model_loader import load_model_for_inference
    from qwen3_vl_retrieval.retrieval.embedding_store import EmbeddingStore
    from qwen3_vl_retrieval.retrieval.first_stage_retriever import FirstStageRetriever
    from qwen3_vl_retrieval.retrieval.second_stage_reranker import SecondStageReranker
    
    index_dir = Path(index_dir)
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    model, processor = load_model_for_inference(
        model_name_or_path=str(Path(model_path).expanduser()),
        lora_path=lora_path,
        device_map=device if device != "cpu" else None,
    )
    logger.info("Model loaded successfully")
    
    # Load embedding store
    embedding_store_path = index_dir / "embeddings"
    if not embedding_store_path.exists():
        raise FileNotFoundError(f"Embedding store not found at {embedding_store_path}")
    
    embedding_store = EmbeddingStore(
        storage_path=str(embedding_store_path),
        dim=128,
    )
    logger.info(f"Loaded embedding store with {len(embedding_store)} documents")
    
    # Load first-stage retriever
    first_stage_path = index_dir / "first_stage_index"
    if not first_stage_path.exists():
        raise FileNotFoundError(f"First-stage index not found at {first_stage_path}")
    
    first_stage = FirstStageRetriever(
        method="bm25",
        index_path=str(first_stage_path),
    )
    logger.info(f"Loaded first-stage index with {len(first_stage)} documents")
    
    # Create second-stage reranker
    second_stage = SecondStageReranker(
        model=model,
        processor=processor,
        embedding_store=embedding_store,
        use_binary_quantization=True,
        device=device,
    )
    
    return model, processor, first_stage, second_stage, embedding_store


def retrieve(
    query: str,
    first_stage: "FirstStageRetriever",
    second_stage: "SecondStageReranker",
    top_k: int = 10,
    first_stage_top_k: int = 100,
    binary_rescore_ratio: int = 10,
    use_two_stage: bool = True,
) -> List[Tuple[str, float]]:
    """
    Perform retrieval for a query.
    
    Args:
        query: Query text
        first_stage: First-stage retriever
        second_stage: Second-stage reranker
        top_k: Number of results to return
        first_stage_top_k: Number of candidates from first stage
        binary_rescore_ratio: Ratio for binary pre-filtering
        use_two_stage: Whether to use two-stage retrieval
        
    Returns:
        List of (doc_id, score) tuples
    """
    if use_two_stage:
        # Stage 1: Fast recall with BM25
        first_stage_results = first_stage.retrieve(query, top_k=first_stage_top_k)
        candidate_doc_ids = [doc_id for doc_id, _, _ in first_stage_results]
        
        if not candidate_doc_ids:
            logger.warning("No candidates from first stage")
            return []
        
        # Stage 2: Precise reranking with MaxSim
        results = second_stage.rerank(
            query=query,
            candidate_doc_ids=candidate_doc_ids,
            top_k=top_k,
            binary_rescore_ratio=binary_rescore_ratio,
        )
    else:
        # Direct retrieval using second stage only
        all_doc_ids = second_stage.embedding_store.list_doc_ids()
        
        if not all_doc_ids:
            logger.warning("No documents in embedding store")
            return []
        
        results = second_stage.rerank(
            query=query,
            candidate_doc_ids=all_doc_ids,
            top_k=top_k,
            binary_rescore_ratio=binary_rescore_ratio,
        )
    
    return results


def format_results(
    query: str,
    results: List[Tuple[str, float]],
    first_stage: "FirstStageRetriever",
) -> str:
    """Format retrieval results for display."""
    lines = [
        f"\n{'='*60}",
        f"Query: {query}",
        f"{'='*60}",
        f"Found {len(results)} results:\n",
    ]
    
    for rank, (doc_id, score) in enumerate(results, 1):
        image_path = first_stage.image_paths.get(doc_id, "N/A")
        lines.append(f"  {rank}. [{score:.4f}] {doc_id}")
        lines.append(f"     Image: {image_path}")
    
    lines.append("")
    return "\n".join(lines)


def run_interactive(
    first_stage,
    second_stage,
    top_k: int,
    first_stage_top_k: int,
    binary_rescore_ratio: int,
    use_two_stage: bool,
):
    """Run interactive query mode."""
    print("\n" + "="*60)
    print("Qwen3-VL RAG Retrieval - Interactive Mode")
    print("="*60)
    print(f"Index contains {len(first_stage)} documents")
    print("Type 'quit' or 'exit' to stop")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("Enter query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            # Perform retrieval
            results = retrieve(
                query=query,
                first_stage=first_stage,
                second_stage=second_stage,
                top_k=top_k,
                first_stage_top_k=first_stage_top_k,
                binary_rescore_ratio=binary_rescore_ratio,
                use_two_stage=use_two_stage,
            )
            
            # Display results
            print(format_results(query, results, first_stage))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")


def main():
    """Main retrieval function."""
    args = parse_args()
    
    # Validate arguments
    if not args.query and not args.query_file and not args.interactive:
        logger.error("Must specify --query, --query_file, or --interactive")
        sys.exit(1)
    
    # Validate index directory
    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        sys.exit(1)
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load retrieval system
    model, processor, first_stage, second_stage, embedding_store = load_retriever(
        model_path=args.model_path,
        lora_path=args.lora_path,
        index_dir=args.index_dir,
        device=device,
    )
    
    use_two_stage = not args.no_two_stage
    
    if args.interactive:
        # Interactive mode
        run_interactive(
            first_stage=first_stage,
            second_stage=second_stage,
            top_k=args.top_k,
            first_stage_top_k=args.first_stage_top_k,
            binary_rescore_ratio=args.binary_rescore_ratio,
            use_two_stage=use_two_stage,
        )
    else:
        # Batch mode
        queries = []
        
        if args.query:
            queries.append(args.query)
        
        if args.query_file:
            with open(args.query_file, "r", encoding="utf-8") as f:
                queries.extend([line.strip() for line in f if line.strip()])
        
        logger.info(f"Processing {len(queries)} queries...")
        
        all_results = []
        
        for query in queries:
            results = retrieve(
                query=query,
                first_stage=first_stage,
                second_stage=second_stage,
                top_k=args.top_k,
                first_stage_top_k=args.first_stage_top_k,
                binary_rescore_ratio=args.binary_rescore_ratio,
                use_two_stage=use_two_stage,
            )
            
            # Display results
            print(format_results(query, results, first_stage))
            
            # Store for output
            all_results.append({
                "query": query,
                "results": [
                    {
                        "doc_id": doc_id,
                        "score": score,
                        "image_path": first_stage.image_paths.get(doc_id, ""),
                    }
                    for doc_id, score in results
                ]
            })
        
        # Save results if output file specified
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
