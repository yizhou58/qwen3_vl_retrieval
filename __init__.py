"""
Qwen3-VL RAG Retrieval System

A two-stage multimodal document retrieval system based on Qwen3-VL-4B.
Uses ColPali-style late interaction architecture with LoRA fine-tuning.
"""

__version__ = "0.1.0"
__author__ = "Qwen3-VL RAG Team"

from typing import List, Tuple, Dict, Optional

# Model exports
from .models import ColQwen3VL

__all__ = [
    "__version__",
    "__author__",
    "ColQwen3VL",
]
