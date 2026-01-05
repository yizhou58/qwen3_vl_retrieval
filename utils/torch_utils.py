"""
PyTorch utility functions for device detection and memory management.

Requirements: 8.2, 8.5
- Support half-precision (bfloat16) inference for memory efficiency
- Support Flash Attention 2 if available
"""

import gc
import os
import random
from typing import Dict, Optional, Union, Literal

import numpy as np
import torch


def get_device() -> torch.device:
    """
    Get the best available device for computation.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_map(
    model_name: str = "auto",
    max_memory: Optional[Dict[int, str]] = None,
) -> Union[str, Dict[str, Union[int, str]]]:
    """
    Get device map for model loading with accelerate.
    
    Args:
        model_name: Model identifier or "auto" for automatic mapping.
        max_memory: Optional dict mapping device IDs to max memory strings.
                   e.g., {0: "20GB", 1: "20GB", "cpu": "30GB"}
    
    Returns:
        Device map configuration for model loading.
    """
    if not torch.cuda.is_available():
        return "cpu"
    
    if max_memory is not None:
        return "auto"
    
    # For single GPU, use device 0
    if torch.cuda.device_count() == 1:
        return {"": 0}
    
    # For multi-GPU, use auto device map
    return "auto"


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache and run garbage collection.
    
    Useful for freeing memory between model operations.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info() -> Dict[str, Union[int, float]]:
    """
    Get current GPU memory usage information.
    
    Returns:
        Dict containing memory info:
        - total: Total GPU memory in bytes
        - allocated: Currently allocated memory in bytes
        - reserved: Reserved memory in bytes
        - free: Free memory in bytes
        - utilization: Memory utilization percentage
    """
    if not torch.cuda.is_available():
        return {
            "total": 0,
            "allocated": 0,
            "reserved": 0,
            "free": 0,
            "utilization": 0.0,
        }
    
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    free = total - allocated
    utilization = (allocated / total) * 100 if total > 0 else 0.0
    
    return {
        "total": total,
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "utilization": utilization,
    }


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_dtype(
    dtype_str: Literal["float32", "float16", "bfloat16", "auto"] = "auto"
) -> torch.dtype:
    """
    Get torch dtype from string specification.
    
    Args:
        dtype_str: String specifying the dtype.
                  "auto" will use bfloat16 if supported, else float16.
    
    Returns:
        Corresponding torch.dtype.
    """
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "auto":
        # Use bfloat16 if supported (Ampere+ GPUs)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")


def is_flash_attention_available() -> bool:
    """
    Check if Flash Attention 2 is available.
    
    Requirements: 8.5 - Support Flash Attention 2 for faster inference.
    
    Returns:
        True if Flash Attention 2 is available, False otherwise.
    """
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def get_attention_implementation() -> str:
    """
    Get the best available attention implementation.
    
    Returns:
        "flash_attention_2" if available, otherwise "sdpa" or "eager".
    """
    if is_flash_attention_available():
        return "flash_attention_2"
    
    # Check for SDPA (Scaled Dot Product Attention) in PyTorch 2.0+
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"
    
    return "eager"


def estimate_model_memory(
    num_params: int,
    dtype: torch.dtype = torch.bfloat16,
    include_optimizer: bool = False,
    optimizer_states: int = 2,
) -> int:
    """
    Estimate memory required for a model.
    
    Args:
        num_params: Number of model parameters.
        dtype: Data type for model weights.
        include_optimizer: Whether to include optimizer state memory.
        optimizer_states: Number of optimizer states (2 for Adam).
    
    Returns:
        Estimated memory in bytes.
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }.get(dtype, 4)
    
    model_memory = num_params * bytes_per_param
    
    if include_optimizer:
        # Optimizer states are typically float32
        optimizer_memory = num_params * 4 * optimizer_states
        return model_memory + optimizer_memory
    
    return model_memory
