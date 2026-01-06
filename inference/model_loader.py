"""
Model Loading Utilities for ColQwen3VL.

Provides comprehensive utilities for loading ColQwen3VL models with:
- LoRA weight loading and merging
- bfloat16/float16 inference support
- Flash Attention 2 detection and configuration
- Quantization support (4-bit, 8-bit)

Requirements: 8.1, 8.2, 8.5
- Support loading fine-tuned LoRA weights on top of base Qwen3-VL model
- Support half-precision (bfloat16) inference for memory efficiency
- Use Flash Attention 2 if available for faster inference
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal

import torch

from ..models.colqwen3vl import ColQwen3VL
from ..models.processing_colqwen3vl import ColQwen3VLProcessor

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Utility class for loading ColQwen3VL models with various configurations.
    
    Supports:
    - Loading base models with optimal dtype and attention
    - Loading and merging LoRA adapters
    - Quantization (4-bit, 8-bit) for memory efficiency
    - Flash Attention 2 for faster inference
    
    Example:
        >>> loader = ModelLoader()
        >>> model, processor = loader.load(
        ...     "Qwen/Qwen3-VL-4B-Instruct",
        ...     lora_path="./lora_weights",
        ...     torch_dtype=torch.bfloat16,
        ... )
    """
    
    def __init__(self):
        """Initialize ModelLoader."""
        self._flash_attn_available = None
        self._sdpa_available = None

    @property
    def flash_attention_available(self) -> bool:
        """Check if Flash Attention 2 is available."""
        if self._flash_attn_available is None:
            try:
                import flash_attn
                self._flash_attn_available = True
                logger.debug("Flash Attention 2 is available")
            except ImportError:
                self._flash_attn_available = False
                logger.debug("Flash Attention 2 is not available")
        return self._flash_attn_available
    
    @property
    def sdpa_available(self) -> bool:
        """Check if SDPA (Scaled Dot Product Attention) is available."""
        if self._sdpa_available is None:
            self._sdpa_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            if self._sdpa_available:
                logger.debug("SDPA is available (PyTorch 2.0+)")
            else:
                logger.debug("SDPA is not available")
        return self._sdpa_available
    
    def detect_best_attention(self) -> str:
        """
        Detect the best available attention implementation.
        
        Priority:
        1. Flash Attention 2 (fastest, requires flash-attn package)
        2. SDPA (PyTorch 2.0+, good performance)
        3. Eager (fallback, slowest)
        
        Returns:
            Attention implementation string
            
        Requirements:
            8.5: Use Flash Attention 2 if available
        """
        if self.flash_attention_available:
            return "flash_attention_2"
        elif self.sdpa_available:
            return "sdpa"
        else:
            return "eager"
    
    def detect_best_dtype(
        self,
        prefer_bfloat16: bool = True,
    ) -> torch.dtype:
        """
        Detect the best available dtype for inference.
        
        Args:
            prefer_bfloat16: Whether to prefer bfloat16 over float16
            
        Returns:
            Best available dtype
            
        Requirements:
            8.2: Support half-precision (bfloat16) inference
        """
        if prefer_bfloat16 and torch.cuda.is_available():
            # Check if GPU supports bfloat16
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        
        if torch.cuda.is_available():
            return torch.float16
        
        return torch.float32

    def load(
        self,
        model_name_or_path: str,
        lora_path: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, dict]] = None,
        attn_implementation: Optional[str] = None,
        max_num_visual_tokens: Optional[int] = None,
        merge_lora: bool = False,
        quantization_config: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[ColQwen3VL, ColQwen3VLProcessor]:
        """
        Load ColQwen3VL model and processor with optimal settings.
        
        Args:
            model_name_or_path: Path to base Qwen3-VL model or HuggingFace model ID
            lora_path: Optional path to LoRA adapter weights
            torch_dtype: Data type for model weights (auto-detected if None)
            device_map: Device mapping ("auto", "cuda", or custom dict)
            attn_implementation: Attention implementation (auto-detected if None)
            max_num_visual_tokens: Maximum number of visual tokens per image
            merge_lora: Whether to merge LoRA weights into base model
            quantization_config: Optional quantization config dict for 4-bit/8-bit
            **kwargs: Additional arguments passed to model loading
            
        Returns:
            Tuple of (model, processor)
            
        Requirements:
            8.1: Support loading fine-tuned LoRA weights
            8.2: Support half-precision (bfloat16) inference
            8.5: Use Flash Attention 2 if available
        """
        # Auto-detect dtype if not specified
        if torch_dtype is None:
            torch_dtype = self.detect_best_dtype()
        
        # Auto-detect attention implementation if not specified
        if attn_implementation is None:
            attn_implementation = self.detect_best_attention()
        
        # Auto-detect device map if not specified
        if device_map is None and torch.cuda.is_available():
            device_map = "auto"
        
        logger.info(f"Loading model from: {model_name_or_path}")
        logger.info(f"  dtype: {torch_dtype}")
        logger.info(f"  attention: {attn_implementation}")
        logger.info(f"  device_map: {device_map}")
        
        # Prepare quantization config if provided
        bnb_config = None
        if quantization_config is not None:
            bnb_config = self._create_quantization_config(quantization_config)
            logger.info(f"  quantization: {quantization_config.get('load_in_4bit', False) and '4-bit' or '8-bit'}")
        
        # Load base model
        model = ColQwen3VL.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            quantization_config=bnb_config,
            **kwargs,
        )
        
        # Load LoRA weights if provided
        if lora_path is not None:
            model = self.load_lora(model, lora_path, merge=merge_lora)
        
        # Load processor
        processor = ColQwen3VLProcessor.from_pretrained(
            model_name_or_path,
            max_num_visual_tokens=max_num_visual_tokens,
        )
        
        logger.info("Model and processor loaded successfully")
        
        return model, processor

    def load_lora(
        self,
        model: ColQwen3VL,
        lora_path: str,
        merge: bool = False,
    ) -> ColQwen3VL:
        """
        Load LoRA adapter weights onto a model.
        
        Args:
            model: Base ColQwen3VL model
            lora_path: Path to LoRA adapter weights
            merge: Whether to merge LoRA weights into base model
            
        Returns:
            Model with LoRA weights loaded
            
        Requirements:
            8.1: Support loading fine-tuned LoRA weights
        """
        logger.info(f"Loading LoRA weights from: {lora_path}")
        
        # Load LoRA weights
        model = model.load_lora_weights(lora_path)
        
        # Optionally merge LoRA into base model
        if merge:
            logger.info("Merging LoRA weights into base model")
            model = model.merge_and_unload_lora()
        
        return model
    
    def _create_quantization_config(
        self,
        config: dict,
    ):
        """
        Create BitsAndBytesConfig for quantization.
        
        Args:
            config: Dict with quantization settings
                - load_in_4bit: bool
                - load_in_8bit: bool
                - bnb_4bit_compute_dtype: str (e.g., "bfloat16")
                - bnb_4bit_quant_type: str (e.g., "nf4")
                - bnb_4bit_use_double_quant: bool
                
        Returns:
            BitsAndBytesConfig instance
        """
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "BitsAndBytesConfig requires transformers >= 4.30.0. "
                "Please upgrade: pip install transformers>=4.30.0"
            )
        
        # Map string dtype to torch dtype
        compute_dtype = config.get("bnb_4bit_compute_dtype", "bfloat16")
        if isinstance(compute_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            compute_dtype = dtype_map.get(compute_dtype, torch.bfloat16)
        
        return BitsAndBytesConfig(
            load_in_4bit=config.get("load_in_4bit", False),
            load_in_8bit=config.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        )

    def save_merged_model(
        self,
        model: ColQwen3VL,
        processor: ColQwen3VLProcessor,
        save_path: str,
        safe_serialization: bool = True,
    ) -> None:
        """
        Save a model with merged LoRA weights.
        
        Args:
            model: ColQwen3VL model (with LoRA merged)
            processor: ColQwen3VLProcessor
            save_path: Path to save the merged model
            safe_serialization: Whether to use safetensors format
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving merged model to: {save_path}")
        
        # Save model
        if hasattr(model.model, "save_pretrained"):
            model.model.save_pretrained(
                save_path,
                safe_serialization=safe_serialization,
            )
        
        # Save projection layer
        proj_path = save_path / "custom_text_proj.pt"
        torch.save(model.custom_text_proj.state_dict(), proj_path)
        
        # Save processor
        if hasattr(processor._processor, "save_pretrained"):
            processor._processor.save_pretrained(save_path)
        
        logger.info("Merged model saved successfully")


# Convenience functions

def load_model_for_inference(
    model_name_or_path: str,
    lora_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[Union[str, dict]] = None,
    use_flash_attention: bool = True,
    max_num_visual_tokens: Optional[int] = None,
    **kwargs,
) -> Tuple[ColQwen3VL, ColQwen3VLProcessor]:
    """
    Load model optimized for inference.
    
    Convenience function that automatically:
    - Uses bfloat16 dtype
    - Enables Flash Attention 2 if available
    - Loads LoRA weights if provided
    
    Args:
        model_name_or_path: Path to base model
        lora_path: Optional path to LoRA weights
        torch_dtype: Data type (default: auto-detect)
        device_map: Device mapping (default: "auto" if CUDA available)
        use_flash_attention: Whether to try using Flash Attention 2
        max_num_visual_tokens: Maximum visual tokens per image
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, processor)
    """
    loader = ModelLoader()
    
    # Determine attention implementation
    attn_impl = None
    if use_flash_attention:
        attn_impl = loader.detect_best_attention()
    
    return loader.load(
        model_name_or_path=model_name_or_path,
        lora_path=lora_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
        max_num_visual_tokens=max_num_visual_tokens,
        **kwargs,
    )


def load_model_quantized(
    model_name_or_path: str,
    lora_path: Optional[str] = None,
    bits: Literal[4, 8] = 4,
    max_num_visual_tokens: Optional[int] = None,
    **kwargs,
) -> Tuple[ColQwen3VL, ColQwen3VLProcessor]:
    """
    Load model with quantization for memory efficiency.
    
    Uses BitsAndBytes for 4-bit or 8-bit quantization.
    
    Args:
        model_name_or_path: Path to base model
        lora_path: Optional path to LoRA weights
        bits: Quantization bits (4 or 8)
        max_num_visual_tokens: Maximum visual tokens per image
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (model, processor)
    """
    loader = ModelLoader()
    
    quantization_config = {
        "load_in_4bit": bits == 4,
        "load_in_8bit": bits == 8,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }
    
    return loader.load(
        model_name_or_path=model_name_or_path,
        lora_path=lora_path,
        quantization_config=quantization_config,
        max_num_visual_tokens=max_num_visual_tokens,
        **kwargs,
    )


def check_system_capabilities() -> Dict:
    """
    Check system capabilities for model loading.
    
    Returns:
        Dict with capability information
    """
    loader = ModelLoader()
    
    capabilities = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "flash_attention_2": loader.flash_attention_available,
        "sdpa": loader.sdpa_available,
        "best_attention": loader.detect_best_attention(),
        "best_dtype": str(loader.detect_best_dtype()),
    }
    
    if torch.cuda.is_available():
        capabilities["cuda_device_name"] = torch.cuda.get_device_name(0)
        capabilities["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
        capabilities["bf16_supported"] = torch.cuda.is_bf16_supported()
    
    return capabilities


def estimate_memory_requirements(
    model_name_or_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    quantization_bits: Optional[int] = None,
) -> Dict:
    """
    Estimate memory requirements for loading a model.
    
    Args:
        model_name_or_path: Model identifier
        torch_dtype: Data type for weights
        quantization_bits: Optional quantization (4 or 8 bits)
        
    Returns:
        Dict with memory estimates
    """
    # Approximate parameter counts for Qwen3-VL models
    param_counts = {
        "Qwen3-VL-4B": 4_000_000_000,
        "Qwen3-VL-7B": 7_000_000_000,
        "Qwen3-VL-14B": 14_000_000_000,
    }
    
    # Try to detect model size from name
    num_params = None
    for name, count in param_counts.items():
        if name.lower() in model_name_or_path.lower():
            num_params = count
            break
    
    if num_params is None:
        num_params = 4_000_000_000  # Default to 4B
    
    # Calculate bytes per parameter
    if quantization_bits == 4:
        bytes_per_param = 0.5  # 4 bits = 0.5 bytes
    elif quantization_bits == 8:
        bytes_per_param = 1.0  # 8 bits = 1 byte
    elif torch_dtype == torch.float32:
        bytes_per_param = 4.0
    elif torch_dtype in (torch.float16, torch.bfloat16):
        bytes_per_param = 2.0
    else:
        bytes_per_param = 2.0
    
    model_memory = num_params * bytes_per_param
    
    # Add overhead for activations, gradients, etc.
    inference_overhead = 1.2  # 20% overhead for inference
    
    return {
        "estimated_params": num_params,
        "bytes_per_param": bytes_per_param,
        "model_memory_gb": model_memory / (1024**3),
        "inference_memory_gb": model_memory * inference_overhead / (1024**3),
        "dtype": str(torch_dtype),
        "quantization_bits": quantization_bits,
    }


# Export all public APIs
__all__ = [
    "ModelLoader",
    "load_model_for_inference",
    "load_model_quantized",
    "check_system_capabilities",
    "estimate_memory_requirements",
]
