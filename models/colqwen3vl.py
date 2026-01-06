"""
ColQwen3VL Model Implementation.

ColPali-style retrieval model based on Qwen3-VL-4B.
Extends Qwen3-VL with a projection layer for multi-vector embeddings.

Requirements: 1.1, 1.2, 1.4
- Extend Qwen3-VL-4B with projection layer mapping to 128-dimensional embeddings
- Handle dynamic visual token counts from M-RoPE mechanism
- Apply L2 normalization to all output embeddings
"""

from typing import ClassVar, Optional, List, Union
import logging

import torch
from torch import nn

# Try multiple import paths for Qwen3-VL
Qwen3VLConfig = None
Qwen3VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
except ImportError:
    pass

if Qwen3VLForConditionalGeneration is None:
    try:
        from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    except ImportError:
        pass

if Qwen3VLForConditionalGeneration is None:
    try:
        # Try AutoModel as fallback
        from transformers import AutoModelForVision2Seq, AutoConfig
        Qwen3VLForConditionalGeneration = AutoModelForVision2Seq
        Qwen3VLConfig = AutoConfig
    except ImportError:
        pass

# Use standard logging to avoid circular import issues
logger = logging.getLogger(__name__)


class ColQwen3VL(nn.Module):
    """
    ColPali-style retrieval model based on Qwen3-VL.
    
    Qwen3-VL is a Decoder-only architecture. For retrieval:
    - Document encoding: Extract visual patch tokens and apply projection
    - Query encoding: Extract query text tokens (excluding system prompts)
    
    Attributes:
        dim: Output embedding dimension (default: 128)
        custom_text_proj: Linear projection layer (FULLY TRAINABLE, not LoRA)
        mask_non_image_embeddings: Whether to mask non-image tokens for doc encoding
        
    Requirements:
        1.1: Projection layer maps hidden states to 128-dimensional embeddings
        1.2: Handle dynamic visual token counts from M-RoPE mechanism
        1.4: Apply L2 normalization to all output embeddings
    """
    
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related
    
    def __init__(
        self,
        config: Optional["Qwen3VLConfig"] = None,
        mask_non_image_embeddings: bool = False,
        dim: int = 128,
    ):
        """
        Initialize ColQwen3VL model.
        
        Args:
            config: Qwen3VLConfig configuration object
            mask_non_image_embeddings: Whether to mask non-image tokens during forward
            dim: Output embedding dimension (default: 128)
        """
        super().__init__()
        
        if Qwen3VLModel is None:
            raise ImportError(
                "Qwen3-VL model requires transformers >= 4.45.0. "
                "Please upgrade: pip install transformers>=4.45.0"
            )
        
        self.config = config
        self.dim = dim
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.padding_side = "left"
        
        # Initialize the base Qwen3-VL model
        self.model = Qwen3VLModel(config) if config is not None else None
        
        # Projection layer: maps hidden_size to embedding dimension
        # This is FULLY TRAINABLE (not LoRA wrapped)
        hidden_size = config.text_config.hidden_size if config is not None else 4096
        self.custom_text_proj = nn.Linear(hidden_size, self.dim)
        
        # Initialize projection layer weights
        nn.init.normal_(self.custom_text_proj.weight, std=0.02)
        nn.init.zeros_(self.custom_text_proj.bias)
        
        # Store image token ID for masking
        self._image_token_id = config.image_token_id if config is not None else 151655
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        mask_non_image_embeddings: bool = False,
        dim: int = 128,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, dict]] = None,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ) -> "ColQwen3VL":
        """
        Load a pretrained ColQwen3VL model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model or model identifier
            mask_non_image_embeddings: Whether to mask non-image tokens
            dim: Output embedding dimension
            torch_dtype: Data type for model weights
            device_map: Device mapping for model parallelism
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")
            **kwargs: Additional arguments passed to from_pretrained
            
        Returns:
            Initialized ColQwen3VL model
        """
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen3-VL model requires transformers >= 4.45.0. "
                "Please upgrade: pip install transformers>=4.45.0"
            )
        
        # Load the base model
        logger.info(f"Loading Qwen3-VL model from {pretrained_model_name_or_path}")
        
        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        load_kwargs.update(kwargs)
        
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            **load_kwargs,
        )
        
        # Create ColQwen3VL instance
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        
        instance.config = base_model.config
        instance.dim = dim
        instance.mask_non_image_embeddings = mask_non_image_embeddings
        instance.padding_side = "left"
        instance.model = base_model
        
        # Initialize projection layer
        hidden_size = base_model.config.text_config.hidden_size
        instance.custom_text_proj = nn.Linear(hidden_size, dim)
        nn.init.normal_(instance.custom_text_proj.weight, std=0.02)
        nn.init.zeros_(instance.custom_text_proj.bias)
        
        # Move projection layer to same device/dtype as model
        if hasattr(base_model, "device"):
            instance.custom_text_proj = instance.custom_text_proj.to(base_model.device)
        if torch_dtype is not None:
            instance.custom_text_proj = instance.custom_text_proj.to(torch_dtype)
        
        instance._image_token_id = base_model.config.image_token_id
        
        logger.info(f"ColQwen3VL model loaded with dim={dim}")
        
        return instance
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass returning L2-normalized multi-vector embeddings.
        
        Logic:
        1. Run base Qwen3-VL model to get hidden_states
        2. Apply custom_text_proj to hidden_states
        3. Normalize embeddings (L2)
        4. IF processing document (pixel_values present):
           - Apply attention mask to zero out padding tokens
           - Optionally mask non-image tokens
        5. IF processing query (no pixel_values):
           - Apply attention mask to zero out padding tokens
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            pixel_values: Image pixel values (variable shape due to dynamic resolution)
            image_grid_thw: Image grid dimensions (batch_size, 3) for T, H, W
            
        Returns:
            Normalized embeddings (batch_size, seq_len, dim)
            
        Requirements:
            1.1: Output embeddings have dimension 128
            1.2: Handle dynamic visual token counts
            1.4: Apply L2 normalization
        """
        # Handle the custom "pixel_values" input through unpadding
        # Qwen3-VL expects concatenated pixel values without padding
        if pixel_values is not None and image_grid_thw is not None:
            # Calculate actual number of patches per image
            # image_grid_thw: (batch_size, 3) where 3 = (temporal, height, width)
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            
            # Unpad pixel_values: concatenate only valid patches
            pixel_values = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(pixel_values, offsets)],
                dim=0,
            )
        
        # Remove kwargs that base model doesn't expect
        kwargs.pop("return_dict", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        
        # Get last hidden states
        # For Qwen3VLForConditionalGeneration, hidden_states is a tuple
        # The last element is the final layer's hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # Last layer
        else:
            raise ValueError("Cannot extract hidden states from model output")
        
        # Apply projection layer
        proj = self.custom_text_proj(hidden_states)  # (batch_size, seq_len, dim)
        
        # L2 normalization (Requirement 1.4)
        # Add small epsilon to avoid division by zero
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Apply attention mask to zero out padding tokens
        proj = proj * attention_mask.unsqueeze(-1)  # (batch_size, seq_len, dim)
        
        # Optionally mask non-image embeddings for document encoding
        if pixel_values is not None and self.mask_non_image_embeddings:
            # Create mask for image tokens only
            image_mask = (input_ids == self._image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        
        return proj
    
    @property
    def patch_size(self) -> int:
        """Get the patch size from the visual encoder."""
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "config"):
            return self.model.visual.config.patch_size
        return 14  # Default patch size
    
    @property
    def spatial_merge_size(self) -> int:
        """Get the spatial merge size from the visual encoder."""
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "config"):
            return getattr(self.model.visual.config, "spatial_merge_size", 2)
        return 2  # Default merge size
    
    @property
    def image_token_id(self) -> int:
        """Get the image token ID."""
        return self._image_token_id
    
    @property
    def hidden_size(self) -> int:
        """Get the hidden size of the model."""
        if self.config is not None:
            return self.config.text_config.hidden_size
        return 4096
    
    def get_input_embeddings(self):
        """Get input embeddings from the base model."""
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set input embeddings for the base model."""
        self.model.set_input_embeddings(value)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        return next(self.parameters()).dtype
    
    def enable_lora_training(
        self,
        rank: int = 32,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> "ColQwen3VL":
        """
        Set up LoRA for efficient fine-tuning.
        
        CRITICAL:
        - Freezes ViT (visual encoder) parameters
        - Adds LoRA to LLM layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
        - Keeps custom_text_proj FULLY TRAINABLE (not LoRA wrapped)
        
        Args:
            rank: LoRA rank (default: 32)
            alpha: LoRA alpha scaling factor (default: 32)
            dropout: LoRA dropout rate (default: 0.05)
            target_modules: List of module names to apply LoRA to. If None, uses default
                           LLM attention and MLP modules.
            
        Returns:
            Self with LoRA applied
            
        Requirements:
            2.1: Apply LoRA adapters to q, k, v, o projection layers and MLP layers
            2.2: Treat Projection_Layer as fully trainable (not LoRA)
            2.3: Freeze visual encoder (ViT) parameters during training
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "PEFT library is required for LoRA training. "
                "Please install: pip install peft>=0.7.0"
            )
        
        # Step 1: Freeze ViT (visual encoder) parameters (Requirement 2.3)
        # The visual encoder is typically at self.model.visual
        if hasattr(self.model, "visual"):
            logger.info("Freezing visual encoder (ViT) parameters")
            for param in self.model.visual.parameters():
                param.requires_grad = False
        else:
            logger.warning("Visual encoder not found at self.model.visual")
        
        # Step 2: Define target modules for LoRA (Requirement 2.1)
        # These are the LLM attention and MLP layers
        if target_modules is None:
            target_modules = [
                "q_proj",      # Query projection
                "k_proj",      # Key projection
                "v_proj",      # Value projection
                "o_proj",      # Output projection
                "gate_proj",   # MLP gate projection
                "up_proj",     # MLP up projection
                "down_proj",   # MLP down projection
            ]
        
        # Step 3: Create LoRA configuration
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",  # For embedding models
        )
        
        logger.info(
            f"Applying LoRA with rank={rank}, alpha={alpha}, "
            f"target_modules={target_modules}"
        )
        
        # Step 4: Apply LoRA to the base model
        # We need to apply PEFT to self.model (the Qwen3VLModel), not to self
        # This ensures custom_text_proj stays outside PEFT and remains fully trainable
        self.model = get_peft_model(self.model, lora_config)
        
        # Step 5: Ensure custom_text_proj remains fully trainable (Requirement 2.2)
        # It's already outside the PEFT model, but let's explicitly set requires_grad
        for param in self.custom_text_proj.parameters():
            param.requires_grad = True
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LoRA enabled: {trainable_params:,} trainable parameters "
            f"out of {total_params:,} total ({100 * trainable_params / total_params:.2f}%)"
        )
        
        return self
    
    def save_lora_weights(self, save_path: str) -> None:
        """
        Save LoRA adapter weights separately from the base model.
        
        Args:
            save_path: Directory path to save LoRA weights
            
        Requirements:
            2.7: Save LoRA adapter weights separately from the base model
        """
        import os
        
        if not hasattr(self.model, "save_pretrained"):
            raise ValueError(
                "Model does not have LoRA adapters. "
                "Call enable_lora_training() first."
            )
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save LoRA adapter weights
        self.model.save_pretrained(save_path)
        
        # Save projection layer weights separately
        proj_path = os.path.join(save_path, "custom_text_proj.pt")
        torch.save(self.custom_text_proj.state_dict(), proj_path)
        
        logger.info(f"LoRA weights saved to {save_path}")
    
    def load_lora_weights(self, load_path: str) -> "ColQwen3VL":
        """
        Load LoRA adapter weights.
        
        Args:
            load_path: Directory path containing LoRA weights
            
        Returns:
            Self with LoRA weights loaded
        """
        import os
        
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "PEFT library is required for loading LoRA weights. "
                "Please install: pip install peft>=0.7.0"
            )
        
        # Load LoRA adapter weights
        self.model = PeftModel.from_pretrained(self.model, load_path)
        
        # Load projection layer weights if available
        proj_path = os.path.join(load_path, "custom_text_proj.pt")
        if os.path.exists(proj_path):
            self.custom_text_proj.load_state_dict(torch.load(proj_path))
            logger.info(f"Loaded projection layer weights from {proj_path}")
        
        logger.info(f"LoRA weights loaded from {load_path}")
        
        return self
    
    def merge_and_unload_lora(self) -> "ColQwen3VL":
        """
        Merge LoRA weights into the base model and unload adapters.
        
        This is useful for inference when you want to avoid the overhead
        of LoRA forward passes.
        
        Returns:
            Self with LoRA merged into base model
        """
        if hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
            logger.info("LoRA weights merged into base model")
        else:
            logger.warning("Model does not have LoRA adapters to merge")
        
        return self
    
    def print_trainable_parameters(self) -> None:
        """Print the number of trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                logger.debug(f"Trainable: {name} - {param.numel():,} params")
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.4f}%"
        )
