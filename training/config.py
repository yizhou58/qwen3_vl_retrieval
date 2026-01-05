"""
Training Configuration Classes.

Configuration classes for ColQwen3VL model training.

Requirements: 2.5, 2.6
- Support gradient checkpointing to reduce memory usage
- Support configurable LoRA rank and alpha
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from pathlib import Path


@dataclass
class LoRAConfig:
    """
    LoRA (Low-Rank Adaptation) configuration.
    
    Requirements:
        2.6: Support configurable LoRA rank and alpha
    """
    
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    modules_to_save: List[str] = field(default_factory=lambda: ["custom_text_proj"])
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"


@dataclass
class ColModelTrainingConfig:
    """Training configuration for ColQwen3VL model."""
    
    model_name_or_path: str = "~/autodl-tmp/checkpoints/Qwen/Qwen3-VL-4B-Instruct/"
    output_dir: str = "./outputs"
    use_lora: bool = True
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_train_epochs: int = 3
    max_steps: int = -1
    
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True
    
    temperature: float = 0.02
    max_length: int = 512
    max_num_visual_tokens: Optional[int] = None
    
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500
    
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    seed: int = 42
    
    dataloader_num_workers: int = 4
    use_flash_attention: bool = True
    use_hard_negatives: bool = False
    num_hard_negatives: int = 5
    
    def __post_init__(self):
        self.model_name_or_path = str(Path(self.model_name_or_path).expanduser())
        self.output_dir = str(Path(self.output_dir).expanduser())
    
    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
    
    def to_training_arguments(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "bf16": self.bf16,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "lr_scheduler_type": self.lr_scheduler_type,
            "seed": self.seed,
            "remove_unused_columns": False,
        }
