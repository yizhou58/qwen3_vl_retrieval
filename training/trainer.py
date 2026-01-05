"""
ColQwen3VL Trainer Implementation.

Custom trainer for ColQwen3VL retrieval model with contrastive learning.

Requirements: 2.5, 2.7
- Use gradient checkpointing to reduce memory usage
- Save LoRA adapter weights separately from base model
"""

from typing import Any, Dict, Optional, Union
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from transformers import Trainer, TrainingArguments
except ImportError:
    Trainer = object
    TrainingArguments = object

from qwen3_vl_retrieval.training.losses import ColbertLoss, HardNegativeLoss
from qwen3_vl_retrieval.training.config import ColModelTrainingConfig

logger = logging.getLogger(__name__)


class ColQwen3VLTrainer(Trainer):
    """Custom trainer for ColQwen3VL retrieval model."""
    
    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        config: ColModelTrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        self.processor = processor
        self.config = config
        
        if config.use_hard_negatives:
            self.loss_fn = HardNegativeLoss(temperature=config.temperature)
        else:
            self.loss_fn = ColbertLoss(temperature=config.temperature)
        
        training_args = TrainingArguments(**config.to_training_arguments())
        
        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs,
        )
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple]:
        """Compute contrastive loss for a batch."""
        query_outputs = model(
            input_ids=inputs["query_input_ids"],
            attention_mask=inputs["query_attention_mask"],
        )
        
        doc_outputs = model(
            input_ids=inputs["doc_input_ids"],
            attention_mask=inputs["doc_attention_mask"],
            pixel_values=inputs["doc_pixel_values"],
            image_grid_thw=inputs["doc_image_grid_thw"],
        )
        
        query_mask = inputs["query_attention_mask"].bool()
        doc_mask = inputs["doc_attention_mask"].bool()
        
        loss = self.loss_fn(query_outputs, doc_outputs, query_mask, doc_mask)
        
        if return_outputs:
            return loss, {"query_embeddings": query_outputs, "doc_embeddings": doc_outputs}
        return loss
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model, handling LoRA adapters separately."""
        output_dir = output_dir or self.args.output_dir
        
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_dir)
            logger.info(f"Saved LoRA adapters to {output_dir}")
        else:
            super().save_model(output_dir, _internal_call)
        
        if self.processor is not None and hasattr(self.processor, '_processor'):
            self.processor._processor.save_pretrained(output_dir)
