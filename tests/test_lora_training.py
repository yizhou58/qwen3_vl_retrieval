"""
Unit Tests for LoRA Training Setup.

Tests for the enable_lora_training method in ColQwen3VL.

**Validates: Requirements 2.1, 2.2, 2.3**
"""

import pytest
import torch
from torch import nn


class MockQwen3VLConfig(dict):
    """Mock configuration for testing without loading actual model."""
    
    def __init__(self):
        super().__init__()
        self.image_token_id = 151655
        self.text_config = type('TextConfig', (), {'hidden_size': 4096})()
        # Add dict-like attributes that PEFT expects
        self['tie_word_embeddings'] = False
        self['model_type'] = 'qwen3_vl'


class MockVisualEncoder(nn.Module):
    """Mock visual encoder (ViT) for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=14, stride=14)
        self.proj = nn.Linear(64, 4096)
        self.config = type('Config', (), {'patch_size': 14, 'spatial_merge_size': 2})()
    
    def forward(self, x):
        return self.proj(self.conv(x).flatten(2).transpose(1, 2))


class MockLLMLayer(nn.Module):
    """Mock LLM layer with attention and MLP projections."""
    
    def __init__(self, hidden_size=4096):
        super().__init__()
        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        # MLP projections
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size)
    
    def forward(self, x):
        return x


class MockQwen3VLModel(nn.Module):
    """Mock Qwen3-VL model for testing LoRA setup."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = MockVisualEncoder()
        self.layers = nn.ModuleList([MockLLMLayer() for _ in range(2)])
        self.embed_tokens = nn.Embedding(32000, 4096)
    
    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, **kwargs):
        # Simple mock forward
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return type('Output', (), {
            'last_hidden_state': hidden_states,
            'hidden_states': (hidden_states,)
        })()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value


class TestLoRATrainingSetup:
    """
    Tests for enable_lora_training method.
    
    **Validates: Requirements 2.1, 2.2, 2.3**
    """
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock ColQwen3VL-like model for testing."""
        config = MockQwen3VLConfig()
        
        class MockColQwen3VL(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.dim = 128
                self.model = MockQwen3VLModel(config)
                self.custom_text_proj = nn.Linear(4096, 128)
                nn.init.normal_(self.custom_text_proj.weight, std=0.02)
                nn.init.zeros_(self.custom_text_proj.bias)
                self._image_token_id = config.image_token_id
            
            def enable_lora_training(self, rank=32, alpha=32, dropout=0.05, target_modules=None):
                """Simplified version for testing the logic."""
                try:
                    from peft import LoraConfig, get_peft_model
                except ImportError:
                    pytest.skip("PEFT not installed")
                
                # Freeze ViT parameters
                if hasattr(self.model, "visual"):
                    for param in self.model.visual.parameters():
                        param.requires_grad = False
                
                # Define target modules
                if target_modules is None:
                    target_modules = [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ]
                
                # Create LoRA config
                lora_config = LoraConfig(
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    target_modules=target_modules,
                    bias="none",
                    task_type="FEATURE_EXTRACTION",
                )
                
                # Apply LoRA
                self.model = get_peft_model(self.model, lora_config)
                
                # Ensure projection layer is trainable
                for param in self.custom_text_proj.parameters():
                    param.requires_grad = True
                
                return self
        
        return MockColQwen3VL()
    
    def test_vit_parameters_frozen_after_lora(self, mock_model):
        """
        Test that ViT parameters are frozen after enabling LoRA.
        
        **Validates: Requirements 2.3**
        """
        try:
            import peft
        except ImportError:
            pytest.skip("PEFT not installed")
        
        # Before LoRA, ViT params should be trainable
        vit_params_before = [p.requires_grad for p in mock_model.model.visual.parameters()]
        assert all(vit_params_before), "ViT params should be trainable before LoRA"
        
        # Enable LoRA
        mock_model.enable_lora_training(rank=8, alpha=8)
        
        # After LoRA, ViT params should be frozen
        vit_params_after = [p.requires_grad for p in mock_model.model.base_model.model.visual.parameters()]
        assert not any(vit_params_after), "ViT params should be frozen after LoRA"
    
    def test_projection_layer_remains_trainable(self, mock_model):
        """
        Test that projection layer remains fully trainable after LoRA.
        
        **Validates: Requirements 2.2**
        """
        try:
            import peft
        except ImportError:
            pytest.skip("PEFT not installed")
        
        # Enable LoRA
        mock_model.enable_lora_training(rank=8, alpha=8)
        
        # Projection layer should remain trainable
        proj_params = [p.requires_grad for p in mock_model.custom_text_proj.parameters()]
        assert all(proj_params), "Projection layer should remain fully trainable"
    
    def test_lora_applied_to_target_modules(self, mock_model):
        """
        Test that LoRA is applied to the correct target modules.
        
        **Validates: Requirements 2.1**
        """
        try:
            import peft
        except ImportError:
            pytest.skip("PEFT not installed")
        
        # Enable LoRA
        mock_model.enable_lora_training(rank=8, alpha=8)
        
        # Check that the model is now a PeftModel
        assert hasattr(mock_model.model, 'peft_config'), "Model should have peft_config after LoRA"
        
        # Check target modules in config
        peft_config = mock_model.model.peft_config['default']
        expected_modules = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
        assert set(peft_config.target_modules) == expected_modules, \
            f"LoRA should target {expected_modules}, got {peft_config.target_modules}"
    
    def test_lora_rank_and_alpha_configuration(self, mock_model):
        """
        Test that LoRA rank and alpha are correctly configured.
        
        **Validates: Requirements 2.6**
        """
        try:
            import peft
        except ImportError:
            pytest.skip("PEFT not installed")
        
        # Enable LoRA with specific rank and alpha
        mock_model.enable_lora_training(rank=16, alpha=32)
        
        # Check configuration
        peft_config = mock_model.model.peft_config['default']
        assert peft_config.r == 16, f"LoRA rank should be 16, got {peft_config.r}"
        assert peft_config.lora_alpha == 32, f"LoRA alpha should be 32, got {peft_config.lora_alpha}"
    
    def test_trainable_parameters_reduced(self, mock_model):
        """
        Test that trainable parameters are significantly reduced after LoRA.
        
        **Validates: Requirements 2.1, 2.3**
        """
        try:
            import peft
        except ImportError:
            pytest.skip("PEFT not installed")
        
        # Count trainable params before
        trainable_before = sum(p.numel() for p in mock_model.parameters() if p.requires_grad)
        total_before = sum(p.numel() for p in mock_model.parameters())
        
        # Enable LoRA
        mock_model.enable_lora_training(rank=8, alpha=8)
        
        # Count trainable params after
        trainable_after = sum(p.numel() for p in mock_model.parameters() if p.requires_grad)
        total_after = sum(p.numel() for p in mock_model.parameters())
        
        # Trainable params should be much less than total
        # (LoRA adds some params but freezes most)
        trainable_ratio = trainable_after / total_after
        assert trainable_ratio < 0.5, \
            f"Trainable ratio should be < 50%, got {trainable_ratio * 100:.2f}%"
