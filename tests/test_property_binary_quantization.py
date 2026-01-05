"""
Property-Based Tests for Binary Quantization Correctness.

**Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

Property 6: For any float32 embedding vector, binary quantization using sign function 
SHALL produce a binary vector where bit i is 1 if embedding[i] >= 0 and 0 otherwise. 
The storage size SHALL be reduced by factor of 32 (float32 to 1-bit). 
Hamming distance between two binary vectors SHALL equal the number of differing bits.
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st, assume
from typing import Tuple

from qwen3_vl_retrieval.retrieval.binary_quantizer import BinaryQuantizer


# Hypothesis strategies
@st.composite
def embedding_strategy(draw):
    """Generate random embeddings with dimension divisible by 8."""
    # Dimension must be divisible by 8 for packing
    dim_multiplier = draw(st.integers(min_value=1, max_value=32))
    dim = dim_multiplier * 8  # 8, 16, 24, ..., 256
    
    n_tokens = draw(st.integers(min_value=1, max_value=64))
    
    # Generate random float embeddings
    embeddings = torch.randn(n_tokens, dim)
    
    return embeddings


@st.composite
def embedding_pair_strategy(draw):
    """Generate a pair of embeddings with the same dimension."""
    dim_multiplier = draw(st.integers(min_value=1, max_value=16))
    dim = dim_multiplier * 8
    
    n_tokens_1 = draw(st.integers(min_value=1, max_value=32))
    n_tokens_2 = draw(st.integers(min_value=1, max_value=32))
    
    emb1 = torch.randn(n_tokens_1, dim)
    emb2 = torch.randn(n_tokens_2, dim)
    
    return emb1, emb2


class TestProperty6BinaryQuantizationCorrectness:
    """
    Property 6: Binary Quantization Correctness
    
    **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    
    For any float32 embedding vector, binary quantization using sign function SHALL 
    produce a binary vector where bit i is 1 if embedding[i] >= 0 and 0 otherwise.
    """
    
    @given(embeddings=embedding_strategy())
    @settings(max_examples=100, deadline=None)
    def test_sign_function_quantization(self, embeddings: torch.Tensor):
        """
        Property: Binary quantization uses sign function correctly.
        
        bit[i] = 1 if embedding[i] >= 0 else 0
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.1**
        """
        quantizer = BinaryQuantizer()
        
        # Quantize
        binary = quantizer.quantize(embeddings)
        
        # Unpack to verify bit values
        unpacked = quantizer.unpack_binary(binary, embeddings.shape[-1])
        
        # Expected: bit[i] = 1 if embedding[i] >= 0 else 0
        expected_bits = (embeddings >= 0).to(torch.uint8)
        
        # Property: unpacked bits should match expected
        assert torch.all(unpacked == expected_bits), \
            "Sign function quantization mismatch"
    
    @given(embeddings=embedding_strategy())
    @settings(max_examples=100, deadline=None)
    def test_storage_reduction_32x(self, embeddings: torch.Tensor):
        """
        Property: Storage is reduced by factor of 32.
        
        float32 = 32 bits per value -> 1 bit per value = 32x reduction
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.4**
        """
        quantizer = BinaryQuantizer()
        
        # Original storage (float32)
        original_bytes = embeddings.numel() * 4  # 4 bytes per float32
        
        # Quantized storage (uint8 packed)
        binary = quantizer.quantize(embeddings)
        quantized_bytes = binary.numel()  # 1 byte per uint8
        
        # Property: reduction factor should be 32
        reduction_factor = original_bytes / quantized_bytes
        assert abs(reduction_factor - 32) < 0.01, \
            f"Expected 32x reduction, got {reduction_factor}x"
    
    @given(embeddings=embedding_strategy())
    @settings(max_examples=100, deadline=None)
    def test_output_shape(self, embeddings: torch.Tensor):
        """
        Property: Output shape is (..., dim // 8).
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.1**
        """
        quantizer = BinaryQuantizer()
        
        binary = quantizer.quantize(embeddings)
        
        expected_shape = embeddings.shape[:-1] + (embeddings.shape[-1] // 8,)
        
        assert binary.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {binary.shape}"
    
    @given(data=embedding_pair_strategy())
    @settings(max_examples=100, deadline=None)
    def test_hamming_distance_equals_differing_bits(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """
        Property: Hamming distance equals the number of differing bits.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.2**
        """
        emb1, emb2 = data
        quantizer = BinaryQuantizer()
        
        # Quantize both
        binary1 = quantizer.quantize(emb1)
        binary2 = quantizer.quantize(emb2)
        
        # Compute Hamming distance using our implementation
        distances = quantizer.hamming_distance(binary1, binary2)
        
        # Compute reference Hamming distance by counting differing bits
        bits1 = quantizer.unpack_binary(binary1, emb1.shape[-1])  # (n1, dim)
        bits2 = quantizer.unpack_binary(binary2, emb2.shape[-1])  # (n2, dim)
        
        # For each pair, count differing bits
        for i in range(bits1.shape[0]):
            for j in range(bits2.shape[0]):
                # XOR and count 1s
                diff_bits = (bits1[i] != bits2[j]).sum().item()
                computed_dist = distances[i, j].item()
                
                assert computed_dist == diff_bits, \
                    f"Hamming distance mismatch at ({i},{j}): " \
                    f"computed={computed_dist}, expected={diff_bits}"
    
    @given(embeddings=embedding_strategy())
    @settings(max_examples=100, deadline=None)
    def test_hamming_distance_self_is_zero(self, embeddings: torch.Tensor):
        """
        Property: Hamming distance of a vector with itself is zero.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.2**
        """
        quantizer = BinaryQuantizer()
        
        binary = quantizer.quantize(embeddings)
        
        # Distance with self
        distances = quantizer.hamming_distance(binary, binary)
        
        # Diagonal should be zero
        for i in range(binary.shape[0]):
            assert distances[i, i].item() == 0, \
                f"Self-distance at {i} should be 0, got {distances[i, i].item()}"
    
    @given(data=embedding_pair_strategy())
    @settings(max_examples=100, deadline=None)
    def test_hamming_distance_symmetric(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """
        Property: Hamming distance is symmetric: d(a, b) = d(b, a).
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.2**
        """
        emb1, emb2 = data
        quantizer = BinaryQuantizer()
        
        binary1 = quantizer.quantize(emb1)
        binary2 = quantizer.quantize(emb2)
        
        dist_1_to_2 = quantizer.hamming_distance(binary1, binary2)
        dist_2_to_1 = quantizer.hamming_distance(binary2, binary1)
        
        # Property: d(a, b) = d(b, a)
        assert torch.allclose(dist_1_to_2, dist_2_to_1.T), \
            "Hamming distance should be symmetric"
    
    @given(embeddings=embedding_strategy())
    @settings(max_examples=100, deadline=None)
    def test_hamming_distance_non_negative(self, embeddings: torch.Tensor):
        """
        Property: Hamming distance is always non-negative.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.2**
        """
        quantizer = BinaryQuantizer()
        
        binary = quantizer.quantize(embeddings)
        
        # Create a slightly different embedding
        perturbed = embeddings + torch.randn_like(embeddings) * 0.1
        binary_perturbed = quantizer.quantize(perturbed)
        
        distances = quantizer.hamming_distance(binary, binary_perturbed)
        
        assert torch.all(distances >= 0), \
            "Hamming distance should be non-negative"
    
    @given(embeddings=embedding_strategy())
    @settings(max_examples=100, deadline=None)
    def test_hamming_distance_bounded(self, embeddings: torch.Tensor):
        """
        Property: Hamming distance is bounded by embedding dimension.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.2**
        """
        quantizer = BinaryQuantizer()
        dim = embeddings.shape[-1]
        
        binary = quantizer.quantize(embeddings)
        
        # Create opposite embeddings (all bits flipped)
        opposite = -embeddings
        binary_opposite = quantizer.quantize(opposite)
        
        distances = quantizer.hamming_distance(binary, binary_opposite)
        
        # Maximum distance is dim (all bits different)
        assert torch.all(distances <= dim), \
            f"Hamming distance should be <= {dim}"
    
    @given(embeddings=embedding_strategy())
    @settings(max_examples=100, deadline=None)
    def test_rescore_matches_maxsim(self, embeddings: torch.Tensor):
        """
        Property: Rescore produces correct MaxSim scores.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.3**
        """
        quantizer = BinaryQuantizer()
        
        # Create query and doc embeddings
        query = embeddings[:min(8, embeddings.shape[0])]  # Use first 8 tokens as query
        doc = embeddings  # Use all as document
        
        # Compute rescore
        score = quantizer.rescore(query, doc)
        
        # Compute reference MaxSim
        similarities = torch.matmul(query, doc.T)
        max_sims = similarities.max(dim=1)[0]
        expected_score = max_sims.sum()
        
        assert torch.allclose(score, expected_score, atol=1e-5), \
            f"Rescore mismatch: computed={score.item():.6f}, expected={expected_score.item():.6f}"


class TestProperty6EdgeCases:
    """
    Edge case tests for Binary Quantization.
    
    **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    
    def test_all_positive_embeddings(self):
        """
        Test: All positive values produce all 1 bits.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.1**
        """
        quantizer = BinaryQuantizer()
        
        embeddings = torch.abs(torch.randn(10, 128)) + 0.1  # All positive
        binary = quantizer.quantize(embeddings)
        
        # All bits should be 1, so each byte should be 255
        assert torch.all(binary == 255), \
            "All positive embeddings should produce all 1 bits (255 per byte)"
    
    def test_all_negative_embeddings(self):
        """
        Test: All negative values produce all 0 bits.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.1**
        """
        quantizer = BinaryQuantizer()
        
        embeddings = -torch.abs(torch.randn(10, 128)) - 0.1  # All negative
        binary = quantizer.quantize(embeddings)
        
        # All bits should be 0
        assert torch.all(binary == 0), \
            "All negative embeddings should produce all 0 bits"
    
    def test_zero_values_quantize_to_one(self):
        """
        Test: Zero values quantize to 1 (>= 0 condition).
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.1**
        """
        quantizer = BinaryQuantizer()
        
        embeddings = torch.zeros(1, 8)  # All zeros
        binary = quantizer.quantize(embeddings)
        
        # Zero >= 0, so all bits should be 1
        assert binary[0, 0].item() == 255, \
            "Zero values should quantize to 1 (all bits set)"
    
    def test_alternating_signs(self):
        """
        Test: Alternating positive/negative produces alternating bits.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.1**
        """
        quantizer = BinaryQuantizer()
        
        # Create alternating pattern: [+, -, +, -, +, -, +, -]
        embeddings = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
        binary = quantizer.quantize(embeddings)
        
        # Expected bits: [1, 0, 1, 0, 1, 0, 1, 0] = 0b10101010 = 170
        assert binary[0, 0].item() == 170, \
            f"Alternating pattern should produce 170, got {binary[0, 0].item()}"
    
    def test_invalid_dimension_raises_error(self):
        """
        Test: Non-divisible-by-8 dimension raises ValueError.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.1**
        """
        quantizer = BinaryQuantizer()
        
        embeddings = torch.randn(10, 127)  # 127 not divisible by 8
        
        with pytest.raises(ValueError, match="divisible by 8"):
            quantizer.quantize(embeddings)
    
    def test_single_token_embedding(self):
        """
        Test: Single token embedding works correctly.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.1**
        """
        quantizer = BinaryQuantizer()
        
        embeddings = torch.randn(1, 128)
        binary = quantizer.quantize(embeddings)
        
        assert binary.shape == (1, 16), \
            f"Expected shape (1, 16), got {binary.shape}"
    
    def test_hamming_distance_opposite_vectors(self):
        """
        Test: Opposite vectors have maximum Hamming distance.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.2**
        """
        quantizer = BinaryQuantizer()
        
        embeddings = torch.randn(1, 128)
        opposite = -embeddings
        
        binary1 = quantizer.quantize(embeddings)
        binary2 = quantizer.quantize(opposite)
        
        distance = quantizer.hamming_distance(binary1, binary2)
        
        # All bits should be different
        assert distance[0, 0].item() == 128, \
            f"Opposite vectors should have distance 128, got {distance[0, 0].item()}"
    
    def test_storage_reduction_factor(self):
        """
        Test: get_storage_reduction_factor returns 32.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.4**
        """
        quantizer = BinaryQuantizer()
        
        assert quantizer.get_storage_reduction_factor() == 32, \
            "Storage reduction factor should be 32"
    
    def test_estimate_storage_bytes(self):
        """
        Test: Storage estimation is correct.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.4**
        """
        quantizer = BinaryQuantizer()
        
        num_tokens = 1000
        dim = 128
        
        # Float32 storage
        float_bytes = quantizer.estimate_storage_bytes(num_tokens, dim, quantized=False)
        assert float_bytes == num_tokens * dim * 4, \
            f"Float storage should be {num_tokens * dim * 4}, got {float_bytes}"
        
        # Binary storage
        binary_bytes = quantizer.estimate_storage_bytes(num_tokens, dim, quantized=True)
        assert binary_bytes == num_tokens * (dim // 8), \
            f"Binary storage should be {num_tokens * (dim // 8)}, got {binary_bytes}"
        
        # Verify 32x reduction
        assert float_bytes / binary_bytes == 32, \
            "Storage reduction should be 32x"


class TestProperty6BatchProcessing:
    """
    Tests for batch processing in Binary Quantization.
    
    **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    
    @given(embeddings=embedding_strategy())
    @settings(max_examples=100, deadline=None)
    def test_batch_quantization_consistent(self, embeddings: torch.Tensor):
        """
        Property: Batch quantization produces same results as individual.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.1**
        """
        quantizer = BinaryQuantizer()
        
        # Batch quantization
        batch_binary = quantizer.quantize(embeddings)
        
        # Individual quantization
        for i in range(embeddings.shape[0]):
            individual_binary = quantizer.quantize(embeddings[i:i+1])
            
            assert torch.all(batch_binary[i] == individual_binary[0]), \
                f"Batch and individual quantization mismatch at index {i}"
    
    def test_batch_rescore(self):
        """
        Test: Batch rescore produces correct scores for multiple documents.
        
        **Feature: qwen3-vl-rag-retrieval, Property 6: Binary Quantization Correctness**
        **Validates: Requirements 4.3**
        """
        quantizer = BinaryQuantizer()
        
        query = torch.randn(10, 128)
        docs = [torch.randn(20, 128) for _ in range(5)]
        
        # Batch rescore
        batch_scores = quantizer.batch_rescore(query, docs)
        
        # Individual rescore
        individual_scores = []
        for doc in docs:
            score = quantizer.rescore(query, doc)
            individual_scores.append(score)
        individual_scores = torch.stack(individual_scores)
        
        assert torch.allclose(batch_scores, individual_scores, atol=1e-5), \
            "Batch and individual rescore should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
