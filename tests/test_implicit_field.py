"""Unit tests for implicit weight field module."""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.implicit_field import (
    ImplicitWeightField, 
    CompressionConfig, 
    FieldArchitecture,
    TensorStatistics,
    MultiScaleImplicitField
)


class TestCompressionConfig(unittest.TestCase):
    """Test cases for CompressionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CompressionConfig()
        
        self.assertEqual(config.bandwidth, 4)
        self.assertEqual(config.hidden_width, 256)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.w0, 30.0)
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.max_steps, 2000)
        self.assertAlmostEqual(config.regularization, 1e-6)
        self.assertAlmostEqual(config.convergence_threshold, 1e-6)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CompressionConfig(
            bandwidth=8,
            hidden_width=512,
            num_layers=3,
            learning_rate=5e-4
        )
        
        self.assertEqual(config.bandwidth, 8)
        self.assertEqual(config.hidden_width, 512)
        self.assertEqual(config.num_layers, 3)
        self.assertEqual(config.learning_rate, 5e-4)


class TestTensorStatistics(unittest.TestCase):
    """Test cases for tensor statistics computation."""
    
    def test_compute_statistics(self):
        """Test tensor statistics computation."""
        # Create test tensor with known properties
        tensor = torch.randn(100, 50)
        stats = ImplicitWeightField.compute_tensor_statistics(tensor)
        
        self.assertIsInstance(stats, TensorStatistics)
        self.assertGreater(stats.mean, -1)
        self.assertLess(stats.mean, 1)
        self.assertGreater(stats.std, 0)
        self.assertGreaterEqual(stats.sparsity, 0)
        self.assertLessEqual(stats.sparsity, 1)
        self.assertGreater(stats.effective_rank, 0)
        self.assertLessEqual(stats.effective_rank, min(tensor.shape))
    
    def test_sparse_tensor_statistics(self):
        """Test statistics for sparse tensor."""
        # Create sparse tensor
        tensor = torch.zeros(100, 100)
        tensor[torch.randperm(100)[:10], torch.randperm(100)[:10]] = torch.randn(10)
        
        stats = ImplicitWeightField.compute_tensor_statistics(tensor)
        
        self.assertGreater(stats.sparsity, 0.9)  # Should be very sparse
        self.assertLess(stats.effective_rank, 20)  # Low rank
    
    def test_low_rank_tensor_statistics(self):
        """Test statistics for low-rank tensor."""
        # Create low-rank tensor
        rank = 5
        U = torch.randn(100, rank)
        V = torch.randn(rank, 100)
        tensor = U @ V
        
        stats = ImplicitWeightField.compute_tensor_statistics(tensor)
        
        # Effective rank should be close to true rank
        self.assertLess(abs(stats.effective_rank - rank), 2)


class TestImplicitWeightField(unittest.TestCase):
    """Test cases for ImplicitWeightField."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_field_creation(self):
        """Test field creation with different configurations."""
        # 2D tensor (linear layer)
        field = ImplicitWeightField(
            tensor_shape=(128, 64),
            config=CompressionConfig()
        )
        self.assertEqual(field.tensor_shape, (128, 64))
        self.assertEqual(field.num_params, 128 * 64)
        
        # 4D tensor (conv layer)
        field = ImplicitWeightField(
            tensor_shape=(64, 32, 3, 3),
            config=CompressionConfig()
        )
        self.assertEqual(field.tensor_shape, (64, 32, 3, 3))
        self.assertEqual(field.num_params, 64 * 32 * 3 * 3)
    
    def test_architecture_selection(self):
        """Test adaptive architecture selection."""
        # Tiny tensor - should be explicit
        field = ImplicitWeightField((10, 10))
        self.assertEqual(field.architecture, FieldArchitecture.EXPLICIT)
        
        # Small tensor - should be linear
        field = ImplicitWeightField((100, 50))
        self.assertEqual(field.architecture, FieldArchitecture.LINEAR_1L)
        
        # Medium tensor - should be SIREN 2L
        field = ImplicitWeightField((1000, 1000))
        self.assertEqual(field.architecture, FieldArchitecture.SIREN_2L)
    
    def test_forward_pass(self):
        """Test forward pass through field."""
        shape = (64, 32)
        field = ImplicitWeightField(shape).to(self.device)
        
        # Generate all coordinates
        coords = field._generate_coordinates()
        coords = coords.to(self.device)
        
        # Forward pass
        weights = field(coords)
        
        # Check output shape
        self.assertEqual(weights.shape, (np.prod(shape), 1))
        
        # Reshape to tensor
        tensor = field.to_tensor()
        self.assertEqual(tensor.shape, shape)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        # Large tensor
        shape = (1024, 512)
        config = CompressionConfig(hidden_width=128)
        field = ImplicitWeightField(shape, config)
        
        ratio = field.compression_ratio()
        
        # Should achieve compression
        self.assertGreater(ratio, 1.0)
        
        # Calculate expected ratio
        original_params = np.prod(shape)
        field_params = field.count_parameters()
        expected_ratio = original_params / field_params
        
        self.assertAlmostEqual(ratio, expected_ratio, places=2)
    
    def test_coordinate_generation(self):
        """Test coordinate generation and normalization."""
        shape = (10, 20, 5)
        field = ImplicitWeightField(shape)
        
        coords = field._generate_coordinates()
        
        # Check shape
        self.assertEqual(coords.shape, (1000, 3))  # 10*20*5 = 1000
        
        # Check normalization to [0, 1]
        self.assertTrue(torch.all(coords >= 0))
        self.assertTrue(torch.all(coords <= 1))
        
        # Check corner cases
        self.assertTrue(torch.any(torch.all(coords == 0, dim=1)))  # (0,0,0)
        self.assertTrue(torch.any(torch.all(coords == 1, dim=1)))  # (1,1,1)
    
    def test_to_tensor_conversion(self):
        """Test conversion from field to tensor."""
        shape = (32, 16, 4)
        field = ImplicitWeightField(shape).to(self.device)
        
        # Convert to tensor
        tensor = field.to_tensor()
        
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.device, self.device)
        
        # Test consistency
        tensor2 = field.to_tensor()
        torch.testing.assert_close(tensor, tensor2)
    
    def test_gradient_flow(self):
        """Test gradient flow through field."""
        field = ImplicitWeightField((64, 32)).to(self.device)
        
        # Generate tensor
        tensor = field.to_tensor()
        
        # Use in computation
        x = torch.randn(10, 32, device=self.device)
        y = x @ tensor.T
        loss = y.sum()
        
        # Check gradients
        loss.backward()
        
        # Field parameters should have gradients
        for param in field.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_explicit_storage(self):
        """Test explicit storage for tiny tensors."""
        shape = (5, 5)
        field = ImplicitWeightField(shape)
        
        self.assertEqual(field.architecture, FieldArchitecture.EXPLICIT)
        
        # Should have explicit weights
        self.assertIsNotNone(field.explicit_weights)
        self.assertEqual(field.explicit_weights.shape, shape)
        
        # Forward pass should return flattened weights
        coords = field._generate_coordinates()
        weights = field(coords)
        
        torch.testing.assert_close(
            weights.squeeze(),
            field.explicit_weights.flatten()
        )
    
    def test_device_transfer(self):
        """Test moving field between devices."""
        field = ImplicitWeightField((128, 64))
        
        # CPU tensor
        tensor_cpu = field.to_tensor()
        self.assertEqual(tensor_cpu.device.type, 'cpu')
        
        if torch.cuda.is_available():
            # Move to GPU
            field_gpu = field.cuda()
            tensor_gpu = field_gpu.to_tensor()
            self.assertEqual(tensor_gpu.device.type, 'cuda')
            
            # Results should match
            torch.testing.assert_close(tensor_cpu, tensor_gpu.cpu())


class TestMultiScaleImplicitField(unittest.TestCase):
    """Test cases for multi-scale implicit field."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_multiscale_creation(self):
        """Test multi-scale field creation."""
        field = MultiScaleImplicitField(
            tensor_shape=(512, 512),
            num_scales=3,
            config=CompressionConfig()
        ).to(self.device)
        
        self.assertEqual(field.num_scales, 3)
        self.assertIsNotNone(field.base_field)
        self.assertEqual(len(field.detail_fields), 3)
        self.assertEqual(len(field.scale_weights), 3)
    
    def test_multiscale_forward(self):
        """Test forward pass through multi-scale field."""
        shape = (256, 128)
        field = MultiScaleImplicitField(shape, num_scales=2).to(self.device)
        
        # Generate tensor
        tensor = field.to_tensor()
        
        self.assertEqual(tensor.shape, shape)
        
        # Test that scales contribute
        with torch.no_grad():
            # Zero out detail fields
            for detail_field in field.detail_fields:
                for param in detail_field.parameters():
                    param.zero_()
            
            base_only = field.to_tensor()
            
            # Restore and check difference
            for param in field.parameters():
                param.data.normal_(0, 0.1)
            
            full_tensor = field.to_tensor()
            
            # Should be different when detail fields contribute
            self.assertFalse(torch.allclose(base_only, full_tensor))
    
    def test_scale_weights_learnable(self):
        """Test that scale weights are learnable."""
        field = MultiScaleImplicitField((128, 128), num_scales=2)
        
        # Check initial weights
        initial_weights = field.scale_weights.clone()
        
        # Optimization step
        optimizer = torch.optim.SGD([field.scale_weights], lr=0.1)
        
        tensor = field.to_tensor()
        loss = tensor.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Weights should change
        self.assertFalse(torch.allclose(initial_weights, field.scale_weights))
    
    def test_compression_ratio_multiscale(self):
        """Test compression ratio for multi-scale field."""
        shape = (1024, 1024)
        
        # Single scale
        single_field = ImplicitWeightField(shape, CompressionConfig(hidden_width=128))
        single_ratio = single_field.compression_ratio()
        
        # Multi scale (should have more parameters)
        multi_field = MultiScaleImplicitField(
            shape, 
            num_scales=2,
            config=CompressionConfig(hidden_width=128)
        )
        multi_ratio = multi_field.compression_ratio()
        
        # Multi-scale should have lower compression due to extra fields
        self.assertLess(multi_ratio, single_ratio)
        
        # But should still compress
        self.assertGreater(multi_ratio, 1.0)


class TestFieldIntegration(unittest.TestCase):
    """Test field integration with real tensors."""
    
    def test_compress_real_tensor(self):
        """Test compressing a real network tensor."""
        # Create a conv layer
        conv = nn.Conv2d(32, 64, kernel_size=3)
        weight = conv.weight.data
        
        # Create field
        field = ImplicitWeightField(weight.shape)
        
        # Train field to fit weights
        optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
        
        for _ in range(100):
            reconstructed = field.to_tensor()
            loss = nn.MSELoss()(reconstructed, weight)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check reconstruction quality
        final_reconstruction = field.to_tensor()
        mse = nn.MSELoss()(final_reconstruction, weight).item()
        
        # Should achieve reasonable reconstruction
        self.assertLess(mse, 0.01)
        
        # Check compression
        original_params = weight.numel()
        compressed_params = field.count_parameters()
        
        if field.architecture != FieldArchitecture.EXPLICIT:
            self.assertLess(compressed_params, original_params)
    
    def test_batch_weight_generation(self):
        """Test generating weights for batched operations."""
        field = ImplicitWeightField((64, 32))
        
        # Generate subset of weights
        indices = torch.tensor([
            [0, 0], [0, 1], [1, 0], [1, 1]
        ], dtype=torch.long)
        
        # Convert to normalized coordinates
        coords = indices.float()
        coords[:, 0] /= 63  # Normalize first dimension
        coords[:, 1] /= 31  # Normalize second dimension
        
        # Apply encoding
        encoded = field.encoder(coords)
        subset_weights = field.field(encoded)
        
        self.assertEqual(subset_weights.shape, (4, 1))


if __name__ == '__main__':
    unittest.main()