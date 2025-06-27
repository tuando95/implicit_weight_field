"""Unit tests for compression pipeline."""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.compressor import ModelCompressor, CompressionResult
from compression.trainer import FieldTrainer, TrainingConfig
from core.implicit_field import ImplicitWeightField, CompressionConfig


class TestFieldTrainer(unittest.TestCase):
    """Test cases for field trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        field = ImplicitWeightField((64, 32))
        config = TrainingConfig(learning_rate=1e-3, max_steps=100)
        trainer = FieldTrainer(field, config)
        
        self.assertEqual(trainer.config.learning_rate, 1e-3)
        self.assertEqual(trainer.config.max_steps, 100)
        self.assertIsNotNone(trainer.optimizer)
    
    def test_training_convergence(self):
        """Test that training reduces loss."""
        # Create target tensor
        shape = (32, 16)
        target = torch.randn(shape)
        
        # Create and train field
        field = ImplicitWeightField(shape)
        config = TrainingConfig(learning_rate=1e-2, max_steps=200)
        trainer = FieldTrainer(field, config)
        
        # Get initial loss
        initial_loss = trainer._compute_loss(target)
        
        # Train
        steps = trainer.train(target, verbose=False)
        
        # Get final loss
        final_loss = trainer._compute_loss(target)
        
        # Loss should decrease
        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 0.1)  # Should achieve reasonable fit
        self.assertGreater(steps, 0)
    
    def test_early_stopping(self):
        """Test early stopping on convergence."""
        # Simple tensor that's easy to fit
        target = torch.ones(10, 10) * 0.5
        
        field = ImplicitWeightField((10, 10))
        config = TrainingConfig(
            learning_rate=1e-2,
            max_steps=1000,
            convergence_threshold=1e-4,
            patience=10
        )
        trainer = FieldTrainer(field, config)
        
        steps = trainer.train(target, verbose=False)
        
        # Should stop early
        self.assertLess(steps, 1000)
    
    def test_loss_tracking(self):
        """Test loss history tracking."""
        target = torch.randn(20, 20)
        
        field = ImplicitWeightField((20, 20))
        config = TrainingConfig(max_steps=50)
        trainer = FieldTrainer(field, config)
        
        trainer.train(target, verbose=False)
        
        # Check loss history
        self.assertGreater(len(trainer.loss_history), 0)
        self.assertLessEqual(len(trainer.loss_history), 50)
        
        # Loss should generally decrease
        if len(trainer.loss_history) > 10:
            early_avg = np.mean(trainer.loss_history[:5])
            late_avg = np.mean(trainer.loss_history[-5:])
            self.assertLess(late_avg, early_avg)
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        target = torch.randn(50, 50) * 10  # Large values
        
        field = ImplicitWeightField((50, 50))
        config = TrainingConfig(
            learning_rate=1.0,  # Large LR
            gradient_clip=1.0
        )
        trainer = FieldTrainer(field, config)
        
        # Train for a few steps
        trainer.train(target, verbose=False)
        
        # Model should still be stable (not NaN)
        reconstructed = field.to_tensor()
        self.assertFalse(torch.any(torch.isnan(reconstructed)))
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics computation."""
        shape = (30, 30)
        target = torch.randn(shape)
        
        field = ImplicitWeightField(shape)
        trainer = FieldTrainer(field, TrainingConfig(max_steps=100))
        
        # Train
        trainer.train(target, verbose=False)
        
        # Evaluate
        metrics = trainer.evaluate(target)
        
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('max_error', metrics)
        self.assertIn('compression_ratio', metrics)
        
        # Sanity checks
        self.assertGreater(metrics['mse'], 0)
        self.assertEqual(metrics['rmse'], np.sqrt(metrics['mse']))
        self.assertGreater(metrics['compression_ratio'], 0)


class TestModelCompressor(unittest.TestCase):
    """Test cases for model compressor."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create simple test model
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 5)
        )
    
    def test_compressor_initialization(self):
        """Test compressor initialization."""
        config = CompressionConfig(hidden_width=64)
        compressor = ModelCompressor(self.model, config)
        
        self.assertEqual(compressor.config.hidden_width, 64)
        self.assertEqual(len(compressor.fields), 0)  # Not compressed yet
    
    def test_full_compression(self):
        """Test compressing entire model."""
        compressor = ModelCompressor(self.model)
        result = compressor.compress()
        
        self.assertIsInstance(result, CompressionResult)
        
        # Should compress linear layers
        self.assertIn('0.weight', result.layer_results)  # First linear
        self.assertIn('2.weight', result.layer_results)  # Second linear
        self.assertIn('4.weight', result.layer_results)  # Third linear
        
        # Check compression metrics
        self.assertGreater(result.total_compression_ratio, 0)
        self.assertGreater(result.original_parameters, 0)
        self.assertGreater(result.compressed_parameters, 0)
    
    def test_selective_compression(self):
        """Test compressing specific layers."""
        compressor = ModelCompressor(self.model)
        
        # Only compress first and last linear layers
        layer_names = ['0.weight', '4.weight']
        result = compressor.compress(layer_names=layer_names)
        
        # Only specified layers should be compressed
        self.assertEqual(len(result.layer_results), 2)
        self.assertIn('0.weight', result.layer_results)
        self.assertIn('4.weight', result.layer_results)
        self.assertNotIn('2.weight', result.layer_results)
    
    def test_min_size_filtering(self):
        """Test minimum tensor size filtering."""
        # Add a tiny layer
        model = nn.Sequential(
            nn.Linear(2, 3),    # 6 parameters - too small
            nn.Linear(100, 200) # 20000 parameters - large enough
        )
        
        config = CompressionConfig()
        compressor = ModelCompressor(model, config, min_tensor_size=1000)
        result = compressor.compress()
        
        # Only large layer should be compressed
        self.assertEqual(len(result.layer_results), 1)
        self.assertIn('1.weight', result.layer_results)
        self.assertNotIn('0.weight', result.layer_results)
    
    def test_get_compressed_model(self):
        """Test getting compressed model."""
        compressor = ModelCompressor(self.model)
        compressor.compress()
        
        compressed_model = compressor.get_compressed_model()
        
        # Should have same structure
        self.assertEqual(len(compressed_model), len(self.model))
        
        # Should produce output
        x = torch.randn(5, 10)
        output = compressed_model(x)
        self.assertEqual(output.shape, (5, 5))
    
    def test_compression_quality(self):
        """Test compression maintains model quality."""
        # Get original output
        x = torch.randn(10, 10)
        with torch.no_grad():
            original_output = self.model(x)
        
        # Compress with tight tolerance
        config = CompressionConfig(
            learning_rate=1e-2,
            max_steps=500,
            convergence_threshold=1e-5
        )
        compressor = ModelCompressor(self.model, config)
        compressor.compress()
        
        # Get compressed output
        compressed_model = compressor.get_compressed_model()
        with torch.no_grad():
            compressed_output = compressed_model(x)
        
        # Outputs should be similar
        mse = nn.MSELoss()(original_output, compressed_output)
        self.assertLess(mse.item(), 0.01)
    
    def test_compression_with_different_layers(self):
        """Test compression with various layer types."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        
        compressor = ModelCompressor(model)
        result = compressor.compress()
        
        # Should compress conv and linear layers
        compressed_count = len(result.layer_results)
        self.assertGreater(compressed_count, 0)
        
        # Should skip batch norm (too small)
        for name in result.layer_results:
            self.assertNotIn('BatchNorm', name)
    
    def test_save_and_load_compressed_model(self):
        """Test saving and loading compressed model."""
        import tempfile
        
        # Compress model
        compressor = ModelCompressor(self.model)
        compressor.compress()
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            compressor.save_compressed_model(tmp.name)
            
            # Load
            loaded_compressor = ModelCompressor(self.model)
            loaded_compressor.load_compressed_model(tmp.name)
            
            # Compare fields
            self.assertEqual(len(loaded_compressor.fields), len(compressor.fields))
            
            # Test loaded model
            loaded_model = loaded_compressor.get_compressed_model()
            x = torch.randn(5, 10)
            output = loaded_model(x)
            self.assertEqual(output.shape, (5, 5))
        
        os.unlink(tmp.name)
    
    def test_compression_statistics(self):
        """Test compression statistics reporting."""
        compressor = ModelCompressor(self.model)
        result = compressor.compress()
        
        # Check layer statistics
        for layer_name, layer_result in result.layer_results.items():
            self.assertGreater(layer_result.compression_ratio, 0)
            self.assertGreaterEqual(layer_result.reconstruction_error, 0)
            self.assertIsNotNone(layer_result.field_architecture)
            self.assertGreater(layer_result.training_steps, 0)
            self.assertIsNotNone(layer_result.original_shape)
    
    def test_device_handling(self):
        """Test compression on different devices."""
        if torch.cuda.is_available():
            model_gpu = self.model.cuda()
            compressor = ModelCompressor(model_gpu, device='cuda')
            result = compressor.compress()
            
            # Fields should be on GPU
            for field in compressor.fields.values():
                self.assertEqual(next(field.parameters()).device.type, 'cuda')
            
            # Compressed model should work on GPU
            compressed_model = compressor.get_compressed_model()
            x = torch.randn(5, 10, device='cuda')
            output = compressed_model(x)
            self.assertEqual(output.device.type, 'cuda')


class TestCompressionIntegration(unittest.TestCase):
    """Integration tests for compression pipeline."""
    
    def test_end_to_end_compression(self):
        """Test complete compression pipeline."""
        # Create a more complex model
        model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1000)
        )
        
        # Compress
        config = CompressionConfig(
            bandwidth=4,
            hidden_width=128,
            max_steps=100  # Fast for testing
        )
        compressor = ModelCompressor(model, config, min_tensor_size=10000)
        result = compressor.compress()
        
        # Verify compression
        self.assertGreater(result.total_compression_ratio, 1.0)
        
        # Test inference
        compressed_model = compressor.get_compressed_model()
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            original_out = model(x)
            compressed_out = compressed_model(x)
        
        # Should produce similar outputs
        self.assertEqual(original_out.shape, compressed_out.shape)
        
        # Check memory savings
        original_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # MB
        compressed_size = result.compressed_parameters * 4 / 1024 / 1024  # MB
        
        print(f"Original size: {original_size:.2f} MB")
        print(f"Compressed size: {compressed_size:.2f} MB")
        print(f"Compression ratio: {result.total_compression_ratio:.2f}x")
        
        self.assertLess(compressed_size, original_size)


if __name__ == '__main__':
    unittest.main()