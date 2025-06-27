"""Unit tests for positional encoding utilities."""

import unittest
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.positional_encoding import FourierFeatures, PositionalEncoding


class TestFourierFeatures(unittest.TestCase):
    """Test cases for Fourier feature encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.input_dim = 3
        self.bandwidth = 4
        self.encoder = FourierFeatures(self.input_dim, self.bandwidth)
    
    def test_initialization(self):
        """Test encoder initialization."""
        self.assertEqual(self.encoder.input_dim, self.input_dim)
        self.assertEqual(self.encoder.bandwidth, self.bandwidth)
        self.assertEqual(self.encoder.output_dim, 2 * self.bandwidth * self.input_dim)
    
    def test_output_dimension(self):
        """Test output dimensions for different configurations."""
        configs = [
            (1, 1),  # Minimal
            (2, 4),  # 2D input
            (3, 8),  # 3D input
            (10, 16),  # High-dimensional
        ]
        
        for input_dim, bandwidth in configs:
            encoder = FourierFeatures(input_dim, bandwidth)
            x = torch.randn(32, input_dim)
            output = encoder(x)
            
            expected_dim = 2 * bandwidth * input_dim
            self.assertEqual(output.shape, (32, expected_dim))
    
    def test_encoding_structure(self):
        """Test the structure of Fourier encodings."""
        # Simple 1D case for easier verification
        encoder = FourierFeatures(1, 2)
        x = torch.tensor([[0.5]])
        
        output = encoder(x)
        
        # Expected: [sin(2^0 * pi * x), cos(2^0 * pi * x), 
        #           sin(2^1 * pi * x), cos(2^1 * pi * x)]
        expected = torch.tensor([[
            np.sin(1 * np.pi * 0.5),
            np.cos(1 * np.pi * 0.5),
            np.sin(2 * np.pi * 0.5),
            np.cos(2 * np.pi * 0.5)
        ]])
        
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    
    def test_frequency_progression(self):
        """Test that frequencies follow geometric progression."""
        encoder = FourierFeatures(1, 4)
        x = torch.tensor([[0.25]])
        output = encoder(x)
        
        # Extract sine components (every other element starting from 0)
        sines = output[0, ::2].numpy()
        
        # Check frequencies: 2^0, 2^1, 2^2, 2^3
        for i in range(4):
            expected_freq = 2**i
            expected_val = np.sin(expected_freq * np.pi * 0.25)
            self.assertAlmostEqual(sines[i], expected_val, places=5)
    
    def test_normalization_invariance(self):
        """Test encoding of normalized vs unnormalized coordinates."""
        encoder = FourierFeatures(2, 4)
        
        # Normalized coordinates [0, 1]
        x_norm = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        
        # Same relative positions, different scale
        x_scaled = x_norm * 10
        
        # Encodings should be different due to different frequencies
        enc_norm = encoder(x_norm)
        enc_scaled = encoder(x_scaled)
        
        self.assertFalse(torch.allclose(enc_norm, enc_scaled))
    
    def test_gradient_flow(self):
        """Test gradient flow through encoding."""
        encoder = FourierFeatures(3, 8)
        x = torch.randn(16, 3, requires_grad=True)
        
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
    
    def test_batch_processing(self):
        """Test batch processing consistency."""
        encoder = FourierFeatures(2, 4)
        
        # Process batch
        batch = torch.randn(10, 2)
        batch_output = encoder(batch)
        
        # Process individually
        individual_outputs = []
        for i in range(10):
            individual_output = encoder(batch[i:i+1])
            individual_outputs.append(individual_output)
        
        individual_outputs = torch.cat(individual_outputs, dim=0)
        
        torch.testing.assert_close(batch_output, individual_outputs)
    
    def test_device_compatibility(self):
        """Test encoding on different devices."""
        encoder = FourierFeatures(3, 4)
        x = torch.randn(32, 3)
        
        # CPU
        output_cpu = encoder(x)
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        # GPU if available
        if torch.cuda.is_available():
            encoder_gpu = encoder.cuda()
            x_gpu = x.cuda()
            output_gpu = encoder_gpu(x_gpu)
            self.assertEqual(output_gpu.device.type, 'cuda')
            
            # Results should match
            torch.testing.assert_close(output_cpu, output_gpu.cpu())
    
    def test_boundary_values(self):
        """Test encoding at boundary values."""
        encoder = FourierFeatures(1, 4)
        
        # Test at boundaries and special points
        special_points = torch.tensor([
            [0.0],    # Zero
            [1.0],    # One
            [-1.0],   # Negative
            [0.5],    # Mid-point
            [1e-8],   # Very small
            [1e8]     # Very large
        ])
        
        # Should not produce NaN or Inf
        output = encoder(special_points)
        self.assertFalse(torch.any(torch.isnan(output)))
        self.assertFalse(torch.any(torch.isinf(output)))


class TestPositionalEncoding(unittest.TestCase):
    """Test cases for general positional encoding."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_fourier_encoding_type(self):
        """Test Fourier encoding through general interface."""
        encoder = PositionalEncoding(
            input_dim=2,
            encoding_type='fourier',
            bandwidth=4
        )
        
        x = torch.randn(32, 2)
        output = encoder(x)
        
        self.assertEqual(output.shape, (32, 16))  # 2 * 4 * 2
        self.assertEqual(encoder.output_dim, 16)
    
    def test_identity_encoding_type(self):
        """Test identity encoding (no encoding)."""
        encoder = PositionalEncoding(
            input_dim=5,
            encoding_type='identity'
        )
        
        x = torch.randn(32, 5)
        output = encoder(x)
        
        # Should be unchanged
        torch.testing.assert_close(output, x)
        self.assertEqual(encoder.output_dim, 5)
    
    def test_learned_encoding_type(self):
        """Test learned positional encoding."""
        encoder = PositionalEncoding(
            input_dim=3,
            encoding_type='learned',
            encoding_dim=64
        )
        
        x = torch.randn(32, 3)
        output = encoder(x)
        
        self.assertEqual(output.shape, (32, 64))
        self.assertEqual(encoder.output_dim, 64)
        
        # Check that it's learnable
        self.assertTrue(any(p.requires_grad for p in encoder.parameters()))
    
    def test_invalid_encoding_type(self):
        """Test error handling for invalid encoding type."""
        with self.assertRaises(ValueError):
            PositionalEncoding(
                input_dim=3,
                encoding_type='invalid'
            )
    
    def test_encoding_consistency(self):
        """Test that same input produces same output."""
        encoder = PositionalEncoding(
            input_dim=2,
            encoding_type='fourier',
            bandwidth=8
        )
        
        x = torch.randn(16, 2)
        output1 = encoder(x)
        output2 = encoder(x)
        
        torch.testing.assert_close(output1, output2)
    
    def test_gradient_preservation(self):
        """Test gradient flow for all encoding types."""
        encoding_configs = [
            ('identity', {}),
            ('fourier', {'bandwidth': 4}),
            ('learned', {'encoding_dim': 32})
        ]
        
        for encoding_type, kwargs in encoding_configs:
            encoder = PositionalEncoding(
                input_dim=3,
                encoding_type=encoding_type,
                **kwargs
            )
            
            x = torch.randn(8, 3, requires_grad=True)
            output = encoder(x)
            loss = output.sum()
            loss.backward()
            
            self.assertIsNotNone(x.grad)
            if encoding_type != 'identity':  # Identity should preserve gradients exactly
                self.assertTrue(torch.any(x.grad != 0))
    
    def test_learned_encoding_training(self):
        """Test that learned encoding parameters update during training."""
        encoder = PositionalEncoding(
            input_dim=2,
            encoding_type='learned',
            encoding_dim=16
        )
        
        # Get initial parameters
        initial_params = [p.clone() for p in encoder.parameters()]
        
        # Simple training step
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
        
        x = torch.randn(32, 2)
        output = encoder(x)
        loss = output.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        for initial, current in zip(initial_params, encoder.parameters()):
            self.assertFalse(torch.allclose(initial, current))
    
    def test_encoding_scale_sensitivity(self):
        """Test sensitivity to input scale for different encodings."""
        x_small = torch.randn(16, 3) * 0.01
        x_large = torch.randn(16, 3) * 100
        
        # Fourier encoding should be sensitive to scale
        fourier_encoder = PositionalEncoding(3, 'fourier', bandwidth=4)
        fourier_small = fourier_encoder(x_small)
        fourier_large = fourier_encoder(x_large)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(fourier_small, fourier_large, atol=1e-2))
        
        # Identity encoding preserves scale
        identity_encoder = PositionalEncoding(3, 'identity')
        identity_small = identity_encoder(x_small)
        identity_large = identity_encoder(x_large)
        
        # Check scale is preserved
        scale_ratio = (identity_large.abs().mean() / identity_small.abs().mean()).item()
        self.assertGreater(scale_ratio, 50)  # Large should be much bigger
    
    def test_coordinate_range_recommendations(self):
        """Test behavior with recommended coordinate ranges."""
        # Fourier encoding expects normalized coordinates [0, 1] or [-1, 1]
        encoder = PositionalEncoding(2, 'fourier', bandwidth=8)
        
        # Normalized coordinates
        x_norm = torch.rand(100, 2)  # [0, 1]
        output_norm = encoder(x_norm)
        
        # Check output is well-behaved (no extreme values)
        self.assertLess(output_norm.abs().max().item(), 2.0)
        self.assertGreater(output_norm.abs().mean().item(), 0.1)
    
    def test_encoding_orthogonality(self):
        """Test orthogonality properties of Fourier encoding."""
        encoder = FourierFeatures(1, 4)
        
        # Sample points
        n_points = 1000
        x = torch.linspace(0, 1, n_points).unsqueeze(1)
        encoded = encoder(x)
        
        # Check approximate orthogonality of basis functions
        # Compute gram matrix
        gram = encoded.T @ encoded / n_points
        
        # Off-diagonal elements should be small
        eye = torch.eye(gram.shape[0])
        off_diagonal = gram * (1 - eye)
        
        self.assertLess(off_diagonal.abs().max().item(), 0.1)


class TestEncodingIntegration(unittest.TestCase):
    """Test encoding integration with SIREN networks."""
    
    def test_fourier_siren_integration(self):
        """Test Fourier encoding + SIREN for function fitting."""
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core.siren import SIREN
        
        # Create high-frequency target function
        x = torch.linspace(-1, 1, 200).unsqueeze(1)
        y = torch.sin(20 * np.pi * x) + 0.5 * torch.cos(40 * np.pi * x)
        
        # Without encoding - should struggle
        net_plain = SIREN(1, 64, 1, 3)
        
        # With Fourier encoding
        encoder = FourierFeatures(1, 8)
        net_encoded = SIREN(encoder.output_dim, 64, 1, 3)
        
        # Train both
        for net, use_encoding in [(net_plain, False), (net_encoded, True)]:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
            
            for _ in range(500):
                if use_encoding:
                    x_input = encoder(x)
                else:
                    x_input = x
                
                pred = net(x_input)
                loss = torch.nn.MSELoss()(pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate final losses
        with torch.no_grad():
            loss_plain = torch.nn.MSELoss()(net_plain(x), y).item()
            loss_encoded = torch.nn.MSELoss()(net_encoded(encoder(x)), y).item()
        
        # Encoded version should perform better on high-frequency function
        self.assertLess(loss_encoded, loss_plain * 0.5)


if __name__ == '__main__':
    unittest.main()