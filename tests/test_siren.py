"""Unit tests for SIREN architecture."""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.siren import SIRENLayer, SIREN


class TestSIRENLayer(unittest.TestCase):
    """Test cases for SIRENLayer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 10
        self.out_features = 20
        self.w0 = 30.0
        torch.manual_seed(42)
    
    def test_layer_creation(self):
        """Test layer creation with different configurations."""
        # First layer
        layer = SIRENLayer(self.in_features, self.out_features, 
                          w0=self.w0, is_first=True)
        self.assertEqual(layer.linear.in_features, self.in_features)
        self.assertEqual(layer.linear.out_features, self.out_features)
        self.assertEqual(layer.w0, self.w0)
        self.assertTrue(layer.is_first)
        
        # Hidden layer
        layer = SIRENLayer(self.in_features, self.out_features, 
                          w0=self.w0, is_first=False)
        self.assertFalse(layer.is_first)
    
    def test_weight_initialization(self):
        """Test proper weight initialization."""
        # First layer initialization
        layer = SIRENLayer(self.in_features, self.out_features, 
                          w0=self.w0, is_first=True)
        weights = layer.linear.weight.data.cpu().numpy()
        expected_bound = 1 / self.in_features
        self.assertTrue(np.all(weights >= -expected_bound))
        self.assertTrue(np.all(weights <= expected_bound))
        
        # Hidden layer initialization
        layer = SIRENLayer(self.in_features, self.out_features, 
                          w0=self.w0, is_first=False)
        weights = layer.linear.weight.data.cpu().numpy()
        expected_bound = np.sqrt(6 / self.in_features) / self.w0
        self.assertTrue(np.all(weights >= -expected_bound * 1.1))  # Small tolerance
        self.assertTrue(np.all(weights <= expected_bound * 1.1))
    
    def test_forward_pass(self):
        """Test forward pass through layer."""
        layer = SIRENLayer(self.in_features, self.out_features, w0=self.w0)
        x = torch.randn(32, self.in_features)
        
        output = layer(x)
        self.assertEqual(output.shape, (32, self.out_features))
        
        # Check that output is in range [-1, 1] (sine activation)
        self.assertTrue(torch.all(output >= -1))
        self.assertTrue(torch.all(output <= 1))
    
    def test_gradient_flow(self):
        """Test gradient flow through layer."""
        layer = SIRENLayer(self.in_features, self.out_features, w0=self.w0)
        x = torch.randn(32, self.in_features, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(layer.linear.weight.grad)
        self.assertIsNotNone(layer.linear.bias.grad)
        
        # Check that gradients are not zero
        self.assertTrue(torch.any(x.grad != 0))
        self.assertTrue(torch.any(layer.linear.weight.grad != 0))


class TestSIREN(unittest.TestCase):
    """Test cases for complete SIREN network."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.in_features = 10
        self.hidden_features = 64
        self.out_features = 1
        self.num_layers = 3
    
    def test_network_creation(self):
        """Test network creation with different configurations."""
        # Basic network
        net = SIREN(self.in_features, self.hidden_features, 
                   self.out_features, self.num_layers)
        
        # Check architecture
        modules = list(net.layers.children())
        self.assertEqual(len(modules), self.num_layers)
        
        # Check first layer
        self.assertIsInstance(modules[0], SIRENLayer)
        self.assertTrue(modules[0].is_first)
        self.assertEqual(modules[0].linear.in_features, self.in_features)
        
        # Check hidden layers
        for i in range(1, self.num_layers - 1):
            self.assertIsInstance(modules[i], SIRENLayer)
            self.assertFalse(modules[i].is_first)
            self.assertEqual(modules[i].linear.in_features, self.hidden_features)
            self.assertEqual(modules[i].linear.out_features, self.hidden_features)
        
        # Check output layer
        self.assertIsInstance(modules[-1], nn.Linear)
        self.assertEqual(modules[-1].out_features, self.out_features)
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        net = SIREN(self.in_features, self.hidden_features, 
                   self.out_features, self.num_layers)
        
        batch_size = 32
        x = torch.randn(batch_size, self.in_features)
        
        output = net(x)
        self.assertEqual(output.shape, (batch_size, self.out_features))
        
        # Output should not be bounded (linear final layer)
        self.assertTrue(torch.any(output < -1) or torch.any(output > 1))
    
    def test_different_architectures(self):
        """Test networks with different depths and widths."""
        configs = [
            (10, 32, 1, 2),   # Shallow, narrow
            (10, 128, 1, 4),  # Deep, wide
            (20, 64, 5, 3),   # Multi-output
        ]
        
        for in_f, hidden_f, out_f, n_layers in configs:
            net = SIREN(in_f, hidden_f, out_f, n_layers)
            x = torch.randn(16, in_f)
            output = net(x)
            self.assertEqual(output.shape, (16, out_f))
    
    def test_frequency_modulation(self):
        """Test different frequency parameters."""
        w0_values = [1.0, 10.0, 30.0, 100.0]
        
        for w0 in w0_values:
            net = SIREN(self.in_features, self.hidden_features, 
                       self.out_features, self.num_layers, w0=w0)
            
            # Check that w0 is properly set
            for module in net.layers.children():
                if isinstance(module, SIRENLayer):
                    self.assertEqual(module.w0, w0)
    
    def test_gradient_stability(self):
        """Test gradient stability through deep networks."""
        # Create a deeper network
        deep_net = SIREN(self.in_features, self.hidden_features, 
                        self.out_features, num_layers=6)
        
        x = torch.randn(32, self.in_features, requires_grad=True)
        target = torch.randn(32, self.out_features)
        
        # Forward pass
        output = deep_net(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check gradient magnitudes
        grad_norms = []
        for name, param in deep_net.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                # Gradients should not explode or vanish
                self.assertLess(grad_norm, 100.0)  # Not exploding
                self.assertGreater(grad_norm, 1e-8)  # Not vanishing
        
        # Check that gradient norms are relatively stable
        grad_norms = np.array(grad_norms)
        cv = grad_norms.std() / grad_norms.mean()  # Coefficient of variation
        self.assertLess(cv, 10.0)  # Reasonable variation
    
    def test_parameter_count(self):
        """Test parameter counting."""
        net = SIREN(self.in_features, self.hidden_features, 
                   self.out_features, self.num_layers)
        
        # Calculate expected parameter count
        # First layer: (in_features + 1) * hidden_features
        # Hidden layers: (num_layers - 2) * (hidden_features + 1) * hidden_features
        # Output layer: (hidden_features + 1) * out_features
        expected_params = (
            (self.in_features + 1) * self.hidden_features +
            (self.num_layers - 2) * (self.hidden_features + 1) * self.hidden_features +
            (self.hidden_features + 1) * self.out_features
        )
        
        actual_params = sum(p.numel() for p in net.parameters())
        self.assertEqual(actual_params, expected_params)
    
    def test_device_compatibility(self):
        """Test network works on different devices."""
        net = SIREN(self.in_features, self.hidden_features, 
                   self.out_features, self.num_layers)
        x = torch.randn(32, self.in_features)
        
        # CPU
        output_cpu = net(x)
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        # GPU if available
        if torch.cuda.is_available():
            net_gpu = net.cuda()
            x_gpu = x.cuda()
            output_gpu = net_gpu(x_gpu)
            self.assertEqual(output_gpu.device.type, 'cuda')
            
            # Results should be the same
            torch.testing.assert_close(
                output_cpu, 
                output_gpu.cpu(), 
                rtol=1e-5, 
                atol=1e-5
            )


class TestSIRENFitting(unittest.TestCase):
    """Test SIREN's ability to fit functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
    
    def test_fit_simple_function(self):
        """Test fitting a simple 1D function."""
        # Create data
        x = torch.linspace(-np.pi, np.pi, 100).unsqueeze(1)
        y = torch.sin(5 * x)  # High-frequency sine
        
        # Create and train network
        net = SIREN(1, 32, 1, 3, w0=30.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        
        # Train for a few steps
        for _ in range(100):
            pred = net(x)
            loss = nn.MSELoss()(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check that loss decreased
        with torch.no_grad():
            final_pred = net(x)
            final_loss = nn.MSELoss()(final_pred, y)
            self.assertLess(final_loss.item(), 0.1)
    
    def test_fit_image_patch(self):
        """Test fitting a 2D image patch."""
        # Create synthetic image patch
        size = 16
        coords = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, size),
            torch.linspace(0, 1, size),
            indexing='ij'
        ), dim=-1).reshape(-1, 2)
        
        # Create pattern
        values = torch.sin(10 * coords[:, 0]) * torch.cos(10 * coords[:, 1])
        values = values.unsqueeze(1)
        
        # Create and train network
        net = SIREN(2, 64, 1, 3, w0=30.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        
        # Train
        for _ in range(200):
            pred = net(coords)
            loss = nn.MSELoss()(pred, values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check reconstruction
        with torch.no_grad():
            final_pred = net(coords)
            final_loss = nn.MSELoss()(final_pred, values)
            self.assertLess(final_loss.item(), 0.01)


if __name__ == '__main__':
    unittest.main()