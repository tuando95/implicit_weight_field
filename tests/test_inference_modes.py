"""Unit tests for inference modes."""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.preload_mode import PreloadInference
from inference.streaming_mode import StreamingInference, LRUCache
from core.implicit_field import ImplicitWeightField, CompressionConfig


class TestLRUCache(unittest.TestCase):
    """Test cases for LRU cache implementation."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache(capacity=3)
        
        # Test put and get
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        self.assertIsNone(cache.get("key3"))
    
    def test_cache_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(capacity=3)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item - should evict key2 (least recently used)
        cache.put("key4", "value4")
        
        self.assertIsNotNone(cache.get("key1"))  # Still present
        self.assertIsNone(cache.get("key2"))     # Evicted
        self.assertIsNotNone(cache.get("key3"))  # Still present
        self.assertIsNotNone(cache.get("key4"))  # Newly added
    
    def test_cache_update(self):
        """Test updating existing cache entries."""
        cache = LRUCache(capacity=2)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Update key1
        cache.put("key1", "new_value1")
        
        self.assertEqual(cache.get("key1"), "new_value1")
        
        # key1 should now be most recently used
        cache.put("key3", "value3")
        
        self.assertIsNotNone(cache.get("key1"))  # Still present
        self.assertIsNone(cache.get("key2"))     # Evicted
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(capacity=2)
        
        # Initial stats
        stats = cache.get_stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 0)
        self.assertEqual(stats['hit_rate'], 0.0)
        
        # Add items and test
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit
        
        stats = cache.get_stats()
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['misses'], 1)
        self.assertAlmostEqual(stats['hit_rate'], 2/3)
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LRUCache(capacity=3)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        cache.clear()
        
        self.assertIsNone(cache.get("key1"))
        self.assertIsNone(cache.get("key2"))
        self.assertEqual(cache.size, 0)


class TestPreloadInference(unittest.TestCase):
    """Test cases for preload inference mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_preload_creation(self):
        """Test preload inference creation."""
        # Create test model with fields
        fields = {
            'layer1.weight': ImplicitWeightField((64, 32)),
            'layer2.weight': ImplicitWeightField((128, 64))
        }
        
        inference = PreloadInference(fields, device=self.device)
        
        self.assertEqual(len(inference.reconstructed_weights), 2)
        self.assertIn('layer1.weight', inference.reconstructed_weights)
        self.assertIn('layer2.weight', inference.reconstructed_weights)
    
    def test_weight_reconstruction(self):
        """Test weight reconstruction on initialization."""
        # Create field
        shape = (32, 16)
        field = ImplicitWeightField(shape).to(self.device)
        
        # Set some pattern in the field
        if hasattr(field, 'explicit_weights'):
            field.explicit_weights.data = torch.randn(shape, device=self.device)
        
        # Create preload inference
        fields = {'test_weight': field}
        inference = PreloadInference(fields, device=self.device)
        
        # Check reconstruction
        weight = inference.get_weight('test_weight')
        self.assertEqual(weight.shape, shape)
        self.assertEqual(weight.device, self.device)
        
        # Should match field output
        expected = field.to_tensor()
        torch.testing.assert_close(weight, expected)
    
    def test_get_weight_performance(self):
        """Test that get_weight is fast (preloaded)."""
        fields = {
            f'layer{i}': ImplicitWeightField((128, 128)) 
            for i in range(10)
        }
        
        inference = PreloadInference(fields)
        
        # Time weight access
        start = time.time()
        for i in range(100):
            for j in range(10):
                weight = inference.get_weight(f'layer{j}')
        elapsed = time.time() - start
        
        # Should be very fast (just dict lookup)
        self.assertLess(elapsed, 0.1)  # 1000 lookups in < 0.1s
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        # Create fields of known sizes
        fields = {
            'small': ImplicitWeightField((10, 10)),
            'large': ImplicitWeightField((1000, 1000))
        }
        
        inference = PreloadInference(fields)
        memory_mb = inference.get_memory_usage_mb()
        
        # Rough check - at least the size of tensors
        min_expected_mb = (100 + 1000000) * 4 / (1024 * 1024)  # float32
        self.assertGreater(memory_mb, min_expected_mb * 0.8)  # Allow some overhead
    
    def test_missing_weight_error(self):
        """Test error handling for missing weights."""
        fields = {'existing': ImplicitWeightField((10, 10))}
        inference = PreloadInference(fields)
        
        with self.assertRaises(KeyError):
            inference.get_weight('non_existing')


class TestStreamingInference(unittest.TestCase):
    """Test cases for streaming inference mode."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_streaming_creation(self):
        """Test streaming inference creation."""
        fields = {
            'layer1': ImplicitWeightField((64, 32)),
            'layer2': ImplicitWeightField((128, 64))
        }
        
        inference = StreamingInference(
            fields, 
            cache_size_mb=10,
            device=self.device
        )
        
        self.assertEqual(len(inference.fields), 2)
        self.assertIsNotNone(inference.weight_cache)
        self.assertIsNotNone(inference.coord_cache)
    
    def test_on_demand_reconstruction(self):
        """Test on-demand weight reconstruction."""
        # Create field with known pattern
        shape = (32, 16)
        field = ImplicitWeightField(shape).to(self.device)
        
        fields = {'test_weight': field}
        inference = StreamingInference(fields, cache_size_mb=1, device=self.device)
        
        # First access - cache miss
        weight1 = inference.get_weight('test_weight')
        self.assertEqual(weight1.shape, shape)
        
        # Second access - cache hit
        weight2 = inference.get_weight('test_weight')
        torch.testing.assert_close(weight1, weight2)
        
        # Check cache stats
        stats = inference.get_cache_stats()
        self.assertEqual(stats['weight_cache']['hits'], 1)
        self.assertEqual(stats['weight_cache']['misses'], 1)
    
    def test_coordinate_caching(self):
        """Test coordinate tensor caching."""
        shape = (64, 32, 3, 3)  # Conv weight
        field = ImplicitWeightField(shape)
        
        fields = {'conv': field}
        inference = StreamingInference(fields, cache_size_mb=10)
        
        # Access weight element
        coord = (10, 20, 1, 1)
        value1 = inference.get_weight_element('conv', coord)
        
        # Second access should use cached coordinate
        value2 = inference.get_weight_element('conv', coord)
        
        self.assertEqual(value1, value2)
        
        # Check coordinate cache was used
        stats = inference.get_cache_stats()
        self.assertGreater(stats['coord_cache']['hits'], 0)
    
    def test_cache_eviction(self):
        """Test cache eviction under memory pressure."""
        # Create multiple large fields
        fields = {
            f'layer{i}': ImplicitWeightField((512, 512))
            for i in range(5)
        }
        
        # Small cache that can't hold all weights
        inference = StreamingInference(fields, cache_size_mb=2)
        
        # Access all weights
        for i in range(5):
            weight = inference.get_weight(f'layer{i}')
        
        # Early weights should be evicted
        stats = inference.get_cache_stats()
        self.assertLess(inference.weight_cache.size, 5)
    
    def test_element_access(self):
        """Test individual element access."""
        shape = (10, 20)
        field = ImplicitWeightField(shape)
        
        # Train field to have known values
        with torch.no_grad():
            if hasattr(field, 'explicit_weights'):
                field.explicit_weights.data = torch.arange(200).reshape(shape).float()
        
        fields = {'matrix': field}
        inference = StreamingInference(fields)
        
        # Access specific elements
        val_0_0 = inference.get_weight_element('matrix', (0, 0))
        val_5_10 = inference.get_weight_element('matrix', (5, 10))
        
        if hasattr(field, 'explicit_weights'):
            self.assertAlmostEqual(val_0_0, 0.0, places=5)
            self.assertAlmostEqual(val_5_10, 110.0, places=5)  # 5*20 + 10
    
    def test_prefetching(self):
        """Test prefetching for sequential access."""
        shape = (100, 100)
        field = ImplicitWeightField(shape)
        fields = {'weight': field}
        
        inference = StreamingInference(fields, enable_prefetch=True)
        
        # Sequential access pattern
        for i in range(10):
            for j in range(10):
                value = inference.get_weight_element('weight', (i, j))
        
        # Prefetching should improve cache hits for sequential access
        stats = inference.get_cache_stats()
        if inference.enable_prefetch:
            # With prefetching, we expect better cache performance
            self.assertGreater(stats['coord_cache']['hit_rate'], 0.5)
    
    def test_memory_efficiency(self):
        """Test memory efficiency compared to preload."""
        # Large model
        fields = {
            f'layer{i}': ImplicitWeightField((1024, 1024))
            for i in range(10)
        }
        
        # Streaming with small cache
        streaming = StreamingInference(fields, cache_size_mb=10)
        streaming_memory = streaming.get_memory_usage_mb()
        
        # Preload
        preload = PreloadInference(fields)
        preload_memory = preload.get_memory_usage_mb()
        
        # Streaming should use much less memory
        self.assertLess(streaming_memory, preload_memory * 0.2)
    
    def test_thread_safety(self):
        """Test thread safety of streaming inference."""
        import threading
        
        fields = {'shared': ImplicitWeightField((256, 256))}
        inference = StreamingInference(fields)
        
        errors = []
        
        def access_weights():
            try:
                for _ in range(100):
                    weight = inference.get_weight('shared')
                    # Random element access
                    i, j = np.random.randint(0, 256, size=2)
                    val = inference.get_weight_element('shared', (i, j))
            except Exception as e:
                errors.append(e)
        
        # Launch multiple threads
        threads = []
        for _ in range(4):
            t = threading.Thread(target=access_weights)
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Should complete without errors
        self.assertEqual(len(errors), 0)


class TestInferenceModeIntegration(unittest.TestCase):
    """Test integration of inference modes with compressed models."""
    
    def test_mode_consistency(self):
        """Test that both modes produce same results."""
        # Create test fields
        fields = {
            'conv1.weight': ImplicitWeightField((32, 16, 3, 3)),
            'fc1.weight': ImplicitWeightField((128, 256)),
            'fc2.weight': ImplicitWeightField((10, 128))
        }
        
        # Train fields to have specific patterns
        for name, field in fields.items():
            if hasattr(field, 'explicit_weights'):
                field.explicit_weights.data.normal_(0, 0.1)
        
        # Create both inference modes
        preload = PreloadInference(fields)
        streaming = StreamingInference(fields, cache_size_mb=100)
        
        # Compare outputs
        for name in fields:
            weight_preload = preload.get_weight(name)
            weight_streaming = streaming.get_weight(name)
            
            torch.testing.assert_close(weight_preload, weight_streaming, rtol=1e-5, atol=1e-5)
    
    def test_inference_with_forward_pass(self):
        """Test inference modes in actual forward pass."""
        # Simple model
        class TestModel(nn.Module):
            def __init__(self, inference_engine):
                super().__init__()
                self.inference = inference_engine
            
            def forward(self, x):
                # Linear layer: y = xW^T + b
                weight = self.inference.get_weight('fc.weight')
                x = x @ weight.t()
                return x
        
        # Create field
        fields = {'fc.weight': ImplicitWeightField((128, 64))}
        
        # Test with both modes
        for inference_cls in [PreloadInference, StreamingInference]:
            inference = inference_cls(fields)
            model = TestModel(inference)
            
            # Forward pass
            x = torch.randn(32, 64)
            output = model(x)
            
            self.assertEqual(output.shape, (32, 128))
            
            # Should be differentiable
            output.sum().backward()


if __name__ == '__main__':
    unittest.main()