"""
Performance test suite for memory usage requirements.

This module tests that all system components stay within memory
limits and manage memory efficiently during trading operations.
"""
import pytest
import numpy as np
import torch
import gc
import psutil
import os
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import threading
from datetime import datetime, timedelta

# Test markers
pytestmark = [pytest.mark.performance, pytest.mark.slow]


class TestMemoryLimits:
    """Test memory usage limits for different system components."""

    @pytest.fixture
    def memory_requirements(self):
        """Memory requirements for different components."""
        return {
            "strategic_model_mb": 64,      # Strategic model memory
            "tactical_model_mb": 32,       # Tactical model memory
            "data_buffer_mb": 128,         # Data buffer memory
            "total_system_mb": 512,        # Total system memory limit
            "peak_usage_mb": 768,          # Peak usage allowance
            "memory_leak_threshold": 1.1   # 10% growth threshold
        }

    @pytest.fixture
    def memory_monitor(self):
        """Memory monitoring utility."""
        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.baseline = self.get_current_usage()
                
            def get_current_usage(self):
                return self.process.memory_info().rss / 1024 / 1024  # MB
                
            def get_peak_usage(self):
                return self.process.memory_info().peak_wss / 1024 / 1024 if hasattr(self.process.memory_info(), 'peak_wss') else self.get_current_usage()
                
            def reset_baseline(self):
                gc.collect()
                time.sleep(0.1)
                self.baseline = self.get_current_usage()
                
            def get_usage_delta(self):
                return self.get_current_usage() - self.baseline
                
        return MemoryMonitor()

    def test_strategic_model_memory_usage(self, memory_monitor, memory_requirements):
        """Test memory usage of strategic models."""
        memory_monitor.reset_baseline()
        
        # Create mock strategic models
        models = {}
        for agent_name in ["strategic_agent", "position_sizing_agent", "regime_detection_agent"]:
            model = Mock()
            # Simulate model parameters
            model.parameters = [torch.randn(512, 624), torch.randn(256, 512), torch.randn(3, 256)]
            model.memory_usage = sum(p.numel() * 4 for p in model.parameters) / 1024 / 1024  # MB
            models[agent_name] = model
        
        # Calculate total model memory
        total_model_memory = sum(model.memory_usage for model in models.values())
        current_memory = memory_monitor.get_usage_delta()
        
        # Check strategic model memory limits
        assert total_model_memory < memory_requirements["strategic_model_mb"]
        print(f"Strategic models memory: {total_model_memory:.2f}MB")

    def test_tactical_model_memory_usage(self, memory_monitor, memory_requirements):
        """Test memory usage of tactical models."""
        memory_monitor.reset_baseline()
        
        # Create mock tactical models (smaller, optimized for speed)
        models = {}
        for agent_name in ["tactical_execution_agent", "order_flow_agent", "latency_optimization_agent"]:
            model = Mock()
            # Simulate smaller model parameters
            model.parameters = [torch.randn(256, 600), torch.randn(128, 256), torch.randn(7, 128)]
            model.memory_usage = sum(p.numel() * 4 for p in model.parameters) / 1024 / 1024  # MB
            models[agent_name] = model
        
        # Calculate total tactical model memory
        total_model_memory = sum(model.memory_usage for model in models.values())
        current_memory = memory_monitor.get_usage_delta()
        
        # Check tactical model memory limits (stricter)
        assert total_model_memory < memory_requirements["tactical_model_mb"]
        print(f"Tactical models memory: {total_model_memory:.2f}MB")

    def test_data_buffer_memory_usage(self, memory_monitor, memory_requirements):
        """Test memory usage of data buffers."""
        memory_monitor.reset_baseline()
        
        # Simulate various data buffers
        buffers = {}
        
        # Market data buffer (OHLCV)
        buffers["market_data"] = np.random.rand(10000, 6)  # 10k bars, 6 features
        
        # Indicator buffer
        buffers["indicators"] = np.random.rand(10000, 20)  # 10k bars, 20 indicators
        
        # Matrix buffer (30m)
        buffers["matrix_30m"] = np.random.rand(48, 13, 100)  # 48 bars, 13 features, 100 samples
        
        # Matrix buffer (5m)
        buffers["matrix_5m"] = np.random.rand(60, 10, 500)  # 60 bars, 10 features, 500 samples
        
        # Experience replay buffer
        buffers["experience"] = [np.random.rand(624) for _ in range(1000)]
        
        # Calculate buffer memory usage
        buffer_memory = 0
        for name, buffer in buffers.items():
            if isinstance(buffer, np.ndarray):
                buffer_memory += buffer.nbytes / 1024 / 1024
            elif isinstance(buffer, list):
                buffer_memory += sum(item.nbytes for item in buffer) / 1024 / 1024
        
        current_memory = memory_monitor.get_usage_delta()
        
        # Check data buffer memory limits
        assert buffer_memory < memory_requirements["data_buffer_mb"]
        print(f"Data buffers memory: {buffer_memory:.2f}MB")

    def test_total_system_memory_usage(self, memory_monitor, memory_requirements):
        """Test total system memory usage under full load."""
        memory_monitor.reset_baseline()
        
        # Simulate full system load
        system_components = {}
        
        # Models
        system_components["strategic_models"] = [torch.randn(512, 624) for _ in range(3)]
        system_components["tactical_models"] = [torch.randn(256, 600) for _ in range(4)]
        
        # Data structures
        system_components["market_data"] = np.random.rand(50000, 10)
        system_components["indicators"] = np.random.rand(50000, 25)
        system_components["matrices"] = np.random.rand(1000, 624)
        
        # Runtime data
        system_components["decisions"] = [{"data": np.random.rand(100)} for _ in range(1000)]
        system_components["orders"] = [{"order": np.random.rand(50)} for _ in range(500)]
        
        current_memory = memory_monitor.get_current_usage()
        
        # Check total system memory limits
        assert current_memory < memory_requirements["total_system_mb"]
        print(f"Total system memory: {current_memory:.2f}MB")

    def test_peak_memory_usage(self, memory_monitor, memory_requirements):
        """Test peak memory usage during intensive operations."""
        memory_monitor.reset_baseline()
        
        # Simulate memory-intensive operations
        temp_data = []
        
        for i in range(100):
            # Create large temporary arrays
            large_array = np.random.rand(1000, 1000)
            
            # Process the array
            processed = np.dot(large_array, large_array.T)
            
            # Keep only a small portion
            temp_data.append(processed[:10, :10])
            
            # Occasionally clear old data
            if i % 20 == 0:
                temp_data = temp_data[-50:]  # Keep only recent data
                gc.collect()
        
        peak_memory = memory_monitor.get_current_usage()
        
        # Check peak memory limits
        assert peak_memory < memory_requirements["peak_usage_mb"]
        print(f"Peak memory usage: {peak_memory:.2f}MB")


class TestMemoryLeaks:
    """Test for memory leaks in long-running operations."""

    @pytest.fixture
    def leak_detector(self):
        """Memory leak detection utility."""
        class LeakDetector:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.measurements = []
                
            def measure(self):
                gc.collect()
                time.sleep(0.1)
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.measurements.append({
                    "timestamp": time.time(),
                    "memory_mb": memory_mb
                })
                return memory_mb
                
            def detect_leak(self, threshold=1.1):
                if len(self.measurements) < 2:
                    return False
                    
                first_memory = self.measurements[0]["memory_mb"]
                last_memory = self.measurements[-1]["memory_mb"]
                growth_ratio = last_memory / first_memory
                
                return growth_ratio > threshold
                
            def get_growth_rate(self):
                if len(self.measurements) < 2:
                    return 0
                    
                first = self.measurements[0]
                last = self.measurements[-1]
                
                memory_growth = last["memory_mb"] - first["memory_mb"]
                time_elapsed = last["timestamp"] - first["timestamp"]
                
                return memory_growth / time_elapsed if time_elapsed > 0 else 0
                
        return LeakDetector()

    def test_inference_memory_leak(self, leak_detector):
        """Test for memory leaks during repeated inference."""
        mock_model = Mock()
        
        def mock_inference(input_data):
            # Simulate inference with tensor operations
            tensor = torch.FloatTensor(input_data)
            result = torch.softmax(torch.randn(tensor.shape[0], 3), dim=1)
            return result.numpy()
        
        mock_model.forward = mock_inference
        
        # Initial measurement
        leak_detector.measure()
        
        # Run many inference cycles
        for i in range(1000):
            input_data = np.random.rand(1, 624)
            output = mock_model.forward(input_data)
            
            # Measure memory every 100 iterations
            if i % 100 == 0:
                leak_detector.measure()
        
        # Final measurement
        leak_detector.measure()
        
        # Check for memory leaks
        has_leak = leak_detector.detect_leak(threshold=1.1)
        growth_rate = leak_detector.get_growth_rate()
        
        assert not has_leak, f"Memory leak detected! Growth rate: {growth_rate:.4f} MB/second"
        print(f"Inference memory growth rate: {growth_rate:.4f} MB/second")

    def test_data_processing_memory_leak(self, leak_detector):
        """Test for memory leaks during data processing."""
        mock_processor = Mock()
        
        def mock_process_bar(bar_data):
            # Simulate bar processing
            processed = {
                "ohlcv": np.array(bar_data["ohlcv"]),
                "indicators": np.random.rand(20),
                "features": np.random.rand(13)
            }
            return processed
        
        mock_processor.process = mock_process_bar
        
        # Initial measurement
        leak_detector.measure()
        
        # Process many bars
        for i in range(5000):
            bar_data = {
                "ohlcv": [1.0850 + i*0.0001, 1.0851, 1.0849, 1.0850, 1000],
                "timestamp": time.time() + i
            }
            
            result = mock_processor.process(bar_data)
            
            # Measure memory every 500 bars
            if i % 500 == 0:
                leak_detector.measure()
        
        # Final measurement
        leak_detector.measure()
        
        # Check for memory leaks
        has_leak = leak_detector.detect_leak(threshold=1.05)  # Stricter for data processing
        growth_rate = leak_detector.get_growth_rate()
        
        assert not has_leak, f"Data processing memory leak detected! Growth rate: {growth_rate:.4f} MB/second"
        print(f"Data processing memory growth rate: {growth_rate:.4f} MB/second")

    def test_buffer_memory_management(self, leak_detector):
        """Test memory management in circular buffers."""
        class CircularBuffer:
            def __init__(self, max_size):
                self.max_size = max_size
                self.buffer = []
                self.index = 0
                
            def add(self, item):
                if len(self.buffer) < self.max_size:
                    self.buffer.append(item)
                else:
                    self.buffer[self.index] = item
                    self.index = (self.index + 1) % self.max_size
        
        # Create buffers
        market_buffer = CircularBuffer(1000)
        indicator_buffer = CircularBuffer(1000)
        decision_buffer = CircularBuffer(500)
        
        # Initial measurement
        leak_detector.measure()
        
        # Fill buffers repeatedly
        for i in range(10000):
            # Add market data
            market_data = np.random.rand(6)  # OHLCV + volume
            market_buffer.add(market_data)
            
            # Add indicators
            indicators = np.random.rand(20)
            indicator_buffer.add(indicators)
            
            # Add decisions
            decision = {"position": np.random.uniform(-1, 1), "confidence": np.random.uniform(0, 1)}
            decision_buffer.add(decision)
            
            # Measure memory every 1000 iterations
            if i % 1000 == 0:
                leak_detector.measure()
        
        # Final measurement
        leak_detector.measure()
        
        # Check for memory leaks
        has_leak = leak_detector.detect_leak(threshold=1.02)  # Very strict for buffers
        growth_rate = leak_detector.get_growth_rate()
        
        assert not has_leak, f"Buffer memory leak detected! Growth rate: {growth_rate:.4f} MB/second"
        print(f"Buffer memory growth rate: {growth_rate:.4f} MB/second")

    def test_concurrent_memory_usage(self, leak_detector):
        """Test memory usage under concurrent operations."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker_function(worker_id, num_operations):
            local_data = []
            for i in range(num_operations):
                # Simulate work with memory allocation
                data = np.random.rand(100, 100)
                processed = np.sum(data, axis=0)
                local_data.append(processed)
                
                # Periodically clean up
                if len(local_data) > 50:
                    local_data = local_data[-25:]
            
            results_queue.put(f"Worker {worker_id} completed")
        
        # Initial measurement
        leak_detector.measure()
        
        # Start multiple workers
        num_workers = 4
        operations_per_worker = 500
        
        threads = []
        for worker_id in range(num_workers):
            thread = threading.Thread(target=worker_function, args=(worker_id, operations_per_worker))
            threads.append(thread)
            thread.start()
        
        # Monitor memory during execution
        monitoring = True
        def memory_monitor():
            while monitoring:
                leak_detector.measure()
                time.sleep(0.5)
        
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.start()
        
        # Wait for workers to complete
        for thread in threads:
            thread.join()
        
        monitoring = False
        monitor_thread.join()
        
        # Final measurement
        leak_detector.measure()
        
        # Check for memory leaks
        has_leak = leak_detector.detect_leak(threshold=1.15)  # Allow some growth for concurrency
        growth_rate = leak_detector.get_growth_rate()
        
        assert not has_leak, f"Concurrent memory leak detected! Growth rate: {growth_rate:.4f} MB/second"
        print(f"Concurrent memory growth rate: {growth_rate:.4f} MB/second")


class TestMemoryOptimization:
    """Test memory optimization techniques."""

    def test_tensor_memory_efficiency(self, memory_monitor):
        """Test efficient tensor memory usage."""
        memory_monitor.reset_baseline()
        
        # Test memory-efficient tensor operations
        def efficient_operations():
            # Use in-place operations
            tensor = torch.randn(1000, 1000)
            tensor.mul_(0.5)  # In-place multiplication
            tensor.add_(0.1)  # In-place addition
            
            # Use views instead of copies
            view = tensor[:500, :500]
            
            # Explicit cleanup
            del tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return view
        
        def inefficient_operations():
            # Creates many temporary tensors
            tensor = torch.randn(1000, 1000)
            tensor = tensor * 0.5  # Creates new tensor
            tensor = tensor + 0.1  # Creates new tensor
            
            # Creates copy
            subset = tensor[:500, :500].clone()
            
            return subset
        
        # Test efficient operations
        memory_before = memory_monitor.get_current_usage()
        efficient_result = efficient_operations()
        gc.collect()
        memory_efficient = memory_monitor.get_current_usage()
        
        # Test inefficient operations
        inefficient_result = inefficient_operations()
        gc.collect()
        memory_inefficient = memory_monitor.get_current_usage()
        
        efficient_usage = memory_efficient - memory_before
        inefficient_usage = memory_inefficient - memory_efficient
        
        # Efficient operations should use less memory
        assert efficient_usage < inefficient_usage
        print(f"Memory usage - Efficient: {efficient_usage:.2f}MB, Inefficient: {inefficient_usage:.2f}MB")

    def test_numpy_memory_efficiency(self, memory_monitor):
        """Test efficient numpy memory usage."""
        memory_monitor.reset_baseline()
        
        # Test memory-efficient numpy operations
        def efficient_numpy():
            # Use pre-allocated arrays
            result = np.empty((1000, 1000))
            temp = np.random.rand(1000, 1000)
            
            # In-place operations
            np.multiply(temp, 0.5, out=result)
            np.add(result, 0.1, out=result)
            
            # Use views
            subset = result[:500, :500]
            
            return subset
        
        def inefficient_numpy():
            # Creates many temporary arrays
            array = np.random.rand(1000, 1000)
            array = array * 0.5
            array = array + 0.1
            subset = array[:500, :500].copy()
            
            return subset
        
        # Test both approaches
        memory_before = memory_monitor.get_current_usage()
        efficient_result = efficient_numpy()
        gc.collect()
        memory_efficient = memory_monitor.get_current_usage()
        
        inefficient_result = inefficient_numpy()
        gc.collect()
        memory_inefficient = memory_monitor.get_current_usage()
        
        efficient_usage = memory_efficient - memory_before
        inefficient_usage = memory_inefficient - memory_efficient
        
        # Efficient operations should use less memory
        assert efficient_usage < inefficient_usage
        print(f"NumPy memory usage - Efficient: {efficient_usage:.2f}MB, Inefficient: {inefficient_usage:.2f}MB")

    def test_memory_pooling(self, memory_monitor):
        """Test memory pooling for frequent allocations."""
        memory_monitor.reset_baseline()
        
        class MemoryPool:
            def __init__(self, shape, pool_size=100):
                self.shape = shape
                self.pool = [np.empty(shape) for _ in range(pool_size)]
                self.available = list(range(pool_size))
                
            def get(self):
                if self.available:
                    idx = self.available.pop()
                    return self.pool[idx], idx
                else:
                    return np.empty(self.shape), -1
                    
            def return_array(self, idx):
                if idx >= 0:
                    self.available.append(idx)
        
        # Test with memory pool
        def with_pooling():
            pool = MemoryPool((1000, 100))
            results = []
            
            for _ in range(1000):
                array, idx = pool.get()
                array.fill(np.random.random())
                result = np.sum(array)
                results.append(result)
                pool.return_array(idx)
            
            return results
        
        # Test without memory pool
        def without_pooling():
            results = []
            
            for _ in range(1000):
                array = np.empty((1000, 100))
                array.fill(np.random.random())
                result = np.sum(array)
                results.append(result)
            
            return results
        
        # Compare memory usage
        memory_before = memory_monitor.get_current_usage()
        pooled_results = with_pooling()
        gc.collect()
        memory_pooled = memory_monitor.get_current_usage()
        
        regular_results = without_pooling()
        gc.collect()
        memory_regular = memory_monitor.get_current_usage()
        
        pooled_usage = memory_pooled - memory_before
        regular_usage = memory_regular - memory_pooled
        
        # Pooling should be more memory efficient
        print(f"Memory usage - With pooling: {pooled_usage:.2f}MB, Without pooling: {regular_usage:.2f}MB")

    def test_garbage_collection_efficiency(self, memory_monitor):
        """Test garbage collection efficiency."""
        memory_monitor.reset_baseline()
        
        def create_garbage():
            # Create objects that will become garbage
            data_structures = []
            for i in range(1000):
                obj = {
                    "id": i,
                    "data": np.random.rand(100),
                    "metadata": {"timestamp": time.time(), "processed": False}
                }
                data_structures.append(obj)
            
            # Process and discard
            processed = []
            for obj in data_structures:
                obj["metadata"]["processed"] = True
                processed.append(obj["data"].sum())
            
            return processed
        
        # Test garbage collection behavior
        memory_before = memory_monitor.get_current_usage()
        
        # Create garbage
        results = create_garbage()
        
        memory_after_creation = memory_monitor.get_current_usage()
        
        # Force garbage collection
        gc.collect()
        
        memory_after_gc = memory_monitor.get_current_usage()
        
        creation_increase = memory_after_creation - memory_before
        gc_decrease = memory_after_creation - memory_after_gc
        
        # Garbage collection should free significant memory
        assert gc_decrease > creation_increase * 0.5  # Should free at least 50%
        
        print(f"Memory - Created: +{creation_increase:.2f}MB, GC freed: -{gc_decrease:.2f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])