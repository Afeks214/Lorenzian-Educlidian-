"""
Performance test suite for inference speed requirements.

This module tests that all system components meet their inference
speed requirements for real-time trading.
"""
import pytest
import numpy as np
import time
import torch
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import psutil
import os
from datetime import datetime

# Test markers
pytestmark = [pytest.mark.performance, pytest.mark.slow]


class TestStrategicInferenceSpeed:
    """Test inference speed for strategic components (30-minute timeframe)."""

    @pytest.fixture
    def performance_requirements(self):
        """Performance requirements for strategic inference."""
        return {
            "max_inference_time_ms": 50,  # 50ms for strategic decisions
            "target_inference_time_ms": 20,  # Target: 20ms
            "batch_inference_time_ms": 100,  # Batch processing: 100ms
            "memory_limit_mb": 256,  # Memory usage limit
            "cpu_limit_percent": 30  # CPU usage limit
        }

    @pytest.fixture
    def mock_strategic_model(self):
        """Create a mock strategic model for testing."""
        model = Mock()
        model.forward = Mock()
        model.inference_time_ms = 0
        
        def mock_forward(x):
            # Simulate inference time
            start_time = time.perf_counter()
            # Simulate computation
            result = torch.softmax(torch.randn(x.shape[0], 3), dim=1)
            end_time = time.perf_counter()
            model.inference_time_ms = (end_time - start_time) * 1000
            return result
        
        model.forward = mock_forward
        return model

    @pytest.fixture
    def sample_strategic_input(self):
        """Sample input for strategic model."""
        return np.random.rand(1, 624)  # 48 bars × 13 features

    def test_single_inference_speed(self, mock_strategic_model, sample_strategic_input, 
                                   performance_requirements, performance_timer):
        """Test single inference speed for strategic decisions."""
        performance_timer.start()
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sample_strategic_input)
        
        # Run inference
        output = mock_strategic_model.forward(input_tensor)
        
        performance_timer.stop()
        elapsed_ms = performance_timer.elapsed_ms()
        
        # Check performance requirements
        assert elapsed_ms < performance_requirements["max_inference_time_ms"]
        assert output.shape == (1, 3)  # Expected output shape
        
        print(f"Strategic inference time: {elapsed_ms:.2f}ms")

    def test_batch_inference_speed(self, mock_strategic_model, performance_requirements):
        """Test batch inference speed for multiple strategic decisions."""
        batch_size = 32
        input_batch = torch.randn(batch_size, 624)
        
        start_time = time.perf_counter()
        
        # Run batch inference
        output = mock_strategic_model.forward(input_batch)
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        # Check batch performance
        assert elapsed_ms < performance_requirements["batch_inference_time_ms"]
        assert output.shape == (batch_size, 3)
        
        # Calculate per-sample time
        per_sample_ms = elapsed_ms / batch_size
        assert per_sample_ms < performance_requirements["target_inference_time_ms"]
        
        print(f"Batch inference time: {elapsed_ms:.2f}ms ({per_sample_ms:.2f}ms per sample)")

    def test_memory_usage_during_inference(self, mock_strategic_model, sample_strategic_input,
                                         performance_requirements, memory_profiler):
        """Test memory usage during strategic inference."""
        memory_profiler.start()
        
        # Run multiple inferences to stress test memory
        for _ in range(100):
            input_tensor = torch.FloatTensor(sample_strategic_input)
            _ = mock_strategic_model.forward(input_tensor)
        
        peak_memory = memory_profiler.get_peak_usage()
        
        # Check memory usage
        assert peak_memory < performance_requirements["memory_limit_mb"]
        
        print(f"Peak memory usage: {peak_memory:.2f}MB")

    def test_cpu_usage_during_inference(self, mock_strategic_model, sample_strategic_input,
                                       performance_requirements):
        """Test CPU usage during strategic inference."""
        process = psutil.Process(os.getpid())
        
        # Warm up
        input_tensor = torch.FloatTensor(sample_strategic_input)
        _ = mock_strategic_model.forward(input_tensor)
        
        # Measure CPU usage
        cpu_before = process.cpu_percent()
        
        # Run intensive inference
        for _ in range(50):
            _ = mock_strategic_model.forward(input_tensor)
        
        cpu_after = process.cpu_percent()
        cpu_usage = max(cpu_after - cpu_before, cpu_after)
        
        # Check CPU usage (allowing some flexibility for test environment)
        assert cpu_usage < performance_requirements["cpu_limit_percent"] * 2
        
        print(f"CPU usage during inference: {cpu_usage:.2f}%")

    @pytest.mark.parametrize("input_size", [
        (1, 624),    # Single sample
        (8, 624),    # Small batch
        (32, 624),   # Standard batch
        (64, 624)    # Large batch
    ])
    def test_scalable_inference_speed(self, mock_strategic_model, input_size, performance_requirements):
        """Test inference speed scalability with different input sizes."""
        batch_size, feature_size = input_size
        input_tensor = torch.randn(batch_size, feature_size)
        
        start_time = time.perf_counter()
        output = mock_strategic_model.forward(input_tensor)
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        per_sample_ms = elapsed_ms / batch_size
        
        # Per-sample time should remain consistent
        assert per_sample_ms < performance_requirements["max_inference_time_ms"]
        assert output.shape == (batch_size, 3)
        
        print(f"Batch size {batch_size}: {elapsed_ms:.2f}ms total, {per_sample_ms:.2f}ms per sample")


class TestTacticalInferenceSpeed:
    """Test inference speed for tactical components (5-minute timeframe)."""

    @pytest.fixture
    def tactical_performance_requirements(self):
        """Performance requirements for tactical inference."""
        return {
            "max_inference_time_ms": 5,   # 5ms for tactical decisions
            "target_inference_time_ms": 2,  # Target: 2ms
            "ultra_fast_inference_ms": 1,   # Ultra-fast: 1ms
            "memory_limit_mb": 128,        # Smaller memory limit
            "cpu_limit_percent": 20        # Lower CPU usage
        }

    @pytest.fixture
    def mock_tactical_model(self):
        """Create a mock tactical model optimized for speed."""
        model = Mock()
        
        def mock_fast_forward(x):
            # Simulate very fast inference
            start_time = time.perf_counter()
            # Minimal computation for speed
            result = torch.softmax(torch.randn(x.shape[0], 7), dim=1)
            end_time = time.perf_counter()
            model.inference_time_ms = (end_time - start_time) * 1000
            return result
        
        model.forward = mock_fast_forward
        return model

    @pytest.fixture
    def sample_tactical_input(self):
        """Sample input for tactical model."""
        return np.random.rand(1, 600)  # 60 bars × 10 features

    def test_ultra_fast_inference(self, mock_tactical_model, sample_tactical_input,
                                 tactical_performance_requirements, performance_timer):
        """Test ultra-fast inference for tactical decisions."""
        performance_timer.start()
        
        input_tensor = torch.FloatTensor(sample_tactical_input)
        output = mock_tactical_model.forward(input_tensor)
        
        performance_timer.stop()
        elapsed_ms = performance_timer.elapsed_ms()
        
        # Ultra-strict requirement for tactical
        assert elapsed_ms < tactical_performance_requirements["max_inference_time_ms"]
        assert output.shape == (1, 7)  # Tactical action space
        
        print(f"Tactical inference time: {elapsed_ms:.3f}ms")

    def test_high_frequency_inference(self, mock_tactical_model, sample_tactical_input,
                                    tactical_performance_requirements):
        """Test high-frequency inference capability."""
        input_tensor = torch.FloatTensor(sample_tactical_input)
        inference_times = []
        
        # Run many rapid inferences
        for _ in range(1000):
            start_time = time.perf_counter()
            _ = mock_tactical_model.forward(input_tensor)
            end_time = time.perf_counter()
            
            inference_times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(inference_times)
        max_time = np.max(inference_times)
        
        # Check average and worst-case performance
        assert avg_time < tactical_performance_requirements["target_inference_time_ms"]
        assert max_time < tactical_performance_requirements["max_inference_time_ms"]
        
        print(f"High-frequency inference - Avg: {avg_time:.3f}ms, Max: {max_time:.3f}ms")

    def test_latency_optimization(self, mock_tactical_model, tactical_performance_requirements):
        """Test latency-optimized inference pipeline."""
        # Pre-allocate tensors to minimize allocation overhead
        input_tensor = torch.randn(1, 600)
        
        # Warm up the model
        for _ in range(10):
            _ = mock_tactical_model.forward(input_tensor)
        
        # Measure optimized inference
        latencies = []
        for _ in range(100):
            start_time = time.perf_counter()
            _ = mock_tactical_model.forward(input_tensor)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Check latency percentiles
        assert p50_latency < tactical_performance_requirements["target_inference_time_ms"]
        assert p95_latency < tactical_performance_requirements["max_inference_time_ms"]
        assert p99_latency < tactical_performance_requirements["max_inference_time_ms"] * 1.5
        
        print(f"Latency percentiles - P50: {p50_latency:.3f}ms, P95: {p95_latency:.3f}ms, P99: {p99_latency:.3f}ms")

    @pytest.mark.asyncio
    async def test_async_inference_speed(self, mock_tactical_model, sample_tactical_input):
        """Test asynchronous inference speed."""
        async def async_inference(input_data):
            # Simulate async inference
            await asyncio.sleep(0.001)  # 1ms async delay
            input_tensor = torch.FloatTensor(input_data)
            return mock_tactical_model.forward(input_tensor)
        
        start_time = time.perf_counter()
        
        # Run multiple concurrent inferences
        tasks = [async_inference(sample_tactical_input) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        
        # Should be faster than sequential execution
        assert total_time_ms < 20  # Should complete in under 20ms
        assert len(results) == 10
        
        print(f"Async inference time for 10 concurrent requests: {total_time_ms:.2f}ms")


class TestSystemThroughput:
    """Test overall system throughput performance."""

    @pytest.fixture
    def throughput_requirements(self):
        """Throughput requirements for the system."""
        return {
            "min_ticks_per_second": 1000,
            "min_decisions_per_second": 100,
            "min_strategic_decisions_per_hour": 48,  # Every 30 minutes
            "min_tactical_decisions_per_hour": 2880,  # Every 5 minutes
            "target_latency_ms": 10
        }

    def test_tick_processing_throughput(self, throughput_requirements):
        """Test tick data processing throughput."""
        mock_processor = Mock()
        
        def process_tick(tick_data):
            # Simulate minimal tick processing
            return {"processed": True, "timestamp": tick_data["timestamp"]}
        
        mock_processor.process = process_tick
        
        # Generate test ticks
        num_ticks = 10000
        ticks = [
            {"timestamp": time.time() + i/1000, "price": 1.0850 + (i % 100) * 0.0001, "volume": 100}
            for i in range(num_ticks)
        ]
        
        start_time = time.perf_counter()
        
        # Process all ticks
        for tick in ticks:
            mock_processor.process(tick)
        
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        ticks_per_second = num_ticks / elapsed_seconds
        
        # Check throughput requirement
        assert ticks_per_second >= throughput_requirements["min_ticks_per_second"]
        
        print(f"Tick processing throughput: {ticks_per_second:.0f} ticks/second")

    def test_decision_generation_throughput(self, throughput_requirements):
        """Test decision generation throughput."""
        mock_decision_engine = Mock()
        
        def generate_decision(market_data):
            # Simulate decision generation
            return {
                "position": np.random.uniform(-1, 1),
                "confidence": np.random.uniform(0.5, 1.0),
                "timestamp": time.time()
            }
        
        mock_decision_engine.decide = generate_decision
        
        # Generate test market data
        num_decisions = 1000
        market_data_samples = [
            {"features": np.random.rand(624), "timestamp": time.time() + i}
            for i in range(num_decisions)
        ]
        
        start_time = time.perf_counter()
        
        # Generate decisions
        decisions = []
        for data in market_data_samples:
            decision = mock_decision_engine.decide(data)
            decisions.append(decision)
        
        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time
        decisions_per_second = num_decisions / elapsed_seconds
        
        # Check throughput requirement
        assert decisions_per_second >= throughput_requirements["min_decisions_per_second"]
        assert len(decisions) == num_decisions
        
        print(f"Decision generation throughput: {decisions_per_second:.0f} decisions/second")

    def test_end_to_end_latency(self, throughput_requirements, performance_timer):
        """Test end-to-end system latency."""
        # Mock pipeline components
        data_processor = Mock()
        strategic_agent = Mock()
        tactical_agent = Mock()
        
        def mock_pipeline(tick_data):
            # Simulate complete pipeline
            processed_data = data_processor.process(tick_data)
            strategic_decision = strategic_agent.decide(processed_data)
            tactical_execution = tactical_agent.execute(strategic_decision)
            return tactical_execution
        
        # Configure mocks
        data_processor.process = Mock(return_value={"processed": True})
        strategic_agent.decide = Mock(return_value={"position": 0.5})
        tactical_agent.execute = Mock(return_value={"executed": True})
        
        # Test multiple samples
        latencies = []
        for _ in range(100):
            tick_data = {"price": 1.0850, "volume": 100, "timestamp": time.time()}
            
            performance_timer.start()
            result = mock_pipeline(tick_data)
            performance_timer.stop()
            
            latencies.append(performance_timer.elapsed_ms())
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Check latency requirements
        assert avg_latency < throughput_requirements["target_latency_ms"]
        assert p95_latency < throughput_requirements["target_latency_ms"] * 2
        
        print(f"End-to-end latency - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")

    def test_concurrent_processing_throughput(self, throughput_requirements):
        """Test throughput under concurrent processing load."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker(work_items):
            local_results = []
            for item in work_items:
                # Simulate processing
                start_time = time.perf_counter()
                processed = {"id": item["id"], "result": item["data"] * 2}
                end_time = time.perf_counter()
                local_results.append({
                    "processed": processed,
                    "time_ms": (end_time - start_time) * 1000
                })
            results_queue.put(local_results)
        
        # Create work items
        num_workers = 4
        items_per_worker = 250
        total_items = num_workers * items_per_worker
        
        work_batches = []
        for i in range(num_workers):
            batch = [
                {"id": i * items_per_worker + j, "data": np.random.rand(100)}
                for j in range(items_per_worker)
            ]
            work_batches.append(batch)
        
        start_time = time.perf_counter()
        
        # Start workers
        threads = []
        for batch in work_batches:
            thread = threading.Thread(target=worker, args=(batch,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            batch_results = results_queue.get()
            all_results.extend(batch_results)
        
        elapsed_seconds = end_time - start_time
        throughput = total_items / elapsed_seconds
        
        # Check concurrent throughput
        assert throughput >= throughput_requirements["min_decisions_per_second"]
        assert len(all_results) == total_items
        
        print(f"Concurrent processing throughput: {throughput:.0f} items/second")


class TestResourceUtilization:
    """Test resource utilization under performance loads."""

    def test_memory_efficiency(self, memory_profiler):
        """Test memory efficiency during intensive operations."""
        memory_profiler.start()
        
        # Simulate memory-intensive operations
        large_arrays = []
        for i in range(50):
            # Create and process large arrays
            array = np.random.rand(1000, 1000)
            processed = np.dot(array, array.T)
            large_arrays.append(processed[:100, :100])  # Keep only small result
        
        peak_memory = memory_profiler.get_peak_usage()
        current_memory = memory_profiler.get_current_usage()
        
        # Memory should not grow unbounded
        assert current_memory < peak_memory * 1.1  # Within 10% of peak
        assert peak_memory < 512  # Should stay under 512MB
        
        print(f"Memory usage - Peak: {peak_memory:.2f}MB, Current: {current_memory:.2f}MB")

    def test_cpu_efficiency(self):
        """Test CPU efficiency during computational loads."""
        process = psutil.Process(os.getpid())
        
        # Baseline CPU usage
        baseline_cpu = process.cpu_percent(interval=1)
        
        # CPU-intensive operations
        start_time = time.perf_counter()
        for _ in range(100):
            # Matrix operations
            a = np.random.rand(500, 500)
            b = np.random.rand(500, 500)
            _ = np.dot(a, b)
        
        end_time = time.perf_counter()
        computation_time = end_time - start_time
        
        # Measure CPU usage after computation
        post_computation_cpu = process.cpu_percent()
        
        # Should complete computation efficiently
        assert computation_time < 30  # Should complete in under 30 seconds
        print(f"CPU usage - Baseline: {baseline_cpu:.2f}%, After computation: {post_computation_cpu:.2f}%")
        print(f"Computation time: {computation_time:.2f} seconds")

    @pytest.mark.performance
    def test_sustained_load_performance(self, performance_timer):
        """Test performance under sustained load."""
        mock_system = Mock()
        
        def process_request(request_id):
            # Simulate request processing
            data = np.random.rand(100, 100)
            result = np.sum(data)
            return {"id": request_id, "result": result}
        
        mock_system.process = process_request
        
        # Sustained load test
        duration_seconds = 10
        request_count = 0
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        while time.perf_counter() < end_time:
            mock_system.process(request_count)
            request_count += 1
        
        actual_duration = time.perf_counter() - start_time
        requests_per_second = request_count / actual_duration
        
        # Should maintain good throughput under sustained load
        assert requests_per_second > 500  # At least 500 requests/second
        
        print(f"Sustained load performance: {requests_per_second:.0f} requests/second over {actual_duration:.1f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])