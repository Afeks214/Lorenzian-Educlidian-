#!/usr/bin/env python3
"""
MARL P99 Latency Validation Test Suite
======================================

This test suite validates the ultra-low latency optimizations implemented in
the async_inference_pool.py file, specifically targeting P99 latency reduction
from 11.227ms to under 2ms.

Test Coverage:
- P99 latency measurement and validation
- Vectorized batch processing performance
- JIT compilation effectiveness
- GPU acceleration benchmarks
- Tensor caching optimization
- Lock-free data structure performance
"""

import asyncio
import time
import numpy as np
import torch
import pytest
import logging
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from tactical.async_inference_pool import (
    AsyncInferencePool,
    get_latency_stats,
    reset_latency_monitoring,
    benchmark_latency,
    fvg_inference_fast,
    momentum_inference_fast,
    entry_inference_fast,
    _tensor_cache,
    _latency_monitor
)

logger = logging.getLogger(__name__)

class TestMARLP99LatencyValidation:
    """Test suite for MARL P99 latency validation."""
    
    @pytest.fixture
    async def inference_pool(self):
        """Create and initialize inference pool for testing."""
        pool = AsyncInferencePool(
            max_workers_per_type=2,
            max_queue_size=1000,
            batch_timeout_ms=5.0,
            max_batch_size=16
        )
        await pool.initialize()
        await pool.start()
        yield pool
        await pool.stop()
    
    @pytest.fixture
    def test_matrix(self):
        """Generate test matrix data."""
        return np.random.randn(60, 7).astype(np.float32)
    
    @pytest.fixture
    def test_synergy_event(self):
        """Generate test synergy event."""
        return {
            "synergy_type": "test",
            "direction": 1,
            "confidence": 0.8,
            "timestamp": time.time()
        }
    
    @pytest.mark.asyncio
    async def test_jit_compilation_performance(self, test_matrix):
        """Test JIT compilation performance improvements."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        matrix_tensor = torch.from_numpy(test_matrix).to(device)
        volume_tensor = matrix_tensor[:, 6]
        
        # Test FVG inference
        start_time = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                result = fvg_inference_fast(matrix_tensor, 1.0)
        fvg_time = (time.perf_counter() - start_time) / 100
        
        # Test momentum inference
        start_time = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                result = momentum_inference_fast(matrix_tensor, volume_tensor)
        momentum_time = (time.perf_counter() - start_time) / 100
        
        # Test entry inference
        start_time = time.perf_counter()
        for _ in range(100):
            with torch.no_grad():
                result = entry_inference_fast(matrix_tensor, volume_tensor)
        entry_time = (time.perf_counter() - start_time) / 100
        
        # Validate JIT performance
        assert fvg_time < 0.001, f"FVG inference too slow: {fvg_time*1000:.3f}ms"
        assert momentum_time < 0.001, f"Momentum inference too slow: {momentum_time*1000:.3f}ms"
        assert entry_time < 0.001, f"Entry inference too slow: {entry_time*1000:.3f}ms"
        
        logger.info(f"JIT Performance: FVG={fvg_time*1000:.3f}ms, Momentum={momentum_time*1000:.3f}ms, Entry={entry_time*1000:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_vectorized_batch_processing(self, inference_pool, test_matrix, test_synergy_event):
        """Test vectorized batch processing performance."""
        # Reset monitoring
        reset_latency_monitoring()
        
        # Test vectorized processing
        start_time = time.perf_counter()
        results = await inference_pool.submit_inference_jobs_vectorized(
            test_matrix, test_synergy_event, "test_vectorized"
        )
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Validate results
        assert len(results) == 3, "Should return results for all 3 agents"
        assert all(isinstance(r["action"], int) for r in results), "Actions should be integers"
        assert all(len(r["probabilities"]) == 3 for r in results), "Should have 3 probabilities"
        assert all(r["reasoning"]["jit_compiled"] for r in results), "Should use JIT compilation"
        assert all(r["reasoning"]["vectorized"] for r in results), "Should use vectorized processing"
        
        # Validate latency target
        assert processing_time_ms < 2.0, f"Vectorized processing too slow: {processing_time_ms:.3f}ms"
        
        logger.info(f"Vectorized processing time: {processing_time_ms:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_p99_latency_benchmark(self, inference_pool, test_matrix, test_synergy_event):
        """Test P99 latency with comprehensive benchmarking."""
        # Reset monitoring
        reset_latency_monitoring()
        
        # Warm up
        for _ in range(10):
            await inference_pool.submit_inference_jobs_vectorized(
                test_matrix, test_synergy_event, "warmup"
            )
        
        # Benchmark with 1000 iterations
        benchmark_results = await benchmark_latency(inference_pool, iterations=1000)
        
        # Validate benchmark results
        assert "latency_results" in benchmark_results
        assert "performance_validation" in benchmark_results
        
        latency_stats = benchmark_results["latency_results"]
        
        # Critical P99 validation
        p99_latency = latency_stats.get("p99_latency_ms", float('inf'))
        assert p99_latency < 2.0, f"P99 latency target missed: {p99_latency:.3f}ms > 2.0ms"
        
        # Additional performance validations
        assert latency_stats.get("p95_latency_ms", float('inf')) < 1.5, "P95 latency should be under 1.5ms"
        assert latency_stats.get("mean_latency_ms", float('inf')) < 1.0, "Mean latency should be under 1.0ms"
        assert latency_stats.get("sub_2ms_percentage", 0) > 95.0, "At least 95% of requests should be sub-2ms"
        
        # Throughput validation
        throughput = benchmark_results["benchmark_info"]["avg_throughput_rps"]
        assert throughput > 500, f"Throughput too low: {throughput:.1f} RPS"
        
        logger.info(f"P99 Latency Benchmark Results:")
        logger.info(f"  P99: {p99_latency:.3f}ms")
        logger.info(f"  P95: {latency_stats.get('p95_latency_ms'):.3f}ms")
        logger.info(f"  Mean: {latency_stats.get('mean_latency_ms'):.3f}ms")
        logger.info(f"  Sub-2ms: {latency_stats.get('sub_2ms_percentage'):.1f}%")
        logger.info(f"  Throughput: {throughput:.1f} RPS")
        
        return benchmark_results
    
    @pytest.mark.asyncio
    async def test_tensor_cache_performance(self, test_matrix):
        """Test tensor cache performance optimization."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clear cache
        _tensor_cache.clear()
        
        # Test cache miss performance
        start_time = time.perf_counter()
        for i in range(100):
            tensor = _tensor_cache.get_tensor(f"test_key_{i}", (60, 7))
            tensor.copy_(torch.from_numpy(test_matrix).to(device), non_blocking=True)
        cache_miss_time = time.perf_counter() - start_time
        
        # Test cache hit performance
        start_time = time.perf_counter()
        for i in range(100):
            tensor = _tensor_cache.get_tensor("test_key_0", (60, 7))  # Same key
            tensor.copy_(torch.from_numpy(test_matrix).to(device), non_blocking=True)
        cache_hit_time = time.perf_counter() - start_time
        
        # Cache hits should be significantly faster
        assert cache_hit_time < cache_miss_time * 0.8, "Cache hits should be faster than cache misses"
        
        logger.info(f"Tensor Cache Performance: Miss={cache_miss_time*10:.3f}ms, Hit={cache_hit_time*10:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_gpu_acceleration_validation(self, inference_pool, test_matrix, test_synergy_event):
        """Test GPU acceleration if available."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for acceleration testing")
        
        # Test GPU-accelerated inference
        results = await inference_pool.submit_inference_jobs_vectorized(
            test_matrix, test_synergy_event, "gpu_test"
        )
        
        # Validate GPU acceleration is used
        assert all(r["reasoning"]["gpu_accelerated"] for r in results), "GPU acceleration should be enabled"
        
        # GPU processing should be very fast
        processing_time = results[0]["processing_time_ms"]
        assert processing_time < 1.0, f"GPU processing should be under 1ms: {processing_time:.3f}ms"
        
        logger.info(f"GPU acceleration validated: {processing_time:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_stress_test_latency_stability(self, inference_pool, test_matrix, test_synergy_event):
        """Test latency stability under stress conditions."""
        reset_latency_monitoring()
        
        # Stress test with rapid-fire requests
        tasks = []
        for i in range(100):
            task = asyncio.create_task(
                inference_pool.submit_inference_jobs_vectorized(
                    test_matrix, test_synergy_event, f"stress_test_{i}"
                )
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Get latency statistics
        stats = get_latency_stats()
        
        # Validate stability under stress
        p99_latency = stats.get("p99_latency_ms", float('inf'))
        assert p99_latency < 3.0, f"P99 latency under stress: {p99_latency:.3f}ms"
        
        # Check for consistency
        max_latency = stats.get("max_latency_ms", float('inf'))
        assert max_latency < 10.0, f"Max latency too high: {max_latency:.3f}ms"
        
        logger.info(f"Stress Test Results: P99={p99_latency:.3f}ms, Max={max_latency:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_latency_monitoring_accuracy(self, inference_pool, test_matrix, test_synergy_event):
        """Test latency monitoring accuracy."""
        reset_latency_monitoring()
        
        # Record known processing times
        measured_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            await inference_pool.submit_inference_jobs_vectorized(
                test_matrix, test_synergy_event, "accuracy_test"
            )
            end_time = time.perf_counter()
            measured_times.append((end_time - start_time) * 1000)
        
        # Get monitoring stats
        stats = get_latency_stats()
        
        # Validate accuracy
        measured_mean = np.mean(measured_times)
        monitored_mean = stats.get("mean_latency_ms", 0)
        
        # Should be within 5% accuracy
        accuracy_diff = abs(measured_mean - monitored_mean) / measured_mean
        assert accuracy_diff < 0.05, f"Monitoring accuracy error: {accuracy_diff:.3%}"
        
        logger.info(f"Monitoring Accuracy: Measured={measured_mean:.3f}ms, Monitored={monitored_mean:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, inference_pool, test_matrix, test_synergy_event):
        """Test performance regression detection."""
        # Baseline benchmark
        baseline_results = await benchmark_latency(inference_pool, iterations=100)
        baseline_p99 = baseline_results["latency_results"]["p99_latency_ms"]
        
        # Second benchmark
        reset_latency_monitoring()
        current_results = await benchmark_latency(inference_pool, iterations=100)
        current_p99 = current_results["latency_results"]["p99_latency_ms"]
        
        # Performance should be consistent
        regression_threshold = 0.5  # 0.5ms
        regression = current_p99 - baseline_p99
        
        assert regression < regression_threshold, \
            f"Performance regression detected: {regression:.3f}ms increase"
        
        logger.info(f"Performance Consistency: Baseline={baseline_p99:.3f}ms, Current={current_p99:.3f}ms")
    
    def test_optimization_features_enabled(self, inference_pool):
        """Test that all optimization features are enabled."""
        report = inference_pool.get_latency_report()
        
        optimization_status = report["optimization_status"]
        
        # Validate all optimizations are enabled
        assert optimization_status["jit_compilation"] == "enabled"
        assert optimization_status["vectorized_processing"] == "enabled"
        assert optimization_status["tensor_caching"] == "enabled"
        assert optimization_status["lock_free_queues"] == "enabled"
        
        logger.info("All optimization features validated as enabled")

if __name__ == "__main__":
    """Run the latency validation tests."""
    pytest.main([__file__, "-v", "-s", "--tb=short"])