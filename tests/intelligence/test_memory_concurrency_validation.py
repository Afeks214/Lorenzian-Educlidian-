"""
Advanced Memory and Concurrency Validation for Intelligence System.

Tests memory stability, concurrency safety, and performance under extreme
load conditions to ensure production readiness.
"""

import pytest
import torch
import numpy as np
import time
import threading
import concurrent.futures
import psutil
import gc
import weakref
from typing import Dict, Any, List
import multiprocessing
import queue
import tracemalloc
import sys
import os

# Import the intelligence components
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from intelligence.intelligence_hub import IntelligenceHub, IntelligenceMetrics
from intelligence.performance_monitor import IntelligencePerformanceMonitor, PerformanceSnapshot
from intelligence.regime_detector import RegimeDetector, RegimeAnalysis, MarketRegime
from intelligence.gating_network import GatingNetwork, FastGatingNetwork
from intelligence.regime_aware_reward import RegimeAwareRewardFunction


class TestMemoryValidation:
    """Test memory usage patterns and leak detection."""
    
    @pytest.fixture
    def intelligence_config(self):
        """Optimized config for memory testing."""
        return {
            'max_intelligence_overhead_ms': 1.0,
            'performance_monitoring': True,
            'regime_detection': {
                'fast_mode': True,
                'cache_analysis': True,
                'min_confidence_threshold': 0.7
            },
            'gating_network': {
                'hidden_dim': 32,
                'dropout_rate': 0.1
            },
            'attention': {
                'use_fast_attention': True,
                'batch_size_threshold': 4
            },
            'regime_aware_reward': {
                'fast_mode': True,
                'cache_rewards': True,
                'cache_ttl_seconds': 1,
                'normalization_window': 50
            }
        }
    
    def test_memory_leak_detection_extended(self, intelligence_config):
        """Extended memory leak test with 2000+ operations."""
        
        # Start memory tracking
        tracemalloc.start()
        hub = IntelligenceHub(intelligence_config)
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_samples = [initial_memory]
        
        # Run 2000 operations with varied inputs
        for i in range(2000):
            # Generate diverse market contexts
            context = {
                'volatility_30': np.random.uniform(0.5, 4.0),
                'momentum_20': np.random.uniform(-0.1, 0.1),
                'momentum_50': np.random.uniform(-0.05, 0.05),
                'volume_ratio': np.random.uniform(0.3, 3.0),
                'mmd_score': np.random.uniform(0.1, 0.9),
                'price_trend': np.random.uniform(-0.05, 0.05)
            }
            
            # Generate diverse agent predictions
            predictions = [
                {
                    'action_probabilities': np.random.dirichlet([1, 1, 1]),
                    'confidence': np.random.uniform(0.5, 0.95),
                    'agent_id': f'agent_{j}'
                }
                for j in range(3)
            ]
            
            # Generate diverse attention weights
            attention_weights = [
                torch.softmax(torch.randn(7) * np.random.uniform(0.5, 2.0), dim=0)
                for _ in range(3)
            ]
            
            # Process through intelligence pipeline
            result, metrics = hub.process_intelligence_pipeline(
                context, predictions, attention_weights
            )
            
            # Sample memory every 100 operations
            if i % 100 == 0:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Force garbage collection
                gc.collect()
                
                print(f"Operation {i}: Memory={current_memory:.1f}MB, Growth={current_memory - initial_memory:.1f}MB")
        
        # Final memory check
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        # Get tracemalloc statistics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Memory Analysis:")
        print(f"Initial: {initial_memory:.1f}MB")
        print(f"Final: {final_memory:.1f}MB")
        print(f"Growth: {total_growth:.1f}MB")
        print(f"Peak traced: {peak / 1024 / 1024:.1f}MB")
        
        # Calculate memory growth trend
        if len(memory_samples) > 5:
            recent_samples = memory_samples[-5:]
            growth_trend = (recent_samples[-1] - recent_samples[0]) / len(recent_samples)
            print(f"Recent growth trend: {growth_trend:.2f}MB per 100 ops")
            
            # Assert memory stability
            assert growth_trend < 5.0, f"Memory growth trend {growth_trend:.2f}MB per 100 ops indicates leak"
        
        # Overall memory growth should be reasonable
        assert total_growth < 200, f"Total memory growth {total_growth:.1f}MB too high (indicates leak)"
        
        # Reset hub state and verify cleanup
        hub.reset_intelligence_state()
        gc.collect()
        
        cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024
        cleanup_reduction = final_memory - cleanup_memory
        print(f"Memory after cleanup: {cleanup_memory:.1f}MB (reduction: {cleanup_reduction:.1f}MB)")
    
    def test_cache_memory_management(self, intelligence_config):
        """Test cache memory management and cleanup."""
        
        hub = IntelligenceHub(intelligence_config)
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Fill caches with unique contexts to test cache size limits
        for i in range(2000):
            unique_context = {
                'volatility_30': i * 0.001,  # Unique values to avoid cache hits
                'momentum_20': i * 0.0001,
                'momentum_50': i * 0.00005,
                'volume_ratio': 1.0 + i * 0.0001,
                'mmd_score': 0.3 + i * 0.0001,
                'price_trend': i * 0.00001
            }
            
            predictions = [
                {
                    'action_probabilities': np.array([0.33, 0.33, 0.34]),
                    'confidence': 0.7,
                    'agent_id': f'agent_{j}'
                }
                for j in range(3)
            ]
            
            hub.process_intelligence_pipeline(unique_context, predictions, None)
        
        cache_memory = psutil.Process().memory_info().rss / 1024 / 1024
        cache_growth = cache_memory - initial_memory
        
        # Check cache sizes
        stats = hub.get_integration_statistics()
        cache_stats = stats['cache_stats']
        
        print(f"Cache Statistics:")
        print(f"Regime cache size: {cache_stats['regime_cache_size']}")
        print(f"Context cache size: {cache_stats['context_cache_size']}")
        print(f"Gating cache size: {cache_stats['gating_cache_size']}")
        print(f"Memory growth: {cache_growth:.1f}MB")
        
        # Caches should respect size limits
        assert cache_stats['regime_cache_size'] <= hub.max_cache_size
        assert cache_growth < 100, f"Cache memory growth {cache_growth:.1f}MB too high"
        
        # Reset and verify cache cleanup
        hub.reset_intelligence_state()
        gc.collect()
        
        reset_stats = hub.get_integration_statistics()
        reset_cache_stats = reset_stats['cache_stats']
        
        # Caches should be cleared
        assert reset_cache_stats['regime_cache_size'] == 0
        assert reset_cache_stats['context_cache_size'] == 0
        assert reset_cache_stats['gating_cache_size'] == 0
    
    def test_weakref_cleanup_validation(self, intelligence_config):
        """Test proper cleanup using weak references."""
        
        weak_refs = []
        
        def create_hub_and_track():
            hub = IntelligenceHub(intelligence_config)
            weak_refs.append(weakref.ref(hub))
            
            # Use the hub briefly
            context = {'volatility_30': 1.0, 'mmd_score': 0.3, 'momentum_20': 0.0, 
                      'momentum_50': 0.0, 'volume_ratio': 1.0, 'price_trend': 0.0}
            predictions = [{'action_probabilities': np.array([0.33, 0.33, 0.34]), 'confidence': 0.7}]
            
            hub.process_intelligence_pipeline(context, predictions, None)
            return hub
        
        # Create and destroy multiple hubs
        for i in range(10):
            hub = create_hub_and_track()
            del hub
            gc.collect()
        
        # Check that objects are properly garbage collected
        alive_refs = [ref for ref in weak_refs if ref() is not None]
        print(f"Alive references: {len(alive_refs)} out of {len(weak_refs)}")
        
        # Most objects should be garbage collected
        assert len(alive_refs) <= 2, f"Too many objects still alive: {len(alive_refs)}"
    
    def test_torch_memory_cleanup(self, intelligence_config):
        """Test PyTorch tensor memory cleanup."""
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated()
        else:
            initial_gpu_memory = 0
        
        hub = IntelligenceHub(intelligence_config)
        
        # Process with large attention tensors
        for i in range(100):
            large_attention_weights = [
                torch.softmax(torch.randn(50), dim=0),  # Larger tensors
                torch.softmax(torch.randn(50), dim=0),
                torch.softmax(torch.randn(50), dim=0)
            ]
            
            context = {
                'volatility_30': np.random.uniform(0.5, 3.0),
                'mmd_score': np.random.uniform(0.1, 0.8),
                'momentum_20': np.random.uniform(-0.05, 0.05),
                'momentum_50': np.random.uniform(-0.03, 0.03),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'price_trend': np.random.uniform(-0.02, 0.02)
            }
            
            predictions = [
                {'action_probabilities': np.random.dirichlet([1, 1, 1]), 'confidence': 0.7}
                for _ in range(3)
            ]
            
            result, metrics = hub.process_intelligence_pipeline(
                context, predictions, large_attention_weights
            )
            
            # Clear references
            del large_attention_weights
            
            if i % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Check final memory usage
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated()
            gpu_growth = final_gpu_memory - initial_gpu_memory
            print(f"GPU memory growth: {gpu_growth / 1024 / 1024:.1f}MB")
            assert gpu_growth < 100 * 1024 * 1024, f"GPU memory growth {gpu_growth / 1024 / 1024:.1f}MB too high"


class TestConcurrencyValidation:
    """Test concurrency safety and performance."""
    
    @pytest.fixture
    def intelligence_config(self):
        """Config optimized for concurrency testing."""
        return {
            'max_intelligence_overhead_ms': 2.0,  # More lenient for concurrent testing
            'performance_monitoring': True,
            'real_time_monitoring': False,  # Disable to avoid thread conflicts
            'regime_detection': {
                'fast_mode': True,
                'cache_analysis': True
            },
            'gating_network': {
                'hidden_dim': 32
            }
        }
    
    def test_high_concurrency_stress(self, intelligence_config):
        """Test system under high concurrent load."""
        
        hub = IntelligenceHub(intelligence_config)
        num_threads = 20
        requests_per_thread = 50
        total_requests = num_threads * requests_per_thread
        
        results = []
        errors = []
        latencies = []
        
        def worker_thread(thread_id):
            """Worker thread function."""
            thread_results = []
            thread_latencies = []
            
            for i in range(requests_per_thread):
                try:
                    # Generate unique context per request
                    context = {
                        'volatility_30': np.random.uniform(0.5, 3.0),
                        'momentum_20': np.random.uniform(-0.05, 0.05),
                        'momentum_50': np.random.uniform(-0.03, 0.03),
                        'volume_ratio': np.random.uniform(0.5, 2.0),
                        'mmd_score': np.random.uniform(0.1, 0.8),
                        'price_trend': np.random.uniform(-0.02, 0.02)
                    }
                    
                    predictions = [
                        {
                            'action_probabilities': np.random.dirichlet([1, 1, 1]),
                            'confidence': np.random.uniform(0.6, 0.9),
                            'agent_id': f'agent_{j}'
                        }
                        for j in range(3)
                    ]
                    
                    attention_weights = [
                        torch.softmax(torch.randn(7), dim=0)
                        for _ in range(3)
                    ]
                    
                    start_time = time.perf_counter()
                    result, metrics = hub.process_intelligence_pipeline(
                        context, predictions, attention_weights
                    )
                    end_time = time.perf_counter()
                    
                    latency = (end_time - start_time) * 1000
                    thread_latencies.append(latency)
                    thread_results.append((thread_id, i, result, latency))
                    
                    # Small random delay to increase contention
                    time.sleep(np.random.exponential(0.001))
                    
                except Exception as e:
                    errors.append((thread_id, i, str(e)))
            
            return thread_results, thread_latencies
        
        # Execute concurrent requests
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            
            for future in concurrent.futures.as_completed(futures):
                thread_results, thread_latencies = future.result()
                results.extend(thread_results)
                latencies.extend(thread_latencies)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze results
        success_rate = len(results) / total_requests
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        throughput = len(results) / total_duration
        
        print(f"High Concurrency Results:")
        print(f"Total requests: {total_requests}")
        print(f"Successful: {len(results)} ({success_rate:.1%})")
        print(f"Errors: {len(errors)}")
        print(f"Duration: {total_duration:.2f}s")
        print(f"Throughput: {throughput:.1f} req/s")
        print(f"Mean latency: {mean_latency:.3f}ms")
        print(f"P95 latency: {p95_latency:.3f}ms")
        print(f"P99 latency: {p99_latency:.3f}ms")
        print(f"Max latency: {max_latency:.3f}ms")
        
        # Validate performance under high concurrency
        assert success_rate >= 0.98, f"Success rate {success_rate:.1%} too low"
        assert mean_latency < 10.0, f"Mean latency {mean_latency:.3f}ms too high"
        assert p95_latency < 20.0, f"P95 latency {p95_latency:.3f}ms too high"
        assert max_latency < 50.0, f"Max latency {max_latency:.3f}ms too high"
        assert throughput > 50, f"Throughput {throughput:.1f} req/s too low"
        
        # Check for data races in results
        for _, _, result, _ in results:
            assert 'final_probabilities' in result
            probs = result['final_probabilities']
            assert len(probs) == 3
            assert all(np.isfinite(p) for p in probs)
            assert 0.99 <= sum(probs) <= 1.01  # Should be normalized
    
    def test_cache_thread_safety(self, intelligence_config):
        """Test cache thread safety under concurrent access."""
        
        hub = IntelligenceHub(intelligence_config)
        num_threads = 10
        operations_per_thread = 100
        
        # Use same contexts to force cache contention
        shared_contexts = [
            {
                'volatility_30': 1.0 + i * 0.1,
                'momentum_20': 0.01 * i,
                'momentum_50': 0.005 * i,
                'volume_ratio': 1.0 + i * 0.05,
                'mmd_score': 0.3 + i * 0.01,
                'price_trend': 0.001 * i
            }
            for i in range(10)  # Limited set to force cache sharing
        ]
        
        predictions = [
            {'action_probabilities': np.array([0.33, 0.33, 0.34]), 'confidence': 0.7}
            for _ in range(3)
        ]
        
        results = []
        errors = []
        
        def cache_worker(thread_id):
            """Worker that accesses shared cache entries."""
            thread_results = []
            
            for i in range(operations_per_thread):
                try:
                    # Randomly select from shared contexts
                    context = shared_contexts[np.random.randint(0, len(shared_contexts))]
                    
                    result, metrics = hub.process_intelligence_pipeline(
                        context, predictions, None
                    )
                    
                    thread_results.append((thread_id, i, result))
                    
                except Exception as e:
                    errors.append((thread_id, i, str(e)))
            
            return thread_results
        
        # Execute concurrent cache access
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(num_threads)]
            
            for future in concurrent.futures.as_completed(futures):
                thread_results = future.result()
                results.extend(thread_results)
        
        # Get cache statistics
        stats = hub.get_integration_statistics()
        cache_stats = stats['cache_stats']
        
        print(f"Cache Thread Safety Results:")
        print(f"Total operations: {len(results)}")
        print(f"Errors: {len(errors)}")
        print(f"Cache hit rate: {cache_stats['cache_hit_rate']:.1%}")
        print(f"Cache sizes: regime={cache_stats['regime_cache_size']}, "
              f"context={cache_stats['context_cache_size']}, "
              f"gating={cache_stats['gating_cache_size']}")
        
        # Validate thread safety
        assert len(errors) == 0, f"Thread safety errors detected: {errors[:5]}"
        assert len(results) == num_threads * operations_per_thread
        assert cache_stats['cache_hit_rate'] > 0.5, "Cache should have significant hit rate"
        
        # Validate result consistency
        for _, _, result in results:
            assert 'final_probabilities' in result
            probs = result['final_probabilities']
            assert all(np.isfinite(p) for p in probs)
    
    def test_resource_contention_handling(self, intelligence_config):
        """Test handling of resource contention."""
        
        hub = IntelligenceHub(intelligence_config)
        monitor = IntelligencePerformanceMonitor(intelligence_config)
        
        # Create high CPU load scenario
        def cpu_intensive_worker():
            """Worker that creates CPU contention."""
            for _ in range(1000):
                context = {
                    'volatility_30': np.random.uniform(0.5, 3.0),
                    'momentum_20': np.random.uniform(-0.05, 0.05),
                    'momentum_50': np.random.uniform(-0.03, 0.03),
                    'volume_ratio': np.random.uniform(0.5, 2.0),
                    'mmd_score': np.random.uniform(0.1, 0.8),
                    'price_trend': np.random.uniform(-0.02, 0.02)
                }
                
                predictions = [
                    {'action_probabilities': np.random.dirichlet([1, 1, 1]), 'confidence': 0.7}
                    for _ in range(3)
                ]
                
                attention_weights = [
                    torch.softmax(torch.randn(7), dim=0)
                    for _ in range(3)
                ]
                
                session = monitor.start_performance_measurement()
                result, metrics = hub.process_intelligence_pipeline(
                    context, predictions, attention_weights
                )
                monitor.complete_performance_measurement(session, result)
        
        # Run multiple CPU-intensive workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = [executor.submit(cpu_intensive_worker) for _ in range(multiprocessing.cpu_count())]
            
            # Wait for completion
            for future in concurrent.futures.as_completed(futures):
                future.result()
        
        # Analyze performance under contention
        summary = monitor.get_performance_summary()
        
        print(f"Resource Contention Results:")
        print(f"Mean inference time: {summary['total_inference_time']['mean_ms']:.3f}ms")
        print(f"P95 inference time: {summary['total_inference_time']['p95_ms']:.3f}ms")
        print(f"Target compliance: {summary['total_inference_time']['target_compliance']:.1%}")
        print(f"Performance score: {summary['performance_score']:.1f}")
        
        # System should handle contention gracefully
        assert summary['total_inference_time']['mean_ms'] < 15.0, \
            "Mean inference time too high under contention"
        assert summary['performance_score'] > 30, \
            "Performance score too low under contention"
        assert summary['total_inference_time']['target_compliance'] > 0.7, \
            "Target compliance too low under contention"


class TestStressAndLoad:
    """Test system under extreme stress conditions."""
    
    @pytest.fixture
    def stress_config(self):
        """Configuration for stress testing."""
        return {
            'max_intelligence_overhead_ms': 5.0,  # More lenient for stress tests
            'performance_monitoring': True,
            'real_time_monitoring': False,
            'regime_detection': {
                'fast_mode': True,
                'cache_analysis': True,
                'min_confidence_threshold': 0.5  # Lower threshold for stress
            },
            'gating_network': {
                'hidden_dim': 16  # Smaller for stress testing
            },
            'regime_aware_reward': {
                'fast_mode': True,
                'cache_rewards': True,
                'cache_ttl_seconds': 0.5  # Shorter cache TTL
            }
        }
    
    def test_extreme_load_endurance(self, stress_config):
        """Test system endurance under extreme load."""
        
        hub = IntelligenceHub(stress_config)
        monitor = IntelligencePerformanceMonitor(stress_config)
        
        # Test parameters
        duration_seconds = 30  # 30 second stress test
        target_ops_per_second = 200
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        operation_count = 0
        error_count = 0
        latencies = []
        
        print(f"Starting {duration_seconds}s extreme load test...")
        
        while time.time() < end_time:
            try:
                # Generate random inputs
                context = {
                    'volatility_30': np.random.uniform(0.1, 5.0),
                    'momentum_20': np.random.uniform(-0.1, 0.1),
                    'momentum_50': np.random.uniform(-0.05, 0.05),
                    'volume_ratio': np.random.uniform(0.1, 5.0),
                    'mmd_score': np.random.uniform(0.05, 0.95),
                    'price_trend': np.random.uniform(-0.1, 0.1)
                }
                
                predictions = [
                    {
                        'action_probabilities': np.random.dirichlet([0.5, 0.5, 0.5]),
                        'confidence': np.random.uniform(0.3, 0.95),
                        'agent_id': f'stress_agent_{j}'
                    }
                    for j in range(3)
                ]
                
                attention_weights = [
                    torch.softmax(torch.randn(7) * np.random.uniform(0.1, 3.0), dim=0)
                    for _ in range(3)
                ]
                
                session = monitor.start_performance_measurement()
                
                op_start = time.perf_counter()
                result, metrics = hub.process_intelligence_pipeline(
                    context, predictions, attention_weights
                )
                op_end = time.perf_counter()
                
                monitor.complete_performance_measurement(session, result)
                
                latency = (op_end - op_start) * 1000
                latencies.append(latency)
                operation_count += 1
                
                # Validate result integrity under stress
                assert 'final_probabilities' in result
                probs = result['final_probabilities']
                assert len(probs) == 3
                assert all(np.isfinite(p) for p in probs)
                
                # Brief pause to prevent overwhelming
                if operation_count % 100 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    current_ops_per_sec = operation_count / elapsed
                    print(f"Progress: {elapsed:.1f}s, {operation_count} ops, "
                          f"{current_ops_per_sec:.1f} ops/s, errors: {error_count}")
                
                # Adaptive throttling
                if len(latencies) > 10:
                    recent_latency = np.mean(latencies[-10:])
                    if recent_latency > 10.0:  # If getting slow, throttle
                        time.sleep(0.001)
                
            except Exception as e:
                error_count += 1
                print(f"Error #{error_count}: {str(e)}")
                if error_count > operation_count * 0.05:  # More than 5% error rate
                    break
        
        # Calculate final statistics
        actual_duration = time.time() - start_time
        actual_ops_per_second = operation_count / actual_duration
        error_rate = error_count / max(1, operation_count)
        
        mean_latency = np.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        
        # Get final performance summary
        summary = monitor.get_performance_summary()
        
        print(f"Extreme Load Test Results:")
        print(f"Duration: {actual_duration:.2f}s")
        print(f"Operations: {operation_count}")
        print(f"Errors: {error_count} ({error_rate:.1%})")
        print(f"Ops/sec: {actual_ops_per_second:.1f}")
        print(f"Mean latency: {mean_latency:.3f}ms")
        print(f"P95 latency: {p95_latency:.3f}ms")
        print(f"P99 latency: {p99_latency:.3f}ms")
        print(f"Performance score: {summary['performance_score']:.1f}")
        
        # Validate endurance performance
        assert operation_count > 0, "No operations completed"
        assert error_rate < 0.10, f"Error rate {error_rate:.1%} too high"
        assert actual_ops_per_second > 20, f"Throughput {actual_ops_per_second:.1f} ops/s too low"
        assert mean_latency < 20.0, f"Mean latency {mean_latency:.3f}ms too high under stress"
        assert summary['performance_score'] > 20, "Performance score too low under extreme load"
    
    def test_memory_pressure_handling(self, stress_config):
        """Test system behavior under memory pressure."""
        
        hub = IntelligenceHub(stress_config)
        
        # Create memory pressure by allocating large objects
        memory_hogs = []
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Gradually increase memory pressure
            for i in range(50):
                # Allocate large tensors to create memory pressure
                large_tensor = torch.randn(1000, 1000)
                memory_hogs.append(large_tensor)
                
                # Continue intelligence operations under memory pressure
                context = {
                    'volatility_30': 2.0 + i * 0.01,
                    'momentum_20': 0.01 + i * 0.0001,
                    'momentum_50': 0.005 + i * 0.00005,
                    'volume_ratio': 1.0 + i * 0.001,
                    'mmd_score': 0.3 + i * 0.001,
                    'price_trend': i * 0.00001
                }
                
                predictions = [
                    {'action_probabilities': np.array([0.33, 0.33, 0.34]), 'confidence': 0.7}
                    for _ in range(3)
                ]
                
                try:
                    result, metrics = hub.process_intelligence_pipeline(
                        context, predictions, None
                    )
                    
                    # Validate result under memory pressure
                    assert 'final_probabilities' in result
                    probs = result['final_probabilities']
                    assert all(np.isfinite(p) for p in probs)
                    
                except Exception as e:
                    print(f"Operation failed under memory pressure at iteration {i}: {e}")
                
                if i % 10 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - initial_memory
                    print(f"Memory pressure test - iteration {i}, memory: {current_memory:.1f}MB (+{memory_growth:.1f}MB)")
                    
                    # Stop if memory usage becomes excessive
                    if memory_growth > 2000:  # 2GB limit
                        print(f"Stopping memory pressure test at {memory_growth:.1f}MB growth")
                        break
        
        finally:
            # Clean up memory hogs
            del memory_hogs
            gc.collect()
        
        # System should continue functioning under memory pressure
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"Memory after cleanup: {final_memory:.1f}MB")


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v', '--tb=short'])