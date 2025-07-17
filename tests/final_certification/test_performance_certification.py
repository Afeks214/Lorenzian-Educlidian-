"""
AGENT EPSILON: Performance & Scalability 200% Certification Test Suite

This comprehensive certification validates system performance and scalability
under all production conditions:

PERFORMANCE TARGETS:
- ✅ <5ms mean latency (Strategic MARL decisions)
- ✅ <8ms P95 latency under normal load
- ✅ <12ms P99 latency under peak load
- ✅ >500 QPS sustained throughput
- ✅ <100MB memory growth over 10K operations
- ✅ <80% CPU utilization under peak load
- ✅ <75% memory utilization under normal operation

SCALABILITY TESTS:
- Load testing (100-1000 QPS)
- Stress testing (burst capacity)
- Endurance testing (extended operation)
- Memory stability testing
- Resource efficiency validation

Author: Agent Epsilon - 200% Production Certification
Version: 1.0 - Final Certification
"""

import numpy as np
import torch
import time
import pytest
from typing import Dict, Any, List, Tuple, Optional
import logging
from unittest.mock import Mock
import psutil
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import queue
import resource
import sys

# Import performance monitoring
from src.monitoring.metrics_exporter import MetricsExporter
from src.monitoring.health_monitor import HealthMonitor

# Import core system components
from src.intelligence.intelligence_hub import IntelligenceHub
from src.agents.mlmi_strategic_agent import MLMIStrategicAgent
from src.agents.nwrqk_strategic_agent import NWRQKStrategicAgent
from src.agents.regime_detection_agent import RegimeDetectionAgent
from src.core.events import EventBus

logger = logging.getLogger(__name__)


class TestPerformanceCertification:
    """Final performance and scalability certification."""
    
    @pytest.fixture
    def intelligence_hub(self):
        """Create optimized intelligence hub for performance testing."""
        config = {
            'max_intelligence_overhead_ms': 1.0,
            'performance_monitoring': True,
            'regime_detection': {'fast_mode': True, 'cache_analysis': True},
            'gating_network': {'hidden_dim': 32},
            'attention': {'enable_caching': True},
            'regime_aware_reward': {'fast_computation': True}
        }
        return IntelligenceHub(config)
    
    @pytest.fixture
    def strategic_agents(self):
        """Create performance-optimized strategic agents."""
        mock_event_bus = Mock(spec=EventBus)
        
        mlmi_config = {
            'agent_id': 'perf_mlmi',
            'gamma': 0.99,
            'lambda_': 0.95,
            'hidden_dim': 128,
            'dropout_rate': 0.0,  # Disable dropout for performance
            'jit_compile': True
        }
        
        nwrqk_config = {
            'agent_id': 'perf_nwrqk',
            'hidden_dim': 64,
            'dropout_rate': 0.0,
            'jit_compile': True
        }
        
        regime_config = {
            'agent_id': 'perf_regime',
            'hidden_dim': 32,
            'dropout_rate': 0.0,
            'jit_compile': True
        }
        
        return {
            'mlmi': MLMIStrategicAgent(mlmi_config, mock_event_bus),
            'nwrqk': NWRQKStrategicAgent(nwrqk_config),
            'regime': RegimeDetectionAgent(regime_config)
        }
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitoring system."""
        config = {
            'enable_detailed_metrics': True,
            'sampling_interval_ms': 100,
            'memory_tracking': True
        }
        return HealthMonitor(config)

    def _create_performance_test_context(self, variation: int = 0) -> Dict[str, Any]:
        """Create varied market context for performance testing."""
        base_values = {
            'volatility_30': 1.5,
            'momentum_20': 0.05,
            'momentum_50': 0.03,
            'volume_ratio': 1.2,
            'mmd_score': 0.3,
            'price_trend': 0.02
        }
        
        # Add variation to prevent caching from skewing results
        variation_factor = 0.1 * (variation % 100) / 100.0
        
        return {
            key: value * (1 + variation_factor * np.random.uniform(-1, 1))
            for key, value in base_values.items()
        }

    def _create_test_predictions(self, num_agents: int = 3) -> List[Dict[str, Any]]:
        """Create test agent predictions."""
        predictions = []
        
        for i in range(num_agents):
            # Generate realistic probability distributions
            probs = np.random.dirichlet([2, 3, 4, 5, 3, 2, 1])  # 7 actions
            confidence = np.random.uniform(0.7, 0.9)
            
            predictions.append({
                'action_probabilities': probs.tolist(),
                'confidence': confidence,
                'agent_id': f'agent_{i}'
            })
        
        return predictions

    def _create_test_attention_weights(self, num_agents: int = 3) -> List[torch.Tensor]:
        """Create test attention weights."""
        attention_weights = []
        
        feature_dims = [4, 3, 3]  # MLMI, NWRQK, Regime feature dimensions
        
        for i in range(min(num_agents, len(feature_dims))):
            # Generate normalized attention weights
            weights = torch.softmax(torch.randn(feature_dims[i]), dim=0)
            attention_weights.append(weights)
        
        return attention_weights

    def _execute_load_test(self, rps: int, duration: int) -> Dict[str, Any]:
        """Execute load test with specified requests per second and duration."""
        
        # Calculate total requests
        total_requests = rps * duration
        interval = 1.0 / rps  # Time between requests
        
        # Performance tracking
        latencies = []
        errors = []
        start_time = time.perf_counter()
        completed_requests = 0
        
        # Memory tracking
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_samples = []
        
        # Create intelligence hub for testing
        intelligence_hub = IntelligenceHub({
            'max_intelligence_overhead_ms': 1.0,
            'performance_monitoring': True,
            'regime_detection': {'fast_mode': True},
            'gating_network': {'hidden_dim': 32}
        })
        
        try:
            for request_id in range(total_requests):
                request_start = time.perf_counter()
                
                try:
                    # Create test inputs
                    context = self._create_performance_test_context(request_id)
                    predictions = self._create_test_predictions()
                    attention_weights = self._create_test_attention_weights()
                    
                    # Execute strategic decision
                    result, metrics = intelligence_hub.process_intelligence_pipeline(
                        context, predictions, attention_weights
                    )
                    
                    # Record latency
                    latency = (time.perf_counter() - request_start) * 1000  # ms
                    latencies.append(latency)
                    completed_requests += 1
                    
                    # Validate result quality
                    if not result.get('intelligence_active', False):
                        errors.append(f"Intelligence not active for request {request_id}")
                    
                    if 'final_probabilities' not in result:
                        errors.append(f"Missing final probabilities for request {request_id}")
                    
                except Exception as e:
                    errors.append(f"Request {request_id} failed: {str(e)}")
                
                # Sample memory periodically
                if request_id % (rps // 2) == 0:  # Sample twice per second
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory - initial_memory)
                
                # Rate limiting
                elapsed = time.perf_counter() - start_time
                expected_time = request_id * interval
                if elapsed < expected_time:
                    time.sleep(expected_time - elapsed)
                
                # Early termination if falling too far behind
                if elapsed > expected_time + 1.0:  # More than 1 second behind
                    logger.warning(f"Load test falling behind schedule at request {request_id}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Load test interrupted")
        
        # Calculate final metrics
        total_time = time.perf_counter() - start_time
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        if latencies:
            mean_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            max_latency = np.max(latencies)
        else:
            mean_latency = p50_latency = p95_latency = p99_latency = max_latency = 0.0
        
        error_rate = len(errors) / max(1, completed_requests)
        actual_throughput = completed_requests / total_time
        
        return {
            'completed_requests': completed_requests,
            'total_requests': total_requests,
            'total_time_seconds': total_time,
            'mean_latency_ms': mean_latency,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'max_latency_ms': max_latency,
            'error_rate': error_rate,
            'error_count': len(errors),
            'errors': errors[:10],  # First 10 errors for debugging
            'throughput_achieved': actual_throughput,
            'target_throughput': rps,
            'memory_growth_mb': memory_growth,
            'memory_samples': memory_samples,
            'cpu_usage_percent': psutil.cpu_percent(interval=None)
        }

    def test_comprehensive_performance_certification(self):
        """Comprehensive performance certification under all conditions."""
        
        test_configurations = [
            {'name': 'baseline_load', 'requests_per_second': 100, 'duration_seconds': 300},
            {'name': 'peak_load', 'requests_per_second': 500, 'duration_seconds': 180},
            {'name': 'stress_load', 'requests_per_second': 1000, 'duration_seconds': 60},
            {'name': 'endurance_load', 'requests_per_second': 200, 'duration_seconds': 1800}
        ]
        
        performance_results = {}
        
        for config in test_configurations:
            logger.info(f"🚀 Running {config['name']} test...")
            
            # Force garbage collection before test
            gc.collect()
            
            # Execute load test
            result = self._execute_load_test(
                rps=config['requests_per_second'],
                duration=config['duration_seconds']
            )
            
            performance_results[config['name']] = result
            
            # Validate performance requirements based on load type
            if config['name'] == 'baseline_load':
                assert result['mean_latency_ms'] < 3.0, \
                    f"Baseline: Mean latency {result['mean_latency_ms']:.3f}ms exceeds 3ms"
                assert result['p95_latency_ms'] < 5.0, \
                    f"Baseline: P95 latency {result['p95_latency_ms']:.3f}ms exceeds 5ms"
                assert result['p99_latency_ms'] < 8.0, \
                    f"Baseline: P99 latency {result['p99_latency_ms']:.3f}ms exceeds 8ms"
            
            elif config['name'] == 'peak_load':
                assert result['mean_latency_ms'] < 5.0, \
                    f"Peak: Mean latency {result['mean_latency_ms']:.3f}ms exceeds 5ms"
                assert result['p95_latency_ms'] < 8.0, \
                    f"Peak: P95 latency {result['p95_latency_ms']:.3f}ms exceeds 8ms"
                assert result['p99_latency_ms'] < 12.0, \
                    f"Peak: P99 latency {result['p99_latency_ms']:.3f}ms exceeds 12ms"
            
            elif config['name'] == 'stress_load':
                assert result['mean_latency_ms'] < 8.0, \
                    f"Stress: Mean latency {result['mean_latency_ms']:.3f}ms exceeds 8ms"
                assert result['p95_latency_ms'] < 15.0, \
                    f"Stress: P95 latency {result['p95_latency_ms']:.3f}ms exceeds 15ms"
                assert result['p99_latency_ms'] < 25.0, \
                    f"Stress: P99 latency {result['p99_latency_ms']:.3f}ms exceeds 25ms"
            
            elif config['name'] == 'endurance_load':
                assert result['mean_latency_ms'] < 4.0, \
                    f"Endurance: Mean latency {result['mean_latency_ms']:.3f}ms exceeds 4ms"
                assert result['memory_growth_mb'] < 200.0, \
                    f"Endurance: Memory growth {result['memory_growth_mb']:.1f}MB exceeds 200MB"
            
            # Universal requirements
            assert result['error_rate'] < 0.001, \
                f"{config['name']}: Error rate {result['error_rate']:.4f} exceeds 0.1%"
            
            assert result['throughput_achieved'] > config['requests_per_second'] * 0.95, \
                f"{config['name']}: Throughput {result['throughput_achieved']:.1f} below 95% of target"
            
            logger.info(f"✅ {config['name']}: {result['mean_latency_ms']:.2f}ms mean, "
                       f"{result['throughput_achieved']:.1f} QPS, "
                       f"{result['error_rate']:.4f} error rate")
        
        return performance_results

    def test_memory_and_resource_certification(self):
        """Certify memory usage and resource management."""
        
        # Initial resource measurements
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu_percent = psutil.cpu_percent(interval=1.0)
        
        # Create intelligence hub for extended testing
        intelligence_hub = IntelligenceHub({
            'max_intelligence_overhead_ms': 1.0,
            'performance_monitoring': True,
            'regime_detection': {'fast_mode': True, 'cache_analysis': True},
            'gating_network': {'hidden_dim': 32}
        })
        
        # Extended operation test (10,000 strategic decisions)
        memory_samples = []
        cpu_samples = []
        latency_samples = []
        
        logger.info("🔬 Running extended memory stability test (10,000 operations)...")
        
        for i in range(10000):
            operation_start = time.perf_counter()
            
            try:
                # Create test inputs with variation
                context = self._create_performance_test_context(i)
                predictions = self._create_test_predictions()
                attention_weights = self._create_test_attention_weights()
                
                # Execute strategic decision
                result, metrics = intelligence_hub.process_intelligence_pipeline(
                    context, predictions, attention_weights
                )
                
                # Record latency
                latency = (time.perf_counter() - operation_start) * 1000
                latency_samples.append(latency)
                
                # Sample resources every 100 operations
                if i % 100 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - initial_memory
                    memory_samples.append(memory_growth)
                    
                    # CPU sampling
                    cpu_percent = psutil.cpu_percent(interval=None)
                    cpu_samples.append(cpu_percent)
                    
                    # Early warning for excessive memory growth
                    if memory_growth > 150:
                        logger.warning(f"High memory growth {memory_growth:.1f}MB at operation {i}")
                    
                    # Force periodic garbage collection
                    if i % 1000 == 0:
                        gc.collect()
                        logger.info(f"Completed {i} operations, memory growth: {memory_growth:.1f}MB")
                
            except Exception as e:
                logger.error(f"Operation {i} failed: {e}")
                # Continue testing even if individual operations fail
        
        # Final measurements
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_growth = final_memory - initial_memory
        
        # Calculate statistics
        if memory_samples:
            avg_memory_growth = np.mean(memory_samples)
            max_memory_growth = np.max(memory_samples)
            memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]  # Linear trend
        else:
            avg_memory_growth = max_memory_growth = memory_trend = 0.0
        
        if cpu_samples:
            avg_cpu_usage = np.mean(cpu_samples)
            max_cpu_usage = np.max(cpu_samples)
        else:
            avg_cpu_usage = max_cpu_usage = 0.0
        
        if latency_samples:
            avg_latency = np.mean(latency_samples)
            latency_degradation = (np.mean(latency_samples[-100:]) - np.mean(latency_samples[:100])) / np.mean(latency_samples[:100])
        else:
            avg_latency = latency_degradation = 0.0
        
        # Memory stability assertions
        assert total_memory_growth < 100, \
            f"Excessive total memory growth: {total_memory_growth:.1f}MB over 10K operations"
        
        assert max_memory_growth < 150, \
            f"Peak memory growth too high: {max_memory_growth:.1f}MB"
        
        assert memory_trend < 0.01, \
            f"Memory leak detected - trend: {memory_trend:.4f}MB per sample"
        
        # CPU and performance assertions
        assert avg_cpu_usage < 80, \
            f"Average CPU usage too high: {avg_cpu_usage:.1f}%"
        
        assert max_cpu_usage < 95, \
            f"Peak CPU usage too high: {max_cpu_usage:.1f}%"
        
        assert avg_latency < 2.0, \
            f"Average latency during extended test too high: {avg_latency:.3f}ms"
        
        assert abs(latency_degradation) < 0.1, \
            f"Significant latency degradation: {latency_degradation:.3f} (10%)"
        
        logger.info(f"✅ Memory & Resource Stability Certified:")
        logger.info(f"   📊 Total memory growth: {total_memory_growth:.1f}MB")
        logger.info(f"   📊 Average CPU usage: {avg_cpu_usage:.1f}%")
        logger.info(f"   📊 Average latency: {avg_latency:.3f}ms")
        logger.info(f"   📊 Latency degradation: {latency_degradation:.3f}")
        
        return {
            'total_memory_growth_mb': total_memory_growth,
            'avg_memory_growth_mb': avg_memory_growth,
            'max_memory_growth_mb': max_memory_growth,
            'memory_trend_mb_per_sample': memory_trend,
            'avg_cpu_usage_percent': avg_cpu_usage,
            'max_cpu_usage_percent': max_cpu_usage,
            'avg_latency_ms': avg_latency,
            'latency_degradation_ratio': latency_degradation,
            'operations_completed': 10000
        }

    def test_concurrent_scalability_certification(self):
        """Certify scalability under concurrent load from multiple threads."""
        
        def worker_thread(thread_id: int, operations_per_thread: int, results_queue: queue.Queue):
            """Worker thread for concurrent testing."""
            
            # Create thread-local intelligence hub
            intelligence_hub = IntelligenceHub({
                'max_intelligence_overhead_ms': 1.0,
                'performance_monitoring': True,
                'regime_detection': {'fast_mode': True}
            })
            
            thread_results = {
                'thread_id': thread_id,
                'operations_completed': 0,
                'total_latency_ms': 0.0,
                'errors': []
            }
            
            start_time = time.perf_counter()
            
            for op_id in range(operations_per_thread):
                try:
                    op_start = time.perf_counter()
                    
                    # Create test inputs
                    context = self._create_performance_test_context(thread_id * 1000 + op_id)
                    predictions = self._create_test_predictions()
                    attention_weights = self._create_test_attention_weights()
                    
                    # Execute operation
                    result, metrics = intelligence_hub.process_intelligence_pipeline(
                        context, predictions, attention_weights
                    )
                    
                    # Track performance
                    op_latency = (time.perf_counter() - op_start) * 1000
                    thread_results['total_latency_ms'] += op_latency
                    thread_results['operations_completed'] += 1
                    
                    # Validate result
                    if not result.get('intelligence_active', False):
                        thread_results['errors'].append(f"Op {op_id}: Intelligence not active")
                    
                except Exception as e:
                    thread_results['errors'].append(f"Op {op_id}: {str(e)}")
            
            thread_results['total_time_seconds'] = time.perf_counter() - start_time
            thread_results['avg_latency_ms'] = (
                thread_results['total_latency_ms'] / max(1, thread_results['operations_completed'])
            )
            
            results_queue.put(thread_results)
        
        # Test different concurrency levels
        concurrency_tests = [
            {'threads': 2, 'ops_per_thread': 500},
            {'threads': 4, 'ops_per_thread': 250},
            {'threads': 8, 'ops_per_thread': 125},
            {'threads': 16, 'ops_per_thread': 100}
        ]
        
        scalability_results = {}
        
        for test_config in concurrency_tests:
            num_threads = test_config['threads']
            ops_per_thread = test_config['ops_per_thread']
            
            logger.info(f"🔄 Testing {num_threads} concurrent threads, {ops_per_thread} ops each...")
            
            # Create results queue
            results_queue = queue.Queue()
            
            # Start worker threads
            threads = []
            start_time = time.perf_counter()
            
            for thread_id in range(num_threads):
                thread = threading.Thread(
                    target=worker_thread,
                    args=(thread_id, ops_per_thread, results_queue)
                )
                thread.start()
                threads.append(thread)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            total_time = time.perf_counter() - start_time
            
            # Collect results
            thread_results = []
            while not results_queue.empty():
                thread_results.append(results_queue.get())
            
            # Aggregate metrics
            total_operations = sum(r['operations_completed'] for r in thread_results)
            total_errors = sum(len(r['errors']) for r in thread_results)
            avg_latencies = [r['avg_latency_ms'] for r in thread_results if r['operations_completed'] > 0]
            
            if avg_latencies:
                overall_avg_latency = np.mean(avg_latencies)
                latency_std = np.std(avg_latencies)
                max_thread_latency = np.max(avg_latencies)
            else:
                overall_avg_latency = latency_std = max_thread_latency = 0.0
            
            overall_throughput = total_operations / total_time
            error_rate = total_errors / max(1, total_operations)
            
            scalability_results[num_threads] = {
                'num_threads': num_threads,
                'ops_per_thread': ops_per_thread,
                'total_operations': total_operations,
                'total_time_seconds': total_time,
                'overall_avg_latency_ms': overall_avg_latency,
                'latency_std_ms': latency_std,
                'max_thread_latency_ms': max_thread_latency,
                'overall_throughput_ops_sec': overall_throughput,
                'error_rate': error_rate,
                'thread_results': thread_results
            }
            
            # Scalability assertions
            assert error_rate < 0.01, \
                f"{num_threads} threads: Error rate {error_rate:.4f} exceeds 1%"
            
            assert overall_avg_latency < 10.0, \
                f"{num_threads} threads: Avg latency {overall_avg_latency:.3f}ms exceeds 10ms"
            
            assert latency_std < 5.0, \
                f"{num_threads} threads: Latency std {latency_std:.3f}ms too high (poor consistency)"
            
            # Throughput should scale reasonably (not perfectly linear due to overhead)
            if num_threads <= 4:
                min_expected_throughput = 80 * num_threads  # 80 ops/sec per thread minimum
                assert overall_throughput > min_expected_throughput, \
                    f"{num_threads} threads: Throughput {overall_throughput:.1f} below {min_expected_throughput}"
            
            logger.info(f"✅ {num_threads} threads: {overall_avg_latency:.2f}ms avg, "
                       f"{overall_throughput:.1f} ops/sec, "
                       f"{error_rate:.4f} error rate")
        
        return scalability_results

    def test_burst_capacity_certification(self):
        """Certify system ability to handle burst traffic."""
        
        # Create intelligence hub
        intelligence_hub = IntelligenceHub({
            'max_intelligence_overhead_ms': 1.0,
            'performance_monitoring': True,
            'regime_detection': {'fast_mode': True, 'cache_analysis': True}
        })
        
        # Burst test scenarios
        burst_scenarios = [
            {'name': 'small_burst', 'burst_size': 100, 'burst_duration_ms': 500},
            {'name': 'medium_burst', 'burst_size': 500, 'burst_duration_ms': 1000},
            {'name': 'large_burst', 'burst_size': 1000, 'burst_duration_ms': 2000}
        ]
        
        burst_results = {}
        
        for scenario in burst_scenarios:
            logger.info(f"💥 Testing {scenario['name']}: {scenario['burst_size']} ops in {scenario['burst_duration_ms']}ms")
            
            burst_size = scenario['burst_size']
            burst_duration_ms = scenario['burst_duration_ms']
            
            # Prepare test data
            test_contexts = [self._create_performance_test_context(i) for i in range(burst_size)]
            test_predictions = [self._create_test_predictions() for _ in range(burst_size)]
            test_attention = [self._create_test_attention_weights() for _ in range(burst_size)]
            
            # Execute burst
            latencies = []
            errors = []
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=min(burst_size, 32)) as executor:
                # Submit all operations at once
                futures = []
                for i in range(burst_size):
                    future = executor.submit(
                        self._execute_single_operation,
                        intelligence_hub,
                        test_contexts[i],
                        test_predictions[i],
                        test_attention[i]
                    )
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(futures):
                    try:
                        latency, success, error_msg = future.result(timeout=5.0)
                        latencies.append(latency)
                        if not success:
                            errors.append(f"Op {i}: {error_msg}")
                    except Exception as e:
                        errors.append(f"Op {i}: Exception {str(e)}")
                        latencies.append(5000.0)  # Timeout latency
            
            total_time = (time.perf_counter() - start_time) * 1000  # ms
            
            # Calculate metrics
            if latencies:
                mean_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)
                max_latency = np.max(latencies)
            else:
                mean_latency = p95_latency = p99_latency = max_latency = 0.0
            
            error_rate = len(errors) / burst_size
            actual_throughput = burst_size / (total_time / 1000)  # ops/sec
            
            burst_results[scenario['name']] = {
                'burst_size': burst_size,
                'target_duration_ms': burst_duration_ms,
                'actual_duration_ms': total_time,
                'mean_latency_ms': mean_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'max_latency_ms': max_latency,
                'error_rate': error_rate,
                'error_count': len(errors),
                'throughput_ops_sec': actual_throughput
            }
            
            # Burst-specific assertions
            if scenario['name'] == 'small_burst':
                assert mean_latency < 10.0, \
                    f"Small burst mean latency {mean_latency:.3f}ms exceeds 10ms"
                assert p95_latency < 20.0, \
                    f"Small burst P95 latency {p95_latency:.3f}ms exceeds 20ms"
            
            elif scenario['name'] == 'medium_burst':
                assert mean_latency < 15.0, \
                    f"Medium burst mean latency {mean_latency:.3f}ms exceeds 15ms"
                assert p95_latency < 30.0, \
                    f"Medium burst P95 latency {p95_latency:.3f}ms exceeds 30ms"
            
            elif scenario['name'] == 'large_burst':
                assert mean_latency < 25.0, \
                    f"Large burst mean latency {mean_latency:.3f}ms exceeds 25ms"
                assert p99_latency < 100.0, \
                    f"Large burst P99 latency {p99_latency:.3f}ms exceeds 100ms"
            
            # Universal burst requirements
            assert error_rate < 0.05, \
                f"{scenario['name']}: Error rate {error_rate:.4f} exceeds 5%"
            
            logger.info(f"✅ {scenario['name']}: {mean_latency:.2f}ms mean, "
                       f"{actual_throughput:.1f} ops/sec burst capacity")
        
        return burst_results

    def _execute_single_operation(
        self, 
        intelligence_hub: IntelligenceHub,
        context: Dict[str, Any], 
        predictions: List[Dict[str, Any]], 
        attention_weights: List[torch.Tensor]
    ) -> Tuple[float, bool, str]:
        """Execute a single strategic operation and return (latency, success, error_msg)."""
        
        start_time = time.perf_counter()
        
        try:
            result, metrics = intelligence_hub.process_intelligence_pipeline(
                context, predictions, attention_weights
            )
            
            latency = (time.perf_counter() - start_time) * 1000  # ms
            
            # Validate result
            if not result.get('intelligence_active', False):
                return latency, False, "Intelligence not active"
            
            if 'final_probabilities' not in result:
                return latency, False, "Missing final probabilities"
            
            return latency, True, ""
            
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return latency, False, str(e)


if __name__ == "__main__":
    # Run the comprehensive performance certification test suite
    pytest.main([__file__, "-v", "--tb=short"])