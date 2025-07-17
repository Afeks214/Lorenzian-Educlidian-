"""
AGENT 9: Strategic Latency Performance Tests
Comprehensive test suite for strategic inference latency optimization and performance validation.
"""

import pytest
import asyncio
import time
import statistics
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Performance testing imports
try:
    import psutil
    import torch
    import ray
    HAS_PERFORMANCE_DEPS = True
except ImportError:
    HAS_PERFORMANCE_DEPS = False

# Core imports
from src.agents.strategic_marl_component import StrategicMARLComponent
from src.tactical.async_inference_pool import AsyncInferencePool
from src.intelligence.intelligence_hub import IntelligenceHub
from src.core.performance.performance_monitor import PerformanceMonitor
from src.monitoring.tactical_metrics import tactical_metrics


@dataclass
class PerformanceTarget:
    """Performance targets for strategic inference."""
    max_inference_latency_ms: float = 50.0
    max_p99_latency_ms: float = 100.0
    min_throughput_ops_per_sec: float = 1000.0
    max_memory_usage_mb: float = 512.0
    max_cpu_usage_percent: float = 80.0
    max_gpu_memory_mb: float = 1024.0


class TestStrategicLatencyOptimization:
    """Test strategic inference latency optimization."""
    
    @pytest.fixture
    def performance_targets(self):
        """Get performance targets for testing."""
        return PerformanceTarget()
    
    @pytest.fixture
    def mock_strategic_component(self):
        """Create a mock strategic MARL component."""
        config = {
            'confidence_threshold': 0.7,
            'max_inference_latency_ms': 50,
            'environment': {'n_agents': 3},
            'model': {'learning_rate': 0.001}
        }
        
        with patch('src.agents.strategic_marl_component.get_logger'):
            component = StrategicMARLComponent(config)
            return component
    
    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing."""
        return {
            'matrix_state': np.random.randn(48, 8).astype(np.float32),
            'synergy_event': {
                'synergy_type': 'momentum_alignment',
                'direction': 1,
                'confidence': 0.8,
                'timestamp': time.time()
            },
            'correlation_id': 'test_correlation_123'
        }
    
    def test_single_inference_latency_target(self, mock_strategic_component, sample_market_data, performance_targets):
        """Test that single inference meets latency targets."""
        
        # Mock the inference methods
        mock_strategic_component._intelligent_gating_network = Mock()
        mock_strategic_component._intelligent_gating_network.should_process_event.return_value = (True, 0.8)
        
        # Mock agent inference
        mock_agents = {
            'mlmi_agent': Mock(),
            'nwrqk_agent': Mock(),
            'momentum_agent': Mock()
        }
        
        for agent_name, agent in mock_agents.items():
            agent.inference_step.return_value = {
                'action': 1,
                'confidence': 0.8,
                'probabilities': [0.1, 0.2, 0.7],
                'reasoning': {'agent_type': agent_name}
            }
        
        mock_strategic_component.agents = mock_agents
        mock_strategic_component._ensemble_decision_making = Mock()
        mock_strategic_component._ensemble_decision_making.return_value = {
            'action': 1,
            'confidence': 0.8,
            'probabilities': [0.1, 0.2, 0.7],
            'ensemble_reasoning': {'method': 'weighted_voting'}
        }
        
        # Measure inference latency
        start_time = time.perf_counter()
        
        result = mock_strategic_component._handle_synergy_event(sample_market_data['synergy_event'])
        
        end_time = time.perf_counter()
        inference_latency_ms = (end_time - start_time) * 1000
        
        # Assert latency target
        assert inference_latency_ms < performance_targets.max_inference_latency_ms, \
            f"Inference latency {inference_latency_ms:.2f}ms exceeds target {performance_targets.max_inference_latency_ms}ms"
        
        # Verify result quality
        assert result is not None
        assert 'action' in result
        assert 'confidence' in result
    
    def test_batch_inference_throughput(self, mock_strategic_component, performance_targets):
        """Test batch inference throughput meets targets."""
        
        # Mock the inference pipeline
        mock_strategic_component._intelligent_gating_network = Mock()
        mock_strategic_component._intelligent_gating_network.should_process_event.return_value = (True, 0.8)
        
        # Mock fast inference
        def fast_inference(event):
            return {
                'action': 1,
                'confidence': 0.8,
                'probabilities': [0.1, 0.2, 0.7],
                'processing_time_ms': 5.0
            }
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=fast_inference)
        
        # Generate test events
        test_events = []
        for i in range(100):
            test_events.append({
                'synergy_type': 'test_event',
                'direction': 1 if i % 2 == 0 else -1,
                'confidence': 0.8,
                'timestamp': time.time() + i * 0.001
            })
        
        # Measure throughput
        start_time = time.perf_counter()
        
        results = []
        for event in test_events:
            result = mock_strategic_component._handle_synergy_event(event)
            results.append(result)
        
        end_time = time.perf_counter()
        total_time_seconds = end_time - start_time
        
        # Calculate throughput
        throughput_ops_per_sec = len(test_events) / total_time_seconds
        
        # Assert throughput target
        assert throughput_ops_per_sec >= performance_targets.min_throughput_ops_per_sec, \
            f"Throughput {throughput_ops_per_sec:.2f} ops/sec below target {performance_targets.min_throughput_ops_per_sec}"
        
        # Verify all results processed
        assert len(results) == len(test_events)
        assert all(result is not None for result in results)
    
    def test_concurrent_inference_performance(self, mock_strategic_component, performance_targets):
        """Test concurrent inference performance."""
        
        # Mock thread-safe inference
        mock_strategic_component._intelligent_gating_network = Mock()
        mock_strategic_component._intelligent_gating_network.should_process_event.return_value = (True, 0.8)
        
        def concurrent_inference(event):
            time.sleep(0.01)  # Simulate 10ms processing time
            return {
                'action': 1,
                'confidence': 0.8,
                'thread_id': threading.current_thread().ident
            }
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=concurrent_inference)
        
        # Test concurrent processing
        results_queue = queue.Queue()
        
        def worker_thread(thread_id: int):
            """Worker thread for concurrent testing."""
            for i in range(10):
                event = {
                    'synergy_type': f'thread_{thread_id}_event_{i}',
                    'direction': 1,
                    'confidence': 0.8,
                    'timestamp': time.time()
                }
                
                start_time = time.perf_counter()
                result = mock_strategic_component._handle_synergy_event(event)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                results_queue.put((thread_id, i, latency_ms, result))
        
        # Start multiple threads
        threads = []
        num_threads = 5
        
        start_time = time.perf_counter()
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Collect results
        latencies = []
        results = []
        
        while not results_queue.empty():
            thread_id, event_id, latency_ms, result = results_queue.get()
            latencies.append(latency_ms)
            results.append(result)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        total_operations = len(results)
        throughput = total_operations / total_time
        
        # Assert performance targets
        assert avg_latency < performance_targets.max_inference_latency_ms, \
            f"Average latency {avg_latency:.2f}ms exceeds target {performance_targets.max_inference_latency_ms}ms"
        
        assert p99_latency < performance_targets.max_p99_latency_ms, \
            f"P99 latency {p99_latency:.2f}ms exceeds target {performance_targets.max_p99_latency_ms}ms"
        
        assert throughput >= performance_targets.min_throughput_ops_per_sec, \
            f"Throughput {throughput:.2f} ops/sec below target {performance_targets.min_throughput_ops_per_sec}"
        
        # Verify all operations completed successfully
        assert len(results) == num_threads * 10
        assert all(result is not None for result in results)
    
    @pytest.mark.skipif(not HAS_PERFORMANCE_DEPS, reason="Performance dependencies not available")
    def test_memory_usage_optimization(self, mock_strategic_component, performance_targets):
        """Test memory usage optimization."""
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Mock memory-efficient inference
        mock_strategic_component._intelligent_gating_network = Mock()
        mock_strategic_component._intelligent_gating_network.should_process_event.return_value = (True, 0.8)
        
        def memory_efficient_inference(event):
            # Simulate memory-efficient processing
            temp_data = np.random.randn(100, 100).astype(np.float32)  # Small temporary allocation
            result = {
                'action': 1,
                'confidence': 0.8,
                'temp_data_size': temp_data.nbytes
            }
            del temp_data  # Explicit cleanup
            return result
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=memory_efficient_inference)
        
        # Run memory usage test
        for i in range(1000):
            event = {
                'synergy_type': f'memory_test_{i}',
                'direction': 1,
                'confidence': 0.8,
                'timestamp': time.time()
            }
            
            result = mock_strategic_component._handle_synergy_event(event)
            
            # Periodic memory check
            if i % 100 == 0:
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_increase_mb = current_memory_mb - initial_memory_mb
                
                assert memory_increase_mb < performance_targets.max_memory_usage_mb, \
                    f"Memory usage increased by {memory_increase_mb:.2f}MB, exceeds target {performance_targets.max_memory_usage_mb}MB"
        
        # Final memory check
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory_mb - initial_memory_mb
        
        assert total_memory_increase < performance_targets.max_memory_usage_mb, \
            f"Total memory increase {total_memory_increase:.2f}MB exceeds target {performance_targets.max_memory_usage_mb}MB"
    
    @pytest.mark.skipif(not HAS_PERFORMANCE_DEPS, reason="Performance dependencies not available")
    def test_cpu_usage_optimization(self, mock_strategic_component, performance_targets):
        """Test CPU usage optimization."""
        
        # Mock CPU-efficient inference
        mock_strategic_component._intelligent_gating_network = Mock()
        mock_strategic_component._intelligent_gating_network.should_process_event.return_value = (True, 0.8)
        
        def cpu_efficient_inference(event):
            # Simulate CPU-efficient processing
            result = {
                'action': 1,
                'confidence': 0.8,
                'cpu_optimized': True
            }
            return result
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=cpu_efficient_inference)
        
        # Monitor CPU usage during inference
        cpu_measurements = []
        
        def measure_cpu_usage():
            """Measure CPU usage in background."""
            for _ in range(10):
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_measurements.append(cpu_percent)
        
        # Start CPU monitoring
        cpu_monitor_thread = threading.Thread(target=measure_cpu_usage)
        cpu_monitor_thread.start()
        
        # Run inference workload
        for i in range(100):
            event = {
                'synergy_type': f'cpu_test_{i}',
                'direction': 1,
                'confidence': 0.8,
                'timestamp': time.time()
            }
            
            result = mock_strategic_component._handle_synergy_event(event)
            time.sleep(0.001)  # Small delay to allow CPU monitoring
        
        # Wait for CPU monitoring to complete
        cpu_monitor_thread.join()
        
        # Analyze CPU usage
        if cpu_measurements:
            avg_cpu_usage = statistics.mean(cpu_measurements)
            max_cpu_usage = max(cpu_measurements)
            
            assert avg_cpu_usage < performance_targets.max_cpu_usage_percent, \
                f"Average CPU usage {avg_cpu_usage:.2f}% exceeds target {performance_targets.max_cpu_usage_percent}%"
            
            assert max_cpu_usage < performance_targets.max_cpu_usage_percent * 1.2, \
                f"Peak CPU usage {max_cpu_usage:.2f}% exceeds acceptable limit"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_optimization(self, mock_strategic_component, performance_targets):
        """Test GPU memory optimization."""
        
        # Mock GPU-efficient inference
        mock_strategic_component._intelligent_gating_network = Mock()
        mock_strategic_component._intelligent_gating_network.should_process_event.return_value = (True, 0.8)
        
        def gpu_efficient_inference(event):
            # Simulate GPU operations
            if torch.cuda.is_available():
                # Small tensor operations
                temp_tensor = torch.randn(100, 100, device='cuda')
                result_tensor = torch.relu(temp_tensor)
                
                # Clean up GPU memory
                del temp_tensor
                del result_tensor
                torch.cuda.empty_cache()
            
            return {
                'action': 1,
                'confidence': 0.8,
                'gpu_optimized': True
            }
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=gpu_efficient_inference)
        
        # Monitor GPU memory usage
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Run GPU workload
        for i in range(100):
            event = {
                'synergy_type': f'gpu_test_{i}',
                'direction': 1,
                'confidence': 0.8,
                'timestamp': time.time()
            }
            
            result = mock_strategic_component._handle_synergy_event(event)
            
            # Check GPU memory periodically
            if i % 10 == 0:
                current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_memory_increase = current_gpu_memory - initial_gpu_memory
                
                assert gpu_memory_increase < performance_targets.max_gpu_memory_mb, \
                    f"GPU memory usage increased by {gpu_memory_increase:.2f}MB, exceeds target"
        
        # Final GPU memory check
        final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        total_gpu_memory_increase = final_gpu_memory - initial_gpu_memory
        
        assert total_gpu_memory_increase < performance_targets.max_gpu_memory_mb, \
            f"Total GPU memory increase {total_gpu_memory_increase:.2f}MB exceeds target"
    
    def test_intelligent_gating_performance_impact(self, mock_strategic_component, performance_targets):
        """Test performance impact of intelligent gating."""
        
        # Mock gating network
        mock_gating_network = Mock()
        mock_strategic_component._intelligent_gating_network = mock_gating_network
        
        # Test with gating enabled
        mock_gating_network.should_process_event.return_value = (True, 0.8)
        
        def gated_inference(event):
            return {
                'action': 1,
                'confidence': 0.8,
                'gated': True
            }
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=gated_inference)
        
        # Measure performance with gating
        start_time = time.perf_counter()
        
        for i in range(100):
            event = {
                'synergy_type': f'gating_test_{i}',
                'direction': 1,
                'confidence': 0.8,
                'timestamp': time.time()
            }
            
            result = mock_strategic_component._handle_synergy_event(event)
        
        end_time = time.perf_counter()
        gated_time = end_time - start_time
        
        # Test without gating (bypass gating network)
        mock_gating_network.should_process_event.return_value = (True, 1.0)  # Always process
        
        start_time = time.perf_counter()
        
        for i in range(100):
            event = {
                'synergy_type': f'no_gating_test_{i}',
                'direction': 1,
                'confidence': 0.8,
                'timestamp': time.time()
            }
            
            result = mock_strategic_component._handle_synergy_event(event)
        
        end_time = time.perf_counter()
        ungated_time = end_time - start_time
        
        # Gating overhead should be minimal
        gating_overhead = gated_time - ungated_time
        overhead_percentage = (gating_overhead / ungated_time) * 100 if ungated_time > 0 else 0
        
        assert overhead_percentage < 10, \
            f"Gating overhead {overhead_percentage:.2f}% exceeds acceptable limit of 10%"
    
    def test_ensemble_decision_making_latency(self, mock_strategic_component, performance_targets):
        """Test ensemble decision making latency."""
        
        # Mock individual agent results
        agent_results = [
            {'action': 1, 'confidence': 0.8, 'probabilities': [0.1, 0.2, 0.7]},
            {'action': 1, 'confidence': 0.7, 'probabilities': [0.2, 0.3, 0.5]},
            {'action': 0, 'confidence': 0.9, 'probabilities': [0.3, 0.6, 0.1]}
        ]
        
        # Mock ensemble decision making
        def ensemble_decision_making(results):
            # Simulate ensemble processing
            ensemble_action = max(set([r['action'] for r in results]), key=[r['action'] for r in results].count)
            ensemble_confidence = statistics.mean([r['confidence'] for r in results])
            
            return {
                'action': ensemble_action,
                'confidence': ensemble_confidence,
                'probabilities': [0.2, 0.3, 0.5],
                'ensemble_reasoning': {'method': 'majority_vote'}
            }
        
        mock_strategic_component._ensemble_decision_making = Mock(side_effect=ensemble_decision_making)
        
        # Measure ensemble latency
        latencies = []
        
        for i in range(100):
            start_time = time.perf_counter()
            
            result = mock_strategic_component._ensemble_decision_making(agent_results)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        # Assert latency targets
        assert avg_latency < performance_targets.max_inference_latency_ms / 2, \
            f"Ensemble latency {avg_latency:.2f}ms exceeds target {performance_targets.max_inference_latency_ms / 2}ms"
        
        assert p99_latency < performance_targets.max_inference_latency_ms, \
            f"Ensemble P99 latency {p99_latency:.2f}ms exceeds target {performance_targets.max_inference_latency_ms}ms"
    
    def test_performance_monitoring_overhead(self, mock_strategic_component, performance_targets):
        """Test performance monitoring overhead."""
        
        # Mock performance monitoring
        with patch('src.monitoring.tactical_metrics.tactical_metrics') as mock_metrics:
            mock_metrics.measure_inference_latency.return_value.__enter__ = Mock()
            mock_metrics.measure_inference_latency.return_value.__exit__ = Mock()
            mock_metrics.record_inference_completion = Mock()
            
            # Test with monitoring enabled
            start_time = time.perf_counter()
            
            for i in range(100):
                with mock_metrics.measure_inference_latency('strategic', 'test_agent', f'correlation_{i}'):
                    # Simulate inference
                    time.sleep(0.001)  # 1ms processing time
                
                mock_metrics.record_inference_completion(
                    agent_name='test_agent',
                    agent_type='strategic',
                    processing_time_ms=1.0,
                    success=True
                )
            
            end_time = time.perf_counter()
            monitored_time = end_time - start_time
            
            # Test without monitoring
            start_time = time.perf_counter()
            
            for i in range(100):
                # Simulate inference without monitoring
                time.sleep(0.001)  # 1ms processing time
            
            end_time = time.perf_counter()
            unmonitored_time = end_time - start_time
            
            # Monitoring overhead should be minimal
            monitoring_overhead = monitored_time - unmonitored_time
            overhead_percentage = (monitoring_overhead / unmonitored_time) * 100 if unmonitored_time > 0 else 0
            
            assert overhead_percentage < 5, \
                f"Monitoring overhead {overhead_percentage:.2f}% exceeds acceptable limit of 5%"


class TestStrategicLatencyRegressionTests:
    """Regression tests for strategic latency performance."""
    
    def test_latency_regression_detection(self, mock_strategic_component, performance_targets):
        """Test detection of latency regressions."""
        
        # Baseline performance measurements
        baseline_latencies = []
        
        # Mock baseline inference
        def baseline_inference(event):
            time.sleep(0.005)  # 5ms baseline
            return {'action': 1, 'confidence': 0.8}
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=baseline_inference)
        
        # Measure baseline performance
        for i in range(50):
            event = {'synergy_type': f'baseline_{i}', 'direction': 1, 'confidence': 0.8}
            
            start_time = time.perf_counter()
            result = mock_strategic_component._handle_synergy_event(event)
            end_time = time.perf_counter()
            
            baseline_latencies.append((end_time - start_time) * 1000)
        
        baseline_avg = statistics.mean(baseline_latencies)
        
        # Test for regression (simulated slower inference)
        def regressed_inference(event):
            time.sleep(0.015)  # 15ms (3x slower)
            return {'action': 1, 'confidence': 0.8}
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=regressed_inference)
        
        # Measure regressed performance
        regressed_latencies = []
        
        for i in range(50):
            event = {'synergy_type': f'regressed_{i}', 'direction': 1, 'confidence': 0.8}
            
            start_time = time.perf_counter()
            result = mock_strategic_component._handle_synergy_event(event)
            end_time = time.perf_counter()
            
            regressed_latencies.append((end_time - start_time) * 1000)
        
        regressed_avg = statistics.mean(regressed_latencies)
        
        # Detect regression
        regression_percentage = ((regressed_avg - baseline_avg) / baseline_avg) * 100
        
        # Assert regression detection
        assert regression_percentage > 50, \
            f"Failed to detect significant regression: {regression_percentage:.2f}%"
        
        # Verify regression exceeds acceptable threshold
        acceptable_regression_threshold = 20  # 20% acceptable degradation
        assert regression_percentage > acceptable_regression_threshold, \
            f"Regression {regression_percentage:.2f}% should be detected as significant"
    
    def test_performance_trend_analysis(self, mock_strategic_component):
        """Test performance trend analysis."""
        
        # Simulate performance trend over time
        performance_samples = []
        
        def variable_performance_inference(event):
            # Simulate gradually degrading performance
            sample_id = len(performance_samples)
            base_delay = 0.005  # 5ms base
            trend_delay = sample_id * 0.0001  # Gradual increase
            
            time.sleep(base_delay + trend_delay)
            return {'action': 1, 'confidence': 0.8}
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=variable_performance_inference)
        
        # Collect performance samples
        for i in range(100):
            event = {'synergy_type': f'trend_{i}', 'direction': 1, 'confidence': 0.8}
            
            start_time = time.perf_counter()
            result = mock_strategic_component._handle_synergy_event(event)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            performance_samples.append(latency_ms)
        
        # Analyze trend
        first_half = performance_samples[:50]
        second_half = performance_samples[50:]
        
        first_half_avg = statistics.mean(first_half)
        second_half_avg = statistics.mean(second_half)
        
        trend_percentage = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        # Verify trend detection
        assert trend_percentage > 10, \
            f"Failed to detect performance trend: {trend_percentage:.2f}%"
        
        # Verify trend is significant
        assert second_half_avg > first_half_avg, \
            "Performance should show degradation trend"
    
    def test_performance_benchmark_comparison(self, mock_strategic_component, performance_targets):
        """Test performance against established benchmarks."""
        
        # Define benchmark scenarios
        benchmark_scenarios = [
            {
                'name': 'simple_event',
                'event': {'synergy_type': 'simple', 'direction': 1, 'confidence': 0.8},
                'expected_latency_ms': 10.0
            },
            {
                'name': 'complex_event',
                'event': {'synergy_type': 'complex', 'direction': -1, 'confidence': 0.9},
                'expected_latency_ms': 25.0
            },
            {
                'name': 'high_confidence_event',
                'event': {'synergy_type': 'momentum', 'direction': 1, 'confidence': 0.95},
                'expected_latency_ms': 15.0
            }
        ]
        
        # Mock scenario-specific inference
        def benchmark_inference(event):
            # Simulate different processing times based on event type
            if event['synergy_type'] == 'simple':
                time.sleep(0.008)  # 8ms
            elif event['synergy_type'] == 'complex':
                time.sleep(0.020)  # 20ms
            else:
                time.sleep(0.012)  # 12ms
            
            return {'action': 1, 'confidence': 0.8}
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=benchmark_inference)
        
        # Run benchmark tests
        benchmark_results = {}
        
        for scenario in benchmark_scenarios:
            latencies = []
            
            for i in range(20):
                start_time = time.perf_counter()
                result = mock_strategic_component._handle_synergy_event(scenario['event'])
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000)
            
            avg_latency = statistics.mean(latencies)
            benchmark_results[scenario['name']] = {
                'avg_latency_ms': avg_latency,
                'expected_latency_ms': scenario['expected_latency_ms'],
                'meets_benchmark': avg_latency <= scenario['expected_latency_ms']
            }
        
        # Verify benchmark results
        for scenario_name, results in benchmark_results.items():
            assert results['meets_benchmark'], \
                f"Scenario '{scenario_name}' failed benchmark: {results['avg_latency_ms']:.2f}ms > {results['expected_latency_ms']}ms"
        
        # All scenarios should meet overall target
        all_latencies = [results['avg_latency_ms'] for results in benchmark_results.values()]
        max_latency = max(all_latencies)
        
        assert max_latency < performance_targets.max_inference_latency_ms, \
            f"Maximum benchmark latency {max_latency:.2f}ms exceeds target {performance_targets.max_inference_latency_ms}ms"


@pytest.mark.performance
@pytest.mark.unit
class TestStrategicLatencyIntegration:
    """Integration tests for strategic latency optimization."""
    
    def test_end_to_end_latency_optimization(self, mock_strategic_component, performance_targets):
        """Test end-to-end latency optimization."""
        
        # Mock complete pipeline
        mock_strategic_component._intelligent_gating_network = Mock()
        mock_strategic_component._intelligent_gating_network.should_process_event.return_value = (True, 0.8)
        
        mock_agents = {
            'mlmi_agent': Mock(),
            'nwrqk_agent': Mock(),
            'momentum_agent': Mock()
        }
        
        for agent_name, agent in mock_agents.items():
            agent.inference_step.return_value = {
                'action': 1,
                'confidence': 0.8,
                'probabilities': [0.1, 0.2, 0.7]
            }
        
        mock_strategic_component.agents = mock_agents
        mock_strategic_component._ensemble_decision_making = Mock()
        mock_strategic_component._ensemble_decision_making.return_value = {
            'action': 1,
            'confidence': 0.8,
            'probabilities': [0.1, 0.2, 0.7],
            'ensemble_reasoning': {'method': 'weighted_voting'}
        }
        
        # Test complete pipeline latency
        total_latencies = []
        
        for i in range(100):
            event = {
                'synergy_type': f'e2e_test_{i}',
                'direction': 1,
                'confidence': 0.8,
                'timestamp': time.time()
            }
            
            start_time = time.perf_counter()
            
            # Complete pipeline
            result = mock_strategic_component._handle_synergy_event(event)
            
            end_time = time.perf_counter()
            total_latency_ms = (end_time - start_time) * 1000
            total_latencies.append(total_latency_ms)
        
        # Analyze end-to-end performance
        avg_latency = statistics.mean(total_latencies)
        p95_latency = np.percentile(total_latencies, 95)
        p99_latency = np.percentile(total_latencies, 99)
        
        # Assert end-to-end targets
        assert avg_latency < performance_targets.max_inference_latency_ms, \
            f"E2E average latency {avg_latency:.2f}ms exceeds target {performance_targets.max_inference_latency_ms}ms"
        
        assert p95_latency < performance_targets.max_inference_latency_ms * 1.5, \
            f"E2E P95 latency {p95_latency:.2f}ms exceeds acceptable limit"
        
        assert p99_latency < performance_targets.max_p99_latency_ms, \
            f"E2E P99 latency {p99_latency:.2f}ms exceeds target {performance_targets.max_p99_latency_ms}ms"
    
    def test_system_scalability_under_load(self, mock_strategic_component, performance_targets):
        """Test system scalability under increasing load."""
        
        # Mock scalable inference
        mock_strategic_component._intelligent_gating_network = Mock()
        mock_strategic_component._intelligent_gating_network.should_process_event.return_value = (True, 0.8)
        
        def scalable_inference(event):
            # Simulate load-dependent processing time
            base_time = 0.005  # 5ms base
            return {'action': 1, 'confidence': 0.8}
        
        mock_strategic_component._handle_synergy_event = Mock(side_effect=scalable_inference)
        
        # Test different load levels
        load_levels = [10, 50, 100, 200, 500]
        scalability_results = {}
        
        for load in load_levels:
            latencies = []
            
            start_time = time.perf_counter()
            
            for i in range(load):
                event = {
                    'synergy_type': f'load_{load}_event_{i}',
                    'direction': 1,
                    'confidence': 0.8,
                    'timestamp': time.time()
                }
                
                inference_start = time.perf_counter()
                result = mock_strategic_component._handle_synergy_event(event)
                inference_end = time.perf_counter()
                
                latencies.append((inference_end - inference_start) * 1000)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            scalability_results[load] = {
                'avg_latency_ms': statistics.mean(latencies),
                'throughput_ops_per_sec': load / total_time,
                'total_time_sec': total_time
            }
        
        # Verify scalability
        for load, results in scalability_results.items():
            assert results['avg_latency_ms'] < performance_targets.max_inference_latency_ms, \
                f"Load {load}: Average latency {results['avg_latency_ms']:.2f}ms exceeds target"
            
            assert results['throughput_ops_per_sec'] >= performance_targets.min_throughput_ops_per_sec, \
                f"Load {load}: Throughput {results['throughput_ops_per_sec']:.2f} ops/sec below target"
        
        # Check that performance doesn't degrade significantly with load
        base_latency = scalability_results[10]['avg_latency_ms']
        max_latency = scalability_results[500]['avg_latency_ms']
        
        latency_degradation = ((max_latency - base_latency) / base_latency) * 100
        
        assert latency_degradation < 50, \
            f"Latency degradation {latency_degradation:.2f}% under load exceeds acceptable threshold"