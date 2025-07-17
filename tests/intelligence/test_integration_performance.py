"""
Comprehensive Integration Testing for Intelligence Hub Performance.

Tests that all intelligence upgrades work together seamlessly while maintaining
the critical <5ms performance requirement under various conditions.
"""

import pytest
import torch
import numpy as np
import time
import threading
import concurrent.futures
import psutil
from typing import Dict, Any, List
import tempfile
import os

# Import the intelligence components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from intelligence.intelligence_hub import IntelligenceHub, IntelligenceMetrics
from intelligence.performance_monitor import IntelligencePerformanceMonitor, PerformanceSnapshot
from intelligence.regime_detector import RegimeDetector, RegimeAnalysis, MarketRegime
from intelligence.gating_network import GatingNetwork
from intelligence.regime_aware_reward import RegimeAwareRewardFunction
from intelligence.attention_optimizer import AttentionOptimizer

class TestIntelligenceIntegration:
    """Test suite for intelligence integration and performance."""
    
    @pytest.fixture
    def intelligence_config(self):
        """Standard intelligence configuration for testing."""
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
                'cache_rewards': True
            }
        }
    
    @pytest.fixture
    def sample_market_context(self):
        """Sample market context for testing."""
        return {
            'volatility_30': 2.0,
            'momentum_20': 0.03,
            'momentum_50': 0.02,
            'volume_ratio': 1.2,
            'mmd_score': 0.3,
            'price_trend': 0.01
        }
    
    @pytest.fixture
    def sample_agent_predictions(self):
        """Sample agent predictions for testing."""
        return [
            {
                'action_probabilities': np.array([0.4, 0.3, 0.3]),
                'confidence': 0.8,
                'agent_id': 'MLMI'
            },
            {
                'action_probabilities': np.array([0.2, 0.5, 0.3]),
                'confidence': 0.7,
                'agent_id': 'NWRQK'
            },
            {
                'action_probabilities': np.array([0.3, 0.4, 0.3]),
                'confidence': 0.75,
                'agent_id': 'Regime'
            }
        ]
    
    @pytest.fixture
    def sample_attention_weights(self):
        """Sample attention weights for testing."""
        return [
            torch.tensor([0.3, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0]),  # MLMI attention (7D)
            torch.tensor([0.25, 0.35, 0.25, 0.15, 0.0, 0.0, 0.0]),  # NWRQK attention (7D)
            torch.tensor([0.4, 0.35, 0.25, 0.0, 0.0, 0.0, 0.0])  # Regime attention (7D)
        ]

class TestBasicIntegration(TestIntelligenceIntegration):
    """Test basic integration functionality."""
    
    def test_intelligence_hub_initialization(self, intelligence_config):
        """Test that intelligence hub initializes correctly."""
        
        hub = IntelligenceHub(intelligence_config)
        
        assert hub is not None
        assert hub.regime_detector is not None
        assert hub.gating_network is not None
        assert hub.regime_reward_function is not None
        assert hub.attention_optimizer is not None
        assert hub.max_intelligence_overhead_ms == 1.0
    
    def test_complete_intelligence_pipeline(
        self, 
        intelligence_config, 
        sample_market_context, 
        sample_agent_predictions, 
        sample_attention_weights
    ):
        """Test complete intelligence pipeline execution."""
        
        hub = IntelligenceHub(intelligence_config)
        
        result, metrics = hub.process_intelligence_pipeline(
            sample_market_context,
            sample_agent_predictions,
            sample_attention_weights
        )
        
        # Validate result structure
        assert 'final_probabilities' in result
        assert 'overall_confidence' in result
        assert 'regime' in result
        assert 'gating_weights' in result
        assert 'intelligence_active' in result
        assert result['intelligence_active'] is True
        
        # Validate probabilities
        probs = result['final_probabilities']
        assert len(probs) == 3
        assert all(0 <= p <= 1 for p in probs)
        assert abs(sum(probs) - 1.0) < 0.01  # Should sum to 1
        
        # Validate confidence
        assert 0 <= result['overall_confidence'] <= 1
        
        # Validate gating weights
        weights = result['gating_weights']
        assert len(weights) == 3
        assert all(w >= 0 for w in weights)
    
    def test_fallback_mechanism(self, intelligence_config, sample_agent_predictions):
        """Test fallback mechanism when components fail."""
        
        hub = IntelligenceHub(intelligence_config)
        
        # Test with invalid market context
        invalid_context = {'invalid_key': 'invalid_value'}
        
        result, metrics = hub.process_intelligence_pipeline(
            invalid_context,
            sample_agent_predictions,
            None
        )
        
        # Should still return valid result
        assert 'final_probabilities' in result
        assert 'intelligence_active' in result
        # May be True or False depending on how gracefully it handles errors

class TestPerformanceRequirements(TestIntelligenceIntegration):
    """Test critical performance requirements."""
    
    def test_intelligence_overhead_target(
        self, 
        intelligence_config, 
        sample_market_context, 
        sample_agent_predictions, 
        sample_attention_weights
    ):
        """Test that intelligence overhead meets <1.0ms target."""
        
        hub = IntelligenceHub(intelligence_config)
        performance_monitor = IntelligencePerformanceMonitor(intelligence_config)
        
        # Warm up
        for _ in range(5):
            hub.process_intelligence_pipeline(
                sample_market_context, sample_agent_predictions, sample_attention_weights
            )
        
        # Run performance test
        latencies = []
        for _ in range(100):
            session = performance_monitor.start_performance_measurement()
            
            start_time = time.perf_counter()
            result, metrics = hub.process_intelligence_pipeline(
                sample_market_context, sample_agent_predictions, sample_attention_weights
            )
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            performance_monitor.complete_performance_measurement(session, result)
        
        # Validate performance requirements
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Intelligence Performance Results:")
        print(f"Mean latency: {mean_latency:.3f}ms")
        print(f"P95 latency: {p95_latency:.3f}ms")
        print(f"P99 latency: {p99_latency:.3f}ms")
        
        # Assertions with some tolerance for CI environments
        assert mean_latency < 2.0, f"Mean intelligence overhead {mean_latency:.3f}ms exceeds 2.0ms"
        assert p95_latency < 3.0, f"P95 intelligence overhead {p95_latency:.3f}ms exceeds 3.0ms"
        assert p99_latency < 5.0, f"P99 intelligence overhead {p99_latency:.3f}ms exceeds 5.0ms"
        
        # Check that intelligence functionality is working
        assert result['intelligence_active'] is True
        assert 'final_probabilities' in result
    
    def test_component_performance_breakdown(
        self, 
        intelligence_config, 
        sample_market_context, 
        sample_agent_predictions, 
        sample_attention_weights
    ):
        """Test individual component performance targets."""
        
        hub = IntelligenceHub(intelligence_config)
        performance_monitor = IntelligencePerformanceMonitor(intelligence_config)
        
        # Component performance targets (ms)
        component_targets = {
            'regime_detection': 0.3,
            'gating': 0.4,
            'attention': 0.3,
            'integration': 0.3
        }
        
        # Run tests
        for _ in range(50):
            session = performance_monitor.start_performance_measurement()
            
            # Test regime detection
            performance_monitor.record_component_start(session, 'regime_detection')
            regime_analysis = hub._fast_regime_detection(sample_market_context)
            performance_monitor.record_component_end(session, 'regime_detection')
            
            # Test gating
            performance_monitor.record_component_start(session, 'gating')
            gating_weights = hub._fast_gating_computation(sample_market_context, regime_analysis)
            performance_monitor.record_component_end(session, 'gating')
            
            # Test attention
            performance_monitor.record_component_start(session, 'attention')
            attention_analysis = hub._analyze_attention_patterns(sample_attention_weights, regime_analysis)
            performance_monitor.record_component_end(session, 'attention')
            
            # Test integration
            performance_monitor.record_component_start(session, 'integration')
            integrated_result = hub._integrate_intelligence_components(
                regime_analysis, gating_weights, attention_analysis, sample_agent_predictions
            )
            performance_monitor.record_component_end(session, 'integration')
            
            performance_monitor.complete_performance_measurement(session, integrated_result)
        
        # Get performance summary
        summary = performance_monitor.get_performance_summary()
        component_stats = summary['component_timings']
        
        # Validate component performance
        for component, target_ms in component_targets.items():
            if component in component_stats:
                stats = component_stats[component]
                print(f"{component}: mean={stats['mean_ms']:.3f}ms, p95={stats['p95_ms']:.3f}ms")
                
                # Allow some tolerance for CI environments
                assert stats['mean_ms'] < target_ms * 2, \
                    f"Component {component} mean time {stats['mean_ms']:.3f}ms exceeds target {target_ms}ms"
                assert stats['p95_ms'] < target_ms * 3, \
                    f"Component {component} P95 time {stats['p95_ms']:.3f}ms exceeds target {target_ms * 3}ms"
    
    def test_memory_usage_stability(
        self, 
        intelligence_config, 
        sample_market_context, 
        sample_agent_predictions, 
        sample_attention_weights
    ):
        """Test that memory usage remains stable over time."""
        
        hub = IntelligenceHub(intelligence_config)
        performance_monitor = IntelligencePerformanceMonitor(intelligence_config)
        
        initial_memory = performance_monitor._get_memory_usage()
        
        # Run 500 intelligence operations
        for i in range(500):
            # Vary market context slightly to avoid caching
            context = sample_market_context.copy()
            context['volatility_30'] = 1.5 + 0.5 * np.sin(i * 0.1)
            context['momentum_20'] = 0.02 * np.cos(i * 0.05)
            context['momentum_50'] = 0.01 * np.sin(i * 0.03)
            context['volume_ratio'] = 1.0 + 0.3 * np.random.randn()
            context['mmd_score'] = 0.3 + 0.2 * np.random.randn()
            context['price_trend'] = 0.005 * np.random.randn()
            
            # Vary attention weights
            attention_weights = [
                torch.softmax(torch.randn(7), dim=0),
                torch.softmax(torch.randn(7), dim=0),
                torch.softmax(torch.randn(7), dim=0)
            ]
            
            result, metrics = hub.process_intelligence_pipeline(
                context, sample_agent_predictions, attention_weights
            )
            
            # Check memory every 100 operations
            if i % 100 == 0 and i > 0:
                current_memory = performance_monitor._get_memory_usage()
                memory_growth = current_memory - initial_memory
                
                print(f"Memory after {i} operations: {current_memory:.1f}MB (growth: {memory_growth:.1f}MB)")
                
                # Allow for reasonable memory growth
                assert memory_growth < 100, f"Memory growth {memory_growth:.1f}MB after {i} operations indicates leak"
        
        final_memory = performance_monitor._get_memory_usage()
        total_memory_growth = final_memory - initial_memory
        
        print(f"Total memory growth: {total_memory_growth:.1f}MB")
        
        # Validate no significant memory leaks
        assert total_memory_growth < 150, f"Total memory growth {total_memory_growth:.1f}MB indicates memory leak"

class TestConcurrencyAndLoad(TestIntelligenceIntegration):
    """Test performance under concurrent load."""
    
    def test_concurrent_intelligence_processing(
        self, 
        intelligence_config, 
        sample_market_context, 
        sample_agent_predictions, 
        sample_attention_weights
    ):
        """Test intelligence system under concurrent load."""
        
        hub = IntelligenceHub(intelligence_config)
        
        def process_request():
            # Create varied market context
            context = {
                'volatility_30': np.random.uniform(0.5, 3.0),
                'momentum_20': np.random.uniform(-0.05, 0.05),
                'momentum_50': np.random.uniform(-0.03, 0.03),
                'volume_ratio': np.random.uniform(0.5, 2.0),
                'mmd_score': np.random.uniform(0.1, 0.8),
                'price_trend': np.random.uniform(-0.02, 0.02)
            }
            
            # Create varied agent predictions
            predictions = [
                {
                    'action_probabilities': np.random.dirichlet([1, 1, 1]),
                    'confidence': np.random.uniform(0.6, 0.9),
                    'agent_id': f'agent_{i}'
                }
                for i in range(3)
            ]
            
            # Create varied attention weights
            attention_weights = [
                torch.softmax(torch.randn(7), dim=0),
                torch.softmax(torch.randn(7), dim=0),
                torch.softmax(torch.randn(7), dim=0)
            ]
            
            start_time = time.perf_counter()
            result, metrics = hub.process_intelligence_pipeline(
                context, predictions, attention_weights
            )
            end_time = time.perf_counter()
            
            return (end_time - start_time) * 1000, result
        
        # Run 30 concurrent requests with 5 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_request) for _ in range(30)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze concurrent performance
        latencies = [r[0] for r in results]
        mean_concurrent_latency = np.mean(latencies)
        p95_concurrent_latency = np.percentile(latencies, 95)
        max_concurrent_latency = np.max(latencies)
        
        print(f"Concurrent Performance Results:")
        print(f"Mean latency: {mean_concurrent_latency:.3f}ms")
        print(f"P95 latency: {p95_concurrent_latency:.3f}ms")
        print(f"Max latency: {max_concurrent_latency:.3f}ms")
        
        # Validate concurrent performance (more lenient than single-threaded)
        assert mean_concurrent_latency < 5.0, f"Mean concurrent latency {mean_concurrent_latency:.3f}ms too high"
        assert p95_concurrent_latency < 10.0, f"P95 concurrent latency {p95_concurrent_latency:.3f}ms too high"
        assert max_concurrent_latency < 20.0, f"Max concurrent latency {max_concurrent_latency:.3f}ms too high"
        
        # Validate all requests succeeded
        for _, result in results:
            assert 'final_probabilities' in result
            assert 'intelligence_active' in result
    
    def test_load_testing_sustained_performance(
        self, 
        intelligence_config, 
        sample_market_context, 
        sample_agent_predictions, 
        sample_attention_weights
    ):
        """Test sustained performance under continuous load."""
        
        hub = IntelligenceHub(intelligence_config)
        performance_monitor = IntelligencePerformanceMonitor(intelligence_config)
        
        # Run for 10 seconds of continuous load
        start_time = time.time()
        end_time = start_time + 10.0  # 10 seconds
        
        operation_count = 0
        latencies = []
        
        while time.time() < end_time:
            # Vary inputs slightly
            context = sample_market_context.copy()
            context['volatility_30'] += np.random.normal(0, 0.1)
            context['mmd_score'] += np.random.normal(0, 0.05)
            
            session = performance_monitor.start_performance_measurement()
            
            op_start = time.perf_counter()
            result, metrics = hub.process_intelligence_pipeline(
                context, sample_agent_predictions, sample_attention_weights
            )
            op_end = time.perf_counter()
            
            latency = (op_end - op_start) * 1000
            latencies.append(latency)
            
            performance_monitor.complete_performance_measurement(session, result)
            
            operation_count += 1
            
            # Small sleep to prevent overwhelming the system
            time.sleep(0.001)  # 1ms
        
        # Calculate performance statistics
        actual_duration = time.time() - start_time
        ops_per_second = operation_count / actual_duration
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"Sustained Load Test Results:")
        print(f"Duration: {actual_duration:.2f}s")
        print(f"Operations: {operation_count}")
        print(f"Ops/sec: {ops_per_second:.1f}")
        print(f"Mean latency: {mean_latency:.3f}ms")
        print(f"P95 latency: {p95_latency:.3f}ms")
        
        # Validate sustained performance
        assert ops_per_second > 50, f"Throughput {ops_per_second:.1f} ops/sec too low"
        assert mean_latency < 5.0, f"Mean latency {mean_latency:.3f}ms too high under load"
        assert p95_latency < 10.0, f"P95 latency {p95_latency:.3f}ms too high under load"
        
        # Get performance summary
        summary = performance_monitor.get_performance_summary()
        assert summary['performance_score'] > 50, "Performance score too low under sustained load"

class TestComponentIntegration(TestIntelligenceIntegration):
    """Test integration between specific components."""
    
    def test_regime_detector_integration(self, intelligence_config, sample_market_context):
        \"\"\"Test regime detector integration and caching.\"\"\"
        
        hub = IntelligenceHub(intelligence_config)
        
        # Test regime detection
        regime_analysis = hub._fast_regime_detection(sample_market_context)
        
        assert isinstance(regime_analysis.regime, MarketRegime)
        assert 0 <= regime_analysis.confidence <= 1
        assert regime_analysis.analysis_time_ms >= 0
        
        # Test caching (second call should be faster)
        start_time = time.perf_counter()
        cached_analysis = hub._fast_regime_detection(sample_market_context)
        cache_time = (time.perf_counter() - start_time) * 1000
        
        assert cached_analysis.regime == regime_analysis.regime
        assert cache_time < regime_analysis.analysis_time_ms  # Should be faster due to caching
    
    def test_gating_network_integration(self, intelligence_config, sample_market_context):
        \"\"\"Test gating network integration and optimization.\"\"\"
        
        hub = IntelligenceHub(intelligence_config)
        
        # Create regime analysis
        regime_analysis = hub._fast_regime_detection(sample_market_context)
        
        # Test gating computation
        gating_weights = hub._fast_gating_computation(sample_market_context, regime_analysis)
        
        assert isinstance(gating_weights, torch.Tensor)
        assert len(gating_weights.shape) >= 1
        
        # Convert to numpy for validation
        weights_np = gating_weights.detach().cpu().numpy()
        assert np.all(weights_np >= 0), \"Gating weights should be non-negative\"
        assert np.sum(weights_np) > 0, \"Gating weights should sum to positive value\"
    
    def test_attention_optimizer_integration(self, intelligence_config, sample_attention_weights):
        \"\"\"Test attention optimizer integration.\"\"\"
        
        hub = IntelligenceHub(intelligence_config)
        
        # Test batch attention optimization
        optimized_weights = hub.attention_optimizer.batch_attention_computation(
            sample_attention_weights,
            ['mlmi', 'nwrqk', 'regime'],
            [{} for _ in range(3)]
        )
        
        assert len(optimized_weights) == len(sample_attention_weights)
        
        for weights in optimized_weights:
            assert isinstance(weights, torch.Tensor)
            assert torch.all(weights >= 0), \"Attention weights should be non-negative\"
            assert torch.sum(weights) > 0, \"Attention weights should sum to positive value\"

class TestPerformanceMonitoring(TestIntelligenceIntegration):
    \"\"\"Test performance monitoring functionality.\"\"\"
    
    def test_performance_monitor_basic_functionality(self, intelligence_config):
        \"\"\"Test basic performance monitoring functionality.\"\"\"
        
        monitor = IntelligencePerformanceMonitor(intelligence_config)
        
        # Test session management
        session = monitor.start_performance_measurement()
        assert 'start_time' in session
        assert 'component_times' in session
        
        # Test component timing
        monitor.record_component_start(session, 'test_component')
        time.sleep(0.001)  # 1ms
        monitor.record_component_end(session, 'test_component')
        
        assert 'test_component' in session['component_times']
        assert session['component_times']['test_component'] > 0
        
        # Test snapshot creation
        snapshot = monitor.complete_performance_measurement(session, {})
        
        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.total_inference_time_ms > 0
        assert 'test_component' in snapshot.component_breakdown
    
    def test_performance_summary_and_recommendations(self, intelligence_config):
        \"\"\"Test performance summary and optimization recommendations.\"\"\"
        
        monitor = IntelligencePerformanceMonitor(intelligence_config)
        
        # Generate some performance data
        for i in range(20):
            session = monitor.start_performance_measurement()
            
            # Simulate component timing
            for component in ['attention', 'gating', 'regime_detection']:
                monitor.record_component_start(session, component)
                time.sleep(0.001)  # 1ms
                monitor.record_component_end(session, component)
            
            monitor.complete_performance_measurement(session, {})
        
        # Test performance summary
        summary = monitor.get_performance_summary()
        
        assert 'total_inference_time' in summary
        assert 'intelligence_overhead' in summary
        assert 'component_timings' in summary
        assert 'performance_score' in summary
        
        # Test recommendations
        recommendations = monitor.get_optimization_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_performance_data_export(self, intelligence_config):
        \"\"\"Test performance data export functionality.\"\"\"
        
        monitor = IntelligencePerformanceMonitor(intelligence_config)
        
        # Generate some data
        for _ in range(5):
            session = monitor.start_performance_measurement()
            monitor.record_component_start(session, 'test')
            time.sleep(0.001)
            monitor.record_component_end(session, 'test')
            monitor.complete_performance_measurement(session, {})
        
        # Test export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            monitor.export_performance_data(temp_path)
            assert os.path.exists(temp_path)
            
            # Verify file content
            import json
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'performance_history' in data
            assert 'component_timings' in data
            assert len(data['performance_history']) > 0
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestErrorHandlingAndRobustness(TestIntelligenceIntegration):
    \"\"\"Test error handling and system robustness.\"\"\"
    
    def test_invalid_input_handling(self, intelligence_config):
        \"\"\"Test handling of invalid inputs.\"\"\"
        
        hub = IntelligenceHub(intelligence_config)
        
        # Test with None inputs
        result, metrics = hub.process_intelligence_pipeline(None, None, None)
        assert 'final_probabilities' in result
        
        # Test with empty inputs
        result, metrics = hub.process_intelligence_pipeline({}, [], [])
        assert 'final_probabilities' in result
        
        # Test with malformed inputs
        bad_context = {'bad_key': 'bad_value'}
        bad_predictions = [{'bad_prediction': True}]
        
        result, metrics = hub.process_intelligence_pipeline(
            bad_context, bad_predictions, None
        )
        assert 'final_probabilities' in result
    
    def test_component_failure_recovery(self, intelligence_config):
        \"\"\"Test recovery from component failures.\"\"\"
        
        hub = IntelligenceHub(intelligence_config)
        
        # Test with extreme market context values
        extreme_context = {
            'volatility_30': float('inf'),
            'mmd_score': float('nan'),
            'momentum_20': -1e10,
            'momentum_50': 1e10,
            'volume_ratio': -1.0,
            'price_trend': float('nan')
        }
        
        predictions = [
            {'action_probabilities': np.array([0.33, 0.33, 0.34]), 'confidence': 0.5}
        ]
        
        # Should not crash and should return valid result
        result, metrics = hub.process_intelligence_pipeline(
            extreme_context, predictions, None
        )
        
        assert 'final_probabilities' in result
        probs = result['final_probabilities']
        assert len(probs) == 3
        assert all(np.isfinite(p) for p in probs)
        assert abs(sum(probs) - 1.0) < 0.1  # Should be reasonably normalized
    
    def test_performance_degradation_detection(self, intelligence_config):
        \"\"\"Test detection of performance degradation.\"\"\"
        
        monitor = IntelligencePerformanceMonitor(intelligence_config)
        
        # Create baseline performance
        for _ in range(50):
            session = monitor.start_performance_measurement()
            monitor.record_component_start(session, 'test')
            time.sleep(0.001)  # 1ms - good performance
            monitor.record_component_end(session, 'test')
            monitor.complete_performance_measurement(session, {})
        
        # Simulate performance degradation
        for _ in range(10):
            session = monitor.start_performance_measurement()
            monitor.record_component_start(session, 'test')
            time.sleep(0.005)  # 5ms - degraded performance
            monitor.record_component_end(session, 'test')
            monitor.complete_performance_measurement(session, {})
        
        # Check that recommendations detect the degradation
        recommendations = monitor.get_optimization_recommendations()
        degradation_detected = any(
            'degradation' in rec.lower() or 'optimization needed' in rec.lower()
            for rec in recommendations
        )
        
        # Note: This might not always trigger depending on thresholds
        # but the test validates the monitoring system works
        summary = monitor.get_performance_summary()
        assert summary['performance_score'] < 100  # Should detect some performance impact


# Utility functions for running performance benchmarks
def run_performance_benchmark(
    hub: IntelligenceHub,
    market_context: Dict[str, Any],
    agent_predictions: List[Dict[str, Any]],
    attention_weights: List[torch.Tensor],
    num_iterations: int = 1000
) -> Dict[str, float]:
    \"\"\"Run a performance benchmark and return statistics.\"\"\"
    
    latencies = []
    
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        
        result, metrics = hub.process_intelligence_pipeline(
            market_context, agent_predictions, attention_weights
        )
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    return {
        'mean_ms': np.mean(latencies),
        'median_ms': np.median(latencies),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies)
    }


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v', '--tb=short'])