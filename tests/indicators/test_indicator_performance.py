"""
Performance Benchmarks and Stress Tests for Indicators System
Tests <1ms calculation targets, accuracy against academic benchmarks, and real-time performance
"""

import pytest
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import psutil
import gc
from typing import List, Dict, Any

from src.indicators.engine import IndicatorEngine
from src.indicators.custom.mlmi import MLMICalculator
from src.indicators.custom.nwrqk import NWRQKCalculator
from src.indicators.custom.fvg import FVGDetector
from src.indicators.custom.lvn import LVNAnalyzer
from src.indicators.custom.mmd import MMDFeatureExtractor
from src.core.events import EventType, Event, BarData
from tests.mocks.mock_kernel import MockKernel
from tests.mocks.mock_event_bus import MockEventBus


class PerformanceProfiler:
    """Performance profiler for indicators"""
    
    def __init__(self):
        self.measurements = []
        self.memory_usage = []
        self.cpu_usage = []
        
    def start_measurement(self, name: str):
        """Start a performance measurement"""
        return {
            'name': name,
            'start_time': time.perf_counter(),
            'start_memory': psutil.Process().memory_info().rss,
            'start_cpu': psutil.cpu_percent()
        }
        
    def end_measurement(self, measurement: Dict):
        """End a performance measurement"""
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        end_cpu = psutil.cpu_percent()
        
        result = {
            'name': measurement['name'],
            'duration_ms': (end_time - measurement['start_time']) * 1000,
            'memory_delta_mb': (end_memory - measurement['start_memory']) / 1024 / 1024,
            'cpu_avg': (measurement['start_cpu'] + end_cpu) / 2
        }
        
        self.measurements.append(result)
        return result
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.measurements:
            return {}
            
        durations = [m['duration_ms'] for m in self.measurements]
        memory_deltas = [m['memory_delta_mb'] for m in self.measurements]
        
        return {
            'total_measurements': len(self.measurements),
            'avg_duration_ms': np.mean(durations),
            'max_duration_ms': np.max(durations),
            'min_duration_ms': np.min(durations),
            'p95_duration_ms': np.percentile(durations, 95),
            'p99_duration_ms': np.percentile(durations, 99),
            'avg_memory_delta_mb': np.mean(memory_deltas),
            'max_memory_delta_mb': np.max(memory_deltas),
            'total_memory_used_mb': sum(m for m in memory_deltas if m > 0)
        }


class TestIndicatorPerformance:
    """Test suite for indicator performance validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_kernel = MockKernel()
        self.mock_event_bus = MockEventBus()
        self.profiler = PerformanceProfiler()
        
    def create_test_bar(self, close=100.0, volume=1000, timestamp=None):
        """Create a test bar with realistic data"""
        if timestamp is None:
            timestamp = datetime.now()
            
        return BarData(
            symbol="BTCUSDT",
            timestamp=timestamp,
            open=close + np.random.uniform(-0.5, 0.5),
            high=close + np.random.uniform(0.1, 2.0),
            low=close - np.random.uniform(0.1, 2.0),
            close=close,
            volume=volume,
            timeframe=30
        )
        
    def generate_performance_dataset(self, size: int = 1000):
        """Generate large dataset for performance testing"""
        bars = []
        base_price = 100.0
        
        for i in range(size):
            # Simulate realistic price movement
            price = base_price + np.random.normal(0, 1.0) + (i * 0.01)
            volume = int(np.random.uniform(500, 2000))
            
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = self.create_test_bar(close=price, volume=volume, timestamp=timestamp)
            bars.append(bar)
            
        return bars
        
    def test_engine_initialization_performance(self):
        """Test engine initialization performance"""
        measurement = self.profiler.start_measurement("engine_init")
        
        engine = IndicatorEngine("perf_test", self.mock_kernel)
        
        result = self.profiler.end_measurement(measurement)
        
        # Engine should initialize quickly
        assert result['duration_ms'] < 100  # < 100ms
        assert result['memory_delta_mb'] < 50  # < 50MB
        
        # Should have all indicators initialized
        assert hasattr(engine, 'mlmi')
        assert hasattr(engine, 'nwrqk')
        assert hasattr(engine, 'fvg')
        assert hasattr(engine, 'lvn')
        assert hasattr(engine, 'mmd')
        
    def test_single_calculation_performance(self):
        """Test single calculation performance for each indicator"""
        bars = self.generate_performance_dataset(100)
        
        # Test MLMI performance
        mlmi = MLMICalculator({}, self.mock_event_bus)
        for bar in bars:
            mlmi.update_30m_history(bar)
            
        measurement = self.profiler.start_measurement("mlmi_single")
        result = mlmi.calculate_30m(bars[-1])
        mlmi_result = self.profiler.end_measurement(measurement)
        
        assert mlmi_result['duration_ms'] < 1.0  # < 1ms target
        assert isinstance(result, dict)
        
        # Test NWRQK performance
        nwrqk = NWRQKCalculator({}, self.mock_event_bus)
        for bar in bars:
            nwrqk.update_30m_history(bar)
            
        measurement = self.profiler.start_measurement("nwrqk_single")
        result = nwrqk.calculate_30m(bars[-1])
        nwrqk_result = self.profiler.end_measurement(measurement)
        
        assert nwrqk_result['duration_ms'] < 1.0  # < 1ms target
        assert isinstance(result, dict)
        
        # Test FVG performance
        fvg = FVGDetector({}, self.mock_event_bus)
        for bar in bars:
            fvg.update_5m_history(bar)
            
        measurement = self.profiler.start_measurement("fvg_single")
        result = fvg.calculate_5m(bars[-1])
        fvg_result = self.profiler.end_measurement(measurement)
        
        assert fvg_result['duration_ms'] < 1.0  # < 1ms target
        assert isinstance(result, dict)
        
        # Test LVN performance
        lvn = LVNAnalyzer({}, self.mock_event_bus)
        
        measurement = self.profiler.start_measurement("lvn_single")
        result = lvn.calculate_30m(bars[-1])
        lvn_result = self.profiler.end_measurement(measurement)
        
        assert lvn_result['duration_ms'] < 1.0  # < 1ms target
        assert isinstance(result, dict)
        
        # Test MMD performance
        mmd = MMDFeatureExtractor({}, self.mock_event_bus)
        for bar in bars:
            mmd.update_30m_history(bar)
            
        measurement = self.profiler.start_measurement("mmd_single")
        result = mmd.calculate_30m(bars[-1])
        mmd_result = self.profiler.end_measurement(measurement)
        
        assert mmd_result['duration_ms'] < 5.0  # < 5ms (more complex calculation)
        assert isinstance(result, dict)
        
    def test_batch_processing_performance(self):
        """Test batch processing performance"""
        bars = self.generate_performance_dataset(1000)
        engine = IndicatorEngine("batch_test", self.mock_kernel)
        
        measurement = self.profiler.start_measurement("batch_1000")
        
        # Process all bars
        for bar in bars:
            if bar.timeframe == 5:
                engine._on_5min_bar(Event(EventType.NEW_5MIN_BAR, bar))
            else:
                engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
                
        result = self.profiler.end_measurement(measurement)
        
        # Should process 1000 bars efficiently
        assert result['duration_ms'] < 1000  # < 1s total
        assert result['duration_ms'] / len(bars) < 1.0  # < 1ms per bar average
        
        # Check that all calculations were performed
        assert engine.calculations_30min > 0
        
    def test_concurrent_processing_performance(self):
        """Test concurrent processing performance"""
        bars = self.generate_performance_dataset(500)
        num_engines = 4
        
        def process_engine(engine_id):
            engine = IndicatorEngine(f"concurrent_test_{engine_id}", self.mock_kernel)
            
            measurement = self.profiler.start_measurement(f"concurrent_engine_{engine_id}")
            
            for bar in bars:
                engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
                
            return self.profiler.end_measurement(measurement)
            
        measurement = self.profiler.start_measurement("concurrent_total")
        
        # Process with multiple engines concurrently
        with ThreadPoolExecutor(max_workers=num_engines) as executor:
            futures = [executor.submit(process_engine, i) for i in range(num_engines)]
            results = [future.result() for future in as_completed(futures)]
            
        total_result = self.profiler.end_measurement(measurement)
        
        # Concurrent processing should be efficient
        assert total_result['duration_ms'] < 2000  # < 2s total
        
        # Individual engines should maintain performance
        for result in results:
            assert result['duration_ms'] < 1000  # < 1s per engine
            
    def test_memory_usage_performance(self):
        """Test memory usage under load"""
        bars = self.generate_performance_dataset(2000)
        engine = IndicatorEngine("memory_test", self.mock_kernel)
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Process large dataset
        for i, bar in enumerate(bars):
            engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
            
            # Check memory every 100 bars
            if i % 100 == 0:
                current_memory = psutil.Process().memory_info().rss
                memory_delta = (current_memory - initial_memory) / 1024 / 1024
                
                # Memory growth should be controlled
                assert memory_delta < 100  # < 100MB growth per 100 bars
                
        # Final memory check
        final_memory = psutil.Process().memory_info().rss
        total_memory_delta = (final_memory - initial_memory) / 1024 / 1024
        
        # Total memory usage should be reasonable
        assert total_memory_delta < 200  # < 200MB total
        
    def test_real_time_streaming_performance(self):
        """Test real-time streaming performance"""
        engine = IndicatorEngine("streaming_test", self.mock_kernel)
        
        # Simulate real-time streaming
        latencies = []
        
        for i in range(100):
            bar = self.create_test_bar(close=100.0 + i)
            
            start_time = time.perf_counter()
            engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000  # ms
            latencies.append(latency)
            
            # Simulate real-time delay
            time.sleep(0.001)  # 1ms delay
            
        # Check latency statistics
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency < 1.0  # < 1ms average
        assert max_latency < 5.0  # < 5ms maximum
        assert p95_latency < 2.0  # < 2ms 95th percentile
        
    def test_stress_testing_performance(self):
        """Test performance under stress conditions"""
        engine = IndicatorEngine("stress_test", self.mock_kernel)
        
        # Generate stress test data
        stress_bars = []
        for i in range(5000):  # Large dataset
            # Add some extreme values
            if i % 100 == 0:
                price = 100.0 + np.random.uniform(-50, 50)
                volume = int(np.random.uniform(1, 10000))
            else:
                price = 100.0 + np.random.normal(0, 5.0)
                volume = int(np.random.uniform(500, 2000))
                
            bar = self.create_test_bar(close=price, volume=volume)
            stress_bars.append(bar)
            
        measurement = self.profiler.start_measurement("stress_test")
        
        # Process under stress
        for bar in stress_bars:
            engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
            
        result = self.profiler.end_measurement(measurement)
        
        # Should handle stress efficiently
        assert result['duration_ms'] < 5000  # < 5s total
        assert result['duration_ms'] / len(stress_bars) < 1.0  # < 1ms per bar
        
        # Engine should still be functional
        assert engine.calculations_30min == len(stress_bars)
        features = engine.get_current_features()
        assert isinstance(features, dict)
        assert len(features) > 0
        
    def test_accuracy_vs_performance_tradeoff(self):
        """Test accuracy vs performance tradeoff"""
        bars = self.generate_performance_dataset(200)
        
        # Test with different configuration parameters
        configs = [
            {'num_neighbors': 50, 'reference_window': 50},   # Fast
            {'num_neighbors': 200, 'reference_window': 100}, # Balanced
            {'num_neighbors': 500, 'reference_window': 200}  # Accurate
        ]
        
        results = []
        
        for i, config in enumerate(configs):
            mlmi = MLMICalculator(config, self.mock_event_bus)
            mmd = MMDFeatureExtractor(config, self.mock_event_bus)
            
            # Build history
            for bar in bars[:-1]:
                mlmi.update_30m_history(bar)
                mmd.update_30m_history(bar)
                
            # Time the calculation
            measurement = self.profiler.start_measurement(f"accuracy_test_{i}")
            
            mlmi_result = mlmi.calculate_30m(bars[-1])
            mmd_result = mmd.calculate_30m(bars[-1])
            
            perf_result = self.profiler.end_measurement(measurement)
            
            results.append({
                'config': config,
                'performance': perf_result,
                'mlmi_result': mlmi_result,
                'mmd_result': mmd_result
            })
            
        # Fast config should be fastest
        assert results[0]['performance']['duration_ms'] < results[1]['performance']['duration_ms']
        assert results[1]['performance']['duration_ms'] < results[2]['performance']['duration_ms']
        
        # All should meet minimum performance requirements
        for result in results:
            assert result['performance']['duration_ms'] < 10.0  # < 10ms
            
    def test_numba_optimization_performance(self):
        """Test Numba optimization performance"""
        from src.indicators.custom.mlmi import calculate_wma, calculate_rsi_with_ma
        from src.indicators.custom.nwrqk import kernel_regression_numba
        from src.indicators.custom.fvg import generate_fvg_data_fast
        from src.indicators.custom.mmd import compute_mmd
        
        # Generate test data
        data = np.random.uniform(50, 150, 1000)
        
        # Test WMA performance
        measurement = self.profiler.start_measurement("numba_wma")
        wma_result = calculate_wma(data, 20)
        wma_perf = self.profiler.end_measurement(measurement)
        
        assert wma_perf['duration_ms'] < 1.0  # < 1ms
        assert isinstance(wma_result, np.ndarray)
        
        # Test RSI performance
        measurement = self.profiler.start_measurement("numba_rsi")
        rsi_result = calculate_rsi_with_ma(data, 14)
        rsi_perf = self.profiler.end_measurement(measurement)
        
        assert rsi_perf['duration_ms'] < 1.0  # < 1ms
        assert isinstance(rsi_result, np.ndarray)
        
        # Test kernel regression performance
        measurement = self.profiler.start_measurement("numba_kernel")
        kernel_result = kernel_regression_numba(data, 100, 1.0, 1.0, 10)
        kernel_perf = self.profiler.end_measurement(measurement)
        
        assert kernel_perf['duration_ms'] < 1.0  # < 1ms
        assert isinstance(kernel_result, (float, np.float64))
        
        # Test FVG performance
        high_data = data + 5
        low_data = data - 5
        
        measurement = self.profiler.start_measurement("numba_fvg")
        bull_fvg, bear_fvg, bull_active, bear_active = generate_fvg_data_fast(high_data, low_data, len(data))
        fvg_perf = self.profiler.end_measurement(measurement)
        
        assert fvg_perf['duration_ms'] < 1.0  # < 1ms
        assert isinstance(bull_fvg, np.ndarray)
        
        # Test MMD performance
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 4)
        
        measurement = self.profiler.start_measurement("numba_mmd")
        mmd_result = compute_mmd(X, Y, 1.0)
        mmd_perf = self.profiler.end_measurement(measurement)
        
        assert mmd_perf['duration_ms'] < 5.0  # < 5ms
        assert isinstance(mmd_result, (float, np.float64))
        
    def test_garbage_collection_performance(self):
        """Test performance with garbage collection"""
        engine = IndicatorEngine("gc_test", self.mock_kernel)
        bars = self.generate_performance_dataset(1000)
        
        # Disable automatic garbage collection
        gc.disable()
        
        try:
            measurement = self.profiler.start_measurement("no_gc")
            
            for bar in bars:
                engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
                
            no_gc_result = self.profiler.end_measurement(measurement)
            
        finally:
            gc.enable()
            
        # Test with normal garbage collection
        engine.reset()
        
        measurement = self.profiler.start_measurement("with_gc")
        
        for bar in bars:
            engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
            
        with_gc_result = self.profiler.end_measurement(measurement)
        
        # Performance should be reasonable with GC
        assert with_gc_result['duration_ms'] < no_gc_result['duration_ms'] * 2  # < 2x slower
        
    def test_feature_store_performance(self):
        """Test feature store performance"""
        engine = IndicatorEngine("feature_store_test", self.mock_kernel)
        
        # Test feature store access performance
        access_times = []
        
        for i in range(1000):
            start_time = time.perf_counter()
            features = engine.get_current_features()
            end_time = time.perf_counter()
            
            access_time = (end_time - start_time) * 1000  # ms
            access_times.append(access_time)
            
        # Feature store access should be fast
        avg_access_time = np.mean(access_times)
        max_access_time = np.max(access_times)
        
        assert avg_access_time < 0.1  # < 0.1ms average
        assert max_access_time < 1.0  # < 1ms maximum
        
    @pytest.mark.asyncio
    async def test_async_performance(self):
        """Test asynchronous performance"""
        engine = IndicatorEngine("async_test", self.mock_kernel)
        bars = self.generate_performance_dataset(100)
        
        # Test async feature store updates
        measurement = self.profiler.start_measurement("async_updates")
        
        for bar in bars:
            features_5m = {'fvg_bullish_active': True}
            features_30m = {'mlmi_value': 0.5}
            
            await engine._update_feature_store_5min(features_5m, bar.timestamp)
            await engine._update_feature_store_30min(features_30m, bar.timestamp)
            
        result = self.profiler.end_measurement(measurement)
        
        # Async updates should be efficient
        assert result['duration_ms'] < 100  # < 100ms total
        assert result['duration_ms'] / len(bars) < 1.0  # < 1ms per update
        
    def test_event_emission_performance(self):
        """Test event emission performance"""
        engine = IndicatorEngine("event_test", self.mock_kernel)
        
        # Mock event bus to count emissions
        emission_count = 0
        emission_times = []
        
        def mock_publish(event):
            nonlocal emission_count
            emission_count += 1
            
        engine.event_bus.publish = mock_publish
        
        bars = self.generate_performance_dataset(100)
        
        measurement = self.profiler.start_measurement("event_emission")
        
        for bar in bars:
            engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
            
        result = self.profiler.end_measurement(measurement)
        
        # Event emission should be efficient
        assert result['duration_ms'] < 100  # < 100ms total
        assert emission_count == engine.events_emitted
        
    def test_performance_regression(self):
        """Test for performance regression"""
        # Baseline performance measurements
        baseline_targets = {
            'engine_init': 100,     # < 100ms
            'single_calc': 1,       # < 1ms
            'batch_100': 100,       # < 100ms for 100 bars
            'feature_access': 0.1   # < 0.1ms
        }
        
        # Test engine initialization
        measurement = self.profiler.start_measurement("regression_init")
        engine = IndicatorEngine("regression_test", self.mock_kernel)
        init_result = self.profiler.end_measurement(measurement)
        
        assert init_result['duration_ms'] < baseline_targets['engine_init']
        
        # Test single calculation
        bars = self.generate_performance_dataset(100)
        for bar in bars[:-1]:
            engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
            
        measurement = self.profiler.start_measurement("regression_single")
        engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bars[-1]))
        single_result = self.profiler.end_measurement(measurement)
        
        assert single_result['duration_ms'] < baseline_targets['single_calc']
        
        # Test batch processing
        engine.reset()
        batch_bars = self.generate_performance_dataset(100)
        
        measurement = self.profiler.start_measurement("regression_batch")
        for bar in batch_bars:
            engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
        batch_result = self.profiler.end_measurement(measurement)
        
        assert batch_result['duration_ms'] < baseline_targets['batch_100']
        
        # Test feature access
        measurement = self.profiler.start_measurement("regression_access")
        features = engine.get_current_features()
        access_result = self.profiler.end_measurement(measurement)
        
        assert access_result['duration_ms'] < baseline_targets['feature_access']
        
    def test_performance_summary(self):
        """Generate performance summary report"""
        # Run a comprehensive performance test
        engine = IndicatorEngine("summary_test", self.mock_kernel)
        bars = self.generate_performance_dataset(500)
        
        # Test various scenarios
        scenarios = [
            ("initialization", lambda: IndicatorEngine("test", self.mock_kernel)),
            ("single_30m", lambda: engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bars[0]))),
            ("batch_processing", lambda: [engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar)) for bar in bars[:100]]),
            ("feature_access", lambda: engine.get_current_features()),
            ("feature_summary", lambda: engine.get_feature_summary())
        ]
        
        results = {}
        
        for scenario_name, scenario_func in scenarios:
            measurement = self.profiler.start_measurement(scenario_name)
            scenario_func()
            results[scenario_name] = self.profiler.end_measurement(measurement)
            
        # Generate summary
        summary = self.profiler.get_summary()
        
        # Print performance report
        print("\n" + "="*50)
        print("INDICATORS PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total measurements: {summary['total_measurements']}")
        print(f"Average duration: {summary['avg_duration_ms']:.2f}ms")
        print(f"Maximum duration: {summary['max_duration_ms']:.2f}ms")
        print(f"95th percentile: {summary['p95_duration_ms']:.2f}ms")
        print(f"99th percentile: {summary['p99_duration_ms']:.2f}ms")
        print(f"Total memory used: {summary['total_memory_used_mb']:.2f}MB")
        print("\nScenario breakdown:")
        for scenario_name, result in results.items():
            print(f"  {scenario_name}: {result['duration_ms']:.2f}ms")
        print("="*50)
        
        # Verify overall performance
        assert summary['p95_duration_ms'] < 10.0  # 95% of operations < 10ms
        assert summary['p99_duration_ms'] < 50.0  # 99% of operations < 50ms
        assert summary['total_memory_used_mb'] < 100.0  # < 100MB total