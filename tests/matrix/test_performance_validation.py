"""
Performance Validation Test Suite for Matrix Assembly System

This test suite provides comprehensive performance validation including
memory efficiency, CPU usage, latency requirements, and scalability
testing for the matrix assembly system.
"""

import pytest
import numpy as np
import time
import threading
import gc
import psutil
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor, as_completed
import resource

# Import components for performance testing
from src.matrix.assembler_30m import MatrixAssembler30m
from src.matrix.assembler_5m import MatrixAssembler5m
from src.matrix.normalizers import RollingNormalizer


class PerformanceMonitor:
    """Utility class for monitoring performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss
        self.initial_cpu_times = self.process.cpu_times()
        self.start_time = time.time()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'increase_mb': (memory_info.rss - self.initial_memory) / 1024 / 1024
        }
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Get CPU usage statistics."""
        cpu_times = self.process.cpu_times()
        elapsed = time.time() - self.start_time
        
        return {
            'user_cpu_pct': ((cpu_times.user - self.initial_cpu_times.user) / elapsed) * 100,
            'system_cpu_pct': ((cpu_times.system - self.initial_cpu_times.system) / elapsed) * 100,
            'elapsed_seconds': elapsed
        }


class MockMarketDataGenerator:
    """High-performance mock data generator for testing."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.current_price = 4145.0
        self.current_volume = 1000
        self.counter = 0
        
    def generate_strategic_batch(self, count: int) -> List[Dict[str, Any]]:
        """Generate batch of strategic features."""
        batch = []
        for i in range(count):
            self.counter += 1
            price_change = np.random.normal(0, 0.001)
            self.current_price *= (1 + price_change)
            
            features = {
                'current_price': self.current_price,
                'mlmi_value': np.random.uniform(20, 80),
                'mlmi_signal': np.random.choice([-1, 0, 1]),
                'nwrqk_value': self.current_price + np.random.normal(0, 10),
                'nwrqk_slope': np.random.normal(0, 0.1),
                'lvn_distance_points': abs(np.random.normal(0, 20)),
                'lvn_nearest_strength': np.random.uniform(0, 100),
                'timestamp': datetime.now() + timedelta(minutes=30 * i)
            }
            batch.append(features)
        return batch
    
    def generate_tactical_batch(self, count: int) -> List[Dict[str, Any]]:
        """Generate batch of tactical features."""
        batch = []
        for i in range(count):
            self.counter += 1
            price_change = np.random.normal(0, 0.0005)
            self.current_price *= (1 + price_change)
            
            features = {
                'current_price': self.current_price,
                'fvg_bullish_active': 1.0 if np.random.random() < 0.1 else 0.0,
                'fvg_bearish_active': 1.0 if np.random.random() < 0.1 else 0.0,
                'fvg_nearest_level': self.current_price + np.random.normal(0, 5),
                'fvg_age': max(0, np.random.exponential(10)),
                'fvg_mitigation_signal': 1.0 if np.random.random() < 0.05 else 0.0,
                'price_momentum_5': np.random.normal(0, 0.5),
                'volume_ratio': np.random.lognormal(0, 0.3),
                'current_volume': max(100, int(self.current_volume * np.random.lognormal(0, 0.3))),
                'timestamp': datetime.now() + timedelta(minutes=5 * i)
            }
            batch.append(features)
        return batch


class TestMemoryEfficiency:
    """Test memory efficiency and resource management."""
    
    @pytest.fixture
    def mock_kernel(self):
        """Mock kernel for testing."""
        kernel = Mock()
        event_bus = Mock()
        kernel.get_event_bus.return_value = event_bus
        return kernel
    
    @pytest.fixture
    def strategic_config(self, mock_kernel):
        """Strategic configuration."""
        return {
            'name': 'PerformanceTest30m',
            'window_size': 100,
            'features': [
                'mlmi_value', 'mlmi_signal', 'nwrqk_value', 'nwrqk_slope',
                'lvn_distance_points', 'lvn_nearest_strength',
                'time_hour_sin', 'time_hour_cos'
            ],
            'kernel': mock_kernel,
            'warmup_period': 50
        }
    
    @pytest.fixture
    def tactical_config(self, mock_kernel):
        """Tactical configuration."""
        return {
            'name': 'PerformanceTest5m',
            'window_size': 200,
            'features': [
                'fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
                'fvg_age', 'fvg_mitigation_signal', 'price_momentum_5', 'volume_ratio'
            ],
            'kernel': mock_kernel,
            'warmup_period': 100
        }
    
    def test_memory_usage_strategic_assembler(self, strategic_config):
        """Test memory usage of strategic assembler."""
        monitor = PerformanceMonitor()
        assembler = MatrixAssembler30m(strategic_config)
        data_generator = MockMarketDataGenerator()
        
        # Track memory usage over time
        memory_samples = []
        
        # Initial memory sample
        memory_samples.append(monitor.get_memory_usage())
        
        # Run increasing loads
        load_sizes = [1000, 5000, 10000, 25000, 50000]
        
        for load_size in load_sizes:
            # Generate batch data
            batch = data_generator.generate_strategic_batch(load_size)
            
            # Process batch
            for features in batch:
                assembler._update_matrix(features)
            
            # Force garbage collection
            gc.collect()
            
            # Sample memory
            memory_samples.append(monitor.get_memory_usage())
        
        # Analyze memory usage
        final_memory = memory_samples[-1]
        peak_memory = max(sample['rss_mb'] for sample in memory_samples)
        
        # Memory requirements
        assert final_memory['increase_mb'] < 100  # Less than 100MB increase
        assert peak_memory < 500  # Less than 500MB total
        
        # Check for memory leaks (memory should stabilize)
        if len(memory_samples) > 3:
            recent_growth = memory_samples[-1]['increase_mb'] - memory_samples[-3]['increase_mb']
            assert recent_growth < 20  # Less than 20MB growth in recent samples
        
        # Validate assembler state
        assert assembler.n_updates == sum(load_sizes)
        assert assembler.is_ready()
        
        # Matrix should maintain constant size
        matrix = assembler.get_matrix()
        assert matrix.shape == (100, 8)  # Window size unchanged
    
    def test_memory_efficiency_tactical_assembler(self, tactical_config):
        """Test memory efficiency of tactical assembler under high load."""
        monitor = PerformanceMonitor()
        assembler = MatrixAssembler5m(tactical_config)
        data_generator = MockMarketDataGenerator()
        
        # High-frequency tactical processing
        memory_samples = []
        
        # Process large number of tactical updates
        for batch_num in range(100):
            batch = data_generator.generate_tactical_batch(1000)
            
            # Process batch
            for features in batch:
                assembler._update_matrix(features)
            
            # Sample memory periodically
            if batch_num % 10 == 0:
                gc.collect()
                memory_samples.append(monitor.get_memory_usage())
        
        # Analyze memory efficiency
        final_memory = memory_samples[-1]
        memory_growth = final_memory['increase_mb'] - memory_samples[0]['increase_mb']
        
        # Should maintain low memory footprint
        assert memory_growth < 50  # Less than 50MB growth
        assert final_memory['rss_mb'] < 300  # Less than 300MB total
        
        # Check processing results
        assert assembler.n_updates == 100000
        assert assembler.is_ready()
        
        # Matrix size should be constant
        matrix = assembler.get_matrix()
        assert matrix.shape == (200, 7)
    
    def test_memory_leak_detection(self, strategic_config):
        """Test for memory leaks in long-running operations."""
        monitor = PerformanceMonitor()
        assembler = MatrixAssembler30m(strategic_config)
        data_generator = MockMarketDataGenerator()
        
        # Run multiple cycles to detect leaks
        memory_baseline = []
        
        for cycle in range(20):
            cycle_start_memory = monitor.get_memory_usage()
            
            # Process data
            batch = data_generator.generate_strategic_batch(2000)
            for features in batch:
                assembler._update_matrix(features)
            
            # Force cleanup
            gc.collect()
            
            cycle_end_memory = monitor.get_memory_usage()
            memory_growth = cycle_end_memory['rss_mb'] - cycle_start_memory['rss_mb']
            memory_baseline.append(memory_growth)
        
        # Analyze for leaks
        if len(memory_baseline) > 10:
            # Check if memory growth is stabilizing
            recent_growth = np.mean(memory_baseline[-5:])
            early_growth = np.mean(memory_baseline[:5])
            
            # Recent growth should not be significantly higher
            assert recent_growth <= early_growth * 1.5
            
            # Individual cycle growth should be minimal
            assert recent_growth < 5.0  # Less than 5MB per cycle
    
    def test_concurrent_memory_usage(self, strategic_config, tactical_config):
        """Test memory usage under concurrent operations."""
        monitor = PerformanceMonitor()
        
        # Create multiple assemblers
        strategic_assembler = MatrixAssembler30m(strategic_config)
        tactical_assembler = MatrixAssembler5m(tactical_config)
        
        data_generator = MockMarketDataGenerator()
        
        # Track memory during concurrent operations
        memory_samples = []
        errors = []
        
        def strategic_worker():
            try:
                batch = data_generator.generate_strategic_batch(5000)
                for features in batch:
                    strategic_assembler._update_matrix(features)
            except Exception as e:
                errors.append(e)
        
        def tactical_worker():
            try:
                batch = data_generator.generate_tactical_batch(10000)
                for features in batch:
                    tactical_assembler._update_matrix(features)
            except Exception as e:
                errors.append(e)
        
        # Initial memory
        memory_samples.append(monitor.get_memory_usage())
        
        # Run concurrent workers
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=strategic_worker))
        for _ in range(5):
            threads.append(threading.Thread(target=tactical_worker))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Final memory
        gc.collect()
        memory_samples.append(monitor.get_memory_usage())
        
        # Validate concurrent memory usage
        assert len(errors) == 0
        
        memory_increase = memory_samples[-1]['increase_mb'] - memory_samples[0]['increase_mb']
        assert memory_increase < 200  # Less than 200MB for concurrent operations
        
        # Both assemblers should be functional
        assert strategic_assembler.n_updates == 15000  # 3 * 5000
        assert tactical_assembler.n_updates == 50000   # 5 * 10000


class TestLatencyRequirements:
    """Test latency requirements for real-time processing."""
    
    @pytest.fixture
    def mock_kernel(self):
        kernel = Mock()
        event_bus = Mock()
        kernel.get_event_bus.return_value = event_bus
        return kernel
    
    def test_strategic_latency_requirements(self, mock_kernel):
        """Test strategic assembler latency requirements."""
        config = {
            'name': 'LatencyTest30m',
            'window_size': 50,
            'features': ['mlmi_value', 'mlmi_signal', 'nwrqk_value', 'nwrqk_slope'],
            'kernel': mock_kernel,
            'warmup_period': 25
        }
        
        assembler = MatrixAssembler30m(config)
        data_generator = MockMarketDataGenerator()
        
        # Warm up
        warmup_batch = data_generator.generate_strategic_batch(30)
        for features in warmup_batch:
            assembler._update_matrix(features)
        
        # Measure latency
        latencies = []
        
        test_batch = data_generator.generate_strategic_batch(5000)
        for features in test_batch:
            start_time = time.perf_counter()
            assembler._update_matrix(features)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Analyze latency
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        # Strategic latency requirements
        assert avg_latency < 2.0   # Average < 2ms
        assert p95_latency < 5.0   # 95th percentile < 5ms
        assert p99_latency < 10.0  # 99th percentile < 10ms
        assert max_latency < 50.0  # Maximum < 50ms
        
        # Validate processing
        assert assembler.n_updates == 5030  # Warmup + test
        assert assembler.is_ready()
    
    def test_tactical_latency_requirements(self, mock_kernel):
        """Test tactical assembler latency requirements."""
        config = {
            'name': 'LatencyTest5m',
            'window_size': 100,
            'features': ['fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
                        'fvg_age', 'price_momentum_5', 'volume_ratio'],
            'kernel': mock_kernel,
            'warmup_period': 50
        }
        
        assembler = MatrixAssembler5m(config)
        data_generator = MockMarketDataGenerator()
        
        # Warm up
        warmup_batch = data_generator.generate_tactical_batch(60)
        for features in warmup_batch:
            assembler._update_matrix(features)
        
        # Measure latency under load
        latencies = []
        
        test_batch = data_generator.generate_tactical_batch(10000)
        for features in test_batch:
            start_time = time.perf_counter()
            assembler._update_matrix(features)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Analyze tactical latency
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        # Tactical latency requirements (more stringent)
        assert avg_latency < 0.5   # Average < 0.5ms
        assert p95_latency < 1.0   # 95th percentile < 1ms
        assert p99_latency < 2.0   # 99th percentile < 2ms
        assert max_latency < 10.0  # Maximum < 10ms
        
        # Validate processing
        assert assembler.n_updates == 10060
        assert assembler.is_ready()
    
    def test_matrix_retrieval_latency(self, mock_kernel):
        """Test matrix retrieval latency."""
        config = {
            'name': 'RetrievalLatencyTest',
            'window_size': 500,  # Larger window to test retrieval performance
            'features': ['mlmi_value', 'mlmi_signal', 'nwrqk_value', 'nwrqk_slope'],
            'kernel': mock_kernel,
            'warmup_period': 250
        }
        
        assembler = MatrixAssembler30m(config)
        data_generator = MockMarketDataGenerator()
        
        # Fill matrix
        batch = data_generator.generate_strategic_batch(600)
        for features in batch:
            assembler._update_matrix(features)
        
        # Measure retrieval latency
        retrieval_latencies = []
        
        for _ in range(5000):
            start_time = time.perf_counter()
            matrix = assembler.get_matrix()
            end_time = time.perf_counter()
            
            retrieval_latency_ms = (end_time - start_time) * 1000
            retrieval_latencies.append(retrieval_latency_ms)
            
            # Validate matrix
            assert matrix is not None
            assert matrix.shape == (500, 4)
        
        # Analyze retrieval latency
        avg_retrieval_latency = np.mean(retrieval_latencies)
        p95_retrieval_latency = np.percentile(retrieval_latencies, 95)
        max_retrieval_latency = np.max(retrieval_latencies)
        
        # Retrieval should be very fast
        assert avg_retrieval_latency < 0.1   # Average < 0.1ms
        assert p95_retrieval_latency < 0.5   # 95th percentile < 0.5ms
        assert max_retrieval_latency < 2.0   # Maximum < 2ms
    
    def test_concurrent_latency_impact(self, mock_kernel):
        """Test latency impact under concurrent load."""
        config = {
            'name': 'ConcurrentLatencyTest',
            'window_size': 100,
            'features': ['fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level'],
            'kernel': mock_kernel,
            'warmup_period': 50
        }
        
        assembler = MatrixAssembler5m(config)
        data_generator = MockMarketDataGenerator()
        
        # Warm up
        warmup_batch = data_generator.generate_tactical_batch(60)
        for features in warmup_batch:
            assembler._update_matrix(features)
        
        # Concurrent latency test
        latency_results = []
        errors = []
        
        def latency_worker(worker_id: int):
            try:
                worker_latencies = []
                batch = data_generator.generate_tactical_batch(2000)
                
                for features in batch:
                    start_time = time.perf_counter()
                    assembler._update_matrix(features)
                    end_time = time.perf_counter()
                    
                    latency_ms = (end_time - start_time) * 1000
                    worker_latencies.append(latency_ms)
                
                latency_results.append((worker_id, worker_latencies))
            except Exception as e:
                errors.append(e)
        
        # Run concurrent workers
        threads = []
        for i in range(8):
            threads.append(threading.Thread(target=latency_worker, args=(i,)))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Analyze concurrent latency
        assert len(errors) == 0
        assert len(latency_results) == 8
        
        all_latencies = []
        for worker_id, latencies in latency_results:
            all_latencies.extend(latencies)
        
        # Concurrent latency should still meet requirements
        avg_concurrent_latency = np.mean(all_latencies)
        p95_concurrent_latency = np.percentile(all_latencies, 95)
        
        # Relaxed requirements under concurrency
        assert avg_concurrent_latency < 2.0  # Average < 2ms
        assert p95_concurrent_latency < 5.0  # 95th percentile < 5ms


class TestThroughputAndScalability:
    """Test throughput and scalability limits."""
    
    @pytest.fixture
    def mock_kernel(self):
        kernel = Mock()
        event_bus = Mock()
        kernel.get_event_bus.return_value = event_bus
        return kernel
    
    def test_strategic_throughput(self, mock_kernel):
        """Test strategic assembler throughput."""
        config = {
            'name': 'ThroughputTest30m',
            'window_size': 100,
            'features': ['mlmi_value', 'mlmi_signal', 'nwrqk_value'],
            'kernel': mock_kernel,
            'warmup_period': 50
        }
        
        assembler = MatrixAssembler30m(config)
        data_generator = MockMarketDataGenerator()
        
        # Throughput test
        num_updates = 100000
        batch = data_generator.generate_strategic_batch(num_updates)
        
        start_time = time.time()
        
        for features in batch:
            assembler._update_matrix(features)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        throughput = num_updates / total_time
        
        # Strategic throughput requirement
        assert throughput > 20000  # At least 20,000 updates per second
        
        # Validate processing
        assert assembler.n_updates == num_updates
        assert assembler.is_ready()
    
    def test_tactical_throughput(self, mock_kernel):
        """Test tactical assembler throughput."""
        config = {
            'name': 'ThroughputTest5m',
            'window_size': 200,
            'features': ['fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level'],
            'kernel': mock_kernel,
            'warmup_period': 100
        }
        
        assembler = MatrixAssembler5m(config)
        data_generator = MockMarketDataGenerator()
        
        # High throughput test
        num_updates = 200000
        batch = data_generator.generate_tactical_batch(num_updates)
        
        start_time = time.time()
        
        for features in batch:
            assembler._update_matrix(features)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        throughput = num_updates / total_time
        
        # Tactical throughput requirement
        assert throughput > 50000  # At least 50,000 updates per second
        
        # Validate processing
        assert assembler.n_updates == num_updates
        assert assembler.is_ready()
    
    def test_concurrent_throughput(self, mock_kernel):
        """Test concurrent throughput with multiple assemblers."""
        config = {
            'name': 'ConcurrentThroughputTest',
            'window_size': 100,
            'features': ['fvg_bullish_active', 'fvg_bearish_active'],
            'kernel': mock_kernel,
            'warmup_period': 50
        }
        
        # Create multiple assemblers
        assemblers = [MatrixAssembler5m(config) for _ in range(4)]
        data_generator = MockMarketDataGenerator()
        
        # Concurrent throughput test
        results = []
        errors = []
        
        def throughput_worker(assembler_idx: int):
            try:
                assembler = assemblers[assembler_idx]
                batch = data_generator.generate_tactical_batch(25000)
                
                start_time = time.time()
                
                for features in batch:
                    assembler._update_matrix(features)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                throughput = len(batch) / total_time
                results.append((assembler_idx, throughput))
            except Exception as e:
                errors.append(e)
        
        # Run concurrent workers
        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=throughput_worker, args=(i,)))
        
        overall_start = time.time()
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        overall_end = time.time()
        overall_time = overall_end - overall_start
        
        # Analyze concurrent throughput
        assert len(errors) == 0
        assert len(results) == 4
        
        total_updates = 4 * 25000  # 100,000 total updates
        overall_throughput = total_updates / overall_time
        
        # Concurrent throughput should scale
        assert overall_throughput > 80000  # At least 80,000 updates per second
        
        # Individual throughput should be reasonable
        individual_throughputs = [throughput for _, throughput in results]
        avg_individual_throughput = np.mean(individual_throughputs)
        assert avg_individual_throughput > 15000  # At least 15,000 per assembler
    
    def test_scalability_limits(self, mock_kernel):
        """Test scalability limits with increasing load."""
        config = {
            'name': 'ScalabilityTest',
            'window_size': 500,
            'features': ['mlmi_value', 'mlmi_signal', 'nwrqk_value', 'nwrqk_slope'],
            'kernel': mock_kernel,
            'warmup_period': 250
        }
        
        assembler = MatrixAssembler30m(config)
        data_generator = MockMarketDataGenerator()
        
        # Test with increasing loads
        load_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        throughputs = []
        
        for load_size in load_sizes:
            batch = data_generator.generate_strategic_batch(load_size)
            
            start_time = time.time()
            
            for features in batch:
                assembler._update_matrix(features)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            throughput = load_size / total_time
            throughputs.append(throughput)
        
        # Throughput should scale reasonably
        assert len(throughputs) == len(load_sizes)
        
        # Should maintain reasonable throughput even at high loads
        final_throughput = throughputs[-1]
        assert final_throughput > 10000  # At least 10,000 updates per second
        
        # Throughput should not degrade too much with scale
        if len(throughputs) > 2:
            throughput_degradation = throughputs[0] / throughputs[-1]
            assert throughput_degradation < 5.0  # Less than 5x degradation


class TestResourceUtilization:
    """Test CPU and system resource utilization."""
    
    @pytest.fixture
    def mock_kernel(self):
        kernel = Mock()
        event_bus = Mock()
        kernel.get_event_bus.return_value = event_bus
        return kernel
    
    def test_cpu_utilization(self, mock_kernel):
        """Test CPU utilization under load."""
        config = {
            'name': 'CPUTest',
            'window_size': 200,
            'features': ['fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
                        'fvg_age', 'price_momentum_5', 'volume_ratio'],
            'kernel': mock_kernel,
            'warmup_period': 100
        }
        
        assembler = MatrixAssembler5m(config)
        data_generator = MockMarketDataGenerator()
        monitor = PerformanceMonitor()
        
        # CPU utilization test
        batch = data_generator.generate_tactical_batch(100000)
        
        start_time = time.time()
        initial_cpu = monitor.get_cpu_usage()
        
        for features in batch:
            assembler._update_matrix(features)
        
        end_time = time.time()
        final_cpu = monitor.get_cpu_usage()
        
        # Analyze CPU usage
        processing_time = end_time - start_time
        cpu_efficiency = len(batch) / processing_time
        
        # CPU utilization should be reasonable
        assert cpu_efficiency > 30000  # At least 30,000 updates per second
        
        # Should not consume excessive CPU
        cpu_usage = final_cpu['user_cpu_pct'] + final_cpu['system_cpu_pct']
        assert cpu_usage < 200  # Less than 200% (2 cores)
    
    def test_normalizer_resource_usage(self):
        """Test resource usage of normalizers."""
        monitor = PerformanceMonitor()
        
        # Create many normalizers
        normalizers = [RollingNormalizer(alpha=0.01, warmup_samples=100) for _ in range(100)]
        
        # Feed data to normalizers
        for i in range(10000):
            value = np.sin(i * 0.01) + np.random.normal(0, 0.1)
            
            for normalizer in normalizers:
                normalizer.update(value)
        
        # Check resource usage
        memory_usage = monitor.get_memory_usage()
        cpu_usage = monitor.get_cpu_usage()
        
        # Normalizers should be efficient
        assert memory_usage['increase_mb'] < 50  # Less than 50MB for 100 normalizers
        
        # Test normalization performance
        test_value = 5.0
        
        start_time = time.time()
        
        for _ in range(10000):
            for normalizer in normalizers:
                normalizer.normalize_zscore(test_value)
        
        end_time = time.time()
        normalization_time = end_time - start_time
        
        # Normalization should be fast
        normalizations_per_second = (10000 * 100) / normalization_time
        assert normalizations_per_second > 100000  # At least 100,000 normalizations per second
    
    def test_system_resource_limits(self, mock_kernel):
        """Test behavior at system resource limits."""
        config = {
            'name': 'ResourceLimitTest',
            'window_size': 1000,
            'features': ['mlmi_value', 'mlmi_signal', 'nwrqk_value', 'nwrqk_slope',
                        'lvn_distance_points', 'lvn_nearest_strength'],
            'kernel': mock_kernel,
            'warmup_period': 500
        }
        
        assembler = MatrixAssembler30m(config)
        data_generator = MockMarketDataGenerator()
        monitor = PerformanceMonitor()
        
        # Test with very large workload
        large_batch = data_generator.generate_strategic_batch(500000)
        
        start_time = time.time()
        initial_memory = monitor.get_memory_usage()
        
        for i, features in enumerate(large_batch):
            assembler._update_matrix(features)
            
            # Check system resources periodically
            if i % 50000 == 0:
                current_memory = monitor.get_memory_usage()
                current_cpu = monitor.get_cpu_usage()
                
                # Should not exceed reasonable limits
                assert current_memory['rss_mb'] < 1000  # Less than 1GB
                assert current_cpu['elapsed_seconds'] < 300  # Less than 5 minutes
        
        end_time = time.time()
        final_memory = monitor.get_memory_usage()
        
        # Validate large-scale processing
        assert assembler.n_updates == 500000
        assert assembler.is_ready()
        
        # Resource usage should be reasonable
        processing_time = end_time - start_time
        memory_increase = final_memory['increase_mb']
        
        assert processing_time < 120  # Less than 2 minutes
        assert memory_increase < 500  # Less than 500MB


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])