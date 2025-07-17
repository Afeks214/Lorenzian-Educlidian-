"""
Integration Test Suite for Matrix Assembly Pipeline

This test suite validates the complete matrix pipeline from raw data through
indicators to matrix assembly and normalized output, including memory usage
and performance validation.
"""

import pytest
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import threading
import time
import gc
import psutil
import os

# Import components for integration testing
from src.matrix.assembler_30m import MatrixAssembler30m
from src.matrix.assembler_5m import MatrixAssembler5m
from src.matrix.normalizers import RollingNormalizer
from src.core.minimal_dependencies import EventType, Event
from src.utils.logger import get_logger


class TestMatrixPipelineIntegration:
    """Integration tests for the complete matrix pipeline."""
    
    @pytest.fixture
    def mock_kernel(self):
        """Mock kernel with event bus."""
        kernel = Mock()
        event_bus = Mock()
        kernel.get_event_bus.return_value = event_bus
        return kernel
    
    @pytest.fixture
    def strategic_config(self, mock_kernel):
        """Strategic (30m) configuration."""
        return {
            'name': 'Strategic30m',
            'window_size': 50,
            'features': [
                'mlmi_value',
                'mlmi_signal',
                'nwrqk_value',
                'nwrqk_slope',
                'lvn_distance_points',
                'lvn_nearest_strength',
                'time_hour_sin',
                'time_hour_cos'
            ],
            'kernel': mock_kernel,
            'warmup_period': 25
        }
    
    @pytest.fixture
    def tactical_config(self, mock_kernel):
        """Tactical (5m) configuration."""
        return {
            'name': 'Tactical5m',
            'window_size': 100,
            'features': [
                'fvg_bullish_active',
                'fvg_bearish_active',
                'fvg_nearest_level',
                'fvg_age',
                'fvg_mitigation_signal',
                'price_momentum_5',
                'volume_ratio'
            ],
            'kernel': mock_kernel,
            'warmup_period': 50
        }
    
    @pytest.fixture
    def dual_assembler_setup(self, strategic_config, tactical_config):
        """Setup both assemblers for integration testing."""
        strategic = MatrixAssembler30m(strategic_config)
        tactical = MatrixAssembler5m(tactical_config)
        return strategic, tactical
    
    @pytest.fixture
    def market_data_generator(self):
        """Generate realistic market data for testing."""
        class MarketDataGenerator:
            def __init__(self):
                self.current_price = 4145.0
                self.current_volume = 1000
                self.timestamp = datetime.now()
                self.trend = 0.0
                self.volatility = 0.001
                
            def generate_bar(self) -> Dict[str, Any]:
                """Generate a single market bar."""
                # Price movement
                price_change = np.random.normal(self.trend, self.volatility)
                self.current_price *= (1 + price_change)
                
                # Volume variation
                volume_multiplier = np.random.lognormal(0, 0.5)
                self.current_volume = max(100, int(1000 * volume_multiplier))
                
                # Advance time
                self.timestamp += timedelta(minutes=5)
                
                return {
                    'timestamp': self.timestamp,
                    'current_price': self.current_price,
                    'close': self.current_price,
                    'current_volume': self.current_volume,
                    'volume': self.current_volume,
                    'open': self.current_price * (1 + np.random.normal(0, 0.0005)),
                    'high': self.current_price * (1 + abs(np.random.normal(0, 0.001))),
                    'low': self.current_price * (1 - abs(np.random.normal(0, 0.001))),
                }
            
            def generate_strategic_features(self, base_bar: Dict[str, Any]) -> Dict[str, Any]:
                """Generate strategic features for 30m timeframe."""
                features = base_bar.copy()
                
                # MLMI features
                features['mlmi_value'] = np.random.uniform(20, 80)
                features['mlmi_signal'] = np.random.choice([-1, 0, 1])
                
                # NWRQK features
                features['nwrqk_value'] = self.current_price + np.random.normal(0, 10)
                features['nwrqk_slope'] = np.random.normal(0, 0.1)
                
                # LVN features
                features['lvn_distance_points'] = abs(np.random.normal(0, 20))
                features['lvn_nearest_strength'] = np.random.uniform(0, 100)
                
                return features
            
            def generate_tactical_features(self, base_bar: Dict[str, Any]) -> Dict[str, Any]:
                """Generate tactical features for 5m timeframe."""
                features = base_bar.copy()
                
                # FVG features
                fvg_probability = 0.1
                features['fvg_bullish_active'] = 1.0 if np.random.random() < fvg_probability else 0.0
                features['fvg_bearish_active'] = 1.0 if np.random.random() < fvg_probability else 0.0
                
                if features['fvg_bullish_active'] or features['fvg_bearish_active']:
                    features['fvg_nearest_level'] = self.current_price + np.random.normal(0, 5)
                    features['fvg_age'] = max(0, np.random.exponential(10))
                    features['fvg_mitigation_signal'] = 1.0 if np.random.random() < 0.05 else 0.0
                else:
                    features['fvg_nearest_level'] = 0.0
                    features['fvg_age'] = 0.0
                    features['fvg_mitigation_signal'] = 0.0
                
                # Price momentum (5-bar)
                features['price_momentum_5'] = np.random.normal(0, 0.5)
                
                # Volume ratio
                features['volume_ratio'] = np.random.lognormal(0, 0.3)
                
                return features
        
        return MarketDataGenerator()


class TestRawDataToMatrixPipeline:
    """Test complete pipeline from raw data to matrix output."""
    
    def test_strategic_pipeline_end_to_end(self, strategic_config, market_data_generator):
        """Test complete strategic pipeline end-to-end."""
        assembler = MatrixAssembler30m(strategic_config)
        
        # Simulate data flow
        matrices = []
        
        for i in range(100):
            # Generate market data
            base_bar = market_data_generator.generate_bar()
            strategic_features = market_data_generator.generate_strategic_features(base_bar)
            
            # Process through assembler
            assembler._update_matrix(strategic_features)
            
            # Collect matrix when ready
            if assembler.is_ready():
                matrix = assembler.get_matrix()
                if matrix is not None:
                    matrices.append(matrix.copy())
        
        # Validate pipeline output
        assert len(matrices) > 0
        
        # Check matrix properties
        final_matrix = matrices[-1]
        assert final_matrix.shape[1] == 8  # 8 strategic features
        assert final_matrix.shape[0] <= 50  # Window size
        
        # Check normalization
        assert np.all(np.isfinite(final_matrix))
        assert np.all(final_matrix >= -3.0)  # Within reasonable bounds
        assert np.all(final_matrix <= 3.0)
        
        # Check temporal consistency
        assert assembler.n_updates == 100
        assert assembler.is_ready()
    
    def test_tactical_pipeline_end_to_end(self, tactical_config, market_data_generator):
        """Test complete tactical pipeline end-to-end."""
        assembler = MatrixAssembler5m(tactical_config)
        
        # Simulate high-frequency tactical data
        matrices = []
        fvg_summaries = []
        
        for i in range(200):
            # Generate market data
            base_bar = market_data_generator.generate_bar()
            tactical_features = market_data_generator.generate_tactical_features(base_bar)
            
            # Process through assembler
            assembler._update_matrix(tactical_features)
            
            # Collect outputs when ready
            if assembler.is_ready():
                matrix = assembler.get_matrix()
                if matrix is not None:
                    matrices.append(matrix.copy())
                
                # Get FVG summary
                fvg_summary = assembler.get_fvg_summary()
                fvg_summaries.append(fvg_summary)
        
        # Validate tactical pipeline
        assert len(matrices) > 0
        assert len(fvg_summaries) > 0
        
        # Check matrix properties
        final_matrix = matrices[-1]
        assert final_matrix.shape[1] == 7  # 7 tactical features
        assert final_matrix.shape[0] <= 100  # Window size
        
        # Check FVG analysis
        final_summary = fvg_summaries[-1]
        assert final_summary['status'] == 'ready'
        assert 'last_20_bars' in final_summary
        
        # Check processing speed
        assert assembler.n_updates == 200
        assert assembler.is_ready()
    
    def test_dual_timeframe_integration(self, dual_assembler_setup, market_data_generator):
        """Test integration of both strategic and tactical assemblers."""
        strategic, tactical = dual_assembler_setup
        
        # Simulate coordinated data flow
        strategic_matrices = []
        tactical_matrices = []
        
        # Strategic updates every 6 tactical updates (30m vs 5m)
        strategic_counter = 0
        
        for i in range(300):
            # Generate base market data
            base_bar = market_data_generator.generate_bar()
            
            # Always update tactical (5m)
            tactical_features = market_data_generator.generate_tactical_features(base_bar)
            tactical._update_matrix(tactical_features)
            
            # Update strategic every 6 bars (30m)
            strategic_counter += 1
            if strategic_counter >= 6:
                strategic_features = market_data_generator.generate_strategic_features(base_bar)
                strategic._update_matrix(strategic_features)
                strategic_counter = 0
            
            # Collect matrices
            if strategic.is_ready():
                strategic_matrix = strategic.get_matrix()
                if strategic_matrix is not None:
                    strategic_matrices.append(strategic_matrix.copy())
            
            if tactical.is_ready():
                tactical_matrix = tactical.get_matrix()
                if tactical_matrix is not None:
                    tactical_matrices.append(tactical_matrix.copy())
        
        # Validate dual timeframe integration
        assert len(strategic_matrices) > 0
        assert len(tactical_matrices) > 0
        
        # Strategic should have fewer updates than tactical
        assert strategic.n_updates < tactical.n_updates
        
        # Both should be ready
        assert strategic.is_ready()
        assert tactical.is_ready()
        
        # Check consistency
        assert len(strategic_matrices) < len(tactical_matrices)
    
    def test_indicator_integration_simulation(self, strategic_config, market_data_generator):
        """Test integration with simulated indicator engine."""
        assembler = MatrixAssembler30m(strategic_config)
        
        # Mock indicator engine output
        class MockIndicatorEngine:
            def __init__(self):
                self.feature_store = {}
                
            def process_bar(self, bar_data: Dict[str, Any]) -> Dict[str, Any]:
                """Simulate indicator processing."""
                features = market_data_generator.generate_strategic_features(bar_data)
                self.feature_store = features
                return features
        
        indicator_engine = MockIndicatorEngine()
        
        # Simulate complete pipeline
        for i in range(80):
            # Raw market data
            raw_bar = market_data_generator.generate_bar()
            
            # Indicator processing
            feature_store = indicator_engine.process_bar(raw_bar)
            
            # Matrix assembly
            assembler._update_matrix(feature_store)
            
            # Validate intermediate state
            if i % 10 == 0:
                stats = assembler.get_statistics()
                assert stats['n_updates'] == i + 1
                assert stats['error_count'] == 0
        
        # Final validation
        assert assembler.is_ready()
        
        final_matrix = assembler.get_matrix()
        assert final_matrix is not None
        assert final_matrix.shape == (50, 8)
        
        # Check feature importance
        importance = assembler.get_feature_importance()
        assert len(importance) == 8
        assert abs(sum(importance.values()) - 1.0) < 1e-6


class TestMemoryEfficiencyValidation:
    """Test memory efficiency under various conditions."""
    
    def test_memory_usage_under_continuous_load(self, strategic_config, market_data_generator):
        """Test memory usage under continuous load."""
        assembler = MatrixAssembler30m(strategic_config)
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Simulate continuous operation
        for i in range(10000):
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_strategic_features(base_bar)
            
            assembler._update_matrix(features)
            
            # Periodic garbage collection
            if i % 1000 == 0:
                gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024
        
        # Matrix should maintain constant size
        assert assembler.matrix.shape == (50, 8)
        
        # Check that normalizers don't grow indefinitely
        for normalizer in assembler.normalizers.values():
            assert normalizer.n_samples <= 10000
    
    def test_memory_efficiency_dual_assemblers(self, dual_assembler_setup, market_data_generator):
        """Test memory efficiency with dual assemblers."""
        strategic, tactical = dual_assembler_setup
        
        # Monitor memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run both assemblers
        for i in range(5000):
            base_bar = market_data_generator.generate_bar()
            
            # Update tactical every bar
            tactical_features = market_data_generator.generate_tactical_features(base_bar)
            tactical._update_matrix(tactical_features)
            
            # Update strategic every 6 bars
            if i % 6 == 0:
                strategic_features = market_data_generator.generate_strategic_features(base_bar)
                strategic._update_matrix(strategic_features)
            
            # Periodic cleanup
            if i % 1000 == 0:
                gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should be reasonable for dual assemblers
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
        
        # Check matrix sizes
        assert strategic.matrix.shape == (50, 8)
        assert tactical.matrix.shape == (100, 7)
    
    def test_memory_leak_detection(self, strategic_config, market_data_generator):
        """Test for memory leaks during long-running operation."""
        assembler = MatrixAssembler30m(strategic_config)
        
        # Monitor memory over time
        process = psutil.Process(os.getpid())
        memory_samples = []
        
        for cycle in range(10):
            cycle_start_memory = process.memory_info().rss
            
            # Run for many updates
            for i in range(1000):
                base_bar = market_data_generator.generate_bar()
                features = market_data_generator.generate_strategic_features(base_bar)
                assembler._update_matrix(features)
            
            # Force garbage collection
            gc.collect()
            
            cycle_end_memory = process.memory_info().rss
            memory_samples.append(cycle_end_memory - cycle_start_memory)
        
        # Check for memory leaks
        # Memory increase should stabilize (not grow continuously)
        if len(memory_samples) > 5:
            recent_samples = memory_samples[-5:]
            memory_growth = max(recent_samples) - min(recent_samples)
            
            # Growth should be minimal
            assert memory_growth < 10 * 1024 * 1024  # Less than 10MB variation
    
    def test_large_scale_matrix_operations(self, strategic_config, market_data_generator):
        """Test memory efficiency with large-scale matrix operations."""
        # Create assembler with larger window
        strategic_config['window_size'] = 1000
        assembler = MatrixAssembler30m(strategic_config)
        
        # Monitor memory during matrix operations
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Fill large matrix
        for i in range(2000):
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_strategic_features(base_bar)
            assembler._update_matrix(features)
        
        # Perform multiple matrix retrievals
        for i in range(100):
            matrix = assembler.get_matrix()
            assert matrix is not None
            assert matrix.shape == (1000, 8)
            
            # Simulate matrix processing
            processed_matrix = matrix * 2.0
            normalized_matrix = processed_matrix / np.std(processed_matrix)
            
            # Clean up references
            del matrix, processed_matrix, normalized_matrix
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Should be reasonable even with large matrices
        assert memory_increase < 200 * 1024 * 1024  # Less than 200MB


class TestPerformanceValidation:
    """Test performance requirements and benchmarks."""
    
    def test_strategic_update_performance(self, strategic_config, market_data_generator):
        """Test strategic update performance requirements."""
        assembler = MatrixAssembler30m(strategic_config)
        
        # Warm up
        for i in range(50):
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_strategic_features(base_bar)
            assembler._update_matrix(features)
        
        # Measure update performance
        update_times = []
        
        for i in range(1000):
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_strategic_features(base_bar)
            
            start_time = time.perf_counter()
            assembler._update_matrix(features)
            end_time = time.perf_counter()
            
            update_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Analyze performance
        avg_update_time = np.mean(update_times)
        p95_update_time = np.percentile(update_times, 95)
        max_update_time = np.max(update_times)
        
        # Strategic requirements (less stringent than tactical)
        assert avg_update_time < 5.0   # Average < 5ms
        assert p95_update_time < 10.0  # 95th percentile < 10ms
        assert max_update_time < 50.0  # Maximum < 50ms
    
    def test_tactical_update_performance(self, tactical_config, market_data_generator):
        """Test tactical update performance requirements."""
        assembler = MatrixAssembler5m(tactical_config)
        
        # Warm up
        for i in range(100):
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_tactical_features(base_bar)
            assembler._update_matrix(features)
        
        # Measure update performance
        update_times = []
        
        for i in range(2000):
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_tactical_features(base_bar)
            
            start_time = time.perf_counter()
            assembler._update_matrix(features)
            end_time = time.perf_counter()
            
            update_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Analyze performance
        avg_update_time = np.mean(update_times)
        p95_update_time = np.percentile(update_times, 95)
        max_update_time = np.max(update_times)
        
        # Tactical requirements (more stringent)
        assert avg_update_time < 1.0   # Average < 1ms
        assert p95_update_time < 2.0   # 95th percentile < 2ms
        assert max_update_time < 10.0  # Maximum < 10ms
    
    def test_matrix_retrieval_performance(self, strategic_config, market_data_generator):
        """Test matrix retrieval performance."""
        assembler = MatrixAssembler30m(strategic_config)
        
        # Fill matrix
        for i in range(100):
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_strategic_features(base_bar)
            assembler._update_matrix(features)
        
        # Measure retrieval performance
        retrieval_times = []
        
        for i in range(1000):
            start_time = time.perf_counter()
            matrix = assembler.get_matrix()
            end_time = time.perf_counter()
            
            retrieval_times.append((end_time - start_time) * 1000)
            
            # Validate matrix
            assert matrix is not None
            assert matrix.shape == (50, 8)
        
        # Analyze retrieval performance
        avg_retrieval_time = np.mean(retrieval_times)
        p95_retrieval_time = np.percentile(retrieval_times, 95)
        
        # Matrix retrieval should be very fast
        assert avg_retrieval_time < 0.1   # Average < 0.1ms
        assert p95_retrieval_time < 0.5   # 95th percentile < 0.5ms
    
    def test_concurrent_performance(self, dual_assembler_setup, market_data_generator):
        """Test concurrent performance with multiple assemblers."""
        strategic, tactical = dual_assembler_setup
        
        # Performance tracking
        results = []
        errors = []
        
        def strategic_worker():
            try:
                times = []
                for i in range(500):
                    base_bar = market_data_generator.generate_bar()
                    features = market_data_generator.generate_strategic_features(base_bar)
                    
                    start_time = time.perf_counter()
                    strategic._update_matrix(features)
                    end_time = time.perf_counter()
                    
                    times.append((end_time - start_time) * 1000)
                
                results.append(('strategic', times))
            except Exception as e:
                errors.append(e)
        
        def tactical_worker():
            try:
                times = []
                for i in range(1000):
                    base_bar = market_data_generator.generate_bar()
                    features = market_data_generator.generate_tactical_features(base_bar)
                    
                    start_time = time.perf_counter()
                    tactical._update_matrix(features)
                    end_time = time.perf_counter()
                    
                    times.append((end_time - start_time) * 1000)
                
                results.append(('tactical', times))
            except Exception as e:
                errors.append(e)
        
        # Run concurrent workers
        threads = []
        for _ in range(2):  # 2 strategic workers
            threads.append(threading.Thread(target=strategic_worker))
        for _ in range(4):  # 4 tactical workers
            threads.append(threading.Thread(target=tactical_worker))
        
        start_time = time.time()
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Validate concurrent performance
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 6  # 2 strategic + 4 tactical
        
        # Check that concurrent execution was efficient
        total_time = end_time - start_time
        assert total_time < 10.0  # Should complete within 10 seconds
        
        # Validate individual worker performance
        for worker_type, times in results:
            avg_time = np.mean(times)
            if worker_type == 'strategic':
                assert avg_time < 10.0  # Relaxed under concurrency
            else:  # tactical
                assert avg_time < 5.0   # Relaxed under concurrency
    
    def test_throughput_benchmark(self, tactical_config, market_data_generator):
        """Test throughput benchmark for tactical processing."""
        assembler = MatrixAssembler5m(tactical_config)
        
        # Throughput test
        num_updates = 50000
        start_time = time.time()
        
        for i in range(num_updates):
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_tactical_features(base_bar)
            assembler._update_matrix(features)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        throughput = num_updates / total_time
        
        # Should achieve high throughput
        assert throughput > 10000  # At least 10,000 updates per second
        
        # Validate final state
        assert assembler.n_updates == num_updates
        assert assembler.is_ready()
        
        # Check that quality is maintained
        final_matrix = assembler.get_matrix()
        assert final_matrix is not None
        assert np.all(np.isfinite(final_matrix))


class TestRobustnessAndErrorHandling:
    """Test robustness and error handling in pipeline."""
    
    def test_malformed_data_handling(self, strategic_config):
        """Test handling of malformed data."""
        assembler = MatrixAssembler30m(strategic_config)
        
        # Test various malformed inputs
        malformed_inputs = [
            {},  # Empty dict
            {'current_price': 'invalid'},  # Invalid price
            {'current_price': np.nan},  # NaN price
            {'current_price': np.inf},  # Infinite price
            {'current_price': -100},  # Negative price
            {'mlmi_value': 'not_a_number'},  # Invalid feature
            {'mlmi_value': 150},  # Out of range value
        ]
        
        for malformed_input in malformed_inputs:
            # Should not crash
            assembler._update_matrix(malformed_input)
        
        # Should have handled all malformed inputs
        assert assembler.n_updates == len(malformed_inputs)
        assert assembler.error_count == 0  # Should handle gracefully
    
    def test_feature_evolution_handling(self, strategic_config, market_data_generator):
        """Test handling of evolving feature sets."""
        assembler = MatrixAssembler30m(strategic_config)
        
        # Start with minimal features
        for i in range(30):
            features = {
                'current_price': 4145.0 + i,
                'mlmi_value': 50.0 + i,
                'timestamp': datetime.now()
            }
            assembler._update_matrix(features)
        
        # Add more features gradually
        for i in range(30):
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_strategic_features(base_bar)
            
            # Remove some features randomly
            if i % 3 == 0:
                features.pop('nwrqk_slope', None)
            if i % 5 == 0:
                features.pop('lvn_distance_points', None)
            
            assembler._update_matrix(features)
        
        # Should handle feature evolution gracefully
        assert assembler.n_updates == 60
        assert assembler.is_ready()
        
        # Check missing feature tracking
        stats = assembler.get_statistics()
        assert 'missing_features' in stats
    
    def test_extreme_market_conditions(self, tactical_config, market_data_generator):
        """Test handling of extreme market conditions."""
        assembler = MatrixAssembler5m(tactical_config)
        
        # Simulate extreme conditions
        extreme_scenarios = [
            # Flash crash
            {'current_price': 4145.0, 'volume_ratio': 50.0, 'price_momentum_5': -10.0},
            # Flash spike
            {'current_price': 4145.0, 'volume_ratio': 30.0, 'price_momentum_5': 15.0},
            # No volume
            {'current_price': 4145.0, 'volume_ratio': 0.0, 'price_momentum_5': 0.0},
            # Extreme FVG age
            {'current_price': 4145.0, 'fvg_age': 1000.0, 'fvg_bullish_active': 1.0},
            # Simultaneous signals
            {'current_price': 4145.0, 'fvg_bullish_active': 1.0, 'fvg_bearish_active': 1.0},
        ]
        
        for scenario in extreme_scenarios:
            # Fill in missing features
            base_bar = market_data_generator.generate_bar()
            features = market_data_generator.generate_tactical_features(base_bar)
            features.update(scenario)
            
            assembler._update_matrix(features)
        
        # Should handle extreme conditions
        assert assembler.n_updates == len(extreme_scenarios)
        
        # Matrix should remain stable
        matrix = assembler.get_matrix()
        if matrix is not None:
            assert np.all(np.isfinite(matrix))
    
    def test_recovery_from_errors(self, strategic_config, market_data_generator):
        """Test recovery from various error conditions."""
        assembler = MatrixAssembler30m(strategic_config)
        
        # Introduce errors and valid data intermittently
        for i in range(100):
            if i % 10 == 0:
                # Invalid data
                invalid_features = {'invalid_key': 'invalid_value'}
                assembler._update_matrix(invalid_features)
            else:
                # Valid data
                base_bar = market_data_generator.generate_bar()
                features = market_data_generator.generate_strategic_features(base_bar)
                assembler._update_matrix(features)
        
        # Should have recovered and processed valid data
        assert assembler.n_updates == 100
        assert assembler.is_ready()
        
        # Should have a valid matrix
        matrix = assembler.get_matrix()
        assert matrix is not None
        assert np.all(np.isfinite(matrix))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])