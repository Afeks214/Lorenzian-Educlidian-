"""
Test Suite for MatrixAssembler30m - Strategic 30-minute Matrix Assembly

This test suite validates the 30-minute matrix assembler with comprehensive
testing covering feature extraction accuracy, window processing, memory
efficiency, and integration with data sources.
"""

import pytest
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import threading
import time
import gc

# Import the components to test
from src.matrix.assembler_30m import MatrixAssembler30m
from src.matrix.normalizers import RollingNormalizer
from src.core.minimal_dependencies import EventType, Event
from src.utils.logger import get_logger


class TestMatrixAssembler30m:
    """Test suite for 30-minute matrix assembler."""
    
    @pytest.fixture
    def mock_kernel(self):
        """Mock kernel with event bus."""
        kernel = Mock()
        event_bus = Mock()
        kernel.get_event_bus.return_value = event_bus
        return kernel
    
    @pytest.fixture
    def basic_config(self, mock_kernel):
        """Basic configuration for testing."""
        return {
            'name': 'TestAssembler30m',
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
            'warmup_period': 25,
            'feature_configs': {
                'nwrqk_slope': {'ema_alpha': 0.05, 'warmup_samples': 50},
                'mlmi_value': {'ema_alpha': 0.02, 'warmup_samples': 100}
            }
        }
    
    @pytest.fixture
    def assembler(self, basic_config):
        """Create assembler instance."""
        return MatrixAssembler30m(basic_config)
    
    @pytest.fixture
    def sample_feature_store(self):
        """Sample feature store with all required features."""
        return {
            'mlmi_value': 55.0,
            'mlmi_signal': 1.0,
            'nwrqk_value': 4150.0,
            'nwrqk_slope': 0.05,
            'lvn_distance_points': 8.5,
            'lvn_nearest_strength': 75.0,
            'current_price': 4145.0,
            'close': 4145.0,
            'timestamp': datetime.now(),
            'volume': 1000
        }


class TestAssemblerInitialization:
    """Test assembler initialization and configuration."""
    
    def test_successful_initialization(self, basic_config):
        """Test successful assembler initialization."""
        assembler = MatrixAssembler30m(basic_config)
        
        assert assembler.name == 'TestAssembler30m'
        assert assembler.window_size == 50
        assert assembler.n_features == 8
        assert len(assembler.feature_names) == 8
        assert assembler.current_price is None
        assert assembler.price_ema is None
        assert assembler.price_ema_alpha == 0.001
        
        # Check normalizers are initialized
        assert len(assembler.normalizers) == 8
        assert isinstance(assembler.normalizers['nwrqk_slope'], RollingNormalizer)
        
        # Check matrix initialization
        assert assembler.matrix.shape == (50, 8)
        assert assembler.matrix.dtype == np.float32
        assert np.all(assembler.matrix == 0)
    
    def test_invalid_configuration(self, mock_kernel):
        """Test error handling for invalid configuration."""
        
        # Missing window_size
        with pytest.raises(ValueError, match="window_size is required"):
            MatrixAssembler30m({
                'name': 'Test',
                'features': ['mlmi_value'],
                'kernel': mock_kernel
            })
        
        # Missing features
        with pytest.raises(ValueError, match="features list is required"):
            MatrixAssembler30m({
                'name': 'Test',
                'window_size': 50,
                'kernel': mock_kernel
            })
        
        # Missing kernel
        with pytest.raises(ValueError, match="Kernel reference is required"):
            MatrixAssembler30m({
                'name': 'Test',
                'window_size': 50,
                'features': ['mlmi_value']
            })
    
    def test_feature_importance_calculation(self, assembler):
        """Test feature importance calculation."""
        importance = assembler.get_feature_importance()
        
        # Check all features have importance scores
        assert len(importance) == 8
        assert all(0 <= score <= 1 for score in importance.values())
        
        # Check scores sum to 1.0
        assert abs(sum(importance.values()) - 1.0) < 1e-6
        
        # Check specific importance values
        assert importance['mlmi_value'] == 0.20
        assert importance['mlmi_signal'] == 0.15
        assert importance['nwrqk_value'] == 0.15
        assert importance['nwrqk_slope'] == 0.20


class TestFeatureExtraction:
    """Test feature extraction functionality."""
    
    def test_extract_features_with_time_features(self, assembler, sample_feature_store):
        """Test feature extraction with time features."""
        # Set timestamp to specific hour
        sample_feature_store['timestamp'] = datetime(2023, 1, 1, 14, 30)  # 2:30 PM
        
        features = assembler.extract_features(sample_feature_store)
        
        assert features is not None
        assert len(features) == 8
        
        # Check time features are correctly extracted
        time_hour_sin_idx = assembler.feature_names.index('time_hour_sin')
        time_hour_cos_idx = assembler.feature_names.index('time_hour_cos')
        
        # Hour should be 14.5 (2:30 PM)
        assert features[time_hour_sin_idx] == 14.5
        assert features[time_hour_cos_idx] == 14.5
        
        # Check other features
        mlmi_value_idx = assembler.feature_names.index('mlmi_value')
        assert features[mlmi_value_idx] == 55.0
    
    def test_extract_features_without_time_features(self, basic_config, mock_kernel):
        """Test feature extraction without time features."""
        basic_config['features'] = ['mlmi_value', 'mlmi_signal', 'nwrqk_value']
        assembler = MatrixAssembler30m(basic_config)
        
        features = assembler.extract_features(sample_feature_store)
        
        # Should return None to use default extraction
        assert features is None
    
    def test_extract_features_missing_price(self, assembler):
        """Test feature extraction with missing price."""
        feature_store = {
            'mlmi_value': 55.0,
            'mlmi_signal': 1.0,
            'time_hour_sin': 14.5,
            'time_hour_cos': 14.5
        }
        
        features = assembler.extract_features(feature_store)
        
        assert features is None
    
    def test_price_ema_update(self, assembler, sample_feature_store):
        """Test price EMA update mechanism."""
        # First update
        assembler.extract_features(sample_feature_store)
        assert assembler.price_ema == 4145.0
        
        # Second update with different price
        sample_feature_store['current_price'] = 4150.0
        assembler.extract_features(sample_feature_store)
        
        # EMA should be updated
        expected_ema = 4145.0 + 0.001 * (4150.0 - 4145.0)
        assert abs(assembler.price_ema - expected_ema) < 1e-6


class TestFeaturePreprocessing:
    """Test feature preprocessing and normalization."""
    
    def test_mlmi_value_preprocessing(self, assembler):
        """Test MLMI value preprocessing."""
        raw_features = [55.0, 1.0, 4150.0, 0.05, 8.5, 75.0, 14.5, 14.5]
        feature_store = {'current_price': 4145.0}
        
        processed = assembler.preprocess_features(raw_features, feature_store)
        
        # MLMI value should be scaled from [0,100] to [-1,1]
        mlmi_idx = assembler.feature_names.index('mlmi_value')
        expected = (55.0 - 50.0) / 50.0  # Min-max scale with center at 50
        assert abs(processed[mlmi_idx] - 0.1) < 1e-6
    
    def test_mlmi_signal_preprocessing(self, assembler):
        """Test MLMI signal preprocessing."""
        raw_features = [55.0, 1.0, 4150.0, 0.05, 8.5, 75.0, 14.5, 14.5]
        feature_store = {'current_price': 4145.0}
        
        processed = assembler.preprocess_features(raw_features, feature_store)
        
        # MLMI signal should remain as is
        mlmi_signal_idx = assembler.feature_names.index('mlmi_signal')
        assert processed[mlmi_signal_idx] == 1.0
    
    def test_nwrqk_value_preprocessing(self, assembler):
        """Test NWRQK value preprocessing."""
        raw_features = [55.0, 1.0, 4150.0, 0.05, 8.5, 75.0, 14.5, 14.5]
        feature_store = {'current_price': 4145.0}
        
        processed = assembler.preprocess_features(raw_features, feature_store)
        
        # NWRQK value should be normalized as percentage from current price
        nwrqk_idx = assembler.feature_names.index('nwrqk_value')
        expected_pct = ((4150.0 - 4145.0) / 4145.0) * 100  # ~0.12%
        expected_normalized = expected_pct / 5.0  # Scale to [-1, 1]
        assert abs(processed[nwrqk_idx] - expected_normalized) < 1e-3
    
    def test_lvn_distance_preprocessing(self, assembler):
        """Test LVN distance preprocessing."""
        raw_features = [55.0, 1.0, 4150.0, 0.05, 8.5, 75.0, 14.5, 14.5]
        feature_store = {'current_price': 4145.0}
        
        processed = assembler.preprocess_features(raw_features, feature_store)
        
        # LVN distance should use exponential decay
        lvn_idx = assembler.feature_names.index('lvn_distance_points')
        expected_pct = (8.5 / 4145.0) * 100  # Distance as percentage
        expected_decay = np.exp(-expected_pct)
        assert abs(processed[lvn_idx] - expected_decay) < 1e-3
    
    def test_time_feature_preprocessing(self, assembler):
        """Test time feature preprocessing."""
        raw_features = [55.0, 1.0, 4150.0, 0.05, 8.5, 75.0, 14.5, 14.5]
        feature_store = {'current_price': 4145.0}
        
        processed = assembler.preprocess_features(raw_features, feature_store)
        
        # Time features should be cyclically encoded
        sin_idx = assembler.feature_names.index('time_hour_sin')
        cos_idx = assembler.feature_names.index('time_hour_cos')
        
        # For hour 14.5, calculate expected sin/cos values
        expected_sin = np.sin(2 * np.pi * 14.5 / 24)
        expected_cos = np.cos(2 * np.pi * 14.5 / 24)
        
        assert abs(processed[sin_idx] - expected_sin) < 1e-6
        assert abs(processed[cos_idx] - expected_cos) < 1e-6
    
    def test_preprocessing_with_invalid_values(self, assembler):
        """Test preprocessing with invalid values."""
        raw_features = [np.nan, np.inf, -np.inf, 0.05, 8.5, 75.0, 14.5, 14.5]
        feature_store = {'current_price': 4145.0}
        
        processed = assembler.preprocess_features(raw_features, feature_store)
        
        # Should handle invalid values gracefully
        assert np.all(np.isfinite(processed))
        assert np.all(processed >= -3.0)
        assert np.all(processed <= 3.0)
    
    def test_preprocessing_error_handling(self, assembler):
        """Test preprocessing error handling."""
        raw_features = [55.0, 1.0, 4150.0, 0.05]  # Wrong number of features
        feature_store = {'current_price': 4145.0}
        
        # Should not raise exception, should return zeros
        processed = assembler.preprocess_features(raw_features, feature_store)
        assert len(processed) == 4
        assert np.all(processed == 0.0)


class TestFeatureValidation:
    """Test feature validation functionality."""
    
    def test_valid_features(self, assembler):
        """Test validation of valid features."""
        valid_features = [55.0, 1.0, 4150.0, 0.05, 8.5, 75.0, 14.5, 14.5]
        
        assert assembler.validate_features(valid_features) is True
    
    def test_invalid_mlmi_value(self, assembler):
        """Test validation of invalid MLMI value."""
        invalid_features = [105.0, 1.0, 4150.0, 0.05, 8.5, 75.0, 14.5, 14.5]  # MLMI > 100
        
        assert assembler.validate_features(invalid_features) is False
    
    def test_invalid_mlmi_signal(self, assembler):
        """Test validation of invalid MLMI signal."""
        invalid_features = [55.0, 0.5, 4150.0, 0.05, 8.5, 75.0, 14.5, 14.5]  # MLMI signal not in [-1,0,1]
        
        assert assembler.validate_features(invalid_features) is False
    
    def test_invalid_lvn_strength(self, assembler):
        """Test validation of invalid LVN strength."""
        invalid_features = [55.0, 1.0, 4150.0, 0.05, 8.5, 120.0, 14.5, 14.5]  # LVN strength > 100
        
        assert assembler.validate_features(invalid_features) is False
    
    def test_non_finite_values(self, assembler):
        """Test validation of non-finite values."""
        invalid_features = [55.0, 1.0, np.nan, 0.05, 8.5, 75.0, 14.5, 14.5]
        
        assert assembler.validate_features(invalid_features) is False
    
    def test_wrong_feature_count(self, assembler):
        """Test validation with wrong feature count."""
        invalid_features = [55.0, 1.0, 4150.0, 0.05]  # Only 4 features instead of 8
        
        assert assembler.validate_features(invalid_features) is False


class TestWindowProcessing:
    """Test window processing and circular buffer management."""
    
    def test_matrix_update_sequence(self, assembler, sample_feature_store):
        """Test matrix update sequence."""
        # Initial state
        assert assembler.n_updates == 0
        assert assembler.is_full is False
        assert assembler.is_ready() is False
        
        # Update with multiple feature stores
        for i in range(30):
            # Vary the features slightly
            feature_store = sample_feature_store.copy()
            feature_store['mlmi_value'] = 50.0 + i
            
            assembler._update_matrix(feature_store)
            
            assert assembler.n_updates == i + 1
            assert assembler.current_index == (i + 1) % assembler.window_size
            
            # Check ready state
            if i + 1 >= assembler._warmup_period:
                assert assembler.is_ready() is True
    
    def test_circular_buffer_wraparound(self, assembler, sample_feature_store):
        """Test circular buffer wraparound."""
        # Fill beyond window size
        for i in range(60):  # More than window_size (50)
            feature_store = sample_feature_store.copy()
            feature_store['mlmi_value'] = float(i)
            
            assembler._update_matrix(feature_store)
        
        assert assembler.n_updates == 60
        assert assembler.is_full is True
        assert assembler.current_index == 10  # 60 % 50
        
        # Check that oldest data is overwritten
        matrix = assembler.get_matrix()
        assert matrix is not None
        assert matrix.shape == (50, 8)
    
    def test_get_matrix_chronological_order(self, assembler, sample_feature_store):
        """Test matrix retrieval in chronological order."""
        # Fill with identifiable data
        for i in range(60):
            feature_store = sample_feature_store.copy()
            feature_store['mlmi_value'] = float(i)
            
            assembler._update_matrix(feature_store)
        
        matrix = assembler.get_matrix()
        
        # Check chronological order (oldest to newest)
        mlmi_idx = assembler.feature_names.index('mlmi_value')
        mlmi_values = matrix[:, mlmi_idx]
        
        # Should contain values 10-59 (last 50 values)
        expected_values = np.arange(10, 60, dtype=np.float32)
        
        # The values will be normalized, so we need to check the pattern
        assert len(mlmi_values) == 50
        assert np.all(np.diff(mlmi_values) >= 0)  # Should be increasing
    
    def test_get_latest_features(self, assembler, sample_feature_store):
        """Test getting latest features."""
        # No updates yet
        assert assembler.get_latest_features() is None
        
        # Add some updates
        for i in range(5):
            feature_store = sample_feature_store.copy()
            feature_store['mlmi_value'] = float(i * 10)
            
            assembler._update_matrix(feature_store)
        
        latest = assembler.get_latest_features()
        assert latest is not None
        assert len(latest) == 8
        
        # Should be the last update (i=4, mlmi_value=40)
        mlmi_idx = assembler.feature_names.index('mlmi_value')
        # The value will be normalized, but should be the highest
        assert latest[mlmi_idx] > 0  # Should be positive after normalization


class TestPerformanceAndMemory:
    """Test performance and memory efficiency."""
    
    def test_update_latency_tracking(self, assembler, sample_feature_store):
        """Test update latency tracking."""
        # Generate some updates
        for i in range(10):
            assembler._update_matrix(sample_feature_store)
        
        # Check latency tracking
        assert len(assembler.update_latencies) == 10
        assert all(latency >= 0 for latency in assembler.update_latencies)
    
    def test_memory_efficiency(self, assembler, sample_feature_store):
        """Test memory efficiency for large number of updates."""
        # Track memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform many updates
        for i in range(1000):
            assembler._update_matrix(sample_feature_store)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 10 MB)
        assert memory_increase < 10 * 1024 * 1024
        
        # Matrix size should remain constant
        assert assembler.matrix.shape == (50, 8)
    
    def test_concurrent_access(self, assembler, sample_feature_store):
        """Test thread safety of concurrent access."""
        results = []
        errors = []
        
        def update_matrix():
            try:
                for i in range(100):
                    assembler._update_matrix(sample_feature_store)
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        def read_matrix():
            try:
                for i in range(100):
                    matrix = assembler.get_matrix()
                    if matrix is not None:
                        assert matrix.shape == (50, 8)
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent threads
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=update_matrix))
            threads.append(threading.Thread(target=read_matrix))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10  # 5 update + 5 read threads
    
    def test_performance_under_load(self, assembler, sample_feature_store):
        """Test performance under continuous load."""
        start_time = time.time()
        
        # Perform many updates
        for i in range(5000):
            assembler._update_matrix(sample_feature_store)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should process at least 1000 updates per second
        updates_per_second = 5000 / total_time
        assert updates_per_second > 1000
        
        # Check average latency
        if assembler.update_latencies:
            avg_latency = np.mean(list(assembler.update_latencies))
            assert avg_latency < 1.0  # Less than 1ms average


class TestStatisticsAndValidation:
    """Test statistics reporting and validation."""
    
    def test_get_statistics(self, assembler, sample_feature_store):
        """Test statistics reporting."""
        # Add some updates
        for i in range(30):
            assembler._update_matrix(sample_feature_store)
        
        stats = assembler.get_statistics()
        
        # Check basic stats
        assert stats['name'] == 'TestAssembler30m'
        assert stats['window_size'] == 50
        assert stats['n_features'] == 8
        assert stats['n_updates'] == 30
        assert stats['is_ready'] is True
        assert stats['features'] == assembler.feature_names
        
        # Check performance stats
        assert 'performance' in stats
        assert 'avg_latency_ms' in stats['performance']
        assert 'max_latency_ms' in stats['performance']
        assert 'p95_latency_ms' in stats['performance']
        
        # Check matrix stats
        assert 'matrix_stats' in stats
        assert stats['matrix_stats']['shape'] == (30, 8)
        assert 'mean' in stats['matrix_stats']
        assert 'std' in stats['matrix_stats']
    
    def test_validate_matrix(self, assembler, sample_feature_store):
        """Test matrix validation."""
        # Add some valid updates
        for i in range(30):
            assembler._update_matrix(sample_feature_store)
        
        is_valid, issues = assembler.validate_matrix()
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_matrix_with_issues(self, assembler):
        """Test matrix validation with issues."""
        # Manually introduce invalid values
        assembler.matrix[0, 0] = np.nan
        assembler.n_updates = 1
        
        is_valid, issues = assembler.validate_matrix()
        
        assert is_valid is False
        assert len(issues) > 0
        assert any("non-finite" in issue.lower() for issue in issues)
    
    def test_reset_functionality(self, assembler, sample_feature_store):
        """Test reset functionality."""
        # Add updates
        for i in range(30):
            assembler._update_matrix(sample_feature_store)
        
        # Reset
        assembler.reset()
        
        # Check reset state
        assert assembler.n_updates == 0
        assert assembler.current_index == 0
        assert assembler.is_full is False
        assert assembler.is_ready() is False
        assert assembler.error_count == 0
        assert len(assembler.update_latencies) == 0
        assert np.all(assembler.matrix == 0)


class TestIntegrationWithDataSources:
    """Test integration with data sources and event system."""
    
    def test_event_subscription(self, assembler):
        """Test event subscription."""
        # Check that assembler subscribed to events
        event_bus = assembler.kernel.get_event_bus()
        event_bus.subscribe.assert_called_once_with(
            EventType.INDICATORS_READY, 
            assembler._on_indicators_ready
        )
    
    def test_indicators_ready_event_handling(self, assembler, sample_feature_store):
        """Test handling of INDICATORS_READY event."""
        # Create event
        event = Event(
            event_type=EventType.INDICATORS_READY,
            payload=sample_feature_store,
            timestamp=datetime.now()
        )
        
        # Handle event
        assembler._on_indicators_ready(event)
        
        # Check that matrix was updated
        assert assembler.n_updates == 1
        assert assembler.current_index == 1
    
    def test_error_handling_in_event_processing(self, assembler):
        """Test error handling in event processing."""
        # Create event with invalid payload
        event = Event(
            event_type=EventType.INDICATORS_READY,
            payload=None,  # Invalid payload
            timestamp=datetime.now()
        )
        
        # Handle event - should not raise exception
        assembler._on_indicators_ready(event)
        
        # Check error was recorded
        assert assembler.error_count > 0
        assert assembler.last_error_time is not None
    
    def test_missing_feature_handling(self, assembler):
        """Test handling of missing features."""
        # Feature store with missing features
        incomplete_feature_store = {
            'mlmi_value': 55.0,
            'current_price': 4145.0
            # Missing other features
        }
        
        # Update with incomplete features
        assembler._update_matrix(incomplete_feature_store)
        
        # Check that missing features are tracked
        assert len(assembler.missing_feature_warnings) > 0
        assert assembler.n_updates == 1  # Should still update with defaults
    
    def test_feature_store_evolution(self, assembler):
        """Test handling of evolving feature stores."""
        # Start with basic features
        basic_store = {
            'mlmi_value': 55.0,
            'mlmi_signal': 1.0,
            'current_price': 4145.0
        }
        
        # Update multiple times
        for i in range(10):
            assembler._update_matrix(basic_store)
        
        # Add more features
        enhanced_store = basic_store.copy()
        enhanced_store.update({
            'nwrqk_value': 4150.0,
            'nwrqk_slope': 0.05,
            'lvn_distance_points': 8.5,
            'lvn_nearest_strength': 75.0
        })
        
        # Continue updating
        for i in range(10):
            assembler._update_matrix(enhanced_store)
        
        # Should handle both scenarios gracefully
        assert assembler.n_updates == 20
        assert assembler.is_ready() is True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_extreme_feature_values(self, assembler):
        """Test handling of extreme feature values."""
        extreme_store = {
            'mlmi_value': 1e10,  # Extremely large
            'mlmi_signal': -1e10,  # Extremely negative
            'nwrqk_value': np.inf,  # Infinity
            'nwrqk_slope': -np.inf,  # Negative infinity
            'lvn_distance_points': np.nan,  # NaN
            'lvn_nearest_strength': 0.0,
            'current_price': 4145.0
        }
        
        # Should handle gracefully
        assembler._update_matrix(extreme_store)
        
        # Check that processed values are within bounds
        latest = assembler.get_latest_features()
        assert latest is not None
        assert np.all(np.isfinite(latest))
        assert np.all(latest >= -3.0)
        assert np.all(latest <= 3.0)
    
    def test_zero_current_price(self, assembler):
        """Test handling of zero current price."""
        zero_price_store = {
            'mlmi_value': 55.0,
            'mlmi_signal': 1.0,
            'current_price': 0.0  # Zero price
        }
        
        # Should handle gracefully
        assembler._update_matrix(zero_price_store)
        
        # Should still update (with default processing)
        assert assembler.n_updates == 1
    
    def test_rapid_successive_updates(self, assembler, sample_feature_store):
        """Test rapid successive updates."""
        # Simulate rapid updates
        for i in range(1000):
            assembler._update_matrix(sample_feature_store)
        
        # Should handle without issues
        assert assembler.n_updates == 1000
        assert assembler.is_ready() is True
        
        # Matrix should be stable
        matrix = assembler.get_matrix()
        assert matrix is not None
        assert matrix.shape == (50, 8)
    
    def test_string_timestamp_parsing(self, assembler):
        """Test parsing of string timestamps."""
        string_timestamp_store = {
            'mlmi_value': 55.0,
            'mlmi_signal': 1.0,
            'current_price': 4145.0,
            'timestamp': '2023-01-01T14:30:00',  # ISO format string
            'time_hour_sin': 14.5,
            'time_hour_cos': 14.5
        }
        
        # Should parse timestamp correctly
        features = assembler.extract_features(string_timestamp_store)
        assert features is not None
        
        # Time features should be processed
        time_sin_idx = assembler.feature_names.index('time_hour_sin')
        time_cos_idx = assembler.feature_names.index('time_hour_cos')
        
        assert features[time_sin_idx] == 14.5
        assert features[time_cos_idx] == 14.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])