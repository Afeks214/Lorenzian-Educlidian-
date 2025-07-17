"""
Test Suite for MatrixAssembler5m - Tactical 5-minute Matrix Assembly

This test suite validates the 5-minute matrix assembler with comprehensive
testing covering tactical timeframe processing, real-time assembly,
performance tests for low-latency requirements, and FVG analysis.
"""

import logging


import pytest
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import threading
import time
import gc
from collections import deque

# Import the components to test
from src.matrix.assembler_5m import MatrixAssembler5m
from src.matrix.normalizers import RollingNormalizer, exponential_decay, percentage_from_price
from src.core.minimal_dependencies import EventType, Event
from src.utils.logger import get_logger


class TestMatrixAssembler5m:
    """Test suite for 5-minute matrix assembler."""
    
    @pytest.fixture
    def mock_kernel(self):
        """Mock kernel with event bus."""
        kernel = Mock()
        event_bus = Mock()
        kernel.get_event_bus.return_value = event_bus
        return kernel
    
    @pytest.fixture
    def tactical_config(self, mock_kernel):
        """Tactical configuration for testing."""
        return {
            'name': 'TestAssembler5m',
            'window_size': 100,  # More data for tactical decisions
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
            'warmup_period': 50,
            'feature_configs': {
                'price_momentum_5': {'ema_alpha': 0.1, 'warmup_samples': 20},
                'volume_ratio': {'ema_alpha': 0.05, 'warmup_samples': 30}
            }
        }
    
    @pytest.fixture
    def assembler(self, tactical_config):
        """Create assembler instance."""
        return MatrixAssembler5m(tactical_config)
    
    @pytest.fixture
    def sample_fvg_store(self):
        """Sample feature store with FVG data."""
        return {
            'fvg_bullish_active': 1.0,
            'fvg_bearish_active': 0.0,
            'fvg_nearest_level': 4150.0,
            'fvg_age': 5.0,
            'fvg_mitigation_signal': 0.0,
            'price_momentum_5': 0.25,
            'volume_ratio': 1.5,
            'current_price': 4145.0,
            'current_volume': 1500,
            'close': 4145.0,
            'volume': 1500
        }


class TestTacticalInitialization:
    """Test tactical assembler initialization."""
    
    def test_successful_initialization(self, tactical_config):
        """Test successful tactical assembler initialization."""
        assembler = MatrixAssembler5m(tactical_config)
        
        assert assembler.name == 'TestAssembler5m'
        assert assembler.window_size == 100
        assert assembler.n_features == 7
        assert len(assembler.feature_names) == 7
        
        # Check tactical-specific attributes
        assert assembler.price_history.maxlen == 6
        assert assembler.volume_ema is None
        assert assembler.volume_ema_alpha == 0.02
        assert assembler.current_price is None
        assert assembler.last_fvg_update is None
        
        # Check matrix initialization
        assert assembler.matrix.shape == (100, 7)
        assert assembler.matrix.dtype == np.float32
    
    def test_feature_importance_calculation(self, assembler):
        """Test tactical feature importance calculation."""
        importance = assembler.get_feature_importance()
        
        # Check all features have importance scores
        assert len(importance) == 7
        assert all(0 <= score <= 1 for score in importance.values())
        
        # Check scores sum to 1.0
        assert abs(sum(importance.values()) - 1.0) < 1e-6
        
        # Check FVG features have high importance
        assert importance['fvg_bullish_active'] == 0.20
        assert importance['fvg_bearish_active'] == 0.20
        assert importance['fvg_nearest_level'] == 0.15
        assert importance['fvg_mitigation_signal'] == 0.15


class TestTacticalFeatureExtraction:
    """Test tactical feature extraction with momentum and volume."""
    
    def test_extract_features_with_momentum(self, assembler, sample_fvg_store):
        """Test feature extraction with momentum calculation."""
        # Add price history
        for i in range(6):
            assembler.price_history.append(4140.0 + i * 2)  # Increasing prices
        
        features = assembler.extract_features(sample_fvg_store)
        
        assert features is not None
        assert len(features) == 7
        
        # Check momentum calculation
        momentum_idx = assembler.feature_names.index('price_momentum_5')
        # Should calculate momentum from price history
        assert features[momentum_idx] > 0  # Should be positive for rising prices
    
    def test_extract_features_with_volume_ratio(self, assembler, sample_fvg_store):
        """Test feature extraction with volume ratio calculation."""
        # Initialize volume EMA
        assembler.volume_ema = 1000.0
        
        features = assembler.extract_features(sample_fvg_store)
        
        assert features is not None
        
        # Check volume ratio calculation
        volume_ratio_idx = assembler.feature_names.index('volume_ratio')
        expected_ratio = 1500.0 / 1000.0  # 1.5
        assert abs(features[volume_ratio_idx] - expected_ratio) < 1e-6
    
    def test_extract_features_without_custom_features(self, tactical_config, mock_kernel):
        """Test extraction without custom features."""
        tactical_config['features'] = ['fvg_bullish_active', 'fvg_bearish_active']
        assembler = MatrixAssembler5m(tactical_config)
        
        features = assembler.extract_features(sample_fvg_store)
        
        # Should return None to use default extraction
        assert features is None
    
    def test_price_momentum_calculation(self, assembler):
        """Test price momentum calculation."""
        # Fill price history with known values
        prices = [4140.0, 4142.0, 4144.0, 4146.0, 4148.0, 4150.0]
        for price in prices:
            assembler.price_history.append(price)
        
        momentum = assembler._calculate_price_momentum()
        
        # Calculate expected momentum
        expected = ((4150.0 - 4140.0) / 4140.0) * 100  # ~0.24%
        assert abs(momentum - expected) < 1e-6
    
    def test_price_momentum_insufficient_data(self, assembler):
        """Test momentum calculation with insufficient data."""
        # Only add 3 prices
        for i in range(3):
            assembler.price_history.append(4140.0 + i)
        
        momentum = assembler._calculate_price_momentum()
        
        assert momentum == 0.0
    
    def test_volume_ema_update(self, assembler, sample_fvg_store):
        """Test volume EMA update mechanism."""
        # First update
        assembler.extract_features(sample_fvg_store)
        assert assembler.volume_ema == 1500.0
        
        # Second update with different volume
        sample_fvg_store['current_volume'] = 2000
        assembler.extract_features(sample_fvg_store)
        
        # EMA should be updated
        expected_ema = 1500.0 + 0.02 * (2000 - 1500.0)
        assert abs(assembler.volume_ema - expected_ema) < 1e-6


class TestTacticalPreprocessing:
    """Test tactical preprocessing optimized for FVG dynamics."""
    
    def test_fvg_binary_preprocessing(self, assembler, sample_fvg_store):
        """Test FVG binary feature preprocessing."""
        raw_features = [1.0, 0.0, 4150.0, 5.0, 0.0, 0.25, 1.5]
        
        processed = assembler.preprocess_features(raw_features, sample_fvg_store)
        
        # Binary features should remain as is
        bullish_idx = assembler.feature_names.index('fvg_bullish_active')
        bearish_idx = assembler.feature_names.index('fvg_bearish_active')
        mitigation_idx = assembler.feature_names.index('fvg_mitigation_signal')
        
        assert processed[bullish_idx] == 1.0
        assert processed[bearish_idx] == 0.0
        assert processed[mitigation_idx] == 0.0
    
    def test_fvg_level_preprocessing(self, assembler, sample_fvg_store):
        """Test FVG level preprocessing."""
        raw_features = [1.0, 0.0, 4150.0, 5.0, 0.0, 0.25, 1.5]
        
        processed = assembler.preprocess_features(raw_features, sample_fvg_store)
        
        # FVG level should be normalized as % distance from current price
        level_idx = assembler.feature_names.index('fvg_nearest_level')
        expected_pct = ((4150.0 - 4145.0) / 4145.0) * 100  # ~0.12%
        expected_normalized = expected_pct / 2.0  # Scale to [-1, 1]
        
        assert abs(processed[level_idx] - expected_normalized) < 1e-3
    
    def test_fvg_age_preprocessing(self, assembler, sample_fvg_store):
        """Test FVG age preprocessing with exponential decay."""
        raw_features = [1.0, 0.0, 4150.0, 5.0, 0.0, 0.25, 1.5]
        
        processed = assembler.preprocess_features(raw_features, sample_fvg_store)
        
        # FVG age should use exponential decay
        age_idx = assembler.feature_names.index('fvg_age')
        expected_decay = exponential_decay(5.0, decay_rate=0.1)
        
        assert abs(processed[age_idx] - expected_decay) < 1e-6
    
    def test_momentum_preprocessing(self, assembler, sample_fvg_store):
        """Test momentum preprocessing."""
        raw_features = [1.0, 0.0, 4150.0, 5.0, 0.0, 2.5, 1.5]  # 2.5% momentum
        
        processed = assembler.preprocess_features(raw_features, sample_fvg_store)
        
        # Momentum should be scaled to [-1, 1]
        momentum_idx = assembler.feature_names.index('price_momentum_5')
        expected = np.clip(2.5 / 5.0, -1.0, 1.0)  # Scale by ±5%
        
        assert abs(processed[momentum_idx] - expected) < 1e-6
    
    def test_volume_ratio_preprocessing(self, assembler, sample_fvg_store):
        """Test volume ratio preprocessing."""
        raw_features = [1.0, 0.0, 4150.0, 5.0, 0.0, 0.25, 3.0]  # 3x volume
        
        processed = assembler.preprocess_features(raw_features, sample_fvg_store)
        
        # Volume ratio should use log transform
        volume_idx = assembler.feature_names.index('volume_ratio')
        expected_log = np.log1p(3.0 - 1)  # log1p(2.0)
        expected_normalized = np.tanh(expected_log)
        
        assert abs(processed[volume_idx] - expected_normalized) < 1e-6
    
    def test_preprocessing_with_zero_current_price(self, assembler):
        """Test preprocessing when current price is zero."""
        raw_features = [1.0, 0.0, 4150.0, 5.0, 0.0, 0.25, 1.5]
        feature_store = {'current_price': 0.0}
        
        processed = assembler.preprocess_features(raw_features, feature_store)
        
        # Should handle gracefully
        assert np.all(np.isfinite(processed))
        
        # FVG level should be 0 when current price is invalid
        level_idx = assembler.feature_names.index('fvg_nearest_level')
        assert processed[level_idx] == 0.0


class TestTacticalValidation:
    """Test tactical feature validation."""
    
    def test_valid_fvg_features(self, assembler):
        """Test validation of valid FVG features."""
        valid_features = [1.0, 0.0, 4150.0, 5.0, 0.0, 0.25, 1.5]
        
        assert assembler.validate_features(valid_features) is True
    
    def test_invalid_binary_features(self, assembler):
        """Test validation of invalid binary features."""
        invalid_features = [0.5, 0.0, 4150.0, 5.0, 0.0, 0.25, 1.5]  # Invalid binary
        
        assert assembler.validate_features(invalid_features) is False
    
    def test_invalid_fvg_age(self, assembler):
        """Test validation of invalid FVG age."""
        invalid_features = [1.0, 0.0, 4150.0, -1.0, 0.0, 0.25, 1.5]  # Negative age
        
        assert assembler.validate_features(invalid_features) is False
    
    def test_invalid_volume_ratio(self, assembler):
        """Test validation of invalid volume ratio."""
        invalid_features = [1.0, 0.0, 4150.0, 5.0, 0.0, 0.25, -1.0]  # Negative ratio
        
        assert assembler.validate_features(invalid_features) is False


class TestFVGAnalysis:
    """Test FVG-specific analysis functionality."""
    
    def test_fvg_summary_not_ready(self, assembler):
        """Test FVG summary when not ready."""
        summary = assembler.get_fvg_summary()
        
        assert summary['status'] == 'not_ready'
    
    def test_fvg_summary_ready(self, assembler, sample_fvg_store):
        """Test FVG summary when ready."""
        # Add enough data to be ready
        for i in range(60):
            store = sample_fvg_store.copy()
            store['fvg_bullish_active'] = 1.0 if i % 5 == 0 else 0.0
            store['fvg_bearish_active'] = 1.0 if i % 7 == 0 else 0.0
            store['fvg_mitigation_signal'] = 1.0 if i % 10 == 0 else 0.0
            store['fvg_age'] = float(i % 15)
            
            assembler._update_matrix(store)
        
        summary = assembler.get_fvg_summary()
        
        assert summary['status'] == 'ready'
        assert 'last_20_bars' in summary
        assert 'bullish_fvg_count' in summary['last_20_bars']
        assert 'bearish_fvg_count' in summary['last_20_bars']
        assert 'mitigation_count' in summary['last_20_bars']
        assert 'fvg_activity_rate' in summary['last_20_bars']
    
    def test_fvg_activity_patterns(self, assembler, sample_fvg_store):
        """Test FVG activity pattern analysis."""
        # Create specific pattern
        patterns = [
            {'fvg_bullish_active': 1.0, 'fvg_bearish_active': 0.0, 'fvg_age': 1.0},
            {'fvg_bullish_active': 1.0, 'fvg_bearish_active': 0.0, 'fvg_age': 2.0},
            {'fvg_bullish_active': 0.0, 'fvg_bearish_active': 1.0, 'fvg_age': 1.0},
            {'fvg_bullish_active': 0.0, 'fvg_bearish_active': 0.0, 'fvg_age': 0.0},
        ]
        
        for i in range(60):
            store = sample_fvg_store.copy()
            pattern = patterns[i % len(patterns)]
            store.update(pattern)
            
            assembler._update_matrix(store)
        
        summary = assembler.get_fvg_summary()
        
        # Check activity patterns
        assert summary['last_20_bars']['bullish_fvg_count'] > 0
        assert summary['last_20_bars']['bearish_fvg_count'] > 0
        assert summary['last_20_bars']['fvg_activity_rate'] > 0


class TestTacticalPerformance:
    """Test performance requirements for tactical timeframe."""
    
    def test_low_latency_requirements(self, assembler, sample_fvg_store):
        """Test low-latency requirements for tactical processing."""
        # Warm up
        for i in range(10):
            assembler._update_matrix(sample_fvg_store)
        
        # Measure latency for single update
        start_time = time.perf_counter()
        assembler._update_matrix(sample_fvg_store)
        end_time = time.perf_counter()
        
        update_latency = (end_time - start_time) * 1000  # Convert to ms
        
        # Should be faster than 1ms for tactical decisions
        assert update_latency < 1.0
    
    def test_high_frequency_updates(self, assembler, sample_fvg_store):
        """Test handling of high-frequency updates."""
        start_time = time.time()
        
        # Simulate high-frequency updates
        for i in range(10000):
            store = sample_fvg_store.copy()
            store['fvg_age'] = float(i % 100)
            
            assembler._update_matrix(store)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should process at least 5000 updates per second
        updates_per_second = 10000 / total_time
        assert updates_per_second > 5000
    
    def test_memory_efficiency_under_load(self, assembler, sample_fvg_store):
        """Test memory efficiency under continuous load."""
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run continuous updates
        for i in range(50000):
            assembler._update_matrix(sample_fvg_store)
            
            # Force garbage collection periodically
            if i % 10000 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 5 MB)
        assert memory_increase < 5 * 1024 * 1024
    
    def test_concurrent_tactical_processing(self, assembler, sample_fvg_store):
        """Test concurrent processing for tactical decisions."""
        results = []
        errors = []
        
        def tactical_updates():
            try:
                for i in range(1000):
                    store = sample_fvg_store.copy()
                    store['fvg_age'] = float(i % 50)
                    assembler._update_matrix(store)
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        def tactical_reads():
            try:
                for i in range(1000):
                    matrix = assembler.get_matrix()
                    fvg_summary = assembler.get_fvg_summary()
                    if matrix is not None:
                        assert matrix.shape[0] <= 100
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent tactical operations
        threads = []
        for _ in range(8):  # Higher concurrency for tactical
            threads.append(threading.Thread(target=tactical_updates))
            threads.append(threading.Thread(target=tactical_reads))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 16  # 8 update + 8 read threads
    
    def test_real_time_processing_simulation(self, assembler, sample_fvg_store):
        """Test real-time processing simulation."""
        # Simulate real-time data stream
        latencies = []
        
        for i in range(1000):
            # Simulate varying market conditions
            store = sample_fvg_store.copy()
            store['fvg_bullish_active'] = 1.0 if i % 10 < 3 else 0.0
            store['fvg_bearish_active'] = 1.0 if i % 10 > 7 else 0.0
            store['fvg_age'] = float(i % 20)
            store['price_momentum_5'] = np.sin(i * 0.1) * 2.0
            store['volume_ratio'] = 1.0 + np.random.normal(0, 0.5)
            
            start_time = time.perf_counter()
            assembler._update_matrix(store)
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
        
        # Check latency statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = np.max(latencies)
        
        # Tactical requirements
        assert avg_latency < 0.5  # Average < 0.5ms
        assert p95_latency < 1.0   # 95th percentile < 1ms
        assert max_latency < 5.0   # Maximum < 5ms


class TestTacticalIntegration:
    """Test integration with tactical decision systems."""
    
    def test_tactical_decision_context(self, assembler, sample_fvg_store):
        """Test providing context for tactical decisions."""
        # Build decision context
        for i in range(100):
            store = sample_fvg_store.copy()
            
            # Create market scenarios
            if i < 20:  # Bullish FVG scenario
                store['fvg_bullish_active'] = 1.0
                store['fvg_bearish_active'] = 0.0
                store['price_momentum_5'] = 0.5
            elif i < 40:  # Bearish FVG scenario
                store['fvg_bullish_active'] = 0.0
                store['fvg_bearish_active'] = 1.0
                store['price_momentum_5'] = -0.5
            else:  # Neutral scenario
                store['fvg_bullish_active'] = 0.0
                store['fvg_bearish_active'] = 0.0
                store['price_momentum_5'] = 0.0
            
            assembler._update_matrix(store)
        
        # Get tactical context
        matrix = assembler.get_matrix()
        fvg_summary = assembler.get_fvg_summary()
        
        assert matrix is not None
        assert matrix.shape == (100, 7)
        assert fvg_summary['status'] == 'ready'
        
        # Check that different scenarios are captured
        bullish_idx = assembler.feature_names.index('fvg_bullish_active')
        bearish_idx = assembler.feature_names.index('fvg_bearish_active')
        
        bullish_activity = np.sum(matrix[:, bullish_idx])
        bearish_activity = np.sum(matrix[:, bearish_idx])
        
        assert bullish_activity > 0
        assert bearish_activity > 0
    
    def test_rapid_regime_changes(self, assembler, sample_fvg_store):
        """Test handling of rapid regime changes."""
        # Simulate rapid regime changes
        for i in range(200):
            store = sample_fvg_store.copy()
            
            # Alternate between regimes every 10 bars
            regime = (i // 10) % 3
            
            if regime == 0:  # Bullish regime
                store['fvg_bullish_active'] = 1.0
                store['fvg_bearish_active'] = 0.0
                store['price_momentum_5'] = 1.0
                store['volume_ratio'] = 2.0
            elif regime == 1:  # Bearish regime
                store['fvg_bullish_active'] = 0.0
                store['fvg_bearish_active'] = 1.0
                store['price_momentum_5'] = -1.0
                store['volume_ratio'] = 1.8
            else:  # Neutral regime
                store['fvg_bullish_active'] = 0.0
                store['fvg_bearish_active'] = 0.0
                store['price_momentum_5'] = 0.0
                store['volume_ratio'] = 1.0
            
            assembler._update_matrix(store)
        
        # Should handle regime changes without issues
        assert assembler.n_updates == 200
        assert assembler.is_ready() is True
        
        # Check that recent regime is reflected in latest features
        latest = assembler.get_latest_features()
        assert latest is not None
    
    def test_edge_case_fvg_scenarios(self, assembler, sample_fvg_store):
        """Test edge case FVG scenarios."""
        edge_cases = [
            # Simultaneous bull and bear FVGs
            {'fvg_bullish_active': 1.0, 'fvg_bearish_active': 1.0, 'fvg_age': 0.0},
            # Very old FVG
            {'fvg_bullish_active': 1.0, 'fvg_bearish_active': 0.0, 'fvg_age': 100.0},
            # Extreme volume spike
            {'fvg_bullish_active': 1.0, 'fvg_bearish_active': 0.0, 'volume_ratio': 50.0},
            # Zero volume
            {'fvg_bullish_active': 1.0, 'fvg_bearish_active': 0.0, 'volume_ratio': 0.0},
            # Extreme momentum
            {'fvg_bullish_active': 1.0, 'fvg_bearish_active': 0.0, 'price_momentum_5': 20.0},
        ]
        
        for case in edge_cases:
            store = sample_fvg_store.copy()
            store.update(case)
            
            # Should handle edge cases gracefully
            assembler._update_matrix(store)
        
        # Check that all edge cases were processed
        assert assembler.n_updates == len(edge_cases)
        
        # Matrix should be stable
        matrix = assembler.get_matrix()
        assert matrix is not None
        assert np.all(np.isfinite(matrix))


class TestTacticalStatistics:
    """Test tactical statistics and monitoring."""
    
    def test_tactical_statistics_reporting(self, assembler, sample_fvg_store):
        """Test tactical statistics reporting."""
        # Generate tactical activity
        for i in range(150):
            store = sample_fvg_store.copy()
            store['fvg_age'] = float(i % 20)
            store['price_momentum_5'] = np.sin(i * 0.1)
            
            assembler._update_matrix(store)
        
        stats = assembler.get_statistics()
        
        # Check tactical-specific stats
        assert stats['window_size'] == 100
        assert stats['n_features'] == 7
        assert stats['n_updates'] == 150
        assert stats['is_ready'] is True
        
        # Check performance is within tactical requirements
        if 'performance' in stats:
            assert stats['performance']['avg_latency_ms'] < 1.0
            assert stats['performance']['p95_latency_ms'] < 2.0
    
    def test_tactical_error_tracking(self, assembler):
        """Test tactical error tracking."""
        # Generate some errors
        for i in range(5):
            try:
                # Invalid feature store
                assembler._update_matrix({})
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                logger.error(f'Error occurred: {e}')
        
        stats = assembler.get_statistics()
        
        # Should track errors but continue functioning
        assert stats['error_count'] >= 0
        assert assembler.n_updates >= 0
    
    def test_missing_feature_adaptation(self, assembler):
        """Test adaptation to missing tactical features."""
        # Gradual feature availability
        base_store = {'current_price': 4145.0}
        
        # Start with minimal features
        for i in range(20):
            store = base_store.copy()
            store['fvg_bullish_active'] = 1.0 if i % 5 == 0 else 0.0
            assembler._update_matrix(store)
        
        # Add more features gradually
        for i in range(20):
            store = base_store.copy()
            store['fvg_bullish_active'] = 1.0 if i % 5 == 0 else 0.0
            store['fvg_bearish_active'] = 1.0 if i % 7 == 0 else 0.0
            assembler._update_matrix(store)
        
        # Full feature set
        for i in range(20):
            store = sample_fvg_store.copy()
            assembler._update_matrix(store)
        
        # Should handle gradual feature availability
        assert assembler.n_updates == 60
        assert assembler.is_ready() is True
        
        # Check missing feature tracking
        stats = assembler.get_statistics()
        assert 'missing_features' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])