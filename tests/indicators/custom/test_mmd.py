"""
Mathematical Validation Tests for MMD (Maximum Mean Discrepancy) Feature Extractor
Tests MMD calculations, regime detection, and statistical properties
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import time

from src.indicators.custom.mmd import (
    MMDFeatureExtractor,
    _compute_dists_sq_numba,
    gaussian_kernel,
    compute_mmd
)
from src.core.minimal_dependencies import EventBus, BarData
from tests.mocks.mock_event_bus import MockEventBus


class TestMMDFeatureExtractor:
    """Test suite for MMD Feature Extractor mathematical validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_event_bus = MockEventBus()
        self.config = {
            'reference_window': 100,
            'test_window': 50,
            'signature_degree': 3,
            'max_history_length': 200
        }
        self.mmd = MMDFeatureExtractor(self.config, self.mock_event_bus)
        
    def create_test_bar(self, close=100.0, high=None, low=None, open=None, volume=1000, timestamp=None):
        """Create a test bar with realistic OHLCV data"""
        if timestamp is None:
            timestamp = datetime.now()
        if high is None:
            high = close + np.random.uniform(0.1, 2.0)
        if low is None:
            low = close - np.random.uniform(0.1, 2.0)
        if open is None:
            open = close + np.random.uniform(-1.0, 1.0)
            
        return BarData(
            symbol="BTCUSDT",
            timestamp=timestamp,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            timeframe=30
        )
    
    def generate_synthetic_regime_data(self, length=200, regime_type='trending'):
        """Generate synthetic data with known regime characteristics"""
        bars = []
        base_price = 100.0
        
        for i in range(length):
            if regime_type == 'trending':
                # Trending regime: consistent direction with low volatility
                price = base_price + (i * 0.1) + np.random.normal(0, 0.2)
                volume = np.random.uniform(800, 1200)
            elif regime_type == 'volatile':
                # Volatile regime: high volatility, no clear trend
                price = base_price + np.random.normal(0, 2.0)
                volume = np.random.uniform(500, 2000)
            elif regime_type == 'ranging':
                # Ranging regime: oscillating around mean
                price = base_price + 5 * np.sin(i * 0.1) + np.random.normal(0, 0.5)
                volume = np.random.uniform(900, 1100)
            else:  # breakout
                # Breakout regime: sudden large move
                if i < length // 2:
                    price = base_price + np.random.normal(0, 0.1)
                else:
                    price = base_price + 10 + np.random.normal(0, 0.1)
                volume = np.random.uniform(1000, 3000)
                
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = self.create_test_bar(
                close=price,
                high=price + np.random.uniform(0.1, 1.0),
                low=price - np.random.uniform(0.1, 1.0),
                volume=int(volume),
                timestamp=timestamp
            )
            bars.append(bar)
            
        return bars
    
    def test_mmd_initialization(self):
        """Test MMD feature extractor initialization"""
        assert self.mmd.reference_window == 100
        assert self.mmd.test_window == 50
        assert self.mmd.reference_data is None
        assert self.mmd.sigma == 1.0
        assert self.mmd.mmd_scores == []
        
    def test_compute_dists_sq_numba(self):
        """Test Numba-accelerated pairwise distance calculation"""
        # Test with simple 2D data
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        dists_sq = _compute_dists_sq_numba(data)
        
        # Should return pairwise squared distances
        expected_pairs = 3 * 2 // 2  # n*(n-1)/2
        assert len(dists_sq) == expected_pairs
        
        # All distances should be non-negative
        assert all(d >= 0 for d in dists_sq)
        
        # Test with single point
        single_point = np.array([[1.0, 2.0]])
        dists_single = _compute_dists_sq_numba(single_point)
        assert len(dists_single) == 0
        
    def test_gaussian_kernel(self):
        """Test Gaussian kernel function"""
        # Test kernel with same points (should be 1.0)
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        kernel_same = gaussian_kernel(x, y, 1.0)
        assert abs(kernel_same - 1.0) < 1e-10
        
        # Test kernel with different points
        y_diff = np.array([2.0, 3.0])
        kernel_diff = gaussian_kernel(x, y_diff, 1.0)
        assert 0.0 < kernel_diff < 1.0
        
        # Test kernel decreases with distance
        y_far = np.array([10.0, 20.0])
        kernel_far = gaussian_kernel(x, y_far, 1.0)
        assert kernel_far < kernel_diff
        
        # Test kernel with different sigma
        kernel_wide = gaussian_kernel(x, y_diff, 2.0)
        kernel_narrow = gaussian_kernel(x, y_diff, 0.5)
        assert kernel_wide > kernel_narrow
        
    def test_compute_mmd(self):
        """Test MMD computation"""
        # Test with identical distributions
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        mmd_identical = compute_mmd(X, Y, 1.0)
        assert mmd_identical < 0.01  # Should be close to 0
        
        # Test with different distributions
        Z = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        mmd_different = compute_mmd(X, Z, 1.0)
        assert mmd_different > mmd_identical
        
        # Test with single points
        single_X = np.array([[1.0, 2.0]])
        single_Y = np.array([[3.0, 4.0]])
        mmd_single = compute_mmd(single_X, single_Y, 1.0)
        assert mmd_single >= 0.0
        
    def test_mmd_feature_extraction_insufficient_data(self):
        """Test MMD feature extraction with insufficient data"""
        # Test with no data
        result = self.mmd.calculate_30m(self.create_test_bar())
        assert result == {'mmd_features': np.zeros(13)}
        
        # Test with insufficient data (< reference_window + test_window)
        for i in range(100):  # Less than 150 required
            bar = self.create_test_bar(close=100.0 + i)
            result = self.mmd.calculate_30m(bar)
            assert np.array_equal(result['mmd_features'], np.zeros(13))
            
    def test_mmd_feature_extraction_sufficient_data(self):
        """Test MMD feature extraction with sufficient data"""
        # Generate sufficient data
        bars = self.generate_synthetic_regime_data(200, 'trending')
        
        # Process bars
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            
        # Final result should be valid
        assert isinstance(result['mmd_features'], np.ndarray)
        assert len(result['mmd_features']) == 13
        assert not np.isnan(result['mmd_features']).any()
        assert not np.isinf(result['mmd_features']).any()
        
    def test_mmd_feature_vector_structure(self):
        """Test MMD feature vector structure"""
        # Generate test data
        bars = self.generate_synthetic_regime_data(200, 'trending')
        
        # Process bars
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            
        features = result['mmd_features']
        
        # Should have 13 features as documented
        assert len(features) == 13
        
        # Features should be reasonable values
        # Features 0-4: OHLCV (should be > 0)
        assert features[0] > 0  # open (close as proxy)
        assert features[1] > 0  # high
        assert features[2] > 0  # low
        assert features[3] > 0  # close
        assert features[4] >= 0  # volume
        
        # Features 5-8: calculated indicators
        # Returns, log_returns, range, volatility can be any value
        assert isinstance(features[5], (int, float))  # returns
        assert isinstance(features[6], (int, float))  # log_returns
        assert isinstance(features[7], (int, float))  # range
        assert isinstance(features[8], (int, float))  # volatility
        
        # Features 9-11: momentum indicators
        assert isinstance(features[9], (int, float))   # momentum_20
        assert isinstance(features[10], (int, float))  # momentum_50
        assert isinstance(features[11], (int, float))  # volume_ratio
        
        # Feature 12: MMD score
        assert features[12] >= 0.0  # MMD should be non-negative
        
    def test_mmd_regime_detection(self):
        """Test MMD regime detection capabilities"""
        # Test with different regime types
        regimes = ['trending', 'volatile', 'ranging', 'breakout']
        regime_scores = []
        
        for regime_type in regimes:
            bars = self.generate_synthetic_regime_data(200, regime_type)
            
            # Reset MMD for each regime
            mmd = MMDFeatureExtractor(self.config, self.mock_event_bus)
            
            # Process bars
            for bar in bars:
                result = mmd.calculate_30m(bar)
                
            # Get final MMD score
            mmd_score = result['mmd_features'][12]
            regime_scores.append(mmd_score)
            
        # Different regimes should potentially give different MMD scores
        assert len(set(regime_scores)) > 1  # At least some variation
        
    def test_mmd_reference_data_initialization(self):
        """Test reference data initialization"""
        # Generate data
        bars = self.generate_synthetic_regime_data(200, 'trending')
        
        # Process bars
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            
        # Reference data should be initialized
        assert self.mmd.reference_data is not None
        assert isinstance(self.mmd.reference_data, np.ndarray)
        assert self.mmd.reference_data.shape[0] == self.mmd.reference_window
        assert self.mmd.reference_data.shape[1] == 4  # returns, log_returns, range, volatility
        
        # Should have reference statistics
        assert hasattr(self.mmd, 'ref_mean')
        assert hasattr(self.mmd, 'ref_std')
        assert isinstance(self.mmd.ref_mean, np.ndarray)
        assert isinstance(self.mmd.ref_std, np.ndarray)
        
    def test_mmd_sigma_estimation(self):
        """Test MMD sigma parameter estimation"""
        # Generate data with known characteristics
        bars = self.generate_synthetic_regime_data(200, 'trending')
        
        # Process bars
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            
        # Sigma should be estimated
        assert self.mmd.sigma > 0.0
        assert self.mmd.sigma < 100.0  # Should be reasonable
        
    def test_mmd_performance_requirements(self):
        """Test MMD performance requirements"""
        # Generate sufficient data
        bars = self.generate_synthetic_regime_data(200)
        
        # Add to history
        for bar in bars[:-1]:
            self.mmd.calculate_30m(bar)
            
        # Time the calculation
        start_time = time.time()
        result = self.mmd.calculate_30m(bars[-1])
        calc_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Should complete within 10ms (more lenient due to MMD complexity)
        assert calc_time < 10.0
        
        # Result should be valid
        assert isinstance(result['mmd_features'], np.ndarray)
        assert len(result['mmd_features']) == 13
        
    def test_mmd_mathematical_properties(self):
        """Test mathematical properties of MMD"""
        # Generate test data
        bars = self.generate_synthetic_regime_data(200, 'trending')
        
        mmd_scores = []
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            if result['mmd_features'][12] > 0:  # Valid MMD score
                mmd_scores.append(result['mmd_features'][12])
                
        # MMD should be non-negative
        assert all(score >= 0.0 for score in mmd_scores)
        
        # MMD should be bounded (typically < 2 for normalized data)
        assert all(score < 10.0 for score in mmd_scores)
        
    def test_mmd_edge_cases(self):
        """Test MMD edge cases"""
        # Test with constant prices
        constant_bars = []
        for i in range(200):
            bar = self.create_test_bar(
                close=100.0,
                high=100.0,
                low=100.0,
                open=100.0,
                volume=1000,
                timestamp=datetime.now() + timedelta(minutes=i*30)
            )
            constant_bars.append(bar)
            
        # Process constant bars
        for bar in constant_bars:
            result = self.mmd.calculate_30m(bar)
            
        # Should handle constant prices
        assert isinstance(result['mmd_features'], np.ndarray)
        assert len(result['mmd_features']) == 13
        
        # Test with extreme volatility
        extreme_bars = []
        for i in range(200):
            price = 100.0 + np.random.normal(0, 50.0)  # Very high volatility
            bar = self.create_test_bar(
                close=price,
                timestamp=datetime.now() + timedelta(minutes=i*30)
            )
            extreme_bars.append(bar)
            
        # Reset for new test
        mmd_extreme = MMDFeatureExtractor(self.config, self.mock_event_bus)
        
        for bar in extreme_bars:
            result = mmd_extreme.calculate_30m(bar)
            
        # Should handle extreme volatility
        assert isinstance(result['mmd_features'], np.ndarray)
        assert len(result['mmd_features']) == 13
        
    def test_mmd_get_current_values(self):
        """Test getting current MMD values"""
        # Initially should return 0
        values = self.mmd.get_current_values()
        assert values == {'mmd_score': 0.0}
        
        # After processing data
        bars = self.generate_synthetic_regime_data(200)
        
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            
        values = self.mmd.get_current_values()
        assert 'mmd_score' in values
        assert isinstance(values['mmd_score'], float)
        
    def test_mmd_reset_functionality(self):
        """Test MMD reset functionality"""
        # Generate some data
        bars = self.generate_synthetic_regime_data(50)
        
        for bar in bars:
            self.mmd.calculate_30m(bar)
            
        # Check that data exists
        assert len(self.mmd.history_30m) > 0
        assert len(self.mmd.mmd_scores) > 0
        
        # Reset
        self.mmd.reset()
        
        # Check that data is cleared
        assert len(self.mmd.history_30m) == 0
        assert len(self.mmd.mmd_scores) == 0
        assert self.mmd.reference_data is None
        
    def test_mmd_nan_handling(self):
        """Test MMD NaN handling"""
        # Generate data with potential NaN values
        bars = self.generate_synthetic_regime_data(200)
        
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            
        # Result should not contain NaN values
        assert not np.isnan(result['mmd_features']).any()
        
    def test_mmd_numerical_stability(self):
        """Test MMD numerical stability"""
        # Test with very small price changes
        bars = []
        base_price = 100.0
        
        for i in range(200):
            price = base_price + (i * 0.0001)  # Very small changes
            bar = self.create_test_bar(
                close=price,
                timestamp=datetime.now() + timedelta(minutes=i*30)
            )
            bars.append(bar)
            
        # Should handle small changes without numerical issues
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            assert not np.isnan(result['mmd_features']).any()
            assert not np.isinf(result['mmd_features']).any()
            
    def test_mmd_memory_efficiency(self):
        """Test MMD memory efficiency"""
        # Generate large dataset
        bars = self.generate_synthetic_regime_data(1000)
        
        # Process all bars
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            
        # Check that memory usage is controlled
        assert len(self.mmd.history_30m) <= self.mmd.max_history_length
        assert len(self.mmd.mmd_scores) <= 100  # Should be limited
        
    def test_mmd_configuration_validation(self):
        """Test MMD configuration validation"""
        # Test with custom configuration
        custom_config = {
            'reference_window': 200,
            'test_window': 100,
            'max_history_length': 500
        }
        
        mmd_custom = MMDFeatureExtractor(custom_config, self.mock_event_bus)
        
        assert mmd_custom.reference_window == 200
        assert mmd_custom.test_window == 100
        assert mmd_custom.max_history_length == 500
        
    def test_mmd_feature_engineering(self):
        """Test MMD feature engineering calculations"""
        # Generate test data
        bars = self.generate_synthetic_regime_data(200)
        
        # Process bars
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            
        # Check that feature engineering was applied
        assert len(self.mmd.history_30m) > 0
        
        # Convert to DataFrame to check calculations
        df = pd.DataFrame([{
            'close': b.close, 'high': b.high, 'low': b.low, 'volume': b.volume
        } for b in self.mmd.history_30m])
        
        # Check that derived features can be calculated
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close'].shift(1)
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252 * 48)
        
        # Should not have all NaN values
        assert not df['returns'].isna().all()
        assert not df['log_returns'].isna().all()
        assert not df['range'].isna().all()
        
    def test_mmd_threading_safety(self):
        """Test MMD thread safety"""
        import threading
        
        bars = self.generate_synthetic_regime_data(100)
        results = []
        
        def calculate_mmd(bar):
            result = self.mmd.calculate_30m(bar)
            results.append(result)
            
        # Create multiple threads
        threads = []
        for bar in bars:
            t = threading.Thread(target=calculate_mmd, args=(bar,))
            threads.append(t)
            
        # Start all threads
        for t in threads:
            t.start()
            
        # Wait for completion
        for t in threads:
            t.join()
            
        # Should have processed all bars
        assert len(results) == len(bars)
        
        # All results should be valid
        for result in results:
            assert isinstance(result['mmd_features'], np.ndarray)
            assert len(result['mmd_features']) == 13
            
    def test_mmd_stress_testing(self):
        """Test MMD under stress conditions"""
        # Generate large dataset with complex patterns
        bars = []
        for i in range(300):
            # Create complex patterns
            if i % 100 < 50:
                regime = 'trending'
            elif i % 100 < 75:
                regime = 'volatile'
            else:
                regime = 'ranging'
                
            if regime == 'trending':
                price = 100.0 + (i * 0.1) + np.random.normal(0, 0.2)
            elif regime == 'volatile':
                price = 100.0 + np.random.normal(0, 5.0)
            else:
                price = 100.0 + 10 * np.sin(i * 0.05) + np.random.normal(0, 0.5)
                
            bar = self.create_test_bar(
                close=price,
                timestamp=datetime.now() + timedelta(minutes=i*30)
            )
            bars.append(bar)
            
        # Process all bars - should not crash
        for bar in bars:
            result = self.mmd.calculate_30m(bar)
            assert isinstance(result, dict)
            assert 'mmd_features' in result
            assert isinstance(result['mmd_features'], np.ndarray)
            assert len(result['mmd_features']) == 13