"""
Test Suite for Matrix Normalizers - Statistical Validation and Numerical Stability

This test suite validates the normalization algorithms with comprehensive
testing covering statistical properties, numerical stability, edge cases,
and performance under extreme conditions.
"""

import logging


import pytest
import numpy as np
from typing import Union, Tuple
import warnings
from unittest.mock import patch

# Import the normalizers to test
from src.matrix.normalizers import (
    z_score_normalize,
    min_max_scale,
    cyclical_encode,
    percentage_from_price,
    exponential_decay,
    log_transform,
    robust_percentile_scale,
    safe_divide,
    RollingNormalizer
)


class TestZScoreNormalization:
    """Test z-score normalization functionality."""
    
    def test_basic_z_score_normalization(self):
        """Test basic z-score normalization."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = 3.0
        std = np.sqrt(2.0)  # Standard deviation
        
        normalized = z_score_normalize(values, mean, std)
        
        # Check that mean of normalized values is approximately 0
        assert abs(np.mean(normalized)) < 1e-10
        
        # Check specific values
        expected = (values - mean) / std
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_z_score_with_clipping(self):
        """Test z-score normalization with clipping."""
        values = np.array([-10.0, 0.0, 10.0])
        mean = 0.0
        std = 1.0
        
        normalized = z_score_normalize(values, mean, std, clip_range=(-2.0, 2.0))
        
        # Values should be clipped to [-2, 2]
        assert np.all(normalized >= -2.0)
        assert np.all(normalized <= 2.0)
        assert normalized[0] == -2.0  # -10 should be clipped to -2
        assert normalized[2] == 2.0   # 10 should be clipped to 2
    
    def test_z_score_with_zero_std(self):
        """Test z-score normalization with zero standard deviation."""
        values = np.array([5.0, 5.0, 5.0])
        mean = 5.0
        std = 0.0
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized = z_score_normalize(values, mean, std)
            
            # Should issue warning
            assert len(w) == 1
            assert "Standard deviation is zero" in str(w[0].message)
        
        # Should return zeros
        np.testing.assert_array_equal(normalized, np.zeros_like(values))
    
    def test_z_score_single_value(self):
        """Test z-score normalization with single value."""
        value = 5.0
        mean = 3.0
        std = 2.0
        
        normalized = z_score_normalize(value, mean, std)
        
        expected = (5.0 - 3.0) / 2.0
        assert normalized == expected
    
    def test_z_score_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small std
        values = np.array([1.0, 1.0000001, 1.0000002])
        mean = 1.0000001
        std = 1e-10
        
        normalized = z_score_normalize(values, mean, std)
        
        # Should not produce inf or nan
        assert np.all(np.isfinite(normalized))


class TestMinMaxScaling:
    """Test min-max scaling functionality."""
    
    def test_basic_min_max_scaling(self):
        """Test basic min-max scaling."""
        values = np.array([0.0, 5.0, 10.0])
        min_val = 0.0
        max_val = 10.0
        
        scaled = min_max_scale(values, min_val, max_val, target_range=(-1.0, 1.0))
        
        # Check scaling
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(scaled, expected)
    
    def test_min_max_with_different_target_range(self):
        """Test min-max scaling with different target range."""
        values = np.array([0.0, 5.0, 10.0])
        min_val = 0.0
        max_val = 10.0
        
        scaled = min_max_scale(values, min_val, max_val, target_range=(0.0, 1.0))
        
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(scaled, expected)
    
    def test_min_max_with_equal_bounds(self):
        """Test min-max scaling with equal min and max values."""
        values = np.array([5.0, 5.0, 5.0])
        min_val = 5.0
        max_val = 5.0
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scaled = min_max_scale(values, min_val, max_val, target_range=(-1.0, 1.0))
            
            # Should issue warning
            assert len(w) == 1
            assert "Min and max values are equal" in str(w[0].message)
        
        # Should return midpoint of target range
        expected = np.array([0.0, 0.0, 0.0])  # Midpoint of [-1, 1]
        np.testing.assert_array_almost_equal(scaled, expected)
    
    def test_min_max_single_value(self):
        """Test min-max scaling with single value."""
        value = 7.5
        min_val = 5.0
        max_val = 10.0
        
        scaled = min_max_scale(value, min_val, max_val, target_range=(-1.0, 1.0))
        
        expected = ((7.5 - 5.0) / (10.0 - 5.0)) * 2.0 - 1.0  # Scale to [-1, 1]
        assert abs(scaled - expected) < 1e-10
    
    def test_min_max_clipping(self):
        """Test min-max scaling with values outside bounds."""
        values = np.array([-5.0, 5.0, 15.0])
        min_val = 0.0
        max_val = 10.0
        
        scaled = min_max_scale(values, min_val, max_val, target_range=(-1.0, 1.0))
        
        # Values should be clipped to target range
        assert np.all(scaled >= -1.0)
        assert np.all(scaled <= 1.0)
        assert scaled[0] == -1.0  # -5 maps to < -1, should be clipped
        assert scaled[2] == 1.0   # 15 maps to > 1, should be clipped


class TestCyclicalEncoding:
    """Test cyclical encoding functionality."""
    
    def test_basic_cyclical_encoding(self):
        """Test basic cyclical encoding."""
        values = np.array([0.0, 6.0, 12.0, 18.0])
        max_value = 24.0  # 24-hour cycle
        
        sin_encoded, cos_encoded = cyclical_encode(values, max_value)
        
        # Check specific values
        expected_sin = np.sin(2 * np.pi * values / max_value)
        expected_cos = np.cos(2 * np.pi * values / max_value)
        
        np.testing.assert_array_almost_equal(sin_encoded, expected_sin)
        np.testing.assert_array_almost_equal(cos_encoded, expected_cos)
    
    def test_cyclical_encoding_single_value(self):
        """Test cyclical encoding with single value."""
        value = 12.0
        max_value = 24.0
        
        sin_encoded, cos_encoded = cyclical_encode(value, max_value)
        
        expected_angle = 2 * np.pi * 12.0 / 24.0  # π
        expected_sin = np.sin(expected_angle)  # 0
        expected_cos = np.cos(expected_angle)  # -1
        
        assert abs(sin_encoded - expected_sin) < 1e-10
        assert abs(cos_encoded - expected_cos) < 1e-10
    
    def test_cyclical_encoding_full_cycle(self):
        """Test cyclical encoding for full cycle."""
        values = np.array([0.0, 12.0, 24.0])
        max_value = 24.0
        
        sin_encoded, cos_encoded = cyclical_encode(values, max_value)
        
        # 0 and 24 should be equivalent (both at start of cycle)
        assert abs(sin_encoded[0] - sin_encoded[2]) < 1e-10
        assert abs(cos_encoded[0] - cos_encoded[2]) < 1e-10
    
    def test_cyclical_encoding_invalid_max_value(self):
        """Test cyclical encoding with invalid max value."""
        with pytest.raises(ValueError, match="max_value must be positive"):
            cyclical_encode(12.0, 0.0)
        
        with pytest.raises(ValueError, match="max_value must be positive"):
            cyclical_encode(12.0, -24.0)
    
    def test_cyclical_encoding_preserves_periodicity(self):
        """Test that cyclical encoding preserves periodicity."""
        values = np.linspace(0, 48, 100)  # Two full cycles
        max_value = 24.0
        
        sin_encoded, cos_encoded = cyclical_encode(values, max_value)
        
        # Check that values separated by max_value are identical
        for i in range(50):
            v1 = values[i]
            v2 = values[i + 50]  # One full cycle later
            
            if abs(v2 - v1 - max_value) < 1e-10:
                assert abs(sin_encoded[i] - sin_encoded[i + 50]) < 1e-10
                assert abs(cos_encoded[i] - cos_encoded[i + 50]) < 1e-10


class TestPercentageFromPrice:
    """Test percentage from price functionality."""
    
    def test_basic_percentage_calculation(self):
        """Test basic percentage calculation."""
        values = np.array([100.0, 105.0, 95.0])
        reference_price = 100.0
        
        percentages = percentage_from_price(values, reference_price)
        
        expected = np.array([0.0, 5.0, -5.0])
        np.testing.assert_array_almost_equal(percentages, expected)
    
    def test_percentage_with_clipping(self):
        """Test percentage calculation with clipping."""
        values = np.array([80.0, 100.0, 130.0])
        reference_price = 100.0
        clip_pct = 15.0
        
        percentages = percentage_from_price(values, reference_price, clip_pct)
        
        # Should clip extreme values
        assert percentages[0] == -15.0  # -20% clipped to -15%
        assert percentages[1] == 0.0    # 0% unchanged
        assert percentages[2] == 15.0   # +30% clipped to +15%
    
    def test_percentage_single_value(self):
        """Test percentage calculation with single value."""
        value = 110.0
        reference_price = 100.0
        
        percentage = percentage_from_price(value, reference_price)
        
        expected = 10.0  # 10% increase
        assert abs(percentage - expected) < 1e-10
    
    def test_percentage_invalid_reference_price(self):
        """Test percentage calculation with invalid reference price."""
        with pytest.raises(ValueError, match="reference_price must be positive"):
            percentage_from_price(110.0, 0.0)
        
        with pytest.raises(ValueError, match="reference_price must be positive"):
            percentage_from_price(110.0, -100.0)
    
    def test_percentage_extreme_values(self):
        """Test percentage calculation with extreme values."""
        values = np.array([0.0, 1e-10, 1e10])
        reference_price = 100.0
        
        percentages = percentage_from_price(values, reference_price, clip_pct=200.0)
        
        # Should handle extreme values
        assert percentages[0] == -100.0  # 0 is -100% of reference
        assert percentages[1] < 0        # Very small positive is negative percentage
        assert percentages[2] == 200.0   # Large value clipped to 200%


class TestExponentialDecay:
    """Test exponential decay functionality."""
    
    def test_basic_exponential_decay(self):
        """Test basic exponential decay."""
        values = np.array([0.0, 1.0, 2.0, 5.0, 10.0])
        decay_rate = 0.1
        
        decayed = exponential_decay(values, decay_rate)
        
        # Check specific values
        expected = np.exp(-decay_rate * values)
        np.testing.assert_array_almost_equal(decayed, expected)
    
    def test_exponential_decay_single_value(self):
        """Test exponential decay with single value."""
        value = 5.0
        decay_rate = 0.2
        
        decayed = exponential_decay(value, decay_rate)
        
        expected = np.exp(-0.2 * 5.0)
        assert abs(decayed - expected) < 1e-10
    
    def test_exponential_decay_negative_values(self):
        """Test exponential decay with negative values."""
        values = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        decay_rate = 0.1
        
        decayed = exponential_decay(values, decay_rate)
        
        # Should use absolute value
        expected = np.exp(-decay_rate * np.abs(values))
        np.testing.assert_array_almost_equal(decayed, expected)
    
    def test_exponential_decay_invalid_decay_rate(self):
        """Test exponential decay with invalid decay rate."""
        with pytest.raises(ValueError, match="decay_rate must be positive"):
            exponential_decay(5.0, 0.0)
        
        with pytest.raises(ValueError, match="decay_rate must be positive"):
            exponential_decay(5.0, -0.1)
    
    def test_exponential_decay_properties(self):
        """Test exponential decay properties."""
        values = np.array([0.0, 1.0, 2.0, 3.0])
        decay_rate = 0.1
        
        decayed = exponential_decay(values, decay_rate)
        
        # All values should be in (0, 1] range
        assert np.all(decayed > 0)
        assert np.all(decayed <= 1)
        
        # Value at 0 should be 1
        assert decayed[0] == 1.0
        
        # Should be monotonically decreasing
        assert np.all(np.diff(decayed) <= 0)


class TestLogTransform:
    """Test logarithmic transformation functionality."""
    
    def test_basic_log_transform(self):
        """Test basic logarithmic transformation."""
        values = np.array([0.0, 1.0, 2.0, 10.0])
        
        transformed = log_transform(values)
        
        # Check that transformation is applied
        assert np.all(np.isfinite(transformed))
        assert transformed[0] >= 0  # log1p(epsilon - 1) should be valid
    
    def test_log_transform_with_epsilon(self):
        """Test log transform with custom epsilon."""
        values = np.array([0.0, 1e-10, 1.0])
        epsilon = 1e-6
        
        transformed = log_transform(values, epsilon)
        
        # Should handle very small values
        assert np.all(np.isfinite(transformed))
        assert transformed[0] >= 0
    
    def test_log_transform_negative_values(self):
        """Test log transform with negative values."""
        values = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        
        transformed = log_transform(values)
        
        # Should convert negative values to small positive
        assert np.all(np.isfinite(transformed))
        assert np.all(transformed >= 0)
    
    def test_log_transform_single_value(self):
        """Test log transform with single value."""
        value = 2.0
        
        transformed = log_transform(value)
        
        expected = np.log1p(2.0 - 1)  # log1p(1) = ln(2)
        assert abs(transformed - expected) < 1e-10


class TestRobustPercentileScale:
    """Test robust percentile scaling functionality."""
    
    def test_basic_robust_scaling(self):
        """Test basic robust percentile scaling."""
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        q25 = 1.0
        q75 = 3.0
        
        scaled = robust_percentile_scale(values, q25, q75)
        
        # Check scaling
        iqr = q75 - q25
        median = (q25 + q75) / 2
        expected = (values - median) / iqr
        
        np.testing.assert_array_almost_equal(scaled, expected)
    
    def test_robust_scaling_with_clipping(self):
        """Test robust scaling with clipping."""
        values = np.array([-10.0, 0.0, 5.0, 10.0])
        q25 = 1.0
        q75 = 3.0
        
        scaled = robust_percentile_scale(values, q25, q75, clip_range=(-1.0, 1.0))
        
        # Values should be clipped
        assert np.all(scaled >= -1.0)
        assert np.all(scaled <= 1.0)
    
    def test_robust_scaling_zero_iqr(self):
        """Test robust scaling with zero IQR."""
        values = np.array([5.0, 5.0, 5.0])
        q25 = 5.0
        q75 = 5.0
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scaled = robust_percentile_scale(values, q25, q75)
            
            # Should issue warning
            assert len(w) == 1
            assert "IQR is zero" in str(w[0].message)
        
        # Should return zeros
        np.testing.assert_array_equal(scaled, np.zeros_like(values))
    
    def test_robust_scaling_single_value(self):
        """Test robust scaling with single value."""
        value = 2.5
        q25 = 1.0
        q75 = 3.0
        
        scaled = robust_percentile_scale(value, q25, q75)
        
        iqr = q75 - q25
        median = (q25 + q75) / 2
        expected = (value - median) / iqr
        
        assert abs(scaled - expected) < 1e-10


class TestSafeDivide:
    """Test safe division functionality."""
    
    def test_basic_safe_divide(self):
        """Test basic safe division."""
        numerator = np.array([10.0, 20.0, 30.0])
        denominator = np.array([2.0, 4.0, 5.0])
        
        result = safe_divide(numerator, denominator)
        
        expected = np.array([5.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_safe_divide_with_zero_denominator(self):
        """Test safe division with zero denominator."""
        numerator = np.array([10.0, 20.0, 30.0])
        denominator = np.array([2.0, 0.0, 5.0])
        default = -999.0
        
        result = safe_divide(numerator, denominator, default)
        
        expected = np.array([5.0, -999.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_safe_divide_single_values(self):
        """Test safe division with single values."""
        numerator = 10.0
        denominator = 2.0
        
        result = safe_divide(numerator, denominator)
        
        assert result == 5.0
    
    def test_safe_divide_single_zero_denominator(self):
        """Test safe division with single zero denominator."""
        numerator = 10.0
        denominator = 0.0
        default = 42.0
        
        result = safe_divide(numerator, denominator, default)
        
        assert result == 42.0
    
    def test_safe_divide_very_small_denominator(self):
        """Test safe division with very small denominator."""
        numerator = 10.0
        denominator = 1e-12  # Very small but not zero
        
        result = safe_divide(numerator, denominator)
        
        # Should return default because denominator is effectively zero
        assert result == 0.0


class TestRollingNormalizer:
    """Test rolling normalizer functionality."""
    
    @pytest.fixture
    def normalizer(self):
        """Create a rolling normalizer instance."""
        return RollingNormalizer(alpha=0.1, warmup_samples=10)
    
    def test_rolling_normalizer_initialization(self, normalizer):
        """Test rolling normalizer initialization."""
        assert normalizer.alpha == 0.1
        assert normalizer.warmup_samples == 10
        assert normalizer.n_samples == 0
        assert normalizer.mean == 0.0
        assert normalizer.variance == 0.0
        assert normalizer.min_val == float('inf')
        assert normalizer.max_val == float('-inf')
    
    def test_rolling_normalizer_first_update(self, normalizer):
        """Test first update to rolling normalizer."""
        value = 5.0
        
        normalizer.update(value)
        
        assert normalizer.n_samples == 1
        assert normalizer.mean == 5.0
        assert normalizer.variance == 0.0
        assert normalizer.min_val == 5.0
        assert normalizer.max_val == 5.0
        assert normalizer.q25 == 5.0
        assert normalizer.q50 == 5.0
        assert normalizer.q75 == 5.0
    
    def test_rolling_normalizer_multiple_updates(self, normalizer):
        """Test multiple updates to rolling normalizer."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for value in values:
            normalizer.update(value)
        
        assert normalizer.n_samples == 5
        assert normalizer.mean > 0
        assert normalizer.variance > 0
        assert normalizer.min_val == 1.0
        assert normalizer.max_val == 5.0
        assert normalizer.std > 0
    
    def test_rolling_normalizer_zscore_normalization(self, normalizer):
        """Test z-score normalization with rolling normalizer."""
        # Add some data
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            normalizer.update(value)
        
        # Test normalization
        test_value = 3.0
        normalized = normalizer.normalize_zscore(test_value)
        
        # Should return a reasonable value
        assert np.isfinite(normalized)
        assert -5.0 <= normalized <= 5.0  # Reasonable range
    
    def test_rolling_normalizer_minmax_normalization(self, normalizer):
        """Test min-max normalization with rolling normalizer."""
        # Add some data
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            normalizer.update(value)
        
        # Test normalization
        test_value = 3.0
        normalized = normalizer.normalize_minmax(test_value)
        
        # Should be in [-1, 1] range
        assert -1.0 <= normalized <= 1.0
    
    def test_rolling_normalizer_robust_normalization(self, normalizer):
        """Test robust normalization with rolling normalizer."""
        # Add enough data for robust normalization
        values = list(range(1, 21))  # 20 values
        for value in values:
            normalizer.update(value)
        
        # Test normalization
        test_value = 10.0
        normalized = normalizer.normalize_robust(test_value)
        
        # Should return a reasonable value
        assert np.isfinite(normalized)
    
    def test_rolling_normalizer_insufficient_data(self, normalizer):
        """Test normalizer behavior with insufficient data."""
        # Only one update
        normalizer.update(5.0)
        
        # Z-score should return 0
        assert normalizer.normalize_zscore(5.0) == 0.0
        
        # Min-max should return midpoint
        assert normalizer.normalize_minmax(5.0) == 0.0
        
        # Robust should return 0
        assert normalizer.normalize_robust(5.0) == 0.0
    
    def test_rolling_normalizer_warmup_period(self):
        """Test normalizer behavior during warmup period."""
        normalizer = RollingNormalizer(alpha=0.1, warmup_samples=5)
        
        # During warmup, should use arithmetic mean
        for i in range(3):
            normalizer.update(float(i))
        
        # Should still be in warmup
        assert normalizer.n_samples < normalizer.warmup_samples
    
    def test_rolling_normalizer_extreme_values(self, normalizer):
        """Test normalizer with extreme values."""
        # Add normal values
        for i in range(10):
            normalizer.update(float(i))
        
        # Add extreme values
        extreme_values = [1e6, -1e6, np.inf, -np.inf]
        
        for value in extreme_values:
            if np.isfinite(value):
                normalizer.update(value)
        
        # Should handle extreme values gracefully
        assert np.isfinite(normalizer.mean)
        assert np.isfinite(normalizer.variance)
    
    def test_rolling_normalizer_consistency(self, normalizer):
        """Test normalizer consistency over time."""
        # Add data and track statistics
        stats_history = []
        
        for i in range(100):
            normalizer.update(np.sin(i * 0.1))  # Sinusoidal data
            
            if i % 10 == 0:
                stats_history.append({
                    'mean': normalizer.mean,
                    'std': normalizer.std,
                    'min': normalizer.min_val,
                    'max': normalizer.max_val
                })
        
        # Statistics should stabilize over time
        assert len(stats_history) > 5
        
        # Check that statistics are reasonable
        final_stats = stats_history[-1]
        assert -2.0 <= final_stats['mean'] <= 2.0
        assert 0.0 <= final_stats['std'] <= 2.0
        assert final_stats['min'] >= -1.1
        assert final_stats['max'] <= 1.1


class TestNumericalStability:
    """Test numerical stability of all normalizers."""
    
    def test_extreme_values_stability(self):
        """Test stability with extreme values."""
        extreme_values = [
            1e-10, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e10,
            -1e-10, -1e-5, -1e-3, -1e-1, -1e1, -1e3, -1e5, -1e10
        ]
        
        for func in [z_score_normalize, min_max_scale, percentage_from_price]:
            for value in extreme_values:
                try:
                    if func == z_score_normalize:
                        result = func(value, 0.0, 1.0)
                    elif func == min_max_scale:
                        result = func(value, -1e6, 1e6)
                    elif func == percentage_from_price:
                        if value > 0:
                            result = func(value, 1.0)
                        else:
                            continue
                    
                    # Result should be finite
                    assert np.isfinite(result)
                except (ValueError, ZeroDivisionError):
                    # Expected for some extreme cases
                    pass
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and infinity values."""
        problem_values = [np.nan, np.inf, -np.inf]
        
        for value in problem_values:
            # Test functions that should handle these gracefully
            try:
                result = exponential_decay(value, 0.1)
                assert np.isfinite(result)
            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f'Error occurred: {e}')
            
            try:
                result = log_transform(value)
                assert np.isfinite(result)
            except (ValueError, TypeError, AttributeError) as e:
                logger.error(f'Error occurred: {e}')
    
    def test_precision_preservation(self):
        """Test that precision is preserved for normal values."""
        normal_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        for value in normal_values:
            # Test that normal values are processed correctly
            result = exponential_decay(value, 0.1)
            assert 0.0 < result <= 1.0
            
            result = log_transform(value)
            assert np.isfinite(result)
            
            if value > 0:
                result = percentage_from_price(value, 1.0)
                assert np.isfinite(result)
    
    def test_rolling_normalizer_stability(self):
        """Test rolling normalizer stability under stress."""
        normalizer = RollingNormalizer(alpha=0.01, warmup_samples=100)
        
        # Add variety of data including edge cases
        test_data = []
        
        # Normal data
        test_data.extend(np.random.normal(0, 1, 1000))
        
        # Extreme values
        test_data.extend([1e6, -1e6, 1e-6, -1e-6])
        
        # Step changes
        test_data.extend([100.0] * 50)
        test_data.extend([-100.0] * 50)
        
        # Gradually add data
        for value in test_data:
            if np.isfinite(value):
                normalizer.update(value)
        
        # Check final state
        assert np.isfinite(normalizer.mean)
        assert np.isfinite(normalizer.variance)
        assert normalizer.variance >= 0
        assert normalizer.std >= 0
        assert normalizer.min_val <= normalizer.max_val
        
        # Test normalization methods
        test_value = 5.0
        
        zscore = normalizer.normalize_zscore(test_value)
        assert np.isfinite(zscore)
        
        minmax = normalizer.normalize_minmax(test_value)
        assert np.isfinite(minmax)
        assert -1.0 <= minmax <= 1.0
        
        robust = normalizer.normalize_robust(test_value)
        assert np.isfinite(robust)


class TestPerformanceUnderLoad:
    """Test performance of normalizers under load."""
    
    def test_vectorized_operations_performance(self):
        """Test performance of vectorized operations."""
        # Large array
        large_array = np.random.normal(0, 1, 100000)
        
        # Test vectorized operations
        import time
        
        start_time = time.time()
        z_normalized = z_score_normalize(large_array, 0.0, 1.0)
        z_time = time.time() - start_time
        
        start_time = time.time()
        minmax_normalized = min_max_scale(large_array, -5.0, 5.0)
        minmax_time = time.time() - start_time
        
        start_time = time.time()
        exp_decayed = exponential_decay(np.abs(large_array), 0.1)
        exp_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 100ms each)
        assert z_time < 0.1
        assert minmax_time < 0.1
        assert exp_time < 0.1
        
        # Results should be valid
        assert np.all(np.isfinite(z_normalized))
        assert np.all(np.isfinite(minmax_normalized))
        assert np.all(np.isfinite(exp_decayed))
    
    def test_rolling_normalizer_performance(self):
        """Test rolling normalizer performance."""
        normalizer = RollingNormalizer(alpha=0.01, warmup_samples=100)
        
        # Time many updates
        import time
        
        start_time = time.time()
        
        for i in range(10000):
            normalizer.update(np.sin(i * 0.001))
        
        update_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert update_time < 1.0  # Less than 1 second for 10k updates
        
        # Test normalization speed
        start_time = time.time()
        
        for i in range(1000):
            normalizer.normalize_zscore(float(i))
        
        normalize_time = time.time() - start_time
        
        # Should be fast
        assert normalize_time < 0.1
    
    def test_memory_efficiency(self):
        """Test memory efficiency of normalizers."""
        # Rolling normalizer should use constant memory
        normalizer = RollingNormalizer(alpha=0.01, warmup_samples=100)
        
        # Add many values
        for i in range(100000):
            normalizer.update(float(i))
        
        # Check that internal state is reasonable
        assert normalizer.n_samples == 100000
        assert np.isfinite(normalizer.mean)
        assert np.isfinite(normalizer.variance)
        
        # Memory usage should be constant regardless of number of updates
        # (This is implicit in the design - no growing collections)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])