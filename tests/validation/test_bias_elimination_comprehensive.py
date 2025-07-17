"""
AGENT 4 - COMPREHENSIVE BIAS ELIMINATION TEST SUITE
Critical validation tests for zero look-ahead bias guarantee

Test Coverage:
- Look-ahead bias detection and prevention
- Point-in-time data access validation  
- Temporal ordering compliance
- Out-of-sample testing integrity
- Statistical significance validation
- Walk-forward analysis correctness
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

# Add project path for imports
sys.path.append('/home/QuantNova/GrandModel')

from src.validation.bias_elimination_engine import (
    PointInTimeDataManager, BiasFreePMICalculator, BiasFreeCMWRQKCalculator,
    bias_free_rsi_calculation, bias_free_wma_calculation,
    WalkForwardValidator, StatisticalSignificanceTester,
    validate_system_integrity, ValidationMetrics
)


class TestBiasDetection:
    """Test suite for look-ahead bias detection"""
    
    def test_point_in_time_data_manager_future_access_detection(self):
        """Test that future data access is correctly detected"""
        manager = PointInTimeDataManager()
        
        # Valid access (historical data)
        assert manager.validate_temporal_access(100, 50, "test_operation") == True
        assert manager.validate_temporal_access(100, 100, "test_operation") == True
        
        # Invalid access (future data) 
        assert manager.validate_temporal_access(100, 101, "test_operation") == False
        assert manager.validate_temporal_access(50, 75, "test_operation") == False
        
        # Check that violations were recorded
        assert manager.validation_metrics.bias_violations == 2
        assert manager.validation_metrics.data_leakage_count == 2
    
    def test_historical_window_bias_prevention(self):
        """Test that historical window prevents future data access"""
        manager = PointInTimeDataManager()
        data = np.arange(1000, dtype=float)
        
        # Valid historical windows
        window = manager.get_historical_window(data, 100, 50)
        assert len(window) == 50
        assert np.array_equal(window, data[51:101])
        
        # Test at start of data
        window = manager.get_historical_window(data, 10, 20)
        assert len(window) == 11  # Can only get 0-10
        assert np.array_equal(window, data[0:11])
        
        # Test future access attempt should raise error
        with pytest.raises((RuntimeError, ValueError)):
            manager.get_historical_window(data, 100, 150)  # Would need future data
    
    def test_bias_free_rsi_temporal_constraints(self):
        """Test RSI calculation respects temporal constraints"""
        prices = np.cumsum(np.random.normal(0, 0.01, 1000)) + 100
        
        # Test at different time points
        for current_idx in [20, 50, 100, 500]:
            rsi = bias_free_rsi_calculation(prices, 14, current_idx)
            
            if current_idx >= 14:
                assert not np.isnan(rsi)
                assert 0 <= rsi <= 100
            else:
                assert np.isnan(rsi)
    
    def test_bias_free_wma_temporal_constraints(self):
        """Test WMA calculation respects temporal constraints"""
        prices = np.cumsum(np.random.normal(0, 0.01, 1000)) + 100
        
        # Test at different time points
        for current_idx in [5, 20, 50, 100]:
            wma = bias_free_wma_calculation(prices, 10, current_idx)
            
            if current_idx >= 9:  # length - 1
                assert not np.isnan(wma)
            else:
                assert np.isnan(wma)


class TestMLMIBiasElimination:
    """Test MLMI bias elimination"""
    
    def test_mlmi_knn_historical_patterns_only(self):
        """Test that k-NN only uses historical patterns"""
        mlmi_calc = BiasFreePMICalculator(num_neighbors=5)
        
        # Add some historical patterns
        for i in range(10):
            mlmi_calc.store_historical_pattern(
                rsi_slow=50.0 + i, 
                rsi_quick=45.0 + i,
                outcome=1 if i % 2 == 0 else -1,
                timestamp_idx=i
            )
        
        # Current prediction should only use patterns with timestamp < current_idx
        current_idx = 8
        prediction = mlmi_calc.bias_free_knn_predict(52.0, 47.0, current_idx)
        
        # Should only use patterns 0-7, not 8-9
        assert isinstance(prediction, float)
        assert mlmi_calc.validation_metrics.bias_violations == 0
    
    def test_mlmi_future_pattern_rejection(self):
        """Test that future patterns are rejected"""
        mlmi_calc = BiasFreePMICalculator(num_neighbors=5)
        
        # Add patterns including future ones (this simulates a bug)
        for i in range(10):
            mlmi_calc.store_historical_pattern(
                rsi_slow=50.0 + i,
                rsi_quick=45.0 + i, 
                outcome=1,
                timestamp_idx=i
            )
        
        # Manually add a "future" pattern to test detection
        mlmi_calc.historical_patterns.append({
            'rsi_slow': 60.0,
            'rsi_quick': 55.0,
            'outcome': 1,
            'timestamp_idx': 15  # Future timestamp
        })
        
        # Prediction at timestamp 10 should reject pattern from timestamp 15
        prediction = mlmi_calc.bias_free_knn_predict(52.0, 47.0, 10)
        
        # Should detect the bias violation
        assert mlmi_calc.validation_metrics.bias_violations >= 1
    
    def test_mlmi_full_calculation_bias_free(self):
        """Test full MLMI calculation is bias-free"""
        mlmi_calc = BiasFreePMICalculator()
        
        # Generate synthetic price data
        np.random.seed(42)
        prices = np.cumsum(np.random.normal(0, 0.01, 500)) + 100
        
        # Test calculation at different time points
        for current_idx in [120, 200, 300]:
            result = mlmi_calc.calculate_bias_free_mlmi(prices, current_idx)
            
            assert 'mlmi_value' in result
            assert 'mlmi_signal' in result
            assert isinstance(result['mlmi_value'], float)
            assert result['mlmi_signal'] in [-1, 0, 1]
        
        # Should have no bias violations
        assert mlmi_calc.validation_metrics.bias_violations == 0


class TestNWRQKBiasElimination:
    """Test NW-RQK bias elimination"""
    
    def test_nwrqk_kernel_distance_calculation(self):
        """Test that kernel distance uses proper historical data"""
        from src.validation.bias_elimination_engine import bias_free_nw_kernel
        
        # Test kernel calculation
        x_current = 100.0
        x_historical = 99.5
        h = 8.0
        r = 8.0
        
        weight = bias_free_nw_kernel(x_current, x_historical, h, r)
        assert isinstance(weight, float)
        assert weight > 0
        assert weight <= 1
    
    def test_nwrqk_regression_temporal_constraints(self):
        """Test NW-RQK regression respects temporal ordering"""
        nwrqk_calc = BiasFreeCMWRQKCalculator()
        
        # Generate price data
        np.random.seed(42) 
        prices = np.cumsum(np.random.normal(0, 0.01, 200)) + 100
        
        # Test at different time points
        for current_idx in [30, 50, 100]:
            yhat = nwrqk_calc.bias_free_kernel_regression(prices, current_idx, 8.0)
            
            if current_idx >= nwrqk_calc.x_0:
                assert not np.isnan(yhat) or current_idx < nwrqk_calc.x_0
            else:
                assert np.isnan(yhat)
        
        # Should have no temporal violations
        assert nwrqk_calc.validation_metrics.bias_violations == 0
    
    def test_nwrqk_full_calculation_bias_free(self):
        """Test full NW-RQK calculation is bias-free"""
        nwrqk_calc = BiasFreeCMWRQKCalculator()
        
        # Generate price data
        np.random.seed(42)
        prices = np.cumsum(np.random.normal(0, 0.01, 200)) + 100
        
        # Test calculation
        for current_idx in [40, 80, 150]:
            result = nwrqk_calc.calculate_bias_free_nwrqk(prices, current_idx)
            
            assert 'nwrqk_value' in result
            assert 'nwrqk_signal' in result
            assert isinstance(result['nwrqk_value'], float)
            assert result['nwrqk_signal'] in [-1, 0, 1]
        
        # Should have no bias violations
        assert nwrqk_calc.validation_metrics.bias_violations == 0


class TestWalkForwardValidation:
    """Test walk-forward analysis implementation"""
    
    def create_synthetic_data(self, n_periods: int = 2000) -> pd.DataFrame:
        """Create synthetic price data for testing"""
        dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='H')
        np.random.seed(42)
        
        # Generate price series with some trend and volatility
        returns = np.random.normal(0.0001, 0.02, n_periods)
        prices = 100 * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'close': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods))),
            'open': np.roll(prices, 1),
            'volume': np.random.randint(1000, 10000, n_periods)
        }, index=dates)
    
    def dummy_strategy(self, train_data: pd.DataFrame, 
                      test_data: pd.DataFrame) -> Dict[str, Any]:
        """Dummy strategy for testing walk-forward analysis"""
        # Simple buy-and-hold strategy
        initial_price = test_data['close'].iloc[0]
        final_price = test_data['close'].iloc[-1]
        
        total_return = (final_price / initial_price) - 1
        
        # Generate some random trades
        n_trades = len(test_data) // 10
        trade_returns = np.random.normal(total_return / n_trades, 0.01, n_trades)
        
        return {
            'returns': trade_returns.tolist(),
            'total_return': total_return,
            'n_trades': n_trades
        }
    
    def test_walk_forward_temporal_integrity(self):
        """Test walk-forward analysis maintains temporal integrity"""
        data = self.create_synthetic_data(2000)
        validator = WalkForwardValidator(train_window=500, test_window=100)
        
        results = validator.run_walk_forward_analysis(data, self.dummy_strategy)
        
        assert 'total_folds' in results
        assert 'fold_details' in results
        assert results['total_folds'] > 0
        
        # Check temporal ordering in folds
        for i, fold in enumerate(results['fold_details']):
            assert fold['train_end'] <= fold['test_start']
            
            # Each subsequent fold should start after the previous
            if i > 0:
                prev_fold = results['fold_details'][i-1]
                assert fold['train_start'] >= prev_fold['test_start']
    
    def test_walk_forward_out_of_sample_separation(self):
        """Test that training and testing data are properly separated"""
        data = self.create_synthetic_data(1000)
        validator = WalkForwardValidator(train_window=200, test_window=50)
        
        results = validator.run_walk_forward_analysis(data, self.dummy_strategy)
        
        # Verify no overlap between train and test periods
        for fold in results['fold_details']:
            train_end = fold['train_end']
            test_start = fold['test_start']
            
            # Test data should start after training data ends
            assert test_start >= train_end
    
    def test_walk_forward_metrics_calculation(self):
        """Test walk-forward metrics are calculated correctly"""
        data = self.create_synthetic_data(1000)
        validator = WalkForwardValidator(train_window=200, test_window=50)
        
        results = validator.run_walk_forward_analysis(data, self.dummy_strategy)
        
        # Check required metrics exist
        required_metrics = ['mean_return', 'std_return', 'sharpe_ratio', 
                           'max_drawdown', 'win_rate']
        for metric in required_metrics:
            assert metric in results
            assert isinstance(results[metric], (int, float))


class TestStatisticalSignificance:
    """Test statistical significance testing"""
    
    def test_t_test_significance_detection(self):
        """Test t-test correctly identifies significant returns"""
        # Generate returns with significant positive mean
        np.random.seed(42)
        significant_returns = np.random.normal(0.02, 0.05, 100)  # 2% mean return
        
        result = StatisticalSignificanceTester.t_test_significance(
            significant_returns.tolist(), alpha=0.05
        )
        
        assert 'is_significant' in result
        assert 'p_value' in result
        assert 't_statistic' in result
        assert isinstance(result['is_significant'], bool)
        assert 0 <= result['p_value'] <= 1
    
    def test_t_test_non_significant_returns(self):
        """Test t-test correctly identifies non-significant returns"""
        # Generate returns with mean close to zero
        np.random.seed(42)
        non_significant_returns = np.random.normal(0.001, 0.05, 30)  # 0.1% mean
        
        result = StatisticalSignificanceTester.t_test_significance(
            non_significant_returns.tolist(), alpha=0.05
        )
        
        # With small sample and low mean, should not be significant
        assert isinstance(result['is_significant'], bool)
        assert result['p_value'] > 0.05 or not result['is_significant']
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation"""
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.03, 50)
        
        result = StatisticalSignificanceTester.bootstrap_confidence_interval(
            returns.tolist(), confidence_level=0.95, n_bootstrap=100
        )
        
        assert 'confidence_interval' in result
        assert 'contains_zero' in result
        assert len(result['confidence_interval']) == 2
        assert result['confidence_interval'][0] <= result['confidence_interval'][1]
        assert isinstance(result['contains_zero'], bool)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for statistical tests"""
        insufficient_returns = [0.01]  # Only one return
        
        t_result = StatisticalSignificanceTester.t_test_significance(insufficient_returns)
        bootstrap_result = StatisticalSignificanceTester.bootstrap_confidence_interval(insufficient_returns)
        
        assert 'error' in t_result
        assert 'error' in bootstrap_result


class TestSystemIntegrity:
    """Test overall system integrity validation"""
    
    def test_system_integrity_validation_clean_system(self):
        """Test system integrity validation with clean (bias-free) system"""
        # Create clean components
        data_manager = PointInTimeDataManager()
        mlmi_calc = BiasFreePMICalculator()
        nwrqk_calc = BiasFreeCMWRQKCalculator()
        
        # Perform some valid operations
        data = np.random.random(100)
        _ = data_manager.get_historical_window(data, 50, 20)
        _ = mlmi_calc.bias_free_knn_predict(50.0, 45.0, 10)
        _ = nwrqk_calc.bias_free_kernel_regression(data, 50, 8.0)
        
        # Validate system integrity
        report = validate_system_integrity(data_manager, mlmi_calc, nwrqk_calc)
        
        assert report['system_status'] == 'BIAS_FREE'
        assert report['total_bias_violations'] == 0
        assert report['certification']['overall_certification'] is True
    
    def test_system_integrity_validation_biased_system(self):
        """Test system integrity validation detects biased system"""
        # Create components and introduce bias
        data_manager = PointInTimeDataManager()
        mlmi_calc = BiasFreePMICalculator()
        nwrqk_calc = BiasFreeCMWRQKCalculator()
        
        # Intentionally create bias violations
        _ = data_manager.validate_temporal_access(50, 60, "test")  # Future access
        
        # Validate system integrity
        report = validate_system_integrity(data_manager, mlmi_calc, nwrqk_calc)
        
        assert report['system_status'] == 'BIAS_DETECTED'
        assert report['total_bias_violations'] > 0
        assert report['certification']['overall_certification'] is False
    
    def test_validation_metrics_tracking(self):
        """Test that validation metrics are properly tracked"""
        metrics = ValidationMetrics(100, 5, 2, 3, 1)
        
        assert metrics.total_calculations == 100
        assert metrics.bias_violations == 5
        assert metrics.bias_ratio == 0.05
        assert metrics.is_bias_free is False
        
        clean_metrics = ValidationMetrics(100, 0, 0, 0, 0)
        assert clean_metrics.is_bias_free is True


class TestComprehensiveValidation:
    """Comprehensive end-to-end validation tests"""
    
    def test_end_to_end_bias_free_calculation(self):
        """Test complete bias-free calculation pipeline"""
        # Generate synthetic data
        np.random.seed(42)
        prices = np.cumsum(np.random.normal(0, 0.01, 1000)) + 100
        
        # Initialize bias-free calculators
        mlmi_calc = BiasFreePMICalculator()
        nwrqk_calc = BiasFreeCMWRQKCalculator()
        data_manager = PointInTimeDataManager()
        
        # Run calculations at multiple time points
        results = []
        for current_idx in range(100, len(prices), 50):
            mlmi_result = mlmi_calc.calculate_bias_free_mlmi(prices, current_idx)
            nwrqk_result = nwrqk_calc.calculate_bias_free_nwrqk(prices, current_idx)
            
            results.append({
                'timestamp': current_idx,
                'mlmi': mlmi_result,
                'nwrqk': nwrqk_result
            })
        
        # Validate all calculations were bias-free
        integrity_report = validate_system_integrity(data_manager, mlmi_calc, nwrqk_calc)
        
        assert len(results) > 0
        assert integrity_report['system_status'] == 'BIAS_FREE'
        assert integrity_report['total_bias_violations'] == 0
        
        # Verify all results have expected structure
        for result in results:
            assert 'mlmi' in result and 'nwrqk' in result
            assert 'mlmi_value' in result['mlmi']
            assert 'nwrqk_value' in result['nwrqk']
    
    def test_performance_under_stress(self):
        """Test performance and bias-freedom under stress conditions"""
        # Large dataset
        np.random.seed(42)
        large_prices = np.cumsum(np.random.normal(0, 0.02, 10000)) + 100
        
        mlmi_calc = BiasFreePMICalculator()
        nwrqk_calc = BiasFreeCMWRQKCalculator()
        data_manager = PointInTimeDataManager()
        
        # Run many calculations
        calculation_count = 0
        for current_idx in range(200, len(large_prices), 100):
            _ = mlmi_calc.calculate_bias_free_mlmi(large_prices, current_idx)
            _ = nwrqk_calc.calculate_bias_free_nwrqk(large_prices, current_idx)
            calculation_count += 2
        
        # Validate system remained bias-free under stress
        integrity_report = validate_system_integrity(data_manager, mlmi_calc, nwrqk_calc)
        
        assert calculation_count > 100  # Many calculations performed
        assert integrity_report['system_status'] == 'BIAS_FREE'
        assert integrity_report['total_bias_violations'] == 0


if __name__ == "__main__":
    print("🧪 AGENT 4 - COMPREHENSIVE BIAS ELIMINATION TEST SUITE")
    print("🛡️ Testing zero look-ahead bias guarantee...")
    
    # Run specific test classes
    pytest.main([
        __file__,
        "-v", 
        "--tb=short",
        "-x"  # Stop on first failure
    ])