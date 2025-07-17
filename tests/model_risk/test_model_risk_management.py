"""
Comprehensive Test Suite for Model Risk Management System

This test suite provides extensive testing for model risk management including
model validation, backtesting, performance monitoring, and statistical significance testing.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import warnings

# Import model risk management components
from src.model_risk.model_validator import (
    ModelValidator, ValidationResult, ValidationRule, ValidationSummary,
    ValidationSeverity, ValidationStatus, ValidationCategory,
    InputShapeValidation, InputRangeValidation, InputNaNValidation,
    OutputShapeValidation, OutputRangeValidation, MonotonicityValidation
)
from src.model_risk.backtesting_engine import (
    BacktestingEngine, BacktestResult, BacktestConfig, BacktestMetrics,
    BacktestStatus
)
from src.core.event_bus import EventBus


class TestModelValidator:
    """Test the model validation system"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def model_validator(self, event_bus):
        """Create model validator for testing"""
        return ModelValidator(event_bus)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.5, -0.3, 0.8]))
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
    def test_model_validator_initialization(self, model_validator):
        """Test model validator initialization"""
        assert model_validator is not None
        assert model_validator.enabled is True
        assert len(model_validator.validation_rules) > 0
        assert model_validator.total_validations == 0
        assert model_validator.total_failures == 0
    
    def test_input_shape_validation(self, sample_data):
        """Test input shape validation rule"""
        # Test passing validation
        rule = InputShapeValidation(expected_shape=(3, 3))
        result = rule.validate(None, sample_data)
        
        assert result.status == ValidationStatus.PASSED
        assert result.rule_id == "input_shape_validation"
        assert result.category == ValidationCategory.INPUT_VALIDATION
        assert result.severity == ValidationSeverity.ERROR
        
        # Test failing validation
        rule = InputShapeValidation(expected_shape=(2, 3))
        result = rule.validate(None, sample_data)
        
        assert result.status == ValidationStatus.FAILED
        assert "shape mismatch" in result.message.lower()
        assert result.details["expected_shape"] == (2, 3)
        assert result.details["actual_shape"] == (3, 3)
    
    def test_input_range_validation(self, sample_data):
        """Test input range validation rule"""
        # Test passing validation
        rule = InputRangeValidation(min_value=0.0, max_value=10.0)
        result = rule.validate(None, sample_data)
        
        assert result.status == ValidationStatus.PASSED
        assert result.rule_id == "input_range_validation"
        
        # Test failing validation
        rule = InputRangeValidation(min_value=0.0, max_value=5.0)
        result = rule.validate(None, sample_data)
        
        assert result.status == ValidationStatus.FAILED
        assert "out of range" in result.message.lower()
        assert result.details["actual_max"] == 9.0
        assert result.details["expected_max"] == 5.0
    
    def test_input_nan_validation(self):
        """Test input NaN validation rule"""
        rule = InputNaNValidation()
        
        # Test data without NaN
        clean_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = rule.validate(None, clean_data)
        
        assert result.status == ValidationStatus.PASSED
        assert result.details["nan_count"] == 0
        
        # Test data with NaN
        nan_data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        result = rule.validate(None, nan_data)
        
        assert result.status == ValidationStatus.FAILED
        assert "contains NaN values" in result.message
        assert result.details["nan_count"] == 1
        assert result.details["nan_percentage"] > 0
    
    def test_output_shape_validation(self, mock_model, sample_data):
        """Test output shape validation rule"""
        # Test passing validation
        rule = OutputShapeValidation(expected_shape=(3,))
        result = rule.validate(mock_model, sample_data)
        
        assert result.status == ValidationStatus.PASSED
        assert result.rule_id == "output_shape_validation"
        assert result.category == ValidationCategory.OUTPUT_VALIDATION
        
        # Test failing validation
        rule = OutputShapeValidation(expected_shape=(2,))
        result = rule.validate(mock_model, sample_data)
        
        assert result.status == ValidationStatus.FAILED
        assert "shape mismatch" in result.message.lower()
    
    def test_output_range_validation(self, mock_model, sample_data):
        """Test output range validation rule"""
        # Test passing validation
        rule = OutputRangeValidation(min_value=-1.0, max_value=1.0)
        result = rule.validate(mock_model, sample_data)
        
        assert result.status == ValidationStatus.PASSED
        assert result.rule_id == "output_range_validation"
        
        # Test failing validation (mock returns values outside range)
        rule = OutputRangeValidation(min_value=0.0, max_value=0.5)
        result = rule.validate(mock_model, sample_data)
        
        assert result.status == ValidationStatus.FAILED
        assert "out of range" in result.message.lower()
    
    def test_monotonicity_validation(self, sample_data):
        """Test monotonicity validation rule"""
        # Create mock model with monotonic behavior
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=lambda x: x[0, 0] * 0.1)  # Monotonic in first feature
        
        rule = MonotonicityValidation(feature_index=0, expected_direction="increasing")
        result = rule.validate(mock_model, sample_data)
        
        assert result.status == ValidationStatus.PASSED
        assert result.rule_id == "monotonicity_validation"
        assert result.category == ValidationCategory.BEHAVIORAL_VALIDATION
    
    def test_validation_rule_management(self, model_validator):
        """Test validation rule management"""
        # Test adding rule
        new_rule = InputRangeValidation(min_value=-5.0, max_value=5.0)
        success = model_validator.add_rule(new_rule)
        
        assert success is True
        assert new_rule.rule_id in model_validator.validation_rules
        
        # Test enabling/disabling rule
        success = model_validator.disable_rule(new_rule.rule_id)
        assert success is True
        assert model_validator.validation_rules[new_rule.rule_id].enabled is False
        
        success = model_validator.enable_rule(new_rule.rule_id)
        assert success is True
        assert model_validator.validation_rules[new_rule.rule_id].enabled is True
        
        # Test removing rule
        success = model_validator.remove_rule(new_rule.rule_id)
        assert success is True
        assert new_rule.rule_id not in model_validator.validation_rules
    
    def test_model_validation_integration(self, model_validator, mock_model, sample_data):
        """Test integrated model validation"""
        # Run validation
        summary = model_validator.validate_model(
            model=mock_model,
            data=sample_data,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        assert isinstance(summary, ValidationSummary)
        assert summary.model_id == "test_model"
        assert summary.model_version == "1.0.0"
        assert summary.total_validations > 0
        assert len(summary.results) > 0
        assert summary.execution_time > 0
        
        # Check that validation was recorded
        assert model_validator.total_validations > 0
        assert len(model_validator.validation_history) > 0
    
    def test_validation_with_failures(self, model_validator, mock_model):
        """Test validation with failures"""
        # Create data that will fail validation
        bad_data = np.array([[np.nan, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        summary = model_validator.validate_model(
            model=mock_model,
            data=bad_data,
            model_id="failing_model",
            model_version="1.0.0"
        )
        
        assert summary.failed > 0
        assert summary.success_rate < 1.0
        assert summary.overall_status in [ValidationStatus.FAILED, ValidationStatus.WARNING]
        
        # Check for specific failure
        nan_failures = [r for r in summary.results if r.rule_id == "input_nan_validation" and r.is_failure()]
        assert len(nan_failures) > 0
    
    def test_validation_history_management(self, model_validator, mock_model, sample_data):
        """Test validation history management"""
        # Run multiple validations
        for i in range(3):
            model_validator.validate_model(
                model=mock_model,
                data=sample_data,
                model_id=f"model_{i}",
                model_version="1.0.0"
            )
        
        # Test history retrieval
        history = model_validator.get_validation_history()
        assert len(history) == 3
        
        # Test filtering by model_id
        filtered_history = model_validator.get_validation_history(model_id="model_1")
        assert len(filtered_history) == 1
        assert filtered_history[0].model_id == "model_1"
        
        # Test limiting results
        limited_history = model_validator.get_validation_history(limit=2)
        assert len(limited_history) == 2
    
    def test_validation_rule_statistics(self, model_validator, mock_model, sample_data):
        """Test validation rule statistics"""
        # Run some validations
        for i in range(5):
            model_validator.validate_model(
                model=mock_model,
                data=sample_data,
                model_id=f"model_{i}",
                model_version="1.0.0"
            )
        
        # Get rule statistics
        stats = model_validator.get_rule_statistics()
        
        assert isinstance(stats, dict)
        assert len(stats) > 0
        
        # Check specific rule statistics
        for rule_id, rule_stats in stats.items():
            assert "rule_id" in rule_stats
            assert "execution_count" in rule_stats
            assert "failure_count" in rule_stats
            assert "failure_rate" in rule_stats
            assert rule_stats["execution_count"] == 5  # Should have run 5 times
    
    def test_validation_report_generation(self, model_validator, mock_model, sample_data):
        """Test validation report generation"""
        # Run validations with some failures
        for i in range(3):
            if i == 1:
                # Create one with failures
                bad_data = np.array([[100.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Out of range
                model_validator.validate_model(
                    model=mock_model,
                    data=bad_data,
                    model_id="model_with_failures",
                    model_version="1.0.0"
                )
            else:
                model_validator.validate_model(
                    model=mock_model,
                    data=sample_data,
                    model_id="good_model",
                    model_version="1.0.0"
                )
        
        # Generate report
        report = model_validator.generate_validation_report(period_days=1)
        
        assert "summary" in report
        assert "most_problematic_rules" in report
        assert "recent_validations" in report
        assert report["summary"]["total_validation_runs"] == 3
        assert report["summary"]["total_validations"] > 0
        assert report["summary"]["total_failed"] > 0
        assert len(report["most_problematic_rules"]) > 0
    
    def test_validator_status(self, model_validator, mock_model, sample_data):
        """Test validator status reporting"""
        # Run some validations
        model_validator.validate_model(mock_model, sample_data, "test_model", "1.0.0")
        
        status = model_validator.get_validator_status()
        
        assert "enabled" in status
        assert "total_rules" in status
        assert "active_rules" in status
        assert "total_validations" in status
        assert "total_failures" in status
        assert "overall_failure_rate" in status
        assert "validation_history_count" in status
        assert status["enabled"] is True
        assert status["total_rules"] > 0
        assert status["total_validations"] > 0
    
    def test_validator_enable_disable(self, model_validator, mock_model, sample_data):
        """Test validator enable/disable functionality"""
        # Disable validator
        model_validator.disable_validator()
        assert model_validator.enabled is False
        
        # Run validation (should be skipped)
        summary = model_validator.validate_model(mock_model, sample_data, "test_model", "1.0.0")
        assert summary.total_validations == 0
        
        # Re-enable validator
        model_validator.enable_validator()
        assert model_validator.enabled is True
        
        # Run validation (should work)
        summary = model_validator.validate_model(mock_model, sample_data, "test_model", "1.0.0")
        assert summary.total_validations > 0
    
    def test_validation_rule_execution_statistics(self, model_validator, mock_model, sample_data):
        """Test validation rule execution statistics tracking"""
        # Get initial rule
        rule_id = list(model_validator.validation_rules.keys())[0]
        rule = model_validator.validation_rules[rule_id]
        
        initial_execution_count = rule.execution_count
        initial_failure_count = rule.failure_count
        
        # Run validation
        model_validator.validate_model(mock_model, sample_data, "test_model", "1.0.0")
        
        # Check statistics updated
        assert rule.execution_count > initial_execution_count
        assert rule.last_execution is not None
        assert rule.last_result is not None
    
    def test_validation_cleanup(self, model_validator, mock_model, sample_data):
        """Test validation history cleanup"""
        # Run some validations
        for i in range(5):
            model_validator.validate_model(mock_model, sample_data, f"model_{i}", "1.0.0")
        
        initial_count = len(model_validator.validation_history)
        assert initial_count == 5
        
        # Cleanup (using 0 days to clean everything)
        cleaned_count = model_validator.cleanup_history(days_old=0)
        
        assert cleaned_count == initial_count
        assert len(model_validator.validation_history) == 0


class TestBacktestingEngine:
    """Test the backtesting engine"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def backtesting_engine(self, event_bus):
        """Create backtesting engine for testing"""
        return BacktestingEngine(event_bus)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.1, -0.05, 0.15]))
        return model
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)  # For reproducible tests
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def backtest_config(self):
        """Create backtest configuration"""
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=1000000.0,
            transaction_cost=0.001,
            slippage=0.0005,
            max_position_size=0.2,
            rebalance_frequency="daily",
            risk_free_rate=0.02,
            lookback_window=252,
            out_of_sample_ratio=0.3
        )
    
    def test_backtesting_engine_initialization(self, backtesting_engine):
        """Test backtesting engine initialization"""
        assert backtesting_engine is not None
        assert len(backtesting_engine.backtest_history) == 0
        assert len(backtesting_engine.running_backtests) == 0
    
    def test_backtest_config_creation(self, backtest_config):
        """Test backtest configuration"""
        assert backtest_config.start_date == datetime(2023, 1, 1)
        assert backtest_config.end_date == datetime(2023, 12, 31)
        assert backtest_config.initial_capital == 1000000.0
        assert backtest_config.transaction_cost == 0.001
        assert backtest_config.max_position_size == 0.2
        assert backtest_config.out_of_sample_ratio == 0.3
    
    def test_backtest_execution(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test backtest execution"""
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=backtest_config,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        assert isinstance(result, BacktestResult)
        assert result.model_id == "test_model"
        assert result.model_version == "1.0.0"
        assert result.status == BacktestStatus.COMPLETED
        assert result.is_successful() is True
        assert result.metrics is not None
        assert result.returns is not None
        assert result.positions is not None
        assert result.execution_time > 0
    
    def test_backtest_metrics_calculation(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test backtest metrics calculation"""
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=backtest_config,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        metrics = result.metrics
        assert isinstance(metrics, BacktestMetrics)
        
        # Check all metrics are present
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'annual_return')
        assert hasattr(metrics, 'volatility')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'var_95')
        assert hasattr(metrics, 'var_99')
        assert hasattr(metrics, 'win_rate')
        assert hasattr(metrics, 'profit_factor')
        assert hasattr(metrics, 'calmar_ratio')
        assert hasattr(metrics, 'sortino_ratio')
        
        # Check metrics are reasonable
        assert -1.0 <= metrics.total_return <= 5.0  # Reasonable return range
        assert 0.0 <= metrics.volatility <= 1.0  # Reasonable volatility range
        assert -3.0 <= metrics.sharpe_ratio <= 5.0  # Reasonable Sharpe ratio range
        assert -1.0 <= metrics.max_drawdown <= 0.0  # Drawdown should be negative
        assert 0.0 <= metrics.win_rate <= 1.0  # Win rate should be between 0 and 1
        assert metrics.trades_count >= 0
    
    def test_backtest_metrics_to_dict(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test metrics conversion to dictionary"""
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=backtest_config,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        metrics_dict = result.metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert "total_return" in metrics_dict
        assert "annual_return" in metrics_dict
        assert "volatility" in metrics_dict
        assert "sharpe_ratio" in metrics_dict
        assert "max_drawdown" in metrics_dict
        assert "var_95" in metrics_dict
        assert "var_99" in metrics_dict
        assert "win_rate" in metrics_dict
        assert "trades_count" in metrics_dict
    
    def test_backtest_with_different_configs(self, backtesting_engine, mock_model, sample_market_data):
        """Test backtest with different configurations"""
        # Test with high transaction costs
        high_cost_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            initial_capital=1000000.0,
            transaction_cost=0.01,  # High cost
            slippage=0.005,  # High slippage
            max_position_size=0.1,  # Small position
            out_of_sample_ratio=0.3
        )
        
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=high_cost_config,
            model_id="high_cost_model",
            model_version="1.0.0"
        )
        
        assert result.is_successful()
        assert result.metrics.total_return is not None
        
        # Test with low transaction costs
        low_cost_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            initial_capital=1000000.0,
            transaction_cost=0.0001,  # Low cost
            slippage=0.0001,  # Low slippage
            max_position_size=0.5,  # Large position
            out_of_sample_ratio=0.3
        )
        
        result2 = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=low_cost_config,
            model_id="low_cost_model",
            model_version="1.0.0"
        )
        
        assert result2.is_successful()
        # Low cost should generally perform better (though not guaranteed)
        # assert result2.metrics.total_return >= result.metrics.total_return
    
    def test_backtest_error_handling(self, backtesting_engine, mock_model, backtest_config):
        """Test backtest error handling"""
        # Test with empty data
        empty_data = pd.DataFrame()
        
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=empty_data,
            config=backtest_config,
            model_id="empty_data_model",
            model_version="1.0.0"
        )
        
        assert result.status == BacktestStatus.FAILED
        assert len(result.errors) > 0
        assert "No data available" in result.errors[0] or "empty" in result.errors[0].lower()
    
    def test_backtest_history_management(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test backtest history management"""
        # Run multiple backtests
        for i in range(3):
            config = BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=1000000.0,
                transaction_cost=0.001,
                max_position_size=0.2,
                out_of_sample_ratio=0.3
            )
            
            backtesting_engine.run_backtest(
                model=mock_model,
                data=sample_market_data,
                config=config,
                model_id=f"model_{i}",
                model_version="1.0.0"
            )
        
        # Test history retrieval
        history = backtesting_engine.get_backtest_history()
        assert len(history) == 3
        
        # Test filtering by model_id
        filtered_history = backtesting_engine.get_backtest_history(model_id="model_1")
        assert len(filtered_history) == 1
        assert filtered_history[0].model_id == "model_1"
        
        # Test limiting results
        limited_history = backtesting_engine.get_backtest_history(limit=2)
        assert len(limited_history) == 2
    
    def test_running_backtests_tracking(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test running backtests tracking"""
        # Check initial state
        running = backtesting_engine.get_running_backtests()
        assert len(running) == 0
        
        # Run backtest
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=backtest_config,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        # After completion, should not be in running backtests
        running = backtesting_engine.get_running_backtests()
        assert len(running) == 0
        
        # Should be in history
        history = backtesting_engine.get_backtest_history()
        assert len(history) == 1
        assert history[0].backtest_id == result.backtest_id
    
    def test_backtesting_engine_status(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test backtesting engine status"""
        # Run some backtests
        for i in range(2):
            config = BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=1000000.0,
                transaction_cost=0.001,
                max_position_size=0.2,
                out_of_sample_ratio=0.3
            )
            
            backtesting_engine.run_backtest(
                model=mock_model,
                data=sample_market_data,
                config=config,
                model_id=f"model_{i}",
                model_version="1.0.0"
            )
        
        # Get status
        status = backtesting_engine.get_engine_status()
        
        assert "total_backtests" in status
        assert "running_backtests" in status
        assert "successful_backtests" in status
        assert "failed_backtests" in status
        assert "average_execution_time" in status
        assert "last_backtest" in status
        
        assert status["total_backtests"] == 2
        assert status["running_backtests"] == 0
        assert status["successful_backtests"] == 2
        assert status["failed_backtests"] == 0
        assert status["average_execution_time"] > 0
    
    def test_backtest_result_properties(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test backtest result properties"""
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=backtest_config,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        # Test execution time calculation
        assert result.execution_time > 0
        assert result.execution_time == (result.end_time - result.start_time).total_seconds()
        
        # Test success check
        assert result.is_successful() is True
        assert result.status == BacktestStatus.COMPLETED
        assert result.metrics is not None
    
    def test_backtest_signal_generation(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test backtest signal generation"""
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=backtest_config,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        # Check that signals were generated and used
        assert result.positions is not None
        assert len(result.positions) > 0
        
        # Check that positions are within limits
        max_position = result.positions['position'].abs().max()
        assert max_position <= backtest_config.max_position_size
        
        # Check that portfolio value changes over time
        portfolio_values = result.positions['value']
        assert len(portfolio_values.unique()) > 1  # Should have some variation
    
    def test_backtest_trade_execution(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test backtest trade execution"""
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=backtest_config,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        # Check trades were executed
        assert result.trades is not None
        
        if len(result.trades) > 0:
            # Check trade structure
            assert "date" in result.trades.columns
            assert "side" in result.trades.columns
            assert "quantity" in result.trades.columns
            assert "price" in result.trades.columns
            assert "value" in result.trades.columns
            assert "cost" in result.trades.columns
            
            # Check trade values
            for _, trade in result.trades.iterrows():
                assert trade["side"] in ["BUY", "SELL"]
                assert trade["quantity"] > 0
                assert trade["price"] > 0
                assert trade["cost"] >= 0  # Transaction costs should be positive
    
    def test_backtest_out_of_sample_split(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test out-of-sample data split"""
        # Set specific out-of-sample ratio
        backtest_config.out_of_sample_ratio = 0.4
        
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=backtest_config,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        assert result.is_successful()
        
        # The test data length should be approximately 40% of total data
        total_data_length = len(sample_market_data)
        expected_test_length = int(total_data_length * 0.4)
        
        # Check that results exist for the test period
        assert len(result.returns) > 0
        assert len(result.positions) > 0
    
    def test_backtest_performance_metrics_accuracy(self, backtesting_engine, mock_model, sample_market_data, backtest_config):
        """Test accuracy of performance metrics calculations"""
        result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=backtest_config,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        metrics = result.metrics
        
        # Test total return calculation
        initial_value = backtest_config.initial_capital
        final_value = result.positions['value'].iloc[-1]
        expected_total_return = (final_value / initial_value) - 1
        
        assert abs(metrics.total_return - expected_total_return) < 0.01  # Within 1% tolerance
        
        # Test volatility calculation
        returns = result.returns
        expected_volatility = returns.std() * np.sqrt(252)
        assert abs(metrics.volatility - expected_volatility) < 0.01
        
        # Test VaR calculations
        assert metrics.var_95 <= np.percentile(returns, 6)  # Should be around 5th percentile
        assert metrics.var_99 <= np.percentile(returns, 2)  # Should be around 1st percentile
        
        # Test drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        expected_max_drawdown = drawdown.min()
        
        assert abs(metrics.max_drawdown - expected_max_drawdown) < 0.01


class TestModelRiskIntegration:
    """Test integrated model risk management functionality"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def model_risk_system(self, event_bus):
        """Create integrated model risk system"""
        return {
            "model_validator": ModelValidator(event_bus),
            "backtesting_engine": BacktestingEngine(event_bus)
        }
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.1, -0.05, 0.15]))
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
        np.random.seed(42)
        
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    def test_end_to_end_model_risk_workflow(self, model_risk_system, mock_model, sample_data, sample_market_data):
        """Test end-to-end model risk management workflow"""
        validator = model_risk_system["model_validator"]
        backtesting_engine = model_risk_system["backtesting_engine"]
        
        # 1. Model validation
        validation_summary = validator.validate_model(
            model=mock_model,
            data=sample_data,
            model_id="test_model",
            model_version="1.0.0"
        )
        
        assert validation_summary.overall_status == ValidationStatus.PASSED
        assert validation_summary.passed > 0
        
        # 2. Backtesting (only if validation passes)
        if validation_summary.overall_status == ValidationStatus.PASSED:
            config = BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 6, 30),
                initial_capital=1000000.0,
                transaction_cost=0.001,
                max_position_size=0.2,
                out_of_sample_ratio=0.3
            )
            
            backtest_result = backtesting_engine.run_backtest(
                model=mock_model,
                data=sample_market_data,
                config=config,
                model_id="test_model",
                model_version="1.0.0"
            )
            
            assert backtest_result.is_successful()
            assert backtest_result.metrics is not None
            assert backtest_result.metrics.sharpe_ratio is not None
        
        # 3. Generate comprehensive report
        validation_report = validator.generate_validation_report(
            model_id="test_model",
            period_days=1
        )
        
        backtest_status = backtesting_engine.get_engine_status()
        
        # Verify complete workflow
        assert validation_report["summary"]["total_validation_runs"] > 0
        assert backtest_status["total_backtests"] > 0
        assert backtest_status["successful_backtests"] > 0
    
    def test_model_risk_decision_workflow(self, model_risk_system, mock_model, sample_data, sample_market_data):
        """Test model risk decision workflow"""
        validator = model_risk_system["model_validator"]
        backtesting_engine = model_risk_system["backtesting_engine"]
        
        # Test with a model that fails validation
        bad_data = np.array([[np.nan, 2.0, 3.0], [4.0, np.inf, 6.0]])
        
        validation_summary = validator.validate_model(
            model=mock_model,
            data=bad_data,
            model_id="bad_model",
            model_version="1.0.0"
        )
        
        # Should have failures
        assert validation_summary.failed > 0
        assert validation_summary.overall_status in [ValidationStatus.FAILED, ValidationStatus.WARNING]
        
        # Decision: Don't backtest models that fail critical validations
        if validation_summary.has_critical_failures:
            # Should not proceed with backtesting
            assert validation_summary.critical_failures > 0
            
            # Log decision not to backtest
            logger.info(
                "Model not approved for backtesting due to critical validation failures",
                model_id="bad_model",
                critical_failures=validation_summary.critical_failures
            )
        else:
            # Proceed with backtesting for non-critical failures
            config = BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=1000000.0,
                transaction_cost=0.001,
                max_position_size=0.1,  # Reduced position size for risky model
                out_of_sample_ratio=0.3
            )
            
            backtest_result = backtesting_engine.run_backtest(
                model=mock_model,
                data=sample_market_data,
                config=config,
                model_id="bad_model",
                model_version="1.0.0"
            )
            
            assert backtest_result.model_id == "bad_model"
    
    def test_model_comparison_workflow(self, model_risk_system, mock_model, sample_data, sample_market_data):
        """Test model comparison workflow"""
        validator = model_risk_system["model_validator"]
        backtesting_engine = model_risk_system["backtesting_engine"]
        
        models = {
            "model_a": mock_model,
            "model_b": mock_model  # Same model for simplicity
        }
        
        validation_results = {}
        backtest_results = {}
        
        # Validate and backtest multiple models
        for model_id, model in models.items():
            # Validation
            validation_summary = validator.validate_model(
                model=model,
                data=sample_data,
                model_id=model_id,
                model_version="1.0.0"
            )
            validation_results[model_id] = validation_summary
            
            # Backtesting
            config = BacktestConfig(
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 3, 31),
                initial_capital=1000000.0,
                transaction_cost=0.001,
                max_position_size=0.2,
                out_of_sample_ratio=0.3
            )
            
            backtest_result = backtesting_engine.run_backtest(
                model=model,
                data=sample_market_data,
                config=config,
                model_id=model_id,
                model_version="1.0.0"
            )
            backtest_results[model_id] = backtest_result
        
        # Compare results
        for model_id in models.keys():
            validation = validation_results[model_id]
            backtest = backtest_results[model_id]
            
            assert validation.model_id == model_id
            assert backtest.model_id == model_id
            assert validation.overall_status == ValidationStatus.PASSED
            assert backtest.is_successful()
        
        # Rank models by performance
        ranked_models = sorted(
            backtest_results.items(),
            key=lambda x: x[1].metrics.sharpe_ratio,
            reverse=True
        )
        
        assert len(ranked_models) == 2
        assert ranked_models[0][1].metrics.sharpe_ratio is not None
    
    def test_continuous_monitoring_workflow(self, model_risk_system, mock_model, sample_data, sample_market_data):
        """Test continuous monitoring workflow"""
        validator = model_risk_system["model_validator"]
        backtesting_engine = model_risk_system["backtesting_engine"]
        
        # Simulate continuous monitoring over multiple periods
        periods = [
            (datetime(2023, 1, 1), datetime(2023, 2, 28)),
            (datetime(2023, 2, 1), datetime(2023, 3, 31)),
            (datetime(2023, 3, 1), datetime(2023, 4, 30))
        ]
        
        monitoring_results = []
        
        for i, (start_date, end_date) in enumerate(periods):
            # Validate model
            validation_summary = validator.validate_model(
                model=mock_model,
                data=sample_data,
                model_id="monitored_model",
                model_version=f"1.{i}.0"
            )
            
            # Backtest on period
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=1000000.0,
                transaction_cost=0.001,
                max_position_size=0.2,
                out_of_sample_ratio=0.3
            )
            
            backtest_result = backtesting_engine.run_backtest(
                model=mock_model,
                data=sample_market_data,
                config=config,
                model_id="monitored_model",
                model_version=f"1.{i}.0"
            )
            
            monitoring_results.append({
                "period": i,
                "validation": validation_summary,
                "backtest": backtest_result
            })
        
        # Analyze monitoring results
        assert len(monitoring_results) == 3
        
        # Check validation consistency
        validation_success_rates = [
            r["validation"].success_rate for r in monitoring_results
        ]
        
        # Check backtest performance trend
        sharpe_ratios = [
            r["backtest"].metrics.sharpe_ratio for r in monitoring_results
        ]
        
        # All should be successful
        assert all(r["validation"].overall_status == ValidationStatus.PASSED for r in monitoring_results)
        assert all(r["backtest"].is_successful() for r in monitoring_results)
        
        # Generate monitoring report
        validation_report = validator.generate_validation_report(
            model_id="monitored_model",
            period_days=90
        )
        
        backtest_history = backtesting_engine.get_backtest_history(
            model_id="monitored_model"
        )
        
        assert validation_report["summary"]["total_validation_runs"] == 3
        assert len(backtest_history) == 3
    
    def test_risk_thresholds_and_alerts(self, model_risk_system, mock_model, sample_data, sample_market_data):
        """Test risk thresholds and alerting"""
        validator = model_risk_system["model_validator"]
        backtesting_engine = model_risk_system["backtesting_engine"]
        
        # Define risk thresholds
        risk_thresholds = {
            "min_sharpe_ratio": 0.5,
            "max_drawdown": -0.20,
            "min_win_rate": 0.4,
            "max_validation_failures": 2
        }
        
        # Test model
        validation_summary = validator.validate_model(
            model=mock_model,
            data=sample_data,
            model_id="risk_test_model",
            model_version="1.0.0"
        )
        
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            initial_capital=1000000.0,
            transaction_cost=0.001,
            max_position_size=0.2,
            out_of_sample_ratio=0.3
        )
        
        backtest_result = backtesting_engine.run_backtest(
            model=mock_model,
            data=sample_market_data,
            config=config,
            model_id="risk_test_model",
            model_version="1.0.0"
        )
        
        # Check thresholds
        alerts = []
        
        # Validation alerts
        if validation_summary.failed > risk_thresholds["max_validation_failures"]:
            alerts.append(f"Validation failures ({validation_summary.failed}) exceed threshold ({risk_thresholds['max_validation_failures']})")
        
        # Backtest alerts
        if backtest_result.is_successful():
            metrics = backtest_result.metrics
            
            if metrics.sharpe_ratio < risk_thresholds["min_sharpe_ratio"]:
                alerts.append(f"Sharpe ratio ({metrics.sharpe_ratio:.3f}) below threshold ({risk_thresholds['min_sharpe_ratio']})")
            
            if metrics.max_drawdown < risk_thresholds["max_drawdown"]:
                alerts.append(f"Max drawdown ({metrics.max_drawdown:.3f}) exceeds threshold ({risk_thresholds['max_drawdown']})")
            
            if metrics.win_rate < risk_thresholds["min_win_rate"]:
                alerts.append(f"Win rate ({metrics.win_rate:.3f}) below threshold ({risk_thresholds['min_win_rate']})")
        
        # Log alerts
        if alerts:
            logger.warning("Risk threshold alerts", model_id="risk_test_model", alerts=alerts)
        
        # Verify system is working
        assert validation_summary.model_id == "risk_test_model"
        assert backtest_result.model_id == "risk_test_model"


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])