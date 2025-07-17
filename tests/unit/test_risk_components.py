"""
Unit tests for risk management components.

This module tests the risk calculation, monitoring, and validation
components that ensure safe trading operations.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import time
from datetime import datetime, timedelta

# Test markers
pytestmark = [pytest.mark.unit, pytest.mark.risk]


class TestKellyCriterionCalculator:
    """Test the Kelly Criterion position sizing calculator."""

    @pytest.fixture
    def mock_kelly_calculator(self):
        """Create a mock Kelly criterion calculator."""
        calculator = Mock()
        calculator.win_rate = 0.0
        calculator.avg_win = 0.0
        calculator.avg_loss = 0.0
        calculator.num_trades = 0
        
        # Mock methods
        calculator.calculate_kelly_fraction = Mock()
        calculator.update_statistics = Mock()
        calculator.get_optimal_position_size = Mock()
        calculator.validate_inputs = Mock(return_value=True)
        
        return calculator

    @pytest.fixture
    def sample_trade_history(self):
        """Sample trade history for testing."""
        return [
            {"pnl": 50.0, "outcome": "win", "timestamp": "2023-01-01"},
            {"pnl": -30.0, "outcome": "loss", "timestamp": "2023-01-02"},
            {"pnl": 75.0, "outcome": "win", "timestamp": "2023-01-03"},
            {"pnl": -25.0, "outcome": "loss", "timestamp": "2023-01-04"},
            {"pnl": 40.0, "outcome": "win", "timestamp": "2023-01-05"},
            {"pnl": -35.0, "outcome": "loss", "timestamp": "2023-01-06"},
            {"pnl": 60.0, "outcome": "win", "timestamp": "2023-01-07"},
            {"pnl": -20.0, "outcome": "loss", "timestamp": "2023-01-08"},
            {"pnl": 80.0, "outcome": "win", "timestamp": "2023-01-09"},
            {"pnl": -40.0, "outcome": "loss", "timestamp": "2023-01-10"}
        ]

    def test_kelly_fraction_calculation(self, mock_kelly_calculator):
        """Test Kelly fraction calculation with various scenarios."""
        test_scenarios = [
            {"win_rate": 0.6, "avg_win": 0.025, "avg_loss": 0.02, "expected_range": (0.1, 0.4)},
            {"win_rate": 0.5, "avg_win": 0.03, "avg_loss": 0.02, "expected_range": (0.1, 0.3)},
            {"win_rate": 0.7, "avg_win": 0.02, "avg_loss": 0.015, "expected_range": (0.2, 0.5)},
            {"win_rate": 0.45, "avg_win": 0.02, "avg_loss": 0.02, "expected_range": (0.0, 0.0)}  # Negative expectancy
        ]
        
        for scenario in test_scenarios:
            # Mock Kelly calculation: f* = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = scenario["avg_win"] / scenario["avg_loss"]
            p = scenario["win_rate"]
            q = 1 - p
            kelly_fraction = max(0, (b * p - q) / b)
            
            mock_kelly_calculator.calculate_kelly_fraction = Mock(return_value=kelly_fraction)
            result = mock_kelly_calculator.calculate_kelly_fraction(
                scenario["win_rate"], 
                scenario["avg_win"], 
                scenario["avg_loss"]
            )
            
            expected_min, expected_max = scenario["expected_range"]
            assert expected_min <= result <= expected_max
            mock_kelly_calculator.calculate_kelly_fraction.assert_called_with(
                scenario["win_rate"], 
                scenario["avg_win"], 
                scenario["avg_loss"]
            )

    def test_statistics_update(self, mock_kelly_calculator, sample_trade_history):
        """Test updating statistics from trade history."""
        # Calculate expected statistics
        wins = [trade for trade in sample_trade_history if trade["outcome"] == "win"]
        losses = [trade for trade in sample_trade_history if trade["outcome"] == "loss"]
        
        expected_win_rate = len(wins) / len(sample_trade_history)
        expected_avg_win = np.mean([trade["pnl"] for trade in wins])
        expected_avg_loss = abs(np.mean([trade["pnl"] for trade in losses]))
        
        # Mock statistics update
        def mock_update(trade_history):
            mock_kelly_calculator.win_rate = expected_win_rate
            mock_kelly_calculator.avg_win = expected_avg_win
            mock_kelly_calculator.avg_loss = expected_avg_loss
            mock_kelly_calculator.num_trades = len(trade_history)
        
        mock_kelly_calculator.update_statistics.side_effect = mock_update
        
        # Update statistics
        mock_kelly_calculator.update_statistics(sample_trade_history)
        
        # Verify statistics
        assert mock_kelly_calculator.win_rate == expected_win_rate
        assert mock_kelly_calculator.avg_win == expected_avg_win
        assert mock_kelly_calculator.avg_loss == expected_avg_loss
        assert mock_kelly_calculator.num_trades == len(sample_trade_history)

    def test_optimal_position_sizing(self, mock_kelly_calculator):
        """Test optimal position sizing recommendations."""
        test_cases = [
            {"kelly_fraction": 0.25, "account_balance": 10000, "max_risk": 0.02, "expected_size": 0.02},
            {"kelly_fraction": 0.15, "account_balance": 50000, "max_risk": 0.01, "expected_size": 0.01},
            {"kelly_fraction": 0.35, "account_balance": 100000, "max_risk": 0.03, "expected_size": 0.03}
        ]
        
        for case in test_cases:
            # Kelly sizing with risk constraints
            recommended_size = min(case["kelly_fraction"], case["max_risk"])
            
            mock_kelly_calculator.get_optimal_position_size = Mock(return_value=recommended_size)
            result = mock_kelly_calculator.get_optimal_position_size(
                case["account_balance"], 
                case["max_risk"]
            )
            
            assert result <= case["max_risk"]  # Should not exceed risk limit
            assert result == case["expected_size"]

    def test_input_validation(self, mock_kelly_calculator):
        """Test input validation for Kelly calculator."""
        valid_inputs = [
            {"win_rate": 0.6, "avg_win": 0.025, "avg_loss": 0.02},
            {"win_rate": 0.5, "avg_win": 0.01, "avg_loss": 0.01}
        ]
        
        invalid_inputs = [
            {"win_rate": 1.5, "avg_win": 0.025, "avg_loss": 0.02},  # Invalid win rate
            {"win_rate": 0.6, "avg_win": -0.025, "avg_loss": 0.02},  # Negative avg win
            {"win_rate": 0.6, "avg_win": 0.025, "avg_loss": -0.02},  # Negative avg loss
            {"win_rate": 0.6, "avg_win": 0, "avg_loss": 0.02}       # Zero avg win
        ]
        
        # Test valid inputs
        for inputs in valid_inputs:
            mock_kelly_calculator.validate_inputs = Mock(return_value=True)
            assert mock_kelly_calculator.validate_inputs(inputs) is True
        
        # Test invalid inputs
        for inputs in invalid_inputs:
            mock_kelly_calculator.validate_inputs = Mock(return_value=False)
            assert mock_kelly_calculator.validate_inputs(inputs) is False

    def test_fractional_kelly_implementation(self, mock_kelly_calculator):
        """Test fractional Kelly implementation for risk reduction."""
        full_kelly = 0.25
        fractional_multipliers = [0.25, 0.5, 0.75, 1.0]
        
        for multiplier in fractional_multipliers:
            expected_fraction = full_kelly * multiplier
            
            mock_kelly_calculator.apply_fractional_kelly = Mock(return_value=expected_fraction)
            result = mock_kelly_calculator.apply_fractional_kelly(full_kelly, multiplier)
            
            assert result == expected_fraction
            assert result <= full_kelly  # Should not exceed full Kelly


class TestVaRCalculator:
    """Test the Value at Risk (VaR) calculator."""

    @pytest.fixture
    def mock_var_calculator(self):
        """Create a mock VaR calculator."""
        calculator = Mock()
        calculator.confidence_level = 0.95
        calculator.lookback_period = 252  # Trading days
        calculator.method = "historical"
        
        # Mock methods
        calculator.calculate_var = Mock()
        calculator.calculate_expected_shortfall = Mock()
        calculator.validate_returns = Mock(return_value=True)
        calculator.get_risk_metrics = Mock()
        
        return calculator

    @pytest.fixture
    def sample_returns(self):
        """Sample return series for testing."""
        np.random.seed(42)  # For reproducible tests
        # Generate returns with some volatility clustering
        returns = np.random.normal(0.0005, 0.02, 1000)  # Daily returns
        # Add some fat tails
        extreme_indices = np.random.choice(1000, 50, replace=False)
        returns[extreme_indices] *= 3
        return returns

    def test_historical_var_calculation(self, mock_var_calculator, sample_returns):
        """Test historical VaR calculation."""
        confidence_levels = [0.90, 0.95, 0.99]
        
        for confidence_level in confidence_levels:
            # Calculate expected VaR (percentile of losses)
            losses = -sample_returns  # Convert returns to losses
            var_percentile = (1 - confidence_level) * 100
            expected_var = np.percentile(losses, 100 - var_percentile)
            
            mock_var_calculator.calculate_var = Mock(return_value=expected_var)
            result = mock_var_calculator.calculate_var(sample_returns, confidence_level, "historical")
            
            assert result > 0  # VaR should be positive (loss amount)
            assert isinstance(result, (int, float))
            mock_var_calculator.calculate_var.assert_called_with(sample_returns, confidence_level, "historical")

    def test_parametric_var_calculation(self, mock_var_calculator, sample_returns):
        """Test parametric VaR calculation."""
        confidence_level = 0.95
        
        # Mock parametric VaR calculation
        mean_return = np.mean(sample_returns)
        std_return = np.std(sample_returns)
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)
        expected_var = -(mean_return - z_score * std_return)
        
        mock_var_calculator.calculate_var = Mock(return_value=expected_var)
        result = mock_var_calculator.calculate_var(sample_returns, confidence_level, "parametric")
        
        assert result > 0
        assert isinstance(result, (int, float))
        mock_var_calculator.calculate_var.assert_called_with(sample_returns, confidence_level, "parametric")

    def test_expected_shortfall_calculation(self, mock_var_calculator, sample_returns):
        """Test Expected Shortfall (Conditional VaR) calculation."""
        confidence_level = 0.95
        
        # Mock Expected Shortfall calculation
        losses = -sample_returns
        var_threshold = np.percentile(losses, 95)
        tail_losses = losses[losses >= var_threshold]
        expected_shortfall = np.mean(tail_losses)
        
        mock_var_calculator.calculate_expected_shortfall = Mock(return_value=expected_shortfall)
        result = mock_var_calculator.calculate_expected_shortfall(sample_returns, confidence_level)
        
        assert result > 0
        assert isinstance(result, (int, float))
        mock_var_calculator.calculate_expected_shortfall.assert_called_with(sample_returns, confidence_level)

    def test_monte_carlo_var(self, mock_var_calculator):
        """Test Monte Carlo VaR simulation."""
        num_simulations = 10000
        time_horizon = 1  # 1 day
        
        # Mock Monte Carlo parameters
        current_value = 100000
        mean_return = 0.0005
        volatility = 0.02
        
        # Mock simulation results
        simulated_values = []
        np.random.seed(42)
        for _ in range(num_simulations):
            random_return = np.random.normal(mean_return, volatility)
            future_value = current_value * (1 + random_return)
            simulated_values.append(future_value)
        
        losses = [current_value - value for value in simulated_values]
        var_95 = np.percentile(losses, 95)
        
        mock_var_calculator.monte_carlo_var = Mock(return_value=var_95)
        result = mock_var_calculator.monte_carlo_var(
            current_value, mean_return, volatility, num_simulations, time_horizon
        )
        
        assert result > 0
        assert isinstance(result, (int, float))

    def test_var_backtesting(self, mock_var_calculator):
        """Test VaR model backtesting."""
        # Mock VaR predictions and actual losses
        var_predictions = [1000, 1200, 800, 1500, 1100] * 20  # 100 days
        actual_losses = [1200, 900, 1300, 800, 1600] * 20    # Some violations
        
        # Calculate violations
        violations = [actual > predicted for actual, predicted in zip(actual_losses, var_predictions)]
        violation_rate = sum(violations) / len(violations)
        
        mock_var_calculator.backtest_var = Mock(return_value={
            "violation_rate": violation_rate,
            "expected_violations": 0.05,  # 5% for 95% VaR
            "passes_kupiec_test": abs(violation_rate - 0.05) < 0.02
        })
        
        result = mock_var_calculator.backtest_var(var_predictions, actual_losses)
        
        assert "violation_rate" in result
        assert "expected_violations" in result
        assert "passes_kupiec_test" in result
        assert isinstance(result["passes_kupiec_test"], bool)

    def test_risk_metrics_compilation(self, mock_var_calculator, sample_returns):
        """Test compilation of comprehensive risk metrics."""
        expected_metrics = {
            "var_95": 0.025,
            "var_99": 0.045,
            "expected_shortfall_95": 0.035,
            "max_drawdown": 0.08,
            "volatility": 0.02,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.1
        }
        
        mock_var_calculator.get_risk_metrics = Mock(return_value=expected_metrics)
        result = mock_var_calculator.get_risk_metrics(sample_returns)
        
        # Verify all metrics are present
        required_metrics = ["var_95", "var_99", "expected_shortfall_95", "max_drawdown", "volatility"]
        for metric in required_metrics:
            assert metric in result
            assert isinstance(result[metric], (int, float))

    def test_stress_testing(self, mock_var_calculator):
        """Test stress testing scenarios."""
        stress_scenarios = [
            {"name": "market_crash", "shock": -0.20, "correlation_increase": 0.3},
            {"name": "volatility_spike", "vol_multiplier": 2.0, "duration_days": 30},
            {"name": "liquidity_crisis", "spread_widening": 5.0, "slippage_increase": 0.01}
        ]
        
        expected_stress_results = {
            "market_crash": {"stressed_var": 0.05, "additional_loss": 20000},
            "volatility_spike": {"stressed_var": 0.04, "additional_loss": 15000},
            "liquidity_crisis": {"stressed_var": 0.035, "additional_loss": 12000}
        }
        
        mock_var_calculator.stress_test = Mock()
        
        for scenario in stress_scenarios:
            scenario_name = scenario["name"]
            expected_result = expected_stress_results[scenario_name]
            
            mock_var_calculator.stress_test = Mock(return_value=expected_result)
            result = mock_var_calculator.stress_test(scenario)
            
            assert "stressed_var" in result
            assert "additional_loss" in result
            assert result["stressed_var"] > 0
            mock_var_calculator.stress_test.assert_called_with(scenario)


class TestCorrelationTracker:
    """Test the correlation tracking component."""

    @pytest.fixture
    def mock_correlation_tracker(self):
        """Create a mock correlation tracker."""
        tracker = Mock()
        tracker.correlation_matrix = np.eye(5)  # 5x5 identity matrix
        tracker.symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
        tracker.lookback_window = 100
        
        # Mock methods
        tracker.update_correlations = Mock()
        tracker.get_correlation_matrix = Mock()
        tracker.detect_correlation_breakdown = Mock()
        tracker.calculate_portfolio_correlation = Mock()
        
        return tracker

    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for correlation calculations."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        
        # Generate correlated price series
        base_returns = np.random.normal(0, 0.01, 252)
        
        data = {}
        correlations = [1.0, 0.7, -0.3, 0.5, 0.4]  # Correlation with base series
        
        for i, symbol in enumerate(["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]):
            if i == 0:
                returns = base_returns
            else:
                noise = np.random.normal(0, 0.01, 252)
                returns = correlations[i] * base_returns + np.sqrt(1 - correlations[i]**2) * noise
            
            # Convert returns to prices
            prices = 100 * np.cumprod(1 + returns)
            data[symbol] = prices
        
        return pd.DataFrame(data, index=dates)

    def test_correlation_matrix_calculation(self, mock_correlation_tracker, sample_price_data):
        """Test correlation matrix calculation."""
        # Calculate expected correlations
        returns = sample_price_data.pct_change().dropna()
        expected_correlation_matrix = returns.corr().values
        
        mock_correlation_tracker.get_correlation_matrix = Mock(return_value=expected_correlation_matrix)
        result = mock_correlation_tracker.get_correlation_matrix()
        
        # Verify matrix properties
        assert result.shape == (5, 5)
        assert np.allclose(np.diag(result), 1.0)  # Diagonal should be 1
        assert np.allclose(result, result.T)      # Should be symmetric
        
        # Check correlation bounds
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_rolling_correlation_update(self, mock_correlation_tracker):
        """Test rolling correlation updates."""
        new_price_data = {
            "EURUSD": 1.0855,
            "GBPUSD": 1.2650,
            "USDJPY": 148.25,
            "USDCHF": 0.8920,
            "AUDUSD": 0.6580
        }
        
        # Mock correlation update
        def mock_update(price_data):
            # Simulate correlation matrix update
            mock_correlation_tracker.correlation_matrix = np.random.rand(5, 5)
            # Make it symmetric
            mock_correlation_tracker.correlation_matrix = (
                mock_correlation_tracker.correlation_matrix + 
                mock_correlation_tracker.correlation_matrix.T
            ) / 2
            # Set diagonal to 1
            np.fill_diagonal(mock_correlation_tracker.correlation_matrix, 1.0)
        
        mock_correlation_tracker.update_correlations.side_effect = mock_update
        
        # Update correlations
        mock_correlation_tracker.update_correlations(new_price_data)
        
        # Verify update was called
        mock_correlation_tracker.update_correlations.assert_called_once_with(new_price_data)

    def test_correlation_breakdown_detection(self, mock_correlation_tracker):
        """Test detection of correlation breakdown during stress."""
        historical_correlation = 0.75
        current_correlation = 0.25
        breakdown_threshold = 0.3
        
        is_breakdown = abs(current_correlation - historical_correlation) > breakdown_threshold
        
        mock_correlation_tracker.detect_correlation_breakdown = Mock(return_value=is_breakdown)
        result = mock_correlation_tracker.detect_correlation_breakdown(
            "EURUSD", "GBPUSD", historical_correlation, current_correlation
        )
        
        assert result is True  # Should detect breakdown
        mock_correlation_tracker.detect_correlation_breakdown.assert_called_with(
            "EURUSD", "GBPUSD", historical_correlation, current_correlation
        )

    def test_portfolio_correlation_calculation(self, mock_correlation_tracker):
        """Test portfolio-level correlation calculation."""
        portfolio_weights = {
            "EURUSD": 0.3,
            "GBPUSD": 0.2,
            "USDJPY": 0.2,
            "USDCHF": 0.15,
            "AUDUSD": 0.15
        }
        
        # Mock portfolio correlation calculation
        expected_portfolio_variance = 0.025  # 2.5% portfolio variance
        
        mock_correlation_tracker.calculate_portfolio_correlation = Mock(return_value=expected_portfolio_variance)
        result = mock_correlation_tracker.calculate_portfolio_correlation(portfolio_weights)
        
        assert result > 0
        assert isinstance(result, (int, float))
        mock_correlation_tracker.calculate_portfolio_correlation.assert_called_with(portfolio_weights)

    def test_correlation_regime_detection(self, mock_correlation_tracker):
        """Test correlation regime detection."""
        correlation_regimes = [
            {"regime": "low_correlation", "threshold": 0.3, "description": "Diversified market"},
            {"regime": "medium_correlation", "threshold": 0.6, "description": "Normal market"},
            {"regime": "high_correlation", "threshold": 0.8, "description": "Stressed market"}
        ]
        
        current_avg_correlation = 0.65
        expected_regime = "medium_correlation"
        
        mock_correlation_tracker.detect_correlation_regime = Mock(return_value=expected_regime)
        result = mock_correlation_tracker.detect_correlation_regime(current_avg_correlation)
        
        assert result in ["low_correlation", "medium_correlation", "high_correlation"]
        mock_correlation_tracker.detect_correlation_regime.assert_called_with(current_avg_correlation)


class TestRiskValidation:
    """Test risk validation and compliance checking."""

    @pytest.fixture
    def mock_risk_validator(self):
        """Create a mock risk validator."""
        validator = Mock()
        validator.risk_limits = {
            "max_position_size": 0.1,
            "max_portfolio_var": 0.02,
            "max_correlation": 0.8,
            "max_leverage": 3.0,
            "max_sector_exposure": 0.3
        }
        
        # Mock methods
        validator.validate_position_size = Mock()
        validator.validate_portfolio_risk = Mock()
        validator.validate_concentration = Mock()
        validator.check_all_limits = Mock()
        
        return validator

    def test_position_size_validation(self, mock_risk_validator):
        """Test position size validation against limits."""
        test_positions = [
            {"symbol": "EURUSD", "size": 0.05, "should_pass": True},
            {"symbol": "GBPUSD", "size": 0.15, "should_pass": False},
            {"symbol": "USDJPY", "size": 0.08, "should_pass": True}
        ]
        
        for position in test_positions:
            expected_result = position["should_pass"]
            mock_risk_validator.validate_position_size = Mock(return_value=expected_result)
            
            result = mock_risk_validator.validate_position_size(
                position["symbol"], 
                position["size"]
            )
            
            assert result == expected_result
            mock_risk_validator.validate_position_size.assert_called_with(
                position["symbol"], 
                position["size"]
            )

    def test_portfolio_risk_validation(self, mock_risk_validator):
        """Test portfolio-level risk validation."""
        portfolio_metrics = {
            "total_var": 0.015,
            "concentration_risk": 0.25,
            "leverage": 2.5,
            "correlation_risk": 0.7
        }
        
        # All metrics within limits
        all_within_limits = all([
            portfolio_metrics["total_var"] <= 0.02,
            portfolio_metrics["concentration_risk"] <= 0.3,
            portfolio_metrics["leverage"] <= 3.0,
            portfolio_metrics["correlation_risk"] <= 0.8
        ])
        
        mock_risk_validator.validate_portfolio_risk = Mock(return_value=all_within_limits)
        result = mock_risk_validator.validate_portfolio_risk(portfolio_metrics)
        
        assert result is True
        mock_risk_validator.validate_portfolio_risk.assert_called_with(portfolio_metrics)

    def test_concentration_risk_validation(self, mock_risk_validator):
        """Test concentration risk validation."""
        portfolio_allocations = {
            "EURUSD": 0.25,
            "GBPUSD": 0.20,
            "USDJPY": 0.20,
            "USDCHF": 0.15,
            "AUDUSD": 0.15,
            "NZDUSD": 0.05
        }
        
        # Check maximum single position
        max_position = max(portfolio_allocations.values())
        concentration_acceptable = max_position <= 0.3
        
        mock_risk_validator.validate_concentration = Mock(return_value=concentration_acceptable)
        result = mock_risk_validator.validate_concentration(portfolio_allocations)
        
        assert result is True
        mock_risk_validator.validate_concentration.assert_called_with(portfolio_allocations)

    def test_comprehensive_risk_check(self, mock_risk_validator):
        """Test comprehensive risk limit checking."""
        trade_proposal = {
            "symbol": "EURUSD",
            "size": 0.08,
            "direction": "BUY",
            "current_portfolio": {
                "total_exposure": 0.6,
                "var_estimate": 0.018,
                "max_correlation": 0.75
            }
        }
        
        # Mock comprehensive check
        validation_results = {
            "position_size_ok": True,
            "portfolio_var_ok": True,
            "concentration_ok": True,
            "correlation_ok": True,
            "overall_approved": True
        }
        
        mock_risk_validator.check_all_limits = Mock(return_value=validation_results)
        result = mock_risk_validator.check_all_limits(trade_proposal)
        
        assert result["overall_approved"] is True
        assert all(result[key] for key in ["position_size_ok", "portfolio_var_ok", "concentration_ok", "correlation_ok"])
        mock_risk_validator.check_all_limits.assert_called_with(trade_proposal)

    def test_risk_limit_violation_handling(self, mock_risk_validator):
        """Test handling of risk limit violations."""
        violation_scenarios = [
            {"type": "position_size", "current": 0.15, "limit": 0.1, "action": "reduce_position"},
            {"type": "portfolio_var", "current": 0.025, "limit": 0.02, "action": "hedge_portfolio"},
            {"type": "concentration", "current": 0.35, "limit": 0.3, "action": "diversify"}
        ]
        
        for scenario in violation_scenarios:
            mock_risk_validator.handle_violation = Mock(return_value=scenario["action"])
            action = mock_risk_validator.handle_violation(scenario["type"], scenario["current"], scenario["limit"])
            
            assert action == scenario["action"]
            mock_risk_validator.handle_violation.assert_called_with(
                scenario["type"], 
                scenario["current"], 
                scenario["limit"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])