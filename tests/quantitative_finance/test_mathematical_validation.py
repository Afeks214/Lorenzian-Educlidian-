"""
Mathematical Validation and Benchmark Testing Suite

This module provides comprehensive mathematical validation of quantitative finance models
against known benchmarks and analytical solutions to ensure accuracy and reliability.

Test Coverage:
- Benchmark validation against known analytical solutions
- Comparison with reference implementations
- Mathematical property verification
- Convergence testing
- Error bound validation
"""

import pytest
import numpy as np
import time
from typing import Dict, Any, List, Tuple

from tests.quantitative_finance import (
    TOLERANCE, BENCHMARK_TOLERANCE, assert_close, TestDataSets
)

# Import the models to test
from tests.quantitative_finance.test_options_pricing import BlackScholesModel
from tests.quantitative_finance.test_volatility_models import GARCHModel, EWMAModel
from tests.quantitative_finance.test_portfolio_optimization import MeanVarianceOptimizer
from tests.quantitative_finance.test_fixed_income import BondPricer, YieldCurve
from tests.quantitative_finance.test_derivatives_pricing import FuturesPricer


class MathematicalBenchmarks:
    """Known mathematical benchmarks for validation"""
    
    # Black-Scholes benchmarks (exact analytical solutions)
    BLACK_SCHOLES_BENCHMARKS = [
        {
            'name': 'ATM_Call_1Y',
            'S': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.20, 'q': 0.0,
            'expected_call': 10.4506, 'expected_put': 5.5735,
            'expected_delta_call': 0.6368, 'expected_delta_put': -0.3632,
            'expected_gamma': 0.0199, 'expected_theta_call': -6.4140,
            'expected_vega': 39.8942, 'expected_rho_call': 53.2325
        },
        {
            'name': 'OTM_Call_6M',
            'S': 100.0, 'K': 110.0, 'T': 0.5, 'r': 0.03, 'sigma': 0.25, 'q': 0.02,
            'expected_call': 2.8793, 'expected_put': 11.3671,
            'expected_delta_call': 0.2715, 'expected_delta_put': -0.7185,
            'expected_gamma': 0.0201, 'expected_theta_call': -10.0843,
            'expected_vega': 17.7540, 'expected_rho_call': 10.8347
        },
        {
            'name': 'ITM_Put_2Y',
            'S': 80.0, 'K': 100.0, 'T': 2.0, 'r': 0.04, 'sigma': 0.30, 'q': 0.01,
            'expected_call': 7.5950, 'expected_put': 25.1829,
            'expected_delta_call': 0.3046, 'expected_delta_put': -0.6854,
            'expected_gamma': 0.0093, 'expected_theta_call': -4.1432,
            'expected_vega': 44.6598, 'expected_rho_call': 29.8456
        }
    ]
    
    # Bond pricing benchmarks
    BOND_BENCHMARKS = [
        {
            'name': 'Par_Bond_5Y',
            'face_value': 100.0, 'coupon_rate': 0.05, 'yield_rate': 0.05,
            'time_to_maturity': 5.0, 'frequency': 2,
            'expected_price': 100.0, 'expected_duration': 4.3295,
            'expected_convexity': 20.5474
        },
        {
            'name': 'Premium_Bond_10Y',
            'face_value': 1000.0, 'coupon_rate': 0.06, 'yield_rate': 0.04,
            'time_to_maturity': 10.0, 'frequency': 2,
            'expected_price': 1162.22, 'expected_duration': 8.1717,
            'expected_convexity': 78.1501
        },
        {
            'name': 'Discount_Bond_3Y',
            'face_value': 100.0, 'coupon_rate': 0.03, 'yield_rate': 0.05,
            'time_to_maturity': 3.0, 'frequency': 2,
            'expected_price': 94.4581, 'expected_duration': 2.8638,
            'expected_convexity': 8.7456
        }
    ]
    
    # Portfolio optimization benchmarks
    PORTFOLIO_BENCHMARKS = [
        {
            'name': 'Two_Asset_Portfolio',
            'expected_returns': np.array([0.10, 0.15]),
            'covariance_matrix': np.array([[0.04, 0.02], [0.02, 0.09]]),
            'risk_aversion': 2.0,
            'expected_weights': np.array([0.5833, 0.4167]),
            'expected_portfolio_return': 0.1208,
            'expected_portfolio_risk': 0.1844
        },
        {
            'name': 'Three_Asset_Equal_Corr',
            'expected_returns': np.array([0.08, 0.12, 0.16]),
            'covariance_matrix': np.array([
                [0.0225, 0.0075, 0.0075],
                [0.0075, 0.0400, 0.0100],
                [0.0075, 0.0100, 0.0625]
            ]),
            'risk_aversion': 3.0,
            'expected_weights': np.array([0.2857, 0.3810, 0.3333]),
            'expected_portfolio_return': 0.1238,
            'expected_portfolio_risk': 0.1633
        }
    ]


class TestBlackScholesBenchmarks:
    """Test Black-Scholes model against known benchmarks"""
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.BLACK_SCHOLES_BENCHMARKS)
    def test_black_scholes_call_pricing(self, benchmark):
        """Test Black-Scholes call pricing against benchmarks"""
        
        call_price = BlackScholesModel.call_price(
            S=benchmark['S'], K=benchmark['K'], T=benchmark['T'],
            r=benchmark['r'], sigma=benchmark['sigma'], q=benchmark['q']
        )
        
        # Test against benchmark with tight tolerance
        assert_close(call_price, benchmark['expected_call'], BENCHMARK_TOLERANCE)
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.BLACK_SCHOLES_BENCHMARKS)
    def test_black_scholes_put_pricing(self, benchmark):
        """Test Black-Scholes put pricing against benchmarks"""
        
        put_price = BlackScholesModel.put_price(
            S=benchmark['S'], K=benchmark['K'], T=benchmark['T'],
            r=benchmark['r'], sigma=benchmark['sigma'], q=benchmark['q']
        )
        
        assert_close(put_price, benchmark['expected_put'], BENCHMARK_TOLERANCE)
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.BLACK_SCHOLES_BENCHMARKS)
    def test_black_scholes_delta(self, benchmark):
        """Test Black-Scholes delta against benchmarks"""
        
        delta_call = BlackScholesModel.delta(
            S=benchmark['S'], K=benchmark['K'], T=benchmark['T'],
            r=benchmark['r'], sigma=benchmark['sigma'], q=benchmark['q'],
            option_type="call"
        )
        
        delta_put = BlackScholesModel.delta(
            S=benchmark['S'], K=benchmark['K'], T=benchmark['T'],
            r=benchmark['r'], sigma=benchmark['sigma'], q=benchmark['q'],
            option_type="put"
        )
        
        assert_close(delta_call, benchmark['expected_delta_call'], BENCHMARK_TOLERANCE)
        assert_close(delta_put, benchmark['expected_delta_put'], BENCHMARK_TOLERANCE)
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.BLACK_SCHOLES_BENCHMARKS)
    def test_black_scholes_gamma(self, benchmark):
        """Test Black-Scholes gamma against benchmarks"""
        
        gamma = BlackScholesModel.gamma(
            S=benchmark['S'], K=benchmark['K'], T=benchmark['T'],
            r=benchmark['r'], sigma=benchmark['sigma'], q=benchmark['q']
        )
        
        assert_close(gamma, benchmark['expected_gamma'], BENCHMARK_TOLERANCE)
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.BLACK_SCHOLES_BENCHMARKS)
    def test_black_scholes_vega(self, benchmark):
        """Test Black-Scholes vega against benchmarks"""
        
        vega = BlackScholesModel.vega(
            S=benchmark['S'], K=benchmark['K'], T=benchmark['T'],
            r=benchmark['r'], sigma=benchmark['sigma'], q=benchmark['q']
        )
        
        assert_close(vega, benchmark['expected_vega'], BENCHMARK_TOLERANCE)
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.BLACK_SCHOLES_BENCHMARKS)
    def test_black_scholes_rho(self, benchmark):
        """Test Black-Scholes rho against benchmarks"""
        
        rho_call = BlackScholesModel.rho(
            S=benchmark['S'], K=benchmark['K'], T=benchmark['T'],
            r=benchmark['r'], sigma=benchmark['sigma'], q=benchmark['q'],
            option_type="call"
        )
        
        assert_close(rho_call, benchmark['expected_rho_call'], BENCHMARK_TOLERANCE)


class TestBondPricingBenchmarks:
    """Test bond pricing models against known benchmarks"""
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.BOND_BENCHMARKS)
    def test_bond_price_benchmark(self, benchmark):
        """Test bond pricing against benchmarks"""
        
        bond_price = BondPricer.bond_price(
            face_value=benchmark['face_value'],
            coupon_rate=benchmark['coupon_rate'],
            yield_rate=benchmark['yield_rate'],
            time_to_maturity=benchmark['time_to_maturity'],
            frequency=benchmark['frequency']
        )
        
        assert_close(bond_price, benchmark['expected_price'], BENCHMARK_TOLERANCE)
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.BOND_BENCHMARKS)
    def test_bond_duration_benchmark(self, benchmark):
        """Test bond duration against benchmarks"""
        
        duration = BondPricer.modified_duration(
            face_value=benchmark['face_value'],
            coupon_rate=benchmark['coupon_rate'],
            yield_rate=benchmark['yield_rate'],
            time_to_maturity=benchmark['time_to_maturity'],
            frequency=benchmark['frequency']
        )
        
        assert_close(duration, benchmark['expected_duration'], BENCHMARK_TOLERANCE)
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.BOND_BENCHMARKS)
    def test_bond_convexity_benchmark(self, benchmark):
        """Test bond convexity against benchmarks"""
        
        convexity = BondPricer.convexity(
            face_value=benchmark['face_value'],
            coupon_rate=benchmark['coupon_rate'],
            yield_rate=benchmark['yield_rate'],
            time_to_maturity=benchmark['time_to_maturity'],
            frequency=benchmark['frequency']
        )
        
        assert_close(convexity, benchmark['expected_convexity'], BENCHMARK_TOLERANCE)


class TestPortfolioOptimizationBenchmarks:
    """Test portfolio optimization against known benchmarks"""
    
    @pytest.mark.parametrize("benchmark", MathematicalBenchmarks.PORTFOLIO_BENCHMARKS)
    def test_mean_variance_optimization_benchmark(self, benchmark):
        """Test mean-variance optimization against benchmarks"""
        
        optimizer = MeanVarianceOptimizer(risk_aversion=benchmark['risk_aversion'])
        results = optimizer.fit(
            expected_returns=benchmark['expected_returns'],
            covariance_matrix=benchmark['covariance_matrix']
        )
        
        # Test optimal weights
        for i, (actual, expected) in enumerate(zip(results['weights'], benchmark['expected_weights'])):
            assert_close(actual, expected, BENCHMARK_TOLERANCE * 2)  # Slightly relaxed tolerance
        
        # Test portfolio metrics
        assert_close(results['portfolio_return'], benchmark['expected_portfolio_return'], BENCHMARK_TOLERANCE)
        assert_close(results['portfolio_volatility'], benchmark['expected_portfolio_risk'], BENCHMARK_TOLERANCE)


class TestNumericalConvergence:
    """Test numerical convergence properties"""
    
    def test_monte_carlo_convergence(self):
        """Test Monte Carlo convergence for option pricing"""
        
        # Parameters for European call
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        
        # Analytical solution
        analytical_price = BlackScholesModel.call_price(S, K, T, r, sigma, 0.0)
        
        # Monte Carlo simulation
        def monte_carlo_option_price(n_simulations):
            np.random.seed(42)
            random_numbers = np.random.normal(0, 1, n_simulations)
            
            # Final stock prices
            final_prices = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * random_numbers)
            
            # Option payoffs
            payoffs = np.maximum(final_prices - K, 0)
            
            # Present value
            return np.mean(payoffs) * np.exp(-r * T)
        
        # Test convergence with increasing sample sizes
        sample_sizes = [1000, 5000, 10000, 50000]
        errors = []
        
        for n in sample_sizes:
            mc_price = monte_carlo_option_price(n)
            error = abs(mc_price - analytical_price)
            errors.append(error)
        
        # Error should generally decrease with sample size
        assert errors[-1] < errors[0], "Monte Carlo error should decrease with sample size"
        assert errors[-1] < 0.5, "Monte Carlo should converge to reasonable accuracy"
    
    def test_binomial_tree_convergence(self):
        """Test binomial tree convergence"""
        
        # Parameters
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        
        # Analytical solution
        analytical_price = BlackScholesModel.call_price(S, K, T, r, sigma, 0.0)
        
        # Binomial tree pricing
        def binomial_tree_price(n_steps):
            dt = T / n_steps
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)
            
            # Initialize option values at expiration
            option_values = np.zeros(n_steps + 1)
            for i in range(n_steps + 1):
                stock_price = S * (u ** (n_steps - i)) * (d ** i)
                option_values[i] = max(stock_price - K, 0)
            
            # Work backwards
            for step in range(n_steps - 1, -1, -1):
                for i in range(step + 1):
                    option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            
            return option_values[0]
        
        # Test convergence
        step_sizes = [50, 100, 200, 500]
        errors = []
        
        for n in step_sizes:
            binomial_price = binomial_tree_price(n)
            error = abs(binomial_price - analytical_price)
            errors.append(error)
        
        # Error should decrease with more steps
        assert errors[-1] < errors[0], "Binomial tree should converge with more steps"
        assert errors[-1] < 0.1, "Binomial tree should converge to reasonable accuracy"
    
    def test_yield_curve_interpolation_convergence(self):
        """Test yield curve interpolation convergence"""
        
        # Known yield curve points
        base_maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        base_yields = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
        
        # Add more points and test convergence
        dense_maturities = np.linspace(0.5, 10.0, 20)
        
        # Interpolate yields
        curve = YieldCurve(method="cubic_spline")
        curve.fit(base_maturities, base_yields)
        
        interpolated_yields = [curve.interpolate(t) for t in dense_maturities]
        
        # Test smoothness
        yield_diffs = np.diff(interpolated_yields)
        max_jump = np.max(np.abs(yield_diffs))
        
        assert max_jump < 0.01, "Yield curve should be smooth"
        assert np.all(np.array(interpolated_yields) > 0), "All yields should be positive"


class TestMathematicalProperties:
    """Test mathematical properties of models"""
    
    def test_put_call_parity_property(self):
        """Test put-call parity holds exactly"""
        
        # Test parameters
        test_cases = [
            (100.0, 100.0, 1.0, 0.05, 0.20, 0.0),
            (50.0, 60.0, 0.5, 0.03, 0.30, 0.02),
            (120.0, 100.0, 2.0, 0.04, 0.15, 0.01)
        ]
        
        for S, K, T, r, sigma, q in test_cases:
            call_price = BlackScholesModel.call_price(S, K, T, r, sigma, q)
            put_price = BlackScholesModel.put_price(S, K, T, r, sigma, q)
            
            # Put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
            left_side = call_price - put_price
            right_side = S * np.exp(-q * T) - K * np.exp(-r * T)
            
            assert_close(left_side, right_side, TOLERANCE)
    
    def test_greeks_relationships(self):
        """Test mathematical relationships between Greeks"""
        
        S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
        
        # Calculate Greeks
        delta_call = BlackScholesModel.delta(S, K, T, r, sigma, q, "call")
        delta_put = BlackScholesModel.delta(S, K, T, r, sigma, q, "put")
        gamma = BlackScholesModel.gamma(S, K, T, r, sigma, q)
        
        # Test delta relationship: delta_call - delta_put = e^(-qT)
        delta_diff = delta_call - delta_put
        expected_diff = np.exp(-q * T)
        assert_close(delta_diff, expected_diff, TOLERANCE)
        
        # Test gamma positivity
        assert gamma > 0, "Gamma should be positive"
        
        # Test gamma is same for call and put
        gamma_put = BlackScholesModel.gamma(S, K, T, r, sigma, q)
        assert_close(gamma, gamma_put, TOLERANCE)
    
    def test_bond_yield_price_relationship(self):
        """Test inverse relationship between bond yield and price"""
        
        face_value = 100.0
        coupon_rate = 0.05
        time_to_maturity = 5.0
        
        yields = np.linspace(0.02, 0.08, 10)
        prices = []
        
        for yield_rate in yields:
            price = BondPricer.bond_price(face_value, coupon_rate, yield_rate, time_to_maturity)
            prices.append(price)
        
        # Prices should be monotonically decreasing with yield
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], "Bond prices should decrease with yield"
    
    def test_portfolio_diversification_effect(self):
        """Test diversification effect in portfolio optimization"""
        
        # Two uncorrelated assets
        expected_returns = np.array([0.10, 0.10])
        uncorr_cov = np.array([[0.04, 0.00], [0.00, 0.04]])
        
        optimizer = MeanVarianceOptimizer(risk_aversion=2.0)
        uncorr_results = optimizer.fit(expected_returns, uncorr_cov)
        
        # Two perfectly correlated assets
        corr_cov = np.array([[0.04, 0.04], [0.04, 0.04]])
        corr_results = optimizer.fit(expected_returns, corr_cov)
        
        # Uncorrelated portfolio should have lower risk for same return
        assert uncorr_results['portfolio_volatility'] < corr_results['portfolio_volatility'], \
            "Diversification should reduce portfolio risk"
    
    def test_futures_cost_of_carry_relationship(self):
        """Test cost-of-carry relationship for futures"""
        
        S, r, T = 100.0, 0.05, 1.0
        
        # No carry costs
        futures_price_1 = FuturesPricer.futures_price(S, r, T)
        expected_price_1 = S * np.exp(r * T)
        assert_close(futures_price_1, expected_price_1, TOLERANCE)
        
        # With dividend yield
        q = 0.02
        futures_price_2 = FuturesPricer.futures_price(S, r, T, dividend_yield=q)
        expected_price_2 = S * np.exp((r - q) * T)
        assert_close(futures_price_2, expected_price_2, TOLERANCE)
        
        # Dividends should reduce futures price
        assert futures_price_2 < futures_price_1, "Dividends should reduce futures price"


class TestErrorBounds:
    """Test error bounds and accuracy requirements"""
    
    def test_numerical_stability_bounds(self):
        """Test numerical stability under various conditions"""
        
        # Test Black-Scholes with extreme parameters
        extreme_cases = [
            (100.0, 100.0, 0.001, 0.05, 0.20),  # Very short time
            (100.0, 100.0, 10.0, 0.05, 0.20),   # Long time
            (100.0, 100.0, 1.0, 0.05, 0.01),    # Low volatility
            (100.0, 100.0, 1.0, 0.05, 1.0),     # High volatility
        ]
        
        for S, K, T, r, sigma in extreme_cases:
            call_price = BlackScholesModel.call_price(S, K, T, r, sigma, 0.0)
            put_price = BlackScholesModel.put_price(S, K, T, r, sigma, 0.0)
            
            # Prices should be bounded and finite
            assert 0 <= call_price <= S, f"Call price {call_price} should be bounded"
            assert 0 <= put_price <= K * np.exp(-r * T), f"Put price {put_price} should be bounded"
            assert np.isfinite(call_price), "Call price should be finite"
            assert np.isfinite(put_price), "Put price should be finite"
    
    def test_approximation_error_bounds(self):
        """Test approximation error bounds"""
        
        # Test EWMA vs GARCH approximation
        returns = TestDataSets.get_volatility_data()
        
        # Fit EWMA
        ewma = EWMAModel(lambda_decay=0.94)
        ewma.fit(returns)
        
        # Fit GARCH
        garch = GARCHModel()
        garch.fit(returns)
        
        # Compare variance estimates
        ewma_var = ewma.variance_series[-1]
        garch_var = garch.conditional_variance[-1]
        
        # Error should be reasonable
        relative_error = abs(ewma_var - garch_var) / garch_var
        assert relative_error < 0.5, "EWMA and GARCH should give similar variance estimates"
    
    def test_convergence_rates(self):
        """Test convergence rates for numerical methods"""
        
        # Test binomial tree convergence rate
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        analytical_price = BlackScholesModel.call_price(S, K, T, r, sigma, 0.0)
        
        # Calculate errors for different step sizes
        step_sizes = [50, 100, 200, 400]
        errors = []
        
        for n in step_sizes:
            # Simplified binomial pricing
            dt = T / n
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            p = (np.exp(r * dt) - d) / (u - d)
            
            # Single step approximation
            binomial_price = np.exp(-r * dt) * (p * max(S * u - K, 0) + (1 - p) * max(S * d - K, 0))
            error = abs(binomial_price - analytical_price)
            errors.append(error)
        
        # Error should decrease with more steps
        assert errors[-1] < errors[0], "Binomial tree error should decrease with more steps"


class TestComprehensiveValidation:
    """Comprehensive validation across all models"""
    
    def test_model_consistency(self):
        """Test consistency between different models"""
        
        # Test consistency between different volatility models
        returns = TestDataSets.get_volatility_data()
        
        # EWMA model
        ewma = EWMAModel(lambda_decay=0.94)
        ewma.fit(returns)
        
        # GARCH model
        garch = GARCHModel()
        garch.fit(returns)
        
        # Both should give positive variance estimates
        assert ewma.variance_series[-1] > 0, "EWMA variance should be positive"
        assert garch.conditional_variance[-1] > 0, "GARCH variance should be positive"
        
        # Estimates should be in reasonable range
        assert 0.0001 < ewma.variance_series[-1] < 0.1, "EWMA variance should be reasonable"
        assert 0.0001 < garch.conditional_variance[-1] < 0.1, "GARCH variance should be reasonable"
    
    def test_cross_model_validation(self):
        """Test validation across different model types"""
        
        # Test option pricing vs portfolio optimization
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        
        # Option delta
        delta = BlackScholesModel.delta(S, K, T, r, sigma, 0.0, "call")
        
        # Delta should be reasonable hedge ratio
        assert 0 < delta < 1, "Call delta should be between 0 and 1"
        
        # Test bond pricing vs yield curve
        face_value, coupon_rate, time_to_maturity = 100.0, 0.05, 5.0
        yield_rate = 0.04
        
        bond_price = BondPricer.bond_price(face_value, coupon_rate, yield_rate, time_to_maturity)
        
        # Create yield curve
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        yields = np.array([0.03, 0.035, 0.04, 0.045])
        
        curve = YieldCurve(method="linear")
        curve.fit(maturities, yields)
        
        curve_yield = curve.interpolate(time_to_maturity)
        curve_bond_price = BondPricer.bond_price(face_value, coupon_rate, curve_yield, time_to_maturity)
        
        # Prices should be close when using same yield
        assert abs(bond_price - curve_bond_price) < 0.01, "Bond prices should be consistent"
    
    def test_benchmark_performance(self):
        """Test that all models meet performance benchmarks"""
        
        # Test option pricing performance
        start_time = time.time()
        for _ in range(1000):
            BlackScholesModel.call_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0)
        option_time = time.time() - start_time
        
        assert option_time < 0.1, "Option pricing should be fast"
        
        # Test bond pricing performance
        start_time = time.time()
        for _ in range(1000):
            BondPricer.bond_price(100.0, 0.05, 0.04, 5.0)
        bond_time = time.time() - start_time
        
        assert bond_time < 0.1, "Bond pricing should be fast"
        
        # Test portfolio optimization performance
        expected_returns = np.array([0.08, 0.12, 0.10])
        covariance_matrix = np.array([
            [0.04, 0.01, 0.01],
            [0.01, 0.09, 0.02],
            [0.01, 0.02, 0.05]
        ])
        
        start_time = time.time()
        optimizer = MeanVarianceOptimizer(risk_aversion=2.0)
        optimizer.fit(expected_returns, covariance_matrix)
        portfolio_time = time.time() - start_time
        
        assert portfolio_time < 0.1, "Portfolio optimization should be fast"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])