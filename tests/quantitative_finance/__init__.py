"""
Quantitative Finance Models Testing Suite

This module contains comprehensive tests for quantitative finance models
including options pricing, volatility modeling, portfolio optimization,
fixed income models, and derivatives pricing.

Test Coverage:
- Options Pricing Models (Black-Scholes, Heston, etc.)
- Volatility Models (GARCH, EWMA, stochastic volatility)
- Portfolio Optimization (mean-variance, Black-Litterman)
- Fixed Income Models (yield curves, duration, convexity)
- Derivatives Pricing (futures, swaps, credit derivatives)

All tests ensure mathematical accuracy and numerical stability
under various market conditions.
"""

# Test configuration
TOLERANCE = 1e-6
MONTE_CARLO_SAMPLES = 10000
BENCHMARK_TOLERANCE = 1e-4

# Market data constants for testing
RISK_FREE_RATE = 0.03
DIVIDEND_YIELD = 0.02
VOLATILITY = 0.20
SPOT_PRICE = 100.0
STRIKE_PRICE = 100.0
TIME_TO_MATURITY = 1.0

# Common test utilities
import numpy as np
import pytest
from typing import Dict, Any, List, Tuple


def assert_close(actual: float, expected: float, tolerance: float = TOLERANCE) -> None:
    """Assert that two values are close within tolerance"""
    assert abs(actual - expected) < tolerance, f"Expected {expected}, got {actual}"


def generate_test_returns(n_samples: int = 1000, volatility: float = 0.20) -> np.ndarray:
    """Generate synthetic returns for testing"""
    np.random.seed(42)
    return np.random.normal(0, volatility / np.sqrt(252), n_samples)


def generate_price_series(
    initial_price: float = 100.0,
    returns: np.ndarray = None,
    n_samples: int = 1000
) -> np.ndarray:
    """Generate price series from returns"""
    if returns is None:
        returns = generate_test_returns(n_samples)
    
    prices = np.zeros(len(returns) + 1)
    prices[0] = initial_price
    
    for i in range(len(returns)):
        prices[i + 1] = prices[i] * (1 + returns[i])
    
    return prices


def create_correlation_matrix(n_assets: int, correlation: float = 0.5) -> np.ndarray:
    """Create correlation matrix for testing"""
    corr_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            corr_matrix[i, j] = correlation
            corr_matrix[j, i] = correlation
    return corr_matrix


def validate_option_greeks(
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
    rho: float,
    option_type: str = "call"
) -> bool:
    """Validate option Greeks satisfy basic mathematical relationships"""
    # Delta bounds
    if option_type == "call":
        if not (0 <= delta <= 1):
            return False
    else:  # put
        if not (-1 <= delta <= 0):
            return False
    
    # Gamma is always positive
    if gamma < 0:
        return False
    
    # Theta is typically negative for long options
    # (time decay reduces option value)
    
    # Vega is always positive
    if vega < 0:
        return False
    
    # Rho bounds depend on option type
    if option_type == "call":
        if rho < 0:
            return False
    else:  # put
        if rho > 0:
            return False
    
    return True


# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "black_scholes_call": 0.001,  # 1ms max
    "black_scholes_put": 0.001,
    "heston_pricing": 0.100,  # 100ms max
    "garch_estimation": 0.050,
    "portfolio_optimization": 0.500,
    "yield_curve_construction": 0.010,
    "monte_carlo_var": 0.200
}


# Test data sets
class TestDataSets:
    """Standard test data sets for quantitative finance models"""
    
    @staticmethod
    def get_market_data() -> Dict[str, Any]:
        """Get standard market data for testing"""
        return {
            "spot_price": SPOT_PRICE,
            "strike_price": STRIKE_PRICE,
            "time_to_maturity": TIME_TO_MATURITY,
            "risk_free_rate": RISK_FREE_RATE,
            "dividend_yield": DIVIDEND_YIELD,
            "volatility": VOLATILITY
        }
    
    @staticmethod
    def get_yield_curve_data() -> Dict[str, List[float]]:
        """Get yield curve data for testing"""
        return {
            "maturities": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            "yields": [0.02, 0.025, 0.03, 0.035, 0.04, 0.038, 0.036]
        }
    
    @staticmethod
    def get_portfolio_data() -> Dict[str, Any]:
        """Get portfolio data for optimization testing"""
        n_assets = 5
        returns = np.random.multivariate_normal(
            mean=[0.08, 0.10, 0.12, 0.06, 0.09],
            cov=np.diag([0.16, 0.25, 0.36, 0.09, 0.20]) * 0.01,
            size=252
        )
        
        return {
            "returns": returns,
            "expected_returns": returns.mean(axis=0),
            "covariance_matrix": np.cov(returns.T),
            "asset_names": ["Stock A", "Stock B", "Stock C", "Bond A", "REIT"]
        }
    
    @staticmethod
    def get_volatility_data() -> np.ndarray:
        """Get volatility data for modeling"""
        np.random.seed(42)
        # Generate GARCH-like volatility clustering
        n_samples = 1000
        returns = np.zeros(n_samples)
        volatility = np.zeros(n_samples)
        volatility[0] = 0.02
        
        for i in range(1, n_samples):
            # GARCH(1,1) process
            volatility[i] = np.sqrt(
                0.000001 + 0.05 * returns[i-1]**2 + 0.9 * volatility[i-1]**2
            )
            returns[i] = volatility[i] * np.random.normal()
        
        return returns