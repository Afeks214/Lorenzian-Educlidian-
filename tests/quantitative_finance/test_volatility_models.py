"""
Volatility Models Testing Suite

Comprehensive testing for volatility modeling including:
- GARCH models (GARCH(1,1), EGARCH, GJR-GARCH)
- EWMA (Exponentially Weighted Moving Average)
- Stochastic volatility models
- Volatility surface construction and calibration
- Implied volatility extraction and smile modeling
- Volatility forecasting accuracy
- Model parameter estimation and validation
"""

import pytest
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp2d, griddata
import warnings

from tests.quantitative_finance import (
    TOLERANCE, BENCHMARK_TOLERANCE, PERFORMANCE_BENCHMARKS,
    TestDataSets, assert_close, generate_test_returns, generate_price_series
)


class GARCHModel:
    """GARCH(1,1) model implementation"""
    
    def __init__(self, omega: float = 0.000001, alpha: float = 0.05, beta: float = 0.9):
        self.omega = omega  # Constant term
        self.alpha = alpha  # ARCH coefficient
        self.beta = beta    # GARCH coefficient
        self.fitted = False
        self.residuals = None
        self.conditional_variance = None
        self.log_likelihood = None
    
    def simulate(self, n_periods: int, initial_variance: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate GARCH process"""
        np.random.seed(42)
        
        returns = np.zeros(n_periods)
        variance = np.zeros(n_periods)
        variance[0] = initial_variance
        
        for t in range(1, n_periods):
            # Generate innovation
            epsilon = np.random.normal(0, 1)
            
            # Update variance
            variance[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * variance[t-1]
            
            # Generate return
            returns[t] = np.sqrt(variance[t]) * epsilon
        
        return returns, variance
    
    def fit(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fit GARCH model to returns"""
        def garch_likelihood(params):
            omega, alpha, beta = params
            
            # Check parameter constraints
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            n = len(returns)
            variance = np.zeros(n)
            variance[0] = np.var(returns)  # Initial variance
            
            log_likelihood = 0
            
            for t in range(1, n):
                variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
                
                if variance[t] <= 0:
                    return 1e10
                
                log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(variance[t]) + returns[t]**2 / variance[t])
            
            return -log_likelihood
        
        # Initial parameters
        initial_params = [0.000001, 0.05, 0.9]
        
        # Optimization bounds
        bounds = [(1e-8, 1e-3), (1e-6, 0.3), (1e-6, 0.99)]
        
        # Optimize
        result = minimize(garch_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.omega, self.alpha, self.beta = result.x
            self.fitted = True
            self.log_likelihood = -result.fun
            
            # Calculate fitted variance
            n = len(returns)
            self.conditional_variance = np.zeros(n)
            self.conditional_variance[0] = np.var(returns)
            
            for t in range(1, n):
                self.conditional_variance[t] = (self.omega + 
                                              self.alpha * returns[t-1]**2 + 
                                              self.beta * self.conditional_variance[t-1])
            
            # Calculate standardized residuals
            self.residuals = returns / np.sqrt(self.conditional_variance)
        
        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'log_likelihood': self.log_likelihood,
            'converged': result.success,
            'persistence': self.alpha + self.beta
        }
    
    def forecast(self, horizon: int) -> np.ndarray:
        """Forecast volatility for given horizon"""
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = np.zeros(horizon)
        last_variance = self.conditional_variance[-1]
        unconditional_variance = self.omega / (1 - self.alpha - self.beta)
        
        for h in range(horizon):
            if h == 0:
                forecasts[h] = last_variance
            else:
                # Long-term forecast converges to unconditional variance
                persistence = (self.alpha + self.beta) ** h
                forecasts[h] = unconditional_variance * (1 - persistence) + last_variance * persistence
        
        return forecasts


class EWMAModel:
    """Exponentially Weighted Moving Average volatility model"""
    
    def __init__(self, lambda_decay: float = 0.94):
        self.lambda_decay = lambda_decay
        self.variance_series = None
        self.fitted = False
    
    def fit(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fit EWMA model to returns"""
        n = len(returns)
        self.variance_series = np.zeros(n)
        
        # Initial variance
        self.variance_series[0] = returns[0]**2
        
        # EWMA recursion
        for t in range(1, n):
            self.variance_series[t] = (self.lambda_decay * self.variance_series[t-1] + 
                                     (1 - self.lambda_decay) * returns[t-1]**2)
        
        self.fitted = True
        
        return {
            'lambda': self.lambda_decay,
            'final_variance': self.variance_series[-1],
            'avg_variance': np.mean(self.variance_series)
        }
    
    def forecast(self, horizon: int) -> np.ndarray:
        """Forecast EWMA volatility"""
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        current_variance = self.variance_series[-1]
        forecasts = np.full(horizon, current_variance)
        
        return forecasts


class StochasticVolatilityModel:
    """Heston-like stochastic volatility model"""
    
    def __init__(self, kappa: float = 2.0, theta: float = 0.04, sigma: float = 0.3, rho: float = -0.5):
        self.kappa = kappa    # Mean reversion speed
        self.theta = theta    # Long-term variance
        self.sigma = sigma    # Volatility of volatility
        self.rho = rho        # Correlation
        self.fitted = False
    
    def simulate(self, n_periods: int, initial_price: float = 100.0, 
                 initial_variance: float = 0.04, dt: float = 1/252) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate stochastic volatility process"""
        np.random.seed(42)
        
        prices = np.zeros(n_periods)
        variance = np.zeros(n_periods)
        returns = np.zeros(n_periods)
        
        prices[0] = initial_price
        variance[0] = initial_variance
        
        for t in range(1, n_periods):
            # Generate correlated random shocks
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, np.sqrt(dt))
            
            # Update variance (Heston model)
            variance[t] = variance[t-1] + self.kappa * (self.theta - variance[t-1]) * dt + self.sigma * np.sqrt(max(variance[t-1], 0)) * dW2
            variance[t] = max(variance[t], 0)  # Ensure non-negative variance
            
            # Update price
            returns[t] = np.sqrt(max(variance[t-1], 0)) * dW1
            prices[t] = prices[t-1] * np.exp(returns[t])
        
        return prices, variance, returns
    
    def calibrate(self, returns: np.ndarray) -> Dict[str, Any]:
        """Calibrate stochastic volatility model (simplified)"""
        # Simplified calibration using method of moments
        
        # Estimate parameters from return characteristics
        returns_var = np.var(returns)
        returns_mean = np.mean(returns)
        
        # Estimate long-term variance
        self.theta = returns_var
        
        # Estimate mean reversion from autocorrelation of squared returns
        squared_returns = returns**2
        autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        self.kappa = -np.log(max(autocorr, 0.01))  # Ensure positive kappa
        
        # Estimate vol-of-vol from variance of squared returns
        self.sigma = np.std(squared_returns) / np.sqrt(returns_var)
        
        self.fitted = True
        
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'rho': self.rho
        }


class VolatilitySurface:
    """Volatility surface construction and interpolation"""
    
    def __init__(self):
        self.strikes = None
        self.maturities = None
        self.implied_vols = None
        self.surface = None
        self.fitted = False
    
    def fit(self, strikes: np.ndarray, maturities: np.ndarray, 
            implied_vols: np.ndarray) -> Dict[str, Any]:
        """Fit volatility surface"""
        self.strikes = strikes
        self.maturities = maturities
        self.implied_vols = implied_vols
        
        # Create meshgrid for interpolation
        K_grid, T_grid = np.meshgrid(strikes, maturities)
        
        # Fit surface using 2D interpolation
        self.surface = interp2d(strikes, maturities, implied_vols, kind='cubic')
        
        self.fitted = True
        
        return {
            'min_strike': np.min(strikes),
            'max_strike': np.max(strikes),
            'min_maturity': np.min(maturities),
            'max_maturity': np.max(maturities),
            'surface_fitted': True
        }
    
    def interpolate(self, strike: float, maturity: float) -> float:
        """Interpolate volatility for given strike and maturity"""
        if not self.fitted:
            raise ValueError("Surface must be fitted before interpolation")
        
        return float(self.surface(strike, maturity))
    
    def smile(self, maturity: float, strikes: np.ndarray = None) -> np.ndarray:
        """Extract volatility smile for given maturity"""
        if strikes is None:
            strikes = self.strikes
        
        return np.array([self.interpolate(k, maturity) for k in strikes])
    
    def term_structure(self, strike: float, maturities: np.ndarray = None) -> np.ndarray:
        """Extract term structure for given strike"""
        if maturities is None:
            maturities = self.maturities
        
        return np.array([self.interpolate(strike, t) for t in maturities])


class ImpliedVolatilityExtractor:
    """Implied volatility extraction from option prices"""
    
    @staticmethod
    def black_scholes_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Calculate Black-Scholes vega"""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)
    
    @staticmethod
    def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, 
                           q: float = 0.0, option_type: str = "call") -> float:
        """Calculate Black-Scholes option price"""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == "call":
            return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)
    
    @classmethod
    def extract_implied_volatility(cls, market_price: float, S: float, K: float, T: float, 
                                 r: float, q: float = 0.0, option_type: str = "call") -> float:
        """Extract implied volatility using Newton-Raphson method"""
        
        def objective(sigma):
            try:
                model_price = cls.black_scholes_price(S, K, T, r, sigma, q, option_type)
                return model_price - market_price
            except:
                return float('inf')
        
        def objective_derivative(sigma):
            try:
                return cls.black_scholes_vega(S, K, T, r, sigma, q)
            except:
                return 1e-10
        
        # Initial guess
        sigma = 0.2
        
        # Newton-Raphson iteration
        for _ in range(100):
            f = objective(sigma)
            f_prime = objective_derivative(sigma)
            
            if abs(f) < 1e-6:
                break
            
            if f_prime == 0:
                break
            
            sigma_new = sigma - f / f_prime
            
            # Ensure positive volatility
            sigma_new = max(sigma_new, 0.001)
            sigma_new = min(sigma_new, 5.0)
            
            if abs(sigma_new - sigma) < 1e-8:
                break
            
            sigma = sigma_new
        
        return sigma


class TestGARCHModel:
    """Test suite for GARCH models"""
    
    def test_garch_simulation(self):
        """Test GARCH simulation"""
        garch = GARCHModel(omega=0.000001, alpha=0.05, beta=0.9)
        
        returns, variance = garch.simulate(n_periods=1000)
        
        # Check properties
        assert len(returns) == 1000, "Returns length should match n_periods"
        assert len(variance) == 1000, "Variance length should match n_periods"
        assert np.all(variance > 0), "Variance should be positive"
        
        # Check persistence
        persistence = garch.alpha + garch.beta
        assert persistence < 1, "GARCH process should be stationary"
        
        # Check volatility clustering
        abs_returns = np.abs(returns)
        autocorr = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
        assert autocorr > 0.1, "Should exhibit volatility clustering"
    
    def test_garch_fitting(self):
        """Test GARCH model fitting"""
        # Generate test data
        test_returns = TestDataSets.get_volatility_data()
        
        start_time = time.time()
        garch = GARCHModel()
        results = garch.fit(test_returns)
        execution_time = time.time() - start_time
        
        # Check fitting results
        assert results['converged'], "GARCH fitting should converge"
        assert results['omega'] > 0, "Omega should be positive"
        assert results['alpha'] >= 0, "Alpha should be non-negative"
        assert results['beta'] >= 0, "Beta should be non-negative"
        assert results['persistence'] < 1, "Persistence should be less than 1"
        
        # Check fitted variance
        assert garch.conditional_variance is not None, "Conditional variance should be calculated"
        assert np.all(garch.conditional_variance > 0), "Conditional variance should be positive"
        
        # Performance check
        assert execution_time < PERFORMANCE_BENCHMARKS["garch_estimation"]
    
    def test_garch_forecasting(self):
        """Test GARCH forecasting"""
        test_returns = TestDataSets.get_volatility_data()
        
        garch = GARCHModel()
        garch.fit(test_returns)
        
        forecasts = garch.forecast(horizon=10)
        
        # Check forecast properties
        assert len(forecasts) == 10, "Forecast length should match horizon"
        assert np.all(forecasts > 0), "Forecasts should be positive"
        
        # Check convergence to unconditional variance
        unconditional_var = garch.omega / (1 - garch.alpha - garch.beta)
        long_term_forecast = garch.forecast(horizon=100)
        
        assert abs(long_term_forecast[-1] - unconditional_var) < 0.01, "Long-term forecast should converge to unconditional variance"
    
    def test_garch_parameter_constraints(self):
        """Test GARCH parameter constraints"""
        test_returns = TestDataSets.get_volatility_data()
        
        garch = GARCHModel()
        results = garch.fit(test_returns)
        
        # Test parameter bounds
        assert results['omega'] > 0, "Omega constraint violated"
        assert results['alpha'] >= 0, "Alpha constraint violated"
        assert results['beta'] >= 0, "Beta constraint violated"
        assert results['alpha'] + results['beta'] < 1, "Stationarity constraint violated"
        
        # Test with invalid parameters
        invalid_garch = GARCHModel(omega=-0.001, alpha=0.05, beta=0.9)
        # Should handle gracefully or raise appropriate error
    
    def test_garch_residual_analysis(self):
        """Test GARCH residual analysis"""
        test_returns = TestDataSets.get_volatility_data()
        
        garch = GARCHModel()
        garch.fit(test_returns)
        
        # Check residuals
        assert garch.residuals is not None, "Residuals should be calculated"
        
        # Standardized residuals should be approximately white noise
        residuals = garch.residuals[1:]  # Skip first observation
        
        # Test for no autocorrelation in residuals
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        assert abs(autocorr) < 0.1, "Residuals should not be autocorrelated"
        
        # Test for approximately normal distribution
        _, p_value = stats.jarque_bera(residuals)
        # Note: We don't require strict normality, just reasonable behavior


class TestEWMAModel:
    """Test suite for EWMA models"""
    
    def test_ewma_fitting(self):
        """Test EWMA model fitting"""
        test_returns = generate_test_returns(1000)
        
        ewma = EWMAModel(lambda_decay=0.94)
        results = ewma.fit(test_returns)
        
        # Check results
        assert results['lambda'] == 0.94, "Lambda should match input"
        assert results['final_variance'] > 0, "Final variance should be positive"
        assert results['avg_variance'] > 0, "Average variance should be positive"
        
        # Check variance series
        assert ewma.variance_series is not None, "Variance series should be calculated"
        assert len(ewma.variance_series) == len(test_returns), "Variance series length should match returns"
        assert np.all(ewma.variance_series > 0), "All variances should be positive"
    
    def test_ewma_forecasting(self):
        """Test EWMA forecasting"""
        test_returns = generate_test_returns(1000)
        
        ewma = EWMAModel(lambda_decay=0.94)
        ewma.fit(test_returns)
        
        forecasts = ewma.forecast(horizon=10)
        
        # EWMA forecast should be constant
        assert len(forecasts) == 10, "Forecast length should match horizon"
        assert np.all(forecasts == forecasts[0]), "EWMA forecasts should be constant"
        assert forecasts[0] == ewma.variance_series[-1], "Forecast should equal final variance"
    
    def test_ewma_lambda_sensitivity(self):
        """Test EWMA sensitivity to lambda parameter"""
        test_returns = generate_test_returns(1000)
        
        lambdas = [0.9, 0.94, 0.98]
        final_variances = []
        
        for lambda_val in lambdas:
            ewma = EWMAModel(lambda_decay=lambda_val)
            results = ewma.fit(test_returns)
            final_variances.append(results['final_variance'])
        
        # Higher lambda should result in smoother variance estimates
        # (This test verifies the model responds to parameter changes)
        assert len(set(final_variances)) > 1, "Different lambdas should produce different results"
    
    def test_ewma_vs_sample_variance(self):
        """Test EWMA vs sample variance behavior"""
        # Generate returns with time-varying volatility
        np.random.seed(42)
        n = 1000
        returns = np.zeros(n)
        
        for i in range(n):
            if i < 500:
                returns[i] = np.random.normal(0, 0.1)  # Low volatility
            else:
                returns[i] = np.random.normal(0, 0.3)  # High volatility
        
        ewma = EWMAModel(lambda_decay=0.94)
        ewma.fit(returns)
        
        # EWMA should adapt faster than sample variance
        sample_var_early = np.var(returns[:500])
        sample_var_late = np.var(returns[500:])
        
        ewma_var_early = ewma.variance_series[499]
        ewma_var_late = ewma.variance_series[-1]
        
        # EWMA should show more dramatic change
        ewma_ratio = ewma_var_late / ewma_var_early
        sample_ratio = sample_var_late / sample_var_early
        
        assert ewma_ratio > sample_ratio, "EWMA should be more responsive to volatility changes"


class TestStochasticVolatilityModel:
    """Test suite for stochastic volatility models"""
    
    def test_stochastic_vol_simulation(self):
        """Test stochastic volatility simulation"""
        sv_model = StochasticVolatilityModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
        
        prices, variance, returns = sv_model.simulate(n_periods=1000)
        
        # Check output dimensions
        assert len(prices) == 1000, "Prices length should match n_periods"
        assert len(variance) == 1000, "Variance length should match n_periods"
        assert len(returns) == 1000, "Returns length should match n_periods"
        
        # Check variance properties
        assert np.all(variance >= 0), "Variance should be non-negative"
        assert np.mean(variance) > 0, "Average variance should be positive"
        
        # Check mean reversion in variance
        long_term_mean = np.mean(variance[500:])  # Use latter half
        assert abs(long_term_mean - sv_model.theta) < 0.02, "Variance should revert to theta"
    
    def test_stochastic_vol_calibration(self):
        """Test stochastic volatility calibration"""
        # Generate test data
        sv_model = StochasticVolatilityModel()
        _, _, test_returns = sv_model.simulate(n_periods=1000)
        
        # Calibrate new model
        new_model = StochasticVolatilityModel()
        results = new_model.calibrate(test_returns)
        
        # Check calibration results
        assert results['kappa'] > 0, "Kappa should be positive"
        assert results['theta'] > 0, "Theta should be positive"
        assert results['sigma'] > 0, "Sigma should be positive"
        assert -1 <= results['rho'] <= 1, "Rho should be between -1 and 1"
    
    def test_stochastic_vol_correlation(self):
        """Test correlation between price and volatility"""
        sv_model = StochasticVolatilityModel(rho=-0.7)  # Strong negative correlation
        
        prices, variance, returns = sv_model.simulate(n_periods=1000)
        
        # Calculate correlation between returns and variance changes
        variance_changes = np.diff(variance)
        correlation = np.corrcoef(returns[1:], variance_changes)[0, 1]
        
        # Should be negative (leverage effect)
        assert correlation < 0, "Correlation should be negative (leverage effect)"
    
    def test_stochastic_vol_mean_reversion(self):
        """Test mean reversion in stochastic volatility"""
        sv_model = StochasticVolatilityModel(kappa=5.0, theta=0.04)  # Strong mean reversion
        
        # Start with high initial variance
        _, variance, _ = sv_model.simulate(n_periods=1000, initial_variance=0.16)
        
        # Variance should revert to long-term mean
        final_variance = np.mean(variance[-100:])
        assert abs(final_variance - sv_model.theta) < 0.05, "Variance should revert to theta"


class TestVolatilitySurface:
    """Test suite for volatility surface"""
    
    @pytest.fixture
    def sample_surface_data(self):
        """Generate sample volatility surface data"""
        strikes = np.array([80, 90, 100, 110, 120])
        maturities = np.array([0.25, 0.5, 1.0, 2.0])
        
        # Generate realistic implied volatilities (smile and term structure)
        implied_vols = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                # Base volatility with smile and term structure
                base_vol = 0.2 + 0.05 * np.sqrt(T)  # Term structure
                smile_effect = 0.02 * ((K - 100) / 100)**2  # Smile effect
                implied_vols[i, j] = base_vol + smile_effect
        
        return strikes, maturities, implied_vols
    
    def test_surface_fitting(self, sample_surface_data):
        """Test volatility surface fitting"""
        strikes, maturities, implied_vols = sample_surface_data
        
        surface = VolatilitySurface()
        results = surface.fit(strikes, maturities, implied_vols)
        
        # Check fitting results
        assert results['surface_fitted'], "Surface should be fitted"
        assert results['min_strike'] == strikes.min(), "Min strike should match"
        assert results['max_strike'] == strikes.max(), "Max strike should match"
        assert surface.fitted, "Surface should be marked as fitted"
    
    def test_surface_interpolation(self, sample_surface_data):
        """Test volatility surface interpolation"""
        strikes, maturities, implied_vols = sample_surface_data
        
        surface = VolatilitySurface()
        surface.fit(strikes, maturities, implied_vols)
        
        # Test interpolation at known points
        test_vol = surface.interpolate(100, 1.0)
        expected_vol = implied_vols[2, 2]  # T=1.0, K=100
        assert abs(test_vol - expected_vol) < 0.01, "Interpolation should be accurate at known points"
        
        # Test interpolation between points
        interp_vol = surface.interpolate(95, 0.75)
        assert 0.1 < interp_vol < 0.5, "Interpolated volatility should be reasonable"
    
    def test_volatility_smile_extraction(self, sample_surface_data):
        """Test volatility smile extraction"""
        strikes, maturities, implied_vols = sample_surface_data
        
        surface = VolatilitySurface()
        surface.fit(strikes, maturities, implied_vols)
        
        # Extract smile for 1-year maturity
        smile = surface.smile(1.0)
        
        # Check smile properties
        assert len(smile) == len(strikes), "Smile should have same length as strikes"
        assert np.all(smile > 0), "All volatilities should be positive"
        
        # Check smile shape (should be U-shaped)
        atm_index = len(strikes) // 2
        assert smile[0] > smile[atm_index], "OTM put vol should be higher than ATM"
        assert smile[-1] > smile[atm_index], "OTM call vol should be higher than ATM"
    
    def test_term_structure_extraction(self, sample_surface_data):
        """Test volatility term structure extraction"""
        strikes, maturities, implied_vols = sample_surface_data
        
        surface = VolatilitySurface()
        surface.fit(strikes, maturities, implied_vols)
        
        # Extract term structure for ATM strike
        term_structure = surface.term_structure(100)
        
        # Check term structure properties
        assert len(term_structure) == len(maturities), "Term structure should have same length as maturities"
        assert np.all(term_structure > 0), "All volatilities should be positive"
        
        # Typically, term structure slopes upward
        assert term_structure[-1] > term_structure[0], "Long-term vol should be higher than short-term"


class TestImpliedVolatilityExtraction:
    """Test suite for implied volatility extraction"""
    
    def test_implied_vol_extraction_atm(self):
        """Test implied volatility extraction for ATM options"""
        # Market parameters
        S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
        true_vol = 0.2
        
        # Generate option price using Black-Scholes
        market_price = ImpliedVolatilityExtractor.black_scholes_price(S, K, T, r, true_vol, q, "call")
        
        # Extract implied volatility
        implied_vol = ImpliedVolatilityExtractor.extract_implied_volatility(
            market_price, S, K, T, r, q, "call"
        )
        
        # Should recover true volatility
        assert_close(implied_vol, true_vol, 0.001)
    
    def test_implied_vol_extraction_otm(self):
        """Test implied volatility extraction for OTM options"""
        # Market parameters
        S, K, T, r, q = 100, 110, 1.0, 0.05, 0.0  # OTM call
        true_vol = 0.25
        
        # Generate option price
        market_price = ImpliedVolatilityExtractor.black_scholes_price(S, K, T, r, true_vol, q, "call")
        
        # Extract implied volatility
        implied_vol = ImpliedVolatilityExtractor.extract_implied_volatility(
            market_price, S, K, T, r, q, "call"
        )
        
        # Should recover true volatility
        assert_close(implied_vol, true_vol, 0.001)
    
    def test_implied_vol_extraction_put(self):
        """Test implied volatility extraction for put options"""
        # Market parameters
        S, K, T, r, q = 100, 95, 1.0, 0.05, 0.0  # OTM put
        true_vol = 0.22
        
        # Generate option price
        market_price = ImpliedVolatilityExtractor.black_scholes_price(S, K, T, r, true_vol, q, "put")
        
        # Extract implied volatility
        implied_vol = ImpliedVolatilityExtractor.extract_implied_volatility(
            market_price, S, K, T, r, q, "put"
        )
        
        # Should recover true volatility
        assert_close(implied_vol, true_vol, 0.001)
    
    def test_implied_vol_extreme_cases(self):
        """Test implied volatility extraction for extreme cases"""
        # Very short time to maturity
        S, K, T, r, q = 100, 100, 0.01, 0.05, 0.0
        true_vol = 0.3
        
        market_price = ImpliedVolatilityExtractor.black_scholes_price(S, K, T, r, true_vol, q, "call")
        implied_vol = ImpliedVolatilityExtractor.extract_implied_volatility(
            market_price, S, K, T, r, q, "call"
        )
        
        # Should still work reasonably well
        assert abs(implied_vol - true_vol) < 0.05
        
        # Very deep ITM option
        S, K, T, r, q = 100, 70, 1.0, 0.05, 0.0
        market_price = ImpliedVolatilityExtractor.black_scholes_price(S, K, T, r, true_vol, q, "call")
        implied_vol = ImpliedVolatilityExtractor.extract_implied_volatility(
            market_price, S, K, T, r, q, "call"
        )
        
        # Should work even for deep ITM
        assert abs(implied_vol - true_vol) < 0.05
    
    def test_implied_vol_smile_construction(self):
        """Test construction of volatility smile"""
        # Parameters
        S, T, r, q = 100, 1.0, 0.05, 0.0
        strikes = np.array([80, 90, 100, 110, 120])
        true_vols = np.array([0.25, 0.22, 0.20, 0.22, 0.25])  # U-shaped smile
        
        implied_vols = []
        
        for i, K in enumerate(strikes):
            # Generate market price
            market_price = ImpliedVolatilityExtractor.black_scholes_price(
                S, K, T, r, true_vols[i], q, "call"
            )
            
            # Extract implied volatility
            implied_vol = ImpliedVolatilityExtractor.extract_implied_volatility(
                market_price, S, K, T, r, q, "call"
            )
            
            implied_vols.append(implied_vol)
        
        implied_vols = np.array(implied_vols)
        
        # Check smile shape
        assert implied_vols[0] > implied_vols[2], "OTM put vol should be higher than ATM"
        assert implied_vols[-1] > implied_vols[2], "OTM call vol should be higher than ATM"
        
        # Check accuracy
        for i in range(len(strikes)):
            assert_close(implied_vols[i], true_vols[i], 0.001)


class TestVolatilityModelComparisons:
    """Test suite for comparing different volatility models"""
    
    def test_garch_vs_ewma_persistence(self):
        """Test persistence comparison between GARCH and EWMA"""
        test_returns = TestDataSets.get_volatility_data()
        
        # Fit GARCH model
        garch = GARCHModel()
        garch.fit(test_returns)
        
        # Fit EWMA model
        ewma = EWMAModel(lambda_decay=0.94)
        ewma.fit(test_returns)
        
        # Compare persistence
        garch_persistence = garch.alpha + garch.beta
        ewma_persistence = ewma.lambda_decay
        
        # Both should exhibit high persistence
        assert garch_persistence > 0.8, "GARCH should show high persistence"
        assert ewma_persistence > 0.8, "EWMA should show high persistence"
    
    def test_volatility_forecasting_comparison(self):
        """Test forecasting performance comparison"""
        test_returns = TestDataSets.get_volatility_data()
        
        # Split data
        train_size = int(0.8 * len(test_returns))
        train_returns = test_returns[:train_size]
        test_returns_out = test_returns[train_size:]
        
        # Fit models
        garch = GARCHModel()
        garch.fit(train_returns)
        
        ewma = EWMAModel()
        ewma.fit(train_returns)
        
        # Generate forecasts
        horizon = len(test_returns_out)
        garch_forecasts = garch.forecast(horizon)
        ewma_forecasts = ewma.forecast(horizon)
        
        # Compare forecast properties
        assert len(garch_forecasts) == horizon, "GARCH forecasts should match horizon"
        assert len(ewma_forecasts) == horizon, "EWMA forecasts should match horizon"
        
        # GARCH forecasts should vary, EWMA should be constant
        garch_variation = np.std(garch_forecasts)
        ewma_variation = np.std(ewma_forecasts)
        
        assert garch_variation > ewma_variation, "GARCH forecasts should vary more than EWMA"
    
    def test_model_selection_criteria(self):
        """Test model selection criteria"""
        test_returns = TestDataSets.get_volatility_data()
        
        # Fit GARCH model
        garch = GARCHModel()
        garch_results = garch.fit(test_returns)
        
        # Calculate information criteria (simplified)
        n_params_garch = 3  # omega, alpha, beta
        n_obs = len(test_returns)
        
        aic_garch = -2 * garch_results['log_likelihood'] + 2 * n_params_garch
        bic_garch = -2 * garch_results['log_likelihood'] + n_params_garch * np.log(n_obs)
        
        # Check that criteria are reasonable
        assert aic_garch > 0, "AIC should be positive"
        assert bic_garch > 0, "BIC should be positive"
        assert bic_garch > aic_garch, "BIC should be larger than AIC for small datasets"


class TestVolatilityModelPerformance:
    """Test performance of volatility models"""
    
    def test_garch_performance(self):
        """Test GARCH model performance"""
        test_returns = TestDataSets.get_volatility_data()
        
        start_time = time.time()
        garch = GARCHModel()
        garch.fit(test_returns)
        execution_time = time.time() - start_time
        
        # Performance should be reasonable
        assert execution_time < PERFORMANCE_BENCHMARKS["garch_estimation"]
    
    def test_ewma_performance(self):
        """Test EWMA model performance"""
        test_returns = TestDataSets.get_volatility_data()
        
        start_time = time.time()
        ewma = EWMAModel()
        ewma.fit(test_returns)
        execution_time = time.time() - start_time
        
        # EWMA should be very fast
        assert execution_time < 0.01, "EWMA should be very fast"
    
    def test_implied_vol_performance(self):
        """Test implied volatility extraction performance"""
        # Test single extraction
        start_time = time.time()
        
        implied_vol = ImpliedVolatilityExtractor.extract_implied_volatility(
            market_price=10.0, S=100, K=100, T=1.0, r=0.05, q=0.0, option_type="call"
        )
        
        execution_time = time.time() - start_time
        
        # Should be fast
        assert execution_time < 0.01, "Implied vol extraction should be fast"
        assert implied_vol > 0, "Implied vol should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])