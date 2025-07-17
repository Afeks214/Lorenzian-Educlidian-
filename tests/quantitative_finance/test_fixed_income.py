"""
Fixed Income Models Testing Suite

Comprehensive testing for fixed income models including:
- Yield Curve Construction and Interpolation
- Duration and Convexity Calculations
- Bond Pricing Models
- Interest Rate Models (Vasicek, CIR, Hull-White)
- Credit Risk Models and Default Probability
- Term Structure Models
- Fixed Income Portfolio Management
- Curve Fitting and Calibration
"""

import pytest
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize, root_scalar
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import norm
import warnings

from tests.quantitative_finance import (
    TOLERANCE, BENCHMARK_TOLERANCE, PERFORMANCE_BENCHMARKS,
    TestDataSets, assert_close
)


class BondPricer:
    """Bond pricing utilities"""
    
    @staticmethod
    def bond_price(face_value: float, coupon_rate: float, yield_rate: float,
                   time_to_maturity: float, frequency: int = 2) -> float:
        """Calculate bond price using present value formula"""
        
        if time_to_maturity <= 0:
            return face_value
        
        # Number of coupon payments
        n_payments = int(time_to_maturity * frequency)
        
        # Coupon payment amount
        coupon_payment = face_value * coupon_rate / frequency
        
        # Yield per period
        yield_per_period = yield_rate / frequency
        
        # Present value of coupon payments
        if yield_per_period == 0:
            pv_coupons = coupon_payment * n_payments
        else:
            pv_coupons = coupon_payment * (1 - (1 + yield_per_period)**(-n_payments)) / yield_per_period
        
        # Present value of face value
        pv_face_value = face_value / (1 + yield_per_period)**n_payments
        
        return pv_coupons + pv_face_value
    
    @staticmethod
    def bond_yield(price: float, face_value: float, coupon_rate: float,
                   time_to_maturity: float, frequency: int = 2) -> float:
        """Calculate bond yield to maturity using root finding"""
        
        def objective(yield_rate):
            return BondPricer.bond_price(face_value, coupon_rate, yield_rate,
                                       time_to_maturity, frequency) - price
        
        # Use root finding to solve for yield
        try:
            result = root_scalar(objective, bracket=[0.001, 0.5], method='brentq')
            return result.root
        except ValueError:
            # If bracket fails, try wider range
            result = root_scalar(objective, bracket=[0.0001, 1.0], method='brentq')
            return result.root
    
    @staticmethod
    def modified_duration(face_value: float, coupon_rate: float, yield_rate: float,
                         time_to_maturity: float, frequency: int = 2) -> float:
        """Calculate modified duration"""
        
        # Calculate Macaulay duration first
        macaulay_duration = BondPricer.macaulay_duration(
            face_value, coupon_rate, yield_rate, time_to_maturity, frequency
        )
        
        # Modified duration = Macaulay duration / (1 + yield/frequency)
        return macaulay_duration / (1 + yield_rate / frequency)
    
    @staticmethod
    def macaulay_duration(face_value: float, coupon_rate: float, yield_rate: float,
                         time_to_maturity: float, frequency: int = 2) -> float:
        """Calculate Macaulay duration"""
        
        if time_to_maturity <= 0:
            return 0
        
        n_payments = int(time_to_maturity * frequency)
        coupon_payment = face_value * coupon_rate / frequency
        yield_per_period = yield_rate / frequency
        
        bond_price = BondPricer.bond_price(face_value, coupon_rate, yield_rate,
                                         time_to_maturity, frequency)
        
        # Weight each cash flow by its time
        weighted_time = 0
        
        for t in range(1, n_payments + 1):
            time_years = t / frequency
            
            if t == n_payments:
                # Final payment includes coupon and face value
                cash_flow = coupon_payment + face_value
            else:
                cash_flow = coupon_payment
            
            # Present value of cash flow
            pv_cash_flow = cash_flow / (1 + yield_per_period)**t
            
            # Weight by time
            weighted_time += time_years * pv_cash_flow
        
        return weighted_time / bond_price
    
    @staticmethod
    def convexity(face_value: float, coupon_rate: float, yield_rate: float,
                  time_to_maturity: float, frequency: int = 2) -> float:
        """Calculate bond convexity"""
        
        if time_to_maturity <= 0:
            return 0
        
        n_payments = int(time_to_maturity * frequency)
        coupon_payment = face_value * coupon_rate / frequency
        yield_per_period = yield_rate / frequency
        
        bond_price = BondPricer.bond_price(face_value, coupon_rate, yield_rate,
                                         time_to_maturity, frequency)
        
        # Calculate convexity
        convexity_sum = 0
        
        for t in range(1, n_payments + 1):
            if t == n_payments:
                cash_flow = coupon_payment + face_value
            else:
                cash_flow = coupon_payment
            
            # Present value of cash flow
            pv_cash_flow = cash_flow / (1 + yield_per_period)**t
            
            # Time factor for convexity
            time_factor = t * (t + 1) / (frequency**2)
            
            convexity_sum += time_factor * pv_cash_flow
        
        return convexity_sum / (bond_price * (1 + yield_per_period)**2)


class YieldCurve:
    """Yield curve construction and interpolation"""
    
    def __init__(self, method: str = "cubic_spline"):
        self.method = method
        self.maturities = None
        self.yields = None
        self.interpolator = None
        self.fitted = False
    
    def fit(self, maturities: np.ndarray, yields: np.ndarray) -> Dict[str, Any]:
        """Fit yield curve"""
        self.maturities = maturities
        self.yields = yields
        
        if self.method == "linear":
            self.interpolator = interp1d(maturities, yields, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
        elif self.method == "cubic_spline":
            self.interpolator = CubicSpline(maturities, yields, bc_type='natural')
        elif self.method == "nelson_siegel":
            self._fit_nelson_siegel()
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")
        
        self.fitted = True
        
        return {
            'method': self.method,
            'n_points': len(maturities),
            'min_maturity': np.min(maturities),
            'max_maturity': np.max(maturities),
            'fitted': True
        }
    
    def _fit_nelson_siegel(self):
        """Fit Nelson-Siegel yield curve model"""
        # Nelson-Siegel model: y(t) = β₀ + β₁ * (1 - exp(-t/τ)) / (t/τ) + β₂ * ((1 - exp(-t/τ)) / (t/τ) - exp(-t/τ))
        
        def nelson_siegel(t, beta0, beta1, beta2, tau):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                term1 = (1 - np.exp(-t / tau)) / (t / tau)
                term2 = term1 - np.exp(-t / tau)
                return beta0 + beta1 * term1 + beta2 * term2
        
        def objective(params):
            beta0, beta1, beta2, tau = params
            predicted = np.array([nelson_siegel(t, beta0, beta1, beta2, tau) for t in self.maturities])
            return np.sum((predicted - self.yields)**2)
        
        # Initial guess
        initial_params = [0.05, -0.02, 0.02, 2.0]
        
        # Bounds for parameters
        bounds = [(0.001, 0.2), (-0.1, 0.1), (-0.1, 0.1), (0.1, 10.0)]
        
        # Optimize
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            self.ns_params = result.x
            self.interpolator = lambda t: nelson_siegel(t, *self.ns_params)
        else:
            # Fallback to cubic spline
            self.interpolator = CubicSpline(self.maturities, self.yields, bc_type='natural')
    
    def interpolate(self, maturity: float) -> float:
        """Interpolate yield for given maturity"""
        if not self.fitted:
            raise ValueError("Yield curve must be fitted before interpolation")
        
        return float(self.interpolator(maturity))
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """Calculate forward rate between two times"""
        if not self.fitted:
            raise ValueError("Yield curve must be fitted before forward rate calculation")
        
        if t1 >= t2:
            raise ValueError("t1 must be less than t2")
        
        # Forward rate: f(t1, t2) = (y(t2) * t2 - y(t1) * t1) / (t2 - t1)
        y1 = self.interpolate(t1)
        y2 = self.interpolate(t2)
        
        return (y2 * t2 - y1 * t1) / (t2 - t1)
    
    def discount_factor(self, maturity: float) -> float:
        """Calculate discount factor for given maturity"""
        yield_rate = self.interpolate(maturity)
        return np.exp(-yield_rate * maturity)
    
    def zero_coupon_bond_price(self, face_value: float, maturity: float) -> float:
        """Calculate zero-coupon bond price"""
        return face_value * self.discount_factor(maturity)


class InterestRateModel:
    """Base class for interest rate models"""
    
    def __init__(self):
        self.fitted = False
        self.parameters = None
    
    def simulate(self, r0: float, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """Simulate interest rate paths"""
        raise NotImplementedError("Subclasses must implement simulate method")
    
    def bond_price(self, r: float, T: float, face_value: float = 1.0) -> float:
        """Calculate bond price under the model"""
        raise NotImplementedError("Subclasses must implement bond_price method")


class VasicekModel(InterestRateModel):
    """Vasicek interest rate model"""
    
    def __init__(self, kappa: float, theta: float, sigma: float):
        super().__init__()
        self.kappa = kappa    # Mean reversion speed
        self.theta = theta    # Long-term mean
        self.sigma = sigma    # Volatility
        self.fitted = True
        self.parameters = {'kappa': kappa, 'theta': theta, 'sigma': sigma}
    
    def simulate(self, r0: float, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """Simulate Vasicek model paths"""
        dt = T / n_steps
        
        # Pre-calculate constants
        drift_factor = np.exp(-self.kappa * dt)
        mean_factor = self.theta * (1 - drift_factor)
        vol_factor = self.sigma * np.sqrt((1 - np.exp(-2 * self.kappa * dt)) / (2 * self.kappa))
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0
        
        # Generate random shocks
        np.random.seed(42)
        shocks = np.random.normal(0, 1, (n_paths, n_steps))
        
        # Simulate paths
        for t in range(n_steps):
            paths[:, t + 1] = (paths[:, t] * drift_factor + 
                             mean_factor + 
                             vol_factor * shocks[:, t])
        
        return paths
    
    def bond_price(self, r: float, T: float, face_value: float = 1.0) -> float:
        """Calculate bond price under Vasicek model"""
        # Analytical solution for Vasicek model
        if T <= 0:
            return face_value
        
        # A(T) and B(T) functions
        B_T = (1 - np.exp(-self.kappa * T)) / self.kappa
        A_T = (B_T - T) * (self.theta - self.sigma**2 / (2 * self.kappa**2)) - \
              (self.sigma**2 * B_T**2) / (4 * self.kappa)
        
        return face_value * np.exp(A_T - B_T * r)
    
    def yield_curve(self, r: float, maturities: np.ndarray) -> np.ndarray:
        """Calculate yield curve under Vasicek model"""
        yields = np.zeros(len(maturities))
        
        for i, T in enumerate(maturities):
            if T <= 0:
                yields[i] = r
            else:
                bond_price = self.bond_price(r, T, 1.0)
                yields[i] = -np.log(bond_price) / T
        
        return yields


class CIRModel(InterestRateModel):
    """Cox-Ingersoll-Ross (CIR) interest rate model"""
    
    def __init__(self, kappa: float, theta: float, sigma: float):
        super().__init__()
        self.kappa = kappa    # Mean reversion speed
        self.theta = theta    # Long-term mean
        self.sigma = sigma    # Volatility
        self.fitted = True
        self.parameters = {'kappa': kappa, 'theta': theta, 'sigma': sigma}
        
        # Check Feller condition
        if 2 * kappa * theta <= sigma**2:
            warnings.warn("Feller condition not satisfied - rates may hit zero")
    
    def simulate(self, r0: float, T: float, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """Simulate CIR model paths using Euler-Maruyama"""
        dt = T / n_steps
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = r0
        
        # Generate random shocks
        np.random.seed(42)
        shocks = np.random.normal(0, 1, (n_paths, n_steps))
        
        # Simulate paths
        for t in range(n_steps):
            r_current = paths[:, t]
            
            # Ensure non-negative rates
            r_current = np.maximum(r_current, 0)
            
            # CIR dynamics: dr = κ(θ - r)dt + σ√r dW
            drift = self.kappa * (self.theta - r_current) * dt
            diffusion = self.sigma * np.sqrt(r_current) * np.sqrt(dt) * shocks[:, t]
            
            paths[:, t + 1] = r_current + drift + diffusion
            
            # Apply reflection at zero boundary
            paths[:, t + 1] = np.maximum(paths[:, t + 1], 0)
        
        return paths
    
    def bond_price(self, r: float, T: float, face_value: float = 1.0) -> float:
        """Calculate bond price under CIR model"""
        if T <= 0:
            return face_value
        
        # CIR model parameters
        h = np.sqrt(self.kappa**2 + 2 * self.sigma**2)
        
        # A(T) and B(T) functions
        numerator = 2 * h * np.exp((self.kappa + h) * T / 2)
        denominator = (self.kappa + h) * (np.exp(h * T) - 1) + 2 * h
        
        B_T = (np.exp(h * T) - 1) / denominator
        A_T = (2 * h * np.exp((self.kappa + h) * T / 2) / denominator)**(2 * self.kappa * self.theta / self.sigma**2)
        
        return face_value * A_T * np.exp(-B_T * r)


class CreditRiskModel:
    """Credit risk model for default probability"""
    
    def __init__(self, model_type: str = "merton"):
        self.model_type = model_type
        self.fitted = False
        self.parameters = None
    
    def merton_default_probability(self, V: float, D: float, T: float, r: float, sigma_V: float) -> float:
        """Calculate default probability using Merton model"""
        # Merton model: default occurs if V(T) < D
        # where V is firm value, D is debt level
        
        d2 = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
        
        return norm.cdf(-d2)
    
    def credit_spread(self, risk_free_rate: float, default_prob: float, recovery_rate: float = 0.4) -> float:
        """Calculate credit spread given default probability"""
        # Credit spread = -ln(1 - default_prob * (1 - recovery_rate)) / T
        # Simplified version for demonstration
        return -np.log(1 - default_prob * (1 - recovery_rate))
    
    def survival_probability(self, hazard_rate: float, T: float) -> float:
        """Calculate survival probability given hazard rate"""
        return np.exp(-hazard_rate * T)
    
    def fit_hazard_rate(self, credit_spreads: np.ndarray, maturities: np.ndarray,
                       recovery_rate: float = 0.4) -> Dict[str, Any]:
        """Fit constant hazard rate model"""
        
        def objective(hazard_rate):
            implied_spreads = []
            for T in maturities:
                survival_prob = self.survival_probability(hazard_rate, T)
                default_prob = 1 - survival_prob
                implied_spread = self.credit_spread(0, default_prob, recovery_rate)
                implied_spreads.append(implied_spread)
            
            return np.sum((np.array(implied_spreads) - credit_spreads)**2)
        
        # Optimize
        result = minimize(objective, [0.01], bounds=[(0.001, 0.5)], method='L-BFGS-B')
        
        if result.success:
            self.hazard_rate = result.x[0]
            self.fitted = True
            self.parameters = {'hazard_rate': self.hazard_rate, 'recovery_rate': recovery_rate}
            
            return {
                'hazard_rate': self.hazard_rate,
                'recovery_rate': recovery_rate,
                'fitted': True
            }
        else:
            return {
                'fitted': False,
                'error': result.message
            }


class TestBondPricing:
    """Test suite for bond pricing"""
    
    def test_bond_price_calculation(self):
        """Test basic bond price calculation"""
        # Standard bond parameters
        face_value = 1000
        coupon_rate = 0.05
        yield_rate = 0.04
        time_to_maturity = 5.0
        
        start_time = time.time()
        bond_price = BondPricer.bond_price(face_value, coupon_rate, yield_rate, time_to_maturity)
        execution_time = time.time() - start_time
        
        # Check price is reasonable
        assert bond_price > 0, "Bond price should be positive"
        assert bond_price > face_value, "Bond should trade at premium (yield < coupon)"
        
        # Check performance
        assert execution_time < 0.01, "Bond pricing should be fast"
        
        # Test with yield > coupon (discount bond)
        discount_price = BondPricer.bond_price(face_value, coupon_rate, 0.06, time_to_maturity)
        assert discount_price < face_value, "Bond should trade at discount (yield > coupon)"
    
    def test_yield_to_maturity(self):
        """Test yield to maturity calculation"""
        # Bond parameters
        face_value = 1000
        coupon_rate = 0.05
        time_to_maturity = 5.0
        
        # Calculate price at known yield
        known_yield = 0.04
        bond_price = BondPricer.bond_price(face_value, coupon_rate, known_yield, time_to_maturity)
        
        # Calculate yield from price
        calculated_yield = BondPricer.bond_yield(bond_price, face_value, coupon_rate, time_to_maturity)
        
        # Should recover original yield
        assert_close(calculated_yield, known_yield, TOLERANCE)
    
    def test_duration_calculation(self):
        """Test duration calculations"""
        # Bond parameters
        face_value = 1000
        coupon_rate = 0.05
        yield_rate = 0.04
        time_to_maturity = 5.0
        
        # Calculate durations
        macaulay_duration = BondPricer.macaulay_duration(face_value, coupon_rate, yield_rate, time_to_maturity)
        modified_duration = BondPricer.modified_duration(face_value, coupon_rate, yield_rate, time_to_maturity)
        
        # Check duration properties
        assert macaulay_duration > 0, "Macaulay duration should be positive"
        assert macaulay_duration < time_to_maturity, "Macaulay duration should be less than maturity"
        assert modified_duration < macaulay_duration, "Modified duration should be less than Macaulay"
        
        # Test relationship: Modified = Macaulay / (1 + y/m)
        expected_modified = macaulay_duration / (1 + yield_rate / 2)
        assert_close(modified_duration, expected_modified, TOLERANCE)
    
    def test_convexity_calculation(self):
        """Test convexity calculation"""
        # Bond parameters
        face_value = 1000
        coupon_rate = 0.05
        yield_rate = 0.04
        time_to_maturity = 5.0
        
        convexity = BondPricer.convexity(face_value, coupon_rate, yield_rate, time_to_maturity)
        
        # Check convexity properties
        assert convexity > 0, "Convexity should be positive"
        
        # Test convexity numerically
        delta_y = 0.0001
        price_up = BondPricer.bond_price(face_value, coupon_rate, yield_rate + delta_y, time_to_maturity)
        price_down = BondPricer.bond_price(face_value, coupon_rate, yield_rate - delta_y, time_to_maturity)
        price_base = BondPricer.bond_price(face_value, coupon_rate, yield_rate, time_to_maturity)
        
        numerical_convexity = (price_up + price_down - 2 * price_base) / (price_base * delta_y**2)
        
        # Should be approximately equal
        assert abs(convexity - numerical_convexity) < 0.5, "Analytical and numerical convexity should match"
    
    def test_bond_price_sensitivity(self):
        """Test bond price sensitivity to parameters"""
        base_params = {
            'face_value': 1000,
            'coupon_rate': 0.05,
            'yield_rate': 0.04,
            'time_to_maturity': 5.0
        }
        
        base_price = BondPricer.bond_price(**base_params)
        
        # Test yield sensitivity
        high_yield_price = BondPricer.bond_price(**{**base_params, 'yield_rate': 0.06})
        assert high_yield_price < base_price, "Higher yield should result in lower price"
        
        # Test maturity sensitivity
        long_maturity_price = BondPricer.bond_price(**{**base_params, 'time_to_maturity': 10.0})
        # For premium bonds, longer maturity typically means higher price
        
        # Test coupon sensitivity
        high_coupon_price = BondPricer.bond_price(**{**base_params, 'coupon_rate': 0.06})
        assert high_coupon_price > base_price, "Higher coupon should result in higher price"
    
    def test_zero_coupon_bond(self):
        """Test zero coupon bond pricing"""
        face_value = 1000
        yield_rate = 0.05
        time_to_maturity = 5.0
        
        # Zero coupon bond
        zero_coupon_price = BondPricer.bond_price(face_value, 0, yield_rate, time_to_maturity)
        
        # Should equal face_value * exp(-yield * time)
        expected_price = face_value * np.exp(-yield_rate * time_to_maturity)
        assert_close(zero_coupon_price, expected_price, 0.01)  # Small tolerance for discrete compounding
        
        # Duration should equal maturity for zero coupon bonds
        zero_duration = BondPricer.macaulay_duration(face_value, 0, yield_rate, time_to_maturity)
        assert_close(zero_duration, time_to_maturity, 0.01)


class TestYieldCurve:
    """Test suite for yield curve construction"""
    
    @pytest.fixture
    def sample_curve_data(self):
        """Get sample yield curve data"""
        return TestDataSets.get_yield_curve_data()
    
    def test_linear_interpolation(self, sample_curve_data):
        """Test linear interpolation"""
        maturities = np.array(sample_curve_data['maturities'])
        yields = np.array(sample_curve_data['yields'])
        
        start_time = time.time()
        curve = YieldCurve(method="linear")
        results = curve.fit(maturities, yields)
        execution_time = time.time() - start_time
        
        # Check fitting results
        assert results['fitted'], "Curve should be fitted"
        assert results['method'] == "linear", "Method should be linear"
        
        # Test interpolation at known points
        for i, (mat, yield_val) in enumerate(zip(maturities, yields)):
            interpolated = curve.interpolate(mat)
            assert_close(interpolated, yield_val, TOLERANCE)
        
        # Test interpolation between points
        mid_point = (maturities[0] + maturities[1]) / 2
        mid_yield = curve.interpolate(mid_point)
        assert yields[0] <= mid_yield <= yields[1] or yields[1] <= mid_yield <= yields[0]
        
        # Performance check
        assert execution_time < PERFORMANCE_BENCHMARKS["yield_curve_construction"]
    
    def test_cubic_spline_interpolation(self, sample_curve_data):
        """Test cubic spline interpolation"""
        maturities = np.array(sample_curve_data['maturities'])
        yields = np.array(sample_curve_data['yields'])
        
        curve = YieldCurve(method="cubic_spline")
        results = curve.fit(maturities, yields)
        
        # Check fitting results
        assert results['fitted'], "Curve should be fitted"
        assert results['method'] == "cubic_spline", "Method should be cubic_spline"
        
        # Test interpolation at known points
        for mat, yield_val in zip(maturities, yields):
            interpolated = curve.interpolate(mat)
            assert_close(interpolated, yield_val, TOLERANCE)
        
        # Test smoothness - cubic spline should be smooth
        test_points = np.linspace(maturities[0], maturities[-1], 100)
        interpolated_yields = [curve.interpolate(t) for t in test_points]
        
        # Check for no extreme jumps
        diffs = np.diff(interpolated_yields)
        assert np.all(np.abs(diffs) < 0.1), "Interpolated yields should be smooth"
    
    def test_nelson_siegel_fitting(self, sample_curve_data):
        """Test Nelson-Siegel model fitting"""
        maturities = np.array(sample_curve_data['maturities'])
        yields = np.array(sample_curve_data['yields'])
        
        curve = YieldCurve(method="nelson_siegel")
        results = curve.fit(maturities, yields)
        
        # Check fitting results
        assert results['fitted'], "Curve should be fitted"
        assert results['method'] == "nelson_siegel", "Method should be nelson_siegel"
        
        # Test interpolation gives reasonable results
        for mat in maturities:
            interpolated = curve.interpolate(mat)
            assert 0.001 < interpolated < 0.2, "Interpolated yield should be reasonable"
    
    def test_forward_rate_calculation(self, sample_curve_data):
        """Test forward rate calculation"""
        maturities = np.array(sample_curve_data['maturities'])
        yields = np.array(sample_curve_data['yields'])
        
        curve = YieldCurve(method="cubic_spline")
        curve.fit(maturities, yields)
        
        # Calculate forward rates
        t1, t2 = 1.0, 2.0
        forward_rate = curve.forward_rate(t1, t2)
        
        # Check forward rate properties
        assert forward_rate > 0, "Forward rate should be positive"
        
        # Test forward rate calculation manually
        y1 = curve.interpolate(t1)
        y2 = curve.interpolate(t2)
        expected_forward = (y2 * t2 - y1 * t1) / (t2 - t1)
        
        assert_close(forward_rate, expected_forward, TOLERANCE)
    
    def test_discount_factor_calculation(self, sample_curve_data):
        """Test discount factor calculation"""
        maturities = np.array(sample_curve_data['maturities'])
        yields = np.array(sample_curve_data['yields'])
        
        curve = YieldCurve(method="cubic_spline")
        curve.fit(maturities, yields)
        
        # Test discount factors
        for mat in maturities:
            discount_factor = curve.discount_factor(mat)
            
            # Check discount factor properties
            assert 0 < discount_factor <= 1, "Discount factor should be between 0 and 1"
            
            # Check relationship with yield
            yield_val = curve.interpolate(mat)
            expected_discount = np.exp(-yield_val * mat)
            assert_close(discount_factor, expected_discount, TOLERANCE)
    
    def test_zero_coupon_bond_pricing(self, sample_curve_data):
        """Test zero coupon bond pricing using yield curve"""
        maturities = np.array(sample_curve_data['maturities'])
        yields = np.array(sample_curve_data['yields'])
        
        curve = YieldCurve(method="cubic_spline")
        curve.fit(maturities, yields)
        
        # Test zero coupon bond pricing
        face_value = 1000
        maturity = 5.0
        
        bond_price = curve.zero_coupon_bond_price(face_value, maturity)
        
        # Check bond price properties
        assert 0 < bond_price < face_value, "Zero coupon bond should trade at discount"
        
        # Verify using discount factor
        discount_factor = curve.discount_factor(maturity)
        expected_price = face_value * discount_factor
        assert_close(bond_price, expected_price, TOLERANCE)


class TestInterestRateModels:
    """Test suite for interest rate models"""
    
    def test_vasicek_simulation(self):
        """Test Vasicek model simulation"""
        # Model parameters
        kappa = 0.5
        theta = 0.05
        sigma = 0.02
        
        vasicek = VasicekModel(kappa, theta, sigma)
        
        # Simulate paths
        r0 = 0.03
        T = 1.0
        n_steps = 252
        n_paths = 1000
        
        start_time = time.time()
        paths = vasicek.simulate(r0, T, n_steps, n_paths)
        execution_time = time.time() - start_time
        
        # Check simulation results
        assert paths.shape == (n_paths, n_steps + 1), "Paths should have correct dimensions"
        assert np.all(paths[:, 0] == r0), "All paths should start at r0"
        
        # Check mean reversion
        final_rates = paths[:, -1]
        mean_final_rate = np.mean(final_rates)
        
        # Should be closer to theta than r0 (mean reversion)
        assert abs(mean_final_rate - theta) < abs(r0 - theta), "Should exhibit mean reversion"
        
        # Performance check
        assert execution_time < 1.0, "Simulation should be reasonably fast"
    
    def test_vasicek_bond_pricing(self):
        """Test Vasicek bond pricing"""
        vasicek = VasicekModel(kappa=0.5, theta=0.05, sigma=0.02)
        
        # Test bond pricing
        r = 0.04
        T = 5.0
        
        bond_price = vasicek.bond_price(r, T)
        
        # Check bond price properties
        assert 0 < bond_price <= 1, "Bond price should be between 0 and 1"
        
        # Test with different rates
        higher_rate_price = vasicek.bond_price(0.06, T)
        assert higher_rate_price < bond_price, "Higher rate should result in lower bond price"
        
        # Test with different maturities
        longer_maturity_price = vasicek.bond_price(r, 10.0)
        # Relationship depends on model parameters
    
    def test_vasicek_yield_curve(self):
        """Test Vasicek yield curve generation"""
        vasicek = VasicekModel(kappa=0.5, theta=0.05, sigma=0.02)
        
        # Generate yield curve
        r = 0.04
        maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        
        yields = vasicek.yield_curve(r, maturities)
        
        # Check yield curve properties
        assert len(yields) == len(maturities), "Yields should match maturities"
        assert np.all(yields > 0), "All yields should be positive"
        
        # Check that yields converge to long-term mean
        assert abs(yields[-1] - vasicek.theta) < abs(yields[0] - vasicek.theta), \
            "Long-term yields should be closer to theta"
    
    def test_cir_simulation(self):
        """Test CIR model simulation"""
        # Model parameters (satisfying Feller condition)
        kappa = 2.0
        theta = 0.05
        sigma = 0.1
        
        cir = CIRModel(kappa, theta, sigma)
        
        # Simulate paths
        r0 = 0.03
        T = 1.0
        n_steps = 252
        n_paths = 1000
        
        paths = cir.simulate(r0, T, n_steps, n_paths)
        
        # Check simulation results
        assert paths.shape == (n_paths, n_steps + 1), "Paths should have correct dimensions"
        assert np.all(paths[:, 0] == r0), "All paths should start at r0"
        assert np.all(paths >= 0), "All rates should be non-negative"
        
        # Check mean reversion
        final_rates = paths[:, -1]
        mean_final_rate = np.mean(final_rates)
        
        # Should be closer to theta than r0 (mean reversion)
        assert abs(mean_final_rate - theta) < abs(r0 - theta) + 0.01, "Should exhibit mean reversion"
    
    def test_cir_bond_pricing(self):
        """Test CIR bond pricing"""
        cir = CIRModel(kappa=2.0, theta=0.05, sigma=0.1)
        
        # Test bond pricing
        r = 0.04
        T = 5.0
        
        bond_price = cir.bond_price(r, T)
        
        # Check bond price properties
        assert 0 < bond_price <= 1, "Bond price should be between 0 and 1"
        
        # Test with different rates
        higher_rate_price = cir.bond_price(0.06, T)
        assert higher_rate_price < bond_price, "Higher rate should result in lower bond price"
    
    def test_model_comparison(self):
        """Test comparison between Vasicek and CIR models"""
        # Similar parameters
        kappa = 1.0
        theta = 0.05
        sigma = 0.02
        
        vasicek = VasicekModel(kappa, theta, sigma)
        cir = CIRModel(kappa, theta, sigma)
        
        # Compare bond prices
        r = 0.04
        T = 5.0
        
        vasicek_price = vasicek.bond_price(r, T)
        cir_price = cir.bond_price(r, T)
        
        # Both should be reasonable
        assert 0 < vasicek_price <= 1, "Vasicek bond price should be reasonable"
        assert 0 < cir_price <= 1, "CIR bond price should be reasonable"
        
        # Prices should be similar but not identical
        assert abs(vasicek_price - cir_price) < 0.1, "Prices should be similar"


class TestCreditRiskModels:
    """Test suite for credit risk models"""
    
    def test_merton_default_probability(self):
        """Test Merton model default probability"""
        credit_model = CreditRiskModel(model_type="merton")
        
        # Firm parameters
        V = 100  # Firm value
        D = 80   # Debt level
        T = 1.0  # Time horizon
        r = 0.05 # Risk-free rate
        sigma_V = 0.3  # Firm value volatility
        
        default_prob = credit_model.merton_default_probability(V, D, T, r, sigma_V)
        
        # Check default probability properties
        assert 0 <= default_prob <= 1, "Default probability should be between 0 and 1"
        
        # Test with higher leverage (higher default probability)
        higher_leverage_prob = credit_model.merton_default_probability(V, 90, T, r, sigma_V)
        assert higher_leverage_prob > default_prob, "Higher leverage should increase default probability"
        
        # Test with higher volatility (higher default probability)
        higher_vol_prob = credit_model.merton_default_probability(V, D, T, r, 0.5)
        assert higher_vol_prob > default_prob, "Higher volatility should increase default probability"
    
    def test_credit_spread_calculation(self):
        """Test credit spread calculation"""
        credit_model = CreditRiskModel()
        
        # Test parameters
        risk_free_rate = 0.03
        default_prob = 0.02
        recovery_rate = 0.4
        
        credit_spread = credit_model.credit_spread(risk_free_rate, default_prob, recovery_rate)
        
        # Check spread properties
        assert credit_spread > 0, "Credit spread should be positive"
        
        # Test with higher default probability
        higher_default_spread = credit_model.credit_spread(risk_free_rate, 0.05, recovery_rate)
        assert higher_default_spread > credit_spread, "Higher default probability should increase spread"
        
        # Test with lower recovery rate
        lower_recovery_spread = credit_model.credit_spread(risk_free_rate, default_prob, 0.2)
        assert lower_recovery_spread > credit_spread, "Lower recovery rate should increase spread"
    
    def test_survival_probability(self):
        """Test survival probability calculation"""
        credit_model = CreditRiskModel()
        
        # Test parameters
        hazard_rate = 0.02
        T = 5.0
        
        survival_prob = credit_model.survival_probability(hazard_rate, T)
        
        # Check survival probability properties
        assert 0 <= survival_prob <= 1, "Survival probability should be between 0 and 1"
        
        # Test with longer time horizon
        longer_survival_prob = credit_model.survival_probability(hazard_rate, 10.0)
        assert longer_survival_prob < survival_prob, "Longer time should decrease survival probability"
        
        # Test with higher hazard rate
        higher_hazard_prob = credit_model.survival_probability(0.05, T)
        assert higher_hazard_prob < survival_prob, "Higher hazard rate should decrease survival probability"
    
    def test_hazard_rate_fitting(self):
        """Test hazard rate model fitting"""
        credit_model = CreditRiskModel()
        
        # Sample credit spreads and maturities
        maturities = np.array([1.0, 2.0, 3.0, 5.0])
        credit_spreads = np.array([0.01, 0.015, 0.02, 0.025])
        
        results = credit_model.fit_hazard_rate(credit_spreads, maturities)
        
        # Check fitting results
        assert results['fitted'], "Model should be fitted successfully"
        assert results['hazard_rate'] > 0, "Hazard rate should be positive"
        assert 0 <= results['recovery_rate'] <= 1, "Recovery rate should be between 0 and 1"
        
        # Test that fitted model can reproduce similar spreads
        fitted_hazard = results['hazard_rate']
        recovery_rate = results['recovery_rate']
        
        for i, (T, actual_spread) in enumerate(zip(maturities, credit_spreads)):
            survival_prob = credit_model.survival_probability(fitted_hazard, T)
            default_prob = 1 - survival_prob
            implied_spread = credit_model.credit_spread(0, default_prob, recovery_rate)
            
            # Should be reasonably close
            assert abs(implied_spread - actual_spread) < 0.005, f"Fitted spread should match actual for maturity {T}"


class TestFixedIncomeIntegration:
    """Test integration of fixed income models"""
    
    def test_bond_yield_curve_consistency(self):
        """Test consistency between bond pricing and yield curve"""
        # Create yield curve
        curve_data = TestDataSets.get_yield_curve_data()
        maturities = np.array(curve_data['maturities'])
        yields = np.array(curve_data['yields'])
        
        curve = YieldCurve(method="cubic_spline")
        curve.fit(maturities, yields)
        
        # Test bond pricing consistency
        face_value = 1000
        coupon_rate = 0.05
        maturity = 5.0
        
        # Get yield from curve
        curve_yield = curve.interpolate(maturity)
        
        # Price bond using this yield
        bond_price = BondPricer.bond_price(face_value, coupon_rate, curve_yield, maturity)
        
        # Calculate implied yield from bond price
        implied_yield = BondPricer.bond_yield(bond_price, face_value, coupon_rate, maturity)
        
        # Should be consistent
        assert_close(implied_yield, curve_yield, 0.001)
    
    def test_interest_rate_model_calibration(self):
        """Test calibration of interest rate models to yield curve"""
        # Create sample yield curve
        curve_data = TestDataSets.get_yield_curve_data()
        maturities = np.array(curve_data['maturities'])
        yields = np.array(curve_data['yields'])
        
        # Current short rate
        r0 = yields[0]
        
        # Test Vasicek model
        vasicek = VasicekModel(kappa=0.5, theta=0.05, sigma=0.02)
        model_yields = vasicek.yield_curve(r0, maturities)
        
        # Model yields should be reasonable
        assert np.all(model_yields > 0), "All model yields should be positive"
        assert len(model_yields) == len(maturities), "Model yields should match maturities"
        
        # Should be roughly similar to market yields (within reasonable range)
        for i, (market_yield, model_yield) in enumerate(zip(yields, model_yields)):
            assert abs(market_yield - model_yield) < 0.05, f"Model yield should be reasonable for maturity {maturities[i]}"
    
    def test_credit_risk_integration(self):
        """Test integration of credit risk with interest rate models"""
        # Risk-free yield curve
        curve_data = TestDataSets.get_yield_curve_data()
        maturities = np.array(curve_data['maturities'])
        yields = np.array(curve_data['yields'])
        
        rf_curve = YieldCurve(method="cubic_spline")
        rf_curve.fit(maturities, yields)
        
        # Credit model
        credit_model = CreditRiskModel()
        
        # Calculate credit spreads for different maturities
        default_prob = 0.02
        recovery_rate = 0.4
        
        credit_spreads = []
        risky_yields = []
        
        for maturity in maturities:
            # Risk-free rate
            rf_rate = rf_curve.interpolate(maturity)
            
            # Credit spread
            credit_spread = credit_model.credit_spread(rf_rate, default_prob, recovery_rate)
            
            # Risky yield
            risky_yield = rf_rate + credit_spread
            
            credit_spreads.append(credit_spread)
            risky_yields.append(risky_yield)
        
        # Check results
        assert np.all(np.array(credit_spreads) > 0), "All credit spreads should be positive"
        assert np.all(np.array(risky_yields) > yields), "Risky yields should be higher than risk-free"
        
        # Credit spreads should be reasonable
        for spread in credit_spreads:
            assert spread < 0.1, "Credit spread should be reasonable"


class TestFixedIncomePerformance:
    """Test performance of fixed income models"""
    
    def test_bond_pricing_performance(self):
        """Test bond pricing performance"""
        # Large number of bonds
        n_bonds = 1000
        
        start_time = time.time()
        for _ in range(n_bonds):
            BondPricer.bond_price(1000, 0.05, 0.04, 5.0)
        execution_time = time.time() - start_time
        
        # Should be fast
        avg_time_per_bond = execution_time / n_bonds
        assert avg_time_per_bond < 0.001, "Bond pricing should be very fast"
    
    def test_yield_curve_performance(self):
        """Test yield curve construction performance"""
        curve_data = TestDataSets.get_yield_curve_data()
        maturities = np.array(curve_data['maturities'])
        yields = np.array(curve_data['yields'])
        
        start_time = time.time()
        curve = YieldCurve(method="cubic_spline")
        curve.fit(maturities, yields)
        execution_time = time.time() - start_time
        
        # Should be fast
        assert execution_time < PERFORMANCE_BENCHMARKS["yield_curve_construction"]
    
    def test_interest_rate_simulation_performance(self):
        """Test interest rate simulation performance"""
        vasicek = VasicekModel(kappa=0.5, theta=0.05, sigma=0.02)
        
        start_time = time.time()
        paths = vasicek.simulate(r0=0.03, T=1.0, n_steps=252, n_paths=1000)
        execution_time = time.time() - start_time
        
        # Should be reasonably fast
        assert execution_time < 2.0, "Interest rate simulation should be reasonably fast"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])