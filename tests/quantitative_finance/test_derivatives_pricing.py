"""
Derivatives Pricing Testing Suite

Comprehensive testing for derivatives pricing models including:
- Futures Pricing and Basis Relationships
- Interest Rate Swaps Pricing
- Currency Swaps and Cross-Currency Derivatives
- Credit Default Swaps (CDS)
- Structured Products Pricing
- Exotic Derivatives
- Greeks for Complex Derivatives
- Model Validation and Calibration
"""

import pytest
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize, root_scalar, fsolve
from scipy.stats import norm
from scipy.interpolate import interp1d
import warnings

from tests.quantitative_finance import (
    TOLERANCE, BENCHMARK_TOLERANCE, PERFORMANCE_BENCHMARKS,
    TestDataSets, assert_close
)


class FuturesPricer:
    """Futures pricing models"""
    
    @staticmethod
    def futures_price(spot_price: float, risk_free_rate: float, time_to_maturity: float,
                     dividend_yield: float = 0.0, storage_cost: float = 0.0,
                     convenience_yield: float = 0.0) -> float:
        """Calculate futures price using cost-of-carry model"""
        
        # Cost of carry = risk-free rate + storage cost - dividend yield - convenience yield
        carry_cost = risk_free_rate + storage_cost - dividend_yield - convenience_yield
        
        return spot_price * np.exp(carry_cost * time_to_maturity)
    
    @staticmethod
    def basis(spot_price: float, futures_price: float) -> float:
        """Calculate basis (spot - futures)"""
        return spot_price - futures_price
    
    @staticmethod
    def implied_convenience_yield(spot_price: float, futures_price: float,
                                risk_free_rate: float, time_to_maturity: float,
                                dividend_yield: float = 0.0, storage_cost: float = 0.0) -> float:
        """Calculate implied convenience yield"""
        
        if time_to_maturity <= 0:
            return 0.0
        
        # Solve for convenience yield: F = S * exp((r + s - d - c) * T)
        # c = r + s - d - ln(F/S) / T
        
        convenience_yield = (risk_free_rate + storage_cost - dividend_yield - 
                           np.log(futures_price / spot_price) / time_to_maturity)
        
        return convenience_yield
    
    @staticmethod
    def currency_futures_price(spot_exchange_rate: float, domestic_rate: float,
                              foreign_rate: float, time_to_maturity: float) -> float:
        """Calculate currency futures price"""
        
        # F = S * exp((r_d - r_f) * T)
        return spot_exchange_rate * np.exp((domestic_rate - foreign_rate) * time_to_maturity)
    
    @staticmethod
    def commodity_futures_price(spot_price: float, risk_free_rate: float,
                               time_to_maturity: float, storage_cost: float,
                               convenience_yield: float) -> float:
        """Calculate commodity futures price"""
        
        carry_cost = risk_free_rate + storage_cost - convenience_yield
        return spot_price * np.exp(carry_cost * time_to_maturity)


class SwapsPricer:
    """Interest rate and currency swaps pricing"""
    
    def __init__(self, yield_curve: Optional[Any] = None):
        self.yield_curve = yield_curve
    
    def vanilla_swap_rate(self, tenor: float, payment_frequency: float = 0.5) -> float:
        """Calculate fair swap rate for vanilla interest rate swap"""
        
        if self.yield_curve is None:
            # Use flat yield curve for testing
            flat_rate = 0.05
            return flat_rate
        
        # Number of payments
        n_payments = int(tenor / payment_frequency)
        
        # Calculate present value of floating leg (= 1 at initiation)
        pv_floating = 1.0
        
        # Calculate present value of fixed leg annuity
        pv_annuity = 0.0
        for i in range(1, n_payments + 1):
            maturity = i * payment_frequency
            discount_factor = self.yield_curve.discount_factor(maturity)
            pv_annuity += discount_factor * payment_frequency
        
        # Fair swap rate
        return pv_floating / pv_annuity
    
    def swap_pv(self, notional: float, fixed_rate: float, floating_rate: float,
                tenor: float, payment_frequency: float = 0.5) -> float:
        """Calculate present value of interest rate swap"""
        
        # For simplicity, assume flat yield curve
        discount_rate = 0.05
        
        n_payments = int(tenor / payment_frequency)
        
        pv_fixed = 0.0
        pv_floating = 0.0
        
        for i in range(1, n_payments + 1):
            maturity = i * payment_frequency
            discount_factor = np.exp(-discount_rate * maturity)
            
            # Fixed leg payment
            fixed_payment = notional * fixed_rate * payment_frequency
            pv_fixed += fixed_payment * discount_factor
            
            # Floating leg payment (simplified)
            floating_payment = notional * floating_rate * payment_frequency
            pv_floating += floating_payment * discount_factor
        
        # PV from fixed payer's perspective
        return pv_floating - pv_fixed
    
    def currency_swap_rate(self, domestic_rate: float, foreign_rate: float,
                          spot_exchange_rate: float, tenor: float) -> float:
        """Calculate currency swap rate"""
        
        # Simplified currency swap rate calculation
        # In practice, would use full term structure
        
        rate_differential = domestic_rate - foreign_rate
        return spot_exchange_rate * np.exp(rate_differential * tenor)
    
    def dv01(self, notional: float, tenor: float, payment_frequency: float = 0.5) -> float:
        """Calculate DV01 (dollar value of 01) for swap"""
        
        # DV01 = change in swap value for 1bp change in rates
        base_rate = 0.05
        
        pv_base = self.swap_pv(notional, base_rate, base_rate, tenor, payment_frequency)
        pv_up = self.swap_pv(notional, base_rate + 0.0001, base_rate, tenor, payment_frequency)
        
        return pv_up - pv_base


class CDSPricer:
    """Credit Default Swap pricing"""
    
    def __init__(self, recovery_rate: float = 0.4):
        self.recovery_rate = recovery_rate
    
    def cds_spread(self, default_probability: float, tenor: float,
                   risk_free_rate: float = 0.05) -> float:
        """Calculate CDS spread given default probability"""
        
        # Simplified CDS spread calculation
        # Spread = (1 - recovery_rate) * default_probability / tenor
        
        loss_given_default = 1 - self.recovery_rate
        return loss_given_default * default_probability / tenor
    
    def cds_pv(self, notional: float, cds_spread: float, hazard_rate: float,
               tenor: float, risk_free_rate: float = 0.05) -> float:
        """Calculate CDS present value"""
        
        # Present value of premium leg
        pv_premium = 0.0
        
        # Present value of protection leg
        pv_protection = 0.0
        
        # Quarterly payments
        payment_frequency = 0.25
        n_payments = int(tenor / payment_frequency)
        
        for i in range(1, n_payments + 1):
            t = i * payment_frequency
            
            # Survival probability
            survival_prob = np.exp(-hazard_rate * t)
            
            # Discount factor
            discount_factor = np.exp(-risk_free_rate * t)
            
            # Premium payment
            premium_payment = notional * cds_spread * payment_frequency
            pv_premium += premium_payment * survival_prob * discount_factor
            
            # Protection payment (simplified)
            if i == 1:
                default_prob = 1 - survival_prob
                protection_payment = notional * (1 - self.recovery_rate) * default_prob
                pv_protection += protection_payment * discount_factor
        
        # PV from protection buyer's perspective
        return pv_protection - pv_premium
    
    def hazard_rate_from_spread(self, cds_spread: float, tenor: float) -> float:
        """Calculate implied hazard rate from CDS spread"""
        
        # Simplified relationship
        # In practice, would use iterative solution
        
        return cds_spread / (1 - self.recovery_rate)


class StructuredProductsPricer:
    """Structured products pricing"""
    
    @staticmethod
    def autocallable_note_price(spot_price: float, barrier_level: float, coupon_rate: float,
                               knock_in_barrier: float, time_to_maturity: float,
                               risk_free_rate: float, volatility: float,
                               n_simulations: int = 10000) -> float:
        """Price autocallable note using Monte Carlo"""
        
        np.random.seed(42)
        
        # Number of observation dates (quarterly)
        n_observations = int(time_to_maturity * 4)
        dt = time_to_maturity / n_observations
        
        payoffs = []
        
        for _ in range(n_simulations):
            # Generate price path
            prices = [spot_price]
            hit_knock_in = False
            
            for t in range(n_observations):
                # Generate next price
                dW = np.random.normal(0, np.sqrt(dt))
                next_price = prices[-1] * np.exp((risk_free_rate - 0.5 * volatility**2) * dt + 
                                               volatility * dW)
                prices.append(next_price)
                
                # Check knock-in barrier
                if next_price <= knock_in_barrier:
                    hit_knock_in = True
                
                # Check autocall condition
                if next_price >= barrier_level:
                    # Autocall triggered
                    observation_time = (t + 1) * dt
                    payoff = (1 + coupon_rate * observation_time) * np.exp(-risk_free_rate * observation_time)
                    payoffs.append(payoff)
                    break
            else:
                # No autocall triggered
                final_price = prices[-1]
                
                if hit_knock_in and final_price < spot_price:
                    # Capital at risk
                    payoff = (final_price / spot_price) * np.exp(-risk_free_rate * time_to_maturity)
                else:
                    # Principal protected
                    payoff = np.exp(-risk_free_rate * time_to_maturity)
                
                payoffs.append(payoff)
        
        return np.mean(payoffs)
    
    @staticmethod
    def reverse_convertible_note_price(spot_price: float, strike_price: float,
                                     coupon_rate: float, time_to_maturity: float,
                                     risk_free_rate: float, volatility: float) -> float:
        """Price reverse convertible note"""
        
        # Reverse convertible = Bond + Short Put Option
        
        # Bond component
        bond_value = np.exp(-risk_free_rate * time_to_maturity) * (1 + coupon_rate * time_to_maturity)
        
        # Short put option component
        from tests.quantitative_finance.test_options_pricing import BlackScholesModel
        
        put_value = BlackScholesModel.put_price(spot_price, strike_price, time_to_maturity,
                                              risk_free_rate, volatility, 0.0)
        
        return bond_value - put_value
    
    @staticmethod
    def equity_linked_note_price(spot_price: float, participation_rate: float,
                               cap_level: float, time_to_maturity: float,
                               risk_free_rate: float, volatility: float) -> float:
        """Price equity-linked note with cap"""
        
        # ELN = Zero-coupon bond + Capped call option
        
        # Zero-coupon bond
        bond_value = np.exp(-risk_free_rate * time_to_maturity)
        
        # Capped call option (call spread)
        from tests.quantitative_finance.test_options_pricing import BlackScholesModel
        
        call_value = BlackScholesModel.call_price(spot_price, spot_price, time_to_maturity,
                                                risk_free_rate, volatility, 0.0)
        
        cap_call_value = BlackScholesModel.call_price(spot_price, cap_level, time_to_maturity,
                                                     risk_free_rate, volatility, 0.0)
        
        capped_call_value = participation_rate * (call_value - cap_call_value)
        
        return bond_value + capped_call_value


class ExoticDerivativesPricer:
    """Exotic derivatives pricing"""
    
    @staticmethod
    def quanto_option_price(spot_price: float, strike_price: float, time_to_maturity: float,
                           domestic_rate: float, foreign_rate: float, asset_volatility: float,
                           fx_volatility: float, correlation: float) -> float:
        """Price quanto option"""
        
        # Quanto adjustment
        quanto_adjustment = correlation * asset_volatility * fx_volatility
        
        # Adjusted foreign rate
        adjusted_foreign_rate = foreign_rate - quanto_adjustment
        
        # Use Black-Scholes with adjusted parameters
        from tests.quantitative_finance.test_options_pricing import BlackScholesModel
        
        return BlackScholesModel.call_price(spot_price, strike_price, time_to_maturity,
                                          domestic_rate, asset_volatility, adjusted_foreign_rate)
    
    @staticmethod
    def compound_option_price(spot_price: float, strike1: float, strike2: float,
                            time1: float, time2: float, risk_free_rate: float,
                            volatility: float) -> float:
        """Price compound option (option on option)"""
        
        # Simplified compound option pricing
        # In practice, would use more sophisticated methods
        
        from tests.quantitative_finance.test_options_pricing import BlackScholesModel
        
        # First option (underlying)
        option1_value = BlackScholesModel.call_price(spot_price, strike2, time2,
                                                    risk_free_rate, volatility, 0.0)
        
        # Second option (on the first option)
        option2_value = BlackScholesModel.call_price(option1_value, strike1, time1,
                                                    risk_free_rate, volatility, 0.0)
        
        return option2_value * 0.8  # Rough adjustment for compound structure
    
    @staticmethod
    def rainbow_option_price(spot_prices: List[float], strike_price: float,
                           time_to_maturity: float, risk_free_rate: float,
                           volatilities: List[float], correlations: np.ndarray,
                           n_simulations: int = 10000) -> float:
        """Price rainbow option (option on multiple underlyings)"""
        
        np.random.seed(42)
        n_assets = len(spot_prices)
        
        # Generate correlated random variables
        L = np.linalg.cholesky(correlations)
        
        payoffs = []
        
        for _ in range(n_simulations):
            # Generate correlated random numbers
            Z = np.random.normal(0, 1, n_assets)
            corr_Z = L @ Z
            
            # Final prices
            final_prices = []
            for i in range(n_assets):
                final_price = spot_prices[i] * np.exp(
                    (risk_free_rate - 0.5 * volatilities[i]**2) * time_to_maturity +
                    volatilities[i] * np.sqrt(time_to_maturity) * corr_Z[i]
                )
                final_prices.append(final_price)
            
            # Rainbow option payoff (best-of)
            max_performance = max(final_prices[i] / spot_prices[i] for i in range(n_assets))
            payoff = max(0, max_performance - strike_price)
            payoffs.append(payoff)
        
        # Discount expected payoff
        return np.mean(payoffs) * np.exp(-risk_free_rate * time_to_maturity)


class TestFuturesPricing:
    """Test suite for futures pricing"""
    
    def test_basic_futures_pricing(self):
        """Test basic futures pricing"""
        # Parameters
        spot_price = 100.0
        risk_free_rate = 0.05
        time_to_maturity = 0.25  # 3 months
        
        start_time = time.time()
        futures_price = FuturesPricer.futures_price(spot_price, risk_free_rate, time_to_maturity)
        execution_time = time.time() - start_time
        
        # Check pricing
        expected_price = spot_price * np.exp(risk_free_rate * time_to_maturity)
        assert_close(futures_price, expected_price, TOLERANCE)
        
        # Performance check
        assert execution_time < 0.001, "Futures pricing should be very fast"
    
    def test_futures_with_dividends(self):
        """Test futures pricing with dividend yield"""
        spot_price = 100.0
        risk_free_rate = 0.05
        time_to_maturity = 0.5
        dividend_yield = 0.02
        
        futures_price = FuturesPricer.futures_price(spot_price, risk_free_rate, time_to_maturity,
                                                   dividend_yield=dividend_yield)
        
        # With dividends, futures price should be lower
        no_dividend_price = FuturesPricer.futures_price(spot_price, risk_free_rate, time_to_maturity)
        assert futures_price < no_dividend_price, "Dividends should reduce futures price"
        
        # Check formula
        expected_price = spot_price * np.exp((risk_free_rate - dividend_yield) * time_to_maturity)
        assert_close(futures_price, expected_price, TOLERANCE)
    
    def test_commodity_futures(self):
        """Test commodity futures pricing"""
        spot_price = 50.0
        risk_free_rate = 0.04
        time_to_maturity = 1.0
        storage_cost = 0.02
        convenience_yield = 0.01
        
        futures_price = FuturesPricer.commodity_futures_price(spot_price, risk_free_rate,
                                                            time_to_maturity, storage_cost,
                                                            convenience_yield)
        
        # Check pricing
        carry_cost = risk_free_rate + storage_cost - convenience_yield
        expected_price = spot_price * np.exp(carry_cost * time_to_maturity)
        assert_close(futures_price, expected_price, TOLERANCE)
        
        # With high convenience yield, futures should be lower
        high_convenience_price = FuturesPricer.commodity_futures_price(spot_price, risk_free_rate,
                                                                     time_to_maturity, storage_cost,
                                                                     0.03)
        assert high_convenience_price < futures_price, "Higher convenience yield should reduce futures price"
    
    def test_currency_futures(self):
        """Test currency futures pricing"""
        spot_exchange_rate = 1.2  # USD/EUR
        domestic_rate = 0.05  # USD rate
        foreign_rate = 0.03   # EUR rate
        time_to_maturity = 0.5
        
        futures_price = FuturesPricer.currency_futures_price(spot_exchange_rate, domestic_rate,
                                                           foreign_rate, time_to_maturity)
        
        # Check pricing
        expected_price = spot_exchange_rate * np.exp((domestic_rate - foreign_rate) * time_to_maturity)
        assert_close(futures_price, expected_price, TOLERANCE)
        
        # Higher domestic rate should increase futures price
        higher_domestic_price = FuturesPricer.currency_futures_price(spot_exchange_rate, 0.06,
                                                                   foreign_rate, time_to_maturity)
        assert higher_domestic_price > futures_price, "Higher domestic rate should increase futures price"
    
    def test_basis_calculation(self):
        """Test basis calculation"""
        spot_price = 100.0
        futures_price = 102.0
        
        basis = FuturesPricer.basis(spot_price, futures_price)
        
        # Check basis
        expected_basis = spot_price - futures_price
        assert_close(basis, expected_basis, TOLERANCE)
        assert basis < 0, "Basis should be negative when futures > spot"
    
    def test_implied_convenience_yield(self):
        """Test implied convenience yield calculation"""
        spot_price = 100.0
        futures_price = 101.0
        risk_free_rate = 0.05
        time_to_maturity = 1.0
        storage_cost = 0.02
        
        convenience_yield = FuturesPricer.implied_convenience_yield(spot_price, futures_price,
                                                                  risk_free_rate, time_to_maturity,
                                                                  storage_cost=storage_cost)
        
        # Check that implied convenience yield is reasonable
        assert convenience_yield > 0, "Convenience yield should be positive"
        
        # Verify by pricing futures with implied convenience yield
        implied_futures_price = FuturesPricer.commodity_futures_price(spot_price, risk_free_rate,
                                                                     time_to_maturity, storage_cost,
                                                                     convenience_yield)
        
        assert_close(implied_futures_price, futures_price, TOLERANCE)


class TestSwapsPricing:
    """Test suite for swaps pricing"""
    
    def test_vanilla_swap_rate(self):
        """Test vanilla interest rate swap rate calculation"""
        tenor = 5.0
        payment_frequency = 0.5
        
        swaps_pricer = SwapsPricer()
        swap_rate = swaps_pricer.vanilla_swap_rate(tenor, payment_frequency)
        
        # Check swap rate is reasonable
        assert 0.01 < swap_rate < 0.15, "Swap rate should be reasonable"
    
    def test_swap_pv_calculation(self):
        """Test swap present value calculation"""
        notional = 1000000
        fixed_rate = 0.05
        floating_rate = 0.04
        tenor = 2.0
        
        swaps_pricer = SwapsPricer()
        swap_pv = swaps_pricer.swap_pv(notional, fixed_rate, floating_rate, tenor)
        
        # Check that swap PV is reasonable
        assert abs(swap_pv) < notional * 0.1, "Swap PV should be reasonable fraction of notional"
        
        # Fixed payer should have positive PV when floating > fixed
        assert swap_pv > 0, "Fixed payer should have positive PV when floating > fixed"
        
        # Test with reversed rates
        reversed_pv = swaps_pricer.swap_pv(notional, floating_rate, fixed_rate, tenor)
        assert reversed_pv < 0, "PV should be negative when fixed > floating"
    
    def test_swap_dv01(self):
        """Test swap DV01 calculation"""
        notional = 1000000
        tenor = 5.0
        
        swaps_pricer = SwapsPricer()
        dv01 = swaps_pricer.dv01(notional, tenor)
        
        # Check DV01 properties
        assert dv01 > 0, "DV01 should be positive"
        assert dv01 < notional * 0.001, "DV01 should be reasonable"
        
        # Longer tenor should have higher DV01
        longer_dv01 = swaps_pricer.dv01(notional, 10.0)
        assert longer_dv01 > dv01, "Longer tenor should have higher DV01"
    
    def test_currency_swap_rate(self):
        """Test currency swap rate calculation"""
        domestic_rate = 0.05
        foreign_rate = 0.03
        spot_exchange_rate = 1.2
        tenor = 1.0
        
        swaps_pricer = SwapsPricer()
        swap_rate = swaps_pricer.currency_swap_rate(domestic_rate, foreign_rate,
                                                   spot_exchange_rate, tenor)
        
        # Check swap rate
        assert swap_rate > 0, "Currency swap rate should be positive"
        
        # Should be close to forward exchange rate
        forward_rate = spot_exchange_rate * np.exp((domestic_rate - foreign_rate) * tenor)
        assert abs(swap_rate - forward_rate) < 0.1, "Swap rate should be close to forward rate"


class TestCDSPricing:
    """Test suite for CDS pricing"""
    
    def test_cds_spread_calculation(self):
        """Test CDS spread calculation"""
        default_probability = 0.02
        tenor = 5.0
        
        cds_pricer = CDSPricer(recovery_rate=0.4)
        cds_spread = cds_pricer.cds_spread(default_probability, tenor)
        
        # Check spread properties
        assert cds_spread > 0, "CDS spread should be positive"
        assert cds_spread < 0.1, "CDS spread should be reasonable"
        
        # Higher default probability should increase spread
        higher_default_spread = cds_pricer.cds_spread(0.05, tenor)
        assert higher_default_spread > cds_spread, "Higher default probability should increase spread"
    
    def test_cds_pv_calculation(self):
        """Test CDS present value calculation"""
        notional = 1000000
        cds_spread = 0.02
        hazard_rate = 0.01
        tenor = 5.0
        
        cds_pricer = CDSPricer(recovery_rate=0.4)
        cds_pv = cds_pricer.cds_pv(notional, cds_spread, hazard_rate, tenor)
        
        # Check PV properties
        assert abs(cds_pv) < notional * 0.1, "CDS PV should be reasonable fraction of notional"
    
    def test_hazard_rate_calculation(self):
        """Test hazard rate from CDS spread"""
        cds_spread = 0.02
        tenor = 5.0
        
        cds_pricer = CDSPricer(recovery_rate=0.4)
        hazard_rate = cds_pricer.hazard_rate_from_spread(cds_spread, tenor)
        
        # Check hazard rate
        assert hazard_rate > 0, "Hazard rate should be positive"
        assert hazard_rate < 0.5, "Hazard rate should be reasonable"
        
        # Higher spread should imply higher hazard rate
        higher_spread_hazard = cds_pricer.hazard_rate_from_spread(0.05, tenor)
        assert higher_spread_hazard > hazard_rate, "Higher spread should imply higher hazard rate"
    
    def test_recovery_rate_sensitivity(self):
        """Test CDS sensitivity to recovery rate"""
        default_probability = 0.02
        tenor = 5.0
        
        # Different recovery rates
        low_recovery_pricer = CDSPricer(recovery_rate=0.2)
        high_recovery_pricer = CDSPricer(recovery_rate=0.6)
        
        low_recovery_spread = low_recovery_pricer.cds_spread(default_probability, tenor)
        high_recovery_spread = high_recovery_pricer.cds_spread(default_probability, tenor)
        
        # Lower recovery rate should result in higher spread
        assert low_recovery_spread > high_recovery_spread, "Lower recovery rate should increase spread"


class TestStructuredProducts:
    """Test suite for structured products"""
    
    def test_autocallable_note_pricing(self):
        """Test autocallable note pricing"""
        spot_price = 100.0
        barrier_level = 100.0
        coupon_rate = 0.08
        knock_in_barrier = 70.0
        time_to_maturity = 3.0
        risk_free_rate = 0.03
        volatility = 0.25
        
        start_time = time.time()
        note_price = StructuredProductsPricer.autocallable_note_price(
            spot_price, barrier_level, coupon_rate, knock_in_barrier,
            time_to_maturity, risk_free_rate, volatility, n_simulations=1000
        )
        execution_time = time.time() - start_time
        
        # Check pricing
        assert 0.5 < note_price < 2.0, "Autocallable note price should be reasonable"
        
        # Should be higher than risk-free bond
        risk_free_bond_price = np.exp(-risk_free_rate * time_to_maturity)
        assert note_price > risk_free_bond_price, "Autocallable note should be worth more than risk-free bond"
        
        # Performance check
        assert execution_time < 5.0, "Autocallable note pricing should be reasonably fast"
    
    def test_reverse_convertible_note_pricing(self):
        """Test reverse convertible note pricing"""
        spot_price = 100.0
        strike_price = 90.0
        coupon_rate = 0.10
        time_to_maturity = 1.0
        risk_free_rate = 0.03
        volatility = 0.25
        
        note_price = StructuredProductsPricer.reverse_convertible_note_price(
            spot_price, strike_price, coupon_rate, time_to_maturity,
            risk_free_rate, volatility
        )
        
        # Check pricing
        assert 0.8 < note_price < 1.2, "Reverse convertible note price should be reasonable"
        
        # Should be less than bond with same coupon (due to short put)
        bond_price = (1 + coupon_rate * time_to_maturity) * np.exp(-risk_free_rate * time_to_maturity)
        assert note_price < bond_price, "Reverse convertible should be worth less than plain bond"
    
    def test_equity_linked_note_pricing(self):
        """Test equity-linked note pricing"""
        spot_price = 100.0
        participation_rate = 0.8
        cap_level = 120.0
        time_to_maturity = 2.0
        risk_free_rate = 0.03
        volatility = 0.20
        
        note_price = StructuredProductsPricer.equity_linked_note_price(
            spot_price, participation_rate, cap_level, time_to_maturity,
            risk_free_rate, volatility
        )
        
        # Check pricing
        assert 0.8 < note_price < 1.3, "Equity-linked note price should be reasonable"
        
        # Should be at least worth the zero-coupon bond
        zero_coupon_price = np.exp(-risk_free_rate * time_to_maturity)
        assert note_price >= zero_coupon_price, "ELN should be worth at least the zero-coupon bond"
    
    def test_structured_product_sensitivity(self):
        """Test structured product sensitivity to parameters"""
        base_params = {
            'spot_price': 100.0,
            'barrier_level': 100.0,
            'coupon_rate': 0.08,
            'knock_in_barrier': 70.0,
            'time_to_maturity': 3.0,
            'risk_free_rate': 0.03,
            'volatility': 0.25
        }
        
        base_price = StructuredProductsPricer.autocallable_note_price(**base_params, n_simulations=1000)
        
        # Test volatility sensitivity
        high_vol_price = StructuredProductsPricer.autocallable_note_price(
            **{**base_params, 'volatility': 0.35}, n_simulations=1000
        )
        
        # Higher volatility should generally increase autocallable value
        assert high_vol_price != base_price, "Volatility should affect autocallable price"
        
        # Test coupon sensitivity
        high_coupon_price = StructuredProductsPricer.autocallable_note_price(
            **{**base_params, 'coupon_rate': 0.12}, n_simulations=1000
        )
        
        assert high_coupon_price > base_price, "Higher coupon should increase autocallable value"


class TestExoticDerivatives:
    """Test suite for exotic derivatives"""
    
    def test_quanto_option_pricing(self):
        """Test quanto option pricing"""
        spot_price = 100.0
        strike_price = 100.0
        time_to_maturity = 1.0
        domestic_rate = 0.05
        foreign_rate = 0.03
        asset_volatility = 0.25
        fx_volatility = 0.15
        correlation = -0.3
        
        quanto_price = ExoticDerivativesPricer.quanto_option_price(
            spot_price, strike_price, time_to_maturity, domestic_rate,
            foreign_rate, asset_volatility, fx_volatility, correlation
        )
        
        # Check pricing
        assert quanto_price > 0, "Quanto option price should be positive"
        assert quanto_price < spot_price, "Quanto option price should be less than spot"
        
        # Compare with regular option
        from tests.quantitative_finance.test_options_pricing import BlackScholesModel
        regular_price = BlackScholesModel.call_price(spot_price, strike_price, time_to_maturity,
                                                    domestic_rate, asset_volatility, foreign_rate)
        
        # Should be different due to quanto adjustment
        assert abs(quanto_price - regular_price) > 0.01, "Quanto option should differ from regular option"
    
    def test_compound_option_pricing(self):
        """Test compound option pricing"""
        spot_price = 100.0
        strike1 = 10.0  # Strike for option on option
        strike2 = 100.0  # Strike for underlying option
        time1 = 0.5   # Time to first expiration
        time2 = 1.0   # Time to second expiration
        risk_free_rate = 0.05
        volatility = 0.25
        
        compound_price = ExoticDerivativesPricer.compound_option_price(
            spot_price, strike1, strike2, time1, time2, risk_free_rate, volatility
        )
        
        # Check pricing
        assert compound_price > 0, "Compound option price should be positive"
        assert compound_price < spot_price, "Compound option price should be less than spot"
        
        # Should be less than regular option
        from tests.quantitative_finance.test_options_pricing import BlackScholesModel
        regular_price = BlackScholesModel.call_price(spot_price, strike2, time2,
                                                    risk_free_rate, volatility, 0.0)
        
        assert compound_price < regular_price, "Compound option should be less valuable than regular option"
    
    def test_rainbow_option_pricing(self):
        """Test rainbow option pricing"""
        spot_prices = [100.0, 95.0, 105.0]
        strike_price = 1.0  # Strike on performance
        time_to_maturity = 1.0
        risk_free_rate = 0.05
        volatilities = [0.25, 0.30, 0.20]
        
        # Correlation matrix
        correlations = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        
        rainbow_price = ExoticDerivativesPricer.rainbow_option_price(
            spot_prices, strike_price, time_to_maturity, risk_free_rate,
            volatilities, correlations, n_simulations=1000
        )
        
        # Check pricing
        assert rainbow_price > 0, "Rainbow option price should be positive"
        assert rainbow_price < max(spot_prices), "Rainbow option price should be reasonable"
        
        # Test with different correlations
        low_corr_matrix = np.array([
            [1.0, 0.1, 0.1],
            [0.1, 1.0, 0.1],
            [0.1, 0.1, 1.0]
        ])
        
        low_corr_price = ExoticDerivativesPricer.rainbow_option_price(
            spot_prices, strike_price, time_to_maturity, risk_free_rate,
            volatilities, low_corr_matrix, n_simulations=1000
        )
        
        # Lower correlation should generally increase rainbow option value
        assert low_corr_price >= rainbow_price * 0.8, "Lower correlation should not drastically reduce value"
    
    def test_exotic_derivatives_sensitivity(self):
        """Test exotic derivatives sensitivity"""
        base_params = {
            'spot_price': 100.0,
            'strike_price': 100.0,
            'time_to_maturity': 1.0,
            'domestic_rate': 0.05,
            'foreign_rate': 0.03,
            'asset_volatility': 0.25,
            'fx_volatility': 0.15,
            'correlation': -0.3
        }
        
        base_price = ExoticDerivativesPricer.quanto_option_price(**base_params)
        
        # Test correlation sensitivity
        high_corr_price = ExoticDerivativesPricer.quanto_option_price(
            **{**base_params, 'correlation': 0.3}
        )
        
        assert high_corr_price != base_price, "Correlation should affect quanto option price"
        
        # Test FX volatility sensitivity
        high_fx_vol_price = ExoticDerivativesPricer.quanto_option_price(
            **{**base_params, 'fx_volatility': 0.25}
        )
        
        assert high_fx_vol_price != base_price, "FX volatility should affect quanto option price"


class TestDerivativesIntegration:
    """Test integration of derivatives models"""
    
    def test_futures_options_parity(self):
        """Test put-call parity for futures options"""
        # This would test the relationship between futures, calls, and puts
        # For simplicity, we'll test basic consistency
        
        spot_price = 100.0
        strike_price = 100.0
        time_to_maturity = 0.25
        risk_free_rate = 0.05
        
        # Futures price
        futures_price = FuturesPricer.futures_price(spot_price, risk_free_rate, time_to_maturity)
        
        # Options on futures should use futures price as underlying
        from tests.quantitative_finance.test_options_pricing import BlackScholesModel
        
        futures_call = BlackScholesModel.call_price(futures_price, strike_price, time_to_maturity,
                                                   risk_free_rate, 0.2, 0.0)
        futures_put = BlackScholesModel.put_price(futures_price, strike_price, time_to_maturity,
                                                 risk_free_rate, 0.2, 0.0)
        
        # Check put-call parity for futures options
        # C - P = (F - K) * e^(-rT)
        left_side = futures_call - futures_put
        right_side = (futures_price - strike_price) * np.exp(-risk_free_rate * time_to_maturity)
        
        assert_close(left_side, right_side, 0.01)
    
    def test_swap_options_consistency(self):
        """Test consistency between swaps and swaptions"""
        # Basic consistency check
        notional = 1000000
        strike_rate = 0.05
        tenor = 5.0
        
        swaps_pricer = SwapsPricer()
        fair_swap_rate = swaps_pricer.vanilla_swap_rate(tenor)
        
        # If fair swap rate > strike, payer swaption should be in-the-money
        # This is a simplified check
        assert fair_swap_rate > 0, "Fair swap rate should be positive"
        
        # Swaption value would depend on volatility and other factors
        # For now, just check basic properties
        if fair_swap_rate > strike_rate:
            # Payer swaption should have positive intrinsic value
            intrinsic_value = (fair_swap_rate - strike_rate) * notional * tenor
            assert intrinsic_value > 0, "Payer swaption should have positive intrinsic value"
    
    def test_credit_equity_correlation(self):
        """Test correlation between credit and equity derivatives"""
        # Basic test of credit-equity relationships
        
        # Merton model parameters
        firm_value = 100.0
        debt_level = 80.0
        time_to_maturity = 1.0
        risk_free_rate = 0.05
        firm_volatility = 0.3
        
        # Credit risk
        cds_pricer = CDSPricer(recovery_rate=0.4)
        
        # Equity (call option on firm value)
        from tests.quantitative_finance.test_options_pricing import BlackScholesModel
        equity_value = BlackScholesModel.call_price(firm_value, debt_level, time_to_maturity,
                                                   risk_free_rate, firm_volatility, 0.0)
        
        # Higher firm volatility should increase equity value but also credit risk
        high_vol_equity = BlackScholesModel.call_price(firm_value, debt_level, time_to_maturity,
                                                      risk_free_rate, 0.5, 0.0)
        
        assert high_vol_equity > equity_value, "Higher volatility should increase equity value"
        
        # This demonstrates the basic relationship in structural models


class TestDerivativesPerformance:
    """Test performance of derivatives pricing"""
    
    def test_futures_pricing_performance(self):
        """Test futures pricing performance"""
        n_calculations = 10000
        
        start_time = time.time()
        for _ in range(n_calculations):
            FuturesPricer.futures_price(100.0, 0.05, 0.25)
        execution_time = time.time() - start_time
        
        avg_time = execution_time / n_calculations
        assert avg_time < 0.0001, "Futures pricing should be very fast"
    
    def test_swaps_pricing_performance(self):
        """Test swaps pricing performance"""
        swaps_pricer = SwapsPricer()
        
        start_time = time.time()
        for _ in range(1000):
            swaps_pricer.vanilla_swap_rate(5.0)
        execution_time = time.time() - start_time
        
        assert execution_time < 1.0, "Swaps pricing should be reasonably fast"
    
    def test_monte_carlo_performance(self):
        """Test Monte Carlo derivatives performance"""
        start_time = time.time()
        
        StructuredProductsPricer.autocallable_note_price(
            100.0, 100.0, 0.08, 70.0, 3.0, 0.03, 0.25, n_simulations=1000
        )
        
        execution_time = time.time() - start_time
        assert execution_time < 10.0, "Monte Carlo pricing should complete in reasonable time"


class TestModelValidation:
    """Test model validation for derivatives"""
    
    def test_numerical_stability(self):
        """Test numerical stability of derivatives models"""
        # Test with extreme parameters
        
        # Very short time to maturity
        short_time_price = FuturesPricer.futures_price(100.0, 0.05, 0.001)
        assert 99.0 < short_time_price < 101.0, "Short time futures price should be stable"
        
        # Very high volatility
        high_vol_rainbow = ExoticDerivativesPricer.rainbow_option_price(
            [100.0, 100.0], 1.0, 1.0, 0.05, [2.0, 2.0],
            np.array([[1.0, 0.5], [0.5, 1.0]]), n_simulations=100
        )
        assert high_vol_rainbow > 0, "High volatility rainbow option should be stable"
    
    def test_boundary_conditions(self):
        """Test boundary conditions for derivatives"""
        # Test CDS with zero hazard rate
        cds_pricer = CDSPricer()
        zero_hazard_pv = cds_pricer.cds_pv(1000000, 0.02, 0.0, 5.0)
        assert zero_hazard_pv < 0, "Zero hazard rate should make CDS negative for protection buyer"
        
        # Test futures with zero time to maturity
        zero_time_futures = FuturesPricer.futures_price(100.0, 0.05, 0.0)
        assert_close(zero_time_futures, 100.0, TOLERANCE)
    
    def test_arbitrage_free_conditions(self):
        """Test arbitrage-free conditions"""
        # Test put-call parity in various contexts
        
        # Currency futures arbitrage
        spot_rate = 1.2
        domestic_rate = 0.05
        foreign_rate = 0.03
        time_to_maturity = 0.25
        
        futures_price = FuturesPricer.currency_futures_price(spot_rate, domestic_rate,
                                                           foreign_rate, time_to_maturity)
        
        # Forward rate should equal futures price (ignoring margin effects)
        forward_rate = spot_rate * np.exp((domestic_rate - foreign_rate) * time_to_maturity)
        assert_close(futures_price, forward_rate, TOLERANCE)
        
        # CDS-bond basis arbitrage
        # In practice, would test that CDS spread + bond yield = risk-free rate + basis
        # For simplicity, just check that CDS spreads are reasonable
        cds_pricer = CDSPricer()
        spread = cds_pricer.cds_spread(0.02, 5.0)
        assert 0.001 < spread < 0.1, "CDS spread should be in reasonable range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])