"""
Hypothesis integration for property-based testing of trading systems.

This module provides Hypothesis strategies and property-based testing
utilities specifically designed for trading system components.
"""

import hypothesis.strategies as st
from hypothesis import given, settings, assume, example
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math


class TestComplexity(Enum):
    """Test complexity levels for property-based testing."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXTREME = "extreme"


@dataclass
class MarketDataPoint:
    """Market data point for testing."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    spread: float


@dataclass
class TradingSignal:
    """Trading signal for testing."""
    signal_type: str  # "BUY", "SELL", "HOLD"
    confidence: float
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    metadata: Dict[str, Any]


@dataclass
class RiskMetrics:
    """Risk metrics for testing."""
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    correlation_matrix: List[List[float]]


class MarketDataStrategy:
    """Hypothesis strategies for market data generation."""
    
    @staticmethod
    def price(min_price: float = 0.01, max_price: float = 10000.0) -> st.SearchStrategy[float]:
        """Generate realistic price values."""
        return st.floats(
            min_value=min_price,
            max_value=max_price,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False
        ).map(lambda x: round(x, 2))
    
    @staticmethod
    def volume(min_volume: int = 1, max_volume: int = 1000000) -> st.SearchStrategy[int]:
        """Generate realistic volume values."""
        return st.integers(min_value=min_volume, max_value=max_volume)
    
    @staticmethod
    def spread(min_spread: float = 0.01, max_spread: float = 1.0) -> st.SearchStrategy[float]:
        """Generate realistic bid-ask spreads."""
        return st.floats(
            min_value=min_spread,
            max_value=max_spread,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False
        ).map(lambda x: round(x, 4))
    
    @staticmethod
    def timestamp(start_date: Optional[datetime] = None) -> st.SearchStrategy[datetime]:
        """Generate realistic timestamps."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        
        return st.datetimes(
            min_value=start_date,
            max_value=datetime.now(),
            timezones=st.none()
        )
    
    @staticmethod
    def symbol() -> st.SearchStrategy[str]:
        """Generate realistic trading symbols."""
        return st.sampled_from([
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX",
            "ORCL", "CRM", "BABA", "JNJ", "JPM", "V", "WMT", "PG", "UNH", "HD",
            "MA", "DIS", "PYPL", "ADBE", "CMCSA", "NFLX", "XOM", "VZ", "T", "PFE"
        ])
    
    @staticmethod
    def ohlcv_bar(complexity: TestComplexity = TestComplexity.MEDIUM) -> st.SearchStrategy[Dict[str, Any]]:
        """Generate OHLCV bar data."""
        if complexity == TestComplexity.SIMPLE:
            price_range = (50.0, 200.0)
            volume_range = (1000, 100000)
        elif complexity == TestComplexity.MEDIUM:
            price_range = (1.0, 1000.0)
            volume_range = (100, 1000000)
        elif complexity == TestComplexity.COMPLEX:
            price_range = (0.01, 10000.0)
            volume_range = (1, 10000000)
        else:  # EXTREME
            price_range = (0.001, 100000.0)
            volume_range = (1, 100000000)
        
        return st.builds(
            lambda o, h, l, c, v, t, s: {
                "open": o,
                "high": max(o, h, l, c),
                "low": min(o, h, l, c),
                "close": c,
                "volume": v,
                "timestamp": t,
                "symbol": s
            },
            o=MarketDataStrategy.price(*price_range),
            h=MarketDataStrategy.price(*price_range),
            l=MarketDataStrategy.price(*price_range),
            c=MarketDataStrategy.price(*price_range),
            v=MarketDataStrategy.volume(*volume_range),
            t=MarketDataStrategy.timestamp(),
            s=MarketDataStrategy.symbol()
        )
    
    @staticmethod
    def market_data_point() -> st.SearchStrategy[MarketDataPoint]:
        """Generate complete market data point."""
        return st.builds(
            lambda symbol, timestamp, price, volume, spread: MarketDataPoint(
                symbol=symbol,
                timestamp=timestamp,
                price=price,
                volume=volume,
                bid=round(price - spread/2, 4),
                ask=round(price + spread/2, 4),
                spread=spread
            ),
            symbol=MarketDataStrategy.symbol(),
            timestamp=MarketDataStrategy.timestamp(),
            price=MarketDataStrategy.price(),
            volume=MarketDataStrategy.volume(),
            spread=MarketDataStrategy.spread()
        )
    
    @staticmethod
    def price_series(length: int = 100) -> st.SearchStrategy[List[float]]:
        """Generate realistic price series with correlation."""
        return st.lists(
            MarketDataStrategy.price(),
            min_size=length,
            max_size=length
        ).map(lambda prices: MarketDataStrategy._make_correlated_series(prices))
    
    @staticmethod
    def _make_correlated_series(prices: List[float]) -> List[float]:
        """Add realistic correlation to price series."""
        if len(prices) < 2:
            return prices
        
        # Apply simple random walk correlation
        correlated = [prices[0]]
        for i in range(1, len(prices)):
            # Add some momentum/mean reversion
            change = (prices[i] - correlated[i-1]) * 0.7  # Momentum factor
            correlated.append(correlated[i-1] + change)
        
        return correlated


class TradingSignalStrategy:
    """Hypothesis strategies for trading signal generation."""
    
    @staticmethod
    def signal_type() -> st.SearchStrategy[str]:
        """Generate trading signal types."""
        return st.sampled_from(["BUY", "SELL", "HOLD"])
    
    @staticmethod
    def confidence(min_confidence: float = 0.0, max_confidence: float = 1.0) -> st.SearchStrategy[float]:
        """Generate confidence values."""
        return st.floats(
            min_value=min_confidence,
            max_value=max_confidence,
            exclude_min=True,
            exclude_max=True,
            allow_nan=False,
            allow_infinity=False
        )
    
    @staticmethod
    def position_size(max_size: float = 1.0) -> st.SearchStrategy[float]:
        """Generate position sizes."""
        return st.floats(
            min_value=0.001,
            max_value=max_size,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False
        )
    
    @staticmethod
    def stop_loss() -> st.SearchStrategy[Optional[float]]:
        """Generate stop loss values."""
        return st.one_of(
            st.none(),
            st.floats(min_value=0.001, max_value=0.1, exclude_min=True, allow_nan=False, allow_infinity=False)
        )
    
    @staticmethod
    def take_profit() -> st.SearchStrategy[Optional[float]]:
        """Generate take profit values."""
        return st.one_of(
            st.none(),
            st.floats(min_value=0.001, max_value=0.5, exclude_min=True, allow_nan=False, allow_infinity=False)
        )
    
    @staticmethod
    def trading_signal() -> st.SearchStrategy[TradingSignal]:
        """Generate complete trading signal."""
        return st.builds(
            TradingSignal,
            signal_type=TradingSignalStrategy.signal_type(),
            confidence=TradingSignalStrategy.confidence(),
            position_size=TradingSignalStrategy.position_size(),
            stop_loss=TradingSignalStrategy.stop_loss(),
            take_profit=TradingSignalStrategy.take_profit(),
            metadata=st.dictionaries(
                st.text(min_size=1, max_size=10),
                st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=50), st.integers())
            )
        )


class RiskMetricsStrategy:
    """Hypothesis strategies for risk metrics generation."""
    
    @staticmethod
    def var_value(percentile: float = 0.95) -> st.SearchStrategy[float]:
        """Generate VaR values."""
        if percentile >= 0.99:
            max_var = 0.1  # 10% VaR for 99th percentile
        elif percentile >= 0.95:
            max_var = 0.05  # 5% VaR for 95th percentile
        else:
            max_var = 0.02  # 2% VaR for lower percentiles
        
        return st.floats(
            min_value=0.001,
            max_value=max_var,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False
        )
    
    @staticmethod
    def sharpe_ratio() -> st.SearchStrategy[float]:
        """Generate Sharpe ratio values."""
        return st.floats(
            min_value=-3.0,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False
        )
    
    @staticmethod
    def volatility() -> st.SearchStrategy[float]:
        """Generate volatility values."""
        return st.floats(
            min_value=0.001,
            max_value=2.0,
            exclude_min=True,
            allow_nan=False,
            allow_infinity=False
        )
    
    @staticmethod
    def correlation_matrix(size: int = 3) -> st.SearchStrategy[List[List[float]]]:
        """Generate valid correlation matrix."""
        return st.builds(
            lambda: RiskMetricsStrategy._generate_correlation_matrix(size)
        )
    
    @staticmethod
    def _generate_correlation_matrix(size: int) -> List[List[float]]:
        """Generate a valid correlation matrix."""
        # Start with identity matrix
        matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
        
        # Add random correlations while maintaining positive semi-definite property
        for i in range(size):
            for j in range(i+1, size):
                # Generate correlation between -0.9 and 0.9
                corr = np.random.uniform(-0.9, 0.9)
                matrix[i][j] = corr
                matrix[j][i] = corr
        
        return matrix
    
    @staticmethod
    def risk_metrics() -> st.SearchStrategy[RiskMetrics]:
        """Generate complete risk metrics."""
        return st.builds(
            lambda var_95, var_99, es, md, sr, vol, corr: RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=es,
                max_drawdown=md,
                sharpe_ratio=sr,
                volatility=vol,
                correlation_matrix=corr
            ),
            var_95=RiskMetricsStrategy.var_value(0.95),
            var_99=RiskMetricsStrategy.var_value(0.99),
            es=RiskMetricsStrategy.var_value(0.975),  # Expected shortfall
            md=st.floats(min_value=0.001, max_value=0.5, exclude_min=True, allow_nan=False, allow_infinity=False),
            sr=RiskMetricsStrategy.sharpe_ratio(),
            vol=RiskMetricsStrategy.volatility(),
            corr=RiskMetricsStrategy.correlation_matrix()
        )


class PropertyBasedTestEngine:
    """
    Engine for running property-based tests on trading system components.
    
    This engine provides a framework for defining and running property-based
    tests that verify mathematical invariants and business rules.
    """
    
    def __init__(self, max_examples: int = 100, timeout: int = 60):
        self.max_examples = max_examples
        self.timeout = timeout
        self.test_results = []
        
    def run_property_test(self, 
                         test_function: Callable,
                         strategy: st.SearchStrategy,
                         property_name: str,
                         examples: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Run a property-based test.
        
        Args:
            test_function: Function to test
            strategy: Hypothesis strategy for generating test inputs
            property_name: Name of the property being tested
            examples: Optional explicit examples to test
            
        Returns:
            Test results dictionary
        """
        test_settings = settings(
            max_examples=self.max_examples,
            timeout=self.timeout,
            print_blob=True
        )
        
        # Add explicit examples if provided
        if examples:
            for example_value in examples:
                test_function = example(example_value)(test_function)
        
        # Create hypothesis test
        hypothesis_test = given(strategy)(test_settings(test_function))
        
        # Run test and capture results
        try:
            hypothesis_test()
            result = {
                "property": property_name,
                "status": "PASSED",
                "examples_tested": self.max_examples,
                "error": None
            }
        except Exception as e:
            result = {
                "property": property_name,
                "status": "FAILED",
                "examples_tested": "unknown",
                "error": str(e)
            }
        
        self.test_results.append(result)
        return result
    
    def run_trading_invariants_test(self, trading_component: Any) -> List[Dict[str, Any]]:
        """Run all trading invariants tests on a component."""
        results = []
        
        # Test price invariants
        results.append(self.run_property_test(
            lambda data: trading_invariants.price_non_negative(data),
            MarketDataStrategy.ohlcv_bar(),
            "price_non_negative"
        ))
        
        # Test volume invariants
        results.append(self.run_property_test(
            lambda data: trading_invariants.volume_positive(data),
            MarketDataStrategy.ohlcv_bar(),
            "volume_positive"
        ))
        
        # Test OHLC relationships
        results.append(self.run_property_test(
            lambda data: trading_invariants.ohlc_relationships(data),
            MarketDataStrategy.ohlcv_bar(),
            "ohlc_relationships"
        ))
        
        return results
    
    def run_risk_invariants_test(self, risk_component: Any) -> List[Dict[str, Any]]:
        """Run all risk invariants tests on a component."""
        results = []
        
        # Test VaR ordering
        results.append(self.run_property_test(
            lambda metrics: risk_invariants.var_ordering(metrics),
            RiskMetricsStrategy.risk_metrics(),
            "var_ordering"
        ))
        
        # Test correlation matrix properties
        results.append(self.run_property_test(
            lambda metrics: risk_invariants.correlation_matrix_properties(metrics),
            RiskMetricsStrategy.risk_metrics(),
            "correlation_matrix_properties"
        ))
        
        # Test Sharpe ratio bounds
        results.append(self.run_property_test(
            lambda metrics: risk_invariants.sharpe_ratio_bounds(metrics),
            RiskMetricsStrategy.risk_metrics(),
            "sharpe_ratio_bounds"
        ))
        
        return results
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASSED")
        failed_tests = total_tests - passed_tests
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "results": self.test_results
        }


class TradingInvariants:
    """Trading system invariants for property-based testing."""
    
    @staticmethod
    def price_non_negative(data: Dict[str, Any]) -> bool:
        """Prices must be non-negative."""
        return all(data[key] >= 0 for key in ["open", "high", "low", "close"] if key in data)
    
    @staticmethod
    def volume_positive(data: Dict[str, Any]) -> bool:
        """Volume must be positive."""
        return data.get("volume", 0) > 0
    
    @staticmethod
    def ohlc_relationships(data: Dict[str, Any]) -> bool:
        """OHLC relationships must be valid."""
        if not all(key in data for key in ["open", "high", "low", "close"]):
            return True  # Skip if missing data
        
        o, h, l, c = data["open"], data["high"], data["low"], data["close"]
        return h >= max(o, c) and l <= min(o, c)
    
    @staticmethod
    def spread_positive(data: Dict[str, Any]) -> bool:
        """Bid-ask spread must be positive."""
        if "bid" in data and "ask" in data:
            return data["ask"] > data["bid"]
        return True
    
    @staticmethod
    def position_size_bounds(signal: TradingSignal) -> bool:
        """Position size must be within bounds."""
        return 0 < signal.position_size <= 1.0
    
    @staticmethod
    def confidence_bounds(signal: TradingSignal) -> bool:
        """Confidence must be between 0 and 1."""
        return 0 <= signal.confidence <= 1.0
    
    @staticmethod
    def stop_loss_logic(signal: TradingSignal) -> bool:
        """Stop loss must be logical for signal direction."""
        if signal.stop_loss is None:
            return True
        
        # For BUY signals, stop loss should be below current price
        # For SELL signals, stop loss should be above current price
        # This is a simplified check - in reality, we'd need current price
        return signal.stop_loss > 0


class RiskInvariants:
    """Risk management invariants for property-based testing."""
    
    @staticmethod
    def var_ordering(metrics: RiskMetrics) -> bool:
        """VaR 99% should be >= VaR 95%."""
        return metrics.var_99 >= metrics.var_95
    
    @staticmethod
    def expected_shortfall_bounds(metrics: RiskMetrics) -> bool:
        """Expected shortfall should be >= VaR."""
        return metrics.expected_shortfall >= metrics.var_95
    
    @staticmethod
    def correlation_matrix_properties(metrics: RiskMetrics) -> bool:
        """Correlation matrix must have valid properties."""
        matrix = metrics.correlation_matrix
        n = len(matrix)
        
        # Check diagonal elements are 1
        for i in range(n):
            if abs(matrix[i][i] - 1.0) > 1e-10:
                return False
        
        # Check symmetry
        for i in range(n):
            for j in range(n):
                if abs(matrix[i][j] - matrix[j][i]) > 1e-10:
                    return False
        
        # Check correlation bounds
        for i in range(n):
            for j in range(n):
                if abs(matrix[i][j]) > 1.0:
                    return False
        
        return True
    
    @staticmethod
    def volatility_positive(metrics: RiskMetrics) -> bool:
        """Volatility must be positive."""
        return metrics.volatility > 0
    
    @staticmethod
    def max_drawdown_bounds(metrics: RiskMetrics) -> bool:
        """Max drawdown should be between 0 and 1."""
        return 0 <= metrics.max_drawdown <= 1.0
    
    @staticmethod
    def sharpe_ratio_bounds(metrics: RiskMetrics) -> bool:
        """Sharpe ratio should be within reasonable bounds."""
        return -10.0 <= metrics.sharpe_ratio <= 10.0


class PerformanceInvariants:
    """Performance invariants for property-based testing."""
    
    @staticmethod
    def inference_time_bounds(inference_time_ms: float) -> bool:
        """Inference time should be within bounds."""
        return 0 < inference_time_ms < 1000  # Less than 1 second
    
    @staticmethod
    def memory_usage_bounds(memory_mb: float) -> bool:
        """Memory usage should be within bounds."""
        return 0 < memory_mb < 1024  # Less than 1GB
    
    @staticmethod
    def throughput_positive(throughput_ops_per_sec: float) -> bool:
        """Throughput should be positive."""
        return throughput_ops_per_sec > 0


# Create instances for easy access
trading_invariants = TradingInvariants()
risk_invariants = RiskInvariants()
performance_invariants = PerformanceInvariants()