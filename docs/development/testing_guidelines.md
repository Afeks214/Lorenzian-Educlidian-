# Testing Guidelines

## Overview

Comprehensive testing is critical for the GrandModel trading system to ensure reliability, performance, and correctness in high-stakes financial environments. This document outlines testing strategies, standards, and best practices for maintaining system quality across all components.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Types and Levels](#test-types-and-levels)
- [Testing Framework Setup](#testing-framework-setup)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Performance Testing](#performance-testing)
- [Financial Accuracy Testing](#financial-accuracy-testing)
- [MARL Testing](#marl-testing)
- [Mock Objects and Fixtures](#mock-objects-and-fixtures)
- [Test Data Management](#test-data-management)
- [Continuous Testing](#continuous-testing)

## Testing Philosophy

### Core Testing Principles

1. **Financial Accuracy First**: All calculations must be precisely correct
2. **Performance Under Load**: Tests must verify real-world performance
3. **Risk-Free Testing**: Never risk real money in automated tests
4. **Deterministic Results**: Tests must produce consistent, repeatable results
5. **Comprehensive Coverage**: Critical paths require 100% test coverage
6. **Fast Feedback**: Unit tests must run quickly for development velocity
7. **Production-Like Testing**: Integration tests simulate real conditions

### Testing Pyramid

```
                    /\
                   /  \
              E2E /____\ (Few, Slow, Expensive)
                 /      \
        Integration /________\ (Some, Medium Speed)
                   /          \
              Unit /____________\ (Many, Fast, Cheap)
```

- **Unit Tests (70%)**: Fast, isolated, test individual components
- **Integration Tests (20%)**: Test component interactions
- **End-to-End Tests (10%)**: Test complete workflows

## Test Types and Levels

### Test Categories

```python
import pytest

# Test markers for categorization
@pytest.mark.unit
def test_kelly_calculation():
    """Unit test for Kelly Criterion calculation"""
    pass

@pytest.mark.integration
def test_order_execution_flow():
    """Integration test for order execution"""
    pass

@pytest.mark.performance
def test_high_frequency_processing():
    """Performance test for high-frequency data processing"""
    pass

@pytest.mark.financial
def test_portfolio_pnl_accuracy():
    """Financial accuracy test for P&L calculations"""
    pass

@pytest.mark.marl
def test_agent_coordination():
    """MARL-specific test for agent coordination"""
    pass

@pytest.mark.regression
def test_previous_bug_fix():
    """Regression test for previously fixed bug"""
    pass

@pytest.mark.smoke
def test_system_startup():
    """Smoke test for basic system functionality"""
    pass
```

### Test Execution Levels

```bash
# Run different test suites
pytest tests/ -m unit                    # Fast unit tests
pytest tests/ -m integration            # Integration tests
pytest tests/ -m "unit or integration"  # Combined test suite
pytest tests/ -m performance --timeout=300  # Performance tests with timeout
pytest tests/ -m financial              # Financial accuracy tests
pytest tests/ --cov=src --cov-report=html  # Coverage analysis
```

## Testing Framework Setup

### Core Testing Dependencies

```python
# requirements-test.txt
pytest==7.4.0
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.11.1
pytest-benchmark==4.0.0
pytest-xdist==3.3.1
hypothesis==6.82.0
factory-boy==3.3.0
freezegun==1.2.2
responses==0.23.3
```

### Pytest Configuration

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (medium speed)
    performance: Performance tests (slow)
    financial: Financial accuracy tests
    marl: MARL-specific tests
    regression: Regression tests
    smoke: Smoke tests for basic functionality
    slow: Slow tests (skip in development)
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
asyncio_mode = auto
```

### Test Directory Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests
│   ├── __init__.py
│   ├── test_core/             # Core component tests
│   ├── test_data/             # Data pipeline tests
│   ├── test_analysis/         # Analysis component tests
│   ├── test_intelligence/     # MARL and AI tests
│   ├── test_execution/        # Execution component tests
│   └── test_risk/             # Risk management tests
├── integration/               # Integration tests
│   ├── __init__.py
│   ├── test_data_flow/        # End-to-end data flow
│   ├── test_trading_workflow/ # Complete trading workflows
│   └── test_system_startup/   # System initialization
├── performance/               # Performance tests
│   ├── __init__.py
│   ├── test_latency/          # Latency benchmarks
│   ├── test_throughput/       # Throughput benchmarks
│   └── test_memory/           # Memory usage tests
├── financial/                 # Financial accuracy tests
│   ├── __init__.py
│   ├── test_calculations/     # Mathematical accuracy
│   ├── test_pnl/             # P&L calculations
│   └── test_risk_metrics/     # Risk calculation accuracy
├── fixtures/                  # Test data and fixtures
│   ├── market_data/          # Sample market data
│   ├── configurations/       # Test configurations
│   └── models/               # Test ML models
└── utils/                     # Test utilities
    ├── __init__.py
    ├── factories.py          # Data factories
    ├── mocks.py              # Mock objects
    └── helpers.py            # Test helper functions
```

## Unit Testing

### Component Unit Tests

```python
# tests/unit/test_risk/test_kelly_calculator.py
import unittest
from decimal import Decimal
import pytest
import numpy as np

from src.risk.calculators.kelly_calculator import KellyCalculator, KellyResult
from src.risk.exceptions import InvalidProbabilityError, InvalidRatioError

class TestKellyCalculator(unittest.TestCase):
    """Unit tests for Kelly Criterion calculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = KellyCalculator()
        self.default_config = {
            'max_risk_fraction': 0.25,
            'min_edge_threshold': 0.01
        }
    
    def test_optimal_kelly_calculation(self):
        """Test Kelly fraction calculation with optimal parameters"""
        # Arrange
        win_probability = 0.6
        win_loss_ratio = 1.5
        
        # Act
        result = self.calculator.calculate(win_probability, win_loss_ratio, self.default_config)
        
        # Assert
        expected_kelly = (0.6 * 1.5 - 0.4) / 1.5
        self.assertAlmostEqual(result.kelly_fraction, expected_kelly, places=6)
        self.assertTrue(result.is_favorable)
        self.assertGreater(result.expected_growth, 0)
    
    def test_unfavorable_bet_returns_zero(self):
        """Test that unfavorable bets return zero position size"""
        # Arrange: Unfavorable odds
        win_probability = 0.4
        win_loss_ratio = 1.0
        
        # Act
        result = self.calculator.calculate(win_probability, win_loss_ratio, self.default_config)
        
        # Assert
        self.assertEqual(result.recommended_fraction, 0.0)
        self.assertFalse(result.is_favorable)
        self.assertLessEqual(result.expected_growth, 0)
    
    def test_risk_adjustment_caps_position_size(self):
        """Test that risk adjustment prevents over-leveraging"""
        # Arrange: Very favorable odds that would suggest large position
        win_probability = 0.9
        win_loss_ratio = 2.0
        
        # Act
        result = self.calculator.calculate(win_probability, win_loss_ratio, self.default_config)
        
        # Assert
        self.assertLessEqual(result.recommended_fraction, self.default_config['max_risk_fraction'])
        self.assertGreater(result.kelly_fraction, result.recommended_fraction)
    
    @pytest.mark.parametrize("win_prob,expected_valid", [
        (0.0, True),   # Edge case: zero probability
        (0.5, True),   # Fair bet
        (1.0, True),   # Edge case: certain win
        (-0.1, False), # Invalid: negative probability
        (1.1, False),  # Invalid: probability > 1
    ])
    def test_probability_validation(self, win_prob, expected_valid):
        """Test probability parameter validation"""
        win_loss_ratio = 1.5
        
        if expected_valid:
            # Should not raise exception
            result = self.calculator.calculate(win_prob, win_loss_ratio, self.default_config)
            self.assertIsInstance(result, KellyResult)
        else:
            # Should raise validation error
            with self.assertRaises(InvalidProbabilityError):
                self.calculator.calculate(win_prob, win_loss_ratio, self.default_config)
    
    def test_win_loss_ratio_validation(self):
        """Test win/loss ratio parameter validation"""
        win_probability = 0.6
        
        # Valid ratios
        valid_ratios = [0.1, 1.0, 2.5, 10.0]
        for ratio in valid_ratios:
            result = self.calculator.calculate(win_probability, ratio, self.default_config)
            self.assertIsInstance(result, KellyResult)
        
        # Invalid ratios
        invalid_ratios = [0.0, -0.5, -1.0]
        for ratio in invalid_ratios:
            with self.assertRaises(InvalidRatioError):
                self.calculator.calculate(win_probability, ratio, self.default_config)
    
    def test_precision_with_decimal_arithmetic(self):
        """Test calculation precision using Decimal arithmetic"""
        # Arrange: Use Decimal for precise financial calculations
        win_probability = Decimal('0.55')
        win_loss_ratio = Decimal('1.2')
        
        # Act
        result = self.calculator.calculate_precise(win_probability, win_loss_ratio, self.default_config)
        
        # Assert: Verify precision to 8 decimal places
        expected_kelly = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        self.assertEqual(result.kelly_fraction, expected_kelly)
    
    def test_edge_case_extreme_values(self):
        """Test behavior with extreme but valid values"""
        # Test very small edge
        result_small_edge = self.calculator.calculate(0.501, 1.0, self.default_config)
        self.assertAlmostEqual(result_small_edge.kelly_fraction, 0.002, places=3)
        
        # Test very large win/loss ratio
        result_large_ratio = self.calculator.calculate(0.6, 100.0, self.default_config)
        self.assertLess(result_large_ratio.recommended_fraction, 0.25)  # Should be capped
    
    def test_performance_benchmark(self):
        """Test calculation performance for high-frequency usage"""
        import time
        
        # Arrange
        win_probability = 0.6
        win_loss_ratio = 1.5
        iterations = 10000
        
        # Act
        start_time = time.time()
        for _ in range(iterations):
            result = self.calculator.calculate(win_probability, win_loss_ratio, self.default_config)
        end_time = time.time()
        
        # Assert: Should complete 10,000 calculations in under 1 second
        total_time = end_time - start_time
        calculations_per_second = iterations / total_time
        self.assertGreater(calculations_per_second, 10000,
                         f"Kelly calculation too slow: {calculations_per_second:.0f} calc/sec")

# Property-based testing with Hypothesis
from hypothesis import given, strategies as st

class TestKellyCalculatorProperties(unittest.TestCase):
    """Property-based tests for Kelly Calculator"""
    
    def setUp(self):
        self.calculator = KellyCalculator()
        self.config = {'max_risk_fraction': 0.25, 'min_edge_threshold': 0.01}
    
    @given(
        win_prob=st.floats(min_value=0.01, max_value=0.99),
        win_loss_ratio=st.floats(min_value=0.1, max_value=10.0)
    )
    def test_kelly_fraction_properties(self, win_prob, win_loss_ratio):
        """Test mathematical properties of Kelly fraction"""
        result = self.calculator.calculate(win_prob, win_loss_ratio, self.config)
        
        # Kelly fraction should be between -1 and 1
        self.assertGreaterEqual(result.kelly_fraction, -1.0)
        self.assertLessEqual(result.kelly_fraction, 1.0)
        
        # Recommended fraction should never exceed max risk
        self.assertLessEqual(result.recommended_fraction, self.config['max_risk_fraction'])
        
        # If favorable, recommended fraction should be positive
        if result.is_favorable:
            self.assertGreater(result.recommended_fraction, 0)
        else:
            self.assertEqual(result.recommended_fraction, 0)
```

### Async Component Testing

```python
# tests/unit/test_data/test_data_handler.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.data.handlers.market_data_handler import MarketDataHandler
from src.core.events import Event, EventType, TickData

@pytest.mark.asyncio
class TestMarketDataHandlerAsync:
    """Test async operations in MarketDataHandler"""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Mock event bus for testing"""
        return MagicMock()
    
    @pytest.fixture
    def data_handler_config(self):
        """Test configuration for data handler"""
        return {
            'source_type': 'test',
            'symbols': ['AAPL', 'MSFT'],
            'connection': {
                'host': 'localhost',
                'port': 9999,
                'timeout': 5.0
            }
        }
    
    @pytest.fixture
    def data_handler(self, data_handler_config, mock_event_bus):
        """Create data handler for testing"""
        return MarketDataHandler(data_handler_config, mock_event_bus)
    
    async def test_successful_connection(self, data_handler):
        """Test successful connection to data source"""
        with patch('src.data.handlers.market_data_handler.TestDataSource') as mock_source:
            # Arrange
            mock_source_instance = AsyncMock()
            mock_source.return_value = mock_source_instance
            mock_source_instance.connect.return_value = True
            
            # Act
            success = await data_handler.connect()
            
            # Assert
            assert success is True
            mock_source_instance.connect.assert_called_once()
    
    async def test_connection_failure_handling(self, data_handler):
        """Test handling of connection failures"""
        with patch('src.data.handlers.market_data_handler.TestDataSource') as mock_source:
            # Arrange
            mock_source_instance = AsyncMock()
            mock_source.return_value = mock_source_instance
            mock_source_instance.connect.side_effect = ConnectionError("Connection failed")
            
            # Act & Assert
            with pytest.raises(ConnectionError):
                await data_handler.connect()
    
    async def test_tick_data_processing(self, data_handler, mock_event_bus):
        """Test processing of incoming tick data"""
        # Arrange
        sample_tick = TickData(
            symbol='AAPL',
            timestamp=datetime.now(),
            price=150.25,
            volume=1000
        )
        
        # Act
        await data_handler.process_tick(sample_tick)
        
        # Assert
        mock_event_bus.publish.assert_called_once()
        published_event = mock_event_bus.publish.call_args[0][0]
        assert published_event.event_type == EventType.NEW_TICK
        assert published_event.payload == sample_tick
    
    async def test_high_frequency_tick_processing(self, data_handler):
        """Test high-frequency tick processing performance"""
        # Arrange
        ticks = [
            TickData(symbol='AAPL', timestamp=datetime.now(), price=150.0 + i * 0.01, volume=100)
            for i in range(10000)
        ]
        
        # Act
        start_time = asyncio.get_event_loop().time()
        
        tasks = [data_handler.process_tick(tick) for tick in ticks]
        await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        
        # Assert
        processing_time = end_time - start_time
        ticks_per_second = len(ticks) / processing_time
        
        assert ticks_per_second > 50000, f"Processing too slow: {ticks_per_second:.0f} ticks/sec"
    
    async def test_reconnection_logic(self, data_handler):
        """Test automatic reconnection on connection loss"""
        with patch('src.data.handlers.market_data_handler.TestDataSource') as mock_source:
            # Arrange
            mock_source_instance = AsyncMock()
            mock_source.return_value = mock_source_instance
            
            # Simulate connection loss then recovery
            mock_source_instance.is_connected.side_effect = [True, False, False, True]
            mock_source_instance.reconnect.return_value = True
            
            # Act
            await data_handler.start_monitoring()
            
            # Wait for reconnection logic to execute
            await asyncio.sleep(0.1)
            
            # Assert
            mock_source_instance.reconnect.assert_called()
```

## Integration Testing

### Component Integration Tests

```python
# tests/integration/test_trading_workflow/test_complete_trade_flow.py
import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta

from src.core.kernel import AlgoSpaceKernel
from src.core.events import EventBus, Event, EventType
from tests.utils.factories import MarketDataFactory, ConfigurationFactory

@pytest.mark.integration
class TestCompleteTradingWorkflow:
    """Integration tests for complete trading workflows"""
    
    @pytest.fixture
    async def trading_system(self):
        """Set up complete trading system for integration testing"""
        # Create test configuration
        config = ConfigurationFactory.create_test_config(
            mode='integration_test',
            symbols=['AAPL', 'MSFT'],
            initial_capital=100000
        )
        
        # Initialize kernel with test configuration
        kernel = AlgoSpaceKernel(config)
        await kernel.initialize()
        
        yield kernel
        
        # Cleanup
        await kernel.shutdown()
    
    async def test_market_data_to_trade_execution(self, trading_system):
        """Test complete flow from market data to trade execution"""
        # Arrange
        market_data = MarketDataFactory.create_trending_market_data(
            symbol='AAPL',
            start_price=150.0,
            trend_direction='up',
            duration_minutes=60
        )
        
        trade_executed = asyncio.Event()
        executed_trades = []
        
        # Subscribe to trade execution events
        def on_trade_executed(event: Event):
            executed_trades.append(event.payload)
            trade_executed.set()
        
        trading_system.event_bus.subscribe(EventType.ORDER_FILLED, on_trade_executed)
        
        # Act
        # Start the trading system
        await trading_system.start()
        
        # Feed market data
        for tick in market_data:
            await trading_system.process_tick(tick)
            await asyncio.sleep(0.001)  # Small delay to allow processing
        
        # Wait for trade execution (with timeout)
        try:
            await asyncio.wait_for(trade_executed.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pytest.fail("No trade executed within timeout period")
        
        # Assert
        assert len(executed_trades) > 0, "No trades were executed"
        
        trade = executed_trades[0]
        assert trade['symbol'] == 'AAPL'
        assert trade['quantity'] > 0
        assert trade['execution_price'] > 0
    
    async def test_risk_management_integration(self, trading_system):
        """Test risk management integration across components"""
        # Arrange
        # Configure aggressive risk limits
        risk_config = {
            'max_position_size': 0.01,  # 1% of portfolio
            'max_daily_loss': 0.02,     # 2% daily loss limit
            'var_limit': 0.015          # 1.5% VaR limit
        }
        
        trading_system.update_risk_config(risk_config)
        
        # Create market data that would trigger large position
        volatile_market_data = MarketDataFactory.create_volatile_market_data(
            symbol='AAPL',
            volatility=0.05,  # 5% volatility
            duration_minutes=30
        )
        
        risk_alerts = []
        
        def on_risk_alert(event: Event):
            risk_alerts.append(event.payload)
        
        trading_system.event_bus.subscribe(EventType.RISK_BREACH, on_risk_alert)
        
        # Act
        await trading_system.start()
        
        for tick in volatile_market_data:
            await trading_system.process_tick(tick)
            await asyncio.sleep(0.001)
        
        await asyncio.sleep(1.0)  # Allow risk calculations to complete
        
        # Assert
        # Verify risk management is working
        portfolio = trading_system.get_component('portfolio_manager')
        total_exposure = portfolio.get_total_exposure()
        
        # Should not exceed risk limits
        assert total_exposure <= risk_config['max_position_size'] * portfolio.total_value
        
        # Risk alerts should have been generated if limits approached
        if total_exposure > 0.8 * risk_config['max_position_size'] * portfolio.total_value:
            assert len(risk_alerts) > 0, "Risk alerts should be generated near limits"
    
    async def test_marl_decision_integration(self, trading_system):
        """Test MARL agent decision integration"""
        # Arrange
        strategic_marl = trading_system.get_component('strategic_marl')
        assert strategic_marl is not None, "Strategic MARL component not available"
        
        # Create market conditions that should trigger MARL decision
        trending_data = MarketDataFactory.create_strong_trend_data(
            symbol='AAPL',
            trend_strength=0.8,
            duration_minutes=45
        )
        
        marl_decisions = []
        
        def on_marl_decision(event: Event):
            marl_decisions.append(event.payload)
        
        trading_system.event_bus.subscribe(EventType.STRATEGIC_DECISION, on_marl_decision)
        
        # Act
        await trading_system.start()
        
        for tick in trending_data:
            await trading_system.process_tick(tick)
            await asyncio.sleep(0.001)
        
        await asyncio.sleep(2.0)  # Allow MARL processing
        
        # Assert
        assert len(marl_decisions) > 0, "MARL agents should make decisions"
        
        decision = marl_decisions[0]
        assert decision['action'] in ['long', 'short', 'hold']
        assert 0 <= decision['confidence'] <= 1
        assert 'agents' in decision
        
        # Verify agent coordination
        agent_decisions = decision['agents']
        assert 'strategic_agent' in agent_decisions
        assert 'tactical_agent' in agent_decisions
        assert 'risk_agent' in agent_decisions
    
    async def test_error_recovery_integration(self, trading_system):
        """Test system error recovery and resilience"""
        # Arrange
        error_count = 0
        recovery_count = 0
        
        def on_system_error(event: Event):
            nonlocal error_count
            error_count += 1
        
        def on_component_recovery(event: Event):
            nonlocal recovery_count
            recovery_count += 1
        
        trading_system.event_bus.subscribe(EventType.SYSTEM_ERROR, on_system_error)
        trading_system.event_bus.subscribe(EventType.COMPONENT_STARTED, on_component_recovery)
        
        # Act
        await trading_system.start()
        
        # Simulate component failure
        data_handler = trading_system.get_component('data_handler')
        await data_handler.simulate_connection_failure()
        
        # Wait for error detection and recovery
        await asyncio.sleep(5.0)
        
        # Assert
        assert error_count > 0, "Error should be detected"
        assert recovery_count > 0, "Component should recover"
        
        # Verify system is still functional
        system_health = trading_system.get_system_health()
        assert system_health['overall_status'] in ['healthy', 'degraded']
```

## Performance Testing

### Latency Benchmarks

```python
# tests/performance/test_latency/test_critical_path_latency.py
import pytest
import time
import asyncio
import statistics
from typing import List

from src.core.events import Event, EventType, TickData
from tests.utils.factories import TickDataFactory

@pytest.mark.performance
class TestCriticalPathLatency:
    """Test latency of critical trading system paths"""
    
    @pytest.fixture
    def tick_processor(self):
        """Create tick processor for latency testing"""
        from src.data.processors.tick_processor import TickProcessor
        config = {'batch_size': 1, 'enable_caching': False}  # Disable optimizations for pure latency test
        return TickProcessor(config)
    
    def test_tick_processing_latency(self, tick_processor):
        """Test single tick processing latency"""
        # Arrange
        test_ticks = TickDataFactory.create_realistic_ticks(count=1000)
        latencies = []
        
        # Act
        for tick in test_ticks:
            start_time = time.perf_counter()
            tick_processor.process(tick)
            end_time = time.perf_counter()
            
            latency_us = (end_time - start_time) * 1_000_000  # Convert to microseconds
            latencies.append(latency_us)
        
        # Assert
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        # Performance requirements (in microseconds)
        assert avg_latency < 100, f"Average latency too high: {avg_latency:.1f}μs"
        assert p95_latency < 500, f"95th percentile latency too high: {p95_latency:.1f}μs"
        assert p99_latency < 1000, f"99th percentile latency too high: {p99_latency:.1f}μs"
        
        # Log performance metrics
        pytest.performance_metrics = {
            'tick_processing_avg_latency_us': avg_latency,
            'tick_processing_p95_latency_us': p95_latency,
            'tick_processing_p99_latency_us': p99_latency
        }
    
    @pytest.mark.asyncio
    async def test_marl_decision_latency(self):
        """Test MARL decision making latency"""
        from src.intelligence.marl_component import MARLComponent
        
        # Arrange
        config = ConfigurationFactory.create_performance_config()
        marl_component = MARLComponent(config)
        await marl_component.initialize()
        
        # Create test market states
        market_states = [
            MarketStateFactory.create_trending_state(),
            MarketStateFactory.create_volatile_state(),
            MarketStateFactory.create_sideways_state()
        ]
        
        latencies = []
        
        # Act
        for market_state in market_states:
            for _ in range(100):  # Multiple iterations for statistical significance
                start_time = time.perf_counter()
                decision = await marl_component.make_decision(market_state)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
                latencies.append(latency_ms)
        
        # Assert
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 10, f"MARL decision latency too high: {avg_latency:.1f}ms"
        assert max_latency < 50, f"Maximum MARL latency too high: {max_latency:.1f}ms"
    
    def test_risk_calculation_latency(self):
        """Test risk calculation latency"""
        from src.risk.calculators.var_calculator import VaRCalculator
        
        # Arrange
        var_calculator = VaRCalculator({'confidence_level': 0.95})
        
        # Generate test portfolio data
        returns_data = [
            np.random.normal(0.001, 0.02, 252)  # 252 days of returns
            for _ in range(100)  # 100 different portfolios
        ]
        
        latencies = []
        
        # Act
        for returns in returns_data:
            start_time = time.perf_counter()
            var_result = var_calculator.calculate(returns)
            end_time = time.perf_counter()
            
            latency_us = (end_time - start_time) * 1_000_000
            latencies.append(latency_us)
        
        # Assert
        avg_latency = statistics.mean(latencies)
        assert avg_latency < 5000, f"VaR calculation too slow: {avg_latency:.1f}μs"

### Throughput Benchmarks

```python
# tests/performance/test_throughput/test_high_frequency_processing.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

@pytest.mark.performance
class TestHighFrequencyThroughput:
    """Test system throughput under high-frequency loads"""
    
    @pytest.mark.asyncio
    async def test_tick_ingestion_throughput(self):
        """Test maximum tick ingestion throughput"""
        from src.data.handlers.market_data_handler import MarketDataHandler
        
        # Arrange
        config = {
            'batch_size': 1000,
            'worker_threads': 4,
            'queue_size': 100000
        }
        
        data_handler = MarketDataHandler(config)
        
        # Generate high-frequency test data
        tick_count = 100000
        test_ticks = TickDataFactory.create_high_frequency_ticks(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            count=tick_count,
            frequency_hz=1000  # 1000 ticks per second per symbol
        )
        
        processed_count = 0
        
        def tick_processed_callback():
            nonlocal processed_count
            processed_count += 1
        
        data_handler.set_tick_processed_callback(tick_processed_callback)
        
        # Act
        start_time = time.time()
        
        # Send all ticks
        for tick in test_ticks:
            await data_handler.process_tick(tick)
        
        # Wait for all processing to complete
        await data_handler.wait_for_queue_empty()
        
        end_time = time.time()
        
        # Assert
        processing_time = end_time - start_time
        throughput = processed_count / processing_time
        
        assert throughput > 50000, f"Tick throughput too low: {throughput:.0f} ticks/sec"
        assert processed_count == tick_count, f"Lost ticks: {tick_count - processed_count}"
    
    def test_concurrent_order_processing(self):
        """Test concurrent order processing throughput"""
        from src.execution.order_manager import OrderManager
        
        # Arrange
        config = {'max_concurrent_orders': 1000}
        order_manager = OrderManager(config)
        
        # Generate test orders
        orders = [
            OrderFactory.create_market_order(symbol=f'STOCK_{i % 100}', quantity=100)
            for i in range(10000)
        ]
        
        # Act
        start_time = time.time()
        
        # Process orders concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(order_manager.process_order, order)
                for order in orders
            ]
            
            # Wait for all orders to complete
            completed_orders = [future.result() for future in futures]
        
        end_time = time.time()
        
        # Assert
        processing_time = end_time - start_time
        orders_per_second = len(completed_orders) / processing_time
        
        assert orders_per_second > 5000, f"Order processing too slow: {orders_per_second:.0f} orders/sec"
        assert len(completed_orders) == len(orders), "Some orders were not processed"
    
    @pytest.mark.asyncio
    async def test_system_under_load(self):
        """Test complete system performance under realistic load"""
        # Arrange
        config = ConfigurationFactory.create_high_performance_config()
        trading_system = AlgoSpaceKernel(config)
        await trading_system.initialize()
        await trading_system.start()
        
        # Generate realistic market load
        duration_seconds = 30
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        ticks_per_second_per_symbol = 100
        
        total_expected_ticks = duration_seconds * len(symbols) * ticks_per_second_per_symbol
        
        # Metrics collection
        metrics = {
            'ticks_processed': 0,
            'orders_executed': 0,
            'decisions_made': 0,
            'errors_occurred': 0
        }
        
        # Set up metric collection
        self._setup_metric_collection(trading_system, metrics)
        
        # Act
        start_time = time.time()
        
        # Generate and send market data
        async def generate_market_data():
            for second in range(duration_seconds):
                for symbol in symbols:
                    for _ in range(ticks_per_second_per_symbol):
                        tick = TickDataFactory.create_realistic_tick(symbol)
                        await trading_system.process_tick(tick)
                        await asyncio.sleep(0.00001)  # 10μs delay between ticks
        
        await generate_market_data()
        
        # Allow processing to complete
        await asyncio.sleep(5.0)
        
        end_time = time.time()
        
        # Assert
        actual_duration = end_time - start_time
        
        # Throughput assertions
        tick_throughput = metrics['ticks_processed'] / actual_duration
        assert tick_throughput > 40000, f"System tick throughput too low: {tick_throughput:.0f} ticks/sec"
        
        # System health assertions
        error_rate = metrics['errors_occurred'] / metrics['ticks_processed'] if metrics['ticks_processed'] > 0 else 1
        assert error_rate < 0.001, f"Error rate too high: {error_rate:.4f}"
        
        # Decision making assertions
        if metrics['decisions_made'] > 0:
            decision_rate = metrics['decisions_made'] / (actual_duration / 60)  # decisions per minute
            assert decision_rate > 0, "System should make some trading decisions under load"
        
        await trading_system.shutdown()
```

## Financial Accuracy Testing

### Mathematical Precision Tests

```python
# tests/financial/test_calculations/test_financial_precision.py
import pytest
from decimal import Decimal, getcontext
import numpy as np
import pandas as pd

# Set high precision for financial calculations
getcontext().prec = 28

@pytest.mark.financial
class TestFinancialCalculationPrecision:
    """Test precision of financial calculations"""
    
    def test_pnl_calculation_precision(self):
        """Test P&L calculation precision with large numbers"""
        # Arrange
        position_size = Decimal('1000000')  # $1M position
        entry_price = Decimal('150.123456789')
        exit_price = Decimal('151.987654321')
        
        # Act
        pnl = (exit_price - entry_price) * position_size
        
        # Assert - verify precision to 8 decimal places
        expected_pnl = Decimal('1864197.532000000')
        assert pnl == expected_pnl
        
        # Verify no floating-point precision loss
        float_pnl = float(exit_price - entry_price) * float(position_size)
        decimal_pnl = float(pnl)
        
        # Decimal should be more precise than float
        assert abs(decimal_pnl - float_pnl) < 0.01  # Allow small difference due to float imprecision
    
    def test_commission_calculation_accuracy(self):
        """Test commission calculation accuracy"""
        # Arrange
        trade_value = Decimal('50000.00')
        commission_rate = Decimal('0.005')  # 0.5%
        
        # Act
        commission = trade_value * commission_rate
        
        # Assert
        expected_commission = Decimal('250.00')
        assert commission == expected_commission
        
        # Test with fractional shares
        fractional_shares = Decimal('100.333')
        share_price = Decimal('150.25')
        total_value = fractional_shares * share_price
        commission = total_value * commission_rate
        
        # Should maintain precision
        assert commission.as_tuple().exponent >= -2  # At least 2 decimal places
    
    def test_percentage_return_calculation(self):
        """Test percentage return calculation accuracy"""
        # Arrange
        initial_value = Decimal('100000.00')
        final_value = Decimal('105250.75')
        
        # Act
        absolute_return = final_value - initial_value
        percentage_return = (absolute_return / initial_value) * 100
        
        # Assert
        expected_percentage = Decimal('5.25075')
        assert percentage_return == expected_percentage
        
        # Test with very small returns
        small_final_value = Decimal('100000.01')
        small_return = ((small_final_value - initial_value) / initial_value) * 100
        
        # Should maintain precision for small returns
        assert small_return == Decimal('0.00001')
    
    def test_compound_interest_calculation(self):
        """Test compound interest calculation accuracy"""
        # Arrange
        principal = Decimal('10000.00')
        annual_rate = Decimal('0.075')  # 7.5%
        compounding_periods = 252  # Daily compounding
        years = Decimal('1')
        
        # Act
        # A = P(1 + r/n)^(nt)
        rate_per_period = annual_rate / compounding_periods
        exponent = compounding_periods * years
        
        # Use logarithms for precise compound calculation
        import math
        compound_factor = Decimal(str(math.exp(float(rate_per_period) * float(exponent))))
        final_amount = principal * compound_factor
        
        # Assert
        # Should be close to theoretical continuous compounding: Pe^(rt)
        continuous_compound = principal * Decimal(str(math.exp(float(annual_rate))))
        
        # Difference should be minimal
        difference = abs(final_amount - continuous_compound)
        assert difference < Decimal('1.00'), f"Compound calculation error too large: {difference}"

### Portfolio Calculation Tests

```python
# tests/financial/test_pnl/test_portfolio_calculations.py
@pytest.mark.financial
class TestPortfolioCalculations:
    """Test portfolio-level financial calculations"""
    
    def test_portfolio_value_calculation(self):
        """Test total portfolio value calculation"""
        # Arrange
        positions = {
            'AAPL': {'quantity': 100, 'avg_price': Decimal('150.00'), 'current_price': Decimal('155.50')},
            'MSFT': {'quantity': 50, 'avg_price': Decimal('300.00'), 'current_price': Decimal('295.25')},
            'CASH': {'quantity': 1, 'avg_price': Decimal('25000.00'), 'current_price': Decimal('25000.00')}
        }
        
        # Act
        total_value = Decimal('0')
        unrealized_pnl = Decimal('0')
        
        for symbol, position in positions.items():
            current_value = position['quantity'] * position['current_price']
            cost_basis = position['quantity'] * position['avg_price']
            
            total_value += current_value
            if symbol != 'CASH':
                unrealized_pnl += current_value - cost_basis
        
        # Assert
        expected_total = Decimal('55762.50')  # 15550 + 14762.50 + 25000
        expected_unrealized = Decimal('312.50')  # 550 + (-237.50)
        
        assert total_value == expected_total
        assert unrealized_pnl == expected_unrealized
    
    def test_risk_metrics_calculation(self):
        """Test portfolio risk metrics calculation"""
        # Arrange - 30 days of returns
        returns = pd.Series([
            0.012, -0.008, 0.005, 0.003, -0.015, 0.020, -0.002,
            0.007, -0.012, 0.018, 0.001, -0.009, 0.013, -0.005,
            0.008, 0.002, -0.007, 0.015, -0.003, 0.011, 0.006,
            -0.004, 0.009, -0.001, 0.014, 0.003, -0.006, 0.010,
            -0.011, 0.004
        ])
        
        # Act
        # Sharpe ratio calculation
        mean_return = returns.mean()
        std_return = returns.std()
        risk_free_rate = 0.02 / 252  # 2% annual risk-free rate, daily
        
        excess_return = mean_return - risk_free_rate
        sharpe_ratio = excess_return / std_return * np.sqrt(252)  # Annualized
        
        # VaR calculation (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Assert
        assert isinstance(sharpe_ratio, (float, np.float64))
        assert sharpe_ratio > -3 and sharpe_ratio < 3  # Reasonable range
        
        assert var_95 < 0  # VaR should be negative (loss)
        assert var_95 > -0.1  # But not unreasonably large
        
        assert max_drawdown <= 0  # Drawdown should be negative or zero
        assert max_drawdown > -0.5  # But not extreme
```

## MARL Testing

### Agent Testing

```python
# tests/unit/test_intelligence/test_marl_agents.py
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from src.intelligence.agents.strategic_agent import StrategicAgent
from src.intelligence.environments.trading_env import TradingEnvironment

@pytest.mark.marl
class TestMARLAgents:
    """Test MARL agent functionality"""
    
    @pytest.fixture
    def agent_config(self):
        """Configuration for test agent"""
        return {
            'observation_space_size': 48 * 13,  # 48 bars, 13 features
            'action_space_size': 3,  # long, neutral, short
            'hidden_layers': [128, 64],
            'learning_rate': 0.001,
            'epsilon': 0.1
        }
    
    @pytest.fixture
    def strategic_agent(self, agent_config):
        """Create strategic agent for testing"""
        return StrategicAgent(agent_config)
    
    def test_agent_initialization(self, strategic_agent, agent_config):
        """Test agent initializes correctly"""
        # Assert
        assert strategic_agent.observation_space_size == agent_config['observation_space_size']
        assert strategic_agent.action_space_size == agent_config['action_space_size']
        assert hasattr(strategic_agent, 'policy_network')
        assert hasattr(strategic_agent, 'target_network')
    
    def test_action_selection_deterministic(self, strategic_agent):
        """Test deterministic action selection"""
        # Arrange
        observation = np.random.randn(48 * 13)
        
        # Act
        action1 = strategic_agent.select_action(observation, training=False)
        action2 = strategic_agent.select_action(observation, training=False)
        
        # Assert - should be deterministic in evaluation mode
        assert action1 == action2
        assert action1 in [0, 1, 2]  # Valid action space
    
    def test_action_selection_stochastic(self, strategic_agent):
        """Test stochastic action selection during training"""
        # Arrange
        observation = np.random.randn(48 * 13)
        
        # Act - multiple actions in training mode
        actions = [
            strategic_agent.select_action(observation, training=True)
            for _ in range(100)
        ]
        
        # Assert - should have some exploration
        unique_actions = set(actions)
        assert len(unique_actions) > 1, "Agent should explore during training"
        assert all(action in [0, 1, 2] for action in actions)
    
    def test_experience_storage(self, strategic_agent):
        """Test experience storage for training"""
        # Arrange
        state = np.random.randn(48 * 13)
        action = 1
        reward = 0.5
        next_state = np.random.randn(48 * 13)
        done = False
        
        # Act
        strategic_agent.store_experience(state, action, reward, next_state, done)
        
        # Assert
        assert len(strategic_agent.experience_buffer) == 1
        experience = strategic_agent.experience_buffer[0]
        assert np.array_equal(experience['state'], state)
        assert experience['action'] == action
        assert experience['reward'] == reward
        assert experience['done'] == done
    
    def test_model_update(self, strategic_agent):
        """Test model update process"""
        # Arrange - fill experience buffer
        for _ in range(100):  # Minimum batch size
            state = np.random.randn(48 * 13)
            action = np.random.choice(3)
            reward = np.random.randn()
            next_state = np.random.randn(48 * 13)
            done = np.random.choice([True, False])
            
            strategic_agent.store_experience(state, action, reward, next_state, done)
        
        # Act
        loss = strategic_agent.update()
        
        # Assert
        assert isinstance(loss, (float, np.float32, np.float64))
        assert loss >= 0  # Loss should be non-negative
    
    def test_model_save_load(self, strategic_agent, tmp_path):
        """Test model persistence"""
        # Arrange
        model_path = tmp_path / "test_agent.pth"
        
        # Get initial parameters
        initial_state_dict = strategic_agent.policy_network.state_dict()
        
        # Act
        strategic_agent.save_model(str(model_path))
        
        # Create new agent and load model
        new_agent = StrategicAgent(strategic_agent.config)
        new_agent.load_model(str(model_path))
        
        # Assert
        loaded_state_dict = new_agent.policy_network.state_dict()
        
        for key in initial_state_dict:
            assert torch.allclose(initial_state_dict[key], loaded_state_dict[key])

### Environment Testing

```python
# tests/unit/test_intelligence/test_trading_environment.py
@pytest.mark.marl
class TestTradingEnvironment:
    """Test MARL trading environment"""
    
    @pytest.fixture
    def env_config(self):
        """Configuration for test environment"""
        return {
            'agents': ['strategic', 'tactical', 'risk'],
            'symbols': ['AAPL', 'MSFT'],
            'initial_capital': 100000,
            'episode_length': 1000,
            'transaction_costs': 0.001
        }
    
    @pytest.fixture
    def trading_env(self, env_config):
        """Create trading environment for testing"""
        return TradingEnvironment(env_config)
    
    def test_environment_initialization(self, trading_env, env_config):
        """Test environment initializes correctly"""
        # Assert
        assert trading_env.agents == env_config['agents']
        assert len(trading_env.observation_spaces) == len(env_config['agents'])
        assert len(trading_env.action_spaces) == len(env_config['agents'])
        
        for agent in env_config['agents']:
            assert agent in trading_env.observation_spaces
            assert agent in trading_env.action_spaces
    
    def test_environment_reset(self, trading_env):
        """Test environment reset functionality"""
        # Act
        observations = trading_env.reset()
        
        # Assert
        assert isinstance(observations, dict)
        assert len(observations) == len(trading_env.agents)
        
        for agent in trading_env.agents:
            assert agent in observations
            assert isinstance(observations[agent], np.ndarray)
            assert observations[agent].shape == trading_env.observation_spaces[agent].shape
    
    def test_environment_step(self, trading_env):
        """Test environment step functionality"""
        # Arrange
        observations = trading_env.reset()
        
        actions = {}
        for agent in trading_env.agents:
            actions[agent] = trading_env.action_spaces[agent].sample()
        
        # Act
        next_observations, rewards, dones, infos = trading_env.step(actions)
        
        # Assert
        assert isinstance(next_observations, dict)
        assert isinstance(rewards, dict)
        assert isinstance(dones, dict)
        assert isinstance(infos, dict)
        
        assert len(next_observations) == len(trading_env.agents)
        assert len(rewards) == len(trading_env.agents)
        assert len(dones) == len(trading_env.agents)
        
        for agent in trading_env.agents:
            assert agent in next_observations
            assert agent in rewards
            assert agent in dones
            
            assert isinstance(rewards[agent], (int, float, np.number))
            assert isinstance(dones[agent], bool)
    
    def test_reward_calculation(self, trading_env):
        """Test reward calculation logic"""
        # Arrange
        initial_portfolio_value = 100000
        trading_env.reset()
        
        # Simulate profitable action
        actions = {
            'strategic': 2,  # Long position
            'tactical': 1,   # Immediate entry
            'risk': 0        # Normal risk
        }
        
        # Mock market data that would make long position profitable
        with patch.object(trading_env, '_get_market_return', return_value=0.02):
            # Act
            _, rewards, _, _ = trading_env.step(actions)
        
        # Assert
        # Strategic agent should get positive reward for profitable trade
        assert rewards['strategic'] > 0
        
        # All agents should receive rewards
        for agent in trading_env.agents:
            assert isinstance(rewards[agent], (int, float, np.number))
    
    def test_observation_space_consistency(self, trading_env):
        """Test observation space consistency"""
        # Arrange
        observations = trading_env.reset()
        
        # Act - take multiple steps
        for _ in range(10):
            actions = {agent: trading_env.action_spaces[agent].sample() 
                      for agent in trading_env.agents}
            observations, _, _, _ = trading_env.step(actions)
        
        # Assert - observations should always match space definition
        for agent in trading_env.agents:
            expected_shape = trading_env.observation_spaces[agent].shape
            actual_shape = observations[agent].shape
            assert actual_shape == expected_shape, f"Observation shape mismatch for {agent}"
```

This comprehensive testing guide ensures the GrandModel trading system maintains high quality, performance, and reliability standards required for production financial applications.

## Related Documentation

- [Coding Standards](coding_standards.md)
- [Component Development Guide](component_guide.md)
- [Performance Optimization](../guides/performance_guide.md)
- [API Documentation](../api/)
- [Architecture Overview](../architecture/system_overview.md)