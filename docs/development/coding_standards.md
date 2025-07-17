# Coding Standards and Guidelines

## Overview

This document establishes coding standards and best practices for the GrandModel project. These standards ensure code consistency, maintainability, and reliability across the entire codebase while optimizing for the high-performance requirements of algorithmic trading systems.

## Table of Contents

- [General Principles](#general-principles)
- [Python Standards](#python-standards)
- [Code Organization](#code-organization)
- [Naming Conventions](#naming-conventions)
- [Documentation Standards](#documentation-standards)
- [Error Handling](#error-handling)
- [Performance Guidelines](#performance-guidelines)
- [Security Standards](#security-standards)
- [Testing Standards](#testing-standards)
- [Code Review Process](#code-review-process)

## General Principles

### Core Development Principles

1. **Clarity over Cleverness**: Write code that is easy to understand and maintain
2. **Performance-Conscious**: Optimize for low latency and high throughput
3. **Defensive Programming**: Assume inputs can be invalid and handle gracefully
4. **Fail Fast**: Detect and report errors as early as possible
5. **Immutability Preferred**: Use immutable data structures when possible
6. **Single Responsibility**: Each function/class should have one clear purpose
7. **DRY (Don't Repeat Yourself)**: Eliminate code duplication through abstraction
8. **YAGNI (You Aren't Gonna Need It)**: Don't implement features until needed

### Trading System Specific Principles

1. **Deterministic Behavior**: Code must produce consistent results
2. **Real-time Performance**: Sub-millisecond latency for critical paths
3. **Risk Awareness**: All trading logic must consider risk implications
4. **Audit Trail**: All actions must be traceable and logged
5. **Graceful Degradation**: System must handle partial failures
6. **Data Integrity**: Financial data must be accurate and validated

## Python Standards

### Python Version and Compatibility

- **Target Version**: Python 3.12+
- **Type Hints**: Mandatory for all public APIs
- **Async/Await**: Use for I/O-bound operations
- **Dataclasses**: Preferred for data structures

### Code Formatting

#### Automated Formatting Tools

```bash
# Install formatting tools
pip install black isort flake8 mypy

# Format code
black src/ --line-length 88
isort src/ --profile black
flake8 src/ --max-line-length 88 --ignore E203,W503
mypy src/ --ignore-missing-imports
```

#### Line Length and Wrapping

```python
# Good: Clear and readable
user_data = fetch_user_data(
    user_id=user_id,
    include_preferences=True,
    include_history=False
)

# Bad: Too long
user_data = fetch_user_data(user_id=user_id, include_preferences=True, include_history=False, timeout=30)

# Good: Function definition with many parameters
def calculate_portfolio_metrics(
    positions: Dict[str, Position],
    market_data: MarketData,
    risk_config: RiskConfig,
    calculation_date: datetime,
    include_unrealized: bool = True
) -> PortfolioMetrics:
    pass
```

#### Import Organization

```python
# Standard library imports first
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Third-party imports second
import numpy as np
import pandas as pd
import torch
import pettingzoo
from pydantic import BaseModel

# Local imports last
from src.core.events import Event, EventType
from src.core.kernel import AlgoSpaceKernel
from src.risk.kelly_calculator import KellyCalculator

# Avoid wildcard imports
# Bad: from module import *
# Good: from module import specific_function, SpecificClass
```

### Type Hints and Annotations

#### Comprehensive Type Hints

```python
from typing import Dict, List, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

# Type variables for generic classes
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Enum for better type safety
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

# Dataclass with type hints
@dataclass
class TradeOrder:
    symbol: str
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = datetime.now()

# Function with comprehensive type hints
def calculate_position_risk(
    position: TradeOrder,
    market_data: Dict[str, float],
    risk_params: Dict[str, Any],
    callback: Optional[Callable[[float], None]] = None
) -> Dict[str, Union[float, bool]]:
    """
    Calculate risk metrics for a trading position.
    
    Args:
        position: The trading position to analyze
        market_data: Current market prices and volatility
        risk_params: Risk calculation parameters
        callback: Optional callback for progress updates
        
    Returns:
        Dictionary containing risk metrics including VaR, exposure, etc.
        
    Raises:
        ValueError: If position data is invalid
        CalculationError: If risk calculation fails
    """
    pass

# Generic class with type parameters
class CircularBuffer(Generic[T]):
    """Type-safe circular buffer implementation"""
    
    def __init__(self, size: int) -> None:
        self._size = size
        self._buffer: List[Optional[T]] = [None] * size
        self._index = 0
    
    def add(self, item: T) -> None:
        self._buffer[self._index] = item
        self._index = (self._index + 1) % self._size
    
    def get_latest(self, count: int) -> List[T]:
        result: List[T] = []
        for i in range(min(count, self._size)):
            idx = (self._index - 1 - i) % self._size
            if self._buffer[idx] is not None:
                result.append(self._buffer[idx])
        return result
```

#### Protocol-Based Type Hints

```python
from typing import Protocol

class TradingStrategy(Protocol):
    """Protocol defining trading strategy interface"""
    
    def generate_signal(self, market_data: MarketData) -> Optional[TradeSignal]:
        """Generate trading signal from market data"""
        ...
    
    def calculate_position_size(self, signal: TradeSignal, portfolio: Portfolio) -> float:
        """Calculate appropriate position size"""
        ...
    
    def validate_trade(self, trade: TradeOrder) -> bool:
        """Validate trade before execution"""
        ...

# Function accepting any strategy implementation
def execute_strategy(strategy: TradingStrategy, data: MarketData) -> None:
    signal = strategy.generate_signal(data)
    if signal:
        # Execute trading logic
        pass
```

## Code Organization

### Project Structure

```
src/
├── __init__.py
├── main.py                 # Application entry point
├── core/                   # Core system components
│   ├── __init__.py
│   ├── kernel.py          # System orchestrator
│   ├── events.py          # Event system
│   ├── component_base.py  # Base component class
│   └── config.py          # Configuration management
├── data/                   # Data pipeline components
│   ├── __init__.py
│   ├── handlers/          # Data source handlers
│   ├── processors/        # Data processing
│   └── validators/        # Data validation
├── analysis/              # Market analysis components
│   ├── __init__.py
│   ├── indicators/        # Technical indicators
│   ├── matrices/          # Feature matrices
│   └── patterns/          # Pattern detection
├── intelligence/          # MARL and AI components
│   ├── __init__.py
│   ├── agents/           # MARL agents
│   ├── environments/     # Trading environments
│   └── models/           # ML models
├── execution/            # Order execution
│   ├── __init__.py
│   ├── orders/           # Order management
│   ├── routing/          # Order routing
│   └── brokers/          # Broker interfaces
├── risk/                 # Risk management
│   ├── __init__.py
│   ├── calculators/      # Risk calculations
│   ├── monitors/         # Risk monitoring
│   └── limits/           # Risk limits
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── logging.py        # Logging utilities
│   ├── metrics.py        # Performance metrics
│   └── helpers.py        # General helpers
└── testing/              # Test utilities
    ├── __init__.py
    ├── fixtures/         # Test fixtures
    ├── mocks/           # Mock objects
    └── factories/       # Data factories
```

### Module Organization

```python
# Good module structure
# src/risk/calculators/var_calculator.py

"""
Value at Risk (VaR) calculation module.

This module provides various VaR calculation methods including
parametric, historical simulation, and Monte Carlo approaches.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from src.core.events import Event, EventType
from src.utils.metrics import PerformanceTracker

# Module-level logger
logger = logging.getLogger(__name__)

# Public API - what other modules should import
__all__ = [
    'VaRCalculator',
    'ParametricVaR',
    'HistoricalVaR',
    'MonteCarloVaR',
    'VaRResult',
    'VaRConfig'
]

# Constants
DEFAULT_CONFIDENCE_LEVEL = 0.95
MIN_HISTORICAL_OBSERVATIONS = 252
MAX_MONTE_CARLO_SIMULATIONS = 100000

# Type definitions
@dataclass(frozen=True)
class VaRConfig:
    """Configuration for VaR calculations"""
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    time_horizon_days: int = 1
    method: str = 'parametric'
    historical_window: int = 252

@dataclass(frozen=True)
class VaRResult:
    """Result of VaR calculation"""
    var_amount: float
    confidence_level: float
    method: str
    calculation_time: datetime
    metadata: Dict[str, Any]

# Abstract base class
class VaRCalculator(ABC):
    """Abstract base class for VaR calculators"""
    
    def __init__(self, config: VaRConfig):
        self.config = config
        self.performance_tracker = PerformanceTracker()
    
    @abstractmethod
    def calculate(self, returns: pd.Series) -> VaRResult:
        """Calculate VaR for given return series"""
        pass

# Concrete implementations
class ParametricVaR(VaRCalculator):
    """Parametric VaR calculation using normal distribution assumption"""
    
    def calculate(self, returns: pd.Series) -> VaRResult:
        start_time = time.time()
        
        try:
            # Implementation details...
            var_amount = self._calculate_parametric_var(returns)
            
            return VaRResult(
                var_amount=var_amount,
                confidence_level=self.config.confidence_level,
                method='parametric',
                calculation_time=datetime.now(),
                metadata={'sample_size': len(returns)}
            )
        finally:
            self.performance_tracker.record_calculation_time(
                'parametric_var', 
                time.time() - start_time
            )
```

## Naming Conventions

### Variables and Functions

```python
# Variables: snake_case
user_account = get_user_account()
portfolio_value = calculate_portfolio_value()
market_data_feed = MarketDataFeed()

# Functions: snake_case, descriptive verbs
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float:
    pass

def validate_order_parameters(order: TradeOrder) -> bool:
    pass

def process_market_data_update(event: Event) -> None:
    pass

# Private functions: leading underscore
def _internal_calculation(data: np.ndarray) -> float:
    pass

def _validate_input_data(data: Dict[str, Any]) -> bool:
    pass
```

### Classes and Types

```python
# Classes: PascalCase
class MarketDataHandler:
    pass

class PortfolioManager:
    pass

class MARLTradingAgent:
    pass

# Protocols: PascalCase with "Protocol" suffix
class TradingStrategyProtocol(Protocol):
    pass

# Enums: PascalCase
class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

# Type aliases: PascalCase
SymbolPriceMap = Dict[str, float]
PortfolioWeights = Dict[str, float]
RiskMetrics = Dict[str, Union[float, bool]]
```

### Constants and Configuration

```python
# Constants: UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 0.1
DEFAULT_TIMEOUT_SECONDS = 30
TRADING_HOURS_START = time(9, 30)
TRADING_HOURS_END = time(16, 0)

# Configuration sections
RISK_MANAGEMENT_CONFIG = {
    'max_daily_var': 0.02,
    'position_size_limit': 0.05,
    'correlation_threshold': 0.7
}

DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'grandmodel'
}
```

### File and Directory Names

```python
# Files: snake_case
market_data_handler.py
portfolio_manager.py
kelly_criterion_calculator.py

# Directories: snake_case
data_processing/
risk_management/
order_execution/
machine_learning/

# Test files: test_ prefix
test_market_data_handler.py
test_portfolio_manager.py
test_integration_full_system.py
```

## Documentation Standards

### Docstring Format

```python
def calculate_kelly_position_size(
    win_probability: float,
    win_loss_ratio: float,
    current_capital: float,
    max_risk_percentage: float = 0.25
) -> Dict[str, Union[float, bool]]:
    """
    Calculate optimal position size using Kelly Criterion.
    
    The Kelly Criterion determines the optimal fraction of capital to risk
    on a trade based on the probability of winning and the win/loss ratio.
    
    Args:
        win_probability: Probability of winning the trade (0.0 to 1.0)
        win_loss_ratio: Ratio of average win to average loss
        current_capital: Current available capital for trading
        max_risk_percentage: Maximum percentage of capital to risk (default: 0.25)
        
    Returns:
        Dictionary containing:
            - 'position_size': Recommended position size in currency units
            - 'kelly_fraction': Raw Kelly fraction (before risk adjustment)
            - 'adjusted_fraction': Risk-adjusted Kelly fraction
            - 'is_favorable': Whether the trade has positive expected value
            - 'expected_growth': Expected logarithmic growth rate
            
    Raises:
        ValueError: If win_probability is not between 0 and 1
        ValueError: If win_loss_ratio is not positive
        ValueError: If current_capital is not positive
        
    Example:
        >>> result = calculate_kelly_position_size(0.6, 1.5, 100000)
        >>> print(f"Position size: ${result['position_size']:,.2f}")
        Position size: $15,000.00
        
    Note:
        The Kelly Criterion assumes reinvestment of gains and can recommend
        aggressive position sizes. The implementation includes a maximum
        risk adjustment to prevent overexposure.
        
    References:
        Kelly, J. L. (1956). A new interpretation of information rate.
        Bell System Technical Journal, 35(4), 917-926.
    """
    # Validate inputs
    if not (0 <= win_probability <= 1):
        raise ValueError(f"Win probability must be between 0 and 1, got {win_probability}")
    
    if win_loss_ratio <= 0:
        raise ValueError(f"Win/loss ratio must be positive, got {win_loss_ratio}")
    
    if current_capital <= 0:
        raise ValueError(f"Current capital must be positive, got {current_capital}")
    
    # Calculate Kelly fraction
    kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
    
    # Determine if trade is favorable
    is_favorable = kelly_fraction > 0
    
    # Apply risk adjustment
    adjusted_fraction = min(kelly_fraction, max_risk_percentage) if is_favorable else 0
    
    # Calculate position size
    position_size = adjusted_fraction * current_capital
    
    # Calculate expected growth rate
    expected_growth = (
        win_probability * np.log(1 + adjusted_fraction * win_loss_ratio) +
        (1 - win_probability) * np.log(1 - adjusted_fraction)
    )
    
    return {
        'position_size': position_size,
        'kelly_fraction': kelly_fraction,
        'adjusted_fraction': adjusted_fraction,
        'is_favorable': is_favorable,
        'expected_growth': expected_growth
    }
```

### Class Documentation

```python
class MARLTradingEnvironment:
    """
    Multi-Agent Reinforcement Learning trading environment.
    
    This environment simulates a trading scenario where multiple agents
    can interact with the market simultaneously. It provides market data,
    executes trades, and calculates rewards based on trading performance.
    
    The environment follows the PettingZoo API for multi-agent environments
    and can be used with various MARL algorithms.
    
    Attributes:
        agents: List of agent names participating in the environment
        observation_spaces: Dictionary mapping agent names to observation spaces
        action_spaces: Dictionary mapping agent names to action spaces
        market_data: Current market data state
        portfolio_state: Current portfolio state for each agent
        
    Example:
        >>> env = MARLTradingEnvironment(config)
        >>> observations = env.reset()
        >>> actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        >>> observations, rewards, dones, infos = env.step(actions)
        
    Note:
        The environment maintains separate portfolio states for each agent
        but shares the same market data feed. Agents can observe each other's
        actions but cannot directly modify other agents' portfolios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MARL trading environment.
        
        Args:
            config: Configuration dictionary containing environment parameters
                - symbols: List of trading symbols
                - initial_capital: Starting capital for each agent
                - transaction_costs: Trading fees and slippage
                - market_data_source: Source of market data
        """
        pass
```

### Module Documentation

```python
"""
Risk Management Module

This module provides comprehensive risk management functionality for the
GrandModel trading system. It includes various risk calculation methods,
real-time monitoring, and automated risk controls.

Key Components:
    - VaR calculators for portfolio risk assessment
    - Kelly Criterion for position sizing
    - Correlation tracking for portfolio diversification
    - Real-time risk monitoring and alerts
    - Automated stop-loss and position limits

The module is designed for high-frequency trading environments and provides
sub-millisecond risk calculations for real-time decision making.

Example:
    >>> from src.risk import RiskManager, VaRCalculator
    >>> risk_manager = RiskManager(config)
    >>> var_calc = VaRCalculator(confidence_level=0.95)
    >>> portfolio_var = var_calc.calculate(portfolio_returns)

Threading Safety:
    All components in this module are thread-safe and can be used
    concurrently across multiple trading threads.

Performance Notes:
    - VaR calculations are cached for identical inputs
    - Historical data is stored in optimized NumPy arrays
    - Critical path operations are JIT-compiled with Numba
"""
```

## Error Handling

### Exception Hierarchy

```python
# Custom exception hierarchy
class GrandModelError(Exception):
    """Base exception for all GrandModel errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()

class DataError(GrandModelError):
    """Errors related to data processing and validation"""
    pass

class TradingError(GrandModelError):
    """Errors related to trading operations"""
    pass

class RiskError(GrandModelError):
    """Errors related to risk management"""
    pass

class ConfigurationError(GrandModelError):
    """Errors related to system configuration"""
    pass

class ConnectionError(GrandModelError):
    """Errors related to external connections"""
    pass

# Specific exceptions
class InvalidOrderError(TradingError):
    """Raised when order parameters are invalid"""
    pass

class RiskLimitExceededError(RiskError):
    """Raised when risk limits are exceeded"""
    pass

class MarketDataError(DataError):
    """Raised when market data is invalid or unavailable"""
    pass
```

### Error Handling Patterns

```python
# Comprehensive error handling with context
def process_trading_signal(signal: TradingSignal, portfolio: Portfolio) -> Optional[TradeOrder]:
    """Process trading signal with comprehensive error handling"""
    
    try:
        # Validate signal
        if not _validate_signal(signal):
            raise InvalidSignalError(
                "Signal validation failed",
                error_code="INVALID_SIGNAL",
                context={'signal': signal.to_dict()}
            )
        
        # Check risk limits
        risk_assessment = _assess_signal_risk(signal, portfolio)
        if not risk_assessment.approved:
            raise RiskLimitExceededError(
                f"Signal exceeds risk limits: {risk_assessment.reason}",
                error_code="RISK_LIMIT_EXCEEDED",
                context={
                    'signal': signal.to_dict(),
                    'risk_assessment': risk_assessment.to_dict()
                }
            )
        
        # Create order
        order = _create_order_from_signal(signal, risk_assessment.recommended_size)
        
        logger.info(f"Created order from signal: {order.id}")
        return order
        
    except (InvalidSignalError, RiskLimitExceededError):
        # Known business logic errors - re-raise
        raise
        
    except Exception as e:
        # Unexpected errors - wrap and add context
        logger.error(f"Unexpected error processing signal: {e}", exc_info=True)
        raise TradingError(
            f"Failed to process trading signal: {str(e)}",
            error_code="SIGNAL_PROCESSING_ERROR",
            context={
                'signal': signal.to_dict(),
                'portfolio_id': portfolio.id
            }
        ) from e

# Retry pattern with exponential backoff
import asyncio
from functools import wraps

def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying functions with exponential backoff"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

# Usage example
@retry_with_backoff(max_attempts=3, exceptions=(ConnectionError, TimeoutError))
async def fetch_market_data(symbol: str) -> MarketData:
    """Fetch market data with automatic retry"""
    # Implementation with potential connection issues
    pass
```

### Logging Standards

```python
import logging
import structlog
from datetime import datetime
from typing import Any, Dict

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Get logger for module
logger = structlog.get_logger(__name__)

# Logging best practices
class TradingLogger:
    """Centralized logging for trading operations"""
    
    def __init__(self, component_name: str):
        self.logger = structlog.get_logger(component_name)
        self.component_name = component_name
    
    def log_trade_execution(self, order: TradeOrder, result: ExecutionResult) -> None:
        """Log trade execution with full context"""
        self.logger.info(
            "Trade executed",
            event="trade_execution",
            order_id=order.id,
            symbol=order.symbol,
            quantity=order.quantity,
            execution_price=result.execution_price,
            execution_time=result.execution_time.isoformat(),
            slippage=result.slippage,
            commission=result.commission,
            portfolio_impact=result.portfolio_impact
        )
    
    def log_risk_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Log risk management alerts"""
        self.logger.warning(
            "Risk alert triggered",
            event="risk_alert",
            alert_type=alert_type,
            timestamp=datetime.now().isoformat(),
            **details
        )
    
    def log_performance_metric(self, metric_name: str, value: float, context: Dict[str, Any] = None) -> None:
        """Log performance metrics"""
        self.logger.info(
            "Performance metric",
            event="performance_metric",
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now().isoformat(),
            **(context or {})
        )
    
    def log_error_with_context(self, error: Exception, operation: str, context: Dict[str, Any] = None) -> None:
        """Log errors with comprehensive context"""
        self.logger.error(
            f"Error in {operation}",
            event="error",
            error_type=type(error).__name__,
            error_message=str(error),
            operation=operation,
            component=self.component_name,
            timestamp=datetime.now().isoformat(),
            **(context or {}),
            exc_info=True
        )
```

## Performance Guidelines

### Optimization Principles

```python
# Use appropriate data structures
from collections import deque, defaultdict
import numpy as np
import pandas as pd

# Good: Use deque for FIFO operations
price_buffer = deque(maxsize=1000)

# Good: Use NumPy for numerical computations
returns = np.array(price_changes) / np.array(prices[:-1])
volatility = np.std(returns) * np.sqrt(252)

# Good: Use pandas for time series operations
price_series = pd.Series(prices, index=timestamps)
rolling_mean = price_series.rolling(window=20).mean()

# Performance-critical code optimization
from numba import jit, njit
import asyncio

@njit  # No-Python mode for maximum performance
def fast_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
    """Fast moving average calculation using Numba JIT"""
    n = len(prices)
    result = np.empty(n)
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        result[i] = np.mean(prices[start_idx:i+1])
    
    return result

# Async operations for I/O bound tasks
async def parallel_data_fetch(symbols: List[str]) -> Dict[str, MarketData]:
    """Fetch market data for multiple symbols in parallel"""
    
    async def fetch_symbol_data(symbol: str) -> Tuple[str, MarketData]:
        data = await market_data_client.fetch(symbol)
        return symbol, data
    
    # Create tasks for parallel execution
    tasks = [fetch_symbol_data(symbol) for symbol in symbols]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    return dict(results)

# Memory-efficient processing
def process_large_dataset_in_chunks(
    data_source: Iterator[DataPoint], 
    chunk_size: int = 10000
) -> Generator[ProcessedData, None, None]:
    """Process large datasets in memory-efficient chunks"""
    
    chunk = []
    for data_point in data_source:
        chunk.append(data_point)
        
        if len(chunk) >= chunk_size:
            # Process chunk
            processed_chunk = _process_chunk(chunk)
            yield processed_chunk
            
            # Clear chunk to free memory
            chunk.clear()
    
    # Process remaining data
    if chunk:
        yield _process_chunk(chunk)
```

### Caching Strategies

```python
from functools import lru_cache, wraps
import time
from typing import Dict, Any, Callable

# Simple LRU cache for expensive calculations
@lru_cache(maxsize=1000)
def calculate_indicator_value(prices_hash: int, indicator_type: str, params: tuple) -> float:
    """Cached indicator calculation"""
    # Expensive calculation here
    pass

# Time-based cache with TTL
class TTLCache:
    """Time-To-Live cache implementation"""
    
    def __init__(self, default_ttl: float = 300):  # 5 minutes default
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Any:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() < entry['expires_at']:
                return entry['value']
            else:
                del self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: float = None) -> None:
        ttl = ttl or self.default_ttl
        self._cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }

# Decorator for caching function results
def cache_with_ttl(ttl: float = 300):
    """Decorator for caching function results with TTL"""
    
    def decorator(func: Callable) -> Callable:
        cache = TTLCache(ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Calculate and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            
            return result
        
        return wrapper
    return decorator

# Usage example
@cache_with_ttl(ttl=60)  # Cache for 1 minute
def expensive_market_calculation(symbol: str, lookback_days: int) -> float:
    """Expensive calculation that benefits from caching"""
    # Complex calculation here
    pass
```

## Security Standards

### Input Validation

```python
from typing import Union
import re
from decimal import Decimal, InvalidOperation

def validate_symbol(symbol: str) -> str:
    """Validate and normalize trading symbol"""
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string")
    
    # Remove whitespace and convert to uppercase
    symbol = symbol.strip().upper()
    
    # Validate format (alphanumeric, 1-10 characters)
    if not re.match(r'^[A-Z0-9]{1,10}$', symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")
    
    return symbol

def validate_price(price: Union[str, float, Decimal]) -> Decimal:
    """Validate and normalize price value"""
    try:
        # Convert to Decimal for precise financial calculations
        price_decimal = Decimal(str(price))
        
        # Check for reasonable price range
        if price_decimal <= 0:
            raise ValueError("Price must be positive")
        
        if price_decimal > Decimal('1000000'):
            raise ValueError("Price exceeds maximum allowed value")
        
        # Round to appropriate precision (4 decimal places for most assets)
        return price_decimal.quantize(Decimal('0.0001'))
        
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid price value: {price}") from e

def validate_order_parameters(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate order parameters with comprehensive checks"""
    
    # Required fields
    required_fields = ['symbol', 'quantity', 'order_type']
    for field in required_fields:
        if field not in order_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate and normalize each field
    validated_order = {
        'symbol': validate_symbol(order_data['symbol']),
        'quantity': validate_quantity(order_data['quantity']),
        'order_type': validate_order_type(order_data['order_type'])
    }
    
    # Optional fields with validation
    if 'price' in order_data:
        validated_order['price'] = validate_price(order_data['price'])
    
    if 'stop_price' in order_data:
        validated_order['stop_price'] = validate_price(order_data['stop_price'])
    
    return validated_order
```

### Secure Configuration Management

```python
import os
from pathlib import Path
from typing import Dict, Any, Optional
import keyring
from cryptography.fernet import Fernet

class SecureConfigManager:
    """Secure configuration management with encryption"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key from secure storage"""
        try:
            # Try to get key from system keyring
            key = keyring.get_password("grandmodel", "config_encryption_key")
            if key:
                return key.encode()
        except Exception:
            pass
        
        # Generate new key
        key = Fernet.generate_key()
        
        try:
            # Store in system keyring
            keyring.set_password("grandmodel", "config_encryption_key", key.decode())
        except Exception:
            # Fallback to environment variable (less secure)
            os.environ["GRANDMODEL_CONFIG_KEY"] = key.decode()
        
        return key
    
    def encrypt_sensitive_value(self, value: str) -> str:
        """Encrypt sensitive configuration value"""
        return self._cipher.encrypt(value.encode()).decode()
    
    def decrypt_sensitive_value(self, encrypted_value: str) -> str:
        """Decrypt sensitive configuration value"""
        return self._cipher.decrypt(encrypted_value.encode()).decode()
    
    def load_config(self) -> Dict[str, Any]:
        """Load and decrypt configuration"""
        # Load configuration from file
        with open(self.config_path) as f:
            config = json.load(f)
        
        # Decrypt sensitive values
        self._decrypt_config_values(config)
        
        return config
    
    def _decrypt_config_values(self, config: Dict[str, Any]) -> None:
        """Recursively decrypt sensitive values in config"""
        for key, value in config.items():
            if isinstance(value, dict):
                self._decrypt_config_values(value)
            elif isinstance(value, str) and value.startswith("encrypted:"):
                encrypted_value = value[10:]  # Remove "encrypted:" prefix
                config[key] = self.decrypt_sensitive_value(encrypted_value)
```

## Testing Standards

### Test Organization

```python
# test_portfolio_manager.py
import unittest
from unittest.mock import Mock, MagicMock, patch
import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from src.execution.portfolio_manager import PortfolioManager
from src.core.events import Event, EventType
from src.testing.factories import TradeOrderFactory, PortfolioFactory

class TestPortfolioManager(unittest.TestCase):
    """Test suite for PortfolioManager component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_event_bus = MagicMock()
        self.config = {
            'initial_capital': 100000,
            'max_positions': 10,
            'margin_requirement': 0.5
        }
        self.portfolio_manager = PortfolioManager(self.config, self.mock_event_bus)
    
    def tearDown(self):
        """Clean up after tests"""
        self.portfolio_manager.stop()
    
    def test_initial_portfolio_state(self):
        """Test portfolio initialization"""
        self.assertEqual(self.portfolio_manager.cash_balance, 100000)
        self.assertEqual(len(self.portfolio_manager.positions), 0)
        self.assertEqual(self.portfolio_manager.total_value, 100000)
    
    def test_process_trade_execution(self):
        """Test processing of trade execution"""
        # Create test trade
        trade_event = TradeOrderFactory.create_execution_event(
            symbol='AAPL',
            quantity=100,
            price=Decimal('150.00')
        )
        
        # Process trade
        self.portfolio_manager.on_order_filled(trade_event)
        
        # Verify position created
        self.assertIn('AAPL', self.portfolio_manager.positions)
        position = self.portfolio_manager.positions['AAPL']
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.average_price, Decimal('150.00'))
    
    @patch('src.execution.portfolio_manager.MarketDataService')
    def test_unrealized_pnl_calculation(self, mock_market_data):
        """Test unrealized P&L calculation with mocked market data"""
        # Setup mock market data
        mock_market_data.get_current_price.return_value = Decimal('160.00')
        
        # Create position
        trade_event = TradeOrderFactory.create_execution_event(
            symbol='AAPL',
            quantity=100,
            price=Decimal('150.00')
        )
        self.portfolio_manager.on_order_filled(trade_event)
        
        # Calculate unrealized P&L
        unrealized_pnl = self.portfolio_manager._calculate_unrealized_pnl()
        
        # Verify calculation
        expected_pnl = (Decimal('160.00') - Decimal('150.00')) * 100
        self.assertEqual(unrealized_pnl, expected_pnl)
    
    def test_risk_limit_enforcement(self):
        """Test portfolio risk limit enforcement"""
        # Configure tight risk limits
        self.portfolio_manager.config['max_position_value'] = 5000
        
        # Attempt large trade
        large_trade = TradeOrderFactory.create_execution_event(
            symbol='TSLA',
            quantity=100,
            price=Decimal('800.00')  # $80,000 position
        )
        
        # Should raise risk limit error
        with self.assertRaises(RiskLimitExceededError):
            self.portfolio_manager.on_order_filled(large_trade)

# Integration test example
class TestPortfolioManagerIntegration(unittest.TestCase):
    """Integration tests for PortfolioManager"""
    
    @pytest.mark.integration
    def test_full_trading_workflow(self):
        """Test complete trading workflow integration"""
        # Setup real components (not mocked)
        event_bus = EventBus()
        portfolio_manager = PortfolioManager(self.config, event_bus)
        risk_manager = RiskManager(self.config, event_bus)
        
        # Initialize components
        portfolio_manager.initialize()
        risk_manager.initialize()
        
        # Execute test workflow
        self._execute_sample_trades(event_bus)
        
        # Verify end state
        self._verify_portfolio_consistency(portfolio_manager)
    
    def _execute_sample_trades(self, event_bus: EventBus):
        """Execute sample trading sequence"""
        # Implementation of sample trade sequence
        pass
    
    def _verify_portfolio_consistency(self, portfolio_manager: PortfolioManager):
        """Verify portfolio state consistency"""
        # Implementation of consistency checks
        pass

# Performance test example
class TestPortfolioManagerPerformance(unittest.TestCase):
    """Performance tests for PortfolioManager"""
    
    @pytest.mark.performance
    def test_high_frequency_trade_processing(self):
        """Test processing of high-frequency trades"""
        import time
        
        portfolio_manager = PortfolioManager(self.config, MagicMock())
        portfolio_manager.initialize()
        
        # Generate many trades
        trades = [
            TradeOrderFactory.create_execution_event(
                symbol=f'STOCK_{i}',
                quantity=100,
                price=Decimal('100.00')
            )
            for i in range(10000)
        ]
        
        # Time the processing
        start_time = time.time()
        
        for trade in trades:
            portfolio_manager.on_order_filled(trade)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify performance requirements
        trades_per_second = len(trades) / processing_time
        self.assertGreater(trades_per_second, 1000, 
                          f"Trade processing too slow: {trades_per_second:.1f} trades/sec")
```

## Code Review Process

### Review Checklist

#### Functionality Review
- [ ] Code meets requirements and specifications
- [ ] Edge cases are properly handled
- [ ] Error handling is comprehensive and appropriate
- [ ] Performance requirements are met
- [ ] Security considerations are addressed

#### Code Quality Review
- [ ] Code follows established coding standards
- [ ] Variable and function names are descriptive
- [ ] Code is well-organized and modular
- [ ] No code duplication (DRY principle)
- [ ] Comments explain why, not what

#### Testing Review
- [ ] Unit tests cover all public methods
- [ ] Edge cases are tested
- [ ] Integration tests verify component interactions
- [ ] Performance tests validate speed requirements
- [ ] All tests pass consistently

#### Documentation Review
- [ ] Public APIs are documented with comprehensive docstrings
- [ ] Complex algorithms are explained
- [ ] Configuration options are documented
- [ ] Examples are provided for key functionality

### Review Process

1. **Self-Review**: Developer reviews own code before submission
2. **Automated Checks**: CI/CD pipeline runs tests and quality checks
3. **Peer Review**: At least one other developer reviews the code
4. **Technical Lead Review**: Complex changes require lead developer approval
5. **Final Validation**: Automated testing in staging environment

### Review Tools and Commands

```bash
# Pre-commit hooks setup
pip install pre-commit
pre-commit install

# Run all quality checks
black src/ --check
isort src/ --check-only --profile black
flake8 src/
mypy src/
pytest tests/ --cov=src --cov-report=html

# Security scanning
bandit -r src/
safety check

# Performance profiling
python -m cProfile -o profile.stats src/main.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('time').print_stats(20)"
```

These coding standards ensure that the GrandModel codebase remains maintainable, performant, and secure while supporting the demanding requirements of algorithmic trading systems.

## Related Documentation

- [Testing Guidelines](testing_guidelines.md)
- [Component Development Guide](component_guide.md)
- [Performance Optimization](../guides/performance_guide.md)
- [Architecture Overview](../architecture/system_overview.md)
- [API Documentation](../api/)