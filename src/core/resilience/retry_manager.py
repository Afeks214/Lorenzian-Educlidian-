"""
Comprehensive Retry Manager with Exponential Backoff
===================================================

Advanced retry mechanism with multiple strategies, circuit breaker integration,
and intelligent failure handling.

Features:
- Multiple retry strategies (exponential backoff, linear, fibonacci)
- Jitter and circuit breaker integration
- Conditional retry based on exception types
- Comprehensive metrics and monitoring
- Deadlock prevention and timeout handling
"""

import asyncio
import random
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union, Type
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import inspect

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    CONSTANT_DELAY = "constant_delay"
    ADAPTIVE = "adaptive"


class RetryOutcome(Enum):
    """Retry attempt outcomes."""
    SUCCESS = "success"
    RETRY = "retry"
    FAILED = "failed"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    TIMEOUT = "timeout"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    # Basic retry parameters
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    
    # Jitter configuration
    jitter: bool = True
    jitter_max: float = 0.1
    
    # Timeout configuration
    timeout: Optional[float] = None
    per_attempt_timeout: Optional[float] = None
    
    # Strategy selection
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Exception handling
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    
    # Circuit breaker integration
    circuit_breaker_integration: bool = True
    respect_circuit_breaker: bool = True
    
    # Conditional retry
    retry_condition: Optional[Callable[[Exception, int], bool]] = None
    
    # Metrics
    track_metrics: bool = True
    
    # Adaptive parameters
    adaptive_multiplier_adjustment: bool = True
    adaptive_max_delay_adjustment: bool = True
    
    # Deadlock prevention
    deadlock_timeout: float = 300.0  # 5 minutes
    max_concurrent_retries: int = 100


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""
    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    circuit_breaker_rejections: int = 0
    timeout_failures: int = 0
    total_delay_time: float = 0.0
    average_attempts_per_operation: float = 0.0
    success_rate: float = 0.0
    
    # Strategy-specific metrics
    strategy_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Exception-specific metrics
    exception_counts: Dict[str, int] = field(default_factory=dict)
    
    # Timing metrics
    min_retry_delay: float = float('inf')
    max_retry_delay: float = 0.0
    average_retry_delay: float = 0.0


class RetryManager:
    """
    Advanced retry manager with multiple strategies and comprehensive monitoring.
    
    Features:
    - Multiple retry strategies with adaptive behavior
    - Circuit breaker integration for cascading failure prevention
    - Comprehensive metrics and monitoring
    - Deadlock prevention and timeout handling
    - Conditional retry based on exception types and custom conditions
    """
    
    def __init__(
        self,
        config: RetryConfig,
        circuit_breaker: Optional[Any] = None,
        name: str = "default"
    ):
        """Initialize retry manager."""
        self.config = config
        self.circuit_breaker = circuit_breaker
        self.name = name
        
        # Metrics
        self.metrics = RetryMetrics()
        
        # Adaptive parameters
        self.adaptive_multiplier = config.multiplier
        self.adaptive_max_delay = config.max_delay
        
        # Fibonacci sequence for fibonacci backoff
        self.fibonacci_sequence = [1, 1]
        
        # Semaphore for concurrent retry limiting
        self.retry_semaphore = asyncio.Semaphore(config.max_concurrent_retries)
        
        # Active retry tracking
        self.active_retries: Dict[str, float] = {}
        
        logger.info(f"Retry manager initialized: {name}")
    
    @asynccontextmanager
    async def retry(self, operation_name: str = "operation"):
        """
        Context manager for retry operations.
        
        Usage:
            async with retry_manager.retry("database_call"):
                result = await database_call()
        """
        async with self.retry_semaphore:
            retry_id = f"{operation_name}_{time.time()}_{random.randint(1000, 9999)}"
            self.active_retries[retry_id] = time.time()
            
            try:
                async with self._execute_with_retry(retry_id, operation_name):
                    yield
            finally:
                self.active_retries.pop(retry_id, None)
    
    @asynccontextmanager
    async def _execute_with_retry(self, retry_id: str, operation_name: str):
        """Execute operation with retry logic."""
        attempt = 0
        total_delay = 0.0
        start_time = time.time()
        last_exception = None
        
        while attempt < self.config.max_attempts:
            attempt += 1
            
            # Check circuit breaker
            if (self.circuit_breaker and 
                self.config.respect_circuit_breaker and
                not await self.circuit_breaker.can_execute()):
                
                self.metrics.circuit_breaker_rejections += 1
                logger.warning(f"Circuit breaker open for {operation_name}")
                raise CircuitBreakerOpenException(f"Circuit breaker open for {operation_name}")
            
            # Check deadlock timeout
            if time.time() - start_time > self.config.deadlock_timeout:
                self.metrics.timeout_failures += 1
                logger.error(f"Deadlock timeout exceeded for {operation_name}")
                raise RetryTimeoutException(f"Deadlock timeout exceeded for {operation_name}")
            
            try:
                # Execute with per-attempt timeout
                if self.config.per_attempt_timeout:
                    async with asyncio.timeout(self.config.per_attempt_timeout):
                        yield
                else:
                    yield
                
                # Success - update metrics
                self.metrics.total_attempts += attempt
                self.metrics.successful_retries += 1
                self.metrics.total_delay_time += total_delay
                
                # Update strategy performance
                strategy_name = self.config.strategy.value
                if strategy_name not in self.metrics.strategy_performance:
                    self.metrics.strategy_performance[strategy_name] = {
                        'success_count': 0,
                        'total_attempts': 0,
                        'average_attempts': 0.0
                    }
                
                perf = self.metrics.strategy_performance[strategy_name]
                perf['success_count'] += 1
                perf['total_attempts'] += attempt
                perf['average_attempts'] = perf['total_attempts'] / perf['success_count']
                
                # Update adaptive parameters
                if self.config.adaptive_multiplier_adjustment:
                    await self._update_adaptive_parameters(True, attempt)
                
                logger.info(f"Operation {operation_name} succeeded on attempt {attempt}")
                return
                
            except Exception as e:
                last_exception = e
                
                # Update exception metrics
                exception_name = type(e).__name__
                self.metrics.exception_counts[exception_name] = (
                    self.metrics.exception_counts.get(exception_name, 0) + 1
                )
                
                # Check if we should retry
                if not self._should_retry(e, attempt):
                    self.metrics.failed_retries += 1
                    logger.error(f"Operation {operation_name} failed (non-retryable): {e}")
                    raise
                
                # Check if we've exhausted attempts
                if attempt >= self.config.max_attempts:
                    self.metrics.failed_retries += 1
                    logger.error(f"Operation {operation_name} failed after {attempt} attempts: {e}")
                    raise
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                total_delay += delay
                
                # Update delay metrics
                self.metrics.min_retry_delay = min(self.metrics.min_retry_delay, delay)
                self.metrics.max_retry_delay = max(self.metrics.max_retry_delay, delay)
                
                logger.warning(f"Operation {operation_name} failed on attempt {attempt}, retrying in {delay:.2f}s: {e}")
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # If we get here, all attempts failed
        self.metrics.failed_retries += 1
        if last_exception:
            raise last_exception
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry based on exception type and conditions."""
        # Check non-retryable exceptions
        if any(isinstance(exception, exc_type) for exc_type in self.config.non_retryable_exceptions):
            return False
        
        # Check retryable exceptions (if specified)
        if self.config.retryable_exceptions:
            if not any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions):
                return False
        
        # Check custom retry condition
        if self.config.retry_condition:
            return self.config.retry_condition(exception, attempt)
        
        # Default: retry most exceptions
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.adaptive_multiplier ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.config.base_delay * self._get_fibonacci_number(attempt)
        
        elif self.config.strategy == RetryStrategy.CONSTANT_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.ADAPTIVE:
            delay = self._calculate_adaptive_delay(attempt)
        
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.adaptive_max_delay)
        
        # Apply jitter
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_max * (2 * random.random() - 1)
            delay += jitter_amount
        
        # Ensure minimum delay
        delay = max(delay, 0.1)
        
        return delay
    
    def _get_fibonacci_number(self, n: int) -> int:
        """Get nth Fibonacci number."""
        while len(self.fibonacci_sequence) < n:
            next_fib = self.fibonacci_sequence[-1] + self.fibonacci_sequence[-2]
            self.fibonacci_sequence.append(next_fib)
        
        return self.fibonacci_sequence[n - 1]
    
    def _calculate_adaptive_delay(self, attempt: int) -> float:
        """Calculate adaptive delay based on recent performance."""
        # Base exponential backoff
        base_delay = self.config.base_delay * (self.adaptive_multiplier ** (attempt - 1))
        
        # Adjust based on recent failure rates
        if self.metrics.total_attempts > 0:
            failure_rate = self.metrics.failed_retries / self.metrics.total_attempts
            
            # Increase delay if high failure rate
            if failure_rate > 0.5:
                base_delay *= 1.5
            elif failure_rate < 0.1:
                base_delay *= 0.8
        
        # Adjust based on circuit breaker state
        if self.circuit_breaker and hasattr(self.circuit_breaker, 'state'):
            if self.circuit_breaker.state.value == 'half_open':
                base_delay *= 2.0  # Be more conservative in half-open state
        
        return base_delay
    
    async def _update_adaptive_parameters(self, success: bool, attempts: int):
        """Update adaptive parameters based on operation outcome."""
        if success:
            if attempts == 1:
                # First attempt success - can be more aggressive
                self.adaptive_multiplier = max(1.5, self.adaptive_multiplier * 0.95)
                self.adaptive_max_delay = max(10.0, self.adaptive_max_delay * 0.9)
            else:
                # Retry success - maintain current parameters
                pass
        else:
            # Failure - be more conservative
            self.adaptive_multiplier = min(5.0, self.adaptive_multiplier * 1.1)
            self.adaptive_max_delay = min(300.0, self.adaptive_max_delay * 1.2)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive retry metrics."""
        # Calculate derived metrics
        if self.metrics.total_attempts > 0:
            self.metrics.success_rate = self.metrics.successful_retries / self.metrics.total_attempts
        
        if self.metrics.successful_retries > 0:
            self.metrics.average_attempts_per_operation = (
                self.metrics.total_attempts / self.metrics.successful_retries
            )
        
        if self.metrics.total_attempts > 0:
            self.metrics.average_retry_delay = (
                self.metrics.total_delay_time / self.metrics.total_attempts
            )
        
        return {
            'name': self.name,
            'config': {
                'max_attempts': self.config.max_attempts,
                'strategy': self.config.strategy.value,
                'base_delay': self.config.base_delay,
                'max_delay': self.config.max_delay,
                'multiplier': self.config.multiplier,
                'adaptive_multiplier': self.adaptive_multiplier,
                'adaptive_max_delay': self.adaptive_max_delay
            },
            'metrics': {
                'total_attempts': self.metrics.total_attempts,
                'successful_retries': self.metrics.successful_retries,
                'failed_retries': self.metrics.failed_retries,
                'circuit_breaker_rejections': self.metrics.circuit_breaker_rejections,
                'timeout_failures': self.metrics.timeout_failures,
                'success_rate': self.metrics.success_rate,
                'average_attempts_per_operation': self.metrics.average_attempts_per_operation,
                'total_delay_time': self.metrics.total_delay_time,
                'average_retry_delay': self.metrics.average_retry_delay,
                'min_retry_delay': self.metrics.min_retry_delay if self.metrics.min_retry_delay != float('inf') else 0,
                'max_retry_delay': self.metrics.max_retry_delay
            },
            'strategy_performance': self.metrics.strategy_performance,
            'exception_counts': self.metrics.exception_counts,
            'active_retries': len(self.active_retries),
            'circuit_breaker_connected': self.circuit_breaker is not None
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = RetryMetrics()
        logger.info(f"Metrics reset for retry manager: {self.name}")
    
    async def test_strategy(self, test_duration: int = 60) -> Dict[str, Any]:
        """Test retry strategy with simulated failures."""
        logger.info(f"Testing retry strategy for {test_duration} seconds")
        
        start_time = time.time()
        test_results = {
            'strategy': self.config.strategy.value,
            'test_duration': test_duration,
            'operations_attempted': 0,
            'operations_succeeded': 0,
            'total_retry_attempts': 0,
            'average_attempts_per_operation': 0.0,
            'success_rate': 0.0
        }
        
        # Simulate operations with varying failure rates
        failure_rates = [0.1, 0.3, 0.5, 0.7]
        
        for failure_rate in failure_rates:
            operations_count = 0
            
            while time.time() - start_time < test_duration / len(failure_rates):
                try:
                    async with self.retry(f"test_operation_{failure_rate}"):
                        # Simulate operation
                        if random.random() < failure_rate:
                            raise Exception(f"Simulated failure (rate: {failure_rate})")
                        
                        await asyncio.sleep(0.01)  # Simulate work
                    
                    operations_count += 1
                    
                except Exception:
                    pass  # Expected failures
                
                await asyncio.sleep(0.1)  # Throttle test operations
        
        # Calculate test results
        current_metrics = self.get_metrics()
        test_results.update({
            'operations_attempted': current_metrics['metrics']['total_attempts'],
            'operations_succeeded': current_metrics['metrics']['successful_retries'],
            'total_retry_attempts': current_metrics['metrics']['total_attempts'],
            'average_attempts_per_operation': current_metrics['metrics']['average_attempts_per_operation'],
            'success_rate': current_metrics['metrics']['success_rate']
        })
        
        logger.info(f"Strategy test completed: {test_results}")
        return test_results


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryTimeoutException(Exception):
    """Exception raised when retry timeout is exceeded."""
    pass


class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted."""
    pass


# Convenience functions for common retry patterns
def create_database_retry_manager(circuit_breaker=None) -> RetryManager:
    """Create retry manager optimized for database operations."""
    config = RetryConfig(
        max_attempts=5,
        base_delay=0.5,
        max_delay=30.0,
        multiplier=2.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter=True,
        timeout=60.0,
        per_attempt_timeout=10.0,
        retryable_exceptions=[
            ConnectionError,
            TimeoutError,
            OSError
        ],
        non_retryable_exceptions=[
            ValueError,
            TypeError,
            KeyError
        ]
    )
    
    return RetryManager(config, circuit_breaker, "database")


def create_api_retry_manager(circuit_breaker=None) -> RetryManager:
    """Create retry manager optimized for API calls."""
    config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=60.0,
        multiplier=2.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        jitter=True,
        timeout=120.0,
        per_attempt_timeout=30.0,
        adaptive_multiplier_adjustment=True
    )
    
    return RetryManager(config, circuit_breaker, "api")


def create_broker_retry_manager(circuit_breaker=None) -> RetryManager:
    """Create retry manager optimized for broker operations."""
    config = RetryConfig(
        max_attempts=3,
        base_delay=0.1,
        max_delay=5.0,
        multiplier=1.5,
        strategy=RetryStrategy.ADAPTIVE,
        jitter=True,
        timeout=30.0,
        per_attempt_timeout=5.0,
        circuit_breaker_integration=True
    )
    
    return RetryManager(config, circuit_breaker, "broker")