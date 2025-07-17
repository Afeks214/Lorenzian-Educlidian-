"""
Circuit breaker patterns for external dependencies and high-failure operations.

Implements sophisticated circuit breaker patterns specifically designed for
trading system dependencies like market data feeds, execution venues,
risk management systems, and other external services.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager, asynccontextmanager
from collections import deque
import statistics

from .base_exceptions import (
    BaseGrandModelError, ErrorSeverity, ErrorCategory, ErrorContext,
    CircuitBreakerError, TimeoutError, NetworkError, DependencyError
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failures detected, circuit open
    HALF_OPEN = "half_open" # Testing if service recovered


class DependencyType(Enum):
    """Types of dependencies"""
    MARKET_DATA = "market_data"
    EXECUTION_VENUE = "execution_venue"
    RISK_SYSTEM = "risk_system"
    PRICING_SERVICE = "pricing_service"
    COMPLIANCE_SYSTEM = "compliance_system"
    POSITION_MANAGEMENT = "position_management"
    PORTFOLIO_SYSTEM = "portfolio_system"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    MESSAGING_SYSTEM = "messaging_system"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3  # For half-open state
    request_timeout: float = 10.0
    max_requests_half_open: int = 5
    failure_rate_threshold: float = 0.5  # 50% failure rate
    monitoring_window: int = 60  # seconds
    dependency_type: DependencyType = DependencyType.EXTERNAL_API
    name: str = "default"
    health_check_interval: float = 5.0
    degraded_mode_enabled: bool = True
    fallback_enabled: bool = True


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_open_count: int = 0
    circuit_half_open_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    average_response_time: float = 0.0
    failure_rate: float = 0.0
    state_change_history: List[Tuple[CircuitState, datetime]] = field(default_factory=list)


class CircuitBreakerException(BaseGrandModelError):
    """Exception raised when circuit breaker is open"""
    
    def __init__(self, circuit_name: str, failure_count: int, **kwargs):
        super().__init__(
            message=f"Circuit breaker '{circuit_name}' is open after {failure_count} failures",
            error_code="CIRCUIT_BREAKER_OPEN",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CIRCUIT_BREAKER,
            recoverable=True,
            **kwargs
        )
        self.circuit_name = circuit_name
        self.failure_count = failure_count


class DependencyCircuitBreaker:
    """Circuit breaker for external dependencies"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.next_attempt_time = None
        self.metrics = CircuitBreakerMetrics()
        self.response_times = deque(maxlen=100)
        self.failure_window = deque(maxlen=100)
        self.lock = threading.RLock()
        self.health_check_thread = None
        self.health_check_running = False
        
        # Start health check thread if enabled
        if self.config.health_check_interval > 0:
            self._start_health_check()
        
        logger.info(f"Circuit breaker '{self.config.name}' initialized for {self.config.dependency_type.value}")
    
    def _start_health_check(self):
        """Start health check thread"""
        self.health_check_running = True
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
    
    def _health_check_loop(self):
        """Health check loop"""
        while self.health_check_running:
            try:
                time.sleep(self.config.health_check_interval)
                if self.state == CircuitState.OPEN and self._should_attempt_reset():
                    self._attempt_reset()
            except Exception as e:
                logger.error(f"Health check error for {self.config.name}: {e}")
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit breaker"""
        if self.next_attempt_time is None:
            return True
        
        return datetime.utcnow() >= self.next_attempt_time
    
    def _attempt_reset(self):
        """Attempt to reset circuit breaker"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.config.name}' moved to half-open state")
                self._record_state_change(CircuitState.HALF_OPEN)
    
    def _record_state_change(self, new_state: CircuitState):
        """Record state change"""
        self.metrics.state_change_history.append((new_state, datetime.utcnow()))
        
        # Keep only recent history
        if len(self.metrics.state_change_history) > 1000:
            self.metrics.state_change_history = self.metrics.state_change_history[-1000:]
        
        if new_state == CircuitState.OPEN:
            self.metrics.circuit_open_count += 1
        elif new_state == CircuitState.HALF_OPEN:
            self.metrics.circuit_half_open_count += 1
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.config.name}' moved to half-open state")
                    self._record_state_change(CircuitState.HALF_OPEN)
                    return True
                return False
            else:  # HALF_OPEN
                return self.success_count < self.config.max_requests_half_open
    
    def record_success(self, response_time: float = 0.0):
        """Record successful execution"""
        with self.lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.utcnow()
            self.last_success_time = datetime.utcnow()
            
            if response_time > 0:
                self.response_times.append(response_time)
                self._update_average_response_time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._close_circuit()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
            
            # Update failure rate
            self._update_failure_rate()
            
            logger.debug(f"Success recorded for {self.config.name}, response time: {response_time:.3f}s")
    
    def record_failure(self, error: Exception, response_time: float = 0.0):
        """Record failed execution"""
        with self.lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.utcnow()
            self.last_failure_time = datetime.utcnow()
            self.failure_count += 1
            
            if response_time > 0:
                self.response_times.append(response_time)
                self._update_average_response_time()
            
            # Record failure in window
            self.failure_window.append(datetime.utcnow())
            
            # Update failure rate
            self._update_failure_rate()
            
            if self.state == CircuitState.CLOSED:
                if self._should_open_circuit():
                    self._open_circuit()
            elif self.state == CircuitState.HALF_OPEN:
                # Return to open state on failure
                self._open_circuit()
            
            logger.warning(f"Failure recorded for {self.config.name}: {error}")
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened"""
        # Check failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate threshold
        if self.metrics.failure_rate >= self.config.failure_rate_threshold:
            return True
        
        return False
    
    def _open_circuit(self):
        """Open the circuit breaker"""
        self.state = CircuitState.OPEN
        self.next_attempt_time = datetime.utcnow() + timedelta(seconds=self.config.recovery_timeout)
        logger.warning(f"Circuit breaker '{self.config.name}' opened due to failures")
        self._record_state_change(CircuitState.OPEN)
    
    def _close_circuit(self):
        """Close the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.next_attempt_time = None
        logger.info(f"Circuit breaker '{self.config.name}' closed - service recovered")
        self._record_state_change(CircuitState.CLOSED)
    
    def _update_average_response_time(self):
        """Update average response time"""
        if self.response_times:
            self.metrics.average_response_time = statistics.mean(self.response_times)
    
    def _update_failure_rate(self):
        """Update failure rate"""
        if self.metrics.total_requests > 0:
            self.metrics.failure_rate = self.metrics.failed_requests / self.metrics.total_requests
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics"""
        with self.lock:
            return CircuitBreakerMetrics(
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                circuit_open_count=self.metrics.circuit_open_count,
                circuit_half_open_count=self.metrics.circuit_half_open_count,
                last_failure_time=self.metrics.last_failure_time,
                last_success_time=self.metrics.last_success_time,
                average_response_time=self.metrics.average_response_time,
                failure_rate=self.metrics.failure_rate,
                state_change_history=self.metrics.state_change_history.copy()
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self.lock:
            return {
                "name": self.config.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
                "next_attempt_time": self.next_attempt_time.isoformat() if self.next_attempt_time else None,
                "dependency_type": self.config.dependency_type.value,
                "metrics": self.get_metrics()
            }
    
    def reset(self):
        """Reset circuit breaker"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.next_attempt_time = None
            self.metrics = CircuitBreakerMetrics()
            self.response_times.clear()
            self.failure_window.clear()
            logger.info(f"Circuit breaker '{self.config.name}' reset")
    
    def force_open(self):
        """Force circuit breaker open"""
        with self.lock:
            self._open_circuit()
            logger.warning(f"Circuit breaker '{self.config.name}' forced open")
    
    def force_close(self):
        """Force circuit breaker closed"""
        with self.lock:
            self._close_circuit()
            logger.info(f"Circuit breaker '{self.config.name}' forced closed")
    
    def shutdown(self):
        """Shutdown circuit breaker"""
        self.health_check_running = False
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=1.0)
        
        logger.info(f"Circuit breaker '{self.config.name}' shut down")
    
    def __enter__(self):
        """Context manager entry"""
        if not self.can_execute():
            raise CircuitBreakerException(
                circuit_name=self.config.name,
                failure_count=self.failure_count
            )
        
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        response_time = time.time() - self.start_time
        
        if exc_type is None:
            self.record_success(response_time)
        else:
            self.record_failure(exc_val, response_time)
        
        return False  # Don't suppress exceptions


class DependencyCircuitBreakerManager:
    """Manages multiple circuit breakers for different dependencies"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, DependencyCircuitBreaker] = {}
        self.lock = threading.RLock()
    
    def register_dependency(self, name: str, config: CircuitBreakerConfig) -> DependencyCircuitBreaker:
        """Register a new dependency circuit breaker"""
        with self.lock:
            circuit_breaker = DependencyCircuitBreaker(config)
            self.circuit_breakers[name] = circuit_breaker
            logger.info(f"Registered circuit breaker for dependency: {name}")
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[DependencyCircuitBreaker]:
        """Get circuit breaker by name"""
        with self.lock:
            return self.circuit_breakers.get(name)
    
    def execute_with_circuit_breaker(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        circuit_breaker = self.get_circuit_breaker(name)
        if circuit_breaker is None:
            raise ValueError(f"Circuit breaker '{name}' not found")
        
        with circuit_breaker:
            return func(*args, **kwargs)
    
    async def async_execute_with_circuit_breaker(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        circuit_breaker = self.get_circuit_breaker(name)
        if circuit_breaker is None:
            raise ValueError(f"Circuit breaker '{name}' not found")
        
        if not circuit_breaker.can_execute():
            raise CircuitBreakerException(
                circuit_name=name,
                failure_count=circuit_breaker.failure_count
            )
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            response_time = time.time() - start_time
            circuit_breaker.record_success(response_time)
            return result
        except Exception as e:
            response_time = time.time() - start_time
            circuit_breaker.record_failure(e, response_time)
            raise
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers"""
        with self.lock:
            states = {}
            for name, circuit_breaker in self.circuit_breakers.items():
                states[name] = circuit_breaker.get_state()
            return states
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all circuit breakers"""
        with self.lock:
            summary = {
                'total_dependencies': len(self.circuit_breakers),
                'healthy_dependencies': 0,
                'degraded_dependencies': 0,
                'failed_dependencies': 0,
                'dependency_details': {}
            }
            
            for name, circuit_breaker in self.circuit_breakers.items():
                state = circuit_breaker.get_state()
                
                if state['state'] == CircuitState.CLOSED.value:
                    summary['healthy_dependencies'] += 1
                    health_status = 'healthy'
                elif state['state'] == CircuitState.HALF_OPEN.value:
                    summary['degraded_dependencies'] += 1
                    health_status = 'degraded'
                else:
                    summary['failed_dependencies'] += 1
                    health_status = 'failed'
                
                summary['dependency_details'][name] = {
                    'status': health_status,
                    'state': state['state'],
                    'failure_count': state['failure_count'],
                    'dependency_type': state['dependency_type']
                }
            
            return summary
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self.lock:
            for circuit_breaker in self.circuit_breakers.values():
                circuit_breaker.reset()
            logger.info("All circuit breakers reset")
    
    def shutdown_all(self):
        """Shutdown all circuit breakers"""
        with self.lock:
            for circuit_breaker in self.circuit_breakers.values():
                circuit_breaker.shutdown()
            logger.info("All circuit breakers shut down")
    
    def __del__(self):
        """Destructor"""
        self.shutdown_all()


# Global circuit breaker manager
_global_circuit_breaker_manager = DependencyCircuitBreakerManager()


def get_circuit_breaker_manager() -> DependencyCircuitBreakerManager:
    """Get global circuit breaker manager"""
    return _global_circuit_breaker_manager


def register_market_data_circuit_breaker(name: str, **kwargs) -> DependencyCircuitBreaker:
    """Register circuit breaker for market data dependency"""
    config = CircuitBreakerConfig(
        name=name,
        dependency_type=DependencyType.MARKET_DATA,
        failure_threshold=3,
        recovery_timeout=15.0,
        request_timeout=5.0,
        **kwargs
    )
    return _global_circuit_breaker_manager.register_dependency(name, config)


def register_execution_venue_circuit_breaker(name: str, **kwargs) -> DependencyCircuitBreaker:
    """Register circuit breaker for execution venue dependency"""
    config = CircuitBreakerConfig(
        name=name,
        dependency_type=DependencyType.EXECUTION_VENUE,
        failure_threshold=2,
        recovery_timeout=20.0,
        request_timeout=10.0,
        **kwargs
    )
    return _global_circuit_breaker_manager.register_dependency(name, config)


def register_risk_system_circuit_breaker(name: str, **kwargs) -> DependencyCircuitBreaker:
    """Register circuit breaker for risk system dependency"""
    config = CircuitBreakerConfig(
        name=name,
        dependency_type=DependencyType.RISK_SYSTEM,
        failure_threshold=1,  # Risk system failures are critical
        recovery_timeout=30.0,
        request_timeout=15.0,
        **kwargs
    )
    return _global_circuit_breaker_manager.register_dependency(name, config)


@contextmanager
def circuit_breaker_protection(name: str):
    """Context manager for circuit breaker protection"""
    circuit_breaker = _global_circuit_breaker_manager.get_circuit_breaker(name)
    if circuit_breaker is None:
        raise ValueError(f"Circuit breaker '{name}' not found")
    
    with circuit_breaker:
        yield


@asynccontextmanager
async def async_circuit_breaker_protection(name: str):
    """Async context manager for circuit breaker protection"""
    circuit_breaker = _global_circuit_breaker_manager.get_circuit_breaker(name)
    if circuit_breaker is None:
        raise ValueError(f"Circuit breaker '{name}' not found")
    
    if not circuit_breaker.can_execute():
        raise CircuitBreakerException(
            circuit_name=name,
            failure_count=circuit_breaker.failure_count
        )
    
    start_time = time.time()
    try:
        yield
        response_time = time.time() - start_time
        circuit_breaker.record_success(response_time)
    except Exception as e:
        response_time = time.time() - start_time
        circuit_breaker.record_failure(e, response_time)
        raise


def with_circuit_breaker(name: str):
    """Decorator for circuit breaker protection"""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with async_circuit_breaker_protection(name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                with circuit_breaker_protection(name):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator