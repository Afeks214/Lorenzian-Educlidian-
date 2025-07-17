"""
Enhanced Circuit Breaker Implementation
======================================

Advanced circuit breaker pattern with machine learning integration,
comprehensive metrics, and adaptive failure detection.

Features:
- Multiple failure detection strategies
- Real-time metrics collection
- Event-driven notifications
- State persistence across restarts
- Integration with monitoring systems
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import redis.asyncio as redis
from contextlib import asynccontextmanager

from ..event_bus import EventBus
from ..events import Event

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states with detailed semantics."""
    CLOSED = "closed"              # Normal operation
    OPEN = "open"                  # Failure detected, blocking requests
    HALF_OPEN = "half_open"        # Testing recovery
    FORCED_OPEN = "forced_open"    # Manually opened for maintenance
    DEGRADED = "degraded"          # Partial functionality mode


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    CRITICAL_ERROR = "critical_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    # Failure thresholds
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    half_open_timeout: float = 30.0
    
    # Timing configuration
    failure_rate_window: int = 60  # seconds
    min_requests_threshold: int = 10
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    backoff_multiplier: float = 2.0
    
    # Health check configuration
    health_check_interval: float = 10.0
    health_check_timeout: float = 5.0
    
    # Monitoring configuration
    metrics_enabled: bool = True
    event_notifications: bool = True
    persist_state: bool = True
    
    # Service-specific settings
    service_name: str = "default"
    service_priority: int = 1  # 1=critical, 2=important, 3=optional
    
    # ML integration
    enable_ml_prediction: bool = True
    prediction_threshold: float = 0.7


@dataclass
class CircuitBreakerMetrics:
    """Metrics collected by circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_count: int = 0
    circuit_opens: int = 0
    circuit_closes: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    average_response_time: float = 0.0
    failure_rate: float = 0.0
    uptime_percentage: float = 100.0
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0


class CircuitBreaker:
    """
    Advanced circuit breaker with ML integration and comprehensive monitoring.
    
    This implementation provides:
    - Adaptive failure detection based on multiple criteria
    - Machine learning integration for failure prediction
    - Comprehensive metrics collection and monitoring
    - State persistence across service restarts
    - Event-driven notifications for state changes
    - Integration with chaos engineering for testing
    """
    
    def __init__(
        self,
        config: CircuitBreakerConfig,
        event_bus: Optional[EventBus] = None,
        redis_client: Optional[redis.Redis] = None,
        ml_predictor: Optional[Callable] = None
    ):
        """Initialize circuit breaker with configuration."""
        self.config = config
        self.event_bus = event_bus
        self.redis_client = redis_client
        self.ml_predictor = ml_predictor
        
        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.next_attempt_time = 0.0
        self.state_change_time = time.time()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        self.request_history: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []
        
        # Redis keys for persistence
        self.state_key = f"circuit_breaker:{config.service_name}:state"
        self.metrics_key = f"circuit_breaker:{config.service_name}:metrics"
        self.history_key = f"circuit_breaker:{config.service_name}:history"
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        logger.info(f"Circuit breaker initialized for {config.service_name}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self):
        """Initialize circuit breaker and restore state."""
        try:
            # Restore state from Redis if available
            if self.redis_client:
                await self._restore_state()
            
            # Start background tasks
            if self.config.health_check_interval > 0:
                self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            if self.config.metrics_enabled:
                self.metrics_task = asyncio.create_task(self._metrics_loop())
            
            logger.info(f"Circuit breaker initialized for {self.config.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker: {e}")
            raise
    
    async def close(self):
        """Close circuit breaker and cleanup resources."""
        try:
            # Cancel background tasks
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self.metrics_task:
                self.metrics_task.cancel()
                try:
                    await self.metrics_task
                except asyncio.CancelledError:
                    pass
            
            # Persist final state
            if self.redis_client:
                await self._persist_state()
            
            logger.info(f"Circuit breaker closed for {self.config.service_name}")
            
        except Exception as e:
            logger.error(f"Error closing circuit breaker: {e}")
    
    @asynccontextmanager
    async def protect(self):
        """
        Context manager for protecting service calls.
        
        Usage:
            async with circuit_breaker.protect():
                result = await external_service_call()
        """
        if not await self.can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker is {self.state.value} for {self.config.service_name}"
            )
        
        start_time = time.time()
        success = False
        
        try:
            # Execute the protected code
            yield
            success = True
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            await self._record_failure(e, execution_time)
            raise
            
        finally:
            if success:
                execution_time = time.time() - start_time
                await self._record_success(execution_time)
    
    async def can_execute(self) -> bool:
        """Check if circuit breaker allows execution."""
        current_time = time.time()
        
        # Check ML prediction if enabled
        if self.config.enable_ml_prediction and self.ml_predictor:
            try:
                prediction = await self._get_ml_prediction()
                if prediction > self.config.prediction_threshold:
                    logger.warning(f"ML predictor suggests high failure probability: {prediction:.2f}")
                    return False
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.state == CircuitBreakerState.OPEN:
            if current_time >= self.next_attempt_time:
                # Transition to half-open for testing
                await self._transition_to_half_open()
                return True
            return False
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        elif self.state == CircuitBreakerState.FORCED_OPEN:
            return False
        
        elif self.state == CircuitBreakerState.DEGRADED:
            # Allow limited requests in degraded mode
            return await self._can_execute_degraded()
        
        return False
    
    async def _record_success(self, execution_time: float):
        """Record successful execution."""
        current_time = time.time()
        
        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = datetime.now()
        
        # Update performance metrics
        self.performance_history.append(execution_time)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        self.metrics.average_response_time = sum(self.performance_history) / len(self.performance_history)
        
        # Update success count
        self.success_count += 1
        self.last_success_time = current_time
        
        # Handle state transitions
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
        
        elif self.state == CircuitBreakerState.DEGRADED:
            # Check if we can exit degraded mode
            if self._should_exit_degraded_mode():
                await self._transition_to_closed()
        
        # Record request history
        self.request_history.append({
            'timestamp': current_time,
            'success': True,
            'execution_time': execution_time,
            'state': self.state.value
        })
        
        # Cleanup old history
        cutoff_time = current_time - self.config.failure_rate_window
        self.request_history = [
            req for req in self.request_history 
            if req['timestamp'] > cutoff_time
        ]
        
        # Update failure rate
        self._update_failure_rate()
        
        # Persist state
        if self.redis_client:
            await self._persist_state()
    
    async def _record_failure(self, exception: Exception, execution_time: float):
        """Record failed execution."""
        current_time = time.time()
        
        # Determine failure type
        failure_type = self._classify_failure(exception)
        
        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = datetime.now()
        
        if failure_type == FailureType.TIMEOUT:
            self.metrics.timeout_count += 1
        
        # Update failure count
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # Record request history
        self.request_history.append({
            'timestamp': current_time,
            'success': False,
            'execution_time': execution_time,
            'failure_type': failure_type.value,
            'exception': str(exception),
            'state': self.state.value
        })
        
        # Cleanup old history
        cutoff_time = current_time - self.config.failure_rate_window
        self.request_history = [
            req for req in self.request_history 
            if req['timestamp'] > cutoff_time
        ]
        
        # Update failure rate
        self._update_failure_rate()
        
        # Check if we should open circuit
        if await self._should_open_circuit():
            await self._transition_to_open()
        
        # Persist state
        if self.redis_client:
            await self._persist_state()
        
        # Send event notification
        if self.event_bus and self.config.event_notifications:
            await self.event_bus.publish(Event(
                type="circuit_breaker_failure",
                data={
                    'service': self.config.service_name,
                    'failure_type': failure_type.value,
                    'failure_count': self.failure_count,
                    'state': self.state.value,
                    'exception': str(exception)
                }
            ))
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure."""
        exception_str = str(exception).lower()
        
        if 'timeout' in exception_str or 'timed out' in exception_str:
            return FailureType.TIMEOUT
        elif 'connection' in exception_str or 'network' in exception_str:
            return FailureType.CONNECTION_ERROR
        elif 'unavailable' in exception_str or '503' in exception_str:
            return FailureType.SERVICE_UNAVAILABLE
        elif 'rate limit' in exception_str or '429' in exception_str:
            return FailureType.RATE_LIMIT
        elif 'auth' in exception_str or '401' in exception_str or '403' in exception_str:
            return FailureType.AUTHENTICATION_ERROR
        elif 'validation' in exception_str or '400' in exception_str:
            return FailureType.VALIDATION_ERROR
        elif 'critical' in exception_str or '500' in exception_str:
            return FailureType.CRITICAL_ERROR
        else:
            return FailureType.CRITICAL_ERROR
    
    async def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened."""
        # Check basic failure threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate
        if self.metrics.failure_rate > 50.0 and len(self.request_history) >= self.config.min_requests_threshold:
            return True
        
        # Check for critical failures
        recent_failures = [
            req for req in self.request_history[-5:] 
            if not req['success']
        ]
        
        critical_failures = [
            req for req in recent_failures 
            if req.get('failure_type') == FailureType.CRITICAL_ERROR.value
        ]
        
        if len(critical_failures) >= 3:
            return True
        
        return False
    
    def _should_exit_degraded_mode(self) -> bool:
        """Check if we should exit degraded mode."""
        # Check recent success rate
        recent_requests = self.request_history[-20:]
        if len(recent_requests) < 10:
            return False
        
        successful_requests = [req for req in recent_requests if req['success']]
        success_rate = len(successful_requests) / len(recent_requests)
        
        return success_rate > 0.8
    
    async def _can_execute_degraded(self) -> bool:
        """Check if execution is allowed in degraded mode."""
        # Implement throttling in degraded mode
        current_time = time.time()
        
        # Allow only 50% of normal traffic
        if hash(str(current_time)) % 2 == 0:
            return True
        
        return False
    
    def _update_failure_rate(self):
        """Update failure rate based on recent requests."""
        if not self.request_history:
            self.metrics.failure_rate = 0.0
            return
        
        total_requests = len(self.request_history)
        failed_requests = len([req for req in self.request_history if not req['success']])
        
        if total_requests > 0:
            self.metrics.failure_rate = (failed_requests / total_requests) * 100.0
        else:
            self.metrics.failure_rate = 0.0
    
    async def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        self.state = CircuitBreakerState.OPEN
        self.state_change_time = time.time()
        self.next_attempt_time = time.time() + self.config.timeout_seconds
        self.metrics.circuit_opens += 1
        
        logger.warning(f"Circuit breaker OPENED for {self.config.service_name}")
        
        # Send event notification
        if self.event_bus and self.config.event_notifications:
            await self.event_bus.publish(Event(
                type="circuit_breaker_opened",
                data={
                    'service': self.config.service_name,
                    'failure_count': self.failure_count,
                    'failure_rate': self.metrics.failure_rate,
                    'next_attempt_time': self.next_attempt_time
                }
            ))
    
    async def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.state_change_time = time.time()
        self.success_count = 0
        
        logger.info(f"Circuit breaker HALF-OPEN for {self.config.service_name}")
        
        # Send event notification
        if self.event_bus and self.config.event_notifications:
            await self.event_bus.publish(Event(
                type="circuit_breaker_half_open",
                data={
                    'service': self.config.service_name,
                    'test_period_seconds': self.config.half_open_timeout
                }
            ))
    
    async def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.state_change_time = time.time()
        self.failure_count = 0
        self.success_count = 0
        self.metrics.circuit_closes += 1
        
        logger.info(f"Circuit breaker CLOSED for {self.config.service_name}")
        
        # Send event notification
        if self.event_bus and self.config.event_notifications:
            await self.event_bus.publish(Event(
                type="circuit_breaker_closed",
                data={
                    'service': self.config.service_name,
                    'recovery_time_seconds': time.time() - self.state_change_time
                }
            ))
    
    async def _get_ml_prediction(self) -> float:
        """Get ML prediction for failure probability."""
        if not self.ml_predictor:
            return 0.0
        
        # Prepare features for ML model
        features = {
            'failure_rate': self.metrics.failure_rate,
            'average_response_time': self.metrics.average_response_time,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'time_since_last_failure': time.time() - self.last_failure_time if self.last_failure_time > 0 else 0,
            'current_state': self.state.value,
            'request_count': len(self.request_history)
        }
        
        try:
            prediction = await self.ml_predictor(features)
            return prediction
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return 0.0
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Perform health check logic here
                # This would typically ping the service or check metrics
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _metrics_loop(self):
        """Background metrics collection loop."""
        while True:
            try:
                await asyncio.sleep(30)  # Update metrics every 30 seconds
                
                # Calculate uptime
                if self.metrics.total_requests > 0:
                    self.metrics.uptime_percentage = (
                        self.metrics.successful_requests / self.metrics.total_requests
                    ) * 100.0
                
                # Persist metrics
                if self.redis_client:
                    await self._persist_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    async def _restore_state(self):
        """Restore circuit breaker state from Redis."""
        try:
            state_data = await self.redis_client.hgetall(self.state_key)
            
            if state_data:
                self.state = CircuitBreakerState(state_data.get(b'state', b'closed').decode())
                self.failure_count = int(state_data.get(b'failure_count', b'0'))
                self.success_count = int(state_data.get(b'success_count', b'0'))
                self.last_failure_time = float(state_data.get(b'last_failure_time', b'0'))
                self.last_success_time = float(state_data.get(b'last_success_time', b'0'))
                self.next_attempt_time = float(state_data.get(b'next_attempt_time', b'0'))
                
                logger.info(f"Restored circuit breaker state for {self.config.service_name}: {self.state.value}")
            
            # Restore metrics
            metrics_data = await self.redis_client.hgetall(self.metrics_key)
            if metrics_data:
                self.metrics.total_requests = int(metrics_data.get(b'total_requests', b'0'))
                self.metrics.successful_requests = int(metrics_data.get(b'successful_requests', b'0'))
                self.metrics.failed_requests = int(metrics_data.get(b'failed_requests', b'0'))
                self.metrics.circuit_opens = int(metrics_data.get(b'circuit_opens', b'0'))
                self.metrics.circuit_closes = int(metrics_data.get(b'circuit_closes', b'0'))
                
        except Exception as e:
            logger.error(f"Failed to restore circuit breaker state: {e}")
    
    async def _persist_state(self):
        """Persist circuit breaker state to Redis."""
        try:
            state_data = {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'next_attempt_time': self.next_attempt_time,
                'state_change_time': self.state_change_time
            }
            
            await self.redis_client.hset(self.state_key, mapping=state_data)
            await self.redis_client.expire(self.state_key, 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Failed to persist circuit breaker state: {e}")
    
    async def _persist_metrics(self):
        """Persist metrics to Redis."""
        try:
            metrics_data = {
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'timeout_count': self.metrics.timeout_count,
                'circuit_opens': self.metrics.circuit_opens,
                'circuit_closes': self.metrics.circuit_closes,
                'average_response_time': self.metrics.average_response_time,
                'failure_rate': self.metrics.failure_rate,
                'uptime_percentage': self.metrics.uptime_percentage
            }
            
            await self.redis_client.hset(self.metrics_key, mapping=metrics_data)
            await self.redis_client.expire(self.metrics_key, 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    async def force_open(self):
        """Manually force circuit breaker open."""
        self.state = CircuitBreakerState.FORCED_OPEN
        self.state_change_time = time.time()
        
        logger.warning(f"Circuit breaker FORCE OPENED for {self.config.service_name}")
        
        if self.redis_client:
            await self._persist_state()
    
    async def force_close(self):
        """Manually force circuit breaker closed."""
        await self._transition_to_closed()
        
        logger.info(f"Circuit breaker FORCE CLOSED for {self.config.service_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            'service': self.config.service_name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'last_success_time': self.last_success_time,
            'next_attempt_time': self.next_attempt_time,
            'state_change_time': self.state_change_time,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'success_rate': self.metrics.success_rate(),
                'failure_rate': self.metrics.failure_rate,
                'average_response_time': self.metrics.average_response_time,
                'uptime_percentage': self.metrics.uptime_percentage,
                'circuit_opens': self.metrics.circuit_opens,
                'circuit_closes': self.metrics.circuit_closes
            },
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'success_threshold': self.config.success_threshold,
                'timeout_seconds': self.config.timeout_seconds,
                'service_priority': self.config.service_priority
            }
        }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerTimeoutException(Exception):
    """Exception raised when circuit breaker times out."""
    pass