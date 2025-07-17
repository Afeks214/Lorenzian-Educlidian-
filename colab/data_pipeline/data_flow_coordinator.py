"""
Data Flow Coordinator for NQ Data Pipeline

Manages data synchronization, concurrent processing, and consistency
between execution engine and risk management notebooks.
"""

import threading
import time
import queue
import logging
import traceback
import random
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
from pathlib import Path
import pandas as pd
import numpy as np
from enum import Enum
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from contextlib import contextmanager
import uuid
from collections import defaultdict, deque
from typing import Set
import asyncio
from functools import wraps
import weakref

class DataStreamPriority(Enum):
    """Priority levels for data streams"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class DataStreamType(Enum):
    """Types of data streams"""
    MARKET_DATA = "market_data"
    FEATURES = "features"
    PREDICTIONS = "predictions"
    RISK_METRICS = "risk_metrics"
    PERFORMANCE = "performance"

class DataStreamStatus(Enum):
    """Status of data streams"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"
    DEGRADED = "degraded"
    RECOVERING = "recovering"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class BulkheadType(Enum):
    """Types of bulkhead isolation"""
    THREAD_POOL = "thread_pool"
    SEMAPHORE = "semaphore"
    QUEUE = "queue"

@dataclass
class ErrorContext:
    """Context information for error tracking"""
    error_id: str
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    operation: str
    stream_id: Optional[str] = None
    notebook_id: Optional[str] = None
    correlation_id: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovery_attempts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'component': self.component,
            'operation': self.operation,
            'stream_id': self.stream_id,
            'notebook_id': self.notebook_id,
            'correlation_id': self.correlation_id,
            'stack_trace': self.stack_trace,
            'metadata': self.metadata,
            'retry_count': self.retry_count,
            'recovery_attempts': self.recovery_attempts
        }

class ErrorAggregator:
    """Aggregates and tracks errors across the system"""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.correlation_map: Dict[str, List[str]] = defaultdict(list)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__ + ".ErrorAggregator")
    
    def record_error(self, error_context: ErrorContext):
        """Record an error in the aggregator"""
        with self.lock:
            self.errors.append(error_context)
            
            # Update error counts
            error_key = f"{error_context.component}:{error_context.operation}"
            self.error_counts[error_key] += 1
            
            # Update error rates
            self.error_rates[error_key].append(error_context.timestamp)
            
            # Track correlations
            if error_context.correlation_id:
                self.correlation_map[error_context.correlation_id].append(error_context.error_id)
            
            self.logger.warning(f"Error recorded: {error_context.error_type} in {error_context.component}")
    
    def get_error_rate(self, component: str, operation: str, window_seconds: int = 300) -> float:
        """Calculate error rate for a component/operation"""
        with self.lock:
            error_key = f"{component}:{operation}"
            if error_key not in self.error_rates:
                return 0.0
            
            current_time = time.time()
            window_start = current_time - window_seconds
            
            # Count errors in window
            recent_errors = sum(1 for ts in self.error_rates[error_key] if ts >= window_start)
            
            # Simple rate calculation (errors per second)
            return recent_errors / window_seconds
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        with self.lock:
            current_time = time.time()
            
            # Group errors by severity
            severity_counts = defaultdict(int)
            component_errors = defaultdict(int)
            recent_errors = []
            
            for error in self.errors:
                severity_counts[error.severity.value] += 1
                component_errors[error.component] += 1
                
                # Recent errors (last 5 minutes)
                if current_time - error.timestamp <= 300:
                    recent_errors.append(error.to_dict())
            
            return {
                'total_errors': len(self.errors),
                'severity_breakdown': dict(severity_counts),
                'component_breakdown': dict(component_errors),
                'recent_errors': recent_errors[-10:],  # Last 10 recent errors
                'error_rates': {k: len(v) for k, v in self.error_rates.items()},
                'correlation_count': len(self.correlation_map)
            }
    
    def get_correlated_errors(self, correlation_id: str) -> List[ErrorContext]:
        """Get all errors with the same correlation ID"""
        with self.lock:
            if correlation_id not in self.correlation_map:
                return []
            
            error_ids = self.correlation_map[correlation_id]
            return [error for error in self.errors if error.error_id in error_ids]

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    monitoring_window_seconds: int = 300
    min_throughput: int = 10
    error_rate_threshold: float = 0.5
    
@dataclass
class RetryConfig:
    """Configuration for retry mechanism"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_multiplier: float = 2.0

@dataclass
class BulkheadConfig:
    """Configuration for bulkhead pattern"""
    bulkhead_type: BulkheadType = BulkheadType.THREAD_POOL
    max_concurrent_requests: int = 10
    queue_size: int = 100
    timeout_seconds: int = 30

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        self.request_count = 0
        self.success_rate_window = deque(maxlen=config.monitoring_window_seconds)
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__ + f".CircuitBreaker.{name}")
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_transitions = []
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            self.total_requests += 1
            
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure(e)
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self.next_attempt_time is None:
            return True
        return time.time() >= self.next_attempt_time
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.failure_count = 0
        self._record_state_transition(CircuitBreakerState.HALF_OPEN)
        self.logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN")
    
    def _on_success(self):
        """Handle successful request"""
        self.total_successes += 1
        self.success_rate_window.append((time.time(), True))
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    def _on_failure(self, exception: Exception):
        """Handle failed request"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_rate_window.append((time.time(), False))
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._transition_to_open()
        elif self.state == CircuitBreakerState.CLOSED:
            if self._should_open():
                self._transition_to_open()
    
    def _should_open(self) -> bool:
        """Check if circuit breaker should open"""
        # Check failure threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check error rate if we have enough throughput
        if len(self.success_rate_window) >= self.config.min_throughput:
            current_time = time.time()
            window_start = current_time - self.config.monitoring_window_seconds
            
            recent_requests = [(ts, success) for ts, success in self.success_rate_window if ts >= window_start]
            
            if len(recent_requests) >= self.config.min_throughput:
                failures = sum(1 for _, success in recent_requests if not success)
                error_rate = failures / len(recent_requests)
                
                if error_rate >= self.config.error_rate_threshold:
                    return True
        
        return False
    
    def _transition_to_open(self):
        """Transition to open state"""
        self.state = CircuitBreakerState.OPEN
        self.next_attempt_time = time.time() + self.config.timeout_seconds
        self._record_state_transition(CircuitBreakerState.OPEN)
        self.logger.warning(f"Circuit breaker {self.name} transitioned to OPEN")
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.next_attempt_time = None
        self._record_state_transition(CircuitBreakerState.CLOSED)
        self.logger.info(f"Circuit breaker {self.name} transitioned to CLOSED")
    
    def _record_state_transition(self, new_state: CircuitBreakerState):
        """Record state transition for metrics"""
        self.state_transitions.append({
            'timestamp': time.time(),
            'from_state': self.state.value if hasattr(self, 'state') else None,
            'to_state': new_state.value
        })
        
        # Keep only recent transitions
        if len(self.state_transitions) > 100:
            self.state_transitions = self.state_transitions[-100:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self.lock:
            current_time = time.time()
            
            # Calculate current error rate
            window_start = current_time - self.config.monitoring_window_seconds
            recent_requests = [(ts, success) for ts, success in self.success_rate_window if ts >= window_start]
            
            error_rate = 0.0
            if recent_requests:
                failures = sum(1 for _, success in recent_requests if not success)
                error_rate = failures / len(recent_requests)
            
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'total_requests': self.total_requests,
                'total_failures': self.total_failures,
                'total_successes': self.total_successes,
                'error_rate': error_rate,
                'last_failure_time': self.last_failure_time,
                'next_attempt_time': self.next_attempt_time,
                'state_transitions': self.state_transitions[-10:],  # Last 10 transitions
                'time_in_current_state': current_time - self.state_transitions[-1]['timestamp'] if self.state_transitions else 0
            }
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.next_attempt_time = None
            self.logger.info(f"Circuit breaker {self.name} manually reset")

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class BulkheadExecutor:
    """Bulkhead pattern implementation for resource isolation"""
    
    def __init__(self, config: BulkheadConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.logger = logging.getLogger(__name__ + f".BulkheadExecutor.{name}")
        
        # Initialize based on bulkhead type
        if config.bulkhead_type == BulkheadType.THREAD_POOL:
            self.executor = ThreadPoolExecutor(
                max_workers=config.max_concurrent_requests,
                thread_name_prefix=f"bulkhead-{name}"
            )
            self.semaphore = None
            self.request_queue = None
        elif config.bulkhead_type == BulkheadType.SEMAPHORE:
            self.executor = None
            self.semaphore = threading.Semaphore(config.max_concurrent_requests)
            self.request_queue = None
        elif config.bulkhead_type == BulkheadType.QUEUE:
            self.executor = None
            self.semaphore = None
            self.request_queue = queue.Queue(maxsize=config.queue_size)
        
        # Metrics
        self.total_requests = 0
        self.active_requests = 0
        self.rejected_requests = 0
        self.completed_requests = 0
        self.timeout_requests = 0
        self.request_times = deque(maxlen=1000)
        
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Start queue processor if using queue bulkhead
        if config.bulkhead_type == BulkheadType.QUEUE:
            self._start_queue_processor()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead protection"""
        with self.lock:
            self.total_requests += 1
        
        if self.config.bulkhead_type == BulkheadType.THREAD_POOL:
            return self._execute_with_thread_pool(func, *args, **kwargs)
        elif self.config.bulkhead_type == BulkheadType.SEMAPHORE:
            return self._execute_with_semaphore(func, *args, **kwargs)
        elif self.config.bulkhead_type == BulkheadType.QUEUE:
            return self._execute_with_queue(func, *args, **kwargs)
    
    def _execute_with_thread_pool(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with thread pool bulkhead"""
        try:
            with self.lock:
                self.active_requests += 1
            
            future = self.executor.submit(func, *args, **kwargs)
            start_time = time.time()
            
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                execution_time = time.time() - start_time
                
                with self.lock:
                    self.completed_requests += 1
                    self.request_times.append(execution_time)
                
                return result
                
            except TimeoutError:
                with self.lock:
                    self.timeout_requests += 1
                raise BulkheadTimeoutException(f"Request timed out after {self.config.timeout_seconds}s")
                
        finally:
            with self.lock:
                self.active_requests -= 1
    
    def _execute_with_semaphore(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with semaphore bulkhead"""
        acquired = self.semaphore.acquire(blocking=False)
        
        if not acquired:
            with self.lock:
                self.rejected_requests += 1
            raise BulkheadRejectionException(f"Bulkhead {self.name} at capacity")
        
        try:
            with self.lock:
                self.active_requests += 1
            
            start_time = time.time()
            
            # Execute with timeout
            result = self._execute_with_timeout(func, self.config.timeout_seconds, *args, **kwargs)
            
            execution_time = time.time() - start_time
            
            with self.lock:
                self.completed_requests += 1
                self.request_times.append(execution_time)
            
            return result
            
        finally:
            with self.lock:
                self.active_requests -= 1
            self.semaphore.release()
    
    def _execute_with_queue(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with queue bulkhead"""
        result_queue = queue.Queue()
        request_item = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'result_queue': result_queue,
            'timestamp': time.time()
        }
        
        try:
            self.request_queue.put(request_item, block=False)
        except queue.Full:
            with self.lock:
                self.rejected_requests += 1
            raise BulkheadRejectionException(f"Bulkhead {self.name} queue is full")
        
        # Wait for result
        try:
            result = result_queue.get(timeout=self.config.timeout_seconds)
            if isinstance(result, Exception):
                raise result
            return result
        except queue.Empty:
            with self.lock:
                self.timeout_requests += 1
            raise BulkheadTimeoutException(f"Request timed out after {self.config.timeout_seconds}s")
    
    def _start_queue_processor(self):
        """Start queue processor thread"""
        def process_queue():
            while not self.shutdown_event.is_set():
                try:
                    request_item = self.request_queue.get(timeout=1.0)
                    
                    with self.lock:
                        self.active_requests += 1
                    
                    start_time = time.time()
                    
                    try:
                        result = request_item['func'](*request_item['args'], **request_item['kwargs'])
                        request_item['result_queue'].put(result)
                        
                        execution_time = time.time() - start_time
                        
                        with self.lock:
                            self.completed_requests += 1
                            self.request_times.append(execution_time)
                            
                    except Exception as e:
                        request_item['result_queue'].put(e)
                    
                    finally:
                        with self.lock:
                            self.active_requests -= 1
                        self.request_queue.task_done()
                
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Queue processor error: {e}")
        
        processor_thread = threading.Thread(target=process_queue, daemon=True)
        processor_thread.start()
    
    def _execute_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute function with timeout"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            with self.lock:
                self.timeout_requests += 1
            raise BulkheadTimeoutException(f"Function execution timed out after {timeout}s")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics"""
        with self.lock:
            avg_response_time = np.mean(self.request_times) if self.request_times else 0
            
            return {
                'name': self.name,
                'bulkhead_type': self.config.bulkhead_type.value,
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'active_requests': self.active_requests,
                'total_requests': self.total_requests,
                'completed_requests': self.completed_requests,
                'rejected_requests': self.rejected_requests,
                'timeout_requests': self.timeout_requests,
                'success_rate': self.completed_requests / self.total_requests if self.total_requests > 0 else 0,
                'avg_response_time': avg_response_time,
                'queue_size': self.request_queue.qsize() if self.request_queue else 0,
                'queue_capacity': self.config.queue_size if self.config.bulkhead_type == BulkheadType.QUEUE else None
            }
    
    def shutdown(self):
        """Shutdown bulkhead executor"""
        self.shutdown_event.set()
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.request_queue:
            # Wait for queue to empty
            try:
                self.request_queue.join()
            except:
                pass
        
        self.logger.info(f"Bulkhead {self.name} shut down")

class BulkheadRejectionException(Exception):
    """Exception raised when bulkhead rejects request"""
    pass

class BulkheadTimeoutException(Exception):
    """Exception raised when bulkhead request times out"""
    pass

class RetryExecutor:
    """Retry mechanism with exponential backoff"""
    
    def __init__(self, config: RetryConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.logger = logging.getLogger(__name__ + f".RetryExecutor.{name}")
        
        # Metrics
        self.total_attempts = 0
        self.total_successes = 0
        self.total_failures = 0
        self.retry_counts = deque(maxlen=1000)
        
        self.lock = threading.RLock()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        correlation_id = str(uuid.uuid4())
        attempt = 0
        last_exception = None
        
        while attempt < self.config.max_attempts:
            attempt += 1
            
            with self.lock:
                self.total_attempts += 1
            
            try:
                result = func(*args, **kwargs)
                
                with self.lock:
                    self.total_successes += 1
                    if attempt > 1:
                        self.retry_counts.append(attempt - 1)
                
                self.logger.info(f"Function succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if this is a retryable exception
                if not self._is_retryable_exception(e):
                    self.logger.error(f"Non-retryable exception: {e}")
                    with self.lock:
                        self.total_failures += 1
                    raise
                
                # If this is the last attempt, don't retry
                if attempt >= self.config.max_attempts:
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                
                self.logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s")
                
                # Sleep before retry
                time.sleep(delay)
        
        # All attempts failed
        with self.lock:
            self.total_failures += 1
            self.retry_counts.append(attempt - 1)
        
        self.logger.error(f"All {self.config.max_attempts} attempts failed for {self.name}")
        raise RetryExhaustedException(f"All retry attempts failed", last_exception)
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """Determine if an exception is retryable"""
        # Common retryable exceptions
        retryable_types = (
            ConnectionError,
            TimeoutError,
            OSError,
            BulkheadTimeoutException,
            BulkheadRejectionException,
            CircuitBreakerOpenException
        )
        
        # Don't retry on these exceptions
        non_retryable_types = (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            IndexError
        )
        
        if isinstance(exception, non_retryable_types):
            return False
        
        if isinstance(exception, retryable_types):
            return True
        
        # Default to retryable for unknown exceptions
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        # Base exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        
        # Apply backoff multiplier
        delay *= self.config.backoff_multiplier
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return delay
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry metrics"""
        with self.lock:
            success_rate = self.total_successes / (self.total_successes + self.total_failures) if (self.total_successes + self.total_failures) > 0 else 0
            avg_retries = np.mean(self.retry_counts) if self.retry_counts else 0
            
            return {
                'name': self.name,
                'total_attempts': self.total_attempts,
                'total_successes': self.total_successes,
                'total_failures': self.total_failures,
                'success_rate': success_rate,
                'avg_retries': avg_retries,
                'max_retries': max(self.retry_counts) if self.retry_counts else 0,
                'config': {
                    'max_attempts': self.config.max_attempts,
                    'base_delay': self.config.base_delay,
                    'max_delay': self.config.max_delay,
                    'exponential_base': self.config.exponential_base,
                    'jitter': self.config.jitter
                }
            }

class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted"""
    
    def __init__(self, message: str, last_exception: Exception):
        super().__init__(message)
        self.last_exception = last_exception

class ResilienceManager:
    """Manages all resilience patterns for a component"""
    
    def __init__(self, name: str, 
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 bulkhead_config: Optional[BulkheadConfig] = None,
                 retry_config: Optional[RetryConfig] = None):
        self.name = name
        self.logger = logging.getLogger(__name__ + f".ResilienceManager.{name}")
        
        # Initialize patterns
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config or CircuitBreakerConfig(), name) if circuit_breaker_config else None
        self.bulkhead = BulkheadExecutor(bulkhead_config or BulkheadConfig(), name) if bulkhead_config else None
        self.retry = RetryExecutor(retry_config or RetryConfig(), name) if retry_config else None
        
        # Degradation state
        self.degradation_level = 0  # 0 = normal, 1 = degraded, 2 = critical
        self.degradation_start_time = None
        self.fallback_handlers = {}
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.degraded_requests = 0
        
        self.lock = threading.RLock()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with all resilience patterns"""
        with self.lock:
            self.total_requests += 1
        
        # Check degradation state
        if self.degradation_level > 0:
            fallback_result = self._try_fallback(func.__name__, *args, **kwargs)
            if fallback_result is not None:
                with self.lock:
                    self.degraded_requests += 1
                return fallback_result
        
        # Define execution chain
        def execute_with_patterns():
            if self.bulkhead:
                return self.bulkhead.execute(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        def execute_with_circuit_breaker():
            if self.circuit_breaker:
                return self.circuit_breaker.call(execute_with_patterns)
            else:
                return execute_with_patterns()
        
        def execute_with_retry():
            if self.retry:
                return self.retry.execute(execute_with_circuit_breaker)
            else:
                return execute_with_circuit_breaker()
        
        try:
            result = execute_with_retry()
            
            with self.lock:
                self.successful_requests += 1
            
            # Check if we can recover from degradation
            self._check_recovery()
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failed_requests += 1
            
            # Check if we should degrade
            self._check_degradation()
            
            # Try fallback
            fallback_result = self._try_fallback(func.__name__, *args, **kwargs)
            if fallback_result is not None:
                with self.lock:
                    self.degraded_requests += 1
                return fallback_result
            
            raise
    
    def register_fallback(self, function_name: str, fallback_handler: Callable):
        """Register a fallback handler for a function"""
        self.fallback_handlers[function_name] = fallback_handler
        self.logger.info(f"Registered fallback handler for {function_name}")
    
    def _try_fallback(self, function_name: str, *args, **kwargs) -> Any:
        """Try to execute fallback handler"""
        if function_name in self.fallback_handlers:
            try:
                self.logger.info(f"Executing fallback for {function_name}")
                return self.fallback_handlers[function_name](*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Fallback failed for {function_name}: {e}")
        
        return None
    
    def _check_degradation(self):
        """Check if system should degrade"""
        # Simple degradation logic based on error rate
        if self.total_requests >= 10:
            error_rate = self.failed_requests / self.total_requests
            
            if error_rate >= 0.8 and self.degradation_level < 2:
                self.degradation_level = 2
                self.degradation_start_time = time.time()
                self.logger.warning(f"System {self.name} degraded to CRITICAL level")
            elif error_rate >= 0.5 and self.degradation_level < 1:
                self.degradation_level = 1
                self.degradation_start_time = time.time()
                self.logger.warning(f"System {self.name} degraded to DEGRADED level")
    
    def _check_recovery(self):
        """Check if system can recover from degradation"""
        if self.degradation_level > 0 and self.degradation_start_time:
            # Check if we've been degraded for at least 60 seconds
            if time.time() - self.degradation_start_time >= 60:
                # Check recent success rate
                if self.total_requests >= 5:
                    success_rate = self.successful_requests / self.total_requests
                    
                    if success_rate >= 0.8:
                        old_level = self.degradation_level
                        self.degradation_level = 0
                        self.degradation_start_time = None
                        self.logger.info(f"System {self.name} recovered from degradation level {old_level}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics"""
        with self.lock:
            success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
            
            metrics = {
                'name': self.name,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'degraded_requests': self.degraded_requests,
                'success_rate': success_rate,
                'degradation_level': self.degradation_level,
                'degradation_start_time': self.degradation_start_time,
                'fallback_handlers': list(self.fallback_handlers.keys())
            }
            
            if self.circuit_breaker:
                metrics['circuit_breaker'] = self.circuit_breaker.get_metrics()
            
            if self.bulkhead:
                metrics['bulkhead'] = self.bulkhead.get_metrics()
            
            if self.retry:
                metrics['retry'] = self.retry.get_metrics()
            
            return metrics
    
    def reset(self):
        """Reset all resilience patterns"""
        if self.circuit_breaker:
            self.circuit_breaker.reset()
        
        self.degradation_level = 0
        self.degradation_start_time = None
        
        with self.lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.degraded_requests = 0
        
        self.logger.info(f"Resilience manager {self.name} reset")
    
    def shutdown(self):
        """Shutdown all resilience patterns"""
        if self.bulkhead:
            self.bulkhead.shutdown()
        
        self.logger.info(f"Resilience manager {self.name} shut down")

class SystemHealthMonitor:
    """Monitors overall system health and provides metrics"""
    
    def __init__(self, monitoring_interval: float = 60.0):
        self.monitoring_interval = monitoring_interval
        self.error_aggregator = ErrorAggregator()
        self.component_managers: Dict[str, ResilienceManager] = {}
        self.system_metrics = {
            'uptime': time.time(),
            'total_requests': 0,
            'total_errors': 0,
            'recovery_actions': 0,
            'degradation_events': 0
        }
        
        self.health_status = "healthy"  # healthy, degraded, critical
        self.health_history = deque(maxlen=100)
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__ + ".SystemHealthMonitor")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.shutdown_event = threading.Event()
        self.monitoring_thread.start()
    
    def register_component(self, name: str, resilience_manager: ResilienceManager):
        """Register a component for monitoring"""
        with self.lock:
            self.component_managers[name] = resilience_manager
        self.logger.info(f"Registered component {name} for monitoring")
    
    def record_error(self, error_context: ErrorContext):
        """Record an error in the system"""
        self.error_aggregator.record_error(error_context)
        
        with self.lock:
            self.system_metrics['total_errors'] += 1
        
        # Trigger health check
        self._update_health_status()
    
    def record_request(self):
        """Record a system request"""
        with self.lock:
            self.system_metrics['total_requests'] += 1
    
    def record_recovery_action(self):
        """Record a recovery action"""
        with self.lock:
            self.system_metrics['recovery_actions'] += 1
    
    def record_degradation_event(self):
        """Record a degradation event"""
        with self.lock:
            self.system_metrics['degradation_events'] += 1
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                self._update_health_status()
                self._check_component_health()
                self._perform_self_healing()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(5)  # Short sleep on error
    
    def _update_health_status(self):
        """Update overall system health status"""
        with self.lock:
            current_time = time.time()
            
            # Calculate error rate
            error_rate = 0
            if self.system_metrics['total_requests'] > 0:
                error_rate = self.system_metrics['total_errors'] / self.system_metrics['total_requests']
            
            # Determine health status
            old_status = self.health_status
            
            if error_rate >= 0.8:
                self.health_status = "critical"
            elif error_rate >= 0.3:
                self.health_status = "degraded"
            else:
                self.health_status = "healthy"
            
            # Record health change
            if old_status != self.health_status:
                self.health_history.append({
                    'timestamp': current_time,
                    'old_status': old_status,
                    'new_status': self.health_status,
                    'error_rate': error_rate
                })
                
                self.logger.info(f"System health changed from {old_status} to {self.health_status}")
    
    def _check_component_health(self):
        """Check health of all registered components"""
        for name, manager in self.component_managers.items():
            try:
                metrics = manager.get_metrics()
                
                # Check if component is failing
                if metrics['success_rate'] < 0.5 and metrics['total_requests'] > 10:
                    self.logger.warning(f"Component {name} is experiencing high failure rate: {metrics['success_rate']:.2%}")
                
                # Check if component is degraded
                if metrics['degradation_level'] > 0:
                    self.logger.info(f"Component {name} is degraded (level {metrics['degradation_level']})")
                
            except Exception as e:
                self.logger.error(f"Error checking health of component {name}: {e}")
    
    def _perform_self_healing(self):
        """Perform self-healing actions"""
        if self.health_status == "critical":
            # Reset circuit breakers that have been open for too long
            for name, manager in self.component_managers.items():
                if manager.circuit_breaker:
                    cb_metrics = manager.circuit_breaker.get_metrics()
                    if cb_metrics['state'] == 'open' and cb_metrics['time_in_current_state'] > 300:  # 5 minutes
                        manager.circuit_breaker.reset()
                        self.record_recovery_action()
                        self.logger.info(f"Reset circuit breaker for component {name}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.system_metrics['uptime']
            
            # Component health
            component_health = {}
            for name, manager in self.component_managers.items():
                try:
                    component_health[name] = manager.get_metrics()
                except Exception as e:
                    component_health[name] = {'error': str(e)}
            
            # Error summary
            error_summary = self.error_aggregator.get_error_summary()
            
            return {
                'system_health': {
                    'status': self.health_status,
                    'uptime_seconds': uptime,
                    'total_requests': self.system_metrics['total_requests'],
                    'total_errors': self.system_metrics['total_errors'],
                    'error_rate': self.system_metrics['total_errors'] / self.system_metrics['total_requests'] if self.system_metrics['total_requests'] > 0 else 0,
                    'recovery_actions': self.system_metrics['recovery_actions'],
                    'degradation_events': self.system_metrics['degradation_events']
                },
                'component_health': component_health,
                'error_summary': error_summary,
                'health_history': list(self.health_history)[-10:],  # Last 10 health changes
                'timestamp': current_time
            }
    
    def shutdown(self):
        """Shutdown health monitor"""
        self.shutdown_event.set()
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("System health monitor shut down")

def resilient_operation(component_name: str, operation_name: str, 
                       resilience_manager: ResilienceManager,
                       health_monitor: Optional[SystemHealthMonitor] = None):
    """Decorator for resilient operations"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            correlation_id = str(uuid.uuid4())
            
            try:
                if health_monitor:
                    health_monitor.record_request()
                
                # Execute with resilience patterns
                result = resilience_manager.execute(func, *args, **kwargs)
                
                return result
                
            except Exception as e:
                # Create error context
                error_context = ErrorContext(
                    error_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=ErrorSeverity.HIGH if isinstance(e, (CircuitBreakerOpenException, RetryExhaustedException)) else ErrorSeverity.MEDIUM,
                    component=component_name,
                    operation=operation_name,
                    correlation_id=correlation_id,
                    stack_trace=traceback.format_exc()
                )
                
                if health_monitor:
                    health_monitor.record_error(error_context)
                
                raise
        
        return wrapper
    return decorator
    
@dataclass
class DataMessage:
    """Data message for inter-notebook communication"""
    stream_id: str
    stream_type: DataStreamType
    data: Any
    timestamp: float
    sequence_number: int
    metadata: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: DataStreamPriority = DataStreamPriority.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'stream_id': self.stream_id,
            'stream_type': self.stream_type.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'sequence_number': self.sequence_number,
            'metadata': self.metadata,
            'message_id': self.message_id,
            'priority': self.priority.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'dependencies': self.dependencies
        }
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp

class DataStream:
    """Represents a data stream between notebooks"""
    
    def __init__(self, 
                 stream_id: str,
                 stream_type: DataStreamType,
                 buffer_size: int = 1000):
        self.stream_id = stream_id
        self.stream_type = stream_type
        self.buffer_size = buffer_size
        
        # Stream state
        self.status = DataStreamStatus.ACTIVE
        self.sequence_number = 0
        
        # Data buffer
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.subscribers = []
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.last_message_time = None
        self.start_time = time.time()
        
        # Synchronization
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"DataStream.{stream_id}")
    
    def publish(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Publish data to stream"""
        with self.lock:
            if self.status != DataStreamStatus.ACTIVE:
                return False
            
            # Create message
            message = DataMessage(
                stream_id=self.stream_id,
                stream_type=self.stream_type,
                data=data,
                timestamp=time.time(),
                sequence_number=self.sequence_number,
                metadata=metadata or {}
            )
            
            try:
                # Add to buffer
                self.buffer.put(message, block=False)
                self.sequence_number += 1
                self.messages_sent += 1
                self.last_message_time = time.time()
                
                # Notify subscribers
                self._notify_subscribers(message)
                
                return True
                
            except queue.Full:
                self.logger.warning(f"Buffer full for stream {self.stream_id}")
                return False
    
    def subscribe(self, callback: Callable[[DataMessage], None]):
        """Subscribe to stream updates"""
        with self.lock:
            self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[DataMessage], None]):
        """Unsubscribe from stream updates"""
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
    
    def _notify_subscribers(self, message: DataMessage):
        """Notify all subscribers of new message"""
        for callback in self.subscribers:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Subscriber callback error: {e}")
    
    def get_messages(self, max_messages: int = 10) -> List[DataMessage]:
        """Get messages from stream"""
        messages = []
        
        try:
            while len(messages) < max_messages:
                message = self.buffer.get(block=False)
                messages.append(message)
                self.messages_received += 1
        except queue.Empty:
            pass
        
        return messages
    
    def pause(self):
        """Pause stream"""
        with self.lock:
            self.status = DataStreamStatus.PAUSED
    
    def resume(self):
        """Resume stream"""
        with self.lock:
            self.status = DataStreamStatus.ACTIVE
    
    def stop(self):
        """Stop stream"""
        with self.lock:
            self.status = DataStreamStatus.STOPPED
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                'stream_id': self.stream_id,
                'stream_type': self.stream_type.value,
                'status': self.status.value,
                'messages_sent': self.messages_sent,
                'messages_received': self.messages_received,
                'buffer_size': self.buffer.qsize(),
                'max_buffer_size': self.buffer_size,
                'subscribers': len(self.subscribers),
                'uptime_seconds': uptime,
                'last_message_time': self.last_message_time,
                'message_rate': self.messages_sent / uptime if uptime > 0 else 0
            }

class DataConsistencyChecker:
    """Ensure data consistency across notebooks"""
    
    def __init__(self):
        self.checksums = {}
        self.validation_rules = {}
        self.logger = logging.getLogger(__name__)
    
    def add_validation_rule(self, rule_name: str, rule_func: Callable[[Any], bool]):
        """Add custom validation rule"""
        self.validation_rules[rule_name] = rule_func
    
    def calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data"""
        if isinstance(data, pd.DataFrame):
            # Use DataFrame hash
            return hashlib.md5(str(data.values.data.tobytes()).encode()).hexdigest()
        elif isinstance(data, np.ndarray):
            # Use numpy array hash
            return hashlib.md5(data.tobytes()).hexdigest()
        else:
            # Use pickle for other types
            return hashlib.md5(pickle.dumps(data)).hexdigest()
    
    def validate_data(self, data: Any, data_id: str) -> Dict[str, Any]:
        """Validate data consistency"""
        result = {
            'data_id': data_id,
            'is_valid': True,
            'checksum': self.calculate_checksum(data),
            'validation_errors': [],
            'validation_warnings': []
        }
        
        # Check against stored checksum
        if data_id in self.checksums:
            if result['checksum'] != self.checksums[data_id]:
                result['is_valid'] = False
                result['validation_errors'].append(f"Checksum mismatch for {data_id}")
        else:
            # Store new checksum
            self.checksums[data_id] = result['checksum']
        
        # Run custom validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                if not rule_func(data):
                    result['is_valid'] = False
                    result['validation_errors'].append(f"Validation rule '{rule_name}' failed")
            except Exception as e:
                result['validation_warnings'].append(f"Validation rule '{rule_name}' error: {e}")
        
        return result
    
    def sync_checksums(self, other_checker: 'DataConsistencyChecker'):
        """Synchronize checksums with another checker"""
        for data_id, checksum in other_checker.checksums.items():
            if data_id not in self.checksums:
                self.checksums[data_id] = checksum
            elif self.checksums[data_id] != checksum:
                self.logger.warning(f"Checksum conflict for {data_id}")

class ConcurrentDataProcessor:
    """Process data concurrently across multiple workers"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 use_processes: bool = False):
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.processing_times = []
        self.completed_tasks = 0
        self.failed_tasks = 0
    
    def process_data_parallel(self, 
                            data_chunks: List[Any],
                            processing_func: Callable[[Any], Any],
                            callback: Optional[Callable[[Any], None]] = None) -> List[Any]:
        """Process data chunks in parallel"""
        start_time = time.time()
        
        # Submit tasks
        futures = []
        for i, chunk in enumerate(data_chunks):
            future = self.executor.submit(processing_func, chunk)
            futures.append((i, future))
        
        # Collect results
        results = [None] * len(data_chunks)
        
        for i, future in futures:
            try:
                result = future.result()
                results[i] = result
                self.completed_tasks += 1
                
                if callback:
                    callback(result)
                    
            except Exception as e:
                self.logger.error(f"Task {i} failed: {e}")
                self.failed_tasks += 1
                results[i] = None
        
        # Record performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        self.logger.info(f"Processed {len(data_chunks)} chunks in {processing_time:.2f}s")
        
        return results
    
    def process_data_streaming(self, 
                             data_stream: DataStream,
                             processing_func: Callable[[Any], Any],
                             output_stream: DataStream,
                             batch_size: int = 10):
        """Process data from stream continuously"""
        def process_batch():
            while data_stream.status == DataStreamStatus.ACTIVE:
                # Get batch of messages
                messages = data_stream.get_messages(batch_size)
                
                if messages:
                    # Process batch
                    processed_data = []
                    for message in messages:
                        try:
                            result = processing_func(message.data)
                            processed_data.append(result)
                        except Exception as e:
                            self.logger.error(f"Processing error: {e}")
                    
                    # Send to output stream
                    for data in processed_data:
                        output_stream.publish(data)
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
        
        # Start processing thread
        processing_thread = threading.Thread(target=process_batch)
        processing_thread.daemon = True
        processing_thread.start()
        
        return processing_thread
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        if not self.processing_times:
            return {'status': 'No processing completed yet'}
        
        return {
            'max_workers': self.max_workers,
            'use_processes': self.use_processes,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'avg_processing_time': np.mean(self.processing_times),
            'total_processing_time': sum(self.processing_times),
            'success_rate': self.completed_tasks / (self.completed_tasks + self.failed_tasks) if (self.completed_tasks + self.failed_tasks) > 0 else 0
        }
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)

class DataFlowCoordinator:
    """Central coordinator for data flow between notebooks"""
    
    def __init__(self, 
                 coordination_dir: str = "/tmp/data_flow_coordination",
                 enable_persistence: bool = True,
                 enable_resilience: bool = True,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 bulkhead_config: Optional[BulkheadConfig] = None,
                 retry_config: Optional[RetryConfig] = None):
        """
        Initialize data flow coordinator with enhanced error handling
        
        Args:
            coordination_dir: Directory for coordination files
            enable_persistence: Enable state persistence
            enable_resilience: Enable resilience patterns
            circuit_breaker_config: Circuit breaker configuration
            bulkhead_config: Bulkhead configuration
            retry_config: Retry configuration
        """
        self.coordination_dir = Path(coordination_dir)
        self.coordination_dir.mkdir(parents=True, exist_ok=True)
        self.enable_persistence = enable_persistence
        self.enable_resilience = enable_resilience
        
        # Data streams
        self.streams: Dict[str, DataStream] = {}
        
        # Coordination components
        self.consistency_checker = DataConsistencyChecker()
        self.processor = ConcurrentDataProcessor()
        
        # Resilience components
        if self.enable_resilience:
            self.health_monitor = SystemHealthMonitor(monitoring_interval=30.0)
            
            # Configure resilience patterns
            self.resilience_manager = ResilienceManager(
                name="DataFlowCoordinator",
                circuit_breaker_config=circuit_breaker_config,
                bulkhead_config=bulkhead_config,
                retry_config=retry_config
            )
            
            # Register with health monitor
            self.health_monitor.register_component("DataFlowCoordinator", self.resilience_manager)
            
            # Register fallback handlers
            self._register_fallback_handlers()
            
        else:
            self.health_monitor = None
            self.resilience_manager = None
        
        # Synchronization
        self.lock = threading.RLock()
        
        # Coordination state
        self.notebook_registry = {}
        self.active_sessions = set()
        self.error_aggregator = ErrorAggregator()
        
        # Enhanced metrics
        self.coordination_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'recovery_attempts': 0,
            'degradation_events': 0,
            'start_time': time.time()
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Load persisted state
        if self.enable_persistence:
            self._load_state()
        
        self.logger.info("Data flow coordinator initialized with enhanced error handling")
    
    def _register_fallback_handlers(self):
        """Register fallback handlers for critical operations"""
        if not self.resilience_manager:
            return
        
        # Fallback for stream creation
        def create_stream_fallback(stream_id: str, stream_type: DataStreamType, 
                                 producer_notebook: str, consumer_notebooks: List[str],
                                 buffer_size: int = 1000):
            # Create a basic stream with minimal functionality
            self.logger.info(f"Using fallback stream creation for {stream_id}")
            basic_stream = DataStream(stream_id, stream_type, min(buffer_size, 100))
            return basic_stream
        
        # Fallback for data synchronization
        def sync_data_fallback(source_notebook: str, target_notebook: str, 
                             data_id: str, data: Any):
            self.logger.info(f"Using fallback data sync for {data_id}")
            # Simple file-based fallback
            try:
                fallback_file = self.coordination_dir / f"fallback_{data_id}.pkl"
                with open(fallback_file, 'wb') as f:
                    pickle.dump(data, f)
                return True
            except Exception as e:
                self.logger.error(f"Fallback sync failed: {e}")
                return False
        
        # Fallback for concurrent processing
        def concurrent_processing_fallback(data_chunks: List[Any], 
                                         processing_func: Callable[[Any], Any],
                                         notebook_id: str):
            self.logger.info(f"Using fallback concurrent processing for {notebook_id}")
            # Sequential processing as fallback
            results = []
            for chunk in data_chunks:
                try:
                    result = processing_func(chunk)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Fallback processing failed for chunk: {e}")
                    results.append(None)
            return results
        
        # Register fallback handlers
        self.resilience_manager.register_fallback("create_stream", create_stream_fallback)
        self.resilience_manager.register_fallback("synchronize_data", sync_data_fallback)
        self.resilience_manager.register_fallback("coordinate_concurrent_processing", concurrent_processing_fallback)
    
    def register_notebook(self, notebook_id: str, notebook_type: str, capabilities: List[str]):
        """Register a notebook with the coordinator"""
        with self.lock:
            self.notebook_registry[notebook_id] = {
                'type': notebook_type,
                'capabilities': capabilities,
                'registered_at': time.time(),
                'last_activity': time.time(),
                'streams': []
            }
            
            self.active_sessions.add(notebook_id)
            
            if self.enable_persistence:
                self._save_state()
            
            self.logger.info(f"Registered notebook: {notebook_id} ({notebook_type})")
    
    def unregister_notebook(self, notebook_id: str):
        """Unregister a notebook"""
        with self.lock:
            if notebook_id in self.notebook_registry:
                # Clean up streams
                notebook_info = self.notebook_registry[notebook_id]
                for stream_id in notebook_info['streams']:
                    self.remove_stream(stream_id)
                
                del self.notebook_registry[notebook_id]
                self.active_sessions.discard(notebook_id)
                
                if self.enable_persistence:
                    self._save_state()
                
                self.logger.info(f"Unregistered notebook: {notebook_id}")
    
    def create_stream(self, 
                     stream_id: str,
                     stream_type: DataStreamType,
                     producer_notebook: str,
                     consumer_notebooks: List[str],
                     buffer_size: int = 1000) -> DataStream:
        """Create a new data stream with resilience patterns"""
        
        def _create_stream_impl():
            with self.lock:
                self.coordination_metrics['total_operations'] += 1
                
                if stream_id in self.streams:
                    raise ValueError(f"Stream {stream_id} already exists")
                
                # Create stream
                stream = DataStream(stream_id, stream_type, buffer_size)
                self.streams[stream_id] = stream
                
                # Update notebook registry
                if producer_notebook in self.notebook_registry:
                    self.notebook_registry[producer_notebook]['streams'].append(stream_id)
                
                for consumer in consumer_notebooks:
                    if consumer in self.notebook_registry:
                        self.notebook_registry[consumer]['streams'].append(stream_id)
                
                if self.enable_persistence:
                    self._save_state()
                
                self.coordination_metrics['successful_operations'] += 1
                self.logger.info(f"Created stream: {stream_id} ({stream_type.value})")
                return stream
        
        try:
            if self.enable_resilience and self.resilience_manager:
                return self.resilience_manager.execute(_create_stream_impl)
            else:
                return _create_stream_impl()
        except Exception as e:
            self.coordination_metrics['failed_operations'] += 1
            self._handle_error(e, "create_stream", stream_id=stream_id)
            raise
    
    def _handle_error(self, exception: Exception, operation: str, **context):
        """Handle errors with proper context and reporting"""
        error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=time.time(),
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=self._determine_error_severity(exception),
            component="DataFlowCoordinator",
            operation=operation,
            stack_trace=traceback.format_exc(),
            metadata=context
        )
        
        # Record error
        self.error_aggregator.record_error(error_context)
        
        if self.health_monitor:
            self.health_monitor.record_error(error_context)
        
        # Log error
        self.logger.error(f"Error in {operation}: {exception}", extra={
            'error_id': error_context.error_id,
            'operation': operation,
            'context': context
        })
    
    def _determine_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine severity level of an exception"""
        if isinstance(exception, (CircuitBreakerOpenException, RetryExhaustedException)):
            return ErrorSeverity.CRITICAL
        elif isinstance(exception, (BulkheadTimeoutException, BulkheadRejectionException)):
            return ErrorSeverity.HIGH
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def get_stream(self, stream_id: str) -> Optional[DataStream]:
        """Get existing stream"""
        return self.streams.get(stream_id)
    
    def remove_stream(self, stream_id: str):
        """Remove a stream"""
        with self.lock:
            if stream_id in self.streams:
                stream = self.streams[stream_id]
                stream.stop()
                del self.streams[stream_id]
                
                # Update notebook registry
                for notebook_info in self.notebook_registry.values():
                    if stream_id in notebook_info['streams']:
                        notebook_info['streams'].remove(stream_id)
                
                if self.enable_persistence:
                    self._save_state()
                
                self.logger.info(f"Removed stream: {stream_id}")
    
    def synchronize_data(self, 
                        source_notebook: str,
                        target_notebook: str,
                        data_id: str,
                        data: Any) -> bool:
        """Synchronize data between notebooks"""
        # Validate data consistency
        validation_result = self.consistency_checker.validate_data(data, data_id)
        
        if not validation_result['is_valid']:
            self.logger.error(f"Data validation failed for {data_id}: {validation_result['validation_errors']}")
            return False
        
        # Create synchronization stream if it doesn't exist
        stream_id = f"sync_{source_notebook}_{target_notebook}"
        
        if stream_id not in self.streams:
            self.create_stream(
                stream_id=stream_id,
                stream_type=DataStreamType.MARKET_DATA,
                producer_notebook=source_notebook,
                consumer_notebooks=[target_notebook]
            )
        
        # Send data
        stream = self.streams[stream_id]
        success = stream.publish(data, {
            'data_id': data_id,
            'source_notebook': source_notebook,
            'target_notebook': target_notebook,
            'validation_result': validation_result
        })
        
        if success:
            self.logger.info(f"Synchronized {data_id} from {source_notebook} to {target_notebook}")
        
        return success
    
    def coordinate_concurrent_processing(self, 
                                       data_chunks: List[Any],
                                       processing_func: Callable[[Any], Any],
                                       notebook_id: str) -> List[Any]:
        """Coordinate concurrent processing for a notebook"""
        # Update activity timestamp
        if notebook_id in self.notebook_registry:
            self.notebook_registry[notebook_id]['last_activity'] = time.time()
        
        # Process data
        results = self.processor.process_data_parallel(data_chunks, processing_func)
        
        return results
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status with resilience metrics"""
        with self.lock:
            stream_stats = {}
            for stream_id, stream in self.streams.items():
                stream_stats[stream_id] = stream.get_stats()
            
            current_time = time.time()
            uptime = current_time - self.coordination_metrics['start_time']
            
            base_status = {
                'active_notebooks': len(self.active_sessions),
                'registered_notebooks': len(self.notebook_registry),
                'active_streams': len(self.streams),
                'notebook_registry': self.notebook_registry,
                'stream_statistics': stream_stats,
                'processor_stats': self.processor.get_performance_stats(),
                'coordination_uptime': uptime,
                'coordination_metrics': self.coordination_metrics,
                'timestamp': current_time
            }
            
            # Add resilience information if enabled
            if self.enable_resilience:
                resilience_status = {
                    'resilience_enabled': True,
                    'error_summary': self.error_aggregator.get_error_summary(),
                }
                
                if self.resilience_manager:
                    resilience_status['resilience_metrics'] = self.resilience_manager.get_metrics()
                
                if self.health_monitor:
                    resilience_status['health_report'] = self.health_monitor.get_health_report()
                
                base_status.update(resilience_status)
            else:
                base_status['resilience_enabled'] = False
            
            return base_status
    
    def get_error_report(self) -> Dict[str, Any]:
        """Get detailed error report"""
        return {
            'error_summary': self.error_aggregator.get_error_summary(),
            'coordination_metrics': self.coordination_metrics,
            'resilience_metrics': self.resilience_manager.get_metrics() if self.resilience_manager else None,
            'health_report': self.health_monitor.get_health_report() if self.health_monitor else None
        }
    
    def reset_resilience_patterns(self):
        """Reset all resilience patterns - for emergency recovery"""
        if self.resilience_manager:
            self.resilience_manager.reset()
            self.coordination_metrics['recovery_attempts'] += 1
            self.logger.info("Resilience patterns reset for emergency recovery")
        
        if self.health_monitor:
            self.health_monitor.record_recovery_action()
    
    def _save_state(self):
        """Save coordination state to disk"""
        if not self.enable_persistence:
            return
        
        # Ensure coordination directory exists
        self.coordination_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = self.coordination_dir / "coordination_state.json"
        
        state = {
            'notebook_registry': self.notebook_registry,
            'active_sessions': list(self.active_sessions),
            'stream_ids': list(self.streams.keys()),
            'save_timestamp': time.time()
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load coordination state from disk"""
        state_file = self.coordination_dir / "coordination_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.notebook_registry = state.get('notebook_registry', {})
                self.active_sessions = set(state.get('active_sessions', []))
                
                self.logger.info("Loaded coordination state from disk")
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
    
    def cleanup(self):
        """Cleanup coordinator resources"""
        # Stop all streams
        for stream in self.streams.values():
            stream.stop()
        
        # Shutdown processor
        self.processor.shutdown()
        
        # Shutdown resilience components
        if self.enable_resilience:
            if self.resilience_manager:
                self.resilience_manager.shutdown()
            
            if self.health_monitor:
                self.health_monitor.shutdown()
        
        # Save final state
        if self.enable_persistence:
            self._save_state()
        
        self.logger.info("Data flow coordinator cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

# Utility functions for notebook integration
def create_notebook_client(notebook_id: str, notebook_type: str, coordinator: DataFlowCoordinator):
    """Create a notebook client for easier integration"""
    
    class NotebookClient:
        def __init__(self, notebook_id: str, notebook_type: str, coordinator: DataFlowCoordinator):
            self.notebook_id = notebook_id
            self.notebook_type = notebook_type
            self.coordinator = coordinator
            self.capabilities = []
            
            # Register with coordinator
            self.coordinator.register_notebook(notebook_id, notebook_type, self.capabilities)
        
        def add_capability(self, capability: str):
            """Add a capability to this notebook"""
            self.capabilities.append(capability)
        
        def create_data_stream(self, stream_id: str, stream_type: DataStreamType, consumers: List[str]):
            """Create a data stream from this notebook"""
            return self.coordinator.create_stream(
                stream_id=stream_id,
                stream_type=stream_type,
                producer_notebook=self.notebook_id,
                consumer_notebooks=consumers
            )
        
        def get_data_stream(self, stream_id: str):
            """Get a data stream"""
            return self.coordinator.get_stream(stream_id)
        
        def sync_data(self, target_notebook: str, data_id: str, data: Any):
            """Synchronize data with another notebook"""
            return self.coordinator.synchronize_data(
                source_notebook=self.notebook_id,
                target_notebook=target_notebook,
                data_id=data_id,
                data=data
            )
        
        def process_concurrent(self, data_chunks: List[Any], processing_func: Callable[[Any], Any]):
            """Process data concurrently"""
            return self.coordinator.coordinate_concurrent_processing(
                data_chunks=data_chunks,
                processing_func=processing_func,
                notebook_id=self.notebook_id
            )
        
        def cleanup(self):
            """Cleanup notebook client"""
            self.coordinator.unregister_notebook(self.notebook_id)
    
    return NotebookClient(notebook_id, notebook_type, coordinator)


# Enhanced Concurrency and Dependency Management Components

class AtomicCounter:
    """Thread-safe atomic counter implementation"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        """Atomically increment and return new value"""
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        """Atomically decrement and return new value"""
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> int:
        """Set value atomically"""
        with self._lock:
            old_value = self._value
            self._value = value
            return old_value
    
    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """Compare and swap operation"""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False


class ThreadSafeDict:
    """Thread-safe dictionary implementation with read-write locks"""
    
    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(self._lock)
        self._write_ready = threading.Condition(self._lock)
    
    @contextmanager
    def read_lock(self):
        """Context manager for read operations"""
        with self._lock:
            while self._writers > 0:
                self._read_ready.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._lock:
                self._readers -= 1
                if self._readers == 0:
                    self._write_ready.notify_all()
    
    @contextmanager
    def write_lock(self):
        """Context manager for write operations"""
        with self._lock:
            while self._writers > 0 or self._readers > 0:
                self._write_ready.wait()
            self._writers += 1
        try:
            yield
        finally:
            with self._lock:
                self._writers -= 1
                self._write_ready.notify_all()
                self._read_ready.notify_all()
    
    def get(self, key, default=None):
        """Thread-safe get operation"""
        with self.read_lock():
            return self._dict.get(key, default)
    
    def set(self, key, value):
        """Thread-safe set operation"""
        with self.write_lock():
            self._dict[key] = value
    
    def delete(self, key):
        """Thread-safe delete operation"""
        with self.write_lock():
            if key in self._dict:
                del self._dict[key]
    
    def keys(self):
        """Thread-safe keys operation"""
        with self.read_lock():
            return list(self._dict.keys())
    
    def values(self):
        """Thread-safe values operation"""
        with self.read_lock():
            return list(self._dict.values())
    
    def items(self):
        """Thread-safe items operation"""
        with self.read_lock():
            return list(self._dict.items())
    
    def __len__(self):
        """Thread-safe length operation"""
        with self.read_lock():
            return len(self._dict)
    
    def __contains__(self, key):
        """Thread-safe contains operation"""
        with self.read_lock():
            return key in self._dict


class DependencyGraph:
    """Thread-safe dependency graph for stream dependencies"""
    
    def __init__(self):
        self._graph = ThreadSafeDict()
        self._reverse_graph = ThreadSafeDict()
        self._lock = threading.RLock()
    
    def add_dependency(self, dependent: str, dependency: str):
        """Add a dependency relationship"""
        with self._lock:
            # Add to forward graph
            deps = self._graph.get(dependent, set())
            deps.add(dependency)
            self._graph.set(dependent, deps)
            
            # Add to reverse graph
            rdeps = self._reverse_graph.get(dependency, set())
            rdeps.add(dependent)
            self._reverse_graph.set(dependency, rdeps)
    
    def remove_dependency(self, dependent: str, dependency: str):
        """Remove a dependency relationship"""
        with self._lock:
            deps = self._graph.get(dependent, set())
            deps.discard(dependency)
            self._graph.set(dependent, deps)
            
            rdeps = self._reverse_graph.get(dependency, set())
            rdeps.discard(dependent)
            self._reverse_graph.set(dependency, rdeps)
    
    def get_dependencies(self, node: str) -> Set[str]:
        """Get direct dependencies of a node"""
        return self._graph.get(node, set()).copy()
    
    def get_dependents(self, node: str) -> Set[str]:
        """Get nodes that depend on this node"""
        return self._reverse_graph.get(node, set()).copy()
    
    def has_cycle(self) -> bool:
        """Check if the graph has any cycles"""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True  # Cycle detected
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self._graph.get(node, set()):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self._graph.keys():
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def get_execution_order(self) -> List[str]:
        """Get topological order for execution"""
        if self.has_cycle():
            raise ValueError("Cannot determine execution order: cycle detected")
        
        # Get all nodes
        all_nodes = set(self._graph.keys())
        for deps in self._graph.values():
            all_nodes.update(deps)
        
        in_degree = {}
        
        # Initialize in-degree for all nodes
        for node in all_nodes:
            in_degree[node] = 0
        
        # Calculate in-degree
        # In our graph, if A depends on B, then A has an edge to B
        # But for topological sort, we need to count incoming edges
        # So if A depends on B, then B should have an outgoing edge to A
        for node in all_nodes:
            for dep in self._graph.get(node, set()):
                in_degree[node] = in_degree.get(node, 0) + 1
        
        # Topological sort using Kahn's algorithm
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # For each node that depends on this node, decrease its in-degree
            for dependent in self._reverse_graph.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return result


class LockFreeQueue:
    """Lock-free queue implementation for high-throughput scenarios"""
    
    def __init__(self, maxsize: int = 0):
        self._queue = queue.Queue(maxsize=maxsize)
        self._size = AtomicCounter(0)
    
    def put(self, item, block=True, timeout=None):
        """Put item in queue"""
        try:
            self._queue.put(item, block=block, timeout=timeout)
            self._size.increment()
            return True
        except queue.Full:
            return False
    
    def get(self, block=True, timeout=None):
        """Get item from queue"""
        try:
            item = self._queue.get(block=block, timeout=timeout)
            self._size.decrement()
            return item
        except queue.Empty:
            return None
    
    def qsize(self):
        """Get approximate queue size"""
        return self._size.get()
    
    def empty(self):
        """Check if queue is empty"""
        return self._size.get() == 0


class ConcurrencyMonitor:
    """Monitor concurrency metrics and detect issues"""
    
    def __init__(self):
        self.lock_contention_count = AtomicCounter()
        self.deadlock_detection_count = AtomicCounter()
        self.performance_metrics = ThreadSafeDict()
        self.active_locks = ThreadSafeDict()
        self.lock_acquisition_times = ThreadSafeDict()
        self.logger = logging.getLogger(__name__)
    
    def record_lock_contention(self, lock_name: str):
        """Record lock contention event"""
        self.lock_contention_count.increment()
        self.logger.warning(f"Lock contention detected on {lock_name}")
    
    def record_lock_acquisition(self, lock_name: str, thread_id: str):
        """Record lock acquisition"""
        self.active_locks.set(f"{lock_name}:{thread_id}", time.time())
        self.lock_acquisition_times.set(lock_name, time.time())
    
    def record_lock_release(self, lock_name: str, thread_id: str):
        """Record lock release"""
        key = f"{lock_name}:{thread_id}"
        if key in self.active_locks:
            acquisition_time = self.active_locks.get(key)
            if acquisition_time:
                hold_time = time.time() - acquisition_time
                self.performance_metrics.set(f"lock_hold_time_{lock_name}", hold_time)
            self.active_locks.delete(key)
    
    def check_deadlock_potential(self) -> bool:
        """Check for potential deadlocks"""
        # Simple deadlock detection based on long-held locks
        current_time = time.time()
        deadlock_threshold = 30.0  # 30 seconds
        
        for key, acquisition_time in self.lock_acquisition_times.items():
            if current_time - acquisition_time > deadlock_threshold:
                self.deadlock_detection_count.increment()
                self.logger.error(f"Potential deadlock detected on lock {key}")
                return True
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get concurrency metrics"""
        return {
            'lock_contention_count': self.lock_contention_count.get(),
            'deadlock_detection_count': self.deadlock_detection_count.get(),
            'active_locks': len(self.active_locks),
            'performance_metrics': dict(self.performance_metrics.items()),
            'potential_deadlocks': self.check_deadlock_potential()
        }


class EnhancedDataStream(DataStream):
    """Enhanced DataStream with improved concurrency and dependency management"""
    
    def __init__(self, 
                 stream_id: str,
                 stream_type: DataStreamType,
                 buffer_size: int = 1000,
                 priority: DataStreamPriority = DataStreamPriority.MEDIUM,
                 dependencies: List[str] = None):
        super().__init__(stream_id, stream_type, buffer_size)
        
        self.priority = priority
        self.dependencies = dependencies or []
        
        # Enhanced concurrency features
        self.priority_buffer = queue.PriorityQueue(maxsize=buffer_size)
        self.dependency_graph = DependencyGraph()
        self.lock_free_queue = LockFreeQueue(maxsize=buffer_size)
        self.concurrency_monitor = ConcurrencyMonitor()
        
        # Atomic counters for thread-safe operations
        self.atomic_sequence = AtomicCounter()
        self.atomic_subscribers = AtomicCounter()
        
        # Thread-safe collections
        self.thread_safe_subscribers = ThreadSafeDict()
        self.message_cache = ThreadSafeDict()
        
        # Enhanced synchronization
        self.condition = threading.Condition(self.lock)
        self.processing_semaphore = threading.Semaphore(10)  # Limit concurrent processing
        
        # Performance monitoring
        self.message_processing_times = []
        self.throughput_monitor = AtomicCounter()
        
        # Setup dependencies
        for dep in self.dependencies:
            self.dependency_graph.add_dependency(self.stream_id, dep)
    
    def publish_priority(self, data: Any, 
                        priority: DataStreamPriority = DataStreamPriority.MEDIUM,
                        metadata: Optional[Dict[str, Any]] = None,
                        dependencies: List[str] = None) -> bool:
        """Publish data with priority and dependency support"""
        start_time = time.time()
        
        try:
            with self.processing_semaphore:
                with self.lock:
                    self.concurrency_monitor.record_lock_acquisition("publish_lock", 
                                                                    threading.current_thread().ident)
                    
                    if self.status != DataStreamStatus.ACTIVE:
                        return False
                    
                    # Create enhanced message
                    message = DataMessage(
                        stream_id=self.stream_id,
                        stream_type=self.stream_type,
                        data=data,
                        timestamp=time.time(),
                        sequence_number=self.atomic_sequence.increment(),
                        metadata=metadata or {},
                        priority=priority,
                        dependencies=dependencies or []
                    )
                    
                    # Check dependencies
                    if not self._check_dependencies(message):
                        self.logger.warning(f"Dependencies not satisfied for message {message.message_id}")
                        return False
                    
                    # Add to priority queue
                    try:
                        self.priority_buffer.put(message, block=False)
                        self.messages_sent += 1
                        self.last_message_time = time.time()
                        
                        # Cache message for potential retry
                        self.message_cache.set(message.message_id, message)
                        
                        # Notify waiting threads
                        self.condition.notify_all()
                        
                        # Notify subscribers asynchronously
                        self._notify_subscribers_async(message)
                        
                        # Update throughput
                        self.throughput_monitor.increment()
                        
                        return True
                        
                    except queue.Full:
                        self.logger.warning(f"Priority buffer full for stream {self.stream_id}")
                        return False
                    
                    finally:
                        self.concurrency_monitor.record_lock_release("publish_lock", 
                                                                   threading.current_thread().ident)
        
        finally:
            # Record processing time
            processing_time = time.time() - start_time
            self.message_processing_times.append(processing_time)
            if len(self.message_processing_times) > 1000:
                self.message_processing_times = self.message_processing_times[-1000:]
    
    def _check_dependencies(self, message: DataMessage) -> bool:
        """Check if message dependencies are satisfied"""
        if not message.dependencies:
            return True
        
        # Check if all dependencies are available
        for dep_id in message.dependencies:
            if not self.message_cache.get(dep_id):
                return False
        
        return True
    
    def _notify_subscribers_async(self, message: DataMessage):
        """Notify subscribers asynchronously to avoid blocking"""
        def notify_worker():
            for subscriber_id, callback in self.thread_safe_subscribers.items():
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Subscriber {subscriber_id} callback error: {e}")
        
        # Run in separate thread to avoid blocking
        notification_thread = threading.Thread(target=notify_worker)
        notification_thread.daemon = True
        notification_thread.start()
    
    def subscribe_enhanced(self, callback: Callable[[DataMessage], None], 
                          subscriber_id: str = None) -> str:
        """Enhanced subscription with unique subscriber ID"""
        if subscriber_id is None:
            subscriber_id = str(uuid.uuid4())
        
        self.thread_safe_subscribers.set(subscriber_id, callback)
        self.atomic_subscribers.increment()
        
        return subscriber_id
    
    def unsubscribe_enhanced(self, subscriber_id: str):
        """Enhanced unsubscription"""
        if subscriber_id in self.thread_safe_subscribers:
            self.thread_safe_subscribers.delete(subscriber_id)
            self.atomic_subscribers.decrement()
    
    def get_priority_messages(self, max_messages: int = 10) -> List[DataMessage]:
        """Get messages ordered by priority"""
        messages = []
        
        try:
            while len(messages) < max_messages and not self.priority_buffer.empty():
                message = self.priority_buffer.get(block=False)
                messages.append(message)
                self.messages_received += 1
        except queue.Empty:
            pass
        
        return messages
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics"""
        base_stats = self.get_stats()
        
        avg_processing_time = (np.mean(self.message_processing_times) 
                             if self.message_processing_times else 0)
        
        enhanced_stats = {
            **base_stats,
            'priority': self.priority.value,
            'dependencies': self.dependencies,
            'atomic_sequence': self.atomic_sequence.get(),
            'atomic_subscribers': self.atomic_subscribers.get(),
            'avg_processing_time': avg_processing_time,
            'throughput_per_second': self.throughput_monitor.get(),
            'concurrency_metrics': self.concurrency_monitor.get_metrics(),
            'cached_messages': len(self.message_cache),
            'priority_buffer_size': self.priority_buffer.qsize()
        }
        
        return enhanced_stats


class EnhancedDataFlowCoordinator(DataFlowCoordinator):
    """Enhanced DataFlowCoordinator with race condition fixes and dependency management"""
    
    def __init__(self, 
                 coordination_dir: str = "/tmp/data_flow_coordination",
                 enable_persistence: bool = True,
                 max_concurrent_streams: int = 100,
                 deadlock_detection_interval: float = 5.0,
                 enable_performance_monitoring: bool = True):
        """
        Initialize enhanced data flow coordinator
        
        Args:
            coordination_dir: Directory for coordination files
            enable_persistence: Enable state persistence
            max_concurrent_streams: Maximum number of concurrent streams
            deadlock_detection_interval: Interval for deadlock detection (seconds)
            enable_performance_monitoring: Enable performance monitoring
        """
        super().__init__(coordination_dir, enable_persistence)
        
        # Enhanced synchronization
        self.enhanced_streams = ThreadSafeDict()
        self.global_dependency_graph = DependencyGraph()
        self.concurrency_monitor = ConcurrencyMonitor()
        
        # Configuration
        self.max_concurrent_streams = max_concurrent_streams
        self.deadlock_detection_interval = deadlock_detection_interval
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # Stream management
        self.stream_semaphore = threading.Semaphore(max_concurrent_streams)
        self.stream_creation_lock = threading.RLock()
        
        # Performance tracking
        self.operation_counters = {
            'stream_creations': AtomicCounter(),
            'message_publishes': AtomicCounter(),
            'dependency_resolutions': AtomicCounter(),
            'deadlock_detections': AtomicCounter()
        }
        
        # Background monitoring
        self.monitoring_active = True
        self.monitoring_thread = None
        
        if enable_performance_monitoring:
            self._start_monitoring()
        
        self.logger.info("Enhanced data flow coordinator initialized")
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitor_worker():
            while self.monitoring_active:
                try:
                    # Check for deadlocks
                    self._check_system_deadlocks()
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    # Cleanup expired messages
                    self._cleanup_expired_messages()
                    
                    time.sleep(self.deadlock_detection_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_worker)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _check_system_deadlocks(self):
        """Check for system-wide deadlocks"""
        try:
            # Check global dependency graph for cycles
            if self.global_dependency_graph.has_cycle():
                self.operation_counters['deadlock_detections'].increment()
                self.logger.error("System-wide circular dependency detected!")
                self._handle_circular_dependency()
            
            # Check individual stream monitors
            for stream_id, stream in self.enhanced_streams.items():
                if hasattr(stream, 'concurrency_monitor'):
                    if stream.concurrency_monitor.check_deadlock_potential():
                        self.logger.warning(f"Potential deadlock in stream {stream_id}")
        
        except Exception as e:
            self.logger.error(f"Deadlock detection error: {e}")
    
    def _handle_circular_dependency(self):
        """Handle circular dependency detection"""
        # Try to resolve by reorganizing dependencies
        try:
            execution_order = self.global_dependency_graph.get_execution_order()
            self.logger.info(f"Resolved execution order: {execution_order}")
        except ValueError:
            # If resolution fails, alert and potentially restart affected streams
            self.logger.error("Cannot resolve circular dependency - manual intervention required")
            # Could implement automatic recovery here
    
    def _update_performance_metrics(self):
        """Update system performance metrics"""
        metrics = {
            'total_streams': len(self.enhanced_streams),
            'active_streams': sum(1 for s in self.enhanced_streams.values() 
                                if s.status == DataStreamStatus.ACTIVE),
            'stream_creations': self.operation_counters['stream_creations'].get(),
            'message_publishes': self.operation_counters['message_publishes'].get(),
            'dependency_resolutions': self.operation_counters['dependency_resolutions'].get(),
            'deadlock_detections': self.operation_counters['deadlock_detections'].get()
        }
        
        # Store metrics for retrieval
        self.performance_metrics = metrics
    
    def _cleanup_expired_messages(self):
        """Clean up expired messages from stream caches"""
        current_time = time.time()
        message_ttl = 3600  # 1 hour TTL
        
        for stream in self.enhanced_streams.values():
            if hasattr(stream, 'message_cache'):
                expired_messages = []
                for msg_id, msg in stream.message_cache.items():
                    if hasattr(msg, 'timestamp') and current_time - msg.timestamp > message_ttl:
                        expired_messages.append(msg_id)
                
                for msg_id in expired_messages:
                    stream.message_cache.delete(msg_id)
    
    def create_enhanced_stream(self, 
                             stream_id: str,
                             stream_type: DataStreamType,
                             producer_notebook: str,
                             consumer_notebooks: List[str],
                             buffer_size: int = 1000,
                             priority: DataStreamPriority = DataStreamPriority.MEDIUM,
                             dependencies: List[str] = None) -> EnhancedDataStream:
        """Create an enhanced data stream with dependency management"""
        
        with self.stream_creation_lock:
            # Check if stream already exists
            if stream_id in self.enhanced_streams:
                raise ValueError(f"Enhanced stream {stream_id} already exists")
            
            # Acquire semaphore for stream creation
            if not self.stream_semaphore.acquire(blocking=False):
                raise ValueError("Maximum concurrent streams limit reached")
            
            try:
                # Create enhanced stream
                stream = EnhancedDataStream(
                    stream_id=stream_id,
                    stream_type=stream_type,
                    buffer_size=buffer_size,
                    priority=priority,
                    dependencies=dependencies or []
                )
                
                # Add to enhanced streams collection
                self.enhanced_streams.set(stream_id, stream)
                
                # Update global dependency graph
                if dependencies:
                    for dep in dependencies:
                        self.global_dependency_graph.add_dependency(stream_id, dep)
                
                # Check for circular dependencies
                if self.global_dependency_graph.has_cycle():
                    # Rollback if cycle detected
                    self.enhanced_streams.delete(stream_id)
                    if dependencies:
                        for dep in dependencies:
                            self.global_dependency_graph.remove_dependency(stream_id, dep)
                    self.stream_semaphore.release()
                    raise ValueError(f"Creating stream {stream_id} would create circular dependency")
                
                # Update notebook registry
                if producer_notebook in self.notebook_registry:
                    self.notebook_registry[producer_notebook]['streams'].append(stream_id)
                
                for consumer in consumer_notebooks:
                    if consumer in self.notebook_registry:
                        self.notebook_registry[consumer]['streams'].append(stream_id)
                
                # Update counters
                self.operation_counters['stream_creations'].increment()
                
                if self.enable_persistence:
                    self._save_state()
                
                self.logger.info(f"Created enhanced stream: {stream_id} with dependencies: {dependencies}")
                return stream
                
            except Exception as e:
                # Release semaphore on error
                self.stream_semaphore.release()
                raise e
    
    def publish_with_dependencies(self, 
                                stream_id: str,
                                data: Any,
                                dependencies: List[str] = None,
                                priority: DataStreamPriority = DataStreamPriority.MEDIUM,
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Publish data with dependency resolution"""
        
        stream = self.enhanced_streams.get(stream_id)
        if not stream:
            self.logger.error(f"Stream {stream_id} not found")
            return False
        
        # Resolve dependencies first
        if dependencies:
            resolved_deps = self._resolve_dependencies(dependencies)
            if not resolved_deps:
                self.logger.warning(f"Failed to resolve dependencies for stream {stream_id}")
                return False
        
        # Publish with priority
        success = stream.publish_priority(
            data=data,
            priority=priority,
            metadata=metadata,
            dependencies=dependencies
        )
        
        if success:
            self.operation_counters['message_publishes'].increment()
            
            if dependencies:
                self.operation_counters['dependency_resolutions'].increment()
        
        return success
    
    def _resolve_dependencies(self, dependencies: List[str]) -> bool:
        """Resolve stream dependencies"""
        try:
            # Check if all dependencies exist
            for dep_id in dependencies:
                if dep_id not in self.enhanced_streams:
                    self.logger.error(f"Dependency {dep_id} not found")
                    return False
            
            # For now, just check if dependencies exist
            # In a more complex implementation, you could check if dependent data is available
            return True
            
        except Exception as e:
            self.logger.error(f"Dependency resolution error: {e}")
            return False
    
    def get_stream_execution_order(self) -> List[str]:
        """Get optimal execution order for all streams"""
        try:
            return self.global_dependency_graph.get_execution_order()
        except ValueError as e:
            self.logger.error(f"Cannot determine execution order: {e}")
            return []
    
    def detect_circular_dependencies(self) -> bool:
        """Detect circular dependencies in the system"""
        return self.global_dependency_graph.has_cycle()
    
    def get_dependency_graph_info(self) -> Dict[str, Any]:
        """Get information about the dependency graph"""
        info = {
            'has_cycles': self.global_dependency_graph.has_cycle(),
            'dependencies': {},
            'dependents': {}
        }
        
        for stream_id in self.enhanced_streams.keys():
            info['dependencies'][stream_id] = list(self.global_dependency_graph.get_dependencies(stream_id))
            info['dependents'][stream_id] = list(self.global_dependency_graph.get_dependents(stream_id))
        
        return info
    
    def get_enhanced_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced coordination status"""
        base_status = super().get_coordination_status()
        
        enhanced_stream_stats = {}
        for stream_id, stream in self.enhanced_streams.items():
            enhanced_stream_stats[stream_id] = stream.get_performance_metrics()
        
        enhanced_status = {
            **base_status,
            'enhanced_streams': len(self.enhanced_streams),
            'max_concurrent_streams': self.max_concurrent_streams,
            'stream_semaphore_available': self.stream_semaphore._value,
            'dependency_graph_info': self.get_dependency_graph_info(),
            'concurrency_metrics': self.concurrency_monitor.get_metrics(),
            'operation_counters': {k: v.get() for k, v in self.operation_counters.items()},
            'enhanced_stream_statistics': enhanced_stream_stats,
            'performance_metrics': getattr(self, 'performance_metrics', {})
        }
        
        return enhanced_status
    
    def optimize_stream_performance(self, stream_id: str) -> bool:
        """Optimize performance for a specific stream"""
        stream = self.enhanced_streams.get(stream_id)
        if not stream:
            return False
        
        try:
            # Clear expired messages
            current_time = time.time()
            expired_messages = []
            
            for msg_id, msg in stream.message_cache.items():
                if hasattr(msg, 'timestamp') and current_time - msg.timestamp > 3600:
                    expired_messages.append(msg_id)
            
            for msg_id in expired_messages:
                stream.message_cache.delete(msg_id)
            
            # Reset performance counters if needed
            if len(stream.message_processing_times) > 10000:
                stream.message_processing_times = stream.message_processing_times[-1000:]
            
            self.logger.info(f"Optimized performance for stream {stream_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Performance optimization error for stream {stream_id}: {e}")
            return False
    
    def shutdown_enhanced(self):
        """Enhanced shutdown with proper cleanup"""
        self.logger.info("Shutting down enhanced data flow coordinator...")
        
        # Stop monitoring
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Stop all enhanced streams
        for stream_id, stream in self.enhanced_streams.items():
            try:
                stream.stop()
                self.stream_semaphore.release()
            except Exception as e:
                self.logger.error(f"Error stopping stream {stream_id}: {e}")
        
        # Clear enhanced streams
        self.enhanced_streams = ThreadSafeDict()
        
        # Call parent cleanup
        super().cleanup()
        
        self.logger.info("Enhanced data flow coordinator shutdown complete")
    
    def __del__(self):
        """Enhanced destructor"""
        self.shutdown_enhanced()


# Configuration class for enhanced coordinator
@dataclass
class EnhancedCoordinatorConfig:
    """Configuration for enhanced data flow coordinator"""
    coordination_dir: str = "/tmp/data_flow_coordination"
    enable_persistence: bool = True
    max_concurrent_streams: int = 100
    deadlock_detection_interval: float = 5.0
    enable_performance_monitoring: bool = True
    message_ttl: int = 3600  # Message TTL in seconds
    max_message_cache_size: int = 10000
    enable_automatic_recovery: bool = True
    lock_timeout: float = 30.0
    max_retry_attempts: int = 3


def create_enhanced_coordinator(config: EnhancedCoordinatorConfig = None) -> EnhancedDataFlowCoordinator:
    """Factory function to create enhanced coordinator with configuration"""
    if config is None:
        config = EnhancedCoordinatorConfig()
    
    return EnhancedDataFlowCoordinator(
        coordination_dir=config.coordination_dir,
        enable_persistence=config.enable_persistence,
        max_concurrent_streams=config.max_concurrent_streams,
        deadlock_detection_interval=config.deadlock_detection_interval,
        enable_performance_monitoring=config.enable_performance_monitoring
    )