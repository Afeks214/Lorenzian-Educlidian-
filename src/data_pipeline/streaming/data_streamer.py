"""
Data streaming implementation with minimal memory footprint
"""

import time
import threading
import queue
import pandas as pd
import numpy as np
from typing import Iterator, Callable, Optional, Any, Dict, List, Union, AsyncIterator, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
import gc
import psutil
from pathlib import Path
import weakref
import asyncio
from contextlib import asynccontextmanager, contextmanager
from collections import deque, defaultdict
import socket
import random
import math
from datetime import datetime, timedelta
import sys
import traceback
from functools import wraps
import json
import os
from threading import RLock, Event, Timer
from enum import Enum
from abc import ABC, abstractmethod
import heapq
import math
import statistics

from ..core.config import DataPipelineConfig
from ..core.exceptions import DataStreamingException
from ..core.data_loader import DataChunk, ScalableDataLoader

# Production-ready exception classes
class CircuitBreakerException(DataStreamingException):
    """Exception raised when circuit breaker is open"""
    pass

class MemoryPressureException(DataStreamingException):
    """Exception raised when memory pressure is critical"""
    pass

class ConnectionPoolException(DataStreamingException):
    """Exception raised when connection pool is exhausted"""
    pass

class RetryExhaustedException(DataStreamingException):
    """Exception raised when all retry attempts are exhausted"""
    pass

# Configure logger for production use
logger = logging.getLogger(__name__)

class Priority(Enum):
    """Priority levels for data chunks"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class ConnectionState(Enum):
    """Connection states for circuit breaker"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class BackpressureConfig:
    """Configuration for backpressure handling"""
    enable_backpressure: bool = True
    max_queue_size: int = 100
    queue_size_threshold: float = 0.8  # Trigger backpressure at 80% capacity
    adaptive_threshold: bool = True
    consumer_capacity_window: int = 10  # Number of samples for capacity tracking
    flow_control_window: int = 100  # Window for flow control metrics
    memory_threshold_mb: float = 500.0
    gc_threshold: float = 0.9  # Trigger GC at 90% memory threshold

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    enable_rate_limiting: bool = True
    # Token bucket parameters
    bucket_capacity: int = 1000  # Maximum tokens in bucket
    refill_rate: float = 100.0  # Tokens per second
    burst_capacity: int = 200  # Maximum burst size
    # Sliding window parameters
    window_size_seconds: int = 60  # Window size for sliding window
    max_requests_per_window: int = 5000  # Maximum requests per window
    # Adaptive parameters
    adaptive_rate_limiting: bool = True
    min_refill_rate: float = 10.0
    max_refill_rate: float = 1000.0
    rate_adjustment_factor: float = 0.1

@dataclass
class FlowControlConfig:
    """Configuration for flow control"""
    enable_flow_control: bool = True
    congestion_threshold: float = 0.85  # Trigger congestion control at 85%
    load_shed_threshold: float = 0.95  # Start load shedding at 95%
    priority_queue_size: int = 1000
    congestion_window_size: int = 50
    flow_control_algorithm: str = "adaptive"  # "adaptive", "fixed", "cubic"
    min_throughput_threshold: float = 0.1  # Minimum acceptable throughput ratio

@dataclass
class StreamConfig:
    """Configuration for data streaming"""
    buffer_size: int = 1000
    max_queue_size: int = 10
    timeout_seconds: float = 30.0
    enable_backpressure: bool = True
    memory_threshold_mb: float = 500.0
    enable_compression: bool = True
    enable_batching: bool = True
    batch_size: int = 100
    # Enhanced configurations
    backpressure_config: BackpressureConfig = field(default_factory=BackpressureConfig)
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)
    flow_control_config: FlowControlConfig = field(default_factory=FlowControlConfig)

class BackpressureController:
    """Controls backpressure based on queue size and consumer capacity"""
    
    def __init__(self, config: BackpressureConfig):
        self.config = config
        self.queue_size_history = deque(maxlen=config.flow_control_window)
        self.consumer_capacity_history = deque(maxlen=config.consumer_capacity_window)
        self.current_threshold = config.queue_size_threshold
        self.last_adjustment = time.time()
        self._lock = threading.Lock()
        
    def should_apply_backpressure(self, queue_size: int, max_queue_size: int,
                                 consumer_throughput: float) -> bool:
        """Determine if backpressure should be applied"""
        if not self.config.enable_backpressure:
            return False
            
        with self._lock:
            # Calculate queue utilization
            queue_utilization = queue_size / max_queue_size if max_queue_size > 0 else 0
            
            # Update history
            self.queue_size_history.append(queue_utilization)
            self.consumer_capacity_history.append(consumer_throughput)
            
            # Adaptive threshold adjustment
            if self.config.adaptive_threshold:
                self._adjust_threshold()
            
            # Check if backpressure should be applied
            return queue_utilization > self.current_threshold
    
    def _adjust_threshold(self):
        """Adjust threshold based on system performance"""
        if len(self.queue_size_history) < 10:
            return
            
        now = time.time()
        if now - self.last_adjustment < 5.0:  # Adjust at most every 5 seconds
            return
            
        # Calculate average queue utilization and consumer capacity
        avg_queue_util = statistics.mean(self.queue_size_history)
        avg_consumer_capacity = statistics.mean(self.consumer_capacity_history) if self.consumer_capacity_history else 1.0
        
        # Adjust threshold based on performance
        if avg_consumer_capacity > 0.8 and avg_queue_util < self.current_threshold * 0.8:
            # System is performing well, can increase threshold
            self.current_threshold = min(0.95, self.current_threshold * 1.05)
        elif avg_consumer_capacity < 0.5 or avg_queue_util > self.current_threshold * 0.9:
            # System is struggling, decrease threshold
            self.current_threshold = max(0.5, self.current_threshold * 0.95)
        
        self.last_adjustment = now
    
    def get_backpressure_signal(self, queue_size: int, max_queue_size: int) -> float:
        """Get backpressure signal strength (0.0 to 1.0)"""
        if not self.config.enable_backpressure:
            return 0.0
            
        queue_utilization = queue_size / max_queue_size if max_queue_size > 0 else 0
        
        if queue_utilization < self.current_threshold:
            return 0.0
        elif queue_utilization > 0.95:
            return 1.0
        else:
            # Linear interpolation between threshold and 0.95
            return (queue_utilization - self.current_threshold) / (0.95 - self.current_threshold)

class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float, burst_capacity: int):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.burst_capacity = burst_capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from bucket"""
        with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        time_elapsed = now - self.last_refill
        
        if time_elapsed > 0:
            tokens_to_add = int(time_elapsed * self.refill_rate)
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def get_tokens(self) -> int:
        """Get current token count"""
        with self._lock:
            self._refill()
            return self.tokens
    
    def adjust_rate(self, new_rate: float):
        """Dynamically adjust refill rate"""
        with self._lock:
            self.refill_rate = new_rate

class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self._lock = threading.Lock()
    
    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit"""
        with self._lock:
            now = time.time()
            
            # Remove expired requests
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if we can allow the request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_current_rate(self) -> float:
        """Get current request rate"""
        with self._lock:
            now = time.time()
            
            # Clean up expired requests
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            return len(self.requests) / self.window_size if self.window_size > 0 else 0.0

class RateLimiter:
    """Combined rate limiter with token bucket and sliding window"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.token_bucket = TokenBucket(
            config.bucket_capacity, 
            config.refill_rate, 
            config.burst_capacity
        )
        self.sliding_window = SlidingWindowRateLimiter(
            config.window_size_seconds, 
            config.max_requests_per_window
        )
        self.performance_history = deque(maxlen=100)
        self.last_rate_adjustment = time.time()
        self._lock = threading.Lock()
    
    def allow_request(self, tokens: int = 1) -> bool:
        """Check if request is allowed under rate limiting"""
        if not self.config.enable_rate_limiting:
            return True
        
        # Both token bucket and sliding window must allow the request
        token_allowed = self.token_bucket.consume(tokens)
        window_allowed = self.sliding_window.allow_request()
        
        return token_allowed and window_allowed
    
    def record_performance(self, success: bool, response_time: float):
        """Record performance metrics for adaptive rate limiting"""
        if not self.config.adaptive_rate_limiting:
            return
            
        with self._lock:
            self.performance_history.append({
                'success': success,
                'response_time': response_time,
                'timestamp': time.time()
            })
            
            # Adjust rate based on performance
            self._adjust_rate()
    
    def _adjust_rate(self):
        """Adjust rate limits based on performance"""
        now = time.time()
        if now - self.last_rate_adjustment < 10.0:  # Adjust at most every 10 seconds
            return
        
        if len(self.performance_history) < 10:
            return
        
        # Calculate success rate and average response time
        recent_metrics = [m for m in self.performance_history if now - m['timestamp'] < 30.0]
        if not recent_metrics:
            return
        
        success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
        avg_response_time = statistics.mean(m['response_time'] for m in recent_metrics)
        
        # Adjust refill rate based on performance
        current_rate = self.token_bucket.refill_rate
        
        if success_rate > 0.95 and avg_response_time < 1.0:
            # System is performing well, can increase rate
            new_rate = min(self.config.max_refill_rate, 
                          current_rate * (1 + self.config.rate_adjustment_factor))
        elif success_rate < 0.8 or avg_response_time > 5.0:
            # System is struggling, decrease rate
            new_rate = max(self.config.min_refill_rate, 
                          current_rate * (1 - self.config.rate_adjustment_factor))
        else:
            new_rate = current_rate
        
        if new_rate != current_rate:
            self.token_bucket.adjust_rate(new_rate)
            logger.info(f"Rate limit adjusted: {current_rate:.2f} -> {new_rate:.2f} tokens/sec")
        
        self.last_rate_adjustment = now
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return {
            'tokens_available': self.token_bucket.get_tokens(),
            'bucket_capacity': self.token_bucket.capacity,
            'refill_rate': self.token_bucket.refill_rate,
            'current_window_rate': self.sliding_window.get_current_rate(),
            'window_limit': self.config.max_requests_per_window,
            'rate_limiting_enabled': self.config.enable_rate_limiting
        }

@dataclass
class PriorityDataChunk:
    """Data chunk with priority for queue management"""
    chunk: DataChunk
    priority: Priority
    timestamp: float
    sequence_id: int
    
    def __lt__(self, other):
        """Compare by priority (higher priority first), then by timestamp"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp

class PriorityQueue:
    """Priority queue for data chunks with load shedding"""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.queue = []
        self.sequence_counter = 0
        self.dropped_chunks = 0
        self._lock = threading.Lock()
    
    def put(self, chunk: DataChunk, priority: Priority = Priority.NORMAL) -> bool:
        """Put a chunk in the priority queue"""
        with self._lock:
            if len(self.queue) >= self.max_size:
                # Apply load shedding - drop lowest priority items
                if self._should_drop_chunk(priority):
                    self.dropped_chunks += 1
                    return False
                
                # Remove lowest priority item to make space
                self._remove_lowest_priority()
            
            priority_chunk = PriorityDataChunk(
                chunk=chunk,
                priority=priority,
                timestamp=time.time(),
                sequence_id=self.sequence_counter
            )
            self.sequence_counter += 1
            
            heapq.heappush(self.queue, priority_chunk)
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[PriorityDataChunk]:
        """Get the highest priority chunk"""
        with self._lock:
            if self.queue:
                return heapq.heappop(self.queue)
            return None
    
    def _should_drop_chunk(self, priority: Priority) -> bool:
        """Determine if chunk should be dropped based on priority"""
        if not self.queue:
            return False
        
        # Find the lowest priority item in the queue
        lowest_priority = min(item.priority for item in self.queue)
        
        # Only drop if incoming priority is lower than or equal to the lowest in queue
        return priority.value <= lowest_priority.value
    
    def _remove_lowest_priority(self):
        """Remove the lowest priority item from the queue"""
        if not self.queue:
            return
        
        # Find index of lowest priority item
        lowest_idx = 0
        lowest_priority = self.queue[0].priority
        
        for i, item in enumerate(self.queue):
            if item.priority.value < lowest_priority.value:
                lowest_idx = i
                lowest_priority = item.priority
        
        # Remove the lowest priority item
        self.queue.pop(lowest_idx)
        heapq.heapify(self.queue)
        self.dropped_chunks += 1
    
    def size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self.queue)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            priority_counts = defaultdict(int)
            for item in self.queue:
                priority_counts[item.priority.name] += 1
            
            return {
                'size': len(self.queue),
                'max_size': self.max_size,
                'dropped_chunks': self.dropped_chunks,
                'priority_distribution': dict(priority_counts)
            }

class CongestionController:
    """Controls congestion based on system metrics"""
    
    def __init__(self, config: FlowControlConfig):
        self.config = config
        self.congestion_window = config.congestion_window_size
        self.throughput_history = deque(maxlen=config.congestion_window_size)
        self.latency_history = deque(maxlen=config.congestion_window_size)
        self.last_adjustment = time.time()
        self.congestion_state = "normal"  # normal, congested, recovery
        self._lock = threading.Lock()
    
    def update_metrics(self, throughput: float, latency: float):
        """Update congestion metrics"""
        with self._lock:
            self.throughput_history.append(throughput)
            self.latency_history.append(latency)
            
            # Determine congestion state
            self._update_congestion_state()
    
    def _update_congestion_state(self):
        """Update congestion state based on metrics"""
        if len(self.throughput_history) < 10:
            return
        
        # Calculate recent averages
        recent_throughput = statistics.mean(list(self.throughput_history)[-10:])
        recent_latency = statistics.mean(list(self.latency_history)[-10:])
        
        # Calculate baseline averages
        baseline_throughput = statistics.mean(self.throughput_history) if self.throughput_history else 0
        baseline_latency = statistics.mean(self.latency_history) if self.latency_history else 0
        
        # Determine congestion based on throughput drop and latency increase
        throughput_ratio = recent_throughput / baseline_throughput if baseline_throughput > 0 else 1.0
        latency_ratio = recent_latency / baseline_latency if baseline_latency > 0 else 1.0
        
        if throughput_ratio < self.config.min_throughput_threshold or latency_ratio > 2.0:
            self.congestion_state = "congested"
        elif throughput_ratio > 0.8 and latency_ratio < 1.5:
            self.congestion_state = "normal"
        else:
            self.congestion_state = "recovery"
    
    def get_congestion_level(self) -> float:
        """Get current congestion level (0.0 to 1.0)"""
        with self._lock:
            if self.congestion_state == "normal":
                return 0.0
            elif self.congestion_state == "recovery":
                return 0.5
            else:  # congested
                return 1.0
    
    def should_apply_congestion_control(self) -> bool:
        """Check if congestion control should be applied"""
        return self.congestion_state in ["congested", "recovery"]
    
    def get_recommended_window_size(self) -> int:
        """Get recommended congestion window size"""
        if self.congestion_state == "normal":
            return min(self.congestion_window * 2, self.config.congestion_window_size * 2)
        elif self.congestion_state == "recovery":
            return self.congestion_window
        else:  # congested
            return max(self.congestion_window // 2, self.config.congestion_window_size // 4)

class FlowController:
    """Main flow control coordinator"""
    
    def __init__(self, config: FlowControlConfig):
        self.config = config
        self.priority_queue = PriorityQueue(config.priority_queue_size)
        self.congestion_controller = CongestionController(config)
        self.flow_metrics = FlowMetrics()
        self.load_shedding_active = False
        self._lock = threading.Lock()
    
    def enqueue_chunk(self, chunk: DataChunk, priority: Priority = Priority.NORMAL) -> bool:
        """Enqueue a data chunk with flow control"""
        if not self.config.enable_flow_control:
            return True
        
        # Check if load shedding should be applied
        if self._should_apply_load_shedding():
            if priority.value < Priority.HIGH.value:
                self.flow_metrics.record_dropped_chunk(priority)
                return False
        
        # Try to enqueue the chunk
        success = self.priority_queue.put(chunk, priority)
        
        if success:
            self.flow_metrics.record_enqueued_chunk(priority)
        else:
            self.flow_metrics.record_dropped_chunk(priority)
        
        return success
    
    def dequeue_chunk(self) -> Optional[DataChunk]:
        """Dequeue a data chunk with flow control"""
        if not self.config.enable_flow_control:
            return None
        
        start_time = time.time()
        priority_chunk = self.priority_queue.get()
        
        if priority_chunk:
            processing_time = time.time() - start_time
            self.flow_metrics.record_dequeued_chunk(priority_chunk.priority, processing_time)
            
            # Update congestion metrics
            current_throughput = self.flow_metrics.get_throughput()
            current_latency = processing_time
            self.congestion_controller.update_metrics(current_throughput, current_latency)
            
            return priority_chunk.chunk
        
        return None
    
    def _should_apply_load_shedding(self) -> bool:
        """Determine if load shedding should be applied"""
        with self._lock:
            queue_utilization = self.priority_queue.size() / self.config.priority_queue_size
            congestion_level = self.congestion_controller.get_congestion_level()
            
            # Apply load shedding if queue is too full or system is congested
            should_shed = (queue_utilization > self.config.load_shed_threshold or
                          congestion_level > self.config.congestion_threshold)
            
            if should_shed != self.load_shedding_active:
                self.load_shedding_active = should_shed
                if should_shed:
                    logger.warning("Load shedding activated")
                else:
                    logger.info("Load shedding deactivated")
            
            return should_shed
    
    def get_flow_control_status(self) -> Dict[str, Any]:
        """Get current flow control status"""
        return {
            'queue_stats': self.priority_queue.get_stats(),
            'congestion_level': self.congestion_controller.get_congestion_level(),
            'congestion_state': self.congestion_controller.congestion_state,
            'load_shedding_active': self.load_shedding_active,
            'flow_metrics': self.flow_metrics.get_metrics(),
            'recommended_window_size': self.congestion_controller.get_recommended_window_size()
        }

class FlowMetrics:
    """Track flow control metrics"""
    
    def __init__(self):
        self.enqueued_chunks = defaultdict(int)
        self.dequeued_chunks = defaultdict(int)
        self.dropped_chunks = defaultdict(int)
        self.processing_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        self.last_throughput_calculation = time.time()
        self.total_processed = 0
        self._lock = threading.Lock()
    
    def record_enqueued_chunk(self, priority: Priority):
        """Record an enqueued chunk"""
        with self._lock:
            self.enqueued_chunks[priority.name] += 1
    
    def record_dequeued_chunk(self, priority: Priority, processing_time: float):
        """Record a dequeued chunk"""
        with self._lock:
            self.dequeued_chunks[priority.name] += 1
            self.processing_times.append(processing_time)
            self.total_processed += 1
    
    def record_dropped_chunk(self, priority: Priority):
        """Record a dropped chunk"""
        with self._lock:
            self.dropped_chunks[priority.name] += 1
    
    def get_throughput(self) -> float:
        """Calculate current throughput"""
        with self._lock:
            now = time.time()
            if now - self.last_throughput_calculation > 1.0:  # Update every second
                time_window = min(now - self.last_throughput_calculation, 60.0)
                throughput = len(self.processing_times) / time_window if time_window > 0 else 0.0
                self.throughput_history.append(throughput)
                self.last_throughput_calculation = now
            
            return statistics.mean(self.throughput_history) if self.throughput_history else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive flow metrics"""
        with self._lock:
            avg_processing_time = statistics.mean(self.processing_times) if self.processing_times else 0.0
            
            return {
                'enqueued_chunks': dict(self.enqueued_chunks),
                'dequeued_chunks': dict(self.dequeued_chunks),
                'dropped_chunks': dict(self.dropped_chunks),
                'total_processed': self.total_processed,
                'avg_processing_time': avg_processing_time,
                'current_throughput': self.get_throughput(),
                'queue_depth': sum(self.enqueued_chunks.values()) - sum(self.dequeued_chunks.values())
            }
    
    def __post_init__(self):
        """Validate configuration values"""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.memory_threshold_mb <= 0:
            raise ValueError("memory_threshold_mb must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.initial_retry_delay <= 0:
            raise ValueError("initial_retry_delay must be positive")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")

@dataclass
class ConnectionMetrics:
    """Metrics for connection monitoring"""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    successful_connections: int = 0
    average_connection_time: float = 0.0
    last_connection_time: Optional[datetime] = None
    circuit_breaker_state: ConnectionState = ConnectionState.CLOSED
    retry_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/monitoring"""
        return {
            'total_connections': self.total_connections,
            'active_connections': self.active_connections,
            'failed_connections': self.failed_connections,
            'successful_connections': self.successful_connections,
            'average_connection_time': self.average_connection_time,
            'last_connection_time': self.last_connection_time.isoformat() if self.last_connection_time else None,
            'circuit_breaker_state': self.circuit_breaker_state.value,
            'retry_count': self.retry_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }

class CircuitBreaker:
    """Circuit breaker for connection resilience"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = ConnectionState.CLOSED
        self._half_open_calls = 0
        self._lock = RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self._state == ConnectionState.OPEN:
                if self._should_attempt_reset():
                    self._state = ConnectionState.HALF_OPEN
                    self._half_open_calls = 0
                else:
                    raise CircuitBreakerException("Circuit breaker is OPEN")
            
            if self._state == ConnectionState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerException("Circuit breaker is HALF_OPEN and max calls exceeded")
                self._half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker"""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        with self._lock:
            self._failure_count = 0
            self._state = ConnectionState.CLOSED
            self._half_open_calls = 0
    
    def _on_failure(self):
        """Handle failed operation"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = ConnectionState.OPEN
    
    @property
    def state(self) -> ConnectionState:
        """Get current circuit breaker state"""
        return self._state
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self._failure_count = 0
            self._state = ConnectionState.CLOSED
            self._half_open_calls = 0
            self._last_failure_time = None

class ConnectionPool:
    """Connection pool for managing connections efficiently"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self._pool = queue.Queue(maxsize=config.pool_size)
        self._all_connections = set()
        self._lock = RLock()
        self._created_connections = 0
        self._active_connections = 0
        self._stats = ConnectionMetrics()
        
        # Initialize pool with connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        logger.info(f"Initializing connection pool with {self.config.pool_size} connections")
        # For now, we'll create placeholder connections
        # In a real implementation, these would be actual database/service connections
        pass
    
    def get_connection(self):
        """Get connection from pool"""
        with self._lock:
            try:
                connection = self._pool.get_nowait()
                self._active_connections += 1
                return connection
            except queue.Empty:
                if self._created_connections < self.config.pool_size + self.config.pool_max_overflow:
                    return self._create_connection()
                raise ConnectionPoolException("Connection pool exhausted")
    
    def return_connection(self, connection):
        """Return connection to pool"""
        with self._lock:
            if connection in self._all_connections:
                try:
                    self._pool.put_nowait(connection)
                    self._active_connections = max(0, self._active_connections - 1)
                except queue.Full:
                    self._close_connection(connection)
    
    def _create_connection(self):
        """Create new connection"""
        # Placeholder for actual connection creation
        connection = f"connection_{self._created_connections}"
        self._created_connections += 1
        self._all_connections.add(connection)
        self._active_connections += 1
        return connection
    
    def _close_connection(self, connection):
        """Close connection and remove from pool"""
        if connection in self._all_connections:
            self._all_connections.remove(connection)
            self._created_connections = max(0, self._created_connections - 1)
    
    def close_all(self):
        """Close all connections"""
        with self._lock:
            while not self._pool.empty():
                try:
                    connection = self._pool.get_nowait()
                    self._close_connection(connection)
                except queue.Empty:
                    break
            
            # Close any remaining connections
            for connection in list(self._all_connections):
                self._close_connection(connection)
    
    @property
    def stats(self) -> ConnectionMetrics:
        """Get connection pool statistics"""
        with self._lock:
            self._stats.total_connections = self._created_connections
            self._stats.active_connections = self._active_connections
            return self._stats

class RetryManager:
    """Manages retry logic with exponential backoff"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self._retry_count = 0
        self._last_retry_time = None
    
    def execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retrying operation after {delay:.2f}s (attempt {attempt + 1}/{self.config.max_retries + 1})")
                    time.sleep(delay)
                
                result = func(*args, **kwargs)
                self._reset_retry_state()
                return result
                
            except Exception as e:
                last_exception = e
                self._retry_count += 1
                self._last_retry_time = time.time()
                
                logger.warning(f"Operation failed on attempt {attempt + 1}: {str(e)}")
                
                if attempt == self.config.max_retries:
                    logger.error(f"All retry attempts exhausted. Last error: {str(e)}")
                    break
        
        raise RetryExhaustedException(f"All retry attempts exhausted. Last error: {str(last_exception)}")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        delay = min(
            self.config.initial_retry_delay * (self.config.backoff_multiplier ** (attempt - 1)),
            self.config.max_retry_delay
        )
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, self.config.jitter_max * delay)
        return delay + jitter
    
    def _reset_retry_state(self):
        """Reset retry state on successful operation"""
        self._retry_count = 0
        self._last_retry_time = None
    
    @property
    def retry_count(self) -> int:
        """Get current retry count"""
        return self._retry_count

class EnhancedMemoryMonitor:
    """Enhanced memory monitor with comprehensive tracking and management"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.threshold_mb = config.memory_threshold_mb
        self.max_memory_mb = config.max_memory_mb
        
        # Memory tracking
        self.peak_usage = 0.0
        self.current_usage = 0.0
        self.pressure_events = 0
        self.last_pressure_time = None
        
        # Memory usage history
        self.usage_history = deque(maxlen=1000)
        self.pressure_history = deque(maxlen=100)
        
        # GC statistics
        self.gc_collections = {'gen0': 0, 'gen1': 0, 'gen2': 0}
        self.last_gc_stats = gc.get_stats()
        
        # Thread safety
        self._lock = RLock()
        
        # Process reference
        self._process = psutil.Process()
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is under pressure with enhanced monitoring"""
        with self._lock:
            try:
                # Get current memory usage
                memory_info = self._process.memory_info()
                self.current_usage = memory_info.rss / 1024 / 1024
                self.peak_usage = max(self.peak_usage, self.current_usage)
                
                # Add to history
                self.usage_history.append({
                    'timestamp': time.time(),
                    'usage_mb': self.current_usage,
                    'rss': memory_info.rss,
                    'vms': memory_info.vms
                })
                
                # Check for pressure
                is_pressure = self.current_usage > self.threshold_mb
                is_critical = self.current_usage > self.max_memory_mb
                
                if is_pressure or is_critical:
                    self.pressure_events += 1
                    self.last_pressure_time = time.time()
                    
                    pressure_info = {
                        'timestamp': time.time(),
                        'usage_mb': self.current_usage,
                        'threshold_mb': self.threshold_mb,
                        'is_critical': is_critical,
                        'pressure_level': 'critical' if is_critical else 'high'
                    }
                    
                    self.pressure_history.append(pressure_info)
                    
                    if is_critical:
                        logger.critical(f"Critical memory usage: {self.current_usage:.2f} MB > {self.max_memory_mb} MB")
                    else:
                        logger.warning(f"Memory pressure detected: {self.current_usage:.2f} MB > {self.threshold_mb} MB")
                    
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Error checking memory pressure: {e}")
                return False
    
    def get_current_usage(self) -> float:
        """Get current memory usage in MB"""
        with self._lock:
            return self.current_usage
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB"""
        with self._lock:
            return self.peak_usage
    
    def get_pressure_events(self) -> int:
        """Get number of memory pressure events"""
        with self._lock:
            return self.pressure_events
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """Get garbage collection statistics"""
        with self._lock:
            try:
                current_stats = gc.get_stats()
                
                # Calculate differences
                gc_diff = {
                    'gen0': current_stats[0]['collections'] - self.last_gc_stats[0]['collections'],
                    'gen1': current_stats[1]['collections'] - self.last_gc_stats[1]['collections'],
                    'gen2': current_stats[2]['collections'] - self.last_gc_stats[2]['collections']
                }
                
                # Update tracking
                for gen, count in gc_diff.items():
                    self.gc_collections[gen] += count
                
                self.last_gc_stats = current_stats
                
                return {
                    'total_collections': self.gc_collections,
                    'recent_collections': gc_diff,
                    'current_stats': current_stats,
                    'gc_threshold': gc.get_threshold(),
                    'gc_count': gc.get_count()
                }
                
            except Exception as e:
                logger.error(f"Error getting GC stats: {e}")
                return {}
    
    def force_cleanup(self):
        """Force memory cleanup"""
        logger.info("Forcing memory cleanup")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Clear internal caches
        self._clear_internal_caches()
        
        # Update usage after cleanup
        self.check_memory_pressure()
        
        logger.info(f"Memory usage after cleanup: {self.current_usage:.2f} MB")
    
    def _clear_internal_caches(self):
        """Clear internal caches to free memory"""
        # Trim usage history if too large
        if len(self.usage_history) > 500:
            recent_entries = list(self.usage_history)[-500:]
            self.usage_history.clear()
            self.usage_history.extend(recent_entries)
        
        # Trim pressure history
        if len(self.pressure_history) > 50:
            recent_entries = list(self.pressure_history)[-50:]
            self.pressure_history.clear()
            self.pressure_history.extend(recent_entries)

class HealthMonitor:
    """Monitor system health and streaming performance"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self._health_checks = []
        self._last_check_time = None
        self._health_status = True
        self._health_issues = []
        
    def check_health(self, streamer):
        """Perform comprehensive health check"""
        self._last_check_time = time.time()
        self._health_issues.clear()
        
        # Check memory health
        if streamer._memory_monitor.check_memory_pressure():
            self._health_issues.append("Memory pressure detected")
        
        # Check circuit breaker state
        if streamer._circuit_breaker.state != ConnectionState.CLOSED:
            self._health_issues.append(f"Circuit breaker is {streamer._circuit_breaker.state.value}")
        
        # Check error rate
        error_rate = streamer._stats.get_error_rate()
        if error_rate > 5.0:  # More than 5% error rate
            self._health_issues.append(f"High error rate: {error_rate:.2f}%")
        
        # Check queue sizes
        if streamer._stream_queue.qsize() > streamer.stream_config.max_queue_size * 0.8:
            self._health_issues.append("Stream queue near capacity")
        
        self._health_status = len(self._health_issues) == 0
        
        if not self._health_status:
            logger.warning(f"Health check failed: {', '.join(self._health_issues)}")
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self._health_status
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get health statistics"""
        return {
            'is_healthy': self._health_status,
            'issues': self._health_issues,
            'last_check_time': self._last_check_time,
            'check_interval': self.config.health_check_interval
        }

class MetricsCollector:
    """Collect and manage performance metrics"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self._metrics = {
            'chunks_processed': 0,
            'errors': 0,
            'processing_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'connection_events': deque(maxlen=1000)
        }
        self._start_time = time.time()
    
    def collect_metrics(self, streamer):
        """Collect metrics from streamer"""
        current_time = time.time()
        
        # Collect basic metrics
        stats = streamer.get_streaming_stats()
        
        # Store metrics
        self._metrics['memory_usage'].append({
            'timestamp': current_time,
            'usage_mb': stats.get('memory_usage_mb', 0)
        })
        
        # Log metrics if enabled
        if self.config.enable_detailed_logging:
            logger.info(f"Metrics: {json.dumps(stats, indent=2)}")
    
    def record_chunk_processed(self, chunk: DataChunk):
        """Record chunk processing metrics"""
        self._metrics['chunks_processed'] += 1
        
        processing_time = time.time() - chunk.timestamp
        self._metrics['processing_times'].append(processing_time)
    
    def record_error(self, operation_id: str, error: Exception):
        """Record error metrics"""
        self._metrics['errors'] += 1
        
        error_info = {
            'timestamp': time.time(),
            'operation_id': operation_id,
            'error_type': type(error).__name__,
            'error_message': str(error)
        }
        
        logger.error(f"Error recorded: {json.dumps(error_info)}")
    
    def record_processing_error(self, chunk_id: str, error: Exception):
        """Record chunk processing error"""
        self.record_error(f"chunk_{chunk_id}", error)
    
    def get_error_rate(self) -> float:
        """Calculate error rate"""
        total_operations = self._metrics['chunks_processed'] + self._metrics['errors']
        if total_operations == 0:
            return 0.0
        return (self._metrics['errors'] / total_operations) * 100

class ResourceTracker:
    """Track resource usage and active operations"""
    
    def __init__(self):
        self._active_operations = {}
        self._total_operations = 0
        self._lock = RLock()
    
    def track_operation(self, operation_id: str):
        """Track start of operation"""
        with self._lock:
            self._active_operations[operation_id] = {
                'start_time': time.time(),
                'thread_id': threading.current_thread().ident
            }
            self._total_operations += 1
    
    def untrack_operation(self, operation_id: str):
        """Track end of operation"""
        with self._lock:
            if operation_id in self._active_operations:
                del self._active_operations[operation_id]
    
    def has_active_operations(self) -> bool:
        """Check if there are active operations"""
        with self._lock:
            return len(self._active_operations) > 0
    
    def get_active_operations(self) -> Dict[str, Any]:
        """Get active operations info"""
        with self._lock:
            return dict(self._active_operations)
    
    def get_total_operations(self) -> int:
        """Get total operations count"""
        with self._lock:
            return self._total_operations
    
    def force_cleanup(self):
        """Force cleanup of active operations"""
        with self._lock:
            logger.warning(f"Force cleanup of {len(self._active_operations)} active operations")
            self._active_operations.clear()

class EnhancedStreamingStats:
    """Enhanced streaming statistics with comprehensive tracking"""
    
    def __init__(self):
        self.chunks_processed = 0
        self.total_rows = 0
        self.bytes_processed = 0
        self.processing_times = deque(maxlen=1000)  # Keep last 1000 times
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # Enhanced statistics
        self.errors_count = 0
        self.retries_count = 0
        self.filtered_chunks = 0
        self.transformed_chunks = 0
        self.memory_pressure_events = 0
        
        # Performance tracking
        self.min_processing_time = float('inf')
        self.max_processing_time = 0.0
        self.throughput_samples = deque(maxlen=100)  # Keep last 100 samples
        
        # Memory tracking
        self.peak_memory_usage = 0
        self.memory_usage_samples = deque(maxlen=100)
        
        # Thread safety
        self._lock = RLock()
    
    def update(self, chunk: DataChunk):
        """
        Update statistics with a processed chunk.
        
        Args:
            chunk: Processed data chunk
        """
        with self._lock:
            self.chunks_processed += 1
            self.total_rows += len(chunk.data)
            
            # Calculate bytes processed
            try:
                chunk_bytes = chunk.data.memory_usage(deep=True).sum()
                self.bytes_processed += chunk_bytes
            except Exception as e:
                logger.warning(f"Error calculating chunk memory usage: {e}")
                # Use approximate calculation
                chunk_bytes = len(chunk.data) * chunk.data.shape[1] * 8  # Approximate
                self.bytes_processed += chunk_bytes
            
            # Track processing time
            current_time = time.time()
            processing_time = current_time - chunk.timestamp
            self.processing_times.append(processing_time)
            
            # Update min/max processing times
            self.min_processing_time = min(self.min_processing_time, processing_time)
            self.max_processing_time = max(self.max_processing_time, processing_time)
            
            # Calculate throughput sample
            time_delta = current_time - self.last_update_time
            if time_delta > 0:
                rows_per_second = len(chunk.data) / time_delta
                self.throughput_samples.append(rows_per_second)
            
            self.last_update_time = current_time
    
    def record_error(self):
        """Record an error occurrence"""
        with self._lock:
            self.errors_count += 1
    
    def record_retry(self):
        """Record a retry occurrence"""
        with self._lock:
            self.retries_count += 1
    
    def record_filtered_chunk(self):
        """Record a filtered chunk"""
        with self._lock:
            self.filtered_chunks += 1
    
    def record_transformed_chunk(self):
        """Record a transformed chunk"""
        with self._lock:
            self.transformed_chunks += 1
    
    def record_memory_pressure(self):
        """Record a memory pressure event"""
        with self._lock:
            self.memory_pressure_events += 1
    
    def update_memory_usage(self, memory_mb: float):
        """Update memory usage tracking"""
        with self._lock:
            self.peak_memory_usage = max(self.peak_memory_usage, memory_mb)
            self.memory_usage_samples.append(memory_mb)
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time per chunk"""
        with self._lock:
            if not self.processing_times:
                return 0.0
            return float(np.mean(self.processing_times))
    
    def get_processing_time_percentiles(self) -> Dict[str, float]:
        """Get processing time percentiles"""
        with self._lock:
            if not self.processing_times:
                return {'p50': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0}
            
            times = list(self.processing_times)
            return {
                'p50': float(np.percentile(times, 50)),
                'p90': float(np.percentile(times, 90)),
                'p95': float(np.percentile(times, 95)),
                'p99': float(np.percentile(times, 99))
            }
    
    def get_throughput(self) -> float:
        """Get overall throughput in rows per second"""
        with self._lock:
            elapsed = time.time() - self.start_time
            return self.total_rows / elapsed if elapsed > 0 else 0.0
    
    def get_current_throughput(self) -> float:
        """Get current throughput based on recent samples"""
        with self._lock:
            if not self.throughput_samples:
                return 0.0
            return float(np.mean(self.throughput_samples))
    
    def get_error_rate(self) -> float:
        """Get error rate as percentage"""
        with self._lock:
            total_operations = self.chunks_processed + self.errors_count
            if total_operations == 0:
                return 0.0
            return (self.errors_count / total_operations) * 100
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._lock:
            return {
                'chunks_processed': self.chunks_processed,
                'total_rows': self.total_rows,
                'bytes_processed': self.bytes_processed,
                'errors_count': self.errors_count,
                'retries_count': self.retries_count,
                'filtered_chunks': self.filtered_chunks,
                'transformed_chunks': self.transformed_chunks,
                'memory_pressure_events': self.memory_pressure_events,
                'processing_times': {
                    'avg': self.get_avg_processing_time(),
                    'min': self.min_processing_time if self.min_processing_time != float('inf') else 0.0,
                    'max': self.max_processing_time,
                    **self.get_processing_time_percentiles()
                },
                'throughput': {
                    'overall': self.get_throughput(),
                    'current': self.get_current_throughput()
                },
                'memory': {
                    'peak_usage_mb': self.peak_memory_usage,
                    'avg_usage_mb': float(np.mean(self.memory_usage_samples)) if self.memory_usage_samples else 0.0
                },
                'error_rate_percent': self.get_error_rate(),
                'uptime_seconds': time.time() - self.start_time
            }
    
    def reset(self):
        """Reset statistics"""
        with self._lock:
            self.chunks_processed = 0
            self.total_rows = 0
            self.bytes_processed = 0
            self.processing_times.clear()
            self.start_time = time.time()
            self.last_update_time = time.time()
            
            # Reset enhanced statistics
            self.errors_count = 0
            self.retries_count = 0
            self.filtered_chunks = 0
            self.transformed_chunks = 0
            self.memory_pressure_events = 0
            
            # Reset performance tracking
            self.min_processing_time = float('inf')
            self.max_processing_time = 0.0
            self.throughput_samples.clear()
            
            # Reset memory tracking
            self.peak_memory_usage = 0
            self.memory_usage_samples.clear()
        self.total_rows = 0
        self.bytes_processed = 0
        self.processing_times = deque(maxlen=1000)
        self.throughput_samples = deque(maxlen=100)
        self.start_time = time.time()
        self.last_throughput_calculation = time.time()
        
        # Enhanced metrics
        self.error_count = 0
        self.retry_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.gc_count = 0
        self.memory_pressure_events = 0
        self.backpressure_events = 0
        self.rate_limit_events = 0
        
        # Performance metrics
        self.latency_percentiles = {}
        self.error_rate_history = deque(maxlen=100)
        self.resource_utilization = {}
        
        self._lock = threading.Lock()
    
    def update(self, chunk: DataChunk):
        """Update statistics with a processed chunk"""
        with self._lock:
            self.chunks_processed += 1
            self.total_rows += len(chunk.data)
            self.bytes_processed += chunk.data.memory_usage(deep=True).sum()
            
            # Track processing time
            processing_time = time.time() - chunk.timestamp
            self.processing_times.append(processing_time)
            
            # Update throughput
            self._update_throughput()
    
    def _update_throughput(self):
        """Update throughput calculations"""
        now = time.time()
        if now - self.last_throughput_calculation > 1.0:  # Update every second
            elapsed = now - self.start_time
            current_throughput = self.total_rows / elapsed if elapsed > 0 else 0.0
            self.throughput_samples.append(current_throughput)
            self.last_throughput_calculation = now
    
    def record_error(self):
        """Record an error event"""
        with self._lock:
            self.error_count += 1
            self.error_rate_history.append(time.time())
    
    def record_retry(self):
        """Record a retry event"""
        with self._lock:
            self.retry_count += 1
    
    def record_cache_hit(self):
        """Record a cache hit"""
        with self._lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        with self._lock:
            self.cache_misses += 1
    
    def record_gc_event(self):
        """Record a garbage collection event"""
        with self._lock:
            self.gc_count += 1
    
    def record_memory_pressure(self):
        """Record a memory pressure event"""
        with self._lock:
            self.memory_pressure_events += 1
    
    def record_backpressure_event(self):
        """Record a backpressure event"""
        with self._lock:
            self.backpressure_events += 1
    
    def record_rate_limit_event(self):
        """Record a rate limit event"""
        with self._lock:
            self.rate_limit_events += 1
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time per chunk"""
        with self._lock:
            return statistics.mean(self.processing_times) if self.processing_times else 0.0
    
    def get_throughput(self) -> float:
        """Get current throughput in rows per second"""
        with self._lock:
            return statistics.mean(self.throughput_samples) if self.throughput_samples else 0.0
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles"""
        with self._lock:
            if not self.processing_times:
                return {}
            
            times = sorted(self.processing_times)
            return {
                'p50': np.percentile(times, 50),
                'p95': np.percentile(times, 95),
                'p99': np.percentile(times, 99),
                'p99.9': np.percentile(times, 99.9)
            }
    
    def get_error_rate(self) -> float:
        """Get current error rate"""
        with self._lock:
            now = time.time()
            recent_errors = [t for t in self.error_rate_history if now - t < 60.0]
            return len(recent_errors) / 60.0  # Errors per second
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        with self._lock:
            total_requests = self.cache_hits + self.cache_misses
            return self.cache_hits / total_requests if total_requests > 0 else 0.0
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics"""
        with self._lock:
            elapsed = time.time() - self.start_time
            
            return {
                'basic_stats': {
                    'chunks_processed': self.chunks_processed,
                    'total_rows': self.total_rows,
                    'bytes_processed': self.bytes_processed,
                    'elapsed_time': elapsed,
                    'throughput_rows_per_sec': self.get_throughput(),
                    'avg_processing_time': self.get_avg_processing_time()
                },
                'performance_metrics': {
                    'latency_percentiles': self.get_latency_percentiles(),
                    'error_rate': self.get_error_rate(),
                    'cache_hit_rate': self.get_cache_hit_rate()
                },
                'event_counts': {
                    'error_count': self.error_count,
                    'retry_count': self.retry_count,
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'gc_count': self.gc_count,
                    'memory_pressure_events': self.memory_pressure_events,
                    'backpressure_events': self.backpressure_events,
                    'rate_limit_events': self.rate_limit_events
                },
                'resource_utilization': self.resource_utilization
            }
    
    def reset(self):
        """Reset statistics"""
        with self._lock:
            self.chunks_processed = 0
            self.total_rows = 0
            self.bytes_processed = 0
            self.processing_times.clear()
            self.throughput_samples.clear()
            self.start_time = time.time()
            self.last_throughput_calculation = time.time()
            
            # Reset enhanced metrics
            self.error_count = 0
            self.retry_count = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.gc_count = 0
            self.memory_pressure_events = 0
            self.backpressure_events = 0
            self.rate_limit_events = 0
            
            self.error_rate_history.clear()
            self.resource_utilization.clear()

class EnhancedMemoryMonitor:
    """Enhanced memory monitoring with predictive capabilities"""
    
    def __init__(self, threshold_mb: float):
        self.threshold_mb = threshold_mb
        self.peak_usage = 0.0
        self.current_usage = 0.0
        self.usage_history = deque(maxlen=100)
        self.gc_history = deque(maxlen=50)
        self.last_gc_time = time.time()
        self.pressure_events = 0
        self._lock = threading.Lock()
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is under pressure with prediction"""
        with self._lock:
            process = psutil.Process()
            self.current_usage = process.memory_info().rss / 1024 / 1024
            self.peak_usage = max(self.peak_usage, self.current_usage)
            
            # Add to history
            self.usage_history.append({
                'usage': self.current_usage,
                'timestamp': time.time()
            })
            
            # Check current pressure
            is_pressure = self.current_usage > self.threshold_mb
            
            # Predict future pressure based on trend
            if len(self.usage_history) >= 10:
                recent_usage = [h['usage'] for h in list(self.usage_history)[-10:]]
                if len(recent_usage) >= 2:
                    # Simple linear trend prediction
                    slope = (recent_usage[-1] - recent_usage[0]) / len(recent_usage)
                    predicted_usage = self.current_usage + slope * 5  # Predict 5 steps ahead
                    
                    # Trigger early if prediction shows pressure
                    if predicted_usage > self.threshold_mb * 0.9:
                        is_pressure = True
            
            if is_pressure:
                self.pressure_events += 1
            
            return is_pressure
    
    def should_trigger_gc(self) -> bool:
        """Determine if garbage collection should be triggered"""
        with self._lock:
            now = time.time()
            
            # Don't trigger GC too frequently
            if now - self.last_gc_time < 5.0:
                return False
            
            # Trigger GC if memory usage is high
            if self.current_usage > self.threshold_mb * 0.8:
                self.last_gc_time = now
                self.gc_history.append({
                    'usage_before': self.current_usage,
                    'timestamp': now
                })
                return True
            
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        with self._lock:
            avg_usage = statistics.mean([h['usage'] for h in self.usage_history]) if self.usage_history else 0.0
            
            # Calculate memory growth rate
            growth_rate = 0.0
            if len(self.usage_history) >= 2:
                recent_usage = [h['usage'] for h in list(self.usage_history)[-10:]]
                if len(recent_usage) >= 2:
                    growth_rate = (recent_usage[-1] - recent_usage[0]) / len(recent_usage)
            
            return {
                'current_usage_mb': self.current_usage,
                'peak_usage_mb': self.peak_usage,
                'threshold_mb': self.threshold_mb,
                'average_usage_mb': avg_usage,
                'growth_rate_mb_per_sample': growth_rate,
                'pressure_events': self.pressure_events,
                'gc_events': len(self.gc_history),
                'utilization_percent': (self.current_usage / self.threshold_mb) * 100
            }
    
    def get_current_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.current_usage
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB"""
        return self.peak_usage

class SystemMetricsCollector:
    """Collect system-wide metrics for monitoring"""
    
    def __init__(self):
        self.collection_interval = 5.0  # seconds
        self.metrics_history = deque(maxlen=1000)
        self.collecting = False
        self.collection_thread = None
        self._lock = threading.Lock()
    
    def start_collection(self):
        """Start collecting system metrics"""
        with self._lock:
            if not self.collecting:
                self.collecting = True
                self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
                self.collection_thread.start()
                logger.info("System metrics collection started")
    
    def stop_collection(self):
        """Stop collecting system metrics"""
        with self._lock:
            self.collecting = False
            if self.collection_thread:
                self.collection_thread.join(timeout=1.0)
                logger.info("System metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.collecting:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        
        # Network metrics (if available)
        network_stats = psutil.net_io_counters()
        
        return {
            'timestamp': time.time(),
            'cpu': {
                'usage_percent': cpu_percent,
                'core_count': cpu_count,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            'memory': {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'used_mb': memory.used / 1024 / 1024,
                'usage_percent': memory.percent
            },
            'disk': {
                'total_gb': disk_usage.total / 1024 / 1024 / 1024,
                'free_gb': disk_usage.free / 1024 / 1024 / 1024,
                'used_gb': disk_usage.used / 1024 / 1024 / 1024,
                'usage_percent': (disk_usage.used / disk_usage.total) * 100
            },
            'network': {
                'bytes_sent': network_stats.bytes_sent,
                'bytes_recv': network_stats.bytes_recv,
                'packets_sent': network_stats.packets_sent,
                'packets_recv': network_stats.packets_recv
            }
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the most recent system metrics"""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else {}
    
    def get_metrics_summary(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Get summary of metrics over a time period"""
        with self._lock:
            now = time.time()
            recent_metrics = [
                m for m in self.metrics_history 
                if now - m['timestamp'] <= duration_seconds
            ]
            
            if not recent_metrics:
                return {}
            
            # Calculate averages
            avg_cpu = statistics.mean([m['cpu']['usage_percent'] for m in recent_metrics])
            avg_memory = statistics.mean([m['memory']['usage_percent'] for m in recent_metrics])
            avg_disk = statistics.mean([m['disk']['usage_percent'] for m in recent_metrics])
            
            return {
                'duration_seconds': duration_seconds,
                'sample_count': len(recent_metrics),
                'cpu_usage_avg': avg_cpu,
                'memory_usage_avg': avg_memory,
                'disk_usage_avg': avg_disk,
                'peak_cpu': max([m['cpu']['usage_percent'] for m in recent_metrics]),
                'peak_memory': max([m['memory']['usage_percent'] for m in recent_metrics])
            }

class DataStreamer:
    """
    Production-ready high-performance data streamer with comprehensive
    memory management, connection recovery, and monitoring capabilities
    """
    
    def __init__(self, config: Optional[DataPipelineConfig] = None,
                 stream_config: Optional[StreamConfig] = None):
        self.config = config or DataPipelineConfig()
        self.stream_config = stream_config or StreamConfig()
        self.data_loader = ScalableDataLoader(self.config)
        
        # Setup logging
        self._setup_logging()
        
        # Streaming state
        self._is_streaming = False
        self._stream_queue = queue.Queue(maxsize=self.stream_config.max_queue_size)
        self._error_queue = queue.Queue()
        self._stats = EnhancedStreamingStats()
        self._lock = RLock()  # Use RLock for recursive locking
        self._shutdown_event = Event()
        
        # Memory monitoring and management
        self._memory_monitor = EnhancedMemoryMonitor(self.stream_config)
        self._memory_cleanup_timer = None
        
        # Connection management
        self._connection_pool = ConnectionPool(self.stream_config)
        self._circuit_breaker = CircuitBreaker(
            self.stream_config.failure_threshold,
            self.stream_config.recovery_timeout,
            self.stream_config.half_open_max_calls
        )
        self._retry_manager = RetryManager(self.stream_config)
        
        # Health monitoring
        self._health_monitor = HealthMonitor(self.stream_config)
        self._health_check_timer = None
        
        # Metrics collection
        self._metrics_collector = MetricsCollector(self.stream_config)
        self._metrics_timer = None
        
        # Resource tracking
        self._resource_tracker = ResourceTracker()
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        # Setup cleanup
        self._setup_cleanup()
        
        logger.info(f"DataStreamer initialized with config: {self.stream_config}")
    
    def _setup_logging(self):
        """Setup structured logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if self.stream_config.enable_detailed_logging:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, self.stream_config.log_level),
            format=log_format
        )
    
    def _initialize_monitoring(self):
        """Initialize monitoring timers"""
        if self.stream_config.enable_metrics:
            self._start_metrics_collection()
        
        self._start_health_monitoring()
        self._start_memory_monitoring()
    
    def _start_metrics_collection(self):
        """Start periodic metrics collection"""
        def collect_metrics():
            if not self._shutdown_event.is_set():
                try:
                    self._metrics_collector.collect_metrics(self)
                    self._metrics_timer = Timer(self.stream_config.metrics_interval, collect_metrics)
                    self._metrics_timer.start()
                except Exception as e:
                    logger.error(f"Error collecting metrics: {e}")
        
        self._metrics_timer = Timer(self.stream_config.metrics_interval, collect_metrics)
        self._metrics_timer.start()
    
    def _start_health_monitoring(self):
        """Start health monitoring"""
        def check_health():
            if not self._shutdown_event.is_set():
                try:
                    self._health_monitor.check_health(self)
                    self._health_check_timer = Timer(self.stream_config.health_check_interval, check_health)
                    self._health_check_timer.start()
                except Exception as e:
                    logger.error(f"Error in health check: {e}")
        
        self._health_check_timer = Timer(self.stream_config.health_check_interval, check_health)
        self._health_check_timer.start()
    
    def _start_memory_monitoring(self):
        """Start memory monitoring"""
        def monitor_memory():
            if not self._shutdown_event.is_set():
                try:
                    if self._memory_monitor.check_memory_pressure():
                        self._handle_memory_pressure()
                    self._memory_cleanup_timer = Timer(self.stream_config.memory_check_interval, monitor_memory)
                    self._memory_cleanup_timer.start()
                except Exception as e:
                    logger.error(f"Error monitoring memory: {e}")
        
        self._memory_cleanup_timer = Timer(self.stream_config.memory_check_interval, monitor_memory)
        self._memory_cleanup_timer.start()
    
    def _handle_memory_pressure(self):
        """Handle memory pressure situations"""
        logger.warning("Memory pressure detected, initiating cleanup")
        
        # Force garbage collection
        gc.collect()
        
        # Clear caches if available
        if hasattr(self.data_loader, 'clear_cache'):
            self.data_loader.clear_cache()
        
        # Reduce queue sizes temporarily
        if self._stream_queue.qsize() > self.stream_config.max_queue_size // 2:
            logger.info("Reducing queue size due to memory pressure")
            # Process some items from queue
            items_to_process = min(10, self._stream_queue.qsize())
            for _ in range(items_to_process):
                try:
                    self._stream_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Log memory usage after cleanup
        current_usage = self._memory_monitor.get_current_usage()
        logger.info(f"Memory usage after cleanup: {current_usage:.2f} MB")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup"""
        self._cleanup()
        return False
    
    def __del__(self):
        """Destructor with proper resource cleanup"""
        try:
            self._cleanup()
        except Exception as e:
            # Use print instead of logger in case logging is already shut down
            print(f"Error during DataStreamer cleanup: {e}")
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Comprehensive cleanup of all resources"""
        logger.info("Starting DataStreamer cleanup...")
        
        try:
            # Signal shutdown to all monitoring threads
            self._shutdown_event.set()
            
            # Stop streaming
            self.stop_streaming()
            
            # Stop all timers
            if self._metrics_timer:
                self._metrics_timer.cancel()
            if self._health_check_timer:
                self._health_check_timer.cancel()
            if self._memory_cleanup_timer:
                self._memory_cleanup_timer.cancel()
            
            # Close connection pool
            self._connection_pool.close_all()
            
            # Clear queues
            self._clear_queues()
            
            # Final garbage collection
            gc.collect()
            
            logger.info("DataStreamer cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Continue with cleanup even if there are errors
    
    def _clear_queues(self):
        """Clear all queues safely"""
        try:
            while not self._stream_queue.empty():
                try:
                    self._stream_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            logger.warning(f"Error clearing stream queue: {e}")
        
        try:
            while not self._error_queue.empty():
                try:
                    self._error_queue.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            logger.warning(f"Error clearing error queue: {e}")
        self._rate_limiter = RateLimiter(self.stream_config.rate_limit_config)
        self._flow_controller = FlowController(self.stream_config.flow_control_config)
        self._system_metrics = SystemMetricsCollector()
        
        # Backward compatibility - keep old memory monitor interface
        self._legacy_memory_monitor = MemoryMonitor(self.stream_config.memory_threshold_mb)
        
        # Start system metrics collection
        self._system_metrics.start_collection()
        
        # Cleanup on exit
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.stop_streaming()
        self._system_metrics.stop_collection()
        logger.info("DataStreamer cleanup completed")
    
    def stream_file(self, file_path: str, 
                   transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                   filter_func: Optional[Callable[[pd.DataFrame], bool]] = None,
                   **kwargs) -> Iterator[DataChunk]:
        """
        Stream data from a file with optional transformation and filtering.
        
        This method provides production-ready streaming with:
        - Comprehensive error handling and recovery
        - Memory management and monitoring
        - Connection resilience with circuit breaker
        - Retry logic with exponential backoff
        - Performance metrics collection
        
        Args:
            file_path: Path to the file to stream
            transform_func: Optional transformation function for data
            filter_func: Optional filter function for data
            **kwargs: Additional arguments passed to data loader
        
        Yields:
            DataChunk: Processed data chunks
        
        Raises:
            DataStreamingException: If streaming fails after all retries
        """
        operation_id = f"stream_file_{int(time.time())}"
        logger.info(f"Starting file streaming operation {operation_id} for {file_path}")
        
        try:
            # Use retry manager for resilient operation
            yield from self._retry_manager.execute_with_retry(
                self._stream_file_internal,
                file_path,
                transform_func,
                filter_func,
                operation_id,
                **kwargs
            )
        except Exception as e:
            logger.error(f"File streaming operation {operation_id} failed: {str(e)}")
            self._metrics_collector.record_error(operation_id, e)
            raise DataStreamingException(f"Failed to stream file {file_path}: {str(e)}")
        finally:
            logger.info(f"File streaming operation {operation_id} completed")
    
    def _stream_file_internal(self, file_path: str, 
                            transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                            filter_func: Optional[Callable[[pd.DataFrame], bool]] = None,
                            operation_id: str = None,
                            **kwargs) -> Iterator[DataChunk]:
        """
        Internal file streaming with circuit breaker protection
        """
        with self._lock:
            if self._is_streaming:
                logger.warning("Streaming already in progress, waiting...")
                return
            
            self._is_streaming = True
            self._stats.reset()
            
        try:
            # Track resources
            self._resource_tracker.track_operation(operation_id or "stream_file")
            
            # Use circuit breaker for resilient streaming
            def stream_with_circuit_breaker():
                chunk_count = 0
                
                for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                    # Check for shutdown signal
                    if self._shutdown_event.is_set():
                        logger.info("Shutdown signal received, stopping streaming")
                        break
                    
                    # Memory management
                    if self._memory_monitor.check_memory_pressure():
                        logger.warning("Memory pressure detected, triggering cleanup")
                        self._handle_memory_pressure()
                    
                    # Process chunk with error handling
                    try:
                        processed_chunk = self._process_chunk_safe(chunk, transform_func, filter_func)
                        
                        if processed_chunk is not None:
                            # Update statistics
                            self._stats.update(processed_chunk)
                            
                            # Record metrics
                            self._metrics_collector.record_chunk_processed(processed_chunk)
                            
                            chunk_count += 1
                            yield processed_chunk
                    
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
                        self._error_queue.put((chunk.chunk_id, e))
                        
                        # Continue processing other chunks
                        continue
                    
                    # Check for streaming stop
                    if not self._is_streaming:
                        logger.info("Streaming stopped by user request")
                        break
                    
                    # Yield control periodically
                    if chunk_count % 100 == 0:
                        time.sleep(0.001)  # Small sleep to prevent CPU hogging
                
                logger.info(f"Processed {chunk_count} chunks from {file_path}")
            
            # Execute with circuit breaker protection
            yield from self._circuit_breaker.call(stream_with_circuit_breaker)
            
        except Exception as e:
            logger.error(f"Error in internal streaming for {file_path}: {str(e)}")
            raise
        finally:
            self._is_streaming = False
            self._resource_tracker.untrack_operation(operation_id or "stream_file")
    
    def _process_chunk_safe(self, chunk: DataChunk,
                          transform_func: Optional[Callable] = None,
                          filter_func: Optional[Callable] = None) -> Optional[DataChunk]:
        """
        Process chunk with comprehensive error handling and resource management
        """
        try:
            # Validate chunk
            if chunk is None or chunk.data is None:
                logger.warning("Received null chunk, skipping")
                return None
            
            if chunk.data.empty:
                logger.debug(f"Empty chunk {chunk.chunk_id}, skipping")
                return None
            
            # Track memory usage before processing
            initial_memory = self._memory_monitor.get_current_usage()
            
            # Process chunk
            processed_chunk = self._process_chunk(chunk, transform_func, filter_func)
            
            # Track memory usage after processing
            final_memory = self._memory_monitor.get_current_usage()
            memory_delta = final_memory - initial_memory
            
            if memory_delta > 50:  # Log significant memory increases
                logger.warning(f"Chunk {chunk.chunk_id} processing increased memory by {memory_delta:.2f} MB")
            
            return processed_chunk
            
        except Exception as e:
            logger.error(f"Error in safe chunk processing: {str(e)}")
            logger.debug(f"Chunk details: {chunk.chunk_id if chunk else 'None'}")
            raise
    
    def stream_multiple_files(self, file_paths: List[str],
                            transform_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                            filter_func: Optional[Callable[[pd.DataFrame], bool]] = None,
                            parallel: bool = True,
                            **kwargs) -> Iterator[DataChunk]:
        """
        Stream data from multiple files
        """
        if parallel and self.config.enable_parallel_processing:
            yield from self._stream_multiple_files_parallel(
                file_paths, transform_func, filter_func, **kwargs
            )
        else:
            yield from self._stream_multiple_files_sequential(
                file_paths, transform_func, filter_func, **kwargs
            )
    
    def _stream_multiple_files_sequential(self, file_paths: List[str],
                                        transform_func: Optional[Callable] = None,
                                        filter_func: Optional[Callable] = None,
                                        **kwargs) -> Iterator[DataChunk]:
        """Stream multiple files sequentially"""
        for file_path in file_paths:
            yield from self.stream_file(file_path, transform_func, filter_func, **kwargs)
    
    def _stream_multiple_files_parallel(self, file_paths: List[str],
                                      transform_func: Optional[Callable] = None,
                                      filter_func: Optional[Callable] = None,
                                      **kwargs) -> Iterator[DataChunk]:
        """Stream multiple files in parallel"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit streaming tasks
            futures = [
                executor.submit(self._stream_file_to_queue, file_path, transform_func, filter_func, **kwargs)
                for file_path in file_paths
            ]
            
            # Collect results from queue
            active_futures = set(futures)
            while active_futures:
                try:
                    chunk = self._stream_queue.get(timeout=self.stream_config.timeout_seconds)
                    if chunk is None:  # End of stream marker
                        break
                    yield chunk
                    self._stream_queue.task_done()
                except queue.Empty:
                    # Check if all futures are done
                    active_futures = {f for f in active_futures if not f.done()}
                    if not active_futures:
                        break
                    continue
                except Exception as e:
                    logger.error(f"Error in parallel streaming: {str(e)}")
                    break
    
    def _stream_file_to_queue(self, file_path: str,
                            transform_func: Optional[Callable] = None,
                            filter_func: Optional[Callable] = None,
                            **kwargs):
        """Stream file chunks to queue"""
        try:
            for chunk in self.stream_file(file_path, transform_func, filter_func, **kwargs):
                self._stream_queue.put(chunk)
        except Exception as e:
            self._error_queue.put(e)
        finally:
            self._stream_queue.put(None)  # End of stream marker
    
    def _process_chunk(self, chunk: DataChunk,
                      transform_func: Optional[Callable] = None,
                      filter_func: Optional[Callable] = None) -> Optional[DataChunk]:
        """
        Process a data chunk with transformation and filtering.
        
        Enhanced with:
        - Memory usage tracking
        - Error handling with detailed logging
        - Performance monitoring
        - Resource leak prevention
        
        Args:
            chunk: Data chunk to process
            transform_func: Optional transformation function
            filter_func: Optional filter function
        
        Returns:
            Optional[DataChunk]: Processed chunk or None if filtered out
        """
        start_time = time.time()
        
        try:
            if chunk is None or chunk.data is None:
                logger.warning("Received null chunk for processing")
                return None
            
            data = chunk.data.copy()  # Create copy to avoid modifying original
            original_memory = data.memory_usage(deep=True).sum()
            
            # Apply filter first (to reduce data size early)
            if filter_func:
                try:
                    if not filter_func(data):
                        logger.debug(f"Chunk {chunk.chunk_id} filtered out")
                        return None
                except Exception as e:
                    logger.error(f"Error applying filter to chunk {chunk.chunk_id}: {str(e)}")
                    # Continue without filtering on error
            
            # Apply transformation
            if transform_func:
                try:
                    transformed_data = transform_func(data)
                    if transformed_data is None:
                        logger.debug(f"Chunk {chunk.chunk_id} transformed to None")
                        return None
                    
                    if hasattr(transformed_data, 'empty') and transformed_data.empty:
                        logger.debug(f"Chunk {chunk.chunk_id} transformed to empty")
                        return None
                    
                    data = transformed_data
                    
                except Exception as e:
                    logger.error(f"Error applying transformation to chunk {chunk.chunk_id}: {str(e)}")
                    logger.debug(f"Transformation error details: {traceback.format_exc()}")
                    # Continue with original data on transformation error
            
            # Calculate memory usage
            processed_memory = data.memory_usage(deep=True).sum()
            memory_delta = processed_memory - original_memory
            
            # Create new chunk with processed data
            processed_chunk = DataChunk(
                data=data,
                chunk_id=chunk.chunk_id,
                start_row=chunk.start_row,
                end_row=chunk.end_row,
                file_path=chunk.file_path,
                timestamp=time.time(),
                memory_usage=processed_memory
            )
            
            # Log processing metrics
            processing_time = time.time() - start_time
            if processing_time > 1.0:  # Log slow processing
                logger.warning(f"Slow chunk processing: {processing_time:.2f}s for chunk {chunk.chunk_id}")
            
            if memory_delta > 1024 * 1024:  # Log significant memory increases (>1MB)
                logger.warning(f"Chunk {chunk.chunk_id} processing increased memory by {memory_delta/1024/1024:.2f} MB")
            
            logger.debug(f"Processed chunk {chunk.chunk_id}: {len(data)} rows, {processing_time:.3f}s")
            
            return processed_chunk
            
        except Exception as e:
            logger.error(f"Critical error processing chunk {chunk.chunk_id if chunk else 'None'}: {str(e)}")
            logger.debug(f"Processing error details: {traceback.format_exc()}")
            
            # Record error for monitoring
            self._metrics_collector.record_processing_error(chunk.chunk_id if chunk else 'unknown', e)
            
            return None
        finally:
            # Ensure any temporary variables are cleaned up
            if 'data' in locals():
                del data
            if 'transformed_data' in locals():
                del transformed_data
    
    def _process_chunk_enhanced(self, chunk: DataChunk,
                               transform_func: Optional[Callable] = None,
                               filter_func: Optional[Callable] = None,
                               priority: Priority = Priority.NORMAL) -> Optional[DataChunk]:
        """Enhanced chunk processing with flow control and monitoring"""
        start_time = time.time()
        
        try:
            # Check if we should apply backpressure
            queue_size = self._stream_queue.qsize()
            consumer_throughput = self._stats.get_throughput()
            
            if self._backpressure_controller.should_apply_backpressure(
                queue_size, self.stream_config.max_queue_size, consumer_throughput
            ):
                self._stats.record_backpressure_event()
                # Apply backpressure by briefly pausing
                backpressure_signal = self._backpressure_controller.get_backpressure_signal(
                    queue_size, self.stream_config.max_queue_size
                )
                time.sleep(backpressure_signal * 0.1)  # Scale pause by signal strength
            
            # Process through flow controller
            if not self._flow_controller.enqueue_chunk(chunk, priority):
                return None  # Chunk was dropped due to load shedding
            
            # Get chunk from flow controller (with priority ordering)
            flow_chunk = self._flow_controller.dequeue_chunk()
            if flow_chunk is None:
                return None
            
            data = flow_chunk.data
            
            # Apply filter first (to reduce data size)
            if filter_func and not filter_func(data):
                return None
            
            # Apply transformation
            if transform_func:
                data = transform_func(data)
                if data is None or data.empty:
                    return None
            
            # Create new chunk with processed data
            processed_chunk = DataChunk(
                data=data,
                chunk_id=flow_chunk.chunk_id,
                start_row=flow_chunk.start_row,
                end_row=flow_chunk.end_row,
                file_path=flow_chunk.file_path,
                timestamp=time.time(),
                memory_usage=0  # Will be calculated
            )
            
            # Record performance metrics
            processing_time = time.time() - start_time
            self._rate_limiter.record_performance(True, processing_time)
            
            return processed_chunk
            
        except Exception as e:
            self._stats.record_error()
            processing_time = time.time() - start_time
            self._rate_limiter.record_performance(False, processing_time)
            logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
            return None
    
    def batch_stream(self, file_paths: List[str],
                    batch_size: Optional[int] = None,
                    transform_func: Optional[Callable] = None,
                    **kwargs) -> Iterator[List[DataChunk]]:
        """
        Stream data in batches for more efficient processing
        """
        actual_batch_size = batch_size or self.stream_config.batch_size
        batch = []
        
        for chunk in self.stream_multiple_files(file_paths, transform_func, **kwargs):
            batch.append(chunk)
            
            if len(batch) >= actual_batch_size:
                yield batch
                batch = []
        
        # Yield remaining batch
        if batch:
            yield batch
    
    def stream_with_priority(self, file_path: str,
                           priority: Priority = Priority.NORMAL,
                           transform_func: Optional[Callable] = None,
                           filter_func: Optional[Callable] = None,
                           **kwargs) -> Iterator[DataChunk]:
        """Stream data with priority level for flow control"""
        return self.stream_file(file_path, transform_func, filter_func, priority, **kwargs)
    
    def configure_backpressure(self, **kwargs):
        """Configure backpressure settings dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.stream_config.backpressure_config, key):
                setattr(self.stream_config.backpressure_config, key, value)
                logger.info(f"Backpressure config updated: {key} = {value}")
    
    def configure_rate_limiting(self, **kwargs):
        """Configure rate limiting settings dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.stream_config.rate_limit_config, key):
                setattr(self.stream_config.rate_limit_config, key, value)
                logger.info(f"Rate limit config updated: {key} = {value}")
    
    def configure_flow_control(self, **kwargs):
        """Configure flow control settings dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.stream_config.flow_control_config, key):
                setattr(self.stream_config.flow_control_config, key, value)
                logger.info(f"Flow control config updated: {key} = {value}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the streaming system"""
        memory_stats = self._memory_monitor.get_memory_stats()
        flow_stats = self._flow_controller.get_flow_control_status()
        
        # Determine health status
        health_score = 1.0
        issues = []
        
        # Check memory utilization
        if memory_stats['utilization_percent'] > 90:
            health_score -= 0.3
            issues.append("High memory utilization")
        
        # Check error rate
        error_rate = self._stats.get_error_rate()
        if error_rate > 0.1:  # More than 0.1 errors per second
            health_score -= 0.2
            issues.append("High error rate")
        
        # Check if load shedding is active
        if flow_stats['load_shedding_active']:
            health_score -= 0.2
            issues.append("Load shedding active")
        
        # Check congestion
        if flow_stats['congestion_level'] > 0.7:
            health_score -= 0.1
            issues.append("High congestion")
        
        # Determine overall status
        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.5:
            status = "warning"
        else:
            status = "critical"
        
        return {
            'status': status,
            'health_score': max(0.0, health_score),
            'issues': issues,
            'recommendations': self._get_health_recommendations(issues)
        }
    
    def _get_health_recommendations(self, issues: List[str]) -> List[str]:
        """Get health recommendations based on issues"""
        recommendations = []
        
        for issue in issues:
            if "memory" in issue.lower():
                recommendations.append("Consider reducing batch size or enabling more aggressive GC")
            elif "error" in issue.lower():
                recommendations.append("Check data quality and transformation functions")
            elif "load shedding" in issue.lower():
                recommendations.append("Consider scaling up resources or reducing input rate")
            elif "congestion" in issue.lower():
                recommendations.append("Optimize processing pipeline or add more workers")
        
        return recommendations
    
    def windowed_stream(self, file_path: str, window_size: int,
                       step_size: Optional[int] = None,
                       transform_func: Optional[Callable] = None,
                       **kwargs) -> Iterator[DataChunk]:
        """
        Stream data in sliding windows
        """
        step_size = step_size or window_size
        window_buffer = []
        
        for chunk in self.stream_file(file_path, transform_func, **kwargs):
            window_buffer.append(chunk.data)
            
            # Check if window is full
            total_rows = sum(len(df) for df in window_buffer)
            if total_rows >= window_size:
                # Combine data in window
                combined_data = pd.concat(window_buffer, ignore_index=True)
                
                # Create windowed chunk
                windowed_chunk = DataChunk(
                    data=combined_data.head(window_size),
                    chunk_id=chunk.chunk_id,
                    start_row=chunk.start_row,
                    end_row=chunk.start_row + window_size,
                    file_path=chunk.file_path,
                    timestamp=time.time(),
                    memory_usage=0
                )
                
                yield windowed_chunk
                
                # Slide window
                remaining_rows = total_rows - step_size
                if remaining_rows > 0:
                    # Keep remaining data for next window
                    combined_data = combined_data.tail(remaining_rows)
                    window_buffer = [combined_data]
                else:
                    window_buffer = []
    
    def time_based_stream(self, file_path: str, time_column: str,
                         window_duration: str,
                         transform_func: Optional[Callable] = None,
                         **kwargs) -> Iterator[DataChunk]:
        """
        Stream data in time-based windows
        """
        current_window_start = None
        window_buffer = []
        
        for chunk in self.stream_file(file_path, transform_func, **kwargs):
            data = chunk.data
            
            # Ensure time column is datetime
            if time_column in data.columns:
                data[time_column] = pd.to_datetime(data[time_column])
                
                # Initialize window start
                if current_window_start is None:
                    current_window_start = data[time_column].min()
                
                # Calculate window end
                window_end = current_window_start + pd.Timedelta(window_duration)
                
                # Filter data within window
                window_data = data[data[time_column] < window_end]
                remaining_data = data[data[time_column] >= window_end]
                
                if not window_data.empty:
                    window_buffer.append(window_data)
                
                # If we have data beyond the window, yield current window
                if not remaining_data.empty:
                    if window_buffer:
                        combined_data = pd.concat(window_buffer, ignore_index=True)
                        
                        window_chunk = DataChunk(
                            data=combined_data,
                            chunk_id=chunk.chunk_id,
                            start_row=chunk.start_row,
                            end_row=chunk.start_row + len(combined_data),
                            file_path=chunk.file_path,
                            timestamp=time.time(),
                            memory_usage=0
                        )
                        
                        yield window_chunk
                    
                    # Start new window
                    current_window_start = remaining_data[time_column].min()
                    window_buffer = [remaining_data]
            else:
                # No time column, just buffer the data
                window_buffer.append(data)
        
        # Yield final window if any data remains
        if window_buffer:
            combined_data = pd.concat(window_buffer, ignore_index=True)
            final_chunk = DataChunk(
                data=combined_data,
                chunk_id=999999,  # Final chunk marker
                start_row=0,
                end_row=len(combined_data),
                file_path=file_path,
                timestamp=time.time(),
                memory_usage=0
            )
            yield final_chunk
    
    def stop_streaming(self):
        """
        Stop the streaming process gracefully.
        
        This method ensures:
        - All active operations are signaled to stop
        - Resources are cleaned up properly
        - Queues are drained safely
        - Metrics are finalized
        """
        logger.info("Stopping data streaming...")
        
        with self._lock:
            self._is_streaming = False
            
        # Signal shutdown to monitoring threads
        self._shutdown_event.set()
        
        # Wait for any active operations to complete
        max_wait_time = 10.0  # Maximum wait time in seconds
        wait_start = time.time()
        
        while (time.time() - wait_start) < max_wait_time:
            if not self._resource_tracker.has_active_operations():
                break
            time.sleep(0.1)
        
        # Force cleanup if operations are still active
        if self._resource_tracker.has_active_operations():
            logger.warning("Forcing cleanup of active operations")
            self._resource_tracker.force_cleanup()
        
        # Clear queues
        self._clear_queues()
        
        logger.info("Data streaming stopped successfully")
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive streaming statistics.
        
        Returns:
            Dict[str, Any]: Comprehensive statistics including:
            - Basic streaming stats
            - Memory usage and monitoring
            - Connection health and metrics
            - Performance metrics
            - Error statistics
        """
        try:
            connection_stats = self._connection_pool.stats.to_dict()
            health_stats = self._health_monitor.get_health_stats()
            
            return {
                # Basic streaming stats
                'is_streaming': self._is_streaming,
                'chunks_processed': self._stats.chunks_processed,
                'total_rows': self._stats.total_rows,
                'bytes_processed': self._stats.bytes_processed,
                'avg_processing_time': self._stats.get_avg_processing_time(),
                'throughput_rows_per_sec': self._stats.get_throughput(),
                
                # Memory statistics
                'memory_usage_mb': self._memory_monitor.get_current_usage(),
                'peak_memory_mb': self._memory_monitor.get_peak_usage(),
                'memory_pressure_events': self._memory_monitor.get_pressure_events(),
                'gc_collections': self._memory_monitor.get_gc_stats(),
                
                # Connection statistics
                'connection_pool': connection_stats,
                'circuit_breaker_state': self._circuit_breaker.state.value,
                'retry_count': self._retry_manager.retry_count,
                
                # Health monitoring
                'health_status': health_stats,
                
                # Queue statistics
                'queue_size': self._stream_queue.qsize(),
                'max_queue_size': self.stream_config.max_queue_size,
                'error_queue_size': self._error_queue.qsize(),
                
                # Performance metrics
                'active_operations': self._resource_tracker.get_active_operations(),
                'total_operations': self._resource_tracker.get_total_operations(),
                
                # Configuration
                'config': {
                    'buffer_size': self.stream_config.buffer_size,
                    'memory_threshold_mb': self.stream_config.memory_threshold_mb,
                    'max_retries': self.stream_config.max_retries,
                    'pool_size': self.stream_config.pool_size,
                    'enable_metrics': self.stream_config.enable_metrics
                },
                
                # Timestamps
                'last_updated': datetime.now().isoformat(),
                'uptime_seconds': time.time() - self._stats.start_time
            }
        except Exception as e:
            logger.error(f"Error getting streaming stats: {str(e)}")
            return {
                'error': str(e),
                'is_streaming': self._is_streaming,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status of the streamer.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        return {
            'healthy': self._health_monitor.is_healthy(),
            'circuit_breaker_state': self._circuit_breaker.state.value,
            'memory_pressure': self._memory_monitor.check_memory_pressure(),
            'active_connections': self._connection_pool.stats.active_connections,
            'error_rate': self._metrics_collector.get_error_rate(),
            'last_check': datetime.now().isoformat()
        }
    
    def reset_circuit_breaker(self):
        """
        Reset the circuit breaker to closed state.
        
        This should be used carefully and only when you're sure
        the underlying issues have been resolved.
        """
        logger.info("Resetting circuit breaker")
        self._circuit_breaker.reset()
    
    def clear_error_queue(self) -> List[tuple]:
        """
        Clear and return all errors from the error queue.
        
        Returns:
            List[tuple]: List of (chunk_id, error) tuples
        """
        errors = []
        while not self._error_queue.empty():
            try:
                errors.append(self._error_queue.get_nowait())
            except queue.Empty:
                break
        return errors
    
    def _get_basic_stats(self) -> Dict[str, Any]:
        """Get basic stats for backward compatibility"""
        return {
            'is_streaming': self._is_streaming,
            'chunks_processed': self._stats.chunks_processed,
            'total_rows': self._stats.total_rows,
            'bytes_processed': self._stats.bytes_processed,
            'avg_processing_time': self._stats.get_avg_processing_time(),
            'throughput_rows_per_sec': self._stats.get_throughput(),
            'memory_usage_mb': self._memory_monitor.get_current_usage(),
            'peak_memory_mb': self._memory_monitor.get_peak_usage()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring dashboards"""
        return {
            'throughput': self._stats.get_throughput(),
            'latency_percentiles': self._stats.get_latency_percentiles(),
            'error_rate': self._stats.get_error_rate(),
            'memory_utilization': self._memory_monitor.get_memory_stats()['utilization_percent'],
            'backpressure_active': self._backpressure_controller.current_threshold < 0.8,
            'rate_limiting_active': self._rate_limiter.config.enable_rate_limiting,
            'load_shedding_active': self._flow_controller.load_shedding_active,
            'system_health': self._system_metrics.get_metrics_summary(60)
        }




# Async streaming support
class AsyncDataStreamer:
    """Async version of DataStreamer for better concurrency"""
    
    def __init__(self, config: Optional[DataPipelineConfig] = None):
        self.config = config or DataPipelineConfig()
        self.data_loader = ScalableDataLoader(self.config)
        self._semaphore = asyncio.Semaphore(self.config.max_workers)
    
    async def stream_file_async(self, file_path: str,
                              transform_func: Optional[Callable] = None,
                              **kwargs) -> AsyncIterator[DataChunk]:
        """Async version of stream_file"""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            
            # Run blocking operations in thread pool
            chunks = await loop.run_in_executor(
                None, 
                lambda: list(self.data_loader.load_chunks(file_path, **kwargs))
            )
            
            for chunk in chunks:
                if transform_func:
                    # Run transformation in thread pool
                    transformed_data = await loop.run_in_executor(
                        None, transform_func, chunk.data
                    )
                    chunk.data = transformed_data
                
                yield chunk
    
    async def stream_multiple_files_async(self, file_paths: List[str],
                                        transform_func: Optional[Callable] = None,
                                        **kwargs) -> AsyncIterator[DataChunk]:
        """Async version of stream_multiple_files"""
        tasks = [
            self.stream_file_async(file_path, transform_func, **kwargs)
            for file_path in file_paths
        ]
        
        async for chunk in self._merge_async_streams(tasks):
            yield chunk
    
    async def _merge_async_streams(self, streams: List[AsyncIterator[DataChunk]]) -> AsyncIterator[DataChunk]:
        """Merge multiple async streams"""
        # This is a simplified implementation
        # In production, you'd want a more sophisticated merging strategy
        for stream in streams:
            async for chunk in stream:
                yield chunk


# Legacy StreamingStats class for backward compatibility
class StreamingStats:
    """Legacy streaming statistics for backward compatibility"""
    
    def __init__(self):
        self.chunks_processed = 0
        self.total_rows = 0
        self.bytes_processed = 0
        self.processing_times = []
        self.start_time = time.time()
    
    def update(self, chunk: DataChunk):
        """Update statistics with a processed chunk"""
        self.chunks_processed += 1
        self.total_rows += len(chunk.data)
        self.bytes_processed += chunk.data.memory_usage(deep=True).sum()
        
        # Track processing time
        processing_time = time.time() - chunk.timestamp
        self.processing_times.append(processing_time)
        
        # Keep only recent processing times (last 100)
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time per chunk"""
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def get_throughput(self) -> float:
        """Get throughput in rows per second"""
        elapsed = time.time() - self.start_time
        return self.total_rows / elapsed if elapsed > 0 else 0.0
    
    def reset(self):
        """Reset statistics"""
        self.chunks_processed = 0
        self.total_rows = 0
        self.bytes_processed = 0
        self.processing_times = []
        self.start_time = time.time()

# Legacy MemoryMonitor class for backward compatibility
class MemoryMonitor:
    """Legacy memory monitoring for backward compatibility"""
    
    def __init__(self, threshold_mb: float):
        self.threshold_mb = threshold_mb
        self.peak_usage = 0.0
        self.current_usage = 0.0
    
    def check_memory_pressure(self) -> bool:
        """Check if memory usage is under pressure"""
        self.current_usage = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_usage = max(self.peak_usage, self.current_usage)
        return self.current_usage > self.threshold_mb
    
    def get_current_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.current_usage
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage in MB"""
        return self.peak_usage

# Utility functions
# Production-ready decorators
def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function calls on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

def log_performance(func):
    """
    Decorator to log function performance.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed successfully in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    return wrapper

def monitor_memory(threshold_mb: float = 100.0):
    """
    Decorator to monitor memory usage of functions.
    
    Args:
        threshold_mb: Memory threshold to log warnings
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = func(*args, **kwargs)
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = final_memory - initial_memory
                
                if memory_delta > threshold_mb:
                    logger.warning(f"{func.__name__} used {memory_delta:.2f} MB of memory")
                
                return result
            except Exception as e:
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_delta = final_memory - initial_memory
                logger.error(f"{func.__name__} failed and used {memory_delta:.2f} MB of memory")
                raise
        return wrapper
    return decorator

# Production-ready utility functions
def create_time_filter(start_time: str, end_time: str, 
                      time_column: str = 'timestamp') -> Callable[[pd.DataFrame], bool]:
    """
    Create a time-based filter function with enhanced error handling.
    
    Args:
        start_time: Start time string (ISO format or pandas parseable)
        end_time: End time string (ISO format or pandas parseable)
        time_column: Name of the time column to filter on
    
    Returns:
        Callable filter function that returns True if data should be included
    
    Raises:
        ValueError: If time strings cannot be parsed
    """
    try:
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
    except Exception as e:
        raise ValueError(f"Invalid time format: {str(e)}")
    
    if start_dt >= end_dt:
        raise ValueError("Start time must be before end time")
    
    def filter_func(df: pd.DataFrame) -> bool:
        try:
            if time_column not in df.columns:
                logger.warning(f"Time column '{time_column}' not found in data")
                return True
            
            df_time = pd.to_datetime(df[time_column], errors='coerce')
            
            # Check for invalid timestamps
            if df_time.isna().all():
                logger.warning(f"All timestamps in column '{time_column}' are invalid")
                return True
            
            # Filter data within time range
            mask = df_time.between(start_dt, end_dt)
            return mask.any()
            
        except Exception as e:
            logger.error(f"Error in time filter: {str(e)}")
            return True  # Include data on error to avoid data loss
    
    return filter_func

def create_column_selector(columns: List[str], 
                          strict: bool = False) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a column selection transform function with enhanced error handling.
    
    Args:
        columns: List of column names to select
        strict: If True, raise error if any column is missing
    
    Returns:
        Callable transform function
    
    Raises:
        ValueError: If strict=True and columns are missing
    """
    if not columns:
        raise ValueError("Column list cannot be empty")
    
    def transform_func(df: pd.DataFrame) -> pd.DataFrame:
        try:
            available_cols = [col for col in columns if col in df.columns]
            missing_cols = [col for col in columns if col not in df.columns]
            
            if missing_cols:
                if strict:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                else:
                    logger.warning(f"Missing columns (ignoring): {missing_cols}")
            
            if not available_cols:
                logger.warning("No matching columns found, returning original DataFrame")
                return df
            
            selected_df = df[available_cols].copy()
            logger.debug(f"Selected {len(available_cols)} columns from {len(df.columns)} total")
            
            return selected_df
            
        except Exception as e:
            logger.error(f"Error in column selector: {str(e)}")
            return df  # Return original data on error
    
    return transform_func

def create_sampling_transform(sample_rate: float, 
                            random_state: Optional[int] = None) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a sampling transform function with enhanced error handling.
    
    Args:
        sample_rate: Sampling rate (0.0 to 1.0)
        random_state: Random seed for reproducible sampling
    
    Returns:
        Callable transform function
    
    Raises:
        ValueError: If sample_rate is invalid
    """
    if not 0.0 <= sample_rate <= 1.0:
        raise ValueError(f"Sample rate must be between 0.0 and 1.0, got {sample_rate}")
    
    def transform_func(df: pd.DataFrame) -> pd.DataFrame:
        try:
            if sample_rate >= 1.0:
                return df
            
            if df.empty:
                logger.debug("Empty DataFrame, returning as-is")
                return df
            
            # Calculate number of samples
            n_samples = max(1, int(len(df) * sample_rate))
            
            if n_samples >= len(df):
                return df
            
            sampled_df = df.sample(n=n_samples, random_state=random_state)
            logger.debug(f"Sampled {len(sampled_df)} rows from {len(df)} total")
            
            return sampled_df
            
        except Exception as e:
            logger.error(f"Error in sampling transform: {str(e)}")
            return df  # Return original data on error
    
    return transform_func

def create_memory_efficient_transform(chunk_size: int = 1000) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a memory-efficient transform that processes data in chunks.
    
    Args:
        chunk_size: Size of chunks to process
    
    Returns:
        Callable transform function
    """
    def transform_func(df: pd.DataFrame) -> pd.DataFrame:
        try:
            if len(df) <= chunk_size:
                return df
            
            # Process in chunks
            processed_chunks = []
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                # Add any processing logic here
                processed_chunks.append(chunk)
            
            return pd.concat(processed_chunks, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error in memory-efficient transform: {str(e)}")
            return df
    
    return transform_func

def create_error_resilient_transform(transform_func: Callable[[pd.DataFrame], pd.DataFrame],
                                   fallback_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
                                   ) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create an error-resilient wrapper for transform functions.
    
    Args:
        transform_func: Primary transform function
        fallback_func: Fallback function if primary fails
    
    Returns:
        Callable transform function with error handling
    """
    def resilient_transform(df: pd.DataFrame) -> pd.DataFrame:
        try:
            return transform_func(df)
        except Exception as e:
            logger.error(f"Primary transform failed: {str(e)}")
            
            if fallback_func:
                try:
                    logger.info("Attempting fallback transform")
                    return fallback_func(df)
                except Exception as fallback_e:
                    logger.error(f"Fallback transform also failed: {str(fallback_e)}")
            
            # Return original data if all transforms fail
            logger.warning("All transforms failed, returning original data")
            return df
    
    return resilient_transform

# Production-ready context managers
@contextmanager
def streaming_context(streamer: DataStreamer):
    """
    Context manager for safe streaming operations.
    
    Args:
        streamer: DataStreamer instance
    
    Yields:
        DataStreamer: The streamer instance
    """
    try:
        logger.info("Starting streaming context")
        yield streamer
    except Exception as e:
        logger.error(f"Error in streaming context: {str(e)}")
        raise
    finally:
        logger.info("Cleaning up streaming context")
        streamer.stop_streaming()

@contextmanager
def memory_monitoring_context(memory_limit_mb: float = 1024.0):
    """
    Context manager for monitoring memory usage during operations.
    
    Args:
        memory_limit_mb: Memory limit in MB
    
    Yields:
        dict: Memory monitoring information
    """
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    monitoring_info = {
        'initial_memory_mb': initial_memory,
        'limit_mb': memory_limit_mb,
        'peak_memory_mb': initial_memory
    }
    
    try:
        yield monitoring_info
    finally:
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        monitoring_info['final_memory_mb'] = final_memory
        monitoring_info['memory_delta_mb'] = final_memory - initial_memory
        
        if final_memory > memory_limit_mb:
            logger.warning(f"Memory limit exceeded: {final_memory:.2f} MB > {memory_limit_mb} MB")
        
        logger.info(f"Memory usage: {initial_memory:.2f} MB -> {final_memory:.2f} MB (Δ {final_memory - initial_memory:.2f} MB)")

# Production-ready factory functions
def create_production_streamer(
    buffer_size: int = 1000,
    memory_threshold_mb: float = 512.0,
    max_retries: int = 3,
    enable_metrics: bool = True,
    log_level: str = "INFO"
) -> DataStreamer:
    """
    Create a production-ready DataStreamer with sensible defaults.
    
    Args:
        buffer_size: Buffer size for streaming
        memory_threshold_mb: Memory threshold for pressure detection
        max_retries: Maximum retry attempts
        enable_metrics: Enable metrics collection
        log_level: Logging level
    
    Returns:
        DataStreamer: Configured production-ready streamer
    """
    stream_config = StreamConfig(
        buffer_size=buffer_size,
        memory_threshold_mb=memory_threshold_mb,
        max_retries=max_retries,
        enable_metrics=enable_metrics,
        log_level=log_level,
        enable_detailed_logging=True,
        health_check_interval=30.0,
        memory_check_interval=5.0
    )
    
    return DataStreamer(stream_config=stream_config)

def create_high_performance_streamer(
    buffer_size: int = 5000,
    memory_threshold_mb: float = 1024.0,
    pool_size: int = 20,
    max_retries: int = 5
) -> DataStreamer:
    """
    Create a high-performance DataStreamer for large-scale processing.
    
    Args:
        buffer_size: Large buffer size for high throughput
        memory_threshold_mb: Higher memory threshold
        pool_size: Connection pool size
        max_retries: Maximum retry attempts
    
    Returns:
        DataStreamer: Configured high-performance streamer
    """
    stream_config = StreamConfig(
        buffer_size=buffer_size,
        memory_threshold_mb=memory_threshold_mb,
        pool_size=pool_size,
        max_retries=max_retries,
        enable_metrics=True,
        enable_detailed_logging=False,  # Reduce logging overhead
        health_check_interval=15.0,
        memory_check_interval=3.0,
        log_level="WARNING"  # Reduce logging noise
    )
    
    return DataStreamer(stream_config=stream_config)

# Enhanced utility functions
def create_priority_transform(priority: Priority) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create a transform that adds priority metadata"""
    def transform_func(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.attrs['priority'] = priority
        return df
    
    return transform_func

def create_adaptive_sampler(base_rate: float, 
                          load_threshold: float = 0.8) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create an adaptive sampling transform that adjusts based on system load"""
    def transform_func(df: pd.DataFrame) -> pd.DataFrame:
        # Get current system load (simplified)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Calculate adaptive sampling rate
        system_load = max(cpu_percent, memory_percent) / 100.0
        
        if system_load > load_threshold:
            # Reduce sampling rate under high load
            adaptive_rate = base_rate * (1 - (system_load - load_threshold) / (1 - load_threshold))
            adaptive_rate = max(0.1, adaptive_rate)  # Minimum 10% sampling
        else:
            adaptive_rate = base_rate
        
        if adaptive_rate >= 1.0:
            return df
        return df.sample(frac=adaptive_rate)
    
    return transform_func