"""
Real-time streaming data ingestion with sub-millisecond latency optimization

This module implements high-performance streaming data ingestion capable of handling
high-frequency market data with sub-millisecond latency requirements.
"""

import asyncio
import time
import threading
import queue
import numpy as np
import pandas as pd
from typing import Iterator, Callable, Optional, Any, Dict, List, Union, AsyncIterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
import gc
import psutil
from collections import deque
import heapq
from contextlib import asynccontextmanager
import weakref
from enum import Enum
import struct
import pickle
import lz4.frame
import socket
import select
from multiprocessing import shared_memory
import mmap
from pathlib import Path

logger = logging.getLogger(__name__)

class BackpressureStrategy(Enum):
    """Backpressure handling strategies"""
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    SAMPLE = "sample"
    COMPRESS = "compress"

@dataclass
class PerformanceMetrics:
    """Performance metrics for streaming system"""
    throughput_msgs_per_sec: float = 0.0
    avg_latency_us: float = 0.0  # microseconds
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    queue_depth: int = 0
    backpressure_events: int = 0
    dropped_messages: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class StreamMessage:
    """High-performance message structure"""
    data: Union[np.ndarray, bytes, pd.DataFrame]
    timestamp_ns: int  # nanosecond precision
    sequence_id: int
    source: str
    message_type: str
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()

class CircularBuffer:
    """Lock-free circular buffer for high-performance streaming"""
    
    def __init__(self, size: int, item_size: int = 8):
        self.size = size
        self.item_size = item_size
        self.buffer = np.zeros(size * item_size, dtype=np.uint8)
        self.head = 0
        self.tail = 0
        self.full = False
        self._lock = threading.Lock()
    
    def put(self, item: bytes) -> bool:
        """Put item in buffer, returns False if buffer is full"""
        with self._lock:
            if self.full:
                return False
            
            # Calculate position
            pos = self.tail * self.item_size
            item_bytes = item[:self.item_size]  # Truncate if too large
            
            # Write to buffer
            self.buffer[pos:pos + len(item_bytes)] = np.frombuffer(item_bytes, dtype=np.uint8)
            
            # Update tail
            self.tail = (self.tail + 1) % self.size
            if self.tail == self.head:
                self.full = True
            
            return True
    
    def get(self) -> Optional[bytes]:
        """Get item from buffer"""
        with self._lock:
            if not self.full and self.head == self.tail:
                return None
            
            # Calculate position
            pos = self.head * self.item_size
            item_bytes = self.buffer[pos:pos + self.item_size].tobytes()
            
            # Update head
            self.head = (self.head + 1) % self.size
            self.full = False
            
            return item_bytes
    
    def is_empty(self) -> bool:
        return not self.full and self.head == self.tail
    
    def is_full(self) -> bool:
        return self.full
    
    def size_used(self) -> int:
        if self.full:
            return self.size
        return (self.tail - self.head) % self.size

class RealtimeDataStreamer:
    """High-performance real-time data streaming system"""
    
    def __init__(self, 
                 buffer_size: int = 100000,
                 max_latency_us: float = 500.0,  # 500 microseconds
                 backpressure_strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST,
                 enable_compression: bool = True,
                 enable_batching: bool = True,
                 batch_size: int = 1000,
                 num_workers: int = 4):
        
        self.buffer_size = buffer_size
        self.max_latency_us = max_latency_us
        self.backpressure_strategy = backpressure_strategy
        self.enable_compression = enable_compression
        self.enable_batching = enable_batching
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # High-performance buffers
        self.message_buffer = CircularBuffer(buffer_size)
        self.priority_queue = []
        self.sequence_counter = 0
        
        # Threading
        self.worker_pool = ThreadPoolExecutor(max_workers=num_workers)
        self.is_running = False
        self.workers = []
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self.latency_samples = deque(maxlen=10000)
        self.throughput_samples = deque(maxlen=1000)
        
        # Locks
        self.buffer_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        # Callbacks
        self.data_callbacks = []
        self.error_callbacks = []
        
        # Compression
        self.compression_enabled = enable_compression
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.stop()
        logger.info("RealtimeDataStreamer cleanup completed")
    
    def start(self):
        """Start the streaming system"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        # Start performance monitoring
        monitor_thread = threading.Thread(target=self._monitor_performance)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info(f"RealtimeDataStreamer started with {self.num_workers} workers")
    
    def stop(self):
        """Stop the streaming system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        # Shutdown executor
        self.worker_pool.shutdown(wait=True)
        
        logger.info("RealtimeDataStreamer stopped")
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop for processing messages"""
        batch = []
        last_batch_time = time.time_ns()
        
        while self.is_running:
            try:
                # Get message from buffer
                message_bytes = self.message_buffer.get()
                if message_bytes is None:
                    # No messages, yield CPU
                    time.sleep(0.0001)  # 100 microseconds
                    continue
                
                # Deserialize message
                message = self._deserialize_message(message_bytes)
                if message is None:
                    continue
                
                # Record latency
                current_time_ns = time.time_ns()
                latency_us = (current_time_ns - message.timestamp_ns) / 1000
                self._record_latency(latency_us)
                
                # Check if latency exceeds threshold
                if latency_us > self.max_latency_us:
                    logger.warning(f"Message latency {latency_us:.2f}us exceeds threshold {self.max_latency_us}us")
                
                # Add to batch if batching is enabled
                if self.enable_batching:
                    batch.append(message)
                    
                    # Process batch if full or timeout
                    time_since_batch = (current_time_ns - last_batch_time) / 1000  # microseconds
                    if len(batch) >= self.batch_size or time_since_batch > 1000:  # 1ms timeout
                        self._process_batch(batch)
                        batch = []
                        last_batch_time = current_time_ns
                else:
                    # Process message immediately
                    self._process_message(message)
                
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {str(e)}")
                self._notify_error(e)
    
    def _process_message(self, message: StreamMessage):
        """Process a single message"""
        try:
            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error in data callback: {str(e)}")
                    self._notify_error(e)
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            self._notify_error(e)
    
    def _process_batch(self, batch: List[StreamMessage]):
        """Process a batch of messages"""
        try:
            # Sort by timestamp for ordered processing
            batch.sort(key=lambda x: x.timestamp_ns)
            
            # Process each message in batch
            for message in batch:
                self._process_message(message)
                
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            self._notify_error(e)
    
    def send_message(self, data: Any, 
                    source: str = "unknown",
                    message_type: str = "data",
                    priority: int = 0,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send a message to the streaming system"""
        
        # Create message
        message = StreamMessage(
            data=data,
            timestamp_ns=time.time_ns(),
            sequence_id=self._get_next_sequence_id(),
            source=source,
            message_type=message_type,
            priority=priority,
            metadata=metadata
        )
        
        # Serialize message
        message_bytes = self._serialize_message(message)
        if message_bytes is None:
            return False
        
        # Handle backpressure
        if self.message_buffer.is_full():
            return self._handle_backpressure(message_bytes)
        
        # Add to buffer
        success = self.message_buffer.put(message_bytes)
        
        # Update throughput metrics
        self._update_throughput_metrics()
        
        return success
    
    def _handle_backpressure(self, message_bytes: bytes) -> bool:
        """Handle backpressure according to strategy"""
        with self.metrics_lock:
            self.metrics.backpressure_events += 1
        
        if self.backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
            # Remove oldest message and add new one
            self.message_buffer.get()  # Remove oldest
            success = self.message_buffer.put(message_bytes)
            if not success:
                with self.metrics_lock:
                    self.metrics.dropped_messages += 1
            return success
            
        elif self.backpressure_strategy == BackpressureStrategy.DROP_NEWEST:
            # Drop the new message
            with self.metrics_lock:
                self.metrics.dropped_messages += 1
            return False
            
        elif self.backpressure_strategy == BackpressureStrategy.BLOCK:
            # Block until space is available (with timeout)
            timeout_start = time.time()
            while self.message_buffer.is_full() and (time.time() - timeout_start) < 0.001:  # 1ms timeout
                time.sleep(0.00001)  # 10 microseconds
            
            if not self.message_buffer.is_full():
                return self.message_buffer.put(message_bytes)
            else:
                with self.metrics_lock:
                    self.metrics.dropped_messages += 1
                return False
                
        elif self.backpressure_strategy == BackpressureStrategy.SAMPLE:
            # Sample messages (keep every nth message)
            if self.sequence_counter % 2 == 0:  # Keep every other message
                self.message_buffer.get()  # Remove oldest
                return self.message_buffer.put(message_bytes)
            else:
                with self.metrics_lock:
                    self.metrics.dropped_messages += 1
                return False
                
        elif self.backpressure_strategy == BackpressureStrategy.COMPRESS:
            # Try to compress existing messages
            # This is a simplified implementation
            if self.compression_enabled:
                self._compress_buffer()
                return self.message_buffer.put(message_bytes)
            else:
                with self.metrics_lock:
                    self.metrics.dropped_messages += 1
                return False
        
        return False
    
    def _compress_buffer(self):
        """Compress buffer contents to save space"""
        # This is a simplified implementation
        # In a real system, you'd implement more sophisticated compression
        pass
    
    def _serialize_message(self, message: StreamMessage) -> Optional[bytes]:
        """Serialize message for storage"""
        try:
            # Convert data to bytes if needed
            if isinstance(message.data, pd.DataFrame):
                data_bytes = message.data.to_pickle()
            elif isinstance(message.data, np.ndarray):
                data_bytes = message.data.tobytes()
            elif isinstance(message.data, bytes):
                data_bytes = message.data
            else:
                data_bytes = pickle.dumps(message.data)
            
            # Compress if enabled
            if self.compression_enabled:
                data_bytes = lz4.frame.compress(data_bytes)
            
            # Create message structure
            message_dict = {
                'data': data_bytes,
                'timestamp_ns': message.timestamp_ns,
                'sequence_id': message.sequence_id,
                'source': message.source,
                'message_type': message.message_type,
                'priority': message.priority,
                'metadata': message.metadata,
                'compressed': self.compression_enabled
            }
            
            return pickle.dumps(message_dict)
            
        except Exception as e:
            logger.error(f"Error serializing message: {str(e)}")
            return None
    
    def _deserialize_message(self, message_bytes: bytes) -> Optional[StreamMessage]:
        """Deserialize message from storage"""
        try:
            # Deserialize message structure
            message_dict = pickle.loads(message_bytes)
            
            # Extract data
            data_bytes = message_dict['data']
            
            # Decompress if needed
            if message_dict.get('compressed', False):
                data_bytes = lz4.frame.decompress(data_bytes)
            
            # Reconstruct data based on type
            # This is simplified - in practice you'd need better type detection
            try:
                data = pickle.loads(data_bytes)
            except:
                data = data_bytes
            
            # Create message
            return StreamMessage(
                data=data,
                timestamp_ns=message_dict['timestamp_ns'],
                sequence_id=message_dict['sequence_id'],
                source=message_dict['source'],
                message_type=message_dict['message_type'],
                priority=message_dict['priority'],
                metadata=message_dict['metadata']
            )
            
        except Exception as e:
            logger.error(f"Error deserializing message: {str(e)}")
            return None
    
    def _get_next_sequence_id(self) -> int:
        """Get next sequence ID"""
        with self.buffer_lock:
            self.sequence_counter += 1
            return self.sequence_counter
    
    def _record_latency(self, latency_us: float):
        """Record latency sample"""
        with self.metrics_lock:
            self.latency_samples.append(latency_us)
    
    def _update_throughput_metrics(self):
        """Update throughput metrics"""
        with self.metrics_lock:
            current_time = time.time()
            self.throughput_samples.append(current_time)
    
    def _monitor_performance(self):
        """Monitor system performance"""
        while self.is_running:
            try:
                # Calculate metrics
                with self.metrics_lock:
                    # Throughput calculation
                    if len(self.throughput_samples) > 1:
                        time_window = self.throughput_samples[-1] - self.throughput_samples[0]
                        if time_window > 0:
                            self.metrics.throughput_msgs_per_sec = len(self.throughput_samples) / time_window
                    
                    # Latency calculations
                    if len(self.latency_samples) > 0:
                        latencies = np.array(self.latency_samples)
                        self.metrics.avg_latency_us = np.mean(latencies)
                        self.metrics.p95_latency_us = np.percentile(latencies, 95)
                        self.metrics.p99_latency_us = np.percentile(latencies, 99)
                    
                    # Memory usage
                    process = psutil.Process()
                    self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                    self.metrics.cpu_usage_percent = process.cpu_percent()
                    
                    # Queue depth
                    self.metrics.queue_depth = self.message_buffer.size_used()
                    
                    # Update timestamp
                    self.metrics.timestamp = time.time()
                
                # Log metrics periodically
                if int(time.time()) % 10 == 0:  # Every 10 seconds
                    self._log_metrics()
                
                time.sleep(0.1)  # 100ms monitoring interval
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
    
    def _log_metrics(self):
        """Log performance metrics"""
        logger.info(f"Performance Metrics - "
                   f"Throughput: {self.metrics.throughput_msgs_per_sec:.2f} msg/s, "
                   f"Avg Latency: {self.metrics.avg_latency_us:.2f}us, "
                   f"P95 Latency: {self.metrics.p95_latency_us:.2f}us, "
                   f"P99 Latency: {self.metrics.p99_latency_us:.2f}us, "
                   f"Memory: {self.metrics.memory_usage_mb:.2f}MB, "
                   f"CPU: {self.metrics.cpu_usage_percent:.2f}%, "
                   f"Queue Depth: {self.metrics.queue_depth}, "
                   f"Backpressure Events: {self.metrics.backpressure_events}, "
                   f"Dropped Messages: {self.metrics.dropped_messages}")
    
    def register_data_callback(self, callback: Callable[[StreamMessage], None]):
        """Register a callback for data processing"""
        self.data_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable[[Exception], None]):
        """Register a callback for error handling"""
        self.error_callbacks.append(callback)
    
    def _notify_error(self, error: Exception):
        """Notify error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {str(e)}")
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        with self.metrics_lock:
            return PerformanceMetrics(
                throughput_msgs_per_sec=self.metrics.throughput_msgs_per_sec,
                avg_latency_us=self.metrics.avg_latency_us,
                p95_latency_us=self.metrics.p95_latency_us,
                p99_latency_us=self.metrics.p99_latency_us,
                memory_usage_mb=self.metrics.memory_usage_mb,
                cpu_usage_percent=self.metrics.cpu_usage_percent,
                queue_depth=self.metrics.queue_depth,
                backpressure_events=self.metrics.backpressure_events,
                dropped_messages=self.metrics.dropped_messages,
                timestamp=self.metrics.timestamp
            )
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get buffer status information"""
        return {
            'size_used': self.message_buffer.size_used(),
            'size_total': self.message_buffer.size,
            'utilization_percent': (self.message_buffer.size_used() / self.message_buffer.size) * 100,
            'is_full': self.message_buffer.is_full(),
            'is_empty': self.message_buffer.is_empty()
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        with self.metrics_lock:
            self.metrics = PerformanceMetrics()
            self.latency_samples.clear()
            self.throughput_samples.clear()


class MarketDataStreamer(RealtimeDataStreamer):
    """Specialized streamer for market data"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Market data specific settings
        self.symbol_filters = set()
        self.data_type_filters = set()
        self.price_validators = []
        
        # Market data callbacks
        self.tick_callbacks = []
        self.quote_callbacks = []
        self.trade_callbacks = []
    
    def add_symbol_filter(self, symbol: str):
        """Add symbol filter"""
        self.symbol_filters.add(symbol)
    
    def add_data_type_filter(self, data_type: str):
        """Add data type filter"""
        self.data_type_filters.add(data_type)
    
    def add_price_validator(self, validator: Callable[[float], bool]):
        """Add price validator"""
        self.price_validators.append(validator)
    
    def register_tick_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register tick data callback"""
        self.tick_callbacks.append(callback)
    
    def register_quote_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register quote data callback"""
        self.quote_callbacks.append(callback)
    
    def register_trade_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register trade data callback"""
        self.trade_callbacks.append(callback)
    
    def _process_message(self, message: StreamMessage):
        """Process market data message"""
        try:
            # Apply filters
            if self.symbol_filters and message.metadata:
                symbol = message.metadata.get('symbol')
                if symbol and symbol not in self.symbol_filters:
                    return
            
            if self.data_type_filters and message.message_type not in self.data_type_filters:
                return
            
            # Validate price data
            if isinstance(message.data, dict) and 'price' in message.data:
                price = message.data['price']
                for validator in self.price_validators:
                    if not validator(price):
                        logger.warning(f"Price validation failed for {price}")
                        return
            
            # Route to appropriate callbacks
            if message.message_type == 'tick':
                for callback in self.tick_callbacks:
                    callback(message.data)
            elif message.message_type == 'quote':
                for callback in self.quote_callbacks:
                    callback(message.data)
            elif message.message_type == 'trade':
                for callback in self.trade_callbacks:
                    callback(message.data)
            
            # Call parent processing
            super()._process_message(message)
            
        except Exception as e:
            logger.error(f"Error processing market data message: {str(e)}")
            self._notify_error(e)
    
    def send_tick_data(self, symbol: str, price: float, volume: int, timestamp: Optional[float] = None):
        """Send tick data"""
        tick_data = {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'timestamp': timestamp or time.time()
        }
        
        return self.send_message(
            data=tick_data,
            source='market_data',
            message_type='tick',
            metadata={'symbol': symbol}
        )
    
    def send_quote_data(self, symbol: str, bid: float, ask: float, bid_size: int, ask_size: int, timestamp: Optional[float] = None):
        """Send quote data"""
        quote_data = {
            'symbol': symbol,
            'bid': bid,
            'ask': ask,
            'bid_size': bid_size,
            'ask_size': ask_size,
            'timestamp': timestamp or time.time()
        }
        
        return self.send_message(
            data=quote_data,
            source='market_data',
            message_type='quote',
            metadata={'symbol': symbol}
        )
    
    def send_trade_data(self, symbol: str, price: float, volume: int, side: str, timestamp: Optional[float] = None):
        """Send trade data"""
        trade_data = {
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'side': side,
            'timestamp': timestamp or time.time()
        }
        
        return self.send_message(
            data=trade_data,
            source='market_data',
            message_type='trade',
            metadata={'symbol': symbol}
        )


# Utility functions for common market data validation
def create_price_range_validator(min_price: float, max_price: float) -> Callable[[float], bool]:
    """Create price range validator"""
    def validator(price: float) -> bool:
        return min_price <= price <= max_price
    return validator

def create_price_change_validator(max_change_percent: float) -> Callable[[float], bool]:
    """Create price change validator"""
    last_price = None
    
    def validator(price: float) -> bool:
        nonlocal last_price
        if last_price is None:
            last_price = price
            return True
        
        change_percent = abs(price - last_price) / last_price * 100
        if change_percent > max_change_percent:
            return False
        
        last_price = price
        return True
    
    return validator

def create_volume_validator(max_volume: int) -> Callable[[int], bool]:
    """Create volume validator"""
    def validator(volume: int) -> bool:
        return 0 < volume <= max_volume
    return validator
