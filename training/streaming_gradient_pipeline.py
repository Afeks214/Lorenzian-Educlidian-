"""
Streaming Gradient Update Pipeline
Memory-efficient streaming pipeline for continuous gradient updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, AsyncIterator
import numpy as np
import logging
import time
import threading
import asyncio
from collections import deque, defaultdict
from dataclasses import dataclass, field
import queue
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
from enum import Enum
import math
import json
from datetime import datetime, timedelta
import pickle
import zlib
import sys
import traceback
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import uuid
import heapq
import zmq
import msgpack
from threading import RLock, Event, Condition
import struct
import mmap
import tempfile
import os

from .online_gradient_accumulation import OnlineGradientAccumulator, OnlineGradientConfig

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming modes for gradient updates"""
    PUSH = "push"
    PULL = "pull"
    HYBRID = "hybrid"


class Priority(Enum):
    """Priority levels for gradient updates"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class StreamingConfig:
    """Configuration for streaming gradient pipeline"""
    # Core streaming settings
    mode: StreamingMode = StreamingMode.HYBRID
    batch_size: int = 32
    buffer_size: int = 1000
    max_workers: int = 4
    
    # Memory management
    max_memory_mb: float = 2048.0
    memory_threshold: float = 0.8
    enable_memory_mapping: bool = True
    
    # Network settings
    zmq_port: int = 5555
    zmq_hwm: int = 1000
    compression_enabled: bool = True
    serialization_format: str = "msgpack"  # msgpack, pickle, json
    
    # Performance settings
    target_throughput_per_second: int = 1000
    max_latency_ms: float = 50.0
    enable_batching: bool = True
    batch_timeout_ms: float = 10.0
    
    # Reliability settings
    enable_persistence: bool = True
    checkpoint_interval: int = 100
    max_retries: int = 3
    retry_delay_ms: float = 100.0
    
    # Monitoring
    enable_metrics: bool = True
    metrics_interval: int = 10
    profiling_enabled: bool = False


@dataclass
class GradientUpdate:
    """Represents a single gradient update"""
    id: str
    gradients: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]
    timestamp: float
    priority: Priority = Priority.NORMAL
    retries: int = 0
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value > other.priority.value


class MemoryMappedBuffer:
    """Memory-mapped buffer for efficient gradient storage"""
    
    def __init__(self, max_size_mb: float = 1024.0):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b'\0' * self.max_size_bytes)
        self.temp_file.flush()
        
        # Create memory map
        self.mmap = mmap.mmap(self.temp_file.fileno(), self.max_size_bytes)
        self.current_position = 0
        self.lock = RLock()
        
        # Index for stored gradients
        self.gradient_index = {}
        
    def store_gradient(self, gradient_id: str, gradient_data: bytes) -> bool:
        """Store gradient in memory-mapped buffer"""
        with self.lock:
            data_size = len(gradient_data)
            
            if self.current_position + data_size + 8 > self.max_size_bytes:
                # Buffer full, trigger compaction
                self._compact_buffer()
                
                if self.current_position + data_size + 8 > self.max_size_bytes:
                    return False  # Still no space
            
            # Store size header (8 bytes)
            self.mmap[self.current_position:self.current_position + 8] = struct.pack('Q', data_size)
            
            # Store data
            start_pos = self.current_position + 8
            self.mmap[start_pos:start_pos + data_size] = gradient_data
            
            # Update index
            self.gradient_index[gradient_id] = (self.current_position, data_size)
            self.current_position += data_size + 8
            
            return True
    
    def retrieve_gradient(self, gradient_id: str) -> Optional[bytes]:
        """Retrieve gradient from memory-mapped buffer"""
        with self.lock:
            if gradient_id not in self.gradient_index:
                return None
            
            position, size = self.gradient_index[gradient_id]
            data_start = position + 8
            
            return bytes(self.mmap[data_start:data_start + size])
    
    def remove_gradient(self, gradient_id: str) -> bool:
        """Remove gradient from index (data remains until compaction)"""
        with self.lock:
            if gradient_id in self.gradient_index:
                del self.gradient_index[gradient_id]
                return True
            return False
    
    def _compact_buffer(self):
        """Compact buffer by removing unused gradients"""
        with self.lock:
            if not self.gradient_index:
                self.current_position = 0
                return
            
            # Create new compact layout
            new_position = 0
            new_index = {}
            
            # Create temporary buffer
            temp_buffer = bytearray(self.max_size_bytes)
            
            for gradient_id, (position, size) in self.gradient_index.items():
                # Copy size header
                temp_buffer[new_position:new_position + 8] = self.mmap[position:position + 8]
                
                # Copy data
                data_start = position + 8
                new_data_start = new_position + 8
                temp_buffer[new_data_start:new_data_start + size] = self.mmap[data_start:data_start + size]
                
                # Update new index
                new_index[gradient_id] = (new_position, size)
                new_position += size + 8
            
            # Replace buffer contents
            self.mmap[:new_position] = temp_buffer[:new_position]
            self.gradient_index = new_index
            self.current_position = new_position
            
            logger.info(f"Buffer compacted: {len(self.gradient_index)} gradients, "
                       f"position: {self.current_position}/{self.max_size_bytes}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                'total_gradients': len(self.gradient_index),
                'current_position': self.current_position,
                'max_size_bytes': self.max_size_bytes,
                'utilization': self.current_position / self.max_size_bytes,
                'avg_gradient_size': (self.current_position / len(self.gradient_index)) 
                                   if self.gradient_index else 0
            }
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'temp_file'):
            self.temp_file.close()
            try:
                os.unlink(self.temp_file.name)
            except:
                pass


class GradientSerializer:
    """Handles gradient serialization and compression"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.serialization_format = config.serialization_format
        self.compression_enabled = config.compression_enabled
        
        # Compression stats
        self.compression_stats = {
            'total_compressed_size': 0,
            'total_original_size': 0,
            'compression_ratio': 0.0,
            'compression_count': 0
        }
        
    def serialize_gradient(self, gradient_update: GradientUpdate) -> bytes:
        """Serialize gradient update to bytes"""
        # Convert tensors to numpy arrays for serialization
        serializable_gradients = {}
        for name, tensor in gradient_update.gradients.items():
            if tensor is not None:
                serializable_gradients[name] = {
                    'data': tensor.cpu().numpy(),
                    'device': str(tensor.device),
                    'dtype': str(tensor.dtype)
                }
            else:
                serializable_gradients[name] = None
        
        # Create serializable object
        serializable_obj = {
            'id': gradient_update.id,
            'gradients': serializable_gradients,
            'metadata': gradient_update.metadata,
            'timestamp': gradient_update.timestamp,
            'priority': gradient_update.priority.value,
            'retries': gradient_update.retries
        }
        
        # Serialize
        if self.serialization_format == 'msgpack':
            serialized = msgpack.packb(serializable_obj)
        elif self.serialization_format == 'pickle':
            serialized = pickle.dumps(serializable_obj)
        elif self.serialization_format == 'json':
            serialized = json.dumps(serializable_obj, default=str).encode('utf-8')
        else:
            raise ValueError(f"Unsupported serialization format: {self.serialization_format}")
        
        # Compress if enabled
        if self.compression_enabled:
            compressed = zlib.compress(serialized)
            
            # Update compression stats
            self.compression_stats['total_original_size'] += len(serialized)
            self.compression_stats['total_compressed_size'] += len(compressed)
            self.compression_stats['compression_count'] += 1
            
            if self.compression_stats['total_original_size'] > 0:
                self.compression_stats['compression_ratio'] = (
                    self.compression_stats['total_compressed_size'] /
                    self.compression_stats['total_original_size']
                )
            
            return compressed
        
        return serialized
    
    def deserialize_gradient(self, data: bytes) -> GradientUpdate:
        """Deserialize bytes to gradient update"""
        # Decompress if needed
        if self.compression_enabled:
            data = zlib.decompress(data)
        
        # Deserialize
        if self.serialization_format == 'msgpack':
            obj = msgpack.unpackb(data, raw=False)
        elif self.serialization_format == 'pickle':
            obj = pickle.loads(data)
        elif self.serialization_format == 'json':
            obj = json.loads(data.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported serialization format: {self.serialization_format}")
        
        # Convert numpy arrays back to tensors
        gradients = {}
        for name, tensor_data in obj['gradients'].items():
            if tensor_data is not None:
                numpy_array = tensor_data['data']
                device = tensor_data['device']
                dtype = getattr(torch, tensor_data['dtype'].split('.')[-1])
                
                tensor = torch.from_numpy(numpy_array).to(device=device, dtype=dtype)
                gradients[name] = tensor
            else:
                gradients[name] = None
        
        # Create gradient update
        return GradientUpdate(
            id=obj['id'],
            gradients=gradients,
            metadata=obj['metadata'],
            timestamp=obj['timestamp'],
            priority=Priority(obj['priority']),
            retries=obj['retries']
        )
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return self.compression_stats.copy()


class GradientStreamPublisher:
    """Publishes gradient updates to stream"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.set_hwm(config.zmq_hwm)
        self.socket.bind(f"tcp://*:{config.zmq_port}")
        
        self.serializer = GradientSerializer(config)
        self.published_count = 0
        self.last_publish_time = time.time()
        
        # Performance tracking
        self.publish_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        logger.info(f"Gradient stream publisher initialized on port {config.zmq_port}")
    
    def publish_gradient(self, gradient_update: GradientUpdate) -> bool:
        """Publish gradient update to stream"""
        start_time = time.time()
        
        try:
            # Serialize gradient
            serialized_data = self.serializer.serialize_gradient(gradient_update)
            
            # Create message with priority
            topic = f"gradient_{gradient_update.priority.name.lower()}"
            message = [topic.encode(), serialized_data]
            
            # Send message
            self.socket.send_multipart(message, zmq.NOBLOCK)
            
            # Update metrics
            self.published_count += 1
            publish_time = (time.time() - start_time) * 1000  # ms
            self.publish_times.append(publish_time)
            
            # Update throughput
            current_time = time.time()
            if current_time - self.last_publish_time >= 1.0:  # Calculate throughput every second
                throughput = self.published_count / (current_time - self.last_publish_time)
                self.throughput_history.append(throughput)
                self.published_count = 0
                self.last_publish_time = current_time
            
            return True
            
        except zmq.Again:
            logger.warning("Publisher queue full, dropping message")
            return False
        except Exception as e:
            logger.error(f"Error publishing gradient: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics"""
        return {
            'published_count': self.published_count,
            'avg_publish_time_ms': np.mean(self.publish_times) if self.publish_times else 0,
            'max_publish_time_ms': np.max(self.publish_times) if self.publish_times else 0,
            'current_throughput': self.throughput_history[-1] if self.throughput_history else 0,
            'avg_throughput': np.mean(self.throughput_history) if self.throughput_history else 0,
            'serialization_stats': self.serializer.get_compression_stats()
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.socket.close()
        self.context.term()


class GradientStreamSubscriber:
    """Subscribes to gradient stream and processes updates"""
    
    def __init__(self, config: StreamingConfig, host: str = "localhost"):
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{config.zmq_port}")
        
        # Subscribe to all priority levels
        for priority in Priority:
            topic = f"gradient_{priority.name.lower()}"
            self.socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        
        self.serializer = GradientSerializer(config)
        self.received_count = 0
        self.processing_enabled = True
        
        # Performance tracking
        self.receive_times = deque(maxlen=1000)
        self.processing_times = deque(maxlen=1000)
        
        logger.info(f"Gradient stream subscriber initialized, connected to {host}:{config.zmq_port}")
    
    def receive_gradient(self, timeout_ms: int = 1000) -> Optional[GradientUpdate]:
        """Receive gradient update from stream"""
        start_time = time.time()
        
        try:
            # Receive message with timeout
            if self.socket.poll(timeout_ms):
                message = self.socket.recv_multipart(zmq.NOBLOCK)
                
                if len(message) >= 2:
                    topic, data = message[0], message[1]
                    
                    # Deserialize gradient
                    gradient_update = self.serializer.deserialize_gradient(data)
                    
                    # Update metrics
                    self.received_count += 1
                    receive_time = (time.time() - start_time) * 1000  # ms
                    self.receive_times.append(receive_time)
                    
                    return gradient_update
            
            return None
            
        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Error receiving gradient: {e}")
            return None
    
    async def receive_gradient_async(self, timeout_ms: int = 1000) -> Optional[GradientUpdate]:
        """Asynchronous gradient receive"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.receive_gradient, timeout_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get subscriber statistics"""
        return {
            'received_count': self.received_count,
            'avg_receive_time_ms': np.mean(self.receive_times) if self.receive_times else 0,
            'max_receive_time_ms': np.max(self.receive_times) if self.receive_times else 0,
            'avg_processing_time_ms': np.mean(self.processing_times) if self.processing_times else 0,
            'deserialization_stats': self.serializer.get_compression_stats()
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.socket.close()
        self.context.term()


class StreamingGradientPipeline:
    """Main streaming gradient pipeline orchestrator"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 config: StreamingConfig,
                 device: torch.device = None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.memory_buffer = MemoryMappedBuffer(config.max_memory_mb)
        self.publisher = GradientStreamPublisher(config)
        self.subscriber = GradientStreamSubscriber(config)
        
        # Initialize online gradient accumulator
        gradient_config = OnlineGradientConfig(
            max_accumulation_steps=config.batch_size,
            target_latency_ms=config.max_latency_ms,
            max_memory_mb=config.max_memory_mb,
            async_processing=True
        )
        
        self.gradient_accumulator = OnlineGradientAccumulator(
            model, optimizer, gradient_config, device
        )
        
        # Priority queues for different processing priorities
        self.priority_queues = {
            Priority.CRITICAL: queue.PriorityQueue(),
            Priority.HIGH: queue.PriorityQueue(),
            Priority.NORMAL: queue.PriorityQueue(),
            Priority.LOW: queue.PriorityQueue()
        }
        
        # Processing threads
        self.processing_threads = []
        self.stop_event = threading.Event()
        
        # Performance monitoring
        self.performance_metrics = {
            'gradients_processed': 0,
            'gradients_published': 0,
            'gradients_received': 0,
            'processing_latency': deque(maxlen=1000),
            'memory_usage': deque(maxlen=100),
            'throughput': deque(maxlen=100)
        }
        
        # Batch processing
        self.batch_buffer = []
        self.batch_timeout = config.batch_timeout_ms / 1000.0
        self.last_batch_time = time.time()
        
        # Start processing threads
        self.start_processing_threads()
        
        logger.info("Streaming gradient pipeline initialized")
    
    def start_processing_threads(self):
        """Start background processing threads"""
        # Publisher thread
        publisher_thread = threading.Thread(
            target=self._publisher_loop,
            daemon=True
        )
        publisher_thread.start()
        self.processing_threads.append(publisher_thread)
        
        # Subscriber thread
        subscriber_thread = threading.Thread(
            target=self._subscriber_loop,
            daemon=True
        )
        subscriber_thread.start()
        self.processing_threads.append(subscriber_thread)
        
        # Gradient processing threads
        for i in range(self.config.max_workers):
            processing_thread = threading.Thread(
                target=self._gradient_processing_loop,
                daemon=True
            )
            processing_thread.start()
            self.processing_threads.append(processing_thread)
        
        # Batch processing thread
        if self.config.enable_batching:
            batch_thread = threading.Thread(
                target=self._batch_processing_loop,
                daemon=True
            )
            batch_thread.start()
            self.processing_threads.append(batch_thread)
        
        logger.info(f"Started {len(self.processing_threads)} processing threads")
    
    def submit_gradient(self, 
                       loss: torch.Tensor,
                       batch_data: torch.Tensor = None,
                       priority: Priority = Priority.NORMAL,
                       metadata: Dict[str, Any] = None) -> str:
        """Submit gradient for processing"""
        start_time = time.time()
        
        try:
            # Compute gradients
            loss.backward()
            
            # Extract gradients
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            
            # Create gradient update
            gradient_update = GradientUpdate(
                id=str(uuid.uuid4()),
                gradients=gradients,
                metadata=metadata or {},
                timestamp=time.time(),
                priority=priority
            )
            
            # Add to appropriate priority queue
            self.priority_queues[priority].put(gradient_update)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_metrics['processing_latency'].append(processing_time)
            
            return gradient_update.id
            
        except Exception as e:
            logger.error(f"Error submitting gradient: {e}")
            return None
    
    def _publisher_loop(self):
        """Publisher thread loop"""
        while not self.stop_event.is_set():
            try:
                # Process each priority queue
                for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                    if not self.priority_queues[priority].empty():
                        try:
                            gradient_update = self.priority_queues[priority].get_nowait()
                            
                            # Publish gradient
                            if self.publisher.publish_gradient(gradient_update):
                                self.performance_metrics['gradients_published'] += 1
                                
                                # Store in memory buffer for persistence
                                if self.config.enable_persistence:
                                    serialized = self.publisher.serializer.serialize_gradient(gradient_update)
                                    self.memory_buffer.store_gradient(gradient_update.id, serialized)
                            
                        except queue.Empty:
                            continue
                
                # Small delay to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in publisher loop: {e}")
                time.sleep(0.01)
    
    def _subscriber_loop(self):
        """Subscriber thread loop"""
        while not self.stop_event.is_set():
            try:
                # Receive gradient update
                gradient_update = self.subscriber.receive_gradient(timeout_ms=100)
                
                if gradient_update:
                    self.performance_metrics['gradients_received'] += 1
                    
                    # Add to batch buffer or process immediately
                    if self.config.enable_batching:
                        self.batch_buffer.append(gradient_update)
                    else:
                        self._process_gradient_update(gradient_update)
                
            except Exception as e:
                logger.error(f"Error in subscriber loop: {e}")
                time.sleep(0.01)
    
    def _batch_processing_loop(self):
        """Batch processing thread loop"""
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Process batch if buffer is full or timeout reached
                if (len(self.batch_buffer) >= self.config.batch_size or
                    (self.batch_buffer and current_time - self.last_batch_time >= self.batch_timeout)):
                    
                    # Process batch
                    batch = self.batch_buffer.copy()
                    self.batch_buffer.clear()
                    self.last_batch_time = current_time
                    
                    # Process batch gradients
                    self._process_gradient_batch(batch)
                
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                time.sleep(0.01)
    
    def _gradient_processing_loop(self):
        """Gradient processing thread loop"""
        while not self.stop_event.is_set():
            try:
                # Process gradients from memory buffer
                # This is a simplified version - in practice, you'd have more sophisticated processing
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in gradient processing loop: {e}")
                time.sleep(0.01)
    
    def _process_gradient_update(self, gradient_update: GradientUpdate):
        """Process a single gradient update"""
        try:
            # Apply gradient using accumulator
            # Convert gradients to loss for accumulator (simplified)
            total_loss = sum(grad.sum() for grad in gradient_update.gradients.values() if grad is not None)
            
            result = self.gradient_accumulator.accumulate_gradient(
                total_loss, 
                metadata=gradient_update.metadata
            )
            
            self.performance_metrics['gradients_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing gradient update: {e}")
    
    def _process_gradient_batch(self, batch: List[GradientUpdate]):
        """Process a batch of gradient updates"""
        try:
            # Aggregate gradients from batch
            aggregated_gradients = {}
            total_loss = 0.0
            
            for gradient_update in batch:
                for name, grad in gradient_update.gradients.items():
                    if grad is not None:
                        if name not in aggregated_gradients:
                            aggregated_gradients[name] = torch.zeros_like(grad)
                        aggregated_gradients[name] += grad
                        total_loss += grad.sum().item()
            
            # Average gradients
            for name in aggregated_gradients:
                aggregated_gradients[name] /= len(batch)
            
            # Apply aggregated gradients
            for name, param in self.model.named_parameters():
                if name in aggregated_gradients:
                    param.grad = aggregated_gradients[name]
            
            # Update model
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            self.performance_metrics['gradients_processed'] += len(batch)
            
        except Exception as e:
            logger.error(f"Error processing gradient batch: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'pipeline_metrics': self.performance_metrics.copy(),
            'publisher_stats': self.publisher.get_stats(),
            'subscriber_stats': self.subscriber.get_stats(),
            'memory_buffer_stats': self.memory_buffer.get_stats(),
            'accumulator_stats': self.gradient_accumulator.get_performance_stats()
        }
        
        # Add derived metrics
        if self.performance_metrics['processing_latency']:
            stats['pipeline_metrics']['avg_processing_latency_ms'] = np.mean(
                self.performance_metrics['processing_latency']
            )
        
        return stats
    
    def save_checkpoint(self, filepath: str):
        """Save pipeline checkpoint"""
        checkpoint = {
            'config': self.config.__dict__,
            'performance_metrics': self.performance_metrics,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        logger.info(f"Pipeline checkpoint saved to {filepath}")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up streaming gradient pipeline")
        
        # Stop processing threads
        self.stop_event.set()
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=1.0)
        
        # Clean up components
        self.gradient_accumulator.stop_async_processing()
        self.publisher.cleanup()
        self.subscriber.cleanup()
        self.memory_buffer.cleanup()
        
        logger.info("Pipeline cleanup completed")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()


def create_streaming_config(
    target_throughput: int = 1000,
    max_latency_ms: float = 50.0,
    max_memory_mb: float = 2048.0,
    enable_compression: bool = True
) -> StreamingConfig:
    """Create optimized streaming configuration"""
    
    # Determine batch size based on throughput requirements
    if target_throughput <= 100:
        batch_size = 16
        max_workers = 2
    elif target_throughput <= 1000:
        batch_size = 32
        max_workers = 4
    else:
        batch_size = 64
        max_workers = 8
    
    return StreamingConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        target_throughput_per_second=target_throughput,
        max_latency_ms=max_latency_ms,
        max_memory_mb=max_memory_mb,
        compression_enabled=enable_compression,
        enable_batching=True,
        enable_persistence=True,
        serialization_format="msgpack"
    )


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create configuration
    config = create_streaming_config(
        target_throughput=500,
        max_latency_ms=30.0,
        max_memory_mb=1024.0,
        enable_compression=True
    )
    
    # Initialize pipeline
    pipeline = StreamingGradientPipeline(model, optimizer, config)
    
    # Simulate streaming updates
    print("Starting streaming gradient pipeline demo...")
    
    try:
        for i in range(100):
            # Simulate batch data
            batch_data = torch.randn(32, 128)
            target = torch.randn(32, 1)
            
            # Forward pass
            output = model(batch_data)
            loss = nn.MSELoss()(output, target)
            
            # Submit gradient
            gradient_id = pipeline.submit_gradient(
                loss, 
                batch_data,
                priority=Priority.NORMAL,
                metadata={'step': i, 'loss': loss.item()}
            )
            
            if i % 10 == 0:
                stats = pipeline.get_performance_stats()
                print(f"Step {i}: Loss={loss.item():.6f}, "
                      f"Gradients Processed={stats['pipeline_metrics']['gradients_processed']}, "
                      f"Avg Latency={stats['pipeline_metrics'].get('avg_processing_latency_ms', 0):.2f}ms")
            
            # Small delay to simulate real-time processing
            time.sleep(0.01)
        
        # Final statistics
        final_stats = pipeline.get_performance_stats()
        print(f"\nFinal Statistics:")
        print(f"Total Gradients Processed: {final_stats['pipeline_metrics']['gradients_processed']}")
        print(f"Total Gradients Published: {final_stats['pipeline_metrics']['gradients_published']}")
        print(f"Publisher Throughput: {final_stats['publisher_stats']['avg_throughput']:.2f} updates/sec")
        print(f"Memory Buffer Utilization: {final_stats['memory_buffer_stats']['utilization']:.2%}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        # Cleanup
        pipeline.cleanup()
        print("Demo completed!")