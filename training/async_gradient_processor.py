"""
Asynchronous Gradient Processing Engine
High-performance async gradient processing for real-time trading systems
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Coroutine
import numpy as np
import logging
import time
import asyncio
from collections import deque, defaultdict
from dataclasses import dataclass, field
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import gc
from enum import Enum
import math
import json
from datetime import datetime, timedelta
import uuid
import sys
import traceback
from contextlib import asynccontextmanager
from functools import wraps
import weakref
import multiprocessing as mp
from threading import RLock, Event, Condition
import signal
import os

from .online_gradient_accumulation import OnlineGradientAccumulator, OnlineGradientConfig
from .streaming_gradient_pipeline import StreamingGradientPipeline, StreamingConfig

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Strategy for asynchronous processing"""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC_COROUTINES = "async_coroutines"
    HYBRID = "hybrid"


class TaskPriority(Enum):
    """Priority levels for gradient processing tasks"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


@dataclass
class AsyncProcessingConfig:
    """Configuration for asynchronous gradient processing"""
    # Core processing settings
    strategy: ProcessingStrategy = ProcessingStrategy.HYBRID
    max_workers: int = 4
    max_concurrent_tasks: int = 100
    
    # Performance settings
    target_latency_ms: float = 20.0
    max_latency_ms: float = 50.0
    throughput_target: int = 1000
    
    # Memory management
    max_memory_mb: float = 2048.0
    memory_cleanup_threshold: float = 0.8
    enable_memory_profiling: bool = True
    
    # Queue management
    queue_size: int = 1000
    priority_queue_enabled: bool = True
    backpressure_enabled: bool = True
    
    # Async settings
    event_loop_policy: str = "default"  # default, uvloop, asyncio
    coroutine_concurrency: int = 50
    
    # Process pool settings
    process_pool_size: int = 2
    shared_memory_size_mb: float = 512.0
    
    # Monitoring and debugging
    enable_profiling: bool = False
    enable_metrics: bool = True
    metrics_interval: float = 1.0
    
    # Fault tolerance
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 0.1


@dataclass
class GradientTask:
    """Represents a gradient processing task"""
    id: str
    gradients: Dict[str, torch.Tensor]
    metadata: Dict[str, Any]
    priority: TaskPriority
    timestamp: float
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    retries: int = 0
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority.value > other.priority.value


class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, threshold: int = 5, timeout: float = 30.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.lock = RLock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection"""
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.threshold:
                    self.state = "open"
                
                raise e
    
    def get_state(self) -> str:
        """Get circuit breaker state"""
        return self.state


class AsyncGradientProcessor:
    """Asynchronous gradient processor with multiple processing strategies"""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 config: AsyncProcessingConfig,
                 device: torch.device = None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize processing components
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=config.process_pool_size)
        
        # Task queues
        self.priority_queue = asyncio.PriorityQueue(maxsize=config.queue_size)
        self.pending_tasks = {}  # task_id -> task
        self.completed_tasks = {}  # task_id -> result
        
        # Event loop
        self.loop = None
        self.processing_tasks = set()
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        )
        
        # Performance monitoring
        self.metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'processing_times': deque(maxlen=1000),
            'queue_sizes': deque(maxlen=1000),
            'memory_usage': deque(maxlen=100),
            'throughput': deque(maxlen=100)
        }
        
        # Synchronization
        self.shutdown_event = asyncio.Event()
        self.processing_lock = asyncio.Lock()
        
        # Initialize gradient accumulator
        gradient_config = OnlineGradientConfig(
            target_latency_ms=config.target_latency_ms,
            max_memory_mb=config.max_memory_mb,
            async_processing=True
        )
        
        self.gradient_accumulator = OnlineGradientAccumulator(
            model, optimizer, gradient_config, device
        )
        
        # Start processing
        self.running = False
        
        logger.info(f"Async gradient processor initialized with {config.strategy} strategy")
    
    async def start(self):
        """Start the async processor"""
        if self.running:
            return
        
        self.running = True
        self.loop = asyncio.get_event_loop()
        
        # Start processing tasks based on strategy
        if self.config.strategy in [ProcessingStrategy.ASYNC_COROUTINES, ProcessingStrategy.HYBRID]:
            # Start coroutine workers
            for i in range(self.config.coroutine_concurrency):
                task = asyncio.create_task(self._coroutine_worker(f"worker-{i}"))
                self.processing_tasks.add(task)
        
        # Start monitoring task
        if self.config.enable_metrics:
            monitor_task = asyncio.create_task(self._monitoring_loop())
            self.processing_tasks.add(monitor_task)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.processing_tasks.add(cleanup_task)
        
        logger.info("Async gradient processor started")
    
    async def stop(self):
        """Stop the async processor"""
        if not self.running:
            return
        
        self.running = False
        self.shutdown_event.set()
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # Cleanup executors
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        # Cleanup gradient accumulator
        self.gradient_accumulator.stop_async_processing()
        
        logger.info("Async gradient processor stopped")
    
    async def submit_gradient(self, 
                             gradients: Dict[str, torch.Tensor],
                             priority: TaskPriority = TaskPriority.NORMAL,
                             metadata: Dict[str, Any] = None,
                             callback: Optional[Callable] = None,
                             timeout: Optional[float] = None) -> str:
        """Submit gradient for asynchronous processing"""
        task = GradientTask(
            id=str(uuid.uuid4()),
            gradients=gradients,
            metadata=metadata or {},
            priority=priority,
            timestamp=time.time(),
            callback=callback,
            timeout=timeout or self.config.max_latency_ms / 1000.0
        )
        
        # Add to priority queue
        await self.priority_queue.put((priority.value, task))
        
        # Track task
        self.pending_tasks[task.id] = task
        self.metrics['tasks_submitted'] += 1
        
        return task.id
    
    async def _coroutine_worker(self, worker_id: str):
        """Coroutine worker for processing gradients"""
        while not self.shutdown_event.is_set():
            try:
                # Get task from priority queue
                priority, task = await asyncio.wait_for(
                    self.priority_queue.get(),
                    timeout=1.0
                )
                
                # Process task
                await self._process_task_async(task, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in coroutine worker {worker_id}: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_task_async(self, task: GradientTask, worker_id: str):
        """Process a single gradient task asynchronously"""
        start_time = time.time()
        
        try:
            async with self.processing_lock:
                # Check circuit breaker
                if self.config.enable_circuit_breaker:
                    if self.circuit_breaker.state == "open":
                        raise Exception("Circuit breaker is open")
                
                # Choose processing strategy
                if self.config.strategy == ProcessingStrategy.ASYNC_COROUTINES:
                    result = await self._process_with_coroutines(task)
                elif self.config.strategy == ProcessingStrategy.THREAD_POOL:
                    result = await self._process_with_thread_pool(task)
                elif self.config.strategy == ProcessingStrategy.PROCESS_POOL:
                    result = await self._process_with_process_pool(task)
                else:  # HYBRID
                    result = await self._process_with_hybrid_strategy(task)
                
                # Record completion
                processing_time = (time.time() - start_time) * 1000  # ms
                self.metrics['processing_times'].append(processing_time)
                self.metrics['tasks_completed'] += 1
                
                # Store result
                self.completed_tasks[task.id] = {
                    'result': result,
                    'processing_time': processing_time,
                    'worker_id': worker_id,
                    'timestamp': time.time()
                }
                
                # Call callback if provided
                if task.callback:
                    await self._call_callback_async(task.callback, result)
                
                # Remove from pending tasks
                self.pending_tasks.pop(task.id, None)
                
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            self.metrics['tasks_failed'] += 1
            
            # Record failure
            self.completed_tasks[task.id] = {
                'error': str(e),
                'processing_time': (time.time() - start_time) * 1000,
                'worker_id': worker_id,
                'timestamp': time.time()
            }
            
            # Remove from pending tasks
            self.pending_tasks.pop(task.id, None)
    
    async def _process_with_coroutines(self, task: GradientTask) -> Dict[str, Any]:
        """Process task using pure coroutines"""
        # Create a synthetic loss from gradients for accumulator
        total_loss = torch.tensor(0.0, device=self.device)
        for grad in task.gradients.values():
            if grad is not None:
                total_loss += grad.abs().sum()
        
        # Use gradient accumulator (this is a simplified approach)
        # In practice, you'd have more sophisticated gradient processing
        result = self.gradient_accumulator.accumulate_gradient(
            total_loss,
            metadata=task.metadata
        )
        
        return result
    
    async def _process_with_thread_pool(self, task: GradientTask) -> Dict[str, Any]:
        """Process task using thread pool"""
        def thread_worker():
            try:
                # Process gradients in thread
                total_loss = torch.tensor(0.0, device=self.device)
                for grad in task.gradients.values():
                    if grad is not None:
                        total_loss += grad.abs().sum()
                
                return self.gradient_accumulator.accumulate_gradient(
                    total_loss,
                    metadata=task.metadata
                )
            except Exception as e:
                logger.error(f"Error in thread worker: {e}")
                raise
        
        # Submit to thread pool
        future = self.thread_pool.submit(thread_worker)
        result = await asyncio.wrap_future(future)
        
        return result
    
    async def _process_with_process_pool(self, task: GradientTask) -> Dict[str, Any]:
        """Process task using process pool"""
        def process_worker(gradients_dict, metadata):
            try:
                # Process gradients in separate process
                # Note: This is a simplified version
                # In practice, you'd need to properly serialize/deserialize torch tensors
                gradient_norm = 0.0
                for grad_data in gradients_dict.values():
                    if grad_data is not None:
                        gradient_norm += np.sum(np.abs(grad_data))
                
                return {
                    'gradient_norm': gradient_norm,
                    'processed_in_process': True,
                    'metadata': metadata
                }
            except Exception as e:
                logger.error(f"Error in process worker: {e}")
                raise
        
        # Convert tensors to numpy for process pool
        gradients_numpy = {}
        for name, grad in task.gradients.items():
            if grad is not None:
                gradients_numpy[name] = grad.cpu().numpy()
            else:
                gradients_numpy[name] = None
        
        # Submit to process pool
        future = self.process_pool.submit(process_worker, gradients_numpy, task.metadata)
        result = await asyncio.wrap_future(future)
        
        return result
    
    async def _process_with_hybrid_strategy(self, task: GradientTask) -> Dict[str, Any]:
        """Process task using hybrid strategy"""
        # Choose strategy based on task priority and current load
        if task.priority in [TaskPriority.CRITICAL, TaskPriority.EMERGENCY]:
            # High priority tasks use coroutines for speed
            return await self._process_with_coroutines(task)
        elif len(self.pending_tasks) > self.config.max_concurrent_tasks * 0.8:
            # High load - use process pool for parallelism
            return await self._process_with_process_pool(task)
        else:
            # Normal load - use thread pool
            return await self._process_with_thread_pool(task)
    
    async def _call_callback_async(self, callback: Callable, result: Any):
        """Call callback function asynchronously"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(result)
            else:
                # Run in thread pool for blocking callbacks
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, callback, result
                )
        except Exception as e:
            logger.error(f"Error calling callback: {e}")
    
    async def _monitoring_loop(self):
        """Monitoring loop for performance metrics"""
        while not self.shutdown_event.is_set():
            try:
                # Update metrics
                self.metrics['queue_sizes'].append(self.priority_queue.qsize())
                
                # Memory usage
                memory_usage = psutil.Process().memory_info().rss / 1024**2
                self.metrics['memory_usage'].append(memory_usage)
                
                # Throughput calculation
                if len(self.metrics['processing_times']) > 0:
                    recent_completions = len([
                        t for t in self.metrics['processing_times'] 
                        if t > 0  # Only count completed tasks
                    ])
                    throughput = recent_completions / self.config.metrics_interval
                    self.metrics['throughput'].append(throughput)
                
                # Memory cleanup if needed
                if memory_usage > self.config.max_memory_mb * self.config.memory_cleanup_threshold:
                    await self._cleanup_memory()
                
                # Log metrics periodically
                if self.config.enable_metrics:
                    await self._log_metrics()
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _cleanup_loop(self):
        """Cleanup loop for removing old completed tasks"""
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Remove old completed tasks (older than 5 minutes)
                old_tasks = [
                    task_id for task_id, result in self.completed_tasks.items()
                    if current_time - result['timestamp'] > 300  # 5 minutes
                ]
                
                for task_id in old_tasks:
                    self.completed_tasks.pop(task_id, None)
                
                # Check for timed out pending tasks
                timed_out_tasks = [
                    task_id for task_id, task in self.pending_tasks.items()
                    if task.timeout and current_time - task.timestamp > task.timeout
                ]
                
                for task_id in timed_out_tasks:
                    task = self.pending_tasks.pop(task_id, None)
                    if task:
                        self.completed_tasks[task_id] = {
                            'error': 'Task timeout',
                            'processing_time': (current_time - task.timestamp) * 1000,
                            'timestamp': current_time
                        }
                        self.metrics['tasks_failed'] += 1
                
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_memory(self):
        """Clean up memory"""
        # Clear old metrics
        if len(self.metrics['processing_times']) > 500:
            self.metrics['processing_times'] = deque(
                list(self.metrics['processing_times'])[-500:], 
                maxlen=1000
            )
        
        if len(self.metrics['memory_usage']) > 50:
            self.metrics['memory_usage'] = deque(
                list(self.metrics['memory_usage'])[-50:], 
                maxlen=100
            )
        
        # Clear old completed tasks
        current_time = time.time()
        old_tasks = [
            task_id for task_id, result in self.completed_tasks.items()
            if current_time - result['timestamp'] > 60  # 1 minute
        ]
        
        for task_id in old_tasks:
            self.completed_tasks.pop(task_id, None)
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory cleanup completed")
    
    async def _log_metrics(self):
        """Log performance metrics"""
        if not self.metrics['processing_times']:
            return
        
        avg_processing_time = np.mean(self.metrics['processing_times'])
        current_queue_size = self.priority_queue.qsize()
        current_memory = self.metrics['memory_usage'][-1] if self.metrics['memory_usage'] else 0
        current_throughput = self.metrics['throughput'][-1] if self.metrics['throughput'] else 0
        
        logger.info(f"Async Processor Metrics - "
                   f"Queue: {current_queue_size}, "
                   f"Avg Latency: {avg_processing_time:.2f}ms, "
                   f"Memory: {current_memory:.2f}MB, "
                   f"Throughput: {current_throughput:.2f} tasks/sec, "
                   f"Completed: {self.metrics['tasks_completed']}, "
                   f"Failed: {self.metrics['tasks_failed']}")
    
    async def get_task_result(self, task_id: str, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """Get result of a submitted task"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            await asyncio.sleep(0.01)
        
        return None  # Timeout
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {
            'tasks_submitted': self.metrics['tasks_submitted'],
            'tasks_completed': self.metrics['tasks_completed'],
            'tasks_failed': self.metrics['tasks_failed'],
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'circuit_breaker_state': self.circuit_breaker.get_state(),
            'strategy': self.config.strategy.value
        }
        
        # Add timing statistics
        if self.metrics['processing_times']:
            stats['processing_times'] = {
                'mean': np.mean(self.metrics['processing_times']),
                'std': np.std(self.metrics['processing_times']),
                'min': np.min(self.metrics['processing_times']),
                'max': np.max(self.metrics['processing_times']),
                'p95': np.percentile(self.metrics['processing_times'], 95)
            }
        
        # Add memory statistics
        if self.metrics['memory_usage']:
            stats['memory_usage'] = {
                'current': self.metrics['memory_usage'][-1],
                'mean': np.mean(self.metrics['memory_usage']),
                'max': np.max(self.metrics['memory_usage'])
            }
        
        # Add throughput statistics
        if self.metrics['throughput']:
            stats['throughput'] = {
                'current': self.metrics['throughput'][-1],
                'mean': np.mean(self.metrics['throughput']),
                'max': np.max(self.metrics['throughput'])
            }
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        stats = self.get_performance_stats()
        
        # Check if system is healthy
        is_healthy = True
        issues = []
        
        # Check processing times
        if stats.get('processing_times', {}).get('mean', 0) > self.config.max_latency_ms:
            is_healthy = False
            issues.append("High processing latency")
        
        # Check memory usage
        if stats.get('memory_usage', {}).get('current', 0) > self.config.max_memory_mb * 0.9:
            is_healthy = False
            issues.append("High memory usage")
        
        # Check failure rate
        total_tasks = stats['tasks_submitted']
        if total_tasks > 0:
            failure_rate = stats['tasks_failed'] / total_tasks
            if failure_rate > 0.1:  # 10% failure rate
                is_healthy = False
                issues.append("High failure rate")
        
        # Check circuit breaker
        if stats['circuit_breaker_state'] == 'open':
            is_healthy = False
            issues.append("Circuit breaker is open")
        
        return {
            'healthy': is_healthy,
            'issues': issues,
            'stats': stats,
            'timestamp': time.time()
        }


def create_async_processing_config(
    target_latency_ms: float = 20.0,
    max_workers: int = 4,
    strategy: ProcessingStrategy = ProcessingStrategy.HYBRID
) -> AsyncProcessingConfig:
    """Create optimized async processing configuration"""
    
    # Adjust concurrency based on available CPU cores
    cpu_count = mp.cpu_count()
    max_workers = min(max_workers, cpu_count)
    
    # Adjust coroutine concurrency based on target latency
    if target_latency_ms <= 10.0:
        coroutine_concurrency = 100
    elif target_latency_ms <= 20.0:
        coroutine_concurrency = 50
    else:
        coroutine_concurrency = 25
    
    return AsyncProcessingConfig(
        strategy=strategy,
        max_workers=max_workers,
        target_latency_ms=target_latency_ms,
        max_latency_ms=target_latency_ms * 2.5,
        coroutine_concurrency=coroutine_concurrency,
        enable_metrics=True,
        enable_circuit_breaker=True
    )


# Example usage
async def main():
    """Example usage of async gradient processor"""
    logging.basicConfig(level=logging.INFO)
    
    # Create model and optimizer
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create configuration
    config = create_async_processing_config(
        target_latency_ms=15.0,
        max_workers=4,
        strategy=ProcessingStrategy.HYBRID
    )
    
    # Initialize processor
    processor = AsyncGradientProcessor(model, optimizer, config)
    
    try:
        # Start processor
        await processor.start()
        
        print("Starting async gradient processing demo...")
        
        # Submit tasks
        task_ids = []
        for i in range(100):
            # Create fake gradients
            gradients = {
                'linear1.weight': torch.randn(256, 128),
                'linear1.bias': torch.randn(256),
                'linear2.weight': torch.randn(128, 256),
                'linear2.bias': torch.randn(128)
            }
            
            # Submit task
            task_id = await processor.submit_gradient(
                gradients,
                priority=TaskPriority.NORMAL,
                metadata={'step': i}
            )
            task_ids.append(task_id)
            
            # Add small delay to simulate real-time processing
            await asyncio.sleep(0.01)
        
        # Wait for some tasks to complete
        await asyncio.sleep(2.0)
        
        # Get results
        completed_count = 0
        for task_id in task_ids:
            result = await processor.get_task_result(task_id, timeout=1.0)
            if result and 'result' in result:
                completed_count += 1
        
        print(f"Completed {completed_count}/{len(task_ids)} tasks")
        
        # Show performance stats
        stats = processor.get_performance_stats()
        print(f"Performance Stats:")
        print(f"  Average Latency: {stats.get('processing_times', {}).get('mean', 0):.2f}ms")
        print(f"  Throughput: {stats.get('throughput', {}).get('mean', 0):.2f} tasks/sec")
        print(f"  Memory Usage: {stats.get('memory_usage', {}).get('current', 0):.2f}MB")
        
        # Health check
        health = await processor.health_check()
        print(f"System Health: {'Healthy' if health['healthy'] else 'Unhealthy'}")
        if health['issues']:
            print(f"Issues: {health['issues']}")
    
    finally:
        # Stop processor
        await processor.stop()
        print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())