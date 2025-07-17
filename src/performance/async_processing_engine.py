#!/usr/bin/env python3
"""
Async Processing Engine for Non-Blocking Operations
Implements async/await patterns, concurrent processing, and streaming data handling
"""

import asyncio
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union, Callable, AsyncIterator, Awaitable
import logging
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque, defaultdict
import multiprocessing as mp
from contextlib import asynccontextmanager
import weakref
import gc
import psutil
from functools import wraps
import traceback
from enum import Enum
import heapq
import pickle
from abc import ABC, abstractmethod
import uvloop  # High-performance event loop

from .advanced_caching_system import MultiLevelCache, get_global_cache
from .jit_optimized_engine import JITModelWrapper, OptimizationConfig

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class TaskResult:
    """Result of an async task"""
    task_id: str
    result: Any
    execution_time: float
    memory_usage: float
    status: str = "completed"
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class AsyncTaskConfig:
    """Configuration for async tasks"""
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    cache_result: bool = True
    cache_ttl: Optional[float] = None
    callback: Optional[Callable] = None

class AsyncTaskQueue:
    """Priority-based async task queue"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = []
        self.task_counter = 0
        self.lock = asyncio.Lock()
        self.not_empty = asyncio.Condition(self.lock)
        self.not_full = asyncio.Condition(self.lock)
        self.closed = False
        
    async def put(self, item: Tuple[TaskPriority, str, Callable, tuple, dict], 
                  timeout: Optional[float] = None) -> bool:
        """Put item in queue"""
        async with self.not_full:
            if timeout is not None:
                deadline = asyncio.get_event_loop().time() + timeout
            else:
                deadline = None
            
            while len(self.queue) >= self.max_size and not self.closed:
                if deadline is not None:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        return False
                    await asyncio.wait_for(self.not_full.wait(), timeout=remaining)
                else:
                    await self.not_full.wait()
            
            if self.closed:
                return False
            
            priority, task_id, func, args, kwargs = item
            # Use negative priority for max heap behavior
            heapq.heappush(self.queue, (-priority.value, self.task_counter, task_id, func, args, kwargs))
            self.task_counter += 1
            
            self.not_empty.notify()
            return True
    
    async def get(self, timeout: Optional[float] = None) -> Optional[Tuple[str, Callable, tuple, dict]]:
        """Get item from queue"""
        async with self.not_empty:
            if timeout is not None:
                deadline = asyncio.get_event_loop().time() + timeout
            else:
                deadline = None
            
            while not self.queue and not self.closed:
                if deadline is not None:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        return None
                    await asyncio.wait_for(self.not_empty.wait(), timeout=remaining)
                else:
                    await self.not_empty.wait()
            
            if not self.queue:
                return None
            
            _, _, task_id, func, args, kwargs = heapq.heappop(self.queue)
            self.not_full.notify()
            return task_id, func, args, kwargs
    
    async def close(self):
        """Close the queue"""
        async with self.lock:
            self.closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()
    
    @property
    def qsize(self) -> int:
        """Get queue size"""
        return len(self.queue)

class AsyncModelInference:
    """Async wrapper for model inference"""
    
    def __init__(self, model: Union[nn.Module, JITModelWrapper], 
                 max_batch_size: int = 32, 
                 batch_timeout: float = 0.01):
        self.model = model
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.inference_queue = asyncio.Queue()
        self.batch_processor_task = None
        self.stats = {
            'total_inferences': 0,
            'batched_inferences': 0,
            'avg_batch_size': 0.0,
            'avg_inference_time': 0.0,
            'total_batches': 0
        }
        self.lock = asyncio.Lock()
        
    async def start(self):
        """Start the batch processor"""
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
        
    async def stop(self):
        """Stop the batch processor"""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
    
    async def infer(self, input_data: torch.Tensor, timeout: float = 5.0) -> torch.Tensor:
        """Perform async inference"""
        future = asyncio.Future()
        
        # Add to queue
        await self.inference_queue.put((input_data, future))
        
        # Wait for result
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Inference timeout after {timeout}s")
            raise
    
    async def _batch_processor(self):
        """Process inference requests in batches"""
        while True:
            try:
                batch_inputs = []
                batch_futures = []
                
                # Collect batch
                deadline = asyncio.get_event_loop().time() + self.batch_timeout
                
                while len(batch_inputs) < self.max_batch_size:
                    remaining_time = deadline - asyncio.get_event_loop().time()
                    if remaining_time <= 0:
                        break
                    
                    try:
                        input_data, future = await asyncio.wait_for(
                            self.inference_queue.get(),
                            timeout=remaining_time
                        )
                        batch_inputs.append(input_data)
                        batch_futures.append(future)
                    except asyncio.TimeoutError:
                        break
                
                if not batch_inputs:
                    continue
                
                # Process batch
                start_time = time.time()
                
                try:
                    # Ensure all inputs have same shape for batching
                    if len(batch_inputs) > 1:
                        batch_tensor = torch.stack(batch_inputs)
                    else:
                        batch_tensor = batch_inputs[0].unsqueeze(0)
                    
                    # Perform inference
                    with torch.no_grad():
                        if hasattr(self.model, 'forward'):
                            batch_result = self.model(batch_tensor)
                        else:
                            batch_result = self.model(batch_tensor)
                    
                    # Handle different result types
                    if isinstance(batch_result, tuple):
                        # Multiple outputs - split each output
                        results = []
                        for i in range(len(batch_inputs)):
                            result_tuple = tuple(output[i] for output in batch_result)
                            results.append(result_tuple)
                    else:
                        # Single output
                        results = [batch_result[i] for i in range(len(batch_inputs))]
                    
                    # Set results
                    for i, future in enumerate(batch_futures):
                        if not future.cancelled():
                            future.set_result(results[i])
                    
                    # Update stats
                    processing_time = time.time() - start_time
                    async with self.lock:
                        self.stats['total_inferences'] += len(batch_inputs)
                        self.stats['batched_inferences'] += len(batch_inputs)
                        self.stats['total_batches'] += 1
                        self.stats['avg_batch_size'] = (
                            self.stats['avg_batch_size'] * (self.stats['total_batches'] - 1) + len(batch_inputs)
                        ) / self.stats['total_batches']
                        self.stats['avg_inference_time'] = (
                            self.stats['avg_inference_time'] * (self.stats['total_batches'] - 1) + processing_time
                        ) / self.stats['total_batches']
                    
                except Exception as e:
                    logger.error(f"Batch inference error: {e}")
                    # Set exception for all futures
                    for future in batch_futures:
                        if not future.cancelled():
                            future.set_exception(e)
                
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        async with self.lock:
            return self.stats.copy()

class AsyncDataStreamer:
    """Async data streamer for real-time processing"""
    
    def __init__(self, data_source: Union[str, Callable], 
                 buffer_size: int = 1000,
                 prefetch_size: int = 10):
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.prefetch_size = prefetch_size
        self.buffer = deque(maxlen=buffer_size)
        self.prefetch_task = None
        self.stats = {
            'items_streamed': 0,
            'buffer_overflows': 0,
            'avg_buffer_size': 0.0,
            'streaming_rate': 0.0
        }
        self.lock = asyncio.Lock()
        self.start_time = time.time()
        
    async def start_streaming(self):
        """Start the data streaming"""
        self.prefetch_task = asyncio.create_task(self._prefetch_data())
        
    async def stop_streaming(self):
        """Stop the data streaming"""
        if self.prefetch_task:
            self.prefetch_task.cancel()
            try:
                await self.prefetch_task
            except asyncio.CancelledError:
                pass
    
    async def get_batch(self, batch_size: int) -> List[Any]:
        """Get a batch of data"""
        async with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.buffer))):
                batch.append(self.buffer.popleft())
            return batch
    
    async def _prefetch_data(self):
        """Prefetch data in background"""
        while True:
            try:
                async with self.lock:
                    current_size = len(self.buffer)
                
                if current_size < self.buffer_size - self.prefetch_size:
                    # Need to prefetch more data
                    if callable(self.data_source):
                        # Data source is a function
                        for _ in range(self.prefetch_size):
                            try:
                                data = await self._call_data_source()
                                async with self.lock:
                                    if len(self.buffer) < self.buffer_size:
                                        self.buffer.append(data)
                                        self.stats['items_streamed'] += 1
                                    else:
                                        self.stats['buffer_overflows'] += 1
                            except Exception as e:
                                logger.error(f"Data source error: {e}")
                                break
                    else:
                        # Data source is a file/stream
                        await self._read_from_file()
                
                # Update stats
                async with self.lock:
                    self.stats['avg_buffer_size'] = (
                        self.stats['avg_buffer_size'] * 0.9 + len(self.buffer) * 0.1
                    )
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time > 0:
                        self.stats['streaming_rate'] = self.stats['items_streamed'] / elapsed_time
                
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Prefetch error: {e}")
                await asyncio.sleep(0.1)
    
    async def _call_data_source(self) -> Any:
        """Call the data source function"""
        if asyncio.iscoroutinefunction(self.data_source):
            return await self.data_source()
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.data_source)
    
    async def _read_from_file(self):
        """Read data from file source"""
        # Implementation depends on file type
        # This is a placeholder for file reading logic
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        async with self.lock:
            return self.stats.copy()

class AsyncWorkflowEngine:
    """Orchestrates async workflows with dependencies"""
    
    def __init__(self, max_workers: int = 10, use_process_pool: bool = False):
        self.max_workers = max_workers
        self.task_queue = AsyncTaskQueue(max_size=1000)
        self.workers = []
        self.use_process_pool = use_process_pool
        self.executor = None
        self.cache = get_global_cache()
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_task_time': 0.0,
            'tasks_per_second': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.lock = asyncio.Lock()
        self.start_time = time.time()
        
    async def start(self):
        """Start the workflow engine"""
        # Create thread or process pool
        if self.use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started async workflow engine with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the workflow engine"""
        # Close task queue
        await self.task_queue.close()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Stopped async workflow engine")
    
    async def submit_task(self, task_id: str, func: Callable, *args, 
                         config: AsyncTaskConfig = None, **kwargs) -> str:
        """Submit a task for execution"""
        if config is None:
            config = AsyncTaskConfig()
        
        # Check cache first
        if config.cache_result:
            cache_key = self._generate_cache_key(task_id, args, kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                async with self.lock:
                    self.stats['cache_hits'] += 1
                
                # Create fake result
                result = TaskResult(
                    task_id=task_id,
                    result=cached_result,
                    execution_time=0.0,
                    memory_usage=0.0,
                    status="cached"
                )
                self.completed_tasks[task_id] = result
                return task_id
        
        # Add to task queue
        await self.task_queue.put((config.priority, task_id, func, args, kwargs))
        
        async with self.lock:
            self.stats['total_tasks'] += 1
            self.stats['cache_misses'] += 1
        
        return task_id
    
    async def get_result(self, task_id: str, timeout: float = 30.0) -> TaskResult:
        """Get task result"""
        deadline = asyncio.get_event_loop().time() + timeout
        
        while asyncio.get_event_loop().time() < deadline:
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                return self.failed_tasks[task_id]
            
            # Wait a bit
            await asyncio.sleep(0.01)
        
        # Timeout
        raise asyncio.TimeoutError(f"Task {task_id} timeout after {timeout}s")
    
    async def _worker(self, worker_name: str):
        """Worker coroutine"""
        logger.info(f"Started worker {worker_name}")
        
        while True:
            try:
                # Get task from queue
                task_data = await self.task_queue.get(timeout=1.0)
                if task_data is None:
                    break
                
                task_id, func, args, kwargs = task_data
                
                # Execute task
                result = await self._execute_task(task_id, func, args, kwargs)
                
                # Store result
                if result.status == "completed":
                    self.completed_tasks[task_id] = result
                    
                    # Cache result if needed
                    if task_id not in self.failed_tasks:
                        cache_key = self._generate_cache_key(task_id, args, kwargs)
                        self.cache.put(cache_key, result.result, ttl=300)
                else:
                    self.failed_tasks[task_id] = result
                
                # Update stats
                async with self.lock:
                    if result.status == "completed":
                        self.stats['completed_tasks'] += 1
                    else:
                        self.stats['failed_tasks'] += 1
                    
                    # Update average task time
                    total_completed = self.stats['completed_tasks']
                    if total_completed > 0:
                        self.stats['avg_task_time'] = (
                            self.stats['avg_task_time'] * (total_completed - 1) + result.execution_time
                        ) / total_completed
                    
                    # Update tasks per second
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time > 0:
                        self.stats['tasks_per_second'] = self.stats['completed_tasks'] / elapsed_time
                
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"Stopped worker {worker_name}")
    
    async def _execute_task(self, task_id: str, func: Callable, args: tuple, kwargs: dict) -> TaskResult:
        """Execute a single task"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            self.active_tasks[task_id] = start_time
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            return TaskResult(
                task_id=task_id,
                result=result,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                status="completed"
            )
            
        except Exception as e:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            logger.error(f"Task {task_id} failed: {e}")
            
            return TaskResult(
                task_id=task_id,
                result=None,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                status="failed",
                error=str(e)
            )
        finally:
            self.active_tasks.pop(task_id, None)
    
    def _generate_cache_key(self, task_id: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for task"""
        key_data = {
            'task_id': task_id,
            'args': args,
            'kwargs': kwargs
        }
        return f"async_task_{hash(str(key_data))}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get workflow engine statistics"""
        async with self.lock:
            return {
                **self.stats,
                'active_tasks': len(self.active_tasks),
                'queue_size': self.task_queue.qsize,
                'uptime': time.time() - self.start_time
            }

class AsyncModelPipeline:
    """Async pipeline for model inference with preprocessing and postprocessing"""
    
    def __init__(self, models: Dict[str, Union[nn.Module, JITModelWrapper]],
                 preprocessing: Optional[Callable] = None,
                 postprocessing: Optional[Callable] = None):
        self.models = models
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.model_inferences = {}
        self.workflow_engine = AsyncWorkflowEngine(max_workers=4)
        
        # Initialize async inference for each model
        for name, model in models.items():
            self.model_inferences[name] = AsyncModelInference(model)
    
    async def start(self):
        """Start the pipeline"""
        await self.workflow_engine.start()
        
        for inference in self.model_inferences.values():
            await inference.start()
    
    async def stop(self):
        """Stop the pipeline"""
        await self.workflow_engine.stop()
        
        for inference in self.model_inferences.values():
            await inference.stop()
    
    async def process(self, input_data: Any, model_name: str) -> Any:
        """Process input through the pipeline"""
        # Preprocessing
        if self.preprocessing:
            preprocessed = await self._run_async_func(self.preprocessing, input_data)
        else:
            preprocessed = input_data
        
        # Model inference
        if model_name not in self.model_inferences:
            raise ValueError(f"Model {model_name} not found")
        
        inference_result = await self.model_inferences[model_name].infer(preprocessed)
        
        # Postprocessing
        if self.postprocessing:
            result = await self._run_async_func(self.postprocessing, inference_result)
        else:
            result = inference_result
        
        return result
    
    async def _run_async_func(self, func: Callable, *args, **kwargs) -> Any:
        """Run function asynchronously"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = {
            'workflow_stats': await self.workflow_engine.get_stats(),
            'model_stats': {}
        }
        
        for name, inference in self.model_inferences.items():
            stats['model_stats'][name] = await inference.get_stats()
        
        return stats

# Decorators for async processing
def async_cached(cache: MultiLevelCache, ttl: Optional[float] = None):
    """Decorator for caching async function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"async_{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = await func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            return result
        return wrapper
    return decorator

def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All retries failed for {func.__name__}")
                        raise last_exception
        return wrapper
    return decorator

# Example usage
async def main():
    """Example usage of async processing engine"""
    # Set up uvloop for better performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    # Create workflow engine
    workflow = AsyncWorkflowEngine(max_workers=4)
    await workflow.start()
    
    # Submit some tasks
    task_id1 = await workflow.submit_task("task1", lambda x: x * 2, 42)
    task_id2 = await workflow.submit_task("task2", lambda x: x + 10, 100)
    
    # Get results
    result1 = await workflow.get_result(task_id1)
    result2 = await workflow.get_result(task_id2)
    
    print(f"Task 1 result: {result1.result}")
    print(f"Task 2 result: {result2.result}")
    
    # Get stats
    stats = await workflow.get_stats()
    print(f"Workflow stats: {stats}")
    
    # Stop workflow
    await workflow.stop()

if __name__ == "__main__":
    asyncio.run(main())