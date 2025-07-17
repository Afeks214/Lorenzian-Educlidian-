"""
Parallel data processing implementation for high-performance data pipeline
"""

import time
import threading
import multiprocessing as mp
from multiprocessing import Pool, Queue, Process, Manager
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Iterator, Union, Tuple
from dataclasses import dataclass, field
import logging
import psutil
import gc
import pickle
import dill
from functools import partial
import asyncio
from contextlib import contextmanager
import signal
import os

from ..core.config import DataPipelineConfig
from ..core.exceptions import DataPreprocessingException
from ..core.data_loader import DataChunk, ScalableDataLoader
from ..streaming.data_streamer import DataStreamer
from ..preprocessing.data_processor import DataProcessor

logger = logging.getLogger(__name__)

@dataclass
class ParallelConfig:
    """Configuration for parallel processing"""
    max_workers: int = mp.cpu_count()
    use_processes: bool = True  # Use processes instead of threads
    chunk_size: int = 1000
    queue_size: int = 100
    timeout_seconds: float = 300.0
    enable_load_balancing: bool = True
    enable_fault_tolerance: bool = True
    memory_limit_per_worker_mb: float = 1024.0
    cpu_affinity: bool = False
    
    # Advanced options
    enable_work_stealing: bool = True
    enable_adaptive_scheduling: bool = True
    enable_result_caching: bool = True
    enable_progress_tracking: bool = True

class ParallelProcessor:
    """
    High-performance parallel data processor with advanced scheduling and fault tolerance
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None,
                 pipeline_config: Optional[DataPipelineConfig] = None):
        self.config = config or ParallelConfig()
        self.pipeline_config = pipeline_config or DataPipelineConfig()
        
        # Initialize components
        self.data_loader = ScalableDataLoader(self.pipeline_config)
        self.data_streamer = DataStreamer(self.pipeline_config)
        self.data_processor = DataProcessor(pipeline_config=self.pipeline_config)
        
        # Parallel processing state
        self.task_queue = queue.Queue(maxsize=self.config.queue_size)
        self.result_queue = queue.Queue()
        self.workers = []
        self.is_running = False
        self.stats = ParallelProcessingStats()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def process_files_parallel(self, file_paths: List[str],
                             processing_func: Callable[[DataChunk], DataChunk],
                             **kwargs) -> Iterator[DataChunk]:
        """
        Process multiple files in parallel
        """
        try:
            self.is_running = True
            self.stats.reset()
            
            if self.config.use_processes:
                yield from self._process_files_with_processes(file_paths, processing_func, **kwargs)
            else:
                yield from self._process_files_with_threads(file_paths, processing_func, **kwargs)
                
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            raise DataPreprocessingException(f"Parallel processing failed: {str(e)}")
        finally:
            self.is_running = False
    
    def _process_files_with_processes(self, file_paths: List[str],
                                    processing_func: Callable[[DataChunk], DataChunk],
                                    **kwargs) -> Iterator[DataChunk]:
        """Process files using multiprocessing"""
        # Create worker processes
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {}
            
            for file_path in file_paths:
                for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                    future = executor.submit(self._process_chunk_worker, chunk, processing_func)
                    future_to_chunk[future] = chunk
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result is not None:
                        self.stats.chunks_processed += 1
                        yield result
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
                    if self.config.enable_fault_tolerance:
                        # Try to recover or skip
                        continue
                    else:
                        raise
    
    def _process_files_with_threads(self, file_paths: List[str],
                                  processing_func: Callable[[DataChunk], DataChunk],
                                  **kwargs) -> Iterator[DataChunk]:
        """Process files using multithreading"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunks for processing
            future_to_chunk = {}
            
            for file_path in file_paths:
                for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                    future = executor.submit(self._process_chunk_worker, chunk, processing_func)
                    future_to_chunk[future] = chunk
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result is not None:
                        self.stats.chunks_processed += 1
                        yield result
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
                    if self.config.enable_fault_tolerance:
                        continue
                    else:
                        raise
    
    def _process_chunk_worker(self, chunk: DataChunk, 
                            processing_func: Callable[[DataChunk], DataChunk]) -> Optional[DataChunk]:
        """Worker function for processing chunks"""
        try:
            # Monitor memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.config.memory_limit_per_worker_mb:
                logger.warning(f"Worker memory usage high: {memory_mb:.2f}MB")
                gc.collect()
            
            # Process chunk
            start_time = time.time()
            result = processing_func(chunk)
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats.update_processing_time(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in worker processing chunk {chunk.chunk_id}: {str(e)}")
            return None
    
    def batch_process_parallel(self, file_paths: List[str],
                             batch_size: int,
                             processing_func: Callable[[List[DataChunk]], List[DataChunk]],
                             **kwargs) -> Iterator[List[DataChunk]]:
        """
        Process data in batches in parallel
        """
        # Collect chunks into batches
        batches = []
        current_batch = []
        
        for file_path in file_paths:
            for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                current_batch.append(chunk)
                
                if len(current_batch) >= batch_size:
                    batches.append(current_batch)
                    current_batch = []
        
        # Add remaining batch
        if current_batch:
            batches.append(current_batch)
        
        # Process batches in parallel
        if self.config.use_processes:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(processing_func, batch) for batch in batches]
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        if result:
                            yield result
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        if not self.config.enable_fault_tolerance:
                            raise
        else:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(processing_func, batch) for batch in batches]
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        if result:
                            yield result
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        if not self.config.enable_fault_tolerance:
                            raise
    
    def map_reduce_parallel(self, file_paths: List[str],
                          map_func: Callable[[DataChunk], Any],
                          reduce_func: Callable[[List[Any]], Any],
                          **kwargs) -> Any:
        """
        Perform map-reduce processing in parallel
        """
        # Map phase
        map_results = []
        
        if self.config.use_processes:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                for file_path in file_paths:
                    for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                        future = executor.submit(map_func, chunk)
                        futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        if result is not None:
                            map_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in map phase: {str(e)}")
                        if not self.config.enable_fault_tolerance:
                            raise
        else:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                for file_path in file_paths:
                    for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                        future = executor.submit(map_func, chunk)
                        futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        if result is not None:
                            map_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in map phase: {str(e)}")
                        if not self.config.enable_fault_tolerance:
                            raise
        
        # Reduce phase
        try:
            return reduce_func(map_results)
        except Exception as e:
            logger.error(f"Error in reduce phase: {str(e)}")
            raise DataPreprocessingException(f"Reduce phase failed: {str(e)}")
    
    def stream_process_parallel(self, file_paths: List[str],
                              processing_func: Callable[[DataChunk], DataChunk],
                              **kwargs) -> Iterator[DataChunk]:
        """
        Stream and process data in parallel
        """
        # Create producer-consumer pattern
        chunk_queue = queue.Queue(maxsize=self.config.queue_size)
        result_queue = queue.Queue()
        
        # Producer thread
        def producer():
            try:
                for file_path in file_paths:
                    for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                        chunk_queue.put(chunk)
                # Signal end of input
                for _ in range(self.config.max_workers):
                    chunk_queue.put(None)
            except Exception as e:
                logger.error(f"Error in producer: {str(e)}")
        
        # Consumer workers
        def consumer():
            try:
                while True:
                    chunk = chunk_queue.get()
                    if chunk is None:
                        break
                    
                    result = processing_func(chunk)
                    if result is not None:
                        result_queue.put(result)
                    
                    chunk_queue.task_done()
            except Exception as e:
                logger.error(f"Error in consumer: {str(e)}")
            finally:
                result_queue.put(None)  # Signal end of processing
        
        # Start producer
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        
        # Start consumers
        consumer_threads = []
        for _ in range(self.config.max_workers):
            consumer_thread = threading.Thread(target=consumer)
            consumer_thread.start()
            consumer_threads.append(consumer_thread)
        
        # Collect results
        active_consumers = self.config.max_workers
        while active_consumers > 0:
            try:
                result = result_queue.get(timeout=self.config.timeout_seconds)
                if result is None:
                    active_consumers -= 1
                else:
                    yield result
            except queue.Empty:
                break
        
        # Wait for threads to complete
        producer_thread.join()
        for thread in consumer_threads:
            thread.join()
    
    def pipeline_parallel(self, file_paths: List[str],
                        pipeline_stages: List[Callable[[DataChunk], DataChunk]],
                        **kwargs) -> Iterator[DataChunk]:
        """
        Process data through a pipeline of stages in parallel
        """
        # Create queues for each stage
        stage_queues = [queue.Queue(maxsize=self.config.queue_size) 
                       for _ in range(len(pipeline_stages) + 1)]
        
        # Stage workers
        def stage_worker(stage_func: Callable, input_queue: queue.Queue, 
                        output_queue: queue.Queue):
            try:
                while True:
                    chunk = input_queue.get()
                    if chunk is None:
                        output_queue.put(None)
                        break
                    
                    result = stage_func(chunk)
                    if result is not None:
                        output_queue.put(result)
                    
                    input_queue.task_done()
            except Exception as e:
                logger.error(f"Error in stage worker: {str(e)}")
        
        # Producer
        def producer():
            try:
                for file_path in file_paths:
                    for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                        stage_queues[0].put(chunk)
                stage_queues[0].put(None)
            except Exception as e:
                logger.error(f"Error in producer: {str(e)}")
        
        # Start producer
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        
        # Start stage workers
        worker_threads = []
        for i, stage_func in enumerate(pipeline_stages):
            worker_thread = threading.Thread(
                target=stage_worker,
                args=(stage_func, stage_queues[i], stage_queues[i + 1])
            )
            worker_thread.start()
            worker_threads.append(worker_thread)
        
        # Collect results from final stage
        while True:
            try:
                result = stage_queues[-1].get(timeout=self.config.timeout_seconds)
                if result is None:
                    break
                yield result
            except queue.Empty:
                break
        
        # Wait for all threads to complete
        producer_thread.join()
        for thread in worker_threads:
            thread.join()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics"""
        return {
            'chunks_processed': self.stats.chunks_processed,
            'avg_processing_time': self.stats.get_avg_processing_time(),
            'total_processing_time': self.stats.get_total_processing_time(),
            'throughput': self.stats.get_throughput(),
            'max_workers': self.config.max_workers,
            'use_processes': self.config.use_processes,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
    
    def shutdown(self):
        """Shutdown parallel processor"""
        self.is_running = False
        logger.info("Parallel processor shutdown")


class ParallelProcessingStats:
    """Statistics for parallel processing operations"""
    
    def __init__(self):
        self.chunks_processed = 0
        self.processing_times = []
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def update_processing_time(self, processing_time: float):
        """Update processing time statistics"""
        with self._lock:
            self.processing_times.append(processing_time)
            
            # Keep only recent times (last 1000)
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time"""
        with self._lock:
            return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def get_total_processing_time(self) -> float:
        """Get total processing time"""
        with self._lock:
            return sum(self.processing_times)
    
    def get_throughput(self) -> float:
        """Get processing throughput"""
        with self._lock:
            elapsed = time.time() - self.start_time
            return self.chunks_processed / elapsed if elapsed > 0 else 0.0
    
    def reset(self):
        """Reset statistics"""
        with self._lock:
            self.chunks_processed = 0
            self.processing_times = []
            self.start_time = time.time()


class WorkerPool:
    """Advanced worker pool with load balancing and fault tolerance"""
    
    def __init__(self, max_workers: int, use_processes: bool = True):
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.workers = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        self.stats = WorkerPoolStats()
    
    def start(self):
        """Start worker pool"""
        self.is_running = True
        
        if self.use_processes:
            self.workers = [
                Process(target=self._worker_process)
                for _ in range(self.max_workers)
            ]
        else:
            self.workers = [
                threading.Thread(target=self._worker_thread)
                for _ in range(self.max_workers)
            ]
        
        for worker in self.workers:
            worker.start()
    
    def _worker_process(self):
        """Worker process function"""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    break
                
                func, args, kwargs = task
                result = func(*args, **kwargs)
                self.result_queue.put(result)
                
                self.task_queue.task_done()
                self.stats.tasks_completed += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker process: {str(e)}")
                self.stats.tasks_failed += 1
    
    def _worker_thread(self):
        """Worker thread function"""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    break
                
                func, args, kwargs = task
                result = func(*args, **kwargs)
                self.result_queue.put(result)
                
                self.task_queue.task_done()
                self.stats.tasks_completed += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker thread: {str(e)}")
                self.stats.tasks_failed += 1
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit task to worker pool"""
        self.task_queue.put((func, args, kwargs))
        self.stats.tasks_submitted += 1
    
    def get_result(self, timeout: Optional[float] = None) -> Any:
        """Get result from worker pool"""
        return self.result_queue.get(timeout=timeout)
    
    def shutdown(self):
        """Shutdown worker pool"""
        self.is_running = False
        
        # Signal workers to stop
        for _ in range(self.max_workers):
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join()
        
        logger.info("Worker pool shutdown")


class WorkerPoolStats:
    """Statistics for worker pool operations"""
    
    def __init__(self):
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.start_time = time.time()
    
    def get_success_rate(self) -> float:
        """Get task success rate"""
        total_tasks = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total_tasks if total_tasks > 0 else 0.0
    
    def get_throughput(self) -> float:
        """Get task throughput"""
        elapsed = time.time() - self.start_time
        return self.tasks_completed / elapsed if elapsed > 0 else 0.0


# Utility functions
def create_parallel_processor(max_workers: Optional[int] = None,
                            use_processes: bool = True) -> ParallelProcessor:
    """Create a parallel processor with optimal configuration"""
    max_workers = max_workers or mp.cpu_count()
    
    config = ParallelConfig(
        max_workers=max_workers,
        use_processes=use_processes,
        enable_fault_tolerance=True,
        enable_load_balancing=True
    )
    
    return ParallelProcessor(config)

def parallel_apply(data_chunks: List[DataChunk],
                  func: Callable[[DataChunk], DataChunk],
                  max_workers: Optional[int] = None) -> List[DataChunk]:
    """Apply function to data chunks in parallel"""
    max_workers = max_workers or mp.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, data_chunks))
    
    return [result for result in results if result is not None]

@contextmanager
def parallel_processing_context(max_workers: Optional[int] = None,
                               use_processes: bool = True):
    """Context manager for parallel processing"""
    processor = create_parallel_processor(max_workers, use_processes)
    
    try:
        yield processor
    finally:
        processor.shutdown()