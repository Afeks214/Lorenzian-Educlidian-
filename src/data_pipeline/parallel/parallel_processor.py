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
from typing import List, Dict, Any, Optional, Callable, Iterator, Union, Tuple, NamedTuple
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
from collections import deque, defaultdict
import heapq
from enum import Enum
import uuid
import resource
import platform
from abc import ABC, abstractmethod
import statistics
from weakref import WeakSet
import json
from datetime import datetime, timedelta
from enum import Enum
import warnings
import traceback
import random
from pathlib import Path

from ..core.config import DataPipelineConfig
from ..core.exceptions import DataPreprocessingException
from ..core.data_loader import DataChunk, ScalableDataLoader
from ..streaming.data_streamer import DataStreamer
from ..preprocessing.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker state enumeration"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    RECOVERING = "recovering"
    TERMINATED = "terminated"


class TaskState(Enum):
    """Task state enumeration"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class AlertLevel(Enum):
    """Alert level enumeration"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class WorkerHealthMetrics:
    """Worker health metrics"""
    worker_id: str
    state: WorkerState
    last_heartbeat: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    task_queue_size: int
    tasks_completed: int
    tasks_failed: int
    avg_task_duration: float
    error_rate: float
    uptime: float
    last_error: Optional[str] = None
    
    def is_healthy(self) -> bool:
        """Check if worker is healthy"""
        return (
            self.state in [WorkerState.HEALTHY, WorkerState.INITIALIZING] and
            self.cpu_percent < 90.0 and
            self.memory_percent < 85.0 and
            self.error_rate < 0.1 and
            (datetime.now() - self.last_heartbeat).total_seconds() < 30
        )
    
    def is_degraded(self) -> bool:
        """Check if worker is degraded"""
        return (
            self.cpu_percent > 80.0 or
            self.memory_percent > 75.0 or
            self.error_rate > 0.05 or
            self.avg_task_duration > 10.0
        )


@dataclass
class TaskMetadata:
    """Task metadata for tracking and recovery"""
    task_id: str
    chunk_id: str
    state: TaskState
    worker_id: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    retry_count: int
    max_retries: int
    error_message: Optional[str]
    checkpoint_data: Optional[Dict[str, Any]]
    
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.retry_count < self.max_retries and self.state == TaskState.FAILED


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    message: str
    component: str
    worker_id: Optional[str]
    timestamp: datetime
    resolved: bool = False
    metadata: Optional[Dict[str, Any]] = None

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4

class SchedulingPolicy(Enum):
    """Scheduling policies for task allocation"""
    ROUND_ROBIN = "round_robin"
    WORK_STEALING = "work_stealing"
    RESOURCE_AWARE = "resource_aware"
    DEADLINE_AWARE = "deadline_aware"
    PREDICTIVE = "predictive"

@dataclass
class TaskMetadata:
    """Metadata for tasks in the parallel processing system"""
    task_id: str
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: Optional[float] = None
    estimated_duration: Optional[float] = None
    memory_requirement: Optional[float] = None  # MB
    cpu_requirement: Optional[float] = None  # CPU cores
    dependency_tasks: List[str] = field(default_factory=list)
    submit_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadBalancingConfig:
    """Configuration for load balancing features"""
    enable_work_stealing: bool = True
    enable_adaptive_scaling: bool = True
    enable_predictive_scheduling: bool = True
    enable_resource_awareness: bool = True
    enable_deadline_scheduling: bool = True
    enable_numa_awareness: bool = True
    
    # Work stealing parameters
    steal_threshold: float = 0.5  # When to steal work (queue utilization)
    steal_batch_size: int = 1  # Number of tasks to steal at once
    steal_backoff_ms: int = 100  # Backoff time between steal attempts
    
    # Adaptive scaling parameters
    scale_up_threshold: float = 0.8  # CPU utilization to scale up
    scale_down_threshold: float = 0.3  # CPU utilization to scale down
    scale_up_cooldown: float = 60.0  # Cooldown between scale-ups
    scale_down_cooldown: float = 300.0  # Cooldown between scale-downs
    min_workers: int = 1
    max_workers: int = mp.cpu_count() * 2
    
    # Performance monitoring
    performance_history_size: int = 1000
    performance_update_interval: float = 5.0  # seconds
    
    # Cache-aware scheduling
    enable_cache_awareness: bool = True
    cache_affinity_weight: float = 0.3
    
    # Resource monitoring
    resource_monitoring_interval: float = 1.0  # seconds
    memory_pressure_threshold: float = 0.8  # Memory utilization threshold
    cpu_pressure_threshold: float = 0.9  # CPU utilization threshold

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
    
    # Load balancing configuration
    load_balancing: LoadBalancingConfig = field(default_factory=LoadBalancingConfig)
    
    # Scheduling policy
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.WORK_STEALING
    
    # Fault tolerance configuration
    max_task_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0
    worker_timeout_seconds: float = 60.0
    heartbeat_interval_seconds: float = 5.0
    health_check_interval_seconds: float = 10.0
    
    # Health monitoring configuration
    enable_health_monitoring: bool = True
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 75.0
    disk_threshold_percent: float = 85.0
    error_rate_threshold: float = 0.05
    task_duration_threshold_seconds: float = 10.0
    
    # Recovery configuration
    enable_auto_recovery: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval_seconds: float = 30.0
    checkpoint_directory: str = "/tmp/parallel_processor_checkpoints"
    
    # Alerting configuration
    enable_alerting: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_critical': 90.0,
        'memory_critical': 85.0,
        'error_rate_critical': 0.1,
        'worker_timeout_critical': 120.0
    })

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
        
        # Fault tolerance and health monitoring
        self.worker_health_monitor = WorkerHealthMonitor(self.config)
        self.task_manager = TaskManager(self.config)
        self.alert_manager = AlertManager(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        
        # Worker tracking
        self.worker_metrics: Dict[str, WorkerHealthMetrics] = {}
        self.active_tasks: Dict[str, TaskMetadata] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Monitoring threads
        self._monitoring_threads = []
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Initialize checkpoint directory
        self._initialize_checkpoint_directory()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _initialize_checkpoint_directory(self):
        """Initialize checkpoint directory"""
        if self.config.enable_checkpointing:
            checkpoint_dir = Path(self.config.checkpoint_directory)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoint directory initialized: {checkpoint_dir}")
    
    def _start_monitoring_threads(self):
        """Start monitoring threads"""
        if self.config.enable_health_monitoring:
            health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
            health_thread.start()
            self._monitoring_threads.append(health_thread)
            
            heartbeat_thread = threading.Thread(target=self._heartbeat_monitoring_loop, daemon=True)
            heartbeat_thread.start()
            self._monitoring_threads.append(heartbeat_thread)
        
        if self.config.enable_checkpointing:
            checkpoint_thread = threading.Thread(target=self._checkpoint_loop, daemon=True)
            checkpoint_thread.start()
            self._monitoring_threads.append(checkpoint_thread)
    
    def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        while self.is_running:
            try:
                self.worker_health_monitor.check_worker_health(self.worker_metrics)
                self._detect_performance_degradation()
                time.sleep(self.config.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.config.health_check_interval_seconds)
    
    def _heartbeat_monitoring_loop(self):
        """Heartbeat monitoring loop"""
        while self.is_running:
            try:
                self._check_worker_heartbeats()
                time.sleep(self.config.heartbeat_interval_seconds)
            except Exception as e:
                logger.error(f"Error in heartbeat monitoring loop: {e}")
                time.sleep(self.config.heartbeat_interval_seconds)
    
    def _checkpoint_loop(self):
        """Checkpoint loop"""
        while self.is_running:
            try:
                self.checkpoint_manager.create_checkpoint(self.active_tasks)
                time.sleep(self.config.checkpoint_interval_seconds)
            except Exception as e:
                logger.error(f"Error in checkpoint loop: {e}")
                time.sleep(self.config.checkpoint_interval_seconds)
    
    def _check_worker_heartbeats(self):
        """Check worker heartbeats and detect failures"""
        current_time = datetime.now()
        
        with self._lock:
            for worker_id, metrics in self.worker_metrics.items():
                time_since_heartbeat = (current_time - metrics.last_heartbeat).total_seconds()
                
                if time_since_heartbeat > self.config.worker_timeout_seconds:
                    logger.warning(f"Worker {worker_id} missed heartbeat, marking as failed")
                    metrics.state = WorkerState.FAILED
                    
                    # Generate alert
                    alert = Alert(
                        id=str(uuid.uuid4()),
                        level=AlertLevel.ERROR,
                        message=f"Worker {worker_id} failed heartbeat check",
                        component="worker_monitor",
                        worker_id=worker_id,
                        timestamp=current_time
                    )
                    self.alert_manager.add_alert(alert)
                    
                    # Trigger recovery if enabled
                    if self.config.enable_auto_recovery:
                        self._recover_failed_worker(worker_id)
    
    def _detect_performance_degradation(self):
        """Detect performance degradation"""
        with self._lock:
            for worker_id, metrics in self.worker_metrics.items():
                if metrics.is_degraded() and metrics.state == WorkerState.HEALTHY:
                    logger.warning(f"Worker {worker_id} performance degraded")
                    metrics.state = WorkerState.DEGRADED
                    
                    # Generate alert
                    alert = Alert(
                        id=str(uuid.uuid4()),
                        level=AlertLevel.WARNING,
                        message=f"Worker {worker_id} performance degraded",
                        component="performance_monitor",
                        worker_id=worker_id,
                        timestamp=datetime.now(),
                        metadata={
                            'cpu_percent': metrics.cpu_percent,
                            'memory_percent': metrics.memory_percent,
                            'error_rate': metrics.error_rate
                        }
                    )
                    self.alert_manager.add_alert(alert)
    
    def _recover_failed_worker(self, worker_id: str):
        """Recover failed worker"""
        try:
            logger.info(f"Attempting to recover failed worker {worker_id}")
            
            # Reassign tasks from failed worker
            self.task_manager.reassign_worker_tasks(worker_id, self.active_tasks)
            
            # Remove worker from metrics
            if worker_id in self.worker_metrics:
                del self.worker_metrics[worker_id]
            
            # Start replacement worker if needed
            if len(self.worker_metrics) < self.config.max_workers:
                self._start_replacement_worker()
                
        except Exception as e:
            logger.error(f"Error recovering worker {worker_id}: {e}")
    
    def _start_replacement_worker(self):
        """Start replacement worker"""
        try:
            # Implementation depends on worker type (process vs thread)
            # This is a placeholder for the actual worker creation logic
            logger.info("Starting replacement worker")
            
        except Exception as e:
            logger.error(f"Error starting replacement worker: {e}")
    
    def process_files_parallel(self, file_paths: List[str],
                             processing_func: Callable[[DataChunk], DataChunk],
                             **kwargs) -> Iterator[DataChunk]:
        """
        Process multiple files in parallel with fault tolerance
        """
        try:
            self.is_running = True
            self.stats.reset()
            
            # Start monitoring threads
            self._start_monitoring_threads()
            
            # Load checkpoint if available
            if self.config.enable_checkpointing:
                self.checkpoint_manager.load_checkpoint(self.active_tasks)
            
            if self.config.use_processes:
                yield from self._process_files_with_processes_ft(file_paths, processing_func, **kwargs)
            else:
                yield from self._process_files_with_threads_ft(file_paths, processing_func, **kwargs)
                
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            
            # Generate critical alert
            alert = Alert(
                id=str(uuid.uuid4()),
                level=AlertLevel.CRITICAL,
                message=f"Parallel processing failed: {str(e)}",
                component="parallel_processor",
                worker_id=None,
                timestamp=datetime.now()
            )
            self.alert_manager.add_alert(alert)
            
            raise DataPreprocessingException(f"Parallel processing failed: {str(e)}")
        finally:
            self.is_running = False
            
            # Wait for monitoring threads to complete
            for thread in self._monitoring_threads:
                thread.join(timeout=1.0)
    
    def _process_files_with_processes_ft(self, file_paths: List[str],
                                       processing_func: Callable[[DataChunk], DataChunk],
                                       **kwargs) -> Iterator[DataChunk]:
        """Process files using multiprocessing with fault tolerance"""
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunks for processing
            future_to_task = {}
            
            for file_path in file_paths:
                for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                    # Create task metadata
                    task_id = str(uuid.uuid4())
                    task_metadata = TaskMetadata(
                        task_id=task_id,
                        chunk_id=chunk.chunk_id,
                        state=TaskState.PENDING,
                        worker_id=None,
                        created_at=datetime.now(),
                        started_at=None,
                        completed_at=None,
                        retry_count=0,
                        max_retries=self.config.max_task_retries,
                        error_message=None,
                        checkpoint_data=None
                    )
                    
                    # Submit task
                    future = executor.submit(self._process_chunk_worker_ft, chunk, processing_func, task_metadata)
                    future_to_task[future] = task_metadata
                    
                    # Track active task
                    with self._lock:
                        self.active_tasks[task_id] = task_metadata
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_metadata = future_to_task[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result is not None:
                        task_metadata.state = TaskState.COMPLETED
                        task_metadata.completed_at = datetime.now()
                        self.stats.chunks_processed += 1
                        yield result
                    else:
                        task_metadata.state = TaskState.FAILED
                        
                except Exception as e:
                    task_metadata.state = TaskState.FAILED
                    task_metadata.error_message = str(e)
                    logger.error(f"Error processing task {task_metadata.task_id}: {str(e)}")
                    
                    # Retry logic
                    if self.config.enable_fault_tolerance and task_metadata.can_retry():
                        retry_result = self._retry_task(task_metadata, processing_func, executor)
                        if retry_result is not None:
                            yield retry_result
                    else:
                        # Generate alert for failed task
                        alert = Alert(
                            id=str(uuid.uuid4()),
                            level=AlertLevel.ERROR,
                            message=f"Task {task_metadata.task_id} failed after {task_metadata.retry_count} retries",
                            component="task_processor",
                            worker_id=task_metadata.worker_id,
                            timestamp=datetime.now()
                        )
                        self.alert_manager.add_alert(alert)
                
                finally:
                    # Remove from active tasks
                    with self._lock:
                        if task_metadata.task_id in self.active_tasks:
                            del self.active_tasks[task_metadata.task_id]
    
    def _process_files_with_threads_ft(self, file_paths: List[str],
                                     processing_func: Callable[[DataChunk], DataChunk],
                                     **kwargs) -> Iterator[DataChunk]:
        """Process files using multithreading with fault tolerance"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all chunks for processing
            future_to_task = {}
            
            for file_path in file_paths:
                for chunk in self.data_loader.load_chunks(file_path, **kwargs):
                    # Create task metadata
                    task_id = str(uuid.uuid4())
                    task_metadata = TaskMetadata(
                        task_id=task_id,
                        chunk_id=chunk.chunk_id,
                        state=TaskState.PENDING,
                        worker_id=None,
                        created_at=datetime.now(),
                        started_at=None,
                        completed_at=None,
                        retry_count=0,
                        max_retries=self.config.max_task_retries,
                        error_message=None,
                        checkpoint_data=None
                    )
                    
                    # Submit task
                    future = executor.submit(self._process_chunk_worker_ft, chunk, processing_func, task_metadata)
                    future_to_task[future] = task_metadata
                    
                    # Track active task
                    with self._lock:
                        self.active_tasks[task_id] = task_metadata
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_metadata = future_to_task[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    if result is not None:
                        task_metadata.state = TaskState.COMPLETED
                        task_metadata.completed_at = datetime.now()
                        self.stats.chunks_processed += 1
                        yield result
                    else:
                        task_metadata.state = TaskState.FAILED
                        
                except Exception as e:
                    task_metadata.state = TaskState.FAILED
                    task_metadata.error_message = str(e)
                    logger.error(f"Error processing task {task_metadata.task_id}: {str(e)}")
                    
                    # Retry logic
                    if self.config.enable_fault_tolerance and task_metadata.can_retry():
                        retry_result = self._retry_task(task_metadata, processing_func, executor)
                        if retry_result is not None:
                            yield retry_result
                    else:
                        # Generate alert for failed task
                        alert = Alert(
                            id=str(uuid.uuid4()),
                            level=AlertLevel.ERROR,
                            message=f"Task {task_metadata.task_id} failed after {task_metadata.retry_count} retries",
                            component="task_processor",
                            worker_id=task_metadata.worker_id,
                            timestamp=datetime.now()
                        )
                        self.alert_manager.add_alert(alert)
                
                finally:
                    # Remove from active tasks
                    with self._lock:
                        if task_metadata.task_id in self.active_tasks:
                            del self.active_tasks[task_metadata.task_id]
    
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
        """Get comprehensive parallel processing statistics"""
        base_stats = {
            'chunks_processed': self.stats.chunks_processed,
            'avg_processing_time': self.stats.get_avg_processing_time(),
            'total_processing_time': self.stats.get_total_processing_time(),
            'throughput': self.stats.get_throughput(),
            'max_workers': self.config.max_workers,
            'use_processes': self.config.use_processes,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        # Add health monitoring stats
        if self.config.enable_health_monitoring:
            healthy_workers = sum(1 for metrics in self.worker_metrics.values() if metrics.is_healthy())
            degraded_workers = sum(1 for metrics in self.worker_metrics.values() if metrics.is_degraded())
            unhealthy_workers = sum(1 for metrics in self.worker_metrics.values() if not metrics.is_healthy())
            
            base_stats.update({
                'worker_health': {
                    'total_workers': len(self.worker_metrics),
                    'healthy_workers': healthy_workers,
                    'degraded_workers': degraded_workers,
                    'unhealthy_workers': unhealthy_workers,
                    'avg_cpu_percent': sum(m.cpu_percent for m in self.worker_metrics.values()) / len(self.worker_metrics) if self.worker_metrics else 0,
                    'avg_memory_percent': sum(m.memory_percent for m in self.worker_metrics.values()) / len(self.worker_metrics) if self.worker_metrics else 0,
                    'avg_error_rate': sum(m.error_rate for m in self.worker_metrics.values()) / len(self.worker_metrics) if self.worker_metrics else 0
                }
            })
        
        # Add fault tolerance stats
        if self.config.enable_fault_tolerance:
            active_tasks = len(self.active_tasks)
            pending_tasks = sum(1 for task in self.active_tasks.values() if task.state == TaskState.PENDING)
            running_tasks = sum(1 for task in self.active_tasks.values() if task.state == TaskState.RUNNING)
            failed_tasks = sum(1 for task in self.active_tasks.values() if task.state == TaskState.FAILED)
            retrying_tasks = sum(1 for task in self.active_tasks.values() if task.state == TaskState.RETRYING)
            
            base_stats.update({
                'fault_tolerance': {
                    'active_tasks': active_tasks,
                    'pending_tasks': pending_tasks,
                    'running_tasks': running_tasks,
                    'failed_tasks': failed_tasks,
                    'retrying_tasks': retrying_tasks
                }
            })
        
        # Add alerting stats
        if self.config.enable_alerting:
            active_alerts = self.alert_manager.get_active_alerts()
            critical_alerts = sum(1 for alert in active_alerts if alert.level == AlertLevel.CRITICAL)
            error_alerts = sum(1 for alert in active_alerts if alert.level == AlertLevel.ERROR)
            warning_alerts = sum(1 for alert in active_alerts if alert.level == AlertLevel.WARNING)
            
            base_stats.update({
                'alerts': {
                    'total_active_alerts': len(active_alerts),
                    'critical_alerts': critical_alerts,
                    'error_alerts': error_alerts,
                    'warning_alerts': warning_alerts
                }
            })
        
        return base_stats
    
    def shutdown(self):
        """Shutdown parallel processor with cleanup"""
        self.is_running = False
        
        # Create final checkpoint if enabled
        if self.config.enable_checkpointing:
            self.checkpoint_manager.create_checkpoint(self.active_tasks)
        
        # Wait for monitoring threads to complete
        for thread in self._monitoring_threads:
            thread.join(timeout=5.0)
        
        # Generate shutdown alert
        if self.config.enable_alerting:
            alert = Alert(
                id=str(uuid.uuid4()),
                level=AlertLevel.INFO,
                message="Parallel processor shutdown initiated",
                component="parallel_processor",
                worker_id=None,
                timestamp=datetime.now()
            )
            self.alert_manager.add_alert(alert)
        
        logger.info("Parallel processor shutdown completed")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            'is_running': self.is_running,
            'worker_count': len(self.worker_metrics),
            'healthy_workers': sum(1 for m in self.worker_metrics.values() if m.is_healthy()),
            'active_tasks': len(self.active_tasks),
            'active_alerts': len(self.alert_manager.get_active_alerts()) if self.config.enable_alerting else 0,
            'last_checkpoint': self.checkpoint_manager.checkpoint_dir if self.config.enable_checkpointing else None
        }


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


class WorkerHealthMonitor:
    """Monitor worker health and performance"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".WorkerHealthMonitor")
    
    def check_worker_health(self, worker_metrics: Dict[str, WorkerHealthMetrics]):
        """Check health of all workers"""
        for worker_id, metrics in worker_metrics.items():
            if not metrics.is_healthy():
                self.logger.warning(f"Worker {worker_id} is unhealthy: {metrics.state}")
                
                # Check for critical conditions
                if metrics.cpu_percent > self.config.alert_thresholds.get('cpu_critical', 90.0):
                    self.logger.critical(f"Worker {worker_id} CPU usage critical: {metrics.cpu_percent}%")
                
                if metrics.memory_percent > self.config.alert_thresholds.get('memory_critical', 85.0):
                    self.logger.critical(f"Worker {worker_id} memory usage critical: {metrics.memory_percent}%")
                
                if metrics.error_rate > self.config.alert_thresholds.get('error_rate_critical', 0.1):
                    self.logger.critical(f"Worker {worker_id} error rate critical: {metrics.error_rate}")


class TaskManager:
    """Manage task lifecycle and recovery"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".TaskManager")
    
    def reassign_worker_tasks(self, failed_worker_id: str, active_tasks: Dict[str, TaskMetadata]):
        """Reassign tasks from failed worker"""
        tasks_to_reassign = []
        
        for task_id, task_metadata in active_tasks.items():
            if task_metadata.worker_id == failed_worker_id and task_metadata.state in [TaskState.RUNNING, TaskState.ASSIGNED]:
                tasks_to_reassign.append(task_id)
        
        for task_id in tasks_to_reassign:
            task_metadata = active_tasks[task_id]
            task_metadata.state = TaskState.PENDING
            task_metadata.worker_id = None
            self.logger.info(f"Reassigned task {task_id} from failed worker {failed_worker_id}")


class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.alerts: List[Alert] = []
        self.logger = logging.getLogger(__name__ + ".AlertManager")
        self._lock = threading.Lock()
    
    def add_alert(self, alert: Alert):
        """Add new alert"""
        with self._lock:
            self.alerts.append(alert)
            
            # Log alert
            if alert.level == AlertLevel.CRITICAL:
                self.logger.critical(f"CRITICAL ALERT: {alert.message}")
            elif alert.level == AlertLevel.ERROR:
                self.logger.error(f"ERROR ALERT: {alert.message}")
            elif alert.level == AlertLevel.WARNING:
                self.logger.warning(f"WARNING ALERT: {alert.message}")
            else:
                self.logger.info(f"INFO ALERT: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]
    
    def resolve_alert(self, alert_id: str):
        """Resolve alert"""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    self.logger.info(f"Resolved alert {alert_id}")
                    break


class CheckpointManager:
    """Manage checkpoints for fault tolerance"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_directory)
        self.logger = logging.getLogger(__name__ + ".CheckpointManager")
    
    def create_checkpoint(self, active_tasks: Dict[str, TaskMetadata]):
        """Create checkpoint of active tasks"""
        if not self.config.enable_checkpointing:
            return
        
        try:
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Serialize task metadata
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'tasks': {
                    task_id: {
                        'task_id': task.task_id,
                        'chunk_id': task.chunk_id,
                        'state': task.state.value,
                        'worker_id': task.worker_id,
                        'created_at': task.created_at.isoformat(),
                        'started_at': task.started_at.isoformat() if task.started_at else None,
                        'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                        'retry_count': task.retry_count,
                        'max_retries': task.max_retries,
                        'error_message': task.error_message,
                        'checkpoint_data': task.checkpoint_data
                    }
                    for task_id, task in active_tasks.items()
                }
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.info(f"Created checkpoint: {checkpoint_file}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            self.logger.error(f"Error creating checkpoint: {e}")
    
    def load_checkpoint(self, active_tasks: Dict[str, TaskMetadata]):
        """Load checkpoint to restore tasks"""
        if not self.config.enable_checkpointing:
            return
        
        try:
            # Find latest checkpoint
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoint_files:
                return
            
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Restore tasks
            for task_id, task_data in checkpoint_data['tasks'].items():
                if task_data['state'] in [TaskState.RUNNING.value, TaskState.ASSIGNED.value]:
                    # Restore incomplete tasks
                    task_metadata = TaskMetadata(
                        task_id=task_data['task_id'],
                        chunk_id=task_data['chunk_id'],
                        state=TaskState.PENDING,  # Reset to pending for retry
                        worker_id=None,
                        created_at=datetime.fromisoformat(task_data['created_at']),
                        started_at=None,
                        completed_at=None,
                        retry_count=task_data['retry_count'],
                        max_retries=task_data['max_retries'],
                        error_message=task_data['error_message'],
                        checkpoint_data=task_data['checkpoint_data']
                    )
                    active_tasks[task_id] = task_metadata
            
            self.logger.info(f"Loaded checkpoint: {latest_checkpoint}")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoint files"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            if len(checkpoint_files) > 5:  # Keep only 5 most recent
                sorted_files = sorted(checkpoint_files, key=lambda f: f.stat().st_mtime)
                for old_file in sorted_files[:-5]:
                    old_file.unlink()
                    self.logger.info(f"Removed old checkpoint: {old_file}")
        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoints: {e}")


# Add the missing methods to ParallelProcessor
def _add_fault_tolerance_methods():
    """Add fault tolerance methods to ParallelProcessor"""
    
    def _process_chunk_worker_ft(self, chunk: DataChunk, 
                               processing_func: Callable[[DataChunk], DataChunk],
                               task_metadata: TaskMetadata) -> Optional[DataChunk]:
        """Fault-tolerant worker function for processing chunks"""
        worker_id = f"worker_{threading.current_thread().ident or os.getpid()}"
        task_metadata.worker_id = worker_id
        task_metadata.state = TaskState.RUNNING
        task_metadata.started_at = datetime.now()
        
        try:
            # Update worker health metrics
            self._update_worker_health_metrics(worker_id)
            
            # Monitor memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.config.memory_limit_per_worker_mb:
                logger.warning(f"Worker {worker_id} memory usage high: {memory_mb:.2f}MB")
                
                # Generate alert for high memory usage
                alert = Alert(
                    id=str(uuid.uuid4()),
                    level=AlertLevel.WARNING,
                    message=f"Worker {worker_id} memory usage high: {memory_mb:.2f}MB",
                    component="worker_monitor",
                    worker_id=worker_id,
                    timestamp=datetime.now()
                )
                self.alert_manager.add_alert(alert)
                
                gc.collect()
            
            # Process chunk
            start_time = time.time()
            result = processing_func(chunk)
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats.update_processing_time(processing_time)
            
            # Update worker metrics
            if worker_id in self.worker_metrics:
                self.worker_metrics[worker_id].tasks_completed += 1
                self.worker_metrics[worker_id].avg_task_duration = (
                    (self.worker_metrics[worker_id].avg_task_duration + processing_time) / 2
                )
            
            return result
            
        except Exception as e:
            # Update worker metrics
            if worker_id in self.worker_metrics:
                self.worker_metrics[worker_id].tasks_failed += 1
                self.worker_metrics[worker_id].last_error = str(e)
                self.worker_metrics[worker_id].error_rate = (
                    self.worker_metrics[worker_id].tasks_failed / 
                    (self.worker_metrics[worker_id].tasks_completed + self.worker_metrics[worker_id].tasks_failed)
                )
            
            logger.error(f"Error in worker {worker_id} processing chunk {chunk.chunk_id}: {str(e)}")
            task_metadata.error_message = str(e)
            raise
    
    def _retry_task(self, task_metadata: TaskMetadata, 
                   processing_func: Callable[[DataChunk], DataChunk],
                   executor) -> Optional[DataChunk]:
        """Retry failed task with exponential backoff"""
        try:
            task_metadata.retry_count += 1
            task_metadata.state = TaskState.RETRYING
            
            # Calculate retry delay with exponential backoff
            delay = self.config.retry_delay_seconds * (self.config.retry_backoff_factor ** (task_metadata.retry_count - 1))
            
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, delay * 0.1)
            delay += jitter
            
            logger.info(f"Retrying task {task_metadata.task_id} (attempt {task_metadata.retry_count}) after {delay:.2f}s")
            time.sleep(delay)
            
            # Recreate chunk from metadata
            chunk = self._recreate_chunk_from_metadata(task_metadata)
            if chunk is None:
                return None
            
            # Resubmit task
            future = executor.submit(self._process_chunk_worker_ft, chunk, processing_func, task_metadata)
            
            try:
                result = future.result(timeout=self.config.timeout_seconds)
                if result is not None:
                    task_metadata.state = TaskState.COMPLETED
                    task_metadata.completed_at = datetime.now()
                    self.stats.chunks_processed += 1
                    
                    logger.info(f"Task {task_metadata.task_id} succeeded on retry {task_metadata.retry_count}")
                    return result
                else:
                    task_metadata.state = TaskState.FAILED
                    return None
                    
            except Exception as e:
                task_metadata.state = TaskState.FAILED
                task_metadata.error_message = str(e)
                logger.error(f"Task {task_metadata.task_id} failed on retry {task_metadata.retry_count}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrying task {task_metadata.task_id}: {str(e)}")
            task_metadata.state = TaskState.FAILED
            task_metadata.error_message = str(e)
            return None
    
    def _recreate_chunk_from_metadata(self, task_metadata: TaskMetadata) -> Optional[DataChunk]:
        """Recreate chunk from task metadata"""
        try:
            # This is a simplified implementation - in practice you'd need more sophisticated
            # chunk recreation logic based on the checkpoint data
            if task_metadata.checkpoint_data and 'chunk_data' in task_metadata.checkpoint_data:
                return task_metadata.checkpoint_data['chunk_data']
            else:
                # Fallback: create a minimal chunk
                return DataChunk(
                    chunk_id=task_metadata.chunk_id,
                    data=None,
                    metadata={}
                )
        except Exception as e:
            logger.error(f"Error recreating chunk from metadata: {str(e)}")
            return None
    
    def _update_worker_health_metrics(self, worker_id: str):
        """Update worker health metrics"""
        try:
            current_time = datetime.now()
            process = psutil.Process()
            
            # Get system metrics
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            # Get disk usage
            disk_usage = psutil.disk_usage('/')
            disk_percent = disk_usage.percent
            
            # Update or create worker metrics
            if worker_id not in self.worker_metrics:
                self.worker_metrics[worker_id] = WorkerHealthMetrics(
                    worker_id=worker_id,
                    state=WorkerState.HEALTHY,
                    last_heartbeat=current_time,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_mb=memory_mb,
                    disk_usage_percent=disk_percent,
                    task_queue_size=0,
                    tasks_completed=0,
                    tasks_failed=0,
                    avg_task_duration=0.0,
                    error_rate=0.0,
                    uptime=0.0
                )
            else:
                metrics = self.worker_metrics[worker_id]
                metrics.last_heartbeat = current_time
                metrics.cpu_percent = cpu_percent
                metrics.memory_percent = memory_percent
                metrics.memory_mb = memory_mb
                metrics.disk_usage_percent = disk_percent
                
                # Update state based on health
                if not metrics.is_healthy():
                    if metrics.state == WorkerState.HEALTHY:
                        metrics.state = WorkerState.UNHEALTHY
                elif metrics.is_degraded():
                    if metrics.state == WorkerState.HEALTHY:
                        metrics.state = WorkerState.DEGRADED
                else:
                    metrics.state = WorkerState.HEALTHY
            
        except Exception as e:
            logger.error(f"Error updating worker health metrics for {worker_id}: {str(e)}")
    
    # Add methods to ParallelProcessor class
    ParallelProcessor._process_chunk_worker_ft = _process_chunk_worker_ft
    ParallelProcessor._retry_task = _retry_task
    ParallelProcessor._recreate_chunk_from_metadata = _recreate_chunk_from_metadata
    ParallelProcessor._update_worker_health_metrics = _update_worker_health_metrics


# Dynamic Load Balancing Classes

class WorkStealingTask:
    """Task wrapper for work-stealing algorithm"""
    
    def __init__(self, task_id: str, func: Callable, args: tuple, kwargs: dict, 
                 metadata: Optional[TaskMetadata] = None):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.metadata = metadata or TaskMetadata(task_id=task_id)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.worker_id: Optional[str] = None
        self.result: Any = None
        self.exception: Optional[Exception] = None
        self.stolen_count: int = 0
    
    def __lt__(self, other):
        """Compare tasks for priority queue"""
        if self.metadata.priority != other.metadata.priority:
            return self.metadata.priority.value > other.metadata.priority.value
        
        # If deadlines exist, prioritize by deadline
        if hasattr(self.metadata, 'deadline') and hasattr(other.metadata, 'deadline'):
            if self.metadata.deadline and other.metadata.deadline:
                return self.metadata.deadline < other.metadata.deadline
            elif self.metadata.deadline:
                return True
            elif other.metadata.deadline:
                return False
        
        # Default to FIFO using submit_time
        return getattr(self.metadata, 'submit_time', 0) < getattr(other.metadata, 'submit_time', 0)
    
    def execute(self) -> Any:
        """Execute the task"""
        self.start_time = time.time()
        try:
            self.result = self.func(*self.args, **self.kwargs)
            return self.result
        except Exception as e:
            self.exception = e
            raise
        finally:
            self.end_time = time.time()
    
    def duration(self) -> Optional[float]:
        """Get task execution duration"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class WorkStealingQueue:
    """Thread-safe work-stealing queue implementation"""
    
    def __init__(self, worker_id: str, max_size: int = 1000):
        self.worker_id = worker_id
        self.max_size = max_size
        self._tasks = deque()
        self._priority_tasks = []
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self.total_tasks_added = 0
        self.total_tasks_stolen = 0
    
    def put(self, task: WorkStealingTask, block: bool = True, timeout: Optional[float] = None):
        """Add task to the queue"""
        with self._condition:
            # Wait if queue is full
            if block:
                while len(self._tasks) >= self.max_size:
                    if not self._condition.wait(timeout):
                        raise queue.Full()
            elif len(self._tasks) >= self.max_size:
                raise queue.Full()
            
            # Add to priority queue if high priority, otherwise to normal queue
            if task.metadata.priority.value >= TaskPriority.HIGH.value:
                heapq.heappush(self._priority_tasks, task)
            else:
                self._tasks.append(task)
            
            self.total_tasks_added += 1
            self._condition.notify()
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[WorkStealingTask]:
        """Get task from the queue (LIFO for better cache locality)"""
        with self._condition:
            while True:
                # First check priority tasks
                if self._priority_tasks:
                    task = heapq.heappop(self._priority_tasks)
                    return task
                
                # Then check normal tasks (LIFO)
                if self._tasks:
                    task = self._tasks.pop()
                    return task
                
                if not block:
                    return None
                
                if not self._condition.wait(timeout):
                    return None
    
    def steal(self, count: int = 1) -> List[WorkStealingTask]:
        """Steal tasks from the queue (FIFO for work stealing)"""
        with self._condition:
            stolen_tasks = []
            
            # Steal from normal queue first (FIFO)
            while len(stolen_tasks) < count and self._tasks:
                task = self._tasks.popleft()
                task.stolen_count += 1
                stolen_tasks.append(task)
            
            # Don't steal priority tasks unless necessary
            if not stolen_tasks and self._priority_tasks:
                task = heapq.heappop(self._priority_tasks)
                task.stolen_count += 1
                stolen_tasks.append(task)
            
            self.total_tasks_stolen += len(stolen_tasks)
            return stolen_tasks
    
    def size(self) -> int:
        """Get current queue size"""
        with self._lock:
            return len(self._tasks) + len(self._priority_tasks)
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            return len(self._tasks) == 0 and len(self._priority_tasks) == 0
    
    def utilization(self) -> float:
        """Get queue utilization ratio"""
        with self._lock:
            return (len(self._tasks) + len(self._priority_tasks)) / self.max_size

class WorkerPerformanceMetrics:
    """Performance metrics for individual workers"""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.tasks_stolen = 0
        self.tasks_given = 0
        self.total_execution_time = 0.0
        self.cpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.last_activity = time.time()
        self.start_time = time.time()
        self.performance_score = 1.0
        self._lock = threading.Lock()
    
    def update_task_completion(self, task: WorkStealingTask):
        """Update metrics after task completion"""
        with self._lock:
            if task.exception:
                self.tasks_failed += 1
            else:
                self.tasks_completed += 1
                if task.duration():
                    self.total_execution_time += task.duration()
            
            self.last_activity = time.time()
            self._update_performance_score()
    
    def update_system_metrics(self, cpu_percent: float, memory_percent: float):
        """Update system resource metrics"""
        with self._lock:
            self.cpu_usage_history.append(cpu_percent)
            self.memory_usage_history.append(memory_percent)
    
    def _update_performance_score(self):
        """Update performance score based on various metrics"""
        success_rate = self.tasks_completed / max(1, self.tasks_completed + self.tasks_failed)
        
        avg_cpu = statistics.mean(self.cpu_usage_history) if self.cpu_usage_history else 50.0
        avg_memory = statistics.mean(self.memory_usage_history) if self.memory_usage_history else 50.0
        
        # Performance score based on success rate and resource efficiency
        efficiency_score = 1.0 - (avg_cpu / 100.0 * 0.6 + avg_memory / 100.0 * 0.4)
        self.performance_score = success_rate * 0.7 + efficiency_score * 0.3
    
    def get_throughput(self) -> float:
        """Get tasks per second throughput"""
        with self._lock:
            elapsed = time.time() - self.start_time
            return self.tasks_completed / max(elapsed, 1.0)
    
    def get_average_execution_time(self) -> float:
        """Get average task execution time"""
        with self._lock:
            return self.total_execution_time / max(1, self.tasks_completed)
    
    def is_idle(self, idle_threshold: float = 30.0) -> bool:
        """Check if worker has been idle for too long"""
        return time.time() - self.last_activity > idle_threshold

class LoadBalancingScheduler:
    """Advanced scheduler with multiple load balancing strategies"""
    
    def __init__(self, config: LoadBalancingConfig):
        self.config = config
        self.worker_queues: Dict[str, WorkStealingQueue] = {}
        self.worker_metrics: Dict[str, WorkerPerformanceMetrics] = {}
        self.task_history: Dict[str, WorkStealingTask] = {}
        self.resource_monitor = ResourceMonitor()
        self.scheduling_history = deque(maxlen=config.performance_history_size)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Predictive scheduling data
        self.task_patterns: Dict[str, List[float]] = defaultdict(list)
        self.worker_performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Start monitoring thread
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start resource monitoring thread"""
        if self._monitoring_thread is None:
            self._monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self._monitoring_thread.start()
    
    def _monitor_resources(self):
        """Monitor system resources and worker performance"""
        while not self._stop_event.wait(self.config.resource_monitoring_interval):
            try:
                # Update system metrics
                self.resource_monitor.update_metrics()
                
                # Update worker metrics
                for worker_id, metrics in self.worker_metrics.items():
                    try:
                        # Get worker process metrics if available
                        cpu_percent = self.resource_monitor.get_worker_cpu_usage(worker_id)
                        memory_percent = self.resource_monitor.get_worker_memory_usage(worker_id)
                        metrics.update_system_metrics(cpu_percent, memory_percent)
                    except Exception as e:
                        logger.warning(f"Error updating metrics for worker {worker_id}: {e}")
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
    
    def register_worker(self, worker_id: str, queue_size: int = 100):
        """Register a new worker with the scheduler"""
        with self._lock:
            if worker_id not in self.worker_queues:
                self.worker_queues[worker_id] = WorkStealingQueue(worker_id, queue_size)
                self.worker_metrics[worker_id] = WorkerPerformanceMetrics(worker_id)
                logger.info(f"Registered worker {worker_id}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker from the scheduler"""
        with self._lock:
            if worker_id in self.worker_queues:
                # Redistribute remaining tasks
                remaining_tasks = []
                while not self.worker_queues[worker_id].is_empty():
                    task = self.worker_queues[worker_id].get(block=False)
                    if task:
                        remaining_tasks.append(task)
                
                # Redistribute tasks to other workers
                for task in remaining_tasks:
                    self.schedule_task(task)
                
                del self.worker_queues[worker_id]
                del self.worker_metrics[worker_id]
                logger.info(f"Unregistered worker {worker_id}")
    
    def schedule_task(self, task: WorkStealingTask) -> bool:
        """Schedule a task using the configured scheduling policy"""
        with self._lock:
            if not self.worker_queues:
                return False
            
            # Choose scheduling strategy based on configuration
            if self.config.enable_resource_awareness:
                return self._schedule_resource_aware(task)
            elif self.config.enable_predictive_scheduling:
                return self._schedule_predictive(task)
            else:
                return self._schedule_round_robin(task)
    
    def _schedule_resource_aware(self, task: WorkStealingTask) -> bool:
        """Schedule task with resource awareness"""
        # Find worker with best resource availability
        best_worker = None
        best_score = float('-inf')
        
        for worker_id, queue in self.worker_queues.items():
            metrics = self.worker_metrics[worker_id]
            
            # Calculate resource availability score
            cpu_availability = 1.0 - (statistics.mean(metrics.cpu_usage_history) / 100.0 
                                     if metrics.cpu_usage_history else 0.5)
            memory_availability = 1.0 - (statistics.mean(metrics.memory_usage_history) / 100.0 
                                        if metrics.memory_usage_history else 0.5)
            
            queue_utilization = queue.utilization()
            utilization_score = 1.0 - queue_utilization
            
            # Combined score with weights
            combined_score = (cpu_availability * 0.4 + 
                            memory_availability * 0.3 + 
                            utilization_score * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_worker = worker_id
        
        if best_worker:
            try:
                self.worker_queues[best_worker].put(task, block=False)
                return True
            except queue.Full:
                return self._schedule_round_robin(task)
        
        return False
    
    def _schedule_predictive(self, task: WorkStealingTask) -> bool:
        """Schedule task using predictive analysis"""
        # Predict task execution time based on historical data
        predicted_duration = self._predict_task_duration(task)
        
        # Find worker with best predicted completion time
        best_worker = None
        best_completion_time = float('inf')
        
        for worker_id, queue in self.worker_queues.items():
            metrics = self.worker_metrics[worker_id]
            
            # Predict completion time
            current_queue_time = queue.size() * metrics.get_average_execution_time()
            predicted_completion = current_queue_time + predicted_duration
            
            # Factor in worker performance
            adjusted_completion = predicted_completion / metrics.performance_score
            
            if adjusted_completion < best_completion_time:
                best_completion_time = adjusted_completion
                best_worker = worker_id
        
        if best_worker:
            try:
                self.worker_queues[best_worker].put(task, block=False)
                return True
            except queue.Full:
                return self._schedule_round_robin(task)
        
        return False
    
    def _schedule_round_robin(self, task: WorkStealingTask) -> bool:
        """Simple round-robin scheduling"""
        # Find worker with smallest queue
        best_worker = None
        min_queue_size = float('inf')
        
        for worker_id, queue in self.worker_queues.items():
            if queue.size() < min_queue_size:
                min_queue_size = queue.size()
                best_worker = worker_id
        
        if best_worker:
            try:
                self.worker_queues[best_worker].put(task, block=False)
                return True
            except queue.Full:
                pass
        
        return False
    
    def _predict_task_duration(self, task: WorkStealingTask) -> float:
        """Predict task execution duration based on historical data"""
        # Use estimated duration if available
        if hasattr(task.metadata, 'estimated_duration') and task.metadata.estimated_duration:
            return task.metadata.estimated_duration
        
        # Use historical data for similar tasks
        task_type = task.func.__name__ if hasattr(task.func, '__name__') else str(type(task.func))
        if task_type in self.task_patterns and self.task_patterns[task_type]:
            return statistics.mean(self.task_patterns[task_type])
        
        # Default estimation
        return 1.0
    
    def attempt_work_stealing(self, requesting_worker: str) -> Optional[WorkStealingTask]:
        """Attempt to steal work from other workers"""
        if not self.config.enable_work_stealing:
            return None
        
        with self._lock:
            # Find workers with high queue utilization
            candidates = []
            for worker_id, queue in self.worker_queues.items():
                if (worker_id != requesting_worker and 
                    queue.utilization() > self.config.steal_threshold):
                    candidates.append((worker_id, queue))
            
            # Sort by queue utilization (highest first)
            candidates.sort(key=lambda x: x[1].utilization(), reverse=True)
            
            # Attempt to steal from the most loaded worker
            for worker_id, queue in candidates:
                stolen_tasks = queue.steal(self.config.steal_batch_size)
                if stolen_tasks:
                    logger.debug(f"Worker {requesting_worker} stole {len(stolen_tasks)} tasks from {worker_id}")
                    
                    # Update metrics
                    self.worker_metrics[worker_id].tasks_given += len(stolen_tasks)
                    self.worker_metrics[requesting_worker].tasks_stolen += len(stolen_tasks)
                    
                    return stolen_tasks[0] if stolen_tasks else None
        
        return None
    
    def get_task_for_worker(self, worker_id: str) -> Optional[WorkStealingTask]:
        """Get next task for a specific worker"""
        with self._lock:
            if worker_id not in self.worker_queues:
                return None
            
            queue = self.worker_queues[worker_id]
            
            # First try to get from own queue
            task = queue.get(block=False)
            if task:
                return task
            
            # If no task available, attempt work stealing
            if self.config.enable_work_stealing:
                stolen_task = self.attempt_work_stealing(worker_id)
                if stolen_task:
                    return stolen_task
        
        return None
    
    def task_completed(self, task: WorkStealingTask, worker_id: str):
        """Notify scheduler of task completion"""
        with self._lock:
            # Update worker metrics
            if worker_id in self.worker_metrics:
                self.worker_metrics[worker_id].update_task_completion(task)
            
            # Update task history for predictive scheduling
            if task.duration():
                task_type = task.func.__name__ if hasattr(task.func, '__name__') else str(type(task.func))
                self.task_patterns[task_type].append(task.duration())
                
                # Keep only recent history
                if len(self.task_patterns[task_type]) > self.config.performance_history_size:
                    self.task_patterns[task_type] = self.task_patterns[task_type][-self.config.performance_history_size:]
            
            # Store task in history
            self.task_history[task.task_id] = task
            
            # Clean up old history
            if len(self.task_history) > self.config.performance_history_size:
                oldest_tasks = sorted(self.task_history.items(), key=lambda x: getattr(x[1].metadata, 'submit_time', 0))
                for task_id, _ in oldest_tasks[:len(self.task_history) - self.config.performance_history_size]:
                    del self.task_history[task_id]
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics"""
        with self._lock:
            stats = {
                'total_workers': len(self.worker_queues),
                'total_tasks_in_queues': sum(q.size() for q in self.worker_queues.values()),
                'worker_stats': {},
                'system_stats': self.resource_monitor.get_system_stats(),
                'scheduling_config': {
                    'work_stealing_enabled': self.config.enable_work_stealing,
                    'adaptive_scaling_enabled': self.config.enable_adaptive_scaling,
                    'predictive_scheduling_enabled': self.config.enable_predictive_scheduling,
                    'resource_awareness_enabled': self.config.enable_resource_awareness
                }
            }
            
            # Worker-specific stats
            for worker_id, queue in self.worker_queues.items():
                metrics = self.worker_metrics[worker_id]
                stats['worker_stats'][worker_id] = {
                    'queue_size': queue.size(),
                    'queue_utilization': queue.utilization(),
                    'tasks_completed': metrics.tasks_completed,
                    'tasks_failed': metrics.tasks_failed,
                    'tasks_stolen': metrics.tasks_stolen,
                    'tasks_given': metrics.tasks_given,
                    'throughput': metrics.get_throughput(),
                    'performance_score': metrics.performance_score,
                    'avg_execution_time': metrics.get_average_execution_time(),
                    'is_idle': metrics.is_idle()
                }
            
            return stats
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Load balancing scheduler shutdown")

class ResourceMonitor:
    """System resource monitoring for load balancing"""
    
    def __init__(self):
        self.system_cpu_history = deque(maxlen=100)
        self.system_memory_history = deque(maxlen=100)
        self.worker_processes: Dict[str, psutil.Process] = {}
        self._lock = threading.Lock()
    
    def register_worker_process(self, worker_id: str, process: psutil.Process):
        """Register a worker process for monitoring"""
        with self._lock:
            self.worker_processes[worker_id] = process
    
    def unregister_worker_process(self, worker_id: str):
        """Unregister a worker process"""
        with self._lock:
            if worker_id in self.worker_processes:
                del self.worker_processes[worker_id]
    
    def update_metrics(self):
        """Update system metrics"""
        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            with self._lock:
                self.system_cpu_history.append(cpu_percent)
                self.system_memory_history.append(memory_percent)
                
        except Exception as e:
            logger.warning(f"Error updating system metrics: {e}")
    
    def get_worker_cpu_usage(self, worker_id: str) -> float:
        """Get CPU usage for a specific worker"""
        with self._lock:
            if worker_id in self.worker_processes:
                try:
                    return self.worker_processes[worker_id].cpu_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return 0.0
            return 0.0
    
    def get_worker_memory_usage(self, worker_id: str) -> float:
        """Get memory usage for a specific worker"""
        with self._lock:
            if worker_id in self.worker_processes:
                try:
                    memory_info = self.worker_processes[worker_id].memory_info()
                    return (memory_info.rss / psutil.virtual_memory().total) * 100.0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return 0.0
            return 0.0
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        with self._lock:
            return {
                'cpu_percent': statistics.mean(self.system_cpu_history) if self.system_cpu_history else 0.0,
                'memory_percent': statistics.mean(self.system_memory_history) if self.system_memory_history else 0.0,
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'cpu_count': psutil.cpu_count(),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0),
                'active_workers': len(self.worker_processes)
            }

class AdaptiveWorkerManager:
    """Manages dynamic worker scaling based on workload and system resources"""
    
    def __init__(self, config: ParallelConfig, scheduler: LoadBalancingScheduler):
        self.config = config
        self.scheduler = scheduler
        self.active_workers: Dict[str, WorkerProcess] = {}
        self.worker_pool: Optional[ProcessPoolExecutor] = None
        self.scaling_history = deque(maxlen=100)
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._scaling_thread: Optional[threading.Thread] = None
        
        # NUMA topology detection
        self.numa_topology = self._detect_numa_topology()
        self.numa_node_workers: Dict[int, List[str]] = defaultdict(list)
        
        # Start scaling monitoring
        self._start_scaling_monitor()
    
    def _detect_numa_topology(self) -> Dict[int, List[int]]:
        """Detect NUMA topology if available"""
        numa_topology = {}
        try:
            if hasattr(os, 'sched_getaffinity'):
                # Try to detect NUMA nodes
                for node in range(8):  # Check up to 8 NUMA nodes
                    try:
                        node_cpus = list(range(node * psutil.cpu_count() // 8, 
                                             (node + 1) * psutil.cpu_count() // 8))
                        if node_cpus:
                            numa_topology[node] = node_cpus
                    except:
                        break
        except:
            pass
        
        return numa_topology
    
    def _start_scaling_monitor(self):
        """Start the adaptive scaling monitoring thread"""
        if self._scaling_thread is None:
            self._scaling_thread = threading.Thread(target=self._monitor_scaling, daemon=True)
            self._scaling_thread.start()
    
    def _monitor_scaling(self):
        """Monitor system metrics and adjust worker count"""
        while not self._stop_event.wait(5.0):  # Check every 5 seconds
            try:
                self._evaluate_scaling_needs()
            except Exception as e:
                logger.error(f"Error in scaling monitor: {e}")
    
    def _evaluate_scaling_needs(self):
        """Evaluate if we need to scale workers up or down"""
        if not self.config.load_balancing.enable_adaptive_scaling:
            return
        
        current_time = time.time()
        system_stats = self.scheduler.resource_monitor.get_system_stats()
        load_stats = self.scheduler.get_load_balancing_stats()
        
        # Calculate metrics
        avg_cpu_usage = system_stats['cpu_percent']
        avg_memory_usage = system_stats['memory_percent']
        total_queue_size = load_stats['total_tasks_in_queues']
        active_workers = len(self.active_workers)
        
        # Calculate queue pressure
        queue_pressure = total_queue_size / max(1, active_workers)
        
        # Check if we need to scale up
        should_scale_up = (
            avg_cpu_usage > self.config.load_balancing.scale_up_threshold * 100 or
            queue_pressure > 10  # More than 10 tasks per worker
        )
        
        # Check if we need to scale down
        should_scale_down = (
            avg_cpu_usage < self.config.load_balancing.scale_down_threshold * 100 and
            queue_pressure < 2 and  # Less than 2 tasks per worker
            active_workers > self.config.load_balancing.min_workers
        )
        
        # Apply cooldown periods
        if should_scale_up and (current_time - self.last_scale_up) > self.config.load_balancing.scale_up_cooldown:
            self._scale_up()
        elif should_scale_down and (current_time - self.last_scale_down) > self.config.load_balancing.scale_down_cooldown:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up the number of workers"""
        with self._lock:
            current_workers = len(self.active_workers)
            if current_workers >= self.config.load_balancing.max_workers:
                return
            
            # Determine optimal number of workers to add
            target_workers = min(
                current_workers + max(1, current_workers // 4),  # Add 25% more workers
                self.config.load_balancing.max_workers
            )
            
            workers_to_add = target_workers - current_workers
            
            # Add workers with NUMA awareness
            for i in range(workers_to_add):
                worker_id = f"worker_{uuid.uuid4().hex[:8]}"
                
                # Choose NUMA node with least workers
                numa_node = self._choose_optimal_numa_node()
                
                if self._start_worker(worker_id, numa_node):
                    self.last_scale_up = time.time()
                    logger.info(f"Scaled up: added worker {worker_id} (total: {len(self.active_workers)})")
                    
                    # Record scaling event
                    self.scaling_history.append({
                        'action': 'scale_up',
                        'timestamp': time.time(),
                        'worker_count': len(self.active_workers),
                        'worker_id': worker_id
                    })
    
    def _scale_down(self):
        """Scale down the number of workers"""
        with self._lock:
            current_workers = len(self.active_workers)
            if current_workers <= self.config.load_balancing.min_workers:
                return
            
            # Determine number of workers to remove
            target_workers = max(
                current_workers - max(1, current_workers // 8),  # Remove 12.5% of workers
                self.config.load_balancing.min_workers
            )
            
            workers_to_remove = current_workers - target_workers
            
            # Choose workers to remove (prefer idle workers)
            workers_to_terminate = self._choose_workers_to_terminate(workers_to_remove)
            
            for worker_id in workers_to_terminate:
                if self._stop_worker(worker_id):
                    self.last_scale_down = time.time()
                    logger.info(f"Scaled down: removed worker {worker_id} (total: {len(self.active_workers)})")
                    
                    # Record scaling event
                    self.scaling_history.append({
                        'action': 'scale_down',
                        'timestamp': time.time(),
                        'worker_count': len(self.active_workers),
                        'worker_id': worker_id
                    })
    
    def _choose_optimal_numa_node(self) -> int:
        """Choose the optimal NUMA node for a new worker"""
        if not self.numa_topology or not self.config.load_balancing.enable_numa_awareness:
            return 0
        
        # Find NUMA node with least workers
        node_worker_counts = {}
        for node in self.numa_topology:
            node_worker_counts[node] = len(self.numa_node_workers[node])
        
        return min(node_worker_counts, key=node_worker_counts.get)
    
    def _choose_workers_to_terminate(self, count: int) -> List[str]:
        """Choose workers to terminate, preferring idle workers"""
        worker_scores = []
        
        for worker_id in self.active_workers:
            # Get worker metrics
            if worker_id in self.scheduler.worker_metrics:
                metrics = self.scheduler.worker_metrics[worker_id]
                
                # Score based on idleness and performance
                idle_score = 1.0 if metrics.is_idle() else 0.0
                performance_score = 1.0 - metrics.performance_score
                queue_size = self.scheduler.worker_queues[worker_id].size()
                
                # Combined score (higher = better candidate for termination)
                score = idle_score * 0.5 + performance_score * 0.3 + (1.0 / (queue_size + 1)) * 0.2
                worker_scores.append((worker_id, score))
        
        # Sort by score (highest first) and return top candidates
        worker_scores.sort(key=lambda x: x[1], reverse=True)
        return [worker_id for worker_id, _ in worker_scores[:count]]
    
    def _start_worker(self, worker_id: str, numa_node: int = 0) -> bool:
        """Start a new worker process"""
        try:
            # Create worker process
            worker_process = WorkerProcess(
                worker_id=worker_id,
                scheduler=self.scheduler,
                config=self.config,
                numa_node=numa_node
            )
            
            # Start the worker
            if worker_process.start():
                self.active_workers[worker_id] = worker_process
                self.numa_node_workers[numa_node].append(worker_id)
                self.scheduler.register_worker(worker_id)
                return True
            
        except Exception as e:
            logger.error(f"Failed to start worker {worker_id}: {e}")
        
        return False
    
    def _stop_worker(self, worker_id: str) -> bool:
        """Stop a worker process"""
        try:
            if worker_id in self.active_workers:
                worker_process = self.active_workers[worker_id]
                
                # Stop the worker
                worker_process.stop()
                
                # Remove from tracking
                del self.active_workers[worker_id]
                
                # Remove from NUMA tracking
                for node_workers in self.numa_node_workers.values():
                    if worker_id in node_workers:
                        node_workers.remove(worker_id)
                
                # Unregister from scheduler
                self.scheduler.unregister_worker(worker_id)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to stop worker {worker_id}: {e}")
        
        return False
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics"""
        with self._lock:
            return {
                'active_workers': len(self.active_workers),
                'min_workers': self.config.load_balancing.min_workers,
                'max_workers': self.config.load_balancing.max_workers,
                'numa_topology': self.numa_topology,
                'numa_node_workers': dict(self.numa_node_workers),
                'scaling_history': list(self.scaling_history)[-10:],  # Last 10 events
                'last_scale_up': self.last_scale_up,
                'last_scale_down': self.last_scale_down
            }
    
    def shutdown(self):
        """Shutdown the worker manager"""
        self._stop_event.set()
        
        # Stop all workers
        workers_to_stop = list(self.active_workers.keys())
        for worker_id in workers_to_stop:
            self._stop_worker(worker_id)
        
        # Wait for scaling thread to finish
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5.0)
        
        logger.info("Adaptive worker manager shutdown")

class WorkerProcess:
    """Individual worker process for task execution"""
    
    def __init__(self, worker_id: str, scheduler: LoadBalancingScheduler, 
                 config: ParallelConfig, numa_node: int = 0):
        self.worker_id = worker_id
        self.scheduler = scheduler
        self.config = config
        self.numa_node = numa_node
        self.process: Optional[Process] = None
        self.is_running = False
        self._stop_event = mp.Event()
    
    def start(self) -> bool:
        """Start the worker process"""
        try:
            # Create and start process
            self.process = Process(
                target=self._worker_main,
                args=(self.worker_id, self._stop_event),
                name=f"Worker-{self.worker_id}"
            )
            
            self.process.start()
            self.is_running = True
            
            # Register with resource monitor
            try:
                psutil_process = psutil.Process(self.process.pid)
                self.scheduler.resource_monitor.register_worker_process(self.worker_id, psutil_process)
            except Exception as e:
                logger.warning(f"Failed to register worker process for monitoring: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start worker process {self.worker_id}: {e}")
            return False
    
    def stop(self):
        """Stop the worker process"""
        if self.is_running:
            self._stop_event.set()
            
            if self.process and self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5.0)
                
                if self.process.is_alive():
                    logger.warning(f"Force killing worker {self.worker_id}")
                    self.process.kill()
                    self.process.join()
            
            self.is_running = False
            
            # Unregister from resource monitor
            self.scheduler.resource_monitor.unregister_worker_process(self.worker_id)
    
    def _worker_main(self, worker_id: str, stop_event: mp.Event):
        """Main worker process function"""
        logger.info(f"Worker {worker_id} started")
        
        # Set process priority and affinity
        try:
            process = psutil.Process()
            if self.config.cpu_affinity and self.numa_node in self.scheduler.resource_monitor.worker_processes:
                cpu_list = self.scheduler.resource_monitor.worker_processes.get(self.numa_node, [])
                if cpu_list:
                    process.cpu_affinity(cpu_list)
        except Exception as e:
            logger.warning(f"Failed to set worker affinity: {e}")
        
        # Worker main loop
        while not stop_event.is_set():
            try:
                # Get next task
                task = self.scheduler.get_task_for_worker(worker_id)
                
                if task is None:
                    # No task available, sleep briefly
                    time.sleep(0.1)
                    continue
                
                # Execute task
                try:
                    task.worker_id = worker_id
                    result = task.execute()
                    
                    # Notify scheduler of completion
                    self.scheduler.task_completed(task, worker_id)
                    
                    logger.debug(f"Worker {worker_id} completed task {task.task_id}")
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed to execute task {task.task_id}: {e}")
                    task.exception = e
                    self.scheduler.task_completed(task, worker_id)
                
            except Exception as e:
                logger.error(f"Error in worker {worker_id} main loop: {e}")
                time.sleep(1.0)  # Prevent tight error loop
        
        logger.info(f"Worker {worker_id} stopped")

# Apply the fault tolerance methods
_add_fault_tolerance_methods()