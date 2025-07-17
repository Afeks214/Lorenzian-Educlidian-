#!/usr/bin/env python3
"""
Automated Standby Warmup System
AGENT 2: Trading Engine RTO Specialist

Advanced standby warmup system designed to maintain hot standby instances
in a ready state to achieve <5s RTO. Implements aggressive preloading,
background warming, and intelligent readiness verification.

Key Features:
- Continuous model warming in background
- Preemptive state synchronization
- Connection pool prewarming
- JIT compilation optimization
- Memory optimization for faster startup
- Health verification and readiness checks
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import torch
import numpy as np
from pathlib import Path
import redis.asyncio as redis

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.events import EventBus, Event, EventType
from src.trading.model_preloader import JITModelPreloader, ModelConfig
from src.trading.state_sync import RedisStateSynchronizer, InstanceRole, StateType
from src.utils.logger import get_logger

logger = get_logger(__name__)

class WarmupState(Enum):
    """Standby warmup states"""
    INITIALIZING = "initializing"
    WARMING_UP = "warming_up"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

class WarmupTask(Enum):
    """Types of warmup tasks"""
    MODEL_LOADING = "model_loading"
    MODEL_WARMING = "model_warming"
    CONNECTION_POOLS = "connection_pools"
    STATE_SYNC = "state_sync"
    MEMORY_OPTIMIZATION = "memory_optimization"
    JIT_COMPILATION = "jit_compilation"
    HEALTH_CHECK = "health_check"

@dataclass
class WarmupConfig:
    """Configuration for standby warmup"""
    # Model configuration
    models_dir: str = "/app/models/jit_optimized"
    warmup_iterations: int = 50  # Increased for thorough warming
    parallel_model_loading: bool = True
    max_model_workers: int = 8
    
    # Background warming
    background_warming_interval: float = 30.0  # 30 seconds
    continuous_warming: bool = True
    warmup_intensity: float = 1.0  # 1.0 = normal, 2.0 = aggressive
    
    # State synchronization
    state_sync_interval: float = 0.1  # 100ms
    preemptive_sync: bool = True
    sync_buffer_size: int = 1000
    
    # Connection pools
    redis_pool_size: int = 20
    http_pool_size: int = 50
    preconnect_pools: bool = True
    
    # Memory optimization
    memory_optimization: bool = True
    gc_interval: float = 60.0  # 60 seconds
    memory_threshold: float = 0.8  # 80%
    
    # Performance targets
    startup_time_target: float = 2.0  # 2 seconds
    model_load_target: float = 1.0  # 1 second
    warmup_target: float = 5.0  # 5 seconds
    
    # Health verification
    health_check_interval: float = 5.0  # 5 seconds
    readiness_verification: bool = True
    deep_health_checks: bool = True
    
    # Instance configuration
    instance_id: str = "standby_warmup"
    priority: int = 1  # 1=critical, 2=important, 3=normal

@dataclass
class WarmupMetrics:
    """Metrics for warmup performance"""
    total_warmups: int = 0
    successful_warmups: int = 0
    failed_warmups: int = 0
    average_warmup_time: float = 0.0
    last_warmup_time: float = 0.0
    models_loaded: int = 0
    models_warmed: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    readiness_score: float = 0.0
    
    def update_warmup_time(self, warmup_time: float):
        """Update warmup time metrics"""
        self.total_warmups += 1
        self.average_warmup_time = (self.average_warmup_time * (self.total_warmups - 1) + warmup_time) / self.total_warmups
        self.last_warmup_time = time.time()

@dataclass
class WarmupTask:
    """Individual warmup task"""
    task_id: str
    task_type: WarmupTask
    priority: int
    estimated_time: float
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    start_time: float = 0.0
    end_time: float = 0.0
    error: Optional[str] = None

class StandbyWarmupSystem:
    """
    Advanced standby warmup system for trading engine
    
    Features:
    - Continuous background warming of models
    - Preemptive state synchronization
    - Connection pool prewarming
    - Memory optimization
    - Intelligent readiness verification
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: WarmupConfig):
        self.config = config
        self.event_bus = EventBus()
        
        # State management
        self.state = WarmupState.INITIALIZING
        self.state_change_time = time.time()
        self.readiness_score = 0.0
        
        # Components
        self.model_preloader: Optional[JITModelPreloader] = None
        self.state_sync: Optional[RedisStateSynchronizer] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Task management
        self.warmup_tasks: Dict[str, WarmupTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.completed_tasks: List[str] = []
        
        # Performance tracking
        self.metrics = WarmupMetrics()
        self.performance_history: List[Tuple[float, float]] = []  # (timestamp, metric)
        
        # Background tasks
        self.warmup_task: Optional[asyncio.Task] = None
        self.background_warming_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        self.gc_task: Optional[asyncio.Task] = None
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_model_workers)
        
        # Model registry
        self.loaded_models: Dict[str, torch.jit.ScriptModule] = {}
        self.model_warmup_status: Dict[str, bool] = {}
        
        # Connection pools
        self.connection_pools: Dict[str, Any] = {}
        
        # Synchronization
        self.warmup_lock = asyncio.Lock()
        self.model_lock = asyncio.Lock()
        
        logger.info(f"Standby warmup system initialized for {config.instance_id}")
    
    async def initialize(self, redis_url: str):
        """Initialize standby warmup system"""
        try:
            logger.info("Initializing standby warmup system...")
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            
            # Initialize model preloader
            preloader_config = {
                'max_workers': self.config.max_model_workers,
                'use_memory_mapping': True,
                'enable_jit_optimization': True,
                'parallel_loading': True,
                'warmup_enabled': True,
                'warmup_iterations': self.config.warmup_iterations
            }
            self.model_preloader = JITModelPreloader(preloader_config)
            
            # Initialize state synchronizer
            self.state_sync = RedisStateSynchronizer(
                redis_url=redis_url,
                instance_id=self.config.instance_id,
                role=InstanceRole.PASSIVE,
                sync_interval=self.config.state_sync_interval
            )
            await self.state_sync.initialize()
            
            # Initialize connection pools
            await self._initialize_connection_pools()
            
            # Create warmup tasks
            await self._create_warmup_tasks()
            
            # Start background tasks
            self.warmup_task = asyncio.create_task(self._warmup_loop())
            self.background_warming_task = asyncio.create_task(self._background_warming_loop())
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.metrics_task = asyncio.create_task(self._metrics_loop())
            
            if self.config.memory_optimization:
                self.gc_task = asyncio.create_task(self._gc_loop())
            
            # Register event handlers
            await self._register_event_handlers()
            
            logger.info("Standby warmup system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize standby warmup system: {e}")
            raise
    
    async def _initialize_connection_pools(self):
        """Initialize connection pools"""
        try:
            logger.info("Initializing connection pools...")
            
            # Redis connection pool
            if self.config.preconnect_pools:
                redis_pool = redis.ConnectionPool.from_url(
                    self.redis_client.connection_pool.connection_kwargs['url'],
                    max_connections=self.config.redis_pool_size
                )
                self.connection_pools['redis'] = redis_pool
            
            # HTTP connection pool (placeholder)
            # In a real implementation, this would initialize HTTP session pools
            self.connection_pools['http'] = {
                'pool_size': self.config.http_pool_size,
                'initialized': True
            }
            
            logger.info("Connection pools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")
            raise
    
    async def _create_warmup_tasks(self):
        """Create warmup tasks"""
        try:
            # Model loading tasks
            models_to_load = [
                'position_sizing_agent',
                'stop_target_agent',
                'risk_monitor_agent',
                'portfolio_optimizer_agent',
                'routing_agent',
                'centralized_critic'
            ]
            
            for model_name in models_to_load:
                # Model loading task
                load_task = WarmupTask(
                    task_id=f"load_{model_name}",
                    task_type=WarmupTask.MODEL_LOADING,
                    priority=1,
                    estimated_time=0.5,
                    max_retries=3
                )
                self.warmup_tasks[load_task.task_id] = load_task
                
                # Model warming task
                warm_task = WarmupTask(
                    task_id=f"warm_{model_name}",
                    task_type=WarmupTask.MODEL_WARMING,
                    priority=2,
                    estimated_time=1.0,
                    dependencies=[load_task.task_id],
                    max_retries=2
                )
                self.warmup_tasks[warm_task.task_id] = warm_task
            
            # Connection pool warming task
            pool_task = WarmupTask(
                task_id="warm_connection_pools",
                task_type=WarmupTask.CONNECTION_POOLS,
                priority=3,
                estimated_time=0.2,
                max_retries=2
            )
            self.warmup_tasks[pool_task.task_id] = pool_task
            
            # State sync task
            sync_task = WarmupTask(
                task_id="initialize_state_sync",
                task_type=WarmupTask.STATE_SYNC,
                priority=2,
                estimated_time=0.3,
                max_retries=3
            )
            self.warmup_tasks[sync_task.task_id] = sync_task
            
            # Memory optimization task
            if self.config.memory_optimization:
                memory_task = WarmupTask(
                    task_id="optimize_memory",
                    task_type=WarmupTask.MEMORY_OPTIMIZATION,
                    priority=4,
                    estimated_time=0.1,
                    max_retries=1
                )
                self.warmup_tasks[memory_task.task_id] = memory_task
            
            # JIT compilation task
            jit_task = WarmupTask(
                task_id="jit_compilation",
                task_type=WarmupTask.JIT_COMPILATION,
                priority=3,
                estimated_time=0.5,
                dependencies=[f"load_{model}" for model in models_to_load],
                max_retries=2
            )
            self.warmup_tasks[jit_task.task_id] = jit_task
            
            # Health check task
            health_task = WarmupTask(
                task_id="health_verification",
                task_type=WarmupTask.HEALTH_CHECK,
                priority=5,
                estimated_time=0.2,
                dependencies=list(self.warmup_tasks.keys()),
                max_retries=1
            )
            self.warmup_tasks[health_task.task_id] = health_task
            
            logger.info(f"Created {len(self.warmup_tasks)} warmup tasks")
            
        except Exception as e:
            logger.error(f"Failed to create warmup tasks: {e}")
            raise
    
    async def _warmup_loop(self):
        """Main warmup loop"""
        try:
            self.state = WarmupState.WARMING_UP
            warmup_start_time = time.time()
            
            # Execute warmup tasks in priority order
            await self._execute_warmup_tasks()
            
            # Verify readiness
            if await self._verify_readiness():
                self.state = WarmupState.READY
                self.readiness_score = 100.0
                
                warmup_time = time.time() - warmup_start_time
                self.metrics.update_warmup_time(warmup_time)
                self.metrics.successful_warmups += 1
                
                logger.info(f"Standby warmup completed successfully in {warmup_time:.2f}s")
                
                # Send ready event
                await self.event_bus.emit(Event(
                    type=EventType.SYSTEM_READY,
                    data={
                        'component': 'standby_warmup',
                        'instance_id': self.config.instance_id,
                        'warmup_time': warmup_time,
                        'readiness_score': self.readiness_score
                    }
                ))
                
            else:
                self.state = WarmupState.DEGRADED
                self.metrics.failed_warmups += 1
                logger.warning("Standby warmup completed with degraded readiness")
                
        except Exception as e:
            self.state = WarmupState.FAILED
            self.metrics.failed_warmups += 1
            logger.error(f"Standby warmup failed: {e}")
    
    async def _execute_warmup_tasks(self):
        """Execute warmup tasks in priority order"""
        try:
            # Sort tasks by priority
            sorted_tasks = sorted(
                self.warmup_tasks.values(),
                key=lambda t: t.priority
            )
            
            # Execute tasks
            for task in sorted_tasks:
                if await self._can_execute_task(task):
                    await self._execute_task(task)
                    
        except Exception as e:
            logger.error(f"Error executing warmup tasks: {e}")
            raise
    
    async def _can_execute_task(self, task: WarmupTask) -> bool:
        """Check if task can be executed"""
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        # Check retry count
        if task.retry_count >= task.max_retries:
            return False
        
        return True
    
    async def _execute_task(self, task: WarmupTask):
        """Execute a warmup task"""
        try:
            task.start_time = time.time()
            task.status = "running"
            
            logger.debug(f"Executing warmup task: {task.task_id}")
            
            # Execute task based on type
            if task.task_type == WarmupTask.MODEL_LOADING:
                await self._execute_model_loading_task(task)
            elif task.task_type == WarmupTask.MODEL_WARMING:
                await self._execute_model_warming_task(task)
            elif task.task_type == WarmupTask.CONNECTION_POOLS:
                await self._execute_connection_pools_task(task)
            elif task.task_type == WarmupTask.STATE_SYNC:
                await self._execute_state_sync_task(task)
            elif task.task_type == WarmupTask.MEMORY_OPTIMIZATION:
                await self._execute_memory_optimization_task(task)
            elif task.task_type == WarmupTask.JIT_COMPILATION:
                await self._execute_jit_compilation_task(task)
            elif task.task_type == WarmupTask.HEALTH_CHECK:
                await self._execute_health_check_task(task)
            
            task.end_time = time.time()
            task.status = "completed"
            self.completed_tasks.append(task.task_id)
            
            logger.debug(f"Task {task.task_id} completed in {task.end_time - task.start_time:.2f}s")
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.retry_count += 1
            
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Retry if possible
            if task.retry_count < task.max_retries:
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
                await asyncio.sleep(1)  # Brief delay before retry
                await self._execute_task(task)
            else:
                logger.error(f"Task {task.task_id} failed after {task.max_retries} attempts")
    
    async def _execute_model_loading_task(self, task: WarmupTask):
        """Execute model loading task"""
        try:
            model_name = task.task_id.replace("load_", "")
            model_path = f"{self.config.models_dir}/{model_name}_jit.pt"
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model in executor
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                model_path
            )
            
            async with self.model_lock:
                self.loaded_models[model_name] = model
                self.model_warmup_status[model_name] = False
                self.metrics.models_loaded += 1
            
            logger.debug(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model for task {task.task_id}: {e}")
            raise
    
    def _load_model_sync(self, model_path: str) -> torch.jit.ScriptModule:
        """Load model synchronously"""
        try:
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    async def _execute_model_warming_task(self, task: WarmupTask):
        """Execute model warming task"""
        try:
            model_name = task.task_id.replace("warm_", "")
            
            async with self.model_lock:
                if model_name not in self.loaded_models:
                    raise ValueError(f"Model {model_name} not loaded")
                
                model = self.loaded_models[model_name]
            
            # Warm model in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._warm_model_sync,
                model,
                model_name
            )
            
            async with self.model_lock:
                self.model_warmup_status[model_name] = True
                self.metrics.models_warmed += 1
            
            logger.debug(f"Model {model_name} warmed successfully")
            
        except Exception as e:
            logger.error(f"Failed to warm model for task {task.task_id}: {e}")
            raise
    
    def _warm_model_sync(self, model: torch.jit.ScriptModule, model_name: str):
        """Warm model synchronously"""
        try:
            # Determine input shape based on model
            if model_name == "routing_agent":
                input_shape = [1, 55]
            else:
                input_shape = [1, 47]
            
            # Warm up model with synthetic data
            with torch.no_grad():
                for _ in range(self.config.warmup_iterations):
                    input_tensor = torch.randn(input_shape)
                    _ = model(input_tensor)
                    
            logger.debug(f"Model {model_name} warmed with {self.config.warmup_iterations} iterations")
            
        except Exception as e:
            logger.error(f"Failed to warm model {model_name}: {e}")
            raise
    
    async def _execute_connection_pools_task(self, task: WarmupTask):
        """Execute connection pools warming task"""
        try:
            # Warm Redis connection pool
            if 'redis' in self.connection_pools:
                for _ in range(10):  # Test 10 connections
                    async with self.redis_client.pipeline() as pipe:
                        await pipe.ping()
                        await pipe.execute()
            
            # Warm HTTP connection pool (placeholder)
            # In real implementation, this would pre-establish HTTP connections
            
            logger.debug("Connection pools warmed successfully")
            
        except Exception as e:
            logger.error(f"Failed to warm connection pools: {e}")
            raise
    
    async def _execute_state_sync_task(self, task: WarmupTask):
        """Execute state synchronization task"""
        try:
            if not self.state_sync:
                raise ValueError("State sync not initialized")
            
            # Initialize state sync if not already done
            if not self.state_sync.is_running:
                await self.state_sync.initialize()
            
            # Perform initial state sync
            await self.state_sync.force_full_sync()
            
            logger.debug("State synchronization initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize state sync: {e}")
            raise
    
    async def _execute_memory_optimization_task(self, task: WarmupTask):
        """Execute memory optimization task"""
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Optimize torch memory usage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Update memory metrics
            self.metrics.memory_usage_mb = self._get_memory_usage()
            
            logger.debug(f"Memory optimization completed. Usage: {self.metrics.memory_usage_mb:.2f}MB")
            
        except Exception as e:
            logger.error(f"Failed to optimize memory: {e}")
            raise
    
    async def _execute_jit_compilation_task(self, task: WarmupTask):
        """Execute JIT compilation task"""
        try:
            # Ensure all models are JIT compiled
            async with self.model_lock:
                for model_name, model in self.loaded_models.items():
                    if not isinstance(model, torch.jit.ScriptModule):
                        logger.warning(f"Model {model_name} is not JIT compiled")
                        continue
                    
                    # Trigger JIT optimization
                    model = torch.jit.optimize_for_inference(model)
                    self.loaded_models[model_name] = model
            
            logger.debug("JIT compilation optimization completed")
            
        except Exception as e:
            logger.error(f"Failed to optimize JIT compilation: {e}")
            raise
    
    async def _execute_health_check_task(self, task: WarmupTask):
        """Execute health check task"""
        try:
            # Check model readiness
            async with self.model_lock:
                for model_name in self.loaded_models:
                    if not self.model_warmup_status.get(model_name, False):
                        raise ValueError(f"Model {model_name} not warmed")
            
            # Check state sync health
            if self.state_sync:
                sync_health = await self.state_sync.get_sync_health()
                if sync_health['health_score'] < 80:
                    raise ValueError(f"State sync health poor: {sync_health['health_score']}")
            
            # Check memory usage
            memory_usage = self._get_memory_usage()
            if memory_usage > 2048:  # 2GB threshold
                logger.warning(f"High memory usage: {memory_usage:.2f}MB")
            
            logger.debug("Health check completed successfully")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    async def _verify_readiness(self) -> bool:
        """Verify system readiness"""
        try:
            readiness_score = 0.0
            max_score = 100.0
            
            # Check model readiness (50% of score)
            model_score = 0.0
            if self.loaded_models:
                warmed_models = sum(1 for status in self.model_warmup_status.values() if status)
                model_score = (warmed_models / len(self.loaded_models)) * 50.0
            readiness_score += model_score
            
            # Check state sync readiness (20% of score)
            if self.state_sync:
                sync_health = await self.state_sync.get_sync_health()
                sync_score = (sync_health['health_score'] / 100.0) * 20.0
                readiness_score += sync_score
            
            # Check memory usage (10% of score)
            memory_usage = self._get_memory_usage()
            if memory_usage < 1024:  # Under 1GB
                readiness_score += 10.0
            elif memory_usage < 2048:  # Under 2GB
                readiness_score += 5.0
            
            # Check task completion (20% of score)
            completed_ratio = len(self.completed_tasks) / len(self.warmup_tasks)
            readiness_score += completed_ratio * 20.0
            
            self.readiness_score = readiness_score
            
            # Consider ready if score > 80%
            is_ready = readiness_score >= 80.0
            
            logger.info(f"Readiness verification: {readiness_score:.1f}% (ready: {is_ready})")
            
            return is_ready
            
        except Exception as e:
            logger.error(f"Error verifying readiness: {e}")
            return False
    
    async def _background_warming_loop(self):
        """Background warming loop"""
        while True:
            try:
                await asyncio.sleep(self.config.background_warming_interval)
                
                if self.state == WarmupState.READY and self.config.continuous_warming:
                    await self._perform_background_warming()
                
            except Exception as e:
                logger.error(f"Error in background warming loop: {e}")
    
    async def _perform_background_warming(self):
        """Perform background warming"""
        try:
            # Warm models in background
            async with self.model_lock:
                for model_name, model in self.loaded_models.items():
                    if self.model_warmup_status.get(model_name, False):
                        # Perform light warming
                        await self._background_warm_model(model, model_name)
            
            # Refresh state sync
            if self.state_sync:
                await self.state_sync.force_full_sync()
            
            logger.debug("Background warming completed")
            
        except Exception as e:
            logger.error(f"Error in background warming: {e}")
    
    async def _background_warm_model(self, model: torch.jit.ScriptModule, model_name: str):
        """Warm model in background"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._light_warm_model,
                model,
                model_name
            )
        except Exception as e:
            logger.error(f"Error warming model {model_name} in background: {e}")
    
    def _light_warm_model(self, model: torch.jit.ScriptModule, model_name: str):
        """Light model warming"""
        try:
            # Determine input shape
            if model_name == "routing_agent":
                input_shape = [1, 55]
            else:
                input_shape = [1, 47]
            
            # Light warming with fewer iterations
            with torch.no_grad():
                for _ in range(5):  # Reduced iterations for background
                    input_tensor = torch.randn(input_shape)
                    _ = model(input_tensor)
                    
        except Exception as e:
            logger.error(f"Error in light warming of {model_name}: {e}")
    
    async def _health_check_loop(self):
        """Health check loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Update metrics
                self.metrics.cpu_usage_percent = self._get_cpu_usage()
                self.metrics.memory_usage_mb = self._get_memory_usage()
                
                # Check health
                if self.config.deep_health_checks:
                    await self._perform_deep_health_check()
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _perform_deep_health_check(self):
        """Perform deep health check"""
        try:
            # Check model health
            async with self.model_lock:
                for model_name, model in self.loaded_models.items():
                    # Test model inference
                    input_shape = [1, 55] if model_name == "routing_agent" else [1, 47]
                    test_input = torch.randn(input_shape)
                    
                    with torch.no_grad():
                        _ = model(test_input)
            
            # Check state sync health
            if self.state_sync:
                sync_health = await self.state_sync.get_sync_health()
                if sync_health['health_score'] < 70:
                    logger.warning(f"State sync health degraded: {sync_health['health_score']}")
            
            # Check memory usage
            memory_usage = self._get_memory_usage()
            if memory_usage > self.config.memory_threshold * 4096:  # 4GB * threshold
                logger.warning(f"High memory usage: {memory_usage:.2f}MB")
            
        except Exception as e:
            logger.error(f"Deep health check failed: {e}")
    
    async def _metrics_loop(self):
        """Metrics collection loop"""
        while True:
            try:
                await asyncio.sleep(10)  # Every 10 seconds
                
                # Update metrics
                self.metrics.cpu_usage_percent = self._get_cpu_usage()
                self.metrics.memory_usage_mb = self._get_memory_usage()
                
                # Log metrics
                await self._log_metrics()
                
                # Persist metrics
                await self._persist_metrics()
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
    
    async def _gc_loop(self):
        """Garbage collection loop"""
        while True:
            try:
                await asyncio.sleep(self.config.gc_interval)
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clean up torch memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.debug("Garbage collection completed")
                
            except Exception as e:
                logger.error(f"Error in GC loop: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    async def _register_event_handlers(self):
        """Register event handlers"""
        # Handle system events
        await self.event_bus.subscribe(
            EventType.SYSTEM_READY,
            self._handle_system_ready
        )
        
        await self.event_bus.subscribe(
            EventType.FAILOVER_INITIATED,
            self._handle_failover_initiated
        )
    
    async def _handle_system_ready(self, event: Event):
        """Handle system ready event"""
        logger.info(f"System ready event received from {event.data.get('component')}")
    
    async def _handle_failover_initiated(self, event: Event):
        """Handle failover initiated event"""
        logger.info("Failover initiated - preparing for promotion")
        
        # Perform final warmup
        await self._perform_final_warmup()
    
    async def _perform_final_warmup(self):
        """Perform final warmup before promotion"""
        try:
            # Intensive model warming
            async with self.model_lock:
                for model_name, model in self.loaded_models.items():
                    await self._final_warm_model(model, model_name)
            
            # Force state sync
            if self.state_sync:
                await self.state_sync.force_full_sync()
            
            logger.info("Final warmup completed")
            
        except Exception as e:
            logger.error(f"Error in final warmup: {e}")
    
    async def _final_warm_model(self, model: torch.jit.ScriptModule, model_name: str):
        """Final intensive model warming"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._intensive_warm_model,
                model,
                model_name
            )
        except Exception as e:
            logger.error(f"Error in final warming of {model_name}: {e}")
    
    def _intensive_warm_model(self, model: torch.jit.ScriptModule, model_name: str):
        """Intensive model warming"""
        try:
            # Determine input shape
            if model_name == "routing_agent":
                input_shape = [1, 55]
            else:
                input_shape = [1, 47]
            
            # Intensive warming
            with torch.no_grad():
                for _ in range(100):  # More iterations for final warmup
                    input_tensor = torch.randn(input_shape)
                    _ = model(input_tensor)
                    
        except Exception as e:
            logger.error(f"Error in intensive warming of {model_name}: {e}")
    
    async def _log_metrics(self):
        """Log performance metrics"""
        try:
            logger.info(f"Standby warmup metrics:")
            logger.info(f"  State: {self.state.value}")
            logger.info(f"  Readiness score: {self.readiness_score:.1f}%")
            logger.info(f"  Models loaded: {self.metrics.models_loaded}")
            logger.info(f"  Models warmed: {self.metrics.models_warmed}")
            logger.info(f"  Memory usage: {self.metrics.memory_usage_mb:.2f}MB")
            logger.info(f"  CPU usage: {self.metrics.cpu_usage_percent:.1f}%")
            logger.info(f"  Completed tasks: {len(self.completed_tasks)}/{len(self.warmup_tasks)}")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    async def _persist_metrics(self):
        """Persist metrics to Redis"""
        try:
            if self.redis_client:
                metrics_data = {
                    'state': self.state.value,
                    'readiness_score': self.readiness_score,
                    'models_loaded': self.metrics.models_loaded,
                    'models_warmed': self.metrics.models_warmed,
                    'memory_usage_mb': self.metrics.memory_usage_mb,
                    'cpu_usage_percent': self.metrics.cpu_usage_percent,
                    'completed_tasks': len(self.completed_tasks),
                    'total_tasks': len(self.warmup_tasks),
                    'timestamp': time.time()
                }
                
                await self.redis_client.hset(
                    f"standby_warmup:{self.config.instance_id}:metrics",
                    mapping=metrics_data
                )
                
                await self.redis_client.expire(
                    f"standby_warmup:{self.config.instance_id}:metrics",
                    3600  # 1 hour
                )
                
        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current warmup status"""
        return {
            'state': self.state.value,
            'readiness_score': self.readiness_score,
            'state_change_time': self.state_change_time,
            'models': {
                'loaded': self.metrics.models_loaded,
                'warmed': self.metrics.models_warmed,
                'total': len(self.loaded_models)
            },
            'tasks': {
                'completed': len(self.completed_tasks),
                'total': len(self.warmup_tasks),
                'failed': len([t for t in self.warmup_tasks.values() if t.status == 'failed'])
            },
            'performance': {
                'memory_usage_mb': self.metrics.memory_usage_mb,
                'cpu_usage_percent': self.metrics.cpu_usage_percent,
                'average_warmup_time': self.metrics.average_warmup_time
            },
            'metrics': {
                'total_warmups': self.metrics.total_warmups,
                'successful_warmups': self.metrics.successful_warmups,
                'failed_warmups': self.metrics.failed_warmups
            }
        }
    
    async def force_warmup(self):
        """Force immediate warmup"""
        logger.info("Forcing immediate warmup")
        
        async with self.warmup_lock:
            await self._perform_background_warming()
    
    async def get_model(self, model_name: str) -> Optional[torch.jit.ScriptModule]:
        """Get loaded model"""
        async with self.model_lock:
            return self.loaded_models.get(model_name)
    
    async def is_ready(self) -> bool:
        """Check if system is ready"""
        return self.state == WarmupState.READY and self.readiness_score >= 80.0
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down standby warmup system")
        
        # Cancel background tasks
        for task in [self.warmup_task, self.background_warming_task, 
                     self.health_check_task, self.metrics_task, self.gc_task]:
            if task:
                task.cancel()
        
        # Close components
        if self.state_sync:
            await self.state_sync.shutdown()
        if self.redis_client:
            await self.redis_client.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Standby warmup system shutdown complete")


# Factory function
def create_standby_warmup_system(config: Dict[str, Any]) -> StandbyWarmupSystem:
    """Create standby warmup system instance"""
    warmup_config = WarmupConfig(**config)
    return StandbyWarmupSystem(warmup_config)


# CLI interface
async def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standby Warmup System")
    parser.add_argument("--redis-url", default="redis://localhost:6379/3")
    parser.add_argument("--instance-id", default="standby_warmup_test")
    parser.add_argument("--models-dir", default="/app/models/jit_optimized")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = WarmupConfig(
        instance_id=args.instance_id,
        models_dir=args.models_dir
    )
    
    # Create and run warmup system
    warmup_system = StandbyWarmupSystem(config)
    
    try:
        await warmup_system.initialize(args.redis_url)
        
        # Wait for warmup to complete
        while not await warmup_system.is_ready():
            await asyncio.sleep(1)
            status = warmup_system.get_status()
            print(f"Warmup status: {status}")
        
        print("Warmup completed successfully!")
        
        # Keep running
        while True:
            await asyncio.sleep(10)
            status = warmup_system.get_status()
            print(f"Status: {status}")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await warmup_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())