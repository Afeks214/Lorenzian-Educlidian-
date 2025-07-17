"""
Bulkhead Pattern Implementation
==============================

Resource isolation using the bulkhead pattern to prevent cascading failures
and ensure system resilience through compartmentalization.

Features:
- Resource pool management
- Thread pool isolation
- Connection pool isolation
- Request rate limiting per resource
- Resource priority management
- Automatic resource scaling
- Failure isolation between services
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be managed by bulkheads."""
    THREAD_POOL = "thread_pool"
    CONNECTION_POOL = "connection_pool"
    SEMAPHORE = "semaphore"
    RATE_LIMITER = "rate_limiter"
    MEMORY_POOL = "memory_pool"
    CPU_POOL = "cpu_pool"


class ResourcePriority(Enum):
    """Priority levels for resource allocation."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead resource management."""
    # Resource limits
    max_concurrent_requests: int = 100
    max_thread_pool_size: int = 50
    max_connection_pool_size: int = 20
    max_memory_mb: int = 500
    
    # Scaling parameters
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_pool_size: int = 5
    max_pool_size: int = 200
    
    # Timeout settings
    acquire_timeout: float = 30.0
    operation_timeout: float = 60.0
    
    # Resource priority settings
    priority_queuing: bool = True
    priority_boost_factor: float = 1.5
    
    # Monitoring settings
    metrics_enabled: bool = True
    health_check_interval: float = 30.0
    
    # Failure handling
    failure_isolation: bool = True
    cascade_failure_threshold: int = 5
    
    # Circuit breaker integration
    circuit_breaker_integration: bool = True


@dataclass
class ResourceMetrics:
    """Metrics for resource utilization."""
    total_requests: int = 0
    active_requests: int = 0
    queued_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    
    # Timing metrics
    average_wait_time: float = 0.0
    average_processing_time: float = 0.0
    max_wait_time: float = 0.0
    max_processing_time: float = 0.0
    
    # Resource utilization
    current_utilization: float = 0.0
    peak_utilization: float = 0.0
    
    # Scaling metrics
    scale_up_events: int = 0
    scale_down_events: int = 0


@dataclass
class ResourcePool:
    """Resource pool for bulkhead isolation."""
    name: str
    resource_type: ResourceType
    priority: ResourcePriority
    max_size: int
    current_size: int
    available: int
    
    # Pool resources
    semaphore: asyncio.Semaphore
    thread_pool: Optional[ThreadPoolExecutor] = None
    connection_pool: Optional[Any] = None
    
    # Metrics
    metrics: ResourceMetrics = field(default_factory=ResourceMetrics)
    
    # Configuration
    config: BulkheadConfig = field(default_factory=BulkheadConfig)
    
    # Tracking
    creation_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Queues
    waiting_requests: List[Any] = field(default_factory=list)
    priority_queue: Dict[ResourcePriority, List[Any]] = field(default_factory=dict)


class BulkheadManager:
    """
    Bulkhead manager for resource isolation and management.
    
    Provides:
    - Resource pool management
    - Request isolation
    - Automatic scaling
    - Priority-based resource allocation
    - Failure isolation
    """
    
    def __init__(self, config: BulkheadConfig):
        """Initialize bulkhead manager."""
        self.config = config
        
        # Resource pools
        self.resource_pools: Dict[str, ResourcePool] = {}
        
        # Global metrics
        self.global_metrics = ResourceMetrics()
        
        # Scaling state
        self.scaling_locks: Dict[str, asyncio.Lock] = {}
        self.scaling_cooldowns: Dict[str, float] = {}
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.scaling_task: Optional[asyncio.Task] = None
        
        # Circuit breaker references
        self.circuit_breakers: Dict[str, weakref.ref] = {}
        
        logger.info("Bulkhead manager initialized")
    
    async def initialize(self):
        """Initialize bulkhead manager."""
        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        
        logger.info("Bulkhead manager initialized")
    
    async def close(self):
        """Close bulkhead manager."""
        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.scaling_task:
            self.scaling_task.cancel()
        
        # Close all resource pools
        for pool in self.resource_pools.values():
            await self._close_resource_pool(pool)
        
        logger.info("Bulkhead manager closed")
    
    def create_resource_pool(
        self,
        name: str,
        resource_type: ResourceType,
        max_size: int,
        priority: ResourcePriority = ResourcePriority.MEDIUM,
        config: Optional[BulkheadConfig] = None
    ) -> ResourcePool:
        """Create a new resource pool."""
        pool_config = config or self.config
        
        # Create semaphore for resource limiting
        semaphore = asyncio.Semaphore(max_size)
        
        # Create thread pool if needed
        thread_pool = None
        if resource_type == ResourceType.THREAD_POOL:
            thread_pool = ThreadPoolExecutor(
                max_workers=max_size,
                thread_name_prefix=f"bulkhead_{name}_"
            )
        
        # Create resource pool
        pool = ResourcePool(
            name=name,
            resource_type=resource_type,
            priority=priority,
            max_size=max_size,
            current_size=max_size,
            available=max_size,
            semaphore=semaphore,
            thread_pool=thread_pool,
            config=pool_config
        )
        
        # Initialize priority queues
        pool.priority_queue = {priority: [] for priority in ResourcePriority}
        
        # Register pool
        self.resource_pools[name] = pool
        self.scaling_locks[name] = asyncio.Lock()
        self.scaling_cooldowns[name] = 0.0
        
        logger.info(f"Created resource pool: {name} ({resource_type.value}, max_size={max_size})")
        return pool
    
    async def _close_resource_pool(self, pool: ResourcePool):
        """Close a resource pool."""
        if pool.thread_pool:
            pool.thread_pool.shutdown(wait=True)
        
        # Close connection pool if it exists
        if pool.connection_pool and hasattr(pool.connection_pool, 'close'):
            await pool.connection_pool.close()
        
        logger.info(f"Closed resource pool: {pool.name}")
    
    @asynccontextmanager
    async def acquire_resource(
        self,
        pool_name: str,
        priority: ResourcePriority = ResourcePriority.MEDIUM,
        timeout: Optional[float] = None
    ):
        """
        Acquire a resource from a pool with priority handling.
        
        Usage:
            async with bulkhead.acquire_resource("database_pool", ResourcePriority.HIGH):
                # Use resource
                pass
        """
        pool = self.resource_pools.get(pool_name)
        if not pool:
            raise ValueError(f"Resource pool {pool_name} not found")
        
        acquire_timeout = timeout or self.config.acquire_timeout
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if self.config.circuit_breaker_integration:
                circuit_breaker = self.circuit_breakers.get(pool_name)
                if circuit_breaker and circuit_breaker():
                    if not await circuit_breaker().can_execute():
                        pool.metrics.rejected_requests += 1
                        raise ResourceRejectedError(f"Circuit breaker open for {pool_name}")
            
            # Priority-based acquisition
            if self.config.priority_queuing:
                await self._acquire_with_priority(pool, priority, acquire_timeout)
            else:
                await asyncio.wait_for(pool.semaphore.acquire(), timeout=acquire_timeout)
            
            # Update metrics
            wait_time = time.time() - start_time
            pool.metrics.active_requests += 1
            pool.metrics.total_requests += 1
            pool.metrics.average_wait_time = self._update_average(
                pool.metrics.average_wait_time,
                wait_time,
                pool.metrics.total_requests
            )
            pool.metrics.max_wait_time = max(pool.metrics.max_wait_time, wait_time)
            
            # Update utilization
            pool.available -= 1
            pool.metrics.current_utilization = 1.0 - (pool.available / pool.current_size)
            pool.metrics.peak_utilization = max(
                pool.metrics.peak_utilization,
                pool.metrics.current_utilization
            )
            
            # Update activity time
            pool.last_activity = datetime.now()
            
            logger.debug(f"Acquired resource from {pool_name} (priority={priority.name})")
            
            # Yield to the context
            operation_start = time.time()
            try:
                yield
                
                # Success
                pool.metrics.successful_requests += 1
                
            except Exception as e:
                pool.metrics.failed_requests += 1
                
                # Handle failure isolation
                if self.config.failure_isolation:
                    await self._handle_failure_isolation(pool, e)
                
                raise
                
            finally:
                # Update processing time
                processing_time = time.time() - operation_start
                pool.metrics.average_processing_time = self._update_average(
                    pool.metrics.average_processing_time,
                    processing_time,
                    pool.metrics.total_requests
                )
                pool.metrics.max_processing_time = max(
                    pool.metrics.max_processing_time,
                    processing_time
                )
                
        except asyncio.TimeoutError:
            pool.metrics.rejected_requests += 1
            raise ResourceTimeoutError(f"Timeout acquiring resource from {pool_name}")
            
        finally:
            # Release resource
            if pool.available < pool.current_size:
                pool.semaphore.release()
                pool.available += 1
                pool.metrics.active_requests -= 1
                pool.metrics.current_utilization = 1.0 - (pool.available / pool.current_size)
    
    async def _acquire_with_priority(
        self,
        pool: ResourcePool,
        priority: ResourcePriority,
        timeout: float
    ):
        """Acquire resource with priority handling."""
        # Add to priority queue
        future = asyncio.Future()
        pool.priority_queue[priority].append(future)
        
        # Calculate adjusted timeout based on priority
        adjusted_timeout = timeout
        if priority == ResourcePriority.CRITICAL:
            adjusted_timeout *= self.config.priority_boost_factor
        
        try:
            # Wait for resource allocation
            await asyncio.wait_for(future, timeout=adjusted_timeout)
            
            # Acquire semaphore
            await pool.semaphore.acquire()
            
        except asyncio.TimeoutError:
            # Remove from queue
            if future in pool.priority_queue[priority]:
                pool.priority_queue[priority].remove(future)
            raise
    
    async def _handle_failure_isolation(self, pool: ResourcePool, exception: Exception):
        """Handle failure isolation for a resource pool."""
        # Track consecutive failures
        if not hasattr(pool, 'consecutive_failures'):
            pool.consecutive_failures = 0
        
        pool.consecutive_failures += 1
        
        # Check if we should isolate the resource
        if pool.consecutive_failures >= self.config.cascade_failure_threshold:
            logger.warning(f"Isolating resource pool {pool.name} due to consecutive failures")
            
            # Reduce pool size temporarily
            if pool.current_size > self.config.min_pool_size:
                await self._scale_pool(pool, pool.current_size // 2)
            
            # Reset failure count
            pool.consecutive_failures = 0
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._update_global_metrics()
                await self._check_resource_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _scaling_loop(self):
        """Background scaling loop."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                if self.config.enable_auto_scaling:
                    await self._auto_scale_pools()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
    
    async def _auto_scale_pools(self):
        """Automatically scale resource pools based on utilization."""
        current_time = time.time()
        
        for pool_name, pool in self.resource_pools.items():
            # Check scaling cooldown
            if current_time < self.scaling_cooldowns.get(pool_name, 0):
                continue
            
            async with self.scaling_locks[pool_name]:
                # Scale up if utilization is high
                if (pool.metrics.current_utilization > self.config.scale_up_threshold and
                    pool.current_size < self.config.max_pool_size):
                    
                    new_size = min(
                        pool.current_size + max(1, pool.current_size // 4),
                        self.config.max_pool_size
                    )
                    
                    await self._scale_pool(pool, new_size)
                    pool.metrics.scale_up_events += 1
                    
                    # Set cooldown
                    self.scaling_cooldowns[pool_name] = current_time + 60.0  # 1 minute cooldown
                    
                    logger.info(f"Scaled up pool {pool_name}: {pool.current_size} -> {new_size}")
                
                # Scale down if utilization is low
                elif (pool.metrics.current_utilization < self.config.scale_down_threshold and
                      pool.current_size > self.config.min_pool_size):
                    
                    new_size = max(
                        pool.current_size - max(1, pool.current_size // 4),
                        self.config.min_pool_size
                    )
                    
                    await self._scale_pool(pool, new_size)
                    pool.metrics.scale_down_events += 1
                    
                    # Set cooldown
                    self.scaling_cooldowns[pool_name] = current_time + 60.0  # 1 minute cooldown
                    
                    logger.info(f"Scaled down pool {pool_name}: {pool.current_size} -> {new_size}")
    
    async def _scale_pool(self, pool: ResourcePool, new_size: int):
        """Scale a resource pool to a new size."""
        old_size = pool.current_size
        
        if new_size > old_size:
            # Scale up
            additional_permits = new_size - old_size
            for _ in range(additional_permits):
                pool.semaphore.release()
            
            pool.available += additional_permits
            
            # Scale thread pool if applicable
            if pool.thread_pool:
                pool.thread_pool._max_workers = new_size
        
        elif new_size < old_size:
            # Scale down
            reduced_permits = old_size - new_size
            
            # Acquire permits to reduce availability
            for _ in range(min(reduced_permits, pool.available)):
                try:
                    await asyncio.wait_for(pool.semaphore.acquire(), timeout=0.1)
                except asyncio.TimeoutError:
                    break
            
            pool.available = max(0, pool.available - reduced_permits)
            
            # Scale thread pool if applicable
            if pool.thread_pool:
                pool.thread_pool._max_workers = new_size
        
        pool.current_size = new_size
        pool.max_size = new_size
        
        # Update utilization
        pool.metrics.current_utilization = 1.0 - (pool.available / pool.current_size)
    
    async def _update_global_metrics(self):
        """Update global metrics across all pools."""
        self.global_metrics.total_requests = sum(
            pool.metrics.total_requests for pool in self.resource_pools.values()
        )
        
        self.global_metrics.active_requests = sum(
            pool.metrics.active_requests for pool in self.resource_pools.values()
        )
        
        self.global_metrics.successful_requests = sum(
            pool.metrics.successful_requests for pool in self.resource_pools.values()
        )
        
        self.global_metrics.failed_requests = sum(
            pool.metrics.failed_requests for pool in self.resource_pools.values()
        )
        
        self.global_metrics.rejected_requests = sum(
            pool.metrics.rejected_requests for pool in self.resource_pools.values()
        )
        
        # Calculate average utilization
        if self.resource_pools:
            self.global_metrics.current_utilization = sum(
                pool.metrics.current_utilization for pool in self.resource_pools.values()
            ) / len(self.resource_pools)
    
    async def _check_resource_health(self):
        """Check health of all resource pools."""
        unhealthy_pools = []
        
        for pool_name, pool in self.resource_pools.items():
            # Check for stuck requests
            if pool.metrics.active_requests > 0:
                inactive_time = (datetime.now() - pool.last_activity).total_seconds()
                if inactive_time > 300:  # 5 minutes
                    unhealthy_pools.append(pool_name)
                    logger.warning(f"Pool {pool_name} appears to have stuck requests")
            
            # Check error rate
            if pool.metrics.total_requests > 0:
                error_rate = pool.metrics.failed_requests / pool.metrics.total_requests
                if error_rate > 0.5:  # 50% error rate
                    unhealthy_pools.append(pool_name)
                    logger.warning(f"Pool {pool_name} has high error rate: {error_rate:.2%}")
        
        # Handle unhealthy pools
        for pool_name in unhealthy_pools:
            await self._handle_unhealthy_pool(pool_name)
    
    async def _handle_unhealthy_pool(self, pool_name: str):
        """Handle an unhealthy resource pool."""
        pool = self.resource_pools[pool_name]
        
        # Try to recover the pool
        if pool.resource_type == ResourceType.THREAD_POOL and pool.thread_pool:
            # Restart thread pool
            pool.thread_pool.shutdown(wait=False)
            pool.thread_pool = ThreadPoolExecutor(
                max_workers=pool.current_size,
                thread_name_prefix=f"bulkhead_{pool_name}_"
            )
            
            logger.info(f"Restarted thread pool for {pool_name}")
        
        # Reset metrics
        pool.metrics.failed_requests = 0
        pool.metrics.active_requests = 0
        pool.last_activity = datetime.now()
    
    def _update_average(self, current_average: float, new_value: float, count: int) -> float:
        """Update rolling average."""
        return ((current_average * (count - 1)) + new_value) / count
    
    def register_circuit_breaker(self, pool_name: str, circuit_breaker: Any):
        """Register a circuit breaker for a resource pool."""
        self.circuit_breakers[pool_name] = weakref.ref(circuit_breaker)
        logger.info(f"Registered circuit breaker for pool {pool_name}")
    
    def get_pool_status(self, pool_name: str) -> Dict[str, Any]:
        """Get status of a specific resource pool."""
        pool = self.resource_pools.get(pool_name)
        if not pool:
            return {}
        
        return {
            'name': pool.name,
            'resource_type': pool.resource_type.value,
            'priority': pool.priority.value,
            'current_size': pool.current_size,
            'available': pool.available,
            'utilization': pool.metrics.current_utilization,
            'metrics': {
                'total_requests': pool.metrics.total_requests,
                'active_requests': pool.metrics.active_requests,
                'successful_requests': pool.metrics.successful_requests,
                'failed_requests': pool.metrics.failed_requests,
                'rejected_requests': pool.metrics.rejected_requests,
                'average_wait_time': pool.metrics.average_wait_time,
                'average_processing_time': pool.metrics.average_processing_time,
                'scale_up_events': pool.metrics.scale_up_events,
                'scale_down_events': pool.metrics.scale_down_events
            },
            'creation_time': pool.creation_time.isoformat(),
            'last_activity': pool.last_activity.isoformat()
        }
    
    def get_all_pools_status(self) -> Dict[str, Any]:
        """Get status of all resource pools."""
        return {
            'pools': {
                name: self.get_pool_status(name)
                for name in self.resource_pools.keys()
            },
            'global_metrics': {
                'total_requests': self.global_metrics.total_requests,
                'active_requests': self.global_metrics.active_requests,
                'successful_requests': self.global_metrics.successful_requests,
                'failed_requests': self.global_metrics.failed_requests,
                'rejected_requests': self.global_metrics.rejected_requests,
                'current_utilization': self.global_metrics.current_utilization
            }
        }
    
    async def drain_pool(self, pool_name: str, timeout: float = 60.0):
        """Drain a resource pool gracefully."""
        pool = self.resource_pools.get(pool_name)
        if not pool:
            return
        
        logger.info(f"Draining pool {pool_name}")
        
        # Wait for active requests to complete
        start_time = time.time()
        while pool.metrics.active_requests > 0 and time.time() - start_time < timeout:
            await asyncio.sleep(1)
        
        # Force close if timeout exceeded
        if pool.metrics.active_requests > 0:
            logger.warning(f"Forced drain of pool {pool_name} due to timeout")
        
        # Close the pool
        await self._close_resource_pool(pool)
        
        # Remove from registry
        del self.resource_pools[pool_name]
        del self.scaling_locks[pool_name]
        del self.scaling_cooldowns[pool_name]
        
        logger.info(f"Pool {pool_name} drained and removed")


class ResourceTimeoutError(Exception):
    """Exception raised when resource acquisition times out."""
    pass


class ResourceRejectedError(Exception):
    """Exception raised when resource request is rejected."""
    pass


class ResourceExhaustedError(Exception):
    """Exception raised when resource pool is exhausted."""
    pass