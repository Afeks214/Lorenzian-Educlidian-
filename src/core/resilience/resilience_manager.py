"""
Unified Resilience Manager
==========================

Central orchestrator for all resilience components including circuit breakers,
retry mechanisms, health monitoring, bulkhead patterns, and chaos engineering.

Features:
- Unified configuration and management
- Automatic service discovery and registration
- Coordinated failure handling
- Comprehensive observability
- Production-ready deployment support
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager
import json
import redis.asyncio as redis

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .adaptive_circuit_breaker import AdaptiveCircuitBreaker, AdaptiveConfig
from .retry_manager import RetryManager, RetryConfig
from .health_monitor import HealthMonitor, HealthCheckConfig
from .bulkhead import BulkheadManager, BulkheadConfig, ResourceType, ResourcePriority
from .chaos_engineering import ChaosEngineer, ChaosConfig
from ..event_bus import EventBus
from ..events import Event

logger = logging.getLogger(__name__)


@dataclass
class ResilienceConfig:
    """Unified configuration for all resilience components."""
    # Service identification
    service_name: str = "grandmodel"
    environment: str = "production"
    
    # Circuit breaker configuration
    circuit_breaker_enabled: bool = True
    adaptive_circuit_breaker_enabled: bool = True
    
    # Retry configuration
    retry_enabled: bool = True
    
    # Health monitoring configuration
    health_monitoring_enabled: bool = True
    
    # Bulkhead configuration
    bulkhead_enabled: bool = True
    
    # Chaos engineering configuration
    chaos_engineering_enabled: bool = False  # Disabled by default in production
    
    # Global settings
    redis_url: str = "redis://localhost:6379/0"
    event_bus_enabled: bool = True
    
    # Observability
    metrics_enabled: bool = True
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    
    # Auto-discovery
    auto_discovery_enabled: bool = True
    discovery_interval: int = 60  # seconds
    
    # Production settings
    production_mode: bool = True
    safety_checks_enabled: bool = True
    emergency_stop_enabled: bool = True


class ResilienceManager:
    """
    Unified resilience manager that orchestrates all resilience components.
    
    This manager provides:
    - Centralized configuration and management
    - Automatic service discovery and registration
    - Coordinated failure handling across all components
    - Comprehensive observability and monitoring
    - Production-ready deployment support
    """
    
    def __init__(self, config: ResilienceConfig):
        """Initialize resilience manager."""
        self.config = config
        
        # Core components
        self.event_bus: Optional[EventBus] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Resilience components
        self.circuit_breakers: Dict[str, Union[CircuitBreaker, AdaptiveCircuitBreaker]] = {}
        self.retry_managers: Dict[str, RetryManager] = {}
        self.health_monitor: Optional[HealthMonitor] = None
        self.bulkhead_manager: Optional[BulkheadManager] = None
        self.chaos_engineer: Optional[ChaosEngineer] = None
        
        # Service registry
        self.registered_services: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.discovery_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Emergency controls
        self.emergency_stop_triggered: bool = False
        self.safety_locks: Dict[str, asyncio.Lock] = {}
        
        logger.info(f"Resilience manager initialized for {config.service_name}")
    
    async def initialize(self):
        """Initialize all resilience components."""
        logger.info("Initializing resilience manager...")
        
        # Initialize Redis connection
        if self.config.redis_url:
            self.redis_client = redis.from_url(self.config.redis_url)
            try:
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Initialize event bus
        if self.config.event_bus_enabled:
            self.event_bus = EventBus(redis_client=self.redis_client)
            await self.event_bus.initialize()
        
        # Initialize health monitor
        if self.config.health_monitoring_enabled:
            health_config = HealthCheckConfig(
                check_interval=30.0,
                timeout=10.0,
                enable_shallow=True,
                enable_deep=True,
                enable_synthetic=False,
                enable_passive=True
            )
            
            self.health_monitor = HealthMonitor(
                config=health_config,
                event_bus=self.event_bus,
                redis_client=self.redis_client
            )
            await self.health_monitor.initialize()
        
        # Initialize bulkhead manager
        if self.config.bulkhead_enabled:
            bulkhead_config = BulkheadConfig(
                max_concurrent_requests=100,
                max_thread_pool_size=50,
                max_connection_pool_size=20,
                enable_auto_scaling=True,
                metrics_enabled=True
            )
            
            self.bulkhead_manager = BulkheadManager(bulkhead_config)
            await self.bulkhead_manager.initialize()
        
        # Initialize chaos engineer (if enabled)
        if self.config.chaos_engineering_enabled:
            chaos_config = ChaosConfig(
                experiment_duration=300,
                enable_safety_checks=self.config.safety_checks_enabled,
                max_concurrent_experiments=1,
                business_hours_only=True
            )
            
            self.chaos_engineer = ChaosEngineer(
                config=chaos_config,
                health_monitor=self.health_monitor,
                bulkhead_manager=self.bulkhead_manager
            )
            await self.chaos_engineer.initialize()
        
        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self.config.auto_discovery_enabled:
            self.discovery_task = asyncio.create_task(self._discovery_loop())
        
        if self.config.metrics_enabled:
            self.metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info("Resilience manager initialization complete")
    
    async def close(self):
        """Close all resilience components."""
        logger.info("Closing resilience manager...")
        
        # Cancel background tasks
        for task in [self.monitoring_task, self.discovery_task, self.metrics_task]:
            if task:
                task.cancel()
        
        # Close components
        if self.health_monitor:
            await self.health_monitor.close()
        
        if self.bulkhead_manager:
            await self.bulkhead_manager.close()
        
        if self.chaos_engineer:
            await self.chaos_engineer.close()
        
        # Close circuit breakers
        for cb in self.circuit_breakers.values():
            await cb.close()
        
        # Close event bus
        if self.event_bus:
            await self.event_bus.close()
        
        # Close Redis
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Resilience manager closed")
    
    async def register_service(
        self,
        service_name: str,
        service_instance: Any,
        service_config: Optional[Dict[str, Any]] = None
    ):
        """Register a service with all resilience components."""
        logger.info(f"Registering service: {service_name}")
        
        config = service_config or {}
        
        # Register with service registry
        self.registered_services[service_name] = {
            'instance': service_instance,
            'config': config,
            'registered_at': datetime.now(),
            'last_health_check': None,
            'circuit_breaker': None,
            'retry_manager': None,
            'resource_pools': []
        }
        
        # Create safety lock
        self.safety_locks[service_name] = asyncio.Lock()
        
        # Register circuit breaker
        if self.config.circuit_breaker_enabled:
            await self._register_circuit_breaker(service_name, config)
        
        # Register retry manager
        if self.config.retry_enabled:
            await self._register_retry_manager(service_name, config)
        
        # Register health checks
        if self.config.health_monitoring_enabled and self.health_monitor:
            await self._register_health_checks(service_name, service_instance, config)
        
        # Register bulkhead resources
        if self.config.bulkhead_enabled and self.bulkhead_manager:
            await self._register_bulkhead_resources(service_name, config)
        
        # Register with chaos engineer
        if self.config.chaos_engineering_enabled and self.chaos_engineer:
            await self._register_chaos_testing(service_name, service_instance, config)
        
        logger.info(f"Service registration complete: {service_name}")
    
    async def _register_circuit_breaker(self, service_name: str, config: Dict[str, Any]):
        """Register circuit breaker for a service."""
        cb_config = CircuitBreakerConfig(
            service_name=service_name,
            failure_threshold=config.get('failure_threshold', 5),
            timeout_seconds=config.get('timeout_seconds', 60),
            max_retries=config.get('max_retries', 3),
            enable_ml_prediction=config.get('enable_ml_prediction', True)
        )
        
        if self.config.adaptive_circuit_breaker_enabled:
            adaptive_config = AdaptiveConfig(
                model_update_interval=300,
                adaptation_rate=0.1,
                enable_anomaly_detection=True
            )
            
            circuit_breaker = AdaptiveCircuitBreaker(
                config=cb_config,
                adaptive_config=adaptive_config,
                event_bus=self.event_bus,
                redis_client=self.redis_client
            )
        else:
            circuit_breaker = CircuitBreaker(
                config=cb_config,
                event_bus=self.event_bus,
                redis_client=self.redis_client
            )
        
        await circuit_breaker.initialize()
        self.circuit_breakers[service_name] = circuit_breaker
        self.registered_services[service_name]['circuit_breaker'] = circuit_breaker
        
        logger.info(f"Circuit breaker registered for {service_name}")
    
    async def _register_retry_manager(self, service_name: str, config: Dict[str, Any]):
        """Register retry manager for a service."""
        retry_config = RetryConfig(
            max_attempts=config.get('max_retry_attempts', 3),
            base_delay=config.get('retry_base_delay', 1.0),
            max_delay=config.get('retry_max_delay', 30.0),
            multiplier=config.get('retry_multiplier', 2.0),
            timeout=config.get('retry_timeout', 60.0),
            circuit_breaker_integration=True
        )
        
        circuit_breaker = self.circuit_breakers.get(service_name)
        retry_manager = RetryManager(retry_config, circuit_breaker, service_name)
        
        self.retry_managers[service_name] = retry_manager
        self.registered_services[service_name]['retry_manager'] = retry_manager
        
        logger.info(f"Retry manager registered for {service_name}")
    
    async def _register_health_checks(self, service_name: str, service_instance: Any, config: Dict[str, Any]):
        """Register health checks for a service."""
        # Create health check functions
        async def shallow_check():
            """Basic connectivity check."""
            try:
                if hasattr(service_instance, 'ping'):
                    await service_instance.ping()
                elif hasattr(service_instance, 'health_check'):
                    await service_instance.health_check()
                return True
            except Exception as e:
                logger.error(f"Shallow health check failed for {service_name}: {e}")
                return False
        
        async def deep_check():
            """Full functionality check."""
            try:
                if hasattr(service_instance, 'deep_health_check'):
                    return await service_instance.deep_health_check()
                else:
                    return await shallow_check()
            except Exception as e:
                logger.error(f"Deep health check failed for {service_name}: {e}")
                return False
        
        # Register with health monitor
        self.health_monitor.register_service(
            service_name=service_name,
            shallow_check=shallow_check,
            deep_check=deep_check
        )
        
        logger.info(f"Health checks registered for {service_name}")
    
    async def _register_bulkhead_resources(self, service_name: str, config: Dict[str, Any]):
        """Register bulkhead resources for a service."""
        # Create thread pool
        thread_pool = self.bulkhead_manager.create_resource_pool(
            name=f"{service_name}_threads",
            resource_type=ResourceType.THREAD_POOL,
            max_size=config.get('max_threads', 20),
            priority=ResourcePriority.MEDIUM
        )
        
        # Create connection pool
        connection_pool = self.bulkhead_manager.create_resource_pool(
            name=f"{service_name}_connections",
            resource_type=ResourceType.CONNECTION_POOL,
            max_size=config.get('max_connections', 10),
            priority=ResourcePriority.HIGH
        )
        
        # Create semaphore pool
        semaphore_pool = self.bulkhead_manager.create_resource_pool(
            name=f"{service_name}_semaphore",
            resource_type=ResourceType.SEMAPHORE,
            max_size=config.get('max_concurrent', 50),
            priority=ResourcePriority.MEDIUM
        )
        
        # Store resource pools
        self.registered_services[service_name]['resource_pools'] = [
            thread_pool, connection_pool, semaphore_pool
        ]
        
        # Register circuit breaker with bulkhead
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker:
            self.bulkhead_manager.register_circuit_breaker(service_name, circuit_breaker)
        
        logger.info(f"Bulkhead resources registered for {service_name}")
    
    async def _register_chaos_testing(self, service_name: str, service_instance: Any, config: Dict[str, Any]):
        """Register service for chaos testing."""
        from .chaos_engineering import FailureType
        
        # Define failure modes based on service type
        failure_modes = config.get('failure_modes', [
            FailureType.NETWORK_DELAY,
            FailureType.TIMEOUT,
            FailureType.EXCEPTION,
            FailureType.PARTIAL_FAILURE
        ])
        
        self.chaos_engineer.register_service(
            service_name=service_name,
            service_instance=service_instance,
            failure_modes=failure_modes
        )
        
        logger.info(f"Chaos testing registered for {service_name}")
    
    @asynccontextmanager
    async def resilient_call(
        self,
        service_name: str,
        operation_name: str = "operation",
        priority: ResourcePriority = ResourcePriority.MEDIUM
    ):
        """
        Execute a resilient call with all protection mechanisms.
        
        Usage:
            async with resilience_manager.resilient_call("database", "query"):
                result = await database.query()
        """
        if service_name not in self.registered_services:
            raise ValueError(f"Service {service_name} not registered")
        
        service_info = self.registered_services[service_name]
        circuit_breaker = service_info.get('circuit_breaker')
        retry_manager = service_info.get('retry_manager')
        
        # Check if service is available
        if self.emergency_stop_triggered:
            raise Exception("Emergency stop is active")
        
        # Use bulkhead protection
        if self.bulkhead_manager:
            semaphore_pool_name = f"{service_name}_semaphore"
            async with self.bulkhead_manager.acquire_resource(semaphore_pool_name, priority):
                # Use circuit breaker protection
                if circuit_breaker:
                    async with circuit_breaker.protect():
                        # Use retry protection
                        if retry_manager:
                            async with retry_manager.retry(operation_name):
                                yield
                        else:
                            yield
                else:
                    # Use retry protection without circuit breaker
                    if retry_manager:
                        async with retry_manager.retry(operation_name):
                            yield
                    else:
                        yield
        else:
            # Use circuit breaker protection without bulkhead
            if circuit_breaker:
                async with circuit_breaker.protect():
                    # Use retry protection
                    if retry_manager:
                        async with retry_manager.retry(operation_name):
                            yield
                    else:
                        yield
            else:
                # Use retry protection without circuit breaker
                if retry_manager:
                    async with retry_manager.retry(operation_name):
                        yield
                else:
                    yield
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(30)
                await self._check_system_health()
                await self._update_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _discovery_loop(self):
        """Background service discovery loop."""
        while True:
            try:
                await asyncio.sleep(self.config.discovery_interval)
                await self._discover_services()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
    
    async def _metrics_loop(self):
        """Background metrics collection loop."""
        while True:
            try:
                await asyncio.sleep(60)
                await self._collect_and_export_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
    
    async def _check_system_health(self):
        """Check overall system health."""
        if not self.health_monitor:
            return
        
        # Get system health summary
        health_summary = self.health_monitor.get_system_health_summary()
        
        # Check for emergency conditions
        if (health_summary['unhealthy_services'] > 0 and
            health_summary['system_health_score'] < 50.0):
            
            logger.warning("System health is degraded")
            
            # Trigger emergency protocols if needed
            if (health_summary['system_health_score'] < 20.0 and
                self.config.emergency_stop_enabled):
                await self._trigger_emergency_stop()
    
    async def _discover_services(self):
        """Discover new services automatically."""
        # This would integrate with service discovery systems like Consul, etcd, etc.
        # For now, we'll just log that discovery is running
        logger.debug("Service discovery scan completed")
    
    async def _collect_and_export_metrics(self):
        """Collect and export metrics."""
        try:
            # Collect metrics from all components
            metrics = {
                'timestamp': time.time(),
                'service_name': self.config.service_name,
                'environment': self.config.environment,
                'registered_services': len(self.registered_services),
                'circuit_breakers': {},
                'retry_managers': {},
                'health_summary': {},
                'bulkhead_status': {},
                'chaos_experiments': {}
            }
            
            # Circuit breaker metrics
            for name, cb in self.circuit_breakers.items():
                metrics['circuit_breakers'][name] = cb.get_status()
            
            # Retry manager metrics
            for name, rm in self.retry_managers.items():
                metrics['retry_managers'][name] = rm.get_metrics()
            
            # Health monitor metrics
            if self.health_monitor:
                metrics['health_summary'] = self.health_monitor.get_system_health_summary()
            
            # Bulkhead metrics
            if self.bulkhead_manager:
                metrics['bulkhead_status'] = self.bulkhead_manager.get_all_pools_status()
            
            # Chaos engineering metrics
            if self.chaos_engineer:
                metrics['chaos_experiments'] = self.chaos_engineer.get_all_experiments_status()
            
            # Export metrics
            await self._export_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def _export_metrics(self, metrics: Dict[str, Any]):
        """Export metrics to monitoring systems."""
        # Store in Redis for real-time access
        if self.redis_client:
            try:
                await self.redis_client.set(
                    f"resilience_metrics:{self.config.service_name}",
                    json.dumps(metrics, default=str),
                    ex=300  # 5 minutes
                )
            except Exception as e:
                logger.error(f"Failed to store metrics in Redis: {e}")
        
        # Send to event bus for processing
        if self.event_bus:
            try:
                await self.event_bus.publish(Event(
                    type="resilience_metrics",
                    data=metrics
                ))
            except Exception as e:
                logger.error(f"Failed to publish metrics event: {e}")
    
    async def _trigger_emergency_stop(self):
        """Trigger emergency stop procedures."""
        if self.emergency_stop_triggered:
            return
        
        self.emergency_stop_triggered = True
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        # Stop chaos experiments
        if self.chaos_engineer:
            await self.chaos_engineer._emergency_stop()
        
        # Open all circuit breakers
        for cb in self.circuit_breakers.values():
            await cb.force_open()
        
        # Send emergency alert
        if self.event_bus:
            await self.event_bus.publish(Event(
                type="emergency_stop",
                data={
                    'service_name': self.config.service_name,
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'System health critical'
                }
            ))
    
    async def reset_emergency_stop(self):
        """Reset emergency stop (manual intervention required)."""
        self.emergency_stop_triggered = False
        
        # Close all circuit breakers
        for cb in self.circuit_breakers.values():
            await cb.force_close()
        
        logger.info("Emergency stop reset")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'service_name': self.config.service_name,
            'environment': self.config.environment,
            'emergency_stop_active': self.emergency_stop_triggered,
            'registered_services': len(self.registered_services),
            'components': {
                'circuit_breakers': len(self.circuit_breakers),
                'retry_managers': len(self.retry_managers),
                'health_monitor': self.health_monitor is not None,
                'bulkhead_manager': self.bulkhead_manager is not None,
                'chaos_engineer': self.chaos_engineer is not None
            },
            'services': {}
        }
        
        # Add service details
        for service_name, service_info in self.registered_services.items():
            status['services'][service_name] = {
                'registered_at': service_info['registered_at'].isoformat(),
                'has_circuit_breaker': service_info['circuit_breaker'] is not None,
                'has_retry_manager': service_info['retry_manager'] is not None,
                'resource_pools': len(service_info['resource_pools'])
            }
        
        return status
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get detailed status for a specific service."""
        if service_name not in self.registered_services:
            return {'error': f'Service {service_name} not registered'}
        
        service_info = self.registered_services[service_name]
        status = {
            'service_name': service_name,
            'registered_at': service_info['registered_at'].isoformat(),
            'circuit_breaker': None,
            'retry_manager': None,
            'health_status': None,
            'resource_pools': []
        }
        
        # Circuit breaker status
        if service_info['circuit_breaker']:
            status['circuit_breaker'] = service_info['circuit_breaker'].get_status()
        
        # Retry manager status
        if service_info['retry_manager']:
            status['retry_manager'] = service_info['retry_manager'].get_metrics()
        
        # Health status
        if self.health_monitor:
            health = self.health_monitor.get_service_health(service_name)
            if health:
                status['health_status'] = {
                    'overall_status': health.overall_status.value,
                    'overall_score': health.overall_score,
                    'last_updated': health.last_updated.isoformat(),
                    'availability': health.availability,
                    'average_response_time': health.average_response_time,
                    'error_rate': health.error_rate
                }
        
        # Resource pool status
        if self.bulkhead_manager:
            for pool in service_info['resource_pools']:
                pool_status = self.bulkhead_manager.get_pool_status(pool.name)
                if pool_status:
                    status['resource_pools'].append(pool_status)
        
        return status
    
    async def run_health_check(self, service_name: str):
        """Run immediate health check for a service."""
        if self.health_monitor:
            await self.health_monitor.force_health_check(service_name)
    
    async def run_chaos_experiment(self, experiment_name: str, service_name: str):
        """Run chaos experiment on a service."""
        if not self.chaos_engineer:
            raise ValueError("Chaos engineering is not enabled")
        
        if service_name not in self.registered_services:
            raise ValueError(f"Service {service_name} not registered")
        
        from .chaos_engineering import FailureInjection, FailureType
        
        # Create basic failure injection
        injection = FailureInjection(
            failure_type=FailureType.NETWORK_DELAY,
            target_service=service_name,
            intensity=0.3,
            duration=30
        )
        
        return await self.chaos_engineer.run_experiment(experiment_name, [injection])
    
    async def drain_service(self, service_name: str):
        """Drain a service gracefully."""
        if service_name not in self.registered_services:
            raise ValueError(f"Service {service_name} not registered")
        
        logger.info(f"Draining service: {service_name}")
        
        # Drain bulkhead resources
        if self.bulkhead_manager:
            for pool in self.registered_services[service_name]['resource_pools']:
                await self.bulkhead_manager.drain_pool(pool.name)
        
        # Unregister from health monitor
        if self.health_monitor:
            self.health_monitor.unregister_service(service_name)
        
        # Close circuit breaker
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker:
            await circuit_breaker.close()
            del self.circuit_breakers[service_name]
        
        # Remove from registry
        del self.registered_services[service_name]
        del self.safety_locks[service_name]
        
        logger.info(f"Service drained: {service_name}")


# Convenience function for easy initialization
async def create_resilience_manager(
    service_name: str,
    environment: str = "production",
    redis_url: str = "redis://localhost:6379/0",
    **kwargs
) -> ResilienceManager:
    """Create and initialize a resilience manager with sensible defaults."""
    config = ResilienceConfig(
        service_name=service_name,
        environment=environment,
        redis_url=redis_url,
        **kwargs
    )
    
    manager = ResilienceManager(config)
    await manager.initialize()
    
    return manager