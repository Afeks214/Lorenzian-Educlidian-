"""
Comprehensive Health Monitoring System
======================================

Advanced health monitoring with real-time service health checks,
failure detection, and recovery coordination.

Features:
- Multi-layered health checks (shallow, deep, synthetic)
- Real-time health status tracking
- Proactive failure detection
- Recovery coordination
- Performance-based health scoring
- Integration with circuit breakers and retry mechanisms
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import json
import statistics
import redis.asyncio as redis

from ..event_bus import EventBus
from ..events import Event

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """Types of health checks."""
    SHALLOW = "shallow"      # Basic connectivity check
    DEEP = "deep"           # Full functionality check
    SYNTHETIC = "synthetic"  # End-to-end workflow check
    PASSIVE = "passive"     # Metrics-based check


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    # Basic configuration
    check_interval: float = 30.0
    timeout: float = 10.0
    
    # Health check types to perform
    enable_shallow: bool = True
    enable_deep: bool = True
    enable_synthetic: bool = False
    enable_passive: bool = True
    
    # Failure thresholds
    failure_threshold: int = 3
    recovery_threshold: int = 2
    
    # Performance thresholds
    response_time_threshold: float = 5.0
    error_rate_threshold: float = 0.1
    
    # Scoring weights
    response_time_weight: float = 0.3
    error_rate_weight: float = 0.4
    availability_weight: float = 0.3
    
    # Recovery settings
    auto_recovery: bool = True
    recovery_delay: float = 60.0
    
    # Alerting
    alert_on_degraded: bool = True
    alert_on_unhealthy: bool = True
    alert_on_critical: bool = True


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_name: str
    check_type: HealthCheckType
    status: HealthStatus
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    score: float = 0.0


@dataclass
class ServiceHealth:
    """Comprehensive health information for a service."""
    service_name: str
    overall_status: HealthStatus
    overall_score: float
    last_updated: datetime
    
    # Check results
    shallow_check: Optional[HealthCheckResult] = None
    deep_check: Optional[HealthCheckResult] = None
    synthetic_check: Optional[HealthCheckResult] = None
    passive_check: Optional[HealthCheckResult] = None
    
    # Performance metrics
    average_response_time: float = 0.0
    error_rate: float = 0.0
    availability: float = 1.0
    
    # Historical data
    uptime_percentage: float = 100.0
    mttr: float = 0.0  # Mean Time To Recovery
    mtbf: float = 0.0  # Mean Time Between Failures
    
    # Failure tracking
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    recovery_time: Optional[datetime] = None
    
    # Alerts
    active_alerts: List[str] = field(default_factory=list)


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Provides:
    - Multi-layered health checks
    - Real-time status tracking
    - Proactive failure detection
    - Recovery coordination
    - Performance-based scoring
    """
    
    def __init__(
        self,
        config: HealthCheckConfig,
        event_bus: Optional[EventBus] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        """Initialize health monitor."""
        self.config = config
        self.event_bus = event_bus
        self.redis_client = redis_client
        
        # Service registry
        self.services: Dict[str, ServiceHealth] = {}
        self.health_check_functions: Dict[str, Dict[HealthCheckType, Callable]] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.error_history: Dict[str, List[bool]] = {}
        
        # Background tasks
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.aggregation_task: Optional[asyncio.Task] = None
        
        # Recovery coordination
        self.recovery_locks: Dict[str, asyncio.Lock] = {}
        self.recovery_attempts: Dict[str, int] = {}
        
        logger.info("Health monitor initialized")
    
    async def initialize(self):
        """Initialize health monitor."""
        # Start aggregation task
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        
        logger.info("Health monitor initialized")
    
    async def close(self):
        """Close health monitor."""
        # Cancel all tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        if self.aggregation_task:
            self.aggregation_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
        
        if self.aggregation_task:
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitor closed")
    
    def register_service(
        self,
        service_name: str,
        shallow_check: Optional[Callable] = None,
        deep_check: Optional[Callable] = None,
        synthetic_check: Optional[Callable] = None,
        passive_check: Optional[Callable] = None
    ):
        """Register a service for health monitoring."""
        # Initialize service health
        self.services[service_name] = ServiceHealth(
            service_name=service_name,
            overall_status=HealthStatus.UNKNOWN,
            overall_score=0.0,
            last_updated=datetime.now()
        )
        
        # Register health check functions
        self.health_check_functions[service_name] = {}
        
        if shallow_check:
            self.health_check_functions[service_name][HealthCheckType.SHALLOW] = shallow_check
        
        if deep_check:
            self.health_check_functions[service_name][HealthCheckType.DEEP] = deep_check
        
        if synthetic_check:
            self.health_check_functions[service_name][HealthCheckType.SYNTHETIC] = synthetic_check
        
        if passive_check:
            self.health_check_functions[service_name][HealthCheckType.PASSIVE] = passive_check
        
        # Initialize performance tracking
        self.performance_history[service_name] = []
        self.error_history[service_name] = []
        
        # Create recovery lock
        self.recovery_locks[service_name] = asyncio.Lock()
        self.recovery_attempts[service_name] = 0
        
        # Start health check task
        self.health_check_tasks[service_name] = asyncio.create_task(
            self._health_check_loop(service_name)
        )
        
        logger.info(f"Service registered for health monitoring: {service_name}")
    
    def unregister_service(self, service_name: str):
        """Unregister a service from health monitoring."""
        # Cancel health check task
        if service_name in self.health_check_tasks:
            self.health_check_tasks[service_name].cancel()
            del self.health_check_tasks[service_name]
        
        # Remove from registries
        self.services.pop(service_name, None)
        self.health_check_functions.pop(service_name, None)
        self.performance_history.pop(service_name, None)
        self.error_history.pop(service_name, None)
        self.recovery_locks.pop(service_name, None)
        self.recovery_attempts.pop(service_name, None)
        
        logger.info(f"Service unregistered from health monitoring: {service_name}")
    
    async def _health_check_loop(self, service_name: str):
        """Main health check loop for a service."""
        while True:
            try:
                await self._perform_health_checks(service_name)
                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error for {service_name}: {e}")
                await asyncio.sleep(self.config.check_interval)
    
    async def _perform_health_checks(self, service_name: str):
        """Perform all configured health checks for a service."""
        service_health = self.services[service_name]
        check_functions = self.health_check_functions.get(service_name, {})
        
        # Perform shallow check
        if self.config.enable_shallow and HealthCheckType.SHALLOW in check_functions:
            result = await self._execute_health_check(
                service_name, 
                HealthCheckType.SHALLOW, 
                check_functions[HealthCheckType.SHALLOW]
            )
            service_health.shallow_check = result
        
        # Perform deep check
        if self.config.enable_deep and HealthCheckType.DEEP in check_functions:
            result = await self._execute_health_check(
                service_name,
                HealthCheckType.DEEP,
                check_functions[HealthCheckType.DEEP]
            )
            service_health.deep_check = result
        
        # Perform synthetic check
        if self.config.enable_synthetic and HealthCheckType.SYNTHETIC in check_functions:
            result = await self._execute_health_check(
                service_name,
                HealthCheckType.SYNTHETIC,
                check_functions[HealthCheckType.SYNTHETIC]
            )
            service_health.synthetic_check = result
        
        # Perform passive check
        if self.config.enable_passive and HealthCheckType.PASSIVE in check_functions:
            result = await self._execute_health_check(
                service_name,
                HealthCheckType.PASSIVE,
                check_functions[HealthCheckType.PASSIVE]
            )
            service_health.passive_check = result
        
        # Update service health
        await self._update_service_health(service_name)
    
    async def _execute_health_check(
        self, 
        service_name: str, 
        check_type: HealthCheckType, 
        check_function: Callable
    ) -> HealthCheckResult:
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            # Execute check with timeout
            result = await asyncio.wait_for(
                check_function(),
                timeout=self.config.timeout
            )
            
            response_time = time.time() - start_time
            
            # Determine status based on result
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'healthy'))
                details = result.get('details', {})
                score = result.get('score', self._calculate_score(response_time, True))
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                details = {}
                score = self._calculate_score(response_time, result)
            
            # Track performance
            self.performance_history[service_name].append(response_time)
            self.error_history[service_name].append(status == HealthStatus.HEALTHY)
            
            # Limit history size
            if len(self.performance_history[service_name]) > 100:
                self.performance_history[service_name].pop(0)
                self.error_history[service_name].pop(0)
            
            return HealthCheckResult(
                service_name=service_name,
                check_type=check_type,
                status=status,
                response_time=response_time,
                timestamp=datetime.now(),
                details=details,
                score=score
            )
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            
            # Track timeout as failure
            self.performance_history[service_name].append(response_time)
            self.error_history[service_name].append(False)
            
            return HealthCheckResult(
                service_name=service_name,
                check_type=check_type,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=datetime.now(),
                error_message="Health check timeout",
                score=0.0
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Track error as failure
            self.performance_history[service_name].append(response_time)
            self.error_history[service_name].append(False)
            
            return HealthCheckResult(
                service_name=service_name,
                check_type=check_type,
                status=HealthStatus.UNHEALTHY,
                response_time=response_time,
                timestamp=datetime.now(),
                error_message=str(e),
                score=0.0
            )
    
    def _calculate_score(self, response_time: float, is_healthy: bool) -> float:
        """Calculate health score based on response time and health status."""
        if not is_healthy:
            return 0.0
        
        # Score based on response time (0-100)
        if response_time <= 1.0:
            return 100.0
        elif response_time <= 5.0:
            return 100.0 - (response_time - 1.0) * 20.0
        else:
            return max(0.0, 20.0 - (response_time - 5.0) * 4.0)
    
    async def _update_service_health(self, service_name: str):
        """Update overall service health based on check results."""
        service_health = self.services[service_name]
        
        # Collect all check results
        checks = []
        if service_health.shallow_check:
            checks.append(service_health.shallow_check)
        if service_health.deep_check:
            checks.append(service_health.deep_check)
        if service_health.synthetic_check:
            checks.append(service_health.synthetic_check)
        if service_health.passive_check:
            checks.append(service_health.passive_check)
        
        if not checks:
            return
        
        # Calculate overall status
        statuses = [check.status for check in checks]
        
        if any(status == HealthStatus.CRITICAL for status in statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.UNHEALTHY for status in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            overall_status = HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # Calculate overall score
        scores = [check.score for check in checks]
        overall_score = statistics.mean(scores) if scores else 0.0
        
        # Update performance metrics
        if self.performance_history[service_name]:
            service_health.average_response_time = statistics.mean(
                self.performance_history[service_name]
            )
        
        if self.error_history[service_name]:
            successful_checks = sum(self.error_history[service_name])
            total_checks = len(self.error_history[service_name])
            service_health.error_rate = 1.0 - (successful_checks / total_checks)
            service_health.availability = successful_checks / total_checks
        
        # Update failure tracking
        previous_status = service_health.overall_status
        
        if overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            if previous_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                # New failure
                service_health.last_failure_time = datetime.now()
                service_health.consecutive_failures = 1
            else:
                # Continuing failure
                service_health.consecutive_failures += 1
        else:
            if previous_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                # Recovery
                service_health.recovery_time = datetime.now()
                service_health.consecutive_failures = 0
                
                # Calculate MTTR
                if service_health.last_failure_time:
                    recovery_duration = (
                        service_health.recovery_time - service_health.last_failure_time
                    ).total_seconds()
                    
                    # Simple MTTR calculation (could be more sophisticated)
                    service_health.mttr = recovery_duration
        
        # Update service health
        service_health.overall_status = overall_status
        service_health.overall_score = overall_score
        service_health.last_updated = datetime.now()
        
        # Handle status changes
        if overall_status != previous_status:
            await self._handle_status_change(service_name, previous_status, overall_status)
        
        # Persist health data
        if self.redis_client:
            await self._persist_health_data(service_name, service_health)
    
    async def _handle_status_change(
        self, 
        service_name: str, 
        previous_status: HealthStatus, 
        new_status: HealthStatus
    ):
        """Handle service status changes."""
        logger.info(f"Service {service_name} status changed: {previous_status.value} -> {new_status.value}")
        
        # Send event notification
        if self.event_bus:
            await self.event_bus.publish(Event(
                type="service_health_changed",
                data={
                    'service_name': service_name,
                    'previous_status': previous_status.value,
                    'new_status': new_status.value,
                    'timestamp': datetime.now().isoformat()
                }
            ))
        
        # Handle alerts
        service_health = self.services[service_name]
        
        if new_status == HealthStatus.DEGRADED and self.config.alert_on_degraded:
            await self._trigger_alert(service_name, "Service degraded", new_status)
        
        elif new_status == HealthStatus.UNHEALTHY and self.config.alert_on_unhealthy:
            await self._trigger_alert(service_name, "Service unhealthy", new_status)
        
        elif new_status == HealthStatus.CRITICAL and self.config.alert_on_critical:
            await self._trigger_alert(service_name, "Service critical", new_status)
        
        # Handle recovery
        if (new_status == HealthStatus.HEALTHY and 
            previous_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL] and
            self.config.auto_recovery):
            
            await self._initiate_recovery(service_name)
    
    async def _trigger_alert(self, service_name: str, message: str, status: HealthStatus):
        """Trigger health alert."""
        service_health = self.services[service_name]
        
        alert_id = f"health_{service_name}_{status.value}_{int(time.time())}"
        
        # Add to active alerts
        service_health.active_alerts.append(alert_id)
        
        # Send alert event
        if self.event_bus:
            await self.event_bus.publish(Event(
                type="health_alert",
                data={
                    'alert_id': alert_id,
                    'service_name': service_name,
                    'message': message,
                    'status': status.value,
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'critical' if status == HealthStatus.CRITICAL else 'warning'
                }
            ))
        
        logger.warning(f"Health alert triggered for {service_name}: {message}")
    
    async def _initiate_recovery(self, service_name: str):
        """Initiate recovery process for a service."""
        async with self.recovery_locks[service_name]:
            service_health = self.services[service_name]
            
            # Check if recovery is needed
            if service_health.overall_status == HealthStatus.HEALTHY:
                return
            
            # Increment recovery attempts
            self.recovery_attempts[service_name] += 1
            
            logger.info(f"Initiating recovery for {service_name} (attempt {self.recovery_attempts[service_name]})")
            
            # Send recovery event
            if self.event_bus:
                await self.event_bus.publish(Event(
                    type="service_recovery_initiated",
                    data={
                        'service_name': service_name,
                        'attempt': self.recovery_attempts[service_name],
                        'timestamp': datetime.now().isoformat()
                    }
                ))
            
            # Wait for recovery delay
            await asyncio.sleep(self.config.recovery_delay)
            
            # Reset recovery attempts on successful recovery
            if service_health.overall_status == HealthStatus.HEALTHY:
                self.recovery_attempts[service_name] = 0
    
    async def _aggregation_loop(self):
        """Background task for health data aggregation."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._aggregate_health_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health aggregation error: {e}")
    
    async def _aggregate_health_data(self):
        """Aggregate health data across all services."""
        total_services = len(self.services)
        healthy_services = sum(
            1 for service in self.services.values() 
            if service.overall_status == HealthStatus.HEALTHY
        )
        
        degraded_services = sum(
            1 for service in self.services.values() 
            if service.overall_status == HealthStatus.DEGRADED
        )
        
        unhealthy_services = sum(
            1 for service in self.services.values() 
            if service.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        )
        
        # Calculate system health score
        if total_services > 0:
            system_health_score = sum(
                service.overall_score for service in self.services.values()
            ) / total_services
        else:
            system_health_score = 0.0
        
        # Send aggregated health event
        if self.event_bus:
            await self.event_bus.publish(Event(
                type="system_health_aggregated",
                data={
                    'total_services': total_services,
                    'healthy_services': healthy_services,
                    'degraded_services': degraded_services,
                    'unhealthy_services': unhealthy_services,
                    'system_health_score': system_health_score,
                    'timestamp': datetime.now().isoformat()
                }
            ))
    
    async def _persist_health_data(self, service_name: str, service_health: ServiceHealth):
        """Persist health data to Redis."""
        try:
            health_data = {
                'service_name': service_name,
                'overall_status': service_health.overall_status.value,
                'overall_score': service_health.overall_score,
                'last_updated': service_health.last_updated.isoformat(),
                'average_response_time': service_health.average_response_time,
                'error_rate': service_health.error_rate,
                'availability': service_health.availability,
                'consecutive_failures': service_health.consecutive_failures,
                'active_alerts': service_health.active_alerts
            }
            
            await self.redis_client.hset(
                f"health:{service_name}",
                mapping=health_data
            )
            
            await self.redis_client.expire(f"health:{service_name}", 3600)  # 1 hour
            
        except Exception as e:
            logger.error(f"Failed to persist health data for {service_name}: {e}")
    
    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health information for a specific service."""
        return self.services.get(service_name)
    
    def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health information for all services."""
        return self.services.copy()
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health."""
        total_services = len(self.services)
        
        if total_services == 0:
            return {
                'total_services': 0,
                'healthy_services': 0,
                'degraded_services': 0,
                'unhealthy_services': 0,
                'system_health_score': 0.0,
                'overall_status': HealthStatus.UNKNOWN.value
            }
        
        healthy_services = sum(
            1 for service in self.services.values() 
            if service.overall_status == HealthStatus.HEALTHY
        )
        
        degraded_services = sum(
            1 for service in self.services.values() 
            if service.overall_status == HealthStatus.DEGRADED
        )
        
        unhealthy_services = sum(
            1 for service in self.services.values() 
            if service.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        )
        
        system_health_score = sum(
            service.overall_score for service in self.services.values()
        ) / total_services
        
        # Determine overall system status
        if unhealthy_services > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_services > 0:
            overall_status = HealthStatus.DEGRADED
        elif healthy_services == total_services:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            'total_services': total_services,
            'healthy_services': healthy_services,
            'degraded_services': degraded_services,
            'unhealthy_services': unhealthy_services,
            'system_health_score': system_health_score,
            'overall_status': overall_status.value,
            'last_updated': datetime.now().isoformat()
        }
    
    async def force_health_check(self, service_name: str):
        """Force immediate health check for a service."""
        if service_name in self.services:
            await self._perform_health_checks(service_name)
            logger.info(f"Forced health check completed for {service_name}")
        else:
            logger.warning(f"Service {service_name} not registered for health monitoring")
    
    async def clear_alerts(self, service_name: str):
        """Clear all active alerts for a service."""
        if service_name in self.services:
            self.services[service_name].active_alerts = []
            logger.info(f"Cleared alerts for {service_name}")
        else:
            logger.warning(f"Service {service_name} not registered for health monitoring")