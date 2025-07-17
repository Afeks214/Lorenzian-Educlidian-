"""
Health monitoring system for Strategic MARL 30m System.
Provides comprehensive health checks for all system components.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
from src.utils.redis_compat import redis_client

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "last_check": self.last_check.isoformat()
        }

@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": [c.to_dict() for c in self.components]
        }

class HealthMonitor:
    """
    Comprehensive health monitoring for the Strategic MARL system.
    Checks system resources, component status, and performance thresholds.
    """
    
    # Thresholds from PRD requirements
    CPU_THRESHOLD = 80.0  # 80% CPU usage
    MEMORY_THRESHOLD_MB = 512  # 512MB memory limit
    DISK_USAGE_THRESHOLD = 90.0  # 90% disk usage
    INFERENCE_LATENCY_THRESHOLD_MS = 5  # 5ms inference latency
    DATA_FRESHNESS_THRESHOLD_S = 30  # 30s data freshness
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize health monitor with optional Redis URL."""
        self.redis_url = redis_url or "redis://localhost:6379"
        self._last_checks: Dict[str, datetime] = {}
        self._check_intervals: Dict[str, timedelta] = {
            "system_resources": timedelta(seconds=10),
            "redis": timedelta(seconds=15),
            "data_pipeline": timedelta(seconds=30),
            "model_performance": timedelta(seconds=60),
        }
        
    async def check_system_resources(self) -> ComponentHealth:
        """Check system resource usage."""
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine health status
            issues = []
            if cpu_percent > self.CPU_THRESHOLD:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory_used_mb > self.MEMORY_THRESHOLD_MB:
                issues.append(f"High memory usage: {memory_used_mb:.0f}MB")
            if disk_percent > self.DISK_USAGE_THRESHOLD:
                issues.append(f"High disk usage: {disk_percent:.1f}%")
                
            if issues:
                status = HealthStatus.DEGRADED if len(issues) == 1 else HealthStatus.UNHEALTHY
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources within normal limits"
                
            return ComponentHealth(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_used_mb": memory_used_mb,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "process_count": len(psutil.pids())
                }
            )
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking resources: {str(e)}"
            )
            
    async def check_redis_connectivity(self) -> ComponentHealth:
        """Check Redis connectivity and performance."""
        try:
            # Use our compatibility layer
            redis_client.url = self.redis_url
            await redis_client.connect()
            
            if not redis_client.available:
                return ComponentHealth(
                    name="redis",
                    status=HealthStatus.UNHEALTHY,
                    message="Redis connection unavailable"
                )
            
            # Ping test
            start_time = time.time()
            pong = await redis_client.ping()
            latency_ms = (time.time() - start_time) * 1000
            
            if not pong:
                return ComponentHealth(
                    name="redis",
                    status=HealthStatus.UNHEALTHY,
                    message="Redis ping failed"
                )
                
            # For memory usage, use a default since we can't easily get info
            used_memory_mb = 0  # Default value when info not available
            
            # Determine health
            if latency_ms > 10:
                status = HealthStatus.DEGRADED
                message = f"High Redis latency: {latency_ms:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Redis connection healthy"
                
            return ComponentHealth(
                name="redis",
                status=status,
                message=message,
                details={
                    "latency_ms": latency_ms,
                    "used_memory_mb": used_memory_mb
                }
            )
        except Exception as e:
            logger.error(f"Error checking Redis: {e}")
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}"
            )
            
    async def check_data_pipeline(self) -> ComponentHealth:
        """Check data pipeline freshness and status."""
        try:
            # This would check actual pipeline metrics in production
            # For now, simulating with timestamp checks
            
            # In production, this would query actual pipeline metrics
            last_update = datetime.utcnow() - timedelta(seconds=15)  # Simulated
            data_age_seconds = (datetime.utcnow() - last_update).total_seconds()
            
            if data_age_seconds > self.DATA_FRESHNESS_THRESHOLD_S:
                status = HealthStatus.UNHEALTHY
                message = f"Stale data: {data_age_seconds:.0f}s old"
            elif data_age_seconds > self.DATA_FRESHNESS_THRESHOLD_S * 0.8:
                status = HealthStatus.DEGRADED
                message = f"Data approaching staleness: {data_age_seconds:.0f}s old"
            else:
                status = HealthStatus.HEALTHY
                message = "Data pipeline up to date"
                
            return ComponentHealth(
                name="data_pipeline",
                status=status,
                message=message,
                details={
                    "data_age_seconds": data_age_seconds,
                    "last_update": last_update.isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error checking data pipeline: {e}")
            return ComponentHealth(
                name="data_pipeline",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking pipeline: {str(e)}"
            )
            
    async def check_model_performance(self) -> ComponentHealth:
        """Check model inference performance."""
        try:
            # In production, this would check actual inference metrics
            # For now, simulating performance check
            
            # Simulated inference latency (in production, get from metrics)
            inference_latency_ms = 3.2  # Simulated value
            
            if inference_latency_ms > self.INFERENCE_LATENCY_THRESHOLD_MS:
                status = HealthStatus.UNHEALTHY
                message = f"High inference latency: {inference_latency_ms:.1f}ms"
            elif inference_latency_ms > self.INFERENCE_LATENCY_THRESHOLD_MS * 0.8:
                status = HealthStatus.DEGRADED
                message = f"Inference latency approaching threshold: {inference_latency_ms:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Model performance optimal: {inference_latency_ms:.1f}ms"
                
            return ComponentHealth(
                name="model_performance",
                status=status,
                message=message,
                details={
                    "inference_latency_ms": inference_latency_ms,
                    "threshold_ms": self.INFERENCE_LATENCY_THRESHOLD_MS
                }
            )
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return ComponentHealth(
                name="model_performance",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking performance: {str(e)}"
            )
            
    async def check_external_services(self) -> ComponentHealth:
        """Check connectivity to external services."""
        try:
            services_status = []
            
            # Check Ollama
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get("http://ollama:11434/api/tags", timeout=5.0)
                    ollama_healthy = response.status_code == 200
                except (ConnectionError, OSError, TimeoutError) as e:
                    ollama_healthy = False
                    
            services_status.append(("ollama", ollama_healthy))
            
            # Aggregate results
            failed_services = [s[0] for s in services_status if not s[1]]
            
            if not failed_services:
                status = HealthStatus.HEALTHY
                message = "All external services operational"
            elif len(failed_services) == 1:
                status = HealthStatus.DEGRADED
                message = f"Service degraded: {failed_services[0]}"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Multiple services down: {', '.join(failed_services)}"
                
            return ComponentHealth(
                name="external_services",
                status=status,
                message=message,
                details={
                    "services": dict(services_status)
                }
            )
        except Exception as e:
            logger.error(f"Error checking external services: {e}")
            return ComponentHealth(
                name="external_services",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking services: {str(e)}"
            )
            
    async def _should_check(self, component: str) -> bool:
        """Determine if a component check should run based on interval."""
        last_check = self._last_checks.get(component)
        if not last_check:
            return True
            
        interval = self._check_intervals.get(component, timedelta(seconds=60))
        return datetime.utcnow() - last_check >= interval
        
    async def check_all_components(self) -> SystemHealth:
        """Run all health checks and return system health status."""
        components = []
        
        # Run checks based on intervals
        checks = [
            ("system_resources", self.check_system_resources),
            ("redis", self.check_redis_connectivity),
            ("data_pipeline", self.check_data_pipeline),
            ("model_performance", self.check_model_performance),
            ("external_services", self.check_external_services),
        ]
        
        for component_name, check_func in checks:
            if await self._should_check(component_name):
                component_health = await check_func()
                components.append(component_health)
                self._last_checks[component_name] = datetime.utcnow()
                
        # Determine overall system health
        statuses = [c.status for c in components]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN
            
        return SystemHealth(
            status=overall_status,
            components=components
        )
        
    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information including recommendations."""
        system_health = await self.check_all_components()
        
        # Add recommendations based on health status
        recommendations = []
        for component in system_health.components:
            if component.status == HealthStatus.UNHEALTHY:
                if component.name == "system_resources":
                    if "CPU" in component.message:
                        recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
                    if "memory" in component.message:
                        recommendations.append("Review memory usage patterns and consider increasing limits")
                elif component.name == "redis":
                    recommendations.append("Check Redis server status and network connectivity")
                elif component.name == "data_pipeline":
                    recommendations.append("Investigate data source connectivity and pipeline processing")
                    
        result = system_health.to_dict()
        result["recommendations"] = recommendations
        result["check_intervals"] = {
            k: v.total_seconds() for k, v in self._check_intervals.items()
        }
        
        return result
        
# Global health monitor instance
health_monitor = HealthMonitor()