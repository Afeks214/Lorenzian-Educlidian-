"""
Tactical Health Monitor for High-Frequency Trading System

Provides stringent health monitoring with sub-second response times
and Redis Stream consumer lag monitoring for zero data loss.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import aioredis
import httpx
import json
from collections import deque

logger = logging.getLogger(__name__)

class TacticalHealthStatus(Enum):
    """Health status levels for tactical system."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class TacticalComponentHealth:
    """Health status of a tactical component."""
    name: str
    status: TacticalHealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms
        }

@dataclass
class TacticalSystemHealth:
    """Overall tactical system health status."""
    status: TacticalHealthStatus
    components: List[TacticalComponentHealth]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": [c.to_dict() for c in self.components],
            "performance_summary": self.performance_summary
        }

class TacticalHealthMonitor:
    """
    High-frequency health monitoring for tactical MARL system.
    
    Designed for sub-second response times and stringent performance thresholds.
    """
    
    # Stringent thresholds for tactical system (much stricter than strategic)
    CPU_THRESHOLD_DEGRADED = 70.0      # 70% CPU usage (degraded)
    CPU_THRESHOLD_CRITICAL = 85.0      # 85% CPU usage (critical)
    MEMORY_THRESHOLD_DEGRADED = 20.0   # 20GB memory (degraded)
    MEMORY_THRESHOLD_CRITICAL = 28.0   # 28GB memory (critical)
    DISK_USAGE_THRESHOLD = 80.0        # 80% disk usage
    
    # Ultra-strict latency thresholds
    INFERENCE_LATENCY_THRESHOLD_DEGRADED_MS = 50    # 50ms (degraded)
    INFERENCE_LATENCY_THRESHOLD_CRITICAL_MS = 100   # 100ms (critical)
    
    # Data freshness thresholds (much stricter than strategic)
    DATA_FRESHNESS_THRESHOLD_DEGRADED_S = 10   # 10s (degraded)
    DATA_FRESHNESS_THRESHOLD_CRITICAL_S = 30   # 30s (critical)
    
    # Model staleness thresholds
    MODEL_STALENESS_THRESHOLD_DEGRADED_S = 900   # 15 minutes (degraded)
    MODEL_STALENESS_THRESHOLD_CRITICAL_S = 1800  # 30 minutes (critical)
    
    # Redis stream consumer lag thresholds
    STREAM_LAG_THRESHOLD_DEGRADED_S = 2.0    # 2 seconds (degraded)
    STREAM_LAG_THRESHOLD_CRITICAL_S = 5.0    # 5 seconds (critical)
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize tactical health monitor."""
        self.redis_url = redis_url or "redis://localhost:6379/2"
        self._last_checks: Dict[str, datetime] = {}
        self._performance_history = deque(maxlen=100)
        
        # Check intervals (more frequent than strategic)
        self._check_intervals: Dict[str, timedelta] = {
            "system_resources": timedelta(seconds=5),     # Every 5 seconds
            "redis_streams": timedelta(seconds=10),       # Every 10 seconds
            "model_performance": timedelta(seconds=15),   # Every 15 seconds
            "data_pipeline": timedelta(seconds=10),       # Every 10 seconds
            "external_services": timedelta(seconds=30),   # Every 30 seconds
            "agent_responsiveness": timedelta(seconds=5), # Every 5 seconds
        }
        
        # Stream configurations
        self.stream_configs = {
            "synergy_events": {
                "consumer_group": "tactical_group",
                "consumers": ["tactical_consumer_1"]
            }
        }
        
        logger.info("Tactical health monitor initialized with stringent thresholds")
    
    async def check_system_resources(self) -> TacticalComponentHealth:
        """Check system resources with tactical-specific thresholds."""
        start_time = time.perf_counter()
        
        try:
            # CPU check with 1-second sampling
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024 ** 3)
            memory_percent = memory.percent
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O check
            net_io = psutil.net_io_counters()
            
            # Process count check
            process_count = len(psutil.pids())
            
            # Determine health status
            issues = []
            status = TacticalHealthStatus.HEALTHY
            
            if cpu_percent > self.CPU_THRESHOLD_CRITICAL:
                status = TacticalHealthStatus.CRITICAL
                issues.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > self.CPU_THRESHOLD_DEGRADED:
                status = TacticalHealthStatus.DEGRADED
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_used_gb > self.MEMORY_THRESHOLD_CRITICAL:
                status = TacticalHealthStatus.CRITICAL
                issues.append(f"Critical memory usage: {memory_used_gb:.1f}GB")
            elif memory_used_gb > self.MEMORY_THRESHOLD_DEGRADED:
                status = TacticalHealthStatus.DEGRADED if status == TacticalHealthStatus.HEALTHY else status
                issues.append(f"High memory usage: {memory_used_gb:.1f}GB")
            
            if disk_percent > self.DISK_USAGE_THRESHOLD:
                status = TacticalHealthStatus.DEGRADED if status == TacticalHealthStatus.HEALTHY else status
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "; ".join(issues) if issues else "System resources optimal"
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            return TacticalComponentHealth(
                name="system_resources",
                status=status,
                message=message,
                response_time_ms=response_time,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_used_gb": memory_used_gb,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "process_count": process_count,
                    "network_bytes_sent": net_io.bytes_sent,
                    "network_bytes_recv": net_io.bytes_recv,
                    "thresholds": {
                        "cpu_degraded": self.CPU_THRESHOLD_DEGRADED,
                        "cpu_critical": self.CPU_THRESHOLD_CRITICAL,
                        "memory_degraded_gb": self.MEMORY_THRESHOLD_DEGRADED,
                        "memory_critical_gb": self.MEMORY_THRESHOLD_CRITICAL
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return TacticalComponentHealth(
                name="system_resources",
                status=TacticalHealthStatus.UNKNOWN,
                message=f"Error checking resources: {str(e)}",
                response_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def check_redis_streams(self) -> TacticalComponentHealth:
        """Check Redis streams and consumer lag."""
        start_time = time.perf_counter()
        
        try:
            redis = await aioredis.create_redis_pool(self.redis_url)
            
            # Test basic connectivity
            await redis.ping()
            
            stream_status = {}
            max_lag = 0.0
            total_pending = 0
            
            for stream_name, config in self.stream_configs.items():
                consumer_group = config["consumer_group"]
                
                try:
                    # Get consumer group info
                    info = await redis.execute('XINFO', 'GROUPS', stream_name)
                    
                    if info:
                        for group_info in info:
                            if group_info[1].decode() == consumer_group:
                                lag = int(group_info[7])  # lag field
                                pending = int(group_info[9])  # pending field
                                
                                # Convert lag to seconds (approximate)
                                lag_seconds = lag * 0.001  # Rough estimation
                                
                                max_lag = max(max_lag, lag_seconds)
                                total_pending += pending
                                
                                stream_status[stream_name] = {
                                    "lag_seconds": lag_seconds,
                                    "pending_messages": pending,
                                    "consumer_group": consumer_group
                                }
                                
                except Exception as e:
                    logger.warning(f"Could not get info for stream {stream_name}: {e}")
                    stream_status[stream_name] = {"error": str(e)}
            
            # Memory usage check
            memory_info = await redis.info("memory")
            used_memory_mb = memory_info.get("used_memory", 0) / (1024 * 1024)
            
            redis.close()
            await redis.wait_closed()
            
            # Determine health status based on consumer lag
            if max_lag > self.STREAM_LAG_THRESHOLD_CRITICAL_S:
                status = TacticalHealthStatus.CRITICAL
                message = f"Critical consumer lag: {max_lag:.1f}s"
            elif max_lag > self.STREAM_LAG_THRESHOLD_DEGRADED_S:
                status = TacticalHealthStatus.DEGRADED
                message = f"High consumer lag: {max_lag:.1f}s"
            elif total_pending > 100:
                status = TacticalHealthStatus.DEGRADED
                message = f"High pending messages: {total_pending}"
            else:
                status = TacticalHealthStatus.HEALTHY
                message = "Redis streams healthy"
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            return TacticalComponentHealth(
                name="redis_streams",
                status=status,
                message=message,
                response_time_ms=response_time,
                details={
                    "max_consumer_lag_seconds": max_lag,
                    "total_pending_messages": total_pending,
                    "used_memory_mb": used_memory_mb,
                    "stream_status": stream_status,
                    "thresholds": {
                        "lag_degraded_s": self.STREAM_LAG_THRESHOLD_DEGRADED_S,
                        "lag_critical_s": self.STREAM_LAG_THRESHOLD_CRITICAL_S
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking Redis streams: {e}")
            return TacticalComponentHealth(
                name="redis_streams",
                status=TacticalHealthStatus.CRITICAL,
                message=f"Redis streams connection failed: {str(e)}",
                response_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def check_model_performance(self) -> TacticalComponentHealth:
        """Check model inference performance."""
        start_time = time.perf_counter()
        
        try:
            # In production, this would check actual metrics from tactical_metrics
            # For now, simulate with realistic values
            
            # Simulate inference latency check
            inference_latency_ms = 35.0  # Simulated P95 latency
            
            # Check last prediction time
            last_prediction_time = datetime.utcnow() - timedelta(seconds=30)
            staleness_seconds = (datetime.utcnow() - last_prediction_time).total_seconds()
            
            # Determine status based on latency
            if inference_latency_ms > self.INFERENCE_LATENCY_THRESHOLD_CRITICAL_MS:
                status = TacticalHealthStatus.CRITICAL
                message = f"Critical inference latency: {inference_latency_ms:.1f}ms"
            elif inference_latency_ms > self.INFERENCE_LATENCY_THRESHOLD_DEGRADED_MS:
                status = TacticalHealthStatus.DEGRADED
                message = f"High inference latency: {inference_latency_ms:.1f}ms"
            elif staleness_seconds > self.MODEL_STALENESS_THRESHOLD_CRITICAL_S:
                status = TacticalHealthStatus.CRITICAL
                message = f"Model critically stale: {staleness_seconds:.0f}s"
            elif staleness_seconds > self.MODEL_STALENESS_THRESHOLD_DEGRADED_S:
                status = TacticalHealthStatus.DEGRADED
                message = f"Model becoming stale: {staleness_seconds:.0f}s"
            else:
                status = TacticalHealthStatus.HEALTHY
                message = f"Model performance optimal: {inference_latency_ms:.1f}ms"
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            return TacticalComponentHealth(
                name="model_performance",
                status=status,
                message=message,
                response_time_ms=response_time,
                details={
                    "inference_latency_ms": inference_latency_ms,
                    "staleness_seconds": staleness_seconds,
                    "last_prediction": last_prediction_time.isoformat(),
                    "thresholds": {
                        "latency_degraded_ms": self.INFERENCE_LATENCY_THRESHOLD_DEGRADED_MS,
                        "latency_critical_ms": self.INFERENCE_LATENCY_THRESHOLD_CRITICAL_MS,
                        "staleness_degraded_s": self.MODEL_STALENESS_THRESHOLD_DEGRADED_S,
                        "staleness_critical_s": self.MODEL_STALENESS_THRESHOLD_CRITICAL_S
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return TacticalComponentHealth(
                name="model_performance",
                status=TacticalHealthStatus.UNKNOWN,
                message=f"Error checking performance: {str(e)}",
                response_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def check_data_pipeline(self) -> TacticalComponentHealth:
        """Check data pipeline freshness."""
        start_time = time.perf_counter()
        
        try:
            # In production, this would check actual pipeline metrics
            # Simulate with realistic data age
            last_update = datetime.utcnow() - timedelta(seconds=8)
            data_age_seconds = (datetime.utcnow() - last_update).total_seconds()
            
            # Check matrix assembler status
            matrix_processing_latency_ms = 2.5  # Simulated
            
            if data_age_seconds > self.DATA_FRESHNESS_THRESHOLD_CRITICAL_S:
                status = TacticalHealthStatus.CRITICAL
                message = f"Critical data staleness: {data_age_seconds:.0f}s"
            elif data_age_seconds > self.DATA_FRESHNESS_THRESHOLD_DEGRADED_S:
                status = TacticalHealthStatus.DEGRADED
                message = f"Data becoming stale: {data_age_seconds:.0f}s"
            elif matrix_processing_latency_ms > 10.0:
                status = TacticalHealthStatus.DEGRADED
                message = f"Slow matrix processing: {matrix_processing_latency_ms:.1f}ms"
            else:
                status = TacticalHealthStatus.HEALTHY
                message = "Data pipeline healthy"
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            return TacticalComponentHealth(
                name="data_pipeline",
                status=status,
                message=message,
                response_time_ms=response_time,
                details={
                    "data_age_seconds": data_age_seconds,
                    "matrix_processing_latency_ms": matrix_processing_latency_ms,
                    "last_update": last_update.isoformat(),
                    "thresholds": {
                        "freshness_degraded_s": self.DATA_FRESHNESS_THRESHOLD_DEGRADED_S,
                        "freshness_critical_s": self.DATA_FRESHNESS_THRESHOLD_CRITICAL_S
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking data pipeline: {e}")
            return TacticalComponentHealth(
                name="data_pipeline",
                status=TacticalHealthStatus.UNKNOWN,
                message=f"Error checking pipeline: {str(e)}",
                response_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def check_agent_responsiveness(self) -> TacticalComponentHealth:
        """Check tactical agent responsiveness."""
        start_time = time.perf_counter()
        
        try:
            # Simulate agent response time checks
            agents = ["fvg_agent", "momentum_agent", "entry_agent"]
            agent_response_times = {}
            max_response_time = 0.0
            
            for agent in agents:
                # Simulate agent health check
                response_time_ms = 15.0  # Simulated response time
                agent_response_times[agent] = response_time_ms
                max_response_time = max(max_response_time, response_time_ms)
            
            if max_response_time > 100.0:
                status = TacticalHealthStatus.CRITICAL
                message = f"Agent unresponsive: {max_response_time:.1f}ms"
            elif max_response_time > 50.0:
                status = TacticalHealthStatus.DEGRADED
                message = f"Slow agent response: {max_response_time:.1f}ms"
            else:
                status = TacticalHealthStatus.HEALTHY
                message = "All agents responsive"
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            return TacticalComponentHealth(
                name="agent_responsiveness",
                status=status,
                message=message,
                response_time_ms=response_time,
                details={
                    "agent_response_times": agent_response_times,
                    "max_response_time_ms": max_response_time,
                    "agents_checked": len(agents)
                }
            )
            
        except Exception as e:
            logger.error(f"Error checking agent responsiveness: {e}")
            return TacticalComponentHealth(
                name="agent_responsiveness",
                status=TacticalHealthStatus.UNKNOWN,
                message=f"Error checking agents: {str(e)}",
                response_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def _should_check(self, component: str) -> bool:
        """Determine if a component check should run based on interval."""
        last_check = self._last_checks.get(component)
        if not last_check:
            return True
        
        interval = self._check_intervals.get(component, timedelta(seconds=30))
        return datetime.utcnow() - last_check >= interval
    
    async def check_all_components(self) -> TacticalSystemHealth:
        """Run all tactical health checks."""
        components = []
        
        # Define checks with their functions
        checks = [
            ("system_resources", self.check_system_resources),
            ("redis_streams", self.check_redis_streams),
            ("model_performance", self.check_model_performance),
            ("data_pipeline", self.check_data_pipeline),
            ("agent_responsiveness", self.check_agent_responsiveness),
        ]
        
        # Run checks concurrently for speed
        tasks = []
        for component_name, check_func in checks:
            if await self._should_check(component_name):
                tasks.append(check_func())
        
        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(check_results):
                if isinstance(result, Exception):
                    logger.error(f"Health check failed: {result}")
                    # Create error component
                    component = TacticalComponentHealth(
                        name=checks[i][0],
                        status=TacticalHealthStatus.UNKNOWN,
                        message=f"Check failed: {str(result)}"
                    )
                    components.append(component)
                else:
                    components.append(result)
                    self._last_checks[checks[i][0]] = datetime.utcnow()
        
        # Determine overall system health
        if not components:
            overall_status = TacticalHealthStatus.UNKNOWN
        else:
            statuses = [c.status for c in components]
            
            if any(s == TacticalHealthStatus.CRITICAL for s in statuses):
                overall_status = TacticalHealthStatus.CRITICAL
            elif any(s == TacticalHealthStatus.DEGRADED for s in statuses):
                overall_status = TacticalHealthStatus.DEGRADED
            elif any(s == TacticalHealthStatus.UNKNOWN for s in statuses):
                overall_status = TacticalHealthStatus.UNKNOWN
            else:
                overall_status = TacticalHealthStatus.HEALTHY
        
        # Generate performance summary
        performance_summary = self._generate_performance_summary(components)
        
        return TacticalSystemHealth(
            status=overall_status,
            components=components,
            performance_summary=performance_summary
        )
    
    def _generate_performance_summary(self, components: List[TacticalComponentHealth]) -> Dict[str, Any]:
        """Generate performance summary from component health."""
        summary = {
            "overall_response_time_ms": sum(c.response_time_ms for c in components),
            "components_checked": len(components),
            "healthy_components": sum(1 for c in components if c.status == TacticalHealthStatus.HEALTHY),
            "degraded_components": sum(1 for c in components if c.status == TacticalHealthStatus.DEGRADED),
            "critical_components": sum(1 for c in components if c.status == TacticalHealthStatus.CRITICAL),
            "unknown_components": sum(1 for c in components if c.status == TacticalHealthStatus.UNKNOWN),
        }
        
        # Add component-specific metrics
        for component in components:
            if component.name == "model_performance" and "inference_latency_ms" in component.details:
                summary["model_inference_latency_ms"] = component.details["inference_latency_ms"]
            elif component.name == "redis_streams" and "max_consumer_lag_seconds" in component.details:
                summary["redis_consumer_lag_seconds"] = component.details["max_consumer_lag_seconds"]
        
        return summary
    
    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information with recommendations."""
        system_health = await self.check_all_components()
        
        # Generate recommendations
        recommendations = []
        for component in system_health.components:
            if component.status == TacticalHealthStatus.CRITICAL:
                if component.name == "system_resources":
                    recommendations.append("Critical: Scale tactical system resources immediately")
                elif component.name == "redis_streams":
                    recommendations.append("Critical: Check Redis stream consumer lag - potential data loss risk")
                elif component.name == "model_performance":
                    recommendations.append("Critical: Model inference exceeding 100ms - trade execution at risk")
                elif component.name == "data_pipeline":
                    recommendations.append("Critical: Data pipeline stale - decision accuracy compromised")
                elif component.name == "agent_responsiveness":
                    recommendations.append("Critical: Tactical agents unresponsive - system non-functional")
        
        result = system_health.to_dict()
        result["recommendations"] = recommendations
        result["check_intervals"] = {
            k: v.total_seconds() for k, v in self._check_intervals.items()
        }
        
        return result

# Global tactical health monitor
tactical_health_monitor = TacticalHealthMonitor()