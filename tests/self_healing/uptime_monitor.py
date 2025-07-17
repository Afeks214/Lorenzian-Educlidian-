"""
99.99% Uptime Monitor - Phase 3B Implementation
Agent Epsilon: Self-Healing Production Systems

Advanced uptime monitoring with:
- 99.99% SLA validation and tracking
- Real-time health checks across all components
- Multi-dimensional health scoring
- Automated alerting and escalation
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import aiohttp
import numpy as np
import redis.asyncio as redis
from pathlib import Path
import uuid
import psutil
import websockets

logger = logging.getLogger(__name__)

@dataclass
class UptimeConfig:
    """Uptime monitoring configuration"""
    target_uptime_percentage: float = 99.99
    check_interval_seconds: int = 1
    health_check_timeout_seconds: int = 5
    downtime_alert_threshold_seconds: int = 30
    escalation_levels: List[int] = field(default_factory=lambda: [60, 300, 900])  # 1min, 5min, 15min
    
@dataclass
class HealthCheck:
    """Individual health check definition"""
    name: str
    endpoint: str
    method: str = "GET"
    timeout: int = 5
    expected_status: int = 200
    critical: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class HealthStatus:
    """Health status record"""
    timestamp: float
    component: str
    status: str  # "healthy", "degraded", "unhealthy", "down"
    response_time_ms: float
    error_message: Optional[str] = None
    http_status: Optional[int] = None
    
@dataclass
class UptimeMetrics:
    """Uptime metrics aggregation"""
    total_checks: int
    successful_checks: int
    failed_checks: int
    uptime_percentage: float
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    total_downtime_seconds: float
    incident_count: int
    
class UptimeMonitor:
    """
    99.99% Uptime Monitor
    
    Monitors system uptime with 99.99% SLA validation,
    real-time health checks, and automated alerting.
    """
    
    def __init__(self, config: UptimeConfig):
        self.config = config
        self.monitor_id = str(uuid.uuid4())
        self.is_monitoring = False
        
        # Health checks configuration
        self.health_checks: List[HealthCheck] = []
        self.health_history: List[HealthStatus] = []
        
        # Uptime tracking
        self.monitoring_start_time = None
        self.total_downtime_seconds = 0
        self.current_incident_start = None
        self.incident_count = 0
        
        # Real-time metrics
        self.response_times = []
        self.error_counts = {}
        self.availability_samples = []
        
        # Redis for coordination
        self.redis_client = None
        self.redis_url = "redis://localhost:6379/4"
        
        # Alert management
        self.active_alerts = {}
        self.escalation_state = {}
        
    async def initialize(self):
        """Initialize uptime monitoring"""
        logger.info(f"🏥 Initializing 99.99% Uptime Monitor - ID: {self.monitor_id}")
        
        # Connect to Redis
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Start monitoring tasks
        self.monitoring_start_time = time.time()
        self.is_monitoring = True
        
        # Start monitoring loops
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._metrics_aggregation_loop())
        asyncio.create_task(self._uptime_calculation_loop())
        
        logger.info("✅ Uptime monitor initialized successfully")
        
    def _setup_default_health_checks(self):
        """Setup default health checks for all system components"""
        
        # Core API endpoints
        self.health_checks.extend([
            HealthCheck(
                name="tactical_api",
                endpoint="http://localhost:8080/health/tactical",
                critical=True
            ),
            HealthCheck(
                name="strategic_api", 
                endpoint="http://localhost:8080/health/strategic",
                critical=True
            ),
            HealthCheck(
                name="risk_api",
                endpoint="http://localhost:8080/health/risk",
                critical=True
            ),
            HealthCheck(
                name="execution_api",
                endpoint="http://localhost:8080/health/execution",
                critical=True
            ),
            HealthCheck(
                name="xai_api",
                endpoint="http://localhost:8080/health/xai",
                critical=True
            )
        ])
        
        # Database health checks
        self.health_checks.extend([
            HealthCheck(
                name="redis_health",
                endpoint="http://localhost:8080/health/redis",
                critical=True
            ),
            HealthCheck(
                name="database_health",
                endpoint="http://localhost:8080/health/database",
                critical=True
            )
        ])
        
        # Infrastructure health checks
        self.health_checks.extend([
            HealthCheck(
                name="load_balancer",
                endpoint="http://localhost:8080/health/loadbalancer",
                critical=True
            ),
            HealthCheck(
                name="monitoring_system",
                endpoint="http://localhost:8080/health/monitoring",
                critical=False
            )
        ])
        
        logger.info(f"📋 Configured {len(self.health_checks)} health checks")
        
    async def _health_check_loop(self):
        """Main health check loop"""
        while self.is_monitoring:
            try:
                start_time = time.time()
                
                # Execute all health checks in parallel
                tasks = []
                for health_check in self.health_checks:
                    task = asyncio.create_task(self._execute_health_check(health_check))
                    tasks.append(task)
                    
                # Wait for all health checks to complete
                health_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(health_results):
                    if isinstance(result, Exception):
                        logger.error(f"Health check {self.health_checks[i].name} failed: {result}")
                        
                        # Create failed health status
                        health_status = HealthStatus(
                            timestamp=time.time(),
                            component=self.health_checks[i].name,
                            status="down",
                            response_time_ms=self.config.health_check_timeout_seconds * 1000,
                            error_message=str(result)
                        )
                        
                        self.health_history.append(health_status)
                        
                    elif isinstance(result, HealthStatus):
                        self.health_history.append(result)
                        
                # Keep only last 10000 health records
                if len(self.health_history) > 10000:
                    self.health_history = self.health_history[-10000:]
                    
                # Calculate system health
                await self._calculate_system_health()
                
                # Check for incidents
                await self._check_for_incidents()
                
                # Sleep until next check
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.check_interval_seconds - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.config.check_interval_seconds)
                
    async def _execute_health_check(self, health_check: HealthCheck) -> HealthStatus:
        """Execute a single health check"""
        start_time = time.perf_counter()
        
        try:
            timeout = aiohttp.ClientTimeout(total=health_check.timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    health_check.method,
                    health_check.endpoint,
                    headers=health_check.headers
                ) as response:
                    
                    response_time_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Determine status
                    if response.status == health_check.expected_status:
                        status = "healthy"
                    elif 200 <= response.status < 400:
                        status = "degraded"
                    else:
                        status = "unhealthy"
                        
                    return HealthStatus(
                        timestamp=time.time(),
                        component=health_check.name,
                        status=status,
                        response_time_ms=response_time_ms,
                        http_status=response.status
                    )
                    
        except asyncio.TimeoutError:
            return HealthStatus(
                timestamp=time.time(),
                component=health_check.name,
                status="down",
                response_time_ms=health_check.timeout * 1000,
                error_message="Timeout"
            )
            
        except Exception as e:
            return HealthStatus(
                timestamp=time.time(),
                component=health_check.name,
                status="down",
                response_time_ms=health_check.timeout * 1000,
                error_message=str(e)
            )
            
    async def _calculate_system_health(self):
        """Calculate overall system health"""
        if not self.health_history:
            return
            
        # Get recent health checks (last 60 seconds)
        recent_time = time.time() - 60
        recent_checks = [h for h in self.health_history if h.timestamp > recent_time]
        
        if not recent_checks:
            return
            
        # Calculate health by component
        component_health = {}
        for check in recent_checks:
            if check.component not in component_health:
                component_health[check.component] = []
            component_health[check.component].append(check)
            
        # Calculate system-wide health score
        total_weight = 0
        healthy_weight = 0
        
        for component, checks in component_health.items():
            # Find corresponding health check config
            health_check_config = next((hc for hc in self.health_checks if hc.name == component), None)
            
            if not health_check_config:
                continue
                
            # Weight: critical components count more
            weight = 10 if health_check_config.critical else 1
            total_weight += weight
            
            # Calculate component health (last 10 checks)
            recent_component_checks = sorted(checks, key=lambda x: x.timestamp)[-10:]
            healthy_checks = sum(1 for c in recent_component_checks if c.status == "healthy")
            component_health_ratio = healthy_checks / len(recent_component_checks)
            
            healthy_weight += weight * component_health_ratio
            
        # Overall health score
        system_health_score = healthy_weight / total_weight if total_weight > 0 else 0
        
        # Store health sample
        health_sample = {
            "timestamp": time.time(),
            "health_score": system_health_score,
            "component_count": len(component_health),
            "critical_components_healthy": self._count_critical_components_healthy(component_health)
        }
        
        self.availability_samples.append(health_sample)
        
        # Keep only last 1000 samples (about 16 minutes at 1s interval)
        if len(self.availability_samples) > 1000:
            self.availability_samples = self.availability_samples[-1000:]
            
        # Publish health metrics
        await self._publish_health_metrics(health_sample)
        
    def _count_critical_components_healthy(self, component_health: Dict) -> int:
        """Count healthy critical components"""
        healthy_critical = 0
        
        for component, checks in component_health.items():
            health_check_config = next((hc for hc in self.health_checks if hc.name == component), None)
            
            if health_check_config and health_check_config.critical:
                # Check if component is healthy (last check)
                if checks and checks[-1].status == "healthy":
                    healthy_critical += 1
                    
        return healthy_critical
        
    async def _check_for_incidents(self):
        """Check for system incidents and manage alerts"""
        if not self.availability_samples:
            return
            
        current_sample = self.availability_samples[-1]
        current_time = time.time()
        
        # Check for system-wide incident
        if current_sample["health_score"] < 0.9:  # Less than 90% health
            if not self.current_incident_start:
                self.current_incident_start = current_time
                logger.warning("🚨 System incident detected")
                
                # Create alert
                await self._create_alert("system_incident", {
                    "health_score": current_sample["health_score"],
                    "incident_start": self.current_incident_start
                })
                
        else:
            # System is healthy - close incident if active
            if self.current_incident_start:
                incident_duration = current_time - self.current_incident_start
                self.total_downtime_seconds += incident_duration
                self.incident_count += 1
                
                logger.info(f"✅ System incident resolved - Duration: {incident_duration:.2f}s")
                
                # Close alert
                await self._close_alert("system_incident")
                
                self.current_incident_start = None
                
        # Check for individual component incidents
        await self._check_component_incidents()
        
    async def _check_component_incidents(self):
        """Check for individual component incidents"""
        if not self.health_history:
            return
            
        # Group recent checks by component
        recent_time = time.time() - 300  # Last 5 minutes
        recent_checks = [h for h in self.health_history if h.timestamp > recent_time]
        
        component_checks = {}
        for check in recent_checks:
            if check.component not in component_checks:
                component_checks[check.component] = []
            component_checks[check.component].append(check)
            
        # Check each component
        for component, checks in component_checks.items():
            if not checks:
                continue
                
            # Sort by timestamp
            checks.sort(key=lambda x: x.timestamp)
            
            # Check last 5 checks
            recent_component_checks = checks[-5:]
            failing_checks = [c for c in recent_component_checks if c.status in ["unhealthy", "down"]]
            
            # Alert if more than 60% of recent checks are failing
            if len(failing_checks) / len(recent_component_checks) > 0.6:
                alert_key = f"component_incident_{component}"
                
                if alert_key not in self.active_alerts:
                    await self._create_alert(alert_key, {
                        "component": component,
                        "failure_rate": len(failing_checks) / len(recent_component_checks),
                        "recent_checks": len(recent_component_checks)
                    })
                    
            else:
                # Component is healthy - close alert if active
                alert_key = f"component_incident_{component}"
                if alert_key in self.active_alerts:
                    await self._close_alert(alert_key)
                    
    async def _create_alert(self, alert_type: str, details: Dict):
        """Create a new alert"""
        alert = {
            "alert_id": str(uuid.uuid4()),
            "alert_type": alert_type,
            "severity": "critical" if "system" in alert_type else "warning",
            "details": details,
            "created_at": time.time(),
            "escalation_level": 0
        }
        
        self.active_alerts[alert_type] = alert
        
        # Publish alert
        await self.redis_client.publish(
            "uptime_alerts",
            json.dumps(alert)
        )
        
        logger.error(f"🚨 ALERT: {alert_type} - {details}")
        
    async def _close_alert(self, alert_type: str):
        """Close an active alert"""
        if alert_type in self.active_alerts:
            alert = self.active_alerts[alert_type]
            
            # Calculate alert duration
            duration = time.time() - alert["created_at"]
            
            # Publish alert closure
            closure_message = {
                "alert_id": alert["alert_id"],
                "alert_type": alert_type,
                "status": "resolved",
                "duration_seconds": duration,
                "resolved_at": time.time()
            }
            
            await self.redis_client.publish(
                "uptime_alerts",
                json.dumps(closure_message)
            )
            
            logger.info(f"✅ RESOLVED: {alert_type} - Duration: {duration:.2f}s")
            
            # Remove from active alerts
            del self.active_alerts[alert_type]
            
    async def _publish_health_metrics(self, health_sample: Dict):
        """Publish health metrics to Redis"""
        metrics = {
            "monitor_id": self.monitor_id,
            "timestamp": health_sample["timestamp"],
            "health_score": health_sample["health_score"],
            "uptime_percentage": self.calculate_current_uptime(),
            "total_downtime_seconds": self.total_downtime_seconds,
            "incident_count": self.incident_count,
            "active_alerts": len(self.active_alerts)
        }
        
        await self.redis_client.publish(
            "uptime_metrics",
            json.dumps(metrics)
        )
        
    async def _metrics_aggregation_loop(self):
        """Aggregate and calculate metrics"""
        while self.is_monitoring:
            try:
                # Calculate current metrics
                metrics = await self.calculate_comprehensive_metrics()
                
                # Log metrics every 60 seconds
                if int(time.time()) % 60 == 0:
                    logger.info(f"📊 Uptime: {metrics.uptime_percentage:.4f}% "
                               f"Avg Response: {metrics.average_response_time_ms:.2f}ms "
                               f"Incidents: {metrics.incident_count}")
                    
                # Store metrics
                await self.redis_client.setex(
                    f"uptime_metrics_{self.monitor_id}",
                    300,  # 5 minutes TTL
                    json.dumps(metrics.__dict__)
                )
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
                await asyncio.sleep(10)
                
    async def _uptime_calculation_loop(self):
        """Continuous uptime calculation"""
        while self.is_monitoring:
            try:
                current_uptime = self.calculate_current_uptime()
                
                # Check SLA violation
                if current_uptime < self.config.target_uptime_percentage:
                    logger.warning(f"⚠️ SLA VIOLATION: Current uptime {current_uptime:.4f}% "
                                  f"below target {self.config.target_uptime_percentage}%")
                    
                    # Create SLA violation alert
                    await self._create_alert("sla_violation", {
                        "current_uptime": current_uptime,
                        "target_uptime": self.config.target_uptime_percentage,
                        "violation_percentage": self.config.target_uptime_percentage - current_uptime
                    })
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Uptime calculation error: {e}")
                await asyncio.sleep(30)
                
    def calculate_current_uptime(self) -> float:
        """Calculate current uptime percentage"""
        if not self.monitoring_start_time:
            return 100.0
            
        total_monitoring_time = time.time() - self.monitoring_start_time
        
        # Add current incident time if active
        current_downtime = self.total_downtime_seconds
        if self.current_incident_start:
            current_downtime += time.time() - self.current_incident_start
            
        if total_monitoring_time <= 0:
            return 100.0
            
        uptime_ratio = (total_monitoring_time - current_downtime) / total_monitoring_time
        return uptime_ratio * 100
        
    async def calculate_comprehensive_metrics(self) -> UptimeMetrics:
        """Calculate comprehensive uptime metrics"""
        if not self.health_history:
            return UptimeMetrics(0, 0, 0, 100.0, 0, 0, 0, 0, 0)
            
        # Calculate basic metrics
        total_checks = len(self.health_history)
        successful_checks = sum(1 for h in self.health_history if h.status == "healthy")
        failed_checks = total_checks - successful_checks
        
        # Response time metrics
        response_times = [h.response_time_ms for h in self.health_history if h.response_time_ms > 0]
        
        if response_times:
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
            
        # Current uptime
        current_uptime = self.calculate_current_uptime()
        
        return UptimeMetrics(
            total_checks=total_checks,
            successful_checks=successful_checks,
            failed_checks=failed_checks,
            uptime_percentage=current_uptime,
            average_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            total_downtime_seconds=self.total_downtime_seconds,
            incident_count=self.incident_count
        )
        
    async def generate_uptime_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive uptime report"""
        logger.info(f"📋 Generating uptime report for last {time_range_hours} hours")
        
        metrics = await self.calculate_comprehensive_metrics()
        
        # Calculate SLA compliance
        sla_compliance = metrics.uptime_percentage >= self.config.target_uptime_percentage
        
        # Availability by component
        component_availability = self._calculate_component_availability(time_range_hours)
        
        # Incident analysis
        incident_analysis = self._analyze_incidents(time_range_hours)
        
        # Performance analysis
        performance_analysis = self._analyze_performance(time_range_hours)
        
        report = {
            "report_metadata": {
                "monitor_id": self.monitor_id,
                "time_range_hours": time_range_hours,
                "report_generated_at": datetime.utcnow().isoformat(),
                "monitoring_duration_hours": (time.time() - self.monitoring_start_time) / 3600 if self.monitoring_start_time else 0
            },
            "sla_compliance": {
                "target_uptime": self.config.target_uptime_percentage,
                "actual_uptime": metrics.uptime_percentage,
                "sla_met": sla_compliance,
                "uptime_deficit": max(0, self.config.target_uptime_percentage - metrics.uptime_percentage)
            },
            "availability_metrics": {
                "total_checks": metrics.total_checks,
                "successful_checks": metrics.successful_checks,
                "failed_checks": metrics.failed_checks,
                "availability_percentage": (metrics.successful_checks / metrics.total_checks * 100) if metrics.total_checks > 0 else 100,
                "total_downtime_seconds": metrics.total_downtime_seconds,
                "incident_count": metrics.incident_count
            },
            "performance_metrics": {
                "average_response_time_ms": metrics.average_response_time_ms,
                "p95_response_time_ms": metrics.p95_response_time_ms,
                "p99_response_time_ms": metrics.p99_response_time_ms
            },
            "component_availability": component_availability,
            "incident_analysis": incident_analysis,
            "performance_analysis": performance_analysis,
            "active_alerts": len(self.active_alerts),
            "recommendations": self._generate_uptime_recommendations(metrics, sla_compliance)
        }
        
        # Log report summary
        logger.info("=" * 80)
        logger.info("📊 UPTIME MONITORING REPORT")
        logger.info("=" * 80)
        logger.info(f"SLA Status: {'✅ PASSED' if sla_compliance else '❌ FAILED'}")
        logger.info(f"Target Uptime: {self.config.target_uptime_percentage}%")
        logger.info(f"Actual Uptime: {metrics.uptime_percentage:.4f}%")
        logger.info(f"Total Incidents: {metrics.incident_count}")
        logger.info(f"Total Downtime: {metrics.total_downtime_seconds:.2f}s")
        logger.info(f"Avg Response Time: {metrics.average_response_time_ms:.2f}ms")
        logger.info(f"P99 Response Time: {metrics.p99_response_time_ms:.2f}ms")
        logger.info("=" * 80)
        
        return report
        
    def _calculate_component_availability(self, time_range_hours: int) -> Dict[str, Dict]:
        """Calculate availability for each component"""
        cutoff_time = time.time() - (time_range_hours * 3600)
        recent_checks = [h for h in self.health_history if h.timestamp > cutoff_time]
        
        component_stats = {}
        
        for check in recent_checks:
            if check.component not in component_stats:
                component_stats[check.component] = {
                    "total_checks": 0,
                    "successful_checks": 0,
                    "response_times": []
                }
                
            stats = component_stats[check.component]
            stats["total_checks"] += 1
            
            if check.status == "healthy":
                stats["successful_checks"] += 1
                
            if check.response_time_ms > 0:
                stats["response_times"].append(check.response_time_ms)
                
        # Calculate final metrics
        for component, stats in component_stats.items():
            if stats["total_checks"] > 0:
                stats["availability_percentage"] = (stats["successful_checks"] / stats["total_checks"]) * 100
                
                if stats["response_times"]:
                    stats["avg_response_time_ms"] = np.mean(stats["response_times"])
                    stats["p95_response_time_ms"] = np.percentile(stats["response_times"], 95)
                else:
                    stats["avg_response_time_ms"] = 0
                    stats["p95_response_time_ms"] = 0
                    
                # Remove raw response times for clean output
                del stats["response_times"]
                
        return component_stats
        
    def _analyze_incidents(self, time_range_hours: int) -> Dict[str, Any]:
        """Analyze incident patterns"""
        # This would analyze incident patterns, frequency, duration, etc.
        # For now, return basic incident metrics
        
        return {
            "total_incidents": self.incident_count,
            "average_incident_duration": self.total_downtime_seconds / self.incident_count if self.incident_count > 0 else 0,
            "longest_incident_duration": 0,  # Would need to track individual incidents
            "incident_frequency_per_hour": self.incident_count / time_range_hours if time_range_hours > 0 else 0
        }
        
    def _analyze_performance(self, time_range_hours: int) -> Dict[str, Any]:
        """Analyze performance trends"""
        cutoff_time = time.time() - (time_range_hours * 3600)
        recent_checks = [h for h in self.health_history if h.timestamp > cutoff_time and h.response_time_ms > 0]
        
        if not recent_checks:
            return {"trend": "no_data"}
            
        # Calculate performance trend
        response_times = [h.response_time_ms for h in recent_checks]
        timestamps = [h.timestamp for h in recent_checks]
        
        # Simple trend analysis
        if len(response_times) > 10:
            first_half = response_times[:len(response_times)//2]
            second_half = response_times[len(response_times)//2:]
            
            first_half_avg = np.mean(first_half)
            second_half_avg = np.mean(second_half)
            
            if second_half_avg > first_half_avg * 1.1:
                trend = "degrading"
            elif second_half_avg < first_half_avg * 0.9:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
            
        return {
            "trend": trend,
            "samples_analyzed": len(response_times),
            "performance_variance": np.var(response_times) if response_times else 0
        }
        
    def _generate_uptime_recommendations(self, metrics: UptimeMetrics, sla_compliance: bool) -> List[str]:
        """Generate uptime improvement recommendations"""
        recommendations = []
        
        if not sla_compliance:
            recommendations.append(
                f"SLA violation detected. Current uptime {metrics.uptime_percentage:.4f}% "
                f"below target {self.config.target_uptime_percentage}%. "
                f"Immediate attention required."
            )
            
        if metrics.incident_count > 0:
            avg_incident_duration = metrics.total_downtime_seconds / metrics.incident_count
            if avg_incident_duration > 300:  # 5 minutes
                recommendations.append(
                    f"Long incident duration detected (avg: {avg_incident_duration:.2f}s). "
                    f"Consider improving incident response and recovery procedures."
                )
                
        if metrics.p99_response_time_ms > 1000:  # 1 second
            recommendations.append(
                f"High P99 response time ({metrics.p99_response_time_ms:.2f}ms). "
                f"Consider performance optimization or scaling."
            )
            
        if len(self.active_alerts) > 0:
            recommendations.append(
                f"{len(self.active_alerts)} active alerts require attention. "
                f"Review and resolve outstanding issues."
            )
            
        if metrics.uptime_percentage > 99.99:
            recommendations.append(
                "Excellent uptime achieved! Consider this configuration as baseline "
                "for future deployments."
            )
            
        return recommendations
        
    async def cleanup(self):
        """Cleanup monitoring resources"""
        logger.info("🧹 Cleaning up uptime monitor")
        
        self.is_monitoring = False
        
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("✅ Uptime monitor cleanup completed")