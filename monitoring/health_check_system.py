#!/usr/bin/env python3
"""
Comprehensive Health Check System for GrandModel MARL Trading System
High-frequency health monitoring with <500ms response times
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis
import psutil
import psycopg2
from prometheus_client import Counter, Histogram, Gauge, Info
import traceback
from concurrent.futures import ThreadPoolExecutor
import threading

# Health check metrics
HEALTH_CHECK_DURATION = Histogram('health_check_duration_seconds', 'Health check execution time', ['service', 'check_type'])
HEALTH_CHECK_SUCCESS = Counter('health_checks_success_total', 'Successful health checks', ['service', 'check_type'])
HEALTH_CHECK_FAILURES = Counter('health_checks_failed_total', 'Failed health checks', ['service', 'check_type'])
HEALTH_STATUS = Gauge('health_status', 'Health status (1=healthy, 0=unhealthy)', ['service', 'check_type'])
SLA_COMPLIANCE = Gauge('sla_compliance_rate_percent', 'SLA compliance rate', ['service', 'sla_type'])
HEALTH_CHECK_INFO = Info('health_check_info', 'Health check information')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class CheckType(Enum):
    """Health check types."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"
    DEEP = "deep"
    DEPENDENCY = "dependency"
    PERFORMANCE = "performance"
    BUSINESS = "business"

@dataclass
class HealthCheckResult:
    """Health check result structure."""
    service: str
    check_type: CheckType
    status: HealthStatus
    timestamp: datetime
    duration_ms: float
    message: str
    details: Dict[str, Any]
    dependencies: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'service': self.service,
            'check_type': self.check_type.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'message': self.message,
            'details': self.details,
            'dependencies': self.dependencies or []
        }

class HealthChecker:
    """Base health checker class."""
    
    def __init__(self, service_name: str, check_type: CheckType):
        self.service_name = service_name
        self.check_type = check_type
        
    async def execute(self) -> HealthCheckResult:
        """Execute the health check."""
        raise NotImplementedError("Subclasses must implement execute method")

class DatabaseHealthChecker(HealthChecker):
    """Database connectivity and performance health checker."""
    
    def __init__(self, service_name: str, db_config: Dict[str, Any]):
        super().__init__(service_name, CheckType.DEPENDENCY)
        self.db_config = db_config
        self.connection_pool = None
        
    async def execute(self) -> HealthCheckResult:
        """Check database health."""
        start_time = time.time()
        
        try:
            # Test connection
            connection = psycopg2.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Test query performance
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            # Check connection count
            cursor.execute("SELECT count(*) FROM pg_stat_activity")
            connection_count = cursor.fetchone()[0]
            
            # Check locks
            cursor.execute("SELECT count(*) FROM pg_locks WHERE NOT granted")
            lock_count = cursor.fetchone()[0]
            
            cursor.close()
            connection.close()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status
            if duration_ms > 1000:  # 1 second threshold
                status = HealthStatus.DEGRADED
                message = f"Database slow response: {duration_ms:.2f}ms"
            elif connection_count > 100:  # Connection threshold
                status = HealthStatus.DEGRADED
                message = f"High connection count: {connection_count}"
            elif lock_count > 10:  # Lock threshold
                status = HealthStatus.DEGRADED
                message = f"High lock count: {lock_count}"
            else:
                status = HealthStatus.HEALTHY
                message = "Database healthy"
                
            details = {
                'connection_count': connection_count,
                'lock_count': lock_count,
                'response_time_ms': duration_ms
            }
            
            return HealthCheckResult(
                service=self.service_name,
                check_type=self.check_type,
                status=status,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=message,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service=self.service_name,
                check_type=self.check_type,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=f"Database connection failed: {str(e)}",
                details={'error': str(e)}
            )

class RedisHealthChecker(HealthChecker):
    """Redis connectivity and performance health checker."""
    
    def __init__(self, service_name: str, redis_config: Dict[str, Any]):
        super().__init__(service_name, CheckType.DEPENDENCY)
        self.redis_config = redis_config
        self.redis_client = None
        
    async def execute(self) -> HealthCheckResult:
        """Check Redis health."""
        start_time = time.time()
        
        try:
            # Create Redis client
            redis_client = redis.Redis(**self.redis_config)
            
            # Test ping
            ping_result = redis_client.ping()
            
            # Test set/get performance
            test_key = f"health_check_{int(time.time())}"
            redis_client.set(test_key, "test_value", ex=60)
            get_result = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            # Get Redis info
            info = redis_client.info()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status
            used_memory_percent = (info['used_memory'] / info['maxmemory']) * 100 if info.get('maxmemory') else 0
            
            if duration_ms > 100:  # 100ms threshold
                status = HealthStatus.DEGRADED
                message = f"Redis slow response: {duration_ms:.2f}ms"
            elif used_memory_percent > 90:  # Memory threshold
                status = HealthStatus.DEGRADED
                message = f"Redis high memory usage: {used_memory_percent:.1f}%"
            elif info.get('connected_clients', 0) > 1000:  # Client threshold
                status = HealthStatus.DEGRADED
                message = f"High client count: {info['connected_clients']}"
            else:
                status = HealthStatus.HEALTHY
                message = "Redis healthy"
                
            details = {
                'used_memory_percent': used_memory_percent,
                'connected_clients': info.get('connected_clients', 0),
                'response_time_ms': duration_ms,
                'redis_version': info.get('redis_version', 'unknown')
            }
            
            return HealthCheckResult(
                service=self.service_name,
                check_type=self.check_type,
                status=status,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=message,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service=self.service_name,
                check_type=self.check_type,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=f"Redis connection failed: {str(e)}",
                details={'error': str(e)}
            )

class MARLAgentHealthChecker(HealthChecker):
    """MARL agent health checker."""
    
    def __init__(self, service_name: str, agent_config: Dict[str, Any]):
        super().__init__(service_name, CheckType.READINESS)
        self.agent_config = agent_config
        self.agent_url = agent_config.get('url', 'http://localhost:8000')
        
    async def execute(self) -> HealthCheckResult:
        """Check MARL agent health."""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                # Test health endpoint
                async with session.get(f"{self.agent_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                    else:
                        health_data = {"status": "unhealthy", "error": f"HTTP {response.status}"}
                
                # Test inference endpoint
                test_payload = {"test": True, "timestamp": time.time()}
                async with session.post(f"{self.agent_url}/inference", json=test_payload) as response:
                    inference_time = time.time() - start_time
                    if response.status == 200:
                        inference_data = await response.json()
                    else:
                        inference_data = {"error": f"HTTP {response.status}"}
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Determine status
                if health_data.get("status") != "healthy":
                    status = HealthStatus.UNHEALTHY
                    message = f"Agent unhealthy: {health_data.get('error', 'unknown')}"
                elif inference_time > 0.020:  # 20ms threshold
                    status = HealthStatus.DEGRADED
                    message = f"Agent slow inference: {inference_time*1000:.2f}ms"
                elif duration_ms > 1000:  # 1 second overall threshold
                    status = HealthStatus.DEGRADED
                    message = f"Agent slow response: {duration_ms:.2f}ms"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Agent healthy"
                
                details = {
                    'inference_time_ms': inference_time * 1000,
                    'health_endpoint': health_data,
                    'inference_endpoint': inference_data,
                    'response_time_ms': duration_ms
                }
                
                return HealthCheckResult(
                    service=self.service_name,
                    check_type=self.check_type,
                    status=status,
                    timestamp=datetime.utcnow(),
                    duration_ms=duration_ms,
                    message=message,
                    details=details
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service=self.service_name,
                check_type=self.check_type,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=f"Agent check failed: {str(e)}",
                details={'error': str(e)}
            )

class SystemResourceHealthChecker(HealthChecker):
    """System resource health checker."""
    
    def __init__(self, service_name: str, thresholds: Dict[str, float]):
        super().__init__(service_name, CheckType.PERFORMANCE)
        self.thresholds = thresholds
        
    async def execute(self) -> HealthCheckResult:
        """Check system resource health."""
        start_time = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Load average
            load_avg = psutil.getloadavg()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status
            issues = []
            
            if cpu_percent > self.thresholds.get('cpu_percent', 80):
                issues.append(f"High CPU: {cpu_percent:.1f}%")
            
            if memory_percent > self.thresholds.get('memory_percent', 85):
                issues.append(f"High memory: {memory_percent:.1f}%")
            
            if disk_percent > self.thresholds.get('disk_percent', 90):
                issues.append(f"High disk: {disk_percent:.1f}%")
            
            if load_avg[0] > self.thresholds.get('load_avg', 4.0):
                issues.append(f"High load: {load_avg[0]:.2f}")
            
            if issues:
                status = HealthStatus.DEGRADED
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'load_avg_1m': load_avg[0],
                'load_avg_5m': load_avg[1],
                'load_avg_15m': load_avg[2],
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv
            }
            
            return HealthCheckResult(
                service=self.service_name,
                check_type=self.check_type,
                status=status,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=message,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service=self.service_name,
                check_type=self.check_type,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=f"System resource check failed: {str(e)}",
                details={'error': str(e)}
            )

class BusinessLogicHealthChecker(HealthChecker):
    """Business logic health checker."""
    
    def __init__(self, service_name: str, business_config: Dict[str, Any]):
        super().__init__(service_name, CheckType.BUSINESS)
        self.business_config = business_config
        
    async def execute(self) -> HealthCheckResult:
        """Check business logic health."""
        start_time = time.time()
        
        try:
            # This would typically check business-specific metrics
            # For the trading system, we'd check:
            # - Trade execution success rate
            # - PnL within expected ranges
            # - Risk metrics compliance
            # - Signal quality metrics
            
            # Placeholder implementation
            issues = []
            
            # Simulate business metric checks
            trade_success_rate = 0.95  # Would be fetched from actual metrics
            pnl_daily = 5000  # Would be fetched from actual metrics
            risk_var = 0.03  # Would be fetched from actual metrics
            
            if trade_success_rate < 0.90:
                issues.append(f"Low trade success rate: {trade_success_rate:.1%}")
            
            if pnl_daily < 0:
                issues.append(f"Negative daily PnL: ${pnl_daily}")
            
            if risk_var > 0.05:
                issues.append(f"High VaR: {risk_var:.1%}")
            
            duration_ms = (time.time() - start_time) * 1000
            
            if issues:
                status = HealthStatus.DEGRADED
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "Business logic healthy"
            
            details = {
                'trade_success_rate': trade_success_rate,
                'daily_pnl': pnl_daily,
                'risk_var': risk_var,
                'check_time': datetime.utcnow().isoformat()
            }
            
            return HealthCheckResult(
                service=self.service_name,
                check_type=self.check_type,
                status=status,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=message,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service=self.service_name,
                check_type=self.check_type,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=f"Business logic check failed: {str(e)}",
                details={'error': str(e)}
            )

class ComprehensiveHealthCheckSystem:
    """Comprehensive health check system for GrandModel."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_checkers = {}
        self.health_history = {}
        self.sla_tracker = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize health checkers
        self._initialize_health_checkers()
        
        # SLA targets
        self.sla_targets = {
            'availability': 99.9,  # 99.9% uptime
            'response_time': 500,   # 500ms max response time
            'error_rate': 1.0,      # 1% max error rate
            'throughput': 100       # 100 ops/sec minimum
        }
        
    def _initialize_health_checkers(self):
        """Initialize all health checkers."""
        
        # Database health checker
        if 'database' in self.config:
            self.health_checkers['database'] = DatabaseHealthChecker(
                'database',
                self.config['database']
            )
        
        # Redis health checker
        if 'redis' in self.config:
            self.health_checkers['redis'] = RedisHealthChecker(
                'redis',
                self.config['redis']
            )
        
        # MARL agent health checkers
        if 'agents' in self.config:
            for agent_name, agent_config in self.config['agents'].items():
                self.health_checkers[f'agent_{agent_name}'] = MARLAgentHealthChecker(
                    f'agent_{agent_name}',
                    agent_config
                )
        
        # System resource health checker
        self.health_checkers['system_resources'] = SystemResourceHealthChecker(
            'system_resources',
            self.config.get('resource_thresholds', {})
        )
        
        # Business logic health checker
        self.health_checkers['business_logic'] = BusinessLogicHealthChecker(
            'business_logic',
            self.config.get('business_config', {})
        )
    
    async def run_health_check(self, service_name: str) -> HealthCheckResult:
        """Run health check for a specific service."""
        
        if service_name not in self.health_checkers:
            return HealthCheckResult(
                service=service_name,
                check_type=CheckType.LIVENESS,
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.utcnow(),
                duration_ms=0,
                message=f"No health checker for service: {service_name}",
                details={}
            )
        
        checker = self.health_checkers[service_name]
        
        # Execute health check
        result = await checker.execute()
        
        # Update metrics
        HEALTH_CHECK_DURATION.labels(
            service=service_name,
            check_type=result.check_type.value
        ).observe(result.duration_ms / 1000)
        
        if result.status == HealthStatus.HEALTHY:
            HEALTH_CHECK_SUCCESS.labels(
                service=service_name,
                check_type=result.check_type.value
            ).inc()
            HEALTH_STATUS.labels(
                service=service_name,
                check_type=result.check_type.value
            ).set(1)
        else:
            HEALTH_CHECK_FAILURES.labels(
                service=service_name,
                check_type=result.check_type.value
            ).inc()
            HEALTH_STATUS.labels(
                service=service_name,
                check_type=result.check_type.value
            ).set(0)
        
        # Store in history
        if service_name not in self.health_history:
            self.health_history[service_name] = []
        
        self.health_history[service_name].append(result)
        
        # Keep only last 100 results
        if len(self.health_history[service_name]) > 100:
            self.health_history[service_name] = self.health_history[service_name][-100:]
        
        # Update SLA tracking
        self._update_sla_tracking(result)
        
        return result
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks concurrently."""
        
        tasks = []
        for service_name in self.health_checkers.keys():
            task = asyncio.create_task(self.run_health_check(service_name))
            tasks.append((service_name, task))
        
        results = {}
        for service_name, task in tasks:
            try:
                result = await task
                results[service_name] = result
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                results[service_name] = HealthCheckResult(
                    service=service_name,
                    check_type=CheckType.LIVENESS,
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.utcnow(),
                    duration_ms=0,
                    message=f"Health check exception: {str(e)}",
                    details={'error': str(e)}
                )
        
        return results
    
    def _update_sla_tracking(self, result: HealthCheckResult):
        """Update SLA tracking metrics."""
        
        service = result.service
        if service not in self.sla_tracker:
            self.sla_tracker[service] = {
                'total_checks': 0,
                'successful_checks': 0,
                'total_response_time': 0,
                'error_count': 0
            }
        
        tracker = self.sla_tracker[service]
        tracker['total_checks'] += 1
        tracker['total_response_time'] += result.duration_ms
        
        if result.status == HealthStatus.HEALTHY:
            tracker['successful_checks'] += 1
        else:
            tracker['error_count'] += 1
        
        # Calculate SLA metrics
        availability = (tracker['successful_checks'] / tracker['total_checks']) * 100
        avg_response_time = tracker['total_response_time'] / tracker['total_checks']
        error_rate = (tracker['error_count'] / tracker['total_checks']) * 100
        
        # Update Prometheus metrics
        SLA_COMPLIANCE.labels(service=service, sla_type='availability').set(availability)
        SLA_COMPLIANCE.labels(service=service, sla_type='response_time').set(
            100 if avg_response_time < self.sla_targets['response_time'] else 0
        )
        SLA_COMPLIANCE.labels(service=service, sla_type='error_rate').set(
            100 if error_rate < self.sla_targets['error_rate'] else 0
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        
        current_time = datetime.utcnow()
        summary = {
            'timestamp': current_time.isoformat(),
            'overall_status': HealthStatus.HEALTHY.value,
            'services': {},
            'sla_compliance': {},
            'degraded_services': [],
            'unhealthy_services': []
        }
        
        for service_name, history in self.health_history.items():
            if not history:
                continue
                
            latest_result = history[-1]
            summary['services'][service_name] = {
                'status': latest_result.status.value,
                'last_check': latest_result.timestamp.isoformat(),
                'message': latest_result.message,
                'duration_ms': latest_result.duration_ms
            }
            
            # Track degraded/unhealthy services
            if latest_result.status == HealthStatus.DEGRADED:
                summary['degraded_services'].append(service_name)
            elif latest_result.status == HealthStatus.UNHEALTHY:
                summary['unhealthy_services'].append(service_name)
        
        # Determine overall status
        if summary['unhealthy_services']:
            summary['overall_status'] = HealthStatus.UNHEALTHY.value
        elif summary['degraded_services']:
            summary['overall_status'] = HealthStatus.DEGRADED.value
        
        # Add SLA compliance summary
        for service_name, tracker in self.sla_tracker.items():
            if tracker['total_checks'] > 0:
                availability = (tracker['successful_checks'] / tracker['total_checks']) * 100
                avg_response_time = tracker['total_response_time'] / tracker['total_checks']
                error_rate = (tracker['error_count'] / tracker['total_checks']) * 100
                
                summary['sla_compliance'][service_name] = {
                    'availability': availability,
                    'avg_response_time_ms': avg_response_time,
                    'error_rate_percent': error_rate,
                    'meets_sla': (
                        availability >= self.sla_targets['availability'] and
                        avg_response_time <= self.sla_targets['response_time'] and
                        error_rate <= self.sla_targets['error_rate']
                    )
                }
        
        return summary
    
    async def start_continuous_monitoring(self, interval_seconds: int = 30):
        """Start continuous health monitoring."""
        
        logger.info(f"Starting continuous health monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                start_time = time.time()
                
                # Run all health checks
                results = await self.run_all_health_checks()
                
                # Log results
                healthy_count = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
                degraded_count = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
                unhealthy_count = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
                
                logger.info(f"Health check completed: {healthy_count} healthy, {degraded_count} degraded, {unhealthy_count} unhealthy")
                
                # Wait for next interval
                execution_time = time.time() - start_time
                sleep_time = max(0, interval_seconds - execution_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(interval_seconds)

# Factory function
def create_health_check_system(config: Dict[str, Any]) -> ComprehensiveHealthCheckSystem:
    """Create health check system instance."""
    return ComprehensiveHealthCheckSystem(config)

# Example configuration
EXAMPLE_CONFIG = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "database": "grandmodel",
        "user": "grandmodel",
        "password": "password"
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0
    },
    "agents": {
        "strategic": {
            "url": "http://strategic-agent:8000"
        },
        "tactical": {
            "url": "http://tactical-agent:8000"
        },
        "risk": {
            "url": "http://risk-agent:8000"
        }
    },
    "resource_thresholds": {
        "cpu_percent": 80,
        "memory_percent": 85,
        "disk_percent": 90,
        "load_avg": 4.0
    },
    "business_config": {
        "min_trade_success_rate": 0.90,
        "max_daily_loss": 10000,
        "max_var": 0.05
    }
}

# Example usage
async def main():
    """Example usage of health check system."""
    config = EXAMPLE_CONFIG
    health_system = create_health_check_system(config)
    
    # Run single health check
    result = await health_system.run_health_check('database')
    print(f"Database health: {result.status.value} - {result.message}")
    
    # Run all health checks
    results = await health_system.run_all_health_checks()
    for service, result in results.items():
        print(f"{service}: {result.status.value} - {result.message}")
    
    # Get health summary
    summary = health_system.get_health_summary()
    print(f"Overall status: {summary['overall_status']}")
    
    # Start continuous monitoring (would run indefinitely)
    # await health_system.start_continuous_monitoring(interval_seconds=30)

if __name__ == "__main__":
    asyncio.run(main())