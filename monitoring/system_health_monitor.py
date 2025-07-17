#!/usr/bin/env python3
"""
System Health Monitoring and Alerting
Comprehensive system health monitoring with proactive alerting
"""

import asyncio
import psutil
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import socket
import ssl
import requests
from urllib.parse import urlparse

# Monitoring imports
from .health_check_system import ComprehensiveHealthCheckSystem, HealthStatus
from .alerting_system import AlertManager, Alert, AlertType, AlertSeverity, AlertStatus
from .adaptive_threshold_monitor import AdaptiveThresholdMonitor

# Metrics
from prometheus_client import Counter, Histogram, Gauge
import redis
import psycopg2

# System health metrics
SYSTEM_HEALTH_SCORE = Gauge('system_health_score', 'Overall system health score')
SYSTEM_UPTIME = Gauge('system_uptime_seconds', 'System uptime in seconds')
SYSTEM_ALERTS_GENERATED = Counter('system_alerts_generated_total', 'Total system alerts generated', ['category'])
SYSTEM_RECOVERY_TIME = Histogram('system_recovery_time_seconds', 'System recovery time', ['component'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthCategory(Enum):
    """Health monitoring categories."""
    SYSTEM_RESOURCES = "system_resources"
    NETWORK = "network"
    STORAGE = "storage"
    SERVICES = "services"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS_LOGIC = "business_logic"

class HealthSeverity(Enum):
    """Health issue severity."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class HealthIssue:
    """Health issue data structure."""
    category: HealthCategory
    severity: HealthSeverity
    title: str
    description: str
    timestamp: datetime
    affected_components: List[str]
    metrics: Dict[str, Any]
    suggested_actions: List[str]
    auto_remediation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'affected_components': self.affected_components,
            'metrics': self.metrics,
            'suggested_actions': self.suggested_actions,
            'auto_remediation': self.auto_remediation
        }

class SystemResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get('thresholds', {})
        self.monitoring_interval = config.get('monitoring_interval', 30)
        
    async def check_system_resources(self) -> List[HealthIssue]:
        """Check system resource health."""
        issues = []
        
        try:
            # CPU monitoring
            cpu_issues = await self._check_cpu_health()
            issues.extend(cpu_issues)
            
            # Memory monitoring
            memory_issues = await self._check_memory_health()
            issues.extend(memory_issues)
            
            # Disk monitoring
            disk_issues = await self._check_disk_health()
            issues.extend(disk_issues)
            
            # Network monitoring
            network_issues = await self._check_network_health()
            issues.extend(network_issues)
            
            # Process monitoring
            process_issues = await self._check_process_health()
            issues.extend(process_issues)
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            issues.append(HealthIssue(
                category=HealthCategory.SYSTEM_RESOURCES,
                severity=HealthSeverity.ERROR,
                title="System Resource Check Failed",
                description=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.utcnow(),
                affected_components=['system_monitor'],
                metrics={'error': str(e)},
                suggested_actions=['Check system monitor configuration', 'Restart monitoring service']
            ))
        
        return issues
    
    async def _check_cpu_health(self) -> List[HealthIssue]:
        """Check CPU health."""
        issues = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Load average
            load_avg = psutil.getloadavg()
            
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            
            # Check thresholds
            cpu_threshold = self.thresholds.get('cpu_percent', 80)
            if cpu_percent > cpu_threshold:
                severity = HealthSeverity.CRITICAL if cpu_percent > 95 else HealthSeverity.WARNING
                
                issues.append(HealthIssue(
                    category=HealthCategory.SYSTEM_RESOURCES,
                    severity=severity,
                    title="High CPU Usage",
                    description=f"CPU usage at {cpu_percent:.1f}% (threshold: {cpu_threshold}%)",
                    timestamp=datetime.utcnow(),
                    affected_components=['cpu'],
                    metrics={
                        'cpu_percent': cpu_percent,
                        'cpu_count': cpu_count,
                        'load_avg_1m': load_avg[0],
                        'load_avg_5m': load_avg[1],
                        'load_avg_15m': load_avg[2]
                    },
                    suggested_actions=[
                        'Identify high CPU processes',
                        'Consider scaling resources',
                        'Optimize application performance'
                    ]
                ))
            
            # Check load average
            load_threshold = self.thresholds.get('load_avg', cpu_count * 0.8)
            if load_avg[0] > load_threshold:
                issues.append(HealthIssue(
                    category=HealthCategory.SYSTEM_RESOURCES,
                    severity=HealthSeverity.WARNING,
                    title="High System Load",
                    description=f"1-minute load average: {load_avg[0]:.2f} (threshold: {load_threshold:.2f})",
                    timestamp=datetime.utcnow(),
                    affected_components=['cpu'],
                    metrics={
                        'load_avg_1m': load_avg[0],
                        'load_threshold': load_threshold,
                        'cpu_count': cpu_count
                    },
                    suggested_actions=[
                        'Check for runaway processes',
                        'Review system load patterns',
                        'Consider load balancing'
                    ]
                ))
            
        except Exception as e:
            logger.error(f"Error checking CPU health: {e}")
        
        return issues
    
    async def _check_memory_health(self) -> List[HealthIssue]:
        """Check memory health."""
        issues = []
        
        try:
            # Virtual memory
            memory = psutil.virtual_memory()
            
            # Swap memory
            swap = psutil.swap_memory()
            
            # Check memory usage
            memory_threshold = self.thresholds.get('memory_percent', 85)
            if memory.percent > memory_threshold:
                severity = HealthSeverity.CRITICAL if memory.percent > 95 else HealthSeverity.WARNING
                
                issues.append(HealthIssue(
                    category=HealthCategory.SYSTEM_RESOURCES,
                    severity=severity,
                    title="High Memory Usage",
                    description=f"Memory usage at {memory.percent:.1f}% (threshold: {memory_threshold}%)",
                    timestamp=datetime.utcnow(),
                    affected_components=['memory'],
                    metrics={
                        'memory_percent': memory.percent,
                        'memory_total_gb': memory.total / (1024**3),
                        'memory_available_gb': memory.available / (1024**3),
                        'memory_used_gb': memory.used / (1024**3)
                    },
                    suggested_actions=[
                        'Identify memory-intensive processes',
                        'Clear system caches',
                        'Consider increasing memory capacity'
                    ]
                ))
            
            # Check swap usage
            swap_threshold = self.thresholds.get('swap_percent', 50)
            if swap.percent > swap_threshold:
                issues.append(HealthIssue(
                    category=HealthCategory.SYSTEM_RESOURCES,
                    severity=HealthSeverity.WARNING,
                    title="High Swap Usage",
                    description=f"Swap usage at {swap.percent:.1f}% (threshold: {swap_threshold}%)",
                    timestamp=datetime.utcnow(),
                    affected_components=['memory', 'swap'],
                    metrics={
                        'swap_percent': swap.percent,
                        'swap_total_gb': swap.total / (1024**3),
                        'swap_used_gb': swap.used / (1024**3)
                    },
                    suggested_actions=[
                        'Increase physical memory',
                        'Optimize memory usage',
                        'Review swap configuration'
                    ]
                ))
            
        except Exception as e:
            logger.error(f"Error checking memory health: {e}")
        
        return issues
    
    async def _check_disk_health(self) -> List[HealthIssue]:
        """Check disk health."""
        issues = []
        
        try:
            # Disk usage for root partition
            disk_usage = psutil.disk_usage('/')
            
            # Disk I/O statistics
            disk_io = psutil.disk_io_counters()
            
            # Check disk usage
            disk_threshold = self.thresholds.get('disk_percent', 90)
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            if disk_percent > disk_threshold:
                severity = HealthSeverity.CRITICAL if disk_percent > 95 else HealthSeverity.WARNING
                
                issues.append(HealthIssue(
                    category=HealthCategory.STORAGE,
                    severity=severity,
                    title="High Disk Usage",
                    description=f"Disk usage at {disk_percent:.1f}% (threshold: {disk_threshold}%)",
                    timestamp=datetime.utcnow(),
                    affected_components=['disk'],
                    metrics={
                        'disk_percent': disk_percent,
                        'disk_total_gb': disk_usage.total / (1024**3),
                        'disk_used_gb': disk_usage.used / (1024**3),
                        'disk_free_gb': disk_usage.free / (1024**3)
                    },
                    suggested_actions=[
                        'Clean up temporary files',
                        'Archive old data',
                        'Expand disk capacity'
                    ]
                ))
            
            # Check disk I/O
            if disk_io:
                # Calculate I/O wait time (simplified)
                read_time = disk_io.read_time / 1000  # Convert to seconds
                write_time = disk_io.write_time / 1000
                
                io_threshold = self.thresholds.get('disk_io_time', 1000)  # 1 second
                total_io_time = read_time + write_time
                
                if total_io_time > io_threshold:
                    issues.append(HealthIssue(
                        category=HealthCategory.STORAGE,
                        severity=HealthSeverity.WARNING,
                        title="High Disk I/O",
                        description=f"High disk I/O time: {total_io_time:.1f}s",
                        timestamp=datetime.utcnow(),
                        affected_components=['disk'],
                        metrics={
                            'read_time_s': read_time,
                            'write_time_s': write_time,
                            'total_io_time_s': total_io_time,
                            'read_count': disk_io.read_count,
                            'write_count': disk_io.write_count
                        },
                        suggested_actions=[
                            'Optimize disk I/O patterns',
                            'Consider SSD upgrade',
                            'Review disk-intensive processes'
                        ]
                    ))
            
        except Exception as e:
            logger.error(f"Error checking disk health: {e}")
        
        return issues
    
    async def _check_network_health(self) -> List[HealthIssue]:
        """Check network health."""
        issues = []
        
        try:
            # Network I/O statistics
            network_io = psutil.net_io_counters()
            
            # Network connections
            connections = psutil.net_connections()
            
            # Check for network errors
            if network_io:
                error_threshold = self.thresholds.get('network_errors', 100)
                total_errors = network_io.errin + network_io.errout
                
                if total_errors > error_threshold:
                    issues.append(HealthIssue(
                        category=HealthCategory.NETWORK,
                        severity=HealthSeverity.WARNING,
                        title="Network Errors Detected",
                        description=f"Network errors: {total_errors} (threshold: {error_threshold})",
                        timestamp=datetime.utcnow(),
                        affected_components=['network'],
                        metrics={
                            'errors_in': network_io.errin,
                            'errors_out': network_io.errout,
                            'total_errors': total_errors,
                            'bytes_sent': network_io.bytes_sent,
                            'bytes_recv': network_io.bytes_recv
                        },
                        suggested_actions=[
                            'Check network hardware',
                            'Review network configuration',
                            'Monitor network quality'
                        ]
                    ))
            
            # Check connection count
            connection_threshold = self.thresholds.get('connection_count', 1000)
            if len(connections) > connection_threshold:
                issues.append(HealthIssue(
                    category=HealthCategory.NETWORK,
                    severity=HealthSeverity.WARNING,
                    title="High Connection Count",
                    description=f"Active connections: {len(connections)} (threshold: {connection_threshold})",
                    timestamp=datetime.utcnow(),
                    affected_components=['network'],
                    metrics={
                        'connection_count': len(connections),
                        'connection_threshold': connection_threshold
                    },
                    suggested_actions=[
                        'Review connection pooling',
                        'Check for connection leaks',
                        'Optimize connection management'
                    ]
                ))
            
        except Exception as e:
            logger.error(f"Error checking network health: {e}")
        
        return issues
    
    async def _check_process_health(self) -> List[HealthIssue]:
        """Check process health."""
        issues = []
        
        try:
            # Get all processes
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Check for high resource usage processes
            cpu_threshold = self.thresholds.get('process_cpu_percent', 50)
            memory_threshold = self.thresholds.get('process_memory_percent', 20)
            
            high_cpu_processes = [p for p in processes if p['cpu_percent'] > cpu_threshold]
            high_memory_processes = [p for p in processes if p['memory_percent'] > memory_threshold]
            
            if high_cpu_processes:
                issues.append(HealthIssue(
                    category=HealthCategory.PERFORMANCE,
                    severity=HealthSeverity.WARNING,
                    title="High CPU Usage Processes",
                    description=f"Found {len(high_cpu_processes)} processes with high CPU usage",
                    timestamp=datetime.utcnow(),
                    affected_components=['processes'],
                    metrics={
                        'high_cpu_processes': high_cpu_processes[:5],  # Top 5
                        'cpu_threshold': cpu_threshold
                    },
                    suggested_actions=[
                        'Investigate high CPU processes',
                        'Consider process optimization',
                        'Review resource allocation'
                    ]
                ))
            
            if high_memory_processes:
                issues.append(HealthIssue(
                    category=HealthCategory.PERFORMANCE,
                    severity=HealthSeverity.WARNING,
                    title="High Memory Usage Processes",
                    description=f"Found {len(high_memory_processes)} processes with high memory usage",
                    timestamp=datetime.utcnow(),
                    affected_components=['processes'],
                    metrics={
                        'high_memory_processes': high_memory_processes[:5],  # Top 5
                        'memory_threshold': memory_threshold
                    },
                    suggested_actions=[
                        'Investigate memory usage',
                        'Check for memory leaks',
                        'Optimize memory allocation'
                    ]
                ))
            
        except Exception as e:
            logger.error(f"Error checking process health: {e}")
        
        return issues

class ServiceHealthMonitor:
    """Monitor critical services."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = config.get('services', [])
        self.timeout = config.get('timeout', 10)
        
    async def check_services(self) -> List[HealthIssue]:
        """Check service health."""
        issues = []
        
        for service_config in self.services:
            try:
                service_issues = await self._check_service(service_config)
                issues.extend(service_issues)
            except Exception as e:
                logger.error(f"Error checking service {service_config.get('name', 'unknown')}: {e}")
        
        return issues
    
    async def _check_service(self, service_config: Dict[str, Any]) -> List[HealthIssue]:
        """Check individual service."""
        issues = []
        service_name = service_config.get('name', 'unknown')
        service_type = service_config.get('type', 'http')
        
        try:
            if service_type == 'http':
                issues.extend(await self._check_http_service(service_config))
            elif service_type == 'tcp':
                issues.extend(await self._check_tcp_service(service_config))
            elif service_type == 'database':
                issues.extend(await self._check_database_service(service_config))
            elif service_type == 'redis':
                issues.extend(await self._check_redis_service(service_config))
            elif service_type == 'process':
                issues.extend(await self._check_process_service(service_config))
            
        except Exception as e:
            issues.append(HealthIssue(
                category=HealthCategory.SERVICES,
                severity=HealthSeverity.ERROR,
                title=f"Service Check Failed: {service_name}",
                description=f"Failed to check service {service_name}: {str(e)}",
                timestamp=datetime.utcnow(),
                affected_components=[service_name],
                metrics={'error': str(e)},
                suggested_actions=[
                    f'Check {service_name} configuration',
                    f'Verify {service_name} is running',
                    'Review service logs'
                ]
            ))
        
        return issues
    
    async def _check_http_service(self, service_config: Dict[str, Any]) -> List[HealthIssue]:
        """Check HTTP service."""
        issues = []
        service_name = service_config['name']
        url = service_config['url']
        expected_status = service_config.get('expected_status', 200)
        
        try:
            start_time = time.time()
            response = requests.get(url, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Check status code
            if response.status_code != expected_status:
                issues.append(HealthIssue(
                    category=HealthCategory.SERVICES,
                    severity=HealthSeverity.ERROR,
                    title=f"HTTP Service Error: {service_name}",
                    description=f"HTTP {response.status_code} (expected {expected_status})",
                    timestamp=datetime.utcnow(),
                    affected_components=[service_name],
                    metrics={
                        'status_code': response.status_code,
                        'expected_status': expected_status,
                        'response_time_ms': response_time,
                        'url': url
                    },
                    suggested_actions=[
                        f'Check {service_name} service status',
                        'Review service logs',
                        'Verify service configuration'
                    ]
                ))
            
            # Check response time
            response_time_threshold = service_config.get('response_time_threshold', 5000)  # 5 seconds
            if response_time > response_time_threshold:
                issues.append(HealthIssue(
                    category=HealthCategory.PERFORMANCE,
                    severity=HealthSeverity.WARNING,
                    title=f"Slow HTTP Response: {service_name}",
                    description=f"Response time: {response_time:.1f}ms (threshold: {response_time_threshold}ms)",
                    timestamp=datetime.utcnow(),
                    affected_components=[service_name],
                    metrics={
                        'response_time_ms': response_time,
                        'response_time_threshold': response_time_threshold,
                        'url': url
                    },
                    suggested_actions=[
                        f'Optimize {service_name} performance',
                        'Check network connectivity',
                        'Review service load'
                    ]
                ))
            
        except requests.RequestException as e:
            issues.append(HealthIssue(
                category=HealthCategory.SERVICES,
                severity=HealthSeverity.CRITICAL,
                title=f"HTTP Service Unavailable: {service_name}",
                description=f"Cannot connect to {url}: {str(e)}",
                timestamp=datetime.utcnow(),
                affected_components=[service_name],
                metrics={'error': str(e), 'url': url},
                suggested_actions=[
                    f'Restart {service_name} service',
                    'Check network connectivity',
                    'Verify service configuration'
                ]
            ))
        
        return issues
    
    async def _check_tcp_service(self, service_config: Dict[str, Any]) -> List[HealthIssue]:
        """Check TCP service."""
        issues = []
        service_name = service_config['name']
        host = service_config['host']
        port = service_config['port']
        
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            
            result = sock.connect_ex((host, port))
            connection_time = (time.time() - start_time) * 1000
            
            sock.close()
            
            if result != 0:
                issues.append(HealthIssue(
                    category=HealthCategory.SERVICES,
                    severity=HealthSeverity.CRITICAL,
                    title=f"TCP Service Unavailable: {service_name}",
                    description=f"Cannot connect to {host}:{port}",
                    timestamp=datetime.utcnow(),
                    affected_components=[service_name],
                    metrics={
                        'host': host,
                        'port': port,
                        'connection_result': result,
                        'connection_time_ms': connection_time
                    },
                    suggested_actions=[
                        f'Check {service_name} service status',
                        'Verify port is open',
                        'Check firewall rules'
                    ]
                ))
            
        except Exception as e:
            issues.append(HealthIssue(
                category=HealthCategory.SERVICES,
                severity=HealthSeverity.ERROR,
                title=f"TCP Service Check Failed: {service_name}",
                description=f"Failed to check {host}:{port}: {str(e)}",
                timestamp=datetime.utcnow(),
                affected_components=[service_name],
                metrics={'error': str(e), 'host': host, 'port': port},
                suggested_actions=[
                    f'Check {service_name} configuration',
                    'Verify network connectivity',
                    'Review service logs'
                ]
            ))
        
        return issues
    
    async def _check_database_service(self, service_config: Dict[str, Any]) -> List[HealthIssue]:
        """Check database service."""
        issues = []
        service_name = service_config['name']
        
        try:
            # Connect to database
            conn = psycopg2.connect(
                host=service_config['host'],
                port=service_config['port'],
                database=service_config['database'],
                user=service_config['user'],
                password=service_config['password'],
                connect_timeout=self.timeout
            )
            
            cursor = conn.cursor()
            
            # Test query
            start_time = time.time()
            cursor.execute("SELECT 1")
            query_time = (time.time() - start_time) * 1000
            
            # Check query performance
            query_threshold = service_config.get('query_threshold', 1000)  # 1 second
            if query_time > query_threshold:
                issues.append(HealthIssue(
                    category=HealthCategory.PERFORMANCE,
                    severity=HealthSeverity.WARNING,
                    title=f"Slow Database Query: {service_name}",
                    description=f"Query time: {query_time:.1f}ms (threshold: {query_threshold}ms)",
                    timestamp=datetime.utcnow(),
                    affected_components=[service_name],
                    metrics={
                        'query_time_ms': query_time,
                        'query_threshold': query_threshold
                    },
                    suggested_actions=[
                        'Optimize database queries',
                        'Check database performance',
                        'Review database configuration'
                    ]
                ))
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            issues.append(HealthIssue(
                category=HealthCategory.SERVICES,
                severity=HealthSeverity.CRITICAL,
                title=f"Database Service Error: {service_name}",
                description=f"Database connection failed: {str(e)}",
                timestamp=datetime.utcnow(),
                affected_components=[service_name],
                metrics={'error': str(e)},
                suggested_actions=[
                    f'Check {service_name} database status',
                    'Verify database credentials',
                    'Review database logs'
                ]
            ))
        
        return issues
    
    async def _check_redis_service(self, service_config: Dict[str, Any]) -> List[HealthIssue]:
        """Check Redis service."""
        issues = []
        service_name = service_config['name']
        
        try:
            redis_client = redis.Redis(
                host=service_config['host'],
                port=service_config['port'],
                db=service_config.get('db', 0),
                socket_timeout=self.timeout
            )
            
            # Test ping
            start_time = time.time()
            redis_client.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Get Redis info
            info = redis_client.info()
            
            # Check memory usage
            memory_threshold = service_config.get('memory_threshold', 0.8)  # 80%
            if info.get('maxmemory', 0) > 0:
                memory_usage = info['used_memory'] / info['maxmemory']
                if memory_usage > memory_threshold:
                    issues.append(HealthIssue(
                        category=HealthCategory.PERFORMANCE,
                        severity=HealthSeverity.WARNING,
                        title=f"High Redis Memory Usage: {service_name}",
                        description=f"Memory usage: {memory_usage:.1%} (threshold: {memory_threshold:.1%})",
                        timestamp=datetime.utcnow(),
                        affected_components=[service_name],
                        metrics={
                            'memory_usage': memory_usage,
                            'memory_threshold': memory_threshold,
                            'used_memory': info['used_memory'],
                            'maxmemory': info['maxmemory']
                        },
                        suggested_actions=[
                            'Clear Redis cache',
                            'Optimize Redis usage',
                            'Increase Redis memory limit'
                        ]
                    ))
            
            # Check ping time
            ping_threshold = service_config.get('ping_threshold', 100)  # 100ms
            if ping_time > ping_threshold:
                issues.append(HealthIssue(
                    category=HealthCategory.PERFORMANCE,
                    severity=HealthSeverity.WARNING,
                    title=f"Slow Redis Response: {service_name}",
                    description=f"Ping time: {ping_time:.1f}ms (threshold: {ping_threshold}ms)",
                    timestamp=datetime.utcnow(),
                    affected_components=[service_name],
                    metrics={
                        'ping_time_ms': ping_time,
                        'ping_threshold': ping_threshold
                    },
                    suggested_actions=[
                        'Check Redis performance',
                        'Review Redis configuration',
                        'Optimize Redis operations'
                    ]
                ))
            
        except Exception as e:
            issues.append(HealthIssue(
                category=HealthCategory.SERVICES,
                severity=HealthSeverity.CRITICAL,
                title=f"Redis Service Error: {service_name}",
                description=f"Redis connection failed: {str(e)}",
                timestamp=datetime.utcnow(),
                affected_components=[service_name],
                metrics={'error': str(e)},
                suggested_actions=[
                    f'Check {service_name} Redis status',
                    'Verify Redis configuration',
                    'Review Redis logs'
                ]
            ))
        
        return issues
    
    async def _check_process_service(self, service_config: Dict[str, Any]) -> List[HealthIssue]:
        """Check process service."""
        issues = []
        service_name = service_config['name']
        process_name = service_config['process_name']
        
        try:
            # Find process
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'status']):
                try:
                    if process_name in proc.info['name']:
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not processes:
                issues.append(HealthIssue(
                    category=HealthCategory.SERVICES,
                    severity=HealthSeverity.CRITICAL,
                    title=f"Process Not Running: {service_name}",
                    description=f"Process '{process_name}' is not running",
                    timestamp=datetime.utcnow(),
                    affected_components=[service_name],
                    metrics={'process_name': process_name},
                    suggested_actions=[
                        f'Start {service_name} process',
                        'Check process configuration',
                        'Review process logs'
                    ]
                ))
            
        except Exception as e:
            issues.append(HealthIssue(
                category=HealthCategory.SERVICES,
                severity=HealthSeverity.ERROR,
                title=f"Process Check Failed: {service_name}",
                description=f"Failed to check process '{process_name}': {str(e)}",
                timestamp=datetime.utcnow(),
                affected_components=[service_name],
                metrics={'error': str(e), 'process_name': process_name},
                suggested_actions=[
                    f'Check {service_name} configuration',
                    'Verify process monitoring',
                    'Review system logs'
                ]
            ))
        
        return issues

class SystemHealthMonitor:
    """Comprehensive system health monitor."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resource_monitor = SystemResourceMonitor(config.get('resources', {}))
        self.service_monitor = ServiceHealthMonitor(config.get('services', {}))
        self.alert_manager = None
        self.health_check_system = None
        self.adaptive_threshold_monitor = None
        
        # Monitoring state
        self.monitoring_active = False
        self.health_issues = []
        self.system_uptime_start = time.time()
        
        # Performance tracking
        self.health_history = deque(maxlen=100)
        self.recovery_tracking = {}
        
    def set_dependencies(self, alert_manager=None, health_check_system=None, adaptive_threshold_monitor=None):
        """Set dependency components."""
        self.alert_manager = alert_manager
        self.health_check_system = health_check_system
        self.adaptive_threshold_monitor = adaptive_threshold_monitor
    
    async def start_monitoring(self):
        """Start system health monitoring."""
        self.monitoring_active = True
        logger.info("Starting system health monitoring")
        
        # Start monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._issue_processing_loop()),
            asyncio.create_task(self._metrics_update_loop())
        ]
        
        await asyncio.gather(*monitoring_tasks)
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        logger.info("Stopping system health monitoring")
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop."""
        while self.monitoring_active:
            try:
                # Check system resources
                resource_issues = await self.resource_monitor.check_system_resources()
                
                # Check services
                service_issues = await self.service_monitor.check_services()
                
                # Combine all issues
                all_issues = resource_issues + service_issues
                
                # Update current issues
                self.health_issues = all_issues
                
                # Process issues
                await self._process_health_issues(all_issues)
                
                # Update health history
                self.health_history.append({
                    'timestamp': datetime.utcnow(),
                    'issue_count': len(all_issues),
                    'severity_distribution': self._calculate_severity_distribution(all_issues)
                })
                
                await asyncio.sleep(self.config.get('monitoring_interval', 30))
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _issue_processing_loop(self):
        """Process health issues and generate alerts."""
        while self.monitoring_active:
            try:
                if self.health_issues and self.alert_manager:
                    for issue in self.health_issues:
                        await self._generate_alert_for_issue(issue)
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Error in issue processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_update_loop(self):
        """Update system metrics."""
        while self.monitoring_active:
            try:
                # Calculate overall health score
                health_score = self._calculate_health_score()
                SYSTEM_HEALTH_SCORE.set(health_score)
                
                # Update uptime
                uptime = time.time() - self.system_uptime_start
                SYSTEM_UPTIME.set(uptime)
                
                # Update alert counts
                for issue in self.health_issues:
                    SYSTEM_ALERTS_GENERATED.labels(category=issue.category.value).inc()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_health_issues(self, issues: List[HealthIssue]):
        """Process health issues."""
        for issue in issues:
            try:
                # Check if issue requires immediate attention
                if issue.severity in [HealthSeverity.CRITICAL, HealthSeverity.ERROR]:
                    logger.warning(f"Health issue detected: {issue.title}")
                    
                    # Attempt auto-remediation if enabled
                    if issue.auto_remediation:
                        await self._attempt_auto_remediation(issue)
                
                # Track issue for trend analysis
                await self._track_issue_trends(issue)
                
            except Exception as e:
                logger.error(f"Error processing health issue: {e}")
    
    async def _generate_alert_for_issue(self, issue: HealthIssue):
        """Generate alert for health issue."""
        try:
            # Map severity to alert severity
            severity_map = {
                HealthSeverity.INFO: AlertSeverity.LOW,
                HealthSeverity.WARNING: AlertSeverity.MEDIUM,
                HealthSeverity.ERROR: AlertSeverity.HIGH,
                HealthSeverity.CRITICAL: AlertSeverity.CRITICAL
            }
            
            # Map category to alert type
            type_map = {
                HealthCategory.SYSTEM_RESOURCES: AlertType.SYSTEM_PERFORMANCE,
                HealthCategory.NETWORK: AlertType.INFRASTRUCTURE,
                HealthCategory.STORAGE: AlertType.INFRASTRUCTURE,
                HealthCategory.SERVICES: AlertType.INFRASTRUCTURE,
                HealthCategory.PERFORMANCE: AlertType.SYSTEM_PERFORMANCE,
                HealthCategory.SECURITY: AlertType.SECURITY,
                HealthCategory.BUSINESS_LOGIC: AlertType.TRADING_PERFORMANCE
            }
            
            alert = Alert(
                alert_id=f"health_{issue.category.value}_{int(time.time())}",
                alert_type=type_map.get(issue.category, AlertType.INFRASTRUCTURE),
                severity=severity_map.get(issue.severity, AlertSeverity.MEDIUM),
                title=issue.title,
                description=issue.description,
                timestamp=issue.timestamp,
                source='system_health_monitor',
                status=AlertStatus.ACTIVE,
                metadata=issue.metrics,
                tags=issue.affected_components
            )
            
            await self.alert_manager.create_alert(alert)
            
        except Exception as e:
            logger.error(f"Error generating alert for issue: {e}")
    
    async def _attempt_auto_remediation(self, issue: HealthIssue):
        """Attempt automatic remediation."""
        try:
            # This is a simplified auto-remediation system
            # In a real implementation, this would be more sophisticated
            
            if issue.category == HealthCategory.SYSTEM_RESOURCES:
                if 'High Memory Usage' in issue.title:
                    # Attempt memory cleanup
                    await self._cleanup_memory()
                elif 'High Disk Usage' in issue.title:
                    # Attempt disk cleanup
                    await self._cleanup_disk()
            
            elif issue.category == HealthCategory.SERVICES:
                if 'Service Unavailable' in issue.title:
                    # Attempt service restart
                    await self._restart_service(issue.affected_components[0])
            
            logger.info(f"Auto-remediation attempted for: {issue.title}")
            
        except Exception as e:
            logger.error(f"Error in auto-remediation: {e}")
    
    async def _cleanup_memory(self):
        """Cleanup memory."""
        try:
            # Clear system caches
            subprocess.run(['sync'], check=True)
            subprocess.run(['echo', '1', '>', '/proc/sys/vm/drop_caches'], shell=True)
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in memory cleanup: {e}")
    
    async def _cleanup_disk(self):
        """Cleanup disk space."""
        try:
            # Clean temporary files
            subprocess.run(['find', '/tmp', '-type', 'f', '-atime', '+7', '-delete'], check=True)
            
            # Clean log files
            subprocess.run(['find', '/var/log', '-name', '*.log', '-size', '+100M', '-delete'], check=True)
            
            logger.info("Disk cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in disk cleanup: {e}")
    
    async def _restart_service(self, service_name: str):
        """Restart service."""
        try:
            # This is a simplified service restart
            # In a real implementation, this would be more sophisticated
            subprocess.run(['systemctl', 'restart', service_name], check=True)
            logger.info(f"Service {service_name} restarted")
            
        except Exception as e:
            logger.error(f"Error restarting service {service_name}: {e}")
    
    async def _track_issue_trends(self, issue: HealthIssue):
        """Track issue trends."""
        try:
            # Track issue frequency
            issue_key = f"{issue.category.value}_{issue.title}"
            
            if issue_key not in self.recovery_tracking:
                self.recovery_tracking[issue_key] = {
                    'count': 0,
                    'first_seen': datetime.utcnow(),
                    'last_seen': datetime.utcnow()
                }
            
            self.recovery_tracking[issue_key]['count'] += 1
            self.recovery_tracking[issue_key]['last_seen'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error tracking issue trends: {e}")
    
    def _calculate_severity_distribution(self, issues: List[HealthIssue]) -> Dict[str, int]:
        """Calculate severity distribution."""
        distribution = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }
        
        for issue in issues:
            distribution[issue.severity.value] += 1
        
        return distribution
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score."""
        if not self.health_issues:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {
            HealthSeverity.INFO: 0.1,
            HealthSeverity.WARNING: 0.3,
            HealthSeverity.ERROR: 0.6,
            HealthSeverity.CRITICAL: 1.0
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in self.health_issues)
        
        # Calculate score (0-100)
        max_weight = len(self.health_issues) * 1.0  # Max if all critical
        health_score = max(0, 100 - (total_weight / max_weight * 100))
        
        return health_score
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            'monitoring_active': self.monitoring_active,
            'health_score': self._calculate_health_score(),
            'issue_count': len(self.health_issues),
            'severity_distribution': self._calculate_severity_distribution(self.health_issues),
            'uptime_seconds': time.time() - self.system_uptime_start,
            'issues': [issue.to_dict() for issue in self.health_issues],
            'last_update': datetime.utcnow().isoformat()
        }

# Factory function
def create_system_health_monitor(config: Dict[str, Any]) -> SystemHealthMonitor:
    """Create system health monitor instance."""
    return SystemHealthMonitor(config)

# Example configuration
EXAMPLE_CONFIG = {
    'monitoring_interval': 30,
    'resources': {
        'thresholds': {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'load_avg': 4.0,
            'swap_percent': 50,
            'disk_io_time': 1000,
            'network_errors': 100,
            'connection_count': 1000,
            'process_cpu_percent': 50,
            'process_memory_percent': 20
        }
    },
    'services': {
        'timeout': 10,
        'services': [
            {
                'name': 'web_server',
                'type': 'http',
                'url': 'http://localhost:8080/health',
                'expected_status': 200,
                'response_time_threshold': 5000
            },
            {
                'name': 'database',
                'type': 'database',
                'host': 'localhost',
                'port': 5432,
                'database': 'grandmodel',
                'user': 'user',
                'password': 'password',
                'query_threshold': 1000
            },
            {
                'name': 'redis',
                'type': 'redis',
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'memory_threshold': 0.8,
                'ping_threshold': 100
            }
        ]
    }
}

# Example usage
async def main():
    """Example usage of system health monitor."""
    config = EXAMPLE_CONFIG
    health_monitor = create_system_health_monitor(config)
    
    # Start monitoring
    await health_monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())