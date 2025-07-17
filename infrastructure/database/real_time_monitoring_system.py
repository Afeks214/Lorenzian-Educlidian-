#!/usr/bin/env python3
"""
Real-Time Database Monitoring and Alerting System
AGENT 4: DATABASE & STORAGE SPECIALIST
Focus: Real-time monitoring, alerting, and automated response for database infrastructure
"""

import asyncio
import asyncpg
import psycopg2
import time
import logging
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from collections import deque, defaultdict
import numpy as np
import psutil
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml
from pathlib import Path
import hashlib
import hmac
import base64

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison: str  # >, <, >=, <=, ==, !=
    severity: AlertSeverity
    duration_seconds: int
    enabled: bool
    tags: Dict[str, str]
    runbook_url: str
    auto_resolve: bool
    suppress_duration_seconds: int

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    timestamp: datetime
    resolved_timestamp: Optional[datetime]
    metric_name: str
    metric_value: float
    threshold: float
    tags: Dict[str, str]
    runbook_url: str
    acknowledgment_user: Optional[str]
    acknowledgment_timestamp: Optional[datetime]
    resolution_reason: Optional[str]
    escalation_level: int
    notification_sent: bool

@dataclass
class DatabaseMetrics:
    """Comprehensive database metrics"""
    timestamp: datetime
    instance_id: str
    
    # Connection metrics
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_connections: int
    max_connections: int
    connection_utilization: float
    
    # Query performance
    queries_per_second: float
    avg_query_duration_ms: float
    slow_queries_count: int
    deadlocks_count: int
    lock_waits_count: int
    
    # I/O metrics
    reads_per_second: float
    writes_per_second: float
    read_latency_ms: float
    write_latency_ms: float
    buffer_hit_ratio: float
    
    # Replication metrics
    replication_lag_bytes: int
    replication_lag_seconds: float
    wal_segments_count: int
    
    # System metrics
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    disk_io_utilization: float
    network_io_mbps: float
    
    # Database-specific metrics
    table_size_gb: float
    index_size_gb: float
    cache_hit_ratio: float
    transaction_rate: float
    rollback_rate: float
    
    # Custom metrics
    custom_metrics: Dict[str, float]

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    channel_id: str
    name: str
    type: str  # email, slack, webhook, pagerduty
    config: Dict[str, Any]
    enabled: bool
    severity_filter: List[AlertSeverity]
    tags_filter: Dict[str, str]
    rate_limit_per_hour: int
    escalation_delay_minutes: int

class RealTimeMonitoringSystem:
    """
    Comprehensive real-time monitoring system for database infrastructure
    """
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=20)
        
        # Metrics storage
        self.metrics_history = deque(maxlen=86400)  # 24 hours at 1-second intervals
        self.current_metrics = None
        
        # Alert management
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=100000)
        self.alert_evaluator = AlertEvaluator(self.logger)
        
        # Notification system
        self.notification_channels = {}
        self.notification_manager = NotificationManager(self.config.get('notifications', {}), self.logger)
        
        # Performance baselines
        self.baseline_calculator = BaselineCalculator(self.logger)
        self.anomaly_detector = AnomalyDetector(self.logger)
        
        # Database connections
        self.db_connections = {}
        
        # Setup components
        self._setup_prometheus_metrics()
        self._setup_redis_connection()
        self._load_alert_rules()
        self._setup_notification_channels()
        
        # Health checks
        self.health_checker = HealthChecker(self.config.get('health_checks', {}), self.logger)
        
        # Auto-remediation
        self.auto_remediation = AutoRemediationEngine(self.config.get('auto_remediation', {}), self.logger)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        default_config = {
            "monitoring": {
                "enabled": True,
                "interval_seconds": 10,
                "detailed_interval_seconds": 60,
                "metrics_retention_hours": 24,
                "anomaly_detection": True,
                "baseline_calculation": True,
                "predictive_alerting": True
            },
            "databases": {
                "primary": {
                    "host": "127.0.0.1",
                    "port": 5432,
                    "database": "grandmodel",
                    "user": "monitoring_user",
                    "password": "monitoring_password",
                    "instance_id": "db-primary",
                    "role": "primary"
                },
                "standby": {
                    "host": "127.0.0.1",
                    "port": 5433,
                    "database": "grandmodel",
                    "user": "monitoring_user",
                    "password": "monitoring_password",
                    "instance_id": "db-standby",
                    "role": "standby"
                }
            },
            "alert_rules": {
                "high_connection_usage": {
                    "name": "High Connection Usage",
                    "description": "Database connection usage is above threshold",
                    "metric_name": "connection_utilization",
                    "threshold": 80.0,
                    "comparison": ">",
                    "severity": "warning",
                    "duration_seconds": 300,
                    "enabled": True,
                    "auto_resolve": True,
                    "suppress_duration_seconds": 900
                },
                "critical_connection_usage": {
                    "name": "Critical Connection Usage",
                    "description": "Database connection usage is critically high",
                    "metric_name": "connection_utilization",
                    "threshold": 95.0,
                    "comparison": ">",
                    "severity": "critical",
                    "duration_seconds": 60,
                    "enabled": True,
                    "auto_resolve": True,
                    "suppress_duration_seconds": 300
                },
                "high_query_latency": {
                    "name": "High Query Latency",
                    "description": "Average query latency is above threshold",
                    "metric_name": "avg_query_duration_ms",
                    "threshold": 100.0,
                    "comparison": ">",
                    "severity": "warning",
                    "duration_seconds": 180,
                    "enabled": True,
                    "auto_resolve": True,
                    "suppress_duration_seconds": 600
                },
                "low_buffer_hit_ratio": {
                    "name": "Low Buffer Hit Ratio",
                    "description": "Database buffer hit ratio is below threshold",
                    "metric_name": "buffer_hit_ratio",
                    "threshold": 95.0,
                    "comparison": "<",
                    "severity": "warning",
                    "duration_seconds": 300,
                    "enabled": True,
                    "auto_resolve": True,
                    "suppress_duration_seconds": 1800
                },
                "replication_lag": {
                    "name": "Replication Lag",
                    "description": "Replication lag is above threshold",
                    "metric_name": "replication_lag_seconds",
                    "threshold": 30.0,
                    "comparison": ">",
                    "severity": "critical",
                    "duration_seconds": 120,
                    "enabled": True,
                    "auto_resolve": True,
                    "suppress_duration_seconds": 600
                }
            },
            "notifications": {
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "alerts@grandmodel.com",
                    "password": "email_password",
                    "from_address": "alerts@grandmodel.com",
                    "to_addresses": ["admin@grandmodel.com", "ops@grandmodel.com"],
                    "severity_filter": ["warning", "critical", "emergency"],
                    "rate_limit_per_hour": 10
                },
                "slack": {
                    "enabled": False,
                    "webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz",
                    "channel": "#alerts",
                    "severity_filter": ["critical", "emergency"],
                    "rate_limit_per_hour": 20
                },
                "webhook": {
                    "enabled": False,
                    "url": "https://api.alertmanager.com/webhook",
                    "headers": {"Authorization": "Bearer token"},
                    "severity_filter": ["critical", "emergency"],
                    "rate_limit_per_hour": 50
                }
            },
            "health_checks": {
                "enabled": True,
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "checks": [
                    {
                        "name": "database_connectivity",
                        "type": "sql",
                        "query": "SELECT 1",
                        "expected_result": 1,
                        "timeout_seconds": 5
                    },
                    {
                        "name": "replication_status",
                        "type": "sql",
                        "query": "SELECT pg_is_in_recovery()",
                        "timeout_seconds": 5
                    }
                ]
            },
            "auto_remediation": {
                "enabled": True,
                "actions": {
                    "restart_connections": {
                        "enabled": True,
                        "triggers": ["high_connection_usage"],
                        "command": "pg_terminate_backend",
                        "cooldown_seconds": 300
                    },
                    "clear_cache": {
                        "enabled": True,
                        "triggers": ["low_buffer_hit_ratio"],
                        "command": "DISCARD ALL",
                        "cooldown_seconds": 600
                    }
                }
            },
            "redis": {
                "host": "127.0.0.1",
                "port": 6379,
                "db": 3,
                "password": None
            },
            "prometheus": {
                "enabled": True,
                "port": 9093,
                "metrics_path": "/metrics"
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    return {**default_config, **loaded_config}
            except Exception as e:
                logging.error(f"Failed to load config file: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('real_time_monitoring')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = Path('/var/log/db_monitoring')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / 'monitoring.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Alert handler
        alert_handler = logging.FileHandler(log_dir / 'alerts.log')
        alert_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        alert_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.addHandler(alert_handler)
        
        return logger
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Database metrics
        self.db_connections_gauge = Gauge(
            'db_connections_total',
            'Total database connections',
            ['instance_id', 'state']
        )
        
        self.db_query_duration_histogram = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['instance_id', 'query_type']
        )
        
        self.db_buffer_hit_ratio_gauge = Gauge(
            'db_buffer_hit_ratio',
            'Database buffer hit ratio',
            ['instance_id']
        )
        
        self.db_replication_lag_gauge = Gauge(
            'db_replication_lag_seconds',
            'Database replication lag in seconds',
            ['instance_id']
        )
        
        # System metrics
        self.db_cpu_usage_gauge = Gauge(
            'db_cpu_usage_percent',
            'Database CPU usage percentage',
            ['instance_id']
        )
        
        self.db_memory_usage_gauge = Gauge(
            'db_memory_usage_percent',
            'Database memory usage percentage',
            ['instance_id']
        )
        
        self.db_disk_usage_gauge = Gauge(
            'db_disk_usage_percent',
            'Database disk usage percentage',
            ['instance_id']
        )
        
        # Alert metrics
        self.alerts_total_counter = Counter(
            'db_alerts_total',
            'Total database alerts',
            ['instance_id', 'severity', 'rule_id']
        )
        
        self.alerts_active_gauge = Gauge(
            'db_alerts_active',
            'Active database alerts',
            ['instance_id', 'severity']
        )
        
        # Monitoring metrics
        self.monitoring_duration_histogram = Histogram(
            'db_monitoring_duration_seconds',
            'Database monitoring collection duration',
            ['instance_id']
        )
        
        self.monitoring_errors_counter = Counter(
            'db_monitoring_errors_total',
            'Total monitoring errors',
            ['instance_id', 'error_type']
        )
        
        # Health check metrics
        self.health_check_success_gauge = Gauge(
            'db_health_check_success',
            'Database health check success (1=success, 0=failure)',
            ['instance_id', 'check_name']
        )
        
        # Performance score
        self.performance_score_gauge = Gauge(
            'db_performance_score',
            'Database performance score (0-100)',
            ['instance_id']
        )
    
    def _setup_redis_connection(self):
        """Setup Redis connection"""
        try:
            redis_config = self.config['redis']
            self.redis_client = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db'],
                password=redis_config['password'],
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis connection established")
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        for rule_id, rule_config in self.config['alert_rules'].items():
            alert_rule = AlertRule(
                rule_id=rule_id,
                name=rule_config['name'],
                description=rule_config['description'],
                metric_name=rule_config['metric_name'],
                threshold=rule_config['threshold'],
                comparison=rule_config['comparison'],
                severity=AlertSeverity(rule_config['severity']),
                duration_seconds=rule_config['duration_seconds'],
                enabled=rule_config['enabled'],
                tags=rule_config.get('tags', {}),
                runbook_url=rule_config.get('runbook_url', ''),
                auto_resolve=rule_config.get('auto_resolve', True),
                suppress_duration_seconds=rule_config.get('suppress_duration_seconds', 300)
            )
            
            self.alert_rules[rule_id] = alert_rule
        
        self.logger.info(f"Loaded {len(self.alert_rules)} alert rules")
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        for channel_id, channel_config in self.config['notifications'].items():
            if channel_config.get('enabled', False):
                channel = NotificationChannel(
                    channel_id=channel_id,
                    name=channel_config.get('name', channel_id),
                    type=channel_id,
                    config=channel_config,
                    enabled=True,
                    severity_filter=[AlertSeverity(s) for s in channel_config.get('severity_filter', ['warning', 'critical', 'emergency'])],
                    tags_filter=channel_config.get('tags_filter', {}),
                    rate_limit_per_hour=channel_config.get('rate_limit_per_hour', 10),
                    escalation_delay_minutes=channel_config.get('escalation_delay_minutes', 15)
                )
                
                self.notification_channels[channel_id] = channel
        
        self.logger.info(f"Setup {len(self.notification_channels)} notification channels")
    
    async def collect_database_metrics(self, instance_id: str, db_config: Dict) -> DatabaseMetrics:
        """Collect comprehensive database metrics"""
        start_time = time.time()
        
        try:
            # Connect to database
            conn = await asyncpg.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                timeout=10
            )
            
            # Collect connection metrics
            connection_stats = await conn.fetchrow("""
                SELECT 
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections,
                    (SELECT count(*) FROM pg_stat_activity) as total_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') as idle_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE wait_event IS NOT NULL) as waiting_connections
            """)
            
            # Collect query performance metrics
            query_stats = await conn.fetchrow("""
                SELECT 
                    COALESCE(sum(calls), 0) as total_queries,
                    COALESCE(avg(mean_time), 0) as avg_query_duration_ms,
                    COALESCE(sum(calls) FILTER (WHERE mean_time > 1000), 0) as slow_queries_count
                FROM pg_stat_statements
                WHERE schemaname IS NULL OR schemaname NOT IN ('information_schema', 'pg_catalog')
            """)
            
            # Collect database statistics
            db_stats = await conn.fetchrow("""
                SELECT 
                    numbackends,
                    xact_commit,
                    xact_rollback,
                    blks_read,
                    blks_hit,
                    tup_returned,
                    tup_fetched,
                    tup_inserted,
                    tup_updated,
                    tup_deleted,
                    deadlocks,
                    blk_read_time,
                    blk_write_time
                FROM pg_stat_database 
                WHERE datname = current_database()
            """)
            
            # Collect replication metrics
            replication_stats = await conn.fetchrow("""
                SELECT 
                    COALESCE(
                        (SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) 
                         WHERE pg_is_in_recovery()), 
                        0
                    ) as replication_lag_seconds,
                    COALESCE(
                        (SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) 
                         FROM pg_stat_replication LIMIT 1), 
                        0
                    ) as replication_lag_bytes
            """)
            
            # Collect table and index sizes
            size_stats = await conn.fetchrow("""
                SELECT 
                    COALESCE(sum(pg_total_relation_size(schemaname||'.'||tablename)), 0) / 1024 / 1024 / 1024 as table_size_gb,
                    COALESCE(sum(pg_indexes_size(schemaname||'.'||tablename)), 0) / 1024 / 1024 / 1024 as index_size_gb
                FROM pg_tables 
                WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
            """)
            
            # Calculate derived metrics
            buffer_hit_ratio = 0
            if db_stats['blks_read'] + db_stats['blks_hit'] > 0:
                buffer_hit_ratio = (db_stats['blks_hit'] / (db_stats['blks_read'] + db_stats['blks_hit'])) * 100
            
            connection_utilization = (connection_stats['total_connections'] / connection_stats['max_connections']) * 100
            
            # Calculate transaction rates
            transaction_rate = db_stats['xact_commit'] + db_stats['xact_rollback']
            rollback_rate = db_stats['xact_rollback'] / max(transaction_rate, 1) * 100
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate I/O metrics
            io_stats = psutil.disk_io_counters()
            reads_per_second = io_stats.read_count if io_stats else 0
            writes_per_second = io_stats.write_count if io_stats else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io_mbps = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024 if net_io else 0
            
            await conn.close()
            
            # Create metrics object
            metrics = DatabaseMetrics(
                timestamp=datetime.now(),
                instance_id=instance_id,
                total_connections=connection_stats['total_connections'],
                active_connections=connection_stats['active_connections'],
                idle_connections=connection_stats['idle_connections'],
                waiting_connections=connection_stats['waiting_connections'],
                max_connections=connection_stats['max_connections'],
                connection_utilization=connection_utilization,
                queries_per_second=query_stats['total_queries'] if query_stats else 0,
                avg_query_duration_ms=query_stats['avg_query_duration_ms'] if query_stats else 0,
                slow_queries_count=query_stats['slow_queries_count'] if query_stats else 0,
                deadlocks_count=db_stats['deadlocks'],
                lock_waits_count=0,  # Would need additional query
                reads_per_second=reads_per_second,
                writes_per_second=writes_per_second,
                read_latency_ms=db_stats['blk_read_time'],
                write_latency_ms=db_stats['blk_write_time'],
                buffer_hit_ratio=buffer_hit_ratio,
                replication_lag_bytes=replication_stats['replication_lag_bytes'],
                replication_lag_seconds=replication_stats['replication_lag_seconds'],
                wal_segments_count=0,  # Would need additional query
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                disk_io_utilization=0,  # Would need calculation
                network_io_mbps=network_io_mbps,
                table_size_gb=size_stats['table_size_gb'],
                index_size_gb=size_stats['index_size_gb'],
                cache_hit_ratio=buffer_hit_ratio,
                transaction_rate=transaction_rate,
                rollback_rate=rollback_rate,
                custom_metrics={}
            )
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)
            
            # Record monitoring duration
            duration = time.time() - start_time
            self.monitoring_duration_histogram.labels(instance_id=instance_id).observe(duration)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics for {instance_id}: {e}")
            self.monitoring_errors_counter.labels(instance_id=instance_id, error_type='collection_failed').inc()
            
            # Return empty metrics on error
            return DatabaseMetrics(
                timestamp=datetime.now(),
                instance_id=instance_id,
                total_connections=0,
                active_connections=0,
                idle_connections=0,
                waiting_connections=0,
                max_connections=0,
                connection_utilization=0,
                queries_per_second=0,
                avg_query_duration_ms=0,
                slow_queries_count=0,
                deadlocks_count=0,
                lock_waits_count=0,
                reads_per_second=0,
                writes_per_second=0,
                read_latency_ms=0,
                write_latency_ms=0,
                buffer_hit_ratio=0,
                replication_lag_bytes=0,
                replication_lag_seconds=0,
                wal_segments_count=0,
                cpu_usage_percent=0,
                memory_usage_percent=0,
                disk_usage_percent=0,
                disk_io_utilization=0,
                network_io_mbps=0,
                table_size_gb=0,
                index_size_gb=0,
                cache_hit_ratio=0,
                transaction_rate=0,
                rollback_rate=0,
                custom_metrics={}
            )
    
    def _update_prometheus_metrics(self, metrics: DatabaseMetrics):
        """Update Prometheus metrics"""
        instance_id = metrics.instance_id
        
        # Connection metrics
        self.db_connections_gauge.labels(instance_id=instance_id, state='total').set(metrics.total_connections)
        self.db_connections_gauge.labels(instance_id=instance_id, state='active').set(metrics.active_connections)
        self.db_connections_gauge.labels(instance_id=instance_id, state='idle').set(metrics.idle_connections)
        self.db_connections_gauge.labels(instance_id=instance_id, state='waiting').set(metrics.waiting_connections)
        
        # Query metrics
        self.db_query_duration_histogram.labels(instance_id=instance_id, query_type='average').observe(metrics.avg_query_duration_ms / 1000)
        
        # Buffer hit ratio
        self.db_buffer_hit_ratio_gauge.labels(instance_id=instance_id).set(metrics.buffer_hit_ratio)
        
        # Replication lag
        self.db_replication_lag_gauge.labels(instance_id=instance_id).set(metrics.replication_lag_seconds)
        
        # System metrics
        self.db_cpu_usage_gauge.labels(instance_id=instance_id).set(metrics.cpu_usage_percent)
        self.db_memory_usage_gauge.labels(instance_id=instance_id).set(metrics.memory_usage_percent)
        self.db_disk_usage_gauge.labels(instance_id=instance_id).set(metrics.disk_usage_percent)
        
        # Performance score
        performance_score = self._calculate_performance_score(metrics)
        self.performance_score_gauge.labels(instance_id=instance_id).set(performance_score)
    
    def _calculate_performance_score(self, metrics: DatabaseMetrics) -> float:
        """Calculate performance score (0-100)"""
        score = 100.0
        
        # Connection utilization penalty
        if metrics.connection_utilization > 90:
            score -= 25
        elif metrics.connection_utilization > 80:
            score -= 15
        elif metrics.connection_utilization > 70:
            score -= 5
        
        # Query performance penalty
        if metrics.avg_query_duration_ms > 1000:
            score -= 30
        elif metrics.avg_query_duration_ms > 500:
            score -= 20
        elif metrics.avg_query_duration_ms > 100:
            score -= 10
        
        # Buffer hit ratio penalty
        if metrics.buffer_hit_ratio < 80:
            score -= 25
        elif metrics.buffer_hit_ratio < 90:
            score -= 15
        elif metrics.buffer_hit_ratio < 95:
            score -= 5
        
        # System resource penalties
        if metrics.cpu_usage_percent > 90:
            score -= 20
        elif metrics.cpu_usage_percent > 80:
            score -= 10
        
        if metrics.memory_usage_percent > 90:
            score -= 20
        elif metrics.memory_usage_percent > 80:
            score -= 10
        
        # Replication lag penalty
        if metrics.replication_lag_seconds > 60:
            score -= 25
        elif metrics.replication_lag_seconds > 30:
            score -= 15
        elif metrics.replication_lag_seconds > 10:
            score -= 5
        
        return max(score, 0.0)
    
    async def evaluate_alerts(self, metrics: DatabaseMetrics):
        """Evaluate alert rules against current metrics"""
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Get metric value
                metric_value = getattr(metrics, rule.metric_name, 0)
                
                # Evaluate rule
                should_alert = self.alert_evaluator.evaluate_rule(rule, metric_value)
                
                if should_alert:
                    await self._create_alert(rule, metrics, metric_value)
                else:
                    await self._resolve_alert(rule_id, metrics.instance_id)
                    
            except Exception as e:
                self.logger.error(f"Alert evaluation failed for rule {rule_id}: {e}")
    
    async def _create_alert(self, rule: AlertRule, metrics: DatabaseMetrics, metric_value: float):
        """Create or update an alert"""
        alert_key = f"{rule.rule_id}_{metrics.instance_id}"
        
        # Check if alert already exists
        if alert_key in self.active_alerts:
            # Update existing alert
            existing_alert = self.active_alerts[alert_key]
            existing_alert.metric_value = metric_value
            existing_alert.timestamp = datetime.now()
            return
        
        # Create new alert
        alert_id = f"{rule.rule_id}_{metrics.instance_id}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            title=rule.name,
            description=f"{rule.description} - Current value: {metric_value:.2f}, Threshold: {rule.threshold:.2f}",
            timestamp=datetime.now(),
            resolved_timestamp=None,
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold,
            tags={**rule.tags, 'instance_id': metrics.instance_id},
            runbook_url=rule.runbook_url,
            acknowledgment_user=None,
            acknowledgment_timestamp=None,
            resolution_reason=None,
            escalation_level=0,
            notification_sent=False
        )
        
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Update metrics
        self.alerts_total_counter.labels(
            instance_id=metrics.instance_id,
            severity=rule.severity.value,
            rule_id=rule.rule_id
        ).inc()
        
        self.alerts_active_gauge.labels(
            instance_id=metrics.instance_id,
            severity=rule.severity.value
        ).inc()
        
        # Send notification
        await self.notification_manager.send_alert_notification(alert)
        
        # Log alert
        self.logger.warning(f"ALERT [{rule.severity.value.upper()}] {metrics.instance_id}: {alert.description}")
        
        # Trigger auto-remediation if configured
        if self.auto_remediation:
            await self.auto_remediation.trigger_remediation(alert, metrics)
    
    async def _resolve_alert(self, rule_id: str, instance_id: str):
        """Resolve an alert"""
        alert_key = f"{rule_id}_{instance_id}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            
            # Check if rule supports auto-resolve
            rule = self.alert_rules.get(rule_id)
            if rule and rule.auto_resolve:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_timestamp = datetime.now()
                alert.resolution_reason = "Auto-resolved: metric returned to normal"
                
                # Remove from active alerts
                del self.active_alerts[alert_key]
                
                # Update metrics
                self.alerts_active_gauge.labels(
                    instance_id=instance_id,
                    severity=alert.severity.value
                ).dec()
                
                self.logger.info(f"RESOLVED [{alert.severity.value.upper()}] {instance_id}: {alert.title}")
    
    async def run_health_checks(self):
        """Run configured health checks"""
        for instance_id, db_config in self.config['databases'].items():
            try:
                health_status = await self.health_checker.check_instance_health(instance_id, db_config)
                
                # Update metrics
                for check_name, success in health_status.items():
                    self.health_check_success_gauge.labels(
                        instance_id=instance_id,
                        check_name=check_name
                    ).set(1 if success else 0)
                
            except Exception as e:
                self.logger.error(f"Health check failed for {instance_id}: {e}")
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Collect metrics for all database instances
                all_metrics = {}
                
                for instance_id, db_config in self.config['databases'].items():
                    metrics = await self.collect_database_metrics(instance_id, db_config)
                    all_metrics[instance_id] = metrics
                    
                    # Store metrics
                    self.metrics_history.append(metrics)
                    self.current_metrics = metrics
                    
                    # Cache in Redis
                    if self.redis_client:
                        try:
                            self.redis_client.setex(
                                f'db_metrics:{instance_id}',
                                300,  # 5 minutes expiry
                                json.dumps(asdict(metrics), default=str)
                            )
                        except Exception as e:
                            self.logger.warning(f"Redis cache failed: {e}")
                    
                    # Evaluate alerts
                    await self.evaluate_alerts(metrics)
                
                # Run health checks
                await self.run_health_checks()
                
                # Log summary
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    self._log_monitoring_summary(all_metrics)
                
                # Calculate sleep time
                execution_time = time.time() - start_time
                sleep_time = max(0, self.config['monitoring']['interval_seconds'] - execution_time)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config['monitoring']['interval_seconds'])
    
    def _log_monitoring_summary(self, all_metrics: Dict[str, DatabaseMetrics]):
        """Log monitoring summary"""
        total_alerts = len(self.active_alerts)
        critical_alerts = len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL])
        
        self.logger.info(f"Monitoring Summary - Active Alerts: {total_alerts} (Critical: {critical_alerts})")
        
        for instance_id, metrics in all_metrics.items():
            performance_score = self._calculate_performance_score(metrics)
            self.logger.info(
                f"Instance {instance_id}: "
                f"Performance Score: {performance_score:.1f}, "
                f"Connections: {metrics.active_connections}/{metrics.max_connections}, "
                f"Latency: {metrics.avg_query_duration_ms:.2f}ms, "
                f"Buffer Hit: {metrics.buffer_hit_ratio:.1f}%, "
                f"Replication Lag: {metrics.replication_lag_seconds:.1f}s"
            )
    
    async def start_monitoring(self):
        """Start the monitoring system"""
        self.is_running = True
        
        # Start Prometheus metrics server
        if self.config['prometheus']['enabled']:
            start_http_server(self.config['prometheus']['port'])
            self.logger.info(f"Prometheus metrics server started on port {self.config['prometheus']['port']}")
        
        self.logger.info("Starting real-time database monitoring system")
        
        # Start monitoring loop
        await self.monitoring_loop()
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        
        # Close connections
        if self.redis_client:
            self.redis_client.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Real-time monitoring system stopped")
    
    def get_monitoring_status(self) -> Dict:
        """Get comprehensive monitoring status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_enabled": self.is_running,
            "total_metrics_collected": len(self.metrics_history),
            "active_alerts": len(self.active_alerts),
            "alert_breakdown": {
                severity.value: len([a for a in self.active_alerts.values() if a.severity == severity])
                for severity in AlertSeverity
            },
            "notification_channels": len(self.notification_channels),
            "health_checks_enabled": self.config['health_checks']['enabled'],
            "auto_remediation_enabled": self.config['auto_remediation']['enabled'],
            "recent_alerts": [asdict(alert) for alert in list(self.alert_history)[-10:]],
            "current_metrics": asdict(self.current_metrics) if self.current_metrics else None
        }


class AlertEvaluator:
    """Evaluates alert rules against metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.rule_state = {}
    
    def evaluate_rule(self, rule: AlertRule, metric_value: float) -> bool:
        """Evaluate a single alert rule"""
        try:
            # Perform comparison
            if rule.comparison == '>':
                condition_met = metric_value > rule.threshold
            elif rule.comparison == '<':
                condition_met = metric_value < rule.threshold
            elif rule.comparison == '>=':
                condition_met = metric_value >= rule.threshold
            elif rule.comparison == '<=':
                condition_met = metric_value <= rule.threshold
            elif rule.comparison == '==':
                condition_met = metric_value == rule.threshold
            elif rule.comparison == '!=':
                condition_met = metric_value != rule.threshold
            else:
                self.logger.error(f"Unknown comparison operator: {rule.comparison}")
                return False
            
            # Check duration requirement
            current_time = time.time()
            
            if condition_met:
                # Start tracking if not already
                if rule.rule_id not in self.rule_state:
                    self.rule_state[rule.rule_id] = current_time
                
                # Check if duration requirement is met
                if current_time - self.rule_state[rule.rule_id] >= rule.duration_seconds:
                    return True
            else:
                # Reset state if condition is not met
                if rule.rule_id in self.rule_state:
                    del self.rule_state[rule.rule_id]
            
            return False
            
        except Exception as e:
            self.logger.error(f"Rule evaluation error: {e}")
            return False


class NotificationManager:
    """Manages alert notifications"""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.notification_history = deque(maxlen=10000)
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
    
    async def send_alert_notification(self, alert: Alert):
        """Send alert notification to all configured channels"""
        try:
            # Check rate limits and send notifications
            for channel_type, channel_config in self.config.items():
                if not channel_config.get('enabled', False):
                    continue
                
                # Check severity filter
                severity_filter = channel_config.get('severity_filter', ['warning', 'critical', 'emergency'])
                if alert.severity.value not in severity_filter:
                    continue
                
                # Check rate limit
                if self._check_rate_limit(channel_type, channel_config):
                    if channel_type == 'email':
                        await self._send_email_notification(alert, channel_config)
                    elif channel_type == 'slack':
                        await self._send_slack_notification(alert, channel_config)
                    elif channel_type == 'webhook':
                        await self._send_webhook_notification(alert, channel_config)
                    
                    self._record_notification(channel_type, alert)
                else:
                    self.logger.warning(f"Rate limit exceeded for {channel_type} notifications")
            
        except Exception as e:
            self.logger.error(f"Notification sending failed: {e}")
    
    def _check_rate_limit(self, channel_type: str, config: Dict) -> bool:
        """Check if rate limit allows sending notification"""
        current_time = time.time()
        rate_limit = config.get('rate_limit_per_hour', 10)
        
        # Clean old entries
        channel_history = self.rate_limits[channel_type]
        cutoff_time = current_time - 3600  # 1 hour ago
        
        while channel_history and channel_history[0] < cutoff_time:
            channel_history.popleft()
        
        return len(channel_history) < rate_limit
    
    def _record_notification(self, channel_type: str, alert: Alert):
        """Record notification in history"""
        self.rate_limits[channel_type].append(time.time())
        
        notification_record = {
            'timestamp': datetime.now().isoformat(),
            'channel_type': channel_type,
            'alert_id': alert.alert_id,
            'severity': alert.severity.value,
            'title': alert.title
        }
        
        self.notification_history.append(notification_record)
    
    async def _send_email_notification(self, alert: Alert, config: Dict):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = config['from_address']
            msg['To'] = ', '.join(config['to_addresses'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] Database Alert: {alert.title}"
            
            body = f"""
            Database Alert Notification
            
            Severity: {alert.severity.value.upper()}
            Title: {alert.title}
            Description: {alert.description}
            Instance: {alert.tags.get('instance_id', 'unknown')}
            Timestamp: {alert.timestamp.isoformat()}
            Metric: {alert.metric_name} = {alert.metric_value:.2f}
            Threshold: {alert.threshold:.2f}
            
            Runbook: {alert.runbook_url}
            
            Alert ID: {alert.alert_id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
    
    async def _send_slack_notification(self, alert: Alert, config: Dict):
        """Send Slack notification"""
        try:
            color = {
                AlertSeverity.INFO: 'good',
                AlertSeverity.WARNING: 'warning',
                AlertSeverity.CRITICAL: 'danger',
                AlertSeverity.EMERGENCY: 'danger'
            }.get(alert.severity, 'warning')
            
            payload = {
                "channel": config['channel'],
                "username": "Database Monitor",
                "icon_emoji": ":warning:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"Database Alert: {alert.title}",
                        "text": alert.description,
                        "fields": [
                            {
                                "title": "Instance",
                                "value": alert.tags.get('instance_id', 'unknown'),
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Metric",
                                "value": f"{alert.metric_name} = {alert.metric_value:.2f}",
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": f"{alert.threshold:.2f}",
                                "short": True
                            }
                        ],
                        "footer": f"Alert ID: {alert.alert_id}",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config['webhook_url'], json=payload) as response:
                    if response.status == 200:
                        self.logger.info(f"Slack notification sent for alert {alert.alert_id}")
                    else:
                        self.logger.error(f"Slack notification failed: {response.status}")
            
        except Exception as e:
            self.logger.error(f"Slack notification failed: {e}")
    
    async def _send_webhook_notification(self, alert: Alert, config: Dict):
        """Send webhook notification"""
        try:
            payload = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "title": alert.title,
                "description": alert.description,
                "timestamp": alert.timestamp.isoformat(),
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "tags": alert.tags,
                "runbook_url": alert.runbook_url
            }
            
            headers = config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(config['url'], json=payload, headers=headers) as response:
                    if response.status == 200:
                        self.logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                    else:
                        self.logger.error(f"Webhook notification failed: {response.status}")
            
        except Exception as e:
            self.logger.error(f"Webhook notification failed: {e}")


class HealthChecker:
    """Performs health checks on database instances"""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    async def check_instance_health(self, instance_id: str, db_config: Dict) -> Dict[str, bool]:
        """Check health of a database instance"""
        health_status = {}
        
        if not self.config.get('enabled', True):
            return health_status
        
        try:
            # Connect to database
            conn = await asyncpg.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                timeout=self.config.get('timeout_seconds', 10)
            )
            
            # Run configured health checks
            for check_config in self.config.get('checks', []):
                try:
                    check_name = check_config['name']
                    check_type = check_config['type']
                    
                    if check_type == 'sql':
                        query = check_config['query']
                        result = await conn.fetchval(query)
                        
                        # Check expected result if provided
                        if 'expected_result' in check_config:
                            health_status[check_name] = result == check_config['expected_result']
                        else:
                            health_status[check_name] = result is not None
                    
                    else:
                        self.logger.warning(f"Unknown health check type: {check_type}")
                        health_status[check_name] = False
                        
                except Exception as e:
                    self.logger.error(f"Health check '{check_name}' failed for {instance_id}: {e}")
                    health_status[check_name] = False
            
            await conn.close()
            
        except Exception as e:
            self.logger.error(f"Health check connection failed for {instance_id}: {e}")
            # Mark all checks as failed
            for check_config in self.config.get('checks', []):
                health_status[check_config['name']] = False
        
        return health_status


class AutoRemediationEngine:
    """Automated remediation for common issues"""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.last_action_time = {}
    
    async def trigger_remediation(self, alert: Alert, metrics: DatabaseMetrics):
        """Trigger automated remediation for an alert"""
        if not self.config.get('enabled', False):
            return
        
        try:
            # Find applicable remediation actions
            for action_name, action_config in self.config.get('actions', {}).items():
                if not action_config.get('enabled', False):
                    continue
                
                # Check if alert rule is in triggers
                if alert.rule_id in action_config.get('triggers', []):
                    # Check cooldown
                    if self._check_cooldown(action_name, action_config):
                        await self._execute_remediation(action_name, action_config, alert, metrics)
                    else:
                        self.logger.info(f"Remediation action '{action_name}' in cooldown")
        
        except Exception as e:
            self.logger.error(f"Auto-remediation failed: {e}")
    
    def _check_cooldown(self, action_name: str, action_config: Dict) -> bool:
        """Check if action is in cooldown period"""
        current_time = time.time()
        last_action = self.last_action_time.get(action_name, 0)
        cooldown_seconds = action_config.get('cooldown_seconds', 300)
        
        return current_time - last_action >= cooldown_seconds
    
    async def _execute_remediation(self, action_name: str, action_config: Dict, alert: Alert, metrics: DatabaseMetrics):
        """Execute remediation action"""
        try:
            command = action_config['command']
            
            self.logger.info(f"Executing remediation action '{action_name}' for alert {alert.alert_id}")
            
            # This is a placeholder - implement actual remediation logic
            if action_name == 'restart_connections':
                # Implement connection restart logic
                pass
            elif action_name == 'clear_cache':
                # Implement cache clear logic
                pass
            
            # Record action time
            self.last_action_time[action_name] = time.time()
            
            self.logger.info(f"Remediation action '{action_name}' completed")
            
        except Exception as e:
            self.logger.error(f"Remediation action '{action_name}' failed: {e}")


class BaselineCalculator:
    """Calculates performance baselines"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def calculate_baseline(self, metrics_history: List[DatabaseMetrics]) -> Dict[str, float]:
        """Calculate baseline metrics"""
        # Implementation for baseline calculation
        pass


class AnomalyDetector:
    """Detects anomalies in metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def detect_anomalies(self, metrics: DatabaseMetrics, baseline: Dict[str, float]) -> List[str]:
        """Detect anomalies in current metrics"""
        # Implementation for anomaly detection
        pass


async def main():
    """Main entry point"""
    monitoring_system = RealTimeMonitoringSystem()
    
    try:
        await monitoring_system.start_monitoring()
    except KeyboardInterrupt:
        print("\nShutting down monitoring system...")
        monitoring_system.stop_monitoring()
    except Exception as e:
        print(f"Error: {e}")
        monitoring_system.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())