#!/usr/bin/env python3
"""
Comprehensive Database Performance Monitor
AGENT 14: DATABASE OPTIMIZATION SPECIALIST
Focus: Real-time performance monitoring, alerting, and optimization recommendations
"""

import asyncio
import asyncpg
import psycopg2
import psutil
import time
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from collections import deque, defaultdict
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import subprocess
import redis
from statistics import median, mean, stdev

@dataclass
class DatabaseMetrics:
    """Comprehensive database metrics"""
    timestamp: datetime
    
    # Connection metrics
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_connections: int
    max_connections: int
    connection_utilization: float
    
    # Query performance
    queries_per_second: float
    avg_query_time_ms: float
    slow_queries_count: int
    query_cache_hit_ratio: float
    
    # Transaction metrics
    transactions_per_second: float
    avg_transaction_time_ms: float
    deadlocks_count: int
    lock_waits_count: int
    
    # I/O metrics
    blocks_read: int
    blocks_hit: int
    buffer_hit_ratio: float
    checkpoint_write_time_ms: float
    wal_write_time_ms: float
    
    # Replication metrics
    replication_lag_bytes: int
    replication_lag_seconds: float
    wal_segments_count: int
    
    # System metrics
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_mb: float
    
    # Table metrics
    table_size_mb: float
    index_size_mb: float
    bloat_ratio: float
    vacuum_stats: Dict[str, Any]
    
    # Lock metrics
    lock_count: int
    lock_wait_time_ms: float
    exclusive_locks: int
    shared_locks: int

@dataclass
class PerformanceAlert:
    """Performance alert information"""
    alert_id: str
    timestamp: datetime
    severity: str  # info, warning, critical
    category: str
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    recommendations: List[str]
    auto_resolution: bool

@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    category: str
    priority: str  # low, medium, high, critical
    title: str
    description: str
    impact_score: float
    implementation_effort: str  # low, medium, high
    estimated_improvement: str
    sql_commands: List[str]
    rollback_plan: str
    validation_queries: List[str]

class DatabasePerformanceMonitor:
    """
    Comprehensive database performance monitoring system
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Metrics storage
        self.metrics_history = deque(maxlen=3600)  # 1 hour at 1-second intervals
        self.current_metrics = None
        
        # Alert system
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        
        # Performance baselines
        self.performance_baselines = {}
        self.anomaly_detection_data = defaultdict(deque)
        
        # Connection pools
        self.connection_pool = None
        self.monitoring_connection = None
        
        # Setup Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Redis for caching (optional)
        self.redis_client = self._setup_redis()
        
        # Performance analytics database
        self.analytics_db = self._setup_analytics_db()
        
        # Background tasks
        self.monitoring_tasks = []
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "database": {
                "host": "127.0.0.1",
                "port": 5432,
                "database": "grandmodel",
                "user": "monitoring_user",
                "password": "monitoring_password",
                "min_connections": 5,
                "max_connections": 20
            },
            "monitoring": {
                "interval_seconds": 1,
                "detailed_interval_seconds": 30,
                "metrics_retention_hours": 24,
                "alert_check_interval": 5,
                "baseline_calculation_interval": 300
            },
            "thresholds": {
                "connection_utilization_warning": 70,
                "connection_utilization_critical": 90,
                "query_time_warning_ms": 100,
                "query_time_critical_ms": 1000,
                "buffer_hit_ratio_warning": 90,
                "buffer_hit_ratio_critical": 80,
                "replication_lag_warning_mb": 10,
                "replication_lag_critical_mb": 100,
                "cpu_usage_warning": 80,
                "cpu_usage_critical": 95,
                "memory_usage_warning": 85,
                "memory_usage_critical": 95,
                "disk_usage_warning": 80,
                "disk_usage_critical": 90,
                "lock_wait_time_warning_ms": 1000,
                "lock_wait_time_critical_ms": 5000
            },
            "redis": {
                "host": "127.0.0.1",
                "port": 6379,
                "db": 1,
                "password": None
            },
            "prometheus": {
                "port": 9091,
                "enabled": True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('db_performance_monitor')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('/var/log/db_performance_monitor.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Database connection metrics
        self.db_connections = Gauge(
            'db_connections_total',
            'Total database connections',
            ['state', 'database']
        )
        
        self.db_connection_utilization = Gauge(
            'db_connection_utilization_percent',
            'Database connection utilization percentage',
            ['database']
        )
        
        # Query performance metrics
        self.db_queries_per_second = Gauge(
            'db_queries_per_second',
            'Database queries per second',
            ['database']
        )
        
        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['database', 'query_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.db_slow_queries = Counter(
            'db_slow_queries_total',
            'Total slow queries',
            ['database', 'threshold']
        )
        
        # Transaction metrics
        self.db_transactions_per_second = Gauge(
            'db_transactions_per_second',
            'Database transactions per second',
            ['database']
        )
        
        self.db_deadlocks = Counter(
            'db_deadlocks_total',
            'Total database deadlocks',
            ['database']
        )
        
        # I/O metrics
        self.db_buffer_hit_ratio = Gauge(
            'db_buffer_hit_ratio_percent',
            'Database buffer hit ratio',
            ['database']
        )
        
        self.db_checkpoint_duration = Histogram(
            'db_checkpoint_duration_seconds',
            'Database checkpoint duration',
            ['database']
        )
        
        # Replication metrics
        self.db_replication_lag = Gauge(
            'db_replication_lag_bytes',
            'Database replication lag in bytes',
            ['master', 'replica']
        )
        
        self.db_replication_lag_seconds = Gauge(
            'db_replication_lag_seconds',
            'Database replication lag in seconds',
            ['master', 'replica']
        )
        
        # System metrics
        self.db_cpu_usage = Gauge(
            'db_cpu_usage_percent',
            'Database server CPU usage',
            ['instance']
        )
        
        self.db_memory_usage = Gauge(
            'db_memory_usage_percent',
            'Database server memory usage',
            ['instance']
        )
        
        self.db_disk_usage = Gauge(
            'db_disk_usage_percent',
            'Database server disk usage',
            ['instance', 'mount_point']
        )
        
        # Lock metrics
        self.db_locks = Gauge(
            'db_locks_total',
            'Total database locks',
            ['database', 'lock_type']
        )
        
        self.db_lock_wait_time = Histogram(
            'db_lock_wait_time_seconds',
            'Database lock wait time',
            ['database']
        )
        
        # Alert metrics
        self.db_alerts = Gauge(
            'db_alerts_active',
            'Active database alerts',
            ['severity', 'category']
        )
        
        # Performance score
        self.db_performance_score = Gauge(
            'db_performance_score',
            'Overall database performance score (0-100)',
            ['database']
        )
    
    def _setup_redis(self) -> Optional[redis.Redis]:
        """Setup Redis client for caching"""
        try:
            redis_config = self.config['redis']
            client = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db'],
                password=redis_config['password'],
                decode_responses=True
            )
            
            # Test connection
            client.ping()
            return client
            
        except Exception as e:
            self.logger.warning(f"Redis setup failed: {e}")
            return None
    
    def _setup_analytics_db(self) -> sqlite3.Connection:
        """Setup analytics database"""
        conn = sqlite3.connect('/var/lib/db_analytics.db')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp INTEGER PRIMARY KEY,
                metrics_json TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                severity TEXT,
                category TEXT,
                metric_name TEXT,
                current_value REAL,
                threshold_value REAL,
                message TEXT,
                resolved INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_recommendations (
                recommendation_id TEXT PRIMARY KEY,
                timestamp INTEGER,
                category TEXT,
                priority TEXT,
                title TEXT,
                description TEXT,
                impact_score REAL,
                implementation_effort TEXT,
                applied INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        return conn
    
    async def create_connection_pool(self):
        """Create monitoring connection pool"""
        try:
            db_config = self.config['database']
            
            self.connection_pool = await asyncpg.create_pool(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                min_size=db_config['min_connections'],
                max_size=db_config['max_connections'],
                command_timeout=30,
                server_settings={
                    'application_name': 'db_performance_monitor'
                }
            )
            
            self.logger.info("Database connection pool created")
            
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise
    
    async def collect_database_metrics(self) -> DatabaseMetrics:
        """Collect comprehensive database metrics"""
        try:
            async with self.connection_pool.acquire() as conn:
                # Get connection statistics
                conn_stats = await conn.fetchrow("""
                    SELECT 
                        setting::int as max_connections,
                        (SELECT count(*) FROM pg_stat_activity) as total_connections,
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') as idle_connections,
                        (SELECT count(*) FROM pg_stat_activity WHERE wait_event IS NOT NULL) as waiting_connections
                    FROM pg_settings WHERE name = 'max_connections'
                """)
                
                # Calculate connection utilization
                connection_utilization = (conn_stats['total_connections'] / conn_stats['max_connections']) * 100
                
                # Get database statistics
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
                        conflicts,
                        temp_files,
                        temp_bytes,
                        deadlocks,
                        checksum_failures,
                        checksum_last_failure,
                        blk_read_time,
                        blk_write_time,
                        stats_reset
                    FROM pg_stat_database 
                    WHERE datname = $1
                """, self.config['database']['database'])
                
                # Calculate buffer hit ratio
                total_blocks = db_stats['blks_read'] + db_stats['blks_hit']
                buffer_hit_ratio = (db_stats['blks_hit'] / max(total_blocks, 1)) * 100
                
                # Get replication statistics
                replication_stats = await conn.fetch("""
                    SELECT 
                        application_name,
                        state,
                        sent_lsn,
                        write_lsn,
                        flush_lsn,
                        replay_lsn,
                        write_lag,
                        flush_lag,
                        replay_lag,
                        sync_priority,
                        sync_state,
                        pg_wal_lsn_diff(sent_lsn, replay_lsn) as lag_bytes
                    FROM pg_stat_replication
                """)
                
                # Get system metrics
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Get lock statistics
                lock_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_locks,
                        COUNT(*) FILTER (WHERE mode LIKE '%ExclusiveLock%') as exclusive_locks,
                        COUNT(*) FILTER (WHERE mode LIKE '%ShareLock%') as shared_locks
                    FROM pg_locks
                """)
                
                # Get table and index sizes
                size_stats = await conn.fetchrow("""
                    SELECT 
                        pg_size_pretty(pg_database_size(current_database())) as database_size,
                        SUM(pg_total_relation_size(schemaname||'.'||tablename)) as total_table_size,
                        SUM(pg_indexes_size(schemaname||'.'||tablename)) as total_index_size
                    FROM pg_tables 
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                """)
                
                # Calculate metrics
                current_time = datetime.now()
                
                # For time-based metrics, we need to compare with previous values
                if hasattr(self, 'previous_stats') and self.previous_stats:
                    time_diff = (current_time - self.previous_stats['timestamp']).total_seconds()
                    
                    queries_per_second = (
                        (db_stats['tup_returned'] - self.previous_stats['tup_returned']) / time_diff
                    ) if time_diff > 0 else 0
                    
                    transactions_per_second = (
                        (db_stats['xact_commit'] - self.previous_stats['xact_commit']) / time_diff
                    ) if time_diff > 0 else 0
                else:
                    queries_per_second = 0
                    transactions_per_second = 0
                
                # Store current stats for next iteration
                self.previous_stats = {
                    'timestamp': current_time,
                    'tup_returned': db_stats['tup_returned'],
                    'xact_commit': db_stats['xact_commit']
                }
                
                # Calculate replication lag
                replication_lag_bytes = 0
                replication_lag_seconds = 0.0
                
                if replication_stats:
                    replication_lag_bytes = max(
                        (rep['lag_bytes'] or 0) for rep in replication_stats
                    )
                    replication_lag_seconds = max(
                        (rep['replay_lag'].total_seconds() if rep['replay_lag'] else 0) 
                        for rep in replication_stats
                    )
                
                # Create metrics object
                metrics = DatabaseMetrics(
                    timestamp=current_time,
                    total_connections=conn_stats['total_connections'],
                    active_connections=conn_stats['active_connections'],
                    idle_connections=conn_stats['idle_connections'],
                    waiting_connections=conn_stats['waiting_connections'],
                    max_connections=conn_stats['max_connections'],
                    connection_utilization=connection_utilization,
                    queries_per_second=queries_per_second,
                    avg_query_time_ms=0.0,  # Would need additional tracking
                    slow_queries_count=0,  # Would need additional tracking
                    query_cache_hit_ratio=buffer_hit_ratio,
                    transactions_per_second=transactions_per_second,
                    avg_transaction_time_ms=0.0,  # Would need additional tracking
                    deadlocks_count=db_stats['deadlocks'],
                    lock_waits_count=0,  # Would need additional tracking
                    blocks_read=db_stats['blks_read'],
                    blocks_hit=db_stats['blks_hit'],
                    buffer_hit_ratio=buffer_hit_ratio,
                    checkpoint_write_time_ms=db_stats['blk_write_time'],
                    wal_write_time_ms=0.0,  # Would need additional tracking
                    replication_lag_bytes=replication_lag_bytes,
                    replication_lag_seconds=replication_lag_seconds,
                    wal_segments_count=0,  # Would need additional tracking
                    cpu_usage_percent=cpu_usage,
                    memory_usage_percent=memory.percent,
                    disk_usage_percent=disk.percent,
                    disk_io_read_mb=0.0,  # Would need additional tracking
                    disk_io_write_mb=0.0,  # Would need additional tracking
                    network_io_mb=0.0,  # Would need additional tracking
                    table_size_mb=size_stats['total_table_size'] / (1024 * 1024) if size_stats['total_table_size'] else 0,
                    index_size_mb=size_stats['total_index_size'] / (1024 * 1024) if size_stats['total_index_size'] else 0,
                    bloat_ratio=0.0,  # Would need additional calculation
                    vacuum_stats={},  # Would need additional tracking
                    lock_count=lock_stats['total_locks'],
                    lock_wait_time_ms=0.0,  # Would need additional tracking
                    exclusive_locks=lock_stats['exclusive_locks'],
                    shared_locks=lock_stats['shared_locks']
                )
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(metrics)
                
                # Store metrics
                self.metrics_history.append(metrics)
                self.current_metrics = metrics
                
                # Store in analytics database
                self._store_metrics_in_analytics_db(metrics)
                
                # Cache in Redis
                if self.redis_client:
                    try:
                        self.redis_client.setex(
                            'db_metrics:latest',
                            60,  # 1 minute expiry
                            json.dumps(asdict(metrics), default=str)
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to cache metrics in Redis: {e}")
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to collect database metrics: {e}")
            raise
    
    def _update_prometheus_metrics(self, metrics: DatabaseMetrics):
        """Update Prometheus metrics"""
        try:
            database_name = self.config['database']['database']
            
            # Connection metrics
            self.db_connections.labels(state='total', database=database_name).set(metrics.total_connections)
            self.db_connections.labels(state='active', database=database_name).set(metrics.active_connections)
            self.db_connections.labels(state='idle', database=database_name).set(metrics.idle_connections)
            self.db_connections.labels(state='waiting', database=database_name).set(metrics.waiting_connections)
            self.db_connection_utilization.labels(database=database_name).set(metrics.connection_utilization)
            
            # Query metrics
            self.db_queries_per_second.labels(database=database_name).set(metrics.queries_per_second)
            self.db_transactions_per_second.labels(database=database_name).set(metrics.transactions_per_second)
            
            # I/O metrics
            self.db_buffer_hit_ratio.labels(database=database_name).set(metrics.buffer_hit_ratio)
            
            # Replication metrics
            if metrics.replication_lag_bytes > 0:
                self.db_replication_lag.labels(master='primary', replica='standby').set(metrics.replication_lag_bytes)
                self.db_replication_lag_seconds.labels(master='primary', replica='standby').set(metrics.replication_lag_seconds)
            
            # System metrics
            self.db_cpu_usage.labels(instance='primary').set(metrics.cpu_usage_percent)
            self.db_memory_usage.labels(instance='primary').set(metrics.memory_usage_percent)
            self.db_disk_usage.labels(instance='primary', mount_point='/').set(metrics.disk_usage_percent)
            
            # Lock metrics
            self.db_locks.labels(database=database_name, lock_type='total').set(metrics.lock_count)
            self.db_locks.labels(database=database_name, lock_type='exclusive').set(metrics.exclusive_locks)
            self.db_locks.labels(database=database_name, lock_type='shared').set(metrics.shared_locks)
            
            # Calculate and set performance score
            performance_score = self._calculate_performance_score(metrics)
            self.db_performance_score.labels(database=database_name).set(performance_score)
            
        except Exception as e:
            self.logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def _calculate_performance_score(self, metrics: DatabaseMetrics) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            score = 100.0
            
            # Connection utilization penalty
            if metrics.connection_utilization > 90:
                score -= 20
            elif metrics.connection_utilization > 70:
                score -= 10
            
            # Buffer hit ratio penalty
            if metrics.buffer_hit_ratio < 80:
                score -= 30
            elif metrics.buffer_hit_ratio < 90:
                score -= 15
            
            # CPU usage penalty
            if metrics.cpu_usage_percent > 95:
                score -= 25
            elif metrics.cpu_usage_percent > 80:
                score -= 10
            
            # Memory usage penalty
            if metrics.memory_usage_percent > 95:
                score -= 20
            elif metrics.memory_usage_percent > 85:
                score -= 10
            
            # Replication lag penalty
            if metrics.replication_lag_bytes > 100 * 1024 * 1024:  # 100MB
                score -= 25
            elif metrics.replication_lag_bytes > 10 * 1024 * 1024:  # 10MB
                score -= 10
            
            # Deadlock penalty
            if metrics.deadlocks_count > 0:
                score -= 15
            
            return max(score, 0.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate performance score: {e}")
            return 0.0
    
    def _store_metrics_in_analytics_db(self, metrics: DatabaseMetrics):
        """Store metrics in analytics database"""
        try:
            cursor = self.analytics_db.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO performance_metrics (timestamp, metrics_json) VALUES (?, ?)",
                (int(metrics.timestamp.timestamp()), json.dumps(asdict(metrics), default=str))
            )
            self.analytics_db.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics in analytics DB: {e}")
    
    async def check_performance_alerts(self, metrics: DatabaseMetrics):
        """Check for performance alerts"""
        try:
            thresholds = self.config['thresholds']
            alerts_to_create = []
            
            # Connection utilization alerts
            if metrics.connection_utilization >= thresholds['connection_utilization_critical']:
                alerts_to_create.append(self._create_alert(
                    'critical',
                    'connections',
                    'connection_utilization',
                    metrics.connection_utilization,
                    thresholds['connection_utilization_critical'],
                    f"Connection utilization is critically high: {metrics.connection_utilization:.1f}%",
                    ["Consider increasing max_connections", "Optimize query performance", "Check for connection leaks"]
                ))
            elif metrics.connection_utilization >= thresholds['connection_utilization_warning']:
                alerts_to_create.append(self._create_alert(
                    'warning',
                    'connections',
                    'connection_utilization',
                    metrics.connection_utilization,
                    thresholds['connection_utilization_warning'],
                    f"Connection utilization is high: {metrics.connection_utilization:.1f}%",
                    ["Monitor connection usage", "Consider connection pooling optimization"]
                ))
            
            # Buffer hit ratio alerts
            if metrics.buffer_hit_ratio <= thresholds['buffer_hit_ratio_critical']:
                alerts_to_create.append(self._create_alert(
                    'critical',
                    'performance',
                    'buffer_hit_ratio',
                    metrics.buffer_hit_ratio,
                    thresholds['buffer_hit_ratio_critical'],
                    f"Buffer hit ratio is critically low: {metrics.buffer_hit_ratio:.1f}%",
                    ["Increase shared_buffers", "Optimize queries", "Add more memory"]
                ))
            elif metrics.buffer_hit_ratio <= thresholds['buffer_hit_ratio_warning']:
                alerts_to_create.append(self._create_alert(
                    'warning',
                    'performance',
                    'buffer_hit_ratio',
                    metrics.buffer_hit_ratio,
                    thresholds['buffer_hit_ratio_warning'],
                    f"Buffer hit ratio is low: {metrics.buffer_hit_ratio:.1f}%",
                    ["Consider increasing shared_buffers", "Review query performance"]
                ))
            
            # CPU usage alerts
            if metrics.cpu_usage_percent >= thresholds['cpu_usage_critical']:
                alerts_to_create.append(self._create_alert(
                    'critical',
                    'system',
                    'cpu_usage',
                    metrics.cpu_usage_percent,
                    thresholds['cpu_usage_critical'],
                    f"CPU usage is critically high: {metrics.cpu_usage_percent:.1f}%",
                    ["Optimize queries", "Scale vertically", "Check for runaway processes"]
                ))
            elif metrics.cpu_usage_percent >= thresholds['cpu_usage_warning']:
                alerts_to_create.append(self._create_alert(
                    'warning',
                    'system',
                    'cpu_usage',
                    metrics.cpu_usage_percent,
                    thresholds['cpu_usage_warning'],
                    f"CPU usage is high: {metrics.cpu_usage_percent:.1f}%",
                    ["Monitor CPU usage trends", "Review query performance"]
                ))
            
            # Memory usage alerts
            if metrics.memory_usage_percent >= thresholds['memory_usage_critical']:
                alerts_to_create.append(self._create_alert(
                    'critical',
                    'system',
                    'memory_usage',
                    metrics.memory_usage_percent,
                    thresholds['memory_usage_critical'],
                    f"Memory usage is critically high: {metrics.memory_usage_percent:.1f}%",
                    ["Add more memory", "Optimize shared_buffers", "Check for memory leaks"]
                ))
            elif metrics.memory_usage_percent >= thresholds['memory_usage_warning']:
                alerts_to_create.append(self._create_alert(
                    'warning',
                    'system',
                    'memory_usage',
                    metrics.memory_usage_percent,
                    thresholds['memory_usage_warning'],
                    f"Memory usage is high: {metrics.memory_usage_percent:.1f}%",
                    ["Monitor memory usage trends", "Consider memory optimization"]
                ))
            
            # Replication lag alerts
            replication_lag_mb = metrics.replication_lag_bytes / (1024 * 1024)
            if replication_lag_mb >= thresholds['replication_lag_critical_mb']:
                alerts_to_create.append(self._create_alert(
                    'critical',
                    'replication',
                    'replication_lag',
                    replication_lag_mb,
                    thresholds['replication_lag_critical_mb'],
                    f"Replication lag is critically high: {replication_lag_mb:.1f}MB",
                    ["Check network connectivity", "Optimize replication settings", "Check standby performance"]
                ))
            elif replication_lag_mb >= thresholds['replication_lag_warning_mb']:
                alerts_to_create.append(self._create_alert(
                    'warning',
                    'replication',
                    'replication_lag',
                    replication_lag_mb,
                    thresholds['replication_lag_warning_mb'],
                    f"Replication lag is high: {replication_lag_mb:.1f}MB",
                    ["Monitor replication lag trends", "Check standby performance"]
                ))
            
            # Process alerts
            for alert in alerts_to_create:
                await self._process_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to check performance alerts: {e}")
    
    def _create_alert(self, severity: str, category: str, metric_name: str, 
                     current_value: float, threshold_value: float, 
                     message: str, recommendations: List[str]) -> PerformanceAlert:
        """Create a performance alert"""
        alert_id = f"{category}_{metric_name}_{int(time.time())}"
        
        return PerformanceAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            recommendations=recommendations,
            auto_resolution=False
        )
    
    async def _process_alert(self, alert: PerformanceAlert):
        """Process a performance alert"""
        try:
            # Check if alert already exists
            existing_alert_key = f"{alert.category}_{alert.metric_name}"
            
            if existing_alert_key in self.active_alerts:
                # Update existing alert
                self.active_alerts[existing_alert_key] = alert
            else:
                # Create new alert
                self.active_alerts[existing_alert_key] = alert
                
                # Log alert
                self.logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
                
                # Store in analytics database
                cursor = self.analytics_db.cursor()
                cursor.execute("""
                    INSERT INTO performance_alerts 
                    (alert_id, timestamp, severity, category, metric_name, current_value, threshold_value, message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id,
                    int(alert.timestamp.timestamp()),
                    alert.severity,
                    alert.category,
                    alert.metric_name,
                    alert.current_value,
                    alert.threshold_value,
                    alert.message
                ))
                self.analytics_db.commit()
                
                # Update Prometheus metrics
                self.db_alerts.labels(severity=alert.severity, category=alert.category).inc()
            
            # Add to history
            self.alert_history.append(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to process alert: {e}")
    
    async def monitor_performance(self):
        """Main performance monitoring loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Collect metrics
                metrics = await self.collect_database_metrics()
                
                # Check alerts
                await self.check_performance_alerts(metrics)
                
                # Log performance summary
                if int(time.time()) % 60 == 0:  # Every minute
                    self.logger.info(
                        f"Performance Summary: "
                        f"Connections: {metrics.total_connections}/{metrics.max_connections} "
                        f"({metrics.connection_utilization:.1f}%), "
                        f"Buffer Hit: {metrics.buffer_hit_ratio:.1f}%, "
                        f"CPU: {metrics.cpu_usage_percent:.1f}%, "
                        f"Memory: {metrics.memory_usage_percent:.1f}%, "
                        f"Replication Lag: {metrics.replication_lag_bytes / (1024*1024):.1f}MB"
                    )
                
                # Calculate sleep time
                execution_time = time.time() - start_time
                sleep_time = max(0, self.config['monitoring']['interval_seconds'] - execution_time)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(self.config['monitoring']['interval_seconds'])
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        self.is_running = True
        
        # Create connection pool
        await self.create_connection_pool()
        
        # Start Prometheus metrics server
        if self.config['prometheus']['enabled']:
            start_http_server(self.config['prometheus']['port'])
            self.logger.info(f"Prometheus metrics server started on port {self.config['prometheus']['port']}")
        
        # Start monitoring
        self.logger.info("Starting database performance monitoring")
        await self.monitor_performance()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_running = False
        
        if self.connection_pool:
            self.connection_pool.close()
        
        if self.analytics_db:
            self.analytics_db.close()
        
        if self.redis_client:
            self.redis_client.close()
        
        self.executor.shutdown(wait=True)
        
        self.logger.info("Database performance monitoring stopped")
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        try:
            if not self.current_metrics:
                return {"error": "No metrics available"}
            
            # Calculate performance trends
            recent_metrics = list(self.metrics_history)[-60:]  # Last 60 seconds
            
            if len(recent_metrics) > 1:
                cpu_trend = recent_metrics[-1].cpu_usage_percent - recent_metrics[0].cpu_usage_percent
                memory_trend = recent_metrics[-1].memory_usage_percent - recent_metrics[0].memory_usage_percent
                connection_trend = recent_metrics[-1].connection_utilization - recent_metrics[0].connection_utilization
            else:
                cpu_trend = memory_trend = connection_trend = 0
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "current_metrics": asdict(self.current_metrics),
                "performance_score": self._calculate_performance_score(self.current_metrics),
                "trends": {
                    "cpu_usage_trend": cpu_trend,
                    "memory_usage_trend": memory_trend,
                    "connection_utilization_trend": connection_trend
                },
                "active_alerts": {
                    alert_key: asdict(alert) for alert_key, alert in self.active_alerts.items()
                },
                "recent_alerts": [asdict(alert) for alert in list(self.alert_history)[-10:]],
                "optimization_recommendations": [asdict(rec) for rec in recommendations[:5]],
                "summary": {
                    "total_alerts": len(self.active_alerts),
                    "critical_alerts": len([a for a in self.active_alerts.values() if a.severity == 'critical']),
                    "warning_alerts": len([a for a in self.active_alerts.values() if a.severity == 'warning']),
                    "health_status": "healthy" if len(self.active_alerts) == 0 else "degraded"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {"error": str(e)}
    
    def _generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on current metrics"""
        recommendations = []
        
        if not self.current_metrics:
            return recommendations
        
        metrics = self.current_metrics
        
        # Connection optimization
        if metrics.connection_utilization > 80:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="conn_opt_1",
                category="connections",
                priority="high",
                title="Optimize Connection Pool Size",
                description="Connection utilization is high. Consider increasing max_connections or optimizing connection pooling.",
                impact_score=8.5,
                implementation_effort="medium",
                estimated_improvement="20-30% reduction in connection wait times",
                sql_commands=["ALTER SYSTEM SET max_connections = 500;", "SELECT pg_reload_conf();"],
                rollback_plan="ALTER SYSTEM SET max_connections = 200; SELECT pg_reload_conf();",
                validation_queries=["SELECT setting FROM pg_settings WHERE name = 'max_connections';"]
            ))
        
        # Buffer optimization
        if metrics.buffer_hit_ratio < 95:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="buffer_opt_1",
                category="memory",
                priority="high",
                title="Increase Shared Buffers",
                description="Buffer hit ratio is low. Increasing shared_buffers can improve performance.",
                impact_score=9.2,
                implementation_effort="low",
                estimated_improvement="15-25% improvement in query performance",
                sql_commands=["ALTER SYSTEM SET shared_buffers = '4GB';", "SELECT pg_reload_conf();"],
                rollback_plan="ALTER SYSTEM SET shared_buffers = '2GB'; SELECT pg_reload_conf();",
                validation_queries=["SELECT setting FROM pg_settings WHERE name = 'shared_buffers';"]
            ))
        
        # Index optimization
        if metrics.queries_per_second > 100:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="index_opt_1",
                category="indexes",
                priority="medium",
                title="Analyze Query Performance",
                description="High query volume detected. Consider analyzing slow queries for index optimization.",
                impact_score=7.8,
                implementation_effort="high",
                estimated_improvement="10-20% reduction in query execution time",
                sql_commands=["SELECT pg_stat_statements_reset();"],
                rollback_plan="N/A",
                validation_queries=["SELECT query, calls, total_time FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"]
            ))
        
        return sorted(recommendations, key=lambda x: x.impact_score, reverse=True)

async def main():
    """Main entry point"""
    monitor = DatabasePerformanceMonitor()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nShutting down performance monitor...")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"Error: {e}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())