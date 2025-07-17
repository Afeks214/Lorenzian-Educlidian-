#!/usr/bin/env python3
"""
Database Health Monitor - Enhanced Health Monitoring System
AGENT 1: DATABASE RTO SPECIALIST - Sub-second detection and alerting
Target: Detect failures within 1 second, trigger failover within 15 seconds
"""

import asyncio
import logging
import time
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import asyncpg
import psutil
import yaml
from prometheus_client import Counter, Histogram, Gauge, start_http_server


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class NodeRole(Enum):
    PRIMARY = "primary"
    STANDBY = "standby"
    UNKNOWN = "unknown"


@dataclass
class DatabaseMetrics:
    """Database performance and health metrics"""
    timestamp: datetime
    host: str
    port: int
    role: NodeRole
    status: HealthStatus
    
    # Connection metrics
    active_connections: int
    idle_connections: int
    total_connections: int
    max_connections: int
    connection_utilization: float
    
    # Performance metrics
    query_response_time_ms: float
    transaction_rate: float
    cache_hit_ratio: float
    
    # Replication metrics
    replication_lag_bytes: Optional[int] = None
    replication_lag_seconds: Optional[float] = None
    wal_sender_count: Optional[int] = None
    
    # System metrics
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Custom health indicators
    last_successful_query: Optional[datetime] = None
    consecutive_failures: int = 0
    error_message: Optional[str] = None


class DatabaseHealthMonitor:
    """
    Enhanced database health monitoring system with sub-second detection
    and automatic failover capabilities
    """
    
    def __init__(self, config_path: str = "/app/config.yml"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # Monitoring state
        self.is_running = False
        self.metrics_history: Dict[str, List[DatabaseMetrics]] = {}
        self.last_health_check = {}
        self.failure_counts = {}
        
        # Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Database connections
        self.connections = {}
        
        # Health check configuration
        self.check_interval = self.config.get('health_check_interval', 1)
        self.sub_second_interval = self.config.get('sub_second_interval', 0.5)
        self.failure_threshold = self.config.get('failure_threshold', 3)
        self.query_timeout = self.config.get('query_timeout', 2)
        self.response_time_threshold = self.config.get('response_time_threshold', 500)
        
        # Failover settings
        self.failover_enabled = self.config.get('failover_enabled', True)
        self.failover_cooldown = self.config.get('failover_cooldown', 300)
        self.last_failover_time = None
        
        # Alert settings
        self.alert_webhook = self.config.get('alert_webhook_url')
        self.alert_cooldown = self.config.get('alert_cooldown', 60)
        self.last_alert_time = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration if file not found
            return {
                'databases': {
                    'primary': {
                        'host': os.getenv('PRIMARY_HOST', 'postgres-primary'),
                        'port': int(os.getenv('PRIMARY_PORT', '5432')),
                        'database': os.getenv('POSTGRES_DB', 'grandmodel'),
                        'user': os.getenv('POSTGRES_USER', 'grandmodel'),
                        'password': os.getenv('POSTGRES_PASSWORD', 'password')
                    },
                    'standby': {
                        'host': os.getenv('STANDBY_HOST', 'postgres-standby'),
                        'port': int(os.getenv('STANDBY_PORT', '5432')),
                        'database': os.getenv('POSTGRES_DB', 'grandmodel'),
                        'user': os.getenv('POSTGRES_USER', 'grandmodel'),
                        'password': os.getenv('POSTGRES_PASSWORD', 'password')
                    }
                },
                'health_check_interval': float(os.getenv('HEALTH_CHECK_INTERVAL', '1')),
                'sub_second_interval': float(os.getenv('SUB_SECOND_INTERVAL', '0.5')),
                'failure_threshold': int(os.getenv('FAILURE_THRESHOLD', '3')),
                'query_timeout': int(os.getenv('QUERY_TIMEOUT', '2')),
                'response_time_threshold': int(os.getenv('RESPONSE_TIME_THRESHOLD', '500')),
                'failover_enabled': os.getenv('FAILOVER_ENABLED', 'true').lower() == 'true',
                'alert_webhook_url': os.getenv('ALERT_WEBHOOK_URL'),
                'patroni_api_url': os.getenv('PATRONI_API_URL', 'http://patroni-primary:8008')
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging"""
        logger = logging.getLogger('db_health_monitor')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('/var/log/db_health_monitor.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        self.health_check_counter = Counter(
            'db_health_checks_total',
            'Total number of database health checks',
            ['database', 'status']
        )
        
        self.response_time_histogram = Histogram(
            'db_response_time_seconds',
            'Database query response time',
            ['database']
        )
        
        self.connection_gauge = Gauge(
            'db_connections_active',
            'Number of active database connections',
            ['database']
        )
        
        self.replication_lag_gauge = Gauge(
            'db_replication_lag_bytes',
            'Replication lag in bytes',
            ['database']
        )
        
        self.failure_counter = Counter(
            'db_failures_total',
            'Total number of database failures',
            ['database', 'error_type']
        )
    
    async def create_connection(self, db_config: Dict) -> Optional[asyncpg.Connection]:
        """Create database connection with error handling"""
        try:
            connection = await asyncpg.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                command_timeout=self.query_timeout
            )
            return connection
        except Exception as e:
            self.logger.error(f"Failed to connect to {db_config['host']}:{db_config['port']}: {e}")
            return None
    
    async def check_database_health(self, db_name: str, db_config: Dict) -> DatabaseMetrics:
        """Perform comprehensive database health check"""
        start_time = time.time()
        
        try:
            # Create connection
            connection = await self.create_connection(db_config)
            if not connection:
                return DatabaseMetrics(
                    timestamp=datetime.now(),
                    host=db_config['host'],
                    port=db_config['port'],
                    role=NodeRole.UNKNOWN,
                    status=HealthStatus.FAILED,
                    active_connections=0,
                    idle_connections=0,
                    total_connections=0,
                    max_connections=0,
                    connection_utilization=0.0,
                    query_response_time_ms=0.0,
                    transaction_rate=0.0,
                    cache_hit_ratio=0.0,
                    consecutive_failures=self.failure_counts.get(db_name, 0) + 1,
                    error_message="Connection failed"
                )
            
            # Basic health check query
            await connection.execute("SELECT 1")
            query_time = (time.time() - start_time) * 1000
            
            # Get connection statistics
            conn_stats = await connection.fetchrow("""
                SELECT 
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') as idle_connections,
                    (SELECT count(*) FROM pg_stat_activity) as total_connections,
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections
            """)
            
            # Get replication status
            replication_info = await self.get_replication_info(connection)
            
            # Get database role (primary/standby)
            role = await self.get_database_role(connection)
            
            # Get performance metrics
            perf_metrics = await self.get_performance_metrics(connection)
            
            # Get system metrics
            system_metrics = self.get_system_metrics()
            
            # Calculate connection utilization
            connection_utilization = (conn_stats['total_connections'] / conn_stats['max_connections']) * 100
            
            # Determine health status
            status = self.determine_health_status(
                query_time, connection_utilization, replication_info, system_metrics
            )
            
            # Update Prometheus metrics
            self.health_check_counter.labels(database=db_name, status=status.value).inc()
            self.response_time_histogram.labels(database=db_name).observe(query_time / 1000)
            self.connection_gauge.labels(database=db_name).set(conn_stats['total_connections'])
            
            if replication_info.get('lag_bytes'):
                self.replication_lag_gauge.labels(database=db_name).set(replication_info['lag_bytes'])
            
            # Reset failure count on success
            self.failure_counts[db_name] = 0
            
            await connection.close()
            
            return DatabaseMetrics(
                timestamp=datetime.now(),
                host=db_config['host'],
                port=db_config['port'],
                role=role,
                status=status,
                active_connections=conn_stats['active_connections'],
                idle_connections=conn_stats['idle_connections'],
                total_connections=conn_stats['total_connections'],
                max_connections=conn_stats['max_connections'],
                connection_utilization=connection_utilization,
                query_response_time_ms=query_time,
                transaction_rate=perf_metrics.get('transaction_rate', 0.0),
                cache_hit_ratio=perf_metrics.get('cache_hit_ratio', 0.0),
                replication_lag_bytes=replication_info.get('lag_bytes'),
                replication_lag_seconds=replication_info.get('lag_seconds'),
                wal_sender_count=replication_info.get('wal_sender_count'),
                cpu_usage_percent=system_metrics.get('cpu_usage', 0.0),
                memory_usage_percent=system_metrics.get('memory_usage', 0.0),
                disk_usage_percent=system_metrics.get('disk_usage', 0.0),
                last_successful_query=datetime.now(),
                consecutive_failures=0
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed for {db_name}: {e}")
            self.failure_counter.labels(database=db_name, error_type=type(e).__name__).inc()
            
            # Increment failure count
            self.failure_counts[db_name] = self.failure_counts.get(db_name, 0) + 1
            
            return DatabaseMetrics(
                timestamp=datetime.now(),
                host=db_config['host'],
                port=db_config['port'],
                role=NodeRole.UNKNOWN,
                status=HealthStatus.FAILED,
                active_connections=0,
                idle_connections=0,
                total_connections=0,
                max_connections=0,
                connection_utilization=0.0,
                query_response_time_ms=0.0,
                transaction_rate=0.0,
                cache_hit_ratio=0.0,
                consecutive_failures=self.failure_counts.get(db_name, 0),
                error_message=str(e)
            )
    
    async def get_replication_info(self, connection: asyncpg.Connection) -> Dict:
        """Get replication status and lag information"""
        try:
            # Check if this is a primary (has WAL senders)
            wal_senders = await connection.fetch("""
                SELECT 
                    application_name,
                    client_addr,
                    state,
                    sent_lsn,
                    write_lsn,
                    flush_lsn,
                    replay_lsn,
                    sync_state,
                    pg_wal_lsn_diff(sent_lsn, write_lsn) as write_lag_bytes,
                    pg_wal_lsn_diff(sent_lsn, flush_lsn) as flush_lag_bytes,
                    pg_wal_lsn_diff(sent_lsn, replay_lsn) as replay_lag_bytes
                FROM pg_stat_replication
            """)
            
            if wal_senders:
                # This is a primary server
                max_lag_bytes = max(row['replay_lag_bytes'] or 0 for row in wal_senders)
                return {
                    'role': 'primary',
                    'wal_sender_count': len(wal_senders),
                    'lag_bytes': max_lag_bytes,
                    'standby_count': len(wal_senders)
                }
            else:
                # Check if this is a standby
                recovery_info = await connection.fetchrow("""
                    SELECT 
                        pg_is_in_recovery() as is_standby,
                        CASE WHEN pg_is_in_recovery() THEN 
                            pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn())
                        END as lag_bytes
                """)
                
                if recovery_info and recovery_info['is_standby']:
                    return {
                        'role': 'standby',
                        'lag_bytes': recovery_info['lag_bytes'] or 0,
                        'is_streaming': True
                    }
                else:
                    return {
                        'role': 'primary',
                        'wal_sender_count': 0,
                        'lag_bytes': 0
                    }
        except Exception as e:
            self.logger.error(f"Failed to get replication info: {e}")
            return {}
    
    async def get_database_role(self, connection: asyncpg.Connection) -> NodeRole:
        """Determine if database is primary or standby"""
        try:
            result = await connection.fetchrow("SELECT pg_is_in_recovery() as is_standby")
            return NodeRole.STANDBY if result['is_standby'] else NodeRole.PRIMARY
        except Exception:
            return NodeRole.UNKNOWN
    
    async def get_performance_metrics(self, connection: asyncpg.Connection) -> Dict:
        """Get database performance metrics"""
        try:
            # Get transaction rate and cache hit ratio
            stats = await connection.fetchrow("""
                SELECT 
                    sum(xact_commit + xact_rollback) as total_transactions,
                    sum(blks_hit) as cache_hits,
                    sum(blks_hit + blks_read) as total_blocks
                FROM pg_stat_database
                WHERE datname = current_database()
            """)
            
            cache_hit_ratio = 0.0
            if stats['total_blocks'] > 0:
                cache_hit_ratio = (stats['cache_hits'] / stats['total_blocks']) * 100
            
            return {
                'transaction_rate': stats['total_transactions'] or 0,
                'cache_hit_ratio': cache_hit_ratio
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {'transaction_rate': 0.0, 'cache_hit_ratio': 0.0}
    
    def get_system_metrics(self) -> Dict:
        """Get system-level metrics"""
        try:
            return {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return {'cpu_usage': 0.0, 'memory_usage': 0.0, 'disk_usage': 0.0}
    
    def determine_health_status(self, query_time: float, connection_utilization: float, 
                              replication_info: Dict, system_metrics: Dict) -> HealthStatus:
        """Determine overall health status based on metrics"""
        # Critical conditions
        if query_time > self.response_time_threshold:
            return HealthStatus.CRITICAL
        
        if connection_utilization > 90:
            return HealthStatus.CRITICAL
        
        if system_metrics.get('cpu_usage', 0) > 95:
            return HealthStatus.CRITICAL
        
        if system_metrics.get('memory_usage', 0) > 95:
            return HealthStatus.CRITICAL
        
        # Warning conditions
        if query_time > (self.response_time_threshold * 0.7):
            return HealthStatus.WARNING
        
        if connection_utilization > 75:
            return HealthStatus.WARNING
        
        if replication_info.get('lag_bytes', 0) > 1024 * 1024:  # 1MB lag
            return HealthStatus.WARNING
        
        if system_metrics.get('cpu_usage', 0) > 80:
            return HealthStatus.WARNING
        
        if system_metrics.get('memory_usage', 0) > 80:
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    async def trigger_failover(self, failed_db: str, metrics: DatabaseMetrics):
        """Trigger automatic failover through Patroni"""
        if not self.failover_enabled:
            self.logger.info("Failover disabled, skipping automatic failover")
            return
        
        # Check failover cooldown
        if self.last_failover_time and (datetime.now() - self.last_failover_time).total_seconds() < self.failover_cooldown:
            self.logger.info(f"Failover cooldown active, skipping failover for {failed_db}")
            return
        
        try:
            self.logger.critical(f"Triggering failover for {failed_db}")
            
            # Call Patroni API for failover
            patroni_url = self.config.get('patroni_api_url', 'http://patroni-primary:8008')
            
            async with aiohttp.ClientSession() as session:
                # First, check cluster status
                async with session.get(f"{patroni_url}/cluster") as response:
                    cluster_info = await response.json()
                    self.logger.info(f"Cluster status: {cluster_info}")
                
                # Trigger failover
                failover_data = {
                    "leader": cluster_info.get("leader"),
                    "candidate": None  # Let Patroni choose the best candidate
                }
                
                async with session.post(f"{patroni_url}/failover", json=failover_data) as response:
                    if response.status == 200:
                        self.logger.info("Failover triggered successfully")
                        self.last_failover_time = datetime.now()
                        
                        # Send alert
                        await self.send_alert(
                            "FAILOVER_TRIGGERED",
                            f"Automatic failover triggered for {failed_db}",
                            metrics
                        )
                    else:
                        self.logger.error(f"Failover failed with status {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to trigger failover: {e}")
    
    async def send_alert(self, alert_type: str, message: str, metrics: DatabaseMetrics):
        """Send alert notification"""
        if not self.alert_webhook:
            return
        
        alert_key = f"{alert_type}_{metrics.host}"
        
        # Check alert cooldown
        if alert_key in self.last_alert_time:
            if (datetime.now() - self.last_alert_time[alert_key]).total_seconds() < self.alert_cooldown:
                return
        
        try:
            alert_data = {
                "alert_type": alert_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "database": {
                    "host": metrics.host,
                    "port": metrics.port,
                    "role": metrics.role.value,
                    "status": metrics.status.value
                },
                "metrics": asdict(metrics)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.alert_webhook, json=alert_data) as response:
                    if response.status == 200:
                        self.logger.info(f"Alert sent successfully: {alert_type}")
                        self.last_alert_time[alert_key] = datetime.now()
                    else:
                        self.logger.error(f"Failed to send alert: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    async def sub_second_health_check(self):
        """Sub-second health check for critical monitoring"""
        self.logger.info("Starting sub-second health monitoring")
        
        while self.is_running:
            try:
                # Quick health check using lightweight queries
                tasks = []
                
                for db_name, db_config in self.config['databases'].items():
                    task = asyncio.create_task(self.quick_health_check(db_name, db_config))
                    tasks.append((db_name, task))
                
                # Wait for all quick checks to complete
                for db_name, task in tasks:
                    try:
                        is_healthy = await task
                        if not is_healthy:
                            self.logger.warning(f"Sub-second health check failed for {db_name}")
                            # Trigger immediate full health check
                            await self.check_database_health(db_name, self.config['databases'][db_name])
                    except Exception as e:
                        self.logger.error(f"Sub-second health check error for {db_name}: {e}")
                
                await asyncio.sleep(self.sub_second_interval)
                
            except Exception as e:
                self.logger.error(f"Error in sub-second health check loop: {e}")
                await asyncio.sleep(self.sub_second_interval)
    
    async def quick_health_check(self, db_name: str, db_config: Dict) -> bool:
        """Quick lightweight health check for sub-second monitoring"""
        try:
            connection = await asyncpg.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                command_timeout=1  # 1 second timeout for quick checks
            )
            
            # Simple SELECT 1 query
            await connection.execute("SELECT 1")
            await connection.close()
            return True
            
        except Exception as e:
            self.logger.debug(f"Quick health check failed for {db_name}: {e}")
            return False
    
    async def health_check_loop(self):
        """Main health check loop"""
        self.logger.info("Starting database health monitoring")
        
        while self.is_running:
            try:
                tasks = []
                
                # Check all configured databases
                for db_name, db_config in self.config['databases'].items():
                    task = asyncio.create_task(self.check_database_health(db_name, db_config))
                    tasks.append((db_name, task))
                
                # Wait for all health checks to complete
                for db_name, task in tasks:
                    try:
                        metrics = await task
                        
                        # Store metrics
                        if db_name not in self.metrics_history:
                            self.metrics_history[db_name] = []
                        
                        self.metrics_history[db_name].append(metrics)
                        
                        # Keep only last 100 metrics
                        if len(self.metrics_history[db_name]) > 100:
                            self.metrics_history[db_name] = self.metrics_history[db_name][-100:]
                        
                        # Log health status
                        self.logger.info(
                            f"Database {db_name} ({metrics.host}:{metrics.port}) - "
                            f"Status: {metrics.status.value}, "
                            f"Response: {metrics.query_response_time_ms:.1f}ms, "
                            f"Connections: {metrics.total_connections}/{metrics.max_connections}, "
                            f"Role: {metrics.role.value}"
                        )
                        
                        # Check for critical failures
                        if metrics.status == HealthStatus.FAILED:
                            consecutive_failures = metrics.consecutive_failures
                            
                            if consecutive_failures >= self.failure_threshold:
                                self.logger.critical(
                                    f"Database {db_name} failed {consecutive_failures} consecutive times"
                                )
                                
                                # Send critical alert
                                await self.send_alert(
                                    "DATABASE_CRITICAL",
                                    f"Database {db_name} has failed {consecutive_failures} consecutive health checks",
                                    metrics
                                )
                                
                                # Trigger failover if this is the primary
                                if metrics.role == NodeRole.PRIMARY:
                                    await self.trigger_failover(db_name, metrics)
                        
                        # Check for warnings
                        elif metrics.status == HealthStatus.WARNING:
                            await self.send_alert(
                                "DATABASE_WARNING",
                                f"Database {db_name} is in warning state",
                                metrics
                            )
                        
                        # Check for performance issues
                        elif metrics.status == HealthStatus.CRITICAL:
                            await self.send_alert(
                                "DATABASE_PERFORMANCE",
                                f"Database {db_name} has performance issues",
                                metrics
                            )
                    
                    except Exception as e:
                        self.logger.error(f"Error processing health check for {db_name}: {e}")
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def start(self):
        """Start the health monitoring system"""
        self.is_running = True
        
        # Start Prometheus metrics server
        start_http_server(8000)
        self.logger.info("Prometheus metrics server started on port 8000")
        
        # Start both health check loops concurrently
        await asyncio.gather(
            self.health_check_loop(),
            self.sub_second_health_check()
        )
    
    def stop(self):
        """Stop the health monitoring system"""
        self.is_running = False
        self.logger.info("Database health monitoring stopped")
    
    async def get_status(self) -> Dict:
        """Get current health status for all databases"""
        status = {}
        
        for db_name, metrics_list in self.metrics_history.items():
            if metrics_list:
                latest_metrics = metrics_list[-1]
                status[db_name] = {
                    'status': latest_metrics.status.value,
                    'role': latest_metrics.role.value,
                    'last_check': latest_metrics.timestamp.isoformat(),
                    'response_time_ms': latest_metrics.query_response_time_ms,
                    'connections': f"{latest_metrics.total_connections}/{latest_metrics.max_connections}",
                    'consecutive_failures': latest_metrics.consecutive_failures
                }
        
        return status


async def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Quick health check for container health check
        monitor = DatabaseHealthMonitor()
        try:
            # Quick check of all databases
            all_healthy = True
            for db_name, db_config in monitor.config['databases'].items():
                metrics = await monitor.check_database_health(db_name, db_config)
                if metrics.status == HealthStatus.FAILED:
                    all_healthy = False
                    break
            
            if all_healthy:
                print("Health check passed")
                sys.exit(0)
            else:
                print("Health check failed")
                sys.exit(1)
        except Exception as e:
            print(f"Health check error: {e}")
            sys.exit(1)
    else:
        # Run full monitoring
        monitor = DatabaseHealthMonitor()
        try:
            await monitor.start()
        except KeyboardInterrupt:
            monitor.stop()
            print("Database health monitor stopped")


if __name__ == "__main__":
    asyncio.run(main())