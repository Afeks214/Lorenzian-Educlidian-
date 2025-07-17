#!/usr/bin/env python3
"""
Connection Pool Health Monitor
AGENT 1: DATABASE RTO SPECIALIST - Connection Pool Monitoring
Target: Monitor connection pool health for faster RTO detection
"""

import asyncio
import asyncpg
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge
import psycopg2
from psycopg2 import pool


@dataclass
class ConnectionPoolMetrics:
    """Connection pool health metrics"""
    timestamp: datetime
    pool_name: str
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_connections: int
    pool_utilization: float
    avg_connection_time_ms: float
    failed_connections: int
    max_connections: int
    min_connections: int
    connection_errors: List[str]


class ConnectionPoolMonitor:
    """
    Enhanced connection pool monitoring for faster failure detection
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self.is_running = False
        self.pools = {}
        self.metrics_history = {}
        
        # Monitoring configuration
        self.check_interval = config.get('pool_check_interval', 1)
        self.connection_timeout = config.get('connection_timeout', 2)
        self.max_retries = config.get('max_retries', 3)
        
        # Pool configuration
        self.pool_configs = config.get('connection_pools', {})
        
        # Setup Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Health thresholds
        self.utilization_warning = config.get('utilization_warning', 80)
        self.utilization_critical = config.get('utilization_critical', 95)
        self.connection_time_warning = config.get('connection_time_warning', 100)
        self.connection_time_critical = config.get('connection_time_critical', 500)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('connection_pool_monitor')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics for pool monitoring"""
        self.pool_connections_gauge = Gauge(
            'db_pool_connections_total',
            'Total connections in pool',
            ['pool_name', 'state']
        )
        
        self.pool_utilization_gauge = Gauge(
            'db_pool_utilization_percent',
            'Pool utilization percentage',
            ['pool_name']
        )
        
        self.connection_time_histogram = Histogram(
            'db_pool_connection_time_seconds',
            'Time to acquire connection from pool',
            ['pool_name']
        )
        
        self.pool_errors_counter = Counter(
            'db_pool_errors_total',
            'Total pool connection errors',
            ['pool_name', 'error_type']
        )
    
    async def create_connection_pool(self, pool_name: str, pool_config: Dict):
        """Create and configure connection pool"""
        try:
            # Create asyncpg connection pool
            pool = await asyncpg.create_pool(
                host=pool_config['host'],
                port=pool_config['port'],
                database=pool_config['database'],
                user=pool_config['user'],
                password=pool_config['password'],
                min_size=pool_config.get('min_size', 10),
                max_size=pool_config.get('max_size', 50),
                command_timeout=self.connection_timeout,
                server_settings={
                    'application_name': f'pool_monitor_{pool_name}',
                    'tcp_keepalives_idle': '600',
                    'tcp_keepalives_interval': '30',
                    'tcp_keepalives_count': '3'
                }
            )
            
            self.pools[pool_name] = pool
            self.logger.info(f"Created connection pool '{pool_name}' with {pool_config.get('min_size', 10)}-{pool_config.get('max_size', 50)} connections")
            
        except Exception as e:
            self.logger.error(f"Failed to create connection pool '{pool_name}': {e}")
            raise
    
    async def check_pool_health(self, pool_name: str) -> ConnectionPoolMetrics:
        """Check health of a specific connection pool"""
        start_time = time.time()
        connection_errors = []
        
        try:
            pool = self.pools.get(pool_name)
            if not pool:
                raise Exception(f"Pool '{pool_name}' not found")
            
            # Get pool statistics
            pool_size = pool.get_size()
            min_size = pool.get_min_size()
            max_size = pool.get_max_size()
            idle_size = pool.get_idle_size()
            
            # Calculate metrics
            active_connections = pool_size - idle_size
            utilization = (pool_size / max_size) * 100 if max_size > 0 else 0
            
            # Test connection acquisition time
            connection_start = time.time()
            try:
                async with pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                connection_time = (time.time() - connection_start) * 1000
            except Exception as e:
                connection_time = 0
                connection_errors.append(f"Connection test failed: {str(e)}")
            
            # Update Prometheus metrics
            self.pool_connections_gauge.labels(pool_name=pool_name, state='total').set(pool_size)
            self.pool_connections_gauge.labels(pool_name=pool_name, state='active').set(active_connections)
            self.pool_connections_gauge.labels(pool_name=pool_name, state='idle').set(idle_size)
            self.pool_utilization_gauge.labels(pool_name=pool_name).set(utilization)
            self.connection_time_histogram.labels(pool_name=pool_name).observe(connection_time / 1000)
            
            # Count errors
            if connection_errors:
                self.pool_errors_counter.labels(pool_name=pool_name, error_type='connection_test').inc()
            
            return ConnectionPoolMetrics(
                timestamp=datetime.now(),
                pool_name=pool_name,
                total_connections=pool_size,
                active_connections=active_connections,
                idle_connections=idle_size,
                waiting_connections=0,  # asyncpg doesn't directly expose this
                pool_utilization=utilization,
                avg_connection_time_ms=connection_time,
                failed_connections=len(connection_errors),
                max_connections=max_size,
                min_connections=min_size,
                connection_errors=connection_errors
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check pool health for '{pool_name}': {e}")
            self.pool_errors_counter.labels(pool_name=pool_name, error_type='health_check').inc()
            
            return ConnectionPoolMetrics(
                timestamp=datetime.now(),
                pool_name=pool_name,
                total_connections=0,
                active_connections=0,
                idle_connections=0,
                waiting_connections=0,
                pool_utilization=0,
                avg_connection_time_ms=0,
                failed_connections=1,
                max_connections=0,
                min_connections=0,
                connection_errors=[str(e)]
            )
    
    async def monitor_pools(self):
        """Monitor all connection pools"""
        while self.is_running:
            try:
                for pool_name in self.pools.keys():
                    metrics = await self.check_pool_health(pool_name)
                    
                    # Store metrics
                    if pool_name not in self.metrics_history:
                        self.metrics_history[pool_name] = []
                    
                    self.metrics_history[pool_name].append(metrics)
                    
                    # Keep only last 100 metrics
                    if len(self.metrics_history[pool_name]) > 100:
                        self.metrics_history[pool_name] = self.metrics_history[pool_name][-100:]
                    
                    # Log metrics
                    self.logger.info(
                        f"Pool '{pool_name}': "
                        f"Utilization: {metrics.pool_utilization:.1f}%, "
                        f"Connections: {metrics.active_connections}/{metrics.total_connections}, "
                        f"Connection Time: {metrics.avg_connection_time_ms:.1f}ms, "
                        f"Errors: {metrics.failed_connections}"
                    )
                    
                    # Check for warnings/critical states
                    await self.check_pool_alerts(metrics)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in pool monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_pool_alerts(self, metrics: ConnectionPoolMetrics):
        """Check for pool alert conditions"""
        alerts = []
        
        # Check utilization
        if metrics.pool_utilization >= self.utilization_critical:
            alerts.append(f"CRITICAL: Pool utilization {metrics.pool_utilization:.1f}% >= {self.utilization_critical}%")
        elif metrics.pool_utilization >= self.utilization_warning:
            alerts.append(f"WARNING: Pool utilization {metrics.pool_utilization:.1f}% >= {self.utilization_warning}%")
        
        # Check connection time
        if metrics.avg_connection_time_ms >= self.connection_time_critical:
            alerts.append(f"CRITICAL: Connection time {metrics.avg_connection_time_ms:.1f}ms >= {self.connection_time_critical}ms")
        elif metrics.avg_connection_time_ms >= self.connection_time_warning:
            alerts.append(f"WARNING: Connection time {metrics.avg_connection_time_ms:.1f}ms >= {self.connection_time_warning}ms")
        
        # Check for connection errors
        if metrics.failed_connections > 0:
            alerts.append(f"ERROR: {metrics.failed_connections} connection failures")
        
        # Log alerts
        for alert in alerts:
            if "CRITICAL" in alert:
                self.logger.critical(f"Pool '{metrics.pool_name}': {alert}")
            elif "WARNING" in alert:
                self.logger.warning(f"Pool '{metrics.pool_name}': {alert}")
            else:
                self.logger.error(f"Pool '{metrics.pool_name}': {alert}")
    
    async def start(self):
        """Start the connection pool monitor"""
        self.is_running = True
        
        # Create connection pools
        for pool_name, pool_config in self.pool_configs.items():
            await self.create_connection_pool(pool_name, pool_config)
        
        # Start monitoring
        await self.monitor_pools()
    
    def stop(self):
        """Stop the connection pool monitor"""
        self.is_running = False
        
        # Close all pools
        for pool_name, pool in self.pools.items():
            try:
                pool.close()
                self.logger.info(f"Closed connection pool '{pool_name}'")
            except Exception as e:
                self.logger.error(f"Error closing pool '{pool_name}': {e}")
    
    def get_pool_status(self) -> Dict:
        """Get current status of all pools"""
        status = {}
        
        for pool_name, metrics_list in self.metrics_history.items():
            if metrics_list:
                latest_metrics = metrics_list[-1]
                status[pool_name] = {
                    'utilization': latest_metrics.pool_utilization,
                    'total_connections': latest_metrics.total_connections,
                    'active_connections': latest_metrics.active_connections,
                    'idle_connections': latest_metrics.idle_connections,
                    'connection_time_ms': latest_metrics.avg_connection_time_ms,
                    'errors': latest_metrics.failed_connections,
                    'last_check': latest_metrics.timestamp.isoformat()
                }
        
        return status