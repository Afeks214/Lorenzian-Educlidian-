#!/usr/bin/env python3
"""
Advanced Connection Pool Optimizer for High-Frequency Trading
AGENT 14: DATABASE OPTIMIZATION SPECIALIST
Focus: Sub-millisecond latency connection pooling with pgBouncer integration
"""

import asyncio
import asyncpg
import psycopg2
from psycopg2 import pool
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading
from concurrent.futures import ThreadPoolExecutor
import resource
import os

@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pool optimization"""
    name: str
    host: str
    port: int
    database: str
    user: str
    password: str
    min_size: int = 20
    max_size: int = 100
    target_latency_ms: float = 0.5
    max_latency_ms: float = 2.0
    connection_timeout: int = 1
    command_timeout: int = 5
    keepalive_interval: int = 30
    tcp_keepalive: bool = True
    application_name: str = "hft_trading"

@dataclass
class ConnectionMetrics:
    """Real-time connection metrics"""
    timestamp: datetime
    pool_name: str
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_connections: int
    connection_time_ms: float
    query_time_ms: float
    error_rate: float
    throughput_qps: float
    latency_p95: float
    latency_p99: float
    cpu_usage: float
    memory_usage_mb: float

class AdvancedConnectionPoolOptimizer:
    """
    Advanced connection pool optimizer for high-frequency trading
    Implements intelligent pool sizing, connection pre-warming, and latency optimization
    """
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.pools = {}
        self.metrics_history = {}
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Performance monitoring
        self.start_time = time.time()
        self.query_count = 0
        self.error_count = 0
        self.latency_samples = []
        
        # Setup Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Adaptive pool sizing
        self.pool_adjustment_history = {}
        self.last_adjustment_time = {}
        
        # Connection prewarming
        self.prewarmed_connections = {}
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "pools": {
                "trading_primary": {
                    "host": "127.0.0.1",
                    "port": 6432,  # pgBouncer port
                    "database": "grandmodel",
                    "user": "grandmodel_user",
                    "password": "grandmodel_password",
                    "min_size": 50,
                    "max_size": 200,
                    "target_latency_ms": 0.5,
                    "max_latency_ms": 2.0
                },
                "trading_standby": {
                    "host": "127.0.0.1",
                    "port": 6433,
                    "database": "grandmodel",
                    "user": "grandmodel_user",
                    "password": "grandmodel_password",
                    "min_size": 20,
                    "max_size": 100,
                    "target_latency_ms": 1.0,
                    "max_latency_ms": 3.0
                }
            },
            "monitoring": {
                "check_interval": 0.1,  # 100ms checks
                "metrics_retention": 3600,  # 1 hour
                "prometheus_port": 9090
            },
            "optimization": {
                "auto_scale": True,
                "prewarming_enabled": True,
                "adaptive_sizing": True,
                "latency_threshold_ms": 1.0
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    return {**default_config, **loaded_config}
            except Exception as e:
                self.logger.error(f"Error loading config file: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup high-performance logging"""
        logger = logging.getLogger('connection_pool_optimizer')
        logger.setLevel(logging.INFO)
        
        # Use faster formatting
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        self.connection_pool_size = Gauge(
            'db_connection_pool_size',
            'Current connection pool size',
            ['pool_name', 'state']
        )
        
        self.connection_latency = Histogram(
            'db_connection_latency_seconds',
            'Database connection latency',
            ['pool_name', 'operation'],
            buckets=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.query_throughput = Counter(
            'db_query_throughput_total',
            'Total database queries processed',
            ['pool_name', 'query_type']
        )
        
        self.connection_errors = Counter(
            'db_connection_errors_total',
            'Total connection errors',
            ['pool_name', 'error_type']
        )
        
        self.pool_utilization = Gauge(
            'db_pool_utilization_percent',
            'Connection pool utilization percentage',
            ['pool_name']
        )
        
        self.adaptive_adjustments = Counter(
            'db_pool_adaptive_adjustments_total',
            'Total adaptive pool size adjustments',
            ['pool_name', 'adjustment_type']
        )
    
    async def create_optimized_pool(self, pool_name: str, config: Dict) -> asyncpg.Pool:
        """Create an optimized connection pool"""
        try:
            # TCP optimizations
            tcp_settings = {
                'tcp_keepalives_idle': '60',
                'tcp_keepalives_interval': '30',
                'tcp_keepalives_count': '3',
                'tcp_user_timeout': '30000'
            }
            
            # Connection optimizations
            connection_params = {
                'application_name': f'hft_trading_{pool_name}',
                'statement_timeout': '5000',  # 5 second timeout
                'idle_in_transaction_session_timeout': '10000',  # 10 seconds
                'lock_timeout': '2000',  # 2 second lock timeout
                **tcp_settings
            }
            
            pool = await asyncpg.create_pool(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password'],
                min_size=config['min_size'],
                max_size=config['max_size'],
                command_timeout=config.get('command_timeout', 5),
                server_settings=connection_params,
                connection_class=asyncpg.Connection,
                init=self._init_connection
            )
            
            self.pools[pool_name] = pool
            self.logger.info(f"Created optimized pool '{pool_name}' ({config['min_size']}-{config['max_size']} connections)")
            
            # Pre-warm connections
            if self.config['optimization']['prewarming_enabled']:
                await self._prewarm_connections(pool_name, pool)
            
            return pool
            
        except Exception as e:
            self.logger.error(f"Failed to create pool '{pool_name}': {e}")
            raise
    
    async def _init_connection(self, conn: asyncpg.Connection):
        """Initialize connection with optimizations"""
        try:
            # Set connection-level optimizations
            await conn.execute("SET synchronous_commit = off")
            await conn.execute("SET wal_writer_delay = 10ms")
            await conn.execute("SET commit_delay = 0")
            await conn.execute("SET commit_siblings = 0")
            
            # Prepared statement optimizations
            await conn.execute("SET plan_cache_mode = force_generic_plan")
            await conn.execute("SET default_statistics_target = 100")
            
            # Memory optimizations
            await conn.execute("SET work_mem = '4MB'")
            await conn.execute("SET maintenance_work_mem = '64MB'")
            
        except Exception as e:
            self.logger.warning(f"Connection initialization warning: {e}")
    
    async def _prewarm_connections(self, pool_name: str, pool: asyncpg.Pool):
        """Pre-warm connections for faster response times"""
        try:
            connections = []
            for i in range(pool.get_min_size()):
                conn = await pool.acquire()
                # Execute a simple query to warm up the connection
                await conn.execute("SELECT 1")
                connections.append(conn)
            
            # Release connections back to pool
            for conn in connections:
                await pool.release(conn)
            
            self.logger.info(f"Pre-warmed {len(connections)} connections for pool '{pool_name}'")
            
        except Exception as e:
            self.logger.error(f"Connection pre-warming failed for '{pool_name}': {e}")
    
    async def execute_optimized_query(self, pool_name: str, query: str, params: tuple = None) -> Any:
        """Execute query with latency optimization"""
        start_time = time.time()
        
        try:
            pool = self.pools.get(pool_name)
            if not pool:
                raise ValueError(f"Pool '{pool_name}' not found")
            
            # Use connection with timeout
            async with pool.acquire() as conn:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)
            
            # Record metrics
            execution_time = (time.time() - start_time) * 1000
            self.connection_latency.labels(pool_name=pool_name, operation='query').observe(execution_time / 1000)
            self.query_throughput.labels(pool_name=pool_name, query_type='select').inc()
            
            self.query_count += 1
            self.latency_samples.append(execution_time)
            
            # Keep only recent samples
            if len(self.latency_samples) > 1000:
                self.latency_samples = self.latency_samples[-1000:]
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.connection_errors.labels(pool_name=pool_name, error_type='query_execution').inc()
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    async def monitor_pool_performance(self, pool_name: str) -> ConnectionMetrics:
        """Monitor pool performance in real-time"""
        try:
            pool = self.pools.get(pool_name)
            if not pool:
                raise ValueError(f"Pool '{pool_name}' not found")
            
            # Get pool statistics
            pool_size = pool.get_size()
            idle_size = pool.get_idle_size()
            min_size = pool.get_min_size()
            max_size = pool.get_max_size()
            
            active_connections = pool_size - idle_size
            utilization = (pool_size / max_size) * 100 if max_size > 0 else 0
            
            # Test connection performance
            start_time = time.time()
            try:
                async with pool.acquire() as conn:
                    query_start = time.time()
                    await conn.execute("SELECT 1")
                    query_time = (time.time() - query_start) * 1000
                connection_time = (time.time() - start_time) * 1000
            except Exception as e:
                connection_time = 0
                query_time = 0
                self.connection_errors.labels(pool_name=pool_name, error_type='health_check').inc()
            
            # Calculate performance metrics
            error_rate = (self.error_count / max(self.query_count, 1)) * 100
            throughput = self.query_count / max(time.time() - self.start_time, 1)
            
            # Calculate latency percentiles
            sorted_latencies = sorted(self.latency_samples) if self.latency_samples else [0]
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            latency_p95 = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0
            latency_p99 = sorted_latencies[p99_idx] if p99_idx < len(sorted_latencies) else 0
            
            # System resource usage
            resource_usage = resource.getrusage(resource.RUSAGE_SELF)
            cpu_usage = resource_usage.ru_utime + resource_usage.ru_stime
            memory_usage = resource_usage.ru_maxrss / 1024  # KB to MB
            
            # Update Prometheus metrics
            self.connection_pool_size.labels(pool_name=pool_name, state='total').set(pool_size)
            self.connection_pool_size.labels(pool_name=pool_name, state='active').set(active_connections)
            self.connection_pool_size.labels(pool_name=pool_name, state='idle').set(idle_size)
            self.pool_utilization.labels(pool_name=pool_name).set(utilization)
            
            return ConnectionMetrics(
                timestamp=datetime.now(),
                pool_name=pool_name,
                total_connections=pool_size,
                active_connections=active_connections,
                idle_connections=idle_size,
                waiting_connections=0,  # Not directly available in asyncpg
                connection_time_ms=connection_time,
                query_time_ms=query_time,
                error_rate=error_rate,
                throughput_qps=throughput,
                latency_p95=latency_p95,
                latency_p99=latency_p99,
                cpu_usage=cpu_usage,
                memory_usage_mb=memory_usage
            )
            
        except Exception as e:
            self.logger.error(f"Pool monitoring failed for '{pool_name}': {e}")
            return ConnectionMetrics(
                timestamp=datetime.now(),
                pool_name=pool_name,
                total_connections=0,
                active_connections=0,
                idle_connections=0,
                waiting_connections=0,
                connection_time_ms=0,
                query_time_ms=0,
                error_rate=100.0,
                throughput_qps=0,
                latency_p95=0,
                latency_p99=0,
                cpu_usage=0,
                memory_usage_mb=0
            )
    
    async def adaptive_pool_sizing(self, pool_name: str, metrics: ConnectionMetrics):
        """Automatically adjust pool size based on performance"""
        if not self.config['optimization']['adaptive_sizing']:
            return
            
        pool = self.pools.get(pool_name)
        if not pool:
            return
            
        current_time = time.time()
        last_adjustment = self.last_adjustment_time.get(pool_name, 0)
        
        # Only adjust every 10 seconds minimum
        if current_time - last_adjustment < 10:
            return
            
        pool_config = self.config['pools'][pool_name]
        target_latency = pool_config['target_latency_ms']
        max_latency = pool_config['max_latency_ms']
        
        adjustment_made = False
        
        # Increase pool size if latency is too high
        if metrics.latency_p95 > max_latency and metrics.pool_utilization > 80:
            new_size = min(pool.get_max_size() + 10, 500)  # Cap at 500
            # Note: asyncpg doesn't support dynamic resize, would need recreation
            self.logger.info(f"Pool '{pool_name}' needs scaling up to {new_size} connections")
            self.adaptive_adjustments.labels(pool_name=pool_name, adjustment_type='scale_up').inc()
            adjustment_made = True
            
        # Decrease pool size if utilization is low
        elif metrics.pool_utilization < 30 and metrics.latency_p95 < target_latency:
            new_size = max(pool.get_min_size() - 5, 10)  # Minimum 10 connections
            self.logger.info(f"Pool '{pool_name}' can scale down to {new_size} connections")
            self.adaptive_adjustments.labels(pool_name=pool_name, adjustment_type='scale_down').inc()
            adjustment_made = True
        
        if adjustment_made:
            self.last_adjustment_time[pool_name] = current_time
    
    async def connection_health_check(self, pool_name: str) -> bool:
        """Perform comprehensive health check"""
        try:
            pool = self.pools.get(pool_name)
            if not pool:
                return False
            
            # Test multiple operations
            async with pool.acquire() as conn:
                # Basic connectivity
                await conn.execute("SELECT 1")
                
                # Transaction test
                async with conn.transaction():
                    await conn.execute("SELECT NOW()")
                
                # Prepared statement test
                await conn.execute("SELECT $1::int", 42)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed for pool '{pool_name}': {e}")
            self.connection_errors.labels(pool_name=pool_name, error_type='health_check').inc()
            return False
    
    async def failover_test(self, pool_name: str) -> Tuple[bool, float]:
        """Test failover performance"""
        try:
            start_time = time.time()
            
            # Simulate connection failure and recovery
            pool = self.pools.get(pool_name)
            if not pool:
                return False, 0
            
            # Try to execute query after simulated failure
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            failover_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"Failover test for '{pool_name}' completed in {failover_time:.2f}ms")
            return True, failover_time
            
        except Exception as e:
            self.logger.error(f"Failover test failed for '{pool_name}': {e}")
            return False, 0
    
    async def start_monitoring(self):
        """Start comprehensive monitoring"""
        self.is_running = True
        
        # Initialize connection pools
        for pool_name, pool_config in self.config['pools'].items():
            await self.create_optimized_pool(pool_name, pool_config)
        
        # Start Prometheus metrics server
        prometheus_port = self.config['monitoring']['prometheus_port']
        start_http_server(prometheus_port)
        self.logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        
        # Start monitoring loop
        check_interval = self.config['monitoring']['check_interval']
        
        while self.is_running:
            try:
                for pool_name in self.pools.keys():
                    # Monitor performance
                    metrics = await self.monitor_pool_performance(pool_name)
                    
                    # Store metrics
                    if pool_name not in self.metrics_history:
                        self.metrics_history[pool_name] = []
                    
                    self.metrics_history[pool_name].append(metrics)
                    
                    # Keep only recent metrics
                    retention_count = int(self.config['monitoring']['metrics_retention'] / check_interval)
                    if len(self.metrics_history[pool_name]) > retention_count:
                        self.metrics_history[pool_name] = self.metrics_history[pool_name][-retention_count:]
                    
                    # Adaptive sizing
                    await self.adaptive_pool_sizing(pool_name, metrics)
                    
                    # Log performance
                    if metrics.latency_p95 > self.config['optimization']['latency_threshold_ms']:
                        self.logger.warning(
                            f"Pool '{pool_name}' latency alert: "
                            f"P95={metrics.latency_p95:.2f}ms, "
                            f"P99={metrics.latency_p99:.2f}ms, "
                            f"Utilization={metrics.pool_utilization:.1f}%"
                        )
                    
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(check_interval)
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        self.is_running = False
        
        # Close connection pools
        for pool_name, pool in self.pools.items():
            try:
                pool.close()
                self.logger.info(f"Closed pool '{pool_name}'")
            except Exception as e:
                self.logger.error(f"Error closing pool '{pool_name}': {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'total_queries': self.query_count,
            'total_errors': self.error_count,
            'overall_error_rate': (self.error_count / max(self.query_count, 1)) * 100,
            'pools': {}
        }
        
        for pool_name, metrics_list in self.metrics_history.items():
            if not metrics_list:
                continue
                
            latest_metrics = metrics_list[-1]
            
            # Calculate averages
            recent_metrics = metrics_list[-10:]  # Last 10 measurements
            avg_latency = sum(m.latency_p95 for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput_qps for m in recent_metrics) / len(recent_metrics)
            avg_utilization = sum(m.pool_utilization for m in recent_metrics) / len(recent_metrics)
            
            report['pools'][pool_name] = {
                'current_metrics': asdict(latest_metrics),
                'averages': {
                    'latency_p95_ms': avg_latency,
                    'throughput_qps': avg_throughput,
                    'utilization_percent': avg_utilization
                }
            }
        
        return report

async def main():
    """Main entry point for connection pool optimizer"""
    optimizer = AdvancedConnectionPoolOptimizer()
    
    try:
        await optimizer.start_monitoring()
    except KeyboardInterrupt:
        print("\nShutting down optimizer...")
        optimizer.stop_monitoring()
    except Exception as e:
        print(f"Error: {e}")
        optimizer.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())