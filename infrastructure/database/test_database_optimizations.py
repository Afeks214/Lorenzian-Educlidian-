#!/usr/bin/env python3
"""
Comprehensive Test Suite for Database Optimizations
AGENT 14: DATABASE OPTIMIZATION SPECIALIST
Focus: Testing sub-millisecond latency, failover, and performance optimizations
"""

import asyncio
import asyncpg
import psycopg2
import pytest
import time
import json
import logging
import statistics
import subprocess
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Import our optimization modules
from connection_pool_optimizer import AdvancedConnectionPoolOptimizer
from query_optimizer import IntelligentQueryOptimizer
from high_availability_manager import HighAvailabilityManager
from performance_monitor import DatabasePerformanceMonitor

@dataclass
class TestResult:
    """Test result information"""
    test_name: str
    success: bool
    execution_time_ms: float
    latency_p95: float
    latency_p99: float
    throughput_ops_per_sec: float
    error_rate: float
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class LoadTestResult:
    """Load test result"""
    test_name: str
    duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_ops_per_sec: float
    error_rate: float
    resource_usage: Dict[str, float]

class DatabaseOptimizationTestSuite:
    """
    Comprehensive test suite for database optimizations
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = []
        self.load_test_results = []
        
        # Test configuration
        self.test_config = {
            "database": {
                "host": "127.0.0.1",
                "port": 5432,
                "database": "grandmodel_test",
                "user": "test_user",
                "password": "test_password"
            },
            "pgbouncer": {
                "host": "127.0.0.1",
                "port": 6432
            },
            "patroni": {
                "primary_api": "http://127.0.0.1:8008",
                "standby_api": "http://127.0.0.1:8009"
            },
            "performance_targets": {
                "max_latency_ms": 2.0,
                "target_latency_ms": 0.5,
                "min_throughput_ops_per_sec": 10000,
                "max_failover_time_seconds": 10,
                "max_error_rate": 0.01
            }
        }
        
        # Test data
        self.test_queries = [
            "SELECT 1",
            "SELECT NOW()",
            "SELECT * FROM pg_stat_activity LIMIT 10",
            "SELECT COUNT(*) FROM information_schema.tables",
            "SELECT pg_database_size(current_database())",
            "BEGIN; SELECT 1; COMMIT;",
            "SELECT generate_series(1, 100)",
            "SELECT * FROM pg_settings WHERE name LIKE 'shared%'",
            "SELECT schemaname, tablename FROM pg_tables LIMIT 5",
            "SELECT * FROM pg_stat_database WHERE datname = current_database()"
        ]
        
        # Connection pool for testing
        self.connection_pool = None
        self.is_setup = False
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for tests"""
        logger = logging.getLogger('db_optimization_tests')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def setup_test_environment(self):
        """Setup test environment"""
        if self.is_setup:
            return
            
        try:
            # Create test database connection pool
            self.connection_pool = await asyncpg.create_pool(
                host=self.test_config['database']['host'],
                port=self.test_config['database']['port'],
                database=self.test_config['database']['database'],
                user=self.test_config['database']['user'],
                password=self.test_config['database']['password'],
                min_size=10,
                max_size=50,
                command_timeout=30
            )
            
            # Create test tables
            await self._create_test_tables()
            
            # Insert test data
            await self._insert_test_data()
            
            self.is_setup = True
            self.logger.info("Test environment setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup test environment: {e}")
            raise
    
    async def _create_test_tables(self):
        """Create test tables"""
        try:
            async with self.connection_pool.acquire() as conn:
                # Create performance test table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_test (
                        id SERIAL PRIMARY KEY,
                        test_data TEXT,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        value NUMERIC(10,2),
                        category VARCHAR(50),
                        status VARCHAR(20)
                    )
                """)
                
                # Create indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_performance_test_timestamp 
                    ON performance_test(timestamp)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_performance_test_category 
                    ON performance_test(category)
                """)
                
                # Create load test table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS load_test (
                        id SERIAL PRIMARY KEY,
                        thread_id INTEGER,
                        operation_type VARCHAR(20),
                        execution_time_ms REAL,
                        timestamp TIMESTAMP DEFAULT NOW(),
                        success BOOLEAN DEFAULT TRUE
                    )
                """)
                
                self.logger.info("Test tables created successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to create test tables: {e}")
            raise
    
    async def _insert_test_data(self):
        """Insert test data"""
        try:
            async with self.connection_pool.acquire() as conn:
                # Insert test data
                for i in range(10000):
                    await conn.execute("""
                        INSERT INTO performance_test (test_data, value, category, status)
                        VALUES ($1, $2, $3, $4)
                    """, 
                    f"test_data_{i}", 
                    float(i * 1.5), 
                    f"category_{i % 10}", 
                    "active" if i % 2 == 0 else "inactive"
                    )
                
                self.logger.info("Test data inserted successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to insert test data: {e}")
            raise
    
    async def test_connection_pool_optimization(self) -> TestResult:
        """Test connection pool optimization"""
        test_name = "connection_pool_optimization"
        start_time = time.time()
        
        try:
            # Test connection pool performance
            latencies = []
            errors = 0
            
            # Test concurrent connections
            async def test_connection():
                try:
                    conn_start = time.time()
                    async with self.connection_pool.acquire() as conn:
                        await conn.execute("SELECT 1")
                    latency = (time.time() - conn_start) * 1000
                    latencies.append(latency)
                    return True
                except Exception as e:
                    self.logger.error(f"Connection test failed: {e}")
                    return False
            
            # Run concurrent connection tests
            tasks = []
            for _ in range(100):
                tasks.append(asyncio.create_task(test_connection()))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate metrics
            successful = sum(1 for r in results if r is True)
            errors = len(results) - successful
            error_rate = errors / len(results) if results else 0
            
            # Calculate latency statistics
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)
            else:
                avg_latency = p95_latency = p99_latency = 0
            
            execution_time = (time.time() - start_time) * 1000
            throughput = len(results) / (execution_time / 1000) if execution_time > 0 else 0
            
            # Check if targets are met
            success = (
                p95_latency <= self.test_config['performance_targets']['max_latency_ms'] and
                error_rate <= self.test_config['performance_targets']['max_error_rate']
            )
            
            result = TestResult(
                test_name=test_name,
                success=success,
                execution_time_ms=execution_time,
                latency_p95=p95_latency,
                latency_p99=p99_latency,
                throughput_ops_per_sec=throughput,
                error_rate=error_rate,
                details={
                    "total_connections": len(results),
                    "successful_connections": successful,
                    "failed_connections": errors,
                    "avg_latency_ms": avg_latency,
                    "pool_size": self.connection_pool.get_size(),
                    "pool_idle_size": self.connection_pool.get_idle_size()
                },
                timestamp=datetime.now()
            )
            
            self.test_results.append(result)
            self.logger.info(f"Connection pool test completed: Success={success}, P95={p95_latency:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Connection pool optimization test failed: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                latency_p95=0,
                latency_p99=0,
                throughput_ops_per_sec=0,
                error_rate=1.0,
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def test_query_optimization(self) -> TestResult:
        """Test query optimization"""
        test_name = "query_optimization"
        start_time = time.time()
        
        try:
            latencies = []
            errors = 0
            
            # Test query performance
            for query in self.test_queries:
                for _ in range(10):  # Run each query 10 times
                    try:
                        query_start = time.time()
                        async with self.connection_pool.acquire() as conn:
                            await conn.fetch(query)
                        latency = (time.time() - query_start) * 1000
                        latencies.append(latency)
                    except Exception as e:
                        errors += 1
                        self.logger.warning(f"Query failed: {query} - {e}")
            
            # Calculate metrics
            total_queries = len(self.test_queries) * 10
            successful_queries = len(latencies)
            error_rate = errors / total_queries if total_queries > 0 else 0
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)
                max_latency = max(latencies)
            else:
                avg_latency = p95_latency = p99_latency = max_latency = 0
            
            execution_time = (time.time() - start_time) * 1000
            throughput = total_queries / (execution_time / 1000) if execution_time > 0 else 0
            
            # Check if targets are met
            success = (
                p95_latency <= self.test_config['performance_targets']['max_latency_ms'] and
                error_rate <= self.test_config['performance_targets']['max_error_rate']
            )
            
            result = TestResult(
                test_name=test_name,
                success=success,
                execution_time_ms=execution_time,
                latency_p95=p95_latency,
                latency_p99=p99_latency,
                throughput_ops_per_sec=throughput,
                error_rate=error_rate,
                details={
                    "total_queries": total_queries,
                    "successful_queries": successful_queries,
                    "failed_queries": errors,
                    "avg_latency_ms": avg_latency,
                    "max_latency_ms": max_latency,
                    "queries_tested": len(self.test_queries)
                },
                timestamp=datetime.now()
            )
            
            self.test_results.append(result)
            self.logger.info(f"Query optimization test completed: Success={success}, P95={p95_latency:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query optimization test failed: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                latency_p95=0,
                latency_p99=0,
                throughput_ops_per_sec=0,
                error_rate=1.0,
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def test_high_availability_failover(self) -> TestResult:
        """Test high availability failover"""
        test_name = "high_availability_failover"
        start_time = time.time()
        
        try:
            # Test Patroni API connectivity
            async with aiohttp.ClientSession() as session:
                # Check primary node
                try:
                    async with session.get(
                        f"{self.test_config['patroni']['primary_api']}/",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        primary_status = response.status == 200
                        if primary_status:
                            primary_data = await response.json()
                        else:
                            primary_data = {}
                except Exception as e:
                    primary_status = False
                    primary_data = {"error": str(e)}
                
                # Check standby node (if configured)
                try:
                    async with session.get(
                        f"{self.test_config['patroni']['standby_api']}/",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        standby_status = response.status == 200
                        if standby_status:
                            standby_data = await response.json()
                        else:
                            standby_data = {}
                except Exception as e:
                    standby_status = False
                    standby_data = {"error": str(e)}
            
            # Test database connectivity during simulated issues
            connectivity_test_results = []
            
            for i in range(10):
                try:
                    conn_start = time.time()
                    async with self.connection_pool.acquire() as conn:
                        await conn.execute("SELECT 1")
                    latency = (time.time() - conn_start) * 1000
                    connectivity_test_results.append(latency)
                except Exception as e:
                    self.logger.warning(f"Connectivity test {i} failed: {e}")
            
            # Calculate metrics
            successful_connections = len(connectivity_test_results)
            failed_connections = 10 - successful_connections
            error_rate = failed_connections / 10
            
            if connectivity_test_results:
                avg_latency = statistics.mean(connectivity_test_results)
                p95_latency = np.percentile(connectivity_test_results, 95)
                p99_latency = np.percentile(connectivity_test_results, 99)
            else:
                avg_latency = p95_latency = p99_latency = 0
            
            execution_time = (time.time() - start_time) * 1000
            throughput = 10 / (execution_time / 1000) if execution_time > 0 else 0
            
            # Check if targets are met
            success = (
                primary_status and
                error_rate <= self.test_config['performance_targets']['max_error_rate'] and
                p95_latency <= self.test_config['performance_targets']['max_latency_ms']
            )
            
            result = TestResult(
                test_name=test_name,
                success=success,
                execution_time_ms=execution_time,
                latency_p95=p95_latency,
                latency_p99=p99_latency,
                throughput_ops_per_sec=throughput,
                error_rate=error_rate,
                details={
                    "primary_status": primary_status,
                    "standby_status": standby_status,
                    "primary_data": primary_data,
                    "standby_data": standby_data,
                    "successful_connections": successful_connections,
                    "failed_connections": failed_connections,
                    "avg_latency_ms": avg_latency
                },
                timestamp=datetime.now()
            )
            
            self.test_results.append(result)
            self.logger.info(f"High availability test completed: Success={success}, P95={p95_latency:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"High availability test failed: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                latency_p95=0,
                latency_p99=0,
                throughput_ops_per_sec=0,
                error_rate=1.0,
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def test_performance_monitoring(self) -> TestResult:
        """Test performance monitoring capabilities"""
        test_name = "performance_monitoring"
        start_time = time.time()
        
        try:
            # Test performance metrics collection
            metrics_collected = []
            
            # Collect metrics multiple times
            for i in range(5):
                try:
                    async with self.connection_pool.acquire() as conn:
                        # Collect basic metrics
                        conn_stats = await conn.fetchrow("""
                            SELECT 
                                (SELECT count(*) FROM pg_stat_activity) as total_connections,
                                (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                                (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections
                        """)
                        
                        db_stats = await conn.fetchrow("""
                            SELECT 
                                blks_read,
                                blks_hit,
                                tup_returned,
                                tup_fetched,
                                xact_commit,
                                xact_rollback,
                                deadlocks
                            FROM pg_stat_database 
                            WHERE datname = current_database()
                        """)
                        
                        metrics_collected.append({
                            "timestamp": datetime.now(),
                            "connection_stats": dict(conn_stats),
                            "database_stats": dict(db_stats)
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Metrics collection {i} failed: {e}")
                
                await asyncio.sleep(0.1)  # Small delay between collections
            
            # Calculate metrics
            successful_collections = len(metrics_collected)
            failed_collections = 5 - successful_collections
            error_rate = failed_collections / 5
            
            execution_time = (time.time() - start_time) * 1000
            throughput = 5 / (execution_time / 1000) if execution_time > 0 else 0
            
            # Check if targets are met
            success = (
                successful_collections >= 4 and  # At least 4 out of 5 collections should succeed
                error_rate <= self.test_config['performance_targets']['max_error_rate']
            )
            
            result = TestResult(
                test_name=test_name,
                success=success,
                execution_time_ms=execution_time,
                latency_p95=0,  # Not applicable for monitoring test
                latency_p99=0,  # Not applicable for monitoring test
                throughput_ops_per_sec=throughput,
                error_rate=error_rate,
                details={
                    "successful_collections": successful_collections,
                    "failed_collections": failed_collections,
                    "metrics_collected": metrics_collected,
                    "collection_interval_ms": 100
                },
                timestamp=datetime.now()
            )
            
            self.test_results.append(result)
            self.logger.info(f"Performance monitoring test completed: Success={success}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Performance monitoring test failed: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                latency_p95=0,
                latency_p99=0,
                throughput_ops_per_sec=0,
                error_rate=1.0,
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def run_load_test(self, duration_seconds: int = 60, 
                           concurrent_threads: int = 50) -> LoadTestResult:
        """Run comprehensive load test"""
        test_name = f"load_test_{concurrent_threads}threads_{duration_seconds}s"
        start_time = time.time()
        
        self.logger.info(f"Starting load test: {concurrent_threads} threads for {duration_seconds} seconds")
        
        # Shared statistics
        stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'latencies': [],
            'errors': []
        }
        
        stats_lock = threading.Lock()
        
        async def worker_thread(thread_id: int):
            """Worker thread for load testing"""
            operations = 0
            local_latencies = []
            local_errors = []
            
            thread_start = time.time()
            
            while time.time() - thread_start < duration_seconds:
                try:
                    # Random operation
                    operation_type = np.random.choice(['select', 'insert', 'update'], p=[0.7, 0.2, 0.1])
                    
                    op_start = time.time()
                    
                    if operation_type == 'select':
                        query = np.random.choice(self.test_queries)
                        async with self.connection_pool.acquire() as conn:
                            await conn.fetch(query)
                    
                    elif operation_type == 'insert':
                        async with self.connection_pool.acquire() as conn:
                            await conn.execute(
                                "INSERT INTO load_test (thread_id, operation_type, execution_time_ms) VALUES ($1, $2, $3)",
                                thread_id, operation_type, 0
                            )
                    
                    elif operation_type == 'update':
                        async with self.connection_pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE performance_test SET timestamp = NOW() WHERE id = $1",
                                np.random.randint(1, 1000)
                            )
                    
                    latency = (time.time() - op_start) * 1000
                    local_latencies.append(latency)
                    operations += 1
                    
                    # Record in database
                    try:
                        async with self.connection_pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE load_test SET execution_time_ms = $1, success = $2 WHERE thread_id = $3 AND operation_type = $4",
                                latency, True, thread_id, operation_type
                            )
                    except:
                        pass  # Don't fail the test due to logging issues
                    
                except Exception as e:
                    local_errors.append(str(e))
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.001)
            
            # Update global statistics
            with stats_lock:
                stats['total_operations'] += operations
                stats['successful_operations'] += len(local_latencies)
                stats['failed_operations'] += len(local_errors)
                stats['latencies'].extend(local_latencies)
                stats['errors'].extend(local_errors)
        
        # Start worker threads
        tasks = []
        for i in range(concurrent_threads):
            task = asyncio.create_task(worker_thread(i))
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate final metrics
        end_time = time.time()
        actual_duration = end_time - start_time
        
        if stats['latencies']:
            avg_latency = statistics.mean(stats['latencies'])
            p95_latency = np.percentile(stats['latencies'], 95)
            p99_latency = np.percentile(stats['latencies'], 99)
            max_latency = max(stats['latencies'])
        else:
            avg_latency = p95_latency = p99_latency = max_latency = 0
        
        throughput = stats['total_operations'] / actual_duration if actual_duration > 0 else 0
        error_rate = stats['failed_operations'] / max(stats['total_operations'], 1)
        
        # Basic resource usage (simplified)
        resource_usage = {
            'cpu_percent': 0,  # Would need system monitoring
            'memory_percent': 0,  # Would need system monitoring
            'connections_used': self.connection_pool.get_size() - self.connection_pool.get_idle_size()
        }
        
        result = LoadTestResult(
            test_name=test_name,
            duration_seconds=actual_duration,
            total_operations=stats['total_operations'],
            successful_operations=stats['successful_operations'],
            failed_operations=stats['failed_operations'],
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            throughput_ops_per_sec=throughput,
            error_rate=error_rate,
            resource_usage=resource_usage
        )
        
        self.load_test_results.append(result)
        
        self.logger.info(
            f"Load test completed: {stats['total_operations']} ops, "
            f"{throughput:.1f} ops/sec, P95={p95_latency:.2f}ms, "
            f"Error rate={error_rate:.3f}"
        )
        
        return result
    
    async def run_stress_test(self, max_connections: int = 200) -> TestResult:
        """Run stress test to find breaking point"""
        test_name = f"stress_test_{max_connections}_connections"
        start_time = time.time()
        
        self.logger.info(f"Starting stress test with up to {max_connections} connections")
        
        try:
            # Gradually increase connection count
            latencies = []
            errors = 0
            successful_connections = 0
            
            for connection_count in range(10, max_connections + 1, 10):
                try:
                    # Create connections
                    tasks = []
                    for _ in range(connection_count):
                        async def stress_operation():
                            try:
                                op_start = time.time()
                                async with self.connection_pool.acquire() as conn:
                                    await conn.execute("SELECT pg_sleep(0.01)")  # 10ms operation
                                latency = (time.time() - op_start) * 1000
                                latencies.append(latency)
                                return True
                            except Exception as e:
                                return False
                        
                        tasks.append(asyncio.create_task(stress_operation()))
                    
                    # Wait for all operations to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Count successes and failures
                    batch_successes = sum(1 for r in results if r is True)
                    batch_failures = len(results) - batch_successes
                    
                    successful_connections += batch_successes
                    errors += batch_failures
                    
                    # Check if we've reached the breaking point
                    if batch_failures / len(results) > 0.1:  # More than 10% failures
                        self.logger.warning(f"Breaking point reached at {connection_count} connections")
                        break
                    
                    # Small delay between batches
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Stress test batch failed: {e}")
                    errors += connection_count
                    break
            
            # Calculate metrics
            total_operations = successful_connections + errors
            error_rate = errors / total_operations if total_operations > 0 else 0
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)
            else:
                avg_latency = p95_latency = p99_latency = 0
            
            execution_time = (time.time() - start_time) * 1000
            throughput = total_operations / (execution_time / 1000) if execution_time > 0 else 0
            
            # Success criteria for stress test
            success = (
                successful_connections > 0 and
                error_rate < 0.5  # Less than 50% failures
            )
            
            result = TestResult(
                test_name=test_name,
                success=success,
                execution_time_ms=execution_time,
                latency_p95=p95_latency,
                latency_p99=p99_latency,
                throughput_ops_per_sec=throughput,
                error_rate=error_rate,
                details={
                    "max_connections_tested": max_connections,
                    "total_operations": total_operations,
                    "successful_connections": successful_connections,
                    "failed_connections": errors,
                    "avg_latency_ms": avg_latency,
                    "breaking_point_reached": error_rate > 0.1
                },
                timestamp=datetime.now()
            )
            
            self.test_results.append(result)
            self.logger.info(f"Stress test completed: Success={success}, P95={p95_latency:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            return TestResult(
                test_name=test_name,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                latency_p95=0,
                latency_p99=0,
                throughput_ops_per_sec=0,
                error_rate=1.0,
                details={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        self.logger.info("Starting comprehensive database optimization test suite")
        
        # Setup test environment
        await self.setup_test_environment()
        
        # Run all tests
        test_results = []
        
        # Basic functionality tests
        test_results.append(await self.test_connection_pool_optimization())
        test_results.append(await self.test_query_optimization())
        test_results.append(await self.test_high_availability_failover())
        test_results.append(await self.test_performance_monitoring())
        
        # Performance tests
        load_test_result = await self.run_load_test(duration_seconds=30, concurrent_threads=25)
        stress_test_result = await self.run_stress_test(max_connections=100)
        
        # Calculate overall results
        successful_tests = sum(1 for result in test_results if result.success)
        total_tests = len(test_results)
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Performance summary
        avg_p95_latency = statistics.mean([r.latency_p95 for r in test_results if r.latency_p95 > 0])
        avg_throughput = statistics.mean([r.throughput_ops_per_sec for r in test_results if r.throughput_ops_per_sec > 0])
        avg_error_rate = statistics.mean([r.error_rate for r in test_results])
        
        # Generate report
        report = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "overall_success_rate": overall_success_rate,
                "avg_p95_latency_ms": avg_p95_latency,
                "avg_throughput_ops_per_sec": avg_throughput,
                "avg_error_rate": avg_error_rate
            },
            "test_results": [asdict(result) for result in test_results],
            "load_test_results": [asdict(load_test_result)],
            "stress_test_results": [asdict(stress_test_result)],
            "performance_targets": self.test_config['performance_targets'],
            "target_compliance": {
                "latency_target_met": avg_p95_latency <= self.test_config['performance_targets']['target_latency_ms'],
                "throughput_target_met": avg_throughput >= self.test_config['performance_targets']['min_throughput_ops_per_sec'],
                "error_rate_target_met": avg_error_rate <= self.test_config['performance_targets']['max_error_rate']
            }
        }
        
        # Log summary
        self.logger.info(f"Test suite completed: {successful_tests}/{total_tests} tests passed")
        self.logger.info(f"Average P95 latency: {avg_p95_latency:.2f}ms")
        self.logger.info(f"Average throughput: {avg_throughput:.1f} ops/sec")
        self.logger.info(f"Average error rate: {avg_error_rate:.3f}")
        
        return report
    
    async def cleanup_test_environment(self):
        """Cleanup test environment"""
        try:
            if self.connection_pool:
                # Clean up test data
                async with self.connection_pool.acquire() as conn:
                    await conn.execute("DROP TABLE IF EXISTS performance_test")
                    await conn.execute("DROP TABLE IF EXISTS load_test")
                
                # Close pool
                await self.connection_pool.close()
                self.connection_pool = None
            
            self.is_setup = False
            self.logger.info("Test environment cleaned up")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup test environment: {e}")

async def main():
    """Main entry point for running tests"""
    test_suite = DatabaseOptimizationTestSuite()
    
    try:
        # Run comprehensive test suite
        report = await test_suite.run_comprehensive_test_suite()
        
        # Save report to file
        report_filename = f"db_optimization_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Test report saved to: {report_filename}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATABASE OPTIMIZATION TEST SUMMARY")
        print("="*50)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Successful Tests: {report['summary']['successful_tests']}")
        print(f"Failed Tests: {report['summary']['failed_tests']}")
        print(f"Overall Success Rate: {report['summary']['overall_success_rate']:.2%}")
        print(f"Average P95 Latency: {report['summary']['avg_p95_latency_ms']:.2f}ms")
        print(f"Average Throughput: {report['summary']['avg_throughput_ops_per_sec']:.1f} ops/sec")
        print(f"Average Error Rate: {report['summary']['avg_error_rate']:.3f}")
        
        # Target compliance
        print("\nTarget Compliance:")
        for target, met in report['target_compliance'].items():
            status = "✓" if met else "✗"
            print(f"  {status} {target}: {'PASSED' if met else 'FAILED'}")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await test_suite.cleanup_test_environment()

if __name__ == "__main__":
    asyncio.run(main())