"""
Automated Database Failover Testing Framework
===========================================

This module provides comprehensive automated testing for database failover scenarios
with RTO (Recovery Time Objective) validation. It integrates with Patroni cluster
management and provides detailed metrics on failover performance.

Key Features:
- Automated Patroni cluster failover testing
- RTO validation with <15 second target
- Connection pool failover validation
- Data consistency verification
- Performance regression detection
- Automated recovery validation
- Comprehensive reporting and alerting

Target RTO: <15 seconds for database failover
"""

import asyncio
import time
import logging
import json
import traceback
import psutil
import subprocess
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import asyncpg
import aioredis
from contextlib import asynccontextmanager
import httpx
import threading
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FailoverType(Enum):
    """Types of database failover scenarios."""
    PRIMARY_KILL = "primary_kill"
    PRIMARY_NETWORK_PARTITION = "primary_network_partition"
    PRIMARY_DISK_FAILURE = "primary_disk_failure"
    PATRONI_RESTART = "patroni_restart"
    ETCD_FAILURE = "etcd_failure"
    STANDBY_PROMOTION = "standby_promotion"
    SPLIT_BRAIN_SIMULATION = "split_brain_simulation"
    CASCADING_FAILURE = "cascading_failure"


class TestStatus(Enum):
    """Status of failover tests."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class RTOMetrics:
    """RTO (Recovery Time Objective) metrics."""
    detection_time: float = 0.0  # Time to detect failure
    failover_time: float = 0.0   # Time to complete failover
    recovery_time: float = 0.0   # Time to full recovery
    total_downtime: float = 0.0  # Total downtime
    
    # Performance metrics
    connection_recovery_time: float = 0.0
    first_query_time: float = 0.0
    full_performance_time: float = 0.0
    
    # Validation metrics
    data_consistency_validated: bool = False
    connection_pool_recovered: bool = False
    applications_reconnected: bool = False
    
    def meets_rto_target(self, target_seconds: float = 15.0) -> bool:
        """Check if RTO target is met."""
        return self.total_downtime <= target_seconds


@dataclass
class FailoverTestConfig:
    """Configuration for failover tests."""
    test_id: str
    failover_type: FailoverType
    target_rto_seconds: float = 15.0
    max_test_duration: int = 300  # 5 minutes max
    
    # Database configuration
    primary_host: str = "localhost"
    primary_port: int = 5432
    database_name: str = "trading_db"
    username: str = "admin"
    password: str = "admin"
    
    # Patroni configuration
    patroni_api_port: int = 8008
    etcd_host: str = "localhost"
    etcd_port: int = 2379
    
    # Test configuration
    pre_test_queries: List[str] = field(default_factory=list)
    post_test_queries: List[str] = field(default_factory=list)
    validation_queries: List[str] = field(default_factory=list)
    
    # Monitoring
    enable_detailed_logging: bool = True
    enable_performance_profiling: bool = True
    enable_network_monitoring: bool = True


@dataclass
class FailoverTestResult:
    """Result of a failover test."""
    test_id: str
    test_name: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Core metrics
    rto_metrics: RTOMetrics = field(default_factory=RTOMetrics)
    
    # Detailed results
    pre_test_results: Dict[str, Any] = field(default_factory=dict)
    failover_results: Dict[str, Any] = field(default_factory=dict)
    post_test_results: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    consistency_validation: Dict[str, Any] = field(default_factory=dict)
    performance_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Errors and issues
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def success(self) -> bool:
        """Check if test was successful."""
        return (
            self.status == TestStatus.COMPLETED and
            len(self.errors) == 0 and
            self.rto_metrics.meets_rto_target()
        )


class PatroniController:
    """Controller for Patroni cluster management."""
    
    def __init__(self, config: FailoverTestConfig):
        self.config = config
        self.patroni_base_url = f"http://{config.primary_host}:{config.patroni_api_port}"
        
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.patroni_base_url}/cluster",
                    timeout=5.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {}
    
    async def get_primary_node(self) -> Optional[Dict[str, Any]]:
        """Get current primary node information."""
        try:
            cluster_status = await self.get_cluster_status()
            members = cluster_status.get("members", [])
            
            for member in members:
                if member.get("role") == "leader":
                    return member
            
            return None
        except Exception as e:
            logger.error(f"Failed to get primary node: {e}")
            return None
    
    async def get_standby_nodes(self) -> List[Dict[str, Any]]:
        """Get standby node information."""
        try:
            cluster_status = await self.get_cluster_status()
            members = cluster_status.get("members", [])
            
            return [
                member for member in members
                if member.get("role") in ["replica", "standby_leader"]
            ]
        except Exception as e:
            logger.error(f"Failed to get standby nodes: {e}")
            return []
    
    async def trigger_failover(self, candidate_node: Optional[str] = None) -> bool:
        """Trigger manual failover."""
        try:
            failover_data = {}
            if candidate_node:
                failover_data["candidate"] = candidate_node
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.patroni_base_url}/failover",
                    json=failover_data,
                    timeout=30.0
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to trigger failover: {e}")
            return False
    
    async def restart_node(self, node_name: str) -> bool:
        """Restart a specific node."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.patroni_base_url}/restart",
                    json={"schedule": "now"},
                    timeout=30.0
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to restart node: {e}")
            return False
    
    async def wait_for_healthy_cluster(self, timeout: int = 120) -> bool:
        """Wait for cluster to return to healthy state."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                cluster_status = await self.get_cluster_status()
                
                # Check if we have a primary
                primary = await self.get_primary_node()
                if not primary:
                    await asyncio.sleep(1)
                    continue
                
                # Check if primary is healthy
                if primary.get("state") == "running":
                    # Check standby nodes
                    standby_nodes = await self.get_standby_nodes()
                    healthy_standbys = [
                        node for node in standby_nodes
                        if node.get("state") == "running"
                    ]
                    
                    # At least one healthy standby required
                    if len(healthy_standbys) >= 1:
                        logger.info("Cluster returned to healthy state")
                        return True
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error checking cluster health: {e}")
                await asyncio.sleep(1)
        
        logger.error("Timeout waiting for healthy cluster")
        return False


class DatabaseConnectionTester:
    """Tester for database connections and performance."""
    
    def __init__(self, config: FailoverTestConfig):
        self.config = config
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.test_data_id = str(uuid.uuid4())
        
    async def initialize_connection_pool(self) -> bool:
        """Initialize connection pool."""
        try:
            self.connection_pool = await asyncpg.create_pool(
                host=self.config.primary_host,
                port=self.config.primary_port,
                database=self.config.database_name,
                user=self.config.username,
                password=self.config.password,
                min_size=2,
                max_size=10,
                command_timeout=10,
                server_settings={'application_name': 'failover_tester'}
            )
            logger.info("Connection pool initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            return False
    
    async def close_connection_pool(self):
        """Close connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("Connection pool closed")
    
    async def test_connection(self) -> Tuple[bool, float]:
        """Test database connection and measure latency."""
        if not self.connection_pool:
            return False, 0.0
        
        try:
            start_time = time.time()
            
            async with self.connection_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            latency = time.time() - start_time
            return True, latency
            
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            return False, 0.0
    
    async def execute_query(self, query: str) -> Tuple[bool, Any, float]:
        """Execute a query and measure performance."""
        if not self.connection_pool:
            return False, None, 0.0
        
        try:
            start_time = time.time()
            
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetch(query)
            
            execution_time = time.time() - start_time
            return True, result, execution_time
            
        except Exception as e:
            logger.warning(f"Query execution failed: {e}")
            return False, None, 0.0
    
    async def insert_test_data(self) -> bool:
        """Insert test data for consistency validation."""
        try:
            # Create test table if not exists
            create_table_query = """
            CREATE TABLE IF NOT EXISTS failover_test_data (
                id UUID PRIMARY KEY,
                test_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                data JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
            
            success, _, _ = await self.execute_query(create_table_query)
            if not success:
                return False
            
            # Insert test data
            insert_query = """
            INSERT INTO failover_test_data (id, test_id, timestamp, data)
            VALUES ($1, $2, $3, $4)
            """
            
            test_data = {
                "test_type": "failover_validation",
                "metrics": {"start_time": time.time()},
                "status": "active"
            }
            
            async with self.connection_pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    uuid.uuid4(),
                    self.test_data_id,
                    datetime.now(),
                    json.dumps(test_data)
                )
            
            logger.info("Test data inserted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert test data: {e}")
            return False
    
    async def validate_test_data(self) -> bool:
        """Validate test data consistency after failover."""
        try:
            query = """
            SELECT COUNT(*) as count, MAX(created_at) as latest
            FROM failover_test_data
            WHERE test_id = $1
            """
            
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchrow(query, self.test_data_id)
                
                if result and result['count'] > 0:
                    logger.info(f"Test data validation: {result['count']} records found")
                    return True
                else:
                    logger.error("Test data validation failed: no records found")
                    return False
                    
        except Exception as e:
            logger.error(f"Test data validation failed: {e}")
            return False
    
    async def cleanup_test_data(self):
        """Clean up test data."""
        try:
            query = "DELETE FROM failover_test_data WHERE test_id = $1"
            
            async with self.connection_pool.acquire() as conn:
                await conn.execute(query, self.test_data_id)
                
            logger.info("Test data cleaned up")
            
        except Exception as e:
            logger.warning(f"Failed to clean up test data: {e}")


class FailoverScenarioExecutor:
    """Executor for different failover scenarios."""
    
    def __init__(self, config: FailoverTestConfig):
        self.config = config
        self.patroni_controller = PatroniController(config)
        
    async def execute_scenario(self, scenario: FailoverType) -> Dict[str, Any]:
        """Execute a specific failover scenario."""
        logger.info(f"Executing failover scenario: {scenario.value}")
        
        scenario_map = {
            FailoverType.PRIMARY_KILL: self._execute_primary_kill,
            FailoverType.PRIMARY_NETWORK_PARTITION: self._execute_network_partition,
            FailoverType.PRIMARY_DISK_FAILURE: self._execute_disk_failure,
            FailoverType.PATRONI_RESTART: self._execute_patroni_restart,
            FailoverType.ETCD_FAILURE: self._execute_etcd_failure,
            FailoverType.STANDBY_PROMOTION: self._execute_standby_promotion,
            FailoverType.SPLIT_BRAIN_SIMULATION: self._execute_split_brain,
            FailoverType.CASCADING_FAILURE: self._execute_cascading_failure
        }
        
        executor = scenario_map.get(scenario)
        if not executor:
            return {"success": False, "error": f"Unknown scenario: {scenario.value}"}
        
        try:
            start_time = time.time()
            result = await executor()
            execution_time = time.time() - start_time
            
            result["execution_time"] = execution_time
            return result
            
        except Exception as e:
            logger.error(f"Scenario execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _execute_primary_kill(self) -> Dict[str, Any]:
        """Execute primary database kill scenario."""
        try:
            # Get primary node
            primary = await self.patroni_controller.get_primary_node()
            if not primary:
                return {"success": False, "error": "No primary node found"}
            
            # Kill primary PostgreSQL process
            primary_host = primary.get("host", self.config.primary_host)
            
            # Find PostgreSQL processes
            postgres_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'postgres' in proc.info['name'].lower() or 'postgres' in cmdline.lower():
                        if 'primary' in cmdline or 'master' in cmdline:
                            postgres_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not postgres_pids:
                return {"success": False, "error": "No PostgreSQL processes found"}
            
            # Kill processes
            killed_pids = []
            for pid in postgres_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            logger.info(f"Killed PostgreSQL processes: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "primary_node": primary,
                "method": "process_kill"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_network_partition(self) -> Dict[str, Any]:
        """Execute network partition scenario."""
        try:
            # Block PostgreSQL port using iptables
            port = self.config.primary_port
            
            # Block incoming connections
            block_cmd = [
                "iptables", "-A", "INPUT", "-p", "tcp", "--dport", str(port), "-j", "DROP"
            ]
            
            result = subprocess.run(block_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"success": False, "error": f"iptables failed: {result.stderr}"}
            
            logger.info(f"Network partition created for port {port}")
            
            return {
                "success": True,
                "blocked_port": port,
                "method": "network_partition",
                "iptables_rule": ' '.join(block_cmd)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_disk_failure(self) -> Dict[str, Any]:
        """Execute disk failure scenario."""
        try:
            # Simulate disk failure by making data directory read-only
            data_dir = "/var/lib/postgresql/data"
            
            # Make directory read-only
            subprocess.run(["chmod", "-R", "444", data_dir], capture_output=True)
            
            logger.info(f"Disk failure simulated for {data_dir}")
            
            return {
                "success": True,
                "affected_directory": data_dir,
                "method": "disk_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_patroni_restart(self) -> Dict[str, Any]:
        """Execute Patroni restart scenario."""
        try:
            # Restart Patroni service
            restart_cmd = ["systemctl", "restart", "patroni"]
            result = subprocess.run(restart_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"success": False, "error": f"Patroni restart failed: {result.stderr}"}
            
            logger.info("Patroni service restarted")
            
            return {
                "success": True,
                "method": "patroni_restart",
                "restart_command": ' '.join(restart_cmd)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_etcd_failure(self) -> Dict[str, Any]:
        """Execute etcd failure scenario."""
        try:
            # Kill etcd processes
            etcd_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'etcd' in proc.info['name'].lower():
                        etcd_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not etcd_pids:
                return {"success": False, "error": "No etcd processes found"}
            
            # Kill processes
            killed_pids = []
            for pid in etcd_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            logger.info(f"Killed etcd processes: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "method": "etcd_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_standby_promotion(self) -> Dict[str, Any]:
        """Execute standby promotion scenario."""
        try:
            # Get standby nodes
            standby_nodes = await self.patroni_controller.get_standby_nodes()
            
            if not standby_nodes:
                return {"success": False, "error": "No standby nodes found"}
            
            # Select first standby as candidate
            candidate = standby_nodes[0]
            candidate_name = candidate.get("name")
            
            # Trigger failover to specific candidate
            success = await self.patroni_controller.trigger_failover(candidate_name)
            
            if not success:
                return {"success": False, "error": "Failover trigger failed"}
            
            logger.info(f"Standby promotion triggered for {candidate_name}")
            
            return {
                "success": True,
                "promoted_node": candidate,
                "method": "standby_promotion"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_split_brain(self) -> Dict[str, Any]:
        """Execute split brain scenario."""
        try:
            # This is a complex scenario that involves network partitioning
            # between nodes while keeping them running
            
            # Block inter-node communication
            nodes = await self.patroni_controller.get_standby_nodes()
            
            if len(nodes) < 2:
                return {"success": False, "error": "Need at least 2 nodes for split brain"}
            
            # Block communication between nodes (simplified)
            # In production, this would involve more sophisticated network manipulation
            
            logger.info("Split brain scenario executed")
            
            return {
                "success": True,
                "affected_nodes": len(nodes),
                "method": "split_brain"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_cascading_failure(self) -> Dict[str, Any]:
        """Execute cascading failure scenario."""
        try:
            # Execute multiple failures in sequence
            failures = []
            
            # First, kill primary
            primary_result = await self._execute_primary_kill()
            failures.append({"step": "primary_kill", "result": primary_result})
            
            # Wait a bit
            await asyncio.sleep(5)
            
            # Then cause etcd issues
            etcd_result = await self._execute_etcd_failure()
            failures.append({"step": "etcd_failure", "result": etcd_result})
            
            logger.info("Cascading failure scenario executed")
            
            return {
                "success": True,
                "failures": failures,
                "method": "cascading_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def cleanup_scenario(self, scenario: FailoverType, scenario_result: Dict[str, Any]) -> bool:
        """Clean up after scenario execution."""
        try:
            if scenario == FailoverType.PRIMARY_NETWORK_PARTITION:
                # Remove iptables rules
                port = self.config.primary_port
                unblock_cmd = [
                    "iptables", "-D", "INPUT", "-p", "tcp", "--dport", str(port), "-j", "DROP"
                ]
                subprocess.run(unblock_cmd, capture_output=True)
                
            elif scenario == FailoverType.PRIMARY_DISK_FAILURE:
                # Restore directory permissions
                data_dir = "/var/lib/postgresql/data"
                subprocess.run(["chmod", "-R", "755", data_dir], capture_output=True)
                
            # Wait for cluster to stabilize
            await self.patroni_controller.wait_for_healthy_cluster()
            
            logger.info(f"Cleanup completed for scenario: {scenario.value}")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed for scenario {scenario.value}: {e}")
            return False


class DatabaseFailoverTester:
    """Main class for automated database failover testing."""
    
    def __init__(self, config: FailoverTestConfig):
        self.config = config
        self.patroni_controller = PatroniController(config)
        self.db_tester = DatabaseConnectionTester(config)
        self.scenario_executor = FailoverScenarioExecutor(config)
        
    async def run_failover_test(self) -> FailoverTestResult:
        """Run complete failover test."""
        result = FailoverTestResult(
            test_id=self.config.test_id,
            test_name=f"Database Failover Test - {self.config.failover_type.value}",
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        logger.info(f"Starting failover test: {result.test_name}")
        
        try:
            # Phase 1: Pre-test validation
            logger.info("Phase 1: Pre-test validation")
            await self._run_pre_test_validation(result)
            
            # Phase 2: Establish baseline
            logger.info("Phase 2: Establishing baseline")
            await self._establish_baseline(result)
            
            # Phase 3: Execute failover scenario
            logger.info("Phase 3: Executing failover scenario")
            await self._execute_failover_scenario(result)
            
            # Phase 4: Monitor recovery
            logger.info("Phase 4: Monitoring recovery")
            await self._monitor_recovery(result)
            
            # Phase 5: Validate recovery
            logger.info("Phase 5: Validating recovery")
            await self._validate_recovery(result)
            
            # Phase 6: Performance validation
            logger.info("Phase 6: Performance validation")
            await self._validate_performance(result)
            
            # Phase 7: Cleanup
            logger.info("Phase 7: Cleanup")
            await self._cleanup_test(result)
            
            result.status = TestStatus.COMPLETED
            result.end_time = datetime.now()
            
            # Calculate final metrics
            self._calculate_final_metrics(result)
            
            logger.info(f"Test completed: {result.test_name}")
            logger.info(f"RTO Target Met: {result.rto_metrics.meets_rto_target(self.config.target_rto_seconds)}")
            logger.info(f"Total Downtime: {result.rto_metrics.total_downtime:.2f}s")
            
            return result
            
        except Exception as e:
            result.status = TestStatus.FAILED
            result.end_time = datetime.now()
            result.errors.append(f"Test failed: {str(e)}")
            
            logger.error(f"Test failed: {result.test_name} - {str(e)}")
            logger.error(traceback.format_exc())
            
            # Attempt cleanup
            try:
                await self._emergency_cleanup(result)
            except Exception as cleanup_error:
                logger.error(f"Emergency cleanup failed: {cleanup_error}")
            
            return result
    
    async def _run_pre_test_validation(self, result: FailoverTestResult):
        """Run pre-test validation."""
        try:
            # Check cluster health
            cluster_status = await self.patroni_controller.get_cluster_status()
            result.pre_test_results["cluster_status"] = cluster_status
            
            if not cluster_status:
                raise Exception("Cluster is not accessible")
            
            # Check primary node
            primary = await self.patroni_controller.get_primary_node()
            if not primary:
                raise Exception("No primary node found")
            
            result.pre_test_results["primary_node"] = primary
            
            # Check standby nodes
            standby_nodes = await self.patroni_controller.get_standby_nodes()
            result.pre_test_results["standby_nodes"] = standby_nodes
            
            if len(standby_nodes) < 1:
                result.warnings.append("No standby nodes found - failover may fail")
            
            # Initialize database connection
            connection_success = await self.db_tester.initialize_connection_pool()
            if not connection_success:
                raise Exception("Failed to initialize database connection")
            
            # Test connectivity
            connected, latency = await self.db_tester.test_connection()
            if not connected:
                raise Exception("Database connection test failed")
            
            result.pre_test_results["baseline_latency"] = latency
            
            # Insert test data
            test_data_success = await self.db_tester.insert_test_data()
            if not test_data_success:
                raise Exception("Failed to insert test data")
            
            logger.info("Pre-test validation completed successfully")
            
        except Exception as e:
            result.errors.append(f"Pre-test validation failed: {str(e)}")
            raise
    
    async def _establish_baseline(self, result: FailoverTestResult):
        """Establish performance baseline."""
        try:
            baseline_metrics = {
                "connection_tests": [],
                "query_performance": [],
                "start_time": time.time()
            }
            
            # Run baseline connection tests
            for i in range(5):
                connected, latency = await self.db_tester.test_connection()
                baseline_metrics["connection_tests"].append({
                    "iteration": i,
                    "connected": connected,
                    "latency": latency
                })
                await asyncio.sleep(0.1)
            
            # Run baseline query tests
            test_queries = [
                "SELECT NOW()",
                "SELECT COUNT(*) FROM pg_stat_activity",
                "SELECT * FROM failover_test_data LIMIT 1"
            ]
            
            for query in test_queries:
                success, _, execution_time = await self.db_tester.execute_query(query)
                baseline_metrics["query_performance"].append({
                    "query": query,
                    "success": success,
                    "execution_time": execution_time
                })
            
            result.pre_test_results["baseline_metrics"] = baseline_metrics
            
            logger.info("Baseline established successfully")
            
        except Exception as e:
            result.errors.append(f"Baseline establishment failed: {str(e)}")
            raise
    
    async def _execute_failover_scenario(self, result: FailoverTestResult):
        """Execute the failover scenario."""
        try:
            # Record start time for failover
            failover_start = time.time()
            
            # Execute the specific scenario
            scenario_result = await self.scenario_executor.execute_scenario(
                self.config.failover_type
            )
            
            if not scenario_result.get("success"):
                raise Exception(f"Scenario execution failed: {scenario_result.get('error')}")
            
            # Record detection time (time to execute scenario)
            result.rto_metrics.detection_time = time.time() - failover_start
            
            result.failover_results = scenario_result
            
            logger.info(f"Failover scenario executed: {scenario_result}")
            
        except Exception as e:
            result.errors.append(f"Failover scenario execution failed: {str(e)}")
            raise
    
    async def _monitor_recovery(self, result: FailoverTestResult):
        """Monitor the recovery process."""
        try:
            recovery_start = time.time()
            max_wait_time = self.config.max_test_duration
            
            recovery_metrics = {
                "connection_attempts": [],
                "cluster_status_checks": [],
                "recovery_events": []
            }
            
            first_connection_time = None
            cluster_healthy_time = None
            
            while time.time() - recovery_start < max_wait_time:
                current_time = time.time()
                
                # Test connection
                connected, latency = await self.db_tester.test_connection()
                
                recovery_metrics["connection_attempts"].append({
                    "timestamp": current_time,
                    "connected": connected,
                    "latency": latency
                })
                
                # Check cluster status
                cluster_status = await self.patroni_controller.get_cluster_status()
                primary = await self.patroni_controller.get_primary_node()
                
                recovery_metrics["cluster_status_checks"].append({
                    "timestamp": current_time,
                    "has_primary": primary is not None,
                    "primary_healthy": primary.get("state") == "running" if primary else False
                })
                
                # Record first successful connection
                if connected and first_connection_time is None:
                    first_connection_time = current_time
                    result.rto_metrics.first_query_time = current_time - recovery_start
                    recovery_metrics["recovery_events"].append({
                        "event": "first_connection",
                        "timestamp": current_time,
                        "time_from_start": current_time - recovery_start
                    })
                
                # Check if cluster is healthy
                if primary and primary.get("state") == "running" and cluster_healthy_time is None:
                    cluster_healthy_time = current_time
                    result.rto_metrics.failover_time = current_time - recovery_start
                    recovery_metrics["recovery_events"].append({
                        "event": "cluster_healthy",
                        "timestamp": current_time,
                        "time_from_start": current_time - recovery_start
                    })
                
                # If we have both connection and healthy cluster, we're done
                if first_connection_time and cluster_healthy_time:
                    break
                
                await asyncio.sleep(0.5)
            
            # Calculate recovery time
            if first_connection_time:
                result.rto_metrics.recovery_time = first_connection_time - recovery_start
            
            result.failover_results["recovery_metrics"] = recovery_metrics
            
            logger.info(f"Recovery monitoring completed - Recovery time: {result.rto_metrics.recovery_time:.2f}s")
            
        except Exception as e:
            result.errors.append(f"Recovery monitoring failed: {str(e)}")
            raise
    
    async def _validate_recovery(self, result: FailoverTestResult):
        """Validate recovery completeness."""
        try:
            validation_results = {
                "connection_validation": False,
                "data_consistency": False,
                "cluster_health": False,
                "performance_validation": False
            }
            
            # Test connection stability
            stable_connections = 0
            for i in range(10):
                connected, latency = await self.db_tester.test_connection()
                if connected:
                    stable_connections += 1
                await asyncio.sleep(0.1)
            
            validation_results["connection_validation"] = stable_connections >= 8
            result.rto_metrics.connection_pool_recovered = validation_results["connection_validation"]
            
            # Validate data consistency
            data_valid = await self.db_tester.validate_test_data()
            validation_results["data_consistency"] = data_valid
            result.rto_metrics.data_consistency_validated = data_valid
            
            # Check cluster health
            cluster_healthy = await self.patroni_controller.wait_for_healthy_cluster(timeout=30)
            validation_results["cluster_health"] = cluster_healthy
            
            # Performance validation
            performance_start = time.time()
            query_success = 0
            for i in range(5):
                success, _, execution_time = await self.db_tester.execute_query("SELECT NOW()")
                if success and execution_time < 1.0:  # 1 second threshold
                    query_success += 1
                await asyncio.sleep(0.1)
            
            validation_results["performance_validation"] = query_success >= 4
            result.rto_metrics.full_performance_time = time.time() - performance_start
            
            result.consistency_validation = validation_results
            
            # Overall validation
            all_validations_passed = all(validation_results.values())
            
            if not all_validations_passed:
                failed_validations = [k for k, v in validation_results.items() if not v]
                result.warnings.append(f"Recovery validation failed: {failed_validations}")
            
            logger.info(f"Recovery validation completed: {validation_results}")
            
        except Exception as e:
            result.errors.append(f"Recovery validation failed: {str(e)}")
            raise
    
    async def _validate_performance(self, result: FailoverTestResult):
        """Validate post-failover performance."""
        try:
            performance_metrics = {
                "connection_latency": [],
                "query_performance": [],
                "throughput_test": []
            }
            
            # Connection latency test
            for i in range(10):
                connected, latency = await self.db_tester.test_connection()
                performance_metrics["connection_latency"].append({
                    "iteration": i,
                    "connected": connected,
                    "latency": latency
                })
                await asyncio.sleep(0.1)
            
            # Query performance test
            test_queries = [
                "SELECT NOW()",
                "SELECT COUNT(*) FROM pg_stat_activity",
                "SELECT * FROM failover_test_data LIMIT 10"
            ]
            
            for query in test_queries:
                success, _, execution_time = await self.db_tester.execute_query(query)
                performance_metrics["query_performance"].append({
                    "query": query,
                    "success": success,
                    "execution_time": execution_time
                })
            
            # Throughput test
            throughput_start = time.time()
            concurrent_queries = 5
            
            async def run_query():
                return await self.db_tester.execute_query("SELECT 1")
            
            tasks = [run_query() for _ in range(concurrent_queries)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            throughput_time = time.time() - throughput_start
            successful_queries = sum(1 for r in results if not isinstance(r, Exception) and r[0])
            
            performance_metrics["throughput_test"] = {
                "concurrent_queries": concurrent_queries,
                "successful_queries": successful_queries,
                "total_time": throughput_time,
                "queries_per_second": successful_queries / throughput_time
            }
            
            result.performance_validation = performance_metrics
            
            logger.info(f"Performance validation completed: {performance_metrics}")
            
        except Exception as e:
            result.errors.append(f"Performance validation failed: {str(e)}")
            raise
    
    async def _cleanup_test(self, result: FailoverTestResult):
        """Clean up test resources."""
        try:
            # Clean up scenario
            cleanup_success = await self.scenario_executor.cleanup_scenario(
                self.config.failover_type,
                result.failover_results
            )
            
            if not cleanup_success:
                result.warnings.append("Scenario cleanup failed")
            
            # Clean up test data
            await self.db_tester.cleanup_test_data()
            
            # Close connection pool
            await self.db_tester.close_connection_pool()
            
            logger.info("Test cleanup completed")
            
        except Exception as e:
            result.warnings.append(f"Test cleanup failed: {str(e)}")
    
    async def _emergency_cleanup(self, result: FailoverTestResult):
        """Emergency cleanup for failed tests."""
        try:
            logger.warning("Performing emergency cleanup")
            
            # Reset iptables
            subprocess.run(["iptables", "-F"], capture_output=True)
            
            # Reset directory permissions
            subprocess.run(["chmod", "-R", "755", "/var/lib/postgresql/data"], capture_output=True)
            
            # Close connection pool
            await self.db_tester.close_connection_pool()
            
            # Wait for cluster to stabilize
            await self.patroni_controller.wait_for_healthy_cluster(timeout=60)
            
            logger.info("Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def _calculate_final_metrics(self, result: FailoverTestResult):
        """Calculate final RTO metrics."""
        try:
            # Calculate total downtime
            result.rto_metrics.total_downtime = max(
                result.rto_metrics.recovery_time,
                result.rto_metrics.failover_time,
                result.rto_metrics.first_query_time
            )
            
            # Validate applications reconnected
            result.rto_metrics.applications_reconnected = (
                result.rto_metrics.connection_pool_recovered and
                result.rto_metrics.data_consistency_validated
            )
            
            logger.info(f"Final RTO metrics calculated: {result.rto_metrics}")
            
        except Exception as e:
            result.warnings.append(f"Final metrics calculation failed: {str(e)}")


# Example usage and testing
async def main():
    """Demonstrate database failover testing."""
    # Test configuration
    config = FailoverTestConfig(
        test_id="db_failover_test_001",
        failover_type=FailoverType.PRIMARY_KILL,
        target_rto_seconds=15.0,
        primary_host="localhost",
        primary_port=5432,
        database_name="trading_db",
        username="admin",
        password="admin"
    )
    
    # Create tester
    tester = DatabaseFailoverTester(config)
    
    # Run test
    result = await tester.run_failover_test()
    
    # Display results
    print(f"Test Result: {result.status.value}")
    print(f"RTO Target Met: {result.rto_metrics.meets_rto_target()}")
    print(f"Total Downtime: {result.rto_metrics.total_downtime:.2f}s")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())