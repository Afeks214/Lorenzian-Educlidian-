#!/usr/bin/env python3
"""
Disaster Recovery Testing Framework

Automated "Game Day" testing to validate disaster recovery procedures.
This script simulates various failure scenarios and measures recovery times.
"""

import os
import sys
import time
import subprocess
import logging
import json
import asyncio
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import docker
import redis
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of failures to simulate."""
    REDIS_CORRUPTION = "redis_corruption"
    CONTAINER_DELETION = "container_deletion"
    NETWORK_PARTITION = "network_partition"
    DISK_FULL = "disk_full"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    SERVICE_CRASH = "service_crash"

@dataclass
class TestResult:
    """Result of a disaster recovery test."""
    failure_type: FailureType
    test_name: str
    success: bool
    failure_time: float
    detection_time: float
    recovery_time: float
    total_downtime: float
    error_message: Optional[str] = None
    recovery_steps: Optional[List[str]] = None

class DisasterRecoveryTester:
    """Automated disaster recovery testing framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize DR tester."""
        self.config = self._load_config(config_path)
        self.docker_client = docker.from_env()
        self.redis_client = None
        self.test_results = []
        
        # Test configuration
        self.tactical_container_name = "grandmodel-tactical"
        self.redis_container_name = "grandmodel-redis-1"
        self.test_timeout = 300  # 5 minutes max per test
        
        logger.info("Disaster Recovery Tester initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load DR test configuration."""
        default_config = {
            "max_downtime_seconds": 60,
            "max_detection_time_seconds": 30,
            "max_recovery_time_seconds": 120,
            "redis_data_corruption_size": 1000000,  # 1MB
            "test_data_retention_days": 7,
            "enable_destructive_tests": True,
            "alert_endpoints": [],
            "recovery_validation_checks": [
                "tactical_service_health",
                "redis_connectivity",
                "jit_models_loaded",
                "event_processing_active"
            ]
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    async def initialize(self):
        """Initialize connections and test environment."""
        logger.info("Initializing DR test environment")
        
        # Connect to Redis
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=2)
            await asyncio.to_thread(self.redis_client.ping)
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # Verify Docker containers are running
        try:
            tactical_container = self.docker_client.containers.get(self.tactical_container_name)
            redis_container = self.docker_client.containers.get(self.redis_container_name)
            
            if tactical_container.status != 'running':
                logger.error(f"Tactical container is not running: {tactical_container.status}")
                raise RuntimeError("Tactical container not running")
            
            if redis_container.status != 'running':
                logger.error(f"Redis container is not running: {redis_container.status}")
                raise RuntimeError("Redis container not running")
            
            logger.info("All containers verified as running")
            
        except docker.errors.NotFound as e:
            logger.error(f"Required container not found: {e}")
            raise
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all disaster recovery tests."""
        logger.info("🚨 Starting Disaster Recovery Game Day Exercise")
        
        await self.initialize()
        
        # Test scenarios
        test_scenarios = [
            (FailureType.SERVICE_CRASH, "Service Process Crash"),
            (FailureType.CONTAINER_DELETION, "Container Deletion"),
            (FailureType.REDIS_CORRUPTION, "Redis Data Corruption"),
            (FailureType.NETWORK_PARTITION, "Network Partition"),
            (FailureType.MEMORY_EXHAUSTION, "Memory Exhaustion"),
        ]
        
        if not self.config["enable_destructive_tests"]:
            logger.warning("Destructive tests disabled in configuration")
            test_scenarios = [(FailureType.SERVICE_CRASH, "Service Process Crash")]
        
        for failure_type, test_name in test_scenarios:
            logger.info(f"🔥 Running test: {test_name}")
            
            try:
                result = await self._run_test(failure_type, test_name)
                self.test_results.append(result)
                
                if result.success:
                    logger.info(f"✅ Test passed: {test_name} (recovery: {result.recovery_time:.1f}s)")
                else:
                    logger.error(f"❌ Test failed: {test_name} - {result.error_message}")
                
                # Wait between tests
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"💥 Test {test_name} crashed: {e}")
                self.test_results.append(TestResult(
                    failure_type=failure_type,
                    test_name=test_name,
                    success=False,
                    failure_time=0,
                    detection_time=0,
                    recovery_time=0,
                    total_downtime=0,
                    error_message=str(e)
                ))
        
        # Generate report
        report = self._generate_report()
        await self._save_report(report)
        
        logger.info("🎯 Disaster Recovery Game Day Exercise completed")
        return self.test_results
    
    async def _run_test(self, failure_type: FailureType, test_name: str) -> TestResult:
        """Run a single disaster recovery test."""
        logger.info(f"Starting test: {test_name}")
        
        # Record baseline state
        baseline_state = await self._capture_system_state()
        
        # Inject failure
        failure_start_time = time.time()
        failure_details = await self._inject_failure(failure_type)
        
        # Wait for failure detection
        detection_start_time = time.time()
        detection_time = await self._wait_for_failure_detection(failure_type)
        
        # Execute recovery procedure
        recovery_start_time = time.time()
        recovery_steps = await self._execute_recovery(failure_type)
        
        # Validate recovery
        validation_start_time = time.time()
        recovery_success = await self._validate_recovery(baseline_state)
        recovery_end_time = time.time()
        
        # Calculate timings
        total_recovery_time = recovery_end_time - recovery_start_time
        total_downtime = recovery_end_time - failure_start_time
        
        return TestResult(
            failure_type=failure_type,
            test_name=test_name,
            success=recovery_success,
            failure_time=failure_start_time,
            detection_time=detection_time,
            recovery_time=total_recovery_time,
            total_downtime=total_downtime,
            recovery_steps=recovery_steps
        )
    
    async def _inject_failure(self, failure_type: FailureType) -> Dict[str, Any]:
        """Inject the specified failure type."""
        logger.info(f"💣 Injecting failure: {failure_type.value}")
        
        if failure_type == FailureType.SERVICE_CRASH:
            return await self._crash_tactical_service()
        elif failure_type == FailureType.CONTAINER_DELETION:
            return await self._delete_tactical_container()
        elif failure_type == FailureType.REDIS_CORRUPTION:
            return await self._corrupt_redis_data()
        elif failure_type == FailureType.NETWORK_PARTITION:
            return await self._create_network_partition()
        elif failure_type == FailureType.MEMORY_EXHAUSTION:
            return await self._exhaust_memory()
        else:
            raise ValueError(f"Unknown failure type: {failure_type}")
    
    async def _crash_tactical_service(self) -> Dict[str, Any]:
        """Crash the tactical service process."""
        try:
            container = self.docker_client.containers.get(self.tactical_container_name)
            
            # Kill the main process inside the container
            result = container.exec_run("pkill -f tactical_main.py")
            
            return {
                "method": "process_kill",
                "exit_code": result.exit_code,
                "output": result.output.decode() if result.output else ""
            }
        except Exception as e:
            logger.error(f"Failed to crash tactical service: {e}")
            raise
    
    async def _delete_tactical_container(self) -> Dict[str, Any]:
        """Delete the tactical container."""
        try:
            container = self.docker_client.containers.get(self.tactical_container_name)
            container.stop(timeout=10)
            container.remove()
            
            return {
                "method": "container_deletion",
                "container_id": container.id
            }
        except Exception as e:
            logger.error(f"Failed to delete tactical container: {e}")
            raise
    
    async def _corrupt_redis_data(self) -> Dict[str, Any]:
        """Corrupt Redis data."""
        try:
            # Create corrupt data in Redis
            corrupt_data = b"x" * self.config["redis_data_corruption_size"]
            
            # Overwrite critical keys
            corrupt_keys = [
                "synergy_events",
                "tactical_models",
                "tactical_state"
            ]
            
            for key in corrupt_keys:
                await asyncio.to_thread(self.redis_client.set, key, corrupt_data)
            
            return {
                "method": "data_corruption",
                "corrupted_keys": corrupt_keys,
                "corruption_size": len(corrupt_data)
            }
        except Exception as e:
            logger.error(f"Failed to corrupt Redis data: {e}")
            raise
    
    async def _create_network_partition(self) -> Dict[str, Any]:
        """Create network partition between tactical and Redis."""
        try:
            # Use iptables to block communication
            tactical_container = self.docker_client.containers.get(self.tactical_container_name)
            redis_container = self.docker_client.containers.get(self.redis_container_name)
            
            # Get container IPs
            tactical_ip = tactical_container.attrs['NetworkSettings']['Networks']['grandmodel_default']['IPAddress']
            redis_ip = redis_container.attrs['NetworkSettings']['Networks']['grandmodel_default']['IPAddress']
            
            # Block traffic between containers - SECURITY FIX
            # Use subprocess with explicit arguments to prevent injection
            block_cmd = [
                "iptables", 
                "-A", "DOCKER-USER", 
                "-s", tactical_ip, 
                "-d", redis_ip, 
                "-j", "DROP"
            ]
            subprocess.run(block_cmd, shell=False, check=True)
            
            return {
                "method": "network_partition",
                "tactical_ip": tactical_ip,
                "redis_ip": redis_ip,
                "block_rule": block_cmd
            }
        except Exception as e:
            logger.error(f"Failed to create network partition: {e}")
            raise
    
    async def _exhaust_memory(self) -> Dict[str, Any]:
        """Exhaust container memory."""
        try:
            container = self.docker_client.containers.get(self.tactical_container_name)
            
            # Run memory exhaustion script inside container
            memory_bomb = """
import numpy as np
import time
arrays = []
for i in range(100):
    arrays.append(np.random.random((1000, 1000, 10)))
    time.sleep(0.1)
"""
            
            # Execute memory bomb
            result = container.exec_run(
                f"python3 -c \"{memory_bomb}\"",
                detach=True
            )
            
            return {
                "method": "memory_exhaustion",
                "exec_id": result.id
            }
        except Exception as e:
            logger.error(f"Failed to exhaust memory: {e}")
            raise
    
    async def _wait_for_failure_detection(self, failure_type: FailureType) -> float:
        """Wait for failure to be detected."""
        start_time = time.time()
        max_wait = self.config["max_detection_time_seconds"]
        
        while time.time() - start_time < max_wait:
            if await self._is_failure_detected():
                detection_time = time.time() - start_time
                logger.info(f"Failure detected in {detection_time:.1f}s")
                return detection_time
            
            await asyncio.sleep(1)
        
        logger.warning(f"Failure not detected within {max_wait}s")
        return max_wait
    
    async def _is_failure_detected(self) -> bool:
        """Check if failure has been detected."""
        try:
            # Check if tactical service is responding
            result = subprocess.run(
                ["curl", "-f", "http://localhost:8001/health"],
                capture_output=True,
                timeout=5
            )
            return result.returncode != 0
        except Exception:
            return True
    
    async def _execute_recovery(self, failure_type: FailureType) -> List[str]:
        """Execute recovery procedure."""
        logger.info("🔧 Executing recovery procedure")
        
        recovery_steps = []
        
        if failure_type == FailureType.SERVICE_CRASH:
            recovery_steps = await self._recover_from_service_crash()
        elif failure_type == FailureType.CONTAINER_DELETION:
            recovery_steps = await self._recover_from_container_deletion()
        elif failure_type == FailureType.REDIS_CORRUPTION:
            recovery_steps = await self._recover_from_redis_corruption()
        elif failure_type == FailureType.NETWORK_PARTITION:
            recovery_steps = await self._recover_from_network_partition()
        elif failure_type == FailureType.MEMORY_EXHAUSTION:
            recovery_steps = await self._recover_from_memory_exhaustion()
        
        return recovery_steps
    
    async def _recover_from_service_crash(self) -> List[str]:
        """Recover from service crash."""
        steps = []
        
        # Docker restart should handle this automatically
        steps.append("Wait for Docker restart policy to trigger")
        await asyncio.sleep(30)
        
        # Verify service is back
        steps.append("Verify service recovery")
        max_wait = 60
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                result = subprocess.run(
                    ["curl", "-f", "http://localhost:8001/health"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    steps.append("Service recovered successfully")
                    return steps
            except Exception:
                pass
            
            await asyncio.sleep(5)
        
        steps.append("Manual service restart required")
        subprocess.run(["docker-compose", "restart", "tactical-marl"], check=True)
        
        return steps
    
    async def _recover_from_container_deletion(self) -> List[str]:
        """Recover from container deletion."""
        steps = []
        
        steps.append("Restart tactical service")
        result = subprocess.run(
            ["docker-compose", "up", "-d", "tactical-marl"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            steps.append(f"Failed to restart: {result.stderr}")
            raise RuntimeError(f"Container restart failed: {result.stderr}")
        
        steps.append("Wait for service initialization")
        await asyncio.sleep(60)  # Allow time for JIT compilation
        
        return steps
    
    async def _recover_from_redis_corruption(self) -> List[str]:
        """Recover from Redis corruption."""
        steps = []
        
        steps.append("Stop tactical service")
        subprocess.run(["docker-compose", "stop", "tactical-marl"], check=True)
        
        steps.append("Flush corrupted Redis data")
        await asyncio.to_thread(self.redis_client.flushdb)
        
        steps.append("Restart tactical service")
        subprocess.run(["docker-compose", "start", "tactical-marl"], check=True)
        
        steps.append("Wait for service recovery")
        await asyncio.sleep(60)
        
        return steps
    
    async def _recover_from_network_partition(self) -> List[str]:
        """Recover from network partition."""
        steps = []
        
        steps.append("Remove network blocking rules")
        # Remove all DOCKER-USER rules (simple approach)
        subprocess.run(["iptables", "-F", "DOCKER-USER"], check=True)
        
        steps.append("Restart tactical service to reset connections")
        subprocess.run(["docker-compose", "restart", "tactical-marl"], check=True)
        
        steps.append("Wait for service recovery")
        await asyncio.sleep(30)
        
        return steps
    
    async def _recover_from_memory_exhaustion(self) -> List[str]:
        """Recover from memory exhaustion."""
        steps = []
        
        steps.append("Kill memory-consuming processes")
        try:
            container = self.docker_client.containers.get(self.tactical_container_name)
            container.exec_run("pkill -f python3")
        except Exception:
            pass
        
        steps.append("Restart tactical service")
        subprocess.run(["docker-compose", "restart", "tactical-marl"], check=True)
        
        steps.append("Wait for service recovery")
        await asyncio.sleep(60)
        
        return steps
    
    async def _validate_recovery(self, baseline_state: Dict[str, Any]) -> bool:
        """Validate that recovery was successful."""
        logger.info("🔍 Validating recovery")
        
        checks = self.config["recovery_validation_checks"]
        
        for check in checks:
            if not await self._run_validation_check(check):
                logger.error(f"Validation check failed: {check}")
                return False
        
        logger.info("✅ All recovery validation checks passed")
        return True
    
    async def _run_validation_check(self, check_name: str) -> bool:
        """Run a specific validation check."""
        try:
            if check_name == "tactical_service_health":
                result = subprocess.run(
                    ["curl", "-f", "http://localhost:8001/health"],
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
            
            elif check_name == "redis_connectivity":
                await asyncio.to_thread(self.redis_client.ping)
                return True
            
            elif check_name == "jit_models_loaded":
                # Check if JIT models are accessible
                response = subprocess.run(
                    ["curl", "-f", "http://localhost:8001/models/status"],
                    capture_output=True,
                    timeout=10
                )
                return response.returncode == 0
            
            elif check_name == "event_processing_active":
                # Check if event processing is active
                response = subprocess.run(
                    ["curl", "-f", "http://localhost:8001/status"],
                    capture_output=True,
                    timeout=10
                )
                return response.returncode == 0
            
            else:
                logger.warning(f"Unknown validation check: {check_name}")
                return False
                
        except Exception as e:
            logger.error(f"Validation check {check_name} failed: {e}")
            return False
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture baseline system state."""
        try:
            return {
                "timestamp": time.time(),
                "containers_running": len(self.docker_client.containers.list()),
                "redis_connected": await asyncio.to_thread(self.redis_client.ping),
                "tactical_service_healthy": await self._run_validation_check("tactical_service_health")
            }
        except Exception as e:
            logger.error(f"Failed to capture system state: {e}")
            return {}
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate disaster recovery test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - passed_tests
        
        avg_recovery_time = 0
        avg_detection_time = 0
        avg_downtime = 0
        
        if self.test_results:
            successful_tests = [r for r in self.test_results if r.success]
            if successful_tests:
                avg_recovery_time = sum(r.recovery_time for r in successful_tests) / len(successful_tests)
                avg_detection_time = sum(r.detection_time for r in successful_tests) / len(successful_tests)
                avg_downtime = sum(r.total_downtime for r in successful_tests) / len(successful_tests)
        
        # Check if targets are met
        targets_met = {
            "max_downtime": avg_downtime <= self.config["max_downtime_seconds"],
            "max_detection_time": avg_detection_time <= self.config["max_detection_time_seconds"],
            "max_recovery_time": avg_recovery_time <= self.config["max_recovery_time_seconds"]
        }
        
        return {
            "test_timestamp": datetime.utcnow().isoformat() + "Z",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "average_recovery_time": avg_recovery_time,
            "average_detection_time": avg_detection_time,
            "average_downtime": avg_downtime,
            "targets_met": targets_met,
            "all_targets_met": all(targets_met.values()),
            "detailed_results": [
                {
                    "failure_type": r.failure_type.value,
                    "test_name": r.test_name,
                    "success": r.success,
                    "recovery_time": r.recovery_time,
                    "detection_time": r.detection_time,
                    "total_downtime": r.total_downtime,
                    "error_message": r.error_message,
                    "recovery_steps": r.recovery_steps
                }
                for r in self.test_results
            ],
            "configuration": self.config
        }
    
    async def _save_report(self, report: Dict[str, Any]):
        """Save disaster recovery test report."""
        report_dir = Path("reports/disaster_recovery")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"dr_test_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Disaster recovery test report saved to: {report_path}")
        
        # Also save as latest report
        latest_path = report_dir / "latest_dr_report.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2)

async def main():
    """Main function for disaster recovery testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Disaster Recovery Testing Framework")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("Running in dry-run mode - no actual failures will be injected")
        return
    
    tester = DisasterRecoveryTester(args.config)
    
    try:
        results = await tester.run_all_tests()
        
        # Print summary
        passed = sum(1 for r in results if r.success)
        total = len(results)
        
        print(f"\n{'='*60}")
        print("DISASTER RECOVERY TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {(passed/total*100):.1f}%")
        
        if passed == total:
            print("✅ All disaster recovery tests passed!")
            sys.exit(0)
        else:
            print("❌ Some disaster recovery tests failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Disaster recovery testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())