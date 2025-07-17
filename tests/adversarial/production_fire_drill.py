"""
Phase 3: Production Fire Drill Automation

This module simulates real production failure scenarios and tests system
recovery capabilities. Designed to validate system resilience under fire.

Mission: Break the system in realistic ways, then validate recovery.
"""

import logging


import asyncio
import time
import json
import subprocess
import signal
import psutil
import docker
import redis.asyncio as redis
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys
import aiohttp
from enum import Enum
import threading
import random

# Add project path
sys.path.append(str(Path(__file__).parent.parent.parent))

class FailureType(Enum):
    """Types of failure scenarios to simulate."""
    REDIS_FAILURE = "redis_failure"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    CPU_STARVATION = "cpu_starvation"
    NETWORK_PARTITION = "network_partition"
    CORRUPT_CONFIG = "corrupt_config"
    MONITORING_FAILURE = "monitoring_failure"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"
    CASCADING_FAILURE = "cascading_failure"

@dataclass
class FireDrillResult:
    """Fire drill test result structure."""
    test_name: str
    failure_type: FailureType
    failure_injection_time: float
    detection_time: Optional[float]
    recovery_start_time: Optional[float]
    full_recovery_time: Optional[float]
    system_availability_during_failure: float
    data_loss_detected: bool
    recovery_successful: bool
    performance_impact: Dict[str, float]
    lessons_learned: List[str]
    breaking_point_reached: bool

class ProductionFireDrill:
    """
    Production fire drill automation system.
    
    Simulates realistic production failures and measures system recovery.
    Tests the system's ability to maintain availability and recover gracefully.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results: List[FireDrillResult] = []
        self.docker_client = None
        self.redis_client = None
        self.baseline_performance = {}
        self.monitoring_active = False
        
    async def setup(self):
        """Setup fire drill environment."""
        try:
            self.docker_client = docker.from_env()
            self.redis_client = redis.from_url("redis://localhost:6379/2")
            await self.redis_client.ping()
            
            # Establish baseline performance
            await self._establish_baseline_performance()
            
            print("🔥 Fire drill environment ready")
            
        except Exception as e:
            print(f"❌ Failed to setup fire drill environment: {e}")
            raise
    
    async def teardown(self):
        """Cleanup fire drill environment."""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.docker_client:
            self.docker_client.close()
    
    async def _establish_baseline_performance(self):
        """Establish baseline performance metrics."""
        print("📊 Establishing baseline performance...")
        
        latencies = []
        success_count = 0
        
        # Make 20 baseline requests
        async with aiohttp.ClientSession() as session:
            for i in range(20):
                start_time = time.perf_counter()
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        latency = (time.perf_counter() - start_time) * 1000
                        latencies.append(latency)
                        if response.status == 200:
                            success_count += 1
                except Exception:
                    pass
                
                await asyncio.sleep(0.1)
        
        self.baseline_performance = {
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "success_rate": success_count / 20,
            "max_latency_ms": max(latencies) if latencies else 0
        }
        
        print(f"✅ Baseline established: {self.baseline_performance}")
    
    async def test_redis_failure_scenario(self) -> FireDrillResult:
        """
        Simulate Redis failure and test system recovery.
        
        Steps:
        1. Stop Redis container
        2. Monitor system behavior
        3. Restart Redis
        4. Measure recovery time
        """
        print(f"\n🔥 FIRE DRILL: Redis Failure Scenario")
        
        failure_injection_time = time.time()
        detection_time = None
        recovery_start_time = None
        full_recovery_time = None
        availability_samples = []
        data_loss_detected = False
        
        try:
            # Step 1: Stop Redis container
            print("💥 Injecting failure: Stopping Redis container...")
            redis_container = self.docker_client.containers.get("grandmodel-redis-1")
            redis_container.stop()
            
            # Step 2: Monitor system behavior during failure
            print("📊 Monitoring system behavior during Redis failure...")
            
            failure_detected = False
            monitoring_start = time.time()
            
            # Monitor for 60 seconds or until recovery
            while time.time() - monitoring_start < 60:
                # Test system availability
                availability = await self._test_system_availability()
                availability_samples.append(availability)
                
                # Check if failure is detected
                if not failure_detected and availability < 0.5:
                    detection_time = time.time()
                    failure_detected = True
                    print(f"🚨 Failure detected at {detection_time - failure_injection_time:.2f}s")
                
                await asyncio.sleep(1)
            
            # Step 3: Start recovery
            print("🔧 Starting recovery: Restarting Redis...")
            recovery_start_time = time.time()
            redis_container.start()
            
            # Wait for Redis to be ready
            redis_ready = False
            while not redis_ready and time.time() - recovery_start_time < 30:
                try:
                    test_redis = redis.from_url("redis://localhost:6379/2")
                    await test_redis.ping()
                    await test_redis.close()
                    redis_ready = True
                except (ImportError, ModuleNotFoundError) as e:
                    await asyncio.sleep(1)
            
            # Step 4: Monitor recovery
            print("📈 Monitoring system recovery...")
            recovery_monitoring_start = time.time()
            
            while time.time() - recovery_monitoring_start < 60:
                availability = await self._test_system_availability()
                availability_samples.append(availability)
                
                # Check if full recovery achieved
                if availability > 0.9 and not full_recovery_time:
                    full_recovery_time = time.time()
                    print(f"✅ Full recovery achieved at {full_recovery_time - failure_injection_time:.2f}s")
                    break
                
                await asyncio.sleep(1)
            
            # Check for data loss
            data_loss_detected = await self._check_data_integrity()
            
            # Calculate metrics
            avg_availability = sum(availability_samples) / len(availability_samples) if availability_samples else 0
            recovery_successful = full_recovery_time is not None
            
            performance_impact = await self._measure_performance_impact()
            
            lessons_learned = [
                f"Redis failure detection took {(detection_time - failure_injection_time):.2f}s" if detection_time else "Failure detection failed",
                f"System availability during failure: {avg_availability:.1%}",
                f"Recovery time: {(full_recovery_time - recovery_start_time):.2f}s" if full_recovery_time and recovery_start_time else "Recovery failed",
                "Data integrity maintained" if not data_loss_detected else "DATA LOSS DETECTED"
            ]
            
            result = FireDrillResult(
                test_name="redis_failure_scenario",
                failure_type=FailureType.REDIS_FAILURE,
                failure_injection_time=failure_injection_time,
                detection_time=detection_time,
                recovery_start_time=recovery_start_time,
                full_recovery_time=full_recovery_time,
                system_availability_during_failure=avg_availability,
                data_loss_detected=data_loss_detected,
                recovery_successful=recovery_successful,
                performance_impact=performance_impact,
                lessons_learned=lessons_learned,
                breaking_point_reached=avg_availability < 0.3 or data_loss_detected
            )
            
        except Exception as e:
            print(f"❌ Redis failure drill failed: {e}")
            result = FireDrillResult(
                test_name="redis_failure_scenario",
                failure_type=FailureType.REDIS_FAILURE,
                failure_injection_time=failure_injection_time,
                detection_time=None,
                recovery_start_time=None,
                full_recovery_time=None,
                system_availability_during_failure=0.0,
                data_loss_detected=True,
                recovery_successful=False,
                performance_impact={},
                lessons_learned=[f"Fire drill execution failed: {str(e)}"],
                breaking_point_reached=True
            )
        
        self.results.append(result)
        self._print_fire_drill_result(result)
        return result
    
    async def test_memory_exhaustion_scenario(self) -> FireDrillResult:
        """
        Simulate memory exhaustion and test system behavior.
        
        Gradually consume system memory and observe system response.
        """
        print(f"\n🧠 FIRE DRILL: Memory Exhaustion Scenario")
        
        failure_injection_time = time.time()
        detection_time = None
        recovery_start_time = None
        full_recovery_time = None
        availability_samples = []
        memory_hogs = []
        
        try:
            print("💥 Injecting failure: Gradually exhausting system memory...")
            
            # Monitor baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"📊 Baseline memory usage: {baseline_memory:.1f}MB")
            
            # Gradually consume memory
            chunk_size = 100 * 1024 * 1024  # 100MB chunks
            max_chunks = 50  # Up to 5GB
            
            failure_detected = False
            
            for chunk_num in range(max_chunks):
                # Allocate memory chunk
                memory_chunk = bytearray(chunk_size)
                memory_hogs.append(memory_chunk)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"🔥 Allocated {(chunk_num + 1) * 100}MB, total: {current_memory:.1f}MB")
                
                # Test system availability
                availability = await self._test_system_availability()
                availability_samples.append(availability)
                
                # Check for failure detection
                if not failure_detected and availability < 0.7:
                    detection_time = time.time()
                    failure_detected = True
                    print(f"🚨 Memory pressure detected at {detection_time - failure_injection_time:.2f}s")
                
                # Check if system becomes unresponsive
                if availability < 0.1:
                    print("💥 System unresponsive due to memory exhaustion")
                    break
                
                await asyncio.sleep(2)  # Give system time to respond
            
            # Start recovery by releasing memory
            print("🔧 Starting recovery: Releasing allocated memory...")
            recovery_start_time = time.time()
            
            # Release memory chunks gradually
            while memory_hogs:
                memory_hogs.pop()
                
                if len(memory_hogs) % 10 == 0:  # Check every 1GB released
                    availability = await self._test_system_availability()
                    availability_samples.append(availability)
                    
                    current_memory = process.memory_info().rss / 1024 / 1024
                    print(f"🔄 Released memory, current: {current_memory:.1f}MB, availability: {availability:.1%}")
                    
                    if availability > 0.9 and not full_recovery_time:
                        full_recovery_time = time.time()
                        print(f"✅ Full recovery achieved")
                        break
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Final availability check
            final_availability = await self._test_system_availability()
            availability_samples.append(final_availability)
            
            if not full_recovery_time and final_availability > 0.9:
                full_recovery_time = time.time()
            
            # Calculate metrics
            avg_availability = sum(availability_samples) / len(availability_samples) if availability_samples else 0
            recovery_successful = full_recovery_time is not None
            data_loss_detected = await self._check_data_integrity()
            performance_impact = await self._measure_performance_impact()
            
            lessons_learned = [
                f"Memory pressure detection took {(detection_time - failure_injection_time):.2f}s" if detection_time else "Memory pressure not detected",
                f"System degraded at ~{baseline_memory + (chunk_num + 1) * 100:.0f}MB memory usage",
                f"Average availability during test: {avg_availability:.1%}",
                f"Recovery time: {(full_recovery_time - recovery_start_time):.2f}s" if full_recovery_time and recovery_start_time else "Recovery incomplete"
            ]
            
            result = FireDrillResult(
                test_name="memory_exhaustion_scenario",
                failure_type=FailureType.MEMORY_EXHAUSTION,
                failure_injection_time=failure_injection_time,
                detection_time=detection_time,
                recovery_start_time=recovery_start_time,
                full_recovery_time=full_recovery_time,
                system_availability_during_failure=avg_availability,
                data_loss_detected=data_loss_detected,
                recovery_successful=recovery_successful,
                performance_impact=performance_impact,
                lessons_learned=lessons_learned,
                breaking_point_reached=avg_availability < 0.5
            )
            
        except Exception as e:
            # Ensure memory cleanup on error
            memory_hogs.clear()
            import gc
            gc.collect()
            
            print(f"❌ Memory exhaustion drill failed: {e}")
            result = FireDrillResult(
                test_name="memory_exhaustion_scenario",
                failure_type=FailureType.MEMORY_EXHAUSTION,
                failure_injection_time=failure_injection_time,
                detection_time=None,
                recovery_start_time=None,
                full_recovery_time=None,
                system_availability_during_failure=0.0,
                data_loss_detected=True,
                recovery_successful=False,
                performance_impact={},
                lessons_learned=[f"Memory drill execution failed: {str(e)}"],
                breaking_point_reached=True
            )
        
        self.results.append(result)
        self._print_fire_drill_result(result)
        return result
    
    async def test_monitoring_failure_scenario(self) -> FireDrillResult:
        """
        Simulate monitoring system failure and test blind operation.
        
        Simulates scenario where monitoring/metrics collection fails
        but the core system must continue operating.
        """
        print(f"\n📊 FIRE DRILL: Monitoring Failure Scenario")
        
        failure_injection_time = time.time()
        detection_time = None
        recovery_start_time = None
        full_recovery_time = None
        availability_samples = []
        
        try:
            print("💥 Injecting failure: Simulating monitoring system failure...")
            
            # Block metrics endpoint to simulate monitoring failure
            original_metrics_available = True
            
            # Test baseline monitoring availability
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.base_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        original_metrics_available = response.status == 200
                except (ConnectionError, OSError, TimeoutError) as e:
                    original_metrics_available = False
            
            print(f"📊 Baseline monitoring available: {original_metrics_available}")
            
            # Simulate monitoring failure by overwhelming metrics endpoint
            monitoring_failure_tasks = []
            
            async def overwhelm_metrics():
                """Generate excessive load on metrics endpoint to cause failure."""
                async with aiohttp.ClientSession() as session:
                    for _ in range(1000):  # Massive request load
                        try:
                            await session.get(f"{self.base_url}/metrics", timeout=aiohttp.ClientTimeout(total=0.1))
                        except (ConnectionError, OSError, TimeoutError) as e:
                            logger.error(f'Error occurred: {e}')
                        await asyncio.sleep(0.001)  # Very rapid requests
            
            # Start overwhelming the monitoring
            for _ in range(5):  # Multiple concurrent overwhelming tasks
                task = asyncio.create_task(overwhelm_metrics())
                monitoring_failure_tasks.append(task)
            
            print("🔥 Monitoring system under extreme load...")
            
            # Monitor system behavior with failed monitoring
            monitoring_start = time.time()
            metrics_available = True
            
            for i in range(60):  # Monitor for 60 seconds
                # Test core system availability (not metrics)
                core_availability = await self._test_core_system_availability()
                availability_samples.append(core_availability)
                
                # Test if metrics are still available
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.base_url}/metrics", timeout=aiohttp.ClientTimeout(total=2)) as response:
                            metrics_currently_available = response.status == 200
                except (ConnectionError, OSError, TimeoutError) as e:
                    metrics_currently_available = False
                
                # Detect monitoring failure
                if metrics_available and not metrics_currently_available and not detection_time:
                    detection_time = time.time()
                    metrics_available = False
                    print(f"🚨 Monitoring failure detected at {detection_time - failure_injection_time:.2f}s")
                
                # Check if core system maintains operation
                if core_availability < 0.8:
                    print(f"⚠️ Core system degraded: {core_availability:.1%} availability")
                
                await asyncio.sleep(1)
            
            # Start recovery by stopping the overwhelming load
            print("🔧 Starting recovery: Stopping monitoring overload...")
            recovery_start_time = time.time()
            
            # Cancel overwhelming tasks
            for task in monitoring_failure_tasks:
                task.cancel()
            
            # Wait for tasks to finish canceling
            await asyncio.gather(*monitoring_failure_tasks, return_exceptions=True)
            
            # Monitor recovery
            for i in range(30):  # Monitor recovery for 30 seconds
                core_availability = await self._test_core_system_availability()
                availability_samples.append(core_availability)
                
                # Test metrics recovery
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self.base_url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as response:
                            metrics_recovered = response.status == 200
                except (ConnectionError, OSError, TimeoutError) as e:
                    metrics_recovered = False
                
                if metrics_recovered and core_availability > 0.9 and not full_recovery_time:
                    full_recovery_time = time.time()
                    print(f"✅ Full recovery achieved - monitoring and core system operational")
                    break
                
                await asyncio.sleep(1)
            
            # Calculate metrics
            avg_availability = sum(availability_samples) / len(availability_samples) if availability_samples else 0
            recovery_successful = full_recovery_time is not None
            data_loss_detected = await self._check_data_integrity()
            performance_impact = await self._measure_performance_impact()
            
            # Core system should maintain high availability even with monitoring failure
            core_system_resilient = avg_availability > 0.8
            
            lessons_learned = [
                f"Monitoring failure detection: {(detection_time - failure_injection_time):.2f}s" if detection_time else "Monitoring failure not detected",
                f"Core system availability during monitoring failure: {avg_availability:.1%}",
                "Core system resilient to monitoring failure" if core_system_resilient else "CRITICAL: Core system affected by monitoring failure",
                f"Recovery time: {(full_recovery_time - recovery_start_time):.2f}s" if full_recovery_time and recovery_start_time else "Recovery incomplete"
            ]
            
            result = FireDrillResult(
                test_name="monitoring_failure_scenario",
                failure_type=FailureType.MONITORING_FAILURE,
                failure_injection_time=failure_injection_time,
                detection_time=detection_time,
                recovery_start_time=recovery_start_time,
                full_recovery_time=full_recovery_time,
                system_availability_during_failure=avg_availability,
                data_loss_detected=data_loss_detected,
                recovery_successful=recovery_successful,
                performance_impact=performance_impact,
                lessons_learned=lessons_learned,
                breaking_point_reached=not core_system_resilient
            )
            
        except Exception as e:
            print(f"❌ Monitoring failure drill failed: {e}")
            result = FireDrillResult(
                test_name="monitoring_failure_scenario",
                failure_type=FailureType.MONITORING_FAILURE,
                failure_injection_time=failure_injection_time,
                detection_time=None,
                recovery_start_time=None,
                full_recovery_time=None,
                system_availability_during_failure=0.0,
                data_loss_detected=True,
                recovery_successful=False,
                performance_impact={},
                lessons_learned=[f"Monitoring drill execution failed: {str(e)}"],
                breaking_point_reached=True
            )
        
        self.results.append(result)
        self._print_fire_drill_result(result)
        return result
    
    async def _test_system_availability(self) -> float:
        """Test overall system availability."""
        endpoints = ["/health", "/status", "/decide"]
        successful_requests = 0
        total_requests = len(endpoints)
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    if endpoint == "/decide":
                        # Test with minimal payload
                        payload = {
                            "matrix_state": [[0.0] * 7 for _ in range(60)],
                            "correlation_id": f"test-{time.time()}"
                        }
                        async with session.post(
                            f"{self.base_url}{endpoint}",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status in [200, 400]:  # 400 is acceptable for malformed data
                                successful_requests += 1
                    else:
                        async with session.get(
                            f"{self.base_url}{endpoint}",
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                successful_requests += 1
                except (ConnectionError, OSError, TimeoutError) as e:
                    pass  # Request failed
        
        return successful_requests / total_requests
    
    async def _test_core_system_availability(self) -> float:
        """Test core system availability (excluding monitoring endpoints)."""
        endpoints = ["/health", "/decide"]
        successful_requests = 0
        total_requests = len(endpoints)
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    if endpoint == "/decide":
                        payload = {
                            "matrix_state": [[0.0] * 7 for _ in range(60)],
                            "correlation_id": f"core-test-{time.time()}"
                        }
                        async with session.post(
                            f"{self.base_url}{endpoint}",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status in [200, 400]:
                                successful_requests += 1
                    else:
                        async with session.get(
                            f"{self.base_url}{endpoint}",
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                successful_requests += 1
                except (ConnectionError, OSError, TimeoutError) as e:
                    logger.error(f'Error occurred: {e}')
        
        return successful_requests / total_requests
    
    async def _check_data_integrity(self) -> bool:
        """Check if data loss occurred during the failure."""
        # Simplified data integrity check
        # In a real system, this would check database consistency, etc.
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check for expected data structure
                        required_fields = ["service", "version", "components"]
                        return all(field in data for field in required_fields)
            return False
        except (json.JSONDecodeError, ValueError) as e:
            return True  # Assume data loss if we can't check
    
    async def _measure_performance_impact(self) -> Dict[str, float]:
        """Measure current performance vs baseline."""
        latencies = []
        success_count = 0
        
        # Make 10 test requests
        async with aiohttp.ClientSession() as session:
            for i in range(10):
                start_time = time.perf_counter()
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        latency = (time.perf_counter() - start_time) * 1000
                        latencies.append(latency)
                        if response.status == 200:
                            success_count += 1
                except Exception:
                    latencies.append(5000)  # 5s timeout penalty
                
                await asyncio.sleep(0.1)
        
        current_performance = {
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "success_rate": success_count / 10,
            "max_latency_ms": max(latencies) if latencies else 0
        }
        
        # Compare to baseline
        return {
            "latency_degradation_pct": (
                (current_performance["avg_latency_ms"] - self.baseline_performance["avg_latency_ms"]) /
                self.baseline_performance["avg_latency_ms"] * 100
            ) if self.baseline_performance["avg_latency_ms"] > 0 else 0,
            "availability_degradation_pct": (
                (self.baseline_performance["success_rate"] - current_performance["success_rate"]) /
                self.baseline_performance["success_rate"] * 100
            ) if self.baseline_performance["success_rate"] > 0 else 0,
            "current_avg_latency_ms": current_performance["avg_latency_ms"],
            "current_success_rate": current_performance["success_rate"]
        }
    
    def _print_fire_drill_result(self, result: FireDrillResult):
        """Print formatted fire drill result."""
        print(f"\n{'='*60}")
        print(f"🔥 FIRE DRILL RESULT: {result.test_name}")
        print(f"{'='*60}")
        print(f"Failure Type: {result.failure_type.value}")
        print(f"")
        print(f"Timeline:")
        print(f"  Failure Injection: 0.0s")
        if result.detection_time:
            print(f"  Failure Detection: {result.detection_time - result.failure_injection_time:.2f}s")
        if result.recovery_start_time:
            print(f"  Recovery Start: {result.recovery_start_time - result.failure_injection_time:.2f}s")
        if result.full_recovery_time:
            print(f"  Full Recovery: {result.full_recovery_time - result.failure_injection_time:.2f}s")
        print(f"")
        print(f"Impact Assessment:")
        print(f"  Availability During Failure: {result.system_availability_during_failure:.1%}")
        print(f"  Data Loss Detected: {'YES' if result.data_loss_detected else 'NO'}")
        print(f"  Recovery Successful: {'YES' if result.recovery_successful else 'NO'}")
        print(f"")
        if result.performance_impact:
            print(f"Performance Impact:")
            for key, value in result.performance_impact.items():
                print(f"  {key}: {value:.1f}")
        print(f"")
        if result.breaking_point_reached:
            print(f"💥 BREAKING POINT REACHED!")
        else:
            print(f"✅ System resilience validated")
        print(f"")
        print(f"Lessons Learned:")
        for lesson in result.lessons_learned:
            print(f"  - {lesson}")
        print(f"{'='*60}")
    
    def generate_fire_drill_report(self) -> str:
        """Generate comprehensive fire drill report."""
        report = []
        report.append("# PRODUCTION FIRE DRILL REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Executive summary
        total_drills = len(self.results)
        breaking_points = sum(1 for r in self.results if r.breaking_point_reached)
        successful_recoveries = sum(1 for r in self.results if r.recovery_successful)
        
        report.append("## EXECUTIVE SUMMARY")
        report.append(f"- **Total Fire Drills**: {total_drills}")
        report.append(f"- **Breaking Points**: {breaking_points}")
        report.append(f"- **Successful Recoveries**: {successful_recoveries}")
        report.append(f"- **Recovery Success Rate**: {(successful_recoveries / total_drills * 100):.1f}%")
        report.append("")
        
        if breaking_points > 0:
            report.append("🚨 **CRITICAL FINDINGS**")
            for result in self.results:
                if result.breaking_point_reached:
                    report.append(f"   - {result.failure_type.value}: System resilience failure")
        else:
            report.append("✅ **All fire drills passed - system resilience validated**")
        
        report.append("")
        report.append("## DETAILED RESULTS")
        
        for result in self.results:
            report.append(f"\n### {result.test_name.upper()}")
            report.append(f"- **Failure Type**: {result.failure_type.value}")
            
            if result.detection_time:
                detection_delay = result.detection_time - result.failure_injection_time
                report.append(f"- **Detection Time**: {detection_delay:.2f}s")
            
            if result.full_recovery_time and result.recovery_start_time:
                recovery_duration = result.full_recovery_time - result.recovery_start_time
                report.append(f"- **Recovery Duration**: {recovery_duration:.2f}s")
            
            report.append(f"- **Availability During Failure**: {result.system_availability_during_failure:.1%}")
            report.append(f"- **Data Integrity**: {'FAILED' if result.data_loss_detected else 'MAINTAINED'}")
            report.append(f"- **Recovery Status**: {'SUCCESS' if result.recovery_successful else 'FAILED'}")
            
            if result.breaking_point_reached:
                report.append("- **🔥 BREAKING POINT**: REACHED")
        
        report.append("")
        report.append("## RECOMMENDATIONS")
        
        # Generate recommendations based on results
        recommendations = set()
        for result in self.results:
            if result.breaking_point_reached:
                if result.failure_type == FailureType.REDIS_FAILURE:
                    recommendations.add("Implement Redis clustering for high availability")
                elif result.failure_type == FailureType.MEMORY_EXHAUSTION:
                    recommendations.add("Add memory monitoring and automatic resource scaling")
                elif result.failure_type == FailureType.MONITORING_FAILURE:
                    recommendations.add("Decouple core system from monitoring dependencies")
            
            if not result.recovery_successful:
                recommendations.add("Implement automated recovery procedures")
            
            if result.system_availability_during_failure < 0.5:
                recommendations.add("Add circuit breakers and graceful degradation")
        
        for rec in recommendations:
            report.append(f"- {rec}")
        
        return "\n".join(report)

# Test execution functions
async def run_production_fire_drill_suite():
    """Run the complete production fire drill suite."""
    fire_drill = ProductionFireDrill()
    await fire_drill.setup()
    
    try:
        print("🔥 STARTING PRODUCTION FIRE DRILL SUITE")
        print("=" * 60)
        
        # Run fire drill scenarios
        await fire_drill.test_redis_failure_scenario()
        await fire_drill.test_memory_exhaustion_scenario()
        await fire_drill.test_monitoring_failure_scenario()
        
        # Generate and save report
        report = fire_drill.generate_fire_drill_report()
        
        report_path = Path(__file__).parent / "fire_drill_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Fire drill report saved to: {report_path}")
        print(report)
        
        return fire_drill.results
        
    finally:
        await fire_drill.teardown()

if __name__ == "__main__":
    asyncio.run(run_production_fire_drill_suite())