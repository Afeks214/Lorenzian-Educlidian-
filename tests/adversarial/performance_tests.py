"""
Phase 3: End-to-End Adversarial Performance Testing Suite

This module contains comprehensive performance and stress tests designed to find
the exact breaking points of the tactical MARL system under extreme conditions.

Mission: Test until the system breaks, then test recovery.
"""

import asyncio
import time
import json
import statistics
import numpy as np
import pytest
import concurrent.futures
import psutil
import aiohttp
import redis.asyncio as redis
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import signal
import threading
import resource
import gc
import sys

# Add project path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.tactical_main import app
from src.tactical.controller import TacticalMARLController
from src.monitoring.tactical_metrics import tactical_metrics

@dataclass
class PerformanceResult:
    """Performance test result structure."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_rps: float
    error_rate: float
    memory_peak_mb: float
    cpu_peak_percent: float
    breaking_point_reached: bool
    breaking_point_description: str = ""

class PerformanceStressTester:
    """
    Comprehensive performance stress tester for the tactical MARL system.
    
    Designed to find exact breaking points under various stress conditions.
    """
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results: List[PerformanceResult] = []
        self.redis_client = None
        self.test_running = False
        
    async def setup(self):
        """Setup test environment."""
        self.redis_client = redis.from_url("redis://localhost:6379/2")
        await self.redis_client.ping()
        
    async def teardown(self):
        """Cleanup test environment."""
        if self.redis_client:
            await self.redis_client.close()
    
    def create_test_matrix(self) -> List[List[float]]:
        """Create a valid 60x7 test matrix."""
        return [
            [
                np.random.choice([0, 1], p=[0.8, 0.2]),  # fvg_bullish_active
                np.random.choice([0, 1], p=[0.8, 0.2]),  # fvg_bearish_active
                np.random.uniform(0.99, 1.01),          # fvg_nearest_level
                np.random.exponential(5),               # fvg_age
                np.random.choice([0, 1], p=[0.9, 0.1]), # fvg_mitigation_signal
                np.random.uniform(-5, 5),               # price_momentum_5
                np.random.uniform(0.5, 2.0),            # volume_ratio
            ] for _ in range(60)
        ]
    
    def create_test_synergy_event(self, correlation_id: str) -> Dict[str, Any]:
        """Create test synergy event data."""
        return {
            "synergy_type": "TYPE_1",
            "direction": 1,
            "confidence": 0.85,
            "signal_sequence": [],
            "market_context": {},
            "correlation_id": correlation_id,
            "timestamp": time.time()
        }

    async def test_latency_under_load(self, target_rps: int, duration_seconds: int = 60) -> PerformanceResult:
        """
        Test /decide endpoint latency under sustained load.
        
        Target: Find exact point where sub-100ms requirement breaks.
        """
        print(f"\n🔥 LATENCY STRESS TEST: {target_rps} RPS for {duration_seconds}s")
        
        # Performance tracking
        latencies = []
        successes = 0
        failures = 0
        memory_usage = []
        cpu_usage = []
        breaking_point_reached = False
        breaking_point_description = ""
        
        # Memory monitor task
        async def monitor_resources():
            while self.test_running:
                process = psutil.Process()
                memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                cpu_usage.append(process.cpu_percent())
                await asyncio.sleep(0.1)
        
        # Request generator
        async def make_request(session: aiohttp.ClientSession, correlation_id: str):
            matrix_state = self.create_test_matrix()
            synergy_context = self.create_test_synergy_event(correlation_id)
            
            payload = {
                "matrix_state": matrix_state,
                "synergy_context": synergy_context,
                "correlation_id": correlation_id
            }
            
            start_time = time.perf_counter()
            try:
                async with session.post(
                    f"{self.base_url}/decide",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    latency = (time.perf_counter() - start_time) * 1000
                    latencies.append(latency)
                    
                    if response.status == 200:
                        return True, latency
                    else:
                        return False, latency
                        
            except Exception as e:
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)
                return False, latency
        
        # Main load generation
        self.test_running = True
        monitor_task = asyncio.create_task(monitor_resources())
        
        start_time = time.time()
        request_interval = 1.0 / target_rps
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            request_count = 0
            
            while time.time() - start_time < duration_seconds:
                correlation_id = f"stress-test-{request_count}-{int(time.time())}"
                
                task = asyncio.create_task(make_request(session, correlation_id))
                tasks.append(task)
                request_count += 1
                
                # Control request rate
                await asyncio.sleep(request_interval)
                
                # Check for breaking point conditions every 100 requests
                if request_count % 100 == 0:
                    recent_latencies = latencies[-100:] if len(latencies) >= 100 else latencies
                    if recent_latencies:
                        avg_recent = statistics.mean(recent_latencies)
                        p95_recent = np.percentile(recent_latencies, 95)
                        
                        # Breaking point: 80% of requests exceed 100ms
                        over_100ms = sum(1 for l in recent_latencies if l > 100)
                        if over_100ms / len(recent_latencies) > 0.8:
                            breaking_point_reached = True
                            breaking_point_description = f"80% of requests exceeded 100ms (avg: {avg_recent:.1f}ms, p95: {p95_recent:.1f}ms)"
                            print(f"💥 BREAKING POINT REACHED: {breaking_point_description}")
                            break
                        
                        # Memory breaking point
                        if memory_usage and memory_usage[-1] > 4000:  # 4GB
                            breaking_point_reached = True
                            breaking_point_description = f"Memory usage exceeded 4GB: {memory_usage[-1]:.1f}MB"
                            print(f"💥 MEMORY BREAKING POINT: {breaking_point_description}")
                            break
            
            # Wait for all tasks to complete
            print(f"⏳ Waiting for {len(tasks)} requests to complete...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failures += 1
                elif result[0]:  # Success
                    successes += 1
                else:
                    failures += 1
        
        self.test_running = False
        monitor_task.cancel()
        
        # Calculate metrics
        total_time = time.time() - start_time
        total_requests = successes + failures
        
        result = PerformanceResult(
            test_name=f"latency_stress_{target_rps}rps",
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            throughput_rps=total_requests / total_time,
            error_rate=failures / total_requests if total_requests > 0 else 0,
            memory_peak_mb=max(memory_usage) if memory_usage else 0,
            cpu_peak_percent=max(cpu_usage) if cpu_usage else 0,
            breaking_point_reached=breaking_point_reached,
            breaking_point_description=breaking_point_description
        )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    async def test_memory_exhaustion(self) -> PerformanceResult:
        """
        Test system behavior under extreme memory pressure.
        
        Progressively increase memory load until system breaks.
        """
        print(f"\n🧠 MEMORY EXHAUSTION TEST")
        
        latencies = []
        successes = 0
        failures = 0
        memory_hogs = []
        breaking_point_reached = False
        breaking_point_description = ""
        
        async with aiohttp.ClientSession() as session:
            request_count = 0
            
            # Progressively allocate memory while making requests
            for memory_mb in range(100, 8000, 100):  # Up to 8GB
                print(f"📈 Allocating {memory_mb}MB of memory...")
                
                # Create memory pressure
                chunk_size = 1024 * 1024  # 1MB chunks
                num_chunks = memory_mb
                memory_hog = [b'x' * chunk_size for _ in range(num_chunks)]
                memory_hogs.append(memory_hog)
                
                # Make 10 requests at this memory level
                for i in range(10):
                    correlation_id = f"memory-test-{memory_mb}-{i}"
                    matrix_state = self.create_test_matrix()
                    synergy_context = self.create_test_synergy_event(correlation_id)
                    
                    payload = {
                        "matrix_state": matrix_state,
                        "synergy_context": synergy_context,
                        "correlation_id": correlation_id
                    }
                    
                    start_time = time.perf_counter()
                    try:
                        async with session.post(
                            f"{self.base_url}/decide",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            latency = (time.perf_counter() - start_time) * 1000
                            latencies.append(latency)
                            
                            if response.status == 200:
                                successes += 1
                            else:
                                failures += 1
                                
                    except Exception as e:
                        latency = (time.perf_counter() - start_time) * 1000
                        latencies.append(latency)
                        failures += 1
                        
                        # Check if this is a memory-related failure
                        if "memory" in str(e).lower() or "out of" in str(e).lower():
                            breaking_point_reached = True
                            breaking_point_description = f"Memory allocation failed at {memory_mb}MB: {str(e)}"
                            print(f"💥 MEMORY BREAKING POINT: {breaking_point_description}")
                            break
                    
                    request_count += 1
                
                if breaking_point_reached:
                    break
                
                # Check response times for degradation
                recent_latencies = latencies[-10:] if len(latencies) >= 10 else latencies
                if recent_latencies and statistics.mean(recent_latencies) > 1000:  # 1 second
                    breaking_point_reached = True
                    breaking_point_description = f"Response time degraded to {statistics.mean(recent_latencies):.1f}ms at {memory_mb}MB"
                    print(f"💥 PERFORMANCE BREAKING POINT: {breaking_point_description}")
                    break
        
        # Cleanup memory
        memory_hogs.clear()
        gc.collect()
        
        total_requests = successes + failures
        
        result = PerformanceResult(
            test_name="memory_exhaustion",
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            throughput_rps=0,  # Not applicable for this test
            error_rate=failures / total_requests if total_requests > 0 else 0,
            memory_peak_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_peak_percent=0,  # Not tracked in this test
            breaking_point_reached=breaking_point_reached,
            breaking_point_description=breaking_point_description
        )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    async def test_cpu_saturation(self, target_rps: int = 100) -> PerformanceResult:
        """
        Test system behavior under CPU saturation.
        
        Create CPU-intensive background tasks while making requests.
        """
        print(f"\n⚡ CPU SATURATION TEST")
        
        latencies = []
        successes = 0
        failures = 0
        breaking_point_reached = False
        breaking_point_description = ""
        
        # CPU saturation function
        def cpu_burner():
            """CPU-intensive task to create saturation."""
            end_time = time.time() + 60  # Run for 60 seconds
            while time.time() < end_time:
                # Prime number calculation (CPU intensive)
                for i in range(10000):
                    for j in range(2, int(i**0.5) + 1):
                        if i % j == 0:
                            break
        
        # Start CPU burner threads (one per CPU core)
        cpu_count = psutil.cpu_count()
        print(f"🔥 Starting {cpu_count} CPU burner threads...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
            # Start CPU burning tasks
            cpu_futures = [executor.submit(cpu_burner) for _ in range(cpu_count)]
            
            # Give CPU burners time to ramp up
            await asyncio.sleep(2)
            
            # Make requests under CPU load
            async with aiohttp.ClientSession() as session:
                for i in range(target_rps):
                    correlation_id = f"cpu-test-{i}"
                    matrix_state = self.create_test_matrix()
                    synergy_context = self.create_test_synergy_event(correlation_id)
                    
                    payload = {
                        "matrix_state": matrix_state,
                        "synergy_context": synergy_context,
                        "correlation_id": correlation_id
                    }
                    
                    start_time = time.perf_counter()
                    try:
                        async with session.post(
                            f"{self.base_url}/decide",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            latency = (time.perf_counter() - start_time) * 1000
                            latencies.append(latency)
                            
                            if response.status == 200:
                                successes += 1
                            else:
                                failures += 1
                                
                    except Exception as e:
                        latency = (time.perf_counter() - start_time) * 1000
                        latencies.append(latency)
                        failures += 1
                        
                        if "timeout" in str(e).lower():
                            breaking_point_reached = True
                            breaking_point_description = f"Request timeout under CPU load at request {i}"
                            print(f"💥 CPU SATURATION BREAKING POINT: {breaking_point_description}")
                            break
                    
                    # Check for extreme latency
                    if latencies and latencies[-1] > 5000:  # 5 seconds
                        breaking_point_reached = True
                        breaking_point_description = f"Request latency exceeded 5 seconds: {latencies[-1]:.1f}ms"
                        print(f"💥 LATENCY BREAKING POINT: {breaking_point_description}")
                        break
                    
                    await asyncio.sleep(1.0 / target_rps)  # Control rate
            
            # Wait for CPU burners to finish
            concurrent.futures.wait(cpu_futures)
        
        total_requests = successes + failures
        
        result = PerformanceResult(
            test_name="cpu_saturation",
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            throughput_rps=total_requests / 60,  # 60 second test
            error_rate=failures / total_requests if total_requests > 0 else 0,
            memory_peak_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_peak_percent=100,  # Saturated
            breaking_point_reached=breaking_point_reached,
            breaking_point_description=breaking_point_description
        )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    async def test_io_bottleneck_simulation(self) -> PerformanceResult:
        """
        Simulate I/O bottlenecks with artificial delays in Redis operations.
        
        This tests how the system handles slow Redis/network conditions.
        """
        print(f"\n💾 I/O BOTTLENECK SIMULATION TEST")
        
        latencies = []
        successes = 0
        failures = 0
        breaking_point_reached = False
        breaking_point_description = ""
        
        # Create a Redis proxy that adds artificial delays
        original_execute_command = self.redis_client.execute_command
        
        async def slow_redis_command(*args, **kwargs):
            # Simulate slow Redis operations
            await asyncio.sleep(0.01)  # 10ms delay per Redis operation
            return await original_execute_command(*args, **kwargs)
        
        # Monkey patch Redis client
        self.redis_client.execute_command = slow_redis_command
        
        try:
            async with aiohttp.ClientSession() as session:
                for i in range(50):  # Limited requests due to slow I/O
                    correlation_id = f"io-test-{i}"
                    matrix_state = self.create_test_matrix()
                    synergy_context = self.create_test_synergy_event(correlation_id)
                    
                    payload = {
                        "matrix_state": matrix_state,
                        "synergy_context": synergy_context,
                        "correlation_id": correlation_id
                    }
                    
                    start_time = time.perf_counter()
                    try:
                        async with session.post(
                            f"{self.base_url}/decide",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            latency = (time.perf_counter() - start_time) * 1000
                            latencies.append(latency)
                            
                            if response.status == 200:
                                successes += 1
                            else:
                                failures += 1
                                
                    except Exception as e:
                        latency = (time.perf_counter() - start_time) * 1000
                        latencies.append(latency)
                        failures += 1
                        
                        if "timeout" in str(e).lower():
                            breaking_point_reached = True
                            breaking_point_description = f"I/O timeout at request {i}: {str(e)}"
                            print(f"💥 I/O BREAKING POINT: {breaking_point_description}")
                            break
                    
                    # Check for unacceptable latency under I/O stress
                    if latencies and latencies[-1] > 2000:  # 2 seconds
                        breaking_point_reached = True
                        breaking_point_description = f"I/O latency exceeded 2 seconds: {latencies[-1]:.1f}ms"
                        print(f"💥 I/O LATENCY BREAKING POINT: {breaking_point_description}")
                        break
        
        finally:
            # Restore original Redis client
            self.redis_client.execute_command = original_execute_command
        
        total_requests = successes + failures
        
        result = PerformanceResult(
            test_name="io_bottleneck",
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            throughput_rps=0,  # Not meaningful for this test
            error_rate=failures / total_requests if total_requests > 0 else 0,
            memory_peak_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_peak_percent=0,
            breaking_point_reached=breaking_point_reached,
            breaking_point_description=breaking_point_description
        )
        
        self.results.append(result)
        self._print_result(result)
        return result
    
    def _print_result(self, result: PerformanceResult):
        """Print formatted test result."""
        print(f"\n{'='*60}")
        print(f"📊 TEST RESULT: {result.test_name}")
        print(f"{'='*60}")
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests}")
        print(f"Failed: {result.failed_requests}")
        print(f"Error Rate: {result.error_rate*100:.1f}%")
        print(f"Throughput: {result.throughput_rps:.1f} RPS")
        print(f"")
        print(f"Latency Metrics:")
        print(f"  Average: {result.avg_latency_ms:.1f}ms")
        print(f"  P95: {result.p95_latency_ms:.1f}ms")
        print(f"  P99: {result.p99_latency_ms:.1f}ms")
        print(f"  Max: {result.max_latency_ms:.1f}ms")
        print(f"")
        print(f"Resource Usage:")
        print(f"  Peak Memory: {result.memory_peak_mb:.1f}MB")
        print(f"  Peak CPU: {result.cpu_peak_percent:.1f}%")
        print(f"")
        if result.breaking_point_reached:
            print(f"💥 BREAKING POINT REACHED!")
            print(f"   {result.breaking_point_description}")
        else:
            print(f"✅ No breaking point reached")
        print(f"{'='*60}")
    
    def generate_report(self) -> str:
        """Generate comprehensive performance test report."""
        report = []
        report.append("# PHASE 3: ADVERSARIAL PERFORMANCE TEST REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("## EXECUTIVE SUMMARY")
        
        breaking_points = [r for r in self.results if r.breaking_point_reached]
        if breaking_points:
            report.append(f"🚨 **{len(breaking_points)} BREAKING POINTS IDENTIFIED**")
            for bp in breaking_points:
                report.append(f"   - {bp.test_name}: {bp.breaking_point_description}")
        else:
            report.append("✅ No breaking points reached in tested scenarios")
        
        report.append("")
        report.append("## DETAILED RESULTS")
        
        for result in self.results:
            report.append(f"\n### {result.test_name.upper()}")
            report.append(f"- **Total Requests**: {result.total_requests}")
            report.append(f"- **Success Rate**: {(1-result.error_rate)*100:.1f}%")
            report.append(f"- **Average Latency**: {result.avg_latency_ms:.1f}ms")
            report.append(f"- **P99 Latency**: {result.p99_latency_ms:.1f}ms")
            report.append(f"- **Throughput**: {result.throughput_rps:.1f} RPS")
            report.append(f"- **Peak Memory**: {result.memory_peak_mb:.1f}MB")
            
            if result.breaking_point_reached:
                report.append(f"- **🔥 Breaking Point**: {result.breaking_point_description}")
            
            # Compliance check
            if result.p99_latency_ms <= 100:
                report.append("- **✅ Sub-100ms Compliance**: PASSED")
            else:
                report.append("- **❌ Sub-100ms Compliance**: FAILED")
        
        report.append("")
        report.append("## RECOMMENDATIONS")
        
        for result in self.results:
            if result.breaking_point_reached:
                if "memory" in result.breaking_point_description.lower():
                    report.append("- **Memory Optimization**: Implement memory pooling and garbage collection tuning")
                elif "timeout" in result.breaking_point_description.lower():
                    report.append("- **Timeout Handling**: Implement circuit breakers and timeout escalation")
                elif "latency" in result.breaking_point_description.lower():
                    report.append("- **Performance Optimization**: Profile and optimize hot paths")
        
        return "\n".join(report)

# Test execution functions
async def run_full_performance_suite():
    """Run the complete performance test suite."""
    tester = PerformanceStressTester()
    await tester.setup()
    
    try:
        print("🚀 STARTING PHASE 3: ADVERSARIAL PERFORMANCE TESTING")
        print("=" * 60)
        
        # Progressive load testing to find latency breaking point
        for rps in [10, 25, 50, 100, 200, 500]:
            print(f"\n🎯 Testing {rps} RPS...")
            result = await tester.test_latency_under_load(rps, duration_seconds=30)
            
            # Stop if we hit a breaking point
            if result.breaking_point_reached:
                print(f"💥 Breaking point reached at {rps} RPS - stopping load escalation")
                break
        
        # Memory exhaustion test
        await tester.test_memory_exhaustion()
        
        # CPU saturation test
        await tester.test_cpu_saturation()
        
        # I/O bottleneck test
        await tester.test_io_bottleneck_simulation()
        
        # Generate and save report
        report = tester.generate_report()
        
        report_path = Path(__file__).parent / "performance_test_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Performance test report saved to: {report_path}")
        print(report)
        
        return tester.results
        
    finally:
        await tester.teardown()

if __name__ == "__main__":
    asyncio.run(run_full_performance_suite())