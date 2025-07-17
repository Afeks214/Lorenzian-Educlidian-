#!/usr/bin/env python3
"""
Agent 6: Production Validation Suite
Comprehensive testing for 99.9% uptime and <10ms latency requirements
"""

import asyncio
import time
import json
import logging
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pytest
import pytest_asyncio
import aiohttp
import redis
import psutil
from prometheus_client.parser import text_string_to_metric_families
import docker
import subprocess
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result structure."""
    test_name: str
    status: str  # PASS, FAIL, SKIP
    duration_ms: float
    metrics: Dict[str, Any]
    errors: List[str] = None
    warnings: List[str] = None

@dataclass
class PerformanceTestResult:
    """Performance test result."""
    test_name: str
    target_latency_ms: float
    actual_latency_ms: float
    target_throughput: float
    actual_throughput: float
    success_rate: float
    passed: bool

class LatencyTestSuite:
    """Ultra-low latency testing suite."""
    
    def __init__(self, target_latency_ms: float = 10.0):
        self.target_latency_ms = target_latency_ms
        self.test_results = []
        
    async def test_api_latency(self, base_url: str, endpoints: List[str]) -> List[PerformanceTestResult]:
        """Test API endpoint latencies."""
        results = []
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                url = f"{base_url}{endpoint}"
                latencies = []
                
                # Warm up
                for _ in range(10):
                    try:
                        start_time = time.perf_counter()
                        async with session.get(url) as response:
                            await response.read()
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        latencies.append(latency_ms)
                    except Exception as e:
                        logger.error(f"Warmup failed for {endpoint}: {e}")
                        
                # Actual test
                test_latencies = []
                errors = 0
                
                for _ in range(100):  # 100 requests for statistical significance
                    try:
                        start_time = time.perf_counter()
                        async with session.get(url) as response:
                            if response.status == 200:
                                await response.read()
                                latency_ms = (time.perf_counter() - start_time) * 1000
                                test_latencies.append(latency_ms)
                            else:
                                errors += 1
                    except Exception as e:
                        errors += 1
                        logger.error(f"Request failed for {endpoint}: {e}")
                        
                if test_latencies:
                    avg_latency = np.mean(test_latencies)
                    p95_latency = np.percentile(test_latencies, 95)
                    p99_latency = np.percentile(test_latencies, 99)
                    success_rate = len(test_latencies) / (len(test_latencies) + errors)
                    
                    # Use P95 for validation (allows some outliers)
                    passed = p95_latency <= self.target_latency_ms
                    
                    results.append(PerformanceTestResult(
                        test_name=f"api_latency_{endpoint.replace('/', '_')}",
                        target_latency_ms=self.target_latency_ms,
                        actual_latency_ms=p95_latency,
                        target_throughput=1000.0 / self.target_latency_ms,  # ops/sec
                        actual_throughput=1000.0 / avg_latency if avg_latency > 0 else 0,
                        success_rate=success_rate,
                        passed=passed
                    ))
                    
                    logger.info(f"Endpoint {endpoint}: Avg={avg_latency:.2f}ms, P95={p95_latency:.2f}ms, P99={p99_latency:.2f}ms")
                    
        return results
        
    async def test_risk_calculation_latency(self, risk_service_url: str) -> PerformanceTestResult:
        """Test risk calculation latency."""
        latencies = []
        errors = 0
        
        # Generate test data
        test_returns = np.random.normal(0.001, 0.02, 1000).tolist()
        payload = {"returns": test_returns, "confidence": 0.95}
        
        async with aiohttp.ClientSession() as session:
            for _ in range(50):  # 50 risk calculations
                try:
                    start_time = time.perf_counter()
                    async with session.post(f"{risk_service_url}/api/risk/calculate", json=payload) as response:
                        if response.status == 200:
                            await response.json()
                            latency_ms = (time.perf_counter() - start_time) * 1000
                            latencies.append(latency_ms)
                        else:
                            errors += 1
                except Exception as e:
                    errors += 1
                    logger.error(f"Risk calculation failed: {e}")
                    
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            success_rate = len(latencies) / (len(latencies) + errors)
            passed = p95_latency <= self.target_latency_ms
            
            return PerformanceTestResult(
                test_name="risk_calculation_latency",
                target_latency_ms=self.target_latency_ms,
                actual_latency_ms=p95_latency,
                target_throughput=100.0,  # target 100 calculations/sec
                actual_throughput=1000.0 / avg_latency if avg_latency > 0 else 0,
                success_rate=success_rate,
                passed=passed
            )
        else:
            return PerformanceTestResult(
                test_name="risk_calculation_latency",
                target_latency_ms=self.target_latency_ms,
                actual_latency_ms=float('inf'),
                target_throughput=100.0,
                actual_throughput=0.0,
                success_rate=0.0,
                passed=False
            )

class LoadTestSuite:
    """Load testing for throughput validation."""
    
    def __init__(self, target_throughput: float = 1000.0):
        self.target_throughput = target_throughput
        
    async def test_concurrent_load(self, base_url: str, duration_seconds: int = 60) -> Dict[str, Any]:
        """Test concurrent load handling."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        successful_requests = 0
        failed_requests = 0
        total_latency = 0
        latencies = []
        
        async def make_request(session: aiohttp.ClientSession):
            nonlocal successful_requests, failed_requests, total_latency
            
            try:
                request_start = time.perf_counter()
                async with session.get(f"{base_url}/health") as response:
                    if response.status == 200:
                        successful_requests += 1
                        latency = (time.perf_counter() - request_start) * 1000
                        total_latency += latency
                        latencies.append(latency)
                    else:
                        failed_requests += 1
            except Exception:
                failed_requests += 1
                
        # Create multiple concurrent sessions
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            while time.time() < end_time:
                # Launch concurrent requests
                for _ in range(10):  # 10 concurrent requests per batch
                    task = asyncio.create_task(make_request(session))
                    tasks.append(task)
                    
                # Wait briefly before next batch
                await asyncio.sleep(0.01)
                
                # Clean up completed tasks periodically
                if len(tasks) > 100:
                    completed_tasks = [t for t in tasks if t.done()]
                    for task in completed_tasks:
                        tasks.remove(task)
                        
            # Wait for remaining tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        total_requests = successful_requests + failed_requests
        actual_duration = time.time() - start_time
        actual_throughput = successful_requests / actual_duration if actual_duration > 0 else 0
        
        return {
            "test_name": "concurrent_load_test",
            "duration_seconds": actual_duration,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "target_throughput": self.target_throughput,
            "actual_throughput": actual_throughput,
            "avg_latency_ms": total_latency / successful_requests if successful_requests > 0 else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
            "passed": actual_throughput >= self.target_throughput and (successful_requests / total_requests) >= 0.99
        }

class UptimeTestSuite:
    """Uptime and availability testing."""
    
    def __init__(self, target_uptime: float = 99.9):
        self.target_uptime = target_uptime
        
    async def test_service_availability(self, services: Dict[str, str], duration_minutes: int = 10) -> Dict[str, Any]:
        """Test service availability over time."""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        service_stats = {service: {"checks": 0, "successes": 0, "failures": 0} for service in services}
        
        async def check_service(service_name: str, url: str):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        service_stats[service_name]["checks"] += 1
                        if response.status == 200:
                            service_stats[service_name]["successes"] += 1
                        else:
                            service_stats[service_name]["failures"] += 1
            except Exception:
                service_stats[service_name]["checks"] += 1
                service_stats[service_name]["failures"] += 1
                
        # Check services every 30 seconds
        while time.time() < end_time:
            tasks = []
            for service_name, url in services.items():
                task = asyncio.create_task(check_service(service_name, url))
                tasks.append(task)
                
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(30)  # Check every 30 seconds
            
        # Calculate uptime percentages
        results = {}
        overall_uptime = 0
        
        for service_name, stats in service_stats.items():
            uptime = (stats["successes"] / stats["checks"] * 100) if stats["checks"] > 0 else 0
            results[service_name] = {
                "uptime_percentage": uptime,
                "total_checks": stats["checks"],
                "successful_checks": stats["successes"],
                "failed_checks": stats["failures"],
                "passed": uptime >= self.target_uptime
            }
            overall_uptime += uptime
            
        overall_uptime = overall_uptime / len(services) if services else 0
        
        results["overall"] = {
            "uptime_percentage": overall_uptime,
            "target_uptime": self.target_uptime,
            "passed": overall_uptime >= self.target_uptime,
            "duration_minutes": duration_minutes
        }
        
        return results

class SecurityTestSuite:
    """Security validation testing."""
    
    async def test_authentication_security(self, auth_url: str) -> Dict[str, Any]:
        """Test authentication security measures."""
        tests = {
            "invalid_credentials": False,
            "rate_limiting": False,
            "token_validation": False,
            "password_policy": False
        }
        
        async with aiohttp.ClientSession() as session:
            # Test invalid credentials
            try:
                async with session.post(f"{auth_url}/auth/login", 
                                      json={"username": "invalid", "password": "invalid"}) as response:
                    tests["invalid_credentials"] = response.status == 401
            except Exception:
                tests["invalid_credentials"] = False
                
            # Test rate limiting (multiple failed attempts)
            failed_attempts = 0
            for _ in range(10):
                try:
                    async with session.post(f"{auth_url}/auth/login",
                                          json={"username": "test", "password": "wrong"}) as response:
                        if response.status == 429:  # Too Many Requests
                            tests["rate_limiting"] = True
                            break
                        failed_attempts += 1
                except Exception:
                    pass
                    
        return {
            "test_name": "authentication_security",
            "tests": tests,
            "passed": all(tests.values())
        }
        
    async def test_api_security_headers(self, base_url: str) -> Dict[str, Any]:
        """Test security headers are present."""
        required_headers = [
            "strict-transport-security",
            "x-content-type-options", 
            "x-frame-options",
            "content-security-policy"
        ]
        
        header_results = {}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{base_url}/health") as response:
                    for header in required_headers:
                        header_results[header] = header.lower() in [h.lower() for h in response.headers.keys()]
            except Exception as e:
                logger.error(f"Failed to check security headers: {e}")
                for header in required_headers:
                    header_results[header] = False
                    
        return {
            "test_name": "security_headers",
            "headers": header_results,
            "passed": all(header_results.values())
        }

class InfrastructureTestSuite:
    """Infrastructure component testing."""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        
    async def test_container_health(self) -> Dict[str, Any]:
        """Test container health status."""
        container_results = {}
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                if 'grandmodel' in container.name.lower():
                    health_status = container.attrs.get('State', {}).get('Health', {}).get('Status')
                    is_running = container.status == 'running'
                    
                    container_results[container.name] = {
                        "status": container.status,
                        "health": health_status,
                        "running": is_running,
                        "healthy": health_status == 'healthy' if health_status else is_running
                    }
                    
        except Exception as e:
            logger.error(f"Failed to check container health: {e}")
            
        overall_healthy = all(result["healthy"] for result in container_results.values())
        
        return {
            "test_name": "container_health",
            "containers": container_results,
            "passed": overall_healthy
        }
        
    async def test_resource_usage(self) -> Dict[str, Any]:
        """Test system resource usage."""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        thresholds = {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "disk_threshold": 90.0
        }
        
        resource_status = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory.percent,
            "disk_usage": disk.percent,
            "cpu_ok": cpu_usage < thresholds["cpu_threshold"],
            "memory_ok": memory.percent < thresholds["memory_threshold"],
            "disk_ok": disk.percent < thresholds["disk_threshold"]
        }
        
        return {
            "test_name": "resource_usage",
            "resources": resource_status,
            "thresholds": thresholds,
            "passed": all([resource_status["cpu_ok"], resource_status["memory_ok"], resource_status["disk_ok"]])
        }
        
    async def test_database_connectivity(self, db_configs: Dict[str, str]) -> Dict[str, Any]:
        """Test database connectivity."""
        db_results = {}
        
        for db_name, connection_string in db_configs.items():
            try:
                if 'redis' in db_name.lower():
                    # Test Redis
                    redis_client = redis.from_url(connection_string)
                    start_time = time.perf_counter()
                    redis_client.ping()
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    db_results[db_name] = {
                        "connected": True,
                        "latency_ms": latency_ms,
                        "healthy": latency_ms < 100  # 100ms threshold
                    }
                else:
                    # Test PostgreSQL or other SQL databases
                    import sqlalchemy as sa
                    engine = sa.create_engine(connection_string)
                    start_time = time.perf_counter()
                    with engine.connect() as conn:
                        conn.execute(sa.text("SELECT 1"))
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    db_results[db_name] = {
                        "connected": True,
                        "latency_ms": latency_ms,
                        "healthy": latency_ms < 100
                    }
                    
            except Exception as e:
                logger.error(f"Database {db_name} connection failed: {e}")
                db_results[db_name] = {
                    "connected": False,
                    "error": str(e),
                    "healthy": False
                }
                
        overall_healthy = all(result["healthy"] for result in db_results.values())
        
        return {
            "test_name": "database_connectivity",
            "databases": db_results,
            "passed": overall_healthy
        }

class ProductionValidationSuite:
    """Main production validation coordinator."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.latency_suite = LatencyTestSuite(config.get('target_latency_ms', 10.0))
        self.load_suite = LoadTestSuite(config.get('target_throughput', 1000.0))
        self.uptime_suite = UptimeTestSuite(config.get('target_uptime', 99.9))
        self.security_suite = SecurityTestSuite()
        self.infrastructure_suite = InfrastructureTestSuite()
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive production validation."""
        start_time = time.time()
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config,
            "tests": {},
            "summary": {}
        }
        
        # Latency tests
        logger.info("Running latency tests...")
        try:
            latency_results = await self.latency_suite.test_api_latency(
                self.config['base_url'],
                self.config['test_endpoints']
            )
            results["tests"]["latency"] = [asdict(result) for result in latency_results]
        except Exception as e:
            logger.error(f"Latency tests failed: {e}")
            results["tests"]["latency"] = {"error": str(e)}
            
        # Load tests
        logger.info("Running load tests...")
        try:
            load_results = await self.load_suite.test_concurrent_load(
                self.config['base_url'],
                self.config.get('load_test_duration', 60)
            )
            results["tests"]["load"] = load_results
        except Exception as e:
            logger.error(f"Load tests failed: {e}")
            results["tests"]["load"] = {"error": str(e)}
            
        # Uptime tests (shorter duration for full suite)
        logger.info("Running uptime tests...")
        try:
            uptime_results = await self.uptime_suite.test_service_availability(
                self.config['services'],
                duration_minutes=2  # Short test for full suite
            )
            results["tests"]["uptime"] = uptime_results
        except Exception as e:
            logger.error(f"Uptime tests failed: {e}")
            results["tests"]["uptime"] = {"error": str(e)}
            
        # Security tests
        logger.info("Running security tests...")
        try:
            auth_results = await self.security_suite.test_authentication_security(
                self.config['base_url']
            )
            header_results = await self.security_suite.test_api_security_headers(
                self.config['base_url']
            )
            results["tests"]["security"] = {
                "authentication": auth_results,
                "headers": header_results
            }
        except Exception as e:
            logger.error(f"Security tests failed: {e}")
            results["tests"]["security"] = {"error": str(e)}
            
        # Infrastructure tests
        logger.info("Running infrastructure tests...")
        try:
            container_results = await self.infrastructure_suite.test_container_health()
            resource_results = await self.infrastructure_suite.test_resource_usage()
            db_results = await self.infrastructure_suite.test_database_connectivity(
                self.config.get('databases', {})
            )
            results["tests"]["infrastructure"] = {
                "containers": container_results,
                "resources": resource_results,
                "databases": db_results
            }
        except Exception as e:
            logger.error(f"Infrastructure tests failed: {e}")
            results["tests"]["infrastructure"] = {"error": str(e)}
            
        # Calculate summary
        total_duration = time.time() - start_time
        passed_tests = []
        failed_tests = []
        
        def check_test_results(test_data, prefix=""):
            if isinstance(test_data, dict):
                if "passed" in test_data:
                    if test_data["passed"]:
                        passed_tests.append(f"{prefix}{test_data.get('test_name', 'unknown')}")
                    else:
                        failed_tests.append(f"{prefix}{test_data.get('test_name', 'unknown')}")
                else:
                    for key, value in test_data.items():
                        check_test_results(value, f"{prefix}{key}.")
            elif isinstance(test_data, list):
                for item in test_data:
                    check_test_results(item, prefix)
                    
        for test_category, test_data in results["tests"].items():
            check_test_results(test_data, f"{test_category}.")
            
        results["summary"] = {
            "total_duration_seconds": total_duration,
            "total_tests": len(passed_tests) + len(failed_tests),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(passed_tests) / (len(passed_tests) + len(failed_tests)) if (len(passed_tests) + len(failed_tests)) > 0 else 0,
            "overall_passed": len(failed_tests) == 0,
            "passed_test_names": passed_tests,
            "failed_test_names": failed_tests
        }
        
        return results
        
    async def run_quick_validation(self) -> Dict[str, Any]:
        """Run quick validation for rapid feedback."""
        start_time = time.time()
        
        # Quick health checks
        health_checks = []
        
        async with aiohttp.ClientSession() as session:
            for service_name, service_url in self.config['services'].items():
                try:
                    start_check = time.perf_counter()
                    async with session.get(f"{service_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        latency_ms = (time.perf_counter() - start_check) * 1000
                        health_checks.append({
                            "service": service_name,
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "latency_ms": latency_ms,
                            "passed": response.status == 200 and latency_ms < 100
                        })
                except Exception as e:
                    health_checks.append({
                        "service": service_name,
                        "status": "error",
                        "error": str(e),
                        "passed": False
                    })
                    
        # Quick resource check
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        resource_check = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory.percent,
            "cpu_ok": cpu_usage < 80,
            "memory_ok": memory.percent < 85,
            "passed": cpu_usage < 80 and memory.percent < 85
        }
        
        all_passed = all(check["passed"] for check in health_checks) and resource_check["passed"]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "test_type": "quick_validation",
            "duration_seconds": time.time() - start_time,
            "health_checks": health_checks,
            "resource_check": resource_check,
            "overall_passed": all_passed,
            "summary": {
                "healthy_services": sum(1 for check in health_checks if check["passed"]),
                "total_services": len(health_checks),
                "resources_ok": resource_check["passed"]
            }
        }

# Example configuration
EXAMPLE_CONFIG = {
    "target_latency_ms": 10.0,
    "target_throughput": 1000.0,
    "target_uptime": 99.9,
    "base_url": "http://localhost:8000",
    "test_endpoints": ["/health", "/api/risk/current", "/api/agents/performance"],
    "services": {
        "strategic_agent": "http://localhost:8001",
        "tactical_agent": "http://localhost:8002", 
        "risk_agent": "http://localhost:8003"
    },
    "databases": {
        "redis": "redis://localhost:6379",
        "postgres": "postgresql://user:pass@localhost:5432/grandmodel"
    },
    "load_test_duration": 60
}

# Factory function
def create_validation_suite(config: Dict[str, Any] = None) -> ProductionValidationSuite:
    """Create production validation suite."""
    if config is None:
        config = EXAMPLE_CONFIG
    return ProductionValidationSuite(config)

# CLI interface
async def main():
    """Main CLI interface for production validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GrandModel Production Validation Suite")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--quick", action="store_true", help="Run quick validation only")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = EXAMPLE_CONFIG
        
    # Create and run validation suite
    suite = create_validation_suite(config)
    
    if args.quick:
        results = await suite.run_quick_validation()
    else:
        results = await suite.run_full_validation()
        
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))
        
    # Exit with appropriate code
    exit_code = 0 if results.get("overall_passed", False) else 1
    exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())