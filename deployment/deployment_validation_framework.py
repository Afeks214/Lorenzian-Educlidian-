#!/usr/bin/env python3
"""
GrandModel Deployment Validation Framework - Agent 20 Implementation
Enterprise-grade deployment validation with comprehensive testing and validation
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml
import requests
import psutil
from kubernetes import client, config
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Test types"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SMOKE = "smoke"
    REGRESSION = "regression"
    LOAD = "load"
    STRESS = "stress"
    CHAOS = "chaos"
    E2E = "e2e"

class TestStatus(Enum):
    """Test status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class ValidationLevel(Enum):
    """Validation levels"""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"

@dataclass
class TestResult:
    """Test result structure"""
    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

@dataclass
class TestSuite:
    """Test suite definition"""
    name: str
    description: str
    test_type: TestType
    tests: List[str] = field(default_factory=list)
    setup_commands: List[str] = field(default_factory=list)
    teardown_commands: List[str] = field(default_factory=list)
    timeout: int = 300  # seconds
    retry_count: int = 0
    parallel: bool = False
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ValidationReport:
    """Validation report structure"""
    id: str
    deployment_id: str
    validation_level: ValidationLevel
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    success_rate: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class HealthChecker:
    """Health check validator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get('timeout', 30)
    
    async def check_service_health(self, service_url: str, expected_status: int = 200) -> TestResult:
        """Check service health endpoint"""
        test_result = TestResult(
            test_id=f"health_check_{service_url.replace('://', '_').replace('/', '_')}",
            test_name=f"Health check for {service_url}",
            test_type=TestType.SMOKE,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            response = requests.get(
                f"{service_url}/health",
                timeout=self.timeout,
                headers={'User-Agent': 'GrandModel-Validator/1.0'}
            )
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            if response.status_code == expected_status:
                test_result.status = TestStatus.PASSED
                test_result.details = {
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'response_body': response.text[:1000] if response.text else None
                }
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Expected status {expected_status}, got {response.status_code}"
                test_result.details = {
                    'status_code': response.status_code,
                    'response_body': response.text[:1000] if response.text else None
                }
                
        except requests.RequestException as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
            
        return test_result
    
    async def check_kubernetes_health(self, namespace: str = "grandmodel") -> List[TestResult]:
        """Check Kubernetes deployment health"""
        results = []
        
        try:
            config.load_incluster_config()
            v1 = client.CoreV1Api()
            apps_v1 = client.AppsV1Api()
            
            # Check pods
            pods_result = TestResult(
                test_id="k8s_pods_health",
                test_name="Kubernetes pods health check",
                test_type=TestType.SMOKE,
                status=TestStatus.RUNNING,
                start_time=datetime.now()
            )
            
            pods = v1.list_namespaced_pod(namespace=namespace)
            healthy_pods = 0
            total_pods = len(pods.items)
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    if pod.status.conditions:
                        for condition in pod.status.conditions:
                            if condition.type == "Ready" and condition.status == "True":
                                healthy_pods += 1
                                break
            
            pods_result.end_time = datetime.now()
            pods_result.duration = (pods_result.end_time - pods_result.start_time).total_seconds()
            
            if healthy_pods == total_pods and total_pods > 0:
                pods_result.status = TestStatus.PASSED
            else:
                pods_result.status = TestStatus.FAILED
                pods_result.error_message = f"Only {healthy_pods}/{total_pods} pods are healthy"
            
            pods_result.details = {
                'total_pods': total_pods,
                'healthy_pods': healthy_pods,
                'unhealthy_pods': total_pods - healthy_pods
            }
            
            results.append(pods_result)
            
            # Check deployments
            deployments_result = TestResult(
                test_id="k8s_deployments_health",
                test_name="Kubernetes deployments health check",
                test_type=TestType.SMOKE,
                status=TestStatus.RUNNING,
                start_time=datetime.now()
            )
            
            deployments = apps_v1.list_namespaced_deployment(namespace=namespace)
            healthy_deployments = 0
            total_deployments = len(deployments.items)
            
            for deployment in deployments.items:
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas == deployment.spec.replicas):
                    healthy_deployments += 1
            
            deployments_result.end_time = datetime.now()
            deployments_result.duration = (deployments_result.end_time - deployments_result.start_time).total_seconds()
            
            if healthy_deployments == total_deployments and total_deployments > 0:
                deployments_result.status = TestStatus.PASSED
            else:
                deployments_result.status = TestStatus.FAILED
                deployments_result.error_message = f"Only {healthy_deployments}/{total_deployments} deployments are healthy"
            
            deployments_result.details = {
                'total_deployments': total_deployments,
                'healthy_deployments': healthy_deployments,
                'unhealthy_deployments': total_deployments - healthy_deployments
            }
            
            results.append(deployments_result)
            
        except Exception as e:
            error_result = TestResult(
                test_id="k8s_health_error",
                test_name="Kubernetes health check error",
                test_type=TestType.SMOKE,
                status=TestStatus.ERROR,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e)
            )
            results.append(error_result)
        
        return results
    
    async def check_database_health(self, database_config: Dict[str, Any]) -> TestResult:
        """Check database health"""
        test_result = TestResult(
            test_id="database_health",
            test_name="Database health check",
            test_type=TestType.SMOKE,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # This would use the actual database connection
            # For now, we'll simulate the check
            await asyncio.sleep(0.1)  # Simulate database connection time
            
            # Check if database is reachable
            db_host = database_config.get('host', 'localhost')
            db_port = database_config.get('port', 5432)
            
            # Simple TCP connection test
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((db_host, db_port))
            sock.close()
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            if result == 0:
                test_result.status = TestStatus.PASSED
                test_result.details = {
                    'host': db_host,
                    'port': db_port,
                    'connection_time': test_result.duration
                }
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Cannot connect to database at {db_host}:{db_port}"
                
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
            
        return test_result

class FunctionalTester:
    """Functional testing validator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout = config.get('timeout', 60)
        self.base_url = config.get('base_url', 'http://localhost:8000')
    
    async def test_api_endpoints(self, endpoints: List[Dict[str, Any]]) -> List[TestResult]:
        """Test API endpoints"""
        results = []
        
        for endpoint in endpoints:
            test_result = TestResult(
                test_id=f"api_test_{endpoint['name']}",
                test_name=f"API test for {endpoint['name']}",
                test_type=TestType.FUNCTIONAL,
                status=TestStatus.RUNNING,
                start_time=datetime.now()
            )
            
            try:
                url = f"{self.base_url}{endpoint['path']}"
                method = endpoint.get('method', 'GET')
                headers = endpoint.get('headers', {})
                data = endpoint.get('data')
                expected_status = endpoint.get('expected_status', 200)
                
                if method == 'GET':
                    response = requests.get(url, headers=headers, timeout=self.timeout)
                elif method == 'POST':
                    response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                elif method == 'PUT':
                    response = requests.put(url, headers=headers, json=data, timeout=self.timeout)
                elif method == 'DELETE':
                    response = requests.delete(url, headers=headers, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                test_result.end_time = datetime.now()
                test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
                
                if response.status_code == expected_status:
                    test_result.status = TestStatus.PASSED
                else:
                    test_result.status = TestStatus.FAILED
                    test_result.error_message = f"Expected {expected_status}, got {response.status_code}"
                
                test_result.details = {
                    'url': url,
                    'method': method,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'response_size': len(response.content) if response.content else 0
                }
                
                # Validate response structure if specified
                if 'response_schema' in endpoint:
                    schema_valid = self.validate_response_schema(response.json(), endpoint['response_schema'])
                    if not schema_valid:
                        test_result.status = TestStatus.FAILED
                        test_result.error_message = "Response schema validation failed"
                
            except Exception as e:
                test_result.end_time = datetime.now()
                test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
                test_result.status = TestStatus.ERROR
                test_result.error_message = str(e)
            
            results.append(test_result)
        
        return results
    
    def validate_response_schema(self, response_data: Any, schema: Dict[str, Any]) -> bool:
        """Validate response against schema"""
        try:
            # Simple schema validation
            if schema.get('type') == 'object':
                if not isinstance(response_data, dict):
                    return False
                
                required_fields = schema.get('required', [])
                for field in required_fields:
                    if field not in response_data:
                        return False
            
            return True
        except Exception:
            return False
    
    async def test_strategic_agent_functionality(self) -> List[TestResult]:
        """Test strategic agent functionality"""
        results = []
        
        # Test strategic agent decision making
        test_result = TestResult(
            test_id="strategic_agent_decision",
            test_name="Strategic agent decision making test",
            test_type=TestType.FUNCTIONAL,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Mock market data
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'price': 100.0,
                'volume': 1000,
                'indicators': {
                    'rsi': 50.0,
                    'macd': 0.1,
                    'bollinger_upper': 105.0,
                    'bollinger_lower': 95.0
                }
            }
            
            # Send to strategic agent
            response = requests.post(
                f"{self.base_url}/api/strategic/analyze",
                json=market_data,
                timeout=self.timeout
            )
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            if response.status_code == 200:
                result_data = response.json()
                if 'decision' in result_data and 'confidence' in result_data:
                    test_result.status = TestStatus.PASSED
                    test_result.details = {
                        'decision': result_data['decision'],
                        'confidence': result_data['confidence'],
                        'response_time': test_result.duration
                    }
                else:
                    test_result.status = TestStatus.FAILED
                    test_result.error_message = "Invalid response format"
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"HTTP {response.status_code}: {response.text}"
                
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
        
        results.append(test_result)
        return results
    
    async def test_risk_management_functionality(self) -> List[TestResult]:
        """Test risk management functionality"""
        results = []
        
        # Test VaR calculation
        test_result = TestResult(
            test_id="risk_var_calculation",
            test_name="VaR calculation test",
            test_type=TestType.FUNCTIONAL,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Mock portfolio data
            portfolio_data = {
                'positions': [
                    {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0},
                    {'symbol': 'GOOGL', 'quantity': 50, 'price': 2500.0},
                    {'symbol': 'MSFT', 'quantity': 75, 'price': 300.0}
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            # Send to risk management
            response = requests.post(
                f"{self.base_url}/api/risk/calculate-var",
                json=portfolio_data,
                timeout=self.timeout
            )
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            if response.status_code == 200:
                result_data = response.json()
                if 'var_value' in result_data and 'confidence_level' in result_data:
                    test_result.status = TestStatus.PASSED
                    test_result.details = {
                        'var_value': result_data['var_value'],
                        'confidence_level': result_data['confidence_level'],
                        'calculation_time': test_result.duration
                    }
                    
                    # Check if VaR calculation is within performance target
                    if test_result.duration > 0.005:  # 5ms threshold
                        test_result.status = TestStatus.FAILED
                        test_result.error_message = f"VaR calculation too slow: {test_result.duration:.3f}s"
                else:
                    test_result.status = TestStatus.FAILED
                    test_result.error_message = "Invalid response format"
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"HTTP {response.status_code}: {response.text}"
                
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
        
        results.append(test_result)
        return results

class PerformanceTester:
    """Performance testing validator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:8000')
        self.concurrent_users = config.get('concurrent_users', 10)
        self.test_duration = config.get('test_duration', 60)
    
    async def run_load_test(self, endpoint: str, requests_per_second: int = 100) -> TestResult:
        """Run load test on endpoint"""
        test_result = TestResult(
            test_id=f"load_test_{endpoint.replace('/', '_')}",
            test_name=f"Load test for {endpoint}",
            test_type=TestType.LOAD,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            url = f"{self.base_url}{endpoint}"
            total_requests = 0
            successful_requests = 0
            failed_requests = 0
            response_times = []
            
            # Run load test
            start_time = time.time()
            while time.time() - start_time < self.test_duration:
                tasks = []
                for _ in range(requests_per_second):
                    tasks.append(self.make_request(url))
                
                # Execute requests concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    total_requests += 1
                    if isinstance(result, Exception):
                        failed_requests += 1
                    else:
                        successful_requests += 1
                        response_times.append(result)
                
                # Wait for next second
                await asyncio.sleep(max(0, 1 - (time.time() % 1)))
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            # Calculate metrics
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
                p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]
            else:
                avg_response_time = 0
                p95_response_time = 0
                p99_response_time = 0
            
            success_rate = successful_requests / total_requests if total_requests > 0 else 0
            
            test_result.details = {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'p99_response_time': p99_response_time,
                'requests_per_second': total_requests / test_result.duration
            }
            
            # Determine test status
            if success_rate >= 0.99 and p95_response_time <= 0.1:  # 99% success rate and 100ms P95
                test_result.status = TestStatus.PASSED
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Performance criteria not met: {success_rate:.2%} success rate, {p95_response_time:.3f}s P95"
            
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
        
        return test_result
    
    async def make_request(self, url: str) -> float:
        """Make a single request and return response time"""
        try:
            start_time = time.time()
            response = requests.get(url, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return response_time
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            raise e
    
    async def run_stress_test(self, endpoint: str) -> TestResult:
        """Run stress test to find breaking point"""
        test_result = TestResult(
            test_id=f"stress_test_{endpoint.replace('/', '_')}",
            test_name=f"Stress test for {endpoint}",
            test_type=TestType.STRESS,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            url = f"{self.base_url}{endpoint}"
            max_rps = 0
            current_rps = 10
            
            # Gradually increase load until breaking point
            while current_rps <= 1000:
                success_rate = await self.test_rps_level(url, current_rps, duration=30)
                
                if success_rate >= 0.95:
                    max_rps = current_rps
                    current_rps *= 2
                else:
                    break
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            test_result.details = {
                'max_sustainable_rps': max_rps,
                'breaking_point_rps': current_rps,
                'test_duration': test_result.duration
            }
            
            if max_rps >= 100:  # Minimum performance requirement
                test_result.status = TestStatus.PASSED
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Max sustainable RPS too low: {max_rps}"
            
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
        
        return test_result
    
    async def test_rps_level(self, url: str, rps: int, duration: int = 30) -> float:
        """Test specific RPS level"""
        total_requests = 0
        successful_requests = 0
        
        start_time = time.time()
        while time.time() - start_time < duration:
            tasks = []
            for _ in range(rps):
                tasks.append(self.make_request(url))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                total_requests += 1
                if not isinstance(result, Exception):
                    successful_requests += 1
            
            await asyncio.sleep(max(0, 1 - (time.time() % 1)))
        
        return successful_requests / total_requests if total_requests > 0 else 0

class SecurityTester:
    """Security testing validator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:8000')
        self.timeout = config.get('timeout', 30)
    
    async def run_security_tests(self) -> List[TestResult]:
        """Run security tests"""
        results = []
        
        # Test authentication
        auth_result = await self.test_authentication()
        results.append(auth_result)
        
        # Test authorization
        authz_result = await self.test_authorization()
        results.append(authz_result)
        
        # Test input validation
        input_result = await self.test_input_validation()
        results.append(input_result)
        
        # Test SQL injection protection
        sql_result = await self.test_sql_injection_protection()
        results.append(sql_result)
        
        # Test XSS protection
        xss_result = await self.test_xss_protection()
        results.append(xss_result)
        
        return results
    
    async def test_authentication(self) -> TestResult:
        """Test authentication mechanisms"""
        test_result = TestResult(
            test_id="security_authentication",
            test_name="Authentication security test",
            test_type=TestType.SECURITY,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Test unauthenticated access
            response = requests.get(f"{self.base_url}/api/protected", timeout=self.timeout)
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            if response.status_code in [401, 403]:
                test_result.status = TestStatus.PASSED
                test_result.details = {
                    'unauthenticated_access_blocked': True,
                    'response_code': response.status_code
                }
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Unauthenticated access allowed: {response.status_code}"
                
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
        
        return test_result
    
    async def test_authorization(self) -> TestResult:
        """Test authorization mechanisms"""
        test_result = TestResult(
            test_id="security_authorization",
            test_name="Authorization security test",
            test_type=TestType.SECURITY,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Test with invalid token
            headers = {'Authorization': 'Bearer invalid_token'}
            response = requests.get(f"{self.base_url}/api/admin", headers=headers, timeout=self.timeout)
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            if response.status_code in [401, 403]:
                test_result.status = TestStatus.PASSED
                test_result.details = {
                    'invalid_token_blocked': True,
                    'response_code': response.status_code
                }
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Invalid token access allowed: {response.status_code}"
                
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
        
        return test_result
    
    async def test_input_validation(self) -> TestResult:
        """Test input validation"""
        test_result = TestResult(
            test_id="security_input_validation",
            test_name="Input validation security test",
            test_type=TestType.SECURITY,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Test with malicious input
            malicious_data = {
                'user_id': '../../../etc/passwd',
                'data': '<script>alert("xss")</script>',
                'query': "'; DROP TABLE users; --"
            }
            
            response = requests.post(
                f"{self.base_url}/api/data",
                json=malicious_data,
                timeout=self.timeout
            )
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            if response.status_code in [400, 422]:
                test_result.status = TestStatus.PASSED
                test_result.details = {
                    'malicious_input_blocked': True,
                    'response_code': response.status_code
                }
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"Malicious input not blocked: {response.status_code}"
                
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
        
        return test_result
    
    async def test_sql_injection_protection(self) -> TestResult:
        """Test SQL injection protection"""
        test_result = TestResult(
            test_id="security_sql_injection",
            test_name="SQL injection protection test",
            test_type=TestType.SECURITY,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Test SQL injection payloads
            sql_payloads = [
                "1' OR '1'='1",
                "1; DROP TABLE users; --",
                "1' UNION SELECT * FROM users --"
            ]
            
            blocked_payloads = 0
            for payload in sql_payloads:
                response = requests.get(
                    f"{self.base_url}/api/user",
                    params={'id': payload},
                    timeout=self.timeout
                )
                
                if response.status_code in [400, 422, 500]:
                    blocked_payloads += 1
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            if blocked_payloads == len(sql_payloads):
                test_result.status = TestStatus.PASSED
                test_result.details = {
                    'sql_injection_blocked': True,
                    'blocked_payloads': blocked_payloads,
                    'total_payloads': len(sql_payloads)
                }
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"SQL injection not fully blocked: {blocked_payloads}/{len(sql_payloads)}"
                
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
        
        return test_result
    
    async def test_xss_protection(self) -> TestResult:
        """Test XSS protection"""
        test_result = TestResult(
            test_id="security_xss_protection",
            test_name="XSS protection test",
            test_type=TestType.SECURITY,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            # Test XSS payloads
            xss_payloads = [
                "<script>alert('xss')</script>",
                "<img src='x' onerror='alert(1)'>",
                "javascript:alert('xss')"
            ]
            
            protected_responses = 0
            for payload in xss_payloads:
                response = requests.post(
                    f"{self.base_url}/api/comment",
                    json={'content': payload},
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    # Check if payload is sanitized in response
                    if payload not in response.text:
                        protected_responses += 1
                else:
                    protected_responses += 1  # Rejected outright
            
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            
            if protected_responses == len(xss_payloads):
                test_result.status = TestStatus.PASSED
                test_result.details = {
                    'xss_protection_active': True,
                    'protected_payloads': protected_responses,
                    'total_payloads': len(xss_payloads)
                }
            else:
                test_result.status = TestStatus.FAILED
                test_result.error_message = f"XSS not fully protected: {protected_responses}/{len(xss_payloads)}"
                
        except Exception as e:
            test_result.end_time = datetime.now()
            test_result.duration = (test_result.end_time - test_result.start_time).total_seconds()
            test_result.status = TestStatus.ERROR
            test_result.error_message = str(e)
        
        return test_result

class DeploymentValidator:
    """Main deployment validation orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_checker = HealthChecker(config)
        self.functional_tester = FunctionalTester(config)
        self.performance_tester = PerformanceTester(config)
        self.security_tester = SecurityTester(config)
        
        # Test suites
        self.test_suites = self.load_test_suites()
    
    def load_test_suites(self) -> Dict[str, TestSuite]:
        """Load test suite definitions"""
        suites = {}
        
        # Smoke test suite
        suites['smoke'] = TestSuite(
            name="smoke",
            description="Basic smoke tests for deployment validation",
            test_type=TestType.SMOKE,
            tests=[
                'health_check_strategic',
                'health_check_tactical',
                'health_check_risk',
                'k8s_pods_health',
                'k8s_deployments_health',
                'database_health'
            ],
            timeout=300,
            parallel=True
        )
        
        # Functional test suite
        suites['functional'] = TestSuite(
            name="functional",
            description="Functional tests for core business logic",
            test_type=TestType.FUNCTIONAL,
            tests=[
                'strategic_agent_decision',
                'tactical_agent_execution',
                'risk_var_calculation',
                'api_endpoints_test'
            ],
            timeout=600,
            parallel=False,
            dependencies=['smoke']
        )
        
        # Performance test suite
        suites['performance'] = TestSuite(
            name="performance",
            description="Performance and load tests",
            test_type=TestType.PERFORMANCE,
            tests=[
                'load_test_strategic',
                'load_test_tactical',
                'load_test_risk',
                'stress_test_system'
            ],
            timeout=1800,
            parallel=True,
            dependencies=['functional']
        )
        
        # Security test suite
        suites['security'] = TestSuite(
            name="security",
            description="Security validation tests",
            test_type=TestType.SECURITY,
            tests=[
                'authentication_test',
                'authorization_test',
                'input_validation_test',
                'sql_injection_test',
                'xss_protection_test'
            ],
            timeout=900,
            parallel=True,
            dependencies=['functional']
        )
        
        return suites
    
    async def validate_deployment(self, deployment_id: str, validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE) -> ValidationReport:
        """Validate deployment"""
        logger.info(f"Starting deployment validation for {deployment_id} at {validation_level.value} level")
        
        report = ValidationReport(
            id=f"validation_{deployment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            deployment_id=deployment_id,
            validation_level=validation_level,
            start_time=datetime.now()
        )
        
        try:
            # Determine which test suites to run based on validation level
            suites_to_run = self.get_suites_for_level(validation_level)
            
            # Run test suites
            for suite_name in suites_to_run:
                suite = self.test_suites[suite_name]
                logger.info(f"Running test suite: {suite_name}")
                
                # Check dependencies
                if suite.dependencies:
                    for dep in suite.dependencies:
                        if dep not in [s.name for s in report.test_results if s.status == TestStatus.PASSED]:
                            logger.warning(f"Dependency {dep} not satisfied for suite {suite_name}")
                            continue
                
                # Run tests in suite
                suite_results = await self.run_test_suite(suite)
                report.test_results.extend(suite_results)
            
            # Calculate report metrics
            report.total_tests = len(report.test_results)
            report.passed_tests = len([r for r in report.test_results if r.status == TestStatus.PASSED])
            report.failed_tests = len([r for r in report.test_results if r.status == TestStatus.FAILED])
            report.skipped_tests = len([r for r in report.test_results if r.status == TestStatus.SKIPPED])
            report.error_tests = len([r for r in report.test_results if r.status == TestStatus.ERROR])
            
            report.success_rate = report.passed_tests / report.total_tests if report.total_tests > 0 else 0
            
            # Generate recommendations
            report.recommendations = self.generate_recommendations(report)
            
            report.end_time = datetime.now()
            
            logger.info(f"Deployment validation completed: {report.success_rate:.2%} success rate")
            
        except Exception as e:
            report.end_time = datetime.now()
            logger.error(f"Deployment validation failed: {e}")
            
            # Add error test result
            error_result = TestResult(
                test_id="validation_error",
                test_name="Validation framework error",
                test_type=TestType.SMOKE,
                status=TestStatus.ERROR,
                start_time=report.start_time,
                end_time=report.end_time,
                error_message=str(e)
            )
            report.test_results.append(error_result)
        
        return report
    
    def get_suites_for_level(self, level: ValidationLevel) -> List[str]:
        """Get test suites for validation level"""
        if level == ValidationLevel.BASIC:
            return ['smoke']
        elif level == ValidationLevel.COMPREHENSIVE:
            return ['smoke', 'functional', 'performance']
        elif level == ValidationLevel.EXHAUSTIVE:
            return ['smoke', 'functional', 'performance', 'security']
        else:
            return ['smoke']
    
    async def run_test_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run a test suite"""
        results = []
        
        try:
            # Run setup commands
            for cmd in suite.setup_commands:
                await self.run_command(cmd)
            
            # Run tests
            if suite.parallel:
                # Run tests in parallel
                tasks = []
                for test_name in suite.tests:
                    task = asyncio.create_task(self.run_test(test_name))
                    tasks.append(task)
                
                test_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in test_results:
                    if isinstance(result, Exception):
                        error_result = TestResult(
                            test_id="parallel_test_error",
                            test_name="Parallel test error",
                            test_type=suite.test_type,
                            status=TestStatus.ERROR,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            error_message=str(result)
                        )
                        results.append(error_result)
                    else:
                        results.extend(result if isinstance(result, list) else [result])
            else:
                # Run tests sequentially
                for test_name in suite.tests:
                    test_results = await self.run_test(test_name)
                    results.extend(test_results if isinstance(test_results, list) else [test_results])
            
            # Run teardown commands
            for cmd in suite.teardown_commands:
                await self.run_command(cmd)
                
        except Exception as e:
            error_result = TestResult(
                test_id=f"suite_error_{suite.name}",
                test_name=f"Test suite {suite.name} error",
                test_type=suite.test_type,
                status=TestStatus.ERROR,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e)
            )
            results.append(error_result)
        
        return results
    
    async def run_test(self, test_name: str) -> List[TestResult]:
        """Run a specific test"""
        try:
            if test_name.startswith('health_check_'):
                service = test_name.replace('health_check_', '')
                service_url = f"{self.config['base_url']}"
                if service == 'strategic':
                    service_url += '/strategic'
                elif service == 'tactical':
                    service_url += '/tactical'
                elif service == 'risk':
                    service_url += '/risk'
                
                result = await self.health_checker.check_service_health(service_url)
                return [result]
            
            elif test_name == 'k8s_pods_health':
                results = await self.health_checker.check_kubernetes_health()
                return results
            
            elif test_name == 'k8s_deployments_health':
                results = await self.health_checker.check_kubernetes_health()
                return [r for r in results if r.test_id == 'k8s_deployments_health']
            
            elif test_name == 'database_health':
                result = await self.health_checker.check_database_health(self.config.get('database', {}))
                return [result]
            
            elif test_name == 'strategic_agent_decision':
                results = await self.functional_tester.test_strategic_agent_functionality()
                return results
            
            elif test_name == 'risk_var_calculation':
                results = await self.functional_tester.test_risk_management_functionality()
                return results
            
            elif test_name.startswith('load_test_'):
                service = test_name.replace('load_test_', '')
                endpoint = f"/api/{service}/test"
                result = await self.performance_tester.run_load_test(endpoint)
                return [result]
            
            elif test_name == 'stress_test_system':
                result = await self.performance_tester.run_stress_test("/api/health")
                return [result]
            
            elif test_name.endswith('_test'):
                security_results = await self.security_tester.run_security_tests()
                return [r for r in security_results if test_name.replace('_test', '') in r.test_id]
            
            else:
                # Unknown test
                error_result = TestResult(
                    test_id=test_name,
                    test_name=test_name,
                    test_type=TestType.SMOKE,
                    status=TestStatus.ERROR,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message=f"Unknown test: {test_name}"
                )
                return [error_result]
                
        except Exception as e:
            error_result = TestResult(
                test_id=test_name,
                test_name=test_name,
                test_type=TestType.SMOKE,
                status=TestStatus.ERROR,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e)
            )
            return [error_result]
    
    async def run_command(self, command: str) -> bool:
        """Run a shell command"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Command succeeded: {command}")
                return True
            else:
                logger.error(f"Command failed: {command} - {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Command error: {command} - {e}")
            return False
    
    def generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if report.failed_tests > 0:
            recommendations.append("Address failed tests before proceeding with deployment")
        
        if report.success_rate < 0.95:
            recommendations.append("Improve test success rate to at least 95%")
        
        # Check performance results
        performance_results = [r for r in report.test_results if r.test_type == TestType.PERFORMANCE]
        for result in performance_results:
            if result.status == TestStatus.FAILED:
                recommendations.append(f"Address performance issues in {result.test_name}")
        
        # Check security results
        security_results = [r for r in report.test_results if r.test_type == TestType.SECURITY]
        for result in security_results:
            if result.status == TestStatus.FAILED:
                recommendations.append(f"Address security issues in {result.test_name}")
        
        if not recommendations:
            recommendations.append("All tests passed successfully - deployment ready")
        
        return recommendations
    
    def export_report(self, report: ValidationReport, format: str = 'json') -> str:
        """Export validation report"""
        if format == 'json':
            return json.dumps({
                'id': report.id,
                'deployment_id': report.deployment_id,
                'validation_level': report.validation_level.value,
                'start_time': report.start_time.isoformat(),
                'end_time': report.end_time.isoformat() if report.end_time else None,
                'total_tests': report.total_tests,
                'passed_tests': report.passed_tests,
                'failed_tests': report.failed_tests,
                'skipped_tests': report.skipped_tests,
                'error_tests': report.error_tests,
                'success_rate': report.success_rate,
                'test_results': [
                    {
                        'test_id': r.test_id,
                        'test_name': r.test_name,
                        'test_type': r.test_type.value,
                        'status': r.status.value,
                        'duration': r.duration,
                        'error_message': r.error_message,
                        'details': r.details
                    }
                    for r in report.test_results
                ],
                'recommendations': report.recommendations
            }, indent=2)
        else:
            return str(report)

# Example usage and testing
async def main():
    """Main function for testing"""
    config = {
        'base_url': 'http://localhost:8000',
        'timeout': 30,
        'concurrent_users': 10,
        'test_duration': 60,
        'database': {
            'host': 'localhost',
            'port': 5432
        }
    }
    
    validator = DeploymentValidator(config)
    
    # Run validation
    report = await validator.validate_deployment(
        deployment_id="test-deployment-001",
        validation_level=ValidationLevel.COMPREHENSIVE
    )
    
    # Export report
    report_json = validator.export_report(report, format='json')
    print(report_json)
    
    # Summary
    print(f"\nValidation Summary:")
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed_tests}")
    print(f"Failed: {report.failed_tests}")
    print(f"Success Rate: {report.success_rate:.2%}")
    print(f"Recommendations: {len(report.recommendations)}")

if __name__ == "__main__":
    asyncio.run(main())