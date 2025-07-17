"""
Automated RTO Validation Testing System for continuous verification of recovery objectives.

This module provides:
- Automated testing scenarios for RTO validation
- Continuous testing framework
- Test result analysis and reporting
- Performance regression detection
- Compliance validation
- Load testing for recovery scenarios
"""

import asyncio
import json
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from src.monitoring.rto_monitor import RTOMonitoringSystem, RTOMetric, RTOStatus, RTOTarget
from src.monitoring.rto_analytics import RTOAnalyticsSystem
from src.core.event_bus import EventBus

logger = logging.getLogger(__name__)

class TestType(Enum):
    """Test types for RTO validation."""
    SMOKE = "smoke"
    FUNCTIONAL = "functional"
    LOAD = "load"
    STRESS = "stress"
    CHAOS = "chaos"
    REGRESSION = "regression"
    COMPLIANCE = "compliance"

class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestScenario:
    """RTO test scenario definition."""
    name: str
    test_type: TestType
    component: str
    description: str
    failure_scenario: str
    expected_rto: float
    tolerance_percentage: float = 20.0
    timeout_seconds: int = 300
    enabled: bool = True
    prerequisites: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "test_type": self.test_type.value,
            "component": self.component,
            "description": self.description,
            "failure_scenario": self.failure_scenario,
            "expected_rto": self.expected_rto,
            "tolerance_percentage": self.tolerance_percentage,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "prerequisites": self.prerequisites,
            "metadata": self.metadata
        }

@dataclass
class TestResult:
    """Test execution result."""
    test_id: str
    scenario_name: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    actual_rto: Optional[float] = None
    expected_rto: Optional[float] = None
    passed: bool = False
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Get test duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def rto_variance(self) -> float:
        """Get RTO variance percentage."""
        if self.actual_rto and self.expected_rto:
            return ((self.actual_rto - self.expected_rto) / self.expected_rto) * 100
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "scenario_name": self.scenario_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "actual_rto": self.actual_rto,
            "expected_rto": self.expected_rto,
            "rto_variance": self.rto_variance,
            "passed": self.passed,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "details": self.details
        }

@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    description: str
    scenarios: List[TestScenario]
    schedule: Optional[str] = None  # Cron expression
    enabled: bool = True
    parallel_execution: bool = False
    max_concurrent_tests: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "schedule": self.schedule,
            "enabled": self.enabled,
            "parallel_execution": self.parallel_execution,
            "max_concurrent_tests": self.max_concurrent_tests
        }

class RTOTestDatabase:
    """Database for RTO test results."""
    
    def __init__(self, db_path: str = "rto_tests.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_results (
                    test_id TEXT PRIMARY KEY,
                    scenario_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration REAL,
                    actual_rto REAL,
                    expected_rto REAL,
                    rto_variance REAL,
                    passed BOOLEAN,
                    error_message TEXT,
                    metrics TEXT,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    run_id TEXT PRIMARY KEY,
                    suite_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_tests INTEGER,
                    passed_tests INTEGER,
                    failed_tests INTEGER,
                    duration REAL,
                    success_rate REAL,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_results_scenario_time 
                ON test_results(scenario_name, start_time)
            """)
    
    def store_test_result(self, result: TestResult):
        """Store test result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO test_results (
                    test_id, scenario_name, status, start_time, end_time, duration,
                    actual_rto, expected_rto, rto_variance, passed, error_message,
                    metrics, details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.test_id,
                result.scenario_name,
                result.status.value,
                result.start_time.isoformat(),
                result.end_time.isoformat() if result.end_time else None,
                result.duration,
                result.actual_rto,
                result.expected_rto,
                result.rto_variance,
                result.passed,
                result.error_message,
                json.dumps(result.metrics),
                json.dumps(result.details)
            ))
    
    def get_test_history(self, scenario_name: str, days: int = 30) -> List[TestResult]:
        """Get test history for a scenario."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM test_results 
                WHERE scenario_name = ? AND start_time >= ?
                ORDER BY start_time DESC
            """, (scenario_name, cutoff.isoformat()))
            
            results = []
            for row in cursor.fetchall():
                result = TestResult(
                    test_id=row['test_id'],
                    scenario_name=row['scenario_name'],
                    status=TestStatus(row['status']),
                    start_time=datetime.fromisoformat(row['start_time']),
                    end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                    actual_rto=row['actual_rto'],
                    expected_rto=row['expected_rto'],
                    passed=bool(row['passed']),
                    error_message=row['error_message'],
                    metrics=json.loads(row['metrics']) if row['metrics'] else {},
                    details=json.loads(row['details']) if row['details'] else {}
                )
                results.append(result)
            
            return results
    
    def get_test_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get test summary statistics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_tests,
                    SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) as passed_tests,
                    SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) as failed_tests,
                    AVG(duration) as avg_duration,
                    AVG(actual_rto) as avg_rto,
                    AVG(rto_variance) as avg_variance
                FROM test_results 
                WHERE start_time >= ?
            """, (cutoff.isoformat(),))
            
            row = cursor.fetchone()
            
            return {
                "total_tests": row['total_tests'] or 0,
                "passed_tests": row['passed_tests'] or 0,
                "failed_tests": row['failed_tests'] or 0,
                "success_rate": (row['passed_tests'] / row['total_tests'] * 100) if row['total_tests'] else 0,
                "avg_duration": row['avg_duration'] or 0,
                "avg_rto": row['avg_rto'] or 0,
                "avg_variance": row['avg_variance'] or 0
            }

class RTOTestExecutor:
    """RTO test executor."""
    
    def __init__(self, rto_monitor: RTOMonitoringSystem):
        self.rto_monitor = rto_monitor
        self.database = RTOTestDatabase()
        self._running_tests = set()
        self._lock = threading.Lock()
    
    async def execute_test(self, scenario: TestScenario) -> TestResult:
        """Execute a single test scenario."""
        test_id = f"{scenario.name}_{int(time.time())}"
        
        with self._lock:
            if test_id in self._running_tests:
                raise ValueError(f"Test {test_id} is already running")
            self._running_tests.add(test_id)
        
        result = TestResult(
            test_id=test_id,
            scenario_name=scenario.name,
            status=TestStatus.RUNNING,
            start_time=datetime.utcnow(),
            expected_rto=scenario.expected_rto
        )
        
        try:
            logger.info(f"Starting RTO test: {scenario.name}")
            
            # Check prerequisites
            if not await self._check_prerequisites(scenario.prerequisites):
                result.status = TestStatus.SKIPPED
                result.error_message = "Prerequisites not met"
                return result
            
            # Execute test scenario
            start_time = time.time()
            
            # Simulate failure and recovery
            rto_metric = await self.rto_monitor.simulate_failure_recovery(
                scenario.component,
                scenario.failure_scenario
            )
            
            end_time = time.time()
            test_duration = end_time - start_time
            
            result.end_time = datetime.utcnow()
            result.actual_rto = rto_metric.actual_seconds
            result.metrics = {
                "test_duration": test_duration,
                "failure_start": rto_metric.failure_start.isoformat() if rto_metric.failure_start else None,
                "recovery_start": rto_metric.recovery_start.isoformat() if rto_metric.recovery_start else None,
                "recovery_complete": rto_metric.recovery_complete.isoformat() if rto_metric.recovery_complete else None,
                "breach_percentage": rto_metric.breach_percentage
            }
            
            # Evaluate test result
            tolerance = scenario.expected_rto * (scenario.tolerance_percentage / 100)
            within_tolerance = abs(result.actual_rto - scenario.expected_rto) <= tolerance
            
            if within_tolerance and rto_metric.status != RTOStatus.CRITICAL:
                result.status = TestStatus.PASSED
                result.passed = True
            else:
                result.status = TestStatus.FAILED
                result.passed = False
                result.error_message = f"RTO {result.actual_rto:.2f}s exceeds tolerance of {tolerance:.2f}s"
            
            result.details = {
                "scenario": scenario.to_dict(),
                "tolerance": tolerance,
                "within_tolerance": within_tolerance,
                "rto_metric": rto_metric.to_dict()
            }
            
            logger.info(f"Test {scenario.name} completed: {result.status.value}")
            
        except asyncio.TimeoutError:
            result.status = TestStatus.FAILED
            result.error_message = f"Test timed out after {scenario.timeout_seconds} seconds"
            result.end_time = datetime.utcnow()
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            result.end_time = datetime.utcnow()
            logger.error(f"Test {scenario.name} failed with error: {e}")
            
        finally:
            with self._lock:
                self._running_tests.discard(test_id)
        
        # Store result
        self.database.store_test_result(result)
        
        return result
    
    async def _check_prerequisites(self, prerequisites: List[str]) -> bool:
        """Check test prerequisites."""
        if not prerequisites:
            return True
        
        for prereq in prerequisites:
            if prereq == "system_healthy":
                # Check if system is healthy
                summary = self.rto_monitor.get_rto_summary(1)
                for component, data in summary.items():
                    if data.get("breach_count", 0) > 0:
                        return False
            elif prereq == "no_active_tests":
                # Check if no other tests are running
                with self._lock:
                    if len(self._running_tests) > 0:
                        return False
        
        return True
    
    async def execute_test_suite(self, test_suite: TestSuite) -> Dict[str, Any]:
        """Execute a test suite."""
        logger.info(f"Starting test suite: {test_suite.name}")
        
        start_time = datetime.utcnow()
        results = []
        
        if test_suite.parallel_execution:
            # Execute tests in parallel
            semaphore = asyncio.Semaphore(test_suite.max_concurrent_tests)
            
            async def execute_with_semaphore(scenario):
                async with semaphore:
                    return await self.execute_test(scenario)
            
            tasks = [execute_with_semaphore(scenario) for scenario in test_suite.scenarios if scenario.enabled]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_result = TestResult(
                        test_id=f"error_{i}_{int(time.time())}",
                        scenario_name=test_suite.scenarios[i].name,
                        status=TestStatus.ERROR,
                        start_time=start_time,
                        end_time=datetime.utcnow(),
                        error_message=str(result)
                    )
                    results[i] = error_result
        else:
            # Execute tests sequentially
            for scenario in test_suite.scenarios:
                if scenario.enabled:
                    result = await self.execute_test(scenario)
                    results.append(result)
        
        end_time = datetime.utcnow()
        
        # Calculate summary
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = sum(1 for r in results if not r.passed)
        total_tests = len(results)
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "suite_name": test_suite.name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "results": [r.to_dict() for r in results]
        }
        
        logger.info(f"Test suite {test_suite.name} completed: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        
        return summary

class RTOValidationFramework:
    """Comprehensive RTO validation framework."""
    
    def __init__(self, rto_monitor: RTOMonitoringSystem, analytics: RTOAnalyticsSystem):
        self.rto_monitor = rto_monitor
        self.analytics = analytics
        self.executor = RTOTestExecutor(rto_monitor)
        self.test_suites = {}
        self._scheduler_task = None
        self._load_default_scenarios()
    
    def _load_default_scenarios(self):
        """Load default test scenarios."""
        # Database scenarios
        db_scenarios = [
            TestScenario(
                name="db_connection_loss",
                test_type=TestType.FUNCTIONAL,
                component="database",
                description="Test database recovery from connection loss",
                failure_scenario="connection_loss",
                expected_rto=25.0,
                tolerance_percentage=20.0
            ),
            TestScenario(
                name="db_primary_failure",
                test_type=TestType.STRESS,
                component="database",
                description="Test database failover from primary failure",
                failure_scenario="primary_failure",
                expected_rto=30.0,
                tolerance_percentage=15.0
            ),
            TestScenario(
                name="db_disk_full",
                test_type=TestType.CHAOS,
                component="database",
                description="Test database recovery from disk full scenario",
                failure_scenario="disk_full",
                expected_rto=20.0,
                tolerance_percentage=25.0
            )
        ]
        
        # Trading engine scenarios
        engine_scenarios = [
            TestScenario(
                name="engine_service_crash",
                test_type=TestType.FUNCTIONAL,
                component="trading_engine",
                description="Test trading engine recovery from service crash",
                failure_scenario="service_crash",
                expected_rto=3.0,
                tolerance_percentage=20.0
            ),
            TestScenario(
                name="engine_memory_leak",
                test_type=TestType.STRESS,
                component="trading_engine",
                description="Test trading engine recovery from memory leak",
                failure_scenario="memory_leak",
                expected_rto=4.0,
                tolerance_percentage=25.0
            ),
            TestScenario(
                name="engine_config_error",
                test_type=TestType.CHAOS,
                component="trading_engine",
                description="Test trading engine recovery from config error",
                failure_scenario="config_error",
                expected_rto=2.5,
                tolerance_percentage=20.0
            )
        ]
        
        # Create test suites
        self.test_suites["database_validation"] = TestSuite(
            name="database_validation",
            description="Database RTO validation suite",
            scenarios=db_scenarios,
            schedule="0 2 * * *",  # Daily at 2 AM
            parallel_execution=False
        )
        
        self.test_suites["trading_engine_validation"] = TestSuite(
            name="trading_engine_validation",
            description="Trading engine RTO validation suite",
            scenarios=engine_scenarios,
            schedule="0 3 * * *",  # Daily at 3 AM
            parallel_execution=False
        )
        
        self.test_suites["smoke_tests"] = TestSuite(
            name="smoke_tests",
            description="Quick smoke tests for RTO validation",
            scenarios=[
                TestScenario(
                    name="db_smoke_test",
                    test_type=TestType.SMOKE,
                    component="database",
                    description="Quick database RTO smoke test",
                    failure_scenario="connection_loss",
                    expected_rto=25.0,
                    tolerance_percentage=30.0
                ),
                TestScenario(
                    name="engine_smoke_test",
                    test_type=TestType.SMOKE,
                    component="trading_engine",
                    description="Quick trading engine RTO smoke test",
                    failure_scenario="service_crash",
                    expected_rto=3.0,
                    tolerance_percentage=30.0
                )
            ],
            schedule="0 * * * *",  # Hourly
            parallel_execution=True,
            max_concurrent_tests=2
        )
    
    async def run_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests for quick validation."""
        return await self.executor.execute_test_suite(self.test_suites["smoke_tests"])
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run full RTO validation."""
        results = {}
        
        for suite_name, test_suite in self.test_suites.items():
            if suite_name != "smoke_tests":
                logger.info(f"Running validation suite: {suite_name}")
                results[suite_name] = await self.executor.execute_test_suite(test_suite)
        
        return results
    
    async def run_load_tests(self, component: str, concurrent_failures: int = 3) -> Dict[str, Any]:
        """Run load tests with concurrent failures."""
        logger.info(f"Running load tests for {component} with {concurrent_failures} concurrent failures")
        
        # Create load test scenarios
        load_scenarios = []
        base_scenario = self.test_suites[f"{component}_validation"].scenarios[0]
        
        for i in range(concurrent_failures):
            scenario = TestScenario(
                name=f"{component}_load_test_{i+1}",
                test_type=TestType.LOAD,
                component=component,
                description=f"Load test {i+1} for {component}",
                failure_scenario=base_scenario.failure_scenario,
                expected_rto=base_scenario.expected_rto * 1.5,  # Allow 50% more time under load
                tolerance_percentage=40.0
            )
            load_scenarios.append(scenario)
        
        load_suite = TestSuite(
            name=f"{component}_load_test",
            description=f"Load testing suite for {component}",
            scenarios=load_scenarios,
            parallel_execution=True,
            max_concurrent_tests=concurrent_failures
        )
        
        return await self.executor.execute_test_suite(load_suite)
    
    async def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression tests to detect performance degradation."""
        logger.info("Running regression tests")
        
        # Get historical performance data
        components = ["database", "trading_engine"]
        regression_results = {}
        
        for component in components:
            # Get recent test results
            recent_results = []
            for suite_name, suite in self.test_suites.items():
                if component in suite_name:
                    for scenario in suite.scenarios:
                        history = self.executor.database.get_test_history(scenario.name, 30)
                        recent_results.extend(history)
            
            if len(recent_results) < 10:
                continue
            
            # Calculate baseline performance
            baseline_rto = statistics.mean([r.actual_rto for r in recent_results[-10:] if r.actual_rto])
            
            # Run current test
            test_scenario = next(
                (s for s in self.test_suites[f"{component}_validation"].scenarios), 
                None
            )
            
            if test_scenario:
                current_result = await self.executor.execute_test(test_scenario)
                
                # Check for regression
                if current_result.actual_rto and current_result.actual_rto > baseline_rto * 1.2:
                    regression_detected = True
                    regression_percentage = ((current_result.actual_rto - baseline_rto) / baseline_rto) * 100
                else:
                    regression_detected = False
                    regression_percentage = 0
                
                regression_results[component] = {
                    "baseline_rto": baseline_rto,
                    "current_rto": current_result.actual_rto,
                    "regression_detected": regression_detected,
                    "regression_percentage": regression_percentage,
                    "test_result": current_result.to_dict()
                }
        
        return regression_results
    
    async def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for RTO validation."""
        logger.info("Generating compliance report")
        
        # Get test summary
        test_summary = self.executor.database.get_test_summary(30)
        
        # Get analytics
        components = ["database", "trading_engine"]
        analytics_results = {}
        
        for component in components:
            analytics_results[component] = self.analytics.get_comprehensive_analysis(component, 30)
        
        # Calculate compliance metrics
        compliance_metrics = {}
        
        for component in components:
            target_rto = RTOTarget.DATABASE.value if component == "database" else RTOTarget.TRADING_ENGINE.value
            
            # Get recent test results
            recent_tests = []
            for suite_name, suite in self.test_suites.items():
                if component in suite_name:
                    for scenario in suite.scenarios:
                        history = self.executor.database.get_test_history(scenario.name, 30)
                        recent_tests.extend(history)
            
            if recent_tests:
                # Calculate compliance rate
                compliant_tests = sum(1 for t in recent_tests if t.actual_rto and t.actual_rto <= target_rto)
                compliance_rate = (compliant_tests / len(recent_tests)) * 100
                
                # Average RTO
                avg_rto = statistics.mean([t.actual_rto for t in recent_tests if t.actual_rto])
                
                # Breach frequency
                breach_count = sum(1 for t in recent_tests if t.actual_rto and t.actual_rto > target_rto)
                breach_frequency = breach_count / len(recent_tests) if recent_tests else 0
                
                compliance_metrics[component] = {
                    "target_rto": target_rto,
                    "total_tests": len(recent_tests),
                    "compliant_tests": compliant_tests,
                    "compliance_rate": compliance_rate,
                    "average_rto": avg_rto,
                    "breach_count": breach_count,
                    "breach_frequency": breach_frequency,
                    "status": "compliant" if compliance_rate >= 95 else "non_compliant"
                }
            else:
                compliance_metrics[component] = {
                    "target_rto": target_rto,
                    "total_tests": 0,
                    "status": "no_data"
                }
        
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "reporting_period": "30 days",
            "test_summary": test_summary,
            "compliance_metrics": compliance_metrics,
            "analytics_results": analytics_results,
            "overall_compliance": self._calculate_overall_compliance(compliance_metrics)
        }
    
    def _calculate_overall_compliance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall compliance status."""
        compliant_components = sum(1 for m in metrics.values() if m.get("status") == "compliant")
        total_components = len(metrics)
        
        if total_components == 0:
            return {"status": "no_data", "percentage": 0}
        
        compliance_percentage = (compliant_components / total_components) * 100
        
        if compliance_percentage >= 95:
            status = "fully_compliant"
        elif compliance_percentage >= 80:
            status = "mostly_compliant"
        elif compliance_percentage >= 50:
            status = "partially_compliant"
        else:
            status = "non_compliant"
        
        return {
            "status": status,
            "percentage": compliance_percentage,
            "compliant_components": compliant_components,
            "total_components": total_components
        }
    
    def add_custom_scenario(self, scenario: TestScenario, suite_name: str = "custom"):
        """Add custom test scenario."""
        if suite_name not in self.test_suites:
            self.test_suites[suite_name] = TestSuite(
                name=suite_name,
                description="Custom test scenarios",
                scenarios=[],
                enabled=True
            )
        
        self.test_suites[suite_name].scenarios.append(scenario)
        logger.info(f"Added custom scenario {scenario.name} to suite {suite_name}")
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation status."""
        return {
            "test_suites": {name: suite.to_dict() for name, suite in self.test_suites.items()},
            "recent_summary": self.executor.database.get_test_summary(7),
            "system_health": self.rto_monitor.get_rto_summary(1)
        }

# Global validation framework
validation_framework = None

def initialize_validation_framework(rto_monitor: RTOMonitoringSystem, analytics: RTOAnalyticsSystem) -> RTOValidationFramework:
    """Initialize global validation framework."""
    global validation_framework
    validation_framework = RTOValidationFramework(rto_monitor, analytics)
    return validation_framework

def get_validation_framework() -> Optional[RTOValidationFramework]:
    """Get global validation framework instance."""
    return validation_framework