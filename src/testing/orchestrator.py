"""
GrandModel Test Orchestrator
============================

Central orchestration system for coordinating all testing activities across
the GrandModel trading system, including test scheduling, execution coordination,
and comprehensive reporting.
"""

import asyncio
import logging
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule

from .framework import TestingFramework, TestType, TestStatus
from .validation import ValidationSuite, ValidationType
from .performance import PerformanceTester, PerformanceTestType
from .quality import QualityAssurance, QualityLevel
from .ci_cd import CIPipeline, PipelineStage


class OrchestratorMode(Enum):
    """Orchestrator execution mode"""
    DEVELOPMENT = "development"
    CONTINUOUS_INTEGRATION = "continuous_integration"
    NIGHTLY_BUILD = "nightly_build"
    RELEASE_CANDIDATE = "release_candidate"
    PRODUCTION_MONITORING = "production_monitoring"


class TestPriority(Enum):
    """Test priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TestPlan:
    """Test execution plan"""
    name: str
    mode: OrchestratorMode
    priority: TestPriority
    test_suites: List[str]
    validation_types: List[ValidationType]
    performance_tests: List[str]
    quality_checks: bool
    ci_cd_pipeline: bool
    schedule_cron: Optional[str] = None
    timeout_minutes: int = 60
    retry_count: int = 1
    parallel_execution: bool = True
    notify_on_failure: bool = True
    
    def __post_init__(self):
        if not self.test_suites:
            self.test_suites = []
        if not self.validation_types:
            self.validation_types = []
        if not self.performance_tests:
            self.performance_tests = []


@dataclass
class OrchestrationResult:
    """Orchestration execution result"""
    test_plan_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    overall_status: TestStatus
    test_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    performance_results: Dict[str, Any]
    quality_results: Dict[str, Any]
    ci_cd_results: Dict[str, Any]
    summary: Dict[str, Any]
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.test_results:
            self.test_results = {}
        if not self.validation_results:
            self.validation_results = {}
        if not self.performance_results:
            self.performance_results = {}
        if not self.quality_results:
            self.quality_results = {}
        if not self.ci_cd_results:
            self.ci_cd_results = {}


class TestOrchestrator:
    """
    Central test orchestration system
    
    Features:
    - Comprehensive test coordination
    - Multiple execution modes
    - Scheduled test execution
    - Parallel test execution
    - Integrated reporting
    - CI/CD pipeline integration
    - Quality gates enforcement
    - Performance monitoring
    - Failure handling and recovery
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/orchestrator_config.yaml"
        self.logger = self._setup_logging()
        self.test_plans: Dict[str, TestPlan] = {}
        self.execution_history: List[OrchestrationResult] = []
        self.scheduled_jobs = []
        
        # Initialize components
        self.testing_framework = TestingFramework()
        self.validation_suite = ValidationSuite()
        self.performance_tester = PerformanceTester()
        self.quality_assurance = QualityAssurance()
        self.ci_cd_pipeline = CIPipeline()
        
        # Initialize paths
        self.project_root = Path(__file__).parent.parent.parent
        self.results_dir = self.project_root / "orchestration_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize default test plans
        self._initialize_default_test_plans()
        
        # Start scheduler
        self.scheduler_running = False
        self.scheduler_thread = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("TestOrchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_default_test_plans(self):
        """Initialize default test plans"""
        # Development mode test plan
        self.test_plans["development"] = TestPlan(
            name="development",
            mode=OrchestratorMode.DEVELOPMENT,
            priority=TestPriority.NORMAL,
            test_suites=["unit_tests", "integration_tests"],
            validation_types=[ValidationType.MATHEMATICAL, ValidationType.STATISTICAL],
            performance_tests=["latency_tests"],
            quality_checks=True,
            ci_cd_pipeline=False,
            timeout_minutes=30,
            parallel_execution=True
        )
        
        # CI mode test plan
        self.test_plans["continuous_integration"] = TestPlan(
            name="continuous_integration",
            mode=OrchestratorMode.CONTINUOUS_INTEGRATION,
            priority=TestPriority.HIGH,
            test_suites=["unit_tests", "integration_tests", "security_tests"],
            validation_types=[ValidationType.MATHEMATICAL, ValidationType.STATISTICAL, ValidationType.MODEL_VALIDATION],
            performance_tests=["latency_tests", "throughput_tests"],
            quality_checks=True,
            ci_cd_pipeline=True,
            timeout_minutes=60,
            parallel_execution=True,
            notify_on_failure=True
        )
        
        # Nightly build test plan
        self.test_plans["nightly_build"] = TestPlan(
            name="nightly_build",
            mode=OrchestratorMode.NIGHTLY_BUILD,
            priority=TestPriority.HIGH,
            test_suites=["unit_tests", "integration_tests", "performance_tests", "security_tests"],
            validation_types=[ValidationType.MATHEMATICAL, ValidationType.STATISTICAL, 
                            ValidationType.BACKTESTING, ValidationType.STRESS_TESTING,
                            ValidationType.MODEL_VALIDATION],
            performance_tests=["latency_tests", "throughput_tests", "load_tests", "scalability_tests"],
            quality_checks=True,
            ci_cd_pipeline=True,
            schedule_cron="0 2 * * *",  # 2 AM daily
            timeout_minutes=180,
            parallel_execution=True,
            notify_on_failure=True
        )
        
        # Release candidate test plan
        self.test_plans["release_candidate"] = TestPlan(
            name="release_candidate",
            mode=OrchestratorMode.RELEASE_CANDIDATE,
            priority=TestPriority.CRITICAL,
            test_suites=["unit_tests", "integration_tests", "performance_tests", "security_tests", "regression_tests"],
            validation_types=[ValidationType.MATHEMATICAL, ValidationType.STATISTICAL, 
                            ValidationType.BACKTESTING, ValidationType.STRESS_TESTING,
                            ValidationType.MODEL_VALIDATION, ValidationType.PERFORMANCE,
                            ValidationType.RISK_VALIDATION],
            performance_tests=["latency_tests", "throughput_tests", "load_tests", "scalability_tests", "endurance_tests"],
            quality_checks=True,
            ci_cd_pipeline=True,
            timeout_minutes=360,
            parallel_execution=True,
            retry_count=2,
            notify_on_failure=True
        )
        
        # Production monitoring test plan
        self.test_plans["production_monitoring"] = TestPlan(
            name="production_monitoring",
            mode=OrchestratorMode.PRODUCTION_MONITORING,
            priority=TestPriority.CRITICAL,
            test_suites=["smoke_tests", "performance_tests"],
            validation_types=[ValidationType.PERFORMANCE, ValidationType.RISK_VALIDATION],
            performance_tests=["latency_tests", "throughput_tests"],
            quality_checks=False,
            ci_cd_pipeline=False,
            schedule_cron="*/15 * * * *",  # Every 15 minutes
            timeout_minutes=10,
            parallel_execution=True,
            notify_on_failure=True
        )
    
    def register_test_plan(self, test_plan: TestPlan):
        """Register a test plan"""
        self.test_plans[test_plan.name] = test_plan
        self.logger.info(f"Registered test plan: {test_plan.name}")
    
    async def execute_test_plan(self, plan_name: str, branch: str = "main") -> OrchestrationResult:
        """
        Execute a test plan
        
        Args:
            plan_name: Name of the test plan to execute
            branch: Git branch to test
            
        Returns:
            OrchestrationResult object
        """
        if plan_name not in self.test_plans:
            raise ValueError(f"Test plan '{plan_name}' not found")
        
        test_plan = self.test_plans[plan_name]
        self.logger.info(f"Executing test plan: {plan_name}")
        
        start_time = datetime.now()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_test_plan_internal(test_plan, branch),
                timeout=test_plan.timeout_minutes * 60
            )
            
            result.start_time = start_time
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
            # Store result
            self.execution_history.append(result)
            
            # Generate comprehensive report
            report_file = self._generate_orchestration_report(result)
            
            # Send notifications if configured
            if test_plan.notify_on_failure and result.overall_status == TestStatus.FAILED:
                await self._send_failure_notifications(result, report_file)
            
            self.logger.info(f"Test plan '{plan_name}' completed with status: {result.overall_status.value}")
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Test plan '{plan_name}' timed out after {test_plan.timeout_minutes} minutes")
            
            result = OrchestrationResult(
                test_plan_name=plan_name,
                start_time=start_time,
                end_time=datetime.now(),
                duration=test_plan.timeout_minutes * 60,
                overall_status=TestStatus.ERROR,
                test_results={},
                validation_results={},
                performance_results={},
                quality_results={},
                ci_cd_results={},
                summary={"error": "Test plan execution timed out"},
                error_message=f"Execution timed out after {test_plan.timeout_minutes} minutes"
            )
            
            self.execution_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Test plan '{plan_name}' failed: {e}")
            
            result = OrchestrationResult(
                test_plan_name=plan_name,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                overall_status=TestStatus.ERROR,
                test_results={},
                validation_results={},
                performance_results={},
                quality_results={},
                ci_cd_results={},
                summary={"error": str(e)},
                error_message=str(e)
            )
            
            self.execution_history.append(result)
            return result
    
    async def _execute_test_plan_internal(self, test_plan: TestPlan, branch: str) -> OrchestrationResult:
        """Internal test plan execution"""
        result = OrchestrationResult(
            test_plan_name=test_plan.name,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=0,
            overall_status=TestStatus.PENDING,
            test_results={},
            validation_results={},
            performance_results={},
            quality_results={},
            ci_cd_results={},
            summary={}
        )
        
        execution_tasks = []
        
        # Prepare execution tasks
        if test_plan.parallel_execution:
            # Execute in parallel
            if test_plan.test_suites:
                execution_tasks.append(self._execute_test_suites(test_plan.test_suites))
            
            if test_plan.validation_types:
                execution_tasks.append(self._execute_validations(test_plan.validation_types))
            
            if test_plan.performance_tests:
                execution_tasks.append(self._execute_performance_tests(test_plan.performance_tests))
            
            if test_plan.quality_checks:
                execution_tasks.append(self._execute_quality_checks())
            
            if test_plan.ci_cd_pipeline:
                execution_tasks.append(self._execute_ci_cd_pipeline(branch))
            
            # Execute all tasks in parallel
            if execution_tasks:
                results = await asyncio.gather(*execution_tasks, return_exceptions=True)
                
                # Process results
                for i, task_result in enumerate(results):
                    if isinstance(task_result, Exception):
                        self.logger.error(f"Task {i} failed: {task_result}")
                        result.overall_status = TestStatus.ERROR
                    else:
                        # Merge results based on task type
                        if i == 0 and test_plan.test_suites:  # Test suites
                            result.test_results = task_result
                        elif (i == 1 and test_plan.validation_types) or (i == 0 and not test_plan.test_suites and test_plan.validation_types):  # Validations
                            result.validation_results = task_result
                        # Continue for other task types...
        else:
            # Execute sequentially
            if test_plan.test_suites:
                result.test_results = await self._execute_test_suites(test_plan.test_suites)
            
            if test_plan.validation_types:
                result.validation_results = await self._execute_validations(test_plan.validation_types)
            
            if test_plan.performance_tests:
                result.performance_results = await self._execute_performance_tests(test_plan.performance_tests)
            
            if test_plan.quality_checks:
                result.quality_results = await self._execute_quality_checks()
            
            if test_plan.ci_cd_pipeline:
                result.ci_cd_results = await self._execute_ci_cd_pipeline(branch)
        
        # Determine overall status
        result.overall_status = self._determine_overall_status(result)
        
        # Generate summary
        result.summary = self._generate_execution_summary(result)
        
        return result
    
    async def _execute_test_suites(self, test_suites: List[str]) -> Dict[str, Any]:
        """Execute test suites"""
        results = {}
        
        for suite_name in test_suites:
            try:
                self.logger.info(f"Executing test suite: {suite_name}")
                
                # Map suite names to test types
                test_type_mapping = {
                    "unit_tests": TestType.UNIT,
                    "integration_tests": TestType.INTEGRATION,
                    "performance_tests": TestType.PERFORMANCE,
                    "security_tests": TestType.SECURITY,
                    "smoke_tests": TestType.SMOKE,
                    "regression_tests": TestType.REGRESSION
                }
                
                test_type = test_type_mapping.get(suite_name, TestType.UNIT)
                
                # Run tests
                test_results = await self.testing_framework.run_all_tests(test_type)
                
                # Get summary
                summary = self.testing_framework.get_test_summary()
                
                results[suite_name] = {
                    "status": "success" if summary.get("failed", 0) == 0 else "failed",
                    "summary": summary,
                    "results": [
                        {
                            "test_id": r.test_id,
                            "test_name": r.test_name,
                            "status": r.status.value,
                            "duration": r.duration,
                            "error_message": r.error_message
                        }
                        for r in test_results
                    ]
                }
                
            except Exception as e:
                self.logger.error(f"Test suite '{suite_name}' failed: {e}")
                results[suite_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    async def _execute_validations(self, validation_types: List[ValidationType]) -> Dict[str, Any]:
        """Execute validations"""
        results = {}
        
        for validation_type in validation_types:
            try:
                self.logger.info(f"Executing validation: {validation_type.value}")
                
                # Run validations of specific type
                validation_results = await self.validation_suite.run_validation_by_type(validation_type)
                
                # Get summary
                summary = self.validation_suite.get_validation_summary()
                
                results[validation_type.value] = {
                    "status": "success" if summary.get("failed", 0) == 0 else "failed",
                    "summary": summary,
                    "results": [
                        {
                            "validation_id": r.validation_id,
                            "validation_name": r.validation_name,
                            "status": r.status.value,
                            "score": r.score,
                            "threshold": r.threshold,
                            "duration": r.duration,
                            "error_message": r.error_message
                        }
                        for r in validation_results
                    ]
                }
                
            except Exception as e:
                self.logger.error(f"Validation '{validation_type.value}' failed: {e}")
                results[validation_type.value] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    async def _execute_performance_tests(self, performance_tests: List[str]) -> Dict[str, Any]:
        """Execute performance tests"""
        results = {}
        
        for test_name in performance_tests:
            try:
                self.logger.info(f"Executing performance test: {test_name}")
                
                # Sample performance tests
                if test_name == "latency_tests":
                    result = await self.performance_tester.test_latency(
                        "sample_latency", 
                        lambda: time.sleep(0.001), 
                        iterations=100
                    )
                elif test_name == "throughput_tests":
                    result = await self.performance_tester.test_throughput(
                        "sample_throughput",
                        lambda: time.sleep(0.001),
                        duration_seconds=10
                    )
                elif test_name == "load_tests":
                    from .performance import LoadTestConfig
                    config = LoadTestConfig(
                        name="sample_load_test",
                        target_function=lambda: time.sleep(0.001),
                        concurrent_users=10,
                        duration_seconds=30
                    )
                    result = await self.performance_tester.test_load(config)
                elif test_name == "scalability_tests":
                    result = await self.performance_tester.test_scalability(
                        "sample_scalability",
                        lambda: time.sleep(0.001),
                        [1, 5, 10, 20]
                    )
                else:
                    # Default latency test
                    result = await self.performance_tester.test_latency(
                        test_name,
                        lambda: time.sleep(0.001),
                        iterations=100
                    )
                
                results[test_name] = {
                    "status": "success" if result.success else "failed",
                    "test_id": result.test_id,
                    "test_type": result.test_type.value,
                    "metrics": {k.value: v for k, v in result.metrics.items()},
                    "duration": result.duration,
                    "error_message": result.error_message
                }
                
            except Exception as e:
                self.logger.error(f"Performance test '{test_name}' failed: {e}")
                results[test_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    async def _execute_quality_checks(self) -> Dict[str, Any]:
        """Execute quality checks"""
        try:
            self.logger.info("Executing quality checks")
            
            # Run quality analysis
            quality_report = await self.quality_assurance.analyze_code_quality()
            
            # Get summary
            summary = self.quality_assurance.get_quality_summary()
            
            return {
                "status": "success" if summary.get("quality_level") not in ["poor", "critical"] else "failed",
                "summary": summary,
                "metrics": {
                    "quality_level": quality_report.metrics.quality_level.value,
                    "quality_score": quality_report.metrics.quality_score,
                    "issues_count": quality_report.metrics.issues_count,
                    "bugs_count": quality_report.metrics.bugs_count,
                    "vulnerabilities_count": quality_report.metrics.vulnerabilities_count,
                    "complexity": quality_report.metrics.complexity,
                    "maintainability": quality_report.metrics.maintainability_index,
                    "test_coverage": quality_report.metrics.test_coverage
                },
                "recommendations": quality_report.recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Quality checks failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _execute_ci_cd_pipeline(self, branch: str) -> Dict[str, Any]:
        """Execute CI/CD pipeline"""
        try:
            self.logger.info("Executing CI/CD pipeline")
            
            # Run CI/CD pipeline
            pipeline_results = await self.ci_cd_pipeline.run_pipeline(branch)
            
            # Get summary
            summary = self.ci_cd_pipeline.get_pipeline_summary()
            
            return {
                "status": "success" if summary.get("overall_status") == "SUCCESS" else "failed",
                "summary": summary,
                "results": [
                    {
                        "stage": r.stage.value,
                        "status": r.status.value,
                        "duration": r.duration,
                        "error_message": r.error_message
                    }
                    for r in pipeline_results
                ]
            }
            
        except Exception as e:
            self.logger.error(f"CI/CD pipeline failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _determine_overall_status(self, result: OrchestrationResult) -> TestStatus:
        """Determine overall execution status"""
        # Check for errors
        if (result.test_results and any(r.get("status") == "error" for r in result.test_results.values()) or
            result.validation_results and any(r.get("status") == "error" for r in result.validation_results.values()) or
            result.performance_results and any(r.get("status") == "error" for r in result.performance_results.values()) or
            result.quality_results.get("status") == "error" or
            result.ci_cd_results.get("status") == "error"):
            return TestStatus.ERROR
        
        # Check for failures
        if (result.test_results and any(r.get("status") == "failed" for r in result.test_results.values()) or
            result.validation_results and any(r.get("status") == "failed" for r in result.validation_results.values()) or
            result.performance_results and any(r.get("status") == "failed" for r in result.performance_results.values()) or
            result.quality_results.get("status") == "failed" or
            result.ci_cd_results.get("status") == "failed"):
            return TestStatus.FAILED
        
        return TestStatus.PASSED
    
    def _generate_execution_summary(self, result: OrchestrationResult) -> Dict[str, Any]:
        """Generate execution summary"""
        summary = {
            "overall_status": result.overall_status.value,
            "execution_time": result.duration,
            "components_executed": []
        }
        
        if result.test_results:
            summary["components_executed"].append("test_suites")
            summary["test_suites_count"] = len(result.test_results)
            summary["test_suites_passed"] = sum(1 for r in result.test_results.values() if r.get("status") == "success")
        
        if result.validation_results:
            summary["components_executed"].append("validations")
            summary["validations_count"] = len(result.validation_results)
            summary["validations_passed"] = sum(1 for r in result.validation_results.values() if r.get("status") == "success")
        
        if result.performance_results:
            summary["components_executed"].append("performance_tests")
            summary["performance_tests_count"] = len(result.performance_results)
            summary["performance_tests_passed"] = sum(1 for r in result.performance_results.values() if r.get("status") == "success")
        
        if result.quality_results:
            summary["components_executed"].append("quality_checks")
            summary["quality_status"] = result.quality_results.get("status", "unknown")
        
        if result.ci_cd_results:
            summary["components_executed"].append("ci_cd_pipeline")
            summary["ci_cd_status"] = result.ci_cd_results.get("status", "unknown")
        
        return summary
    
    def _generate_orchestration_report(self, result: OrchestrationResult) -> str:
        """Generate orchestration report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"orchestration_report_{timestamp}.json"
        
        report_data = {
            "test_plan_name": result.test_plan_name,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "duration": result.duration,
            "overall_status": result.overall_status.value,
            "test_results": result.test_results,
            "validation_results": result.validation_results,
            "performance_results": result.performance_results,
            "quality_results": result.quality_results,
            "ci_cd_results": result.ci_cd_results,
            "summary": result.summary,
            "error_message": result.error_message
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Orchestration report generated: {report_file}")
        return str(report_file)
    
    async def _send_failure_notifications(self, result: OrchestrationResult, report_file: str):
        """Send failure notifications"""
        # Placeholder for notification system
        self.logger.warning(f"Test plan '{result.test_plan_name}' failed - notifications would be sent")
        self.logger.warning(f"Report available at: {report_file}")
    
    def start_scheduler(self):
        """Start the test scheduler"""
        self.scheduler_running = True
        
        def run_scheduler():
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        # Schedule test plans with cron expressions
        for plan_name, test_plan in self.test_plans.items():
            if test_plan.schedule_cron:
                # Simple cron parsing (would need proper cron library for production)
                if test_plan.schedule_cron == "0 2 * * *":  # Daily at 2 AM
                    schedule.every().day.at("02:00").do(self._scheduled_execution, plan_name)
                elif test_plan.schedule_cron == "*/15 * * * *":  # Every 15 minutes
                    schedule.every(15).minutes.do(self._scheduled_execution, plan_name)
        
        # Start scheduler thread
        import threading
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Test scheduler started")
    
    def stop_scheduler(self):
        """Stop the test scheduler"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Test scheduler stopped")
    
    def _scheduled_execution(self, plan_name: str):
        """Execute scheduled test plan"""
        self.logger.info(f"Executing scheduled test plan: {plan_name}")
        
        # Run in new event loop (since we're in a thread)
        import asyncio
        asyncio.run(self.execute_test_plan(plan_name))
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution history"""
        history = []
        
        for result in self.execution_history[-limit:]:
            history.append({
                "test_plan_name": result.test_plan_name,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "duration": result.duration,
                "overall_status": result.overall_status.value,
                "summary": result.summary
            })
        
        return history
    
    def get_orchestration_summary(self) -> Dict[str, Any]:
        """Get orchestration summary"""
        if not self.execution_history:
            return {}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for r in self.execution_history if r.overall_status == TestStatus.PASSED)
        failed_executions = sum(1 for r in self.execution_history if r.overall_status == TestStatus.FAILED)
        error_executions = sum(1 for r in self.execution_history if r.overall_status == TestStatus.ERROR)
        
        avg_duration = sum(r.duration for r in self.execution_history) / total_executions
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "error_executions": error_executions,
            "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
            "average_duration": avg_duration,
            "test_plans_configured": len(self.test_plans),
            "scheduler_running": self.scheduler_running
        }


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize test orchestrator
        orchestrator = TestOrchestrator()
        
        # Execute development test plan
        result = await orchestrator.execute_test_plan("development")
        
        # Print summary
        summary = orchestrator.get_orchestration_summary()
        print(f"\nOrchestration Summary:")
        print(f"Total Executions: {summary.get('total_executions', 0)}")
        print(f"Successful Executions: {summary.get('successful_executions', 0)}")
        print(f"Failed Executions: {summary.get('failed_executions', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"Average Duration: {summary.get('average_duration', 0):.2f}s")
        print(f"Test Plans Configured: {summary.get('test_plans_configured', 0)}")
        
        # Show latest execution
        print(f"\nLatest Execution:")
        print(f"Test Plan: {result.test_plan_name}")
        print(f"Overall Status: {result.overall_status.value}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Summary: {result.summary}")
    
    asyncio.run(main())