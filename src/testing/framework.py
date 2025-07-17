"""
GrandModel Testing Framework
============================

Core testing framework providing automated testing capabilities for the GrandModel
trading system including unit tests, integration tests, and end-to-end scenarios.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import yaml
from datetime import datetime, timedelta

import pytest
import coverage
import torch
import numpy as np
import pandas as pd


class TestType(Enum):
    """Test type enumeration"""
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STRESS = "stress"
    SMOKE = "smoke"
    REGRESSION = "regression"


class TestStatus(Enum):
    """Test status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Test result data structure"""
    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = None
    assertions: int = 0
    coverage: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    test_type: TestType
    test_patterns: List[str]
    setup_functions: List[Callable] = None
    teardown_functions: List[Callable] = None
    timeout: int = 300
    parallel: bool = True
    coverage_threshold: float = 80.0
    retry_count: int = 0
    
    def __post_init__(self):
        if self.setup_functions is None:
            self.setup_functions = []
        if self.teardown_functions is None:
            self.teardown_functions = []


class TestingFramework:
    """
    Comprehensive testing framework for GrandModel system
    
    Features:
    - Automated test discovery and execution
    - Unit, integration, and end-to-end testing
    - Performance benchmarking
    - Coverage analysis
    - Parallel test execution
    - Real-time reporting
    - CI/CD integration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/testing_config.yaml"
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.test_results: List[TestResult] = []
        self.test_suites: Dict[str, TestSuite] = {}
        self.coverage_data = None
        self.start_time = None
        self.end_time = None
        
        # Initialize core components
        self.project_root = Path(__file__).parent.parent.parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Setup coverage
        self.coverage = coverage.Coverage(
            source=[str(self.project_root / "src")],
            omit=[
                "*/tests/*",
                "*/venv/*",
                "*/__pycache__/*",
                "*/setup.py"
            ]
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load testing configuration"""
        default_config = {
            "test_discovery": {
                "patterns": ["test_*.py", "*_test.py"],
                "exclude_patterns": ["__pycache__", "*.pyc"]
            },
            "execution": {
                "parallel": True,
                "max_workers": 4,
                "timeout": 300,
                "retry_count": 0
            },
            "coverage": {
                "threshold": 80.0,
                "fail_under": 70.0,
                "show_missing": True
            },
            "reporting": {
                "formats": ["html", "json", "junit"],
                "include_coverage": True,
                "include_performance": True
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    self._deep_merge(default_config, user_config)
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("TestingFramework")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def register_test_suite(self, suite: TestSuite) -> None:
        """Register a test suite"""
        self.test_suites[suite.name] = suite
        self.logger.info(f"Registered test suite: {suite.name}")
    
    def discover_tests(self, test_type: Optional[TestType] = None) -> List[str]:
        """
        Discover tests based on patterns and type
        
        Args:
            test_type: Optional test type filter
            
        Returns:
            List of test file paths
        """
        test_files = []
        patterns = self.config["test_discovery"]["patterns"]
        exclude_patterns = self.config["test_discovery"]["exclude_patterns"]
        
        # Map test types to directory patterns
        type_mapping = {
            TestType.UNIT: ["tests/unit", "tests/*/test_*.py"],
            TestType.INTEGRATION: ["tests/integration", "tests/*/integration_*.py"],
            TestType.END_TO_END: ["tests/e2e", "tests/*/e2e_*.py"],
            TestType.PERFORMANCE: ["tests/performance", "tests/*/perf_*.py"],
            TestType.SECURITY: ["tests/security", "tests/*/security_*.py"],
            TestType.STRESS: ["tests/stress", "tests/*/stress_*.py"],
            TestType.SMOKE: ["tests/smoke", "tests/*/smoke_*.py"],
            TestType.REGRESSION: ["tests/regression", "tests/*/regression_*.py"]
        }
        
        search_patterns = []
        if test_type and test_type in type_mapping:
            search_patterns = type_mapping[test_type]
        else:
            search_patterns = patterns
        
        for pattern in search_patterns:
            if pattern.startswith("tests/"):
                # Directory-based search
                test_path = self.test_dir / pattern.replace("tests/", "")
                if test_path.is_dir():
                    for file_pattern in patterns:
                        test_files.extend(test_path.glob(file_pattern))
            else:
                # Pattern-based search
                test_files.extend(self.test_dir.rglob(pattern))
        
        # Filter out excluded patterns
        filtered_files = []
        for test_file in test_files:
            if not any(exclude in str(test_file) for exclude in exclude_patterns):
                filtered_files.append(str(test_file))
        
        self.logger.info(f"Discovered {len(filtered_files)} test files")
        return filtered_files
    
    async def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """
        Run a specific test suite
        
        Args:
            suite_name: Name of the test suite to run
            
        Returns:
            List of test results
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        suite = self.test_suites[suite_name]
        self.logger.info(f"Running test suite: {suite_name}")
        
        # Run setup functions
        for setup_func in suite.setup_functions:
            try:
                if asyncio.iscoroutinefunction(setup_func):
                    await setup_func()
                else:
                    setup_func()
            except Exception as e:
                self.logger.error(f"Setup function failed: {e}")
                raise
        
        try:
            # Discover tests for this suite
            test_files = []
            for pattern in suite.test_patterns:
                test_files.extend(self.test_dir.rglob(pattern))
            
            # Run tests
            results = await self._run_tests(
                test_files=[str(f) for f in test_files],
                test_type=suite.test_type,
                timeout=suite.timeout,
                parallel=suite.parallel
            )
            
            # Check coverage threshold
            if self.coverage_data:
                coverage_percentage = self.coverage_data.get("totals", {}).get("percent_covered", 0)
                if coverage_percentage < suite.coverage_threshold:
                    self.logger.warning(
                        f"Coverage {coverage_percentage:.1f}% below threshold {suite.coverage_threshold:.1f}%"
                    )
            
            return results
            
        finally:
            # Run teardown functions
            for teardown_func in suite.teardown_functions:
                try:
                    if asyncio.iscoroutinefunction(teardown_func):
                        await teardown_func()
                    else:
                        teardown_func()
                except Exception as e:
                    self.logger.error(f"Teardown function failed: {e}")
    
    async def run_all_tests(self, test_type: Optional[TestType] = None) -> List[TestResult]:
        """
        Run all tests or tests of specific type
        
        Args:
            test_type: Optional test type filter
            
        Returns:
            List of test results
        """
        self.logger.info("Starting comprehensive test execution")
        self.start_time = time.time()
        
        # Start coverage collection
        self.coverage.start()
        
        try:
            # Discover tests
            test_files = self.discover_tests(test_type)
            
            if not test_files:
                self.logger.warning("No test files discovered")
                return []
            
            # Run tests
            results = await self._run_tests(
                test_files=test_files,
                test_type=test_type,
                timeout=self.config["execution"]["timeout"],
                parallel=self.config["execution"]["parallel"]
            )
            
            self.test_results.extend(results)
            return results
            
        finally:
            # Stop coverage collection
            self.coverage.stop()
            self.coverage.save()
            
            # Generate coverage report
            self._generate_coverage_report()
            
            self.end_time = time.time()
            self.logger.info(f"Test execution completed in {self.end_time - self.start_time:.2f} seconds")
    
    async def _run_tests(self, test_files: List[str], test_type: Optional[TestType] = None,
                        timeout: int = 300, parallel: bool = True) -> List[TestResult]:
        """
        Internal method to run tests
        
        Args:
            test_files: List of test file paths
            test_type: Optional test type
            timeout: Test timeout in seconds
            parallel: Whether to run tests in parallel
            
        Returns:
            List of test results
        """
        results = []
        
        if parallel and len(test_files) > 1:
            # Run tests in parallel
            max_workers = min(self.config["execution"]["max_workers"], len(test_files))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for test_file in test_files:
                    future = executor.submit(self._run_single_test, test_file, test_type, timeout)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.extend(result)
                    except Exception as e:
                        self.logger.error(f"Test execution failed: {e}")
        else:
            # Run tests sequentially
            for test_file in test_files:
                try:
                    result = self._run_single_test(test_file, test_type, timeout)
                    results.extend(result)
                except Exception as e:
                    self.logger.error(f"Test execution failed for {test_file}: {e}")
        
        return results
    
    def _run_single_test(self, test_file: str, test_type: Optional[TestType] = None,
                        timeout: int = 300) -> List[TestResult]:
        """
        Run a single test file
        
        Args:
            test_file: Path to test file
            test_type: Optional test type
            timeout: Test timeout in seconds
            
        Returns:
            List of test results
        """
        self.logger.info(f"Running test file: {test_file}")
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file={self.reports_dir}/pytest_report_{int(time.time())}.json"
        ]
        
        # Add test type markers if specified
        if test_type:
            cmd.extend(["-m", test_type.value])
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Parse results
            return self._parse_pytest_results(result, test_file, test_type)
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Test timeout for {test_file}")
            return [TestResult(
                test_id=f"timeout_{test_file}",
                test_name=test_file,
                test_type=test_type or TestType.UNIT,
                status=TestStatus.ERROR,
                duration=timeout,
                error_message="Test timeout"
            )]
        except Exception as e:
            self.logger.error(f"Error running test {test_file}: {e}")
            return [TestResult(
                test_id=f"error_{test_file}",
                test_name=test_file,
                test_type=test_type or TestType.UNIT,
                status=TestStatus.ERROR,
                duration=0,
                error_message=str(e)
            )]
    
    def _parse_pytest_results(self, result: subprocess.CompletedProcess,
                            test_file: str, test_type: Optional[TestType] = None) -> List[TestResult]:
        """
        Parse pytest results from JSON report
        
        Args:
            result: Subprocess result
            test_file: Test file path
            test_type: Test type
            
        Returns:
            List of test results
        """
        results = []
        
        try:
            # Find the JSON report file
            json_files = list(self.reports_dir.glob("pytest_report_*.json"))
            if not json_files:
                self.logger.warning("No pytest JSON report found")
                return results
            
            # Read the latest report
            latest_report = max(json_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            # Parse test results
            for test in report_data.get("tests", []):
                test_result = TestResult(
                    test_id=test.get("nodeid", ""),
                    test_name=test.get("nodeid", "").split("::")[-1],
                    test_type=test_type or TestType.UNIT,
                    status=TestStatus(test.get("outcome", "failed")),
                    duration=test.get("duration", 0),
                    error_message=test.get("call", {}).get("longrepr", None),
                    metadata={
                        "file": test_file,
                        "line": test.get("lineno", 0),
                        "keywords": test.get("keywords", [])
                    }
                )
                results.append(test_result)
            
            # Clean up temporary report
            latest_report.unlink()
            
        except Exception as e:
            self.logger.error(f"Error parsing pytest results: {e}")
            # Fallback to basic result parsing
            if result.returncode == 0:
                status = TestStatus.PASSED
            else:
                status = TestStatus.FAILED
            
            results.append(TestResult(
                test_id=test_file,
                test_name=test_file,
                test_type=test_type or TestType.UNIT,
                status=status,
                duration=0,
                error_message=result.stderr if result.stderr else None
            ))
        
        return results
    
    def _generate_coverage_report(self) -> None:
        """Generate coverage report"""
        try:
            # Generate HTML coverage report
            html_dir = self.reports_dir / "coverage_html"
            html_dir.mkdir(exist_ok=True)
            
            self.coverage.html_report(directory=str(html_dir))
            
            # Generate JSON coverage report
            json_file = self.reports_dir / "coverage.json"
            self.coverage.json_report(outfile=str(json_file))
            
            # Load coverage data
            if json_file.exists():
                with open(json_file, 'r') as f:
                    self.coverage_data = json.load(f)
            
            self.logger.info(f"Coverage report generated in {html_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating coverage report: {e}")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get test execution summary
        
        Returns:
            Dictionary with test summary
        """
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
        
        total_duration = sum(r.duration for r in self.test_results)
        
        coverage_percentage = 0
        if self.coverage_data:
            coverage_percentage = self.coverage_data.get("totals", {}).get("percent_covered", 0)
        
        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration,
            "coverage_percentage": coverage_percentage,
            "execution_time": self.end_time - self.start_time if self.end_time and self.start_time else 0
        }
    
    def generate_report(self, format: str = "html") -> str:
        """
        Generate test report in specified format
        
        Args:
            format: Report format ("html", "json", "junit")
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "html":
            return self._generate_html_report(timestamp)
        elif format == "json":
            return self._generate_json_report(timestamp)
        elif format == "junit":
            return self._generate_junit_report(timestamp)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_html_report(self, timestamp: str) -> str:
        """Generate HTML report"""
        report_file = self.reports_dir / f"test_report_{timestamp}.html"
        
        # Generate HTML report using existing test reporting system
        try:
            from .test_reporting_system import TestReportingSystem
            
            reporting_system = TestReportingSystem()
            # Convert results to compatible format
            # Implementation details would depend on the existing system
            
        except ImportError:
            # Fallback to basic HTML generation
            html_content = self._generate_basic_html_report()
            
            with open(report_file, 'w') as f:
                f.write(html_content)
        
        return str(report_file)
    
    def _generate_json_report(self, timestamp: str) -> str:
        """Generate JSON report"""
        report_file = self.reports_dir / f"test_report_{timestamp}.json"
        
        report_data = {
            "timestamp": timestamp,
            "summary": self.get_test_summary(),
            "results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "test_type": r.test_type.value,
                    "status": r.status.value,
                    "duration": r.duration,
                    "error_message": r.error_message,
                    "metadata": r.metadata
                }
                for r in self.test_results
            ],
            "coverage": self.coverage_data
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(report_file)
    
    def _generate_junit_report(self, timestamp: str) -> str:
        """Generate JUnit XML report"""
        report_file = self.reports_dir / f"test_report_{timestamp}.xml"
        
        # Generate JUnit XML format
        # Implementation would create proper XML structure
        
        return str(report_file)
    
    def _generate_basic_html_report(self) -> str:
        """Generate basic HTML report"""
        summary = self.get_test_summary()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GrandModel Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                .skipped {{ color: gray; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>GrandModel Test Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {summary.get('total_tests', 0)}</p>
                <p class="passed">Passed: {summary.get('passed', 0)}</p>
                <p class="failed">Failed: {summary.get('failed', 0)}</p>
                <p class="error">Errors: {summary.get('errors', 0)}</p>
                <p class="skipped">Skipped: {summary.get('skipped', 0)}</p>
                <p>Success Rate: {summary.get('success_rate', 0):.1f}%</p>
                <p>Coverage: {summary.get('coverage_percentage', 0):.1f}%</p>
                <p>Duration: {summary.get('total_duration', 0):.2f}s</p>
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Error</th>
                </tr>
        """
        
        for result in self.test_results:
            status_class = result.status.value
            html += f"""
                <tr>
                    <td>{result.test_name}</td>
                    <td>{result.test_type.value}</td>
                    <td class="{status_class}">{result.status.value}</td>
                    <td>{result.duration:.3f}s</td>
                    <td>{result.error_message or ''}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html


# Example usage and test suite definitions
def create_default_test_suites() -> List[TestSuite]:
    """Create default test suites for GrandModel"""
    return [
        TestSuite(
            name="unit_tests",
            test_type=TestType.UNIT,
            test_patterns=["test_*.py", "*_test.py"],
            coverage_threshold=80.0,
            parallel=True
        ),
        TestSuite(
            name="integration_tests",
            test_type=TestType.INTEGRATION,
            test_patterns=["integration_*.py", "test_integration_*.py"],
            coverage_threshold=70.0,
            parallel=False,
            timeout=600
        ),
        TestSuite(
            name="performance_tests",
            test_type=TestType.PERFORMANCE,
            test_patterns=["perf_*.py", "test_performance_*.py"],
            coverage_threshold=50.0,
            parallel=True,
            timeout=900
        ),
        TestSuite(
            name="security_tests",
            test_type=TestType.SECURITY,
            test_patterns=["security_*.py", "test_security_*.py"],
            coverage_threshold=60.0,
            parallel=False,
            timeout=1200
        )
    ]


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize framework
        framework = TestingFramework()
        
        # Register default test suites
        for suite in create_default_test_suites():
            framework.register_test_suite(suite)
        
        # Run all tests
        results = await framework.run_all_tests()
        
        # Generate reports
        html_report = framework.generate_report("html")
        json_report = framework.generate_report("json")
        
        # Print summary
        summary = framework.get_test_summary()
        print(f"\nTest Summary:")
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        print(f"Coverage: {summary.get('coverage_percentage', 0):.1f}%")
        print(f"Reports: {html_report}, {json_report}")
    
    asyncio.run(main())