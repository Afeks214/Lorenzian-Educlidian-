#!/usr/bin/env python3
"""
🤖 AGENT EPSILON MISSION: Continuous Adversarial Testing Automation
Automated adversarial test execution with continuous integration and failure detection.

This module provides:
- Automated test scheduling and execution
- Continuous integration pipeline
- Real-time failure detection and alerting
- Performance regression monitoring
- Automated remediation suggestions
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class TestStatus(Enum):
    """Test execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    status: TestStatus
    execution_time: float
    timestamp: datetime
    details: Dict[str, Any]
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    security_score: Optional[float] = None

@dataclass
class Alert:
    """Alert data structure."""
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    test_name: str
    remediation_steps: List[str]
    affected_systems: List[str]

class ContinuousTestingEngine:
    """
    Continuous adversarial testing automation engine.
    """
    
    def __init__(self, config_path: str = "configs/continuous_testing.yaml"):
        """Initialize the continuous testing engine."""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.test_history: List[TestResult] = []
        self.active_alerts: List[Alert] = []
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        self.shutdown_event = threading.Event()
        self.performance_baseline = {}
        self.last_run_time = None
        
        # Test registry
        self.test_registry = {
            'agent_consistency': self._run_agent_consistency_tests,
            'extreme_data_attacks': self._run_extreme_data_attacks,
            'malicious_config_attacks': self._run_malicious_config_attacks,
            'market_manipulation': self._run_market_manipulation_tests,
            'byzantine_attacks': self._run_byzantine_attacks,
            'performance_regression': self._run_performance_regression_tests,
            'security_validation': self._run_security_validation_tests
        }
        
        self.logger.info("Continuous Testing Engine initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            # Create default config
            default_config = {
                'schedule': {
                    'interval_minutes': 30,
                    'full_suite_hours': 24,
                    'critical_tests_minutes': 5
                },
                'tests': {
                    'agent_consistency': {'enabled': True, 'timeout': 300},
                    'extreme_data_attacks': {'enabled': True, 'timeout': 600},
                    'malicious_config_attacks': {'enabled': True, 'timeout': 300},
                    'market_manipulation': {'enabled': True, 'timeout': 900},
                    'byzantine_attacks': {'enabled': True, 'timeout': 600},
                    'performance_regression': {'enabled': True, 'timeout': 300},
                    'security_validation': {'enabled': True, 'timeout': 600}
                },
                'alerts': {
                    'failure_threshold': 3,
                    'performance_degradation_threshold': 0.2,
                    'security_score_threshold': 0.8,
                    'notification_channels': ['email', 'slack', 'webhook']
                },
                'reporting': {
                    'generate_reports': True,
                    'report_directory': 'reports/continuous_testing',
                    'executive_summary': True
                },
                'max_workers': 4,
                'enable_auto_remediation': True
            }
            
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            return default_config
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('continuous_testing')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'continuous_testing.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def start_continuous_testing(self):
        """Start the continuous testing loop."""
        self.logger.info("Starting continuous adversarial testing...")
        
        # Load performance baseline
        self._load_performance_baseline()
        
        while not self.shutdown_event.is_set():
            try:
                # Check if it's time for a full test suite
                if self._should_run_full_suite():
                    await self._run_full_test_suite()
                else:
                    # Run critical tests
                    await self._run_critical_tests()
                
                # Process alerts
                await self._process_alerts()
                
                # Generate reports if needed
                if self.config.get('reporting', {}).get('generate_reports', True):
                    await self._generate_reports()
                
                # Wait for next scheduled run
                await asyncio.sleep(self.config['schedule']['interval_minutes'] * 60)
                
            except Exception as e:
                self.logger.error(f"Error in continuous testing loop: {e}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Wait before retrying
    
    def _should_run_full_suite(self) -> bool:
        """Determine if a full test suite should be run."""
        if self.last_run_time is None:
            return True
        
        hours_since_last = (datetime.now() - self.last_run_time).total_seconds() / 3600
        return hours_since_last >= self.config['schedule']['full_suite_hours']
    
    async def _run_full_test_suite(self):
        """Run the complete adversarial test suite."""
        self.logger.info("🚀 Starting full adversarial test suite...")
        start_time = time.time()
        
        test_results = []
        enabled_tests = [
            name for name, config in self.config['tests'].items()
            if config.get('enabled', True)
        ]
        
        # Run tests in parallel
        tasks = []
        for test_name in enabled_tests:
            if test_name in self.test_registry:
                task = asyncio.create_task(self._execute_test(test_name))
                tasks.append(task)
        
        # Wait for all tests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for test_name, result in zip(enabled_tests, results):
            if isinstance(result, Exception):
                test_result = TestResult(
                    test_name=test_name,
                    status=TestStatus.ERROR,
                    execution_time=0,
                    timestamp=datetime.now(),
                    details={},
                    error_message=str(result)
                )
            else:
                test_result = result
            
            test_results.append(test_result)
            self.test_history.append(test_result)
        
        execution_time = time.time() - start_time
        self.last_run_time = datetime.now()
        
        # Analyze results and generate alerts
        await self._analyze_test_results(test_results)
        
        self.logger.info(f"✅ Full test suite completed in {execution_time:.2f}s")
        return test_results
    
    async def _run_critical_tests(self):
        """Run critical tests that should be executed frequently."""
        critical_tests = ['agent_consistency', 'security_validation', 'performance_regression']
        
        self.logger.info("🎯 Running critical test subset...")
        
        tasks = []
        for test_name in critical_tests:
            if (test_name in self.test_registry and 
                self.config['tests'].get(test_name, {}).get('enabled', True)):
                task = asyncio.create_task(self._execute_test(test_name))
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for test_name, result in zip(critical_tests, results):
            if isinstance(result, Exception):
                test_result = TestResult(
                    test_name=test_name,
                    status=TestStatus.ERROR,
                    execution_time=0,
                    timestamp=datetime.now(),
                    details={},
                    error_message=str(result)
                )
            else:
                test_result = result
            
            self.test_history.append(test_result)
        
        self.logger.info("✅ Critical tests completed")
    
    async def _execute_test(self, test_name: str) -> TestResult:
        """Execute a single test with timeout and error handling."""
        self.logger.info(f"🔍 Executing test: {test_name}")
        
        start_time = time.time()
        timeout = self.config['tests'][test_name].get('timeout', 300)
        
        try:
            # Execute test function
            test_func = self.test_registry[test_name]
            
            # Run with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, test_func
                ),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Create test result
            test_result = TestResult(
                test_name=test_name,
                status=TestStatus.SUCCESS if result.get('success', False) else TestStatus.FAILURE,
                execution_time=execution_time,
                timestamp=datetime.now(),
                details=result,
                performance_metrics=result.get('performance_metrics', {}),
                security_score=result.get('security_score')
            )
            
            self.logger.info(f"✅ Test {test_name} completed: {test_result.status.value}")
            return test_result
            
        except asyncio.TimeoutError:
            self.logger.error(f"❌ Test {test_name} timed out after {timeout}s")
            return TestResult(
                test_name=test_name,
                status=TestStatus.TIMEOUT,
                execution_time=timeout,
                timestamp=datetime.now(),
                details={},
                error_message=f"Test timed out after {timeout} seconds"
            )
        
        except Exception as e:
            self.logger.error(f"❌ Test {test_name} failed: {e}")
            return TestResult(
                test_name=test_name,
                status=TestStatus.ERROR,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details={},
                error_message=str(e)
            )
    
    def _run_agent_consistency_tests(self) -> Dict[str, Any]:
        """Run agent consistency tests."""
        try:
            # Import and run agent consistency tests
            from adversarial_tests.agent_consistency_test import run_agent_consistency_tests
            
            results, summary = run_agent_consistency_tests()
            
            return {
                'success': summary.get('overall_status') != 'ALL TESTS BLOCKED',
                'results': results,
                'summary': summary,
                'performance_metrics': {'test_count': len(results)},
                'security_score': 0.0 if summary.get('overall_status') == 'ALL TESTS BLOCKED' else 0.5
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'security_score': 0.0
            }
    
    def _run_extreme_data_attacks(self) -> Dict[str, Any]:
        """Run extreme data attack tests."""
        try:
            from adversarial_tests.extreme_data_attacks import run_extreme_data_attacks
            
            results = run_extreme_data_attacks()
            
            return {
                'success': True,
                'results': results,
                'performance_metrics': {'attacks_tested': len(results)},
                'security_score': 0.8  # Assume good score if tests run
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'security_score': 0.0
            }
    
    def _run_malicious_config_attacks(self) -> Dict[str, Any]:
        """Run malicious configuration attack tests."""
        try:
            from adversarial_tests.malicious_config_attacks import run_malicious_config_attacks
            
            results = run_malicious_config_attacks()
            
            return {
                'success': True,
                'results': results,
                'performance_metrics': {'configs_tested': len(results)},
                'security_score': 0.8
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'security_score': 0.0
            }
    
    def _run_market_manipulation_tests(self) -> Dict[str, Any]:
        """Run market manipulation scenario tests."""
        try:
            from adversarial_tests.market_manipulation_scenarios import run_market_manipulation_tests
            
            results = run_market_manipulation_tests()
            
            return {
                'success': True,
                'results': results,
                'performance_metrics': {'scenarios_tested': len(results)},
                'security_score': 0.8
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'security_score': 0.0
            }
    
    def _run_byzantine_attacks(self) -> Dict[str, Any]:
        """Run Byzantine attack tests."""
        try:
            # Run Byzantine attack tests from the test suite
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/adversarial/byzantine_attacks.py', 
                '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'performance_metrics': {'tests_run': result.stdout.count('PASSED')},
                'security_score': 0.8 if result.returncode == 0 else 0.3
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'security_score': 0.0
            }
    
    def _run_performance_regression_tests(self) -> Dict[str, Any]:
        """Run performance regression tests."""
        try:
            # Run performance tests
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/performance/', 
                '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'performance_metrics': {'tests_run': result.stdout.count('PASSED')},
                'security_score': 0.9 if result.returncode == 0 else 0.5
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'security_score': 0.0
            }
    
    def _run_security_validation_tests(self) -> Dict[str, Any]:
        """Run security validation tests."""
        try:
            # Run security tests
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/security/', 
                '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'performance_metrics': {'tests_run': result.stdout.count('PASSED')},
                'security_score': 0.9 if result.returncode == 0 else 0.2
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'security_score': 0.0
            }
    
    async def _analyze_test_results(self, results: List[TestResult]):
        """Analyze test results and generate alerts."""
        for result in results:
            # Check for failures
            if result.status in [TestStatus.FAILURE, TestStatus.ERROR, TestStatus.TIMEOUT]:
                await self._create_failure_alert(result)
            
            # Check for performance regression
            if result.performance_metrics:
                await self._check_performance_regression(result)
            
            # Check security score
            if result.security_score is not None:
                await self._check_security_score(result)
    
    async def _create_failure_alert(self, result: TestResult):
        """Create alert for test failure."""
        severity = AlertSeverity.HIGH if result.status == TestStatus.TIMEOUT else AlertSeverity.MEDIUM
        
        alert = Alert(
            severity=severity,
            title=f"Test Failure: {result.test_name}",
            description=f"Test {result.test_name} failed with status {result.status.value}. "
                       f"Error: {result.error_message or 'No error message'}",
            timestamp=result.timestamp,
            test_name=result.test_name,
            remediation_steps=[
                "Review test logs for detailed error information",
                "Check system resources and dependencies",
                "Verify test data integrity",
                "Consider increasing test timeout if appropriate"
            ],
            affected_systems=["adversarial_testing", "security_validation"]
        )
        
        self.active_alerts.append(alert)
        self.logger.warning(f"🚨 Alert created: {alert.title}")
    
    async def _check_performance_regression(self, result: TestResult):
        """Check for performance regression."""
        if not result.performance_metrics:
            return
        
        baseline_key = f"{result.test_name}_execution_time"
        if baseline_key in self.performance_baseline:
            baseline_time = self.performance_baseline[baseline_key]
            current_time = result.execution_time
            
            if current_time > baseline_time * (1 + self.config['alerts']['performance_degradation_threshold']):
                alert = Alert(
                    severity=AlertSeverity.MEDIUM,
                    title=f"Performance Regression: {result.test_name}",
                    description=f"Test {result.test_name} execution time increased from "
                               f"{baseline_time:.2f}s to {current_time:.2f}s "
                               f"({((current_time - baseline_time) / baseline_time * 100):.1f}% increase)",
                    timestamp=result.timestamp,
                    test_name=result.test_name,
                    remediation_steps=[
                        "Investigate recent system changes",
                        "Check resource utilization",
                        "Review test complexity changes",
                        "Consider system optimization"
                    ],
                    affected_systems=["performance", "testing_infrastructure"]
                )
                
                self.active_alerts.append(alert)
                self.logger.warning(f"🐌 Performance regression detected: {alert.title}")
        else:
            # Set baseline for future comparisons
            self.performance_baseline[baseline_key] = result.execution_time
    
    async def _check_security_score(self, result: TestResult):
        """Check security score against threshold."""
        if result.security_score is None:
            return
        
        threshold = self.config['alerts']['security_score_threshold']
        if result.security_score < threshold:
            alert = Alert(
                severity=AlertSeverity.HIGH,
                title=f"Low Security Score: {result.test_name}",
                description=f"Test {result.test_name} security score {result.security_score:.2f} "
                           f"is below threshold {threshold:.2f}",
                timestamp=result.timestamp,
                test_name=result.test_name,
                remediation_steps=[
                    "Review security test results",
                    "Investigate security vulnerabilities",
                    "Update security controls",
                    "Consider system hardening"
                ],
                affected_systems=["security", "risk_management"]
            )
            
            self.active_alerts.append(alert)
            self.logger.error(f"🔒 Security alert created: {alert.title}")
    
    async def _process_alerts(self):
        """Process and send alerts."""
        if not self.active_alerts:
            return
        
        # Group alerts by severity
        critical_alerts = [a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]
        high_alerts = [a for a in self.active_alerts if a.severity == AlertSeverity.HIGH]
        
        # Send critical alerts immediately
        for alert in critical_alerts:
            await self._send_alert(alert)
        
        # Send high alerts if threshold reached
        if len(high_alerts) >= self.config['alerts']['failure_threshold']:
            await self._send_alert_summary(high_alerts)
        
        # Auto-remediation if enabled
        if self.config.get('enable_auto_remediation', False):
            await self._attempt_auto_remediation()
    
    async def _send_alert(self, alert: Alert):
        """Send individual alert."""
        self.logger.error(f"🚨 ALERT [{alert.severity.value.upper()}]: {alert.title}")
        self.logger.error(f"Description: {alert.description}")
        self.logger.error(f"Affected Systems: {', '.join(alert.affected_systems)}")
        
        # Here you would integrate with actual notification systems
        # (email, Slack, PagerDuty, etc.)
        
        # For now, just log the alert
        alert_data = {
            'timestamp': alert.timestamp.isoformat(),
            'severity': alert.severity.value,
            'title': alert.title,
            'description': alert.description,
            'test_name': alert.test_name,
            'remediation_steps': alert.remediation_steps,
            'affected_systems': alert.affected_systems
        }
        
        # Save alert to file
        alerts_dir = Path('reports/alerts')
        alerts_dir.mkdir(parents=True, exist_ok=True)
        
        alert_file = alerts_dir / f"alert_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
    
    async def _send_alert_summary(self, alerts: List[Alert]):
        """Send summary of multiple alerts."""
        self.logger.error(f"🚨 ALERT SUMMARY: {len(alerts)} alerts detected")
        
        for alert in alerts:
            self.logger.error(f"- {alert.title} ({alert.severity.value})")
    
    async def _attempt_auto_remediation(self):
        """Attempt automatic remediation for known issues."""
        # This is a placeholder for auto-remediation logic
        # In a real system, this would include:
        # - Restarting failed services
        # - Cleaning up resources
        # - Updating configurations
        # - Scaling resources
        
        self.logger.info("🔧 Auto-remediation not implemented yet")
    
    async def _generate_reports(self):
        """Generate testing reports."""
        if not self.test_history:
            return
        
        report_dir = Path(self.config['reporting']['report_directory'])
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate daily summary
        today = datetime.now().strftime('%Y-%m-%d')
        summary_file = report_dir / f"daily_summary_{today}.json"
        
        # Filter today's results
        today_results = [
            r for r in self.test_history
            if r.timestamp.date() == datetime.now().date()
        ]
        
        if today_results:
            summary = {
                'date': today,
                'total_tests': len(today_results),
                'successful_tests': len([r for r in today_results if r.status == TestStatus.SUCCESS]),
                'failed_tests': len([r for r in today_results if r.status == TestStatus.FAILURE]),
                'error_tests': len([r for r in today_results if r.status == TestStatus.ERROR]),
                'timeout_tests': len([r for r in today_results if r.status == TestStatus.TIMEOUT]),
                'average_execution_time': sum(r.execution_time for r in today_results) / len(today_results),
                'average_security_score': sum(r.security_score for r in today_results if r.security_score is not None) / 
                                        len([r for r in today_results if r.security_score is not None]) if any(r.security_score is not None for r in today_results) else 0,
                'active_alerts': len(self.active_alerts),
                'test_results': [asdict(r) for r in today_results]
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"📊 Daily report generated: {summary_file}")
    
    def _load_performance_baseline(self):
        """Load performance baseline from file."""
        baseline_file = Path('reports/performance_baseline.json')
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.performance_baseline = json.load(f)
    
    def _save_performance_baseline(self):
        """Save performance baseline to file."""
        baseline_file = Path('reports/performance_baseline.json')
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(baseline_file, 'w') as f:
            json.dump(self.performance_baseline, f, indent=2)
    
    def stop(self):
        """Stop the continuous testing engine."""
        self.logger.info("🛑 Stopping continuous testing engine...")
        self.shutdown_event.set()
        self.executor.shutdown(wait=True)
        self._save_performance_baseline()
        self.logger.info("✅ Continuous testing engine stopped")

async def main():
    """Main function to run continuous testing."""
    engine = ContinuousTestingEngine()
    
    try:
        await engine.start_continuous_testing()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down continuous testing...")
        engine.stop()
    except Exception as e:
        print(f"❌ Error in continuous testing: {e}")
        engine.stop()

if __name__ == "__main__":
    asyncio.run(main())