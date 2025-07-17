"""
Infrastructure Validation Script

Validates the complete adversarial testing infrastructure to ensure
all components are working correctly and meet performance requirements.
"""

import asyncio
import sys
import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import psutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from adversarial_tests.infrastructure import (
    TestOrchestrator,
    TestingDashboard,
    AdversarialDetector,
    ParallelExecutor,
    TestTask,
    TestPriority,
    ExecutionMode,
    ResourceQuota,
    AttackType,
    ThreatLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InfrastructureValidator:
    """Validates the adversarial testing infrastructure"""
    
    def __init__(self):
        self.validation_results = {}
        self.performance_metrics = {}
        self.errors = []
        
    def log_validation_result(self, test_name: str, passed: bool, 
                            details: Dict = None, error: str = None):
        """Log validation result"""
        self.validation_results[test_name] = {
            "passed": passed,
            "details": details or {},
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        if passed:
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED - {error}")
            self.errors.append(f"{test_name}: {error}")
    
    def log_performance_metric(self, metric_name: str, value: float, 
                             threshold: float = None, unit: str = ""):
        """Log performance metric"""
        self.performance_metrics[metric_name] = {
            "value": value,
            "threshold": threshold,
            "unit": unit,
            "meets_threshold": value <= threshold if threshold else None,
            "timestamp": datetime.now().isoformat()
        }
        
        if threshold:
            status = "✅" if value <= threshold else "⚠️"
            logger.info(f"{status} {metric_name}: {value:.3f}{unit} (threshold: {threshold}{unit})")
        else:
            logger.info(f"📊 {metric_name}: {value:.3f}{unit}")
    
    async def validate_test_orchestrator(self):
        """Validate test orchestrator functionality"""
        logger.info("=== VALIDATING TEST ORCHESTRATOR ===")
        
        try:
            # Create orchestrator
            orchestrator = TestOrchestrator(max_parallel_tests=3)
            
            # Test 1: Session creation
            session_id = await orchestrator.create_session("Validation Session")
            self.log_validation_result(
                "orchestrator_session_creation",
                session_id is not None,
                {"session_id": session_id}
            )
            
            # Test 2: Task addition
            async def simple_test():
                await asyncio.sleep(0.1)
                return {"result": "success"}
            
            task = TestTask(
                test_id="validation_test",
                test_name="Validation Test",
                test_function=simple_test,
                priority=TestPriority.HIGH
            )
            
            await orchestrator.add_test_task(session_id, task)
            session = orchestrator.sessions[session_id]
            
            self.log_validation_result(
                "orchestrator_task_addition",
                len(session.tasks) == 1,
                {"task_count": len(session.tasks)}
            )
            
            # Test 3: Session execution
            start_time = time.time()
            results = await orchestrator.execute_session(session_id)
            execution_time = time.time() - start_time
            
            self.log_validation_result(
                "orchestrator_session_execution",
                len(results["results"]) == 1 and results["results"][0]["status"] == "completed",
                {"execution_time": execution_time, "results": results}
            )
            
            self.log_performance_metric("orchestrator_execution_time", execution_time, 5.0, "s")
            
            # Test 4: System metrics
            metrics = await orchestrator.get_system_metrics()
            self.log_validation_result(
                "orchestrator_system_metrics",
                "resource_usage" in metrics and "orchestrator_metrics" in metrics,
                {"metrics_keys": list(metrics.keys())}
            )
            
        except Exception as e:
            self.log_validation_result("orchestrator_validation", False, error=str(e))
    
    async def validate_adversarial_detector(self):
        """Validate adversarial detector functionality"""
        logger.info("=== VALIDATING ADVERSARIAL DETECTOR ===")
        
        try:
            # Create detector
            detector = AdversarialDetector()
            await detector.start_monitoring()
            
            # Test 1: Model analysis
            model = nn.Sequential(
                nn.Linear(10, 5),
                nn.ReLU(),
                nn.Linear(5, 1)
            )
            
            attacks = await detector.analyze_model(model, "test_model", {"accuracy": 0.95})
            self.log_validation_result(
                "detector_model_analysis",
                isinstance(attacks, list),
                {"attacks_count": len(attacks)}
            )
            
            # Test 2: Gradient analysis
            gradients = {
                "layer1.weight": torch.randn(5, 10) * 0.01,
                "layer1.bias": torch.randn(5) * 0.01,
                "layer2.weight": torch.randn(1, 5) * 0.01,
                "layer2.bias": torch.randn(1) * 0.01
            }
            
            attacks = await detector.analyze_gradients(gradients, "test_agent")
            self.log_validation_result(
                "detector_gradient_analysis",
                isinstance(attacks, list),
                {"attacks_count": len(attacks)}
            )
            
            # Test 3: Byzantine detection
            decision = {"action": "buy", "amount": 100}
            attacks = await detector.analyze_agent_decisions("test_agent", decision, 0.8)
            self.log_validation_result(
                "detector_byzantine_analysis",
                isinstance(attacks, list),
                {"attacks_count": len(attacks)}
            )
            
            # Test 4: Detection summary
            summary = detector.get_detection_summary()
            self.log_validation_result(
                "detector_summary",
                "total_recent_attacks" in summary and "monitoring_active" in summary,
                {"summary_keys": list(summary.keys())}
            )
            
            # Test 5: Performance test
            start_time = time.time()
            for i in range(50):
                await detector.analyze_gradients(gradients, f"perf_agent_{i}")
            analysis_time = time.time() - start_time
            
            self.log_performance_metric("detector_gradient_analysis_time", analysis_time, 2.0, "s")
            
            await detector.stop_monitoring()
            
        except Exception as e:
            self.log_validation_result("detector_validation", False, error=str(e))
    
    async def validate_parallel_executor(self):
        """Validate parallel executor functionality"""
        logger.info("=== VALIDATING PARALLEL EXECUTOR ===")
        
        try:
            # Create executor
            executor = ParallelExecutor(max_workers=2, enable_containers=False)
            await executor.start()
            
            # Test 1: Single execution
            def test_function(value: int):
                return value * 2
            
            start_time = time.time()
            context = await executor.execute_test(
                test_function,
                test_args=(42,),
                execution_mode=ExecutionMode.THREAD
            )
            execution_time = time.time() - start_time
            
            self.log_validation_result(
                "executor_single_execution",
                context.exit_code == 0 and "84" in context.stdout,
                {"exit_code": context.exit_code, "output": context.stdout}
            )
            
            self.log_performance_metric("executor_single_execution_time", execution_time, 1.0, "s")
            
            # Test 2: Batch execution
            test_functions = [lambda i=i: test_function(i) for i in range(5)]
            
            start_time = time.time()
            batch_results = await executor.execute_batch(
                test_functions,
                execution_mode=ExecutionMode.THREAD,
                max_parallel=2
            )
            batch_time = time.time() - start_time
            
            successful_results = sum(1 for r in batch_results if r.exit_code == 0)
            
            self.log_validation_result(
                "executor_batch_execution",
                len(batch_results) == 5 and successful_results == 5,
                {"total_results": len(batch_results), "successful": successful_results}
            )
            
            self.log_performance_metric("executor_batch_execution_time", batch_time, 3.0, "s")
            
            # Test 3: Resource management
            quota = ResourceQuota(cpu_cores=1.0, memory_mb=100)
            context = await executor.execute_test(
                test_function,
                test_args=(100,),
                execution_mode=ExecutionMode.PROCESS,
                resource_quota=quota
            )
            
            self.log_validation_result(
                "executor_resource_management",
                context.exit_code == 0,
                {"exit_code": context.exit_code, "quota": quota.__dict__}
            )
            
            # Test 4: System status
            status = executor.get_system_status()
            self.log_validation_result(
                "executor_system_status",
                "active_executions" in status and "metrics" in status,
                {"status_keys": list(status.keys())}
            )
            
            await executor.stop()
            
        except Exception as e:
            self.log_validation_result("executor_validation", False, error=str(e))
    
    async def validate_testing_dashboard(self):
        """Validate testing dashboard functionality"""
        logger.info("=== VALIDATING TESTING DASHBOARD ===")
        
        try:
            # Create orchestrator for dashboard
            orchestrator = TestOrchestrator(max_parallel_tests=2)
            
            # Create dashboard
            dashboard = TestingDashboard(orchestrator, port=5005)
            
            # Test 1: Database operations
            from adversarial_tests.infrastructure.testing_dashboard import TestMetrics
            
            test_metric = TestMetrics(
                test_id="dashboard_test",
                test_name="Dashboard Test",
                execution_time=1.5,
                memory_usage=100.0,
                cpu_usage=50.0,
                status="completed",
                timestamp=datetime.now(),
                session_id="test_session"
            )
            
            dashboard.db.insert_test_metrics(test_metric)
            retrieved_metrics = dashboard.db.get_test_metrics(session_id="test_session")
            
            self.log_validation_result(
                "dashboard_database_operations",
                len(retrieved_metrics) == 1 and retrieved_metrics[0].test_id == "dashboard_test",
                {"metrics_count": len(retrieved_metrics)}
            )
            
            # Test 2: Analytics generation
            analytics = dashboard._generate_performance_analytics()
            self.log_validation_result(
                "dashboard_analytics_generation",
                "total_tests" in analytics and "success_rate" in analytics,
                {"analytics_keys": list(analytics.keys())}
            )
            
            # Test 3: Report generation
            report = dashboard.generate_report(session_id="test_session")
            self.log_validation_result(
                "dashboard_report_generation",
                "report_generated" in report and "summary" in report,
                {"report_keys": list(report.keys())}
            )
            
        except Exception as e:
            self.log_validation_result("dashboard_validation", False, error=str(e))
    
    async def validate_integration(self):
        """Validate integration between components"""
        logger.info("=== VALIDATING COMPONENT INTEGRATION ===")
        
        try:
            # Create all components
            orchestrator = TestOrchestrator(max_parallel_tests=2)
            detector = AdversarialDetector(orchestrator.event_bus)
            executor = ParallelExecutor(max_workers=2, enable_containers=False)
            dashboard = TestingDashboard(orchestrator, port=5006)
            
            # Start components
            await detector.start_monitoring()
            await executor.start()
            
            # Test 1: Event system integration
            events_received = []
            
            async def event_handler(event):
                events_received.append(event)
            
            orchestrator.event_bus.subscribe("session_created", event_handler)
            orchestrator.event_bus.subscribe("task_completed", event_handler)
            
            # Create and execute session
            session_id = await orchestrator.create_session("Integration Test")
            
            async def integration_test():
                # Simulate test that uses detector
                model = nn.Linear(5, 1)
                attacks = await detector.analyze_model(model, "integration_model", {"accuracy": 0.9})
                return {"attacks_detected": len(attacks)}
            
            task = TestTask(
                test_id="integration_test",
                test_name="Integration Test",
                test_function=integration_test,
                priority=TestPriority.HIGH
            )
            
            await orchestrator.add_test_task(session_id, task)
            results = await orchestrator.execute_session(session_id)
            
            # Wait for events
            await asyncio.sleep(0.5)
            
            self.log_validation_result(
                "integration_event_system",
                len(events_received) >= 2,
                {"events_received": len(events_received)}
            )
            
            # Test 2: Cross-component data flow
            self.log_validation_result(
                "integration_data_flow",
                len(results["results"]) == 1 and results["results"][0]["status"] == "completed",
                {"execution_results": results}
            )
            
            # Test 3: System metrics aggregation
            orchestrator_metrics = await orchestrator.get_system_metrics()
            detector_summary = detector.get_detection_summary()
            executor_status = executor.get_system_status()
            
            self.log_validation_result(
                "integration_metrics_aggregation",
                all(metrics is not None for metrics in [orchestrator_metrics, detector_summary, executor_status]),
                {
                    "orchestrator_active": orchestrator_metrics is not None,
                    "detector_active": detector_summary is not None,
                    "executor_active": executor_status is not None
                }
            )
            
            # Cleanup
            await detector.stop_monitoring()
            await executor.stop()
            
        except Exception as e:
            self.log_validation_result("integration_validation", False, error=str(e))
    
    async def validate_performance_requirements(self):
        """Validate performance requirements"""
        logger.info("=== VALIDATING PERFORMANCE REQUIREMENTS ===")
        
        try:
            # Test 1: Orchestrator performance
            orchestrator = TestOrchestrator(max_parallel_tests=5)
            
            # Create many tasks
            session_id = await orchestrator.create_session("Performance Test")
            
            tasks = []
            for i in range(20):
                task = TestTask(
                    test_id=f"perf_test_{i}",
                    test_name=f"Performance Test {i}",
                    test_function=lambda: {"result": f"test_{i}"},
                    priority=TestPriority.MEDIUM
                )
                tasks.append(task)
            
            # Measure task addition performance
            start_time = time.time()
            for task in tasks:
                await orchestrator.add_test_task(session_id, task)
            addition_time = time.time() - start_time
            
            self.log_performance_metric("orchestrator_task_addition_rate", 20 / addition_time, 50.0, " tasks/s")
            
            # Measure execution performance
            start_time = time.time()
            results = await orchestrator.execute_session(session_id)
            execution_time = time.time() - start_time
            
            self.log_performance_metric("orchestrator_execution_throughput", 20 / execution_time, 5.0, " tests/s")
            
            # Test 2: Detector performance
            detector = AdversarialDetector()
            await detector.start_monitoring()
            
            gradients = {
                "layer1.weight": torch.randn(100, 100) * 0.01,
                "layer1.bias": torch.randn(100) * 0.01
            }
            
            start_time = time.time()
            for i in range(100):
                await detector.analyze_gradients(gradients, f"perf_agent_{i}")
            analysis_time = time.time() - start_time
            
            self.log_performance_metric("detector_gradient_analysis_rate", 100 / analysis_time, 50.0, " analyses/s")
            
            await detector.stop_monitoring()
            
            # Test 3: Executor performance
            executor = ParallelExecutor(max_workers=4, enable_containers=False)
            await executor.start()
            
            def quick_test(value: int):
                return value * 2
            
            test_functions = [lambda i=i: quick_test(i) for i in range(20)]
            
            start_time = time.time()
            batch_results = await executor.execute_batch(
                test_functions,
                execution_mode=ExecutionMode.THREAD,
                max_parallel=4
            )
            batch_time = time.time() - start_time
            
            self.log_performance_metric("executor_batch_throughput", 20 / batch_time, 10.0, " tests/s")
            
            await executor.stop()
            
        except Exception as e:
            self.log_validation_result("performance_validation", False, error=str(e))
    
    async def validate_system_resources(self):
        """Validate system resource requirements"""
        logger.info("=== VALIDATING SYSTEM RESOURCES ===")
        
        try:
            # Check system resources
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            disk_gb = psutil.disk_usage('/').total / (1024**3)
            
            self.log_validation_result(
                "system_cpu_resources",
                cpu_count >= 2,
                {"cpu_count": cpu_count, "minimum_required": 2}
            )
            
            self.log_validation_result(
                "system_memory_resources",
                memory_gb >= 4.0,
                {"memory_gb": memory_gb, "minimum_required": 4.0}
            )
            
            self.log_validation_result(
                "system_disk_resources",
                disk_gb >= 10.0,
                {"disk_gb": disk_gb, "minimum_required": 10.0}
            )
            
            # Check Python version
            python_version = sys.version_info
            self.log_validation_result(
                "python_version",
                python_version.major == 3 and python_version.minor >= 8,
                {"python_version": f"{python_version.major}.{python_version.minor}"}
            )
            
            # Check required packages
            required_packages = [
                "torch", "numpy", "psutil", "asyncio", "sqlite3", 
                "flask", "pandas", "matplotlib", "seaborn", "sklearn"
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    missing_packages.append(package)
            
            self.log_validation_result(
                "required_packages",
                len(missing_packages) == 0,
                {"missing_packages": missing_packages}
            )
            
        except Exception as e:
            self.log_validation_result("system_resources_validation", False, error=str(e))
    
    async def run_complete_validation(self):
        """Run complete validation suite"""
        logger.info("🔍 STARTING INFRASTRUCTURE VALIDATION")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Run all validation tests
        await self.validate_system_resources()
        await self.validate_test_orchestrator()
        await self.validate_adversarial_detector()
        await self.validate_parallel_executor()
        await self.validate_testing_dashboard()
        await self.validate_integration()
        await self.validate_performance_requirements()
        
        total_time = time.time() - start_time
        
        # Generate validation report
        self.generate_validation_report(total_time)
    
    def generate_validation_report(self, total_time: float):
        """Generate comprehensive validation report"""
        logger.info("="*80)
        logger.info("📋 VALIDATION REPORT")
        logger.info("="*80)
        
        # Count results
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results.values() if r["passed"])
        failed_tests = total_tests - passed_tests
        
        # Performance metrics summary
        performance_summary = {}
        for metric_name, metric_data in self.performance_metrics.items():
            meets_threshold = metric_data.get("meets_threshold")
            if meets_threshold is not None:
                performance_summary[metric_name] = "✅ PASS" if meets_threshold else "⚠️ FAIL"
            else:
                performance_summary[metric_name] = "📊 INFO"
        
        # Generate report
        report = {
            "validation_timestamp": datetime.now().isoformat(),
            "total_validation_time": total_time,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                "infrastructure_status": "OPERATIONAL" if failed_tests == 0 else "ISSUES DETECTED"
            },
            "validation_results": self.validation_results,
            "performance_metrics": self.performance_metrics,
            "performance_summary": performance_summary,
            "errors": self.errors,
            "recommendations": self.generate_recommendations()
        }
        
        # Save report
        report_filename = f"infrastructure_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📊 VALIDATION SUMMARY")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
        print(f"Validation Time: {total_time:.2f}s")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED - INFRASTRUCTURE IS OPERATIONAL")
        else:
            print(f"\n⚠️  {failed_tests} TESTS FAILED - ISSUES DETECTED")
            print("\nFAILED TESTS:")
            for error in self.errors:
                print(f"  ❌ {error}")
        
        print(f"\n📄 Full validation report saved to: {report_filename}")
        logger.info("="*80)
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Check for failures
        failures = [name for name, result in self.validation_results.items() if not result["passed"]]
        
        if not failures:
            recommendations.append("✅ All validation tests passed - infrastructure is production-ready")
        else:
            recommendations.append("⚠️ Some validation tests failed - address issues before production use")
        
        # Performance recommendations
        slow_metrics = [
            name for name, metric in self.performance_metrics.items()
            if metric.get("meets_threshold") is False
        ]
        
        if slow_metrics:
            recommendations.append(f"⚡ Performance optimization needed for: {', '.join(slow_metrics)}")
        
        # System recommendations
        if "system_resources_validation" in failures:
            recommendations.append("🖥️ Upgrade system resources (CPU, memory, or disk)")
        
        if "required_packages" in failures:
            recommendations.append("📦 Install missing Python packages")
        
        recommendations.extend([
            "🔄 Run validation regularly to ensure continued functionality",
            "📊 Monitor performance metrics in production",
            "🛡️ Enable all security features in production deployment"
        ])
        
        return recommendations


async def main():
    """Main validation function"""
    validator = InfrastructureValidator()
    
    try:
        await validator.run_complete_validation()
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())