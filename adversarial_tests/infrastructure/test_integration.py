"""
Integration Tests for Adversarial Testing Infrastructure

Comprehensive integration tests validating the complete adversarial testing
infrastructure including orchestration, monitoring, detection, and execution.
"""

import asyncio
import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys
import json
import time
from typing import Dict, List, Any
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from adversarial_tests.infrastructure import (
    TestOrchestrator,
    TestingDashboard,
    AdversarialDetector,
    ParallelExecutor,
    TestTask,
    TestPriority,
    TestStatus,
    ExecutionMode,
    ResourceQuota,
    AttackType,
    ThreatLevel
)

from src.core.event_bus import EventBus
from src.core.events import Event


class TestInfrastructureIntegration:
    """Integration tests for the complete testing infrastructure"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create test orchestrator"""
        orchestrator = TestOrchestrator(max_parallel_tests=3)
        yield orchestrator
        # Cleanup any sessions
        for session_id in list(orchestrator.sessions.keys()):
            await orchestrator.cancel_session(session_id)
    
    @pytest.fixture
    async def detector(self):
        """Create adversarial detector"""
        detector = AdversarialDetector()
        await detector.start_monitoring()
        yield detector
        await detector.stop_monitoring()
    
    @pytest.fixture
    async def executor(self):
        """Create parallel executor"""
        executor = ParallelExecutor(max_workers=2, enable_containers=False)
        await executor.start()
        yield executor
        await executor.stop()
    
    @pytest.fixture
    async def dashboard(self, orchestrator):
        """Create testing dashboard"""
        dashboard = TestingDashboard(orchestrator, port=5002)
        yield dashboard
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.Softmax(dim=1)
        )
        return model
    
    async def test_orchestrator_basic_functionality(self, orchestrator):
        """Test basic orchestrator functionality"""
        # Create session
        session_id = await orchestrator.create_session("Test Session")
        assert session_id in orchestrator.sessions
        
        # Add test task
        async def simple_test():
            await asyncio.sleep(0.1)
            return {"result": "success"}
        
        task = TestTask(
            test_id="test_1",
            test_name="Simple Test",
            test_function=simple_test,
            priority=TestPriority.HIGH,
            timeout=5.0
        )
        
        await orchestrator.add_test_task(session_id, task)
        assert len(orchestrator.sessions[session_id].tasks) == 1
        
        # Execute session
        results = await orchestrator.execute_session(session_id)
        assert results["session_id"] == session_id
        assert len(results["results"]) == 1
        assert results["results"][0]["status"] == TestStatus.COMPLETED.value
        
        # Get session status
        status = await orchestrator.get_session_status(session_id)
        assert status["total_tasks"] == 1
        assert status["completed_tasks"] == 1
    
    async def test_adversarial_detector_model_analysis(self, detector, sample_model):
        """Test adversarial detector model analysis"""
        # Create baseline fingerprint
        performance_metrics = {"accuracy": 0.95, "loss": 0.05}
        attacks = await detector.analyze_model(sample_model, "test_model", performance_metrics)
        
        # Should be no attacks on first analysis (creates baseline)
        assert len(attacks) == 0
        
        # Modify model slightly
        with torch.no_grad():
            sample_model[0].weight.data += 0.1
        
        # Analyze again
        attacks = await detector.analyze_model(sample_model, "test_model", performance_metrics)
        
        # Should detect model modification
        assert len(attacks) > 0
        assert any(attack.attack_type == AttackType.MODEL_POISONING for attack in attacks)
    
    async def test_adversarial_detector_gradient_analysis(self, detector):
        """Test adversarial detector gradient analysis"""
        # Create normal gradients
        normal_gradients = {
            "layer1.weight": torch.randn(20, 10) * 0.01,
            "layer1.bias": torch.randn(20) * 0.01,
            "layer2.weight": torch.randn(5, 20) * 0.01,
            "layer2.bias": torch.randn(5) * 0.01
        }
        
        # Add normal gradients multiple times
        for i in range(15):
            attacks = await detector.analyze_gradients(normal_gradients, "test_agent")
            assert len(attacks) == 0  # Should be no attacks with normal gradients
        
        # Create abnormal gradients (gradient explosion)
        abnormal_gradients = {
            "layer1.weight": torch.randn(20, 10) * 100,  # Very large gradients
            "layer1.bias": torch.randn(20) * 100,
            "layer2.weight": torch.randn(5, 20) * 100,
            "layer2.bias": torch.randn(5) * 100
        }
        
        # Analyze abnormal gradients
        attacks = await detector.analyze_gradients(abnormal_gradients, "test_agent")
        
        # Should detect gradient anomaly
        assert len(attacks) > 0
        assert any(attack.attack_type == AttackType.GRADIENT_MANIPULATION for attack in attacks)
    
    async def test_adversarial_detector_byzantine_detection(self, detector):
        """Test Byzantine behavior detection"""
        # Add normal agent decisions
        for i in range(10):
            await detector.analyze_agent_decisions(
                "agent_1", 
                {"action": "buy", "amount": 100 + i},
                0.8 + i * 0.01
            )
            await detector.analyze_agent_decisions(
                "agent_2",
                {"action": "sell", "amount": 90 + i},
                0.75 + i * 0.01
            )
        
        # Add deviant agent decisions
        for i in range(10):
            await detector.analyze_agent_decisions(
                "agent_3",
                {"action": "buy", "amount": 1000000},  # Extremely large amount
                0.1  # Poor performance
            )
        
        # Get detection summary
        summary = detector.get_detection_summary()
        
        # Should detect Byzantine behavior
        assert summary["total_recent_attacks"] > 0
        assert AttackType.BYZANTINE_ATTACK.value in summary["attack_types"]
    
    async def test_parallel_executor_basic_execution(self, executor):
        """Test basic parallel executor functionality"""
        def cpu_test(duration: int = 1):
            time.sleep(duration)
            return f"CPU test completed: {duration}s"
        
        # Execute single test
        context = await executor.execute_test(
            cpu_test,
            test_args=(1,),
            execution_mode=ExecutionMode.THREAD,
            timeout=10.0
        )
        
        assert context.exit_code == 0
        assert "CPU test completed" in context.stdout
        assert context.execution_time > 0
        
        # Execute batch
        test_functions = [
            lambda: cpu_test(1),
            lambda: cpu_test(2),
            lambda: cpu_test(1)
        ]
        
        batch_results = await executor.execute_batch(
            test_functions,
            execution_mode=ExecutionMode.THREAD,
            max_parallel=2
        )
        
        assert len(batch_results) == 3
        assert all(result.exit_code == 0 for result in batch_results)
    
    async def test_parallel_executor_resource_management(self, executor):
        """Test parallel executor resource management"""
        def memory_test(size_mb: int = 50):
            # Allocate memory
            data = [0] * (size_mb * 1024 * 1024 // 8)
            return f"Memory test: {len(data)} elements"
        
        # Execute with resource quota
        quota = ResourceQuota(
            cpu_cores=1.0,
            memory_mb=100,
            disk_mb=100
        )
        
        context = await executor.execute_test(
            memory_test,
            test_args=(50,),
            execution_mode=ExecutionMode.PROCESS,
            resource_quota=quota,
            timeout=30.0
        )
        
        assert context.exit_code == 0
        assert "Memory test" in context.stdout
        
        # Check system status
        status = executor.get_system_status()
        assert status["total_executions"] >= 1
        assert "resource_availability" in status
    
    async def test_dashboard_metrics_collection(self, dashboard):
        """Test dashboard metrics collection"""
        # Create test metrics
        test_metrics = [
            {
                "test_id": "test_1",
                "test_name": "Test 1",
                "execution_time": 1.5,
                "memory_usage": 100.0,
                "cpu_usage": 50.0,
                "status": "completed",
                "timestamp": datetime.now(),
                "session_id": "session_1"
            },
            {
                "test_id": "test_2",
                "test_name": "Test 2",
                "execution_time": 2.0,
                "memory_usage": 150.0,
                "cpu_usage": 60.0,
                "status": "failed",
                "timestamp": datetime.now(),
                "session_id": "session_1",
                "error_message": "Test failed"
            }
        ]
        
        # Add metrics to database
        from adversarial_tests.infrastructure.testing_dashboard import TestMetrics
        for metric_data in test_metrics:
            metric = TestMetrics(**metric_data)
            dashboard.db.insert_test_metrics(metric)
        
        # Retrieve metrics
        retrieved_metrics = dashboard.db.get_test_metrics(session_id="session_1")
        assert len(retrieved_metrics) == 2
        
        # Generate analytics
        analytics = dashboard._generate_performance_analytics()
        assert "total_tests" in analytics
        assert "success_rate" in analytics
    
    async def test_integrated_workflow(self, orchestrator, detector, executor):
        """Test complete integrated workflow"""
        # Create session
        session_id = await orchestrator.create_session("Integrated Test Session")
        
        # Define test functions
        async def security_test():
            # Simulate security test
            await asyncio.sleep(0.1)
            return {"security_check": "passed"}
        
        def performance_test():
            # Simulate performance test
            time.sleep(0.5)
            return {"performance_score": 0.95}
        
        async def adversarial_test():
            # Simulate adversarial test
            model = nn.Linear(10, 1)
            gradients = {
                "weight": torch.randn(1, 10) * 0.01,
                "bias": torch.randn(1) * 0.01
            }
            
            # Analyze with detector
            attacks = await detector.analyze_gradients(gradients, "test_agent")
            
            return {
                "attacks_detected": len(attacks),
                "model_analyzed": True
            }
        
        # Create test tasks
        tasks = [
            TestTask(
                test_id="security_test",
                test_name="Security Test",
                test_function=security_test,
                priority=TestPriority.HIGH,
                timeout=10.0
            ),
            TestTask(
                test_id="performance_test",
                test_name="Performance Test",
                test_function=performance_test,
                priority=TestPriority.MEDIUM,
                timeout=10.0
            ),
            TestTask(
                test_id="adversarial_test",
                test_name="Adversarial Test",
                test_function=adversarial_test,
                priority=TestPriority.HIGH,
                timeout=10.0
            )
        ]
        
        # Add tasks to session
        for task in tasks:
            await orchestrator.add_test_task(session_id, task)
        
        # Execute session
        results = await orchestrator.execute_session(session_id)
        
        # Verify results
        assert len(results["results"]) == 3
        assert all(result["status"] == TestStatus.COMPLETED.value for result in results["results"])
        
        # Check detector summary
        detector_summary = detector.get_detection_summary()
        assert "total_recent_attacks" in detector_summary
        
        # Check executor status
        executor_status = executor.get_system_status()
        assert "active_executions" in executor_status
    
    async def test_error_handling_and_recovery(self, orchestrator):
        """Test error handling and recovery"""
        # Create session
        session_id = await orchestrator.create_session("Error Test Session")
        
        # Test that times out
        async def timeout_test():
            await asyncio.sleep(10)  # Longer than timeout
            return {"result": "should_not_reach"}
        
        # Test that fails
        def failing_test():
            raise ValueError("Test failure simulation")
        
        # Test that succeeds
        async def success_test():
            return {"result": "success"}
        
        # Create tasks
        tasks = [
            TestTask(
                test_id="timeout_test",
                test_name="Timeout Test",
                test_function=timeout_test,
                timeout=1.0  # Short timeout
            ),
            TestTask(
                test_id="failing_test",
                test_name="Failing Test",
                test_function=failing_test
            ),
            TestTask(
                test_id="success_test",
                test_name="Success Test",
                test_function=success_test
            )
        ]
        
        # Add tasks
        for task in tasks:
            await orchestrator.add_test_task(session_id, task)
        
        # Execute session
        results = await orchestrator.execute_session(session_id)
        
        # Verify handling
        assert len(results["results"]) == 3
        
        # Check specific results
        result_by_id = {r["test_id"]: r for r in results["results"]}
        
        assert result_by_id["timeout_test"]["status"] == TestStatus.TIMEOUT.value
        assert result_by_id["failing_test"]["status"] == TestStatus.FAILED.value
        assert result_by_id["success_test"]["status"] == TestStatus.COMPLETED.value
    
    async def test_resource_monitoring_and_limits(self, executor):
        """Test resource monitoring and limits"""
        # Get initial resource availability
        initial_availability = executor.resource_monitor.get_resource_availability()
        assert "cpu_available_percent" in initial_availability
        assert "memory_available_gb" in initial_availability
        
        # Execute resource-intensive test
        def resource_intensive_test():
            # Allocate some memory
            data = [0] * (10 * 1024 * 1024)  # 10MB
            
            # Do some CPU work
            result = sum(i * i for i in range(100000))
            
            return f"Resource test completed: {result}"
        
        # Execute with monitoring
        context = await executor.execute_test(
            resource_intensive_test,
            execution_mode=ExecutionMode.PROCESS,
            resource_quota=ResourceQuota(cpu_cores=1.0, memory_mb=50),
            timeout=30.0
        )
        
        assert context.exit_code == 0
        assert "Resource test completed" in context.stdout
        
        # Check final resource availability
        final_availability = executor.resource_monitor.get_resource_availability()
        assert "cpu_available_percent" in final_availability
        assert "memory_available_gb" in final_availability
    
    async def test_event_system_integration(self, orchestrator, detector):
        """Test event system integration"""
        events_received = []
        
        async def event_handler(event: Event):
            events_received.append(event)
        
        # Subscribe to events
        orchestrator.event_bus.subscribe("session_created", event_handler)
        orchestrator.event_bus.subscribe("task_completed", event_handler)
        detector.event_bus.subscribe("adversarial_attack_detected", event_handler)
        
        # Create session and execute test
        session_id = await orchestrator.create_session("Event Test Session")
        
        async def simple_test():
            return {"result": "success"}
        
        task = TestTask(
            test_id="event_test",
            test_name="Event Test",
            test_function=simple_test
        )
        
        await orchestrator.add_test_task(session_id, task)
        results = await orchestrator.execute_session(session_id)
        
        # Wait for events to be processed
        await asyncio.sleep(0.1)
        
        # Verify events were received
        assert len(events_received) >= 2  # At least session_created and task_completed
        event_types = [event.type for event in events_received]
        assert "session_created" in event_types
        assert "task_completed" in event_types


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for the testing infrastructure"""
    
    @pytest.mark.benchmark
    async def test_orchestrator_performance(self, orchestrator):
        """Benchmark orchestrator performance"""
        session_id = await orchestrator.create_session("Performance Test")
        
        # Create many small tasks
        tasks = []
        for i in range(100):
            task = TestTask(
                test_id=f"perf_test_{i}",
                test_name=f"Performance Test {i}",
                test_function=lambda: {"result": f"test_{i}"},
                priority=TestPriority.MEDIUM
            )
            tasks.append(task)
        
        # Measure task addition time
        start_time = time.time()
        for task in tasks:
            await orchestrator.add_test_task(session_id, task)
        addition_time = time.time() - start_time
        
        # Measure execution time
        start_time = time.time()
        results = await orchestrator.execute_session(session_id)
        execution_time = time.time() - start_time
        
        # Verify performance
        assert addition_time < 5.0  # Should add 100 tasks in < 5 seconds
        assert execution_time < 30.0  # Should execute 100 tasks in < 30 seconds
        assert len(results["results"]) == 100
        
        print(f"Task addition time: {addition_time:.2f}s")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"Throughput: {100 / execution_time:.2f} tests/second")
    
    @pytest.mark.benchmark
    async def test_detector_performance(self, detector):
        """Benchmark detector performance"""
        # Create model for testing
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Benchmark model analysis
        start_time = time.time()
        for i in range(10):
            attacks = await detector.analyze_model(
                model, 
                f"perf_model_{i}", 
                {"accuracy": 0.95}
            )
        model_analysis_time = time.time() - start_time
        
        # Benchmark gradient analysis
        gradients = {
            f"layer_{i}": torch.randn(100, 100) * 0.01
            for i in range(10)
        }
        
        start_time = time.time()
        for i in range(100):
            attacks = await detector.analyze_gradients(gradients, f"perf_agent_{i}")
        gradient_analysis_time = time.time() - start_time
        
        # Verify performance
        assert model_analysis_time < 10.0  # < 10 seconds for 10 models
        assert gradient_analysis_time < 5.0  # < 5 seconds for 100 gradient analyses
        
        print(f"Model analysis time: {model_analysis_time:.2f}s")
        print(f"Gradient analysis time: {gradient_analysis_time:.2f}s")
    
    @pytest.mark.benchmark
    async def test_executor_performance(self, executor):
        """Benchmark executor performance"""
        def quick_test(value: int) -> int:
            return value * 2
        
        # Benchmark single execution
        start_time = time.time()
        context = await executor.execute_test(
            quick_test,
            test_args=(42,),
            execution_mode=ExecutionMode.ASYNC
        )
        single_execution_time = time.time() - start_time
        
        # Benchmark batch execution
        test_functions = [lambda i=i: quick_test(i) for i in range(50)]
        
        start_time = time.time()
        batch_results = await executor.execute_batch(
            test_functions,
            execution_mode=ExecutionMode.THREAD,
            max_parallel=5
        )
        batch_execution_time = time.time() - start_time
        
        # Verify performance
        assert single_execution_time < 0.1  # < 100ms for single execution
        assert batch_execution_time < 10.0  # < 10 seconds for 50 tests
        assert len(batch_results) == 50
        
        print(f"Single execution time: {single_execution_time:.3f}s")
        print(f"Batch execution time: {batch_execution_time:.2f}s")
        print(f"Batch throughput: {50 / batch_execution_time:.2f} tests/second")


# Demo function
async def run_integration_demo():
    """Run comprehensive integration demo"""
    print("=== ADVERSARIAL TESTING INFRASTRUCTURE DEMO ===")
    
    # Initialize components
    orchestrator = TestOrchestrator(max_parallel_tests=5)
    detector = AdversarialDetector(orchestrator.event_bus)
    executor = ParallelExecutor(max_workers=3, enable_containers=False)
    dashboard = TestingDashboard(orchestrator, port=5003)
    
    try:
        # Start components
        await detector.start_monitoring()
        await executor.start()
        
        print("\n1. Testing Orchestrator...")
        session_id = await orchestrator.create_session("Demo Session")
        
        # Define test functions
        async def security_test():
            await asyncio.sleep(0.5)
            return {"security_status": "passed"}
        
        def performance_test():
            time.sleep(0.3)
            return {"performance_score": 0.92}
        
        async def adversarial_test():
            model = nn.Linear(10, 1)
            gradients = {"weight": torch.randn(1, 10) * 0.01, "bias": torch.randn(1) * 0.01}
            attacks = await detector.analyze_gradients(gradients, "demo_agent")
            return {"attacks_detected": len(attacks)}
        
        # Add tasks
        tasks = [
            TestTask("security", "Security Test", security_test, priority=TestPriority.HIGH),
            TestTask("performance", "Performance Test", performance_test, priority=TestPriority.MEDIUM),
            TestTask("adversarial", "Adversarial Test", adversarial_test, priority=TestPriority.HIGH)
        ]
        
        for task in tasks:
            await orchestrator.add_test_task(session_id, task)
        
        # Execute session
        results = await orchestrator.execute_session(session_id)
        print(f"   Executed {len(results['results'])} tests")
        print(f"   Success rate: {results['metrics']['success_rate']:.1f}%")
        
        print("\n2. Testing Adversarial Detector...")
        # Create and analyze model
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        attacks = await detector.analyze_model(model, "demo_model", {"accuracy": 0.95})
        print(f"   Model analysis: {len(attacks)} attacks detected")
        
        # Analyze gradients
        gradients = {
            "layer1.weight": torch.randn(5, 10) * 0.01,
            "layer1.bias": torch.randn(5) * 0.01,
            "layer2.weight": torch.randn(1, 5) * 0.01,
            "layer2.bias": torch.randn(1) * 0.01
        }
        attacks = await detector.analyze_gradients(gradients, "demo_agent")
        print(f"   Gradient analysis: {len(attacks)} attacks detected")
        
        print("\n3. Testing Parallel Executor...")
        # Execute tests in parallel
        def cpu_test(duration: int):
            time.sleep(duration)
            return f"CPU test {duration}s"
        
        test_functions = [lambda d=d: cpu_test(d) for d in [1, 2, 1, 2, 1]]
        batch_results = await executor.execute_batch(
            test_functions,
            execution_mode=ExecutionMode.THREAD,
            max_parallel=3
        )
        print(f"   Parallel execution: {len(batch_results)} tests completed")
        successful = sum(1 for r in batch_results if r.exit_code == 0)
        print(f"   Success rate: {successful/len(batch_results)*100:.1f}%")
        
        print("\n4. Testing Dashboard...")
        # Generate report
        report = dashboard.generate_report(session_id)
        print(f"   Report generated: {report['summary']['total_tests']} tests")
        print(f"   Average execution time: {report['summary']['average_execution_time']:.2f}s")
        
        print("\n5. System Status...")
        # Get system metrics
        orchestrator_metrics = await orchestrator.get_system_metrics()
        detector_summary = detector.get_detection_summary()
        executor_status = executor.get_system_status()
        
        print(f"   Orchestrator: {orchestrator_metrics['active_sessions']} active sessions")
        print(f"   Detector: {detector_summary['total_recent_attacks']} recent attacks")
        print(f"   Executor: {executor_status['active_executions']} active executions")
        
        print("\n=== DEMO COMPLETED SUCCESSFULLY ===")
        print("🎯 All infrastructure components operational")
        print("🚀 Ready for production adversarial testing")
        
    finally:
        # Cleanup
        await detector.stop_monitoring()
        await executor.stop()


if __name__ == "__main__":
    asyncio.run(run_integration_demo())