#!/usr/bin/env python3
"""
Automated Testing Pipeline for Training Infrastructure
Comprehensive testing framework for models, data, and system components
"""

import os
import sys
import json
import time
import logging
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import unittest
import pytest
import torch
import numpy as np
from contextlib import contextmanager

class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    BENCHMARK = "benchmark"
    SYSTEM = "system"
    SMOKE = "smoke"

class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestResult:
    """Test result data"""
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    error_message: Optional[str] = None
    output: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    tests: List[str]
    test_type: TestType
    timeout: int = 300
    parallel: bool = False
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class TestPipeline:
    """Automated testing pipeline"""
    
    def __init__(self, test_dir: str = "/home/QuantNova/GrandModel/colab/infrastructure/testing"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Test tracking
        self.test_results: List[TestResult] = []
        self.test_suites: List[TestSuite] = []
        self.running_tests: Dict[str, threading.Thread] = {}
        
        # Setup test directories
        self.setup_test_directories()
        
        # Load test configuration
        self.load_test_suites()
    
    def setup_test_directories(self):
        """Setup test directory structure"""
        for test_type in TestType:
            (self.test_dir / test_type.value).mkdir(exist_ok=True)
        
        # Create reports directory
        (self.test_dir / "reports").mkdir(exist_ok=True)
        
        # Create fixtures directory
        (self.test_dir / "fixtures").mkdir(exist_ok=True)
    
    def load_test_suites(self):
        """Load test suite configurations"""
        # Define default test suites
        default_suites = [
            TestSuite(
                name="unit_tests",
                tests=["test_model_components", "test_data_pipeline", "test_utilities"],
                test_type=TestType.UNIT,
                timeout=120,
                parallel=True
            ),
            TestSuite(
                name="integration_tests",
                tests=["test_training_pipeline", "test_inference_pipeline", "test_backup_system"],
                test_type=TestType.INTEGRATION,
                timeout=300,
                parallel=False,
                dependencies=["unit_tests"]
            ),
            TestSuite(
                name="performance_tests",
                tests=["test_training_speed", "test_memory_usage", "test_gpu_utilization"],
                test_type=TestType.PERFORMANCE,
                timeout=600,
                parallel=True,
                dependencies=["unit_tests"]
            ),
            TestSuite(
                name="smoke_tests",
                tests=["test_basic_functionality", "test_system_health"],
                test_type=TestType.SMOKE,
                timeout=60,
                parallel=False
            )
        ]
        
        self.test_suites = default_suites
    
    def create_test_file(self, test_name: str, test_type: TestType, test_content: str):
        """Create a test file"""
        test_file = self.test_dir / test_type.value / f"{test_name}.py"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        self.logger.info(f"Created test file: {test_file}")
    
    def run_test(self, test_name: str, test_type: TestType, timeout: int = 300) -> TestResult:
        """Run a single test"""
        start_time = time.time()
        
        try:
            # Find test file
            test_file = self.test_dir / test_type.value / f"{test_name}.py"
            
            if not test_file.exists():
                return TestResult(
                    test_name=test_name,
                    test_type=test_type,
                    status=TestStatus.ERROR,
                    duration=0.0,
                    error_message=f"Test file not found: {test_file}"
                )
            
            # Run test with pytest
            cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.test_dir)
            )
            
            duration = time.time() - start_time
            
            # Determine status
            if result.returncode == 0:
                status = TestStatus.PASSED
            elif result.returncode == 5:  # No tests collected
                status = TestStatus.SKIPPED
            else:
                status = TestStatus.FAILED
            
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=status,
                duration=duration,
                error_message=result.stderr if result.stderr else None,
                output=result.stdout
            )
        
        except subprocess.TimeoutExpired:
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=time.time() - start_time,
                error_message=f"Test timed out after {timeout}s"
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def run_test_suite(self, suite: TestSuite) -> List[TestResult]:
        """Run a test suite"""
        self.logger.info(f"Running test suite: {suite.name}")
        
        results = []
        
        if suite.parallel:
            # Run tests in parallel
            threads = []
            thread_results = {}
            
            for test_name in suite.tests:
                def run_test_thread(name=test_name):
                    thread_results[name] = self.run_test(name, suite.test_type, suite.timeout)
                
                thread = threading.Thread(target=run_test_thread)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Collect results
            for test_name in suite.tests:
                results.append(thread_results[test_name])
        
        else:
            # Run tests sequentially
            for test_name in suite.tests:
                result = self.run_test(test_name, suite.test_type, suite.timeout)
                results.append(result)
        
        return results
    
    def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all test suites"""
        all_results = {}
        
        # Sort suites by dependencies
        sorted_suites = self._sort_suites_by_dependencies()
        
        for suite in sorted_suites:
            # Check if dependencies passed
            if not self._check_dependencies(suite, all_results):
                self.logger.warning(f"Skipping suite {suite.name} due to failed dependencies")
                continue
            
            results = self.run_test_suite(suite)
            all_results[suite.name] = results
            
            # Update test results
            self.test_results.extend(results)
        
        return all_results
    
    def _sort_suites_by_dependencies(self) -> List[TestSuite]:
        """Sort test suites by dependencies"""
        sorted_suites = []
        remaining_suites = self.test_suites.copy()
        
        while remaining_suites:
            # Find suites with no unmet dependencies
            ready_suites = []
            for suite in remaining_suites:
                if all(dep in [s.name for s in sorted_suites] for dep in suite.dependencies):
                    ready_suites.append(suite)
            
            if not ready_suites:
                # Circular dependency or missing dependency
                self.logger.warning("Circular dependency detected in test suites")
                ready_suites = remaining_suites
            
            # Add ready suites to sorted list
            sorted_suites.extend(ready_suites)
            
            # Remove from remaining
            for suite in ready_suites:
                remaining_suites.remove(suite)
        
        return sorted_suites
    
    def _check_dependencies(self, suite: TestSuite, all_results: Dict[str, List[TestResult]]) -> bool:
        """Check if suite dependencies passed"""
        for dep in suite.dependencies:
            if dep not in all_results:
                return False
            
            # Check if all tests in dependency suite passed
            dep_results = all_results[dep]
            if any(result.status == TestStatus.FAILED for result in dep_results):
                return False
        
        return True
    
    def benchmark_model(self, model: torch.nn.Module, input_shape: tuple, 
                       num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        throughput = num_iterations / total_time
        
        return {
            'total_time': total_time,
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'memory_usage': torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        }
    
    def performance_test(self, test_name: str, test_func: Callable, 
                        expected_time: float = None, 
                        expected_memory: float = None) -> TestResult:
        """Run performance test"""
        start_time = time.time()
        
        try:
            # Monitor memory before
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            
            # Run test function
            result = test_func()
            
            # Calculate metrics
            duration = time.time() - start_time
            
            metrics = {'duration': duration}
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
                metrics['peak_memory_gb'] = peak_memory
            
            # Check performance criteria
            status = TestStatus.PASSED
            error_message = None
            
            if expected_time and duration > expected_time:
                status = TestStatus.FAILED
                error_message = f"Performance test failed: {duration:.2f}s > {expected_time:.2f}s"
            
            if expected_memory and metrics.get('peak_memory_gb', 0) > expected_memory:
                status = TestStatus.FAILED
                error_message = f"Memory test failed: {metrics['peak_memory_gb']:.2f}GB > {expected_memory:.2f}GB"
            
            return TestResult(
                test_name=test_name,
                test_type=TestType.PERFORMANCE,
                status=status,
                duration=duration,
                error_message=error_message,
                metrics=metrics
            )
        
        except Exception as e:
            return TestResult(
                test_name=test_name,
                test_type=TestType.PERFORMANCE,
                status=TestStatus.ERROR,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in self.test_results if r.status == TestStatus.FAILED])
        error_tests = len([r for r in self.test_results if r.status == TestStatus.ERROR])
        skipped_tests = len([r for r in self.test_results if r.status == TestStatus.SKIPPED])
        
        # Calculate metrics by test type
        type_metrics = {}
        for test_type in TestType:
            type_results = [r for r in self.test_results if r.test_type == test_type]
            if type_results:
                type_metrics[test_type.value] = {
                    'total': len(type_results),
                    'passed': len([r for r in type_results if r.status == TestStatus.PASSED]),
                    'failed': len([r for r in type_results if r.status == TestStatus.FAILED]),
                    'avg_duration': np.mean([r.duration for r in type_results])
                }
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'skipped': skipped_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'total_duration': sum(r.duration for r in self.test_results)
            },
            'type_metrics': type_metrics,
            'failed_tests': [
                {
                    'name': r.test_name,
                    'type': r.test_type.value,
                    'error': r.error_message,
                    'duration': r.duration
                }
                for r in self.test_results if r.status == TestStatus.FAILED
            ],
            'performance_metrics': [
                {
                    'name': r.test_name,
                    'duration': r.duration,
                    'metrics': r.metrics
                }
                for r in self.test_results 
                if r.test_type == TestType.PERFORMANCE and r.metrics
            ],
            'timestamp': time.time()
        }
    
    def save_test_report(self, filename: str = None):
        """Save test report to file"""
        if filename is None:
            filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = self.test_dir / "reports" / filename
        report = self.generate_test_report()
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Test report saved to: {report_path}")
        return report_path
    
    def create_default_tests(self):
        """Create default test files"""
        # Unit test example
        unit_test_content = '''#!/usr/bin/env python3
"""Unit tests for model components"""

import unittest
import torch
import torch.nn as nn

class TestModelComponents(unittest.TestCase):
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = nn.Linear(10, 1)
        self.assertIsNotNone(model)
        self.assertEqual(model.in_features, 10)
        self.assertEqual(model.out_features, 1)
    
    def test_model_forward(self):
        """Test model forward pass"""
        model = nn.Linear(10, 1)
        x = torch.randn(32, 10)
        y = model(x)
        self.assertEqual(y.shape, (32, 1))
    
    def test_model_training(self):
        """Test model can be trained"""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
        self.assertIsNotNone(loss.item())

if __name__ == '__main__':
    unittest.main()
'''
        
        self.create_test_file("test_model_components", TestType.UNIT, unit_test_content)
        
        # Performance test example
        performance_test_content = '''#!/usr/bin/env python3
"""Performance tests"""

import time
import torch
import torch.nn as nn

def test_training_speed():
    """Test training speed"""
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Test training speed
    start_time = time.time()
    for _ in range(100):
        x = torch.randn(32, 1000)
        y = torch.randn(32, 1)
        
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    duration = time.time() - start_time
    
    # Should complete in reasonable time
    assert duration < 30.0, f"Training too slow: {duration:.2f}s"

def test_inference_speed():
    """Test inference speed"""
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(1000):
            x = torch.randn(1, 1000)
            _ = model(x)
    
    duration = time.time() - start_time
    
    # Should complete in reasonable time
    assert duration < 5.0, f"Inference too slow: {duration:.2f}s"

if __name__ == '__main__':
    test_training_speed()
    test_inference_speed()
    print("All performance tests passed!")
'''
        
        self.create_test_file("test_training_speed", TestType.PERFORMANCE, performance_test_content)
        
        self.logger.info("Default test files created")
    
    @contextmanager
    def test_context(self, test_name: str):
        """Context manager for test execution"""
        start_time = time.time()
        try:
            yield
            self.logger.info(f"Test {test_name} completed successfully in {time.time() - start_time:.2f}s")
        except Exception as e:
            self.logger.error(f"Test {test_name} failed after {time.time() - start_time:.2f}s: {e}")
            raise

# Factory functions
def create_test_pipeline(test_dir: str = None) -> TestPipeline:
    """Create test pipeline with default configuration"""
    return TestPipeline(test_dir or "/home/QuantNova/GrandModel/colab/infrastructure/testing")

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test pipeline
    pipeline = create_test_pipeline()
    
    # Create default tests
    pipeline.create_default_tests()
    
    # Run all tests
    results = pipeline.run_all_tests()
    
    # Generate and save report
    report = pipeline.generate_test_report()
    print(f"Test Results: {json.dumps(report['summary'], indent=2)}")
    
    # Save report
    report_path = pipeline.save_test_report()
    print(f"Report saved to: {report_path}")