"""
pytest-benchmark Integration Plugin - Agent 3

This module provides seamless integration between pytest-benchmark and the 
performance regression detection system.
"""

import os
import json
import pytest
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import subprocess
import structlog

from .performance_regression_system import (
    PerformanceRegressionDetector,
    PerformanceBenchmark,
    PerformanceBudget,
    performance_detector
)

logger = structlog.get_logger()

class BenchmarkIntegrationPlugin:
    """
    Pytest plugin for integrating with performance regression detection
    """
    
    def __init__(self):
        self.detector = performance_detector
        self.git_commit = self._get_git_commit()
        self.git_branch = self._get_git_branch()
        self.environment = os.getenv('PYTEST_ENVIRONMENT', 'local')
        self.benchmark_results = []
        
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def pytest_benchmark_generate_json(self, config, benchmarks, include_data):
        """Hook called when benchmark JSON is generated"""
        for bench in benchmarks:
            # Convert pytest-benchmark result to our format
            benchmark = PerformanceBenchmark(
                test_name=bench['name'],
                timestamp=datetime.now(),
                min_time=bench['min'],
                max_time=bench['max'],
                mean_time=bench['mean'],
                median_time=bench['median'],
                stddev_time=bench['stddev'],
                rounds=bench['rounds'],
                iterations=bench['iterations'],
                git_commit=self.git_commit,
                branch=self.git_branch,
                environment=self.environment,
                additional_metrics=bench.get('stats', {})
            )
            
            # Record benchmark
            self.detector.record_benchmark(benchmark)
            self.benchmark_results.append(benchmark)
            
            # Check for regression
            regression_result = self.detector.detect_regression(benchmark)
            if regression_result and regression_result.regression_detected:
                self._handle_regression(regression_result)
    
    def _handle_regression(self, regression_result):
        """Handle detected performance regression"""
        severity = regression_result.regression_severity
        
        logger.error("Performance regression detected",
                    test_name=regression_result.test_name,
                    severity=severity,
                    current_performance=regression_result.current_performance,
                    baseline_performance=regression_result.baseline_performance,
                    recommendation=regression_result.recommendation)
        
        # Check if we should fail the test
        if severity in ['HIGH', 'CRITICAL']:
            # Add failure message to pytest
            pytest.fail(
                f"Performance regression detected in {regression_result.test_name}: "
                f"{severity} severity. Current: {regression_result.current_performance:.4f}s, "
                f"Baseline: {regression_result.baseline_performance:.4f}s. "
                f"Recommendation: {regression_result.recommendation}"
            )
    
    def pytest_runtest_teardown(self, item, nextitem):
        """Hook called after each test"""
        # Check if test had benchmarks and handle any post-test processing
        if hasattr(item, 'benchmark_results'):
            for result in item.benchmark_results:
                logger.info("Benchmark completed", 
                           test_name=result.test_name,
                           mean_time=result.mean_time)

# Pytest plugin functions
def pytest_configure(config):
    """Configure the benchmark integration plugin"""
    if not hasattr(config, '_benchmark_integration'):
        config._benchmark_integration = BenchmarkIntegrationPlugin()

def pytest_benchmark_generate_json(config, benchmarks, include_data):
    """Hook for benchmark JSON generation"""
    if hasattr(config, '_benchmark_integration'):
        config._benchmark_integration.pytest_benchmark_generate_json(
            config, benchmarks, include_data
        )

def pytest_runtest_teardown(item, nextitem):
    """Hook for test teardown"""
    if hasattr(item.config, '_benchmark_integration'):
        item.config._benchmark_integration.pytest_runtest_teardown(item, nextitem)

# Utility functions for test configuration
def setup_performance_budget(test_name: str, max_time_ms: float, max_regression_percent: float = 25.0):
    """Set up performance budget for a test"""
    budget = PerformanceBudget(
        test_name=test_name,
        max_time_ms=max_time_ms,
        max_regression_percent=max_regression_percent
    )
    performance_detector.set_performance_budget(budget)

def benchmark_with_regression_detection(benchmark_func, test_name: str):
    """Decorator to add regression detection to benchmark functions"""
    def wrapper(*args, **kwargs):
        # Run the benchmark
        result = benchmark_func(*args, **kwargs)
        
        # The actual regression detection happens in the pytest hook
        # This is mainly for manual benchmarking outside of pytest
        
        return result
    
    return wrapper

# Configuration helpers
def configure_performance_budgets():
    """Configure performance budgets for all tests"""
    budgets = {
        # Core system benchmarks
        'test_var_calculation': PerformanceBudget(
            test_name='test_var_calculation',
            max_time_ms=5.0,
            max_regression_percent=20.0
        ),
        'test_correlation_update': PerformanceBudget(
            test_name='test_correlation_update',
            max_time_ms=2.0,
            max_regression_percent=25.0
        ),
        'test_portfolio_optimization': PerformanceBudget(
            test_name='test_portfolio_optimization',
            max_time_ms=50.0,
            max_regression_percent=30.0
        ),
        
        # MARL system benchmarks
        'test_strategic_agent_inference': PerformanceBudget(
            test_name='test_strategic_agent_inference',
            max_time_ms=100.0,
            max_regression_percent=15.0
        ),
        'test_tactical_agent_inference': PerformanceBudget(
            test_name='test_tactical_agent_inference',
            max_time_ms=50.0,
            max_regression_percent=15.0
        ),
        
        # Data processing benchmarks
        'test_matrix_assembler': PerformanceBudget(
            test_name='test_matrix_assembler',
            max_time_ms=10.0,
            max_regression_percent=20.0
        ),
        'test_indicator_calculation': PerformanceBudget(
            test_name='test_indicator_calculation',
            max_time_ms=5.0,
            max_regression_percent=25.0
        ),
        
        # API benchmarks
        'test_api_response_time': PerformanceBudget(
            test_name='test_api_response_time',
            max_time_ms=200.0,
            max_regression_percent=40.0
        ),
        
        # XAI system benchmarks
        'test_explanation_generation': PerformanceBudget(
            test_name='test_explanation_generation',
            max_time_ms=500.0,
            max_regression_percent=30.0
        )
    }
    
    for budget in budgets.values():
        performance_detector.set_performance_budget(budget)
    
    logger.info("Performance budgets configured", 
               budget_count=len(budgets))

# Initialize budgets on import
configure_performance_budgets()