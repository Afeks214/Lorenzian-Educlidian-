"""
GrandModel Comprehensive Testing Framework
==========================================

AGENT 19 MISSION: Comprehensive Testing & Validation Implementation

This module provides a complete testing and validation framework for the GrandModel
trading system, including unit tests, integration tests, performance testing,
CI/CD pipeline integration, and quality assurance monitoring.
"""

# Performance Regression Detection System - Agent 3
from .performance_regression_system import (
    PerformanceRegressionDetector,
    PerformanceBenchmark,
    RegressionResult,
    PerformanceBudget,
    performance_detector
)

from .pytest_benchmark_integration import (
    BenchmarkIntegrationPlugin,
    setup_performance_budget,
    benchmark_with_regression_detection,
    configure_performance_budgets
)

from .performance_alerting_system import (
    PerformanceAlertingSystem,
    AlertChannel,
    AlertRule,
    Alert,
    alerting_system
)

from .ci_performance_gates import (
    CIPerformanceGates,
    BuildInfo,
    PerformanceGate,
    GateResult,
    BisectionResult,
    ci_gates
)

from .performance_dashboard import (
    PerformanceDashboard,
    dashboard
)

# Test Reporting System - Agent 6
from .test_reporting_system import TestReportingSystem

# Agent 19 Extensions - Comprehensive Testing Framework
from .framework import TestingFramework
from .validation import ValidationSuite
from .performance import PerformanceTester
from .quality import QualityAssurance
from .ci_cd import CIPipeline
from .orchestrator import TestOrchestrator

__all__ = [
    # Core system
    'PerformanceRegressionDetector',
    'PerformanceBenchmark',
    'RegressionResult',
    'PerformanceBudget',
    'performance_detector',
    
    # pytest integration
    'BenchmarkIntegrationPlugin',
    'setup_performance_budget',
    'benchmark_with_regression_detection',
    'configure_performance_budgets',
    
    # Alerting system
    'PerformanceAlertingSystem',
    'AlertChannel',
    'AlertRule',
    'Alert',
    'alerting_system',
    
    # CI/CD gates
    'CIPerformanceGates',
    'BuildInfo',
    'PerformanceGate',
    'GateResult',
    'BisectionResult',
    'ci_gates',
    
    # Dashboard
    'PerformanceDashboard',
    'dashboard',
    
    # Test Reporting
    'TestReportingSystem',
    
    # Agent 19 Framework
    'TestingFramework',
    'ValidationSuite',
    'PerformanceTester',
    'QualityAssurance',
    'CIPipeline',
    'TestOrchestrator'
]