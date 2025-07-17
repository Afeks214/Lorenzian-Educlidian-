"""
Adversarial Testing Infrastructure

This module provides comprehensive testing infrastructure for adversarial 
attack detection, model validation, and parallel test execution.

Components:
- test_orchestrator: Async test execution framework with parallel testing
- testing_dashboard: Real-time monitoring and analytics dashboard
- adversarial_detector: Model poisoning and gradient manipulation detection
- parallel_executor: Concurrent test execution with resource management

Example Usage:
    from adversarial_tests.infrastructure import TestOrchestrator, AdversarialDetector
    
    # Create orchestrator
    orchestrator = TestOrchestrator(max_parallel_tests=5)
    
    # Create adversarial detector
    detector = AdversarialDetector(orchestrator.event_bus)
    
    # Start monitoring
    await detector.start_monitoring()
    
    # Execute tests
    session_id = await orchestrator.create_session("Security Tests")
    # ... add test tasks ...
    results = await orchestrator.execute_session(session_id)
"""

# Core components
from .test_orchestrator import (
    TestOrchestrator, 
    TestTask, 
    TestSession, 
    TestStatus, 
    TestPriority,
    ResourceManager
)

# Optional dashboard components (requires flask)
try:
    from .testing_dashboard import (
        TestingDashboard,
        TestMetrics,
        SessionMetrics,
        MetricsDatabase
    )
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    # Create placeholder classes
    class TestingDashboard:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flask is required for TestingDashboard. Install with: pip install flask flask-socketio")
    
    class TestMetrics:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flask is required for TestMetrics. Install with: pip install flask flask-socketio")
    
    class SessionMetrics:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flask is required for SessionMetrics. Install with: pip install flask flask-socketio")
    
    class MetricsDatabase:
        def __init__(self, *args, **kwargs):
            raise ImportError("Flask is required for MetricsDatabase. Install with: pip install flask flask-socketio")

from .adversarial_detector import (
    AdversarialDetector,
    AttackSignature,
    AttackType,
    ThreatLevel,
    ModelFingerprint,
    GradientMonitor,
    ModelIntegrityChecker,
    ByzantineDetector
)

from .parallel_executor import (
    ParallelExecutor,
    ExecutionContext,
    ExecutionMode,
    ResourceQuota,
    WorkerNode,
    ResourceMonitor,
    ContainerManager,
    LoadBalancer
)

__all__ = [
    # Core infrastructure
    'TestOrchestrator',
    'TestingDashboard', 
    'AdversarialDetector',
    'ParallelExecutor',
    
    # Test orchestrator components
    'TestTask',
    'TestSession',
    'TestStatus',
    'TestPriority',
    'ResourceManager',
    
    # Dashboard components
    'TestMetrics',
    'SessionMetrics',
    'MetricsDatabase',
    
    # Detector components
    'AttackSignature',
    'AttackType',
    'ThreatLevel',
    'ModelFingerprint',
    'GradientMonitor',
    'ModelIntegrityChecker',
    'ByzantineDetector',
    
    # Executor components
    'ExecutionContext',
    'ExecutionMode',
    'ResourceQuota',
    'WorkerNode',
    'ResourceMonitor',
    'ContainerManager',
    'LoadBalancer'
]

__version__ = '1.0.0'
__author__ = 'Adversarial Testing Infrastructure Team'
__description__ = 'Comprehensive adversarial testing infrastructure for trading systems'