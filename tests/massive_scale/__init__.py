"""
Massive Scale Testing Architecture - Phase 3A
Agent Epsilon: Production Performance Validation

Million TPS Testing Capability:
- Distributed load generation across multiple nodes
- Real-time performance monitoring and alerting
- Automatic scalability validation
- Resource usage optimization

Components:
- Million TPS simulation framework
- Distributed testing infrastructure
- Auto-scaling test validation
- Performance degradation detection
"""

from .distributed_load_generator import DistributedLoadGenerator
from .million_tps_simulator import MillionTPSSimulator
from .scalability_validator import ScalabilityValidator
from .performance_monitor import MassiveScalePerformanceMonitor

__all__ = [
    "DistributedLoadGenerator",
    "MillionTPSSimulator", 
    "ScalabilityValidator",
    "MassiveScalePerformanceMonitor"
]