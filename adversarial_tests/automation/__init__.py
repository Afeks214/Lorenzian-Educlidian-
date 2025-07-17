#!/usr/bin/env python3
"""
🤖 AGENT EPSILON MISSION: Automation Framework Package
Comprehensive automation and production readiness framework for adversarial testing.

This package provides:
- Continuous adversarial testing automation
- Security certification framework
- Production readiness validation
- Automated reporting system
- Integrated automation pipeline
"""

from .continuous_testing import ContinuousTestingEngine
from .security_certification import SecurityCertificationFramework
from .production_validator import ProductionReadinessValidator
from .reporting_system import AutomatedReportingSystem

__version__ = "1.0.0"
__author__ = "Agent Epsilon"
__description__ = "Automated Security Testing and Production Readiness Framework"

__all__ = [
    'ContinuousTestingEngine',
    'SecurityCertificationFramework', 
    'ProductionReadinessValidator',
    'AutomatedReportingSystem',
    'AutomationPipeline'
]

# Import the main automation pipeline
from .automation_pipeline import AutomationPipeline