"""
Contract-Driven Testing Framework
=================================

Consumer-driven contract testing framework for service interactions.
Implements Pact-style contract testing with versioning and validation.

Key Components:
- ContractRegistry: Manages contract definitions and versions
- ContractValidator: Validates service interactions against contracts
- ContractGenerator: Generates contracts from service interactions
- ContractVersioning: Handles contract evolution and compatibility
"""

from .contract_registry import ContractRegistry
from .contract_validator import ContractValidator
from .contract_generator import ContractGenerator
from .contract_versioning import ContractVersioning
from .service_interaction_tester import ServiceInteractionTester

__all__ = [
    'ContractRegistry',
    'ContractValidator',
    'ContractGenerator',
    'ContractVersioning',
    'ServiceInteractionTester'
]