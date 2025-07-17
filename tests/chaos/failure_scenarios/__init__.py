"""
Failure Scenarios Package
========================

Comprehensive failure scenarios for chaos engineering testing.
Contains predefined failure scenarios for various system components.
"""

from .database_failures import *
from .network_failures import *
from .memory_failures import *
from .cpu_failures import *
from .disk_failures import *
from .service_failures import *
from .coordination_failures import *
from .byzantine_failures import *

__all__ = [
    'DatabaseFailureScenarios',
    'NetworkFailureScenarios', 
    'MemoryFailureScenarios',
    'CPUFailureScenarios',
    'DiskFailureScenarios',
    'ServiceFailureScenarios',
    'CoordinationFailureScenarios',
    'ByzantineFailureScenarios'
]