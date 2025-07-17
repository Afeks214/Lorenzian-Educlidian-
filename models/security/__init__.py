"""
Security Components Module for Tactical MARL Architecture

This module provides cryptographically secure components to eliminate:
- CVE-2025-TACTICAL-001: Dynamic Attention Vulnerability
- CVE-2025-TACTICAL-002: Temperature Scaling Exploit
- CVE-2025-TACTICAL-003: Memory Race Conditions
- CVE-2025-TACTICAL-004: Fixed Kernel Vulnerabilities
- CVE-2025-TACTICAL-005: Initialization Attacks
"""

from .secure_attention import SecureAttentionSystem
from .adaptive_temperature import AdaptiveTemperatureScaling
from .memory_security import SecureMemoryManager
from .multi_scale_kernels import AdaptiveMultiScaleKernels
from .secure_initialization import SecureInitializer
from .attack_detection import RealTimeAttackDetector
from .crypto_validation import CryptographicValidator

__all__ = [
    'SecureAttentionSystem',
    'AdaptiveTemperatureScaling', 
    'SecureMemoryManager',
    'AdaptiveMultiScaleKernels',
    'SecureInitializer',
    'RealTimeAttackDetector',
    'CryptographicValidator'
]