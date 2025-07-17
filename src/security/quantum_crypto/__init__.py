"""
Post-Quantum Cryptography Library
================================

NIST-approved post-quantum cryptographic algorithms implementation for 
future-proof security against quantum computing threats.

Features:
- CRYSTALS-Kyber (ML-KEM) - Key Encapsulation Mechanism
- CRYSTALS-Dilithium (ML-DSA) - Digital Signature Algorithm
- SPHINCS+ (SLH-DSA) - Hash-based Digital Signatures
- Quantum-safe random number generation
- Cryptographic agility framework
- Hybrid classical/quantum cryptography

Author: Agent Gamma - Post-Quantum Cryptography Specialist
Version: 1.0.0 - Production Ready
"""

from .core import (
    PostQuantumCryptoProvider,
    QuantumSafeRNG,
    CryptographicAgility,
    PQCAlgorithm,
    SecurityLevel
)

from .kyber import (
    CrystalsKyber,
    ML_KEM_512,
    ML_KEM_768,
    ML_KEM_1024
)

from .dilithium import (
    CrystalsDilithium,
    ML_DSA_44,
    ML_DSA_65,
    ML_DSA_87
)

from .sphincs import (
    SphincsPlus,
    SLH_DSA_128S,
    SLH_DSA_192S,
    SLH_DSA_256S
)

from .migration import (
    CryptoMigrationManager,
    HybridCryptoProvider
)

from .validation import (
    QuantumThreatAssessment,
    PostQuantumValidator
)

__all__ = [
    # Core components
    'PostQuantumCryptoProvider',
    'QuantumSafeRNG',
    'CryptographicAgility',
    'PQCAlgorithm',
    'SecurityLevel',
    
    # CRYSTALS-Kyber (ML-KEM)
    'CrystalsKyber',
    'ML_KEM_512',
    'ML_KEM_768',
    'ML_KEM_1024',
    
    # CRYSTALS-Dilithium (ML-DSA)
    'CrystalsDilithium',
    'ML_DSA_44',
    'ML_DSA_65',
    'ML_DSA_87',
    
    # SPHINCS+ (SLH-DSA)
    'SphincsPlus',
    'SLH_DSA_128S',
    'SLH_DSA_192S',
    'SLH_DSA_256S',
    
    # Migration and hybrid
    'CryptoMigrationManager',
    'HybridCryptoProvider',
    
    # Validation
    'QuantumThreatAssessment',
    'PostQuantumValidator'
]

# Version information
__version__ = '1.0.0'
__author__ = 'Agent Gamma - Post-Quantum Cryptography Specialist'
__description__ = 'NIST-approved post-quantum cryptographic algorithms'