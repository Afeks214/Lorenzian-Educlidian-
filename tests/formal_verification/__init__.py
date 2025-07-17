"""
Formal Verification Framework
============================

World-class formal verification system for trading algorithms and critical systems.
Implements TLA+ specifications, Coq mathematical proofs, and theorem proving.

Phase 2A: Formal Verification Implementation
- TLA+ specifications for concurrent algorithms
- Coq/Rocq mathematical proof validation
- Theorem proving framework for algorithmic correctness
- Automated formal verification pipeline

Author: Agent Gamma - Formal Verification & Defense-Grade Security
Mission: Phase 2A Implementation
"""

from .tla_plus_framework import TLAPlusSpecificationFramework
from .coq_proof_system import CoqProofValidationSystem
from .theorem_prover import TheoremProvingFramework
from .verification_pipeline import AutomatedVerificationPipeline

__all__ = [
    'TLAPlusSpecificationFramework',
    'CoqProofValidationSystem', 
    'TheoremProvingFramework',
    'AutomatedVerificationPipeline'
]