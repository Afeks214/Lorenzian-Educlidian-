"""
Immutable Audit Infrastructure for Advanced XAI
Agent Epsilon: Advanced XAI Implementation Specialist

This module implements blockchain-based immutable audit trails for trading decisions,
ensuring tamper-proof logging and cryptographic integrity for regulatory compliance.
"""

from .blockchain_audit import BlockchainAuditSystem, AuditBlock, AuditTransaction, AuditChain
from .cryptographic_integrity import CryptographicIntegrity, SignatureValidator, HashValidator
from .distributed_consensus import DistributedAuditConsensus, ConsensusNode, ConsensusProtocol

__all__ = [
    'BlockchainAuditSystem',
    'AuditBlock',
    'AuditTransaction',
    'AuditChain',
    'CryptographicIntegrity',
    'SignatureValidator',
    'HashValidator',
    'DistributedAuditConsensus',
    'ConsensusNode',
    'ConsensusProtocol'
]