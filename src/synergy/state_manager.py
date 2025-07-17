"""
Synergy State Management System
==============================

This module provides state management for synergy detection including:
- Synergy lifecycle tracking
- Signal invalidation logic
- Confidence scoring system
- Integration handoff coordination
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import structlog

from .base import Signal, SynergyPattern

logger = structlog.get_logger()


class SynergyState(Enum):
    """Synergy lifecycle states."""
    BUILDING = "building"           # Chain is being built
    COMPLETE = "complete"           # Chain is complete and valid
    INVALIDATED = "invalidated"     # Chain has been invalidated
    CONSUMED = "consumed"           # Chain has been consumed by execution
    EXPIRED = "expired"            # Chain has expired due to timeout


@dataclass
class SynergyConfidence:
    """Confidence scoring for synergy patterns."""
    base_confidence: float = 1.0        # Base confidence (always 1.0 for hard rules)
    strength_factor: float = 0.0        # Average signal strength
    timing_factor: float = 0.0          # Timing coherence factor
    coherence_factor: float = 0.0       # Direction/trend coherence
    final_confidence: float = 0.0       # Final computed confidence
    
    def compute_final_confidence(self) -> float:
        """Compute final confidence score."""
        # Weighted combination of factors
        self.final_confidence = (
            self.base_confidence * 0.4 +
            self.strength_factor * 0.3 +
            self.timing_factor * 0.2 +
            self.coherence_factor * 0.1
        )
        return min(1.0, max(0.0, self.final_confidence))


@dataclass
class SynergyStateRecord:
    """Complete state record for a synergy pattern."""
    synergy_pattern: SynergyPattern
    state: SynergyState
    confidence: SynergyConfidence
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if synergy has expired."""
        return datetime.now() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if synergy is still valid."""
        return (
            self.state in [SynergyState.BUILDING, SynergyState.COMPLETE] and
            not self.is_expired()
        )


class SynergyStateManager:
    """Manages synergy state, lifecycle, and confidence scoring."""
    
    def __init__(self, expiration_minutes: int = 30):
        """
        Initialize state manager.
        
        Args:
            expiration_minutes: Minutes before synergy expires
        """
        self.expiration_minutes = expiration_minutes
        self.active_synergies: Dict[str, SynergyStateRecord] = {}
        self.synergy_counter = 0
        
        # Performance metrics
        self.metrics = {
            'synergies_created': 0,
            'synergies_invalidated': 0,
            'synergies_consumed': 0,
            'synergies_expired': 0,
            'avg_confidence': 0.0
        }
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_synergies())
        
        logger.info(
            "SynergyStateManager initialized",
            expiration_minutes=expiration_minutes
        )
    
    def create_synergy_record(self, synergy_pattern: SynergyPattern) -> str:
        """
        Create a new synergy state record.
        
        Args:
            synergy_pattern: The detected synergy pattern
            
        Returns:
            Unique synergy ID
        """
        self.synergy_counter += 1
        synergy_id = f"synergy_{self.synergy_counter:06d}"
        
        # Calculate confidence
        confidence = self._calculate_confidence(synergy_pattern)
        
        # Create state record
        now = datetime.now()
        record = SynergyStateRecord(
            synergy_pattern=synergy_pattern,
            state=SynergyState.COMPLETE,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(minutes=self.expiration_minutes),
            metadata={
                'synergy_id': synergy_id,
                'detection_latency_ms': 0,  # Will be populated by detector
                'signal_count': len(synergy_pattern.signals)
            }
        )
        
        self.active_synergies[synergy_id] = record
        self.metrics['synergies_created'] += 1
        
        logger.info(
            "Synergy state record created",
            synergy_id=synergy_id,
            type=synergy_pattern.synergy_type,
            confidence=confidence.final_confidence,
            expires_at=record.expires_at.isoformat()
        )
        
        return synergy_id
    
    def _calculate_confidence(self, synergy_pattern: SynergyPattern) -> SynergyConfidence:
        """Calculate confidence score for synergy pattern."""
        confidence = SynergyConfidence()
        
        # Strength factor: average signal strength
        if synergy_pattern.signals:
            confidence.strength_factor = sum(s.strength for s in synergy_pattern.signals) / len(synergy_pattern.signals)
        
        # Timing factor: how quickly the chain was completed
        if synergy_pattern.bars_to_complete <= 3:
            confidence.timing_factor = 1.0
        elif synergy_pattern.bars_to_complete <= 5:
            confidence.timing_factor = 0.8
        elif synergy_pattern.bars_to_complete <= 8:
            confidence.timing_factor = 0.6
        else:
            confidence.timing_factor = 0.4
        
        # Coherence factor: direction consistency and trend alignment
        if synergy_pattern.signals:
            directions = [s.direction for s in synergy_pattern.signals]
            if all(d == directions[0] for d in directions):
                confidence.coherence_factor = 1.0
            else:
                confidence.coherence_factor = 0.0
        
        # Compute final confidence
        confidence.compute_final_confidence()
        
        return confidence
    
    def invalidate_synergy(self, synergy_id: str, reason: str = "manual"):
        """
        Invalidate a synergy pattern.
        
        Args:
            synergy_id: ID of synergy to invalidate
            reason: Reason for invalidation
        """
        if synergy_id not in self.active_synergies:
            return
        
        record = self.active_synergies[synergy_id]
        record.state = SynergyState.INVALIDATED
        record.updated_at = datetime.now()
        record.metadata['invalidation_reason'] = reason
        
        self.metrics['synergies_invalidated'] += 1
        
        logger.info(
            "Synergy invalidated",
            synergy_id=synergy_id,
            reason=reason,
            original_confidence=record.confidence.final_confidence
        )
    
    def consume_synergy(self, synergy_id: str, execution_data: Dict[str, Any]):
        """
        Mark synergy as consumed by execution system.
        
        Args:
            synergy_id: ID of synergy being consumed
            execution_data: Execution context data
        """
        if synergy_id not in self.active_synergies:
            return
        
        record = self.active_synergies[synergy_id]
        record.state = SynergyState.CONSUMED
        record.updated_at = datetime.now()
        record.metadata['execution_data'] = execution_data
        record.metadata['consumed_at'] = datetime.now().isoformat()
        
        self.metrics['synergies_consumed'] += 1
        
        logger.info(
            "Synergy consumed",
            synergy_id=synergy_id,
            confidence=record.confidence.final_confidence,
            execution_data=execution_data
        )
    
    def get_synergy_state(self, synergy_id: str) -> Optional[SynergyStateRecord]:
        """Get synergy state record."""
        return self.active_synergies.get(synergy_id)
    
    def get_active_synergies(self) -> List[SynergyStateRecord]:
        """Get all active (valid) synergies."""
        return [
            record for record in self.active_synergies.values()
            if record.is_valid()
        ]
    
    def validate_integration_handoff(self, synergy_id: str, target_system: str) -> bool:
        """
        Validate handoff to execution system.
        
        Args:
            synergy_id: ID of synergy being handed off
            target_system: Target system name
            
        Returns:
            True if handoff is valid
        """
        record = self.get_synergy_state(synergy_id)
        if not record:
            logger.warning(
                "Handoff validation failed: synergy not found",
                synergy_id=synergy_id,
                target_system=target_system
            )
            return False
        
        if not record.is_valid():
            logger.warning(
                "Handoff validation failed: synergy not valid",
                synergy_id=synergy_id,
                state=record.state.value,
                target_system=target_system
            )
            return False
        
        # Check minimum confidence threshold
        if record.confidence.final_confidence < 0.5:
            logger.warning(
                "Handoff validation failed: confidence too low",
                synergy_id=synergy_id,
                confidence=record.confidence.final_confidence,
                target_system=target_system
            )
            return False
        
        # Update metadata
        record.metadata['handoff_target'] = target_system
        record.metadata['handoff_timestamp'] = datetime.now().isoformat()
        
        logger.info(
            "Integration handoff validated",
            synergy_id=synergy_id,
            target_system=target_system,
            confidence=record.confidence.final_confidence
        )
        
        return True
    
    async def _cleanup_expired_synergies(self):
        """Background task to cleanup expired synergies."""
        while True:
            try:
                expired_ids = []
                
                for synergy_id, record in self.active_synergies.items():
                    if record.is_expired():
                        expired_ids.append(synergy_id)
                
                for synergy_id in expired_ids:
                    record = self.active_synergies.pop(synergy_id)
                    record.state = SynergyState.EXPIRED
                    self.metrics['synergies_expired'] += 1
                    
                    logger.debug(
                        "Synergy expired and cleaned up",
                        synergy_id=synergy_id,
                        created_at=record.created_at.isoformat()
                    )
                
                # Sleep for 1 minute before next cleanup
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(
                    "Error in synergy cleanup task",
                    error=str(e)
                )
                await asyncio.sleep(60)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get state manager metrics."""
        # Calculate average confidence
        valid_synergies = self.get_active_synergies()
        if valid_synergies:
            self.metrics['avg_confidence'] = sum(
                r.confidence.final_confidence for r in valid_synergies
            ) / len(valid_synergies)
        
        return {
            **self.metrics,
            'active_synergies': len(valid_synergies),
            'total_synergies': len(self.active_synergies)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        return {
            'state_manager': {
                'active': True,
                'expiration_minutes': self.expiration_minutes,
                'metrics': self.get_metrics()
            },
            'active_synergies': [
                {
                    'id': record.metadata.get('synergy_id'),
                    'type': record.synergy_pattern.synergy_type,
                    'state': record.state.value,
                    'confidence': record.confidence.final_confidence,
                    'created_at': record.created_at.isoformat(),
                    'expires_at': record.expires_at.isoformat()
                }
                for record in self.get_active_synergies()
            ]
        }
    
    async def shutdown(self):
        """Shutdown state manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info(
            "SynergyStateManager shutdown",
            final_metrics=self.get_metrics()
        )