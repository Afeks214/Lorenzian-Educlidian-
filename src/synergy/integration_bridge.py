"""
Synergy Integration Bridge
=========================

This module provides proper integration between synergy detection and execution
systems, fixing timestamp alignment failures and ensuring proper event handoffs.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog

from src.core.minimal_dependencies import EventType, Event
from .base import SynergyPattern
from .state_manager import SynergyStateManager

logger = structlog.get_logger()


@dataclass
class IntegrationEvent:
    """Event for integration handoffs."""
    event_type: str
    synergy_id: str
    payload: Dict[str, Any]
    timestamp: datetime
    source_system: str
    target_system: str
    metadata: Dict[str, Any]


class SynergyIntegrationBridge:
    """
    Bridges synergy detection with execution systems.
    
    Handles:
    - Timestamp alignment between systems
    - Event validation and transformation
    - Proper handoff coordination
    - Integration failure recovery
    """
    
    def __init__(self, state_manager: SynergyStateManager, event_bus):
        """
        Initialize integration bridge.
        
        Args:
            state_manager: Synergy state manager
            event_bus: System event bus
        """
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.integration_handlers: Dict[str, Callable] = {}
        self.handoff_timeout_seconds = 30
        
        # Performance metrics
        self.metrics = {
            'handoffs_attempted': 0,
            'handoffs_successful': 0,
            'handoffs_failed': 0,
            'handoffs_timeout': 0,
            'avg_handoff_latency_ms': 0.0
        }
        
        # Active handoffs tracking
        self.active_handoffs: Dict[str, IntegrationEvent] = {}
        
        logger.info("SynergyIntegrationBridge initialized")
    
    def register_integration_handler(self, system_name: str, handler: Callable):
        """
        Register a handler for integration with a specific system.
        
        Args:
            system_name: Name of the target system
            handler: Async handler function
        """
        self.integration_handlers[system_name] = handler
        logger.info(f"Integration handler registered for {system_name}")
    
    async def initiate_handoff(self, synergy_pattern: SynergyPattern, target_system: str) -> bool:
        """
        Initiate synergy handoff to target system.
        
        Args:
            synergy_pattern: The synergy pattern to hand off
            target_system: Target system name
            
        Returns:
            True if handoff successful
        """
        start_time = datetime.now()
        self.metrics['handoffs_attempted'] += 1
        
        try:
            # Validate synergy state
            if not synergy_pattern.synergy_id:
                logger.error("Cannot handoff synergy without ID")
                return False
            
            # Validate handoff with state manager
            if not self.state_manager.validate_integration_handoff(
                synergy_pattern.synergy_id, target_system
            ):
                self.metrics['handoffs_failed'] += 1
                return False
            
            # Create integration event
            integration_event = IntegrationEvent(
                event_type='SYNERGY_HANDOFF',
                synergy_id=synergy_pattern.synergy_id,
                payload=self._create_handoff_payload(synergy_pattern),
                timestamp=datetime.now(),
                source_system='synergy_detector',
                target_system=target_system,
                metadata={
                    'original_synergy_type': synergy_pattern.synergy_type,
                    'handoff_initiated_at': start_time.isoformat(),
                    'expected_timeout': (start_time + timedelta(seconds=self.handoff_timeout_seconds)).isoformat()
                }
            )
            
            # Track active handoff
            self.active_handoffs[synergy_pattern.synergy_id] = integration_event
            
            # Execute handoff
            success = await self._execute_handoff(integration_event)
            
            if success:
                self.metrics['handoffs_successful'] += 1
                # Update latency metrics
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                self._update_latency_metrics(latency_ms)
                
                logger.info(
                    "Synergy handoff successful",
                    synergy_id=synergy_pattern.synergy_id,
                    target_system=target_system,
                    latency_ms=latency_ms
                )
            else:
                self.metrics['handoffs_failed'] += 1
                logger.error(
                    "Synergy handoff failed",
                    synergy_id=synergy_pattern.synergy_id,
                    target_system=target_system
                )
            
            # Remove from active handoffs
            self.active_handoffs.pop(synergy_pattern.synergy_id, None)
            
            return success
            
        except Exception as e:
            self.metrics['handoffs_failed'] += 1
            logger.error(
                "Error during synergy handoff",
                synergy_id=synergy_pattern.synergy_id,
                target_system=target_system,
                error=str(e)
            )
            return False
    
    def _create_handoff_payload(self, synergy_pattern: SynergyPattern) -> Dict[str, Any]:
        """Create payload for handoff event."""
        return {
            'synergy_type': synergy_pattern.synergy_type,
            'direction': synergy_pattern.direction,
            'confidence': synergy_pattern.confidence,
            'completion_time': synergy_pattern.completion_time.isoformat(),
            'bars_to_complete': synergy_pattern.bars_to_complete,
            'is_sequential': synergy_pattern.is_sequential(),
            'signals': [
                {
                    'type': signal.signal_type,
                    'direction': signal.direction,
                    'timestamp': signal.timestamp.isoformat(),
                    'value': signal.value,
                    'strength': signal.strength,
                    'metadata': signal.metadata
                }
                for signal in synergy_pattern.signals
            ],
            'state_info': {
                'synergy_id': synergy_pattern.synergy_id,
                'state_managed': synergy_pattern.state_managed
            }
        }
    
    async def _execute_handoff(self, integration_event: IntegrationEvent) -> bool:
        """Execute the actual handoff."""
        target_system = integration_event.target_system
        
        # Check if we have a registered handler
        if target_system in self.integration_handlers:
            try:
                # Call registered handler
                result = await self.integration_handlers[target_system](integration_event)
                return bool(result)
            except Exception as e:
                logger.error(
                    "Error in integration handler",
                    target_system=target_system,
                    error=str(e)
                )
                return False
        else:
            # Default: emit event to event bus
            try:
                event = Event(
                    event_type=EventType.SYNERGY_DETECTED,
                    payload=integration_event.payload,
                    timestamp=integration_event.timestamp,
                    metadata={
                        **integration_event.metadata,
                        'integration_bridge': True,
                        'target_system': target_system
                    }
                )
                
                await self.event_bus.publish(event)
                return True
                
            except Exception as e:
                logger.error(
                    "Error publishing integration event",
                    target_system=target_system,
                    error=str(e)
                )
                return False
    
    def _update_latency_metrics(self, latency_ms: float):
        """Update latency metrics."""
        # Update average latency
        n = self.metrics['handoffs_successful']
        avg = self.metrics['avg_handoff_latency_ms']
        self.metrics['avg_handoff_latency_ms'] = ((avg * (n - 1)) + latency_ms) / n
    
    async def handle_timeout_cleanup(self):
        """Clean up timed out handoffs."""
        current_time = datetime.now()
        timeout_threshold = current_time - timedelta(seconds=self.handoff_timeout_seconds)
        
        timed_out_handoffs = []
        
        for synergy_id, integration_event in self.active_handoffs.items():
            if integration_event.timestamp < timeout_threshold:
                timed_out_handoffs.append(synergy_id)
        
        for synergy_id in timed_out_handoffs:
            integration_event = self.active_handoffs.pop(synergy_id)
            self.metrics['handoffs_timeout'] += 1
            
            # Invalidate synergy in state manager
            self.state_manager.invalidate_synergy(synergy_id, "handoff_timeout")
            
            logger.warning(
                "Synergy handoff timed out",
                synergy_id=synergy_id,
                target_system=integration_event.target_system
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics."""
        total_handoffs = self.metrics['handoffs_attempted']
        success_rate = (
            self.metrics['handoffs_successful'] / max(1, total_handoffs)
        )
        
        return {
            **self.metrics,
            'success_rate': success_rate,
            'active_handoffs': len(self.active_handoffs),
            'registered_handlers': list(self.integration_handlers.keys())
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration bridge status."""
        return {
            'active': True,
            'metrics': self.get_metrics(),
            'timeout_seconds': self.handoff_timeout_seconds,
            'active_handoffs': [
                {
                    'synergy_id': event.synergy_id,
                    'target_system': event.target_system,
                    'initiated_at': event.timestamp.isoformat(),
                    'payload_size': len(str(event.payload))
                }
                for event in self.active_handoffs.values()
            ]
        }


# Example integration handler for execution system
async def execution_system_handler(integration_event: IntegrationEvent) -> bool:
    """Example handler for execution system integration."""
    try:
        # Extract synergy data
        synergy_data = integration_event.payload
        
        # Validate synergy is sequential
        if not synergy_data.get('is_sequential', False):
            logger.warning(
                "Non-sequential synergy handed off to execution",
                synergy_id=integration_event.synergy_id
            )
            return False
        
        # Validate signal chain
        signals = synergy_data.get('signals', [])
        if len(signals) != 3:
            logger.error(
                "Invalid signal chain length",
                synergy_id=integration_event.synergy_id,
                signal_count=len(signals)
            )
            return False
        
        # Check signal sequence
        signal_types = [s['type'] for s in signals]
        if signal_types != ['nwrqk', 'mlmi', 'fvg']:
            logger.error(
                "Invalid signal sequence",
                synergy_id=integration_event.synergy_id,
                sequence=signal_types
            )
            return False
        
        # Process for execution
        logger.info(
            "Sequential synergy processed for execution",
            synergy_id=integration_event.synergy_id,
            direction=synergy_data.get('direction'),
            confidence=synergy_data.get('confidence')
        )
        
        return True
        
    except Exception as e:
        logger.error(
            "Error in execution system handler",
            synergy_id=integration_event.synergy_id,
            error=str(e)
        )
        return False