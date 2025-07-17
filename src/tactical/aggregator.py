"""
Tactical Decision Aggregator

Aggregates decisions from multiple tactical agents using weighted voting
and synergy-type-specific weightings.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class TacticalDecisionAggregator:
    """
    Aggregates tactical agent decisions with confidence weighting.
    
    Implements weighted voting with synergy-type-specific agent weights
    and 65% execution threshold requirement.
    """
    
    def __init__(self):
        """Initialize decision aggregator."""
        self.execution_threshold = 0.65
        
        # Synergy-type-specific agent weights
        self.synergy_weights = {
            'TYPE_1': [0.5, 0.3, 0.2],   # FVG-heavy synergy
            'TYPE_2': [0.4, 0.4, 0.2],   # Balanced FVG+Momentum
            'TYPE_3': [0.3, 0.5, 0.2],   # Momentum-heavy synergy
            'TYPE_4': [0.35, 0.35, 0.3], # Entry timing critical
            'manual': [0.33, 0.33, 0.34] # Default equal weights
        }
        
        logger.info("Tactical decision aggregator initialized")
    
    async def aggregate_decisions(
        self,
        agent_decisions: List[Dict[str, Any]],
        synergy_event: Any
    ) -> Dict[str, Any]:
        """
        Aggregate multi-agent decisions with synergy context.
        
        Args:
            agent_decisions: List of agent decision dictionaries
            synergy_event: Synergy event context
            
        Returns:
            Aggregated decision dictionary
        """
        
        if len(agent_decisions) != 3:
            raise ValueError("Expected exactly 3 agent decisions")
        
        # Get synergy-specific weights
        synergy_type = synergy_event.synergy_type
        weights = self.synergy_weights.get(synergy_type, self.synergy_weights['manual'])
        
        # Extract actions and confidences
        actions = [decision['action'] for decision in agent_decisions]
        confidences = [decision['confidence'] for decision in agent_decisions]
        
        # Weighted voting
        weighted_actions = self._calculate_weighted_votes(actions, confidences, weights)
        
        # Find consensus action
        max_action = max(weighted_actions, key=weighted_actions.get)
        max_score = weighted_actions[max_action]
        
        # Determine if should execute
        should_execute = max_score >= self.execution_threshold
        
        # HARD SYNERGY ALIGNMENT GATE - Game Theory Resistant
        if should_execute and max_action != 0:  # Not hold
            direction_match = (
                (max_action > 0 and synergy_event.direction > 0) or
                (max_action < 0 and synergy_event.direction < 0)
            )
            
            if not direction_match:
                # HARD GATE: Counter-synergy trades require >95% confidence
                # This prevents consensus override gaming
                if max_score < 0.95:
                    should_execute = False
                    max_score = 0.0  # Complete veto - no partial execution
                    logger.warning(f"STRATEGIC VETO: Counter-synergy trade blocked. "
                                 f"Required confidence: 0.95, actual: {max_score:.3f}")
                else:
                    # Ultra-high confidence override allowed but heavily logged
                    logger.critical(f"STRATEGIC OVERRIDE: Ultra-high confidence counter-synergy trade "
                                   f"approved with confidence: {max_score:.3f}")
            else:
                # Aligned trades get bonus confidence for being strategic
                strategic_bonus = min(0.1, (1.0 - max_score) * 0.5)
                max_score = min(1.0, max_score + strategic_bonus)
        
        # Map action to string
        action_map = {-1: "short", 0: "hold", 1: "long"}
        action_str = action_map.get(max_action, "hold")
        
        return {
            'action': action_str,
            'confidence': max_score,
            'should_execute': should_execute,
            'consensus_breakdown': weighted_actions,
            'synergy_alignment': self._check_synergy_alignment(max_action, synergy_event),
            'execution_threshold': self.execution_threshold,
            'agent_weights': weights,
            'strategic_gate_enforced': max_action != 0 and not self._check_synergy_alignment(max_action, synergy_event),
            'strategic_override_threshold': 0.95,
            'original_action': max_action
        }
    
    def _calculate_weighted_votes(
        self,
        actions: List[int],
        confidences: List[float],
        weights: List[float]
    ) -> Dict[int, float]:
        """Calculate weighted voting scores."""
        weighted_actions = {}
        
        for action, confidence, weight in zip(actions, confidences, weights):
            if action not in weighted_actions:
                weighted_actions[action] = 0.0
            weighted_actions[action] += confidence * weight
        
        return weighted_actions
    
    def _check_synergy_alignment(self, action: int, synergy_event: Any) -> bool:
        """Check if action aligns with synergy direction."""
        if action == 0:  # Hold is always aligned
            return True
        
        return (action > 0 and synergy_event.direction > 0) or \
               (action < 0 and synergy_event.direction < 0)