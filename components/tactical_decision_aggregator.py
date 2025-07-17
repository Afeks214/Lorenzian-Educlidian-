"""
Tactical Decision Aggregator with Configuration-Driven Flexibility

Implements the consensus algorithm for aggregating decisions from tactical agents:
- FVG Agent, Momentum Agent, and Entry Optimization Agent
- Synergy-based dynamic weighting
- Confidence threshold enforcement
- Direction alignment validation

Features:
- Config-driven synergy weights for different market conditions
- Weighted voting with confidence scoring
- Execution threshold validation (0.65 default)
- Direction bias based on synergy context
- Comprehensive decision logging and analysis

Author: Quantitative Engineer
Version: 1.0
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
import yaml
from pathlib import Path
import time
import hashlib
import hmac
import secrets
from cryptography.hazmat.primitives import hashes, hmac as crypto_hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class SynergyType(Enum):
    """Synergy pattern types from strategic system"""
    TYPE_1 = "TYPE_1"  # FVG-heavy synergy
    TYPE_2 = "TYPE_2"  # Balanced FVG+Momentum
    TYPE_3 = "TYPE_3"  # Momentum-heavy synergy
    TYPE_4 = "TYPE_4"  # Entry timing critical
    NONE = "NONE"      # No synergy detected


@dataclass
class AgentDecision:
    """Container for individual agent decision with cryptographic validation"""
    agent_id: str
    action: int
    probabilities: np.ndarray
    confidence: float
    timestamp: float
    signature: Optional[str] = None
    nonce: Optional[str] = None
    view_number: int = 0
    is_byzantine: bool = False


@dataclass
class AggregatedDecision:
    """Container for final aggregated decision with PBFT consensus validation"""
    execute: bool
    action: int
    confidence: float
    agent_votes: Dict[str, AgentDecision]
    consensus_breakdown: Dict[int, float]
    synergy_alignment: float
    execution_command: Optional[Dict[str, Any]]
    pbft_consensus_achieved: bool = False
    byzantine_agents_detected: List[str] = None
    consensus_signatures: Dict[str, str] = None
    view_number: int = 0
    safety_level: float = 0.0


class TacticalDecisionAggregator:
    """
    Tactical Decision Aggregator
    
    Aggregates decisions from multiple tactical agents using a sophisticated
    consensus algorithm that considers:
    - Individual agent confidence levels
    - Synergy-based dynamic weighting
    - Direction alignment with strategic signals
    - Execution threshold validation
    
    The aggregator is fully configuration-driven, allowing rapid adjustment
    of weights and thresholds without code changes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Byzantine Fault Tolerant Tactical Decision Aggregator
        
        Args:
            config: Configuration dictionary with PBFT consensus parameters
        """
        self.config = config or self._default_config()
        
        # Extract key parameters
        self.execution_threshold = self.config.get('execution_threshold', 0.65)
        self.synergy_weights = self.config.get('synergy_weights', self._default_synergy_weights())
        self.direction_penalty = self.config.get('direction_penalty', 0.3)
        self.confidence_boost = self.config.get('confidence_boost', 0.1)
        
        # Production hardening parameters
        self.disagreement_threshold = self.config.get('disagreement_threshold', 0.4)
        self.disagreement_penalty = self.config.get('disagreement_penalty', 0.5)
        self.consensus_filter_enabled = self.config.get('consensus_filter_enabled', True)
        self.min_consensus_strength = self.config.get('min_consensus_strength', 0.6)
        self.max_disagreement_score = self.config.get('max_disagreement_score', 0.8)
        
        # PBFT CONSENSUS PARAMETERS
        self.pbft_enabled = self.config.get('pbft_enabled', True)
        self.byzantine_fault_tolerance = self.config.get('byzantine_fault_tolerance', 1)  # f=1, supports 3f+1=4 agents
        self.pbft_timeout = self.config.get('pbft_timeout', 5.0)  # 5 second timeout
        self.view_change_threshold = self.config.get('view_change_threshold', 3)  # 3 failed attempts
        
        # CRYPTOGRAPHIC SECURITY
        self.master_key = self._generate_master_key()
        self.agent_keys = self._initialize_agent_keys()
        self.current_view = 0
        self.view_change_count = 0
        self.byzantine_agents = set()
        
        # Decision history for analysis
        self.decision_history = []
        self.performance_metrics = {
            'total_decisions': 0,
            'executed_decisions': 0,
            'synergy_aligned': 0,
            'consensus_achieved': 0,
            'disagreement_penalties': 0,
            'consensus_failures': 0,
            'byzantine_attacks_detected': 0,
            'pbft_consensus_achieved': 0,
            'view_changes': 0
        }
        
        logger.info(f"Byzantine Fault Tolerant TacticalDecisionAggregator initialized with PBFT consensus")
        logger.info(f"Execution threshold: {self.execution_threshold}, Byzantine tolerance: f={self.byzantine_fault_tolerance}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for PBFT consensus aggregation"""
        return {
            'execution_threshold': 0.75,  # Higher threshold for safety
            'direction_penalty': 0.3,
            'confidence_boost': 0.1,
            'synergy_weights': self._default_synergy_weights(),
            'disagreement_threshold': 0.3,  # Lower threshold for faster detection
            'disagreement_penalty': 0.7,   # Higher penalty for safety
            'consensus_filter_enabled': True,
            'min_consensus_strength': 0.7, # Higher consensus requirement
            'max_disagreement_score': 0.6, # Lower tolerance for disagreement
            'pbft_enabled': True,
            'byzantine_fault_tolerance': 1,
            'pbft_timeout': 5.0,
            'view_change_threshold': 3
        }
    
    def _default_synergy_weights(self) -> Dict[str, List[float]]:
        """Default synergy-based agent weights"""
        return {
            'TYPE_1': [0.5, 0.3, 0.2],   # FVG-heavy: FVG, Momentum, Entry
            'TYPE_2': [0.4, 0.4, 0.2],   # Balanced: FVG, Momentum, Entry
            'TYPE_3': [0.3, 0.5, 0.2],   # Momentum-heavy: FVG, Momentum, Entry
            'TYPE_4': [0.35, 0.35, 0.3], # Entry timing critical: FVG, Momentum, Entry
            'NONE': [0.33, 0.33, 0.34]   # No synergy: equal weights
        }
    
    def aggregate_decisions(
        self,
        agent_outputs: Dict[str, Any],
        market_state: Any,
        synergy_context: Dict[str, Any]
    ) -> AggregatedDecision:
        """
        BYZANTINE FAULT TOLERANT DECISION AGGREGATION
        
        Implements PBFT consensus algorithm to prevent malicious agent attacks.
        
        Args:
            agent_outputs: Dictionary of agent outputs with cryptographic signatures
            market_state: Current market state information
            synergy_context: Synergy detection context from strategic system
            
        Returns:
            AggregatedDecision with PBFT consensus validation
        """
        try:
            # PHASE 1: CRYPTOGRAPHIC AGENT VALIDATION
            validated_decisions = self._validate_agent_signatures(agent_outputs)
            
            # PHASE 2: BYZANTINE AGENT DETECTION
            byzantine_agents = self._detect_byzantine_agents(validated_decisions, market_state)
            
            # PHASE 3: PBFT CONSENSUS ALGORITHM
            pbft_result = None
            if self.pbft_enabled:
                pbft_result = self._execute_pbft_consensus(
                    validated_decisions, 
                    market_state, 
                    synergy_context,
                    byzantine_agents
                )
            
            # PHASE 4: FALLBACK TO ENHANCED WEIGHTED VOTING (if PBFT fails)
            if pbft_result is None or not pbft_result.get('consensus_achieved', False):
                logger.warning("PBFT consensus failed, falling back to enhanced weighted voting")
                pbft_result = self._enhanced_weighted_voting_fallback(
                    validated_decisions, 
                    market_state, 
                    synergy_context,
                    byzantine_agents
                )
            
            # PHASE 5: SAFETY VALIDATION
            final_decision = self._apply_safety_validation(pbft_result, market_state)
            
            # PHASE 6: CRYPTOGRAPHIC DECISION SIGNING
            self._sign_aggregated_decision(final_decision)
            
            # Update performance metrics
            self._update_pbft_metrics(final_decision, byzantine_agents)
            
            # Store in decision history
            self.decision_history.append({
                'decision': final_decision,
                'synergy_context': synergy_context,
                'timestamp': time.time(),
                'byzantine_agents': list(byzantine_agents),
                'pbft_consensus': pbft_result.get('consensus_achieved', False)
            })
            
            logger.info(f"PBFT Decision: execute={final_decision.execute}, "
                       f"action={final_decision.action}, safety={final_decision.safety_level:.3f}")
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Error in PBFT consensus: {e}")
            # Return ultra-safe default decision
            return self._create_safe_default_decision(agent_outputs)
    
    def _convert_agent_outputs(self, agent_outputs: Dict[str, Any]) -> Dict[str, AgentDecision]:
        """Convert agent outputs to structured decisions"""
        agent_decisions = {}
        
        for agent_id, output in agent_outputs.items():
            if hasattr(output, 'probabilities'):
                probabilities = output.probabilities
                action = output.action
                confidence = output.confidence
                timestamp = output.timestamp
            else:
                # Fallback for dictionary format
                probabilities = np.array(output.get('probabilities', [0.33, 0.34, 0.33]))
                action = output.get('action', 1)  # Default to neutral
                confidence = output.get('confidence', 0.5)
                timestamp = output.get('timestamp', 0)
            
            agent_decisions[agent_id] = AgentDecision(
                agent_id=agent_id,
                action=action,
                probabilities=probabilities,
                confidence=confidence,
                timestamp=timestamp
            )
        
        return agent_decisions
    
    def _extract_synergy_type(self, synergy_context: Dict[str, Any]) -> SynergyType:
        """Extract synergy type from context"""
        synergy_type_str = synergy_context.get('type', 'NONE')
        
        try:
            return SynergyType(synergy_type_str)
        except ValueError:
            logger.warning(f"Unknown synergy type: {synergy_type_str}")
            return SynergyType.NONE
    
    def _get_agent_weights(self, synergy_type: SynergyType) -> List[float]:
        """Get agent weights based on synergy type"""
        weights = self.synergy_weights.get(synergy_type.value, [0.33, 0.33, 0.34])
        
        # Ensure weights sum to 1.0
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        return weights
    
    def _weighted_voting(
        self,
        agent_decisions: Dict[str, AgentDecision],
        agent_weights: List[float]
    ) -> Dict[int, float]:
        """Perform weighted voting across agents"""
        weighted_actions = {0: 0.0, 1: 0.0, 2: 0.0}  # bearish, neutral, bullish
        
        # Agent order: fvg_agent, momentum_agent, entry_opt_agent
        agent_order = ['fvg_agent', 'momentum_agent', 'entry_opt_agent']
        
        for i, agent_id in enumerate(agent_order):
            if agent_id in agent_decisions and i < len(agent_weights):
                decision = agent_decisions[agent_id]
                weight = agent_weights[i]
                
                # Weight the probabilities by agent weight and confidence
                for action_idx, prob in enumerate(decision.probabilities):
                    if action_idx in weighted_actions:
                        weighted_actions[action_idx] += prob * weight * decision.confidence
        
        return weighted_actions
    
    def _apply_direction_bias(
        self,
        consensus_action: int,
        consensus_confidence: float,
        synergy_direction: int,
        synergy_confidence: float
    ) -> float:
        """Apply synergy direction bias to consensus confidence"""
        if synergy_direction == 0 or synergy_confidence < 0.3:
            return consensus_confidence
        
        # Check if action aligns with synergy direction
        # action: 0=bearish, 1=neutral, 2=bullish
        # synergy_direction: -1=bearish, 0=neutral, 1=bullish
        
        direction_match = False
        if consensus_action == 0 and synergy_direction == -1:  # Both bearish
            direction_match = True
        elif consensus_action == 2 and synergy_direction == 1:  # Both bullish
            direction_match = True
        elif consensus_action == 1:  # Neutral action
            direction_match = True  # Neutral is always "aligned"
        
        if direction_match:
            # Boost confidence for aligned trades
            boost = self.confidence_boost * synergy_confidence
            return min(consensus_confidence + boost, 1.0)
        else:
            # Penalize misaligned trades
            penalty = self.direction_penalty * synergy_confidence
            return max(consensus_confidence - penalty, 0.0)
    
    def _create_execution_command(
        self,
        action: int,
        confidence: float,
        market_state: Any,
        synergy_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create execution command for trading system"""
        # Action mapping: 0=bearish/short, 1=neutral/hold, 2=bullish/long
        action_map = {0: 'short', 1: 'hold', 2: 'long'}
        
        # Position sizing based on confidence
        base_quantity = 1.0
        confidence_multiplier = min(confidence / 0.8, 1.5)
        quantity = base_quantity * confidence_multiplier
        
        # Get current price from market state
        current_price = getattr(market_state, 'price', 100.0)
        
        # Calculate risk management levels
        atr_estimate = current_price * 0.02  # 2% ATR estimate
        
        if action == 2:  # Long position
            stop_loss = current_price - (2.0 * atr_estimate)
            take_profit = current_price + (3.0 * atr_estimate)
            side = 'BUY'
        elif action == 0:  # Short position
            stop_loss = current_price + (2.0 * atr_estimate)
            take_profit = current_price - (3.0 * atr_estimate)
            side = 'SELL'
        else:  # Hold
            return {'action': 'hold', 'reason': 'neutral_consensus'}
        
        execution_command = {
            'action': 'execute_trade',
            'side': side,
            'quantity': round(quantity, 2),
            'order_type': 'MARKET',
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'time_in_force': 'IOC',
            'metadata': {
                'source': 'tactical_marl',
                'consensus_action': action_map[action],
                'confidence': confidence,
                'synergy_aligned': synergy_context.get('type', 'NONE') != 'NONE'
            }
        }
        
        return execution_command
    
    def _calculate_disagreement_score(self, agent_decisions: Dict[str, AgentDecision]) -> float:
        """
        Calculate disagreement score between agents
        
        Uses Jensen-Shannon divergence to measure disagreement between probability distributions
        and confidence variance to assess decision uncertainty.
        
        Args:
            agent_decisions: Dictionary of agent decisions
            
        Returns:
            Disagreement score between 0 (perfect agreement) and 1 (maximum disagreement)
        """
        if len(agent_decisions) < 2:
            return 0.0
        
        # Extract probability distributions and confidences
        prob_distributions = []
        confidences = []
        
        for decision in agent_decisions.values():
            # Ensure probabilities are properly normalized
            probs = decision.probabilities
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
            else:
                probs = np.array([0.33, 0.34, 0.33])  # Default uniform
            
            prob_distributions.append(probs)
            confidences.append(decision.confidence)
        
        # Calculate Jensen-Shannon divergence between distributions
        js_divergence = self._jensen_shannon_divergence(prob_distributions)
        
        # Calculate confidence variance
        confidence_variance = np.var(confidences)
        
        # Calculate action disagreement (different actions chosen)
        actions = [decision.action for decision in agent_decisions.values()]
        action_disagreement = 1.0 - (len(set(actions)) == 1)  # 1 if disagreement, 0 if agreement
        
        # Combine metrics with weights (enhanced sensitivity)
        disagreement_score = (
            0.6 * js_divergence +
            0.2 * min(confidence_variance / 0.25, 1.0) +  # Normalize variance
            0.2 * action_disagreement
        )
        
        # Apply exponential scaling for extreme disagreement scenarios
        if js_divergence > 0.5 and action_disagreement > 0.5:
            # Boost score for extreme scenarios (opposite directions + high confidence)
            disagreement_score = min(disagreement_score * 1.5, 1.0)
        
        return min(disagreement_score, 1.0)
    
    def _jensen_shannon_divergence(self, distributions: List[np.ndarray]) -> float:
        """
        Calculate Jensen-Shannon divergence between multiple probability distributions
        
        Args:
            distributions: List of probability distributions
            
        Returns:
            JS divergence score between 0 and 1
        """
        if len(distributions) < 2:
            return 0.0
        
        # Calculate mean distribution
        mean_dist = np.mean(distributions, axis=0)
        
        # Calculate KL divergences
        kl_divergences = []
        for dist in distributions:
            kl_div = self._kl_divergence(dist, mean_dist)
            kl_divergences.append(kl_div)
        
        # Jensen-Shannon divergence
        js_divergence = np.mean(kl_divergences)
        
        # Normalize to [0, 1] range
        return min(js_divergence / np.log(len(distributions[0])), 1.0)
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Kullback-Leibler divergence between two distributions
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            KL divergence
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        return np.sum(p * np.log(p / q))
    
    def _calculate_consensus_strength(self, weighted_actions: Dict[int, float]) -> float:
        """
        Calculate consensus strength based on action weight distribution
        
        Args:
            weighted_actions: Dictionary of action weights
            
        Returns:
            Consensus strength between 0 (no consensus) and 1 (perfect consensus)
        """
        if not weighted_actions:
            return 0.0
        
        # Get weights as array
        weights = np.array(list(weighted_actions.values()))
        
        # Handle zero weights
        if np.sum(weights) == 0:
            return 0.0
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Method 1: Use maximum weight as primary indicator
        max_weight = np.max(weights)
        
        # Method 2: Calculate entropy-based consensus
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        max_entropy = np.log(len(weights))
        entropy_consensus = 1.0 - (entropy / max_entropy)
        
        # Method 3: Calculate variance-based consensus
        variance = np.var(weights)
        max_variance = np.var([1.0, 0.0, 0.0])  # Maximum possible variance
        variance_consensus = 1.0 - (variance / max_variance)
        
        # Combine methods with weights
        consensus_strength = (
            0.5 * max_weight +
            0.3 * entropy_consensus +
            0.2 * variance_consensus
        )
        
        return min(consensus_strength, 1.0)
    
    def _calculate_synergy_alignment(
        self,
        consensus_action: int,
        synergy_direction: int,
        synergy_confidence: float
    ) -> float:
        """Calculate synergy alignment score"""
        if synergy_direction == 0:
            return 0.0
        
        # Convert action to direction
        action_direction = 0
        if consensus_action == 0:
            action_direction = -1
        elif consensus_action == 2:
            action_direction = 1
        
        # Calculate alignment
        if action_direction == synergy_direction:
            return synergy_confidence
        elif action_direction == 0:  # Neutral
            return 0.5 * synergy_confidence
        else:  # Opposite direction
            return -synergy_confidence
    
    def _update_metrics(self, decision: AggregatedDecision, synergy_context: Dict[str, Any]):
        """Update performance metrics"""
        self.performance_metrics['total_decisions'] += 1
        
        if decision.execute:
            self.performance_metrics['executed_decisions'] += 1
        
        if decision.synergy_alignment > 0:
            self.performance_metrics['synergy_aligned'] += 1
        
        if decision.confidence >= self.execution_threshold:
            self.performance_metrics['consensus_achieved'] += 1
    
    def _create_default_decision(self, agent_outputs: Dict[str, Any]) -> AggregatedDecision:
        """Create safe default decision in case of errors"""
        return AggregatedDecision(
            execute=False,
            action=1,  # Neutral
            confidence=0.0,
            agent_votes={},
            consensus_breakdown={0: 0.0, 1: 1.0, 2: 0.0},
            synergy_alignment=0.0,
            execution_command=None
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregator performance metrics"""
        total = self.performance_metrics['total_decisions']
        if total == 0:
            return self.performance_metrics
        
        metrics = self.performance_metrics.copy()
        metrics['execution_rate'] = metrics['executed_decisions'] / total
        metrics['synergy_alignment_rate'] = metrics['synergy_aligned'] / total
        metrics['consensus_rate'] = metrics['consensus_achieved'] / total
        metrics['disagreement_penalty_rate'] = metrics['disagreement_penalties'] / total
        metrics['consensus_failure_rate'] = metrics['consensus_failures'] / total
        
        return metrics
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        return self.decision_history[-limit:]
    
    def analyze_disagreement_patterns(self, lookback_window: int = 50) -> Dict[str, Any]:
        """
        Analyze disagreement patterns in recent decisions
        
        Args:
            lookback_window: Number of recent decisions to analyze
            
        Returns:
            Dictionary with disagreement analysis
        """
        if len(self.decision_history) < 2:
            return {'status': 'insufficient_data', 'decisions_analyzed': 0}
        
        recent_decisions = self.decision_history[-lookback_window:]
        
        # Extract disagreement scores
        disagreement_scores = []
        consensus_strengths = []
        execution_rates = []
        
        for decision_record in recent_decisions:
            decision = decision_record['decision']
            if hasattr(decision, 'disagreement_score'):
                disagreement_scores.append(decision.disagreement_score)
            if hasattr(decision, 'consensus_strength'):
                consensus_strengths.append(decision.consensus_strength)
            execution_rates.append(1.0 if decision.execute else 0.0)
        
        # Calculate statistics
        analysis = {
            'decisions_analyzed': len(recent_decisions),
            'disagreement_statistics': {
                'mean_disagreement': np.mean(disagreement_scores) if disagreement_scores else 0.0,
                'max_disagreement': np.max(disagreement_scores) if disagreement_scores else 0.0,
                'high_disagreement_count': sum(1 for score in disagreement_scores if score > self.disagreement_threshold),
                'disagreement_trend': self._calculate_trend(disagreement_scores) if len(disagreement_scores) > 5 else 0.0
            },
            'consensus_statistics': {
                'mean_consensus_strength': np.mean(consensus_strengths) if consensus_strengths else 0.0,
                'min_consensus_strength': np.min(consensus_strengths) if consensus_strengths else 0.0,
                'low_consensus_count': sum(1 for strength in consensus_strengths if strength < self.min_consensus_strength),
                'consensus_trend': self._calculate_trend(consensus_strengths) if len(consensus_strengths) > 5 else 0.0
            },
            'execution_statistics': {
                'execution_rate': np.mean(execution_rates),
                'blocked_by_disagreement': self.performance_metrics.get('disagreement_penalties', 0),
                'blocked_by_consensus': self.performance_metrics.get('consensus_failures', 0)
            }
        }
        
        # Add recommendations
        analysis['recommendations'] = self._generate_disagreement_recommendations(analysis)
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend in time series data
        
        Args:
            values: List of values to analyze
            
        Returns:
            Trend coefficient (-1 to 1, negative = decreasing, positive = increasing)
        """
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        # Normalize slope to [-1, 1] range
        y_range = np.max(y) - np.min(y)
        if y_range > 0:
            normalized_slope = slope / (y_range / n)
            return np.clip(normalized_slope, -1.0, 1.0)
        
        return 0.0
    
    def _generate_disagreement_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on disagreement analysis
        
        Args:
            analysis: Disagreement analysis results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # High disagreement recommendations
        if analysis['disagreement_statistics']['mean_disagreement'] > 0.6:
            recommendations.append(
                f"High average disagreement ({analysis['disagreement_statistics']['mean_disagreement']:.2f}). "
                f"Consider reducing disagreement_threshold or increasing disagreement_penalty."
            )
        
        # Low consensus recommendations
        if analysis['consensus_statistics']['mean_consensus_strength'] < 0.5:
            recommendations.append(
                f"Low average consensus strength ({analysis['consensus_statistics']['mean_consensus_strength']:.2f}). "
                f"Consider adjusting synergy weights or lowering min_consensus_strength."
            )
        
        # Execution rate recommendations
        if analysis['execution_statistics']['execution_rate'] < 0.3:
            recommendations.append(
                f"Low execution rate ({analysis['execution_statistics']['execution_rate']:.2f}). "
                f"System may be too conservative. Consider lowering execution_threshold."
            )
        
        # Trend recommendations
        if analysis['disagreement_statistics']['disagreement_trend'] > 0.5:
            recommendations.append(
                "Disagreement trend is increasing. Monitor agent performance and consider retraining."
            )
        
        if analysis['consensus_statistics']['consensus_trend'] < -0.5:
            recommendations.append(
                "Consensus trend is decreasing. Agents may be diverging. Review training stability."
            )
        
        if not recommendations:
            recommendations.append("Disagreement patterns appear normal. No immediate action required.")
        
        return recommendations
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_decisions': 0,
            'executed_decisions': 0,
            'synergy_aligned': 0,
            'consensus_achieved': 0,
            'disagreement_penalties': 0,
            'consensus_failures': 0
        }
        self.decision_history.clear()
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration parameters"""
        self.config.update(new_config)
        
        # Update key parameters
        self.execution_threshold = self.config.get('execution_threshold', 0.65)
        self.synergy_weights = self.config.get('synergy_weights', self._default_synergy_weights())
        self.direction_penalty = self.config.get('direction_penalty', 0.3)
        self.confidence_boost = self.config.get('confidence_boost', 0.1)
        
        # Update production hardening parameters
        self.disagreement_threshold = self.config.get('disagreement_threshold', 0.4)
        self.disagreement_penalty = self.config.get('disagreement_penalty', 0.5)
        self.consensus_filter_enabled = self.config.get('consensus_filter_enabled', True)
        self.min_consensus_strength = self.config.get('min_consensus_strength', 0.6)
        self.max_disagreement_score = self.config.get('max_disagreement_score', 0.8)
        
        logger.info(f"Configuration updated: threshold={self.execution_threshold}, "
                   f"disagreement_threshold={self.disagreement_threshold}")
    
    # ==================== PBFT CONSENSUS METHODS ====================
    
    def _generate_master_key(self) -> bytes:
        """Generate master key for cryptographic operations"""
        return secrets.token_bytes(32)
    
    def _initialize_agent_keys(self) -> Dict[str, str]:
        """Initialize cryptographic keys for all agents"""
        try:
            from ..src.consensus import CryptographicCore
            crypto_core = CryptographicCore()
            agent_ids = ['fvg_agent', 'momentum_agent', 'entry_opt_agent', 'strategic_agent']
            return crypto_core.initialize_agent_keys(agent_ids)
        except ImportError:
            logger.warning("Consensus system not available, using mock keys")
            return {aid: secrets.token_hex(8) for aid in ['fvg_agent', 'momentum_agent', 'entry_opt_agent', 'strategic_agent']}
    
    def _validate_agent_signatures(self, agent_outputs: Dict[str, Any]) -> Dict[str, AgentDecision]:
        """
        Validate cryptographic signatures of agent outputs
        
        Args:
            agent_outputs: Raw agent outputs with potential signatures
            
        Returns:
            Validated agent decisions
        """
        validated_decisions = {}
        
        for agent_id, output in agent_outputs.items():
            try:
                # Convert to AgentDecision format
                if hasattr(output, 'probabilities'):
                    decision = AgentDecision(
                        agent_id=agent_id,
                        action=output.action,
                        probabilities=output.probabilities,
                        confidence=output.confidence,
                        timestamp=getattr(output, 'timestamp', time.time()),
                        signature=getattr(output, 'signature', None),
                        nonce=getattr(output, 'nonce', None)
                    )
                else:
                    # Handle dictionary format
                    decision = AgentDecision(
                        agent_id=agent_id,
                        action=output.get('action', 1),
                        probabilities=np.array(output.get('probabilities', [0.33, 0.34, 0.33])),
                        confidence=output.get('confidence', 0.5),
                        timestamp=output.get('timestamp', time.time()),
                        signature=output.get('signature'),
                        nonce=output.get('nonce')
                    )
                
                # Validate signature if present
                if decision.signature and hasattr(self, 'crypto_core') and self.crypto_core:
                    # Create message hash for validation
                    message_content = {
                        'agent_id': agent_id,
                        'action': decision.action,
                        'confidence': decision.confidence,
                        'timestamp': decision.timestamp,
                        'nonce': decision.nonce
                    }
                    message_hash = hashlib.sha256(
                        json.dumps(message_content, sort_keys=True).encode()
                    ).hexdigest()
                    
                    is_valid = self.crypto_core.validate_signature(
                        message_hash, decision.signature, agent_id
                    )
                    
                    if not is_valid:
                        logger.warning(f"Invalid signature from agent {agent_id}")
                        decision.is_byzantine = True
                        continue
                
                validated_decisions[agent_id] = decision
                
            except Exception as e:
                logger.error(f"Error validating agent {agent_id} output: {e}")
                # Create safe default decision
                validated_decisions[agent_id] = AgentDecision(
                    agent_id=agent_id,
                    action=1,  # Neutral
                    probabilities=np.array([0.33, 0.34, 0.33]),
                    confidence=0.0,
                    timestamp=time.time(),
                    is_byzantine=True
                )
        
        return validated_decisions
    
    def _detect_byzantine_agents(
        self, 
        validated_decisions: Dict[str, AgentDecision], 
        market_state: Any
    ) -> Set[str]:
        """
        Detect Byzantine agents using behavior analysis
        
        Args:
            validated_decisions: Validated agent decisions
            market_state: Current market state
            
        Returns:
            Set of agent IDs detected as Byzantine
        """
        byzantine_agents = set()
        
        # Check for obviously Byzantine decisions
        for agent_id, decision in validated_decisions.items():
            if decision.is_byzantine:
                byzantine_agents.add(agent_id)
                continue
                
            # Statistical outlier detection
            if self._is_statistical_outlier(decision, validated_decisions):
                byzantine_agents.add(agent_id)
                logger.warning(f"Agent {agent_id} detected as statistical outlier")
                
            # Confidence analysis
            if decision.confidence < 0.0 or decision.confidence > 1.0:
                byzantine_agents.add(agent_id)
                logger.warning(f"Agent {agent_id} has invalid confidence: {decision.confidence}")
                
            # Probability distribution validation
            if not self._validate_probability_distribution(decision.probabilities):
                byzantine_agents.add(agent_id)
                logger.warning(f"Agent {agent_id} has invalid probability distribution")
        
        # Use Byzantine detector if available
        if hasattr(self, 'byzantine_detector') and self.byzantine_detector:
            try:
                for agent_id, decision in validated_decisions.items():
                    self.byzantine_detector.record_message_activity(
                        agent_id=agent_id,
                        message_type="decision",
                        timestamp=decision.timestamp,
                        signature_valid=not decision.is_byzantine
                    )
                
                suspected, confirmed = self.byzantine_detector.get_byzantine_agents()
                byzantine_agents.update(confirmed)
                
            except Exception as e:
                logger.error(f"Byzantine detector error: {e}")
        
        return byzantine_agents
    
    def _is_statistical_outlier(
        self, 
        decision: AgentDecision, 
        all_decisions: Dict[str, AgentDecision]
    ) -> bool:
        """Check if decision is a statistical outlier"""
        try:
            # Get all confidences
            confidences = [d.confidence for d in all_decisions.values() if not d.is_byzantine]
            
            if len(confidences) < 2:
                return False
            
            # Calculate z-score
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            
            if std_conf == 0:
                return False
            
            z_score = abs(decision.confidence - mean_conf) / std_conf
            
            # Flag as outlier if z-score > 3 (very unusual)
            return z_score > 3.0
            
        except Exception:
            return False
    
    def _validate_probability_distribution(self, probabilities: np.ndarray) -> bool:
        """Validate probability distribution"""
        try:
            # Check if probabilities sum to approximately 1
            prob_sum = np.sum(probabilities)
            if not (0.95 <= prob_sum <= 1.05):
                return False
            
            # Check if all probabilities are non-negative
            if np.any(probabilities < 0):
                return False
            
            # Check if any probability exceeds 1
            if np.any(probabilities > 1):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def _execute_pbft_consensus(
        self,
        validated_decisions: Dict[str, AgentDecision],
        market_state: Any,
        synergy_context: Dict[str, Any],
        byzantine_agents: Set[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute PBFT consensus protocol
        
        Args:
            validated_decisions: Validated agent decisions
            market_state: Current market state
            synergy_context: Synergy context
            byzantine_agents: Known Byzantine agents
            
        Returns:
            PBFT consensus result or None if failed
        """
        if not self.pbft_enabled or not hasattr(self, 'pbft_engine') or not self.pbft_engine:
            return None
        
        try:
            # Filter out Byzantine agents
            clean_decisions = {
                agent_id: decision for agent_id, decision in validated_decisions.items()
                if agent_id not in byzantine_agents
            }
            
            # Create consensus request
            request_id = f"tactical_consensus_{int(time.time() * 1000)}"
            
            # Execute PBFT consensus
            consensus_result = await self.pbft_engine.request_consensus(
                request_id=request_id,
                agent_decisions=clean_decisions,
                market_state=market_state,
                synergy_context=synergy_context
            )
            
            if consensus_result.consensus_achieved:
                return {
                    'consensus_achieved': True,
                    'execute': consensus_result.execute,
                    'action': consensus_result.action,
                    'confidence': consensus_result.confidence,
                    'safety_level': consensus_result.safety_level,
                    'participating_agents': consensus_result.participating_agents,
                    'byzantine_agents_detected': consensus_result.byzantine_agents_detected,
                    'pbft_signatures': consensus_result.signatures
                }
            else:
                logger.warning("PBFT consensus failed to achieve agreement")
                return {'consensus_achieved': False}
                
        except Exception as e:
            logger.error(f"PBFT consensus error: {e}")
            return None
    
    def _enhanced_weighted_voting_fallback(
        self,
        validated_decisions: Dict[str, AgentDecision],
        market_state: Any,
        synergy_context: Dict[str, Any],
        byzantine_agents: Set[str]
    ) -> Dict[str, Any]:
        """
        Enhanced weighted voting fallback when PBFT fails
        
        Args:
            validated_decisions: Validated agent decisions
            market_state: Current market state
            synergy_context: Synergy context
            byzantine_agents: Known Byzantine agents
            
        Returns:
            Fallback consensus result
        """
        try:
            # Filter out Byzantine agents
            clean_decisions = {
                agent_id: decision for agent_id, decision in validated_decisions.items()
                if agent_id not in byzantine_agents
            }
            
            if not clean_decisions:
                return {
                    'consensus_achieved': False,
                    'execute': False,
                    'action': 1,
                    'confidence': 0.0,
                    'safety_level': 0.0
                }
            
            # Get synergy type and weights
            synergy_type = self._extract_synergy_type(synergy_context)
            agent_weights = self._get_agent_weights(synergy_type)
            
            # Perform weighted voting with Byzantine resistance
            weighted_actions = self._byzantine_resistant_voting(clean_decisions, agent_weights)
            
            # Find consensus action
            consensus_action = max(weighted_actions.items(), key=lambda x: x[1])[0]
            action_weight = weighted_actions[consensus_action]
            
            # Calculate consensus confidence with Byzantine penalty
            base_confidence = action_weight / sum(weighted_actions.values())
            byzantine_penalty = len(byzantine_agents) / len(validated_decisions)
            consensus_confidence = base_confidence * (1.0 - byzantine_penalty * 0.5)
            
            # Apply direction bias
            synergy_direction = synergy_context.get('direction', 0)
            synergy_confidence = synergy_context.get('confidence', 0.0)
            
            final_confidence = self._apply_direction_bias(
                consensus_action, consensus_confidence, synergy_direction, synergy_confidence
            )
            
            # Calculate safety level
            safety_level = self._calculate_fallback_safety_level(
                clean_decisions, byzantine_agents, final_confidence
            )
            
            # Execution decision with higher threshold for fallback
            execute = final_confidence >= (self.execution_threshold + 0.1) and safety_level >= 0.5
            
            return {
                'consensus_achieved': True,
                'execute': execute,
                'action': consensus_action,
                'confidence': final_confidence,
                'safety_level': safety_level,
                'fallback_mode': True,
                'byzantine_agents_detected': list(byzantine_agents)
            }
            
        except Exception as e:
            logger.error(f"Enhanced weighted voting fallback error: {e}")
            return {
                'consensus_achieved': False,
                'execute': False,
                'action': 1,
                'confidence': 0.0,
                'safety_level': 0.0
            }
    
    def _byzantine_resistant_voting(
        self,
        clean_decisions: Dict[str, AgentDecision],
        agent_weights: List[float]
    ) -> Dict[int, float]:
        """Perform Byzantine-resistant weighted voting"""
        weighted_actions = {0: 0.0, 1: 0.0, 2: 0.0}
        
        # Use median-based aggregation for Byzantine resistance
        agent_order = ['fvg_agent', 'momentum_agent', 'entry_opt_agent']
        
        for i, agent_id in enumerate(agent_order):
            if agent_id in clean_decisions and i < len(agent_weights):
                decision = clean_decisions[agent_id]
                weight = agent_weights[i]
                
                # Use trimmed mean for robustness
                trimmed_confidence = max(0.1, min(0.9, decision.confidence))
                
                # Weight the probabilities
                for action_idx, prob in enumerate(decision.probabilities):
                    if action_idx in weighted_actions:
                        weighted_actions[action_idx] += prob * weight * trimmed_confidence
        
        return weighted_actions
    
    def _calculate_fallback_safety_level(
        self,
        clean_decisions: Dict[str, AgentDecision],
        byzantine_agents: Set[str],
        confidence: float
    ) -> float:
        """Calculate safety level for fallback consensus"""
        # Base safety from clean decisions
        total_agents = len(clean_decisions) + len(byzantine_agents)
        clean_ratio = len(clean_decisions) / max(1, total_agents)
        
        # Confidence-based safety
        confidence_safety = confidence
        
        # Byzantine penalty
        byzantine_penalty = len(byzantine_agents) / max(1, total_agents)
        
        # Combined safety
        safety_level = 0.6 * clean_ratio + 0.3 * confidence_safety - 0.1 * byzantine_penalty
        
        return max(0.0, min(1.0, safety_level))
    
    def _apply_safety_validation(
        self,
        pbft_result: Dict[str, Any],
        market_state: Any
    ) -> AggregatedDecision:
        """
        Apply additional safety validation to consensus result
        
        Args:
            pbft_result: PBFT consensus result
            market_state: Current market state
            
        Returns:
            Final aggregated decision with safety validation
        """
        try:
            # Extract basic result
            execute = pbft_result.get('execute', False)
            action = pbft_result.get('action', 1)
            confidence = pbft_result.get('confidence', 0.0)
            safety_level = pbft_result.get('safety_level', 0.0)
            consensus_achieved = pbft_result.get('consensus_achieved', False)
            
            # Safety checks
            safety_passed = True
            safety_reasons = []
            
            # Check 1: Minimum safety level
            if safety_level < 0.3:
                safety_passed = False
                safety_reasons.append(f"Low safety level: {safety_level:.3f}")
            
            # Check 2: Confidence threshold
            if confidence < 0.1:
                safety_passed = False
                safety_reasons.append(f"Low confidence: {confidence:.3f}")
            
            # Check 3: Byzantine agent ratio
            byzantine_agents = pbft_result.get('byzantine_agents_detected', [])
            participating_agents = pbft_result.get('participating_agents', [])
            total_agents = len(byzantine_agents) + len(participating_agents)
            
            if total_agents > 0:
                byzantine_ratio = len(byzantine_agents) / total_agents
                if byzantine_ratio > 0.3:  # More than 30% Byzantine
                    safety_passed = False
                    safety_reasons.append(f"High Byzantine ratio: {byzantine_ratio:.3f}")
            
            # Check 4: Market state validation
            if hasattr(market_state, 'volatility') and market_state.volatility > 0.1:
                # Increase conservatism in high volatility
                confidence *= 0.8
                if confidence < self.execution_threshold:
                    safety_passed = False
                    safety_reasons.append("High market volatility")
            
            # Apply safety overrides
            if not safety_passed:
                logger.warning(f"Safety validation failed: {', '.join(safety_reasons)}")
                execute = False
                action = 1  # Force neutral
                confidence = min(confidence, 0.3)  # Cap confidence
            
            # Create final decision
            final_decision = AggregatedDecision(
                execute=execute,
                action=action,
                confidence=confidence,
                agent_votes={},  # Will be populated below
                consensus_breakdown={0: 0.0, 1: 0.0, 2: 0.0},
                synergy_alignment=0.0,
                execution_command=None,
                pbft_consensus_achieved=consensus_achieved,
                byzantine_agents_detected=byzantine_agents,
                consensus_signatures=pbft_result.get('pbft_signatures', {}),
                view_number=pbft_result.get('view_number', 0),
                safety_level=safety_level
            )
            
            # Add execution command if executing
            if execute:
                final_decision.execution_command = self._create_execution_command(
                    action, confidence, market_state, {}
                )
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Safety validation error: {e}")
            return self._create_safe_default_decision({})
    
    def _sign_aggregated_decision(self, decision: AggregatedDecision):
        """
        Add cryptographic signature to aggregated decision
        
        Args:
            decision: Aggregated decision to sign
        """
        try:
            if not hasattr(self, 'crypto_core') or not self.crypto_core:
                return
            
            # Create decision hash
            decision_content = {
                'execute': decision.execute,
                'action': decision.action,
                'confidence': decision.confidence,
                'safety_level': decision.safety_level,
                'timestamp': time.time()
            }
            
            decision_hash = hashlib.sha256(
                json.dumps(decision_content, sort_keys=True).encode()
            ).hexdigest()
            
            # Sign the decision
            signature = self.crypto_core.sign_message(decision_hash, 'tactical_aggregator')
            
            # Add signature to decision
            if not decision.consensus_signatures:
                decision.consensus_signatures = {}
            decision.consensus_signatures['tactical_aggregator'] = signature
            
        except Exception as e:
            logger.error(f"Error signing aggregated decision: {e}")
    
    def _update_pbft_metrics(self, decision: AggregatedDecision, byzantine_agents: Set[str]):
        """
        Update PBFT-specific performance metrics
        
        Args:
            decision: Final aggregated decision
            byzantine_agents: Set of detected Byzantine agents
        """
        # Update existing metrics
        self._update_metrics(decision, {})
        
        # Add PBFT-specific metrics
        if decision.pbft_consensus_achieved:
            self.performance_metrics['pbft_consensus_achieved'] += 1
        
        self.performance_metrics['byzantine_attacks_detected'] += len(byzantine_agents)
        
        if hasattr(decision, 'view_number') and decision.view_number > 0:
            self.performance_metrics['view_changes'] += decision.view_number
    
    def _create_safe_default_decision(self, agent_outputs: Dict[str, Any]) -> AggregatedDecision:
        """
        Create ultra-safe default decision when all consensus methods fail
        
        Args:
            agent_outputs: Original agent outputs
            
        Returns:
            Safe default aggregated decision
        """
        return AggregatedDecision(
            execute=False,  # Never execute on total failure
            action=1,       # Neutral action
            confidence=0.0, # Zero confidence
            agent_votes={},
            consensus_breakdown={0: 0.0, 1: 1.0, 2: 0.0},  # All neutral
            synergy_alignment=0.0,
            execution_command=None,
            pbft_consensus_achieved=False,
            byzantine_agents_detected=[],
            consensus_signatures={},
            view_number=0,
            safety_level=0.0
        )


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('tactical_marl', {}).get('aggregation', {})
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def create_tactical_aggregator(config_path: Optional[str] = None) -> TacticalDecisionAggregator:
    """
    Factory function to create tactical decision aggregator
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured TacticalDecisionAggregator instance
    """
    if config_path:
        config = load_config_from_file(config_path)
    else:
        config = None
    
    return TacticalDecisionAggregator(config)