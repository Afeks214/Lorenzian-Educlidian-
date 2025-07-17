"""
Decision Aggregator for Strategic MARL

Implements weighted ensemble decision-making with uncertainty quantification
using configuration-driven weights for flexibility.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
import yaml
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AggregatedDecision:
    """Container for aggregated decision output"""
    ensemble_probabilities: np.ndarray  # Combined agent outputs
    individual_actions: Dict[str, np.ndarray]  # Per-agent decisions
    confidence: float  # Overall confidence
    uncertainty: float  # Decision uncertainty
    should_proceed: bool  # Final binary decision
    reasoning: Dict[str, Any]  # Explanation features


class DecisionAggregator:
    """
    Weighted Ensemble Decision Aggregator
    
    Combines multiple agent outputs using learned or configured weights,
    providing uncertainty quantification and decision reasoning.
    
    Key Features:
    - Configuration-driven weights for rapid tuning
    - Confidence calculation using max probability and entropy
    - Historical performance tracking for adaptive weighting
    - Comprehensive reasoning output for explainability
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize Decision Aggregator
        
        Args:
            config_path: Path to configuration YAML file
            config: Direct configuration dictionary (overrides config_path)
        """
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Extract key parameters
        self.ensemble_weights = np.array(self.config['ensemble']['weights'])
        self.confidence_threshold = self.config['ensemble']['confidence_threshold']
        self.learning_rate = self.config['ensemble'].get('learning_rate', 1e-3)
        
        # Normalize weights
        self.ensemble_weights = self.ensemble_weights / self.ensemble_weights.sum()
        
        # Performance tracking for adaptive weighting
        self.agent_performance = {
            'mlmi_expert': {'correct': 0, 'total': 0},
            'nwrqk_expert': {'correct': 0, 'total': 0},
            'regime_expert': {'correct': 0, 'total': 0},
        }
        
        # Historical data for analysis
        self.decision_history = []
        
        logger.info(f"DecisionAggregator initialized with weights: {self.ensemble_weights}")
    
    def aggregate(
        self, 
        agent_outputs: Dict[str, np.ndarray],
        synergy_info: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> AggregatedDecision:
        """
        Aggregate multiple agent outputs into ensemble decision
        
        Args:
            agent_outputs: Dictionary mapping agent names to probability vectors
            synergy_info: Current synergy pattern information
            market_context: Market conditions (volatility, etc.)
            
        Returns:
            AggregatedDecision with ensemble output and metadata
        """
        # Validate inputs
        self._validate_agent_outputs(agent_outputs)
        
        # Extract individual probabilities
        agent_names = ['mlmi_expert', 'nwrqk_expert', 'regime_expert']
        agent_probs = [agent_outputs[name] for name in agent_names]
        
        # Calculate weighted ensemble
        ensemble_probs = self._weighted_average(agent_probs)
        
        # Calculate confidence and uncertainty
        confidence = float(np.max(ensemble_probs))
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-8))
        uncertainty = entropy / np.log(len(ensemble_probs))  # Normalized entropy
        
        # Determine binary decision
        should_proceed = confidence > self.confidence_threshold
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            agent_outputs, 
            ensemble_probs,
            synergy_info,
            market_context
        )
        
        # Create decision object
        decision = AggregatedDecision(
            ensemble_probabilities=ensemble_probs,
            individual_actions=agent_outputs.copy(),
            confidence=confidence,
            uncertainty=uncertainty,
            should_proceed=should_proceed,
            reasoning=reasoning
        )
        
        # Store in history
        self.decision_history.append({
            'timestamp': self._get_timestamp(),
            'decision': decision,
            'synergy': synergy_info,
            'market': market_context
        })
        
        return decision
    
    def update_weights(self, performance_feedback: Dict[str, float]):
        """
        Update ensemble weights based on agent performance
        
        Args:
            performance_feedback: Dictionary with agent performance scores
        """
        # Update performance tracking
        for agent, score in performance_feedback.items():
            if agent in self.agent_performance:
                self.agent_performance[agent]['total'] += 1
                if score > 0:
                    self.agent_performance[agent]['correct'] += 1
        
        # Calculate performance-based adjustments
        accuracies = []
        for agent in ['mlmi_expert', 'nwrqk_expert', 'regime_expert']:
            stats = self.agent_performance[agent]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
            else:
                accuracy = 0.5  # Default
            accuracies.append(accuracy)
        
        # Update weights using exponential moving average
        new_weights = np.array(accuracies)
        new_weights = new_weights / new_weights.sum()
        
        self.ensemble_weights = (
            (1 - self.learning_rate) * self.ensemble_weights +
            self.learning_rate * new_weights
        )
        
        # Renormalize
        self.ensemble_weights = self.ensemble_weights / self.ensemble_weights.sum()
        
        logger.info(f"Updated weights: {self.ensemble_weights}")
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about historical decisions"""
        if not self.decision_history:
            return {}
        
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        
        confidences = [d['decision'].confidence for d in recent_decisions]
        uncertainties = [d['decision'].uncertainty for d in recent_decisions]
        proceed_rate = sum(d['decision'].should_proceed for d in recent_decisions) / len(recent_decisions)
        
        # Agent agreement analysis
        agreements = []
        for d in recent_decisions:
            actions = list(d['decision'].individual_actions.values())
            # Check if all agents agree on the highest probability action
            max_actions = [np.argmax(a) for a in actions]
            agreements.append(len(set(max_actions)) == 1)
        
        agreement_rate = sum(agreements) / len(agreements) if agreements else 0
        
        return {
            'avg_confidence': np.mean(confidences),
            'avg_uncertainty': np.mean(uncertainties),
            'proceed_rate': proceed_rate,
            'agent_agreement_rate': agreement_rate,
            'current_weights': self.ensemble_weights.tolist(),
            'total_decisions': len(self.decision_history)
        }
    
    # Private helper methods
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'ensemble': {
                'weights': [0.4, 0.35, 0.25],  # MLMI, NWRQK, Regime
                'confidence_threshold': 0.65,
                'learning_rate': 1e-3,
            }
        }
    
    def _validate_agent_outputs(self, agent_outputs: Dict[str, np.ndarray]):
        """Validate agent outputs are valid probability distributions"""
        required_agents = ['mlmi_expert', 'nwrqk_expert', 'regime_expert']
        
        for agent in required_agents:
            if agent not in agent_outputs:
                raise ValueError(f"Missing output for agent: {agent}")
            
            output = agent_outputs[agent]
            if not isinstance(output, np.ndarray) or output.shape != (3,):
                raise ValueError(f"Invalid output shape for {agent}: {output.shape}")
            
            # Check if valid probability distribution
            if not np.allclose(output.sum(), 1.0, atol=1e-6):
                logger.warning(f"Normalizing output for {agent}: sum={output.sum()}")
                agent_outputs[agent] = output / output.sum()
    
    def _weighted_average(self, agent_probs: List[np.ndarray]) -> np.ndarray:
        """Calculate weighted average of agent probabilities"""
        # Stack probabilities
        prob_matrix = np.stack(agent_probs)  # Shape: (n_agents, 3)
        
        # Apply weights
        weighted_probs = prob_matrix * self.ensemble_weights.reshape(-1, 1)
        
        # Sum and normalize
        ensemble_probs = weighted_probs.sum(axis=0)
        ensemble_probs = ensemble_probs / ensemble_probs.sum()
        
        return ensemble_probs
    
    def _generate_reasoning(
        self,
        agent_outputs: Dict[str, np.ndarray],
        ensemble_probs: np.ndarray,
        synergy_info: Optional[Dict[str, Any]],
        market_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive reasoning for the decision"""
        action_names = ['bearish', 'neutral', 'bullish']
        
        # Determine ensemble action
        ensemble_action_idx = np.argmax(ensemble_probs)
        ensemble_action = action_names[ensemble_action_idx]
        
        # Analyze agent agreements
        agent_agreements = {}
        for agent, probs in agent_outputs.items():
            agent_action_idx = np.argmax(probs)
            agent_agreements[agent] = {
                'prediction': action_names[agent_action_idx],
                'confidence': float(probs[agent_action_idx]),
                'agrees_with_ensemble': agent_action_idx == ensemble_action_idx
            }
        
        # Identify dominant signal
        agent_confidences = {
            agent: float(np.max(probs)) 
            for agent, probs in agent_outputs.items()
        }
        dominant_agent = max(agent_confidences, key=agent_confidences.get)
        
        # Analyze uncertainty sources
        uncertainty_sources = []
        
        # Check for disagreement
        unique_predictions = set(ag['prediction'] for ag in agent_agreements.values())
        if len(unique_predictions) > 1:
            uncertainty_sources.append('agent_disagreement')
        
        # Check for low confidence
        if all(conf < 0.6 for conf in agent_confidences.values()):
            uncertainty_sources.append('low_individual_confidence')
        
        # Check market conditions
        if market_context and market_context.get('volatility_30', 0) > 1.5:
            uncertainty_sources.append('high_volatility')
        
        reasoning = {
            'ensemble_action': ensemble_action,
            'agent_agreements': agent_agreements,
            'dominant_signal': dominant_agent,
            'dominant_confidence': agent_confidences[dominant_agent],
            'uncertainty_sources': uncertainty_sources,
            'synergy_type': synergy_info.get('type', 'None') if synergy_info else 'None',
            'synergy_alignment': self._check_synergy_alignment(
                ensemble_action_idx, synergy_info
            ),
            'market_regime': self._classify_market_regime(market_context),
            'decision_quality': self._assess_decision_quality(
                ensemble_probs, agent_agreements
            )
        }
        
        return reasoning
    
    def _check_synergy_alignment(
        self, 
        action_idx: int, 
        synergy_info: Optional[Dict[str, Any]]
    ) -> str:
        """Check if action aligns with synergy direction"""
        if not synergy_info:
            return 'no_synergy'
        
        synergy_direction = synergy_info.get('direction', 0)
        
        # Map action index to direction: 0=bearish(-1), 1=neutral(0), 2=bullish(1)
        action_direction = action_idx - 1
        
        if synergy_direction == 0:
            return 'neutral_synergy'
        elif synergy_direction * action_direction > 0:
            return 'aligned'
        elif synergy_direction * action_direction < 0:
            return 'misaligned'
        else:
            return 'neutral_action'
    
    def _classify_market_regime(self, market_context: Optional[Dict[str, Any]]) -> str:
        """Classify market regime based on context"""
        if not market_context:
            return 'unknown'
        
        volatility = market_context.get('volatility_30', 1.0)
        
        if volatility < 0.5:
            return 'low_volatility'
        elif volatility < 1.5:
            return 'normal'
        elif volatility < 2.5:
            return 'high_volatility'
        else:
            return 'extreme_volatility'
    
    def _assess_decision_quality(
        self, 
        ensemble_probs: np.ndarray,
        agent_agreements: Dict[str, Dict]
    ) -> str:
        """Assess overall decision quality"""
        confidence = float(np.max(ensemble_probs))
        agreement_count = sum(
            1 for ag in agent_agreements.values() 
            if ag['agrees_with_ensemble']
        )
        
        if confidence > 0.8 and agreement_count == 3:
            return 'high_quality'
        elif confidence > 0.65 and agreement_count >= 2:
            return 'good_quality'
        elif confidence > 0.5:
            return 'moderate_quality'
        else:
            return 'low_quality'
    
    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()