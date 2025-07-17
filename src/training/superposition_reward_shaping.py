"""
Superposition Reward Shaping for MARL Training

This module implements sophisticated reward shaping functions specifically designed
for training agents that produce superposition outputs while maintaining sequential
coordination in cascade systems.

Key Features:
- Confidence-based reward weighting for superposition states
- Entropy regularization rewards for maintaining decision diversity
- Consistency penalties for conflicting superposition states
- Coordination rewards for sequential agent alignment
- Adaptive reward scaling based on superposition quality

Author: AGENT 9 - Superposition-Aware Training Framework
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import math
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class RewardComponent(Enum):
    """Types of reward components in superposition reward shaping"""
    BASE_PERFORMANCE = "base_performance"
    CONFIDENCE_QUALITY = "confidence_quality"
    ENTROPY_REGULARIZATION = "entropy_regularization"
    CONSISTENCY_PENALTY = "consistency_penalty"
    COORDINATION_BONUS = "coordination_bonus"
    SUPERPOSITION_DIVERSITY = "superposition_diversity"
    TEMPORAL_ALIGNMENT = "temporal_alignment"
    DECISION_STABILITY = "decision_stability"


@dataclass
class SuperpositionRewardConfig:
    """Configuration for superposition reward shaping"""
    # Base reward weights
    base_performance_weight: float = 1.0
    confidence_quality_weight: float = 0.3
    entropy_regularization_weight: float = 0.2
    consistency_penalty_weight: float = 0.4
    coordination_bonus_weight: float = 0.2
    
    # Superposition quality thresholds
    min_entropy_threshold: float = 0.5
    max_entropy_threshold: float = 2.0
    confidence_variance_threshold: float = 0.3
    consistency_threshold: float = 0.7
    
    # Coordination parameters
    coordination_window: int = 3  # Number of agents to consider for coordination
    temporal_discount: float = 0.95
    alignment_threshold: float = 0.6
    
    # Adaptive scaling
    adaptive_scaling: bool = True
    scale_factor_range: Tuple[float, float] = (0.5, 2.0)
    adaptation_rate: float = 0.01
    
    # Curriculum learning support
    curriculum_enabled: bool = True
    difficulty_progression: float = 0.1
    superposition_complexity_factor: float = 0.05


@dataclass
class SuperpositionRewardBreakdown:
    """Detailed breakdown of superposition rewards"""
    base_performance: float
    confidence_quality: float
    entropy_regularization: float
    consistency_penalty: float
    coordination_bonus: float
    superposition_diversity: float
    temporal_alignment: float
    decision_stability: float
    
    # Metadata
    total_reward: float
    scaling_factor: float
    quality_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'base_performance': self.base_performance,
            'confidence_quality': self.confidence_quality,
            'entropy_regularization': self.entropy_regularization,
            'consistency_penalty': self.consistency_penalty,
            'coordination_bonus': self.coordination_bonus,
            'superposition_diversity': self.superposition_diversity,
            'temporal_alignment': self.temporal_alignment,
            'decision_stability': self.decision_stability,
            'total_reward': self.total_reward,
            'scaling_factor': self.scaling_factor,
            'quality_score': self.quality_score
        }


class SuperpositionRewardShaper:
    """Main reward shaping system for superposition-aware training"""
    
    def __init__(self, config: SuperpositionRewardConfig, agent_names: List[str],
                 superposition_dim: int):
        self.config = config
        self.agent_names = agent_names
        self.superposition_dim = superposition_dim
        
        # Reward component calculators
        self.calculators = self._initialize_calculators()
        
        # Adaptive scaling state
        self.scaling_factors = {agent: 1.0 for agent in agent_names}
        self.performance_history = {agent: deque(maxlen=100) for agent in agent_names}
        self.quality_history = {agent: deque(maxlen=100) for agent in agent_names}
        
        # Coordination tracking
        self.coordination_history = deque(maxlen=1000)
        self.agent_sequence_rewards = {agent: deque(maxlen=100) for agent in agent_names}
        
        # Curriculum learning state
        self.curriculum_step = 0
        self.current_difficulty = 0.0
        
        # Statistics
        self.reward_stats = {
            'total_rewards_shaped': 0,
            'component_contributions': defaultdict(float),
            'quality_improvements': defaultdict(float)
        }
        
        logger.info(f"Initialized SuperpositionRewardShaper for {len(agent_names)} agents")
    
    def _initialize_calculators(self) -> Dict[RewardComponent, Callable]:
        """Initialize reward component calculators"""
        return {
            RewardComponent.BASE_PERFORMANCE: self._calculate_base_performance,
            RewardComponent.CONFIDENCE_QUALITY: self._calculate_confidence_quality,
            RewardComponent.ENTROPY_REGULARIZATION: self._calculate_entropy_regularization,
            RewardComponent.CONSISTENCY_PENALTY: self._calculate_consistency_penalty,
            RewardComponent.COORDINATION_BONUS: self._calculate_coordination_bonus,
            RewardComponent.SUPERPOSITION_DIVERSITY: self._calculate_superposition_diversity,
            RewardComponent.TEMPORAL_ALIGNMENT: self._calculate_temporal_alignment,
            RewardComponent.DECISION_STABILITY: self._calculate_decision_stability
        }
    
    def shape_rewards(self, agent_name: str, base_reward: float,
                     superposition_output: Dict[str, torch.Tensor],
                     coordination_context: Dict[str, Any],
                     episode_context: Dict[str, Any]) -> SuperpositionRewardBreakdown:
        """
        Shape rewards for superposition-aware training
        
        Args:
            agent_name: Name of the agent
            base_reward: Base reward from environment
            superposition_output: Superposition network outputs
            coordination_context: Context for coordination evaluation
            episode_context: Episode-level context
            
        Returns:
            Detailed reward breakdown
        """
        # Extract superposition components
        decision_states = superposition_output.get('decision_states')
        confidence_weights = superposition_output.get('confidence_weights')
        entropy = superposition_output.get('entropy')
        consistency_score = superposition_output.get('consistency_score')
        
        # Calculate individual reward components
        components = {}
        
        # Base performance (original reward)
        components[RewardComponent.BASE_PERFORMANCE] = self._calculate_base_performance(
            base_reward, agent_name, episode_context
        )
        
        # Confidence quality reward
        components[RewardComponent.CONFIDENCE_QUALITY] = self._calculate_confidence_quality(
            confidence_weights, agent_name, episode_context
        )
        
        # Entropy regularization
        components[RewardComponent.ENTROPY_REGULARIZATION] = self._calculate_entropy_regularization(
            entropy, agent_name, episode_context
        )
        
        # Consistency penalty
        components[RewardComponent.CONSISTENCY_PENALTY] = self._calculate_consistency_penalty(
            decision_states, consistency_score, agent_name, episode_context
        )
        
        # Coordination bonus
        components[RewardComponent.COORDINATION_BONUS] = self._calculate_coordination_bonus(
            superposition_output, coordination_context, agent_name, episode_context
        )
        
        # Superposition diversity
        components[RewardComponent.SUPERPOSITION_DIVERSITY] = self._calculate_superposition_diversity(
            decision_states, confidence_weights, agent_name, episode_context
        )
        
        # Temporal alignment
        components[RewardComponent.TEMPORAL_ALIGNMENT] = self._calculate_temporal_alignment(
            superposition_output, coordination_context, agent_name, episode_context
        )
        
        # Decision stability
        components[RewardComponent.DECISION_STABILITY] = self._calculate_decision_stability(
            superposition_output, coordination_context, agent_name, episode_context
        )
        
        # Calculate total reward
        total_reward = self._combine_reward_components(components, agent_name)
        
        # Apply adaptive scaling
        scaling_factor = self._get_adaptive_scaling_factor(agent_name, components)
        total_reward *= scaling_factor
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(components, superposition_output)
        
        # Update statistics
        self._update_statistics(agent_name, components, total_reward, quality_score)
        
        # Create reward breakdown
        breakdown = SuperpositionRewardBreakdown(
            base_performance=components[RewardComponent.BASE_PERFORMANCE],
            confidence_quality=components[RewardComponent.CONFIDENCE_QUALITY],
            entropy_regularization=components[RewardComponent.ENTROPY_REGULARIZATION],
            consistency_penalty=components[RewardComponent.CONSISTENCY_PENALTY],
            coordination_bonus=components[RewardComponent.COORDINATION_BONUS],
            superposition_diversity=components[RewardComponent.SUPERPOSITION_DIVERSITY],
            temporal_alignment=components[RewardComponent.TEMPORAL_ALIGNMENT],
            decision_stability=components[RewardComponent.DECISION_STABILITY],
            total_reward=total_reward,
            scaling_factor=scaling_factor,
            quality_score=quality_score
        )
        
        return breakdown
    
    def _calculate_base_performance(self, base_reward: float, agent_name: str, 
                                  episode_context: Dict[str, Any]) -> float:
        """Calculate base performance reward component"""
        # Apply curriculum scaling if enabled
        if self.config.curriculum_enabled:
            curriculum_factor = 1.0 + self.current_difficulty * self.config.difficulty_progression
            base_reward *= curriculum_factor
        
        return base_reward * self.config.base_performance_weight
    
    def _calculate_confidence_quality(self, confidence_weights: torch.Tensor, 
                                    agent_name: str, episode_context: Dict[str, Any]) -> float:
        """Calculate confidence quality reward"""
        if confidence_weights is None:
            return 0.0
        
        # Reward well-calibrated confidence
        confidence_entropy = -(confidence_weights * torch.log(confidence_weights + 1e-8)).sum()
        
        # Penalty for overly uniform confidence (lack of commitment)
        max_confidence = torch.max(confidence_weights)
        commitment_bonus = max_confidence.item() if max_confidence > 0.7 else 0.0
        
        # Penalty for extreme confidence variance
        confidence_var = confidence_weights.var().item()
        variance_penalty = max(0, confidence_var - self.config.confidence_variance_threshold)
        
        # Combined confidence quality
        confidence_quality = (
            0.4 * (1.0 - confidence_entropy.item() / math.log(self.superposition_dim)) +
            0.3 * commitment_bonus +
            0.3 * (1.0 - variance_penalty)
        )
        
        return confidence_quality * self.config.confidence_quality_weight
    
    def _calculate_entropy_regularization(self, entropy: torch.Tensor, 
                                        agent_name: str, episode_context: Dict[str, Any]) -> float:
        """Calculate entropy regularization reward"""
        if entropy is None:
            return 0.0
        
        entropy_value = entropy.item() if isinstance(entropy, torch.Tensor) else entropy
        
        # Reward entropy within target range
        if self.config.min_entropy_threshold <= entropy_value <= self.config.max_entropy_threshold:
            # Optimal entropy range
            entropy_reward = 1.0
        elif entropy_value < self.config.min_entropy_threshold:
            # Penalty for too low entropy (lack of exploration)
            entropy_reward = entropy_value / self.config.min_entropy_threshold
        else:
            # Penalty for too high entropy (lack of commitment)
            excess_entropy = entropy_value - self.config.max_entropy_threshold
            entropy_reward = 1.0 - (excess_entropy / self.config.max_entropy_threshold)
        
        return entropy_reward * self.config.entropy_regularization_weight
    
    def _calculate_consistency_penalty(self, decision_states: torch.Tensor, 
                                     consistency_score: torch.Tensor,
                                     agent_name: str, episode_context: Dict[str, Any]) -> float:
        """Calculate consistency penalty for conflicting superposition states"""
        if decision_states is None or consistency_score is None:
            return 0.0
        
        # Base consistency score
        base_consistency = consistency_score.item() if isinstance(consistency_score, torch.Tensor) else consistency_score
        
        # Calculate pairwise disagreement between decision states
        if len(decision_states.shape) >= 3:  # [batch, superposition_dim, action_dim]
            batch_size = decision_states.shape[0]
            disagreement_penalty = 0.0
            
            for b in range(batch_size):
                states = decision_states[b]  # [superposition_dim, action_dim]
                
                # Calculate pairwise KL divergence
                total_kl = 0.0
                n_pairs = 0
                
                for i in range(states.shape[0]):
                    for j in range(i + 1, states.shape[0]):
                        probs_i = F.softmax(states[i], dim=0)
                        probs_j = F.softmax(states[j], dim=0)
                        
                        kl_div = F.kl_div(
                            torch.log(probs_i + 1e-8),
                            probs_j,
                            reduction='sum'
                        )
                        
                        total_kl += kl_div.item()
                        n_pairs += 1
                
                avg_kl = total_kl / n_pairs if n_pairs > 0 else 0.0
                disagreement_penalty += avg_kl
            
            disagreement_penalty /= batch_size
        else:
            disagreement_penalty = 0.0
        
        # Combine consistency measures
        consistency_reward = base_consistency - disagreement_penalty
        
        # Apply penalty if below threshold
        if consistency_reward < self.config.consistency_threshold:
            penalty = (self.config.consistency_threshold - consistency_reward) ** 2
            return -penalty * self.config.consistency_penalty_weight
        
        return 0.0  # No penalty if above threshold
    
    def _calculate_coordination_bonus(self, superposition_output: Dict[str, torch.Tensor],
                                    coordination_context: Dict[str, Any],
                                    agent_name: str, episode_context: Dict[str, Any]) -> float:
        """Calculate coordination bonus for sequential alignment"""
        if not coordination_context:
            return 0.0
        
        # Get agent position in sequence
        agent_idx = self.agent_names.index(agent_name)
        
        # No coordination bonus for first agent
        if agent_idx == 0:
            return 0.0
        
        # Get previous agent's context
        prev_agent_context = coordination_context.get('previous_agent')
        if not prev_agent_context:
            return 0.0
        
        # Calculate confidence alignment
        current_confidence = superposition_output.get('confidence_weights')
        prev_confidence = prev_agent_context.get('confidence_weights')
        
        if current_confidence is not None and prev_confidence is not None:
            # Cosine similarity between confidence distributions
            confidence_similarity = F.cosine_similarity(
                current_confidence.flatten(),
                prev_confidence.flatten(),
                dim=0
            ).item()
            
            # Reward high similarity if both agents are confident
            if confidence_similarity > self.config.alignment_threshold:
                current_max_conf = torch.max(current_confidence).item()
                prev_max_conf = torch.max(prev_confidence).item()
                
                if current_max_conf > 0.7 and prev_max_conf > 0.7:
                    coordination_bonus = confidence_similarity * (current_max_conf + prev_max_conf) / 2
                    
                    # Apply temporal discount
                    temporal_discount = self.config.temporal_discount ** agent_idx
                    coordination_bonus *= temporal_discount
                    
                    return coordination_bonus * self.config.coordination_bonus_weight
        
        return 0.0
    
    def _calculate_superposition_diversity(self, decision_states: torch.Tensor,
                                         confidence_weights: torch.Tensor,
                                         agent_name: str, episode_context: Dict[str, Any]) -> float:
        """Calculate reward for maintaining superposition diversity"""
        if decision_states is None or confidence_weights is None:
            return 0.0
        
        # Calculate diversity across superposition states
        if len(decision_states.shape) >= 3:  # [batch, superposition_dim, action_dim]
            batch_size = decision_states.shape[0]
            diversity_reward = 0.0
            
            for b in range(batch_size):
                states = decision_states[b]  # [superposition_dim, action_dim]
                weights = confidence_weights[b]  # [superposition_dim]
                
                # Calculate entropy of each decision state
                state_entropies = []
                for i in range(states.shape[0]):
                    probs = F.softmax(states[i], dim=0)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                    state_entropies.append(entropy)
                
                # Weight entropies by confidence
                weighted_entropy = sum(e * w.item() for e, w in zip(state_entropies, weights))
                
                # Diversity bonus for maintaining multiple viable options
                diversity_bonus = weighted_entropy / math.log(states.shape[1])  # Normalize by max entropy
                diversity_reward += diversity_bonus
            
            diversity_reward /= batch_size
        else:
            diversity_reward = 0.0
        
        return diversity_reward * 0.1  # Small weight for diversity
    
    def _calculate_temporal_alignment(self, superposition_output: Dict[str, torch.Tensor],
                                    coordination_context: Dict[str, Any],
                                    agent_name: str, episode_context: Dict[str, Any]) -> float:
        """Calculate temporal alignment reward"""
        if not coordination_context:
            return 0.0
        
        # Get temporal context
        temporal_context = coordination_context.get('temporal_context', {})
        
        # Check if decisions are temporally consistent
        current_action = superposition_output.get('collapsed_action')
        previous_actions = temporal_context.get('previous_actions', [])
        
        if current_action is not None and previous_actions:
            # Reward consistency in action trends
            recent_actions = previous_actions[-3:]  # Look at last 3 actions
            
            if len(recent_actions) >= 2:
                # Calculate action trend
                action_trend = np.mean(np.diff(recent_actions))
                current_action_val = current_action.item() if isinstance(current_action, torch.Tensor) else current_action
                
                # Reward alignment with trend
                if abs(action_trend) > 0.1:  # Significant trend
                    expected_action = recent_actions[-1] + action_trend
                    alignment_score = 1.0 - abs(current_action_val - expected_action) / len(previous_actions)
                    
                    return max(0, alignment_score) * 0.1
        
        return 0.0
    
    def _calculate_decision_stability(self, superposition_output: Dict[str, torch.Tensor],
                                    coordination_context: Dict[str, Any],
                                    agent_name: str, episode_context: Dict[str, Any]) -> float:
        """Calculate decision stability reward"""
        # Get agent's decision history
        agent_history = self.agent_sequence_rewards.get(agent_name, deque())
        
        if len(agent_history) < 2:
            return 0.0
        
        # Calculate stability in superposition outputs
        current_confidence = superposition_output.get('confidence_weights')
        
        if current_confidence is not None:
            # Compare with recent confidence patterns
            recent_confidences = [h.get('confidence_weights') for h in list(agent_history)[-5:]]
            recent_confidences = [c for c in recent_confidences if c is not None]
            
            if recent_confidences:
                # Calculate variance in confidence patterns
                confidence_variations = []
                for prev_conf in recent_confidences:
                    if prev_conf.shape == current_confidence.shape:
                        variation = F.mse_loss(current_confidence, prev_conf).item()
                        confidence_variations.append(variation)
                
                if confidence_variations:
                    avg_variation = np.mean(confidence_variations)
                    stability_reward = 1.0 / (1.0 + avg_variation)  # Higher reward for lower variation
                    
                    return stability_reward * 0.1
        
        return 0.0
    
    def _combine_reward_components(self, components: Dict[RewardComponent, float],
                                 agent_name: str) -> float:
        """Combine all reward components into total reward"""
        total_reward = 0.0
        
        for component, value in components.items():
            total_reward += value
        
        return total_reward
    
    def _get_adaptive_scaling_factor(self, agent_name: str, 
                                   components: Dict[RewardComponent, float]) -> float:
        """Calculate adaptive scaling factor for agent"""
        if not self.config.adaptive_scaling:
            return 1.0
        
        # Get current scaling factor
        current_factor = self.scaling_factors[agent_name]
        
        # Calculate quality score for this episode
        quality_score = self._calculate_quality_score(components, {})
        
        # Update scaling based on quality trend
        quality_history = self.quality_history[agent_name]
        quality_history.append(quality_score)
        
        if len(quality_history) > 10:
            # Calculate trend
            recent_quality = np.mean(list(quality_history)[-5:])
            older_quality = np.mean(list(quality_history)[-10:-5])
            
            quality_trend = recent_quality - older_quality
            
            # Adjust scaling factor
            if quality_trend > 0:
                # Improving quality - increase scaling
                adjustment = self.config.adaptation_rate * quality_trend
                new_factor = current_factor + adjustment
            else:
                # Declining quality - decrease scaling
                adjustment = self.config.adaptation_rate * abs(quality_trend)
                new_factor = current_factor - adjustment
            
            # Clamp to valid range
            new_factor = np.clip(new_factor, 
                               self.config.scale_factor_range[0], 
                               self.config.scale_factor_range[1])
            
            self.scaling_factors[agent_name] = new_factor
            return new_factor
        
        return current_factor
    
    def _calculate_quality_score(self, components: Dict[RewardComponent, float],
                               superposition_output: Dict[str, torch.Tensor]) -> float:
        """Calculate overall quality score for superposition output"""
        # Base quality from reward components
        base_quality = 0.0
        
        if RewardComponent.CONFIDENCE_QUALITY in components:
            base_quality += 0.3 * components[RewardComponent.CONFIDENCE_QUALITY]
        
        if RewardComponent.ENTROPY_REGULARIZATION in components:
            base_quality += 0.2 * components[RewardComponent.ENTROPY_REGULARIZATION]
        
        if RewardComponent.CONSISTENCY_PENALTY in components:
            base_quality += 0.3 * (1.0 - abs(components[RewardComponent.CONSISTENCY_PENALTY]))
        
        if RewardComponent.COORDINATION_BONUS in components:
            base_quality += 0.2 * components[RewardComponent.COORDINATION_BONUS]
        
        return max(0.0, min(1.0, base_quality))
    
    def _update_statistics(self, agent_name: str, components: Dict[RewardComponent, float],
                          total_reward: float, quality_score: float):
        """Update reward shaping statistics"""
        self.reward_stats['total_rewards_shaped'] += 1
        
        # Update component contributions
        for component, value in components.items():
            self.reward_stats['component_contributions'][component.value] += abs(value)
        
        # Update quality tracking
        self.reward_stats['quality_improvements'][agent_name] = quality_score
        
        # Update agent performance history
        self.performance_history[agent_name].append(total_reward)
    
    def update_curriculum(self, episode_count: int, global_performance: float):
        """Update curriculum learning parameters"""
        if not self.config.curriculum_enabled:
            return
        
        # Increase difficulty based on performance
        if global_performance > 0.7:  # Good performance threshold
            self.current_difficulty += self.config.superposition_complexity_factor
        elif global_performance < 0.3:  # Poor performance threshold
            self.current_difficulty = max(0.0, self.current_difficulty - self.config.superposition_complexity_factor)
        
        self.curriculum_step += 1
        
        logger.debug(f"Curriculum updated: step={self.curriculum_step}, "
                    f"difficulty={self.current_difficulty:.3f}")
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward shaping statistics"""
        return {
            'total_rewards_shaped': self.reward_stats['total_rewards_shaped'],
            'component_contributions': dict(self.reward_stats['component_contributions']),
            'quality_improvements': dict(self.reward_stats['quality_improvements']),
            'scaling_factors': dict(self.scaling_factors),
            'curriculum_step': self.curriculum_step,
            'current_difficulty': self.current_difficulty,
            'performance_trends': {
                agent: {
                    'mean': np.mean(list(history)) if history else 0.0,
                    'trend': np.mean(list(history)[-10:]) - np.mean(list(history)[-20:-10]) if len(history) >= 20 else 0.0
                }
                for agent, history in self.performance_history.items()
            }
        }
    
    def save_state(self, filepath: str):
        """Save reward shaper state"""
        state = {
            'config': self.config,
            'scaling_factors': self.scaling_factors,
            'performance_history': {
                agent: list(history) for agent, history in self.performance_history.items()
            },
            'quality_history': {
                agent: list(history) for agent, history in self.quality_history.items()
            },
            'coordination_history': list(self.coordination_history),
            'curriculum_step': self.curriculum_step,
            'current_difficulty': self.current_difficulty,
            'reward_stats': dict(self.reward_stats)
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Reward shaper state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load reward shaper state"""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.scaling_factors = state['scaling_factors']
        self.performance_history = {
            agent: deque(history, maxlen=100) for agent, history in state['performance_history'].items()
        }
        self.quality_history = {
            agent: deque(history, maxlen=100) for agent, history in state['quality_history'].items()
        }
        self.coordination_history = deque(state['coordination_history'], maxlen=1000)
        self.curriculum_step = state['curriculum_step']
        self.current_difficulty = state['current_difficulty']
        self.reward_stats = defaultdict(float, state['reward_stats'])
        
        logger.info(f"Reward shaper state loaded from {filepath}")


class MultiAgentSuperpositionRewardSystem:
    """System for coordinating reward shaping across multiple agents"""
    
    def __init__(self, agent_names: List[str], config: SuperpositionRewardConfig,
                 superposition_dim: int):
        self.agent_names = agent_names
        self.config = config
        self.superposition_dim = superposition_dim
        
        # Individual agent reward shapers
        self.agent_shapers = {
            agent: SuperpositionRewardShaper(config, agent_names, superposition_dim)
            for agent in agent_names
        }
        
        # Global coordination tracking
        self.global_coordination_scores = deque(maxlen=1000)
        self.system_performance_history = deque(maxlen=1000)
        
    def shape_multi_agent_rewards(self, agent_rewards: Dict[str, float],
                                 superposition_outputs: Dict[str, Dict[str, torch.Tensor]],
                                 coordination_context: Dict[str, Any],
                                 episode_context: Dict[str, Any]) -> Dict[str, SuperpositionRewardBreakdown]:
        """Shape rewards for all agents simultaneously"""
        shaped_rewards = {}
        
        for agent_name in self.agent_names:
            base_reward = agent_rewards.get(agent_name, 0.0)
            superposition_output = superposition_outputs.get(agent_name, {})
            
            # Add global coordination context
            agent_coordination_context = coordination_context.copy()
            agent_coordination_context['global_performance'] = np.mean(list(agent_rewards.values()))
            
            # Shape reward
            shaped_reward = self.agent_shapers[agent_name].shape_rewards(
                agent_name=agent_name,
                base_reward=base_reward,
                superposition_output=superposition_output,
                coordination_context=agent_coordination_context,
                episode_context=episode_context
            )
            
            shaped_rewards[agent_name] = shaped_reward
        
        # Update global coordination tracking
        self._update_global_coordination(shaped_rewards, episode_context)
        
        return shaped_rewards
    
    def _update_global_coordination(self, shaped_rewards: Dict[str, SuperpositionRewardBreakdown],
                                   episode_context: Dict[str, Any]):
        """Update global coordination metrics"""
        # Calculate system-wide coordination score
        coordination_scores = [r.coordination_bonus for r in shaped_rewards.values()]
        global_coordination = np.mean(coordination_scores) if coordination_scores else 0.0
        
        self.global_coordination_scores.append(global_coordination)
        
        # Calculate system performance
        total_rewards = [r.total_reward for r in shaped_rewards.values()]
        system_performance = np.mean(total_rewards) if total_rewards else 0.0
        
        self.system_performance_history.append(system_performance)
        
        # Update curriculum for all agents
        if len(self.system_performance_history) >= 10:
            recent_performance = np.mean(list(self.system_performance_history)[-10:])
            for shaper in self.agent_shapers.values():
                shaper.update_curriculum(len(self.system_performance_history), recent_performance)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide reward shaping statistics"""
        agent_stats = {}
        for agent_name, shaper in self.agent_shapers.items():
            agent_stats[agent_name] = shaper.get_reward_statistics()
        
        return {
            'agent_statistics': agent_stats,
            'global_coordination_score': np.mean(list(self.global_coordination_scores)) if self.global_coordination_scores else 0.0,
            'system_performance_trend': np.mean(list(self.system_performance_history)[-10:]) - np.mean(list(self.system_performance_history)[-20:-10]) if len(self.system_performance_history) >= 20 else 0.0,
            'coordination_stability': np.std(list(self.global_coordination_scores)) if self.global_coordination_scores else 0.0
        }