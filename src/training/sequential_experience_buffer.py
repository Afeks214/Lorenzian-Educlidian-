"""
Sequential Experience Buffer for Superposition-Aware MARL Training

This module implements a specialized experience buffer that handles sequential
agent coordination and superposition state management for training cascade
agent systems.

Key Features:
- Sequential trajectory management with proper temporal ordering
- Superposition state storage and retrieval
- Coordination-aware experience sampling
- Temporal dependency tracking
- Cascade-specific experience replay strategies

Author: AGENT 9 - Superposition-Aware Training Framework
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
import random
import pickle
import logging
from pathlib import Path
import threading
import time
from enum import Enum

logger = logging.getLogger(__name__)


class SequencePhase(Enum):
    """Phases of sequential agent execution"""
    REGIME_DETECTION = "regime_detection"
    STRUCTURE_ANALYSIS = "structure_analysis"
    TACTICAL_DECISION = "tactical_decision"
    RISK_ASSESSMENT = "risk_assessment"
    EXECUTION = "execution"


@dataclass
class SuperpositionState:
    """Container for superposition state information"""
    decision_states: torch.Tensor  # [superposition_dim, action_dim]
    confidence_weights: torch.Tensor  # [superposition_dim]
    entropy: float
    consistency_score: float
    collapsed_action: int
    collapsed_log_prob: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'decision_states': self.decision_states.cpu().numpy(),
            'confidence_weights': self.confidence_weights.cpu().numpy(),
            'entropy': self.entropy,
            'consistency_score': self.consistency_score,
            'collapsed_action': self.collapsed_action,
            'collapsed_log_prob': self.collapsed_log_prob
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuperpositionState':
        """Create from dictionary"""
        return cls(
            decision_states=torch.from_numpy(data['decision_states']),
            confidence_weights=torch.from_numpy(data['confidence_weights']),
            entropy=data['entropy'],
            consistency_score=data['consistency_score'],
            collapsed_action=data['collapsed_action'],
            collapsed_log_prob=data['collapsed_log_prob']
        )


@dataclass
class SequentialTransition:
    """Single transition in sequential execution"""
    agent_name: str
    sequence_position: int
    phase: SequencePhase
    observation: np.ndarray
    superposition_state: SuperpositionState
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    info: Dict[str, Any]
    
    # Coordination information
    prev_agent_confidence: Optional[torch.Tensor] = None
    coordination_signal: Optional[torch.Tensor] = None
    sequence_reward: float = 0.0
    
    # Temporal dependencies
    temporal_discount: float = 1.0
    sequence_step: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_name': self.agent_name,
            'sequence_position': self.sequence_position,
            'phase': self.phase.value,
            'observation': self.observation,
            'superposition_state': self.superposition_state.to_dict(),
            'action': self.action,
            'reward': self.reward,
            'next_observation': self.next_observation,
            'done': self.done,
            'info': self.info,
            'prev_agent_confidence': self.prev_agent_confidence.cpu().numpy() if self.prev_agent_confidence is not None else None,
            'coordination_signal': self.coordination_signal.cpu().numpy() if self.coordination_signal is not None else None,
            'sequence_reward': self.sequence_reward,
            'temporal_discount': self.temporal_discount,
            'sequence_step': self.sequence_step
        }


@dataclass
class SequenceEpisode:
    """Complete episode with sequential transitions"""
    episode_id: int
    transitions: List[SequentialTransition]
    total_reward: float
    sequence_length: int
    coordination_quality: float
    superposition_metrics: Dict[str, float]
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def get_agent_transitions(self, agent_name: str) -> List[SequentialTransition]:
        """Get all transitions for specific agent"""
        return [t for t in self.transitions if t.agent_name == agent_name]
    
    def get_phase_transitions(self, phase: SequencePhase) -> List[SequentialTransition]:
        """Get all transitions for specific phase"""
        return [t for t in self.transitions if t.phase == phase]


class SequentialExperienceBuffer:
    """Experience buffer specialized for sequential agent coordination"""
    
    def __init__(self, capacity: int, agent_names: List[str], sequence_length: int,
                 prioritized: bool = True, coordination_weight: float = 0.3):
        """
        Initialize sequential experience buffer
        
        Args:
            capacity: Maximum number of episodes to store
            agent_names: Names of agents in execution order
            sequence_length: Expected sequence length
            prioritized: Whether to use prioritized sampling
            coordination_weight: Weight for coordination-based sampling
        """
        self.capacity = capacity
        self.agent_names = agent_names
        self.sequence_length = sequence_length
        self.prioritized = prioritized
        self.coordination_weight = coordination_weight
        
        # Storage
        self.episodes = deque(maxlen=capacity)
        self.agent_buffers = {agent: deque(maxlen=capacity // len(agent_names)) 
                             for agent in agent_names}
        
        # Prioritized sampling
        if prioritized:
            self.priorities = deque(maxlen=capacity)
            self.priority_alpha = 0.6
            self.priority_beta = 0.4
            self.priority_beta_increment = 0.001
            self.max_priority = 1.0
        
        # Coordination tracking
        self.coordination_history = deque(maxlen=1000)
        self.sequence_rewards = deque(maxlen=1000)
        
        # Statistics
        self.total_episodes = 0
        self.total_transitions = 0
        self.coordination_scores = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"Initialized SequentialExperienceBuffer with capacity {capacity}")
    
    def add_episode(self, episode: SequenceEpisode):
        """Add complete episode to buffer"""
        with self._lock:
            self.episodes.append(episode)
            
            # Add to agent-specific buffers
            for agent_name in self.agent_names:
                agent_transitions = episode.get_agent_transitions(agent_name)
                if agent_transitions:
                    self.agent_buffers[agent_name].extend(agent_transitions)
            
            # Update priorities
            if self.prioritized:
                priority = self._calculate_episode_priority(episode)
                self.priorities.append(priority)
                self.max_priority = max(self.max_priority, priority)
            
            # Update statistics
            self.total_episodes += 1
            self.total_transitions += len(episode.transitions)
            self.coordination_scores.append(episode.coordination_quality)
            self.sequence_rewards.append(episode.total_reward)
            
            # Update coordination history
            self.coordination_history.append({
                'episode_id': episode.episode_id,
                'coordination_quality': episode.coordination_quality,
                'sequence_reward': episode.total_reward,
                'superposition_metrics': episode.superposition_metrics
            })
    
    def sample_episode_batch(self, batch_size: int) -> List[SequenceEpisode]:
        """Sample batch of episodes"""
        with self._lock:
            if len(self.episodes) < batch_size:
                return list(self.episodes)
            
            if self.prioritized:
                return self._prioritized_sample_episodes(batch_size)
            else:
                return random.sample(self.episodes, batch_size)
    
    def sample_sequential_batch(self, batch_size: int, agent_name: str) -> List[SequentialTransition]:
        """Sample sequential transitions for specific agent"""
        with self._lock:
            agent_transitions = list(self.agent_buffers[agent_name])
            
            if len(agent_transitions) < batch_size:
                return agent_transitions
            
            # Sample with temporal awareness
            return self._temporal_aware_sample(agent_transitions, batch_size, agent_name)
    
    def sample_coordination_batch(self, batch_size: int) -> List[Tuple[SequentialTransition, SequentialTransition]]:
        """Sample transition pairs for coordination training"""
        with self._lock:
            coordination_pairs = []
            
            for episode in self.episodes:
                for i in range(len(episode.transitions) - 1):
                    current_transition = episode.transitions[i]
                    next_transition = episode.transitions[i + 1]
                    
                    # Check if transitions are from consecutive agents
                    current_pos = current_transition.sequence_position
                    next_pos = next_transition.sequence_position
                    
                    if next_pos == current_pos + 1:
                        coordination_pairs.append((current_transition, next_transition))
            
            if len(coordination_pairs) < batch_size:
                return coordination_pairs
            
            # Sample based on coordination quality
            return self._coordination_aware_sample(coordination_pairs, batch_size)
    
    def sample_superposition_batch(self, batch_size: int) -> List[SequentialTransition]:
        """Sample transitions with focus on superposition quality"""
        with self._lock:
            all_transitions = []
            for episode in self.episodes:
                all_transitions.extend(episode.transitions)
            
            if len(all_transitions) < batch_size:
                return all_transitions
            
            # Sample based on superposition metrics
            return self._superposition_aware_sample(all_transitions, batch_size)
    
    def _prioritized_sample_episodes(self, batch_size: int) -> List[SequenceEpisode]:
        """Sample episodes using prioritized experience replay"""
        priorities = np.array(self.priorities)
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.episodes), batch_size, p=probs)
        
        # Update beta
        self.priority_beta = min(1.0, self.priority_beta + self.priority_beta_increment)
        
        return [self.episodes[idx] for idx in indices]
    
    def _temporal_aware_sample(self, transitions: List[SequentialTransition], 
                              batch_size: int, agent_name: str) -> List[SequentialTransition]:
        """Sample transitions with temporal awareness"""
        # Get agent position in sequence
        agent_pos = self.agent_names.index(agent_name)
        
        # Weight recent transitions more heavily
        weights = []
        for i, transition in enumerate(transitions):
            # Recency weight
            recency_weight = 1.0 / (len(transitions) - i + 1)
            
            # Sequence position weight
            pos_weight = 1.0 if transition.sequence_position == agent_pos else 0.5
            
            # Reward weight
            reward_weight = max(0.1, (transition.reward + 1) / 2)
            
            # Combined weight
            weight = recency_weight * pos_weight * reward_weight
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights /= weights.sum()
        
        # Sample
        indices = np.random.choice(len(transitions), batch_size, p=weights, replace=False)
        return [transitions[idx] for idx in indices]
    
    def _coordination_aware_sample(self, coordination_pairs: List[Tuple[SequentialTransition, SequentialTransition]], 
                                  batch_size: int) -> List[Tuple[SequentialTransition, SequentialTransition]]:
        """Sample coordination pairs based on coordination quality"""
        # Calculate coordination scores for each pair
        coordination_scores = []
        for current, next_trans in coordination_pairs:
            if current.prev_agent_confidence is not None and next_trans.prev_agent_confidence is not None:
                # Calculate confidence alignment
                confidence_sim = torch.cosine_similarity(
                    current.superposition_state.confidence_weights,
                    next_trans.superposition_state.confidence_weights,
                    dim=0
                ).item()
                
                # Reward alignment
                reward_alignment = 1.0 if (current.reward > 0 and next_trans.reward > 0) else 0.0
                
                # Sequence reward
                sequence_score = (current.sequence_reward + next_trans.sequence_reward) / 2
                
                coordination_score = 0.4 * confidence_sim + 0.3 * reward_alignment + 0.3 * sequence_score
            else:
                coordination_score = 0.5  # Default score
            
            coordination_scores.append(coordination_score)
        
        # Sample based on coordination scores
        coordination_scores = np.array(coordination_scores)
        probs = coordination_scores / coordination_scores.sum()
        
        indices = np.random.choice(len(coordination_pairs), batch_size, p=probs, replace=False)
        return [coordination_pairs[idx] for idx in indices]
    
    def _superposition_aware_sample(self, transitions: List[SequentialTransition], 
                                   batch_size: int) -> List[SequentialTransition]:
        """Sample transitions based on superposition quality"""
        # Calculate superposition quality scores
        quality_scores = []
        for transition in transitions:
            sp_state = transition.superposition_state
            
            # Entropy score (higher is better up to a point)
            entropy_score = min(sp_state.entropy / 2.0, 1.0)
            
            # Consistency score
            consistency_score = sp_state.consistency_score
            
            # Confidence variance (lower is better)
            confidence_var = sp_state.confidence_weights.var().item()
            variance_score = 1.0 / (1.0 + confidence_var)
            
            # Combined quality score
            quality_score = 0.4 * entropy_score + 0.3 * consistency_score + 0.3 * variance_score
            quality_scores.append(quality_score)
        
        # Sample based on quality scores
        quality_scores = np.array(quality_scores)
        probs = quality_scores / quality_scores.sum()
        
        indices = np.random.choice(len(transitions), batch_size, p=probs, replace=False)
        return [transitions[idx] for idx in indices]
    
    def _calculate_episode_priority(self, episode: SequenceEpisode) -> float:
        """Calculate priority for episode"""
        # Base priority from total reward
        reward_priority = abs(episode.total_reward)
        
        # Coordination quality bonus
        coordination_bonus = episode.coordination_quality * 0.5
        
        # Superposition quality bonus
        superposition_bonus = np.mean(list(episode.superposition_metrics.values())) * 0.3
        
        # Sequence completion bonus
        completion_bonus = 0.2 if episode.sequence_length == self.sequence_length else 0.0
        
        return reward_priority + coordination_bonus + superposition_bonus + completion_bonus
    
    def get_coordination_statistics(self) -> Dict[str, float]:
        """Get coordination statistics"""
        with self._lock:
            if not self.coordination_history:
                return {}
            
            recent_history = list(self.coordination_history)[-100:]
            
            coordination_qualities = [h['coordination_quality'] for h in recent_history]
            sequence_rewards = [h['sequence_reward'] for h in recent_history]
            
            return {
                'mean_coordination_quality': np.mean(coordination_qualities),
                'std_coordination_quality': np.std(coordination_qualities),
                'mean_sequence_reward': np.mean(sequence_rewards),
                'std_sequence_reward': np.std(sequence_rewards),
                'episodes_stored': len(self.episodes),
                'total_transitions': self.total_transitions,
                'coordination_trend': self._calculate_coordination_trend()
            }
    
    def get_superposition_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get superposition statistics by agent"""
        with self._lock:
            agent_stats = {}
            
            for agent_name in self.agent_names:
                agent_transitions = list(self.agent_buffers[agent_name])
                
                if not agent_transitions:
                    agent_stats[agent_name] = {}
                    continue
                
                # Extract superposition metrics
                entropies = [t.superposition_state.entropy for t in agent_transitions]
                consistencies = [t.superposition_state.consistency_score for t in agent_transitions]
                confidence_vars = [t.superposition_state.confidence_weights.var().item() for t in agent_transitions]
                
                agent_stats[agent_name] = {
                    'mean_entropy': np.mean(entropies),
                    'std_entropy': np.std(entropies),
                    'mean_consistency': np.mean(consistencies),
                    'std_consistency': np.std(consistencies),
                    'mean_confidence_var': np.mean(confidence_vars),
                    'std_confidence_var': np.std(confidence_vars),
                    'transitions_stored': len(agent_transitions)
                }
            
            return agent_stats
    
    def _calculate_coordination_trend(self) -> float:
        """Calculate coordination quality trend"""
        if len(self.coordination_history) < 20:
            return 0.0
        
        recent_qualities = [h['coordination_quality'] for h in list(self.coordination_history)[-20:]]
        older_qualities = [h['coordination_quality'] for h in list(self.coordination_history)[-40:-20]]
        
        if not older_qualities:
            return 0.0
        
        recent_mean = np.mean(recent_qualities)
        older_mean = np.mean(older_qualities)
        
        return recent_mean - older_mean
    
    def update_priorities(self, episode_indices: List[int], priorities: List[float]):
        """Update priorities for specific episodes"""
        with self._lock:
            if not self.prioritized:
                return
            
            for idx, priority in zip(episode_indices, priorities):
                if 0 <= idx < len(self.priorities):
                    self.priorities[idx] = priority
                    self.max_priority = max(self.max_priority, priority)
    
    def clear(self):
        """Clear all stored experiences"""
        with self._lock:
            self.episodes.clear()
            for agent_buffer in self.agent_buffers.values():
                agent_buffer.clear()
            
            if self.prioritized:
                self.priorities.clear()
            
            self.coordination_history.clear()
            self.sequence_rewards.clear()
            
            self.total_episodes = 0
            self.total_transitions = 0
            self.coordination_scores.clear()
    
    def save(self, filepath: str):
        """Save buffer to disk"""
        with self._lock:
            save_data = {
                'episodes': [ep.__dict__ for ep in self.episodes],
                'agent_buffers': {
                    agent: [t.to_dict() for t in transitions] 
                    for agent, transitions in self.agent_buffers.items()
                },
                'priorities': list(self.priorities) if self.prioritized else None,
                'coordination_history': list(self.coordination_history),
                'sequence_rewards': list(self.sequence_rewards),
                'total_episodes': self.total_episodes,
                'total_transitions': self.total_transitions,
                'coordination_scores': self.coordination_scores,
                'config': {
                    'capacity': self.capacity,
                    'agent_names': self.agent_names,
                    'sequence_length': self.sequence_length,
                    'prioritized': self.prioritized,
                    'coordination_weight': self.coordination_weight
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Sequential experience buffer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load buffer from disk"""
        with self._lock:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore episodes
            self.episodes = deque(maxlen=self.capacity)
            for ep_data in save_data['episodes']:
                # Reconstruct episode (simplified)
                episode = SequenceEpisode(**ep_data)
                self.episodes.append(episode)
            
            # Restore agent buffers
            for agent, transitions_data in save_data['agent_buffers'].items():
                if agent in self.agent_buffers:
                    self.agent_buffers[agent] = deque(maxlen=self.capacity // len(self.agent_names))
                    for t_data in transitions_data:
                        # Reconstruct transition (simplified)
                        transition = SequentialTransition(**t_data)
                        self.agent_buffers[agent].append(transition)
            
            # Restore other data
            if self.prioritized and save_data['priorities']:
                self.priorities = deque(save_data['priorities'], maxlen=self.capacity)
            
            self.coordination_history = deque(save_data['coordination_history'], maxlen=1000)
            self.sequence_rewards = deque(save_data['sequence_rewards'], maxlen=1000)
            self.total_episodes = save_data['total_episodes']
            self.total_transitions = save_data['total_transitions']
            self.coordination_scores = save_data['coordination_scores']
            
            logger.info(f"Sequential experience buffer loaded from {filepath}")
    
    def __len__(self) -> int:
        """Get number of stored episodes"""
        return len(self.episodes)
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"SequentialExperienceBuffer(episodes={len(self.episodes)}/{self.capacity}, "
                f"transitions={self.total_transitions}, "
                f"coordination_quality={np.mean(self.coordination_scores[-10:]) if self.coordination_scores else 0:.3f})")


class SequentialBatchBuilder:
    """Builder for creating training batches from sequential experiences"""
    
    def __init__(self, buffer: SequentialExperienceBuffer):
        self.buffer = buffer
    
    def build_agent_batch(self, agent_name: str, batch_size: int) -> Dict[str, torch.Tensor]:
        """Build training batch for specific agent"""
        transitions = self.buffer.sample_sequential_batch(batch_size, agent_name)
        
        if not transitions:
            return {}
        
        # Convert to tensors
        observations = torch.stack([torch.from_numpy(t.observation) for t in transitions])
        actions = torch.tensor([t.action for t in transitions])
        rewards = torch.tensor([t.reward for t in transitions])
        
        # Superposition states
        decision_states = torch.stack([t.superposition_state.decision_states for t in transitions])
        confidence_weights = torch.stack([t.superposition_state.confidence_weights for t in transitions])
        entropies = torch.tensor([t.superposition_state.entropy for t in transitions])
        
        # Coordination features
        coordination_signals = []
        for t in transitions:
            if t.coordination_signal is not None:
                coordination_signals.append(t.coordination_signal)
            else:
                coordination_signals.append(torch.zeros(len(self.buffer.agent_names)))
        
        coordination_signals = torch.stack(coordination_signals)
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'decision_states': decision_states,
            'confidence_weights': confidence_weights,
            'entropies': entropies,
            'coordination_signals': coordination_signals,
            'sequence_positions': torch.tensor([t.sequence_position for t in transitions])
        }
    
    def build_coordination_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Build batch for coordination training"""
        coordination_pairs = self.buffer.sample_coordination_batch(batch_size)
        
        if not coordination_pairs:
            return {}
        
        # Split pairs
        current_transitions, next_transitions = zip(*coordination_pairs)
        
        # Build batch for current agents
        current_batch = self._build_transition_batch(current_transitions)
        
        # Build batch for next agents
        next_batch = self._build_transition_batch(next_transitions)
        
        return {
            'current_agent': current_batch,
            'next_agent': next_batch,
            'coordination_scores': torch.tensor([
                (t1.sequence_reward + t2.sequence_reward) / 2 
                for t1, t2 in coordination_pairs
            ])
        }
    
    def _build_transition_batch(self, transitions: List[SequentialTransition]) -> Dict[str, torch.Tensor]:
        """Build batch from list of transitions"""
        return {
            'observations': torch.stack([torch.from_numpy(t.observation) for t in transitions]),
            'actions': torch.tensor([t.action for t in transitions]),
            'rewards': torch.tensor([t.reward for t in transitions]),
            'decision_states': torch.stack([t.superposition_state.decision_states for t in transitions]),
            'confidence_weights': torch.stack([t.superposition_state.confidence_weights for t in transitions]),
            'sequence_positions': torch.tensor([t.sequence_position for t in transitions])
        }