"""
Tactical Experience Buffer with Prioritized Experience Replay
Implements TD-error based prioritization for high-frequency tactical learning
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple, deque
import random
from dataclasses import dataclass
import pickle
import logging

logger = logging.getLogger(__name__)


# Experience tuple for tactical system
TacticalExperience = namedtuple('TacticalExperience', [
    'state',            # np.ndarray - (60, 7) matrix
    'actions',          # Dict[str, int] - {agent_name: action}
    'rewards',          # Dict[str, float] - {agent_name: reward}
    'next_state',       # np.ndarray - (60, 7) matrix
    'done',             # bool - episode termination
    'log_probs',        # Dict[str, float] - {agent_name: log_prob}
    'value',            # float - critic value estimate
    'td_error',         # float - temporal difference error
    'priority'          # float - priority for sampling
])


@dataclass
class TacticalBatch:
    """Batched experience data for tactical training."""
    states: np.ndarray                    # (batch_size, 60, 7)
    actions: Dict[str, np.ndarray]        # {agent_name: (batch_size,)}
    rewards: Dict[str, np.ndarray]        # {agent_name: (batch_size,)}
    next_states: np.ndarray               # (batch_size, 60, 7)
    dones: np.ndarray                     # (batch_size,)
    log_probs: Dict[str, np.ndarray]      # {agent_name: (batch_size,)}
    values: np.ndarray                    # (batch_size,)
    td_errors: np.ndarray                 # (batch_size,)
    priorities: np.ndarray                # (batch_size,)
    weights: np.ndarray                   # (batch_size,) - importance sampling
    indices: np.ndarray                   # (batch_size,) - buffer indices


class SumTree:
    """
    Efficient sum tree data structure for prioritized sampling.
    Provides O(log n) operations for priority-based sampling.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0
        
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]
        
    def add(self, priority: float, data: Any):
        """Add new experience with given priority."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, idx: int, priority: float):
        """Update priority of experience at given index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        
    def get(self, s: float) -> Tuple[int, float, Any]:
        """Get experience for given cumulative sum."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        if data_idx < 0 or data_idx >= self.n_entries:
            return 0, 0, None
            
        return idx, self.tree[idx], self.data[data_idx]


class TacticalExperienceBuffer:
    """
    Tactical Experience Buffer with Prioritized Experience Replay.
    
    Features:
    - TD-error based prioritization for efficient learning
    - Importance sampling with beta annealing
    - Circular buffer for memory efficiency
    - Batch sampling optimized for tactical training
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        max_priority: float = 1.0
    ):
        """
        Initialize Tactical Experience Buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Prioritization exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            beta_increment: Beta increment per sample
            epsilon: Small value to prevent zero priorities
            max_priority: Maximum priority value
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = max_priority
        
        # Initialize sum tree
        self.tree = SumTree(capacity)
        
        # Agent names for tactical system
        self.agent_names = ['fvg', 'momentum', 'entry']
        
        # Statistics
        self.total_samples = 0
        self.priority_stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': float('inf'),
            'max': 0.0
        }
        
    def add_experience(
        self,
        state: np.ndarray,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_state: np.ndarray,
        done: bool,
        log_probs: Dict[str, float],
        value: float,
        td_error: Optional[float] = None
    ):
        """
        Add experience to buffer.
        
        Args:
            state: Current state (60, 7) matrix
            actions: Agent actions {agent_name: action}
            rewards: Agent rewards {agent_name: reward}
            next_state: Next state (60, 7) matrix
            done: Episode termination flag
            log_probs: Action log probabilities {agent_name: log_prob}
            value: Critic value estimate
            td_error: Temporal difference error (computed if None)
        """
        # Validate inputs
        if not self._validate_experience(state, actions, rewards, next_state, log_probs):
            logger.warning("Invalid experience, skipping")
            return
        
        # Compute TD error if not provided
        if td_error is None:
            td_error = self._compute_td_error(rewards, value, done)
        
        # Compute priority
        priority = self._compute_priority(td_error)
        
        # Create experience
        experience = TacticalExperience(
            state=state.copy(),
            actions=actions.copy(),
            rewards=rewards.copy(),
            next_state=next_state.copy(),
            done=done,
            log_probs=log_probs.copy(),
            value=value,
            td_error=td_error,
            priority=priority
        )
        
        # Add to tree
        self.tree.add(priority, experience)
        
        # Update statistics
        self._update_priority_stats(priority)
        
    def sample(self, batch_size: int) -> TacticalBatch:
        """
        Sample batch of experiences using prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            TacticalBatch with sampled experiences
        """
        if self.tree.n_entries == 0:
            raise ValueError("Buffer is empty, cannot sample")
        
        # Sample from tree
        experiences = []
        indices = []
        priorities = []
        
        priority_segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            # Sample from segment
            a = priority_segment * i
            b = priority_segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            
            if experience is not None:
                experiences.append(experience)
                indices.append(idx)
                priorities.append(priority)
        
        # Handle case where we couldn't sample enough experiences
        while len(experiences) < batch_size:
            # Fill with random experiences
            idx = random.randint(0, self.tree.n_entries - 1)
            _, priority, experience = self.tree.get(random.uniform(0, self.tree.total()))
            if experience is not None:
                experiences.append(experience)
                indices.append(idx)
                priorities.append(priority)
        
        # Compute importance sampling weights
        weights = self._compute_is_weights(priorities)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.total_samples += batch_size
        
        # Convert to batch format
        return self._create_batch(experiences, indices, priorities, weights)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on new TD errors.
        
        Args:
            indices: Buffer indices to update
            td_errors: New TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            priority = self._compute_priority(td_error)
            self.tree.update(idx, priority)
            self._update_priority_stats(priority)
    
    def _validate_experience(
        self,
        state: np.ndarray,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_state: np.ndarray,
        log_probs: Dict[str, float]
    ) -> bool:
        """Validate experience format."""
        # Check state shapes
        if state.shape != (60, 7) or next_state.shape != (60, 7):
            logger.error(f"Invalid state shape: {state.shape}, {next_state.shape}")
            return False
        
        # Check agent data
        for agent_name in self.agent_names:
            if agent_name not in actions:
                logger.error(f"Missing action for agent: {agent_name}")
                return False
            if agent_name not in rewards:
                logger.error(f"Missing reward for agent: {agent_name}")
                return False
            if agent_name not in log_probs:
                logger.error(f"Missing log_prob for agent: {agent_name}")
                return False
        
        # Check value ranges
        for agent_name in self.agent_names:
            if actions[agent_name] not in [0, 1, 2]:
                logger.error(f"Invalid action for {agent_name}: {actions[agent_name]}")
                return False
            if not np.isfinite(rewards[agent_name]):
                logger.error(f"Invalid reward for {agent_name}: {rewards[agent_name]}")
                return False
            if not np.isfinite(log_probs[agent_name]):
                logger.error(f"Invalid log_prob for {agent_name}: {log_probs[agent_name]}")
                return False
        
        return True
    
    def _compute_td_error(self, rewards: Dict[str, float], value: float, done: bool) -> float:
        """Compute temporal difference error."""
        # Simple TD error approximation
        # In practice, this would be computed by the trainer
        avg_reward = sum(rewards.values()) / len(rewards)
        return abs(avg_reward - value)
    
    def _compute_priority(self, td_error: float) -> float:
        """Compute priority from TD error."""
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        return min(priority, self.max_priority)
    
    def _compute_is_weights(self, priorities: List[float]) -> np.ndarray:
        """Compute importance sampling weights."""
        if self.tree.n_entries == 0:
            return np.ones(len(priorities))
        
        # Compute sampling probabilities
        priorities = np.array(priorities)
        probs = priorities / self.tree.total()
        
        # Compute weights
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        
        # Normalize by maximum weight
        weights = weights / weights.max()
        
        return weights
    
    def _create_batch(
        self,
        experiences: List[TacticalExperience],
        indices: List[int],
        priorities: List[float],
        weights: np.ndarray
    ) -> TacticalBatch:
        """Create batch from sampled experiences."""
        batch_size = len(experiences)
        
        # Initialize arrays
        states = np.zeros((batch_size, 60, 7), dtype=np.float32)
        actions = {name: np.zeros(batch_size, dtype=np.int32) for name in self.agent_names}
        rewards = {name: np.zeros(batch_size, dtype=np.float32) for name in self.agent_names}
        next_states = np.zeros((batch_size, 60, 7), dtype=np.float32)
        dones = np.zeros(batch_size, dtype=bool)
        log_probs = {name: np.zeros(batch_size, dtype=np.float32) for name in self.agent_names}
        values = np.zeros(batch_size, dtype=np.float32)
        td_errors = np.zeros(batch_size, dtype=np.float32)
        
        # Fill arrays
        for i, exp in enumerate(experiences):
            states[i] = exp.state
            for name in self.agent_names:
                actions[name][i] = exp.actions[name]
                rewards[name][i] = exp.rewards[name]
                log_probs[name][i] = exp.log_probs[name]
            next_states[i] = exp.next_state
            dones[i] = exp.done
            values[i] = exp.value
            td_errors[i] = exp.td_error
        
        return TacticalBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            log_probs=log_probs,
            values=values,
            td_errors=td_errors,
            priorities=np.array(priorities, dtype=np.float32),
            weights=weights.astype(np.float32),
            indices=np.array(indices, dtype=np.int32)
        )
    
    def _update_priority_stats(self, priority: float):
        """Update priority statistics."""
        self.priority_stats['min'] = min(self.priority_stats['min'], priority)
        self.priority_stats['max'] = max(self.priority_stats['max'], priority)
        
        # Update running mean and std
        alpha = 0.01
        self.priority_stats['mean'] = (1 - alpha) * self.priority_stats['mean'] + alpha * priority
        
        # Simple variance update
        diff = priority - self.priority_stats['mean']
        self.priority_stats['std'] = (1 - alpha) * self.priority_stats['std'] + alpha * abs(diff)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            'capacity': self.capacity,
            'size': self.tree.n_entries,
            'total_samples': self.total_samples,
            'beta': self.beta,
            'alpha': self.alpha,
            'priority_stats': self.priority_stats.copy(),
            'tree_total': self.tree.total(),
            'utilization': self.tree.n_entries / self.capacity
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get buffer state for checkpointing."""
        return {
            'capacity': self.capacity,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'epsilon': self.epsilon,
            'max_priority': self.max_priority,
            'total_samples': self.total_samples,
            'priority_stats': self.priority_stats,
            'tree_n_entries': self.tree.n_entries,
            'tree_write': self.tree.write
        }
    
    def save_buffer(self, filepath: str):
        """Save buffer to file."""
        state = self.get_state()
        state['tree_data'] = self.tree.data[:self.tree.n_entries]
        state['tree_tree'] = self.tree.tree
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Buffer saved to {filepath}")
    
    def load_buffer(self, filepath: str):
        """Load buffer from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore configuration
        self.capacity = state['capacity']
        self.alpha = state['alpha']
        self.beta = state['beta']
        self.beta_increment = state['beta_increment']
        self.epsilon = state['epsilon']
        self.max_priority = state['max_priority']
        self.total_samples = state['total_samples']
        self.priority_stats = state['priority_stats']
        
        # Restore tree
        self.tree = SumTree(self.capacity)
        self.tree.n_entries = state['tree_n_entries']
        self.tree.write = state['tree_write']
        self.tree.data[:self.tree.n_entries] = state['tree_data']
        self.tree.tree = state['tree_tree']
        
        logger.info(f"Buffer loaded from {filepath}")
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.tree = SumTree(self.capacity)
        self.total_samples = 0
        self.priority_stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': float('inf'),
            'max': 0.0
        }
        logger.info("Buffer cleared")
    
    def __len__(self) -> int:
        """Get number of experiences in buffer."""
        return self.tree.n_entries
    
    def __bool__(self) -> bool:
        """Check if buffer has any experiences."""
        return self.tree.n_entries > 0