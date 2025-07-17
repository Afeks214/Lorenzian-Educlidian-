"""
Prioritized Experience Replay Buffer for MAPPO Training
Implements TD-error based prioritization with importance sampling
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple, deque
import random
from dataclasses import dataclass
import pickle


# Experience tuple for storing transitions
Experience = namedtuple('Experience', [
    'states',           # Dict[str, torch.Tensor] - agent states
    'actions',          # Dict[str, int] - agent actions
    'rewards',          # Dict[str, float] - agent rewards
    'next_states',      # Dict[str, torch.Tensor] - next agent states
    'dones',           # Dict[str, bool] - episode termination flags
    'log_probs',       # Dict[str, float] - action log probabilities
    'values',          # Dict[str, float] - state values from critic
    'td_errors'        # Dict[str, float] - TD errors for prioritization
])


@dataclass
class BatchedExperience:
    """Batched experience data for training."""
    states: Dict[str, torch.Tensor]
    actions: Dict[str, torch.Tensor]
    rewards: Dict[str, torch.Tensor]
    next_states: Dict[str, torch.Tensor]
    dones: Dict[str, torch.Tensor]
    log_probs: Dict[str, torch.Tensor]
    values: Dict[str, torch.Tensor]
    advantages: Dict[str, torch.Tensor]
    returns: Dict[str, torch.Tensor]
    weights: torch.Tensor  # Importance sampling weights


class SumTree:
    """
    Binary tree data structure for efficient prioritized sampling.
    Stores priorities in leaf nodes and sums in internal nodes.
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
        dataIdx = idx - self.capacity + 1
        
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedExperienceBuffer:
    """
    Prioritized Experience Replay Buffer with TD-error based prioritization.
    
    Implements proportional prioritization where probability of sampling
    transition i is proportional to priority^alpha.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6
    ):
        """
        Initialize prioritized experience buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling exponent
            beta_frames: Number of frames to anneal beta to 1.0
            epsilon: Small constant to ensure non-zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        
        # Sum tree for efficient sampling
        self.tree = SumTree(capacity)
        
        # Min tree for normalization
        self.min_tree = SumTree(capacity)
        self.min_tree.tree.fill(float('inf'))
        
        # Maximum priority for new experiences
        self.max_priority = 1.0
        
    def beta(self) -> float:
        """Get current beta value (annealed from beta_start to 1.0)."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        
    def add(
        self,
        states: Dict[str, torch.Tensor],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_states: Dict[str, torch.Tensor],
        dones: Dict[str, bool],
        log_probs: Dict[str, float],
        values: Optional[Dict[str, float]] = None,
        td_errors: Optional[Dict[str, float]] = None
    ):
        """
        Add new experience to buffer with maximum priority.
        
        Args:
            states: Agent state observations
            actions: Actions taken by agents
            rewards: Rewards received
            next_states: Next state observations
            dones: Episode termination flags
            log_probs: Log probabilities of actions
            values: State value estimates
            td_errors: TD errors for prioritization
        """
        # Calculate priority based on TD error if provided
        if td_errors is not None:
            # Use mean absolute TD error across agents
            priority = np.mean([abs(td) for td in td_errors.values()]) + self.epsilon
            self.max_priority = max(self.max_priority, priority)
        else:
            # Use maximum priority for new experiences
            priority = self.max_priority
            
        # Create experience tuple
        exp = Experience(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            log_probs=log_probs,
            values=values or {},
            td_errors=td_errors or {}
        )
        
        # Add to sum tree with priority^alpha
        self.tree.add(priority ** self.alpha, exp)
        
        # Update min tree
        self.min_tree.add(priority ** self.alpha, exp)
        
    def sample(self, batch_size: int) -> Tuple[BatchedExperience, np.ndarray]:
        """
        Sample batch of experiences with importance sampling weights.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (batched_experience, indices_for_update)
        """
        self.frame += 1
        
        # Initialize batch storage
        batch = []
        indices = []
        priorities = []
        
        # Calculate sampling range
        segment = self.tree.total() / batch_size
        
        # Sample experiences
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            idx, priority, exp = self.tree.get(s)
            
            if exp is not None:
                batch.append(exp)
                indices.append(idx)
                priorities.append(priority)
                
        # Calculate importance sampling weights
        probs = np.array(priorities) / self.tree.total()
        
        # Normalize by max weight for stability
        min_prob = self.min_tree.tree[self.min_tree.tree != float('inf')].min() / self.tree.total()
        max_weight = (min_prob * self.tree.n_entries) ** (-self.beta())
        
        weights = (probs * self.tree.n_entries) ** (-self.beta())
        weights = weights / max_weight
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Convert batch to tensors
        batched_exp = self._batch_experiences(batch)
        batched_exp.weights = weights
        
        return batched_exp, np.array(indices)
        
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on new TD errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: New TD errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.min_tree.update(idx, priority)
            self.max_priority = max(self.max_priority, abs(td_error) + self.epsilon)
            
    def _batch_experiences(self, experiences: List[Experience]) -> BatchedExperience:
        """Convert list of experiences to batched tensors."""
        # Get agent names from first experience
        agent_names = list(experiences[0].states.keys())
        
        # Initialize dictionaries for batched data
        states = {name: [] for name in agent_names}
        actions = {name: [] for name in agent_names}
        rewards = {name: [] for name in agent_names}
        next_states = {name: [] for name in agent_names}
        dones = {name: [] for name in agent_names}
        log_probs = {name: [] for name in agent_names}
        values = {name: [] for name in agent_names}
        
        # Collect data from experiences
        for exp in experiences:
            for name in agent_names:
                states[name].append(exp.states[name])
                actions[name].append(exp.actions[name])
                rewards[name].append(exp.rewards[name])
                next_states[name].append(exp.next_states[name])
                dones[name].append(float(exp.dones[name]))
                log_probs[name].append(exp.log_probs[name])
                if name in exp.values:
                    values[name].append(exp.values[name])
                else:
                    values[name].append(0.0)
                    
        # Convert to tensors
        for name in agent_names:
            states[name] = torch.stack(states[name])
            actions[name] = torch.tensor(actions[name], dtype=torch.long)
            rewards[name] = torch.tensor(rewards[name], dtype=torch.float32)
            next_states[name] = torch.stack(next_states[name])
            dones[name] = torch.tensor(dones[name], dtype=torch.float32)
            log_probs[name] = torch.tensor(log_probs[name], dtype=torch.float32)
            values[name] = torch.tensor(values[name], dtype=torch.float32)
            
        # Placeholder for advantages and returns (computed during training)
        advantages = {name: torch.zeros_like(rewards[name]) for name in agent_names}
        returns = {name: torch.zeros_like(rewards[name]) for name in agent_names}
        
        return BatchedExperience(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            log_probs=log_probs,
            values=values,
            advantages=advantages,
            returns=returns,
            weights=torch.ones(len(experiences))
        )
        
    def __len__(self) -> int:
        """Get current buffer size."""
        return self.tree.n_entries
        
    def is_ready(self, min_size: int = 1000) -> bool:
        """Check if buffer has enough experiences for training."""
        return len(self) >= min_size
        
    def save(self, filepath: str):
        """Save buffer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tree': self.tree,
                'min_tree': self.min_tree,
                'max_priority': self.max_priority,
                'frame': self.frame
            }, f)
            
    def load(self, filepath: str):
        """Load buffer from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.tree = data['tree']
            self.min_tree = data['min_tree']
            self.max_priority = data['max_priority']
            self.frame = data['frame']


class UniformExperienceBuffer:
    """
    Standard uniform experience replay buffer (for comparison).
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(
        self,
        states: Dict[str, torch.Tensor],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        next_states: Dict[str, torch.Tensor],
        dones: Dict[str, bool],
        log_probs: Dict[str, float],
        values: Optional[Dict[str, float]] = None,
        td_errors: Optional[Dict[str, float]] = None
    ):
        """Add experience to buffer."""
        exp = Experience(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            log_probs=log_probs,
            values=values or {},
            td_errors=td_errors or {}
        )
        self.buffer.append(exp)
        
    def sample(self, batch_size: int) -> Tuple[BatchedExperience, None]:
        """Sample batch uniformly."""
        batch = random.sample(self.buffer, batch_size)
        batched_exp = self._batch_experiences(batch)
        batched_exp.weights = torch.ones(batch_size)
        return batched_exp, None
        
    def _batch_experiences(self, experiences: List[Experience]) -> BatchedExperience:
        """Convert list of experiences to batched tensors."""
        # Implementation identical to PrioritizedExperienceBuffer._batch_experiences
        return PrioritizedExperienceBuffer._batch_experiences(self, experiences)
        
    def __len__(self) -> int:
        return len(self.buffer)
        
    def is_ready(self, min_size: int = 1000) -> bool:
        return len(self) >= min_size