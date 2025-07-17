"""
Random Baseline Agent

Random agent for baseline comparison. Generates random valid action distributions.
Useful for establishing a performance floor and testing environment robustness.
"""

import numpy as np
from typing import Dict, Any, Optional


class RandomAgent:
    """
    Random agent generating valid probability distributions
    
    Provides a performance baseline and helps verify that learned policies
    are actually adding value beyond random chance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize random agent
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        
        # Random seed for reproducibility
        self.random_seed = self.config.get('random_seed', None)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        # Action space configuration
        self.action_dim = self.config.get('action_dim', 3)
        self.distribution_type = self.config.get('distribution', 'dirichlet')
        
        # Dirichlet concentration parameter
        self.alpha = self.config.get('dirichlet_alpha', 1.0)
        
        # Statistics tracking
        self.action_history = []
        self.reset()
        
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Generate random action
        
        Args:
            observation: Observation dict (ignored by random agent)
            
        Returns:
            Random valid probability distribution
        """
        if self.distribution_type == 'dirichlet':
            # Dirichlet distribution for valid probabilities
            action = np.random.dirichlet(np.ones(self.action_dim) * self.alpha)
            
        elif self.distribution_type == 'uniform':
            # Uniform random then normalize
            action = np.random.rand(self.action_dim)
            action = action / action.sum()
            
        elif self.distribution_type == 'softmax':
            # Random logits through softmax
            logits = np.random.randn(self.action_dim)
            exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
            action = exp_logits / exp_logits.sum()
            
        else:
            # Default to uniform normalized
            action = np.ones(self.action_dim) / self.action_dim
            
        # Record action
        self.action_history.append(action)
        
        return action
        
    def get_action_batch(self, observations: np.ndarray) -> np.ndarray:
        """
        Generate batch of random actions
        
        Args:
            observations: Batch of observations (ignored)
            
        Returns:
            Batch of random actions
        """
        batch_size = len(observations) if hasattr(observations, '__len__') else 1
        
        actions = []
        for _ in range(batch_size):
            actions.append(self.get_action({}))
            
        return np.array(actions)
        
    def reset(self):
        """Reset agent state"""
        self.action_history = []
        self.total_actions = 0
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        if not self.action_history:
            return {
                'total_actions': 0,
                'mean_action': np.zeros(self.action_dim),
                'std_action': np.zeros(self.action_dim),
                'entropy': 0.0
            }
            
        actions = np.array(self.action_history)
        
        # Calculate statistics
        mean_action = actions.mean(axis=0)
        std_action = actions.std(axis=0)
        
        # Calculate average entropy
        entropies = []
        for action in actions:
            # Avoid log(0)
            safe_action = np.clip(action, 1e-10, 1.0)
            entropy = -np.sum(safe_action * np.log(safe_action))
            entropies.append(entropy)
            
        avg_entropy = np.mean(entropies)
        
        return {
            'total_actions': len(self.action_history),
            'mean_action': mean_action.tolist(),
            'std_action': std_action.tolist(),
            'entropy': avg_entropy,
            'distribution_type': self.distribution_type
        }


class BiasedRandomAgent(RandomAgent):
    """
    Biased random agent with configurable preferences
    
    Useful for testing against agents with slight biases rather than
    purely uniform random behavior.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Bias configuration
        self.bias_type = self.config.get('bias_type', 'none')
        self.bias_strength = self.config.get('bias_strength', 0.2)
        
        # Specific biases
        self.action_bias = self.config.get('action_bias', [1.0, 1.0, 1.0])
        self.action_bias = np.array(self.action_bias) / np.sum(self.action_bias)
        
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate biased random action"""
        # Get base random action
        base_action = super().get_action(observation)
        
        if self.bias_type == 'none':
            return base_action
            
        elif self.bias_type == 'bullish':
            # Bias toward bullish actions
            bias_vector = np.array([0.1, 0.2, 0.7])
            
        elif self.bias_type == 'bearish':
            # Bias toward bearish actions
            bias_vector = np.array([0.7, 0.2, 0.1])
            
        elif self.bias_type == 'neutral':
            # Bias toward neutral/hold
            bias_vector = np.array([0.1, 0.8, 0.1])
            
        elif self.bias_type == 'custom':
            # Use provided bias
            bias_vector = self.action_bias
            
        else:
            # No bias
            return base_action
            
        # Blend base action with bias
        biased_action = (
            (1 - self.bias_strength) * base_action + 
            self.bias_strength * bias_vector
        )
        
        # Ensure valid distribution
        biased_action = biased_action / biased_action.sum()
        
        return biased_action


class ContextualRandomAgent(RandomAgent):
    """
    Context-aware random agent
    
    Generates random actions but with variance that depends on
    market context (e.g., higher randomness in volatile markets).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Context sensitivity
        self.volatility_sensitivity = self.config.get('volatility_sensitivity', 1.0)
        self.base_alpha = self.alpha
        
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate context-sensitive random action"""
        # Extract context if available
        shared_context = observation.get('shared_context', None)
        
        if shared_context is not None and len(shared_context) > 2:
            # Assume volatility is in position 2 (log scale)
            volatility = np.exp(shared_context[2])
            
            # Adjust randomness based on volatility
            # Higher volatility -> lower alpha -> more random
            self.alpha = self.base_alpha / (1.0 + self.volatility_sensitivity * volatility)
            self.alpha = max(0.1, self.alpha)  # Minimum alpha
        else:
            self.alpha = self.base_alpha
            
        # Generate action with adjusted randomness
        action = np.random.dirichlet(np.ones(self.action_dim) * self.alpha)
        
        # Record action
        self.action_history.append(action)
        
        return action