"""
Base Abstract Classes for Strategic MARL Actors
Defines the shared structure and interface for all specialized actors
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class BaseStrategicActor(nn.Module, ABC):
    """
    Abstract base class for all strategic actors in the MARL system.
    
    Defines the common structure: Conv1D → Temporal → Attention → Output
    Each specialized actor must implement the specific architecture details.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        action_dim: int = 3,  # [bullish, neutral, bearish]
        dropout_rate: float = 0.1,
        temperature_init: float = 1.0
    ):
        """
        Initialize base strategic actor.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            action_dim: Number of possible actions (default 3)
            dropout_rate: Dropout probability for regularization
            temperature_init: Initial temperature for softmax
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.dropout_rate = dropout_rate
        
        # Learnable temperature parameter for exploration control
        self.temperature = nn.Parameter(
            torch.tensor(temperature_init, dtype=torch.float32)
        )
        
        # Build the network architecture
        self._build_network()
        
    @abstractmethod
    def _build_network(self):
        """Build the specific network architecture for this actor."""
        pass
    
    @abstractmethod
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using Conv1D layers.
        
        Args:
            x: Input tensor of shape (batch, input_dim, sequence_length)
            
        Returns:
            Extracted features tensor
        """
        pass
    
    @abstractmethod
    def _temporal_modeling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal modeling (LSTM/GRU).
        
        Args:
            x: Feature tensor from Conv1D layers
            
        Returns:
            Temporal features tensor
        """
        pass
    
    def forward(
        self, 
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the actor network.
        
        Args:
            state: Input state tensor
            deterministic: If True, return argmax action instead of sampling
            
        Returns:
            Dictionary containing:
                - action: Selected action
                - action_probs: Probability distribution over actions
                - log_prob: Log probability of selected action
                - entropy: Entropy of action distribution
                - value: State value (if actor has value head)
        """
        # Ensure correct input shape
        if state.dim() == 2:
            # Add sequence dimension if not present
            state = state.unsqueeze(-1)
        
        # Feature extraction
        features = self._extract_features(state)
        
        # Temporal modeling
        temporal_features = self._temporal_modeling(features)
        
        # Attention mechanism
        attended_features = self._apply_attention(temporal_features)
        
        # Process through output layers if they exist
        features = attended_features
        if hasattr(self, 'output_layers'):
            for layer in self.output_layers:
                features = layer(features)
        
        # Output head
        logits = self.output_head(features)
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature.clamp(min=0.1)
        
        # Get action probabilities
        action_probs = F.softmax(scaled_logits, dim=-1)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        
        # Sample or take argmax
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        output = {
            'action': action,
            'action_probs': action_probs,
            'log_prob': log_prob,
            'entropy': entropy,
            'logits': logits,
            'attention_weights': self._get_attention_weights()
        }
        
        # Add value if actor has critic head
        if hasattr(self, 'value_head'):
            output['value'] = self.value_head(features).squeeze(-1)
        
        return output
    
    def _apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mechanism to temporal features.
        
        Args:
            x: Temporal features tensor
            
        Returns:
            Attended features tensor
        """
        if hasattr(self, 'attention'):
            # Store attention weights for visualization
            attended, self._attention_weights = self.attention(x, x, x)
            # Take the last timestep output instead of averaging
            if attended.dim() == 3:
                return attended[:, -1, :]  # Take last timestep
            else:
                return attended
        return x[:, -1, :] if x.dim() == 3 else x
    
    def _get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the most recent attention weights for visualization."""
        return getattr(self, '_attention_weights', None)
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from the policy.
        
        Args:
            state: Input state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        output = self.forward(state, deterministic)
        value = output.get('value', torch.zeros_like(output['action'], dtype=torch.float32))
        return output['action'], output['log_prob'], value
    
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate given actions under current policy.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            
        Returns:
            Dictionary with log_probs, entropy, and values
        """
        output = self.forward(states)
        
        dist = Categorical(output['action_probs'])
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return {
            'log_probs': log_probs,
            'entropy': entropy,
            'values': output.get('value', torch.zeros_like(actions, dtype=torch.float32)),
            'action_probs': output['action_probs']
        }
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for given state.
        
        Args:
            state: Input state
            
        Returns:
            Value estimate
        """
        if not hasattr(self, 'value_head'):
            raise ValueError("This actor does not have a value head")
        
        output = self.forward(state, deterministic=True)
        return output['value']
    
    def reset_hidden_state(self):
        """Reset any hidden states in recurrent layers."""
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'gru'):
            self.gru.reset_parameters()