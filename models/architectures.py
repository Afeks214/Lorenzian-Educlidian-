"""
Specialized Neural Network Architectures for Strategic MARL System
Implements MLMIActor, NWRQKActor, MMDActor, and CentralizedCritic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .base import BaseStrategicActor
from .components import (
    MultiHeadAttention, ConvBlock, TemporalBlock, 
    ResidualBlock, PositionalEncoding
)


class MLMIActor(BaseStrategicActor):
    """
    MLMI Strategic Actor - Specialized for momentum analysis.
    Uses smaller Conv1D kernels to capture short-term price accelerations.
    
    Input: 4D features [mlmi_value, mlmi_signal, momentum_20, momentum_50]
    Output: 3D action probabilities [bullish, neutral, bearish]
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: List[int] = [256, 128, 64],
        action_dim: int = 3,
        dropout_rate: float = 0.1,
        temperature_init: float = 1.0,
        conv_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 5],  # Smaller kernels for short-term patterns
        sequence_length: int = 48
    ):
        """Initialize MLMI Actor with momentum-specific architecture."""
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.sequence_length = sequence_length
        
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            dropout_rate=dropout_rate,
            temperature_init=temperature_init
        )
        
    def _build_network(self):
        """Build MLMI-specific network architecture."""
        # Feature extraction with small kernels
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_dim
        
        for out_channels, kernel_size in zip(self.conv_channels[:2], self.kernel_sizes):
            self.conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    activation='gelu',
                    norm='batch'
                )
            )
            in_channels = out_channels
        
        # Projection layer to match dimensions
        self.feature_proj = nn.Linear(in_channels, self.hidden_dims[0])
        
        # Temporal modeling - LSTM for momentum patterns
        self.temporal = TemporalBlock(
            input_size=self.hidden_dims[0],
            hidden_size=self.hidden_dims[0],
            num_layers=2,
            rnn_type='lstm',
            dropout=self.dropout_rate,
            bidirectional=False
        )
        
        # Attention mechanism for focusing on key momentum shifts
        self.attention = MultiHeadAttention(
            embed_dim=self.hidden_dims[0],
            num_heads=8,
            dropout=self.dropout_rate
        )
        
        # Position encoding for temporal awareness
        self.pos_encoding = PositionalEncoding(
            embed_dim=self.hidden_dims[0],
            dropout=self.dropout_rate
        )
        
        # Output layers with residual connections
        self.output_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.output_layers.append(
                ResidualBlock(
                    input_dim=self.hidden_dims[i],
                    hidden_dim=self.hidden_dims[i+1],
                    dropout=self.dropout_rate,
                    activation='gelu'
                )
            )
        
        # Final output head
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[-1] // 2, self.action_dim)
        )
        
        # Optional value head for actor-critic
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract momentum-specific features using Conv1D."""
        # Ensure correct shape for conv1d: (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(-1).expand(-1, -1, self.sequence_length)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.expand(-1, -1, self.sequence_length)
        
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape for temporal modeling: (batch, length, channels)
        x = x.transpose(1, 2)
        
        return x
    
    def _temporal_modeling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LSTM for momentum pattern recognition."""
        # Project features to correct dimension
        x = self.feature_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # LSTM processing
        temporal_output, _ = self.temporal(x)
        
        return temporal_output


class NWRQKActor(BaseStrategicActor):
    """
    NWRQK Strategic Actor - Specialized for support/resistance level analysis.
    Uses larger Conv1D kernels to identify broader patterns around levels.
    
    Input: 6D features [nwrqk_pred, lvn_strength, fvg_size, price_distance, volume_imbalance, level_touches]
    Output: 3D action probabilities [bullish, neutral, bearish]
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = [256, 128, 64],
        action_dim: int = 3,
        dropout_rate: float = 0.1,
        temperature_init: float = 1.0,
        conv_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [7, 9],  # Larger kernels for broader patterns
        sequence_length: int = 48
    ):
        """Initialize NWRQK Actor with level-specific architecture."""
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.sequence_length = sequence_length
        
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            dropout_rate=dropout_rate,
            temperature_init=temperature_init
        )
        
    def _build_network(self):
        """Build NWRQK-specific network architecture."""
        # Feature extraction with larger kernels
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_dim
        
        for out_channels, kernel_size in zip(self.conv_channels[:2], self.kernel_sizes):
            self.conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    activation='swish',  # Smoother activation for levels
                    norm='layer'  # Layer norm for stability
                )
            )
            in_channels = out_channels
        
        # Additional conv layer for level pattern extraction
        self.conv_layers.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=self.conv_channels[-1],
                kernel_size=5,
                activation='swish',
                norm='layer'
            )
        )
        
        # Projection layer to match dimensions
        self.feature_proj = nn.Linear(self.conv_channels[-1], self.hidden_dims[0])
        
        # Temporal modeling - LSTM with more capacity for complex patterns
        self.temporal = TemporalBlock(
            input_size=self.hidden_dims[0],
            hidden_size=self.hidden_dims[0],
            num_layers=3,  # Deeper for complex level interactions
            rnn_type='lstm',
            dropout=self.dropout_rate,
            bidirectional=True  # Bidirectional for past/future level awareness
        )
        
        # Multi-head attention for level relationships
        self.attention = MultiHeadAttention(
            embed_dim=self.hidden_dims[0],
            num_heads=8,  # 8 heads for 256 dim (256/8=32 per head)
            dropout=self.dropout_rate
        )
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(
            embed_dim=self.hidden_dims[0],
            dropout=self.dropout_rate
        )
        
        # Deep output network
        self.output_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.output_layers.append(
                ResidualBlock(
                    input_dim=self.hidden_dims[i],
                    hidden_dim=self.hidden_dims[i+1],
                    dropout=self.dropout_rate,
                    activation='swish'
                )
            )
        
        # Final output head
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[-1], self.action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract level-specific features using Conv1D."""
        # Ensure correct shape
        if x.dim() == 2:
            x = x.unsqueeze(-1).expand(-1, -1, self.sequence_length)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.expand(-1, -1, self.sequence_length)
        
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape for temporal modeling
        x = x.transpose(1, 2)
        
        return x
    
    def _temporal_modeling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bidirectional LSTM for level pattern recognition."""
        # Project features to correct dimension
        x = self.feature_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # LSTM processing
        temporal_output, _ = self.temporal(x)
        
        return temporal_output


class MMDActor(BaseStrategicActor):
    """
    MMD (Regime Detection) Strategic Actor - Specialized for market regime analysis.
    Uses GRU for efficiency with lower-dimensional regime features.
    
    Input: 3D features [mmd_score, volatility_30, volume_profile_skew]
    Output: 3D action probabilities [bullish, neutral, bearish]
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dims: List[int] = [128, 64, 32],  # Smaller network for 3D input
        action_dim: int = 3,
        dropout_rate: float = 0.1,
        temperature_init: float = 1.0,
        conv_channels: List[int] = [16, 32, 64],
        kernel_sizes: List[int] = [5, 7],
        sequence_length: int = 48
    ):
        """Initialize MMD Actor with regime-specific architecture."""
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.sequence_length = sequence_length
        
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            dropout_rate=dropout_rate,
            temperature_init=temperature_init
        )
        
    def _build_network(self):
        """Build MMD-specific network architecture."""
        # Feature extraction
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_dim
        
        for out_channels, kernel_size in zip(self.conv_channels[:2], self.kernel_sizes):
            self.conv_layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    activation='relu',  # Simple activation for regime
                    norm='instance'  # Instance norm for regime stability
                )
            )
            in_channels = out_channels
        
        # Projection layer to match dimensions
        self.feature_proj = nn.Linear(in_channels, self.hidden_dims[0])
        
        # Temporal modeling - GRU for efficiency
        self.temporal = TemporalBlock(
            input_size=self.hidden_dims[0],
            hidden_size=self.hidden_dims[0],
            num_layers=2,
            rnn_type='gru',  # GRU for efficiency
            dropout=self.dropout_rate,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            embed_dim=self.hidden_dims[0],
            num_heads=4,  # Fewer heads for simpler features
            dropout=self.dropout_rate
        )
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(
            embed_dim=self.hidden_dims[0],
            dropout=self.dropout_rate
        )
        
        # Output layers
        self.output_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.output_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                    nn.LayerNorm(self.hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate)
                )
            )
        
        # Final output head with volatility adjustment
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[-1], self.action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dims[-1] // 2, 1)
        )
        
        # Volatility adjustment layer
        self.volatility_adjust = nn.Parameter(torch.ones(1))
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract regime-specific features."""
        # Ensure correct shape
        if x.dim() == 2:
            x = x.unsqueeze(-1).expand(-1, -1, self.sequence_length)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.expand(-1, -1, self.sequence_length)
        
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Reshape for temporal modeling
        x = x.transpose(1, 2)
        
        return x
    
    def _temporal_modeling(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GRU for regime pattern recognition."""
        # Project features to correct dimension
        x = self.feature_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # GRU processing
        temporal_output, _ = self.temporal(x)
        
        return temporal_output
    
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with volatility adjustment."""
        output = super().forward(state, deterministic)
        
        # Apply volatility adjustment to logits
        if hasattr(self, 'volatility_adjust'):
            # Extract volatility from state (assuming it's the second feature)
            if state.dim() >= 2:
                # Take mean volatility across sequence dimension
                volatility = state[:, 1].mean(dim=-1) if state.dim() == 3 else state[:, 1]
            else:
                volatility = torch.ones(1, device=state.device)
            volatility_factor = torch.clamp(1.0 / (volatility + 1e-6), 0.5, 2.0)
            
            # Adjust logits based on volatility
            output['logits'] = output['logits'] * volatility_factor.unsqueeze(-1) * self.volatility_adjust
            
            # Recompute action probabilities with adjusted logits
            scaled_logits = output['logits'] / self.temperature.clamp(min=0.1)
            output['action_probs'] = F.softmax(scaled_logits, dim=-1)
        
        return output


class CentralizedCritic(nn.Module):
    """
    Centralized Critic for MAPPO.
    Sees combined observations from all strategic agents and outputs state value.
    Enhanced with layer normalization for stability.
    """
    
    def __init__(
        self,
        state_dim: int,  # Total dimension of all agent observations combined
        n_agents: int = 3,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize centralized critic.
        
        Args:
            state_dim: Total state dimension (sum of all agent state dims)
            n_agents: Number of agents
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        # Build network
        layers = []
        input_dim = state_dim
        
        # Deep MLP with residual connections
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # First layer
                layers.append(nn.Linear(input_dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            else:
                # Residual blocks for deeper layers
                layers.append(
                    ResidualBlock(
                        input_dim=hidden_dims[i-1],
                        hidden_dim=hidden_dim,
                        dropout=dropout_rate,
                        activation='relu'
                    )
                )
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention pooling over agent contributions
        self.agent_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Value heads
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # Auxiliary heads for better learning
        self.agent_value_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], 1) for _ in range(n_agents)
        ])
        
    def forward(
        self, 
        states: torch.Tensor,
        agent_states: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through centralized critic.
        
        Args:
            states: Combined state tensor (batch, state_dim)
            agent_states: Optional dict of individual agent states
            
        Returns:
            Dictionary containing:
                - value: Central state value estimate
                - agent_values: Individual agent value estimates (if agent_states provided)
        """
        # Extract features
        features = self.feature_extractor(states)
        
        # Compute central value
        central_value = self.value_head(features).squeeze(-1)
        
        output = {'value': central_value}
        
        # Compute agent-specific values if individual states provided
        if agent_states is not None:
            agent_values = {}
            for i, (agent_name, agent_state) in enumerate(agent_states.items()):
                if i < len(self.agent_value_heads):
                    # For now, we'll use the agent value heads directly on the last hidden layer
                    # In practice, you might want separate feature extractors per agent
                    # or pad the agent states to the full state dimension
                    agent_values[agent_name] = self.agent_value_heads[i](features).squeeze(-1)
            output['agent_values'] = agent_values
        
        return output
    
    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        """Get value estimate for given states."""
        return self.forward(states)['value']