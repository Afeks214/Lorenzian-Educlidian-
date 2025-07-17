"""
Shared Neural Network Components for Strategic MARL System
Includes attention mechanisms, normalization layers, and other reusable components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for focusing on important features.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim)
            value: Value tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (attended values, attention weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections and reshape for multi-head
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # Final projection
        output = self.out_proj(attended)
        
        # Average attention weights across heads for visualization
        avg_attn_weights = attn_weights.mean(dim=1)
        
        return output, avg_attn_weights


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence data.
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            -(math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # If dimensions match, use residual connection
        self.use_residual = (input_dim == hidden_dim)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Projection layer if dimensions don't match
        if not self.use_residual:
            self.proj = nn.Linear(input_dim, hidden_dim)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        # Project input if dimensions don't match
        if self.use_residual:
            residual = x
        else:
            residual = self.proj(x)
        
        # First layer
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.dropout(x)
        
        # Residual connection
        x = x + residual
        x = self.norm2(x)
        
        return x


class ConvBlock(nn.Module):
    """
    1D Convolutional block for feature extraction.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = 'same',
        groups: int = 1,
        activation: str = 'relu',
        norm: str = 'batch'
    ):
        super().__init__()
        
        # Calculate padding for 'same' convolution
        if padding == 'same':
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, groups=groups
        )
        
        # Normalization
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(out_channels)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(out_channels)
        else:
            self.norm = nn.Identity()
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        x = self.conv(x)
        
        # Handle layer norm which expects different shape
        if isinstance(self.norm, nn.LayerNorm):
            # LayerNorm expects shape [batch, length, channels]
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm(x)
            
        x = self.activation(x)
        return x


class TemporalBlock(nn.Module):
    """
    Temporal modeling block with LSTM/GRU options.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        rnn_type: str = 'lstm',
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.rnn_type = rnn_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Create RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output projection if bidirectional
        if bidirectional:
            self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.output_proj = nn.Identity()
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal block.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            hidden: Optional hidden state
            
        Returns:
            Tuple of (output, hidden_state)
        """
        output, hidden = self.rnn(x, hidden)
        output = self.output_proj(output)
        return output, hidden


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration (alternative to entropy regularization).
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Factorized noise parameters
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
    def reset_noise(self):
        """Sample new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)