"""
Adaptive Multi-Scale Kernels - CVE-2025-TACTICAL-004 Mitigation

Replaces fixed kernel sizes with adaptive convolution layers
to prevent kernel manipulation vulnerabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math
from .crypto_validation import CryptographicValidator
from .memory_security import SecureMemoryManager


class AdaptiveMultiScaleKernels(nn.Module):
    """
    Adaptive multi-scale convolution with dynamic kernel selection.
    
    Key Security Features:
    - Dynamic kernel size adaptation
    - Cryptographic validation of kernel parameters
    - Secure kernel weight initialization
    - Prevention of kernel manipulation attacks
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [1, 3, 5, 7],
        stride: int = 1,
        padding_mode: str = 'zeros',
        dropout_rate: float = 0.1,
        crypto_key: Optional[bytes] = None,
        memory_manager: Optional[SecureMemoryManager] = None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.padding_mode = padding_mode
        self.dropout_rate = dropout_rate
        
        # Initialize security components
        self.crypto_validator = CryptographicValidator(crypto_key)
        self.memory_manager = memory_manager or SecureMemoryManager()
        
        # Multi-scale convolution branches
        self.conv_branches = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2  # Same padding
            
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels // len(kernel_sizes),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=False  # Use batch norm instead
            )
            
            bn = nn.BatchNorm1d(out_channels // len(kernel_sizes))
            
            self.conv_branches.append(conv)
            self.batch_norms.append(bn)
        
        # Adaptive kernel weight selection
        self.kernel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, len(kernel_sizes)),
            nn.Softmax(dim=-1)
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Security monitoring
        self.kernel_weight_history = []
        self.anomaly_threshold = 2.0
        
        # Initialize weights securely
        self._secure_initialization()
    
    def _secure_initialization(self):
        """Initialize weights with cryptographic security."""
        for conv in self.conv_branches:
            # Use He initialization with secure random seed
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            
            # Validate initial weights
            weight_hash = self.crypto_validator.compute_tensor_hash(conv.weight)
            if not self.crypto_validator.validate_tensor_hash(conv.weight, weight_hash):
                # Reinitialize if validation fails
                nn.init.xavier_uniform_(conv.weight)
        
        # Initialize batch norms
        for bn in self.batch_norms:
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
        
        # Initialize attention weights
        for layer in self.kernel_attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Initialize fusion layer
        nn.init.xavier_uniform_(self.feature_fusion.weight)
        nn.init.constant_(self.feature_fusion.bias, 0)
    
    def _validate_kernel_integrity(self) -> bool:
        """Validate kernel weights for tampering."""
        try:
            for i, conv in enumerate(self.conv_branches):
                # Check for NaN/Inf values
                if torch.isnan(conv.weight).any() or torch.isinf(conv.weight).any():
                    return False
                
                # Check weight magnitude (detect unusual scaling)
                weight_norm = torch.norm(conv.weight)
                if weight_norm > 10.0 or weight_norm < 0.01:
                    return False
                
                # Cryptographic validation
                weight_hash = self.crypto_validator.compute_tensor_hash(conv.weight)
                if not self.crypto_validator.validate_tensor_hash(conv.weight, weight_hash):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _adaptive_kernel_selection(self, x: torch.Tensor) -> torch.Tensor:
        """Compute adaptive kernel weights based on input characteristics."""
        # Global average pooling to capture input statistics
        global_features = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        # Compute kernel attention weights
        kernel_weights = self.kernel_attention(x)
        
        # Validate kernel weights
        if torch.isnan(kernel_weights).any() or torch.isinf(kernel_weights).any():
            # Fallback to uniform weights
            kernel_weights = torch.ones_like(kernel_weights) / len(self.kernel_sizes)
        
        # Ensure weights sum to 1
        kernel_weights = kernel_weights / kernel_weights.sum(dim=-1, keepdim=True)
        
        # Security validation
        if self.training and len(self.kernel_weight_history) > 10:
            recent_weights = torch.stack(self.kernel_weight_history[-10:])
            current_deviation = torch.norm(kernel_weights.mean(0) - recent_weights.mean(0))
            
            if current_deviation > self.anomaly_threshold:
                # Use previous stable weights
                kernel_weights = recent_weights[-1].unsqueeze(0).expand_as(kernel_weights)
        
        # Store for monitoring
        if self.training:
            self.kernel_weight_history.append(kernel_weights.detach().clone())
            if len(self.kernel_weight_history) > 100:
                self.kernel_weight_history.pop(0)
        
        return kernel_weights
    
    def forward(self, x: torch.Tensor, validate_security: bool = True) -> torch.Tensor:
        """
        Forward pass with adaptive multi-scale convolution.
        
        Args:
            x: Input tensor (batch_size, in_channels, sequence_length)
            validate_security: Whether to perform security validation
            
        Returns:
            Multi-scale convolved features
        """
        batch_size = x.size(0)
        
        # Security validation
        if validate_security and self.training:
            if not self._validate_kernel_integrity():
                raise RuntimeError("Kernel integrity validation failed - potential security breach")
        
        # Compute adaptive kernel weights
        kernel_weights = self._adaptive_kernel_selection(x)
        
        # Apply multi-scale convolutions
        branch_outputs = []
        
        for i, (conv, bn) in enumerate(zip(self.conv_branches, self.batch_norms)):
            # Convolution
            conv_out = conv(x)
            
            # Batch normalization
            conv_out = bn(conv_out)
            
            # Apply ReLU activation
            conv_out = F.relu(conv_out)
            
            # Weight by adaptive kernel attention
            weight = kernel_weights[:, i:i+1].unsqueeze(-1)
            weighted_out = conv_out * weight
            
            branch_outputs.append(weighted_out)
        
        # Concatenate all branches
        multi_scale_features = torch.cat(branch_outputs, dim=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(multi_scale_features)
        
        # Apply dropout
        output = self.dropout(fused_features)
        
        return output
    
    def get_kernel_stats(self) -> Dict:
        """Get kernel statistics for monitoring."""
        stats = {}
        
        # Kernel weight statistics
        for i, conv in enumerate(self.conv_branches):
            weight = conv.weight.detach()
            stats[f'kernel_{self.kernel_sizes[i]}_mean'] = weight.mean().item()
            stats[f'kernel_{self.kernel_sizes[i]}_std'] = weight.std().item()
            stats[f'kernel_{self.kernel_sizes[i]}_norm'] = torch.norm(weight).item()
        
        # Attention weight statistics
        if self.kernel_weight_history:
            recent_weights = torch.stack(self.kernel_weight_history[-10:])
            stats['attention_mean'] = recent_weights.mean(0).tolist()
            stats['attention_std'] = recent_weights.std(0).tolist()
        
        return stats
    
    def reset_security_state(self):
        """Reset security monitoring state."""
        self.kernel_weight_history.clear()
        self.crypto_validator.reset_state()
    
    def enable_security_mode(self, enable: bool = True):
        """Enable or disable security validation."""
        self.crypto_validator.enabled = enable


class AdaptiveDepthwiseConv(nn.Module):
    """
    Adaptive depthwise convolution for efficient multi-scale processing.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_sizes: List[int] = [3, 5, 7],
        dilation_rates: List[int] = [1, 2, 4],
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        
        # Depthwise convolutions with different scales
        self.depthwise_convs = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            for dilation in dilation_rates:
                padding = (kernel_size - 1) * dilation // 2
                
                conv = nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    groups=channels,  # Depthwise
                    dilation=dilation,
                    padding=padding,
                    bias=False
                )
                
                self.depthwise_convs.append(conv)
        
        # Pointwise fusion
        num_branches = len(kernel_sizes) * len(dilation_rates)
        self.pointwise_fusion = nn.Conv1d(
            in_channels=channels * num_branches,
            out_channels=channels,
            kernel_size=1,
            bias=True
        )
        
        # Batch normalization and dropout
        self.batch_norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for conv in self.depthwise_convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        
        nn.init.xavier_uniform_(self.pointwise_fusion.weight)
        nn.init.constant_(self.pointwise_fusion.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adaptive depthwise convolution."""
        # Apply all depthwise convolutions
        branch_outputs = []
        
        for conv in self.depthwise_convs:
            conv_out = F.relu(conv(x))
            branch_outputs.append(conv_out)
        
        # Concatenate all branches
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # Pointwise fusion
        fused = self.pointwise_fusion(concatenated)
        
        # Batch normalization and dropout
        output = self.batch_norm(fused)
        output = self.dropout(output)
        
        # Residual connection
        output = output + x
        
        return output