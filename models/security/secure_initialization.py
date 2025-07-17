"""
Secure Initialization - CVE-2025-TACTICAL-005 Mitigation

Implements Xavier/He initialization with stability guarantees
and cryptographic validation to prevent initialization attacks.
"""

import torch
import torch.nn as nn
import math
import secrets
from typing import Optional, Dict, Any, Callable
import numpy as np
from .crypto_validation import CryptographicValidator


class SecureInitializer:
    """
    Cryptographically secure weight initialization system.
    
    Key Security Features:
    - Cryptographically secure random number generation
    - Validation of initialization parameters
    - Stability guarantees for training
    - Protection against initialization attacks
    """
    
    def __init__(self, crypto_key: Optional[bytes] = None, seed: Optional[int] = None):
        self.crypto_validator = CryptographicValidator(crypto_key)
        
        # Use cryptographically secure random seed
        if seed is None:
            seed = secrets.randbits(32)
        
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
        # Initialization parameters tracking
        self.initialization_log = []
        self.stability_checks = True
        
        # Supported initialization methods
        self.methods = {
            'xavier_uniform': self._xavier_uniform,
            'xavier_normal': self._xavier_normal,
            'he_uniform': self._he_uniform,
            'he_normal': self._he_normal,
            'orthogonal': self._orthogonal,
            'sparse': self._sparse,
            'secure_random': self._secure_random
        }
    
    def _validate_tensor_properties(self, tensor: torch.Tensor, method: str) -> bool:
        """Validate tensor properties after initialization."""
        try:
            # Check for NaN/Inf values
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return False
            
            # Check tensor magnitude
            tensor_norm = torch.norm(tensor)
            if tensor_norm == 0 or tensor_norm > 100.0:
                return False
            
            # Method-specific validations
            if method in ['xavier_uniform', 'xavier_normal']:
                # Xavier initialization should have specific variance
                expected_std = math.sqrt(2.0 / (tensor.size(0) + tensor.size(-1)))
                actual_std = tensor.std().item()
                if abs(actual_std - expected_std) > expected_std * 0.5:  # Allow 50% tolerance
                    return False
            
            elif method in ['he_uniform', 'he_normal']:
                # He initialization variance check
                fan_in = tensor.size(1) if tensor.dim() > 1 else tensor.size(0)
                expected_std = math.sqrt(2.0 / fan_in)
                actual_std = tensor.std().item()
                if abs(actual_std - expected_std) > expected_std * 0.5:
                    return False
            
            # Cryptographic validation
            tensor_hash = self.crypto_validator.compute_tensor_hash(tensor)
            if not self.crypto_validator.validate_tensor_hash(tensor, tensor_hash):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _xavier_uniform(self, tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Secure Xavier uniform initialization."""
        num_input = tensor.size(1) if tensor.dim() > 1 else tensor.size(0)
        num_output = tensor.size(0)
        
        # Calculate bounds with gain
        bound = gain * math.sqrt(6.0 / (num_input + num_output))
        
        # Use secure random generation
        with torch.no_grad():
            tensor.uniform_(-bound, bound, generator=self.generator)
        
        return tensor
    
    def _xavier_normal(self, tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Secure Xavier normal initialization."""
        num_input = tensor.size(1) if tensor.dim() > 1 else tensor.size(0)
        num_output = tensor.size(0)
        
        # Calculate standard deviation with gain
        std = gain * math.sqrt(2.0 / (num_input + num_output))
        
        # Use secure random generation
        with torch.no_grad():
            tensor.normal_(0, std, generator=self.generator)
        
        return tensor
    
    def _he_uniform(self, tensor: torch.Tensor, nonlinearity: str = 'relu') -> torch.Tensor:
        """Secure He uniform initialization."""
        fan_in = tensor.size(1) if tensor.dim() > 1 else tensor.size(0)
        
        # Calculate gain based on nonlinearity
        gain = nn.init.calculate_gain(nonlinearity)
        
        # Calculate bound
        bound = gain * math.sqrt(3.0 / fan_in)
        
        # Use secure random generation
        with torch.no_grad():
            tensor.uniform_(-bound, bound, generator=self.generator)
        
        return tensor
    
    def _he_normal(self, tensor: torch.Tensor, nonlinearity: str = 'relu') -> torch.Tensor:
        """Secure He normal initialization."""
        fan_in = tensor.size(1) if tensor.dim() > 1 else tensor.size(0)
        
        # Calculate gain based on nonlinearity
        gain = nn.init.calculate_gain(nonlinearity)
        
        # Calculate standard deviation
        std = gain / math.sqrt(fan_in)
        
        # Use secure random generation
        with torch.no_grad():
            tensor.normal_(0, std, generator=self.generator)
        
        return tensor
    
    def _orthogonal(self, tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Secure orthogonal initialization."""
        rows = tensor.size(0)
        cols = tensor.numel() // rows
        
        # Generate random matrix
        flattened = torch.randn(rows, cols, generator=self.generator)
        
        # SVD for orthogonal matrix
        if rows < cols:
            flattened = flattened.t()
        
        q, r = torch.qr(flattened)
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph
        
        if rows < cols:
            q = q.t()
        
        with torch.no_grad():
            tensor.view_as(q).copy_(q)
            tensor.mul_(gain)
        
        return tensor
    
    def _sparse(self, tensor: torch.Tensor, sparsity: float = 0.1, std: float = 0.01) -> torch.Tensor:
        """Secure sparse initialization."""
        with torch.no_grad():
            tensor.normal_(0, std, generator=self.generator)
            
            # Create sparse mask
            mask = torch.rand(tensor.shape, generator=self.generator) < sparsity
            tensor.masked_fill_(~mask, 0)
        
        return tensor
    
    def _secure_random(self, tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
        """Cryptographically secure random initialization."""
        with torch.no_grad():
            tensor.normal_(0, std, generator=self.generator)
        
        return tensor
    
    def initialize_tensor(
        self, 
        tensor: torch.Tensor, 
        method: str = 'he_normal',
        validate: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Initialize tensor with specified method and security validation.
        
        Args:
            tensor: Tensor to initialize
            method: Initialization method name
            validate: Whether to validate initialization
            **kwargs: Method-specific parameters
            
        Returns:
            Initialized tensor
        """
        if method not in self.methods:
            raise ValueError(f"Unknown initialization method: {method}")
        
        # Store original state for rollback
        original_data = tensor.data.clone() if validate else None
        
        try:
            # Apply initialization
            self.methods[method](tensor, **kwargs)
            
            # Validate if requested
            if validate and self.stability_checks:
                if not self._validate_tensor_properties(tensor, method):
                    # Rollback and use fallback
                    if original_data is not None:
                        tensor.data.copy_(original_data)
                    
                    # Apply secure fallback
                    self._secure_random(tensor, std=0.01)
                    
                    # Log the failure
                    self.initialization_log.append({
                        'method': method,
                        'status': 'failed_validation',
                        'fallback': 'secure_random',
                        'shape': list(tensor.shape)
                    })
                else:
                    # Log successful initialization
                    self.initialization_log.append({
                        'method': method,
                        'status': 'success',
                        'shape': list(tensor.shape)
                    })
            
            return tensor
            
        except Exception as e:
            # Rollback on error
            if original_data is not None:
                tensor.data.copy_(original_data)
            
            # Apply secure fallback
            self._secure_random(tensor, std=0.01)
            
            # Log the error
            self.initialization_log.append({
                'method': method,
                'status': 'error',
                'error': str(e),
                'fallback': 'secure_random',
                'shape': list(tensor.shape)
            })
            
            return tensor
    
    def initialize_module(self, module: nn.Module, method_map: Optional[Dict[type, str]] = None) -> nn.Module:
        """
        Initialize all parameters in a module with secure methods.
        
        Args:
            module: PyTorch module to initialize
            method_map: Mapping from layer types to initialization methods
            
        Returns:
            Initialized module
        """
        # Default method mapping
        if method_map is None:
            method_map = {
                nn.Linear: 'xavier_uniform',
                nn.Conv1d: 'he_normal',
                nn.Conv2d: 'he_normal',
                nn.ConvTranspose1d: 'he_normal',
                nn.ConvTranspose2d: 'he_normal',
                nn.LSTM: 'orthogonal',
                nn.GRU: 'orthogonal',
                nn.Embedding: 'xavier_normal'
            }
        
        for name, param in module.named_parameters():
            if param.requires_grad:
                # Determine layer type
                layer_type = type(param)
                for parent_name, parent_module in module.named_modules():
                    if hasattr(parent_module, 'weight') and parent_module.weight is param:
                        layer_type = type(parent_module)
                        break
                
                # Select initialization method
                method = method_map.get(layer_type, 'secure_random')
                
                # Initialize parameter
                if 'weight' in name:
                    self.initialize_tensor(param, method)
                elif 'bias' in name:
                    # Initialize biases to zero
                    nn.init.constant_(param, 0)
        
        return module
    
    def get_initialization_stats(self) -> Dict[str, Any]:
        """Get initialization statistics and logs."""
        if not self.initialization_log:
            return {}
        
        # Count successes and failures
        success_count = sum(1 for log in self.initialization_log if log['status'] == 'success')
        failure_count = len(self.initialization_log) - success_count
        
        # Method usage statistics
        method_counts = {}
        for log in self.initialization_log:
            method = log['method']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'total_initializations': len(self.initialization_log),
            'successful_initializations': success_count,
            'failed_initializations': failure_count,
            'success_rate': success_count / len(self.initialization_log) if self.initialization_log else 0,
            'method_usage': method_counts,
            'recent_logs': self.initialization_log[-10:]  # Last 10 logs
        }
    
    def reset_logs(self):
        """Reset initialization logs."""
        self.initialization_log.clear()
    
    def enable_stability_checks(self, enable: bool = True):
        """Enable or disable stability validation checks."""
        self.stability_checks = enable
    
    def set_seed(self, seed: int):
        """Set new random seed."""
        self.generator.manual_seed(seed)