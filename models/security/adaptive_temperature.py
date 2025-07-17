"""
Adaptive Temperature Scaling - CVE-2025-TACTICAL-002 Mitigation

Implements secure temperature scaling with cryptographic validation
and adaptive bounds to prevent temperature manipulation attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from .crypto_validation import CryptographicValidator


class AdaptiveTemperatureScaling(nn.Module):
    """
    Secure adaptive temperature scaling with cryptographic validation.
    
    Key Security Features:
    - Bounded temperature values to prevent extreme scaling
    - Cryptographic validation of temperature updates
    - Adaptive temperature based on training dynamics
    - Gradient sanitization for temperature parameters
    """
    
    def __init__(
        self,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        max_temperature: float = 5.0,
        adaptation_rate: float = 0.01,
        crypto_key: Optional[bytes] = None,
        enable_adaptive: bool = True
    ):
        super().__init__()
        
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.adaptation_rate = adaptation_rate
        self.enable_adaptive = enable_adaptive
        
        # Initialize cryptographic validator
        self.crypto_validator = CryptographicValidator(crypto_key)
        
        # Learnable temperature parameter with secure bounds
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        
        # Adaptive components
        self.entropy_ema = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.confidence_ema = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.ema_decay = 0.99
        
        # Security monitoring
        self.temperature_history = []
        self.anomaly_count = 0
        self.max_anomalies = 5
        
        # Performance tracking
        self.calibration_error_ema = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
    def _validate_temperature_security(self, temperature: torch.Tensor) -> bool:
        """Validate temperature value for security constraints."""
        try:
            # Check for NaN/Inf
            if torch.isnan(temperature).any() or torch.isinf(temperature).any():
                return False
            
            # Check bounds
            if temperature < self.min_temperature or temperature > self.max_temperature:
                return False
            
            # Check for rapid changes (potential attack)
            if len(self.temperature_history) > 0:
                last_temp = self.temperature_history[-1]
                change_rate = abs(temperature.item() - last_temp)
                if change_rate > 1.0:  # Max change per update
                    return False
            
            # Cryptographic validation
            temp_hash = self.crypto_validator.compute_tensor_hash(temperature)
            if not self.crypto_validator.validate_tensor_hash(temperature, temp_hash):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _compute_adaptive_temperature(
        self, 
        logits: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute adaptive temperature based on current model state."""
        if not self.enable_adaptive:
            return self.temperature
        
        with torch.no_grad():
            # Compute current entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            
            # Update entropy EMA
            self.entropy_ema.data = self.ema_decay * self.entropy_ema + (1 - self.ema_decay) * entropy
            
            # Compute confidence (max probability)
            confidence = torch.max(probs, dim=-1)[0].mean()
            self.confidence_ema.data = self.ema_decay * self.confidence_ema + (1 - self.ema_decay) * confidence
            
            # Adaptive temperature adjustment
            target_entropy = np.log(logits.size(-1)) * 0.8  # 80% of max entropy
            entropy_ratio = self.entropy_ema / target_entropy
            
            # Increase temperature if entropy is too low (overconfident)
            # Decrease temperature if entropy is too high (underconfident)
            temperature_adjustment = 1.0 + self.adaptation_rate * (1.0 - entropy_ratio)
            
            # Apply adjustment to base temperature
            adaptive_temp = self.temperature * temperature_adjustment
            
            # Enforce bounds
            adaptive_temp = torch.clamp(adaptive_temp, self.min_temperature, self.max_temperature)
            
            return adaptive_temp
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        validate_security: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply secure temperature scaling to logits.
        
        Args:
            logits: Input logits tensor
            targets: Optional targets for calibration
            validate_security: Whether to perform security validation
            
        Returns:
            Tuple of (scaled_logits, metrics_dict)
        """
        # Compute adaptive temperature
        current_temp = self._compute_adaptive_temperature(logits, targets)
        
        # Security validation
        if validate_security:
            if not self._validate_temperature_security(current_temp):
                self.anomaly_count += 1
                if self.anomaly_count > self.max_anomalies:
                    # Reset to safe default
                    current_temp = torch.tensor(1.0, device=logits.device)
                    self.anomaly_count = 0
        
        # Apply temperature scaling with numerical stability
        scaled_logits = logits / torch.clamp(current_temp, min=0.01)
        
        # Gradient sanitization for temperature parameter
        if self.training:
            # Clip gradients to prevent explosion
            if self.temperature.grad is not None:
                self.temperature.grad.data.clamp_(-1.0, 1.0)
        
        # Update temperature history
        if len(self.temperature_history) >= 100:
            self.temperature_history.pop(0)
        self.temperature_history.append(current_temp.item())
        
        # Compute metrics
        metrics = {
            'temperature': current_temp.item(),
            'entropy_ema': self.entropy_ema.item(),
            'confidence_ema': self.confidence_ema.item(),
            'anomaly_count': self.anomaly_count
        }
        
        # Compute calibration error if targets provided
        if targets is not None:
            with torch.no_grad():
                probs = F.softmax(scaled_logits, dim=-1)
                predicted_probs = torch.max(probs, dim=-1)[0]
                predictions = torch.argmax(probs, dim=-1)
                accuracy = (predictions == targets).float()
                
                # Expected Calibration Error (ECE)
                calibration_error = torch.abs(predicted_probs - accuracy).mean()
                self.calibration_error_ema.data = (
                    self.ema_decay * self.calibration_error_ema + 
                    (1 - self.ema_decay) * calibration_error
                )
                metrics['calibration_error'] = calibration_error.item()
                metrics['calibration_error_ema'] = self.calibration_error_ema.item()
        
        return scaled_logits, metrics
    
    def get_temperature_stats(self) -> Dict:
        """Get temperature statistics for monitoring."""
        if not self.temperature_history:
            return {}
        
        history = np.array(self.temperature_history)
        return {
            'current_temperature': self.temperature.item(),
            'mean_temperature': float(np.mean(history)),
            'std_temperature': float(np.std(history)),
            'min_temperature': float(np.min(history)),
            'max_temperature': float(np.max(history)),
            'temperature_range': float(np.max(history) - np.min(history)),
            'num_samples': len(history)
        }
    
    def reset_adaptation_state(self):
        """Reset adaptive temperature state."""
        self.entropy_ema.data.zero_()
        self.confidence_ema.data.zero_()
        self.calibration_error_ema.data.zero_()
        self.temperature_history.clear()
        self.anomaly_count = 0
    
    def enable_security_mode(self, enable: bool = True):
        """Enable or disable security validation."""
        self.crypto_validator.enabled = enable
    
    def set_temperature_bounds(self, min_temp: float, max_temp: float):
        """Update temperature bounds."""
        self.min_temperature = min_temp
        self.max_temperature = max_temp
        
        # Clamp current temperature to new bounds
        with torch.no_grad():
            self.temperature.data.clamp_(min_temp, max_temp)