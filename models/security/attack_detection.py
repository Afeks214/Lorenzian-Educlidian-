"""
Real-Time Attack Detection System

Implements real-time detection of adversarial attacks and anomalies
in neural network operations for high-frequency trading systems.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum


class AttackType(Enum):
    """Types of detected attacks."""
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing" 
    WEIGHT_MANIPULATION = "weight_manipulation"
    INPUT_ADVERSARIAL = "input_adversarial"
    ACTIVATION_ANOMALY = "activation_anomaly"
    LOSS_MANIPULATION = "loss_manipulation"
    MEMORY_CORRUPTION = "memory_corruption"
    TIMING_ATTACK = "timing_attack"


@dataclass
class AttackAlert:
    """Attack detection alert."""
    attack_type: AttackType
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    timestamp: float
    details: Dict[str, Any]
    confidence: float
    module_name: Optional[str] = None


class RealTimeAttackDetector(nn.Module):
    """
    Real-time attack detection system with <1ms latency requirements.
    
    Key Features:
    - Multi-layer anomaly detection
    - Real-time gradient monitoring
    - Statistical outlier detection
    - Adversarial input detection
    - Memory corruption detection
    """
    
    def __init__(
        self,
        detection_window: int = 100,
        anomaly_threshold: float = 3.0,
        gradient_threshold: float = 10.0,
        memory_threshold: float = 0.8,
        enable_logging: bool = True
    ):
        super().__init__()
        
        self.detection_window = detection_window
        self.anomaly_threshold = anomaly_threshold
        self.gradient_threshold = gradient_threshold
        self.memory_threshold = memory_threshold
        self.enable_logging = enable_logging
        
        # Detection state
        self.is_monitoring = True
        self.detection_start_time = time.time()
        
        # Historical data for anomaly detection
        self.gradient_history = deque(maxlen=detection_window)
        self.loss_history = deque(maxlen=detection_window)
        self.activation_history = deque(maxlen=detection_window)
        self.timing_history = deque(maxlen=detection_window)
        
        # Statistical baselines
        self.gradient_baseline = None
        self.loss_baseline = None
        self.activation_baseline = None
        
        # Alert management
        self.active_alerts = []
        self.alert_history = deque(maxlen=1000)
        self.alert_lock = threading.Lock()
        
        # Performance tracking
        self.detection_count = 0
        self.false_positive_count = 0
        self.detection_latency = deque(maxlen=100)
        
        # Hooks for monitoring
        self.hooks = []
        self.module_stats = {}
    
    def _compute_statistical_baseline(self, data: List[float], window: int = 50) -> Dict[str, float]:
        """Compute statistical baseline for anomaly detection."""
        if len(data) < window:
            return {'mean': 0.0, 'std': 1.0, 'q95': 0.0, 'q99': 0.0}
        
        recent_data = np.array(list(data)[-window:])
        return {
            'mean': float(np.mean(recent_data)),
            'std': float(np.std(recent_data)),
            'q95': float(np.percentile(recent_data, 95)),
            'q99': float(np.percentile(recent_data, 99))
        }
    
    def _detect_gradient_anomaly(self, gradients: Dict[str, torch.Tensor]) -> Optional[AttackAlert]:
        """Detect gradient-based attacks."""
        try:
            start_time = time.time()
            
            for name, grad in gradients.items():
                if grad is None:
                    continue
                
                # Compute gradient statistics
                grad_norm = torch.norm(grad).item()
                grad_max = torch.max(torch.abs(grad)).item()
                grad_mean = torch.mean(torch.abs(grad)).item()
                
                # Check for gradient explosion
                if grad_norm > self.gradient_threshold:
                    return AttackAlert(
                        attack_type=AttackType.GRADIENT_EXPLOSION,
                        severity='HIGH',
                        timestamp=time.time(),
                        details={
                            'parameter_name': name,
                            'gradient_norm': grad_norm,
                            'threshold': self.gradient_threshold,
                            'max_value': grad_max
                        },
                        confidence=0.9,
                        module_name=name
                    )
                
                # Check for gradient vanishing
                if grad_norm < 1e-8 and grad.numel() > 1:
                    return AttackAlert(
                        attack_type=AttackType.GRADIENT_VANISHING,
                        severity='MEDIUM',
                        timestamp=time.time(),
                        details={
                            'parameter_name': name,
                            'gradient_norm': grad_norm,
                            'mean_value': grad_mean
                        },
                        confidence=0.8,
                        module_name=name
                    )
                
                # Statistical anomaly detection
                self.gradient_history.append(grad_norm)
                if len(self.gradient_history) >= 20:
                    baseline = self._compute_statistical_baseline(list(self.gradient_history))
                    if grad_norm > baseline['mean'] + self.anomaly_threshold * baseline['std']:
                        return AttackAlert(
                            attack_type=AttackType.GRADIENT_EXPLOSION,
                            severity='MEDIUM',
                            timestamp=time.time(),
                            details={
                                'parameter_name': name,
                                'gradient_norm': grad_norm,
                                'baseline_mean': baseline['mean'],
                                'baseline_std': baseline['std'],
                                'z_score': (grad_norm - baseline['mean']) / baseline['std']
                            },
                            confidence=0.7,
                            module_name=name
                        )
            
            # Record detection latency
            detection_time = (time.time() - start_time) * 1000  # ms
            self.detection_latency.append(detection_time)
            
            return None
            
        except Exception:
            return None
    
    def _detect_activation_anomaly(self, activations: torch.Tensor, module_name: str) -> Optional[AttackAlert]:
        """Detect anomalies in layer activations."""
        try:
            # Quick statistical checks
            if torch.isnan(activations).any():
                return AttackAlert(
                    attack_type=AttackType.ACTIVATION_ANOMALY,
                    severity='CRITICAL',
                    timestamp=time.time(),
                    details={
                        'anomaly_type': 'nan_values',
                        'nan_count': torch.isnan(activations).sum().item()
                    },
                    confidence=1.0,
                    module_name=module_name
                )
            
            if torch.isinf(activations).any():
                return AttackAlert(
                    attack_type=AttackType.ACTIVATION_ANOMALY,
                    severity='CRITICAL',
                    timestamp=time.time(),
                    details={
                        'anomaly_type': 'inf_values',
                        'inf_count': torch.isinf(activations).sum().item()
                    },
                    confidence=1.0,
                    module_name=module_name
                )
            
            # Compute activation statistics
            activation_norm = torch.norm(activations).item()
            activation_max = torch.max(torch.abs(activations)).item()
            activation_mean = torch.mean(torch.abs(activations)).item()
            
            # Check for extreme values
            if activation_max > 1000.0:  # Configurable threshold
                return AttackAlert(
                    attack_type=AttackType.ACTIVATION_ANOMALY,
                    severity='HIGH',
                    timestamp=time.time(),
                    details={
                        'anomaly_type': 'extreme_values',
                        'max_activation': activation_max,
                        'norm': activation_norm
                    },
                    confidence=0.8,
                    module_name=module_name
                )
            
            # Store for statistical analysis
            self.activation_history.append(activation_norm)
            
            return None
            
        except Exception:
            return None
    
    def _detect_input_adversarial(self, inputs: torch.Tensor) -> Optional[AttackAlert]:
        """Detect adversarial inputs using statistical methods."""
        try:
            # Quick bounds check
            if torch.any(inputs < -10) or torch.any(inputs > 10):
                return AttackAlert(
                    attack_type=AttackType.INPUT_ADVERSARIAL,
                    severity='MEDIUM',
                    timestamp=time.time(),
                    details={
                        'anomaly_type': 'out_of_bounds',
                        'min_value': torch.min(inputs).item(),
                        'max_value': torch.max(inputs).item()
                    },
                    confidence=0.6
                )
            
            # Statistical outlier detection
            input_mean = torch.mean(inputs).item()
            input_std = torch.std(inputs).item()
            
            # Z-score based detection
            z_scores = torch.abs((inputs - input_mean) / (input_std + 1e-8))
            max_z_score = torch.max(z_scores).item()
            
            if max_z_score > 5.0:  # Very high z-score
                return AttackAlert(
                    attack_type=AttackType.INPUT_ADVERSARIAL,
                    severity='MEDIUM',
                    timestamp=time.time(),
                    details={
                        'anomaly_type': 'statistical_outlier',
                        'max_z_score': max_z_score,
                        'input_mean': input_mean,
                        'input_std': input_std
                    },
                    confidence=0.5
                )
            
            return None
            
        except Exception:
            return None
    
    def _detect_timing_attack(self, operation_time: float, operation_name: str) -> Optional[AttackAlert]:
        """Detect timing-based attacks."""
        try:
            self.timing_history.append(operation_time)
            
            if len(self.timing_history) >= 20:
                baseline = self._compute_statistical_baseline(list(self.timing_history))
                
                # Detect unusually slow operations (potential attack)
                if operation_time > baseline['mean'] + 3 * baseline['std']:
                    return AttackAlert(
                        attack_type=AttackType.TIMING_ATTACK,
                        severity='LOW',
                        timestamp=time.time(),
                        details={
                            'operation_name': operation_name,
                            'operation_time': operation_time,
                            'baseline_mean': baseline['mean'],
                            'baseline_std': baseline['std']
                        },
                        confidence=0.4
                    )
            
            return None
            
        except Exception:
            return None
    
    def register_module_hooks(self, module: nn.Module, module_name: str = ""):
        """Register hooks for real-time monitoring."""
        def forward_hook(module, input, output):
            if self.is_monitoring:
                # Detect activation anomalies
                if isinstance(output, torch.Tensor):
                    alert = self._detect_activation_anomaly(output, module_name)
                    if alert:
                        self._add_alert(alert)
        
        def backward_hook(module, grad_input, grad_output):
            if self.is_monitoring and grad_output[0] is not None:
                # Quick gradient check
                grad_norm = torch.norm(grad_output[0]).item()
                if grad_norm > self.gradient_threshold:
                    alert = AttackAlert(
                        attack_type=AttackType.GRADIENT_EXPLOSION,
                        severity='HIGH',
                        timestamp=time.time(),
                        details={'gradient_norm': grad_norm},
                        confidence=0.8,
                        module_name=module_name
                    )
                    self._add_alert(alert)
        
        # Register hooks
        forward_handle = module.register_forward_hook(forward_hook)
        backward_handle = module.register_backward_hook(backward_hook)
        
        self.hooks.extend([forward_handle, backward_handle])
    
    def _add_alert(self, alert: AttackAlert):
        """Add alert to the system."""
        with self.alert_lock:
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Keep only recent active alerts
            current_time = time.time()
            self.active_alerts = [
                a for a in self.active_alerts 
                if current_time - a.timestamp < 60  # Keep alerts for 1 minute
            ]
    
    def detect_attacks(
        self,
        inputs: Optional[torch.Tensor] = None,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        loss: Optional[torch.Tensor] = None,
        operation_name: str = "forward",
        operation_time: Optional[float] = None
    ) -> List[AttackAlert]:
        """
        Perform comprehensive attack detection.
        
        Args:
            inputs: Input tensors
            gradients: Gradient dictionary
            loss: Current loss value
            operation_name: Name of the operation
            operation_time: Time taken for operation
            
        Returns:
            List of detected attacks
        """
        if not self.is_monitoring:
            return []
        
        alerts = []
        self.detection_count += 1
        
        # Input-based detection
        if inputs is not None:
            alert = self._detect_input_adversarial(inputs)
            if alert:
                alerts.append(alert)
        
        # Gradient-based detection
        if gradients is not None:
            alert = self._detect_gradient_anomaly(gradients)
            if alert:
                alerts.append(alert)
        
        # Loss-based detection
        if loss is not None:
            loss_value = loss.item()
            self.loss_history.append(loss_value)
            
            # Check for loss manipulation
            if loss_value < 0 or loss_value > 1000:
                alert = AttackAlert(
                    attack_type=AttackType.LOSS_MANIPULATION,
                    severity='HIGH',
                    timestamp=time.time(),
                    details={'loss_value': loss_value},
                    confidence=0.9
                )
                alerts.append(alert)
        
        # Timing-based detection
        if operation_time is not None:
            alert = self._detect_timing_attack(operation_time, operation_name)
            if alert:
                alerts.append(alert)
        
        # Add alerts to system
        for alert in alerts:
            self._add_alert(alert)
        
        return alerts
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        with self.alert_lock:
            active_count = len(self.active_alerts)
            critical_count = sum(1 for a in self.active_alerts if a.severity == 'CRITICAL')
            high_count = sum(1 for a in self.active_alerts if a.severity == 'HIGH')
        
        avg_latency = np.mean(self.detection_latency) if self.detection_latency else 0
        
        return {
            'monitoring_active': self.is_monitoring,
            'active_alerts': active_count,
            'critical_alerts': critical_count,
            'high_severity_alerts': high_count,
            'total_detections': self.detection_count,
            'average_detection_latency_ms': avg_latency,
            'uptime_seconds': time.time() - self.detection_start_time
        }
    
    def get_recent_alerts(self, count: int = 10) -> List[AttackAlert]:
        """Get recent alerts."""
        with self.alert_lock:
            return list(self.alert_history)[-count:]
    
    def clear_alerts(self):
        """Clear all active alerts."""
        with self.alert_lock:
            self.active_alerts.clear()
    
    def enable_monitoring(self, enable: bool = True):
        """Enable or disable monitoring."""
        self.is_monitoring = enable
    
    def cleanup_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()