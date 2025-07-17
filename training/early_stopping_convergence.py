"""
Advanced Early Stopping and Convergence Detection System
Implements multiple convergence criteria, adaptive patience, and intelligent stopping
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings

logger = logging.getLogger(__name__)

class ConvergenceMethod(Enum):
    """Methods for detecting convergence"""
    LOSS_PLATEAU = "loss_plateau"
    GRADIENT_NORM = "gradient_norm"
    PARAMETER_CHANGE = "parameter_change"
    VALIDATION_IMPROVEMENT = "validation_improvement"
    STATISTICAL_TEST = "statistical_test"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    LEARNING_CURVE = "learning_curve"

class StoppingReason(Enum):
    """Reasons for stopping training"""
    CONVERGENCE = "convergence"
    EARLY_STOPPING = "early_stopping"
    MAX_EPOCHS = "max_epochs"
    USER_INTERRUPT = "user_interrupt"
    DIVERGENCE = "divergence"
    PLATEAU = "plateau"
    VALIDATION_LOSS_INCREASE = "validation_loss_increase"

@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection"""
    # Basic early stopping
    patience: int = 10
    min_delta: float = 1e-4
    restore_best_weights: bool = True
    
    # Adaptive patience
    adaptive_patience: bool = True
    patience_factor: float = 1.5
    max_patience: int = 50
    
    # Convergence detection methods
    enabled_methods: List[ConvergenceMethod] = None
    
    # Loss plateau detection
    plateau_patience: int = 15
    plateau_threshold: float = 1e-5
    plateau_window: int = 10
    
    # Gradient norm convergence
    gradient_norm_threshold: float = 1e-6
    gradient_norm_window: int = 20
    
    # Parameter change detection
    param_change_threshold: float = 1e-7
    param_change_window: int = 10
    
    # Statistical testing
    statistical_alpha: float = 0.05
    statistical_window: int = 50
    
    # Validation monitoring
    validation_patience: int = 20
    validation_threshold: float = 1e-3
    
    # Divergence detection
    divergence_threshold: float = 100.0
    max_loss_increase: float = 10.0
    
    # Learning curve analysis
    learning_curve_window: int = 100
    learning_curve_smoothing: float = 0.1
    
    # Ensemble variance (for multiple runs)
    ensemble_variance_threshold: float = 1e-4
    
    # Saving and logging
    save_convergence_data: bool = True
    convergence_log_file: str = "convergence_log.json"

@dataclass
class ConvergenceResult:
    """Result of convergence detection"""
    converged: bool
    method: ConvergenceMethod
    confidence: float
    stopping_reason: StoppingReason
    final_value: float
    convergence_epoch: int
    convergence_step: int
    analysis_data: Dict[str, Any]

class LossPlateauDetector:
    """Detect loss plateau using multiple criteria"""
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.loss_history = deque(maxlen=config.plateau_window * 2)
        self.best_loss = float('inf')
        self.plateau_count = 0
        self.last_improvement = 0
        
    def update(self, loss: float, epoch: int) -> bool:
        """Update with new loss value and check for plateau"""
        self.loss_history.append(loss)
        
        if loss < self.best_loss - self.config.min_delta:
            self.best_loss = loss
            self.plateau_count = 0
            self.last_improvement = epoch
        else:
            self.plateau_count += 1
        
        # Check for plateau
        if len(self.loss_history) >= self.config.plateau_window:
            recent_losses = list(self.loss_history)[-self.config.plateau_window:]
            
            # Check if losses are within threshold
            loss_variance = np.var(recent_losses)
            if loss_variance < self.config.plateau_threshold:
                return True
        
        return False
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get detailed plateau analysis"""
        if len(self.loss_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_losses = list(self.loss_history)[-self.config.plateau_window:]
        
        return {
            'current_loss': self.loss_history[-1],
            'best_loss': self.best_loss,
            'plateau_count': self.plateau_count,
            'last_improvement': self.last_improvement,
            'loss_variance': np.var(recent_losses) if len(recent_losses) > 1 else 0,
            'loss_trend': np.polyfit(range(len(recent_losses)), recent_losses, 1)[0] if len(recent_losses) > 1 else 0
        }

class GradientNormDetector:
    """Detect convergence based on gradient norm"""
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.gradient_norms = deque(maxlen=config.gradient_norm_window * 2)
        self.converged = False
        
    def update(self, model: torch.nn.Module) -> bool:
        """Update with model gradients and check convergence"""
        # Calculate gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_norms.append(total_norm)
        
        # Check convergence
        if len(self.gradient_norms) >= self.config.gradient_norm_window:
            recent_norms = list(self.gradient_norms)[-self.config.gradient_norm_window:]
            avg_norm = np.mean(recent_norms)
            
            if avg_norm < self.config.gradient_norm_threshold:
                self.converged = True
                return True
        
        return False
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get gradient norm analysis"""
        if not self.gradient_norms:
            return {'status': 'no_gradients'}
        
        return {
            'current_norm': self.gradient_norms[-1],
            'avg_norm': np.mean(self.gradient_norms),
            'norm_trend': np.polyfit(range(len(self.gradient_norms)), list(self.gradient_norms), 1)[0] if len(self.gradient_norms) > 1 else 0,
            'threshold': self.config.gradient_norm_threshold,
            'converged': self.converged
        }

class ParameterChangeDetector:
    """Detect convergence based on parameter changes"""
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.previous_params = None
        self.param_changes = deque(maxlen=config.param_change_window * 2)
        self.converged = False
        
    def update(self, model: torch.nn.Module) -> bool:
        """Update with model parameters and check convergence"""
        current_params = {}
        
        # Get current parameters
        for name, param in model.named_parameters():
            current_params[name] = param.data.clone()
        
        # Calculate parameter changes
        if self.previous_params is not None:
            total_change = 0.0
            total_params = 0
            
            for name, param in current_params.items():
                if name in self.previous_params:
                    change = torch.norm(param - self.previous_params[name]).item()
                    total_change += change
                    total_params += param.numel()
            
            # Normalize by number of parameters
            if total_params > 0:
                normalized_change = total_change / total_params
                self.param_changes.append(normalized_change)
        
        self.previous_params = current_params
        
        # Check convergence
        if len(self.param_changes) >= self.config.param_change_window:
            recent_changes = list(self.param_changes)[-self.config.param_change_window:]
            avg_change = np.mean(recent_changes)
            
            if avg_change < self.config.param_change_threshold:
                self.converged = True
                return True
        
        return False
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get parameter change analysis"""
        if not self.param_changes:
            return {'status': 'no_changes'}
        
        return {
            'current_change': self.param_changes[-1],
            'avg_change': np.mean(self.param_changes),
            'change_trend': np.polyfit(range(len(self.param_changes)), list(self.param_changes), 1)[0] if len(self.param_changes) > 1 else 0,
            'threshold': self.config.param_change_threshold,
            'converged': self.converged
        }

class StatisticalConvergenceDetector:
    """Detect convergence using statistical tests"""
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.loss_history = deque(maxlen=config.statistical_window * 2)
        self.converged = False
        
    def update(self, loss: float) -> bool:
        """Update with loss and perform statistical test"""
        self.loss_history.append(loss)
        
        if len(self.loss_history) >= self.config.statistical_window:
            # Split into two halves
            half_size = len(self.loss_history) // 2
            first_half = list(self.loss_history)[:half_size]
            second_half = list(self.loss_history)[half_size:]
            
            # Perform t-test
            try:
                statistic, p_value = stats.ttest_ind(first_half, second_half)
                
                # If p-value is high, the means are not significantly different
                if p_value > self.config.statistical_alpha:
                    self.converged = True
                    return True
                    
            except Exception as e:
                logger.debug(f"Statistical test failed: {e}")
        
        return False
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get statistical analysis"""
        if len(self.loss_history) < self.config.statistical_window:
            return {'status': 'insufficient_data'}
        
        # Split into two halves
        half_size = len(self.loss_history) // 2
        first_half = list(self.loss_history)[:half_size]
        second_half = list(self.loss_history)[half_size:]
        
        try:
            statistic, p_value = stats.ttest_ind(first_half, second_half)
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'alpha': self.config.statistical_alpha,
                'first_half_mean': np.mean(first_half),
                'second_half_mean': np.mean(second_half),
                'converged': self.converged
            }
        except Exception as e:
            return {'status': 'test_failed', 'error': str(e)}

class LearningCurveAnalyzer:
    """Analyze learning curve for convergence patterns"""
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        self.loss_history = deque(maxlen=config.learning_curve_window * 2)
        self.smoothed_losses = deque(maxlen=config.learning_curve_window * 2)
        self.converged = False
        
    def update(self, loss: float) -> bool:
        """Update with loss and analyze learning curve"""
        self.loss_history.append(loss)
        
        # Apply exponential smoothing
        if self.smoothed_losses:
            smoothed = (self.config.learning_curve_smoothing * loss + 
                       (1 - self.config.learning_curve_smoothing) * self.smoothed_losses[-1])
        else:
            smoothed = loss
        
        self.smoothed_losses.append(smoothed)
        
        # Analyze curve
        if len(self.smoothed_losses) >= self.config.learning_curve_window:
            recent_losses = list(self.smoothed_losses)[-self.config.learning_curve_window:]
            
            # Check for convergence patterns
            if self._detect_convergence_pattern(recent_losses):
                self.converged = True
                return True
        
        return False
    
    def _detect_convergence_pattern(self, losses: List[float]) -> bool:
        """Detect convergence patterns in loss curve"""
        if len(losses) < 20:
            return False
        
        # Check for slope approaching zero
        x = np.arange(len(losses))
        slope, _ = np.polyfit(x, losses, 1)
        
        # Check for oscillation around a mean
        detrended = losses - np.polyval([slope, losses[0]], x)
        oscillation_variance = np.var(detrended)
        
        # Check for decreasing variance
        first_half_var = np.var(losses[:len(losses)//2])
        second_half_var = np.var(losses[len(losses)//2:])
        
        # Convergence criteria
        slope_converged = abs(slope) < 1e-6
        low_oscillation = oscillation_variance < 1e-4
        decreasing_variance = second_half_var < first_half_var * 0.5
        
        return slope_converged and (low_oscillation or decreasing_variance)
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get learning curve analysis"""
        if len(self.smoothed_losses) < 10:
            return {'status': 'insufficient_data'}
        
        losses = list(self.smoothed_losses)
        x = np.arange(len(losses))
        slope, intercept = np.polyfit(x, losses, 1)
        
        return {
            'current_loss': losses[-1],
            'slope': slope,
            'intercept': intercept,
            'variance': np.var(losses),
            'smoothing_factor': self.config.learning_curve_smoothing,
            'converged': self.converged
        }

class EarlyStoppingConvergenceDetector:
    """
    Advanced early stopping and convergence detection system
    """
    
    def __init__(self, config: ConvergenceConfig):
        self.config = config
        
        # Initialize enabled methods
        if config.enabled_methods is None:
            self.enabled_methods = [
                ConvergenceMethod.LOSS_PLATEAU,
                ConvergenceMethod.VALIDATION_IMPROVEMENT,
                ConvergenceMethod.GRADIENT_NORM
            ]
        else:
            self.enabled_methods = config.enabled_methods
        
        # Initialize detectors
        self.detectors = {}
        
        if ConvergenceMethod.LOSS_PLATEAU in self.enabled_methods:
            self.detectors[ConvergenceMethod.LOSS_PLATEAU] = LossPlateauDetector(config)
        
        if ConvergenceMethod.GRADIENT_NORM in self.enabled_methods:
            self.detectors[ConvergenceMethod.GRADIENT_NORM] = GradientNormDetector(config)
        
        if ConvergenceMethod.PARAMETER_CHANGE in self.enabled_methods:
            self.detectors[ConvergenceMethod.PARAMETER_CHANGE] = ParameterChangeDetector(config)
        
        if ConvergenceMethod.STATISTICAL_TEST in self.enabled_methods:
            self.detectors[ConvergenceMethod.STATISTICAL_TEST] = StatisticalConvergenceDetector(config)
        
        if ConvergenceMethod.LEARNING_CURVE in self.enabled_methods:
            self.detectors[ConvergenceMethod.LEARNING_CURVE] = LearningCurveAnalyzer(config)
        
        # Training state
        self.training_history = []
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.current_patience = config.patience
        self.converged = False
        self.stopping_reason = None
        
        # Validation monitoring
        self.validation_history = []
        self.validation_patience_counter = 0
        self.best_validation_loss = float('inf')
        
        # Divergence detection
        self.divergence_detected = False
        self.initial_loss = None
        
        logger.info(f"Early stopping detector initialized with methods: {self.enabled_methods}")
    
    def update(self, 
               train_loss: float,
               validation_loss: Optional[float] = None,
               model: Optional[torch.nn.Module] = None,
               epoch: int = 0,
               step: int = 0) -> Tuple[bool, ConvergenceResult]:
        """
        Update detector with new training data
        
        Returns:
            Tuple of (should_stop, convergence_result)
        """
        # Initialize if first update
        if self.initial_loss is None:
            self.initial_loss = train_loss
        
        # Store training history
        self.training_history.append({
            'epoch': epoch,
            'step': step,
            'train_loss': train_loss,
            'validation_loss': validation_loss,
            'timestamp': time.time()
        })
        
        # Check for divergence
        if self._check_divergence(train_loss):
            result = ConvergenceResult(
                converged=False,
                method=ConvergenceMethod.LOSS_PLATEAU,
                confidence=1.0,
                stopping_reason=StoppingReason.DIVERGENCE,
                final_value=train_loss,
                convergence_epoch=epoch,
                convergence_step=step,
                analysis_data=self._get_full_analysis()
            )
            return True, result
        
        # Update all detectors
        convergence_votes = {}
        
        # Loss plateau detector
        if ConvergenceMethod.LOSS_PLATEAU in self.detectors:
            detector = self.detectors[ConvergenceMethod.LOSS_PLATEAU]
            if detector.update(train_loss, epoch):
                convergence_votes[ConvergenceMethod.LOSS_PLATEAU] = 1.0
        
        # Gradient norm detector
        if ConvergenceMethod.GRADIENT_NORM in self.detectors and model is not None:
            detector = self.detectors[ConvergenceMethod.GRADIENT_NORM]
            if detector.update(model):
                convergence_votes[ConvergenceMethod.GRADIENT_NORM] = 1.0
        
        # Parameter change detector
        if ConvergenceMethod.PARAMETER_CHANGE in self.detectors and model is not None:
            detector = self.detectors[ConvergenceMethod.PARAMETER_CHANGE]
            if detector.update(model):
                convergence_votes[ConvergenceMethod.PARAMETER_CHANGE] = 1.0
        
        # Statistical test detector
        if ConvergenceMethod.STATISTICAL_TEST in self.detectors:
            detector = self.detectors[ConvergenceMethod.STATISTICAL_TEST]
            if detector.update(train_loss):
                convergence_votes[ConvergenceMethod.STATISTICAL_TEST] = 1.0
        
        # Learning curve detector
        if ConvergenceMethod.LEARNING_CURVE in self.detectors:
            detector = self.detectors[ConvergenceMethod.LEARNING_CURVE]
            if detector.update(train_loss):
                convergence_votes[ConvergenceMethod.LEARNING_CURVE] = 1.0
        
        # Validation improvement check
        validation_converged = False
        if validation_loss is not None:
            validation_converged = self._check_validation_convergence(validation_loss, epoch)
            if validation_converged:
                convergence_votes[ConvergenceMethod.VALIDATION_IMPROVEMENT] = 1.0
        
        # Early stopping check
        early_stopping_triggered = self._check_early_stopping(train_loss, validation_loss, epoch)
        
        # Determine overall convergence
        should_stop = False
        stopping_reason = None
        convergence_method = None
        confidence = 0.0
        
        if convergence_votes:
            # Calculate confidence based on voting
            confidence = sum(convergence_votes.values()) / len(self.enabled_methods)
            
            # Choose method with highest confidence
            convergence_method = max(convergence_votes.keys(), key=lambda k: convergence_votes[k])
            
            # Stop if confidence is high enough
            if confidence > 0.5:  # At least half the methods agree
                should_stop = True
                stopping_reason = StoppingReason.CONVERGENCE
        
        elif early_stopping_triggered:
            should_stop = True
            stopping_reason = StoppingReason.EARLY_STOPPING
            confidence = 1.0
            convergence_method = ConvergenceMethod.VALIDATION_IMPROVEMENT
        
        # Create result
        result = ConvergenceResult(
            converged=len(convergence_votes) > 0,
            method=convergence_method or ConvergenceMethod.LOSS_PLATEAU,
            confidence=confidence,
            stopping_reason=stopping_reason or StoppingReason.CONVERGENCE,
            final_value=train_loss,
            convergence_epoch=epoch,
            convergence_step=step,
            analysis_data=self._get_full_analysis()
        )
        
        # Log convergence data
        if self.config.save_convergence_data:
            self._save_convergence_data(result)
        
        return should_stop, result
    
    def _check_divergence(self, train_loss: float) -> bool:
        """Check if training is diverging"""
        if self.initial_loss is None:
            return False
        
        # Check for explosive growth
        if train_loss > self.initial_loss * self.config.divergence_threshold:
            self.divergence_detected = True
            return True
        
        # Check for recent increase
        if len(self.training_history) > 10:
            recent_losses = [h['train_loss'] for h in self.training_history[-10:]]
            if train_loss > min(recent_losses) * self.config.max_loss_increase:
                self.divergence_detected = True
                return True
        
        return False
    
    def _check_validation_convergence(self, validation_loss: float, epoch: int) -> bool:
        """Check validation loss convergence"""
        self.validation_history.append({
            'epoch': epoch,
            'validation_loss': validation_loss
        })
        
        if validation_loss < self.best_validation_loss - self.config.validation_threshold:
            self.best_validation_loss = validation_loss
            self.validation_patience_counter = 0
            return False
        else:
            self.validation_patience_counter += 1
            return self.validation_patience_counter >= self.config.validation_patience
    
    def _check_early_stopping(self, train_loss: float, validation_loss: Optional[float], epoch: int) -> bool:
        """Check traditional early stopping criteria"""
        # Use validation loss if available, otherwise train loss
        current_loss = validation_loss if validation_loss is not None else train_loss
        
        if current_loss < self.best_loss - self.config.min_delta:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.patience_counter = 0
            
            # Adaptive patience reduction
            if self.config.adaptive_patience:
                self.current_patience = max(
                    self.config.patience,
                    int(self.current_patience / self.config.patience_factor)
                )
            
            return False
        else:
            self.patience_counter += 1
            
            # Adaptive patience increase
            if self.config.adaptive_patience and self.patience_counter > self.current_patience // 2:
                self.current_patience = min(
                    self.config.max_patience,
                    int(self.current_patience * self.config.patience_factor)
                )
            
            return self.patience_counter >= self.current_patience
    
    def _get_full_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis from all detectors"""
        analysis = {
            'training_history': self.training_history[-100:],  # Last 100 points
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'current_patience': self.current_patience,
            'divergence_detected': self.divergence_detected,
            'validation_history': self.validation_history[-50:],  # Last 50 points
            'detector_analysis': {}
        }
        
        # Get analysis from each detector
        for method, detector in self.detectors.items():
            analysis['detector_analysis'][method.value] = detector.get_analysis()
        
        return analysis
    
    def _save_convergence_data(self, result: ConvergenceResult):
        """Save convergence data to file"""
        try:
            convergence_data = {
                'timestamp': time.time(),
                'result': {
                    'converged': result.converged,
                    'method': result.method.value if result.method else None,
                    'confidence': result.confidence,
                    'stopping_reason': result.stopping_reason.value if result.stopping_reason else None,
                    'final_value': result.final_value,
                    'convergence_epoch': result.convergence_epoch,
                    'convergence_step': result.convergence_step
                },
                'config': {
                    'patience': self.config.patience,
                    'min_delta': self.config.min_delta,
                    'enabled_methods': [m.value for m in self.enabled_methods]
                }
            }
            
            # Append to log file
            log_file = Path(self.config.convergence_log_file)
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    existing_data = json.load(f)
                existing_data.append(convergence_data)
            else:
                existing_data = [convergence_data]
            
            with open(log_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save convergence data: {e}")
    
    def plot_convergence_analysis(self, save_path: Optional[str] = None):
        """Plot convergence analysis"""
        if not self.training_history:
            logger.warning("No training history available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
        axes[0, 0].axhline(y=self.best_loss, color='r', linestyle='--', label='Best Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation loss
        if self.validation_history:
            val_epochs = [h['epoch'] for h in self.validation_history]
            val_losses = [h['validation_loss'] for h in self.validation_history]
            
            axes[0, 1].plot(val_epochs, val_losses, 'g-', label='Validation Loss')
            axes[0, 1].axhline(y=self.best_validation_loss, color='r', linestyle='--', label='Best Validation Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Gradient norms (if available)
        if ConvergenceMethod.GRADIENT_NORM in self.detectors:
            detector = self.detectors[ConvergenceMethod.GRADIENT_NORM]
            if detector.gradient_norms:
                axes[1, 0].plot(list(detector.gradient_norms), 'purple', label='Gradient Norm')
                axes[1, 0].axhline(y=self.config.gradient_norm_threshold, color='r', linestyle='--', label='Threshold')
                axes[1, 0].set_title('Gradient Norm')
                axes[1, 0].set_xlabel('Update')
                axes[1, 0].set_ylabel('Norm')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
        
        # Parameter changes (if available)
        if ConvergenceMethod.PARAMETER_CHANGE in self.detectors:
            detector = self.detectors[ConvergenceMethod.PARAMETER_CHANGE]
            if detector.param_changes:
                axes[1, 1].plot(list(detector.param_changes), 'orange', label='Parameter Change')
                axes[1, 1].axhline(y=self.config.param_change_threshold, color='r', linestyle='--', label='Threshold')
                axes[1, 1].set_title('Parameter Changes')
                axes[1, 1].set_xlabel('Update')
                axes[1, 1].set_ylabel('Change')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get comprehensive convergence summary"""
        return {
            'config': {
                'patience': self.config.patience,
                'min_delta': self.config.min_delta,
                'enabled_methods': [m.value for m in self.enabled_methods]
            },
            'state': {
                'best_loss': self.best_loss,
                'best_epoch': self.best_epoch,
                'patience_counter': self.patience_counter,
                'current_patience': self.current_patience,
                'converged': self.converged,
                'divergence_detected': self.divergence_detected
            },
            'training_statistics': {
                'total_epochs': len(self.training_history),
                'final_loss': self.training_history[-1]['train_loss'] if self.training_history else None,
                'loss_improvement': (self.initial_loss - self.best_loss) / self.initial_loss if self.initial_loss else 0
            },
            'detector_analysis': {
                method.value: detector.get_analysis() 
                for method, detector in self.detectors.items()
            }
        }


def create_convergence_config(
    patience: int = 10,
    min_delta: float = 1e-4,
    enable_all_methods: bool = False
) -> ConvergenceConfig:
    """Create optimized convergence configuration"""
    
    if enable_all_methods:
        enabled_methods = list(ConvergenceMethod)
    else:
        enabled_methods = [
            ConvergenceMethod.LOSS_PLATEAU,
            ConvergenceMethod.VALIDATION_IMPROVEMENT,
            ConvergenceMethod.GRADIENT_NORM
        ]
    
    config = ConvergenceConfig(
        patience=patience,
        min_delta=min_delta,
        enabled_methods=enabled_methods,
        adaptive_patience=True,
        save_convergence_data=True,
        restore_best_weights=True
    )
    
    return config


# Example usage functions
def setup_basic_early_stopping(patience: int = 10, min_delta: float = 1e-4) -> EarlyStoppingConvergenceDetector:
    """Setup basic early stopping"""
    config = create_convergence_config(patience, min_delta)
    return EarlyStoppingConvergenceDetector(config)


def setup_advanced_convergence_detection(
    patience: int = 20,
    enable_all_methods: bool = True
) -> EarlyStoppingConvergenceDetector:
    """Setup advanced convergence detection with all methods"""
    config = create_convergence_config(patience, enable_all_methods=enable_all_methods)
    return EarlyStoppingConvergenceDetector(config)