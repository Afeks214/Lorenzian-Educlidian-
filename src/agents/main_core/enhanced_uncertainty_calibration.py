"""
Enhanced Real-Time Uncertainty Calibration System

This module implements an advanced uncertainty calibration system that:
1. Continuously updates calibration in real-time based on execution outcomes
2. Adapts to different market regimes with regime-aware calibration adjustments
3. Uses multi-model ensemble calibration with dynamic weighting
4. Monitors calibration performance with comprehensive metrics
5. Provides adaptive calibration parameters based on trading conditions
6. Integrates execution outcome feedback for continuous learning

The system maintains calibration accuracy during live trading by learning from
actual execution results and adjusting uncertainty estimates accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from collections import deque, defaultdict
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pickle
import json
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications for calibration adjustment."""
    TRENDING = "trending"
    VOLATILE = "volatile"
    RANGING = "ranging"
    TRANSITIONING = "transitioning"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class CalibrationMethod(Enum):
    """Available calibration methods."""
    TEMPERATURE = "temperature"
    PLATT = "platt"
    ISOTONIC = "isotonic"
    BETA = "beta"
    HISTOGRAM = "histogram"
    ENSEMBLE = "ensemble"


@dataclass
class CalibrationSample:
    """Single calibration sample with metadata."""
    timestamp: float
    predicted_probability: float
    actual_outcome: bool
    confidence_score: float
    uncertainty_metrics: Dict[str, float]
    market_regime: MarketRegime
    execution_context: Dict[str, Any]
    trade_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration performance metrics."""
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    log_loss: float
    reliability_score: float
    sharpness_score: float
    resolution_score: float
    calibration_slope: float
    calibration_intercept: float
    confidence_interval_coverage: Dict[str, float]
    regime_specific_metrics: Dict[MarketRegime, Dict[str, float]]
    
    
@dataclass
class AdaptiveCalibrationConfig:
    """Configuration for adaptive calibration parameters."""
    learning_rate: float = 0.01
    adaptation_window: int = 100
    min_samples_for_adaptation: int = 50
    regime_sensitivity: float = 0.1
    volatility_sensitivity: float = 0.05
    performance_decay_factor: float = 0.95
    confidence_threshold_bounds: Tuple[float, float] = (0.5, 0.95)
    ensemble_weight_bounds: Tuple[float, float] = (0.05, 0.8)


class EnhancedUncertaintyCalibrator:
    """
    Enhanced real-time uncertainty calibration system with regime awareness,
    multi-model ensemble, and continuous learning from execution outcomes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibration_config = AdaptiveCalibrationConfig(
            **config.get('adaptive_calibration', {})
        )
        
        # Core calibration components
        self.calibration_methods = self._initialize_calibration_methods()
        self.ensemble_weights = self._initialize_ensemble_weights()
        
        # Real-time data storage
        self.calibration_samples = deque(maxlen=config.get('max_samples', 10000))
        self.regime_samples = defaultdict(lambda: deque(maxlen=2000))
        self.execution_outcomes = deque(maxlen=5000)
        
        # Performance tracking
        self.calibration_metrics = self._initialize_metrics()
        self.regime_metrics = defaultdict(lambda: self._initialize_metrics())
        self.performance_history = deque(maxlen=1000)
        
        # Adaptive parameters
        self.adaptive_thresholds = self._initialize_adaptive_thresholds()
        self.regime_adjustments = self._initialize_regime_adjustments()
        
        # Real-time learning components
        self.learning_scheduler = self._initialize_learning_scheduler()
        self.outcome_buffer = deque(maxlen=500)
        
        # Monitoring and alerting
        self.monitoring_enabled = config.get('monitoring_enabled', True)
        self.alert_thresholds = config.get('alert_thresholds', {})
        
        # Thread safety
        self.calibration_lock = threading.RLock()
        self.update_thread = None
        self.running = False
        
        # Performance optimization
        self.cache_size = config.get('cache_size', 1000)
        self.calibration_cache = {}
        
        logger.info("Enhanced uncertainty calibration system initialized")
        
    def _initialize_calibration_methods(self) -> Dict[CalibrationMethod, Any]:
        """Initialize all calibration methods."""
        methods = {
            CalibrationMethod.TEMPERATURE: EnhancedTemperatureCalibrator(),
            CalibrationMethod.PLATT: EnhancedPlattCalibrator(),
            CalibrationMethod.ISOTONIC: EnhancedIsotonicCalibrator(),
            CalibrationMethod.BETA: EnhancedBetaCalibrator(),
            CalibrationMethod.HISTOGRAM: EnhancedHistogramCalibrator(
                n_bins=self.config.get('histogram_bins', 15)
            )
        }
        return methods
        
    def _initialize_ensemble_weights(self) -> Dict[CalibrationMethod, float]:
        """Initialize ensemble weights for different calibration methods."""
        default_weights = {
            CalibrationMethod.TEMPERATURE: 0.25,
            CalibrationMethod.PLATT: 0.20,
            CalibrationMethod.ISOTONIC: 0.30,
            CalibrationMethod.BETA: 0.15,
            CalibrationMethod.HISTOGRAM: 0.10
        }
        
        config_weights = self.config.get('ensemble_weights', {})
        
        for method, weight in config_weights.items():
            if isinstance(method, str):
                method = CalibrationMethod(method)
            default_weights[method] = weight
            
        # Normalize weights
        total_weight = sum(default_weights.values())
        for method in default_weights:
            default_weights[method] /= total_weight
            
        return default_weights
        
    def _initialize_metrics(self) -> CalibrationMetrics:
        """Initialize calibration metrics structure."""
        return CalibrationMetrics(
            expected_calibration_error=0.0,
            maximum_calibration_error=0.0,
            brier_score=0.0,
            log_loss=0.0,
            reliability_score=0.0,
            sharpness_score=0.0,
            resolution_score=0.0,
            calibration_slope=1.0,
            calibration_intercept=0.0,
            confidence_interval_coverage={
                '50%': 0.5, '68%': 0.68, '80%': 0.8, '95%': 0.95
            },
            regime_specific_metrics={}
        )
        
    def _initialize_adaptive_thresholds(self) -> Dict[str, float]:
        """Initialize adaptive confidence thresholds."""
        return {
            'base_threshold': self.config.get('base_threshold', 0.65),
            'regime_adjustment': 0.0,
            'volatility_adjustment': 0.0,
            'performance_adjustment': 0.0,
            'adaptive_factor': 1.0
        }
        
    def _initialize_regime_adjustments(self) -> Dict[MarketRegime, Dict[str, float]]:
        """Initialize regime-specific calibration adjustments."""
        return {
            MarketRegime.TRENDING: {
                'threshold_adjustment': 0.0,
                'temperature_adjustment': 0.9,
                'ensemble_weight_adjustment': 1.0
            },
            MarketRegime.VOLATILE: {
                'threshold_adjustment': 0.05,
                'temperature_adjustment': 1.2,
                'ensemble_weight_adjustment': 0.8
            },
            MarketRegime.RANGING: {
                'threshold_adjustment': 0.02,
                'temperature_adjustment': 1.0,
                'ensemble_weight_adjustment': 1.0
            },
            MarketRegime.TRANSITIONING: {
                'threshold_adjustment': 0.08,
                'temperature_adjustment': 1.3,
                'ensemble_weight_adjustment': 0.7
            },
            MarketRegime.CRISIS: {
                'threshold_adjustment': 0.15,
                'temperature_adjustment': 1.5,
                'ensemble_weight_adjustment': 0.6
            },
            MarketRegime.RECOVERY: {
                'threshold_adjustment': 0.03,
                'temperature_adjustment': 1.1,
                'ensemble_weight_adjustment': 0.9
            }
        }
        
    def _initialize_learning_scheduler(self) -> Dict[str, Any]:
        """Initialize learning rate scheduler for adaptive calibration."""
        return {
            'base_learning_rate': self.calibration_config.learning_rate,
            'current_learning_rate': self.calibration_config.learning_rate,
            'decay_factor': 0.99,
            'min_learning_rate': 0.001,
            'adaptation_counter': 0,
            'performance_trend': deque(maxlen=10)
        }
        
    def start_real_time_calibration(self):
        """Start real-time calibration update thread."""
        if self.running:
            return
            
        self.running = True
        self.update_thread = threading.Thread(
            target=self._continuous_calibration_update,
            daemon=True
        )
        self.update_thread.start()
        logger.info("Real-time calibration system started")
        
    def stop_real_time_calibration(self):
        """Stop real-time calibration update thread."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("Real-time calibration system stopped")
        
    def calibrate_uncertainty(
        self,
        raw_probabilities: torch.Tensor,
        uncertainty_metrics: Dict[str, float],
        market_context: Optional[Dict[str, Any]] = None,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply enhanced calibration to uncertainty estimates.
        
        Args:
            raw_probabilities: Raw probability predictions
            uncertainty_metrics: Uncertainty decomposition metrics
            market_context: Current market conditions and regime
            execution_context: Execution-specific context
            
        Returns:
            Tuple of (calibrated_probabilities, calibration_info)
        """
        with self.calibration_lock:
            # Detect market regime
            market_regime = self._detect_market_regime(market_context)
            
            # Apply regime-aware calibration
            calibrated_probs = self._apply_regime_aware_calibration(
                raw_probabilities,
                uncertainty_metrics,
                market_regime,
                execution_context
            )
            
            # Calculate adaptive confidence threshold
            adaptive_threshold = self._calculate_adaptive_threshold(
                market_regime,
                uncertainty_metrics,
                market_context
            )
            
            # Generate calibration metadata
            calibration_info = {
                'market_regime': market_regime.value,
                'adaptive_threshold': adaptive_threshold,
                'ensemble_weights': dict(self.ensemble_weights),
                'calibration_quality': self._assess_calibration_quality(market_regime),
                'regime_adjustments': self.regime_adjustments[market_regime],
                'uncertainty_adjustment': self._calculate_uncertainty_adjustment(
                    uncertainty_metrics
                )
            }
            
            return calibrated_probs, calibration_info
            
    def _detect_market_regime(
        self, 
        market_context: Optional[Dict[str, Any]]
    ) -> MarketRegime:
        """Detect current market regime from context."""
        if not market_context:
            return MarketRegime.RANGING
            
        # Extract market indicators
        volatility = market_context.get('volatility', 1.0)
        volume = market_context.get('volume', 1.0)
        trend_strength = market_context.get('trend_strength', 0.0)
        market_stress = market_context.get('stress_indicator', 0.0)
        
        # Regime classification logic
        if market_stress > 0.8:
            return MarketRegime.CRISIS
        elif market_stress > 0.6 and volatility > 2.0:
            return MarketRegime.VOLATILE
        elif trend_strength > 0.6 and volatility < 1.5:
            return MarketRegime.TRENDING
        elif volatility > 2.5 or volume > 2.0:
            return MarketRegime.VOLATILE
        elif trend_strength < 0.2 and volatility < 1.2:
            return MarketRegime.RANGING
        else:
            return MarketRegime.TRANSITIONING
            
    def _apply_regime_aware_calibration(
        self,
        raw_probabilities: torch.Tensor,
        uncertainty_metrics: Dict[str, float],
        market_regime: MarketRegime,
        execution_context: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        """Apply regime-aware ensemble calibration."""
        
        # Get regime-specific adjustments
        regime_adjustments = self.regime_adjustments[market_regime]
        
        # Apply individual calibration methods
        calibrated_results = {}
        for method, calibrator in self.calibration_methods.items():
            try:
                # Apply regime-specific adjustments to calibrator
                calibrated_probs = calibrator.calibrate(
                    raw_probabilities,
                    uncertainty_metrics,
                    regime_adjustments
                )
                calibrated_results[method] = calibrated_probs
            except Exception as e:
                logger.warning(f"Calibration method {method.value} failed: {e}")
                calibrated_results[method] = raw_probabilities
                
        # Ensemble combination with regime-aware weights
        ensemble_result = self._ensemble_calibration(
            calibrated_results,
            market_regime,
            uncertainty_metrics
        )
        
        return ensemble_result
        
    def _ensemble_calibration(
        self,
        calibrated_results: Dict[CalibrationMethod, torch.Tensor],
        market_regime: MarketRegime,
        uncertainty_metrics: Dict[str, float]
    ) -> torch.Tensor:
        """Combine calibration results using dynamic ensemble weighting."""
        
        # Get regime-specific weight adjustments
        regime_weight_adjustment = self.regime_adjustments[market_regime]['ensemble_weight_adjustment']
        
        # Calculate uncertainty-based weight adjustments
        epistemic_uncertainty = uncertainty_metrics.get('epistemic_uncertainty', 0.0)
        aleatoric_uncertainty = uncertainty_metrics.get('aleatoric_uncertainty', 0.0)
        
        # Adjust weights based on uncertainty and regime
        adjusted_weights = {}
        for method, base_weight in self.ensemble_weights.items():
            if method not in calibrated_results:
                continue
                
            # Regime adjustment
            weight = base_weight * regime_weight_adjustment
            
            # Uncertainty-based adjustments
            if method == CalibrationMethod.TEMPERATURE:
                # Temperature scaling works better with low epistemic uncertainty
                weight *= (1.0 - epistemic_uncertainty * 0.5)
            elif method == CalibrationMethod.ISOTONIC:
                # Isotonic regression works better with high epistemic uncertainty
                weight *= (1.0 + epistemic_uncertainty * 0.3)
            elif method == CalibrationMethod.BETA:
                # Beta calibration works better with extreme probabilities
                weight *= (1.0 + aleatoric_uncertainty * 0.2)
                
            adjusted_weights[method] = weight
            
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for method in adjusted_weights:
                adjusted_weights[method] /= total_weight
        else:
            # Fallback to uniform weights
            n_methods = len(adjusted_weights)
            for method in adjusted_weights:
                adjusted_weights[method] = 1.0 / n_methods
                
        # Combine results
        ensemble_result = torch.zeros_like(list(calibrated_results.values())[0])
        
        for method, weight in adjusted_weights.items():
            if method in calibrated_results:
                ensemble_result += weight * calibrated_results[method]
                
        return ensemble_result
        
    def _calculate_adaptive_threshold(
        self,
        market_regime: MarketRegime,
        uncertainty_metrics: Dict[str, float],
        market_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate adaptive confidence threshold based on context."""
        
        # Base threshold
        threshold = self.adaptive_thresholds['base_threshold']
        
        # Regime adjustment
        regime_adjustment = self.regime_adjustments[market_regime]['threshold_adjustment']
        threshold += regime_adjustment
        
        # Uncertainty adjustment
        epistemic_uncertainty = uncertainty_metrics.get('epistemic_uncertainty', 0.0)
        if epistemic_uncertainty > 0.3:
            uncertainty_adjustment = min(0.1, (epistemic_uncertainty - 0.3) * 0.2)
            threshold += uncertainty_adjustment
            
        # Performance-based adjustment
        performance_adjustment = self.adaptive_thresholds['performance_adjustment']
        threshold += performance_adjustment
        
        # Market volatility adjustment
        if market_context:
            volatility = market_context.get('volatility', 1.0)
            if volatility > 1.5:
                volatility_adjustment = min(0.05, (volatility - 1.5) * 0.02)
                threshold += volatility_adjustment
                
        # Clamp to valid range
        min_threshold, max_threshold = self.calibration_config.confidence_threshold_bounds
        threshold = np.clip(threshold, min_threshold, max_threshold)
        
        return threshold
        
    def _calculate_uncertainty_adjustment(
        self,
        uncertainty_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate uncertainty-based adjustments."""
        
        epistemic = uncertainty_metrics.get('epistemic_uncertainty', 0.0)
        aleatoric = uncertainty_metrics.get('aleatoric_uncertainty', 0.0)
        total = uncertainty_metrics.get('total_uncertainty', 0.0)
        
        return {
            'epistemic_adjustment': epistemic * 0.1,
            'aleatoric_adjustment': aleatoric * 0.05,
            'total_adjustment': total * 0.08,
            'confidence_penalty': min(0.1, total * 0.15)
        }
        
    def add_execution_outcome(
        self,
        predicted_probability: float,
        actual_outcome: bool,
        confidence_score: float,
        uncertainty_metrics: Dict[str, float],
        market_context: Dict[str, Any],
        execution_context: Dict[str, Any],
        trade_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add execution outcome for real-time calibration learning.
        
        Args:
            predicted_probability: Original predicted probability
            actual_outcome: Whether the prediction was correct
            confidence_score: Confidence score at time of prediction
            uncertainty_metrics: Uncertainty decomposition
            market_context: Market conditions during prediction
            execution_context: Execution-specific context
            trade_metadata: Additional trade information
        """
        
        # Create calibration sample
        sample = CalibrationSample(
            timestamp=time.time(),
            predicted_probability=predicted_probability,
            actual_outcome=actual_outcome,
            confidence_score=confidence_score,
            uncertainty_metrics=uncertainty_metrics,
            market_regime=self._detect_market_regime(market_context),
            execution_context=execution_context,
            trade_metadata=trade_metadata or {}
        )
        
        with self.calibration_lock:
            # Add to main sample collection
            self.calibration_samples.append(sample)
            
            # Add to regime-specific collection
            self.regime_samples[sample.market_regime].append(sample)
            
            # Add to outcome buffer for immediate processing
            self.outcome_buffer.append(sample)
            
        logger.debug(f"Added execution outcome: prob={predicted_probability:.3f}, "
                    f"actual={actual_outcome}, regime={sample.market_regime.value}")
                    
    def _continuous_calibration_update(self):
        """Continuous calibration update loop."""
        update_interval = self.config.get('update_interval', 5.0)
        
        while self.running:
            try:
                # Process pending outcomes
                if len(self.outcome_buffer) >= self.calibration_config.min_samples_for_adaptation:
                    self._process_outcome_buffer()
                    
                # Update calibration models
                self._update_calibration_models()
                
                # Update ensemble weights
                self._update_ensemble_weights()
                
                # Update adaptive thresholds
                self._update_adaptive_thresholds()
                
                # Calculate performance metrics
                self._calculate_performance_metrics()
                
                # Check for alerts
                if self.monitoring_enabled:
                    self._check_calibration_alerts()
                    
                # Sleep until next update
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous calibration update: {e}")
                time.sleep(update_interval)
                
    def _process_outcome_buffer(self):
        """Process the outcome buffer for immediate calibration updates."""
        with self.calibration_lock:
            if not self.outcome_buffer:
                return
                
            # Extract recent outcomes
            recent_outcomes = list(self.outcome_buffer)
            self.outcome_buffer.clear()
            
            # Group by regime
            regime_outcomes = defaultdict(list)
            for outcome in recent_outcomes:
                regime_outcomes[outcome.market_regime].append(outcome)
                
            # Update calibration for each regime
            for regime, outcomes in regime_outcomes.items():
                if len(outcomes) >= 10:  # Minimum samples for regime update
                    self._update_regime_calibration(regime, outcomes)
                    
    def _update_regime_calibration(
        self,
        regime: MarketRegime,
        outcomes: List[CalibrationSample]
    ):
        """Update calibration for a specific regime."""
        
        # Extract predictions and actual outcomes
        predictions = [sample.predicted_probability for sample in outcomes]
        actuals = [sample.actual_outcome for sample in outcomes]
        
        # Update each calibration method for this regime
        for method, calibrator in self.calibration_methods.items():
            try:
                calibrator.update_with_regime(
                    torch.tensor(predictions),
                    torch.tensor(actuals, dtype=torch.float32),
                    regime
                )
            except Exception as e:
                logger.warning(f"Failed to update {method.value} for regime {regime.value}: {e}")
                
    def _update_calibration_models(self):
        """Update all calibration models with recent data."""
        
        if len(self.calibration_samples) < 100:
            return
            
        # Get recent samples
        recent_samples = list(self.calibration_samples)[-1000:]
        
        # Extract data for fitting
        predictions = torch.tensor([s.predicted_probability for s in recent_samples])
        outcomes = torch.tensor([s.actual_outcome for s in recent_samples], dtype=torch.float32)
        
        # Update each calibration method
        for method, calibrator in self.calibration_methods.items():
            try:
                calibrator.fit(predictions, outcomes)
            except Exception as e:
                logger.warning(f"Failed to update calibration method {method.value}: {e}")
                
    def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance."""
        
        if len(self.calibration_samples) < 200:
            return
            
        # Evaluate each method's performance
        method_performances = {}
        
        for method, calibrator in self.calibration_methods.items():
            try:
                performance = self._evaluate_calibration_method(method, calibrator)
                method_performances[method] = performance
            except Exception as e:
                logger.warning(f"Failed to evaluate {method.value}: {e}")
                method_performances[method] = 0.5  # Default score
                
        # Update weights based on performance
        if method_performances:
            # Convert to weights (higher performance = higher weight)
            total_performance = sum(method_performances.values())
            
            if total_performance > 0:
                for method, performance in method_performances.items():
                    new_weight = performance / total_performance
                    
                    # Smooth update
                    current_weight = self.ensemble_weights[method]
                    learning_rate = self.learning_scheduler['current_learning_rate']
                    
                    updated_weight = (1 - learning_rate) * current_weight + learning_rate * new_weight
                    
                    # Apply bounds
                    min_weight, max_weight = self.calibration_config.ensemble_weight_bounds
                    updated_weight = np.clip(updated_weight, min_weight, max_weight)
                    
                    self.ensemble_weights[method] = updated_weight
                    
                # Normalize weights
                total_weight = sum(self.ensemble_weights.values())
                for method in self.ensemble_weights:
                    self.ensemble_weights[method] /= total_weight
                    
    def _evaluate_calibration_method(
        self,
        method: CalibrationMethod,
        calibrator: Any
    ) -> float:
        """Evaluate the performance of a calibration method."""
        
        # Get recent samples for evaluation
        recent_samples = list(self.calibration_samples)[-500:]
        
        if len(recent_samples) < 50:
            return 0.5
            
        # Extract data
        predictions = torch.tensor([s.predicted_probability for s in recent_samples])
        outcomes = torch.tensor([s.actual_outcome for s in recent_samples], dtype=torch.float32)
        uncertainty_metrics = [s.uncertainty_metrics for s in recent_samples]
        
        # Apply calibration
        try:
            calibrated_probs = []
            for i, pred in enumerate(predictions):
                cal_prob = calibrator.calibrate(
                    pred.unsqueeze(0).unsqueeze(-1),
                    uncertainty_metrics[i],
                    {}  # No regime adjustments for evaluation
                )
                calibrated_probs.append(cal_prob.item())
                
            calibrated_probs = torch.tensor(calibrated_probs)
            
            # Calculate calibration error (lower is better)
            ece = self._calculate_expected_calibration_error(calibrated_probs, outcomes)
            
            # Convert to performance score (higher is better)
            performance = 1.0 / (1.0 + ece)
            
            return performance
            
        except Exception as e:
            logger.warning(f"Error evaluating {method.value}: {e}")
            return 0.5
            
    def _calculate_expected_calibration_error(
        self,
        predictions: torch.Tensor,
        outcomes: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        n_samples = len(predictions)
        ece = 0.0
        
        for i in range(n_bins):
            # Find samples in this bin
            if i == 0:
                mask = predictions <= bin_boundaries[i + 1]
            elif i == n_bins - 1:
                mask = predictions > bin_boundaries[i]
            else:
                mask = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
                
            if mask.sum() > 0:
                bin_confidence = predictions[mask].mean()
                bin_accuracy = outcomes[mask].mean()
                bin_weight = mask.sum().float() / n_samples
                
                ece += bin_weight * torch.abs(bin_accuracy - bin_confidence)
                
        return ece.item()
        
    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on performance."""
        
        if len(self.performance_history) < 10:
            return
            
        # Calculate recent performance trend
        recent_performance = list(self.performance_history)[-10:]
        performance_trend = np.mean([p['calibration_quality'] for p in recent_performance])
        
        # Update performance adjustment
        target_performance = 0.8
        performance_error = target_performance - performance_trend
        
        # Proportional adjustment
        adjustment = performance_error * 0.01
        
        # Apply adjustment
        current_adjustment = self.adaptive_thresholds['performance_adjustment']
        new_adjustment = current_adjustment + adjustment
        
        # Apply bounds
        self.adaptive_thresholds['performance_adjustment'] = np.clip(new_adjustment, -0.05, 0.05)
        
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        
        if len(self.calibration_samples) < 100:
            return
            
        # Get recent samples
        recent_samples = list(self.calibration_samples)[-1000:]
        
        # Calculate overall metrics
        overall_metrics = self._calculate_calibration_metrics(recent_samples)
        self.calibration_metrics = overall_metrics
        
        # Calculate regime-specific metrics
        for regime in MarketRegime:
            regime_samples = [s for s in recent_samples if s.market_regime == regime]
            if len(regime_samples) >= 20:
                regime_metrics = self._calculate_calibration_metrics(regime_samples)
                self.regime_metrics[regime] = regime_metrics
                
        # Update performance history
        performance_record = {
            'timestamp': time.time(),
            'calibration_quality': overall_metrics.expected_calibration_error,
            'brier_score': overall_metrics.brier_score,
            'log_loss': overall_metrics.log_loss,
            'sample_count': len(recent_samples)
        }
        
        self.performance_history.append(performance_record)
        
    def _calculate_calibration_metrics(
        self,
        samples: List[CalibrationSample]
    ) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics for given samples."""
        
        predictions = torch.tensor([s.predicted_probability for s in samples])
        outcomes = torch.tensor([s.actual_outcome for s in samples], dtype=torch.float32)
        
        # Expected Calibration Error
        ece = self._calculate_expected_calibration_error(predictions, outcomes)
        
        # Maximum Calibration Error
        mce = self._calculate_maximum_calibration_error(predictions, outcomes)
        
        # Brier Score
        brier_score = torch.mean((predictions - outcomes) ** 2).item()
        
        # Log Loss
        epsilon = 1e-15
        clipped_predictions = torch.clamp(predictions, epsilon, 1 - epsilon)
        log_loss = -torch.mean(
            outcomes * torch.log(clipped_predictions) + 
            (1 - outcomes) * torch.log(1 - clipped_predictions)
        ).item()
        
        # Reliability, Sharpness, and Resolution
        reliability = self._calculate_reliability(predictions, outcomes)
        sharpness = self._calculate_sharpness(predictions)
        resolution = self._calculate_resolution(predictions, outcomes)
        
        # Calibration slope and intercept
        slope, intercept = self._calculate_calibration_slope(predictions, outcomes)
        
        # Confidence interval coverage
        coverage = self._calculate_confidence_interval_coverage(samples)
        
        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier_score,
            log_loss=log_loss,
            reliability_score=reliability,
            sharpness_score=sharpness,
            resolution_score=resolution,
            calibration_slope=slope,
            calibration_intercept=intercept,
            confidence_interval_coverage=coverage,
            regime_specific_metrics={}
        )
        
    def _calculate_maximum_calibration_error(
        self,
        predictions: torch.Tensor,
        outcomes: torch.Tensor,
        n_bins: int = 10
    ) -> float:
        """Calculate Maximum Calibration Error."""
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        mce = 0.0
        
        for i in range(n_bins):
            if i == 0:
                mask = predictions <= bin_boundaries[i + 1]
            elif i == n_bins - 1:
                mask = predictions > bin_boundaries[i]
            else:
                mask = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
                
            if mask.sum() > 0:
                bin_confidence = predictions[mask].mean()
                bin_accuracy = outcomes[mask].mean()
                error = torch.abs(bin_accuracy - bin_confidence).item()
                mce = max(mce, error)
                
        return mce
        
    def _calculate_reliability(
        self,
        predictions: torch.Tensor,
        outcomes: torch.Tensor
    ) -> float:
        """Calculate reliability component of Brier score decomposition."""
        
        # Group by prediction bins
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        reliability = 0.0
        
        for i in range(n_bins):
            if i == 0:
                mask = predictions <= bin_boundaries[i + 1]
            elif i == n_bins - 1:
                mask = predictions > bin_boundaries[i]
            else:
                mask = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
                
            if mask.sum() > 0:
                bin_prob = predictions[mask].mean()
                bin_freq = outcomes[mask].mean()
                bin_size = mask.sum().float() / len(predictions)
                
                reliability += bin_size * (bin_prob - bin_freq) ** 2
                
        return reliability.item()
        
    def _calculate_sharpness(self, predictions: torch.Tensor) -> float:
        """Calculate sharpness (variance of predictions)."""
        return torch.var(predictions).item()
        
    def _calculate_resolution(
        self,
        predictions: torch.Tensor,
        outcomes: torch.Tensor
    ) -> float:
        """Calculate resolution component of Brier score decomposition."""
        
        # Group by prediction bins
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        resolution = 0.0
        overall_mean = outcomes.mean()
        
        for i in range(n_bins):
            if i == 0:
                mask = predictions <= bin_boundaries[i + 1]
            elif i == n_bins - 1:
                mask = predictions > bin_boundaries[i]
            else:
                mask = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
                
            if mask.sum() > 0:
                bin_freq = outcomes[mask].mean()
                bin_size = mask.sum().float() / len(predictions)
                
                resolution += bin_size * (bin_freq - overall_mean) ** 2
                
        return resolution.item()
        
    def _calculate_calibration_slope(
        self,
        predictions: torch.Tensor,
        outcomes: torch.Tensor
    ) -> Tuple[float, float]:
        """Calculate calibration slope and intercept via linear regression."""
        
        # Convert to numpy for sklearn
        X = predictions.numpy().reshape(-1, 1)
        y = outcomes.numpy()
        
        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X, y)
        
        slope = reg.coef_[0]
        intercept = reg.intercept_
        
        return slope, intercept
        
    def _calculate_confidence_interval_coverage(
        self,
        samples: List[CalibrationSample]
    ) -> Dict[str, float]:
        """Calculate confidence interval coverage statistics."""
        
        # This would require storing confidence intervals from original predictions
        # For now, return default values
        return {
            '50%': 0.5,
            '68%': 0.68,
            '80%': 0.8,
            '95%': 0.95
        }
        
    def _assess_calibration_quality(self, regime: MarketRegime) -> float:
        """Assess overall calibration quality for a regime."""
        
        if regime in self.regime_metrics:
            metrics = self.regime_metrics[regime]
            
            # Combine metrics into overall quality score
            ece_score = 1.0 - metrics.expected_calibration_error
            brier_score = 1.0 - metrics.brier_score
            reliability_score = 1.0 - metrics.reliability_score
            
            # Weighted average
            quality = (ece_score * 0.4 + brier_score * 0.3 + reliability_score * 0.3)
            
            return max(0.0, min(1.0, quality))
        
        return 0.5  # Default quality
        
    def _check_calibration_alerts(self):
        """Check for calibration alerts and warnings."""
        
        if not self.alert_thresholds:
            return
            
        # Check ECE threshold
        ece_threshold = self.alert_thresholds.get('ece_threshold', 0.1)
        if self.calibration_metrics.expected_calibration_error > ece_threshold:
            logger.warning(f"High Expected Calibration Error: "
                          f"{self.calibration_metrics.expected_calibration_error:.3f}")
            
        # Check Brier score threshold
        brier_threshold = self.alert_thresholds.get('brier_threshold', 0.3)
        if self.calibration_metrics.brier_score > brier_threshold:
            logger.warning(f"High Brier Score: {self.calibration_metrics.brier_score:.3f}")
            
        # Check calibration slope
        slope_threshold = self.alert_thresholds.get('slope_threshold', 0.2)
        if abs(self.calibration_metrics.calibration_slope - 1.0) > slope_threshold:
            logger.warning(f"Calibration slope deviation: "
                          f"{self.calibration_metrics.calibration_slope:.3f}")
            
    def get_calibration_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive calibration diagnostics."""
        
        with self.calibration_lock:
            diagnostics = {
                'overall_metrics': {
                    'expected_calibration_error': self.calibration_metrics.expected_calibration_error,
                    'maximum_calibration_error': self.calibration_metrics.maximum_calibration_error,
                    'brier_score': self.calibration_metrics.brier_score,
                    'log_loss': self.calibration_metrics.log_loss,
                    'reliability_score': self.calibration_metrics.reliability_score,
                    'sharpness_score': self.calibration_metrics.sharpness_score,
                    'resolution_score': self.calibration_metrics.resolution_score,
                    'calibration_slope': self.calibration_metrics.calibration_slope,
                    'calibration_intercept': self.calibration_metrics.calibration_intercept
                },
                'regime_metrics': {
                    regime.value: {
                        'expected_calibration_error': metrics.expected_calibration_error,
                        'brier_score': metrics.brier_score,
                        'sample_count': len(self.regime_samples[regime])
                    } for regime, metrics in self.regime_metrics.items()
                },
                'ensemble_weights': dict(self.ensemble_weights),
                'adaptive_thresholds': dict(self.adaptive_thresholds),
                'sample_counts': {
                    'total_samples': len(self.calibration_samples),
                    'regime_samples': {
                        regime.value: len(samples) 
                        for regime, samples in self.regime_samples.items()
                    }
                },
                'performance_trend': list(self.performance_history)[-10:],
                'learning_scheduler': dict(self.learning_scheduler)
            }
            
        return diagnostics
        
    def save_calibration_state(self, filepath: str):
        """Save calibration state to file."""
        
        state = {
            'calibration_methods': self.calibration_methods,
            'ensemble_weights': self.ensemble_weights,
            'adaptive_thresholds': self.adaptive_thresholds,
            'regime_adjustments': self.regime_adjustments,
            'calibration_samples': list(self.calibration_samples)[-1000:],  # Recent samples
            'performance_history': list(self.performance_history),
            'config': self.config,
            'timestamp': time.time()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
            
        logger.info(f"Calibration state saved to {filepath}")
        
    def load_calibration_state(self, filepath: str):
        """Load calibration state from file."""
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.calibration_methods = state['calibration_methods']
        self.ensemble_weights = state['ensemble_weights']
        self.adaptive_thresholds = state['adaptive_thresholds']
        self.regime_adjustments = state['regime_adjustments']
        
        # Restore samples
        for sample in state['calibration_samples']:
            self.calibration_samples.append(sample)
            self.regime_samples[sample.market_regime].append(sample)
            
        self.performance_history = deque(state['performance_history'], maxlen=1000)
        
        logger.info(f"Calibration state loaded from {filepath}")


# Enhanced calibration method implementations

class EnhancedTemperatureCalibrator:
    """Enhanced temperature scaling with regime awareness."""
    
    def __init__(self):
        self.temperature = 1.0
        self.regime_temperatures = {regime: 1.0 for regime in MarketRegime}
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Dict[str, float],
        regime_adjustments: Dict[str, float]
    ) -> torch.Tensor:
        """Apply temperature scaling with regime adjustments."""
        
        # Adjust temperature based on regime
        temp_adjustment = regime_adjustments.get('temperature_adjustment', 1.0)
        adjusted_temp = self.temperature * temp_adjustment
        
        # Further adjust based on uncertainty
        epistemic = uncertainty_metrics.get('epistemic_uncertainty', 0)
        adjusted_temp *= (1.0 + epistemic * 0.3)
        
        # Apply temperature scaling
        if probs.min() > 0 and probs.max() < 1:
            logits = torch.log(probs / (1 - probs + 1e-8))
            scaled_logits = logits / adjusted_temp
            return torch.sigmoid(scaled_logits)
        else:
            return torch.sigmoid(probs / adjusted_temp)
            
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit temperature parameter."""
        # Simplified fitting - in practice would use proper optimization
        accuracy = (predictions.round() == targets).float().mean()
        if accuracy > 0.8:
            self.temperature *= 1.05
        else:
            self.temperature *= 0.95
            
        self.temperature = torch.clamp(torch.tensor(self.temperature), 0.5, 2.0).item()
        
    def update_with_regime(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        regime: MarketRegime
    ):
        """Update temperature for specific regime."""
        # This would implement regime-specific temperature learning
        pass


class EnhancedPlattCalibrator:
    """Enhanced Platt scaling with regime awareness."""
    
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.regime_params = {regime: {'a': 1.0, 'b': 0.0} for regime in MarketRegime}
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Dict[str, float],
        regime_adjustments: Dict[str, float]
    ) -> torch.Tensor:
        """Apply Platt scaling with regime adjustments."""
        
        # Convert to logits
        if probs.min() > 0 and probs.max() < 1:
            logits = torch.log(probs / (1 - probs + 1e-8))
        else:
            logits = probs
            
        # Apply linear transformation
        calibrated_logits = self.a * logits + self.b
        
        # Adjust for uncertainty
        epistemic = uncertainty_metrics.get('epistemic_uncertainty', 0)
        if epistemic > 0:
            calibrated_logits *= (1.0 - epistemic * 0.2)
            
        return torch.sigmoid(calibrated_logits)
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit Platt scaling parameters."""
        # Simplified fitting
        error = predictions - targets
        self.a *= (1.0 - 0.01 * error.mean())
        self.b -= 0.01 * error.mean()
        
    def update_with_regime(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        regime: MarketRegime
    ):
        """Update parameters for specific regime."""
        pass


class EnhancedIsotonicCalibrator:
    """Enhanced isotonic regression with regime awareness."""
    
    def __init__(self):
        self.isotonic_regressor = None
        self.regime_regressors = {regime: None for regime in MarketRegime}
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Dict[str, float],
        regime_adjustments: Dict[str, float]
    ) -> torch.Tensor:
        """Apply isotonic calibration with regime adjustments."""
        
        if self.isotonic_regressor is None:
            return probs
            
        # Apply isotonic regression
        probs_np = probs.cpu().numpy().flatten()
        calibrated_np = self.isotonic_regressor.transform(probs_np)
        calibrated = torch.from_numpy(calibrated_np).to(probs.device)
        calibrated = calibrated.reshape(probs.shape)
        
        # Adjust for uncertainty
        epistemic = uncertainty_metrics.get('epistemic_uncertainty', 0)
        if epistemic > 0:
            calibrated = calibrated * (1 - epistemic * 0.15) + 0.5 * epistemic * 0.15
            
        return calibrated
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit isotonic regression."""
        X = predictions.cpu().numpy()
        y = targets.cpu().numpy()
        
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(X, y)
        
    def update_with_regime(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        regime: MarketRegime
    ):
        """Update regressor for specific regime."""
        pass


class EnhancedBetaCalibrator:
    """Enhanced beta calibration with regime awareness."""
    
    def __init__(self):
        self.alpha = 1.0
        self.beta_param = 1.0
        self.regime_params = {regime: {'alpha': 1.0, 'beta': 1.0} for regime in MarketRegime}
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Dict[str, float],
        regime_adjustments: Dict[str, float]
    ) -> torch.Tensor:
        """Apply beta calibration with regime adjustments."""
        
        probs = torch.clamp(probs, 1e-7, 1-1e-7)
        
        calibrated = []
        for p in probs.flatten():
            cal_p = beta.cdf(p.item(), self.alpha, self.beta_param)
            calibrated.append(cal_p)
            
        calibrated = torch.tensor(calibrated, device=probs.device)
        calibrated = calibrated.reshape(probs.shape)
        
        return calibrated
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit beta parameters."""
        # Simplified fitting
        pass
        
    def update_with_regime(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        regime: MarketRegime
    ):
        """Update parameters for specific regime."""
        pass


class EnhancedHistogramCalibrator:
    """Enhanced histogram calibration with regime awareness."""
    
    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_calibrations = torch.ones(n_bins) * 0.5
        self.regime_calibrations = {
            regime: torch.ones(n_bins) * 0.5 for regime in MarketRegime
        }
        
    def calibrate(
        self,
        probs: torch.Tensor,
        uncertainty_metrics: Dict[str, float],
        regime_adjustments: Dict[str, float]
    ) -> torch.Tensor:
        """Apply histogram calibration with regime adjustments."""
        
        calibrated = torch.zeros_like(probs)
        
        for i in range(self.n_bins):
            if i == 0:
                mask = probs <= self.bin_boundaries[i + 1]
            elif i == self.n_bins - 1:
                mask = probs > self.bin_boundaries[i]
            else:
                mask = (probs > self.bin_boundaries[i]) & (probs <= self.bin_boundaries[i + 1])
                
            calibrated[mask] = self.bin_calibrations[i]
            
        return calibrated
        
    def fit(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Fit histogram bins."""
        for i in range(self.n_bins):
            if i == 0:
                mask = predictions <= self.bin_boundaries[i + 1]
            elif i == self.n_bins - 1:
                mask = predictions > self.bin_boundaries[i]
            else:
                mask = (predictions > self.bin_boundaries[i]) & (predictions <= self.bin_boundaries[i + 1])
                
            if mask.sum() > 0:
                self.bin_calibrations[i] = targets[mask].mean()
                
    def update_with_regime(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        regime: MarketRegime
    ):
        """Update calibration for specific regime."""
        pass