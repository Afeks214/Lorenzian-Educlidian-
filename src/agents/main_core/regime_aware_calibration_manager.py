"""
Regime-Aware Calibration Manager

This module provides a comprehensive regime-aware calibration management system
that automatically detects market regimes and adjusts calibration strategies
accordingly. It integrates with the existing MC Dropout and uncertainty systems
to provide regime-specific calibration adjustments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
from collections import deque, defaultdict
import threading

from .enhanced_uncertainty_calibration import (
    EnhancedUncertaintyCalibrator,
    MarketRegime,
    CalibrationSample,
    CalibrationMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class RegimeContext:
    """Context information for regime detection and calibration."""
    volatility: float
    volume: float
    trend_strength: float
    market_stress: float
    liquidity: float
    spread: float
    momentum: float
    correlation: float
    timestamp: float
    
    
@dataclass
class RegimeTransition:
    """Information about regime transitions."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_time: float
    confidence: float
    transition_speed: float
    

class RegimeDetector:
    """Advanced regime detection using multiple indicators."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_window = config.get('lookback_window', 50)
        self.transition_threshold = config.get('transition_threshold', 0.7)
        
        # Historical regime data
        self.regime_history = deque(maxlen=1000)
        self.indicator_history = deque(maxlen=500)
        
        # Regime detection thresholds
        self.regime_thresholds = {
            'volatility': {
                'high': 2.0,
                'medium': 1.5,
                'low': 0.8
            },
            'trend_strength': {
                'strong': 0.6,
                'moderate': 0.3,
                'weak': 0.1
            },
            'stress': {
                'crisis': 0.8,
                'high': 0.6,
                'medium': 0.4,
                'low': 0.2
            }
        }
        
        # Regime transition detection
        self.transition_detector = RegimeTransitionDetector(config)
        
    def detect_regime(self, market_context: Dict[str, Any]) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime with confidence score.
        
        Args:
            market_context: Current market conditions
            
        Returns:
            Tuple of (detected_regime, confidence_score)
        """
        
        # Extract regime context
        regime_context = self._extract_regime_context(market_context)
        
        # Store in history
        self.indicator_history.append(regime_context)
        
        # Calculate regime probabilities
        regime_probs = self._calculate_regime_probabilities(regime_context)
        
        # Select regime with highest probability
        best_regime = max(regime_probs, key=regime_probs.get)
        confidence = regime_probs[best_regime]
        
        # Check for regime transitions
        transition_info = self.transition_detector.detect_transition(
            self.regime_history,
            best_regime,
            confidence
        )
        
        # Store regime in history
        self.regime_history.append({
            'regime': best_regime,
            'confidence': confidence,
            'timestamp': time.time(),
            'transition_info': transition_info
        })
        
        logger.debug(f"Detected regime: {best_regime.value} "
                    f"(confidence: {confidence:.3f})")
        
        return best_regime, confidence
        
    def _extract_regime_context(self, market_context: Dict[str, Any]) -> RegimeContext:
        """Extract regime-relevant context from market data."""
        
        return RegimeContext(
            volatility=market_context.get('volatility', 1.0),
            volume=market_context.get('volume', 1.0),
            trend_strength=market_context.get('trend_strength', 0.0),
            market_stress=market_context.get('stress_indicator', 0.0),
            liquidity=market_context.get('liquidity', 1.0),
            spread=market_context.get('bid_ask_spread', 0.01),
            momentum=market_context.get('momentum', 0.0),
            correlation=market_context.get('correlation', 0.0),
            timestamp=time.time()
        )
        
    def _calculate_regime_probabilities(
        self, 
        context: RegimeContext
    ) -> Dict[MarketRegime, float]:
        """Calculate probability of each regime given current context."""
        
        regime_scores = {}
        
        # Crisis regime detection
        crisis_score = self._calculate_crisis_score(context)
        regime_scores[MarketRegime.CRISIS] = crisis_score
        
        # Volatile regime detection
        volatile_score = self._calculate_volatile_score(context)
        regime_scores[MarketRegime.VOLATILE] = volatile_score
        
        # Trending regime detection
        trending_score = self._calculate_trending_score(context)
        regime_scores[MarketRegime.TRENDING] = trending_score
        
        # Ranging regime detection
        ranging_score = self._calculate_ranging_score(context)
        regime_scores[MarketRegime.RANGING] = ranging_score
        
        # Transitioning regime detection
        transitioning_score = self._calculate_transitioning_score(context)
        regime_scores[MarketRegime.TRANSITIONING] = transitioning_score
        
        # Recovery regime detection
        recovery_score = self._calculate_recovery_score(context)
        regime_scores[MarketRegime.RECOVERY] = recovery_score
        
        # Normalize scores to probabilities
        total_score = sum(regime_scores.values())
        if total_score > 0:
            regime_probs = {
                regime: score / total_score 
                for regime, score in regime_scores.items()
            }
        else:
            # Default to ranging if no clear signals
            regime_probs = {regime: 1.0/len(regime_scores) for regime in regime_scores}
            
        return regime_probs
        
    def _calculate_crisis_score(self, context: RegimeContext) -> float:
        """Calculate crisis regime score."""
        
        score = 0.0
        
        # High market stress
        if context.market_stress > self.regime_thresholds['stress']['crisis']:
            score += 0.4
        elif context.market_stress > self.regime_thresholds['stress']['high']:
            score += 0.2
            
        # Extreme volatility
        if context.volatility > self.regime_thresholds['volatility']['high'] * 1.5:
            score += 0.3
            
        # Low liquidity
        if context.liquidity < 0.3:
            score += 0.2
            
        # Wide spreads
        if context.spread > 0.05:
            score += 0.1
            
        return min(score, 1.0)
        
    def _calculate_volatile_score(self, context: RegimeContext) -> float:
        """Calculate volatile regime score."""
        
        score = 0.0
        
        # High volatility
        if context.volatility > self.regime_thresholds['volatility']['high']:
            score += 0.4
        elif context.volatility > self.regime_thresholds['volatility']['medium']:
            score += 0.2
            
        # High volume
        if context.volume > 1.8:
            score += 0.2
            
        # Moderate stress
        if (self.regime_thresholds['stress']['medium'] < 
            context.market_stress < self.regime_thresholds['stress']['high']):
            score += 0.2
            
        # Wide spreads
        if context.spread > 0.02:
            score += 0.1
            
        # Low correlation breakdown
        if abs(context.correlation) < 0.3:
            score += 0.1
            
        return min(score, 1.0)
        
    def _calculate_trending_score(self, context: RegimeContext) -> float:
        """Calculate trending regime score."""
        
        score = 0.0
        
        # Strong trend
        if context.trend_strength > self.regime_thresholds['trend_strength']['strong']:
            score += 0.5
        elif context.trend_strength > self.regime_thresholds['trend_strength']['moderate']:
            score += 0.3
            
        # Moderate volatility
        if (self.regime_thresholds['volatility']['low'] < 
            context.volatility < self.regime_thresholds['volatility']['high']):
            score += 0.2
            
        # Strong momentum
        if abs(context.momentum) > 0.4:
            score += 0.2
            
        # Normal liquidity
        if context.liquidity > 0.7:
            score += 0.1
            
        return min(score, 1.0)
        
    def _calculate_ranging_score(self, context: RegimeContext) -> float:
        """Calculate ranging regime score."""
        
        score = 0.0
        
        # Weak trend
        if context.trend_strength < self.regime_thresholds['trend_strength']['weak']:
            score += 0.4
            
        # Low volatility
        if context.volatility < self.regime_thresholds['volatility']['medium']:
            score += 0.3
            
        # Low momentum
        if abs(context.momentum) < 0.2:
            score += 0.2
            
        # Low stress
        if context.market_stress < self.regime_thresholds['stress']['medium']:
            score += 0.1
            
        return min(score, 1.0)
        
    def _calculate_transitioning_score(self, context: RegimeContext) -> float:
        """Calculate transitioning regime score."""
        
        score = 0.0
        
        # Check for regime instability in recent history
        if len(self.regime_history) >= 5:
            recent_regimes = [r['regime'] for r in list(self.regime_history)[-5:]]
            unique_regimes = set(recent_regimes)
            
            if len(unique_regimes) > 2:
                score += 0.3
                
        # Moderate to high volatility
        if (self.regime_thresholds['volatility']['medium'] < 
            context.volatility < self.regime_thresholds['volatility']['high']):
            score += 0.2
            
        # Moderate trend strength
        if (self.regime_thresholds['trend_strength']['weak'] < 
            context.trend_strength < self.regime_thresholds['trend_strength']['strong']):
            score += 0.2
            
        # Moderate stress
        if (self.regime_thresholds['stress']['low'] < 
            context.market_stress < self.regime_thresholds['stress']['high']):
            score += 0.2
            
        # Variable correlation
        if 0.3 < abs(context.correlation) < 0.7:
            score += 0.1
            
        return min(score, 1.0)
        
    def _calculate_recovery_score(self, context: RegimeContext) -> float:
        """Calculate recovery regime score."""
        
        score = 0.0
        
        # Check for recent crisis or volatile regime
        if len(self.regime_history) >= 3:
            recent_regimes = [r['regime'] for r in list(self.regime_history)[-3:]]
            
            if (MarketRegime.CRISIS in recent_regimes or 
                MarketRegime.VOLATILE in recent_regimes):
                score += 0.3
                
        # Decreasing stress
        if (self.regime_thresholds['stress']['low'] < 
            context.market_stress < self.regime_thresholds['stress']['medium']):
            score += 0.2
            
        # Moderate volatility (decreasing)
        if (self.regime_thresholds['volatility']['medium'] < 
            context.volatility < self.regime_thresholds['volatility']['high']):
            score += 0.2
            
        # Improving liquidity
        if context.liquidity > 0.6:
            score += 0.2
            
        # Positive momentum
        if context.momentum > 0.1:
            score += 0.1
            
        return min(score, 1.0)


class RegimeTransitionDetector:
    """Detects regime transitions and their characteristics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transition_threshold = config.get('transition_threshold', 0.7)
        self.min_regime_duration = config.get('min_regime_duration', 10)
        
    def detect_transition(
        self,
        regime_history: deque,
        current_regime: MarketRegime,
        confidence: float
    ) -> Optional[RegimeTransition]:
        """Detect regime transitions."""
        
        if len(regime_history) < 2:
            return None
            
        # Get previous regime
        prev_regime_info = regime_history[-1]
        prev_regime = prev_regime_info['regime']
        
        # Check if regime has changed
        if current_regime != prev_regime:
            # Calculate transition characteristics
            transition_speed = self._calculate_transition_speed(regime_history)
            
            transition = RegimeTransition(
                from_regime=prev_regime,
                to_regime=current_regime,
                transition_time=time.time(),
                confidence=confidence,
                transition_speed=transition_speed
            )
            
            logger.info(f"Regime transition detected: {prev_regime.value} -> "
                       f"{current_regime.value} (confidence: {confidence:.3f})")
            
            return transition
            
        return None
        
    def _calculate_transition_speed(self, regime_history: deque) -> float:
        """Calculate the speed of regime transition."""
        
        if len(regime_history) < 3:
            return 0.5
            
        # Look at confidence changes over recent periods
        recent_confidences = [r['confidence'] for r in list(regime_history)[-5:]]
        
        if len(recent_confidences) < 2:
            return 0.5
            
        # Calculate confidence volatility as proxy for transition speed
        conf_std = np.std(recent_confidences)
        
        # Normalize to 0-1 scale
        speed = min(conf_std * 2, 1.0)
        
        return speed


class RegimeAwareCalibrationManager:
    """
    Comprehensive regime-aware calibration management system that integrates
    with the enhanced uncertainty calibration system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.regime_detector = RegimeDetector(config.get('regime_detection', {}))
        self.enhanced_calibrator = EnhancedUncertaintyCalibrator(config.get('calibration', {}))
        
        # Regime-specific calibration configurations
        self.regime_calibration_configs = self._initialize_regime_configs()
        
        # Performance tracking by regime
        self.regime_performance = defaultdict(lambda: {
            'total_decisions': 0,
            'correct_decisions': 0,
            'avg_confidence': 0.0,
            'calibration_quality': 0.0
        })
        
        # Regime transition management
        self.transition_handler = RegimeTransitionHandler(config)
        
        # Current regime state
        self.current_regime = MarketRegime.RANGING
        self.regime_confidence = 0.5
        self.regime_stability = 1.0
        
        # Thread safety
        self.calibration_lock = threading.RLock()
        
        logger.info("Regime-aware calibration manager initialized")
        
    def _initialize_regime_configs(self) -> Dict[MarketRegime, Dict[str, Any]]:
        """Initialize regime-specific calibration configurations."""
        
        configs = {}
        
        # Crisis regime - very conservative
        configs[MarketRegime.CRISIS] = {
            'confidence_threshold': 0.8,
            'temperature_multiplier': 1.8,
            'ensemble_weights': {
                'temperature': 0.1,
                'platt': 0.1,
                'isotonic': 0.4,
                'beta': 0.2,
                'histogram': 0.2
            },
            'uncertainty_penalty': 0.3
        }
        
        # Volatile regime - moderately conservative
        configs[MarketRegime.VOLATILE] = {
            'confidence_threshold': 0.75,
            'temperature_multiplier': 1.4,
            'ensemble_weights': {
                'temperature': 0.15,
                'platt': 0.15,
                'isotonic': 0.35,
                'beta': 0.2,
                'histogram': 0.15
            },
            'uncertainty_penalty': 0.2
        }
        
        # Trending regime - standard settings
        configs[MarketRegime.TRENDING] = {
            'confidence_threshold': 0.65,
            'temperature_multiplier': 1.0,
            'ensemble_weights': {
                'temperature': 0.25,
                'platt': 0.2,
                'isotonic': 0.3,
                'beta': 0.15,
                'histogram': 0.1
            },
            'uncertainty_penalty': 0.1
        }
        
        # Ranging regime - slightly more aggressive
        configs[MarketRegime.RANGING] = {
            'confidence_threshold': 0.6,
            'temperature_multiplier': 0.9,
            'ensemble_weights': {
                'temperature': 0.3,
                'platt': 0.25,
                'isotonic': 0.25,
                'beta': 0.1,
                'histogram': 0.1
            },
            'uncertainty_penalty': 0.05
        }
        
        # Transitioning regime - very conservative
        configs[MarketRegime.TRANSITIONING] = {
            'confidence_threshold': 0.85,
            'temperature_multiplier': 1.6,
            'ensemble_weights': {
                'temperature': 0.1,
                'platt': 0.15,
                'isotonic': 0.4,
                'beta': 0.25,
                'histogram': 0.1
            },
            'uncertainty_penalty': 0.25
        }
        
        # Recovery regime - moderately aggressive
        configs[MarketRegime.RECOVERY] = {
            'confidence_threshold': 0.7,
            'temperature_multiplier': 1.1,
            'ensemble_weights': {
                'temperature': 0.2,
                'platt': 0.2,
                'isotonic': 0.3,
                'beta': 0.15,
                'histogram': 0.15
            },
            'uncertainty_penalty': 0.15
        }
        
        return configs
        
    def calibrate_with_regime_awareness(
        self,
        raw_probabilities: torch.Tensor,
        uncertainty_metrics: Dict[str, float],
        market_context: Dict[str, Any],
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform regime-aware uncertainty calibration.
        
        Args:
            raw_probabilities: Raw probability predictions
            uncertainty_metrics: Uncertainty decomposition metrics
            market_context: Current market conditions
            execution_context: Execution-specific context
            
        Returns:
            Tuple of (calibrated_probabilities, calibration_info)
        """
        
        with self.calibration_lock:
            # Detect current regime
            detected_regime, regime_conf = self.regime_detector.detect_regime(market_context)
            
            # Handle regime transitions
            transition_info = self.transition_handler.handle_transition(
                self.current_regime,
                detected_regime,
                regime_conf
            )
            
            # Update current regime
            self.current_regime = detected_regime
            self.regime_confidence = regime_conf
            
            # Get regime-specific calibration config
            regime_config = self.regime_calibration_configs[detected_regime]
            
            # Apply regime-specific adjustments to enhanced calibrator
            self._apply_regime_adjustments(regime_config)
            
            # Perform calibration with regime context
            calibrated_probs, calibration_info = self.enhanced_calibrator.calibrate_uncertainty(
                raw_probabilities,
                uncertainty_metrics,
                market_context,
                execution_context
            )
            
            # Add regime-specific information
            calibration_info.update({
                'regime': detected_regime.value,
                'regime_confidence': regime_conf,
                'regime_config': regime_config,
                'transition_info': transition_info,
                'regime_stability': self.regime_stability
            })
            
            return calibrated_probs, calibration_info
            
    def _apply_regime_adjustments(self, regime_config: Dict[str, Any]):
        """Apply regime-specific adjustments to the enhanced calibrator."""
        
        # Update confidence threshold
        if 'confidence_threshold' in regime_config:
            self.enhanced_calibrator.adaptive_thresholds['base_threshold'] = \
                regime_config['confidence_threshold']
                
        # Update ensemble weights
        if 'ensemble_weights' in regime_config:
            for method_name, weight in regime_config['ensemble_weights'].items():
                try:
                    from .enhanced_uncertainty_calibration import CalibrationMethod
                    method = CalibrationMethod(method_name)
                    self.enhanced_calibrator.ensemble_weights[method] = weight
                except ValueError:
                    logger.warning(f"Unknown calibration method: {method_name}")
                    
        # Update temperature multiplier
        if 'temperature_multiplier' in regime_config:
            temp_multiplier = regime_config['temperature_multiplier']
            # Apply to temperature calibrator
            temp_calibrator = self.enhanced_calibrator.calibration_methods.get(
                CalibrationMethod.TEMPERATURE
            )
            if temp_calibrator:
                temp_calibrator.temperature *= temp_multiplier
                
    def add_regime_outcome(
        self,
        predicted_probability: float,
        actual_outcome: bool,
        confidence_score: float,
        uncertainty_metrics: Dict[str, float],
        market_context: Dict[str, Any],
        execution_context: Dict[str, Any]
    ):
        """
        Add execution outcome with regime information for learning.
        
        Args:
            predicted_probability: Original predicted probability
            actual_outcome: Whether the prediction was correct
            confidence_score: Confidence score at time of prediction
            uncertainty_metrics: Uncertainty decomposition
            market_context: Market conditions during prediction
            execution_context: Execution-specific context
        """
        
        # Detect regime for this outcome
        regime, regime_conf = self.regime_detector.detect_regime(market_context)
        
        # Update regime performance tracking
        self._update_regime_performance(regime, actual_outcome, confidence_score)
        
        # Add to enhanced calibrator
        self.enhanced_calibrator.add_execution_outcome(
            predicted_probability,
            actual_outcome,
            confidence_score,
            uncertainty_metrics,
            market_context,
            execution_context
        )
        
    def _update_regime_performance(
        self,
        regime: MarketRegime,
        actual_outcome: bool,
        confidence_score: float
    ):
        """Update performance tracking for a specific regime."""
        
        perf = self.regime_performance[regime]
        
        # Update counters
        perf['total_decisions'] += 1
        if actual_outcome:
            perf['correct_decisions'] += 1
            
        # Update running average confidence
        total_decisions = perf['total_decisions']
        current_avg = perf['avg_confidence']
        
        perf['avg_confidence'] = (
            (current_avg * (total_decisions - 1) + confidence_score) / total_decisions
        )
        
        # Update calibration quality (placeholder)
        accuracy = perf['correct_decisions'] / total_decisions
        perf['calibration_quality'] = accuracy  # Simplified metric
        
    def get_regime_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive regime diagnostics."""
        
        diagnostics = {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_stability': self.regime_stability,
            'regime_performance': {
                regime.value: dict(performance)
                for regime, performance in self.regime_performance.items()
            },
            'regime_history': [
                {
                    'regime': r['regime'].value,
                    'confidence': r['confidence'],
                    'timestamp': r['timestamp']
                }
                for r in list(self.regime_detector.regime_history)[-10:]
            ],
            'calibration_diagnostics': self.enhanced_calibrator.get_calibration_diagnostics()
        }
        
        return diagnostics
        
    def start_regime_monitoring(self):
        """Start regime monitoring and calibration."""
        self.enhanced_calibrator.start_real_time_calibration()
        logger.info("Regime-aware calibration monitoring started")
        
    def stop_regime_monitoring(self):
        """Stop regime monitoring and calibration."""
        self.enhanced_calibrator.stop_real_time_calibration()
        logger.info("Regime-aware calibration monitoring stopped")


class RegimeTransitionHandler:
    """Handles regime transitions and calibration adjustments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transition_cooldown = config.get('transition_cooldown', 30)  # seconds
        self.min_confidence_for_transition = config.get('min_confidence_for_transition', 0.6)
        
        # Track transitions
        self.transition_history = deque(maxlen=100)
        self.last_transition_time = 0
        
    def handle_transition(
        self,
        current_regime: MarketRegime,
        detected_regime: MarketRegime,
        confidence: float
    ) -> Optional[Dict[str, Any]]:
        """Handle regime transition with appropriate calibration adjustments."""
        
        # Check if regime has changed
        if current_regime == detected_regime:
            return None
            
        # Check confidence threshold
        if confidence < self.min_confidence_for_transition:
            return {
                'transition_blocked': True,
                'reason': 'Low confidence',
                'confidence': confidence
            }
            
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_transition_time < self.transition_cooldown:
            return {
                'transition_blocked': True,
                'reason': 'Cooldown period',
                'time_remaining': self.transition_cooldown - (current_time - self.last_transition_time)
            }
            
        # Allow transition
        self.last_transition_time = current_time
        
        transition_info = {
            'transition_allowed': True,
            'from_regime': current_regime.value,
            'to_regime': detected_regime.value,
            'confidence': confidence,
            'timestamp': current_time
        }
        
        # Store transition
        self.transition_history.append(transition_info)
        
        logger.info(f"Regime transition: {current_regime.value} -> {detected_regime.value}")
        
        return transition_info