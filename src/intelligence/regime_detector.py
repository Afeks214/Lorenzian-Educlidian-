"""
Advanced Market Regime Detection System for Contextual Intelligence

This module implements a sophisticated market regime detection system that uses multiple
indicators to classify market conditions and provides regime-aware context for intelligent
decision-making in the trading system.

Key Features:
- Multi-indicator regime classification (volatility, momentum, volume, MMD analysis)
- Confidence scoring for regime detection certainty
- Regime transition probability tracking
- Real-time regime context for reward system adaptation

Author: Agent Gamma - The Contextual Judge
Version: 1.0 - Regime-Aware Intelligence
"""

import numpy as np
import torch
from enum import Enum
from typing import Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications for context-aware decision making."""
    HIGH_VOLATILITY = "high_volatility"      # σ > 2.5, high uncertainty
    LOW_VOLATILITY = "low_volatility"        # σ < 0.8, ranging/quiet market  
    CRISIS = "crisis"                        # σ > 3.5, rapid price moves, high volume
    RECOVERY = "recovery"                    # Emerging from crisis, stabilizing
    BULL_TREND = "bull_trend"               # Strong upward momentum
    BEAR_TREND = "bear_trend"               # Strong downward momentum
    SIDEWAYS = "sideways"                   # No clear trend, low momentum

@dataclass
class RegimeAnalysis:
    """Complete regime analysis with confidence and characteristics."""
    regime: MarketRegime
    confidence: float                        # Confidence in regime classification [0,1]
    volatility: float                       # Current volatility measure
    momentum: float                         # Momentum strength
    volume_profile: float                   # Volume characteristics
    regime_duration: int                    # How long in current regime (periods)
    transition_probability: Dict[MarketRegime, float]  # Transition probabilities

class RegimeDetector:
    """
    Advanced market regime detection using multiple indicators.
    
    Combines volatility, momentum, volume, and MMD analysis for robust
    regime classification with confidence scoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Regime thresholds (configurable)
        self.volatility_thresholds = {
            'crisis': config.get('crisis_volatility', 3.5),
            'high_vol': config.get('high_volatility', 2.5), 
            'low_vol': config.get('low_volatility', 0.8)
        }
        
        self.momentum_thresholds = {
            'strong_trend': config.get('strong_momentum', 0.05),
            'weak_trend': config.get('weak_momentum', 0.02)
        }
        
        # Regime history tracking
        self.regime_history = []
        self.current_regime = None
        self.regime_start_time = None
        self.regime_duration_counter = 0
        
        # Transition probability matrix
        self.transition_matrix = self._initialize_transition_matrix()
        
        # Performance tracking
        self.detection_count = 0
        self.confidence_history = []
        
        self.logger.info(
            f"RegimeDetector initialized with thresholds: "
            f"vol_crisis={self.volatility_thresholds['crisis']}, "
            f"vol_high={self.volatility_thresholds['high_vol']}, "
            f"vol_low={self.volatility_thresholds['low_vol']}"
        )
        
    def _initialize_transition_matrix(self) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
        """Initialize regime transition probability matrix with market-realistic values."""
        transitions = {}
        
        for from_regime in MarketRegime:
            transitions[from_regime] = {}
            for to_regime in MarketRegime:
                if from_regime == to_regime:
                    # High regime persistence
                    transitions[from_regime][to_regime] = 0.75
                else:
                    # Base transition probability
                    transitions[from_regime][to_regime] = 0.04
        
        # Market-specific transition probabilities
        transitions[MarketRegime.BULL_TREND][MarketRegime.SIDEWAYS] = 0.15
        transitions[MarketRegime.BEAR_TREND][MarketRegime.SIDEWAYS] = 0.15
        transitions[MarketRegime.SIDEWAYS][MarketRegime.BULL_TREND] = 0.12
        transitions[MarketRegime.SIDEWAYS][MarketRegime.BEAR_TREND] = 0.12
        transitions[MarketRegime.CRISIS][MarketRegime.RECOVERY] = 0.20
        transitions[MarketRegime.RECOVERY][MarketRegime.BULL_TREND] = 0.18
        transitions[MarketRegime.HIGH_VOLATILITY][MarketRegime.CRISIS] = 0.10
        transitions[MarketRegime.LOW_VOLATILITY][MarketRegime.SIDEWAYS] = 0.20
        
        return transitions
        
    def detect_regime(self, market_context: Dict[str, Any]) -> RegimeAnalysis:
        """
        Detect current market regime based on multiple indicators.
        
        Args:
            market_context: Dictionary containing market indicators
                Expected keys:
                - volatility_30: 30-period volatility
                - momentum_20: 20-period momentum
                - momentum_50: 50-period momentum  
                - volume_ratio: Volume relative to average
                - mmd_score: Maximum Mean Discrepancy score
                
        Returns:
            RegimeAnalysis with regime classification and confidence
        """
        try:
            # Extract market indicators with defaults
            volatility_30 = market_context.get('volatility_30', 1.0)
            momentum_20 = market_context.get('momentum_20', 0.0)
            momentum_50 = market_context.get('momentum_50', 0.0)
            volume_ratio = market_context.get('volume_ratio', 1.0)
            mmd_score = market_context.get('mmd_score', 0.0)
            
            # Validate inputs
            volatility_30 = max(0.01, min(10.0, volatility_30))
            momentum_20 = max(-1.0, min(1.0, momentum_20))
            momentum_50 = max(-1.0, min(1.0, momentum_50))
            volume_ratio = max(0.1, min(20.0, volume_ratio))
            mmd_score = max(0.0, min(5.0, mmd_score))
            
            # Primary regime classification based on volatility
            regime = self._classify_by_volatility(volatility_30)
            
            # Refine classification with momentum analysis
            regime = self._refine_with_momentum(regime, momentum_20, momentum_50)
            
            # Further refinement with volume and MMD
            regime = self._refine_with_volume_mmd(regime, volume_ratio, mmd_score)
            
            # Calculate confidence based on indicator alignment
            confidence = self._calculate_confidence(
                volatility_30, momentum_20, momentum_50, volume_ratio, mmd_score, regime
            )
            
            # Calculate transition probabilities
            transition_probs = self._calculate_transition_probabilities(regime, market_context)
            
            # Update regime tracking
            regime_duration = self._update_regime_tracking(regime)
            
            regime_analysis = RegimeAnalysis(
                regime=regime,
                confidence=confidence,
                volatility=volatility_30,
                momentum=momentum_20,
                volume_profile=volume_ratio,
                regime_duration=regime_duration,
                transition_probability=transition_probs
            )
            
            # Track performance
            self.detection_count += 1
            self.confidence_history.append(confidence)
            if len(self.confidence_history) > 1000:
                self.confidence_history.pop(0)
            
            self.logger.debug(
                f"Regime detected: {regime.value}, confidence: {confidence:.3f}, "
                f"volatility: {volatility_30:.3f}, momentum: {momentum_20:.3f}"
            )
            
            return regime_analysis
            
        except Exception as e:
            self.logger.error(f"Error in regime detection: {e}")
            # Return default regime
            return RegimeAnalysis(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                volatility=1.0,
                momentum=0.0,
                volume_profile=1.0,
                regime_duration=1,
                transition_probability=self._get_uniform_transition_probs()
            )
    
    def _classify_by_volatility(self, volatility: float) -> MarketRegime:
        """Primary classification based on volatility."""
        if volatility > self.volatility_thresholds['crisis']:
            return MarketRegime.CRISIS
        elif volatility > self.volatility_thresholds['high_vol']:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.volatility_thresholds['low_vol']:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.SIDEWAYS
    
    def _refine_with_momentum(self, regime: MarketRegime, momentum_20: float, momentum_50: float) -> MarketRegime:
        """Refine regime classification with momentum analysis."""
        avg_momentum = (momentum_20 + momentum_50) / 2
        momentum_strength = abs(avg_momentum)
        
        # Strong trends override volatility-based classification (except crisis)
        if regime != MarketRegime.CRISIS:
            if avg_momentum > self.momentum_thresholds['strong_trend']:
                return MarketRegime.BULL_TREND
            elif avg_momentum < -self.momentum_thresholds['strong_trend']:
                return MarketRegime.BEAR_TREND
        
        # Recovery detection (transitioning from crisis/high vol to stability)
        if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
            if momentum_strength < self.momentum_thresholds['weak_trend']:
                return MarketRegime.RECOVERY
                
        return regime
    
    def _refine_with_volume_mmd(self, regime: MarketRegime, volume_ratio: float, mmd_score: float) -> MarketRegime:
        """Further refinement with volume and MMD analysis."""
        # High volume + high MMD suggests regime transition or crisis
        if volume_ratio > 2.0 and mmd_score > 0.5:
            if regime not in [MarketRegime.CRISIS, MarketRegime.RECOVERY]:
                return MarketRegime.HIGH_VOLATILITY
        
        # Very high volume with extreme MMD indicates crisis
        if volume_ratio > 5.0 and mmd_score > 1.0:
            return MarketRegime.CRISIS
            
        # Low volume and low MMD in non-trending regimes suggests sideways
        if volume_ratio < 0.8 and mmd_score < 0.2:
            if regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY]:
                return MarketRegime.SIDEWAYS
        
        return regime
    
    def _calculate_confidence(self, volatility: float, momentum_20: float, momentum_50: float, 
                            volume_ratio: float, mmd_score: float, regime: MarketRegime) -> float:
        """Calculate confidence in regime classification based on indicator alignment."""
        # Base confidence from volatility clarity
        vol_confidence = self._volatility_confidence(volatility, regime)
        
        # Momentum consistency
        momentum_consistency = 1.0 - min(1.0, abs(momentum_20 - momentum_50))
        
        # Volume confirmation
        volume_confirmation = self._volume_confirmation(volume_ratio, regime)
        
        # MMD stability (lower MMD = higher confidence in current regime for stable regimes)
        mmd_confidence = self._mmd_confidence(mmd_score, regime)
        
        # Weighted average
        total_confidence = (
            0.4 * vol_confidence +
            0.3 * momentum_consistency + 
            0.2 * volume_confirmation +
            0.1 * mmd_confidence
        )
        
        return np.clip(total_confidence, 0.1, 0.95)
    
    def _volatility_confidence(self, volatility: float, regime: MarketRegime) -> float:
        """Calculate confidence based on how clearly volatility indicates the regime."""
        if regime == MarketRegime.CRISIS:
            if volatility > 4.0:
                return 0.95
            elif volatility > 3.5:
                return 0.85
            else:
                return 0.6
                
        elif regime == MarketRegime.HIGH_VOLATILITY:
            if 2.0 < volatility < 4.0:
                return 0.8
            else:
                return 0.6
                
        elif regime == MarketRegime.LOW_VOLATILITY:
            if volatility < 0.5:
                return 0.9
            elif volatility < 0.8:
                return 0.75
            else:
                return 0.5
                
        else:  # Other regimes
            return 0.7
    
    def _volume_confirmation(self, volume_ratio: float, regime: MarketRegime) -> float:
        """Calculate volume-based confidence for regime."""
        if regime == MarketRegime.CRISIS:
            return min(1.0, volume_ratio / 3.0)  # Crisis should have high volume
        elif regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            return min(1.0, volume_ratio / 2.0)  # Trends should have elevated volume
        elif regime == MarketRegime.SIDEWAYS:
            return max(0.3, 1.0 - abs(volume_ratio - 1.0))  # Sideways should have normal volume
        else:
            return 0.7  # Default confidence
    
    def _mmd_confidence(self, mmd_score: float, regime: MarketRegime) -> float:
        """Calculate MMD-based confidence for regime."""
        if regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND, MarketRegime.CRISIS]:
            # Trending/crisis regimes should have higher MMD
            return min(1.0, mmd_score / 0.5)
        else:
            # Stable regimes should have lower MMD
            return max(0.2, 1.0 - mmd_score)
    
    def _calculate_transition_probabilities(self, current_regime: MarketRegime, market_context: Dict[str, Any]) -> Dict[MarketRegime, float]:
        """Calculate transition probabilities to other regimes."""
        if self.current_regime is None:
            # First detection - use base probabilities
            return self.transition_matrix.get(current_regime, self._get_uniform_transition_probs())
        
        # Get base transition probabilities
        base_probs = self.transition_matrix[current_regime].copy()
        
        # Adjust based on current market conditions
        volatility = market_context.get('volatility_30', 1.0)
        volume_ratio = market_context.get('volume_ratio', 1.0)
        
        # High volatility increases crisis probability
        if volatility > 2.0:
            base_probs[MarketRegime.CRISIS] *= 2.0
            base_probs[MarketRegime.HIGH_VOLATILITY] *= 1.5
        
        # High volume increases trend probability
        if volume_ratio > 1.5:
            base_probs[MarketRegime.BULL_TREND] *= 1.3
            base_probs[MarketRegime.BEAR_TREND] *= 1.3
        
        # Normalize probabilities
        total_prob = sum(base_probs.values())
        if total_prob > 0:
            for regime in base_probs:
                base_probs[regime] /= total_prob
        
        return base_probs
    
    def _get_uniform_transition_probs(self) -> Dict[MarketRegime, float]:
        """Get uniform transition probabilities."""
        prob = 1.0 / len(MarketRegime)
        return {regime: prob for regime in MarketRegime}
    
    def _update_regime_tracking(self, regime: MarketRegime) -> int:
        """Update regime tracking and return duration."""
        if self.current_regime != regime:
            # Regime change detected
            if self.current_regime is not None:
                self.regime_history.append({
                    'regime': self.current_regime,
                    'duration': self.regime_duration_counter,
                    'end_time': time.time()
                })
            
            self.current_regime = regime
            self.regime_start_time = time.time()
            self.regime_duration_counter = 1
            
            # Keep history manageable
            if len(self.regime_history) > 100:
                self.regime_history.pop(0)
                
            self.logger.info(f"Regime transition: -> {regime.value}")
        else:
            # Same regime, increment duration
            self.regime_duration_counter += 1
        
        return self.regime_duration_counter
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime detection statistics."""
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.5
        
        # Calculate regime persistence
        regime_durations = [r['duration'] for r in self.regime_history]
        avg_duration = np.mean(regime_durations) if regime_durations else 1
        
        # Calculate regime distribution
        regime_counts = {}
        for regime in MarketRegime:
            regime_counts[regime.value] = sum(1 for r in self.regime_history if r['regime'] == regime)
        
        return {
            'total_detections': self.detection_count,
            'average_confidence': avg_confidence,
            'current_regime': self.current_regime.value if self.current_regime else 'none',
            'current_duration': self.regime_duration_counter,
            'average_regime_duration': avg_duration,
            'regime_distribution': regime_counts,
            'recent_regimes': [r['regime'].value for r in self.regime_history[-10:]]
        }
    
    def reset_history(self) -> None:
        """Reset regime detection history."""
        self.regime_history.clear()
        self.confidence_history.clear()
        self.current_regime = None
        self.regime_start_time = None
        self.regime_duration_counter = 0
        self.detection_count = 0
        
        self.logger.info("Regime detector history reset")

def create_regime_detector(config: Optional[Dict[str, Any]] = None) -> RegimeDetector:
    """
    Factory function to create a RegimeDetector with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RegimeDetector instance
    """
    default_config = {
        'crisis_volatility': 3.5,
        'high_volatility': 2.5,
        'low_volatility': 0.8,
        'strong_momentum': 0.05,
        'weak_momentum': 0.02
    }
    
    if config:
        default_config.update(config)
    
    return RegimeDetector(default_config)