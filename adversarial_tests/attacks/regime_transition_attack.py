#!/usr/bin/env python3
"""
🚨 AGENT GAMMA MISSION - REGIME TRANSITION ATTACK MODULE
Advanced MARL Attack Development: Regime Detection Exploitation

This module implements sophisticated attacks targeting regime detection
and transition mechanisms in Strategic and Tactical MARL systems:
- False regime signal generation
- Regime transition timing attacks
- Market regime classification manipulation
- Regime detection agent vulnerability exploitation

Key Attack Vectors:
1. False Bull Market Signals: Create fake bullish regime indicators
2. False Bear Market Signals: Create fake bearish regime indicators
3. Regime Transition Confusion: Inject conflicting regime signals
4. Volatility Regime Manipulation: Manipulate volatility-based regimes
5. MMD Score Poisoning: Poison Maximum Mean Discrepancy calculations

MISSION OBJECTIVE: Achieve >80% attack success rate against regime detection defenses
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
from collections import deque
import scipy.stats as stats
from sklearn.mixture import GaussianMixture

# Attack Result Tracking
@dataclass
class RegimeAttackResult:
    """Results from a regime transition attack attempt."""
    attack_type: str
    success: bool
    confidence: float
    regime_disruption_score: float
    false_regime_strength: float
    transition_confusion_rate: float
    original_regime_detection: Dict[str, Any]
    attacked_regime_detection: Dict[str, Any]
    affected_regime_indicators: List[str]
    execution_time_ms: float
    attack_payload: Dict[str, Any]
    timestamp: datetime

class RegimeAttackType(Enum):
    """Types of regime transition attacks."""
    FALSE_BULL_SIGNAL = "false_bull_signal"
    FALSE_BEAR_SIGNAL = "false_bear_signal"
    TRANSITION_CONFUSION = "transition_confusion"
    VOLATILITY_MANIPULATION = "volatility_manipulation"
    MMD_POISONING = "mmd_poisoning"

class RegimeTransitionAttacker:
    """
    Advanced Regime Transition Attack System.
    
    This system implements sophisticated attacks targeting regime detection
    and transition mechanisms in MARL systems.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the Regime Transition Attacker.
        
        Args:
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.device = device
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Attack history and analytics
        self.attack_history = []
        self.success_rates = {attack_type: 0.0 for attack_type in RegimeAttackType}
        self.regime_metrics = {
            'total_attempts': 0,
            'successful_attacks': 0,
            'avg_disruption_score': 0.0,
            'max_disruption_score': 0.0,
            'regimes_compromised': set()
        }
        
        # Regime detection parameters
        self.regime_threshold = 0.3
        self.transition_window = 10
        self.volatility_regimes = ['low', 'medium', 'high']
        self.market_regimes = ['bull', 'bear', 'sideways', 'transitional']
        
        # Historical regime data for pattern learning
        self.regime_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=200)
        
        # Regime attack parameters
        self.false_signal_strength = 0.8
        self.confusion_factor = 0.6
        self.manipulation_persistence = 5  # Number of periods to maintain attack
        
        self.logger.info(f"RegimeTransitionAttacker initialized: device={device}")
    
    def generate_false_bull_signal_attack(
        self,
        market_data: np.ndarray,
        regime_indicators: Dict[str, Any],
        signal_strength: float = None
    ) -> Tuple[Dict[str, Any], RegimeAttackResult]:
        """
        🎯 ATTACK 1: FALSE BULL MARKET SIGNAL
        
        Creates false bullish regime indicators to mislead the regime detection
        system into believing a bull market is beginning.
        
        Args:
            market_data: Market data matrix [sequence_length, features]
            regime_indicators: Current regime indicators
            signal_strength: Strength of false signal
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        if signal_strength is None:
            signal_strength = self.false_signal_strength
        
        # Store original data and indicators
        original_data = market_data.copy()
        original_indicators = regime_indicators.copy()
        
        # Create attacked versions
        attacked_data = market_data.copy()
        attacked_indicators = regime_indicators.copy()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': RegimeAttackType.FALSE_BULL_SIGNAL.value,
            'signal_strength': signal_strength,
            'bull_signal_injections': {},
            'momentum_manipulations': {},
            'trend_fabrications': {}
        }
        
        sequence_length, n_features = market_data.shape
        
        # Inject false bullish momentum
        if n_features > 9:  # Assuming momentum columns exist
            momentum_20_col = 9
            momentum_50_col = 10
            
            # Create artificial bullish momentum
            for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                # Strong positive momentum
                attacked_data[i, momentum_20_col] = abs(attacked_data[i, momentum_20_col]) * signal_strength + 0.02
                attacked_data[i, momentum_50_col] = abs(attacked_data[i, momentum_50_col]) * signal_strength + 0.015
            
            attack_payload['momentum_manipulations'] = {
                'momentum_20_boost': signal_strength * 0.02,
                'momentum_50_boost': signal_strength * 0.015,
                'persistence_periods': self.manipulation_persistence
            }
        
        # Inject false bullish trend
        if n_features > 0:  # Price column
            price_col = 0
            
            # Create artificial upward trend
            for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                trend_boost = signal_strength * 0.01 * (i - (sequence_length - self.manipulation_persistence))
                attacked_data[i, price_col] = attacked_data[i, price_col] * (1 + trend_boost)
            
            attack_payload['trend_fabrications'] = {
                'trend_boost_per_period': signal_strength * 0.01,
                'cumulative_trend_boost': signal_strength * 0.01 * self.manipulation_persistence
            }
        
        # Manipulate volatility to appear controlled (bullish characteristic)
        if n_features > 11:  # Volatility column
            vol_col = 11
            
            # Reduce volatility to create "controlled bull market" appearance
            for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                attacked_data[i, vol_col] = attacked_data[i, vol_col] * (1 - signal_strength * 0.3)
            
            attack_payload['volatility_manipulation'] = {
                'volatility_reduction': signal_strength * 0.3,
                'controlled_bull_appearance': True
            }
        
        # Modify regime indicators
        attacked_indicators['market_regime'] = 'bull'
        attacked_indicators['regime_confidence'] = 0.95
        attacked_indicators['volatility_regime'] = 'low'
        attacked_indicators['momentum_regime'] = 'strong_aligned'
        
        # Create false MMD score (low MMD = stable regime)
        attacked_indicators['mmd_score'] = 0.02  # Very low MMD
        
        # Inject false bull signal indicators
        attack_payload['bull_signal_injections'] = {
            'market_regime': 'bull',
            'regime_confidence': 0.95,
            'volatility_regime': 'low',
            'momentum_regime': 'strong_aligned',
            'mmd_score': 0.02,
            'bull_signal_strength': signal_strength
        }
        
        # Calculate regime disruption score
        regime_disruption = self._calculate_regime_disruption_score(
            original_indicators, attacked_indicators, 'bull'
        )
        
        # Calculate false signal strength
        false_signal_strength = self._calculate_false_signal_strength(
            original_data, attacked_data, 'bull'
        )
        
        # Determine attack success
        success = regime_disruption > self.regime_threshold and false_signal_strength > 0.5
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = RegimeAttackResult(
            attack_type=RegimeAttackType.FALSE_BULL_SIGNAL.value,
            success=success,
            confidence=regime_disruption,
            regime_disruption_score=regime_disruption,
            false_regime_strength=false_signal_strength,
            transition_confusion_rate=0.0,  # Not applicable for this attack
            original_regime_detection=original_indicators,
            attacked_regime_detection=attacked_indicators,
            affected_regime_indicators=['momentum_20', 'momentum_50', 'volatility_30', 'mmd_score'],
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"False bull signal attack executed: "
            f"success={success}, regime_disruption={regime_disruption:.3f}, "
            f"false_signal_strength={false_signal_strength:.3f}"
        )
        
        # Return attacked data and indicators
        attack_payload['attacked_data'] = attacked_data
        attack_payload['attacked_indicators'] = attacked_indicators
        
        return attack_payload, attack_result
    
    def generate_false_bear_signal_attack(
        self,
        market_data: np.ndarray,
        regime_indicators: Dict[str, Any],
        signal_strength: float = None
    ) -> Tuple[Dict[str, Any], RegimeAttackResult]:
        """
        🎯 ATTACK 2: FALSE BEAR MARKET SIGNAL
        
        Creates false bearish regime indicators to mislead the regime detection
        system into believing a bear market is beginning.
        
        Args:
            market_data: Market data matrix [sequence_length, features]
            regime_indicators: Current regime indicators
            signal_strength: Strength of false signal
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        if signal_strength is None:
            signal_strength = self.false_signal_strength
        
        # Store original data and indicators
        original_data = market_data.copy()
        original_indicators = regime_indicators.copy()
        
        # Create attacked versions
        attacked_data = market_data.copy()
        attacked_indicators = regime_indicators.copy()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': RegimeAttackType.FALSE_BEAR_SIGNAL.value,
            'signal_strength': signal_strength,
            'bear_signal_injections': {},
            'momentum_reversals': {},
            'volatility_spikes': {}
        }
        
        sequence_length, n_features = market_data.shape
        
        # Inject false bearish momentum
        if n_features > 9:  # Assuming momentum columns exist
            momentum_20_col = 9
            momentum_50_col = 10
            
            # Create artificial bearish momentum
            for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                # Strong negative momentum
                attacked_data[i, momentum_20_col] = -abs(attacked_data[i, momentum_20_col]) * signal_strength - 0.02
                attacked_data[i, momentum_50_col] = -abs(attacked_data[i, momentum_50_col]) * signal_strength - 0.015
            
            attack_payload['momentum_reversals'] = {
                'momentum_20_reversal': -signal_strength * 0.02,
                'momentum_50_reversal': -signal_strength * 0.015,
                'persistence_periods': self.manipulation_persistence
            }
        
        # Inject false bearish trend
        if n_features > 0:  # Price column
            price_col = 0
            
            # Create artificial downward trend
            for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                trend_decline = signal_strength * 0.01 * (i - (sequence_length - self.manipulation_persistence))
                attacked_data[i, price_col] = attacked_data[i, price_col] * (1 - trend_decline)
            
            attack_payload['trend_fabrications'] = {
                'trend_decline_per_period': -signal_strength * 0.01,
                'cumulative_trend_decline': -signal_strength * 0.01 * self.manipulation_persistence
            }
        
        # Manipulate volatility to appear elevated (bearish characteristic)
        if n_features > 11:  # Volatility column
            vol_col = 11
            
            # Increase volatility to create "panic bear market" appearance
            for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                attacked_data[i, vol_col] = attacked_data[i, vol_col] * (1 + signal_strength * 0.8)
            
            attack_payload['volatility_spikes'] = {
                'volatility_increase': signal_strength * 0.8,
                'panic_bear_appearance': True
            }
        
        # Modify regime indicators
        attacked_indicators['market_regime'] = 'bear'
        attacked_indicators['regime_confidence'] = 0.92
        attacked_indicators['volatility_regime'] = 'high'
        attacked_indicators['momentum_regime'] = 'strong_aligned'  # Aligned but negative
        
        # Create false MMD score (high MMD = regime transition)
        attacked_indicators['mmd_score'] = 0.8  # High MMD indicating regime change
        
        # Inject false bear signal indicators
        attack_payload['bear_signal_injections'] = {
            'market_regime': 'bear',
            'regime_confidence': 0.92,
            'volatility_regime': 'high',
            'momentum_regime': 'strong_aligned',
            'mmd_score': 0.8,
            'bear_signal_strength': signal_strength
        }
        
        # Calculate regime disruption score
        regime_disruption = self._calculate_regime_disruption_score(
            original_indicators, attacked_indicators, 'bear'
        )
        
        # Calculate false signal strength
        false_signal_strength = self._calculate_false_signal_strength(
            original_data, attacked_data, 'bear'
        )
        
        # Determine attack success
        success = regime_disruption > self.regime_threshold and false_signal_strength > 0.5
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = RegimeAttackResult(
            attack_type=RegimeAttackType.FALSE_BEAR_SIGNAL.value,
            success=success,
            confidence=regime_disruption,
            regime_disruption_score=regime_disruption,
            false_regime_strength=false_signal_strength,
            transition_confusion_rate=0.0,  # Not applicable for this attack
            original_regime_detection=original_indicators,
            attacked_regime_detection=attacked_indicators,
            affected_regime_indicators=['momentum_20', 'momentum_50', 'volatility_30', 'mmd_score'],
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"False bear signal attack executed: "
            f"success={success}, regime_disruption={regime_disruption:.3f}, "
            f"false_signal_strength={false_signal_strength:.3f}"
        )
        
        # Return attacked data and indicators
        attack_payload['attacked_data'] = attacked_data
        attack_payload['attacked_indicators'] = attacked_indicators
        
        return attack_payload, attack_result
    
    def generate_transition_confusion_attack(
        self,
        market_data: np.ndarray,
        regime_indicators: Dict[str, Any],
        confusion_factor: float = None
    ) -> Tuple[Dict[str, Any], RegimeAttackResult]:
        """
        🎯 ATTACK 3: REGIME TRANSITION CONFUSION
        
        Injects conflicting regime signals to confuse the regime detection
        system and create uncertainty about the current market regime.
        
        Args:
            market_data: Market data matrix [sequence_length, features]
            regime_indicators: Current regime indicators
            confusion_factor: Strength of confusion injection
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        if confusion_factor is None:
            confusion_factor = self.confusion_factor
        
        # Store original data and indicators
        original_data = market_data.copy()
        original_indicators = regime_indicators.copy()
        
        # Create attacked versions
        attacked_data = market_data.copy()
        attacked_indicators = regime_indicators.copy()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': RegimeAttackType.TRANSITION_CONFUSION.value,
            'confusion_factor': confusion_factor,
            'conflicting_signals': {},
            'regime_oscillations': {},
            'uncertainty_injections': {}
        }
        
        sequence_length, n_features = market_data.shape
        
        # Create conflicting momentum signals
        if n_features > 9:
            momentum_20_col = 9
            momentum_50_col = 10
            
            # Make momentum signals conflict with each other
            for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                # Oscillating conflict pattern
                conflict_phase = (i % 4) / 4.0 * 2 * np.pi
                
                # Momentum 20 suggests bull, momentum 50 suggests bear (or vice versa)
                if np.sin(conflict_phase) > 0:
                    attacked_data[i, momentum_20_col] = abs(attacked_data[i, momentum_20_col]) * confusion_factor
                    attacked_data[i, momentum_50_col] = -abs(attacked_data[i, momentum_50_col]) * confusion_factor
                else:
                    attacked_data[i, momentum_20_col] = -abs(attacked_data[i, momentum_20_col]) * confusion_factor
                    attacked_data[i, momentum_50_col] = abs(attacked_data[i, momentum_50_col]) * confusion_factor
            
            attack_payload['conflicting_signals']['momentum_conflict'] = {
                'momentum_20_pattern': 'oscillating',
                'momentum_50_pattern': 'inverse_oscillating',
                'conflict_strength': confusion_factor
            }
        
        # Create conflicting volatility signals
        if n_features > 11:
            vol_col = 11
            
            # Alternate between high and low volatility rapidly
            for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                if i % 2 == 0:
                    # High volatility
                    attacked_data[i, vol_col] = attacked_data[i, vol_col] * (1 + confusion_factor)
                else:
                    # Low volatility
                    attacked_data[i, vol_col] = attacked_data[i, vol_col] * (1 - confusion_factor * 0.5)
            
            attack_payload['conflicting_signals']['volatility_conflict'] = {
                'alternating_pattern': True,
                'high_vol_multiplier': (1 + confusion_factor),
                'low_vol_multiplier': (1 - confusion_factor * 0.5)
            }
        
        # Create conflicting volume signals
        if n_features > 12:
            volume_col = 12
            
            # Create volume that conflicts with price movement
            for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                if i > 0:
                    price_change = attacked_data[i, 0] - attacked_data[i-1, 0]
                    
                    # Volume should be opposite to price change (conflicting signal)
                    if price_change > 0:
                        # Price up, volume down (bearish divergence)
                        attacked_data[i, volume_col] = attacked_data[i, volume_col] * (1 - confusion_factor * 0.3)
                    else:
                        # Price down, volume up (bullish divergence)
                        attacked_data[i, volume_col] = attacked_data[i, volume_col] * (1 + confusion_factor * 0.3)
            
            attack_payload['conflicting_signals']['volume_price_divergence'] = {
                'divergence_type': 'inverse_correlation',
                'divergence_strength': confusion_factor * 0.3
            }
        
        # Modify regime indicators to show maximum uncertainty
        attacked_indicators['market_regime'] = 'transitional'
        attacked_indicators['regime_confidence'] = 0.25  # Very low confidence
        attacked_indicators['volatility_regime'] = 'transitional'
        attacked_indicators['momentum_regime'] = 'conflicting'
        
        # Create conflicting MMD score (medium MMD = uncertain regime)
        attacked_indicators['mmd_score'] = 0.45  # Medium MMD indicating uncertainty
        
        # Inject confusion indicators
        attack_payload['uncertainty_injections'] = {
            'market_regime': 'transitional',
            'regime_confidence': 0.25,
            'volatility_regime': 'transitional',
            'momentum_regime': 'conflicting',
            'mmd_score': 0.45,
            'confusion_strength': confusion_factor
        }
        
        # Calculate regime disruption score
        regime_disruption = self._calculate_regime_disruption_score(
            original_indicators, attacked_indicators, 'confusion'
        )
        
        # Calculate transition confusion rate
        transition_confusion = self._calculate_transition_confusion_rate(
            original_data, attacked_data
        )
        
        # Determine attack success
        success = regime_disruption > self.regime_threshold and transition_confusion > 0.4
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = RegimeAttackResult(
            attack_type=RegimeAttackType.TRANSITION_CONFUSION.value,
            success=success,
            confidence=regime_disruption,
            regime_disruption_score=regime_disruption,
            false_regime_strength=0.0,  # Not applicable for confusion attack
            transition_confusion_rate=transition_confusion,
            original_regime_detection=original_indicators,
            attacked_regime_detection=attacked_indicators,
            affected_regime_indicators=['momentum_20', 'momentum_50', 'volatility_30', 'volume_ratio'],
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Transition confusion attack executed: "
            f"success={success}, regime_disruption={regime_disruption:.3f}, "
            f"confusion_rate={transition_confusion:.3f}"
        )
        
        # Return attacked data and indicators
        attack_payload['attacked_data'] = attacked_data
        attack_payload['attacked_indicators'] = attacked_indicators
        
        return attack_payload, attack_result
    
    def generate_volatility_manipulation_attack(
        self,
        market_data: np.ndarray,
        regime_indicators: Dict[str, Any],
        target_volatility_regime: str = 'high'
    ) -> Tuple[Dict[str, Any], RegimeAttackResult]:
        """
        🎯 ATTACK 4: VOLATILITY REGIME MANIPULATION
        
        Manipulates volatility patterns to force the system into a specific
        volatility regime classification.
        
        Args:
            market_data: Market data matrix [sequence_length, features]
            regime_indicators: Current regime indicators
            target_volatility_regime: Target volatility regime
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        # Store original data and indicators
        original_data = market_data.copy()
        original_indicators = regime_indicators.copy()
        
        # Create attacked versions
        attacked_data = market_data.copy()
        attacked_indicators = regime_indicators.copy()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': RegimeAttackType.VOLATILITY_MANIPULATION.value,
            'target_volatility_regime': target_volatility_regime,
            'volatility_manipulations': {},
            'regime_forcing': {}
        }
        
        sequence_length, n_features = market_data.shape
        
        # Manipulate volatility based on target regime
        if n_features > 11:  # Volatility column
            vol_col = 11
            
            if target_volatility_regime == 'high':
                # Force high volatility regime
                for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                    # Amplify volatility significantly
                    attacked_data[i, vol_col] = max(attacked_data[i, vol_col] * 3.0, 0.05)
                
                attack_payload['volatility_manipulations']['high_vol_forcing'] = {
                    'amplification_factor': 3.0,
                    'minimum_volatility': 0.05,
                    'target_regime': 'high'
                }
                
                # Update regime indicators
                attacked_indicators['volatility_regime'] = 'high'
                attacked_indicators['market_regime'] = 'volatile'
            
            elif target_volatility_regime == 'low':
                # Force low volatility regime
                for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                    # Suppress volatility significantly
                    attacked_data[i, vol_col] = min(attacked_data[i, vol_col] * 0.2, 0.005)
                
                attack_payload['volatility_manipulations']['low_vol_forcing'] = {
                    'suppression_factor': 0.2,
                    'maximum_volatility': 0.005,
                    'target_regime': 'low'
                }
                
                # Update regime indicators
                attacked_indicators['volatility_regime'] = 'low'
                attacked_indicators['market_regime'] = 'trending'
            
            elif target_volatility_regime == 'transitional':
                # Force transitional volatility regime
                for i in range(max(1, sequence_length - self.manipulation_persistence), sequence_length):
                    # Create rapid volatility changes
                    if i % 2 == 0:
                        attacked_data[i, vol_col] = attacked_data[i, vol_col] * 2.5
                    else:
                        attacked_data[i, vol_col] = attacked_data[i, vol_col] * 0.3
                
                attack_payload['volatility_manipulations']['transitional_vol_forcing'] = {
                    'oscillation_high': 2.5,
                    'oscillation_low': 0.3,
                    'target_regime': 'transitional'
                }
                
                # Update regime indicators
                attacked_indicators['volatility_regime'] = 'transitional'
                attacked_indicators['market_regime'] = 'transitional'
        
        # Adjust related indicators to support volatility regime
        if target_volatility_regime == 'high':
            attacked_indicators['regime_confidence'] = 0.85
            attacked_indicators['mmd_score'] = 0.7  # High MMD for regime change
        elif target_volatility_regime == 'low':
            attacked_indicators['regime_confidence'] = 0.90
            attacked_indicators['mmd_score'] = 0.1  # Low MMD for stable regime
        else:  # transitional
            attacked_indicators['regime_confidence'] = 0.30
            attacked_indicators['mmd_score'] = 0.55  # Medium MMD for transition
        
        # Record regime forcing
        attack_payload['regime_forcing'] = {
            'forced_volatility_regime': target_volatility_regime,
            'supporting_indicators': {
                'regime_confidence': attacked_indicators['regime_confidence'],
                'mmd_score': attacked_indicators['mmd_score'],
                'market_regime': attacked_indicators['market_regime']
            }
        }
        
        # Calculate regime disruption score
        regime_disruption = self._calculate_regime_disruption_score(
            original_indicators, attacked_indicators, target_volatility_regime
        )
        
        # Calculate volatility manipulation strength
        volatility_manipulation = self._calculate_volatility_manipulation_strength(
            original_data, attacked_data
        )
        
        # Determine attack success
        success = regime_disruption > self.regime_threshold and volatility_manipulation > 0.6
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = RegimeAttackResult(
            attack_type=RegimeAttackType.VOLATILITY_MANIPULATION.value,
            success=success,
            confidence=regime_disruption,
            regime_disruption_score=regime_disruption,
            false_regime_strength=volatility_manipulation,
            transition_confusion_rate=0.0,  # Not applicable for this attack
            original_regime_detection=original_indicators,
            attacked_regime_detection=attacked_indicators,
            affected_regime_indicators=['volatility_30', 'mmd_score'],
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Volatility manipulation attack executed: "
            f"success={success}, regime_disruption={regime_disruption:.3f}, "
            f"volatility_manipulation={volatility_manipulation:.3f}"
        )
        
        # Return attacked data and indicators
        attack_payload['attacked_data'] = attacked_data
        attack_payload['attacked_indicators'] = attacked_indicators
        
        return attack_payload, attack_result
    
    def generate_mmd_poisoning_attack(
        self,
        market_data: np.ndarray,
        regime_indicators: Dict[str, Any],
        target_mmd_score: float = 0.9
    ) -> Tuple[Dict[str, Any], RegimeAttackResult]:
        """
        🎯 ATTACK 5: MMD SCORE POISONING
        
        Poisons the Maximum Mean Discrepancy (MMD) calculation to force
        a specific regime detection outcome.
        
        Args:
            market_data: Market data matrix [sequence_length, features]
            regime_indicators: Current regime indicators
            target_mmd_score: Target MMD score to achieve
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        # Store original data and indicators
        original_data = market_data.copy()
        original_indicators = regime_indicators.copy()
        
        # Create attacked versions
        attacked_data = market_data.copy()
        attacked_indicators = regime_indicators.copy()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': RegimeAttackType.MMD_POISONING.value,
            'target_mmd_score': target_mmd_score,
            'mmd_manipulations': {},
            'distribution_shifts': {}
        }
        
        sequence_length, n_features = market_data.shape
        
        # Create distribution shifts to achieve target MMD
        if target_mmd_score > 0.7:
            # High MMD: Create strong distribution shift
            shift_strength = 2.0
            
            # Create two distinct distributions in the data
            split_point = sequence_length // 2
            
            # First half: one distribution
            for i in range(split_point):
                for j in range(n_features):
                    attacked_data[i, j] = attacked_data[i, j] - shift_strength * np.std(attacked_data[:, j])
            
            # Second half: shifted distribution
            for i in range(split_point, sequence_length):
                for j in range(n_features):
                    attacked_data[i, j] = attacked_data[i, j] + shift_strength * np.std(attacked_data[:, j])
            
            attack_payload['distribution_shifts']['high_mmd_shift'] = {
                'shift_strength': shift_strength,
                'split_point': split_point,
                'distribution_type': 'bimodal'
            }
        
        elif target_mmd_score < 0.3:
            # Low MMD: Create minimal distribution shift
            shift_strength = 0.1
            
            # Create very similar distributions
            for i in range(sequence_length):
                for j in range(n_features):
                    # Add small random shift
                    attacked_data[i, j] = attacked_data[i, j] + shift_strength * np.random.normal(0, 0.1)
            
            attack_payload['distribution_shifts']['low_mmd_shift'] = {
                'shift_strength': shift_strength,
                'distribution_type': 'uniform_small_shift'
            }
        
        else:
            # Medium MMD: Create moderate distribution shift
            shift_strength = 1.0
            
            # Create gradual distribution shift
            for i in range(sequence_length):
                shift_factor = (i / sequence_length) * shift_strength
                for j in range(n_features):
                    attacked_data[i, j] = attacked_data[i, j] + shift_factor * np.std(attacked_data[:, j])
            
            attack_payload['distribution_shifts']['medium_mmd_shift'] = {
                'shift_strength': shift_strength,
                'distribution_type': 'gradual_shift'
            }
        
        # Directly set the MMD score
        attacked_indicators['mmd_score'] = target_mmd_score
        
        # Adjust related indicators based on MMD score
        if target_mmd_score > 0.7:
            attacked_indicators['market_regime'] = 'transitional'
            attacked_indicators['regime_confidence'] = 0.95
        elif target_mmd_score < 0.3:
            attacked_indicators['market_regime'] = 'stable'
            attacked_indicators['regime_confidence'] = 0.90
        else:
            attacked_indicators['market_regime'] = 'uncertain'
            attacked_indicators['regime_confidence'] = 0.50
        
        # Record MMD manipulations
        attack_payload['mmd_manipulations'] = {
            'original_mmd': original_indicators.get('mmd_score', 0.0),
            'target_mmd': target_mmd_score,
            'poisoned_mmd': attacked_indicators['mmd_score'],
            'regime_implication': attacked_indicators['market_regime']
        }
        
        # Calculate regime disruption score
        regime_disruption = self._calculate_regime_disruption_score(
            original_indicators, attacked_indicators, 'mmd'
        )
        
        # Calculate MMD poisoning strength
        mmd_poisoning_strength = abs(target_mmd_score - original_indicators.get('mmd_score', 0.0))
        
        # Determine attack success
        success = regime_disruption > self.regime_threshold and mmd_poisoning_strength > 0.3
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = RegimeAttackResult(
            attack_type=RegimeAttackType.MMD_POISONING.value,
            success=success,
            confidence=regime_disruption,
            regime_disruption_score=regime_disruption,
            false_regime_strength=mmd_poisoning_strength,
            transition_confusion_rate=0.0,  # Not applicable for this attack
            original_regime_detection=original_indicators,
            attacked_regime_detection=attacked_indicators,
            affected_regime_indicators=['mmd_score'],
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"MMD poisoning attack executed: "
            f"success={success}, regime_disruption={regime_disruption:.3f}, "
            f"mmd_poisoning_strength={mmd_poisoning_strength:.3f}"
        )
        
        # Return attacked data and indicators
        attack_payload['attacked_data'] = attacked_data
        attack_payload['attacked_indicators'] = attacked_indicators
        
        return attack_payload, attack_result
    
    def _calculate_regime_disruption_score(
        self,
        original_indicators: Dict[str, Any],
        attacked_indicators: Dict[str, Any],
        attack_type: str
    ) -> float:
        """Calculate regime disruption score."""
        disruption_score = 0.0
        count = 0
        
        # Compare key regime indicators
        regime_keys = ['market_regime', 'volatility_regime', 'momentum_regime', 'regime_confidence', 'mmd_score']
        
        for key in regime_keys:
            if key in original_indicators and key in attacked_indicators:
                original_val = original_indicators[key]
                attacked_val = attacked_indicators[key]
                
                if isinstance(original_val, str) and isinstance(attacked_val, str):
                    # Categorical disruption
                    if original_val != attacked_val:
                        disruption_score += 1.0
                elif isinstance(original_val, (int, float)) and isinstance(attacked_val, (int, float)):
                    # Numerical disruption
                    relative_change = abs(attacked_val - original_val) / (abs(original_val) + 1e-8)
                    disruption_score += min(relative_change, 1.0)
                
                count += 1
        
        return disruption_score / max(count, 1)
    
    def _calculate_false_signal_strength(
        self,
        original_data: np.ndarray,
        attacked_data: np.ndarray,
        signal_type: str
    ) -> float:
        """Calculate false signal strength."""
        if original_data.shape != attacked_data.shape:
            return 0.0
        
        # Calculate data manipulation strength
        data_change = np.mean(np.abs(attacked_data - original_data))
        data_std = np.std(original_data)
        
        normalized_change = data_change / (data_std + 1e-8)
        
        # Scale based on signal type
        if signal_type in ['bull', 'bear']:
            return min(normalized_change * 2.0, 1.0)  # Amplify for directional signals
        else:
            return min(normalized_change, 1.0)
    
    def _calculate_transition_confusion_rate(
        self,
        original_data: np.ndarray,
        attacked_data: np.ndarray
    ) -> float:
        """Calculate transition confusion rate."""
        if original_data.shape != attacked_data.shape:
            return 0.0
        
        sequence_length, n_features = original_data.shape
        
        # Calculate signal inconsistency
        confusion_score = 0.0
        
        for i in range(1, sequence_length):
            for j in range(n_features):
                # Check for signal direction changes
                original_direction = np.sign(original_data[i, j] - original_data[i-1, j])
                attacked_direction = np.sign(attacked_data[i, j] - attacked_data[i-1, j])
                
                if original_direction != attacked_direction:
                    confusion_score += 1.0
        
        total_comparisons = (sequence_length - 1) * n_features
        return confusion_score / max(total_comparisons, 1)
    
    def _calculate_volatility_manipulation_strength(
        self,
        original_data: np.ndarray,
        attacked_data: np.ndarray
    ) -> float:
        """Calculate volatility manipulation strength."""
        if original_data.shape != attacked_data.shape or original_data.shape[1] < 12:
            return 0.0
        
        # Focus on volatility column (assumed to be column 11)
        vol_col = 11
        
        original_vol = original_data[:, vol_col]
        attacked_vol = attacked_data[:, vol_col]
        
        # Calculate relative change in volatility
        vol_change = np.mean(np.abs(attacked_vol - original_vol))
        original_vol_std = np.std(original_vol)
        
        manipulation_strength = vol_change / (original_vol_std + 1e-8)
        
        return min(manipulation_strength, 1.0)
    
    def _record_attack_result(self, result: RegimeAttackResult):
        """Record attack result for analytics."""
        self.attack_history.append(result)
        
        # Update metrics
        self.regime_metrics['total_attempts'] += 1
        if result.success:
            self.regime_metrics['successful_attacks'] += 1
        
        # Update success rates
        attack_type = RegimeAttackType(result.attack_type)
        type_attempts = len([r for r in self.attack_history if r.attack_type == result.attack_type])
        type_successes = len([r for r in self.attack_history if r.attack_type == result.attack_type and r.success])
        self.success_rates[attack_type] = type_successes / type_attempts
        
        # Update disruption metrics
        self.regime_metrics['avg_disruption_score'] = np.mean([r.regime_disruption_score for r in self.attack_history])
        self.regime_metrics['max_disruption_score'] = max(self.regime_metrics['max_disruption_score'], result.regime_disruption_score)
        self.regime_metrics['regimes_compromised'].add(result.attack_type)
        
        # Keep history manageable
        if len(self.attack_history) > 1000:
            self.attack_history = self.attack_history[-500:]
    
    def get_attack_analytics(self) -> Dict[str, Any]:
        """Get comprehensive attack analytics."""
        if not self.attack_history:
            return {'status': 'no_attacks_recorded'}
        
        recent_attacks = self.attack_history[-100:]  # Last 100 attacks
        
        return {
            'total_attacks': len(self.attack_history),
            'recent_attacks': len(recent_attacks),
            'overall_success_rate': self.regime_metrics['successful_attacks'] / self.regime_metrics['total_attempts'],
            'success_rates_by_type': {attack_type.value: rate for attack_type, rate in self.success_rates.items()},
            'regime_metrics': self.regime_metrics.copy(),
            'recent_performance': {
                'avg_disruption_score': np.mean([r.regime_disruption_score for r in recent_attacks]),
                'max_disruption_score': max([r.regime_disruption_score for r in recent_attacks]),
                'avg_false_signal_strength': np.mean([r.false_regime_strength for r in recent_attacks if r.false_regime_strength > 0]),
                'avg_confusion_rate': np.mean([r.transition_confusion_rate for r in recent_attacks if r.transition_confusion_rate > 0]),
                'success_rate': len([r for r in recent_attacks if r.success]) / len(recent_attacks)
            },
            'attack_type_distribution': {
                attack_type.value: len([r for r in recent_attacks if r.attack_type == attack_type.value])
                for attack_type in RegimeAttackType
            },
            'regimes_compromised': list(self.regime_metrics['regimes_compromised'])
        }

# Example usage and testing functions
def run_regime_attack_demo():
    """Demonstrate regime transition attack capabilities."""
    print("🚨" * 50)
    print("AGENT GAMMA MISSION - REGIME TRANSITION ATTACK DEMO")
    print("🚨" * 50)
    
    attacker = RegimeTransitionAttacker()
    
    # Generate mock market data
    sequence_length = 48
    n_features = 13
    
    # Mock market data
    market_data = np.random.randn(sequence_length, n_features)
    market_data = np.cumsum(market_data, axis=0)  # Add temporal dependency
    
    # Mock regime indicators
    regime_indicators = {
        'market_regime': 'sideways',
        'regime_confidence': 0.6,
        'volatility_regime': 'medium',
        'momentum_regime': 'weak',
        'mmd_score': 0.3
    }
    
    print("\n🎯 ATTACK 1: FALSE BULL SIGNAL")
    payload, result = attacker.generate_false_bull_signal_attack(
        market_data.copy(), regime_indicators.copy()
    )
    print(f"Success: {result.success}, Regime Disruption: {result.regime_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 2: FALSE BEAR SIGNAL")
    payload, result = attacker.generate_false_bear_signal_attack(
        market_data.copy(), regime_indicators.copy()
    )
    print(f"Success: {result.success}, Regime Disruption: {result.regime_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 3: TRANSITION CONFUSION")
    payload, result = attacker.generate_transition_confusion_attack(
        market_data.copy(), regime_indicators.copy()
    )
    print(f"Success: {result.success}, Confusion Rate: {result.transition_confusion_rate:.3f}")
    
    print("\n🎯 ATTACK 4: VOLATILITY MANIPULATION")
    payload, result = attacker.generate_volatility_manipulation_attack(
        market_data.copy(), regime_indicators.copy(), 'high'
    )
    print(f"Success: {result.success}, Regime Disruption: {result.regime_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 5: MMD POISONING")
    payload, result = attacker.generate_mmd_poisoning_attack(
        market_data.copy(), regime_indicators.copy(), 0.9
    )
    print(f"Success: {result.success}, Regime Disruption: {result.regime_disruption_score:.3f}")
    
    print("\n📊 REGIME ATTACK ANALYTICS")
    analytics = attacker.get_attack_analytics()
    print(f"Overall Success Rate: {analytics['overall_success_rate']:.2%}")
    print(f"Average Disruption Score: {analytics['regime_metrics']['avg_disruption_score']:.3f}")
    print(f"Regimes Compromised: {analytics['regimes_compromised']}")
    
    return attacker

if __name__ == "__main__":
    run_regime_attack_demo()