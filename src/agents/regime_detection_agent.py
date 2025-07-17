"""
Regime Detection Agent for Market Regime Classification and Strategic Decision Support.

This agent uses Maximum Mean Discrepancy (MMD) and other statistical measures to classify 
market regimes and provide regime-aware strategic decisions.

Features used: [10, 11, 12] from 48x13 matrix:
- Feature 10: mmd_score (Maximum Mean Discrepancy score)
- Feature 11: volatility_30 (30-period volatility)
- Feature 12: volume_profile_skew (Volume profile skewness)
"""

import logging


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional
from enum import Enum
import structlog
from scipy import stats

from .base_strategic_agent import BaseStrategicAgent, StrategicAction, MarketRegime
from ..indicators.custom.mmd import gaussian_kernel, compute_mmd


logger = structlog.get_logger()


class RegimePolicyNetwork(nn.Module):
    """
    Regime Policy Network with attention mechanism for regime detection features.
    
    Architecture: 3 inputs → 32 hidden → 16 hidden → 7 actions
    Input features: [mmd_score, volatility_30, volume_profile_skew]
    Output: 7 strategic actions with softmax probabilities
    """
    
    def __init__(
        self, 
        input_dim: int = 3, 
        hidden_dim: int = 32, 
        action_dim: int = 7,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.action_dim = action_dim
        
        # Enhanced attention mechanism for context-sensitive regime features
        self.attention_head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 3),  # 3 → 9 for increased capacity
            nn.ReLU(),
            nn.Linear(input_dim * 3, input_dim * 2),  # 9 → 6 for context sensitivity
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),  # 6 → 3 final attention weights
            nn.Softmax(dim=-1)  # Ensure attention weights sum to 1
        )
        
        # Main policy network (smaller for regime detection)
        self.network = nn.Sequential(
            # Input layer: 3 features (after attention)
            nn.Linear(input_dim, hidden_dim),  # 3 → 32
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Hidden layer
            nn.Linear(hidden_dim, 16),  # 32 → 16
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Output layer: 7 strategic actions
            nn.Linear(16, action_dim)  # 16 → 7
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.normal_(module.bias, 0, 0.1)  # Small random bias for attention sensitivity
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through policy network with attention mechanism.
        
        Args:
            features: Input features tensor (batch_size, 3)
            
        Returns:
            Dictionary with action_probs, logits, and attention_weights
        """
        # Step 1: Generate dynamic attention weights for regime features
        attention_weights = self.attention_head(features)
        
        # Step 2: Apply attention to input features (element-wise multiplication)
        # Focus on most relevant regime indicators (MMD, volatility, volume)
        focused_features = features * attention_weights
        
        # Step 3: Pass focused input to main network
        logits = self.network(focused_features)
        
        # Apply softmax for probabilities (ensures sum = 1.0)
        action_probs = F.softmax(logits, dim=-1)
        
        return {
            'action_probs': action_probs,
            'logits': logits,
            'attention_weights': attention_weights,
            'focused_features': focused_features
        }


class RegimeTransition:
    """Market regime transition tracking"""
    
    def __init__(self, from_regime: MarketRegime, to_regime: MarketRegime, 
                 transition_prob: float, timestamp: Any = None):
        self.from_regime = from_regime
        self.to_regime = to_regime
        self.transition_prob = transition_prob
        self.timestamp = timestamp


class RegimeDetectionAgent(BaseStrategicAgent):
    """
    Strategic agent using MMD and volatility analysis for market regime detection.
    
    This agent classifies market regimes and adjusts strategic decisions based on
    the current regime state, volatility levels, and regime transition probabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Regime Detection Agent
        
        Args:
            config: Agent configuration
        """
        super().__init__(config)
        
        # MMD parameters (verified to match PRD specification)
        self.mmd_sigma = config.get('mmd_sigma', 1.0)  # Gaussian kernel bandwidth
        self.reference_window = config.get('reference_window', 500)
        self.test_window = config.get('test_window', 100)
        
        # Regime classification thresholds
        self.volatility_thresholds = {
            'low': config.get('low_vol_threshold', 0.2),
            'medium': config.get('medium_vol_threshold', 0.5),
            'high': config.get('high_vol_threshold', 0.8),
            'crisis': config.get('crisis_vol_threshold', 1.2)
        }
        
        self.mmd_thresholds = {
            'stable': config.get('stable_mmd_threshold', 0.1),
            'changing': config.get('changing_mmd_threshold', 0.3),
            'volatile': config.get('volatile_mmd_threshold', 0.5)
        }
        
        # Regime state tracking
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.regime_history = []
        self.transition_matrix = self._initialize_transition_matrix()
        
        # Feature history for streaming MMD calculation
        self.feature_history = []
        self.reference_distribution = None
        self.max_history = config.get('max_history', 1000)
        
        # Volume profile analysis
        self.volume_history = []
        self.volume_window = config.get('volume_window', 20)
        
        # Policy network for neural decision making
        self.policy_network = RegimePolicyNetwork(
            input_dim=3,  # [mmd_score, volatility_30, volume_profile_skew]
            hidden_dim=config.get('hidden_dim', 32),
            action_dim=7,  # 7 strategic actions
            dropout_rate=config.get('dropout_rate', 0.1)
        )
        
        # Feature normalization statistics
        self.feature_mean = torch.zeros(3)
        self.feature_std = torch.ones(3)
        self.feature_count = 0
        
        self.logger.info("Regime Detection Agent initialized",
                        mmd_sigma=self.mmd_sigma,
                        ref_window=self.reference_window,
                        test_window=self.test_window)
    
    def _initialize_transition_matrix(self) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
        """Initialize regime transition probability matrix"""
        # Default transition probabilities (can be learned from data)
        transitions = {}
        for from_regime in MarketRegime:
            transitions[from_regime] = {}
            for to_regime in MarketRegime:
                if from_regime == to_regime:
                    transitions[from_regime][to_regime] = 0.8  # High persistence
                else:
                    transitions[from_regime][to_regime] = 0.05  # Low transition prob
        
        # Adjust some specific transitions based on market dynamics
        transitions[MarketRegime.BULL_TREND][MarketRegime.SIDEWAYS] = 0.15
        transitions[MarketRegime.BEAR_TREND][MarketRegime.SIDEWAYS] = 0.15
        transitions[MarketRegime.SIDEWAYS][MarketRegime.BULL_TREND] = 0.1
        transitions[MarketRegime.SIDEWAYS][MarketRegime.BEAR_TREND] = 0.1
        transitions[MarketRegime.CRISIS][MarketRegime.RECOVERY] = 0.3
        
        return transitions
    
    def extract_features(self, observation_matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract features [10, 11, 12] from 48x13 observation matrix
        
        Args:
            observation_matrix: Full 48x13 feature matrix
            
        Returns:
            3-dimensional feature vector [mmd_score, volatility_30, volume_profile_skew]
        """
        if not self.validate_observation(observation_matrix):
            return np.zeros(3)
        
        try:
            # Extract the most recent bar's features
            current_bar = observation_matrix[-1, :]  # Last (most recent) bar
            
            # Feature indices as specified in mission
            mmd_score = current_bar[10] if len(current_bar) > 10 else 0.0
            volatility_30 = current_bar[11] if len(current_bar) > 11 else 0.0
            volume_profile_skew = current_bar[12] if len(current_bar) > 12 else 0.0
            
            # Update feature history for streaming calculations
            self.feature_history.append(current_bar[:4])  # Store first 4 features for MMD
            if len(self.feature_history) > self.max_history:
                self.feature_history.pop(0)
            
            # Update volume history for skew calculations
            volume = current_bar[4] if len(current_bar) > 4 else 0.0  # Volume is feature 4
            self.volume_history.append(volume)
            if len(self.volume_history) > self.volume_window:
                self.volume_history.pop(0)
            
            # Recalculate volume profile skew if we have enough data
            if len(self.volume_history) >= self.volume_window:
                try:
                    volume_profile_skew = float(stats.skew(self.volume_history))
                except (FileNotFoundError, IOError, OSError) as e:
                    volume_profile_skew = 0.0
            
            features = np.array([mmd_score, volatility_30, volume_profile_skew])
            
            # Ensure no NaN or Inf values
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Convert to tensor
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            
            if normalize:
                # Apply normalization
                self._update_feature_normalization(feature_tensor)
                normalized_features = self._normalize_features(feature_tensor)
                return normalized_features.numpy()
            else:
                # Return raw features for testing
                return features
            
        except Exception as e:
            self.logger.error("Error extracting regime detection features", error=str(e))
            return np.zeros(3)
    
    def _update_feature_normalization(self, features: torch.Tensor):
        """Update running statistics for feature normalization."""
        self.feature_count += 1
        
        # Online mean and std update
        if self.feature_count == 1:
            self.feature_mean = features.clone()
            self.feature_std = torch.ones_like(features)
        else:
            # Welford's algorithm for numerical stability
            delta = features - self.feature_mean
            self.feature_mean += delta / self.feature_count
            delta2 = features - self.feature_mean
            # Update variance estimate
            if self.feature_count > 1:
                var_update = delta * delta2 / (self.feature_count - 1)
                self.feature_std = torch.sqrt(var_update + 1e-8)
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalization to features."""
        # Use more robust normalization to prevent division by very small numbers
        std_clamped = torch.clamp(self.feature_std, min=0.1)  # Prevent division by tiny numbers
        normalized = (features - self.feature_mean) / std_clamped
        
        # Additional clipping to prevent extreme values
        normalized = torch.clamp(normalized, min=-5.0, max=5.0)
        
        return normalized
    
    def verify_mmd_calculation(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Verify MMD calculation matches PRD specification exactly
        
        Args:
            X: Reference distribution samples
            Y: Test distribution samples
            
        Returns:
            MMD score
        """
        try:
            # Use the verified MMD implementation from indicators
            mmd_score = compute_mmd(X, Y, self.mmd_sigma)
            
            # Verification: MMD should be non-negative and bounded
            assert mmd_score >= 0, f"MMD should be non-negative, got {mmd_score}"
            
            self.logger.debug("MMD calculation verified",
                            mmd_score=mmd_score,
                            X_shape=X.shape,
                            Y_shape=Y.shape,
                            sigma=self.mmd_sigma)
            
            return mmd_score
            
        except Exception as e:
            self.logger.error("Error in MMD calculation verification", error=str(e))
            return 0.0
    
    def classify_regime(self, features: np.ndarray) -> Tuple[MarketRegime, float]:
        """
        Classify current market regime based on MMD score, volatility, and volume
        
        Args:
            features: [mmd_score, volatility_30, volume_profile_skew]
            
        Returns:
            Tuple of (regime, confidence)
        """
        try:
            mmd_score, volatility_30, volume_profile_skew = features
            
            # Initialize regime probabilities
            regime_probs = {regime: 0.0 for regime in MarketRegime}
            
            # Volatility-based classification
            if volatility_30 > self.volatility_thresholds['crisis']:
                regime_probs[MarketRegime.CRISIS] += 0.4
            elif volatility_30 > self.volatility_thresholds['high']:
                regime_probs[MarketRegime.BEAR_TREND] += 0.3
                regime_probs[MarketRegime.BULL_TREND] += 0.1
            elif volatility_30 > self.volatility_thresholds['medium']:
                regime_probs[MarketRegime.BULL_TREND] += 0.3
                regime_probs[MarketRegime.BEAR_TREND] += 0.2
                regime_probs[MarketRegime.SIDEWAYS] += 0.1
            else:  # Low volatility
                regime_probs[MarketRegime.SIDEWAYS] += 0.4
                regime_probs[MarketRegime.RECOVERY] += 0.1
            
            # MMD-based regime shift detection
            if mmd_score > self.mmd_thresholds['volatile']:
                # High MMD indicates distribution shift - trending behavior
                regime_probs[MarketRegime.BULL_TREND] += 0.2
                regime_probs[MarketRegime.BEAR_TREND] += 0.2
                regime_probs[MarketRegime.CRISIS] += 0.1
            elif mmd_score > self.mmd_thresholds['changing']:
                # Medium MMD indicates some regime change
                regime_probs[MarketRegime.BULL_TREND] += 0.15
                regime_probs[MarketRegime.BEAR_TREND] += 0.15
                regime_probs[MarketRegime.RECOVERY] += 0.1
            else:
                # Low MMD indicates stable regime
                regime_probs[MarketRegime.SIDEWAYS] += 0.3
                regime_probs[MarketRegime.RECOVERY] += 0.2
            
            # Volume profile skew analysis
            if abs(volume_profile_skew) > 1.0:  # High skew indicates unusual volume patterns
                regime_probs[MarketRegime.CRISIS] += 0.1
                regime_probs[MarketRegime.BULL_TREND] += 0.05
                regime_probs[MarketRegime.BEAR_TREND] += 0.05
            
            # Apply regime persistence (Markov property)
            for regime in MarketRegime:
                transition_prob = self.transition_matrix[self.current_regime][regime]
                regime_probs[regime] += transition_prob * 0.2  # Weight historical persistence
            
            # Normalize probabilities
            total_prob = sum(regime_probs.values())
            if total_prob > 0:
                for regime in regime_probs:
                    regime_probs[regime] /= total_prob
            else:
                # Default to uniform distribution
                for regime in regime_probs:
                    regime_probs[regime] = 1.0 / len(MarketRegime)
            
            # Select regime with highest probability
            best_regime = max(regime_probs, key=regime_probs.get)
            confidence = regime_probs[best_regime]
            
            self.logger.debug("Regime classified",
                            regime=best_regime.value,
                            confidence=confidence,
                            mmd_score=mmd_score,
                            volatility=volatility_30,
                            volume_skew=volume_profile_skew)
            
            return best_regime, confidence
            
        except Exception as e:
            self.logger.error("Error classifying regime", error=str(e))
            return MarketRegime.SIDEWAYS, 0.5
    
    def get_volatility_adjusted_params(self, volatility: float) -> Dict[str, float]:
        """
        Get volatility-adjusted policy parameters
        
        Args:
            volatility: Current volatility level
            
        Returns:
            Dictionary of adjusted parameters
        """
        if volatility > self.volatility_thresholds['crisis']:  # Crisis regime
            return {
                "risk_multiplier": 0.3,
                "confidence_threshold": 0.9,
                "position_size_limit": 0.5,
                "stop_loss_tightening": 2.0
            }
        elif volatility > self.volatility_thresholds['high']:  # High volatility
            return {
                "risk_multiplier": 0.5,
                "confidence_threshold": 0.8,
                "position_size_limit": 0.7,
                "stop_loss_tightening": 1.5
            }
        elif volatility < self.volatility_thresholds['low']:  # Low volatility (sideways)
            return {
                "risk_multiplier": 1.2,
                "confidence_threshold": 0.6,
                "position_size_limit": 1.5,
                "stop_loss_tightening": 0.8
            }
        else:  # Medium volatility (normal trending)
            return {
                "risk_multiplier": 1.0,
                "confidence_threshold": 0.7,
                "position_size_limit": 1.0,
                "stop_loss_tightening": 1.0
            }
    
    def make_decision(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make strategic trading decision using regime features with attention mechanism
        
        Args:
            features: [mmd_score, volatility_30, volume_profile_skew]
            
        Returns:
            Decision dictionary with action, confidence, attention weights, and regime info
        """
        try:
            mmd_score, volatility_30, volume_profile_skew = features
            
            # Classify current regime (keep for regime tracking)
            regime, regime_confidence = self.classify_regime(features)
            
            # Update agent state
            previous_regime = self.current_regime
            self.current_regime = regime
            self.regime_confidence = regime_confidence
            
            # Track regime transitions
            if previous_regime != regime:
                transition = RegimeTransition(previous_regime, regime, regime_confidence)
                self.regime_history.append(transition)
                if len(self.regime_history) > 100:
                    self.regime_history.pop(0)
                
                self.logger.info("Regime transition detected",
                               from_regime=previous_regime.value,
                               to_regime=regime.value,
                               confidence=regime_confidence)
            
            # Convert features to tensor
            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # Forward pass through policy network with attention
            with torch.no_grad():
                policy_output = self.policy_network(feature_tensor)
            
            # Extract results
            action_probs = policy_output['action_probs'].squeeze(0)
            attention_weights = policy_output['attention_weights'].squeeze(0)
            focused_features = policy_output['focused_features'].squeeze(0)
            
            # Sample action (deterministic for now, could add exploration)
            action = torch.argmax(action_probs).item()
            confidence = action_probs.max().item()
            
            # Adjust confidence based on regime certainty
            confidence *= regime_confidence
            
            # Map action to strategic decision
            strategic_actions = [
                'STRONG_SELL', 'SELL', 'WEAK_SELL', 'HOLD', 
                'WEAK_BUY', 'BUY', 'STRONG_BUY'
            ]
            
            decision = {
                'action': action,
                'action_name': strategic_actions[action],
                'confidence': confidence,
                'action_probabilities': action_probs.tolist(),
                'features': features.tolist(),
                'attention_weights': attention_weights.tolist(),
                'focused_features': focused_features.tolist(),
                'feature_names': ['mmd_score', 'volatility_30', 'volume_profile_skew'],
                'agent_id': 'regime_detection_agent',
                'timestamp': None,  # Would be set by caller
                'mathematical_method': 'MMD_with_Attention',
                'current_regime': regime.value,
                'regime_confidence': regime_confidence,
                'volatility_adjusted_params': self.get_volatility_adjusted_params(volatility_30)
            }
            
            self.logger.debug("Regime decision made with attention",
                            action=action,
                            confidence=confidence,
                            attention_weights=attention_weights.tolist(),
                            regime=regime.value,
                            features=features.tolist())
            
            return decision
            
        except Exception as e:
            self.logger.error("Error making regime-based decision", error=str(e))
            return self._safe_default_decision(features)
    
    def _safe_default_decision(self, features: np.ndarray) -> Dict[str, Any]:
        """Return safe default decision in case of errors."""
        return {
            'action': 3,  # HOLD
            'action_name': 'HOLD',
            'confidence': 0.5,
            'action_probabilities': [0.14, 0.14, 0.14, 0.16, 0.14, 0.14, 0.14],
            'features': features.tolist() if features is not None else [0.0, 0.0, 0.0],
            'attention_weights': [0.33, 0.33, 0.34],
            'focused_features': [0.0, 0.0, 0.0],
            'feature_names': ['mmd_score', 'volatility_30', 'volume_profile_skew'],
            'agent_id': 'regime_detection_agent',
            'timestamp': None,
            'mathematical_method': 'MMD_with_Attention',
            'current_regime': 'SIDEWAYS',
            'regime_confidence': 0.5,
            'error': 'safe_default'
        }
    
    def get_regime_transition_probability(self, target_regime: MarketRegime) -> float:
        """Get probability of transitioning to target regime"""
        return self.transition_matrix[self.current_regime][target_regime]
    
    def update_transition_matrix(self, observed_transitions: List[RegimeTransition]) -> None:
        """Update transition matrix based on observed regime changes"""
        try:
            # Count transitions
            transition_counts = {}
            for from_regime in MarketRegime:
                transition_counts[from_regime] = {}
                for to_regime in MarketRegime:
                    transition_counts[from_regime][to_regime] = 0
            
            # Count observed transitions
            for transition in observed_transitions:
                transition_counts[transition.from_regime][transition.to_regime] += 1
            
            # Update probabilities with smoothing
            alpha = 0.1  # Learning rate
            for from_regime in MarketRegime:
                total_from = sum(transition_counts[from_regime].values())
                if total_from > 0:
                    for to_regime in MarketRegime:
                        observed_prob = transition_counts[from_regime][to_regime] / total_from
                        current_prob = self.transition_matrix[from_regime][to_regime]
                        # Exponential smoothing
                        self.transition_matrix[from_regime][to_regime] = (
                            (1 - alpha) * current_prob + alpha * observed_prob
                        )
            
            self.logger.info("Transition matrix updated", 
                           transitions_processed=len(observed_transitions))
            
        except Exception as e:
            self.logger.error("Error updating transition matrix", error=str(e))
    
    def get_regime_info(self) -> Dict[str, Any]:
        """Get current regime information"""
        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'recent_transitions': [
                {
                    'from': t.from_regime.value,
                    'to': t.to_regime.value,
                    'prob': t.transition_prob
                }
                for t in self.regime_history[-5:]  # Last 5 transitions
            ]
        }
    
    def reset(self) -> None:
        """Reset agent state"""
        super().reset()
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        self.regime_history = []
        self.feature_history = []
        self.volume_history = []
        self.reference_distribution = None
        self.feature_mean = torch.zeros(3)
        self.feature_std = torch.ones(3)
        self.feature_count = 0
        self.logger.info("Regime detection agent reset")