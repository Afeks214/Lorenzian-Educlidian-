"""
NWRQK Strategic Agent for Support/Resistance Detection and Strategic Trading Decisions.

This agent uses the corrected Nadaraya-Watson Rational Quadratic Kernel (NWRQK) 
to detect support and resistance levels and make strategic trading decisions.

Features used: [2, 3, 4, 5] from 48x13 matrix:
- Feature 2: nwrqk_value (current NWRQK indicator value)
- Feature 3: nwrqk_slope (NWRQK trend slope)  
- Feature 4: lvn_distance (Level Value Nodes distance)
- Feature 5: lvn_strength (Level Value Nodes strength)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List
from enum import Enum
import structlog

from .base_strategic_agent import BaseStrategicAgent, StrategicAction, MarketRegime
from ..indicators.custom.nwrqk import rational_quadratic_kernel


logger = structlog.get_logger()


class NWRQKPolicyNetwork(nn.Module):
    """
    NWRQK Policy Network with attention mechanism for support/resistance features.
    
    Architecture: 4 inputs → 64 hidden → 32 hidden → 7 actions
    Input features: [nwrqk_value, nwrqk_slope, lvn_distance, lvn_strength]
    Output: 7 strategic actions with softmax probabilities
    """
    
    def __init__(
        self, 
        input_dim: int = 4, 
        hidden_dim: int = 64, 
        action_dim: int = 7,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.action_dim = action_dim
        
        # Enhanced attention mechanism for context-sensitive NWRQK features
        self.attention_head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),  # 4 → 8 for increased capacity
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),  # 8 → 4 for context sensitivity
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),  # 4 → 4 final attention weights
            nn.Softmax(dim=-1)  # Ensure attention weights sum to 1
        )
        
        # Main policy network
        self.network = nn.Sequential(
            # Input layer: 4 features (after attention)
            nn.Linear(input_dim, hidden_dim),  # 4 → 64
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Hidden layer
            nn.Linear(hidden_dim, 32),  # 64 → 32
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Output layer: 7 strategic actions
            nn.Linear(32, action_dim)  # 32 → 7
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
            features: Input features tensor (batch_size, 4)
            
        Returns:
            Dictionary with action_probs, logits, and attention_weights
        """
        # Step 1: Generate dynamic attention weights for NWRQK features
        attention_weights = self.attention_head(features)
        
        # Step 2: Apply attention to input features (element-wise multiplication)
        # Focus on most relevant support/resistance indicators
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


class SupportResistanceLevel:
    """Support or resistance level identified by NWRQK"""
    
    def __init__(self, price: float, strength: float, level_type: str):
        self.price = price
        self.strength = strength  # 0.0 to 1.0
        self.level_type = level_type  # 'support' or 'resistance'
        self.touches = 1
        self.last_test_time = None
        
    def update_strength(self, new_strength: float):
        """Update strength based on new kernel density"""
        self.strength = max(self.strength, new_strength)
        self.touches += 1


class NWRQKStrategicAgent(BaseStrategicAgent):
    """
    Strategic agent using NWRQK kernel for support/resistance detection.
    
    This agent implements the corrected NWRQK calculation with proper distance
    computation and uses it to identify key support and resistance levels for
    strategic position decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NWRQK Strategic Agent
        
        Args:
            config: Agent configuration
        """
        super().__init__(config)
        
        # NWRQK parameters (PRD specification)
        self.alpha = config.get('alpha', 1.0)  # RQ kernel parameter
        self.h = config.get('h', 1.0)  # Bandwidth parameter
        self.lookback_window = config.get('lookback_window', 50)
        
        # Support/Resistance detection parameters
        self.density_threshold = config.get('density_threshold', 0.3)
        self.breakout_threshold = config.get('breakout_threshold', 0.02)  # 2% price move
        self.min_level_separation = config.get('min_level_separation', 0.01)  # 1% min separation
        
        # Level tracking
        self.support_levels: List[SupportResistanceLevel] = []
        self.resistance_levels: List[SupportResistanceLevel] = []
        self.max_levels = config.get('max_levels', 5)
        
        # Feature history for slope calculation
        self.nwrqk_history = []
        self.slope_window = config.get('slope_window', 5)
        
        # Policy network for neural decision making
        self.policy_network = NWRQKPolicyNetwork(
            input_dim=4,  # [nwrqk_value, nwrqk_slope, lvn_distance, lvn_strength]
            hidden_dim=config.get('hidden_dim', 64),
            action_dim=7,  # 7 strategic actions
            dropout_rate=config.get('dropout_rate', 0.1)
        )
        
        # Feature normalization statistics
        self.feature_mean = torch.zeros(4)
        self.feature_std = torch.ones(4)
        self.feature_count = 0
        
        self.logger.info("NWRQK Strategic Agent initialized",
                        alpha=self.alpha,
                        h=self.h,
                        lookback=self.lookback_window)
    
    def extract_features(self, observation_matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract features [2, 3, 4, 5] from 48x13 observation matrix
        
        Args:
            observation_matrix: Full 48x13 feature matrix
            
        Returns:
            4-dimensional feature vector [nwrqk_value, nwrqk_slope, lvn_distance, lvn_strength]
        """
        if not self.validate_observation(observation_matrix):
            return np.zeros(4)
        
        try:
            # Extract the most recent bar's features
            current_bar = observation_matrix[-1, :]  # Last (most recent) bar
            
            # Feature indices as specified in mission
            nwrqk_value = current_bar[2] if len(current_bar) > 2 else 0.0
            nwrqk_slope = current_bar[3] if len(current_bar) > 3 else 0.0  
            lvn_distance = current_bar[4] if len(current_bar) > 4 else 0.0
            lvn_strength = current_bar[5] if len(current_bar) > 5 else 0.0
            
            # Calculate NWRQK slope from history if available
            self.nwrqk_history.append(nwrqk_value)
            if len(self.nwrqk_history) > self.slope_window:
                self.nwrqk_history.pop(0)
            
            if len(self.nwrqk_history) >= 2:
                # Calculate slope using linear regression over window
                x = np.arange(len(self.nwrqk_history))
                y = np.array(self.nwrqk_history)
                if np.std(x) > 0:
                    slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
                    nwrqk_slope = slope
                
            features = np.array([nwrqk_value, nwrqk_slope, lvn_distance, lvn_strength])
            
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
            self.logger.error("Error extracting NWRQK features", error=str(e))
            return np.zeros(4)
    
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
    
    def detect_support_resistance(self, price_data: np.ndarray, nwrqk_values: np.ndarray) -> None:
        """
        Detect support and resistance levels using NWRQK kernel density estimation
        
        Args:
            price_data: Historical price data
            nwrqk_values: Corresponding NWRQK values
        """
        if len(price_data) < self.lookback_window:
            return
        
        try:
            # Use recent price data for level detection
            recent_prices = price_data[-self.lookback_window:]
            recent_nwrqk = nwrqk_values[-self.lookback_window:]
            current_price = price_data[-1]
            
            # Clear old levels
            self.support_levels = []
            self.resistance_levels = []
            
            # Find local extremes using kernel density
            for i in range(len(recent_prices)):
                price = recent_prices[i]
                
                # Calculate kernel density at this price level
                density = 0.0
                for j in range(len(recent_prices)):
                    if i != j:
                        density += rational_quadratic_kernel(price, recent_prices[j], self.alpha, self.h)
                
                density /= len(recent_prices)
                
                # Identify support/resistance based on density and price context
                if density > self.density_threshold:
                    # Determine if support or resistance based on price action
                    prices_above = np.sum(recent_prices > price)
                    prices_below = np.sum(recent_prices < price)
                    
                    if prices_above > prices_below and price < current_price:
                        # More prices above this level and below current price = support
                        level = SupportResistanceLevel(price, density, 'support')
                        self._add_support_level(level)
                    elif prices_below > prices_above and price > current_price:
                        # More prices below this level and above current price = resistance  
                        level = SupportResistanceLevel(price, density, 'resistance')
                        self._add_resistance_level(level)
                        
        except Exception as e:
            self.logger.error("Error detecting support/resistance", error=str(e))
    
    def _add_support_level(self, level: SupportResistanceLevel) -> None:
        """Add support level, maintaining max levels and minimum separation"""
        # Check minimum separation from existing levels
        for existing in self.support_levels:
            if abs(level.price - existing.price) / existing.price < self.min_level_separation:
                # Too close to existing level, update if stronger
                if level.strength > existing.strength:
                    existing.price = level.price
                    existing.strength = level.strength
                return
        
        # Add new level
        self.support_levels.append(level)
        
        # Sort by strength and keep only top levels
        self.support_levels.sort(key=lambda x: x.strength, reverse=True)
        if len(self.support_levels) > self.max_levels:
            self.support_levels = self.support_levels[:self.max_levels]
    
    def _add_resistance_level(self, level: SupportResistanceLevel) -> None:
        """Add resistance level, maintaining max levels and minimum separation"""
        # Check minimum separation from existing levels
        for existing in self.resistance_levels:
            if abs(level.price - existing.price) / existing.price < self.min_level_separation:
                # Too close to existing level, update if stronger
                if level.strength > existing.strength:
                    existing.price = level.price
                    existing.strength = level.strength
                return
        
        # Add new level
        self.resistance_levels.append(level)
        
        # Sort by strength and keep only top levels
        self.resistance_levels.sort(key=lambda x: x.strength, reverse=True)
        if len(self.resistance_levels) > self.max_levels:
            self.resistance_levels = self.resistance_levels[:self.max_levels]
    
    def calculate_breakout_probability(self, current_price: float, level: SupportResistanceLevel) -> float:
        """
        Calculate probability of breakout/breakdown through a level
        
        Args:
            current_price: Current market price
            level: Support or resistance level
            
        Returns:
            Breakout probability (0.0 to 1.0)
        """
        try:
            # Distance to level (normalized by price)
            distance_to_level = abs(current_price - level.price) / current_price
            
            # Kernel strength indicates how "solid" the level is
            kernel_strength = level.strength
            
            # Base probability inversely related to distance and strength
            base_prob = 1.0 / (1.0 + np.exp(10 * (distance_to_level - 0.005)))  # Sigmoid
            
            # Adjust for level strength (stronger levels harder to break)
            strength_adjustment = 1.0 - (kernel_strength * 0.5)
            
            # Final probability
            probability = base_prob * strength_adjustment
            return max(0.0, min(1.0, probability))
            
        except Exception as e:
            self.logger.error("Error calculating breakout probability", error=str(e))
            return 0.5
    
    def make_decision(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make strategic trading decision using NWRQK features with attention mechanism
        
        Args:
            features: [nwrqk_value, nwrqk_slope, lvn_distance, lvn_strength]
            
        Returns:
            Decision dictionary with action, confidence, and attention weights
        """
        try:
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
                'feature_names': ['nwrqk_value', 'nwrqk_slope', 'lvn_distance', 'lvn_strength'],
                'agent_id': 'nwrqk_strategic_agent',
                'timestamp': None,  # Would be set by caller
                'mathematical_method': 'NWRQK_with_Attention',
                'support_resistance_levels': self.get_support_resistance_levels()
            }
            
            self.logger.debug("NWRQK decision made with attention",
                            action=action,
                            confidence=confidence,
                            attention_weights=attention_weights.tolist(),
                            features=features.tolist())
            
            return decision
            
        except Exception as e:
            self.logger.error("Error making NWRQK decision", error=str(e))
            return self._safe_default_decision(features)
    
    def _safe_default_decision(self, features: np.ndarray) -> Dict[str, Any]:
        """Return safe default decision in case of errors."""
        return {
            'action': 3,  # HOLD
            'action_name': 'HOLD',
            'confidence': 0.5,
            'action_probabilities': [0.14, 0.14, 0.14, 0.16, 0.14, 0.14, 0.14],
            'features': features.tolist() if features is not None else [0.0, 0.0, 0.0, 0.0],
            'attention_weights': [0.25, 0.25, 0.25, 0.25],
            'focused_features': [0.0, 0.0, 0.0, 0.0],
            'feature_names': ['nwrqk_value', 'nwrqk_slope', 'lvn_distance', 'lvn_strength'],
            'agent_id': 'nwrqk_strategic_agent',
            'timestamp': None,
            'mathematical_method': 'NWRQK_with_Attention',
            'error': 'safe_default'
        }
    
    def get_position_sizing_recommendation(self, confidence: float, volatility: float) -> float:
        """
        Get position sizing recommendation based on confidence and volatility
        
        Args:
            confidence: Decision confidence (0.0 to 1.0)
            volatility: Current market volatility
            
        Returns:
            Position size multiplier (0.0 to 2.0)
        """
        try:
            # Base size from confidence
            base_size = confidence
            
            # Volatility adjustment (reduce size in high volatility)
            vol_adjustment = 1.0 / (1.0 + volatility * 2.0)
            
            # Support/resistance level adjustment
            level_adjustment = 1.0
            if self.support_levels or self.resistance_levels:
                # Increase size when trading with strong levels
                max_strength = 0.0
                for level in self.support_levels + self.resistance_levels:
                    max_strength = max(max_strength, level.strength)
                level_adjustment = 1.0 + (max_strength * 0.5)
            
            size = base_size * vol_adjustment * level_adjustment
            return max(0.1, min(2.0, size))
            
        except Exception as e:
            self.logger.error("Error calculating position size", error=str(e))
            return 1.0
    
    def get_support_resistance_levels(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get current support and resistance levels"""
        return {
            'support': [{'price': l.price, 'strength': l.strength, 'touches': l.touches} 
                       for l in self.support_levels],
            'resistance': [{'price': l.price, 'strength': l.strength, 'touches': l.touches}
                          for l in self.resistance_levels]
        }
    
    def reset(self) -> None:
        """Reset agent state"""
        super().reset()
        self.support_levels = []
        self.resistance_levels = []
        self.nwrqk_history = []
        self.feature_mean = torch.zeros(4)
        self.feature_std = torch.ones(4)
        self.feature_count = 0
        self.logger.info("NWRQK agent reset")