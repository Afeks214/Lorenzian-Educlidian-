"""
Base Strategic Agent for 30-minute MARL trading decisions.

This module provides the abstract base class for all strategic agents in the system,
ensuring consistent interfaces and behavior.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from enum import Enum
import structlog

logger = structlog.get_logger()


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend" 
    SIDEWAYS = "sideways"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class StrategicAction(Enum):
    """Strategic action space"""
    STRONG_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    STRONG_BUY = 4


class BaseStrategicAgent(ABC):
    """
    Abstract base class for strategic agents operating on 30-minute timeframes.
    
    Strategic agents make high-level decisions about market direction and position sizing
    based on regime analysis, trend identification, and support/resistance levels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base strategic agent
        
        Args:
            config: Agent configuration parameters
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.observation_dim = config.get('observation_dim', 624)  # 48 bars × 13 features
        self.action_dim = config.get('action_dim', 5)  # Strategic actions
        self.logger = structlog.get_logger(self.name)
        
        # Performance tracking
        self.decisions_made = 0
        self.last_decision = None
        self.confidence_history = []
        
        self.logger.info("Strategic agent initialized", 
                        name=self.name, 
                        obs_dim=self.observation_dim)
    
    @abstractmethod
    def extract_features(self, observation_matrix: np.ndarray) -> np.ndarray:
        """
        Extract agent-specific features from the 48x13 observation matrix
        
        Args:
            observation_matrix: Full 48x13 feature matrix
            
        Returns:
            Agent-specific feature vector
        """
        pass
    
    @abstractmethod
    def make_decision(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Make strategic trading decision based on extracted features
        
        Args:
            features: Agent-specific feature vector
            
        Returns:
            Tuple of (action, confidence)
            - action: StrategicAction enum value (0-4)
            - confidence: Confidence level (0.0 to 1.0)
        """
        pass
    
    def step(self, observation_matrix: np.ndarray) -> Tuple[int, float]:
        """
        Main agent step function - extracts features and makes decision
        
        Args:
            observation_matrix: Full 48x13 observation matrix
            
        Returns:
            Tuple of (action, confidence)
        """
        try:
            # Extract agent-specific features
            features = self.extract_features(observation_matrix)
            
            # Make decision
            action, confidence = self.make_decision(features)
            
            # Track performance
            self.decisions_made += 1
            self.last_decision = {
                'action': action,
                'confidence': confidence,
                'features': features
            }
            self.confidence_history.append(confidence)
            if len(self.confidence_history) > 1000:
                self.confidence_history.pop(0)
            
            self.logger.debug("Decision made",
                            action=action,
                            confidence=confidence,
                            decisions_count=self.decisions_made)
            
            return action, confidence
            
        except Exception as e:
            self.logger.error("Error in agent step", error=str(e))
            # Return neutral action with low confidence
            return StrategicAction.HOLD.value, 0.1
    
    def get_action_probabilities(self, features: np.ndarray) -> np.ndarray:
        """
        Get probability distribution over actions
        
        Args:
            features: Agent-specific feature vector
            
        Returns:
            Probability distribution over actions (sums to 1.0)
        """
        # Default implementation - override for more sophisticated probability calculation
        action, confidence = self.make_decision(features)
        probs = np.zeros(self.action_dim)
        probs[action] = confidence
        # Distribute remaining probability uniformly
        remaining_prob = 1.0 - confidence
        for i in range(self.action_dim):
            if i != action:
                probs[i] = remaining_prob / (self.action_dim - 1)
        return probs
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.0
        return {
            'decisions_made': self.decisions_made,
            'average_confidence': avg_confidence,
            'last_decision': self.last_decision
        }
    
    def reset(self) -> None:
        """Reset agent state"""
        self.decisions_made = 0
        self.last_decision = None
        self.confidence_history = []
        self.logger.info("Agent reset")
    
    def validate_observation(self, observation_matrix: np.ndarray) -> bool:
        """
        Validate observation matrix format
        
        Args:
            observation_matrix: Input observation matrix
            
        Returns:
            True if valid, False otherwise
        """
        if observation_matrix is None:
            return False
        if not isinstance(observation_matrix, np.ndarray):
            return False
        if observation_matrix.shape != (48, 13):
            self.logger.warning("Invalid observation shape", 
                              expected=(48, 13),
                              actual=observation_matrix.shape)
            return False
        if np.any(np.isnan(observation_matrix)) or np.any(np.isinf(observation_matrix)):
            self.logger.warning("Observation contains NaN or Inf values")
            return False
        return True
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name}(decisions={self.decisions_made})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}("
                f"name={self.name}, "
                f"decisions={self.decisions_made}, "
                f"obs_dim={self.observation_dim})")