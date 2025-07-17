"""
Base Strategic Agent for 30-minute MARL trading decisions.

This module provides the abstract base class for all strategic agents in the system,
ensuring consistent interfaces and behavior with comprehensive error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from enum import Enum
import structlog
from functools import wraps

from src.core.errors.agent_error_decorators import (
    strategic_agent_decorator, AgentType, AgentErrorConfig
)
from src.core.errors.agent_recovery_strategies import (
    create_strategic_recovery_manager, RecoveryConfig
)
from src.core.errors.graceful_degradation import (
    create_strategic_degradation, DegradationConfig
)
from src.core.errors.error_monitoring import record_error
from src.core.errors.base_exceptions import (
    ValidationError, DataError, SystemError, ErrorContext
)

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
    Enhanced with comprehensive error handling, recovery, and degradation mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base strategic agent with enhanced error handling
        
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
        
        # Error handling setup
        self._setup_error_handling()
        
        # Recovery system
        self._setup_recovery_system()
        
        # Degradation system
        self._setup_degradation_system()
        
        self.logger.info("Strategic agent initialized with enhanced error handling", 
                        name=self.name, 
                        obs_dim=self.observation_dim)
    
    def _setup_error_handling(self):
        """Setup error handling decorators and configurations"""
        error_config = AgentErrorConfig(
            agent_type=AgentType.STRATEGIC,
            max_retries=self.config.get('max_retries', 3),
            retry_delay=self.config.get('retry_delay', 1.0),
            circuit_breaker_threshold=self.config.get('circuit_breaker_threshold', 5),
            graceful_degradation=self.config.get('graceful_degradation', True)
        )
        
        self.error_decorator = strategic_agent_decorator(error_config)
    
    def _setup_recovery_system(self):
        """Setup recovery management system"""
        recovery_config = RecoveryConfig(
            max_recovery_attempts=self.config.get('max_recovery_attempts', 3),
            recovery_timeout=self.config.get('recovery_timeout', 30.0),
            gradual_recovery_steps=self.config.get('gradual_recovery_steps', 3)
        )
        
        self.recovery_manager = create_strategic_recovery_manager(recovery_config)
    
    def _setup_degradation_system(self):
        """Setup graceful degradation system"""
        degradation_config = DegradationConfig(
            agent_type=AgentType.STRATEGIC,
            essential_features=['market_regime_detection', 'risk_assessment'],
            fallback_values={
                'market_regime': MarketRegime.SIDEWAYS.value,
                'confidence': 0.1,
                'action': StrategicAction.HOLD.value
            }
        )
        
        self.degradation_system = create_strategic_degradation(degradation_config)
    
    def _create_error_context(self, operation: str, **kwargs) -> ErrorContext:
        """Create error context for tracking"""
        return ErrorContext(
            service_name=self.name,
            additional_data={
                'operation': operation,
                'agent_type': 'strategic',
                'decisions_made': self.decisions_made,
                **kwargs
            }
        )
    
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
        Main agent step function - extracts features and makes decision with error handling
        
        Args:
            observation_matrix: Full 48x13 observation matrix
            
        Returns:
            Tuple of (action, confidence)
        """
        context = self._create_error_context('step', observation_shape=observation_matrix.shape)
        
        try:
            # Validate observation matrix
            if not self.validate_observation(observation_matrix):
                raise ValidationError("Invalid observation matrix", field="observation_matrix")
            
            # Extract agent-specific features with error handling
            features = self._extract_features_with_error_handling(observation_matrix)
            
            # Make decision with error handling
            action, confidence = self._make_decision_with_error_handling(features)
            
            # Track performance
            self.decisions_made += 1
            self.last_decision = {
                'action': action,
                'confidence': confidence,
                'features': features,
                'timestamp': self.logger.bind().msg
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
            # Record error for monitoring
            record_error(e, context, AgentType.STRATEGIC, self.name)
            
            # Attempt recovery
            recovery_result = self.recovery_manager.recover(e, context, self.__dict__)
            
            if recovery_result.get('success', False):
                # Retry with recovered state
                try:
                    return self.step(observation_matrix)
                except Exception as retry_error:
                    self.logger.error("Retry failed after recovery", error=str(retry_error))
            
            # Use degradation system for fallback
            return self._get_degraded_decision(e, context)
    
    def _extract_features_with_error_handling(self, observation_matrix: np.ndarray) -> np.ndarray:
        """Extract features with error handling and fallback"""
        try:
            return self.degradation_system.execute_feature(
                'feature_extraction',
                observation_matrix
            )
        except Exception as e:
            self.logger.warning("Feature extraction failed, using fallback", error=str(e))
            # Return basic features as fallback
            return np.mean(observation_matrix, axis=0)
    
    def _make_decision_with_error_handling(self, features: np.ndarray) -> Tuple[int, float]:
        """Make decision with error handling and fallback"""
        try:
            return self.degradation_system.execute_feature(
                'decision_making',
                features
            )
        except Exception as e:
            self.logger.warning("Decision making failed, using fallback", error=str(e))
            # Return safe default decision
            return StrategicAction.HOLD.value, 0.1
    
    def _get_degraded_decision(self, error: Exception, context: ErrorContext) -> Tuple[int, float]:
        """Get degraded decision when all else fails"""
        self.logger.error("Using degraded decision due to error", error=str(error))
        
        # Check if we can use degradation system
        if self.degradation_system.is_feature_enabled('emergency_decision'):
            try:
                return self.degradation_system.execute_feature('emergency_decision')
            except Exception as degraded_error:
                self.logger.error("Even degraded decision failed", error=str(degraded_error))
        
        # Ultimate fallback - conservative decision
        return StrategicAction.HOLD.value, 0.0
    
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