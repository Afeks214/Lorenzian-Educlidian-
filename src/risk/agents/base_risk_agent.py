"""
Base Risk Agent for Risk Management MARL System.

This module provides the abstract base class for all risk management agents,
specialized for risk calculations, portfolio management, and real-time risk monitoring.

Inherits from BaseStrategicAgent but adapts the interface for risk-specific operations
with the 10-dimensional risk state vector as specified in the PRD.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
from enum import Enum
import structlog
from dataclasses import dataclass
from datetime import datetime

from src.agents.base_strategic_agent import BaseStrategicAgent
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


class RiskAction(Enum):
    """Risk management action classifications"""
    NO_ACTION = 0
    REDUCE_POSITION = 1
    CLOSE_ALL = 2
    HEDGE = 3
    REBALANCE = 4


@dataclass
class RiskState:
    """Risk state vector following PRD specification"""
    account_equity_normalized: float  # Account equity / initial capital
    open_positions_count: int         # Number of open positions
    volatility_regime: float          # Current volatility percentile (0-1)
    correlation_risk: float           # Portfolio correlation coefficient
    var_estimate_5pct: float          # 5% Value at Risk estimate
    current_drawdown_pct: float       # Current drawdown percentage
    margin_usage_pct: float           # Margin utilization percentage
    time_of_day_risk: float          # Time-based risk factor (0-1)
    market_stress_level: float       # Aggregate market stress indicator
    liquidity_conditions: float     # Market liquidity assessment
    
    def to_vector(self) -> np.ndarray:
        """Convert risk state to 10-dimensional vector"""
        return np.array([
            self.account_equity_normalized,
            self.open_positions_count,
            self.volatility_regime,
            self.correlation_risk,
            self.var_estimate_5pct,
            self.current_drawdown_pct,
            self.margin_usage_pct,
            self.time_of_day_risk,
            self.market_stress_level,
            self.liquidity_conditions
        ], dtype=np.float64)
    
    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'RiskState':
        """Create RiskState from 10-dimensional vector"""
        if len(vector) != 10:
            raise ValueError(f"Expected 10-dimensional vector, got {len(vector)}")
        
        return cls(
            account_equity_normalized=float(vector[0]),
            open_positions_count=int(vector[1]),
            volatility_regime=float(vector[2]),
            correlation_risk=float(vector[3]),
            var_estimate_5pct=float(vector[4]),
            current_drawdown_pct=float(vector[5]),
            margin_usage_pct=float(vector[6]),
            time_of_day_risk=float(vector[7]),
            market_stress_level=float(vector[8]),
            liquidity_conditions=float(vector[9])
        )


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for agent performance tracking"""
    total_risk_decisions: int
    risk_events_detected: int
    false_positive_rate: float
    avg_response_time_ms: float
    risk_adjusted_return: float
    max_drawdown: float
    sharpe_ratio: float
    var_accuracy: float
    correlation_prediction_accuracy: float


class BaseRiskAgent(BaseStrategicAgent):
    """
    Abstract base class for risk management agents.
    
    Specialized for risk calculations with:
    - 10-dimensional risk state vector input
    - Risk-specific action spaces (varies by agent)
    - Real-time performance monitoring <10ms
    - Integration with correlation tracker and VaR calculator
    - Event-driven risk response system
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Optional[EventBus] = None):
        """
        Initialize base risk agent
        
        Args:
            config: Risk agent configuration parameters
            event_bus: Event bus for real-time risk communication
        """
        # Override observation_dim for risk vector
        config['observation_dim'] = 10  # 10-dimensional risk state vector
        super().__init__(config)
        
        self.event_bus = event_bus
        self.risk_config = config.get('risk_config', {})
        
        # Risk-specific parameters
        self.max_response_time_ms = config.get('max_response_time_ms', 10.0)
        self.risk_tolerance = config.get('risk_tolerance', 0.02)  # 2% default
        self.enable_emergency_stop = config.get('enable_emergency_stop', True)
        
        # Performance tracking
        self.response_times = []
        self.risk_decisions_made = 0
        self.risk_events_detected = 0
        self.false_positives = 0
        
        # Risk calculation results cache
        self.last_risk_state: Optional[RiskState] = None
        self.last_action_time: Optional[datetime] = None
        
        # Event subscriptions
        if self.event_bus:
            self._setup_event_subscriptions()
        
        self.logger.info("Risk agent initialized",
                        name=self.name,
                        max_response_time=self.max_response_time_ms,
                        risk_tolerance=self.risk_tolerance)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for real-time risk updates"""
        if not self.event_bus:
            return
            
        # Subscribe to risk-relevant events
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_var_update)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_breach)
        self.event_bus.subscribe(EventType.MARKET_STRESS, self._handle_market_stress)
    
    def _handle_var_update(self, event: Event):
        """Handle VaR update events"""
        # Override in derived classes for specific VaR handling
        pass
    
    def _handle_position_update(self, event: Event):
        """Handle position update events"""
        # Override in derived classes for specific position handling
        pass
    
    def _handle_risk_breach(self, event: Event):
        """Handle risk breach events"""
        # Override in derived classes for specific risk breach handling
        pass
    
    def _handle_market_stress(self, event: Event):
        """Handle market stress events"""
        # Override in derived classes for specific market stress handling
        pass
    
    @abstractmethod
    def calculate_risk_action(self, risk_state: RiskState) -> Tuple[Union[int, np.ndarray], float]:
        """
        Calculate risk management action based on risk state
        
        Args:
            risk_state: Current risk state
            
        Returns:
            Tuple of (action, confidence)
            - action: Risk action (int for discrete, np.ndarray for continuous)
            - confidence: Confidence level (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def validate_risk_constraints(self, risk_state: RiskState) -> bool:
        """
        Validate current risk state against agent constraints
        
        Args:
            risk_state: Current risk state
            
        Returns:
            True if risk state is acceptable, False if action needed
        """
        pass
    
    def extract_features(self, observation_matrix: np.ndarray) -> np.ndarray:
        """
        Extract risk features from observation matrix.
        For risk agents, this typically just validates and returns the 10D risk vector.
        
        Args:
            observation_matrix: 10-dimensional risk state vector
            
        Returns:
            Validated risk state vector
        """
        # Risk agents expect 10D vector, not matrix
        if observation_matrix.ndim == 2:
            # If somehow we get a matrix, take the first row or flatten appropriately
            observation_vector = observation_matrix.flatten()[:10]
        else:
            observation_vector = observation_matrix[:10]
        
        if len(observation_vector) != 10:
            self.logger.warning("Invalid risk vector dimension",
                              expected=10,
                              actual=len(observation_vector))
            # Pad or truncate to 10 dimensions
            if len(observation_vector) < 10:
                observation_vector = np.pad(observation_vector, (0, 10 - len(observation_vector)))
            else:
                observation_vector = observation_vector[:10]
        
        return observation_vector
    
    def make_decision(self, features: np.ndarray) -> Tuple[Union[int, np.ndarray], float]:
        """
        Make risk management decision based on extracted features
        
        Args:
            features: 10-dimensional risk state vector
            
        Returns:
            Tuple of (action, confidence)
        """
        start_time = datetime.now()
        
        try:
            # Convert to RiskState
            risk_state = RiskState.from_vector(features)
            self.last_risk_state = risk_state
            
            # Validate risk constraints
            constraints_ok = self.validate_risk_constraints(risk_state)
            if not constraints_ok:
                self.risk_events_detected += 1
            
            # Calculate risk action
            action, confidence = self.calculate_risk_action(risk_state)
            
            # Track performance
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.response_times.append(response_time)
            self.risk_decisions_made += 1
            self.last_action_time = datetime.now()
            
            # Check response time performance
            if response_time > self.max_response_time_ms:
                self.logger.warning("Response time exceeded target",
                                  response_time=response_time,
                                  target=self.max_response_time_ms)
            
            # Publish risk decision event
            if self.event_bus:
                self.event_bus.publish(
                    self.event_bus.create_event(
                        EventType.RISK_UPDATE,
                        {
                            'agent': self.name,
                            'action': action,
                            'confidence': confidence,
                            'risk_state': risk_state,
                            'response_time_ms': response_time
                        },
                        self.name
                    )
                )
            
            return action, confidence
            
        except Exception as e:
            self.logger.error("Error in risk decision", error=str(e))
            # Return safe action with low confidence
            return self._get_safe_action(), 0.1
    
    def _get_safe_action(self) -> Union[int, np.ndarray]:
        """Get safe default action for error cases"""
        # Override in derived classes - default to no action
        return RiskAction.NO_ACTION.value
    
    def step_risk(self, risk_vector: np.ndarray) -> Tuple[Union[int, np.ndarray], float]:
        """
        Risk-specific step function taking 10D risk vector directly
        
        Args:
            risk_vector: 10-dimensional risk state vector
            
        Returns:
            Tuple of (action, confidence)
        """
        return self.step(risk_vector)
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get comprehensive risk performance metrics"""
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        false_positive_rate = (self.false_positives / max(1, self.risk_events_detected))
        
        return RiskMetrics(
            total_risk_decisions=self.risk_decisions_made,
            risk_events_detected=self.risk_events_detected,
            false_positive_rate=false_positive_rate,
            avg_response_time_ms=avg_response_time,
            risk_adjusted_return=0.0,  # To be calculated by derived classes
            max_drawdown=0.0,  # To be calculated by derived classes
            sharpe_ratio=0.0,  # To be calculated by derived classes
            var_accuracy=0.0,  # To be calculated by derived classes
            correlation_prediction_accuracy=0.0  # To be calculated by derived classes
        )
    
    def validate_observation(self, observation_vector: np.ndarray) -> bool:
        """
        Validate risk vector format
        
        Args:
            observation_vector: Input risk vector
            
        Returns:
            True if valid, False otherwise
        """
        if observation_vector is None:
            return False
        if not isinstance(observation_vector, np.ndarray):
            return False
        
        # For risk agents, expect 10D vector (not matrix)
        if observation_vector.ndim == 1:
            expected_shape = (10,)
        elif observation_vector.ndim == 2:
            # If matrix provided, should be (1, 10) or (10, 1)
            if observation_vector.shape not in [(1, 10), (10, 1)]:
                self.logger.warning("Invalid risk vector shape",
                                  expected="(10,), (1,10), or (10,1)",
                                  actual=observation_vector.shape)
                return False
        else:
            self.logger.warning("Invalid risk vector dimensions",
                              expected="1D or 2D",
                              actual=f"{observation_vector.ndim}D")
            return False
        
        if np.any(np.isnan(observation_vector)) or np.any(np.isinf(observation_vector)):
            self.logger.warning("Risk vector contains NaN or Inf values")
            return False
        
        return True
    
    def reset(self) -> None:
        """Reset risk agent state"""
        super().reset()
        self.response_times = []
        self.risk_decisions_made = 0
        self.risk_events_detected = 0
        self.false_positives = 0
        self.last_risk_state = None
        self.last_action_time = None
        self.logger.info("Risk agent reset")
    
    def emergency_stop(self, reason: str) -> bool:
        """
        Execute emergency stop protocol
        
        Args:
            reason: Reason for emergency stop
            
        Returns:
            True if emergency stop executed successfully
        """
        if not self.enable_emergency_stop:
            self.logger.warning("Emergency stop requested but disabled", reason=reason)
            return False
        
        self.logger.critical("EMERGENCY STOP EXECUTED", reason=reason)
        
        if self.event_bus:
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.EMERGENCY_STOP,
                    {
                        'agent': self.name,
                        'reason': reason,
                        'timestamp': datetime.now()
                    },
                    self.name
                )
            )
        
        return True
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name}(risk_decisions={self.risk_decisions_made})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}("
                f"name={self.name}, "
                f"risk_decisions={self.risk_decisions_made}, "
                f"response_time_target={self.max_response_time_ms}ms)")