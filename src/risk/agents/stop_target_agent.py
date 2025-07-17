"""
Stop/Target Agent (π₂) for Dynamic Stop-Loss and Take-Profit Management.

This module implements Agent 2 from the Risk Management MARL System PRD, providing
intelligent and adaptive stop-loss and take-profit level management with real-time
market condition adaptation.

Features:
- Box(0.5, 3.0, (2,)) continuous action space for [stop_multiplier, target_multiplier]
- ATR-based dynamic stop and target calculation
- Trailing stop mechanism with intelligent adaptation
- Volatility regime recognition and adjustment
- Time decay and position performance tracking
- <10ms response time guarantee

Technical Specifications:
- Action Space: Box(0.5, 3.0, (2,)) → [stop_multiplier, target_multiplier] × ATR
- Observation Space: Box(-∞, +∞, (16,)) → Extended risk vector + position context
- Performance Target: Dynamic stop-loss management with trailing capabilities
- Response Time: <10ms for stop/target adjustments

Author: Agent 3 - Stop/Target Agent Developer
Version: 1.0
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import structlog

# Import gymnasium with fallback
try:
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    
from .base_risk_agent import BaseRiskAgent, RiskState, RiskMetrics
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


class VolatilityRegime(Enum):
    """Market volatility regime classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class TrendMode(Enum):
    """Market trend mode classification"""
    RANGING = "ranging"
    TRENDING = "trending"
    REVERSAL = "reversal"


@dataclass
class PositionContext:
    """Extended position context for stop/target decisions"""
    entry_price: float
    current_price: float
    position_size: float
    time_in_trade_minutes: int
    unrealized_pnl_pct: float
    avg_true_range: float
    price_velocity: float
    volume_profile: float
    
    def to_vector(self) -> np.ndarray:
        """Convert position context to vector"""
        return np.array([
            self.entry_price,
            self.current_price,
            self.position_size,
            self.time_in_trade_minutes,
            self.unrealized_pnl_pct,
            self.avg_true_range,
            self.price_velocity,
            self.volume_profile
        ], dtype=np.float64)


@dataclass
class StopTargetLevels:
    """Current stop and target levels"""
    stop_loss_price: float
    take_profit_price: float
    stop_multiplier: float
    target_multiplier: float
    trailing_stop_active: bool
    last_update_time: datetime
    

class StopTargetAgent(BaseRiskAgent):
    """
    Stop/Target Agent (π₂) - Dynamic Stop-Loss and Take-Profit Management
    
    Implements intelligent stop-loss and take-profit level management with:
    - ATR-based dynamic calculation
    - Trailing stop mechanism
    - Volatility regime adaptation
    - Time decay adjustments
    - Market condition responsiveness
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Optional[EventBus] = None):
        """
        Initialize Stop/Target Agent
        
        Args:
            config: Agent configuration
            event_bus: Event bus for real-time communication
        """
        # Override observation_dim for extended risk vector + position context
        config['observation_dim'] = 16  # 10D risk vector + 6D position context
        config['action_dim'] = 2  # [stop_multiplier, target_multiplier]
        
        super().__init__(config, event_bus)
        
        # Create action space if gymnasium is available
        if HAS_GYMNASIUM:
            self.action_space = spaces.Box(
                low=np.array([0.5, 0.5]),
                high=np.array([3.0, 3.0]),
                dtype=np.float32
            )
        
        # Stop/Target specific configuration
        self.atr_period = config.get('atr_period', 14)
        self.min_stop_multiplier = config.get('min_stop_multiplier', 0.5)
        self.max_stop_multiplier = config.get('max_stop_multiplier', 3.0)
        self.min_target_multiplier = config.get('min_target_multiplier', 0.5)
        self.max_target_multiplier = config.get('max_target_multiplier', 3.0)
        
        # Trailing stop configuration
        self.enable_trailing_stops = config.get('enable_trailing_stops', True)
        self.trailing_activation_pct = config.get('trailing_activation_pct', 1.0)  # 1% profit
        self.trailing_step_pct = config.get('trailing_step_pct', 0.5)  # 0.5% steps
        
        # Volatility regime thresholds
        self.volatility_thresholds = {
            'low': config.get('vol_low_threshold', 0.25),
            'medium': config.get('vol_medium_threshold', 0.50),
            'high': config.get('vol_high_threshold', 0.75)
        }
        
        # Time decay configuration
        self.time_decay_enabled = config.get('enable_time_decay', True)
        self.max_hold_time_minutes = config.get('max_hold_time_minutes', 240)  # 4 hours
        self.time_decay_factor = config.get('time_decay_factor', 0.1)
        
        # Performance tracking
        self.stops_triggered = 0
        self.targets_hit = 0
        self.trailing_stops_activated = 0
        self.total_stop_adjustments = 0
        self.avg_hold_time_minutes = 0.0
        
        # Current state
        self.current_levels: Optional[StopTargetLevels] = None
        self.position_history: List[PositionContext] = []
        self.atr_values: List[float] = []
        
        self.logger.info("Stop/Target Agent initialized",
                        atr_period=self.atr_period,
                        trailing_enabled=self.enable_trailing_stops,
                        time_decay_enabled=self.time_decay_enabled)
    
    def calculate_atr(self, high_prices: List[float], 
                      low_prices: List[float], 
                      close_prices: List[float]) -> float:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high_prices: Recent high prices
            low_prices: Recent low prices  
            close_prices: Recent close prices
            
        Returns:
            Current ATR value
        """
        if len(high_prices) < 2:
            return 0.01  # Minimum ATR
        
        true_ranges = []
        for i in range(1, len(high_prices)):
            tr1 = high_prices[i] - low_prices[i]
            tr2 = abs(high_prices[i] - close_prices[i-1])
            tr3 = abs(low_prices[i] - close_prices[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) >= self.atr_period:
            atr = np.mean(true_ranges[-self.atr_period:])
        else:
            atr = np.mean(true_ranges)
        
        # Store for trend analysis
        self.atr_values.append(atr)
        if len(self.atr_values) > 100:
            self.atr_values = self.atr_values[-100:]
        
        return max(atr, 0.001)  # Minimum ATR threshold
    
    def detect_volatility_regime(self, risk_state: RiskState) -> VolatilityRegime:
        """
        Detect current volatility regime
        
        Args:
            risk_state: Current risk state
            
        Returns:
            Detected volatility regime
        """
        vol_percentile = risk_state.volatility_regime
        
        if vol_percentile <= self.volatility_thresholds['low']:
            return VolatilityRegime.LOW
        elif vol_percentile <= self.volatility_thresholds['medium']:
            return VolatilityRegime.MEDIUM
        elif vol_percentile <= self.volatility_thresholds['high']:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def detect_trend_mode(self, position_context: PositionContext) -> TrendMode:
        """
        Detect current trend mode
        
        Args:
            position_context: Current position context
            
        Returns:
            Detected trend mode
        """
        velocity = abs(position_context.price_velocity)
        
        if velocity < 0.001:
            return TrendMode.RANGING
        elif velocity > 0.01:
            return TrendMode.TRENDING
        else:
            return TrendMode.REVERSAL
    
    def calculate_volatility_adjustment(self, regime: VolatilityRegime) -> Tuple[float, float]:
        """
        Calculate volatility-based adjustments for stop and target multipliers
        
        Args:
            regime: Current volatility regime
            
        Returns:
            Tuple of (stop_adjustment, target_adjustment)
        """
        if regime == VolatilityRegime.LOW:
            return 0.8, 1.2  # Tighter stops, wider targets
        elif regime == VolatilityRegime.MEDIUM:
            return 1.0, 1.0  # Neutral
        elif regime == VolatilityRegime.HIGH:
            return 1.5, 0.8  # Wider stops, tighter targets
        else:  # EXTREME
            return 2.0, 0.6  # Much wider stops, much tighter targets
    
    def calculate_time_decay_adjustment(self, time_in_trade: int) -> float:
        """
        Calculate time-based adjustment to tighten stops over time
        
        Args:
            time_in_trade: Minutes in trade
            
        Returns:
            Time decay multiplier (< 1.0 tightens stops)
        """
        if not self.time_decay_enabled or time_in_trade <= 0:
            return 1.0
        
        # Gradually tighten stops as time progresses
        time_ratio = min(time_in_trade / self.max_hold_time_minutes, 1.0)
        decay_factor = 1.0 - (self.time_decay_factor * time_ratio)
        
        return max(decay_factor, 0.3)  # Never tighten more than 70%
    
    def calculate_trailing_stop(self, position_context: PositionContext,
                               current_stop: float) -> Tuple[float, bool]:
        """
        Calculate trailing stop adjustment
        
        Args:
            position_context: Current position context
            current_stop: Current stop level
            
        Returns:
            Tuple of (new_stop_level, trailing_activated)
        """
        if not self.enable_trailing_stops:
            return current_stop, False
        
        unrealized_pnl = position_context.unrealized_pnl_pct
        
        # Check if trailing should be activated
        if unrealized_pnl < self.trailing_activation_pct:
            return current_stop, False
        
        # Calculate new trailing stop level
        is_long = position_context.position_size > 0
        current_price = position_context.current_price
        
        if is_long:
            # For long positions, trail stop upward
            trail_distance = current_price * (self.trailing_step_pct / 100)
            new_stop = current_price - trail_distance
            
            # Only move stop up, never down
            if new_stop > current_stop:
                self.trailing_stops_activated += 1
                return new_stop, True
        else:
            # For short positions, trail stop downward
            trail_distance = current_price * (self.trailing_step_pct / 100)
            new_stop = current_price + trail_distance
            
            # Only move stop down, never up
            if new_stop < current_stop:
                self.trailing_stops_activated += 1
                return new_stop, True
        
        return current_stop, False
    
    def calculate_risk_action(self, risk_state: RiskState) -> Tuple[np.ndarray, float]:
        """
        Calculate stop/target multiplier action based on risk state
        
        Args:
            risk_state: Current risk state
            
        Returns:
            Tuple of (action_array, confidence)
            action_array: [stop_multiplier, target_multiplier]
        """
        try:
            # Default multipliers
            stop_multiplier = 1.5
            target_multiplier = 2.0
            confidence = 0.8
            
            # Detect volatility regime
            vol_regime = self.detect_volatility_regime(risk_state)
            
            # Apply volatility adjustments
            vol_stop_adj, vol_target_adj = self.calculate_volatility_adjustment(vol_regime)
            stop_multiplier *= vol_stop_adj
            target_multiplier *= vol_target_adj
            
            # Market stress adjustments
            if risk_state.market_stress_level > 0.7:
                stop_multiplier *= 1.3  # Wider stops during stress
                target_multiplier *= 0.8  # Tighter targets during stress
                confidence *= 0.9
            
            # Correlation risk adjustments
            if risk_state.correlation_risk > 0.8:
                stop_multiplier *= 1.2  # Wider stops for high correlation
                confidence *= 0.85
            
            # VaR-based adjustments
            if risk_state.var_estimate_5pct > 0.05:  # High VaR
                stop_multiplier *= 1.25
                target_multiplier *= 0.9
                confidence *= 0.8
            
            # Drawdown adjustments
            if risk_state.current_drawdown_pct > 0.1:  # 10% drawdown
                stop_multiplier *= 0.8  # Tighter stops during drawdown
                confidence *= 0.7
            
            # Margin usage adjustments
            if risk_state.margin_usage_pct > 0.8:  # High margin usage
                stop_multiplier *= 0.9  # Slightly tighter stops
                target_multiplier *= 1.1  # Slightly wider targets
            
            # Liquidity adjustments
            if risk_state.liquidity_conditions < 0.3:  # Poor liquidity
                stop_multiplier *= 1.4  # Much wider stops
                target_multiplier *= 0.7  # Tighter targets
                confidence *= 0.6
            
            # Clamp to valid ranges
            stop_multiplier = np.clip(stop_multiplier, self.min_stop_multiplier, self.max_stop_multiplier)
            target_multiplier = np.clip(target_multiplier, self.min_target_multiplier, self.max_target_multiplier)
            confidence = np.clip(confidence, 0.1, 1.0)
            
            action = np.array([stop_multiplier, target_multiplier], dtype=np.float32)
            
            self.total_stop_adjustments += 1
            
            return action, confidence
            
        except Exception as e:
            self.logger.error("Error calculating stop/target action", error=str(e))
            # Return conservative default
            return np.array([1.0, 1.5], dtype=np.float32), 0.3
    
    def calculate_stop_target_levels(self, position_context: PositionContext,
                                   action: np.ndarray) -> StopTargetLevels:
        """
        Calculate actual stop and target price levels
        
        Args:
            position_context: Current position context
            action: [stop_multiplier, target_multiplier]
            
        Returns:
            Calculated stop/target levels
        """
        stop_multiplier, target_multiplier = action
        entry_price = position_context.entry_price
        current_price = position_context.current_price
        atr = position_context.avg_true_range
        is_long = position_context.position_size > 0
        
        # Time decay adjustment
        time_decay = self.calculate_time_decay_adjustment(position_context.time_in_trade_minutes)
        stop_multiplier *= time_decay
        
        if is_long:
            # Long position
            stop_loss_price = entry_price - (atr * stop_multiplier)
            take_profit_price = entry_price + (atr * target_multiplier)
        else:
            # Short position
            stop_loss_price = entry_price + (atr * stop_multiplier)
            take_profit_price = entry_price - (atr * target_multiplier)
        
        # Check for trailing stop adjustment
        if self.current_levels:
            stop_loss_price, trailing_active = self.calculate_trailing_stop(
                position_context, stop_loss_price
            )
        else:
            trailing_active = False
        
        return StopTargetLevels(
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            stop_multiplier=stop_multiplier,
            target_multiplier=target_multiplier,
            trailing_stop_active=trailing_active,
            last_update_time=datetime.now()
        )
    
    def validate_risk_constraints(self, risk_state: RiskState) -> bool:
        """
        Validate risk constraints for stop/target management
        
        Args:
            risk_state: Current risk state
            
        Returns:
            True if constraints are satisfied
        """
        # Check for extreme risk conditions requiring immediate action
        if risk_state.var_estimate_5pct > 0.15:  # 15% VaR
            return False
        
        if risk_state.current_drawdown_pct > 0.2:  # 20% drawdown
            return False
        
        if risk_state.market_stress_level > 0.9:  # Extreme market stress
            return False
        
        if risk_state.correlation_risk > 0.95:  # Extreme correlation
            return False
        
        return True
    
    def step_position(self, risk_vector: np.ndarray, 
                     position_context: PositionContext) -> Tuple[StopTargetLevels, float]:
        """
        Position-specific step function with extended context
        
        Args:
            risk_vector: 10-dimensional risk state vector
            position_context: Current position context
            
        Returns:
            Tuple of (stop_target_levels, confidence)
        """
        # Combine risk vector with position context
        extended_observation = np.concatenate([
            risk_vector,
            position_context.to_vector()[:6]  # Take first 6 elements to make 16D total
        ])
        
        # Get action from base agent
        action, confidence = self.step(extended_observation)
        
        # Calculate actual stop/target levels
        levels = self.calculate_stop_target_levels(position_context, action)
        self.current_levels = levels
        
        # Track position history
        self.position_history.append(position_context)
        if len(self.position_history) > 1000:
            self.position_history = self.position_history[-1000:]
        
        return levels, confidence
    
    def make_decision(self, features: np.ndarray) -> Tuple[Union[int, np.ndarray], float]:
        """
        Override make_decision to handle 16D extended observation
        
        Args:
            features: 16-dimensional extended observation vector
            
        Returns:
            Tuple of (action, confidence)
        """
        start_time = datetime.now()
        
        try:
            # Extract risk state from first 10 dimensions
            risk_vector = features[:10]
            risk_state = RiskState.from_vector(risk_vector)
            self.last_risk_state = risk_state
            
            # Validate risk constraints
            constraints_ok = self.validate_risk_constraints(risk_state)
            if not constraints_ok:
                self.risk_events_detected += 1
            
            # Calculate risk action using full 16D observation
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

    def extract_features(self, observation_matrix: np.ndarray) -> np.ndarray:
        """
        Extract features from extended observation (16D)
        
        Args:
            observation_matrix: 16-dimensional extended observation
            
        Returns:
            Validated feature vector
        """
        # For stop/target agent, expect 16D vector
        if observation_matrix.ndim == 2:
            observation_vector = observation_matrix.flatten()[:16]
        else:
            observation_vector = observation_matrix[:16]
        
        if len(observation_vector) != 16:
            self.logger.warning("Invalid extended observation dimension",
                              expected=16,
                              actual=len(observation_vector))
            # Pad or truncate to 16 dimensions
            if len(observation_vector) < 16:
                observation_vector = np.pad(observation_vector, (0, 16 - len(observation_vector)))
            else:
                observation_vector = observation_vector[:16]
        
        return observation_vector
    
    def validate_observation(self, observation_vector: np.ndarray) -> bool:
        """
        Validate extended observation format (16D)
        
        Args:
            observation_vector: Input observation vector
            
        Returns:
            True if valid, False otherwise
        """
        if observation_vector is None:
            return False
        if not isinstance(observation_vector, np.ndarray):
            return False
        
        # For stop/target agent, expect 16D vector
        if observation_vector.ndim == 1:
            expected_shape = (16,)
        elif observation_vector.ndim == 2:
            if observation_vector.shape not in [(1, 16), (16, 1)]:
                self.logger.warning("Invalid extended observation shape",
                                  expected="(16,), (1,16), or (16,1)",
                                  actual=observation_vector.shape)
                return False
        else:
            return False
        
        if np.any(np.isnan(observation_vector)) or np.any(np.isinf(observation_vector)):
            return False
        
        return True
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get stop/target specific risk metrics"""
        base_metrics = super().get_risk_metrics()
        
        # Calculate stop/target specific metrics
        if self.position_history:
            avg_hold_time = np.mean([p.time_in_trade_minutes for p in self.position_history[-100:]])
        else:
            avg_hold_time = 0.0
        
        self.avg_hold_time_minutes = avg_hold_time
        
        # Override with stop/target specific calculations
        base_metrics.total_risk_decisions = self.total_stop_adjustments
        base_metrics.risk_events_detected = self.stops_triggered + self.targets_hit
        
        return base_metrics
    
    def _get_safe_action(self) -> np.ndarray:
        """Get safe default action for error cases"""
        return np.array([1.5, 2.0], dtype=np.float32)  # Conservative stop/target
    
    def emergency_stop_all_positions(self, reason: str) -> bool:
        """
        Emergency stop protocol - close all positions immediately
        
        Args:
            reason: Reason for emergency stop
            
        Returns:
            True if emergency stop executed
        """
        success = self.emergency_stop(reason)
        
        if success and self.event_bus:
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.POSITION_CLOSE_ALL,
                    {
                        'agent': self.name,
                        'reason': f"Emergency stop: {reason}",
                        'timestamp': datetime.now()
                    },
                    self.name
                )
            )
        
        return success
    
    def _handle_position_update(self, event: Event):
        """Handle position update events"""
        try:
            data = event.data
            if 'stop_triggered' in data:
                self.stops_triggered += 1
                self.logger.info("Stop loss triggered", position=data.get('position_id'))
            
            if 'target_hit' in data:
                self.targets_hit += 1
                self.logger.info("Take profit hit", position=data.get('position_id'))
                
        except Exception as e:
            self.logger.error("Error handling position update", error=str(e))
    
    def reset(self) -> None:
        """Reset stop/target agent state"""
        super().reset()
        self.stops_triggered = 0
        self.targets_hit = 0
        self.trailing_stops_activated = 0
        self.total_stop_adjustments = 0
        self.current_levels = None
        self.position_history = []
        self.atr_values = []
        self.logger.info("Stop/Target agent reset")
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name}(stops={self.stops_triggered}, targets={self.targets_hit})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"StopTargetAgent("
                f"stops_triggered={self.stops_triggered}, "
                f"targets_hit={self.targets_hit}, "
                f"trailing_active={self.trailing_stops_activated}, "
                f"adjustments={self.total_stop_adjustments})")