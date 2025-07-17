"""
Risk Management Agent (π₃) for Execution Engine MARL System

Implements comprehensive execution risk management with real-time monitoring,
ATR-based dynamic risk calculations, and emergency stop protocols.

Technical Specifications:
- Action Space: Box(2) continuous → [stop_loss_mult (0.5,3.0), take_profit_mult (1.0,5.0)]
- Observation Space: Box(-∞, +∞, (15,)) → Execution context + risk metrics
- Performance Target: <100μs risk checks, 100% compliance with position limits
- Neural Network: 15D input → 256→128→64→2 output

Key Features:
- ATR-based dynamic stop-loss and take-profit calculations
- Real-time VaR monitoring and position limit enforcement  
- Emergency stop protocols with risk override capabilities
- Ultra-low latency risk checks (<100μs target)
- Comprehensive risk metrics tracking and alerting

Author: Agent 3 - Risk Management Agent Implementation
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
import time
import structlog
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import deque

from src.core.event_bus import EventBus, Event, EventType
from src.core.events import Event as CoreEvent
from src.indicators.custom.atr import ATRIndicator, ATRReading

logger = structlog.get_logger()


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    
    @property
    def name_str(self):
        """Get string name for display"""
        names = {1: "low", 2: "moderate", 3: "high", 4: "critical", 5: "emergency"}
        return names.get(self.value, "unknown")


class RiskAction(Enum):
    """Risk management actions"""
    MAINTAIN = "maintain"
    REDUCE = "reduce"
    CLOSE = "close"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ExecutionRiskContext:
    """
    15-dimensional execution risk context
    
    Features:
    0-2: Position metrics (current_var, position_concentration, leverage_ratio)
    3-5: Market conditions (volatility_regime, correlation_risk, liquidity_stress)
    6-8: Performance metrics (unrealized_pnl_pct, drawdown_current, sharpe_ratio)
    9-11: Risk limits (var_limit_utilization, margin_utilization, position_limit_utilization)
    12-14: ATR metrics (atr_percentile, atr_trend, volatility_shock_indicator)
    """
    # Position metrics
    current_var: float = 0.0
    position_concentration: float = 0.0
    leverage_ratio: float = 0.0
    
    # Market conditions
    volatility_regime: float = 0.0
    correlation_risk: float = 0.0
    liquidity_stress: float = 0.0
    
    # Performance metrics
    unrealized_pnl_pct: float = 0.0
    drawdown_current: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk limits
    var_limit_utilization: float = 0.0
    margin_utilization: float = 0.0
    position_limit_utilization: float = 0.0
    
    # ATR metrics
    atr_percentile: float = 0.0
    atr_trend: float = 0.0
    volatility_shock_indicator: float = 0.0
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input"""
        return torch.tensor([
            self.current_var, self.position_concentration, self.leverage_ratio,
            self.volatility_regime, self.correlation_risk, self.liquidity_stress,
            self.unrealized_pnl_pct, self.drawdown_current, self.sharpe_ratio,
            self.var_limit_utilization, self.margin_utilization, self.position_limit_utilization,
            self.atr_percentile, self.atr_trend, self.volatility_shock_indicator
        ], dtype=torch.float32)


@dataclass
class RiskParameters:
    """ATR-based risk parameters"""
    stop_loss_multiplier: float
    take_profit_multiplier: float
    atr_value: float
    stop_loss_price: float
    take_profit_price: float
    risk_level: RiskLevel
    confidence: float
    calculation_time_ns: int


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_var_pct: float = 0.02  # 2% VaR limit
    max_leverage: float = 3.0
    max_position_concentration: float = 0.25  # 25% of portfolio
    max_correlation_risk: float = 0.8
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    emergency_var_pct: float = 0.05  # 5% emergency VaR threshold


class RiskManagementNetwork(nn.Module):
    """
    Ultra-fast neural network for risk management decisions
    
    Architecture: 15D input → 256→128→64→2 output
    Target inference time: <100μs
    """
    
    def __init__(self, input_dim: int = 15, hidden_dims: List[int] = None, output_dim: int = 2):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build sequential network for maximum speed
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),  # In-place for memory efficiency
                nn.LayerNorm(hidden_dim)  # Stabilizes training
            ])
            prev_dim = hidden_dim
            
        # Output layer with activation to ensure valid range
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Sigmoid()  # Output to (0,1), then scale to target ranges
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Initialize for fast convergence
        self._initialize_weights()
        
        # JIT compilation for speed
        self._compiled = False
        
    def _initialize_weights(self):
        """Initialize weights for fast convergence and stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimization for inference speed
        
        Args:
            x: Input tensor [batch_size, 15] or [15] for single inference
            
        Returns:
            Risk parameters [batch_size, 2] or [2]: [stop_loss_mult, take_profit_mult]
        """
        # Ensure correct input shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Forward pass
        output = self.network(x)
        
        # Scale outputs to target ranges:
        # stop_loss_mult: 0.5 to 3.0
        # take_profit_mult: 1.0 to 5.0
        stop_loss_mult = 0.5 + 2.5 * output[:, 0]  # Scale (0,1) to (0.5, 3.0)
        take_profit_mult = 1.0 + 4.0 * output[:, 1]  # Scale (0,1) to (1.0, 5.0)
        
        result = torch.stack([stop_loss_mult, take_profit_mult], dim=1)
        
        if squeeze_output:
            result = result.squeeze(0)
            
        return result
    
    def compile_for_inference(self):
        """Compile network for maximum inference speed"""
        if not self._compiled:
            try:
                # JIT script compilation
                example_input = torch.randn(1, self.input_dim)
                self.traced_model = torch.jit.trace(self, example_input)
                self._compiled = True
                logger.info("Risk management network compiled for inference")
            except Exception as e:
                logger.warning("Failed to compile network", error=str(e))
                
    def fast_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Ultra-fast inference path"""
        if self._compiled and hasattr(self, 'traced_model'):
            return self.traced_model(x.unsqueeze(0) if x.dim() == 1 else x)
        else:
            return self.forward(x)


class RiskMonitor:
    """
    Real-time risk monitoring system with ultra-low latency checks
    """
    
    def __init__(self, limits: RiskLimits, event_bus: Optional[EventBus] = None):
        self.limits = limits
        self.event_bus = event_bus
        
        # Performance tracking
        self.check_times = deque(maxlen=10000)
        self.violations_detected = 0
        self.emergency_stops_triggered = 0
        
        # Current state
        self.last_risk_check: Optional[datetime] = None
        self.last_risk_level = RiskLevel.LOW
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Risk monitor initialized", limits=limits)
    
    def check_risk_limits(self, context: ExecutionRiskContext) -> Tuple[RiskLevel, List[str]]:
        """
        Ultra-fast risk limit checking (<100μs target)
        
        Args:
            context: Current execution risk context
            
        Returns:
            Tuple of (risk_level, violations_list)
        """
        start_time = time.perf_counter_ns()
        
        try:
            with self._lock:
                violations = []
                risk_level = RiskLevel.LOW
                
                # VaR checks
                if context.current_var > self.limits.emergency_var_pct:
                    violations.append(f"EMERGENCY: VaR {context.current_var:.3f} > {self.limits.emergency_var_pct:.3f}")
                    risk_level = RiskLevel.EMERGENCY
                elif context.current_var > self.limits.max_var_pct:
                    violations.append(f"VaR limit exceeded: {context.current_var:.3f} > {self.limits.max_var_pct:.3f}")
                    risk_level = risk_level if risk_level > RiskLevel.CRITICAL else RiskLevel.CRITICAL
                
                # Leverage checks
                if context.leverage_ratio > self.limits.max_leverage:
                    violations.append(f"Leverage limit exceeded: {context.leverage_ratio:.2f} > {self.limits.max_leverage:.2f}")
                    risk_level = risk_level if risk_level > RiskLevel.HIGH else RiskLevel.HIGH
                
                # Position concentration checks
                if context.position_concentration > self.limits.max_position_concentration:
                    violations.append(f"Position concentration limit exceeded: {context.position_concentration:.3f} > {self.limits.max_position_concentration:.3f}")
                    risk_level = risk_level if risk_level > RiskLevel.HIGH else RiskLevel.HIGH
                
                # Correlation risk checks
                if context.correlation_risk > self.limits.max_correlation_risk:
                    violations.append(f"Correlation risk limit exceeded: {context.correlation_risk:.3f} > {self.limits.max_correlation_risk:.3f}")
                    risk_level = risk_level if risk_level > RiskLevel.MODERATE else RiskLevel.MODERATE
                
                # Drawdown checks
                if context.drawdown_current > self.limits.max_drawdown_pct:
                    violations.append(f"Drawdown limit exceeded: {context.drawdown_current:.3f} > {self.limits.max_drawdown_pct:.3f}")
                    risk_level = risk_level if risk_level > RiskLevel.CRITICAL else RiskLevel.CRITICAL
                
                # Volatility shock checks
                if context.volatility_shock_indicator > 0.8:
                    violations.append(f"Volatility shock detected: {context.volatility_shock_indicator:.3f}")
                    risk_level = risk_level if risk_level > RiskLevel.HIGH else RiskLevel.HIGH
                
                # Liquidity stress checks
                if context.liquidity_stress > 0.7:
                    violations.append(f"Liquidity stress detected: {context.liquidity_stress:.3f}")
                    risk_level = risk_level if risk_level > RiskLevel.MODERATE else RiskLevel.MODERATE
                
                # Track violations
                if violations:
                    self.violations_detected += 1
                
                self.last_risk_level = risk_level
                self.last_risk_check = datetime.now()
                
                # Performance tracking
                end_time = time.perf_counter_ns()
                check_time_ns = end_time - start_time
                self.check_times.append(check_time_ns)
                
                # Log slow checks
                if check_time_ns > 100_000:  # 100μs threshold
                    logger.warning("Slow risk check", check_time_ns=check_time_ns, target_ns=100_000)
                
                return risk_level, violations
                
        except Exception as e:
            logger.error("Error in risk limit check", error=str(e))
            return RiskLevel.EMERGENCY, [f"Risk check error: {str(e)}"]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get risk monitor performance statistics"""
        if not self.check_times:
            return {}
        
        check_times_us = [t / 1000 for t in self.check_times]
        
        return {
            'total_checks': len(self.check_times),
            'violations_detected': self.violations_detected,
            'emergency_stops': self.emergency_stops_triggered,
            'avg_check_time_ns': np.mean(self.check_times),
            'avg_check_time_us': np.mean(check_times_us),
            'max_check_time_us': np.max(check_times_us),
            'p95_check_time_us': np.percentile(check_times_us, 95),
            'p99_check_time_us': np.percentile(check_times_us, 99),
            'target_100us_met': np.mean(self.check_times) < 100_000,
            'last_risk_level': self.last_risk_level.name_str,
            'last_check_time': self.last_risk_check
        }


class RiskManagementAgent:
    """
    Risk Management Agent (π₃) for Execution Engine MARL System
    
    High-performance agent for execution risk management with:
    - ATR-based dynamic risk parameter calculation
    - <100μs risk checks with real-time monitoring
    - VaR monitoring and position limit enforcement
    - Emergency stop protocols with risk override capabilities
    - Comprehensive risk metrics tracking and alerting
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Optional[EventBus] = None):
        """
        Initialize Risk Management Agent
        
        Args:
            config: Agent configuration
            event_bus: Event bus for communication
        """
        self.config = config
        self.event_bus = event_bus
        
        # Agent parameters
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.min_stop_multiplier = config.get('min_stop_multiplier', 0.5)
        self.max_stop_multiplier = config.get('max_stop_multiplier', 3.0)
        self.min_take_profit_multiplier = config.get('min_take_profit_multiplier', 1.0)
        self.max_take_profit_multiplier = config.get('max_take_profit_multiplier', 5.0)
        
        # Risk limits
        self.risk_limits = RiskLimits(
            max_var_pct=config.get('max_var_pct', 0.02),
            max_leverage=config.get('max_leverage', 3.0),
            max_position_concentration=config.get('max_position_concentration', 0.25),
            max_correlation_risk=config.get('max_correlation_risk', 0.8),
            max_drawdown_pct=config.get('max_drawdown_pct', 0.15),
            emergency_var_pct=config.get('emergency_var_pct', 0.05)
        )
        
        # Initialize components
        self.network = RiskManagementNetwork(
            input_dim=15,
            hidden_dims=config.get('hidden_dims', [256, 128, 64]),
            output_dim=2
        )
        
        self.risk_monitor = RiskMonitor(self.risk_limits, event_bus)
        
        # ATR indicator for risk calculations (optional)
        self.atr_indicator = None  # Will be set up separately if needed
        
        # Performance tracking
        self.inference_times = deque(maxlen=10000)
        self.risk_decisions_made = 0
        self.emergency_stops_executed = 0
        self.risk_parameters_updated = 0
        
        # Current state
        self.current_risk_parameters: Optional[RiskParameters] = None
        self.last_atr_reading: Optional[ATRReading] = None
        
        # Compile network for speed
        self.network.compile_for_inference()
        
        logger.info("Risk Management Agent (π₃) initialized",
                   risk_limits=self.risk_limits,
                   min_stop_mult=self.min_stop_multiplier,
                   max_stop_mult=self.max_stop_multiplier,
                   min_tp_mult=self.min_take_profit_multiplier,
                   max_tp_mult=self.max_take_profit_multiplier)
    
    def calculate_risk_parameters(self, 
                                execution_context: ExecutionRiskContext,
                                current_price: float,
                                position_size: float) -> Tuple[RiskParameters, Dict[str, Any]]:
        """
        Calculate ATR-based dynamic risk parameters
        
        Args:
            execution_context: 15-dimensional execution risk context
            current_price: Current market price
            position_size: Current position size
            
        Returns:
            Tuple of (risk_parameters, decision_info)
        """
        start_time = time.perf_counter_ns()
        
        try:
            # Convert context to tensor
            context_tensor = execution_context.to_tensor()
            
            # Neural network inference
            with torch.no_grad():
                risk_multipliers = self.network.fast_inference(context_tensor)
                
            # Handle tensor output properly
            if risk_multipliers.dim() > 1:
                risk_multipliers = risk_multipliers.squeeze(0)
            
            risk_multipliers_np = risk_multipliers.numpy()
            stop_loss_mult, take_profit_mult = risk_multipliers_np[0], risk_multipliers_np[1]
            
            # Get ATR value for calculations
            atr_value = self._get_current_atr(execution_context)
            
            # Calculate actual stop/target prices
            is_long_position = position_size > 0
            
            if is_long_position:
                stop_loss_price = current_price - (atr_value * stop_loss_mult)
                take_profit_price = current_price + (atr_value * take_profit_mult)
            else:
                stop_loss_price = current_price + (atr_value * stop_loss_mult)
                take_profit_price = current_price - (atr_value * take_profit_mult)
            
            # Perform risk checks
            risk_level, violations = self.risk_monitor.check_risk_limits(execution_context)
            
            # Calculate confidence based on risk conditions
            confidence = self._calculate_confidence(execution_context, risk_level)
            
            # Performance tracking
            end_time = time.perf_counter_ns()
            calculation_time_ns = end_time - start_time
            self.inference_times.append(calculation_time_ns)
            self.risk_decisions_made += 1
            
            # Create risk parameters
            risk_parameters = RiskParameters(
                stop_loss_multiplier=stop_loss_mult,
                take_profit_multiplier=take_profit_mult,
                atr_value=atr_value,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                risk_level=risk_level,
                confidence=confidence,
                calculation_time_ns=calculation_time_ns
            )
            
            self.current_risk_parameters = risk_parameters
            self.risk_parameters_updated += 1
            
            # Decision info
            decision_info = {
                'stop_loss_multiplier': stop_loss_mult,
                'take_profit_multiplier': take_profit_mult,
                'atr_value': atr_value,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'risk_level': risk_level.name_str,
                'confidence': confidence,
                'violations': violations,
                'calculation_time_ns': calculation_time_ns,
                'calculation_time_us': calculation_time_ns / 1000,
                'position_direction': 'long' if is_long_position else 'short'
            }
            
            # Log performance if needed
            if self.risk_decisions_made % 1000 == 0:
                self._log_performance_stats()
                
            # Emit event if event bus available
            if self.event_bus:
                self._emit_risk_decision_event(risk_parameters, decision_info)
                
            # Handle emergency conditions
            if risk_level == RiskLevel.EMERGENCY:
                self._handle_emergency_condition(violations)
                
            return risk_parameters, decision_info
            
        except Exception as e:
            logger.error("Error calculating risk parameters", error=str(e))
            return self._get_emergency_risk_parameters(current_price, position_size), {'error': str(e)}
    
    def _get_current_atr(self, context: ExecutionRiskContext) -> float:
        """Get current ATR value for calculations"""
        # If we have a real ATR indicator, use it
        if self.atr_indicator and self.atr_indicator.get_current_atr():
            return self.atr_indicator.get_current_atr()
        
        # Otherwise estimate from context ATR percentile
        # Assume typical price is 100 and scale by percentile
        base_atr = 1.0  # 1% of price as base ATR
        percentile_multiplier = 0.5 + (context.atr_percentile / 100.0)  # 0.5 to 1.5x
        
        return base_atr * percentile_multiplier
    
    def _calculate_confidence(self, context: ExecutionRiskContext, risk_level: RiskLevel) -> float:
        """Calculate confidence based on risk conditions"""
        base_confidence = 0.8
        
        # Reduce confidence based on risk level
        if risk_level == RiskLevel.EMERGENCY:
            base_confidence *= 0.3
        elif risk_level == RiskLevel.CRITICAL:
            base_confidence *= 0.5
        elif risk_level == RiskLevel.HIGH:
            base_confidence *= 0.7
        elif risk_level == RiskLevel.MODERATE:
            base_confidence *= 0.85
        
        # Adjust for specific risk factors
        if context.volatility_shock_indicator > 0.7:
            base_confidence *= 0.8
        
        if context.liquidity_stress > 0.6:
            base_confidence *= 0.9
        
        if context.correlation_risk > 0.7:
            base_confidence *= 0.85
        
        if context.drawdown_current > 0.1:
            base_confidence *= 0.8
        
        return np.clip(base_confidence, 0.1, 1.0)
    
    def _handle_emergency_condition(self, violations: List[str]):
        """Handle emergency risk conditions"""
        logger.critical("EMERGENCY RISK CONDITION DETECTED", violations=violations)
        
        self.emergency_stops_executed += 1
        
        if self.event_bus:
            # Emit emergency stop event
            emergency_event = CoreEvent(
                type=EventType.RISK_EMERGENCY,
                data={
                    'agent': 'risk_management_agent',
                    'violations': violations,
                    'action': 'emergency_stop',
                    'timestamp': time.time()
                }
            )
            self.event_bus.emit(emergency_event)
    
    def _get_emergency_risk_parameters(self, current_price: float, position_size: float) -> RiskParameters:
        """Get emergency risk parameters for error cases"""
        # Very tight stops in emergency
        stop_mult = 0.5
        tp_mult = 1.0
        atr_value = current_price * 0.005  # 0.5% emergency ATR
        
        is_long = position_size > 0
        if is_long:
            stop_price = current_price - (atr_value * stop_mult)
            tp_price = current_price + (atr_value * tp_mult)
        else:
            stop_price = current_price + (atr_value * stop_mult)
            tp_price = current_price - (atr_value * tp_mult)
        
        return RiskParameters(
            stop_loss_multiplier=stop_mult,
            take_profit_multiplier=tp_mult,
            atr_value=atr_value,
            stop_loss_price=stop_price,
            take_profit_price=tp_price,
            risk_level=RiskLevel.EMERGENCY,
            confidence=0.1,
            calculation_time_ns=0
        )
    
    def _emit_risk_decision_event(self, risk_parameters: RiskParameters, decision_info: Dict[str, Any]):
        """Emit risk management decision event"""
        try:
            event = CoreEvent(
                type=EventType.AGENT_DECISION,
                data={
                    'agent': 'risk_management_agent',
                    'risk_parameters': risk_parameters,
                    'decision_info': decision_info,
                    'timestamp': time.time()
                }
            )
            self.event_bus.emit(event)
        except Exception as e:
            logger.warning("Failed to emit risk decision event", error=str(e))
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        if not self.inference_times:
            return
            
        avg_time_ns = np.mean(list(self.inference_times))
        max_time_ns = np.max(list(self.inference_times))
        p95_time_ns = np.percentile(list(self.inference_times), 95)
        
        risk_stats = self.risk_monitor.get_performance_stats()
        
        logger.info("Risk Management Agent Performance",
                   decisions_made=self.risk_decisions_made,
                   emergency_stops=self.emergency_stops_executed,
                   avg_inference_time_us=avg_time_ns / 1000,
                   max_inference_time_us=max_time_ns / 1000,
                   p95_inference_time_us=p95_time_ns / 1000,
                   target_100us_met=avg_time_ns < 100_000,
                   risk_monitor_stats=risk_stats)
    
    def check_position_limits(self, 
                            current_positions: Dict[str, float],
                            proposed_position: float,
                            symbol: str) -> Tuple[bool, float, str]:
        """
        Check if proposed position complies with risk limits
        
        Args:
            current_positions: Current position sizes by symbol
            proposed_position: Proposed new position size
            symbol: Trading symbol
            
        Returns:
            Tuple of (allowed, adjusted_position, reason)
        """
        try:
            # Calculate total portfolio exposure
            total_exposure = sum(abs(pos) for pos in current_positions.values())
            new_exposure = total_exposure - abs(current_positions.get(symbol, 0)) + abs(proposed_position)
            
            # Check concentration limits
            position_concentration = abs(proposed_position) / max(new_exposure, 1.0)
            
            if position_concentration > self.risk_limits.max_position_concentration:
                # Adjust position to fit within concentration limits
                max_allowed = new_exposure * self.risk_limits.max_position_concentration
                adjusted_position = np.sign(proposed_position) * max_allowed
                return False, adjusted_position, f"Position concentration limit exceeded"
            
            # Check leverage limits (simplified)
            if new_exposure > self.risk_limits.max_leverage * 100000:  # Assuming 100k account
                return False, 0, "Leverage limit exceeded"
            
            return True, proposed_position, "Position allowed"
            
        except Exception as e:
            logger.error("Error checking position limits", error=str(e))
            return False, 0, f"Error in position check: {str(e)}"
    
    def monitor_var_limits(self, current_var: float) -> Tuple[bool, RiskAction]:
        """
        Monitor VaR limits and recommend actions
        
        Args:
            current_var: Current Value at Risk
            
        Returns:
            Tuple of (limit_ok, recommended_action)
        """
        if current_var > self.risk_limits.emergency_var_pct:
            return False, RiskAction.EMERGENCY_STOP
        elif current_var > self.risk_limits.max_var_pct:
            return False, RiskAction.CLOSE
        elif current_var > self.risk_limits.max_var_pct * 0.8:
            return True, RiskAction.REDUCE
        else:
            return True, RiskAction.MAINTAIN
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.inference_times:
            inference_metrics = {}
        else:
            recent_times = list(self.inference_times)
            inference_metrics = {
                'avg_inference_time_ns': np.mean(recent_times),
                'avg_inference_time_us': np.mean(recent_times) / 1000,
                'max_inference_time_us': np.max(recent_times) / 1000,
                'p50_inference_time_us': np.percentile(recent_times, 50) / 1000,
                'p95_inference_time_us': np.percentile(recent_times, 95) / 1000,
                'p99_inference_time_us': np.percentile(recent_times, 99) / 1000,
                'target_100us_met': np.mean(recent_times) < 100_000
            }
        
        risk_monitor_stats = self.risk_monitor.get_performance_stats()
        
        return {
            'total_decisions': self.risk_decisions_made,
            'risk_parameters_updated': self.risk_parameters_updated,
            'emergency_stops_executed': self.emergency_stops_executed,
            'network_compiled': self.network._compiled,
            'current_risk_level': self.risk_monitor.last_risk_level.name_str if self.risk_monitor.last_risk_level else 'unknown',
            'inference_metrics': inference_metrics,
            'risk_monitor_metrics': risk_monitor_stats
        }
    
    def update_risk_limits(self, new_limits: Dict[str, float]):
        """Update risk limits dynamically"""
        for key, value in new_limits.items():
            if hasattr(self.risk_limits, key):
                setattr(self.risk_limits, key, value)
                logger.info("Risk limit updated", parameter=key, new_value=value)
    
    def emergency_stop_all_positions(self, reason: str) -> bool:
        """
        Execute emergency stop for all positions
        
        Args:
            reason: Reason for emergency stop
            
        Returns:
            True if emergency stop executed successfully
        """
        try:
            logger.critical("EXECUTING EMERGENCY STOP", reason=reason)
            
            self.emergency_stops_executed += 1
            
            if self.event_bus:
                emergency_event = CoreEvent(
                    type=EventType.RISK_EMERGENCY,
                    data={
                        'agent': 'risk_management_agent',
                        'action': 'emergency_stop_all',
                        'reason': reason,
                        'timestamp': time.time()
                    }
                )
                self.event_bus.emit(emergency_event)
            
            return True
            
        except Exception as e:
            logger.error("Error executing emergency stop", error=str(e), reason=reason)
            return False
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.inference_times.clear()
        self.risk_decisions_made = 0
        self.emergency_stops_executed = 0
        self.risk_parameters_updated = 0
        self.current_risk_parameters = None
        self.risk_monitor.check_times.clear()
        self.risk_monitor.violations_detected = 0
        self.risk_monitor.emergency_stops_triggered = 0
        logger.info("Risk Management Agent metrics reset")
    
    def __str__(self) -> str:
        """String representation"""
        return f"RiskManagementAgent(decisions={self.risk_decisions_made}, emergencies={self.emergency_stops_executed})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"RiskManagementAgent("
                f"decisions={self.risk_decisions_made}, "
                f"emergencies={self.emergency_stops_executed}, "
                f"current_risk={self.risk_monitor.last_risk_level.name_str if self.risk_monitor.last_risk_level else 'unknown'})")


# Factory function for easy instantiation
def create_risk_management_agent(config: Dict[str, Any], event_bus: Optional[EventBus] = None) -> RiskManagementAgent:
    """Create and initialize a Risk Management Agent"""
    return RiskManagementAgent(config, event_bus)


# Performance benchmark for validation
def benchmark_risk_management_performance(agent: RiskManagementAgent, 
                                        num_iterations: int = 10000) -> Dict[str, Any]:
    """
    Benchmark risk management agent performance
    
    Args:
        agent: Risk management agent to benchmark
        num_iterations: Number of iterations to run
        
    Returns:
        Performance benchmark results
    """
    logger.info("Starting risk management performance benchmark", iterations=num_iterations)
    
    # Reset metrics
    agent.reset_metrics()
    
    # Generate random execution contexts for testing
    contexts = []
    for _ in range(num_iterations):
        context = ExecutionRiskContext(
            current_var=np.random.uniform(0.005, 0.03),
            position_concentration=np.random.uniform(0.05, 0.4),
            leverage_ratio=np.random.uniform(1.0, 4.0),
            volatility_regime=np.random.uniform(0.2, 0.9),
            correlation_risk=np.random.uniform(0.1, 0.9),
            liquidity_stress=np.random.uniform(0.0, 0.8),
            unrealized_pnl_pct=np.random.uniform(-0.1, 0.1),
            drawdown_current=np.random.uniform(0.0, 0.2),
            sharpe_ratio=np.random.uniform(-2.0, 3.0),
            var_limit_utilization=np.random.uniform(0.3, 1.0),
            margin_utilization=np.random.uniform(0.2, 0.9),
            position_limit_utilization=np.random.uniform(0.1, 0.8),
            atr_percentile=np.random.uniform(10, 90),
            atr_trend=np.random.uniform(-0.5, 0.5),
            volatility_shock_indicator=np.random.uniform(0.0, 1.0)
        )
        contexts.append(context)
    
    # Run benchmark
    start_time = time.perf_counter()
    
    for context in contexts:
        risk_params, decision_info = agent.calculate_risk_parameters(
            execution_context=context,
            current_price=100.0,  # Standard test price
            position_size=1.0     # Standard test position
        )
    
    end_time = time.perf_counter()
    total_time_s = end_time - start_time
    
    # Get performance metrics
    metrics = agent.get_performance_metrics()
    
    # Benchmark results
    benchmark_results = {
        'total_iterations': num_iterations,
        'total_time_s': total_time_s,
        'iterations_per_second': num_iterations / total_time_s,
        'avg_time_per_iteration_us': (total_time_s / num_iterations) * 1_000_000,
        'target_100us_met': metrics.get('inference_metrics', {}).get('target_100us_met', False),
        'performance_metrics': metrics
    }
    
    logger.info("Risk management performance benchmark completed",
               iterations_per_second=benchmark_results['iterations_per_second'],
               avg_time_us=benchmark_results['avg_time_per_iteration_us'],
               target_met=benchmark_results['target_100us_met'])
    
    return benchmark_results