"""
Sequential Risk Agents

This module implements sequential-aware risk agents that operate in the specific sequence:
position sizing → stop/target → risk monitor → portfolio optimizer.

Each agent is designed to:
1. Receive enriched context from predecessors
2. Make informed decisions based on prior agent outputs
3. Maintain consistency with the overall risk management strategy
4. Integrate seamlessly with VaR correlation system

Agent Architecture:
- Base: Sequential-aware neural network architectures
- Context: Rich context processing from predecessor agents
- Decision: Sequential decision making with constraint satisfaction
- Integration: VaR correlation system integration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
import structlog

from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.risk.core.correlation_tracker import CorrelationTracker
from src.risk.core.var_calculator import VaRCalculator
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


@dataclass
class SequentialContext:
    """Context passed between sequential agents"""
    position_decisions: Dict[str, float]
    stop_target_decisions: Dict[str, Tuple[float, float]]
    risk_monitor_decisions: Dict[str, float]
    portfolio_optimization_decisions: Dict[str, float]
    risk_state: RiskState
    correlation_context: Dict[str, Any]
    var_context: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


class SequentialRiskAgent(BaseRiskAgent, ABC):
    """
    Base class for sequential risk agents
    
    Extends BaseRiskAgent with sequential-specific functionality:
    - Context processing from predecessor agents
    - Sequential decision making
    - Constraint satisfaction
    - Performance monitoring
    """
    
    def __init__(self, 
                 agent_id: str,
                 config: Dict[str, Any],
                 event_bus: EventBus,
                 correlation_tracker: CorrelationTracker,
                 var_calculator: VaRCalculator):
        super().__init__(agent_id, config, event_bus)
        
        self.correlation_tracker = correlation_tracker
        self.var_calculator = var_calculator
        self.sequence_position = self._get_sequence_position()
        
        # Sequential context processing
        self.context_processor = self._create_context_processor()
        self.constraint_validator = self._create_constraint_validator()
        
        # Performance tracking
        self.decision_times = []
        self.context_processing_times = []
        self.constraint_violations = []
        
        # Agent-specific configuration
        self.max_context_age_seconds = config.get('max_context_age_seconds', 10)
        self.context_validation_threshold = config.get('context_validation_threshold', 0.8)
        
        logger.info(f"Sequential Risk Agent {agent_id} initialized",
                   sequence_position=self.sequence_position,
                   config_keys=list(config.keys()))
    
    @abstractmethod
    def _get_sequence_position(self) -> int:
        """Get position in sequential execution order"""
        pass
    
    @abstractmethod
    def _create_context_processor(self) -> nn.Module:
        """Create context processor for predecessor outputs"""
        pass
    
    @abstractmethod
    def _create_constraint_validator(self) -> 'ConstraintValidator':
        """Create constraint validator for sequential consistency"""
        pass
    
    def process_sequential_context(self, context: SequentialContext) -> torch.Tensor:
        """Process sequential context from predecessor agents"""
        start_time = datetime.now()
        
        # Validate context freshness
        if (datetime.now() - context.timestamp).total_seconds() > self.max_context_age_seconds:
            logger.warning(f"Stale context received by {self.agent_id}",
                          context_age_seconds=(datetime.now() - context.timestamp).total_seconds())
        
        # Process context through neural network
        context_tensor = self.context_processor(self._encode_context(context))
        
        # Track processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.context_processing_times.append(processing_time)
        
        return context_tensor
    
    def _encode_context(self, context: SequentialContext) -> torch.Tensor:
        """Encode sequential context into tensor format"""
        # Base risk state
        risk_vector = torch.tensor(context.risk_state.to_vector(), dtype=torch.float32)
        
        # Predecessor decisions
        position_vector = torch.tensor(list(context.position_decisions.values()), dtype=torch.float32)
        
        # Stop/target decisions (flatten tuples)
        stop_target_flat = []
        for stop, target in context.stop_target_decisions.values():
            stop_target_flat.extend([stop, target])
        stop_target_vector = torch.tensor(stop_target_flat, dtype=torch.float32)
        
        # Risk monitor decisions
        risk_monitor_vector = torch.tensor(list(context.risk_monitor_decisions.values()), dtype=torch.float32)
        
        # Portfolio optimization decisions
        portfolio_vector = torch.tensor(list(context.portfolio_optimization_decisions.values()), dtype=torch.float32)
        
        # Correlation context
        correlation_values = [
            context.correlation_context.get('avg_correlation', 0.0),
            context.correlation_context.get('max_correlation', 0.0),
            float(context.correlation_context.get('regime', 0)),  # Encoded regime
        ]
        correlation_vector = torch.tensor(correlation_values, dtype=torch.float32)
        
        # VaR context
        var_values = [
            context.var_context.get('portfolio_var', 0.0),
            context.var_context.get('var_percentage', 0.0),
            context.var_context.get('calculation_time_ms', 0.0),
        ]
        var_vector = torch.tensor(var_values, dtype=torch.float32)
        
        # Concatenate all vectors
        context_tensor = torch.cat([
            risk_vector,
            position_vector,
            stop_target_vector,
            risk_monitor_vector,
            portfolio_vector,
            correlation_vector,
            var_vector
        ])
        
        return context_tensor
    
    def validate_constraints(self, action: RiskAction, context: SequentialContext) -> bool:
        """Validate action against sequential constraints"""
        try:
            return self.constraint_validator.validate(action, context)
        except Exception as e:
            logger.error(f"Constraint validation failed for {self.agent_id}", error=str(e))
            self.constraint_violations.append(str(e))
            return False
    
    def get_sequential_performance_metrics(self) -> Dict[str, Any]:
        """Get sequential-specific performance metrics"""
        base_metrics = self.get_performance_metrics()
        
        sequential_metrics = {
            'sequence_position': self.sequence_position,
            'avg_decision_time_ms': np.mean(self.decision_times) if self.decision_times else 0,
            'avg_context_processing_time_ms': np.mean(self.context_processing_times) if self.context_processing_times else 0,
            'constraint_violations': len(self.constraint_violations),
            'context_validation_rate': 1.0 - len(self.constraint_violations) / max(len(self.decision_times), 1)
        }
        
        return {**base_metrics, **sequential_metrics}


class ConstraintValidator:
    """Validates sequential constraints across agents"""
    
    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints
        self.violation_history = []
    
    def validate(self, action: RiskAction, context: SequentialContext) -> bool:
        """Validate action against constraints"""
        violations = []
        
        # Check position size constraints
        if hasattr(action, 'position_adjustments'):
            for asset, adjustment in action.position_adjustments.items():
                if abs(adjustment) > self.constraints.get('max_position_adjustment', 0.5):
                    violations.append(f"Position adjustment {adjustment} exceeds limit for {asset}")
        
        # Check stop/target constraints
        if hasattr(action, 'stop_multiplier') and hasattr(action, 'target_multiplier'):
            if action.stop_multiplier > self.constraints.get('max_stop_multiplier', 5.0):
                violations.append(f"Stop multiplier {action.stop_multiplier} exceeds limit")
            if action.target_multiplier > self.constraints.get('max_target_multiplier', 10.0):
                violations.append(f"Target multiplier {action.target_multiplier} exceeds limit")
        
        # Check risk monitor constraints
        if hasattr(action, 'risk_reduction_factor'):
            if action.risk_reduction_factor > self.constraints.get('max_risk_reduction', 0.8):
                violations.append(f"Risk reduction {action.risk_reduction_factor} exceeds limit")
        
        # Check portfolio constraints
        if hasattr(action, 'portfolio_weights'):
            weight_sum = sum(action.portfolio_weights.values())
            if abs(weight_sum - 1.0) > self.constraints.get('weight_sum_tolerance', 0.01):
                violations.append(f"Portfolio weights sum to {weight_sum}, not 1.0")
        
        self.violation_history.extend(violations)
        return len(violations) == 0


class PositionSizingAgent(SequentialRiskAgent):
    """
    π₁ Position Sizing Agent
    
    First agent in sequence, responsible for:
    - Determining position sizes based on risk context
    - Considering upstream strategic/tactical signals
    - Setting foundation for subsequent agents
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus, 
                 correlation_tracker: CorrelationTracker, var_calculator: VaRCalculator):
        super().__init__('position_sizing', config, event_bus, correlation_tracker, var_calculator)
        
        # Position sizing specific configuration
        self.max_position_size = config.get('max_position_size', 0.2)
        self.min_position_size = config.get('min_position_size', 0.01)
        self.position_granularity = config.get('position_granularity', 0.01)
        
        # Initialize neural network
        self.position_network = self._create_position_network()
        
        # Kelly criterion integration
        self.use_kelly_criterion = config.get('use_kelly_criterion', True)
        self.kelly_multiplier = config.get('kelly_multiplier', 0.25)  # 25% of Kelly
    
    def _get_sequence_position(self) -> int:
        return 1
    
    def _create_context_processor(self) -> nn.Module:
        """Create context processor for upstream signals"""
        return UpstreamContextProcessor(
            input_dim=20,  # Strategic + tactical signals
            hidden_dim=64,
            output_dim=32
        )
    
    def _create_constraint_validator(self) -> ConstraintValidator:
        """Create constraint validator for position sizing"""
        constraints = {
            'max_position_adjustment': 0.5,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'max_total_exposure': 1.0
        }
        return ConstraintValidator(constraints)
    
    def _create_position_network(self) -> nn.Module:
        """Create position sizing neural network"""
        return PositionSizingNetwork(
            input_dim=52,  # Risk state + context
            hidden_dim=128,
            output_dim=10,  # Number of assets
            max_position_size=self.max_position_size
        )
    
    def decide_action(self, risk_state: RiskState, context: Optional[SequentialContext] = None) -> RiskAction:
        """Decide position sizing action"""
        start_time = datetime.now()
        
        # Process context if available
        if context:
            context_features = self.process_sequential_context(context)
        else:
            context_features = torch.zeros(32)
        
        # Combine risk state and context
        risk_features = torch.tensor(risk_state.to_vector(), dtype=torch.float32)
        combined_features = torch.cat([risk_features, context_features])
        
        # Generate position sizes
        position_sizes = self.position_network(combined_features)
        
        # Apply Kelly criterion if enabled
        if self.use_kelly_criterion:
            position_sizes = self._apply_kelly_criterion(position_sizes, risk_state)
        
        # Create action
        action = RiskAction(
            action_type='position_sizing',
            position_adjustments={
                f'asset_{i}': float(position_sizes[i]) 
                for i in range(len(position_sizes))
            },
            confidence=0.8,
            expected_impact=torch.mean(torch.abs(position_sizes)).item(),
            constraints_satisfied=True
        )
        
        # Validate constraints
        if context and not self.validate_constraints(action, context):
            logger.warning(f"Constraint violation in {self.agent_id}, adjusting action")
            action = self._adjust_action_for_constraints(action, context)
        
        # Track decision time
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self.decision_times.append(decision_time)
        
        return action
    
    def _apply_kelly_criterion(self, position_sizes: torch.Tensor, risk_state: RiskState) -> torch.Tensor:
        """Apply Kelly criterion for position sizing"""
        # Get volatility estimate (simplified)
        volatility = risk_state.volatility_regime
        
        # Calculate Kelly fraction (simplified)
        kelly_fraction = 0.1 / max(volatility, 0.01)  # Simplified Kelly
        
        # Apply Kelly multiplier and cap
        kelly_adjusted = position_sizes * kelly_fraction * self.kelly_multiplier
        return torch.clamp(kelly_adjusted, -self.max_position_size, self.max_position_size)
    
    def _adjust_action_for_constraints(self, action: RiskAction, context: SequentialContext) -> RiskAction:
        """Adjust action to satisfy constraints"""
        # Scale down position adjustments if they exceed limits
        adjusted_positions = {}
        for asset, adjustment in action.position_adjustments.items():
            adjusted_positions[asset] = np.clip(
                adjustment, 
                -self.max_position_size, 
                self.max_position_size
            )
        
        return RiskAction(
            action_type=action.action_type,
            position_adjustments=adjusted_positions,
            confidence=action.confidence * 0.8,  # Reduce confidence
            expected_impact=action.expected_impact,
            constraints_satisfied=True
        )


class StopTargetAgent(SequentialRiskAgent):
    """
    π₂ Stop/Target Agent
    
    Second agent in sequence, responsible for:
    - Setting stop loss and target profit levels
    - Considering position sizes from π₁
    - Optimizing risk-reward ratios
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus,
                 correlation_tracker: CorrelationTracker, var_calculator: VaRCalculator):
        super().__init__('stop_target', config, event_bus, correlation_tracker, var_calculator)
        
        # Stop/target specific configuration
        self.min_stop_multiplier = config.get('min_stop_multiplier', 0.5)
        self.max_stop_multiplier = config.get('max_stop_multiplier', 5.0)
        self.min_target_multiplier = config.get('min_target_multiplier', 1.0)
        self.max_target_multiplier = config.get('max_target_multiplier', 10.0)
        
        # Initialize neural network
        self.stop_target_network = self._create_stop_target_network()
        
        # Risk-reward optimization
        self.target_risk_reward_ratio = config.get('target_risk_reward_ratio', 2.0)
        
    def _get_sequence_position(self) -> int:
        return 2
    
    def _create_context_processor(self) -> nn.Module:
        """Create context processor for position sizing context"""
        return PositionContextProcessor(
            input_dim=30,  # Position decisions + risk context
            hidden_dim=64,
            output_dim=32
        )
    
    def _create_constraint_validator(self) -> ConstraintValidator:
        """Create constraint validator for stop/target"""
        constraints = {
            'max_stop_multiplier': self.max_stop_multiplier,
            'min_stop_multiplier': self.min_stop_multiplier,
            'max_target_multiplier': self.max_target_multiplier,
            'min_target_multiplier': self.min_target_multiplier,
            'min_risk_reward_ratio': 1.0
        }
        return ConstraintValidator(constraints)
    
    def _create_stop_target_network(self) -> nn.Module:
        """Create stop/target neural network"""
        return StopTargetNetwork(
            input_dim=52,  # Risk state + context
            hidden_dim=128,
            output_dim=2,  # Stop and target multipliers
            min_stop=self.min_stop_multiplier,
            max_stop=self.max_stop_multiplier,
            min_target=self.min_target_multiplier,
            max_target=self.max_target_multiplier
        )
    
    def decide_action(self, risk_state: RiskState, context: Optional[SequentialContext] = None) -> RiskAction:
        """Decide stop/target action"""
        start_time = datetime.now()
        
        # Process context from position sizing agent
        if context:
            context_features = self.process_sequential_context(context)
        else:
            context_features = torch.zeros(32)
        
        # Combine risk state and context
        risk_features = torch.tensor(risk_state.to_vector(), dtype=torch.float32)
        combined_features = torch.cat([risk_features, context_features])
        
        # Generate stop/target multipliers
        stop_target_multipliers = self.stop_target_network(combined_features)
        stop_multiplier = float(stop_target_multipliers[0])
        target_multiplier = float(stop_target_multipliers[1])
        
        # Optimize risk-reward ratio
        stop_multiplier, target_multiplier = self._optimize_risk_reward_ratio(
            stop_multiplier, target_multiplier, risk_state
        )
        
        # Create action
        action = RiskAction(
            action_type='stop_target',
            stop_multiplier=stop_multiplier,
            target_multiplier=target_multiplier,
            confidence=0.85,
            expected_impact=abs(stop_multiplier - 1.0) + abs(target_multiplier - 1.0),
            constraints_satisfied=True
        )
        
        # Validate constraints
        if context and not self.validate_constraints(action, context):
            logger.warning(f"Constraint violation in {self.agent_id}, adjusting action")
            action = self._adjust_action_for_constraints(action, context)
        
        # Track decision time
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self.decision_times.append(decision_time)
        
        return action
    
    def _optimize_risk_reward_ratio(self, stop_multiplier: float, target_multiplier: float, 
                                  risk_state: RiskState) -> Tuple[float, float]:
        """Optimize risk-reward ratio"""
        # Adjust based on volatility regime
        volatility_adjustment = 1.0 + risk_state.volatility_regime * 0.5
        
        # Adjust stop based on volatility
        adjusted_stop = stop_multiplier * volatility_adjustment
        
        # Ensure minimum risk-reward ratio
        min_target = adjusted_stop * self.target_risk_reward_ratio
        adjusted_target = max(target_multiplier, min_target)
        
        # Clamp to valid ranges
        adjusted_stop = np.clip(adjusted_stop, self.min_stop_multiplier, self.max_stop_multiplier)
        adjusted_target = np.clip(adjusted_target, self.min_target_multiplier, self.max_target_multiplier)
        
        return adjusted_stop, adjusted_target
    
    def _adjust_action_for_constraints(self, action: RiskAction, context: SequentialContext) -> RiskAction:
        """Adjust action to satisfy constraints"""
        # Clamp multipliers to valid ranges
        adjusted_stop = np.clip(action.stop_multiplier, self.min_stop_multiplier, self.max_stop_multiplier)
        adjusted_target = np.clip(action.target_multiplier, self.min_target_multiplier, self.max_target_multiplier)
        
        # Ensure minimum risk-reward ratio
        if adjusted_target < adjusted_stop * self.target_risk_reward_ratio:
            adjusted_target = adjusted_stop * self.target_risk_reward_ratio
        
        return RiskAction(
            action_type=action.action_type,
            stop_multiplier=adjusted_stop,
            target_multiplier=adjusted_target,
            confidence=action.confidence * 0.9,
            expected_impact=action.expected_impact,
            constraints_satisfied=True
        )


class RiskMonitorAgent(SequentialRiskAgent):
    """
    π₃ Risk Monitor Agent
    
    Third agent in sequence, responsible for:
    - Monitoring and responding to risk events
    - Considering position and stop/target decisions
    - Triggering emergency protocols
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus,
                 correlation_tracker: CorrelationTracker, var_calculator: VaRCalculator):
        super().__init__('risk_monitor', config, event_bus, correlation_tracker, var_calculator)
        
        # Risk monitoring specific configuration
        self.risk_thresholds = config.get('risk_thresholds', {
            'var_breach': 0.05,
            'correlation_spike': 0.8,
            'drawdown_limit': 0.15,
            'leverage_limit': 4.0
        })
        
        # Initialize neural network
        self.risk_monitor_network = self._create_risk_monitor_network()
        
        # Emergency protocol configuration
        self.emergency_threshold = config.get('emergency_threshold', 0.9)
        self.emergency_actions = config.get('emergency_actions', [
            'reduce_positions', 'hedge_portfolio', 'halt_trading'
        ])
    
    def _get_sequence_position(self) -> int:
        return 3
    
    def _create_context_processor(self) -> nn.Module:
        """Create context processor for position and stop/target context"""
        return RiskMonitorContextProcessor(
            input_dim=40,  # Position + stop/target + risk context
            hidden_dim=64,
            output_dim=32
        )
    
    def _create_constraint_validator(self) -> ConstraintValidator:
        """Create constraint validator for risk monitoring"""
        constraints = {
            'max_risk_reduction': 0.8,
            'max_emergency_actions': 3,
            'min_alert_threshold': 0.1
        }
        return ConstraintValidator(constraints)
    
    def _create_risk_monitor_network(self) -> nn.Module:
        """Create risk monitoring neural network"""
        return RiskMonitorNetwork(
            input_dim=52,  # Risk state + context
            hidden_dim=128,
            output_dim=5,  # Risk monitoring actions
            risk_thresholds=self.risk_thresholds
        )
    
    def decide_action(self, risk_state: RiskState, context: Optional[SequentialContext] = None) -> RiskAction:
        """Decide risk monitoring action"""
        start_time = datetime.now()
        
        # Process context from previous agents
        if context:
            context_features = self.process_sequential_context(context)
        else:
            context_features = torch.zeros(32)
        
        # Combine risk state and context
        risk_features = torch.tensor(risk_state.to_vector(), dtype=torch.float32)
        combined_features = torch.cat([risk_features, context_features])
        
        # Generate risk monitoring actions
        risk_actions = self.risk_monitor_network(combined_features)
        
        # Interpret actions
        alert_level = float(risk_actions[0])
        hedge_ratio = float(risk_actions[1])
        reduce_ratio = float(risk_actions[2])
        emergency_flag = float(risk_actions[3])
        correlation_adjust = float(risk_actions[4])
        
        # Check for emergency conditions
        emergency_triggered = self._check_emergency_conditions(
            risk_state, context, emergency_flag
        )
        
        # Create action
        action = RiskAction(
            action_type='risk_monitor',
            alert_level=alert_level,
            hedge_ratio=hedge_ratio,
            risk_reduction_factor=reduce_ratio,
            emergency_triggered=emergency_triggered,
            correlation_adjustment=correlation_adjust,
            confidence=0.9,
            expected_impact=alert_level + hedge_ratio + reduce_ratio,
            constraints_satisfied=True
        )
        
        # Validate constraints
        if context and not self.validate_constraints(action, context):
            logger.warning(f"Constraint violation in {self.agent_id}, adjusting action")
            action = self._adjust_action_for_constraints(action, context)
        
        # Track decision time
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self.decision_times.append(decision_time)
        
        return action
    
    def _check_emergency_conditions(self, risk_state: RiskState, 
                                   context: Optional[SequentialContext],
                                   emergency_flag: float) -> bool:
        """Check for emergency conditions"""
        # Check individual risk thresholds
        emergency_conditions = [
            risk_state.var_estimate_5pct > self.risk_thresholds['var_breach'],
            risk_state.correlation_risk > self.risk_thresholds['correlation_spike'],
            risk_state.current_drawdown_pct > self.risk_thresholds['drawdown_limit'],
            risk_state.margin_usage_pct > self.risk_thresholds['leverage_limit'] / 4.0,
            emergency_flag > self.emergency_threshold
        ]
        
        # Check context-specific conditions
        if context:
            # Check correlation shock from tracker
            correlation_regime = context.correlation_context.get('regime', 'NORMAL')
            if correlation_regime in ['CRISIS', 'SHOCK']:
                emergency_conditions.append(True)
            
            # Check VaR performance
            var_calc_time = context.var_context.get('calculation_time_ms', 0)
            if var_calc_time > 10.0:  # 10ms threshold
                emergency_conditions.append(True)
        
        return any(emergency_conditions)
    
    def _adjust_action_for_constraints(self, action: RiskAction, context: SequentialContext) -> RiskAction:
        """Adjust action to satisfy constraints"""
        # Clamp risk reduction factor
        adjusted_reduction = np.clip(action.risk_reduction_factor, 0.0, 0.8)
        
        # Adjust hedge ratio
        adjusted_hedge = np.clip(action.hedge_ratio, 0.0, 1.0)
        
        # Adjust alert level
        adjusted_alert = np.clip(action.alert_level, 0.1, 1.0)
        
        return RiskAction(
            action_type=action.action_type,
            alert_level=adjusted_alert,
            hedge_ratio=adjusted_hedge,
            risk_reduction_factor=adjusted_reduction,
            emergency_triggered=action.emergency_triggered,
            correlation_adjustment=action.correlation_adjustment,
            confidence=action.confidence * 0.85,
            expected_impact=action.expected_impact,
            constraints_satisfied=True
        )


class PortfolioOptimizerAgent(SequentialRiskAgent):
    """
    π₄ Portfolio Optimizer Agent
    
    Fourth agent in sequence, responsible for:
    - Final portfolio optimization
    - Considering all previous agent decisions
    - Generating final risk superposition
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus,
                 correlation_tracker: CorrelationTracker, var_calculator: VaRCalculator):
        super().__init__('portfolio_optimizer', config, event_bus, correlation_tracker, var_calculator)
        
        # Portfolio optimization specific configuration
        self.optimization_method = config.get('optimization_method', 'mean_variance')
        self.risk_aversion = config.get('risk_aversion', 2.0)
        self.diversification_penalty = config.get('diversification_penalty', 0.1)
        
        # Initialize neural network
        self.portfolio_network = self._create_portfolio_network()
        
        # Constraint configuration
        self.max_asset_weight = config.get('max_asset_weight', 0.3)
        self.min_asset_weight = config.get('min_asset_weight', 0.0)
        
    def _get_sequence_position(self) -> int:
        return 4
    
    def _create_context_processor(self) -> nn.Module:
        """Create context processor for full sequential context"""
        return PortfolioContextProcessor(
            input_dim=60,  # Full sequential context
            hidden_dim=128,
            output_dim=64
        )
    
    def _create_constraint_validator(self) -> ConstraintValidator:
        """Create constraint validator for portfolio optimization"""
        constraints = {
            'weight_sum_tolerance': 0.01,
            'max_asset_weight': self.max_asset_weight,
            'min_asset_weight': self.min_asset_weight,
            'max_concentration': 0.5
        }
        return ConstraintValidator(constraints)
    
    def _create_portfolio_network(self) -> nn.Module:
        """Create portfolio optimization neural network"""
        return PortfolioOptimizerNetwork(
            input_dim=74,  # Risk state + full context
            hidden_dim=256,
            output_dim=10,  # Asset weights
            max_weight=self.max_asset_weight,
            min_weight=self.min_asset_weight
        )
    
    def decide_action(self, risk_state: RiskState, context: Optional[SequentialContext] = None) -> RiskAction:
        """Decide portfolio optimization action"""
        start_time = datetime.now()
        
        # Process full sequential context
        if context:
            context_features = self.process_sequential_context(context)
        else:
            context_features = torch.zeros(64)
        
        # Combine risk state and context
        risk_features = torch.tensor(risk_state.to_vector(), dtype=torch.float32)
        combined_features = torch.cat([risk_features, context_features])
        
        # Generate portfolio weights
        portfolio_weights = self.portfolio_network(combined_features)
        
        # Apply optimization constraints
        portfolio_weights = self._apply_optimization_constraints(
            portfolio_weights, risk_state, context
        )
        
        # Create action
        action = RiskAction(
            action_type='portfolio_optimizer',
            portfolio_weights={
                f'asset_{i}': float(portfolio_weights[i])
                for i in range(len(portfolio_weights))
            },
            confidence=0.95,
            expected_impact=torch.std(portfolio_weights).item(),
            constraints_satisfied=True
        )
        
        # Validate constraints
        if context and not self.validate_constraints(action, context):
            logger.warning(f"Constraint violation in {self.agent_id}, adjusting action")
            action = self._adjust_action_for_constraints(action, context)
        
        # Track decision time
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self.decision_times.append(decision_time)
        
        return action
    
    def _apply_optimization_constraints(self, weights: torch.Tensor, 
                                      risk_state: RiskState,
                                      context: Optional[SequentialContext]) -> torch.Tensor:
        """Apply portfolio optimization constraints"""
        # Normalize weights to sum to 1
        weights = F.softmax(weights, dim=0)
        
        # Apply individual asset weight constraints
        weights = torch.clamp(weights, self.min_asset_weight, self.max_asset_weight)
        
        # Re-normalize after clamping
        weights = weights / torch.sum(weights)
        
        # Apply diversification penalty
        if self.diversification_penalty > 0:
            concentration = torch.sum(weights ** 2)
            if concentration > 0.5:  # High concentration
                # Apply penalty by reducing largest weights
                sorted_weights, indices = torch.sort(weights, descending=True)
                penalty_factor = 1.0 - self.diversification_penalty
                sorted_weights[:3] *= penalty_factor  # Reduce top 3 weights
                weights[indices] = sorted_weights
                weights = weights / torch.sum(weights)
        
        return weights
    
    def _adjust_action_for_constraints(self, action: RiskAction, context: SequentialContext) -> RiskAction:
        """Adjust action to satisfy constraints"""
        weights = list(action.portfolio_weights.values())
        weights = np.array(weights)
        
        # Clamp individual weights
        weights = np.clip(weights, self.min_asset_weight, self.max_asset_weight)
        
        # Re-normalize
        weights = weights / np.sum(weights)
        
        # Create adjusted action
        adjusted_weights = {
            f'asset_{i}': float(weights[i])
            for i in range(len(weights))
        }
        
        return RiskAction(
            action_type=action.action_type,
            portfolio_weights=adjusted_weights,
            confidence=action.confidence * 0.9,
            expected_impact=action.expected_impact,
            constraints_satisfied=True
        )


# Neural Network Architectures

class UpstreamContextProcessor(nn.Module):
    """Neural network for processing upstream context"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PositionContextProcessor(nn.Module):
    """Neural network for processing position context"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RiskMonitorContextProcessor(nn.Module):
    """Neural network for processing risk monitor context"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PortfolioContextProcessor(nn.Module):
    """Neural network for processing full portfolio context"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PositionSizingNetwork(nn.Module):
    """Neural network for position sizing decisions"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, max_position_size: float):
        super().__init__()
        self.max_position_size = max_position_size
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)
        return torch.tanh(output) * self.max_position_size


class StopTargetNetwork(nn.Module):
    """Neural network for stop/target decisions"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 min_stop: float, max_stop: float, min_target: float, max_target: float):
        super().__init__()
        self.min_stop = min_stop
        self.max_stop = max_stop
        self.min_target = min_target
        self.max_target = max_target
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)
        
        # Apply different ranges for stop and target
        stop_multiplier = torch.sigmoid(output[0]) * (self.max_stop - self.min_stop) + self.min_stop
        target_multiplier = torch.sigmoid(output[1]) * (self.max_target - self.min_target) + self.min_target
        
        return torch.stack([stop_multiplier, target_multiplier])


class RiskMonitorNetwork(nn.Module):
    """Neural network for risk monitoring decisions"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, risk_thresholds: Dict[str, float]):
        super().__init__()
        self.risk_thresholds = risk_thresholds
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)
        return torch.sigmoid(output)  # All outputs in [0, 1]


class PortfolioOptimizerNetwork(nn.Module):
    """Neural network for portfolio optimization decisions"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 max_weight: float, min_weight: float):
        super().__init__()
        self.max_weight = max_weight
        self.min_weight = min_weight
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.network(x)
        # Use softmax to ensure weights sum to 1
        return F.softmax(output, dim=0)


def create_sequential_risk_agents(config: Dict[str, Any], 
                                 event_bus: EventBus,
                                 correlation_tracker: CorrelationTracker,
                                 var_calculator: VaRCalculator) -> Dict[str, SequentialRiskAgent]:
    """
    Factory function to create all sequential risk agents
    
    Args:
        config: Configuration dictionary for agents
        event_bus: Event bus for communication
        correlation_tracker: Correlation tracking system
        var_calculator: VaR calculation system
        
    Returns:
        Dictionary of sequential risk agents
    """
    agents = {}
    
    # Create individual agent configurations
    position_config = config.get('position_sizing', {})
    stop_target_config = config.get('stop_target', {})
    risk_monitor_config = config.get('risk_monitor', {})
    portfolio_optimizer_config = config.get('portfolio_optimizer', {})
    
    # Create agents
    agents['position_sizing'] = PositionSizingAgent(
        position_config, event_bus, correlation_tracker, var_calculator
    )
    
    agents['stop_target'] = StopTargetAgent(
        stop_target_config, event_bus, correlation_tracker, var_calculator
    )
    
    agents['risk_monitor'] = RiskMonitorAgent(
        risk_monitor_config, event_bus, correlation_tracker, var_calculator
    )
    
    agents['portfolio_optimizer'] = PortfolioOptimizerAgent(
        portfolio_optimizer_config, event_bus, correlation_tracker, var_calculator
    )
    
    logger.info("Sequential risk agents created", agent_count=len(agents))
    return agents