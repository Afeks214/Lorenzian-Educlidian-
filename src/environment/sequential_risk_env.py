"""
Sequential Risk MARL Environment

This module implements a sequential risk management environment where agents operate in a
specific sequence: position sizing → stop/target → risk monitor → portfolio optimizer.
Each agent receives enriched context from its predecessors and builds upon prior decisions.

Key Features:
- Sequential agent execution with context propagation
- Integration with existing VaR correlation system
- Real-time risk assessment and emergency protocols
- Strategic and tactical context integration
- Comprehensive risk superposition output

Agent Sequence:
1. Position Sizing Agent (π₁): Determines position sizes based on risk context
2. Stop/Target Agent (π₂): Sets stop losses and targets based on position context
3. Risk Monitor Agent (π₃): Monitors and responds to risk events
4. Portfolio Optimizer Agent (π₄): Optimizes final portfolio allocation

Integration Points:
- VaR correlation tracking system (<5ms performance)
- Correlation shock detection and response
- Emergency protocol activation
- Strategic/tactical context from upstream MARLs
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
import time
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
import asyncio
import threading
from dataclasses import dataclass, field
import json

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import structlog

# Import existing risk management components
from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime, CorrelationShock
from src.risk.core.var_calculator import VaRCalculator, VaRResult
from src.risk.core.state_processor import RiskStateProcessor, StateProcessingConfig
from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.risk.marl.centralized_critic import CentralizedCritic, GlobalRiskState, OperatingMode
from src.safety.trading_system_controller import get_controller

logger = structlog.get_logger()


class SequentialPhase(Enum):
    """Sequential phases of risk management execution"""
    POSITION_SIZING = "position_sizing"
    STOP_TARGET = "stop_target"
    RISK_MONITOR = "risk_monitor"
    PORTFOLIO_OPTIMIZER = "portfolio_optimizer"
    AGGREGATION = "aggregation"
    COMPLETE = "complete"


@dataclass
class SequentialContext:
    """Context passed between sequential agents"""
    position_sizing_decisions: Dict[str, float] = field(default_factory=dict)
    stop_loss_levels: Dict[str, float] = field(default_factory=dict)
    target_levels: Dict[str, float] = field(default_factory=dict)
    risk_alerts: List[str] = field(default_factory=list)
    risk_actions: List[str] = field(default_factory=list)
    portfolio_weights: Dict[str, float] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    correlation_context: Dict[str, Any] = field(default_factory=dict)
    upstream_context: Dict[str, Any] = field(default_factory=dict)  # Strategic/tactical context


@dataclass
class RiskSuperposition:
    """Comprehensive risk superposition output for execution layer"""
    timestamp: datetime
    position_allocations: Dict[str, float]
    stop_loss_orders: Dict[str, float]
    target_profit_orders: Dict[str, float]
    risk_limits: Dict[str, float]
    emergency_protocols: List[str]
    correlation_adjustments: Dict[str, float]
    var_estimates: Dict[str, float]
    execution_priority: List[str]
    risk_attribution: Dict[str, float]
    confidence_scores: Dict[str, float]
    sequential_metadata: Dict[str, Any]


@dataclass
class UpstreamContext:
    """Context received from strategic and tactical MARL systems"""
    strategic_signals: Dict[str, float] = field(default_factory=dict)
    tactical_signals: Dict[str, float] = field(default_factory=dict)
    market_regime: str = "normal"
    volatility_forecast: Dict[str, float] = field(default_factory=dict)
    correlation_forecast: Dict[str, float] = field(default_factory=dict)
    execution_urgency: float = 0.5
    risk_budget: float = 0.05
    time_horizon: int = 1


class SequentialRiskEnvironment(AECEnv):
    """
    Sequential Risk MARL Environment
    
    Implements a sequential risk management environment where agents operate in a
    specific order, with each agent receiving enriched context from predecessors.
    
    Agent Sequence:
    1. Position Sizing Agent → determines position sizes
    2. Stop/Target Agent → sets stop losses and targets
    3. Risk Monitor Agent → monitors and responds to risks
    4. Portfolio Optimizer → optimizes final allocation
    
    Key Features:
    - Context propagation between agents
    - VaR correlation system integration
    - Emergency protocol handling
    - Real-time performance monitoring
    - Rich risk superposition output
    """
    
    metadata = {
        'name': 'sequential_risk_v1',
        'render_modes': ['human', 'rgb_array'],
        'is_parallelizable': False,
        'render_fps': 4
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Sequential Risk Environment
        
        Args:
            config: Configuration dictionary with risk parameters
        """
        super().__init__()
        
        self.config = config
        self.initial_capital = config.get('initial_capital', 1_000_000.0)
        self.max_steps = config.get('max_steps', 1000)
        self.risk_tolerance = config.get('risk_tolerance', 0.05)
        self.performance_target_ms = config.get('performance_target_ms', 5.0)
        
        # Sequential agent configuration
        self.agents = ['position_sizing', 'stop_target', 'risk_monitor', 'portfolio_optimizer']
        self.possible_agents = self.agents.copy()
        self.num_agents = len(self.agents)
        
        # Asset universe
        self.asset_universe = config.get('asset_universe', [
            'SPY', 'QQQ', 'IWM', 'VTI', 'TLT', 'GLD', 'VIX', 'UUP', 'EFA', 'EEM'
        ])
        
        # Initialize core components
        self.event_bus = EventBus()
        self._initialize_risk_components()
        
        # Sequential execution state
        self.current_phase = SequentialPhase.POSITION_SIZING
        self.sequential_context = SequentialContext()
        self.upstream_context = UpstreamContext()
        
        # Environment state
        self.current_step = 0
        self.episode_start_time = None
        self.sequence_completion_times = deque(maxlen=100)
        
        # Agent selector for sequential execution
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        
        # Action and observation spaces
        self._action_spaces = self._define_action_spaces()
        self._observation_spaces = self._define_observation_spaces()
        
        # Episode tracking
        self.episode_rewards = {agent: [] for agent in self.agents}
        self.episode_actions = {agent: [] for agent in self.agents}
        self.risk_superpositions = deque(maxlen=1000)
        
        # Performance monitoring
        self.step_times = deque(maxlen=100)
        self.var_calculation_times = deque(maxlen=100)
        self.correlation_update_times = deque(maxlen=100)
        
        # Emergency protocols
        self.emergency_active = False
        self.emergency_triggers = []
        self.emergency_recovery_actions = []
        
        # Termination conditions
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("Sequential Risk Environment initialized",
                   agents=self.agents,
                   asset_universe=len(self.asset_universe),
                   performance_target_ms=self.performance_target_ms)
    
    def _initialize_risk_components(self):
        """Initialize risk management components"""
        # Enhanced correlation tracker with sequential awareness
        self.correlation_tracker = CorrelationTracker(
            event_bus=self.event_bus,
            ewma_lambda=0.94,
            shock_threshold=0.3,  # Lower threshold for sequential environment
            shock_window_minutes=5,  # Faster detection
            performance_target_ms=self.performance_target_ms
        )
        
        # VaR calculator with real-time updates
        self.var_calculator = VaRCalculator(
            correlation_tracker=self.correlation_tracker,
            event_bus=self.event_bus,
            confidence_levels=[0.95, 0.99],
            time_horizons=[1, 10],
            default_method="parametric"
        )
        
        # State processor for sequential context
        state_config = StateProcessingConfig(
            lookback_window=50,  # Shorter window for faster response
            normalization_method='zscore',
            outlier_threshold=2.5,
            smoothing_alpha=0.2
        )
        self.state_processor = RiskStateProcessor(state_config, self.event_bus)
        
        # Centralized critic for global risk assessment
        critic_config = self.config.get('critic_config', {
            'hidden_dim': 256,
            'num_layers': 4,
            'learning_rate': 0.0005,
            'target_update_freq': 50
        })
        self.centralized_critic = CentralizedCritic(critic_config, self.event_bus)
        
        # Initialize asset universe
        self.correlation_tracker.initialize_assets(self.asset_universe)
        
        # Register performance monitoring callbacks
        self.correlation_tracker.register_leverage_callback(self._handle_leverage_reduction)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for sequential processing"""
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_breach)
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_var_update)
        self.event_bus.subscribe(EventType.CORRELATION_SHOCK, self._handle_correlation_shock)
        self.event_bus.subscribe(EventType.EMERGENCY_PROTOCOL, self._handle_emergency_protocol)
    
    def _define_action_spaces(self) -> Dict[str, spaces.Space]:
        """Define action spaces for sequential agents"""
        return {
            # Position Sizing Agent: Enhanced continuous position adjustments
            'position_sizing': spaces.Box(
                low=-0.5, high=0.5, 
                shape=(len(self.asset_universe),), 
                dtype=np.float32
            ),
            
            # Stop/Target Agent: Stop loss and target profit multipliers
            'stop_target': spaces.Box(
                low=0.5, high=5.0, 
                shape=(2,), 
                dtype=np.float32
            ),
            
            # Risk Monitor Agent: Enhanced risk response actions
            'risk_monitor': spaces.Box(
                low=0.0, high=1.0, 
                shape=(5,), 
                dtype=np.float32
            ),  # [alert_level, hedge_ratio, reduce_ratio, emergency_flag, correlation_adjust]
            
            # Portfolio Optimizer: Final portfolio weights
            'portfolio_optimizer': spaces.Box(
                low=0.0, high=1.0, 
                shape=(len(self.asset_universe),), 
                dtype=np.float32
            )
        }
    
    def _define_observation_spaces(self) -> Dict[str, spaces.Space]:
        """Define observation spaces for sequential agents"""
        # Each agent receives enriched context from predecessors
        base_risk_dim = 10  # Base risk state
        
        return {
            # Position Sizing Agent: Base risk state + upstream context
            'position_sizing': spaces.Box(
                low=-5.0, high=5.0, 
                shape=(base_risk_dim + 10,), 
                dtype=np.float32
            ),
            
            # Stop/Target Agent: Base + position sizing context
            'stop_target': spaces.Box(
                low=-5.0, high=5.0, 
                shape=(base_risk_dim + 10 + len(self.asset_universe),), 
                dtype=np.float32
            ),
            
            # Risk Monitor Agent: Base + position + stop/target context
            'risk_monitor': spaces.Box(
                low=-5.0, high=5.0, 
                shape=(base_risk_dim + 10 + len(self.asset_universe) + 2,), 
                dtype=np.float32
            ),
            
            # Portfolio Optimizer: Full sequential context
            'portfolio_optimizer': spaces.Box(
                low=-5.0, high=5.0, 
                shape=(base_risk_dim + 10 + len(self.asset_universe) + 2 + 5,), 
                dtype=np.float32
            )
        }
    
    @property
    def action_space(self) -> spaces.Space:
        """Return action space for current agent"""
        return self._action_spaces[self.agent_selection]
    
    @property
    def observation_space(self) -> spaces.Space:
        """Return observation space for current agent"""
        return self._observation_spaces[self.agent_selection]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> None:
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset sequential execution state
        self.current_phase = SequentialPhase.POSITION_SIZING
        self.sequential_context = SequentialContext()
        self.upstream_context = UpstreamContext()
        
        # Reset environment state
        self.current_step = 0
        self.episode_start_time = datetime.now()
        self.emergency_active = False
        self.emergency_triggers.clear()
        self.emergency_recovery_actions.clear()
        
        # Reset agent selector
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        
        # Reset risk components
        self.correlation_tracker.reset_statistics()
        self.state_processor.reset_statistics()
        self.centralized_critic.reset()
        
        # Reset episode tracking
        self.episode_rewards = {agent: [] for agent in self.agents}
        self.episode_actions = {agent: [] for agent in self.agents}
        self.risk_superpositions.clear()
        
        # Reset termination conditions
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Initialize with upstream context if provided
        if options and 'upstream_context' in options:
            self.upstream_context = UpstreamContext(**options['upstream_context'])
        
        logger.info("Sequential Risk Environment reset",
                   phase=self.current_phase.value,
                   agent_selection=self.agent_selection,
                   upstream_context=bool(self.upstream_context.strategic_signals))
    
    def step(self, action: Union[int, np.ndarray]) -> None:
        """Execute one step in sequential environment"""
        step_start_time = time.time()
        
        # Validate action
        if not self._validate_action(action):
            logger.error("Invalid action received",
                        agent=self.agent_selection,
                        action=action,
                        expected_space=self.action_space)
            self.rewards[self.agent_selection] = -10.0
            self._advance_sequential_state()
            return
        
        # Store action
        self.episode_actions[self.agent_selection].append(action)
        
        # Apply sequential agent action
        self._apply_sequential_action(self.agent_selection, action)
        
        # Update sequential context
        self._update_sequential_context()
        
        # Check for emergency protocols
        self._check_emergency_protocols()
        
        # Calculate VaR with timing
        var_start_time = time.time()
        var_result = self._calculate_real_time_var()
        var_calc_time = (time.time() - var_start_time) * 1000
        self.var_calculation_times.append(var_calc_time)
        
        # Generate risk state for current context
        risk_state = self._generate_sequential_risk_state()
        
        # Process risk state
        normalized_state, processing_metadata = self.state_processor.process_state(
            risk_state.to_vector()
        )
        
        # Evaluate global risk
        global_risk_state = self._create_global_risk_state(risk_state)
        global_risk_value, operating_mode = self.centralized_critic.evaluate_global_risk(
            global_risk_state
        )
        
        # Calculate sequential reward
        reward = self._calculate_sequential_reward(
            self.agent_selection, action, risk_state, var_result, global_risk_value
        )
        
        self.rewards[self.agent_selection] = reward
        self.episode_rewards[self.agent_selection].append(reward)
        
        # Update info with rich context
        self.infos[self.agent_selection] = {
            'step': self.current_step,
            'phase': self.current_phase.value,
            'var_calculation_time_ms': var_calc_time,
            'global_risk_value': global_risk_value,
            'operating_mode': operating_mode.value,
            'emergency_active': self.emergency_active,
            'sequential_context': self._serialize_context(),
            'processing_time_ms': (time.time() - step_start_time) * 1000,
            'correlation_regime': self.correlation_tracker.current_regime.value,
            'performance_met': var_calc_time < self.performance_target_ms
        }
        
        # Check termination conditions
        self._check_termination_conditions(operating_mode)
        
        # Advance sequential state
        self._advance_sequential_state()
        
        # Generate risk superposition if sequence is complete
        if self.current_phase == SequentialPhase.COMPLETE:
            superposition = self._generate_risk_superposition()
            self.risk_superpositions.append(superposition)
            
            # Publish superposition event
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.RISK_SUPERPOSITION,
                    superposition,
                    'SequentialRiskEnvironment'
                )
            )
        
        # Update timing
        self.current_step += 1
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        
        logger.debug("Sequential step completed",
                    agent=self.agent_selection,
                    phase=self.current_phase.value,
                    step_time_ms=step_time * 1000,
                    var_time_ms=var_calc_time,
                    reward=reward)
    
    def observe(self, agent: str) -> np.ndarray:
        """Get observation for specified agent with sequential context"""
        base_risk_state = self._generate_sequential_risk_state()
        base_vector = base_risk_state.to_vector()
        
        # Add upstream context
        upstream_vector = self._encode_upstream_context()
        
        if agent == 'position_sizing':
            # Base risk state + upstream context
            obs = np.concatenate([base_vector, upstream_vector])
        
        elif agent == 'stop_target':
            # Base + upstream + position sizing context
            position_vector = self._encode_position_sizing_context()
            obs = np.concatenate([base_vector, upstream_vector, position_vector])
        
        elif agent == 'risk_monitor':
            # Base + upstream + position + stop/target context
            position_vector = self._encode_position_sizing_context()
            stop_target_vector = self._encode_stop_target_context()
            obs = np.concatenate([base_vector, upstream_vector, position_vector, stop_target_vector])
        
        elif agent == 'portfolio_optimizer':
            # Full sequential context
            position_vector = self._encode_position_sizing_context()
            stop_target_vector = self._encode_stop_target_context()
            risk_monitor_vector = self._encode_risk_monitor_context()
            obs = np.concatenate([
                base_vector, upstream_vector, position_vector, 
                stop_target_vector, risk_monitor_vector
            ])
        
        else:
            obs = base_vector
        
        return obs.astype(np.float32)
    
    def _validate_action(self, action: Union[int, np.ndarray]) -> bool:
        """Validate action against current agent's action space"""
        try:
            return self.action_space.contains(action)
        except (TypeError, ValueError):
            return False
    
    def _apply_sequential_action(self, agent: str, action: Union[int, np.ndarray]):
        """Apply agent action to sequential context"""
        if agent == 'position_sizing':
            # Apply position sizing decisions
            for i, asset in enumerate(self.asset_universe):
                self.sequential_context.position_sizing_decisions[asset] = float(action[i])
        
        elif agent == 'stop_target':
            # Apply stop/target levels
            stop_multiplier = float(action[0])
            target_multiplier = float(action[1])
            
            for asset in self.asset_universe:
                position_size = self.sequential_context.position_sizing_decisions.get(asset, 0.0)
                if position_size != 0.0:
                    # Calculate stop/target levels based on position size and volatility
                    base_level = abs(position_size) * 0.02  # 2% base level
                    self.sequential_context.stop_loss_levels[asset] = base_level * stop_multiplier
                    self.sequential_context.target_levels[asset] = base_level * target_multiplier
        
        elif agent == 'risk_monitor':
            # Apply risk monitoring actions
            alert_level = float(action[0])
            hedge_ratio = float(action[1])
            reduce_ratio = float(action[2])
            emergency_flag = float(action[3])
            correlation_adjust = float(action[4])
            
            # Update risk actions based on action values
            if alert_level > 0.7:
                self.sequential_context.risk_alerts.append('high_risk_alert')
            if hedge_ratio > 0.5:
                self.sequential_context.risk_actions.append('hedge_positions')
            if reduce_ratio > 0.6:
                self.sequential_context.risk_actions.append('reduce_positions')
            if emergency_flag > 0.8:
                self.sequential_context.risk_actions.append('emergency_stop')
                self.emergency_active = True
            
            # Apply correlation adjustments
            if correlation_adjust > 0.5:
                for asset in self.asset_universe:
                    current_position = self.sequential_context.position_sizing_decisions.get(asset, 0.0)
                    self.sequential_context.position_sizing_decisions[asset] = current_position * (1 - correlation_adjust * 0.3)
        
        elif agent == 'portfolio_optimizer':
            # Apply final portfolio optimization
            weights = np.array(action)
            weight_sum = np.sum(weights)
            
            if weight_sum > 0:
                normalized_weights = weights / weight_sum
                for i, asset in enumerate(self.asset_universe):
                    self.sequential_context.portfolio_weights[asset] = float(normalized_weights[i])
            
            # Apply portfolio optimization to position sizing
            for asset in self.asset_universe:
                base_position = self.sequential_context.position_sizing_decisions.get(asset, 0.0)
                portfolio_weight = self.sequential_context.portfolio_weights.get(asset, 0.0)
                
                # Combine position sizing with portfolio optimization
                optimized_position = base_position * portfolio_weight
                self.sequential_context.position_sizing_decisions[asset] = optimized_position
    
    def _update_sequential_context(self):
        """Update sequential context with latest risk metrics"""
        # Update correlation context
        correlation_matrix = self.correlation_tracker.get_correlation_matrix()
        if correlation_matrix is not None:
            self.sequential_context.correlation_context = {
                'avg_correlation': np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
                'max_correlation': np.max(correlation_matrix),
                'regime': self.correlation_tracker.current_regime.value
            }
        
        # Update performance metrics
        var_result = self.var_calculator.get_latest_var()
        if var_result:
            self.sequential_context.performance_metrics = {
                'var_estimate': var_result.portfolio_var,
                'var_percentage': var_result.portfolio_var / max(self.initial_capital, 1.0),
                'calculation_time_ms': var_result.performance_ms
            }
        
        # Update execution metadata
        self.sequential_context.execution_metadata = {
            'timestamp': datetime.now().isoformat(),
            'sequence_step': self.current_step,
            'phase': self.current_phase.value,
            'emergency_active': self.emergency_active
        }
    
    def _calculate_real_time_var(self) -> Optional[VaRResult]:
        """Calculate VaR with real-time timing monitoring"""
        # Create dummy position data for VaR calculation
        positions = []
        for asset, position_size in self.sequential_context.position_sizing_decisions.items():
            if position_size != 0.0:
                # Create position data (simplified)
                position_value = abs(position_size) * self.initial_capital * 0.1  # 10% of capital per unit
                positions.append({
                    'symbol': asset,
                    'quantity': position_size,
                    'market_value': position_value,
                    'price': 100.0,  # Dummy price
                    'volatility': 0.15  # 15% volatility
                })
        
        if not positions:
            return None
        
        # Update VaR calculator with current positions
        self.var_calculator.portfolio_value = sum(pos['market_value'] for pos in positions)
        
        # Calculate VaR asynchronously
        try:
            loop = asyncio.get_event_loop()
            var_result = loop.run_until_complete(
                self.var_calculator.calculate_var(
                    confidence_level=0.95,
                    time_horizon=1,
                    method="parametric"
                )
            )
            return var_result
        except Exception as e:
            logger.error("VaR calculation failed", error=str(e))
            return None
    
    def _generate_sequential_risk_state(self) -> RiskState:
        """Generate risk state with sequential context"""
        # Calculate portfolio metrics from sequential context
        total_position_value = sum(abs(pos) for pos in self.sequential_context.position_sizing_decisions.values())
        position_count = sum(1 for pos in self.sequential_context.position_sizing_decisions.values() if pos != 0.0)
        
        # Get correlation risk
        correlation_risk = self.sequential_context.correlation_context.get('max_correlation', 0.0)
        
        # Get VaR estimate
        var_estimate = self.sequential_context.performance_metrics.get('var_percentage', 0.0)
        
        # Calculate drawdown (simplified)
        current_value = self.initial_capital + total_position_value * 0.01  # Simplified P&L
        drawdown = max(0.0, (self.initial_capital - current_value) / self.initial_capital)
        
        return RiskState(
            account_equity_normalized=current_value / self.initial_capital,
            open_positions_count=position_count,
            volatility_regime=0.5,  # Simplified volatility regime
            correlation_risk=correlation_risk,
            var_estimate_5pct=var_estimate,
            current_drawdown_pct=drawdown,
            margin_usage_pct=min(1.0, total_position_value / self.initial_capital),
            time_of_day_risk=0.5,  # Simplified time risk
            market_stress_level=len(self.sequential_context.risk_alerts) * 0.2,
            liquidity_conditions=1.0 - len(self.sequential_context.risk_actions) * 0.2
        )
    
    def _create_global_risk_state(self, risk_state: RiskState) -> GlobalRiskState:
        """Create global risk state for centralized critic"""
        base_vector = risk_state.to_vector()
        
        return GlobalRiskState(
            position_sizing_risk=base_vector,
            stop_target_risk=base_vector,
            risk_monitor_risk=base_vector,
            portfolio_optimizer_risk=base_vector,
            total_portfolio_var=self.sequential_context.performance_metrics.get('var_percentage', 0.0),
            portfolio_correlation_max=self.sequential_context.correlation_context.get('max_correlation', 0.0),
            aggregate_leverage=sum(abs(pos) for pos in self.sequential_context.position_sizing_decisions.values()),
            liquidity_risk_score=len(self.sequential_context.risk_actions) * 0.2,
            systemic_risk_level=len(self.sequential_context.risk_alerts) * 0.2,
            timestamp=datetime.now(),
            market_hours_factor=0.5
        )
    
    def _calculate_sequential_reward(self, 
                                   agent: str, 
                                   action: Union[int, np.ndarray],
                                   risk_state: RiskState,
                                   var_result: Optional[VaRResult],
                                   global_risk_value: float) -> float:
        """Calculate reward for sequential agent"""
        base_reward = 0.0
        
        # Performance component
        performance_reward = global_risk_value * 2.0
        
        # VaR performance component
        var_performance_reward = 0.0
        if var_result and var_result.performance_ms < self.performance_target_ms:
            var_performance_reward = 1.0
        
        # Sequential coordination reward
        coordination_reward = 0.0
        if self.current_phase != SequentialPhase.POSITION_SIZING:
            # Reward smooth transitions between phases
            coordination_reward = 0.5
        
        # Agent-specific rewards
        agent_reward = 0.0
        
        if agent == 'position_sizing':
            # Reward appropriate position sizing
            position_sum = sum(abs(pos) for pos in self.sequential_context.position_sizing_decisions.values())
            if position_sum > 0.0 and position_sum < 5.0:  # Reasonable position sizing
                agent_reward = 1.0
        
        elif agent == 'stop_target':
            # Reward setting appropriate stop/target levels
            if len(self.sequential_context.stop_loss_levels) > 0:
                agent_reward = 1.0
        
        elif agent == 'risk_monitor':
            # Reward appropriate risk monitoring
            if len(self.sequential_context.risk_alerts) > 0:
                agent_reward = 2.0
            elif not self.emergency_active:
                agent_reward = 0.5
        
        elif agent == 'portfolio_optimizer':
            # Reward diversification
            if len(self.sequential_context.portfolio_weights) > 0:
                weights = np.array(list(self.sequential_context.portfolio_weights.values()))
                if np.sum(weights) > 0:
                    normalized_weights = weights / np.sum(weights)
                    entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-8))
                    agent_reward = entropy * 0.5
        
        total_reward = (
            base_reward + 
            performance_reward + 
            var_performance_reward + 
            coordination_reward + 
            agent_reward
        )
        
        return float(total_reward)
    
    def _check_emergency_protocols(self):
        """Check and activate emergency protocols"""
        # Check correlation shock
        if self.sequential_context.correlation_context.get('max_correlation', 0.0) > 0.9:
            self.emergency_active = True
            self.emergency_triggers.append('correlation_shock')
        
        # Check VaR breach
        if self.sequential_context.performance_metrics.get('var_percentage', 0.0) > self.risk_tolerance:
            self.emergency_active = True
            self.emergency_triggers.append('var_breach')
        
        # Check excessive risk actions
        if len(self.sequential_context.risk_actions) > 3:
            self.emergency_active = True
            self.emergency_triggers.append('excessive_risk_actions')
    
    def _check_termination_conditions(self, operating_mode: OperatingMode):
        """Check termination conditions"""
        # Emergency termination
        if self.emergency_active or operating_mode == OperatingMode.EMERGENCY:
            for agent in self.agents:
                self.terminations[agent] = True
            return
        
        # Max steps truncation
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                self.truncations[agent] = True
    
    def _advance_sequential_state(self):
        """Advance to next sequential state"""
        # Map current agent to next phase
        phase_map = {
            'position_sizing': SequentialPhase.STOP_TARGET,
            'stop_target': SequentialPhase.RISK_MONITOR,
            'risk_monitor': SequentialPhase.PORTFOLIO_OPTIMIZER,
            'portfolio_optimizer': SequentialPhase.COMPLETE
        }
        
        # Advance phase
        self.current_phase = phase_map.get(self.agent_selection, SequentialPhase.COMPLETE)
        
        # Advance agent selection
        self.agent_selection = self.agent_selector.next()
        
        # Reset to start if sequence is complete
        if self.current_phase == SequentialPhase.COMPLETE:
            self.current_phase = SequentialPhase.POSITION_SIZING
            self.agent_selector = agent_selector(self.agents)
            self.agent_selection = self.agent_selector.next()
    
    def _encode_upstream_context(self) -> np.ndarray:
        """Encode upstream context from strategic/tactical systems"""
        # Encode strategic signals
        strategic_values = list(self.upstream_context.strategic_signals.values())[:5]
        strategic_values.extend([0.0] * (5 - len(strategic_values)))
        
        # Encode tactical signals
        tactical_values = list(self.upstream_context.tactical_signals.values())[:5]
        tactical_values.extend([0.0] * (5 - len(tactical_values)))
        
        return np.array(strategic_values + tactical_values)
    
    def _encode_position_sizing_context(self) -> np.ndarray:
        """Encode position sizing context"""
        positions = [self.sequential_context.position_sizing_decisions.get(asset, 0.0) 
                    for asset in self.asset_universe]
        return np.array(positions)
    
    def _encode_stop_target_context(self) -> np.ndarray:
        """Encode stop/target context"""
        stops = [self.sequential_context.stop_loss_levels.get(asset, 0.0) 
                for asset in self.asset_universe[:1]]  # First asset only for size
        targets = [self.sequential_context.target_levels.get(asset, 0.0) 
                  for asset in self.asset_universe[:1]]  # First asset only for size
        return np.array(stops + targets)
    
    def _encode_risk_monitor_context(self) -> np.ndarray:
        """Encode risk monitor context"""
        return np.array([
            len(self.sequential_context.risk_alerts),
            len(self.sequential_context.risk_actions),
            1.0 if self.emergency_active else 0.0,
            self.sequential_context.correlation_context.get('max_correlation', 0.0),
            self.sequential_context.performance_metrics.get('var_percentage', 0.0)
        ])
    
    def _serialize_context(self) -> Dict[str, Any]:
        """Serialize sequential context for info"""
        return {
            'position_sizing_decisions': self.sequential_context.position_sizing_decisions,
            'stop_loss_levels': self.sequential_context.stop_loss_levels,
            'target_levels': self.sequential_context.target_levels,
            'risk_alerts': self.sequential_context.risk_alerts,
            'risk_actions': self.sequential_context.risk_actions,
            'portfolio_weights': self.sequential_context.portfolio_weights,
            'correlation_context': self.sequential_context.correlation_context,
            'performance_metrics': self.sequential_context.performance_metrics
        }
    
    def _generate_risk_superposition(self) -> RiskSuperposition:
        """Generate comprehensive risk superposition output"""
        return RiskSuperposition(
            timestamp=datetime.now(),
            position_allocations=self.sequential_context.position_sizing_decisions.copy(),
            stop_loss_orders=self.sequential_context.stop_loss_levels.copy(),
            target_profit_orders=self.sequential_context.target_levels.copy(),
            risk_limits={
                'max_position_size': 0.2,
                'max_portfolio_var': self.risk_tolerance,
                'max_correlation': 0.8,
                'max_drawdown': 0.15
            },
            emergency_protocols=self.emergency_triggers.copy(),
            correlation_adjustments=self.sequential_context.correlation_context.copy(),
            var_estimates=self.sequential_context.performance_metrics.copy(),
            execution_priority=list(self.asset_universe),
            risk_attribution=self.sequential_context.position_sizing_decisions.copy(),
            confidence_scores={
                'position_sizing': 0.8,
                'stop_target': 0.9,
                'risk_monitor': 0.7,
                'portfolio_optimizer': 0.85
            },
            sequential_metadata=self.sequential_context.execution_metadata.copy()
        )
    
    def _handle_risk_breach(self, event: Event):
        """Handle risk breach events"""
        self.emergency_active = True
        self.emergency_triggers.append(f"risk_breach_{event.event_type}")
        logger.warning("Risk breach detected in sequential environment", event=event.payload)
    
    def _handle_var_update(self, event: Event):
        """Handle VaR update events"""
        var_data = event.payload
        self.sequential_context.performance_metrics['var_estimate'] = var_data.get('portfolio_var', 0.0)
    
    def _handle_correlation_shock(self, event: Event):
        """Handle correlation shock events"""
        shock_data = event.payload
        self.emergency_active = True
        self.emergency_triggers.append(f"correlation_shock_{shock_data.get('severity', 'unknown')}")
        logger.critical("Correlation shock in sequential environment", shock=shock_data)
    
    def _handle_emergency_protocol(self, event: Event):
        """Handle emergency protocol events"""
        self.emergency_active = True
        self.emergency_recovery_actions.append(event.payload)
        logger.critical("Emergency protocol activated", protocol=event.payload)
    
    def _handle_leverage_reduction(self, new_leverage: float):
        """Handle automated leverage reduction"""
        # Reduce all positions proportionally
        reduction_factor = new_leverage / max(1.0, sum(abs(pos) for pos in self.sequential_context.position_sizing_decisions.values()))
        
        for asset in self.sequential_context.position_sizing_decisions:
            self.sequential_context.position_sizing_decisions[asset] *= reduction_factor
        
        logger.info("Leverage reduction applied", new_leverage=new_leverage, reduction_factor=reduction_factor)
    
    def update_upstream_context(self, upstream_context: Dict[str, Any]):
        """Update upstream context from strategic/tactical systems"""
        self.upstream_context = UpstreamContext(**upstream_context)
        logger.info("Upstream context updated", context_keys=list(upstream_context.keys()))
    
    def get_risk_superposition(self) -> Optional[RiskSuperposition]:
        """Get latest risk superposition"""
        return self.risk_superpositions[-1] if self.risk_superpositions else None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'episode_length': self.current_step,
            'avg_step_time_ms': np.mean(self.step_times) * 1000 if self.step_times else 0,
            'avg_var_calculation_time_ms': np.mean(self.var_calculation_times) if self.var_calculation_times else 0,
            'var_performance_target_met': np.mean(self.var_calculation_times) < self.performance_target_ms if self.var_calculation_times else False,
            'emergency_activations': len(self.emergency_triggers),
            'risk_superpositions_generated': len(self.risk_superpositions),
            'sequential_completions': len(self.sequence_completion_times),
            'current_phase': self.current_phase.value,
            'correlation_regime': self.correlation_tracker.current_regime.value,
            'position_count': len([pos for pos in self.sequential_context.position_sizing_decisions.values() if pos != 0.0])
        }
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n=== Sequential Risk Environment Step {self.current_step} ===")
            print(f"Current Phase: {self.current_phase.value}")
            print(f"Current Agent: {self.agent_selection}")
            print(f"Emergency Active: {self.emergency_active}")
            print(f"Position Sizing Decisions: {len(self.sequential_context.position_sizing_decisions)}")
            print(f"Stop/Target Levels: {len(self.sequential_context.stop_loss_levels)}")
            print(f"Risk Alerts: {len(self.sequential_context.risk_alerts)}")
            print(f"Risk Actions: {len(self.sequential_context.risk_actions)}")
            print(f"Portfolio Weights: {len(self.sequential_context.portfolio_weights)}")
            print(f"Correlation Regime: {self.correlation_tracker.current_regime.value}")
            print(f"VaR Performance: {np.mean(self.var_calculation_times):.2f}ms" if self.var_calculation_times else "N/A")
            print(f"Risk Superpositions: {len(self.risk_superpositions)}")
            print("=" * 50)
        
        elif mode == 'rgb_array':
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up environment resources"""
        if hasattr(self.state_processor, 'shutdown'):
            self.state_processor.shutdown()
        
        if hasattr(self.correlation_tracker, 'shutdown'):
            self.correlation_tracker.shutdown()
        
        logger.info("Sequential Risk Environment closed")


def create_sequential_risk_environment(config: Optional[Dict[str, Any]] = None) -> SequentialRiskEnvironment:
    """
    Factory function to create sequential risk environment
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured SequentialRiskEnvironment instance
    """
    default_config = {
        'initial_capital': 1_000_000.0,
        'max_steps': 1000,
        'risk_tolerance': 0.05,
        'performance_target_ms': 5.0,
        'asset_universe': ['SPY', 'QQQ', 'IWM', 'VTI', 'TLT', 'GLD', 'VIX', 'UUP', 'EFA', 'EEM'],
        'critic_config': {
            'hidden_dim': 256,
            'num_layers': 4,
            'learning_rate': 0.0005,
            'target_update_freq': 50
        }
    }
    
    if config:
        default_config.update(config)
    
    return SequentialRiskEnvironment(default_config)