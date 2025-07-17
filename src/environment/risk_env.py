"""
PettingZoo Risk Management Environment

A comprehensive PettingZoo AEC environment for multi-agent risk management.
This environment integrates with the existing risk management system including
VaR correlation tracking, centralized critic, and specialized risk agents.

Features:
- 4 specialized risk agents with unique action/observation spaces
- Integration with existing CorrelationTracker and VaRCalculator
- Real-time risk state processing and validation
- Comprehensive reward system for multi-agent coordination
- Turn-based execution with proper agent selection
- Emergency protocols and risk event handling

Agent Roles:
- π₁ (position_sizing): Manages portfolio position sizes
- π₂ (stop_target): Sets stop losses and profit targets
- π₃ (risk_monitor): Monitors and responds to risk events
- π₄ (portfolio_optimizer): Optimizes portfolio allocation
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

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import structlog

# Import existing risk management components
from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime, CorrelationShock
from src.risk.core.var_calculator import VaRCalculator
from src.risk.core.state_processor import RiskStateProcessor, StateProcessingConfig
from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.risk.marl.centralized_critic import CentralizedCritic, GlobalRiskState, OperatingMode

logger = structlog.get_logger()


class RiskEnvironmentState(Enum):
    """State machine for managing risk agent turn sequence"""
    AWAITING_POSITION_SIZING = "awaiting_position_sizing"
    AWAITING_STOP_TARGET = "awaiting_stop_target"
    AWAITING_RISK_MONITOR = "awaiting_risk_monitor"
    AWAITING_PORTFOLIO_OPTIMIZER = "awaiting_portfolio_optimizer"
    READY_FOR_AGGREGATION = "ready_for_aggregation"
    EMERGENCY_HALT = "emergency_halt"


class RiskScenario(Enum):
    """Risk scenarios for testing and training"""
    NORMAL_OPERATIONS = "normal"
    CORRELATION_SPIKE = "correlation_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"
    LEVERAGE_UNWIND = "leverage_unwind"
    BLACK_SWAN = "black_swan"


@dataclass
class PortfolioState:
    """Current portfolio state for risk management"""
    positions: Dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    total_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 0.0
    var_estimate: float = 0.0
    expected_shortfall: float = 0.0
    max_correlation: float = 0.0
    drawdown: float = 0.0
    margin_usage: float = 0.0
    position_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MarketConditions:
    """Current market conditions for risk assessment"""
    volatility_regime: float = 0.5  # 0-1 scale
    correlation_level: float = 0.3  # Average portfolio correlation
    liquidity_score: float = 0.8    # 0-1, higher is better
    stress_level: float = 0.1       # 0-1, higher is more stressed
    time_of_day_risk: float = 0.5   # 0-1, based on market hours
    regime_stability: float = 0.8    # How stable the current regime is
    vix_level: float = 20.0         # Volatility index level
    credit_spread: float = 0.02     # Credit spread indicator
    momentum_factor: float = 0.0    # Market momentum
    sentiment_score: float = 0.5    # Market sentiment (0-1)


class RiskManagementEnv(AECEnv):
    """
    PettingZoo AEC Environment for Risk Management MARL System
    
    This environment manages 4 specialized risk agents in a turn-based setting,
    integrating with existing risk management infrastructure.
    
    Agent Specifications:
    - π₁ (position_sizing): Discrete(7) actions for position adjustments
    - π₂ (stop_target): Box(2,) continuous actions for stop/target multipliers
    - π₃ (risk_monitor): Discrete(5) actions for risk monitoring responses
    - π₄ (portfolio_optimizer): Box(10,) continuous actions for portfolio weights
    
    Observation Space: 10-dimensional risk state vector (standardized)
    """
    
    metadata = {
        'name': 'risk_management_v1',
        'render_modes': ['human', 'rgb_array'],
        'is_parallelizable': False,
        'render_fps': 4
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Risk Management Environment
        
        Args:
            config: Configuration dictionary containing:
                - initial_capital: Starting capital (default: 1M)
                - max_steps: Maximum episode steps (default: 1000)
                - risk_tolerance: Risk tolerance level (default: 0.05)
                - asset_universe: List of tradeable assets
                - scenario: Risk scenario to simulate
                - performance_target_ms: Performance target in milliseconds
        """
        super().__init__()
        
        self.config = config
        self.initial_capital = config.get('initial_capital', 1_000_000.0)
        self.max_steps = config.get('max_steps', 1000)
        self.risk_tolerance = config.get('risk_tolerance', 0.05)
        self.performance_target_ms = config.get('performance_target_ms', 5.0)
        
        # Asset universe
        self.asset_universe = config.get('asset_universe', [
            'SPY', 'QQQ', 'IWM', 'VTI', 'TLT', 'GLD', 'VIX', 'UUP', 'EFA', 'EEM'
        ])
        
        # Risk scenario
        self.risk_scenario = RiskScenario(config.get('scenario', 'normal'))
        
        # Agent configuration
        self.agents = ['position_sizing', 'stop_target', 'risk_monitor', 'portfolio_optimizer']
        self.possible_agents = self.agents.copy()
        self.num_agents = len(self.agents)
        
        # Initialize components
        self.event_bus = EventBus()
        self._initialize_risk_components()
        
        # Environment state
        self.env_state = RiskEnvironmentState.AWAITING_POSITION_SIZING
        self.current_step = 0
        self.episode_start_time = None
        self.last_action_time = None
        
        # Portfolio and market state
        self.portfolio_state = PortfolioState()
        self.market_conditions = MarketConditions()
        
        # Agent selector for turn-based execution
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.next()
        
        # Define action and observation spaces
        self._action_spaces = self._define_action_spaces()
        self._observation_spaces = self._define_observation_spaces()
        
        # Episode tracking
        self.episode_rewards = {agent: [] for agent in self.agents}
        self.episode_actions = {agent: [] for agent in self.agents}
        self.risk_events = []
        self.emergency_stops = 0
        
        # Performance tracking
        self.step_times = deque(maxlen=100)
        self.action_counts = defaultdict(int)
        
        # Termination conditions
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        logger.info("Risk Management Environment initialized", 
                   agents=self.agents, 
                   initial_capital=self.initial_capital,
                   asset_universe=len(self.asset_universe),
                   scenario=self.risk_scenario.value)
    
    def _initialize_risk_components(self):
        """Initialize risk management components"""
        # Correlation tracker
        self.correlation_tracker = CorrelationTracker(
            event_bus=self.event_bus,
            ewma_lambda=0.94,
            shock_threshold=0.5,
            shock_window_minutes=10,
            performance_target_ms=self.performance_target_ms
        )
        
        # VaR calculator
        self.var_calculator = VaRCalculator(
            correlation_tracker=self.correlation_tracker,
            event_bus=self.event_bus
        )
        
        # State processor
        state_config = StateProcessingConfig(
            lookback_window=100,
            normalization_method='zscore',
            outlier_threshold=3.0,
            smoothing_alpha=0.1
        )
        self.state_processor = RiskStateProcessor(state_config, self.event_bus)
        
        # Centralized critic
        critic_config = self.config.get('critic_config', {
            'hidden_dim': 128,
            'num_layers': 3,
            'learning_rate': 0.001,
            'target_update_freq': 100
        })
        self.centralized_critic = CentralizedCritic(critic_config, self.event_bus)
        
        # Initialize asset universe in correlation tracker
        self.correlation_tracker.initialize_assets(self.asset_universe)
    
    def _define_action_spaces(self) -> Dict[str, spaces.Space]:
        """Define action spaces for each risk agent"""
        return {
            # π₁ Position Sizing: Discrete actions for position adjustments
            'position_sizing': spaces.Discrete(7),  # [reduce_large, reduce_med, reduce_small, hold, increase_small, increase_med, increase_large]
            
            # π₂ Stop/Target: Continuous multipliers for stop losses and targets
            'stop_target': spaces.Box(low=0.5, high=5.0, shape=(2,), dtype=np.float32),  # [stop_multiplier, target_multiplier]
            
            # π₃ Risk Monitor: Discrete actions for risk responses
            'risk_monitor': spaces.Discrete(5),  # [no_action, alert, reduce_risk, emergency_stop, hedge]
            
            # π₄ Portfolio Optimizer: Continuous weights for portfolio allocation
            'portfolio_optimizer': spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)  # Asset weights
        }
    
    def _define_observation_spaces(self) -> Dict[str, spaces.Space]:
        """Define observation spaces for each risk agent"""
        # All agents observe the same 10-dimensional risk state vector
        base_obs_space = spaces.Box(low=-5.0, high=5.0, shape=(10,), dtype=np.float32)
        
        return {
            'position_sizing': base_obs_space,
            'stop_target': base_obs_space,
            'risk_monitor': base_obs_space,
            'portfolio_optimizer': base_obs_space
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
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset environment state
        self.env_state = RiskEnvironmentState.AWAITING_POSITION_SIZING
        self.current_step = 0
        self.episode_start_time = datetime.now()
        self.last_action_time = None
        self.emergency_stops = 0
        
        # Reset portfolio state
        self.portfolio_state = PortfolioState(
            positions={asset: 0.0 for asset in self.asset_universe},
            cash=self.initial_capital,
            total_value=self.initial_capital,
            position_count=0,
            last_updated=datetime.now()
        )
        
        # Reset market conditions
        self.market_conditions = MarketConditions()
        
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
        self.risk_events.clear()
        
        # Reset termination conditions
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Apply risk scenario modifications
        self._apply_risk_scenario()
        
        logger.info("Environment reset", 
                   scenario=self.risk_scenario.value,
                   initial_capital=self.initial_capital,
                   agent_selection=self.agent_selection)
    
    def step(self, action: Union[int, np.ndarray]) -> None:
        """
        Execute one step in the environment
        
        Args:
            action: Action from the current agent
        """
        step_start_time = time.time()
        
        # Validate action
        if not self._validate_action(action):
            logger.error("Invalid action received", 
                        agent=self.agent_selection,
                        action=action,
                        expected_space=self.action_space)
            # Apply penalty and move to next agent
            self.rewards[self.agent_selection] = -10.0
            self._advance_agent_selection()
            return
        
        # Store action
        self.episode_actions[self.agent_selection].append(action)
        self.action_counts[self.agent_selection] += 1
        
        # Apply agent action
        portfolio_changes = self._apply_agent_action(self.agent_selection, action)
        
        # Simulate market dynamics
        market_changes = self._simulate_market_step()
        
        # Update portfolio state
        self._update_portfolio_state(portfolio_changes, market_changes)
        
        # Check for risk events
        current_risk_events = self._check_risk_events()
        self.risk_events.extend(current_risk_events)
        
        # Generate current risk state
        risk_state = self._generate_risk_state()
        
        # Process risk state
        normalized_state, processing_metadata = self.state_processor.process_state(risk_state.to_vector())
        
        # Evaluate global risk with centralized critic
        global_risk_state = self._create_global_risk_state(risk_state)
        global_risk_value, operating_mode = self.centralized_critic.evaluate_global_risk(global_risk_state)
        
        # Calculate reward for current agent
        reward = self._calculate_agent_reward(
            self.agent_selection, 
            action, 
            risk_state, 
            global_risk_value, 
            current_risk_events
        )
        
        self.rewards[self.agent_selection] = reward
        self.episode_rewards[self.agent_selection].append(reward)
        
        # Update info
        self.infos[self.agent_selection] = {
            'step': self.current_step,
            'portfolio_value': self.portfolio_state.total_value,
            'var_estimate': self.portfolio_state.var_estimate,
            'global_risk_value': global_risk_value,
            'operating_mode': operating_mode.value,
            'risk_events': current_risk_events,
            'processing_time_ms': (time.time() - step_start_time) * 1000,
            'leverage': self.portfolio_state.leverage,
            'drawdown': self.portfolio_state.drawdown,
            'correlation': self.portfolio_state.max_correlation
        }
        
        # Check termination conditions
        self._check_termination_conditions(current_risk_events, operating_mode)
        
        # Advance to next agent
        self._advance_agent_selection()
        
        # Update step counter and timing
        self.current_step += 1
        self.last_action_time = datetime.now()
        self.step_times.append(time.time() - step_start_time)
        
        logger.debug("Step completed", 
                    agent=self.agent_selection,
                    step=self.current_step,
                    reward=reward,
                    portfolio_value=self.portfolio_state.total_value)
    
    def observe(self, agent: str) -> np.ndarray:
        """
        Get observation for specified agent
        
        Args:
            agent: Agent identifier
            
        Returns:
            Normalized risk state vector
        """
        risk_state = self._generate_risk_state()
        normalized_state, _ = self.state_processor.process_state(risk_state.to_vector())
        return normalized_state.astype(np.float32)
    
    def _validate_action(self, action: Union[int, np.ndarray]) -> bool:
        """Validate action against current agent's action space"""
        try:
            return self.action_space.contains(action)
        except (TypeError, ValueError):
            return False
    
    def _apply_agent_action(self, agent: str, action: Union[int, np.ndarray]) -> Dict[str, Any]:
        """
        Apply agent action to portfolio
        
        Args:
            agent: Agent identifier
            action: Action to apply
            
        Returns:
            Dictionary of portfolio changes
        """
        changes = {
            'position_adjustments': {},
            'stop_loss_updates': {},
            'target_updates': {},
            'risk_actions': [],
            'portfolio_rebalancing': {},
            'emergency_actions': []
        }
        
        if agent == 'position_sizing':
            # π₁: Position sizing actions
            action_map = {
                0: -0.5,   # reduce_large
                1: -0.25,  # reduce_med
                2: -0.1,   # reduce_small
                3: 0.0,    # hold
                4: 0.1,    # increase_small
                5: 0.25,   # increase_med
                6: 0.5     # increase_large
            }
            
            adjustment = action_map[action]
            changes['position_adjustments'] = {
                asset: adjustment for asset in self.asset_universe
            }
        
        elif agent == 'stop_target':
            # π₂: Stop/target multipliers
            changes['stop_loss_updates'] = {'multiplier': float(action[0])}
            changes['target_updates'] = {'multiplier': float(action[1])}
        
        elif agent == 'risk_monitor':
            # π₃: Risk monitoring actions
            action_map = {
                0: 'no_action',
                1: 'alert',
                2: 'reduce_risk',
                3: 'emergency_stop',
                4: 'hedge'
            }
            
            risk_action = action_map[action]
            changes['risk_actions'].append(risk_action)
            
            if risk_action == 'reduce_risk':
                changes['position_adjustments'] = {
                    asset: -0.2 for asset in self.asset_universe
                }
            elif risk_action == 'emergency_stop':
                changes['emergency_actions'].append('halt_trading')
                changes['position_adjustments'] = {
                    asset: -1.0 for asset in self.asset_universe
                }
                self.emergency_stops += 1
        
        elif agent == 'portfolio_optimizer':
            # π₄: Portfolio optimization weights
            # Normalize weights to sum to 1
            weights = np.array(action)
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                normalized_weights = weights / weight_sum
                changes['portfolio_rebalancing'] = {
                    asset: float(normalized_weights[i]) 
                    for i, asset in enumerate(self.asset_universe)
                }
        
        return changes
    
    def _simulate_market_step(self) -> Dict[str, Any]:
        """Simulate market dynamics for one step"""
        # Generate correlated returns based on current market conditions
        base_volatility = 0.01  # 1% daily volatility
        
        # Adjust volatility based on market conditions
        vol_multiplier = 1.0 + self.market_conditions.stress_level * 2.0
        
        # Generate market-wide shock
        market_shock = np.random.normal(0, base_volatility * vol_multiplier)
        
        # Generate asset-specific returns
        returns = {}
        for asset in self.asset_universe:
            # Correlation component
            corr_component = market_shock * self.market_conditions.correlation_level
            
            # Idiosyncratic component
            idio_component = np.random.normal(0, base_volatility * 0.5) * (
                1 - self.market_conditions.correlation_level
            )
            
            returns[asset] = corr_component + idio_component
        
        # Update market conditions
        self.market_conditions.volatility_regime = np.clip(
            self.market_conditions.volatility_regime + np.random.normal(0, 0.02),
            0.0, 1.0
        )
        
        self.market_conditions.stress_level = np.clip(
            self.market_conditions.stress_level + np.random.normal(0, 0.01),
            0.0, 1.0
        )
        
        # Update correlation based on stress level
        self.market_conditions.correlation_level = np.clip(
            self.market_conditions.correlation_level + 
            self.market_conditions.stress_level * 0.1 + 
            np.random.normal(0, 0.02),
            0.0, 1.0
        )
        
        return {
            'returns': returns,
            'market_shock': market_shock,
            'volatility_change': vol_multiplier - 1.0
        }
    
    def _update_portfolio_state(self, portfolio_changes: Dict[str, Any], market_changes: Dict[str, Any]):
        """Update portfolio state based on actions and market movements"""
        # Apply position adjustments
        for asset, adjustment in portfolio_changes.get('position_adjustments', {}).items():
            current_position = self.portfolio_state.positions.get(asset, 0.0)
            new_position = current_position * (1 + adjustment)
            self.portfolio_state.positions[asset] = max(0.0, new_position)  # No short positions
        
        # Apply portfolio rebalancing
        if 'portfolio_rebalancing' in portfolio_changes:
            total_value = self.portfolio_state.total_value
            for asset, weight in portfolio_changes['portfolio_rebalancing'].items():
                target_value = total_value * weight
                self.portfolio_state.positions[asset] = target_value
        
        # Apply market returns
        total_pnl = 0.0
        for asset, position in self.portfolio_state.positions.items():
            if position > 0 and asset in market_changes['returns']:
                pnl = position * market_changes['returns'][asset]
                total_pnl += pnl
        
        # Update portfolio metrics
        self.portfolio_state.unrealized_pnl = total_pnl
        self.portfolio_state.total_value = (
            self.portfolio_state.cash + 
            sum(self.portfolio_state.positions.values()) + 
            total_pnl
        )
        
        # Update position count
        self.portfolio_state.position_count = sum(
            1 for pos in self.portfolio_state.positions.values() if pos > 0
        )
        
        # Update leverage
        total_position_value = sum(self.portfolio_state.positions.values())
        self.portfolio_state.leverage = total_position_value / max(self.portfolio_state.total_value, 1.0)
        
        # Update drawdown
        peak_value = max(self.initial_capital, self.portfolio_state.total_value)
        current_drawdown = (peak_value - self.portfolio_state.total_value) / peak_value
        self.portfolio_state.drawdown = max(self.portfolio_state.drawdown, current_drawdown)
        
        # Update VaR estimate (simplified)
        self.portfolio_state.var_estimate = min(
            self.risk_tolerance * self.portfolio_state.total_value,
            self.portfolio_state.leverage * 0.02 * self.portfolio_state.total_value
        )
        
        # Update correlation
        if hasattr(self.correlation_tracker, 'correlation_matrix') and self.correlation_tracker.correlation_matrix is not None:
            self.portfolio_state.max_correlation = np.max(
                self.correlation_tracker.correlation_matrix
            )
        
        self.portfolio_state.last_updated = datetime.now()
    
    def _check_risk_events(self) -> List[str]:
        """Check for risk events and violations"""
        events = []
        
        # Drawdown check
        if self.portfolio_state.drawdown > 0.15:
            events.append('excessive_drawdown')
        
        # Leverage check
        if self.portfolio_state.leverage > 4.0:
            events.append('excessive_leverage')
        
        # VaR breach check
        var_limit = self.risk_tolerance * self.portfolio_state.total_value
        if self.portfolio_state.var_estimate > var_limit:
            events.append('var_breach')
        
        # Correlation risk check
        if self.portfolio_state.max_correlation > 0.85:
            events.append('correlation_spike')
        
        # Market stress check
        if self.market_conditions.stress_level > 0.8:
            events.append('market_stress')
        
        # Liquidity check
        if self.market_conditions.liquidity_score < 0.3:
            events.append('liquidity_crisis')
        
        return events
    
    def _generate_risk_state(self) -> RiskState:
        """Generate current risk state vector"""
        return RiskState(
            account_equity_normalized=self.portfolio_state.total_value / self.initial_capital,
            open_positions_count=self.portfolio_state.position_count,
            volatility_regime=self.market_conditions.volatility_regime,
            correlation_risk=self.portfolio_state.max_correlation,
            var_estimate_5pct=self.portfolio_state.var_estimate / max(self.portfolio_state.total_value, 1.0),
            current_drawdown_pct=self.portfolio_state.drawdown,
            margin_usage_pct=min(1.0, self.portfolio_state.leverage / 4.0),
            time_of_day_risk=self.market_conditions.time_of_day_risk,
            market_stress_level=self.market_conditions.stress_level,
            liquidity_conditions=self.market_conditions.liquidity_score
        )
    
    def _create_global_risk_state(self, risk_state: RiskState) -> GlobalRiskState:
        """Create global risk state for centralized critic"""
        base_vector = risk_state.to_vector()
        
        return GlobalRiskState(
            position_sizing_risk=base_vector,
            stop_target_risk=base_vector,
            risk_monitor_risk=base_vector,
            portfolio_optimizer_risk=base_vector,
            total_portfolio_var=self.portfolio_state.var_estimate / max(self.portfolio_state.total_value, 1.0),
            portfolio_correlation_max=self.portfolio_state.max_correlation,
            aggregate_leverage=self.portfolio_state.leverage,
            liquidity_risk_score=1.0 - self.market_conditions.liquidity_score,
            systemic_risk_level=self.market_conditions.stress_level,
            timestamp=datetime.now(),
            market_hours_factor=self.market_conditions.time_of_day_risk
        )
    
    def _calculate_agent_reward(self, 
                               agent: str, 
                               action: Union[int, np.ndarray], 
                               risk_state: RiskState,
                               global_risk_value: float,
                               risk_events: List[str]) -> float:
        """Calculate reward for specific agent"""
        base_reward = 0.0
        
        # Portfolio performance component
        performance_reward = (
            (self.portfolio_state.total_value - self.initial_capital) / self.initial_capital
        ) * 10.0
        
        # Risk penalty component
        risk_penalty = len(risk_events) * -2.0
        
        # Global risk component
        global_risk_bonus = global_risk_value * 5.0
        
        # Agent-specific rewards
        agent_specific_reward = 0.0
        
        if agent == 'position_sizing':
            # Reward appropriate position sizing
            if 'excessive_leverage' in risk_events and action < 3:  # Reducing
                agent_specific_reward = 2.0
            elif 'excessive_leverage' not in risk_events and action > 3:  # Increasing
                agent_specific_reward = 1.0
        
        elif agent == 'stop_target':
            # Reward tighter stops in volatile conditions
            if self.market_conditions.volatility_regime > 0.7:
                if action[0] < 1.5:  # Tight stop
                    agent_specific_reward = 1.0
        
        elif agent == 'risk_monitor':
            # Reward appropriate risk monitoring
            if risk_events and action > 0:  # Acting on risks
                agent_specific_reward = 3.0
            elif not risk_events and action == 0:  # No action when safe
                agent_specific_reward = 0.5
            else:
                agent_specific_reward = -1.0  # Inappropriate action
        
        elif agent == 'portfolio_optimizer':
            # Reward diversification
            weights = np.array(action)
            weight_entropy = -np.sum(weights * np.log(weights + 1e-8))
            agent_specific_reward = weight_entropy * 0.5
        
        total_reward = (
            base_reward + 
            performance_reward + 
            risk_penalty + 
            global_risk_bonus + 
            agent_specific_reward
        )
        
        return float(total_reward)
    
    def _check_termination_conditions(self, risk_events: List[str], operating_mode: OperatingMode):
        """Check if episode should terminate"""
        # Emergency stop termination
        if self.emergency_stops > 0 or operating_mode == OperatingMode.EMERGENCY:
            for agent in self.agents:
                self.terminations[agent] = True
            return
        
        # Excessive loss termination
        if self.portfolio_state.total_value < 0.5 * self.initial_capital:
            for agent in self.agents:
                self.terminations[agent] = True
            return
        
        # Max steps truncation
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                self.truncations[agent] = True
            return
    
    def _advance_agent_selection(self):
        """Advance to next agent in sequence"""
        self.agent_selection = self.agent_selector.next()
        
        # Update environment state based on current agent
        state_map = {
            'position_sizing': RiskEnvironmentState.AWAITING_POSITION_SIZING,
            'stop_target': RiskEnvironmentState.AWAITING_STOP_TARGET,
            'risk_monitor': RiskEnvironmentState.AWAITING_RISK_MONITOR,
            'portfolio_optimizer': RiskEnvironmentState.AWAITING_PORTFOLIO_OPTIMIZER
        }
        
        self.env_state = state_map.get(self.agent_selection, RiskEnvironmentState.AWAITING_POSITION_SIZING)
    
    def _apply_risk_scenario(self):
        """Apply risk scenario modifications to environment"""
        if self.risk_scenario == RiskScenario.CORRELATION_SPIKE:
            self.market_conditions.correlation_level = 0.9
            self.market_conditions.stress_level = 0.8
        
        elif self.risk_scenario == RiskScenario.LIQUIDITY_CRISIS:
            self.market_conditions.liquidity_score = 0.2
            self.market_conditions.stress_level = 0.9
        
        elif self.risk_scenario == RiskScenario.FLASH_CRASH:
            self.market_conditions.volatility_regime = 0.95
            self.market_conditions.stress_level = 0.95
        
        elif self.risk_scenario == RiskScenario.BLACK_SWAN:
            self.market_conditions.correlation_level = 0.95
            self.market_conditions.volatility_regime = 0.99
            self.market_conditions.stress_level = 0.99
            self.market_conditions.liquidity_score = 0.1
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n=== Risk Management Environment Step {self.current_step} ===")
            print(f"Current Agent: {self.agent_selection}")
            print(f"Environment State: {self.env_state.value}")
            print(f"Portfolio Value: ${self.portfolio_state.total_value:,.2f}")
            print(f"Initial Capital: ${self.initial_capital:,.2f}")
            print(f"P&L: ${self.portfolio_state.total_value - self.initial_capital:,.2f}")
            print(f"Drawdown: {self.portfolio_state.drawdown:.2%}")
            print(f"Leverage: {self.portfolio_state.leverage:.2f}x")
            print(f"VaR: ${self.portfolio_state.var_estimate:,.2f}")
            print(f"Positions: {self.portfolio_state.position_count}")
            print(f"Max Correlation: {self.portfolio_state.max_correlation:.3f}")
            print(f"Market Stress: {self.market_conditions.stress_level:.2f}")
            print(f"Volatility Regime: {self.market_conditions.volatility_regime:.2f}")
            print(f"Liquidity Score: {self.market_conditions.liquidity_score:.2f}")
            print(f"Emergency Stops: {self.emergency_stops}")
            print(f"Risk Events: {len(self.risk_events)}")
            if self.risk_events:
                print(f"Recent Events: {self.risk_events[-3:]}")
            print("=" * 50)
        
        elif mode == 'rgb_array':
            # Return RGB array for video recording
            # Implementation would create visual representation
            return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up environment resources"""
        if hasattr(self.state_processor, 'shutdown'):
            self.state_processor.shutdown()
        
        if hasattr(self.correlation_tracker, 'shutdown'):
            self.correlation_tracker.shutdown()
        
        logger.info("Risk Management Environment closed")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'episode_length': self.current_step,
            'portfolio_return': (self.portfolio_state.total_value - self.initial_capital) / self.initial_capital,
            'max_drawdown': self.portfolio_state.drawdown,
            'risk_events_count': len(self.risk_events),
            'emergency_stops': self.emergency_stops,
            'avg_step_time_ms': np.mean(self.step_times) * 1000 if self.step_times else 0,
            'action_counts': dict(self.action_counts),
            'current_leverage': self.portfolio_state.leverage,
            'current_var': self.portfolio_state.var_estimate,
            'correlation_level': self.portfolio_state.max_correlation,
            'stress_level': self.market_conditions.stress_level
        }


def create_risk_environment(config: Optional[Dict[str, Any]] = None) -> RiskManagementEnv:
    """
    Factory function to create risk management environment
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RiskManagementEnv instance
    """
    default_config = {
        'initial_capital': 1_000_000.0,
        'max_steps': 1000,
        'risk_tolerance': 0.05,
        'scenario': 'normal',
        'performance_target_ms': 5.0,
        'asset_universe': ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VIX', 'UUP', 'EFA', 'EEM', 'DIA'],
        'critic_config': {
            'hidden_dim': 128,
            'num_layers': 3,
            'learning_rate': 0.001,
            'target_update_freq': 100
        }
    }
    
    if config:
        default_config.update(config)
    
    return RiskManagementEnv(default_config)