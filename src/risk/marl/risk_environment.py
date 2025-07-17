"""
Multi-Agent Risk Management Environment

Gym-compatible environment for training and running the 4-agent MARL risk system.
Provides coordinated simulation of portfolio risk management with realistic
market dynamics and risk scenarios.

Features:
- 4 specialized risk agents: Position Sizing (π₁), Stop/Target (π₂), Risk Monitor (π₃), Portfolio Optimizer (π₄)
- 10-dimensional risk state space
- Coordinated multi-agent rewards
- Real-time market simulation
- Emergency scenario testing
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import structlog
from datetime import datetime, timedelta
import random
from collections import defaultdict

from src.core.events import EventBus, Event, EventType
from src.risk.marl.centralized_critic import CentralizedCritic, GlobalRiskState
from src.risk.core.state_processor import RiskStateProcessor, StateProcessingConfig
from src.risk.agents.base_risk_agent import RiskState, RiskAction

logger = structlog.get_logger()


class MarketRegime(Enum):
    """Market regimes for simulation"""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_vol"
    CRISIS = "crisis"
    TRENDING = "trending"
    SIDEWAYS = "sideways"


class RiskScenario(Enum):
    """Risk scenarios for testing"""
    NORMAL_OPERATIONS = "normal"
    CORRELATION_SPIKE = "correlation_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"
    LEVERAGE_UNWIND = "leverage_unwind"
    BLACK_SWAN = "black_swan"


@dataclass
class PortfolioState:
    """Current portfolio state for simulation"""
    positions: Dict[str, float]  # symbol -> position size
    cash: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: float
    var_estimate: float
    max_correlation: float
    drawdown: float
    margin_usage: float


@dataclass
class MarketConditions:
    """Current market conditions"""
    regime: MarketRegime
    volatility_percentile: float  # 0-1
    correlation_level: float  # Average pairwise correlation
    liquidity_score: float  # 0-1, higher is better
    stress_level: float  # 0-1, higher is more stressed
    time_factor: float  # 0-1, based on time of day


class RiskEnvironment(gym.Env):
    """
    Multi-agent risk management environment
    
    Observation Space: 10-dimensional risk vector for each agent
    Action Spaces:
    - π₁ (Position Sizing): Discrete(5) - [reduce_large, reduce_small, hold, increase_small, increase_large]
    - π₂ (Stop/Target): Box(0.5, 3.0, (2,)) - [stop_multiplier, target_multiplier]  
    - π₃ (Risk Monitor): Discrete(4) - [no_action, alert, reduce_risk, emergency_stop]
    - π₄ (Portfolio Optimizer): Box(0.0, 1.0, (5,)) - [equity_weight, fixed_income, commodities, cash, alternatives]
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk environment
        
        Args:
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config
        self.max_steps = config.get('max_steps', 10000)
        self.initial_capital = config.get('initial_capital', 1000000.0)  # $1M default
        self.risk_free_rate = config.get('risk_free_rate', 0.02)  # 2% annual
        
        # Agent configuration
        self.num_agents = 4
        self.agent_names = ['position_sizing', 'stop_target', 'risk_monitor', 'portfolio_optimizer']
        
        # Define action spaces for each agent
        self.action_spaces = {
            'position_sizing': spaces.Discrete(5),  # π₁
            'stop_target': spaces.Box(low=0.5, high=3.0, shape=(2,), dtype=np.float32),  # π₂
            'risk_monitor': spaces.Discrete(4),  # π₃
            'portfolio_optimizer': spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)  # π₄
        }
        
        # Observation space: 10-dimensional risk vector
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(10,), dtype=np.float32
        )
        
        # Initialize components
        self.event_bus = EventBus()
        
        # State processor
        state_config = StateProcessingConfig()
        self.state_processor = RiskStateProcessor(state_config, self.event_bus)
        
        # Centralized critic
        critic_config = config.get('critic_config', {})
        self.centralized_critic = CentralizedCritic(critic_config, self.event_bus)
        
        # Environment state
        self.current_step = 0
        self.portfolio_state = None
        self.market_conditions = None
        self.risk_scenario = RiskScenario.NORMAL_OPERATIONS
        
        # Performance tracking
        self.episode_rewards = []
        self.agent_actions_history = {name: [] for name in self.agent_names}
        self.risk_events = []
        self.emergency_stops = 0
        
        # Market simulation parameters
        self.market_symbols = config.get('symbols', ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'])
        self.correlation_matrix = self._initialize_correlation_matrix()
        self.volatility_targets = {symbol: 0.15 + 0.1 * random.random() for symbol in self.market_symbols}
        
        logger.info("Risk environment initialized",
                   max_steps=self.max_steps,
                   num_agents=self.num_agents,
                   symbols=self.market_symbols)
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state
        
        Returns:
            Initial observations for all agents
        """
        self.current_step = 0
        self.emergency_stops = 0
        
        # Reset portfolio state
        self.portfolio_state = PortfolioState(
            positions={symbol: 0.0 for symbol in self.market_symbols},
            cash=self.initial_capital,
            total_value=self.initial_capital,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            leverage=0.0,
            var_estimate=0.0,
            max_correlation=0.0,
            drawdown=0.0,
            margin_usage=0.0
        )
        
        # Reset market conditions
        self.market_conditions = MarketConditions(
            regime=MarketRegime.NORMAL,
            volatility_percentile=0.5,
            correlation_level=0.3,
            liquidity_score=0.8,
            stress_level=0.1,
            time_factor=0.5
        )
        
        # Reset components
        self.state_processor.reset_statistics()
        self.centralized_critic.reset()
        
        # Clear history
        self.episode_rewards.clear()
        for name in self.agent_names:
            self.agent_actions_history[name].clear()
        self.risk_events.clear()
        
        # Generate initial risk state
        risk_state = self._generate_risk_state()
        
        # Process and return observations
        observations = {}
        for agent_name in self.agent_names:
            normalized_state, _ = self.state_processor.process_state(risk_state.to_vector())
            observations[agent_name] = normalized_state.astype(np.float32)
        
        logger.info("Environment reset")
        return observations
    
    def step(self, actions: Dict[str, Union[int, np.ndarray]]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        bool,                   # done
        Dict[str, Any]         # info
    ]:
        """
        Execute one step in the environment
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        self.current_step += 1
        step_start_time = datetime.now()
        
        # Validate actions
        if not self._validate_actions(actions):
            logger.error("Invalid actions received", actions=actions)
            # Return safe state
            return self._get_safe_step_result()
        
        # Store actions in history
        for agent_name, action in actions.items():
            self.agent_actions_history[agent_name].append(action)
        
        # Apply agent actions to portfolio
        portfolio_changes = self._apply_agent_actions(actions)
        
        # Simulate market step
        market_changes = self._simulate_market_step()
        
        # Update portfolio state
        self._update_portfolio_state(portfolio_changes, market_changes)
        
        # Check for risk scenarios and emergencies
        risk_events = self._check_risk_events()
        
        # Generate new risk state
        risk_state = self._generate_risk_state()
        
        # Process risk state for all agents
        normalized_state, processing_metadata = self.state_processor.process_state(risk_state.to_vector())
        
        # Evaluate global risk with centralized critic
        global_risk_state = self._create_global_risk_state(risk_state, actions)
        global_risk_value, operating_mode = self.centralized_critic.evaluate_global_risk(global_risk_state)
        
        # Calculate rewards for each agent
        rewards = self._calculate_rewards(actions, risk_state, global_risk_value, risk_events)
        
        # Prepare observations for next step
        observations = {}
        for agent_name in self.agent_names:
            observations[agent_name] = normalized_state.astype(np.float32)
        
        # Check if episode is done
        done = self._is_episode_done(risk_events, operating_mode)
        
        # Compile info
        info = {
            'step': self.current_step,
            'portfolio_value': self.portfolio_state.total_value,
            'portfolio_drawdown': self.portfolio_state.drawdown,
            'global_risk_value': global_risk_value,
            'operating_mode': operating_mode.value,
            'risk_events': risk_events,
            'market_regime': self.market_conditions.regime.value,
            'processing_time_ms': (datetime.now() - step_start_time).total_seconds() * 1000,
            'var_estimate': self.portfolio_state.var_estimate,
            'leverage': self.portfolio_state.leverage,
            'correlation': self.portfolio_state.max_correlation
        }
        
        # Track episode rewards
        total_reward = sum(rewards.values())
        self.episode_rewards.append(total_reward)
        
        return observations, rewards, done, info
    
    def _validate_actions(self, actions: Dict[str, Union[int, np.ndarray]]) -> bool:
        """Validate agent actions are in correct format"""
        
        if set(actions.keys()) != set(self.agent_names):
            return False
        
        # Validate each agent's action format
        try:
            # Position sizing: discrete action 0-4
            if not (isinstance(actions['position_sizing'], (int, np.integer)) and 0 <= actions['position_sizing'] <= 4):
                return False
            
            # Stop/target: 2D continuous [0.5, 3.0]
            stop_target = actions['stop_target']
            if not (isinstance(stop_target, np.ndarray) and stop_target.shape == (2,) and
                    np.all(0.5 <= stop_target) and np.all(stop_target <= 3.0)):
                return False
            
            # Risk monitor: discrete action 0-3
            if not (isinstance(actions['risk_monitor'], (int, np.integer)) and 0 <= actions['risk_monitor'] <= 3):
                return False
            
            # Portfolio optimizer: 5D continuous [0.0, 1.0]
            portfolio_weights = actions['portfolio_optimizer']
            if not (isinstance(portfolio_weights, np.ndarray) and portfolio_weights.shape == (5,) and
                    np.all(0.0 <= portfolio_weights) and np.all(portfolio_weights <= 1.0)):
                return False
            
            return True
            
        except (TypeError, AttributeError, ValueError):
            return False
    
    def _apply_agent_actions(self, actions: Dict[str, Union[int, np.ndarray]]) -> Dict[str, Any]:
        """Apply agent actions to portfolio and return changes"""
        
        changes = {
            'position_adjustments': {},
            'stop_loss_updates': {},
            'target_updates': {},
            'risk_actions': [],
            'portfolio_rebalancing': {}
        }
        
        # π₁ Position Sizing Agent
        position_action = actions['position_sizing']
        if position_action == 0:  # reduce_large
            changes['position_adjustments'] = {symbol: -0.2 for symbol in self.market_symbols}
        elif position_action == 1:  # reduce_small
            changes['position_adjustments'] = {symbol: -0.1 for symbol in self.market_symbols}
        elif position_action == 2:  # hold
            changes['position_adjustments'] = {}
        elif position_action == 3:  # increase_small
            changes['position_adjustments'] = {symbol: 0.1 for symbol in self.market_symbols}
        elif position_action == 4:  # increase_large
            changes['position_adjustments'] = {symbol: 0.2 for symbol in self.market_symbols}
        
        # π₂ Stop/Target Agent
        stop_target = actions['stop_target']
        changes['stop_loss_updates'] = {'multiplier': float(stop_target[0])}
        changes['target_updates'] = {'multiplier': float(stop_target[1])}
        
        # π₃ Risk Monitor Agent
        risk_action = actions['risk_monitor']
        if risk_action == 0:  # no_action
            pass
        elif risk_action == 1:  # alert
            changes['risk_actions'].append('alert_generated')
        elif risk_action == 2:  # reduce_risk
            changes['risk_actions'].append('risk_reduction')
            # Apply 25% position reduction
            changes['position_adjustments'] = {
                symbol: changes['position_adjustments'].get(symbol, 0) - 0.25 
                for symbol in self.market_symbols
            }
        elif risk_action == 3:  # emergency_stop
            changes['risk_actions'].append('emergency_stop')
            self.emergency_stops += 1
            # Close all positions
            changes['position_adjustments'] = {symbol: -1.0 for symbol in self.market_symbols}
        
        # π₄ Portfolio Optimizer Agent
        portfolio_weights = actions['portfolio_optimizer']
        # Normalize weights to sum to 1
        weight_sum = np.sum(portfolio_weights)
        if weight_sum > 0:
            normalized_weights = portfolio_weights / weight_sum
            changes['portfolio_rebalancing'] = {
                'equity': float(normalized_weights[0]),
                'fixed_income': float(normalized_weights[1]),
                'commodities': float(normalized_weights[2]),
                'cash': float(normalized_weights[3]),
                'alternatives': float(normalized_weights[4])
            }
        
        return changes
    
    def _simulate_market_step(self) -> Dict[str, Any]:
        """Simulate one step of market dynamics"""
        
        # Update market regime with some probability
        if random.random() < 0.01:  # 1% chance per step
            self.market_conditions.regime = random.choice(list(MarketRegime))
        
        # Generate returns based on current regime
        base_volatility = 0.01  # 1% daily volatility baseline
        
        if self.market_conditions.regime == MarketRegime.HIGH_VOLATILITY:
            volatility_multiplier = 2.0
        elif self.market_conditions.regime == MarketRegime.CRISIS:
            volatility_multiplier = 3.0
        else:
            volatility_multiplier = 1.0
        
        # Generate correlated returns
        returns = {}
        base_return = np.random.normal(0, base_volatility * volatility_multiplier)
        
        for symbol in self.market_symbols:
            # Add correlation and idiosyncratic noise
            correlation_component = base_return * self.market_conditions.correlation_level
            idiosyncratic = np.random.normal(0, base_volatility * 0.5) * (1 - self.market_conditions.correlation_level)
            returns[symbol] = correlation_component + idiosyncratic
        
        # Update market conditions
        self.market_conditions.volatility_percentile = np.clip(
            self.market_conditions.volatility_percentile + np.random.normal(0, 0.05),
            0.0, 1.0
        )
        
        self.market_conditions.stress_level = np.clip(
            self.market_conditions.stress_level + np.random.normal(0, 0.02),
            0.0, 1.0
        )
        
        return {
            'returns': returns,
            'volatility_change': volatility_multiplier - 1.0,
            'correlation_update': self._update_correlation_matrix()
        }
    
    def _update_portfolio_state(self, portfolio_changes: Dict[str, Any], market_changes: Dict[str, Any]):
        """Update portfolio state based on actions and market movements"""
        
        # Apply position adjustments
        for symbol, adjustment in portfolio_changes.get('position_adjustments', {}).items():
            if symbol in self.portfolio_state.positions:
                new_position = self.portfolio_state.positions[symbol] * (1 + adjustment)
                self.portfolio_state.positions[symbol] = max(0.0, new_position)  # No short selling
        
        # Apply market returns to positions
        total_pnl = 0.0
        for symbol, position in self.portfolio_state.positions.items():
            if position > 0 and symbol in market_changes['returns']:
                pnl = position * market_changes['returns'][symbol]
                total_pnl += pnl
        
        # Update portfolio metrics
        self.portfolio_state.unrealized_pnl = total_pnl
        self.portfolio_state.total_value = self.portfolio_state.cash + sum(self.portfolio_state.positions.values()) + total_pnl
        
        # Update drawdown
        peak_value = max(self.initial_capital, self.portfolio_state.total_value)
        current_drawdown = (peak_value - self.portfolio_state.total_value) / peak_value
        self.portfolio_state.drawdown = max(self.portfolio_state.drawdown, current_drawdown)
        
        # Update leverage
        total_position_value = sum(self.portfolio_state.positions.values())
        self.portfolio_state.leverage = total_position_value / max(self.portfolio_state.total_value, 1.0)
        
        # Update VaR estimate (simplified)
        self.portfolio_state.var_estimate = min(0.05 * self.portfolio_state.total_value, 
                                               self.portfolio_state.leverage * 0.02 * self.portfolio_state.total_value)
        
        # Update max correlation
        self.portfolio_state.max_correlation = np.max(self.correlation_matrix) if self.correlation_matrix.size > 0 else 0.0
    
    def _check_risk_events(self) -> List[str]:
        """Check for risk events and violations"""
        events = []
        
        # Check drawdown limits
        if self.portfolio_state.drawdown > 0.15:  # 15% drawdown threshold
            events.append('excessive_drawdown')
        
        # Check leverage limits
        if self.portfolio_state.leverage > 3.0:  # 3x leverage limit
            events.append('excessive_leverage')
        
        # Check VaR limits
        var_limit = 0.05 * self.portfolio_state.total_value  # 5% of portfolio
        if self.portfolio_state.var_estimate > var_limit:
            events.append('var_breach')
        
        # Check correlation risk
        if self.portfolio_state.max_correlation > 0.9:
            events.append('correlation_spike')
        
        # Check market stress conditions
        if self.market_conditions.stress_level > 0.8:
            events.append('market_stress')
        
        return events
    
    def _generate_risk_state(self) -> RiskState:
        """Generate current risk state vector"""
        
        return RiskState(
            account_equity_normalized=self.portfolio_state.total_value / self.initial_capital,
            open_positions_count=sum(1 for pos in self.portfolio_state.positions.values() if pos > 0),
            volatility_regime=self.market_conditions.volatility_percentile,
            correlation_risk=self.portfolio_state.max_correlation,
            var_estimate_5pct=self.portfolio_state.var_estimate / self.portfolio_state.total_value,
            current_drawdown_pct=self.portfolio_state.drawdown,
            margin_usage_pct=min(1.0, self.portfolio_state.leverage / 4.0),  # Normalize to [0,1]
            time_of_day_risk=self.market_conditions.time_factor,
            market_stress_level=self.market_conditions.stress_level,
            liquidity_conditions=self.market_conditions.liquidity_score
        )
    
    def _create_global_risk_state(self, risk_state: RiskState, actions: Dict[str, Union[int, np.ndarray]]) -> GlobalRiskState:
        """Create global risk state for centralized critic"""
        
        # For simplicity, use the same risk vector for all agents
        # In practice, each agent would have specialized risk perspectives
        base_vector = risk_state.to_vector()
        
        return GlobalRiskState(
            position_sizing_risk=base_vector,
            stop_target_risk=base_vector,
            risk_monitor_risk=base_vector,
            portfolio_optimizer_risk=base_vector,
            total_portfolio_var=self.portfolio_state.var_estimate / self.portfolio_state.total_value,
            portfolio_correlation_max=self.portfolio_state.max_correlation,
            aggregate_leverage=self.portfolio_state.leverage,
            liquidity_risk_score=1.0 - self.market_conditions.liquidity_score,
            systemic_risk_level=self.market_conditions.stress_level,
            timestamp=datetime.now(),
            market_hours_factor=self.market_conditions.time_factor
        )
    
    def _calculate_rewards(self, 
                          actions: Dict[str, Union[int, np.ndarray]], 
                          risk_state: RiskState,
                          global_risk_value: float,
                          risk_events: List[str]) -> Dict[str, float]:
        """Calculate rewards for each agent"""
        
        base_reward = 0.0
        
        # Portfolio performance component
        return_reward = (self.portfolio_state.total_value - self.initial_capital) / self.initial_capital * 10
        
        # Risk management component
        risk_penalty = 0.0
        for event in risk_events:
            if event == 'excessive_drawdown':
                risk_penalty -= 5.0
            elif event == 'excessive_leverage':
                risk_penalty -= 3.0
            elif event == 'var_breach':
                risk_penalty -= 2.0
            elif event == 'correlation_spike':
                risk_penalty -= 1.0
        
        # Global risk value component
        global_risk_bonus = global_risk_value * 2.0  # Positive for good risk management
        
        # Agent-specific rewards
        rewards = {}
        
        # π₁ Position Sizing Agent
        position_action = actions['position_sizing']
        if 'excessive_leverage' in risk_events and position_action in [0, 1]:  # Reducing positions
            position_reward = 1.0
        elif 'excessive_leverage' not in risk_events and position_action in [3, 4]:  # Increasing positions
            position_reward = 0.5
        else:
            position_reward = 0.0
        
        rewards['position_sizing'] = base_reward + return_reward + risk_penalty + global_risk_bonus + position_reward
        
        # π₂ Stop/Target Agent
        stop_target = actions['stop_target']
        # Reward tighter stops in high volatility
        if self.market_conditions.volatility_percentile > 0.7 and stop_target[0] < 1.5:
            stop_reward = 0.5
        else:
            stop_reward = 0.0
        
        rewards['stop_target'] = base_reward + return_reward + risk_penalty + global_risk_bonus + stop_reward
        
        # π₃ Risk Monitor Agent
        risk_action = actions['risk_monitor']
        if risk_events and risk_action > 0:  # Taking action when risks present
            monitor_reward = 1.0
        elif not risk_events and risk_action == 0:  # No action when no risks
            monitor_reward = 0.2
        else:
            monitor_reward = -0.5  # Inappropriate action
        
        rewards['risk_monitor'] = base_reward + return_reward + risk_penalty + global_risk_bonus + monitor_reward
        
        # π₄ Portfolio Optimizer Agent
        portfolio_weights = actions['portfolio_optimizer']
        # Reward diversification
        weight_entropy = -np.sum(portfolio_weights * np.log(portfolio_weights + 1e-8))
        diversification_reward = weight_entropy * 0.2
        
        rewards['portfolio_optimizer'] = base_reward + return_reward + risk_penalty + global_risk_bonus + diversification_reward
        
        return rewards
    
    def _is_episode_done(self, risk_events: List[str], operating_mode) -> bool:
        """Determine if episode should end"""
        
        # End on emergency stop
        if 'emergency_stop' in [action for actions in self.agent_actions_history['risk_monitor'] for action in [actions] if actions == 3]:
            return True
        
        # End on excessive losses
        if self.portfolio_state.total_value < 0.5 * self.initial_capital:  # 50% loss
            return True
        
        # End on max steps
        if self.current_step >= self.max_steps:
            return True
        
        return False
    
    def _get_safe_step_result(self) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        """Return safe step result for error cases"""
        safe_obs = {name: np.zeros(10, dtype=np.float32) for name in self.agent_names}
        safe_rewards = {name: -1.0 for name in self.agent_names}
        return safe_obs, safe_rewards, True, {'error': 'invalid_actions'}
    
    def _initialize_correlation_matrix(self) -> np.ndarray:
        """Initialize correlation matrix for market simulation"""
        n = len(self.market_symbols)
        correlation_matrix = np.eye(n)
        
        # Add some baseline correlations
        for i in range(n):
            for j in range(i+1, n):
                correlation = 0.3 + 0.4 * random.random()  # 0.3 to 0.7
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _update_correlation_matrix(self) -> Dict[str, float]:
        """Update correlation matrix and return changes"""
        # Simple correlation evolution
        changes = {}
        n = len(self.market_symbols)
        
        for i in range(n):
            for j in range(i+1, n):
                change = np.random.normal(0, 0.01)  # Small random changes
                self.correlation_matrix[i, j] = np.clip(
                    self.correlation_matrix[i, j] + change, 0.0, 1.0
                )
                self.correlation_matrix[j, i] = self.correlation_matrix[i, j]
                changes[f'{i}_{j}'] = change
        
        return changes
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n=== Risk Environment Step {self.current_step} ===")
            print(f"Portfolio Value: ${self.portfolio_state.total_value:,.2f}")
            print(f"Drawdown: {self.portfolio_state.drawdown:.2%}")
            print(f"Leverage: {self.portfolio_state.leverage:.2f}x")
            print(f"VaR: ${self.portfolio_state.var_estimate:,.2f}")
            print(f"Max Correlation: {self.portfolio_state.max_correlation:.3f}")
            print(f"Market Regime: {self.market_conditions.regime.value}")
            print(f"Stress Level: {self.market_conditions.stress_level:.2f}")
    
    def close(self):
        """Clean up environment resources"""
        if hasattr(self.state_processor, 'shutdown'):
            self.state_processor.shutdown()
        logger.info("Risk environment closed")