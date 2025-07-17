"""
PettingZoo Environment Wrapper for Execution MARL System
========================================================

This module provides a PettingZoo-compatible environment for training the 5-agent
execution MARL system with proper multi-agent coordination and realistic market dynamics.

Agents:
- π₁: Position Sizing Agent (position_sizing)
- π₂: Stop/Target Agent (stop_target)  
- π₃: Risk Monitor Agent (risk_monitor)
- π₄: Portfolio Optimizer Agent (portfolio_optimizer)
- π₅: Routing Agent (routing)

Environment Features:
- AEC (Agent-Environment-Cycle) API compliance
- Realistic market microstructure simulation
- Performance-based reward structure
- Multi-agent coordination incentives
- Risk management constraints
- Execution quality metrics (fill rate, slippage, latency)

Author: Claude Code
Date: 2025-07-17
"""

import asyncio
import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import structlog
from functools import lru_cache

# PettingZoo imports
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import gymnasium as gym

# Import existing execution system components
from src.execution.unified_execution_marl_system import (
    UnifiedExecutionMARLSystem, ExecutionDecision, ExecutionPerformanceMetrics
)
from src.execution.agents.centralized_critic import (
    ExecutionContext, MarketFeatures, CombinedState
)
from src.execution.agents.routing_agent import BrokerType, RoutingState
from src.risk.agents.base_risk_agent import RiskState
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


@dataclass
class MarketState:
    """Current market state for simulation"""
    timestamp: datetime = field(default_factory=datetime.now)
    price: float = 100.0
    volume: float = 1000.0
    bid: float = 99.95
    ask: float = 100.05
    spread_bps: float = 5.0
    volatility: float = 0.15
    trend: float = 0.0
    liquidity_depth: float = 0.8
    market_impact: float = 0.0
    
    # Regime indicators
    regime_bull: float = 0.0
    regime_bear: float = 0.0
    regime_neutral: float = 1.0
    regime_volatility: float = 0.0
    
    # Order flow
    buy_pressure: float = 0.5
    sell_pressure: float = 0.5
    institutional_flow: float = 0.0
    retail_flow: float = 0.0


@dataclass
class ExecutionEnvironmentConfig:
    """Configuration for execution environment"""
    
    # Environment parameters
    max_steps: int = 1000
    initial_portfolio_value: float = 100000.0
    max_position_size: float = 0.2  # 20% of portfolio
    transaction_cost_bps: float = 2.0  # 2 basis points
    min_trade_size: float = 100.0  # Minimum trade size
    
    # Market simulation parameters
    price_volatility: float = 0.15
    market_impact_factor: float = 0.001
    liquidity_factor: float = 0.8
    
    # Reward parameters
    pnl_reward_weight: float = 1.0
    risk_penalty_weight: float = 0.5
    coordination_bonus_weight: float = 0.2
    execution_quality_weight: float = 0.3
    
    # Performance targets
    target_fill_rate: float = 0.998
    target_slippage_bps: float = 2.0
    target_latency_us: float = 500.0
    
    # Risk limits
    max_var_threshold: float = 0.05
    max_drawdown_threshold: float = 0.1
    correlation_threshold: float = 0.8


class ExecutionEnvironment(AECEnv):
    """
    PettingZoo AEC Environment for Execution MARL System
    
    This environment simulates a realistic execution trading scenario where
    5 agents coordinate to make optimal execution decisions.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'execution_marl_v0',
        'is_parallelizable': False,
        'render_fps': 1
    }
    
    def __init__(self, config: Optional[ExecutionEnvironmentConfig] = None):
        """
        Initialize execution environment
        
        Args:
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config or ExecutionEnvironmentConfig()
        
        # Agent names (5 agents)
        self.possible_agents = [
            'position_sizing',
            'stop_target', 
            'risk_monitor',
            'portfolio_optimizer',
            'routing'
        ]
        
        # Initialize agent selector
        self.agent_selector = agent_selector(self.possible_agents)
        
        # Action and observation spaces
        self._setup_spaces()
        
        # Initialize execution system
        self.execution_system = self._create_execution_system()
        
        # Environment state
        self.market_state = MarketState()
        self.portfolio_state = self._initialize_portfolio_state()
        self.execution_history = deque(maxlen=1000)
        
        # Episode tracking
        self.current_step = 0
        self.episode_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.episode_start_time = datetime.now()
        
        # Performance metrics
        self.performance_metrics = ExecutionPerformanceMetrics()
        
        # Agent states
        self.agent_actions = {}
        self.agent_observations = {}
        self.agent_rewards = {}
        self.agent_dones = {}
        self.agent_infos = {}
        
        logger.info("ExecutionEnvironment initialized", 
                   agents=len(self.possible_agents),
                   max_steps=self.config.max_steps)
    
    def _setup_spaces(self):
        """Setup action and observation spaces for all agents"""
        
        # Position Sizing Agent
        self.action_spaces = {
            'position_sizing': spaces.Box(
                low=np.array([0.0, 0.0, 0.0]),  # [size_fraction, kelly_fraction, confidence]
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32
            ),
            'stop_target': spaces.Box(
                low=np.array([0.0, 0.0, 0.0]),  # [stop_multiplier, target_multiplier, confidence]
                high=np.array([5.0, 10.0, 1.0]),
                dtype=np.float32
            ),
            'risk_monitor': spaces.Discrete(4),  # [NO_ACTION, ALERT, REDUCE_POSITION, EMERGENCY_STOP]
            'portfolio_optimizer': spaces.Box(
                low=np.array([0.0, 0.0, 0.0]),  # [allocation_adjustment, rebalance_signal, confidence]
                high=np.array([2.0, 1.0, 1.0]),
                dtype=np.float32
            ),
            'routing': spaces.Discrete(4)  # [IB, ALPACA, TDA, SCHWAB]
        }
        
        # Common observation components
        market_obs_dim = 16  # Market state features
        portfolio_obs_dim = 10  # Portfolio state features
        execution_obs_dim = 8  # Execution context features
        
        # Observation spaces (customized per agent)
        self.observation_spaces = {
            'position_sizing': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(market_obs_dim + portfolio_obs_dim + execution_obs_dim + 5,),  # +5 for position sizing specific
                dtype=np.float32
            ),
            'stop_target': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(market_obs_dim + portfolio_obs_dim + execution_obs_dim + 4,),  # +4 for stop/target specific
                dtype=np.float32
            ),
            'risk_monitor': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(market_obs_dim + portfolio_obs_dim + execution_obs_dim + 6,),  # +6 for risk monitoring specific
                dtype=np.float32
            ),
            'portfolio_optimizer': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(market_obs_dim + portfolio_obs_dim + execution_obs_dim + 5,),  # +5 for portfolio optimization specific
                dtype=np.float32
            ),
            'routing': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(market_obs_dim + portfolio_obs_dim + execution_obs_dim + 12,),  # +12 for routing specific (broker metrics)
                dtype=np.float32
            )
        }
    
    def _create_execution_system(self) -> UnifiedExecutionMARLSystem:
        """Create unified execution MARL system"""
        
        execution_config = {
            'max_workers': 5,
            'position_sizing': {
                'kelly_lookback_periods': 252,
                'max_position_size': self.config.max_position_size,
                'min_position_size': 0.01,
                'risk_free_rate': 0.02
            },
            'stop_target': {
                'atr_period': 14,
                'default_stop_multiplier': 2.0,
                'default_target_multiplier': 3.0,
                'max_stop_loss': 0.05
            },
            'risk_monitor': {
                'var_threshold': self.config.max_var_threshold,
                'correlation_threshold': self.config.correlation_threshold,
                'drawdown_threshold': self.config.max_drawdown_threshold,
                'emergency_stop_threshold': 0.15
            },
            'portfolio_optimizer': {
                'rebalance_threshold': 0.05,
                'target_volatility': 0.12,
                'max_correlation': self.config.correlation_threshold,
                'min_liquidity': 0.1
            },
            'routing': {
                'broker_ids': ['IB', 'ALPACA', 'TDA', 'SCHWAB'],
                'learning_rate': 2e-4,
                'training_mode': True,
                'exploration_epsilon': 0.1,
                'target_qoe_score': 0.85,
                'max_routing_latency_us': 100.0
            }
        }
        
        return UnifiedExecutionMARLSystem(execution_config)
    
    def _initialize_portfolio_state(self) -> Dict[str, Any]:
        """Initialize portfolio state"""
        return {
            'portfolio_value': self.config.initial_portfolio_value,
            'available_capital': self.config.initial_portfolio_value * 0.5,
            'current_position': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'var_estimate': 0.02,
            'expected_return': 0.001,
            'volatility': self.config.price_volatility,
            'sharpe_ratio': 1.0,
            'max_drawdown': 0.0,
            'drawdown_current': 0.0,
            'time_since_last_trade': 0,
            'risk_budget_used': 0.0,
            'correlation_risk': 0.2,
            'liquidity_score': 0.9,
            'stop_loss_level': 0.0,
            'take_profit_level': 0.0,
            'position_entry_price': 0.0,
            'position_entry_time': datetime.now()
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.current_step = 0
        self.episode_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.episode_start_time = datetime.now()
        
        # Reset market state
        self.market_state = MarketState()
        
        # Reset portfolio state
        self.portfolio_state = self._initialize_portfolio_state()
        
        # Reset execution history
        self.execution_history.clear()
        
        # Reset performance metrics
        self.performance_metrics = ExecutionPerformanceMetrics()
        
        # Reset agent states
        self.agent_actions = {}
        self.agent_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.agent_dones = {agent: False for agent in self.possible_agents}
        self.agent_infos = {agent: {} for agent in self.possible_agents}
        
        # Reset agent selector
        self.agent_selector.reset()
        
        # Generate initial observations
        self._generate_observations()
        
        # Set agents and agent selector
        self.agents = self.possible_agents[:]
        self.agent_selection = self.agent_selector.next()
        
        # Set terminations and truncations
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        
        logger.info("Environment reset", 
                   step=self.current_step,
                   agents=len(self.agents))
    
    def step(self, action: Union[int, np.ndarray]):
        """
        Execute one step of the environment
        
        Args:
            action: Action taken by current agent
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # Agent is done, return empty step
            return self._was_dead_step(action)
        
        # Store action for current agent
        self.agent_actions[self.agent_selection] = action
        
        # Check if all agents have acted
        if len(self.agent_actions) == len(self.possible_agents):
            # All agents have acted, execute unified decision
            self._execute_unified_step()
            
            # Clear actions for next step
            self.agent_actions = {}
            
            # Increment step counter
            self.current_step += 1
            
            # Check for episode termination
            self._check_termination()
        
        # Select next agent
        self.agent_selection = self.agent_selector.next()
        
        # Update cumulative rewards
        self._accumulate_rewards()
    
    def _execute_unified_step(self):
        """Execute unified decision across all agents"""
        try:
            # Create execution context from current state
            execution_context = self._create_execution_context()
            
            # Create market features
            market_features = self._create_market_features()
            
            # Create order data for routing agent
            order_data = self._create_order_data()
            
            # Execute unified decision (async call made sync for PettingZoo)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                decision = loop.run_until_complete(
                    self.execution_system.execute_unified_decision(
                        execution_context, market_features, order_data
                    )
                )
            finally:
                loop.close()
            
            # Update environment state based on decision
            self._update_environment_state(decision)
            
            # Calculate rewards for all agents
            self._calculate_rewards(decision)
            
            # Update observations
            self._generate_observations()
            
            # Store execution in history
            self.execution_history.append({
                'step': self.current_step,
                'decision': decision,
                'market_state': self.market_state,
                'portfolio_state': self.portfolio_state.copy(),
                'timestamp': datetime.now()
            })
            
            logger.debug("Unified step executed",
                        step=self.current_step,
                        final_position=decision.final_position_size,
                        risk_approved=decision.risk_approved,
                        latency_us=decision.total_latency_us)
            
        except Exception as e:
            logger.error("Error in unified step execution", error=str(e))
            # Set emergency rewards
            self.agent_rewards = {agent: -10.0 for agent in self.possible_agents}
    
    def _create_execution_context(self) -> ExecutionContext:
        """Create execution context from current state"""
        return ExecutionContext(
            portfolio_value=self.portfolio_state['portfolio_value'],
            available_capital=self.portfolio_state['available_capital'],
            current_position=self.portfolio_state['current_position'],
            unrealized_pnl=self.portfolio_state['unrealized_pnl'],
            realized_pnl=self.portfolio_state['realized_pnl'],
            var_estimate=self.portfolio_state['var_estimate'],
            expected_return=self.portfolio_state['expected_return'],
            volatility=self.portfolio_state['volatility'],
            sharpe_ratio=self.portfolio_state['sharpe_ratio'],
            max_drawdown=self.portfolio_state['max_drawdown'],
            drawdown_current=self.portfolio_state['drawdown_current'],
            time_since_last_trade=self.portfolio_state['time_since_last_trade'],
            risk_budget_used=self.portfolio_state['risk_budget_used'],
            correlation_risk=self.portfolio_state['correlation_risk'],
            liquidity_score=self.portfolio_state['liquidity_score']
        )
    
    def _create_market_features(self) -> MarketFeatures:
        """Create market features from current market state"""
        return MarketFeatures(
            # Order flow features
            buy_volume=self.market_state.buy_pressure * self.market_state.volume,
            sell_volume=self.market_state.sell_pressure * self.market_state.volume,
            order_flow_imbalance=self.market_state.buy_pressure - self.market_state.sell_pressure,
            large_order_flow=self.market_state.institutional_flow,
            retail_flow=self.market_state.retail_flow,
            institutional_flow=self.market_state.institutional_flow,
            flow_toxicity=abs(self.market_state.buy_pressure - self.market_state.sell_pressure),
            flow_persistence=self.market_state.trend,
            
            # Price action features
            price_momentum_1m=self.market_state.trend,
            price_momentum_5m=self.market_state.trend * 0.8,
            support_level=self.market_state.price * 0.98,
            resistance_level=self.market_state.price * 1.02,
            trend_strength=abs(self.market_state.trend),
            mean_reversion_signal=-self.market_state.trend * 0.5,
            breakout_probability=max(0, self.market_state.trend * 2),
            reversal_probability=max(0, -self.market_state.trend * 2),
            
            # Volatility features
            atm_vol=self.market_state.volatility,
            vol_skew=self.market_state.volatility * 0.1,
            vol_term_structure=self.market_state.volatility * 1.1,
            vol_smile_curvature=0.02,
            realized_garch=self.market_state.volatility,
            vol_risk_premium=self.market_state.volatility * 0.2,
            vol_persistence=0.8,
            vol_clustering=self.market_state.volatility ** 2,
            
            # Cross-asset features
            correlation_spy=0.7,
            correlation_vix=-0.5,
            correlation_bonds=0.2,
            correlation_dollar=0.3,
            regime_equity=self.market_state.regime_bull,
            regime_volatility=self.market_state.regime_volatility,
            regime_interest_rate=0.0,
            regime_risk_off=self.market_state.regime_bear
        )
    
    def _create_order_data(self) -> Dict[str, Any]:
        """Create order data for routing agent"""
        return {
            'symbol': 'SPY',
            'side': 'BUY' if self.portfolio_state['current_position'] >= 0 else 'SELL',
            'quantity': abs(self.portfolio_state['current_position']) or 100,
            'order_type': 'MARKET',
            'time_in_force': 'IOC',
            'urgency': 'NORMAL',
            'execution_style': 'AGGRESSIVE'
        }
    
    def _update_environment_state(self, decision: ExecutionDecision):
        """Update environment state based on execution decision"""
        
        # Update position based on decision
        if decision.risk_approved and not decision.emergency_stop:
            position_change = decision.final_position_size - self.portfolio_state['current_position']
            
            if abs(position_change) > 0.001:  # Minimum position change threshold
                # Execute the position change
                execution_price = self.market_state.price
                
                # Apply slippage
                if position_change > 0:  # Long position
                    execution_price *= (1 + decision.estimated_slippage_bps / 10000)
                else:  # Short position
                    execution_price *= (1 - decision.estimated_slippage_bps / 10000)
                
                # Update portfolio
                trade_value = abs(position_change) * execution_price
                transaction_cost = trade_value * (self.config.transaction_cost_bps / 10000)
                
                self.portfolio_state['current_position'] = decision.final_position_size
                self.portfolio_state['realized_pnl'] -= transaction_cost
                self.portfolio_state['position_entry_price'] = execution_price
                self.portfolio_state['position_entry_time'] = datetime.now()
                self.portfolio_state['time_since_last_trade'] = 0
                
                # Update stop/target levels
                self.portfolio_state['stop_loss_level'] = decision.stop_loss_level
                self.portfolio_state['take_profit_level'] = decision.take_profit_level
        
        # Update market state (simulate market dynamics)
        self._simulate_market_dynamics()
        
        # Update portfolio value and PnL
        self._update_portfolio_pnl()
        
        # Update risk metrics
        self._update_risk_metrics()
    
    def _simulate_market_dynamics(self):
        """Simulate realistic market dynamics"""
        dt = 1.0 / (252 * 24 * 60)  # 1 minute in years
        
        # Random walk with drift
        drift = np.random.normal(0, 0.0001)
        noise = np.random.normal(0, self.market_state.volatility * np.sqrt(dt))
        
        # Market impact from current position
        position_impact = self.portfolio_state['current_position'] * self.config.market_impact_factor
        
        # Price update
        price_change = drift + noise - position_impact
        self.market_state.price *= (1 + price_change)
        
        # Update spread based on volatility
        self.market_state.spread_bps = 2.0 + self.market_state.volatility * 10
        self.market_state.bid = self.market_state.price - (self.market_state.spread_bps / 2) / 10000 * self.market_state.price
        self.market_state.ask = self.market_state.price + (self.market_state.spread_bps / 2) / 10000 * self.market_state.price
        
        # Update volume and liquidity
        self.market_state.volume = max(500, np.random.normal(1000, 200))
        self.market_state.liquidity_depth = max(0.1, min(1.0, np.random.normal(0.8, 0.1)))
        
        # Update trend and momentum
        self.market_state.trend = self.market_state.trend * 0.9 + price_change * 0.1
        
        # Update buy/sell pressure
        self.market_state.buy_pressure = max(0, min(1, np.random.normal(0.5, 0.1)))
        self.market_state.sell_pressure = 1 - self.market_state.buy_pressure
    
    def _update_portfolio_pnl(self):
        """Update portfolio PnL based on current position and market price"""
        if self.portfolio_state['current_position'] != 0:
            entry_price = self.portfolio_state['position_entry_price']
            current_price = self.market_state.price
            position_size = self.portfolio_state['current_position']
            
            # Calculate unrealized PnL
            if position_size > 0:  # Long position
                self.portfolio_state['unrealized_pnl'] = (current_price - entry_price) * position_size
            else:  # Short position
                self.portfolio_state['unrealized_pnl'] = (entry_price - current_price) * abs(position_size)
            
            # Update portfolio value
            self.portfolio_state['portfolio_value'] = (
                self.config.initial_portfolio_value + 
                self.portfolio_state['realized_pnl'] + 
                self.portfolio_state['unrealized_pnl']
            )
            
            # Update drawdown
            peak_value = max(self.portfolio_state['portfolio_value'], 
                           getattr(self, '_peak_portfolio_value', self.config.initial_portfolio_value))
            self._peak_portfolio_value = peak_value
            
            current_drawdown = (peak_value - self.portfolio_state['portfolio_value']) / peak_value
            self.portfolio_state['drawdown_current'] = current_drawdown
            self.portfolio_state['max_drawdown'] = max(self.portfolio_state['max_drawdown'], current_drawdown)
    
    def _update_risk_metrics(self):
        """Update risk metrics"""
        # Simple VaR estimate based on current position and volatility
        position_value = abs(self.portfolio_state['current_position']) * self.market_state.price
        portfolio_value = self.portfolio_state['portfolio_value']
        
        if portfolio_value > 0:
            position_weight = position_value / portfolio_value
            var_estimate = position_weight * self.market_state.volatility * 1.65  # 95% VaR
            self.portfolio_state['var_estimate'] = var_estimate
            
            # Risk budget utilization
            self.portfolio_state['risk_budget_used'] = var_estimate / self.config.max_var_threshold
        
        # Update time since last trade
        self.portfolio_state['time_since_last_trade'] += 1
    
    def _calculate_rewards(self, decision: ExecutionDecision):
        """Calculate rewards for all agents based on execution decision"""
        
        # Base reward from PnL change
        total_pnl = self.portfolio_state['realized_pnl'] + self.portfolio_state['unrealized_pnl']
        pnl_change = total_pnl - getattr(self, '_last_total_pnl', 0.0)
        self._last_total_pnl = total_pnl
        
        # Normalize PnL reward
        pnl_reward = pnl_change / self.config.initial_portfolio_value * 100  # Scale to reasonable range
        
        # Risk penalty
        risk_penalty = 0.0
        if self.portfolio_state['var_estimate'] > self.config.max_var_threshold:
            risk_penalty = -1.0 * (self.portfolio_state['var_estimate'] / self.config.max_var_threshold - 1.0)
        
        if self.portfolio_state['drawdown_current'] > self.config.max_drawdown_threshold:
            risk_penalty -= 2.0 * (self.portfolio_state['drawdown_current'] / self.config.max_drawdown_threshold - 1.0)
        
        # Execution quality rewards
        execution_quality_reward = 0.0
        if decision.fill_rate >= self.config.target_fill_rate:
            execution_quality_reward += 0.1
        if decision.estimated_slippage_bps <= self.config.target_slippage_bps:
            execution_quality_reward += 0.1
        if decision.total_latency_us <= self.config.target_latency_us:
            execution_quality_reward += 0.1
        
        # Coordination bonus (if agents agree on direction)
        coordination_bonus = 0.0
        if decision.risk_approved and not decision.emergency_stop:
            coordination_bonus = 0.1
        
        # Agent-specific rewards
        base_reward = (
            pnl_reward * self.config.pnl_reward_weight +
            risk_penalty * self.config.risk_penalty_weight +
            execution_quality_reward * self.config.execution_quality_weight +
            coordination_bonus * self.config.coordination_bonus_weight
        )
        
        # Assign rewards to agents
        self.agent_rewards = {
            'position_sizing': base_reward + (0.1 if decision.position_sizing else 0.0),
            'stop_target': base_reward + (0.1 if decision.stop_target else 0.0),
            'risk_monitor': base_reward + (0.2 if decision.risk_approved else -0.5),
            'portfolio_optimizer': base_reward + (0.1 if decision.portfolio_optimizer else 0.0),
            'routing': base_reward + (0.1 if decision.routing else 0.0)
        }
        
        # Emergency stop penalty
        if decision.emergency_stop:
            self.agent_rewards['risk_monitor'] = 1.0  # Reward for emergency stop
            for agent in self.agent_rewards:
                if agent != 'risk_monitor':
                    self.agent_rewards[agent] = -1.0
    
    def _generate_observations(self):
        """Generate observations for all agents"""
        
        # Common observation components
        market_obs = self._get_market_observation()
        portfolio_obs = self._get_portfolio_observation()
        execution_obs = self._get_execution_observation()
        
        # Agent-specific observations
        self.agent_observations = {
            'position_sizing': np.concatenate([
                market_obs, portfolio_obs, execution_obs,
                self._get_position_sizing_observation()
            ]),
            'stop_target': np.concatenate([
                market_obs, portfolio_obs, execution_obs,
                self._get_stop_target_observation()
            ]),
            'risk_monitor': np.concatenate([
                market_obs, portfolio_obs, execution_obs,
                self._get_risk_monitor_observation()
            ]),
            'portfolio_optimizer': np.concatenate([
                market_obs, portfolio_obs, execution_obs,
                self._get_portfolio_optimizer_observation()
            ]),
            'routing': np.concatenate([
                market_obs, portfolio_obs, execution_obs,
                self._get_routing_observation()
            ])
        }
    
    def _get_market_observation(self) -> np.ndarray:
        """Get market observation vector"""
        return np.array([
            self.market_state.price / 100.0,  # Normalized price
            self.market_state.bid / 100.0,
            self.market_state.ask / 100.0,
            self.market_state.spread_bps / 10.0,  # Normalized spread
            self.market_state.volume / 1000.0,  # Normalized volume
            self.market_state.volatility,
            self.market_state.trend,
            self.market_state.liquidity_depth,
            self.market_state.market_impact,
            self.market_state.buy_pressure,
            self.market_state.sell_pressure,
            self.market_state.regime_bull,
            self.market_state.regime_bear,
            self.market_state.regime_neutral,
            self.market_state.regime_volatility,
            self.market_state.institutional_flow
        ], dtype=np.float32)
    
    def _get_portfolio_observation(self) -> np.ndarray:
        """Get portfolio observation vector"""
        return np.array([
            self.portfolio_state['portfolio_value'] / self.config.initial_portfolio_value,
            self.portfolio_state['current_position'],
            self.portfolio_state['unrealized_pnl'] / self.config.initial_portfolio_value,
            self.portfolio_state['realized_pnl'] / self.config.initial_portfolio_value,
            self.portfolio_state['var_estimate'],
            self.portfolio_state['drawdown_current'],
            self.portfolio_state['max_drawdown'],
            self.portfolio_state['risk_budget_used'],
            self.portfolio_state['correlation_risk'],
            self.portfolio_state['liquidity_score']
        ], dtype=np.float32)
    
    def _get_execution_observation(self) -> np.ndarray:
        """Get execution observation vector"""
        return np.array([
            self.portfolio_state['time_since_last_trade'] / 100.0,
            self.portfolio_state['volatility'],
            self.portfolio_state['expected_return'],
            self.portfolio_state['sharpe_ratio'],
            self.portfolio_state['stop_loss_level'],
            self.portfolio_state['take_profit_level'],
            float(self.current_step) / self.config.max_steps,
            len(self.execution_history) / 1000.0
        ], dtype=np.float32)
    
    def _get_position_sizing_observation(self) -> np.ndarray:
        """Get position sizing specific observation"""
        return np.array([
            self.portfolio_state['available_capital'] / self.config.initial_portfolio_value,
            abs(self.portfolio_state['current_position']) / self.config.max_position_size,
            self.portfolio_state['var_estimate'] / self.config.max_var_threshold,
            self.market_state.volatility,
            self.market_state.trend
        ], dtype=np.float32)
    
    def _get_stop_target_observation(self) -> np.ndarray:
        """Get stop/target specific observation"""
        return np.array([
            self.portfolio_state['stop_loss_level'],
            self.portfolio_state['take_profit_level'],
            self.portfolio_state['unrealized_pnl'] / max(abs(self.portfolio_state['current_position']) * self.market_state.price, 1.0),
            self.market_state.volatility
        ], dtype=np.float32)
    
    def _get_risk_monitor_observation(self) -> np.ndarray:
        """Get risk monitor specific observation"""
        return np.array([
            self.portfolio_state['var_estimate'] / self.config.max_var_threshold,
            self.portfolio_state['drawdown_current'] / self.config.max_drawdown_threshold,
            self.portfolio_state['risk_budget_used'],
            self.portfolio_state['correlation_risk'],
            abs(self.portfolio_state['current_position']) / self.config.max_position_size,
            self.market_state.volatility
        ], dtype=np.float32)
    
    def _get_portfolio_optimizer_observation(self) -> np.ndarray:
        """Get portfolio optimizer specific observation"""
        return np.array([
            self.portfolio_state['portfolio_value'] / self.config.initial_portfolio_value,
            self.portfolio_state['current_position'],
            self.portfolio_state['correlation_risk'],
            self.portfolio_state['liquidity_score'],
            self.portfolio_state['sharpe_ratio']
        ], dtype=np.float32)
    
    def _get_routing_observation(self) -> np.ndarray:
        """Get routing specific observation"""
        # Mock broker performance metrics
        return np.array([
            50.0,  # avg_latency_ms (IB)
            0.998,  # fill_rate (IB)
            1.5,   # slippage_bps (IB)
            60.0,  # avg_latency_ms (ALPACA)
            0.995,  # fill_rate (ALPACA)
            2.0,   # slippage_bps (ALPACA)
            70.0,  # avg_latency_ms (TDA)
            0.993,  # fill_rate (TDA)
            2.5,   # slippage_bps (TDA)
            80.0,  # avg_latency_ms (SCHWAB)
            0.990,  # fill_rate (SCHWAB)
            3.0    # slippage_bps (SCHWAB)
        ], dtype=np.float32)
    
    def _check_termination(self):
        """Check if episode should terminate"""
        
        # Episode length termination
        if self.current_step >= self.config.max_steps:
            self.truncations = {agent: True for agent in self.agents}
            return
        
        # Risk-based termination
        if self.portfolio_state['drawdown_current'] > self.config.max_drawdown_threshold:
            self.terminations = {agent: True for agent in self.agents}
            return
        
        # Portfolio value termination
        if self.portfolio_state['portfolio_value'] <= self.config.initial_portfolio_value * 0.5:
            self.terminations = {agent: True for agent in self.agents}
            return
    
    def _accumulate_rewards(self):
        """Accumulate rewards for each agent"""
        for agent in self.possible_agents:
            self.episode_rewards[agent] += self.agent_rewards.get(agent, 0.0)
    
    def _was_dead_step(self, action):
        """Handle step for terminated/truncated agent"""
        # Agent was dead, return zero reward
        return None
    
    def observe(self, agent: str) -> np.ndarray:
        """
        Get observation for specific agent
        
        Args:
            agent: Agent name
            
        Returns:
            Agent observation
        """
        if agent not in self.agent_observations:
            self._generate_observations()
        
        return self.agent_observations.get(agent, np.zeros(self.observation_spaces[agent].shape))
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment
        
        Args:
            mode: Render mode
            
        Returns:
            Rendered array if mode is 'rgb_array'
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_state['portfolio_value']:.2f}")
            print(f"Current Position: {self.portfolio_state['current_position']:.3f}")
            print(f"Unrealized PnL: ${self.portfolio_state['unrealized_pnl']:.2f}")
            print(f"Market Price: ${self.market_state.price:.2f}")
            print(f"Drawdown: {self.portfolio_state['drawdown_current']:.2%}")
            print(f"VaR: {self.portfolio_state['var_estimate']:.2%}")
            print("-" * 50)
        
        return None
    
    def close(self):
        """Close the environment"""
        if hasattr(self, 'execution_system'):
            # Cleanup execution system
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.execution_system.shutdown())
            finally:
                loop.close()
        
        logger.info("Environment closed")
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get episode statistics"""
        return {
            'episode_length': self.current_step,
            'total_rewards': self.episode_rewards.copy(),
            'final_portfolio_value': self.portfolio_state['portfolio_value'],
            'total_pnl': self.portfolio_state['realized_pnl'] + self.portfolio_state['unrealized_pnl'],
            'max_drawdown': self.portfolio_state['max_drawdown'],
            'final_position': self.portfolio_state['current_position'],
            'execution_count': len(self.execution_history),
            'performance_metrics': self.execution_system.get_performance_report(),
            'episode_duration': datetime.now() - self.episode_start_time
        }


# Wrapper functions for PettingZoo compliance
def env(config: Optional[ExecutionEnvironmentConfig] = None):
    """
    Create execution environment
    
    Args:
        config: Environment configuration
        
    Returns:
        Wrapped environment
    """
    env = ExecutionEnvironment(config)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(config: Optional[ExecutionEnvironmentConfig] = None):
    """
    Create raw execution environment
    
    Args:
        config: Environment configuration
        
    Returns:
        Raw environment
    """
    return ExecutionEnvironment(config)


# Example usage
if __name__ == "__main__":
    # Create environment with custom config
    config = ExecutionEnvironmentConfig(
        max_steps=1000,
        initial_portfolio_value=100000.0,
        max_position_size=0.2,
        transaction_cost_bps=2.0
    )
    
    env = env(config)
    
    # Test basic functionality
    env.reset()
    
    print("Environment created successfully!")
    print(f"Agents: {env.possible_agents}")
    print(f"Action spaces: {list(env.action_spaces.keys())}")
    print(f"Observation spaces: {list(env.observation_spaces.keys())}")
    
    # Run a few steps
    for i in range(10):
        for agent in env.agent_iter():
            observation = env.observe(agent)
            action = env.action_spaces[agent].sample()
            env.step(action)
            
            if env.terminations[agent] or env.truncations[agent]:
                break
        
        if any(env.terminations.values()) or any(env.truncations.values()):
            break
    
    # Get episode statistics
    stats = env.get_episode_statistics()
    print(f"\nEpisode Statistics:")
    print(f"Episode Length: {stats['episode_length']}")
    print(f"Total Rewards: {stats['total_rewards']}")
    print(f"Final Portfolio Value: ${stats['final_portfolio_value']:.2f}")
    
    env.close()