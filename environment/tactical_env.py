"""
Enhanced Tactical 5-Minute MARL Environment

Implements a sophisticated PettingZoo AECEnv with state machine logic for coordinating
three tactical agents: FVG Agent, Momentum Agent, and Entry Optimization Agent.

Features:
- State machine: AWAITING_FVG -> AWAITING_MOMENTUM -> AWAITING_ENTRY_OPT -> READY_FOR_AGGREGATION
- 60x7 matrix construction with exact PRD mathematical formulas
- Superposition output collection and decision aggregation
- Configuration-driven flexibility
- Production-grade error handling and logging

Author: Quantitative Engineer
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from enum import Enum
import logging
from dataclasses import dataclass
import yaml
from pathlib import Path

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium import spaces
import torch

from src.core.events import Event, EventType
from src.core.event_bus import EventBus
from src.indicators.custom.fvg import FVGDetector
from src.matrix.assembler_5m import MatrixAssembler5m
from components.tactical_decision_aggregator import TacticalDecisionAggregator
from training.tactical_reward_system import TacticalRewardSystem

logger = logging.getLogger(__name__)


class TacticalState(Enum):
    """Internal state machine for tactical environment"""
    AWAITING_FVG = "awaiting_fvg"
    AWAITING_MOMENTUM = "awaiting_momentum"
    AWAITING_ENTRY_OPT = "awaiting_entry_opt"
    READY_FOR_AGGREGATION = "ready_for_aggregation"
    EPISODE_DONE = "episode_done"


@dataclass
class AgentOutput:
    """Container for agent superposition output"""
    agent_id: str
    action: int
    probabilities: np.ndarray
    confidence: float
    timestamp: float


@dataclass
class MarketState:
    """Container for current market state"""
    matrix: np.ndarray  # 60x7 matrix
    price: float
    volume: float
    timestamp: float
    features: Dict[str, Any]


class TacticalMarketEnv(AECEnv):
    """
    Enhanced Tactical 5-Minute MARL Environment
    
    Manages three tactical agents in a turn-based decision-making process:
    1. FVG Agent: Analyzes Fair Value Gap patterns
    2. Momentum Agent: Evaluates price momentum and trend continuation
    3. Entry Optimization Agent: Optimizes entry timing and execution
    
    The environment coordinates superposition outputs from all agents and
    triggers decision aggregation when all agents have acted.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "name": "tactical_marl_v1"}
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Tactical Market Environment
        
        Args:
            config: Configuration dictionary or path to config file
        """
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config)
        
        # Agent definitions (CRITICAL: Only tactical agents)
        self.possible_agents = ['fvg_agent', 'momentum_agent', 'entry_opt_agent']
        self.agents = self.possible_agents.copy()
        
        # Initialize agent selector for turn-based coordination
        self.agent_selector = agent_selector(self.agents)
        
        # State machine
        self.tactical_state = TacticalState.AWAITING_FVG
        
        # Agent outputs collection
        self.agent_outputs: Dict[str, AgentOutput] = {}
        
        # Current market state
        self.market_state: Optional[MarketState] = None
        
        # Action and observation spaces
        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.possible_agents  # [bearish, neutral, bullish]
        }
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(60, 7), dtype=np.float32
            ) for agent in self.possible_agents
        }
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.episode_count = 0
        self.step_count = 0
        self.decision_latencies = deque(maxlen=1000)
        
        # Episode state
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        logger.info(f"TacticalMarketEnv initialized with {len(self.agents)} agents")
        
    def _load_config(self, config: Any) -> Dict[str, Any]:
        """Load configuration from file or dict"""
        if isinstance(config, dict):
            return config
        elif isinstance(config, str) or isinstance(config, Path):
            with open(config, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for tactical environment"""
        return {
            'tactical_marl': {
                'environment': {
                    'matrix_shape': [60, 7],
                    'feature_names': [
                        'fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level',
                        'fvg_age', 'fvg_mitigation_signal', 'price_momentum_5', 'volume_ratio'
                    ],
                    'max_episode_steps': 1000,
                    'decision_timeout_ms': 100
                },
                'agents': {
                    'fvg_agent': {
                        'attention_weights': [0.4, 0.4, 0.1, 0.05, 0.05],
                        'focus_features': ['fvg_bullish_active', 'fvg_bearish_active', 'fvg_nearest_level']
                    },
                    'momentum_agent': {
                        'attention_weights': [0.05, 0.05, 0.1, 0.3, 0.5],
                        'focus_features': ['price_momentum_5', 'volume_ratio']
                    },
                    'entry_opt_agent': {
                        'attention_weights': [0.2, 0.2, 0.2, 0.2, 0.2],
                        'focus_features': ['fvg_mitigation_signal', 'volume_ratio']
                    }
                }
            }
        }
    
    def _initialize_components(self):
        """Initialize core components for tactical trading"""
        try:
            # Matrix assembler for 5-minute data
            self.matrix_assembler = MatrixAssembler5m(
                config=self.config.get('tactical_marl', {}).get('environment', {})
            )
            
            # FVG detector for gap analysis
            self.fvg_detector = FVGDetector(
                config=self.config.get('fvg_config', {}),
                event_bus=None  # Will be set externally if needed
            )
            
            # Decision aggregator for consensus
            self.decision_aggregator = TacticalDecisionAggregator(
                config=self.config.get('tactical_marl', {}).get('aggregation', {})
            )
            
            # Reward system for performance evaluation
            self.reward_system = TacticalRewardSystem(
                config=self.config.get('tactical_marl', {}).get('rewards', {})
            )
            
            logger.info("All tactical components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tactical components: {e}")
            raise
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Reset environment for new episode
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Initial observations for all agents
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.episode_count += 1
        self.step_count = 0
        self.tactical_state = TacticalState.AWAITING_FVG
        
        # Reset agents
        self.agents = self.possible_agents.copy()
        self.agent_selector.reset()
        
        # Clear agent outputs
        self.agent_outputs.clear()
        
        # Reset rewards and dones
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Generate initial market state
        self._update_market_state()
        
        # Generate initial observations
        observations = {}
        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
        
        logger.info(f"Episode {self.episode_count} reset complete")
        return observations
    
    def _update_market_state(self):
        """Update current market state with latest data"""
        try:
            # Get current matrix from assembler
            matrix = self.matrix_assembler.get_matrix()
            if matrix is None:
                # Generate synthetic data for testing
                matrix = self._generate_synthetic_matrix()
            
            # Extract current market features
            features = self._extract_market_features(matrix)
            
            # Update market state
            self.market_state = MarketState(
                matrix=matrix,
                price=features.get('current_price', 100.0),
                volume=features.get('current_volume', 1000.0),
                timestamp=self.step_count,
                features=features
            )
            
        except Exception as e:
            logger.error(f"Failed to update market state: {e}")
            # Fallback to synthetic data
            self.market_state = MarketState(
                matrix=self._generate_synthetic_matrix(),
                price=100.0,
                volume=1000.0,
                timestamp=self.step_count,
                features={}
            )
    
    def _generate_synthetic_matrix(self) -> np.ndarray:
        """Generate synthetic 60x7 matrix for testing"""
        matrix = np.random.randn(60, 7).astype(np.float32)
        
        # Add some realistic patterns
        # FVG signals (binary)
        matrix[:, 0] = (np.random.rand(60) > 0.8).astype(np.float32)  # fvg_bullish_active
        matrix[:, 1] = (np.random.rand(60) > 0.8).astype(np.float32)  # fvg_bearish_active
        matrix[:, 2] = np.random.normal(0, 0.5, 60)  # fvg_nearest_level
        matrix[:, 3] = np.random.exponential(2, 60)  # fvg_age
        matrix[:, 4] = (np.random.rand(60) > 0.9).astype(np.float32)  # fvg_mitigation_signal
        matrix[:, 5] = np.random.normal(0, 0.3, 60)  # price_momentum_5
        matrix[:, 6] = np.random.lognormal(0, 0.3, 60)  # volume_ratio
        
        return matrix
    
    def _extract_market_features(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Extract market features from matrix"""
        if matrix is None or matrix.shape[0] == 0:
            return {}
        
        # Get latest bar features
        latest_bar = matrix[-1]
        feature_names = self.config['tactical_marl']['environment']['feature_names']
        
        features = {}
        for i, name in enumerate(feature_names):
            if i < len(latest_bar):
                features[name] = float(latest_bar[i])
        
        # Add derived features
        features['current_price'] = 100.0 + np.random.normal(0, 1)
        features['current_volume'] = 1000.0 + np.random.normal(0, 100)
        
        return features
    
    def observe(self, agent: str) -> np.ndarray:
        """
        Generate observation for specific agent
        
        Args:
            agent: Agent identifier
            
        Returns:
            60x7 matrix observation
        """
        return self._get_observation(agent)
    
    def _get_observation(self, agent: str) -> np.ndarray:
        """Get observation for specific agent"""
        if self.market_state is None:
            return np.zeros((60, 7), dtype=np.float32)
        
        # Base observation is the full matrix
        obs = self.market_state.matrix.copy()
        
        # Apply agent-specific attention weights
        agent_config = self.config['tactical_marl']['agents'].get(agent, {})
        attention_weights = agent_config.get('attention_weights', [1.0] * 7)
        
        # Apply attention weighting
        for i, weight in enumerate(attention_weights):
            if i < obs.shape[1]:
                obs[:, i] *= weight
        
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Dict]]:
        """
        Execute one step of the environment
        
        This method implements sophisticated multi-agent coordination:
        1. Receives superposition output from current agent
        2. Stores output in agent_outputs
        3. Advances to next agent
        4. After all agents act, triggers decision aggregation and reward calculation
        
        Args:
            action: Agent's action (probability distribution will be extracted)
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        if not self.agents:
            raise ValueError("Environment has no active agents")
        
        current_agent = self.agent_selection
        
        # Convert action to superposition probabilities
        probabilities = self._action_to_probabilities(action)
        
        # Store agent output
        agent_output = AgentOutput(
            agent_id=current_agent,
            action=action,
            probabilities=probabilities,
            confidence=float(np.max(probabilities)),
            timestamp=self.step_count
        )
        self.agent_outputs[current_agent] = agent_output
        
        # Update state machine
        self._update_state_machine(current_agent)
        
        # Advance to next agent
        self.agent_selector.next()
        
        # Check if all agents have acted
        if len(self.agent_outputs) == len(self.possible_agents):
            # Trigger decision aggregation and reward calculation
            self._process_agent_decisions()
            
            # Reset for next decision cycle
            self.agent_outputs.clear()
            self.tactical_state = TacticalState.AWAITING_FVG
            self.agent_selector.reset()
        
        # Update market state
        self._update_market_state()
        
        # Generate observations for all agents
        observations = {}
        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
        
        # Increment step count
        self.step_count += 1
        
        # Check episode termination
        max_steps = self.config['tactical_marl']['environment'].get('max_episode_steps', 1000)
        if self.step_count >= max_steps:
            self.dones = {agent: True for agent in self.agents}
            self.tactical_state = TacticalState.EPISODE_DONE
        
        return observations, self.rewards, self.dones, self.infos
    
    def _action_to_probabilities(self, action: int) -> np.ndarray:
        """Convert discrete action to probability distribution"""
        # For now, convert discrete action to one-hot probability
        # In production, this would come from the agent's policy output
        probs = np.zeros(3, dtype=np.float32)
        probs[action] = 1.0
        return probs
    
    def _update_state_machine(self, agent: str):
        """Update internal state machine based on agent completion"""
        if agent == 'fvg_agent':
            self.tactical_state = TacticalState.AWAITING_MOMENTUM
        elif agent == 'momentum_agent':
            self.tactical_state = TacticalState.AWAITING_ENTRY_OPT
        elif agent == 'entry_opt_agent':
            self.tactical_state = TacticalState.READY_FOR_AGGREGATION
    
    def _process_agent_decisions(self):
        """Process all agent decisions and calculate rewards"""
        try:
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            # Decision aggregation
            decision_result = self.decision_aggregator.aggregate_decisions(
                agent_outputs=self.agent_outputs,
                market_state=self.market_state,
                synergy_context=self.infos.get('synergy_context', {})
            )
            
            # Reward calculation
            reward_result = self.reward_system.calculate_tactical_reward(
                decision_result=decision_result,
                market_state=self.market_state,
                agent_outputs=self.agent_outputs
            )
            
            # Distribute rewards to agents
            self._distribute_rewards(reward_result)
            
            # Update infos with decision and reward details
            self._update_infos(decision_result, reward_result)
            
            # Track performance
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if start_time and end_time:
                latency = start_time.elapsed_time(end_time)
                self.decision_latencies.append(latency)
            
            logger.debug(f"Decision processed: {decision_result.get('execute', False)}")
            
        except Exception as e:
            logger.error(f"Error processing agent decisions: {e}")
            # Fallback to zero rewards
            self.rewards = {agent: 0.0 for agent in self.agents}
    
    def _distribute_rewards(self, reward_result: Dict[str, Any]):
        """Distribute rewards to agents based on reward result"""
        # Base reward for all agents
        base_reward = reward_result.get('total_reward', 0.0)
        
        # Agent-specific rewards
        agent_specific = reward_result.get('agent_specific', {})
        
        for agent in self.agents:
            self.rewards[agent] = base_reward
            
            # Add agent-specific bonuses
            if agent in agent_specific:
                agent_rewards = agent_specific[agent]
                for bonus_name, bonus_value in agent_rewards.items():
                    self.rewards[agent] += bonus_value
    
    def _update_infos(self, decision_result: Dict[str, Any], reward_result: Dict[str, Any]):
        """Update info dictionaries with decision and reward details"""
        for agent in self.agents:
            self.infos[agent].update({
                'decision_result': decision_result,
                'reward_components': reward_result,
                'step_count': self.step_count,
                'tactical_state': self.tactical_state.value
            })
    
    @property
    def agent_selection(self) -> str:
        """Get current agent selection"""
        return self.agent_selector.agent_selection
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment state"""
        if mode == "human":
            print(f"Episode: {self.episode_count}, Step: {self.step_count}")
            print(f"State: {self.tactical_state.value}")
            print(f"Current Agent: {self.agent_selection}")
            print(f"Agent Outputs: {len(self.agent_outputs)}/{len(self.possible_agents)}")
            
            if self.market_state:
                print(f"Price: {self.market_state.price:.2f}")
                print(f"Volume: {self.market_state.volume:.0f}")
        
        return None
    
    def close(self):
        """Clean up environment resources"""
        self.agents.clear()
        self.agent_outputs.clear()
        logger.info("TacticalMarketEnv closed")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get environment performance metrics"""
        metrics = {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'tactical_state': self.tactical_state.value,
            'agent_count': len(self.agents),
            'decision_latencies': {
                'mean': np.mean(self.decision_latencies) if self.decision_latencies else 0,
                'p95': np.percentile(self.decision_latencies, 95) if self.decision_latencies else 0,
                'p99': np.percentile(self.decision_latencies, 99) if self.decision_latencies else 0
            }
        }
        
        return metrics


def make_tactical_env(config: Optional[Dict[str, Any]] = None) -> TacticalMarketEnv:
    """
    Factory function to create tactical environment
    
    Args:
        config: Environment configuration
        
    Returns:
        Configured TacticalMarketEnv instance
    """
    env = TacticalMarketEnv(config)
    
    # Apply PettingZoo wrappers if needed
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)
    
    return env