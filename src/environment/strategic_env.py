"""
PettingZoo Environment Wrapper for Strategic MARL System

This module provides a PettingZoo AECEnv wrapper that integrates with the existing
strategic agent architecture, using the three specialized agents (MLMI, NWRQK, Regime)
and the strategic MARL component for decision-making.

Key Features:
- Inherits from PettingZoo AECEnv for compatibility
- Integrates with existing strategic agent base classes
- Supports agent-specific feature extraction and observation spaces
- Implements proper turn-based execution with state machine
- Provides reward calculation and episode management
- Includes performance monitoring and error handling
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
from collections import deque

# PettingZoo imports
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces

# Core system imports
from src.core.component_base import ComponentBase
from src.core.events import EventType, Event
from src.core.minimal_dependencies import MinimalComponentBase

# Strategic agent imports
from src.agents.strategic_agent_base import (
    MLMIStrategicAgent, 
    NWRQKStrategicAgent, 
    RegimeDetectionAgent,
    AgentPrediction
)
from src.agents.strategic_marl_component import StrategicMARLComponent

# Matrix assembler for data generation
try:
    from src.matrix.assembler_30m_enhanced import MatrixAssembler30mEnhanced
except ImportError:
    MatrixAssembler30mEnhanced = None

# Synergy detection
try:
    from src.synergy.detector import SynergyDetector
except ImportError:
    SynergyDetector = None

logger = logging.getLogger(__name__)


class EnvironmentPhase(Enum):
    """Environment execution phases"""
    SETUP = "setup"
    AGENT_DECISION = "agent_decision"
    AGGREGATION = "aggregation"
    REWARD_CALCULATION = "reward_calculation"
    EPISODE_END = "episode_end"


@dataclass
class EnvironmentState:
    """Environment state tracking"""
    phase: EnvironmentPhase
    current_agent_index: int
    episode_step: int
    episode_reward: float
    agent_decisions: Dict[str, AgentPrediction]
    matrix_data: Optional[np.ndarray]
    shared_context: Optional[Dict[str, Any]]
    timestamp: datetime


class StrategicMARLEnvironment(AECEnv):
    """
    PettingZoo AEC Environment for Strategic MARL System
    
    This environment provides a turn-based interface for the three strategic agents
    (MLMI, NWRQK, Regime) to make decisions based on market matrix data.
    
    Key Features:
    - Agent-specific observation spaces based on feature indices
    - Turn-based execution with proper state management
    - Integration with existing strategic components
    - Reward calculation based on ensemble decision performance
    - Episode management with configurable termination conditions
    """
    
    metadata = {
        "name": "strategic_marl_v1",
        "is_parallelizable": False,
        "render_modes": ["human", "rgb_array", "ansi"],
    }
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        kernel: Optional[Any] = None,
        matrix_assembler: Optional[Any] = None,
        synergy_detector: Optional[Any] = None
    ):
        """
        Initialize Strategic MARL Environment
        
        Args:
            config: Environment configuration dictionary
            kernel: AlgoSpace kernel for system integration
            matrix_assembler: Matrix assembler for data generation
            synergy_detector: Synergy detector for pattern recognition
        """
        super().__init__()
        
        # Configuration
        self.config = config or self._get_default_config()
        self.kernel = kernel
        
        # Environment setup
        self.name = "StrategicMARLEnvironment"
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Agent configuration
        self.possible_agents = ["mlmi_expert", "nwrqk_expert", "regime_expert"]
        self.agent_name_mapping = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }
        
        # Feature indices for each agent (from strategic config)
        self.feature_indices = self.config.get("feature_indices", {
            "mlmi_expert": [0, 1, 9, 10],      # MLMI features
            "nwrqk_expert": [2, 3, 4, 5],      # NWRQK features  
            "regime_expert": [10, 11, 12]      # Regime features
        })
        
        # Initialize components
        self.matrix_assembler = matrix_assembler
        self.synergy_detector = synergy_detector
        self.strategic_component = None
        
        # Initialize agents
        self.agents_dict = {}
        self._initialize_agents()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Environment state
        self.env_state = EnvironmentState(
            phase=EnvironmentPhase.SETUP,
            current_agent_index=0,
            episode_step=0,
            episode_reward=0.0,
            agent_decisions={},
            matrix_data=None,
            shared_context=None,
            timestamp=datetime.now()
        )
        
        # Episode management
        self.max_episode_steps = self.config.get("max_episode_steps", 1000)
        self.episode_count = 0
        self.total_episodes = self.config.get("total_episodes", 10000)
        
        # Performance tracking
        self.performance_metrics = {
            "episode_rewards": deque(maxlen=100),
            "episode_lengths": deque(maxlen=100),
            "agent_decision_times": deque(maxlen=1000),
            "ensemble_confidences": deque(maxlen=1000),
            "successful_episodes": 0,
            "failed_episodes": 0
        }
        
        # Reward calculation
        self.reward_scale = self.config.get("reward_scale", 1.0)
        self.confidence_bonus = self.config.get("confidence_bonus", 0.1)
        
        self.logger.info(f"StrategicMARLEnvironment initialized with {len(self.possible_agents)} agents")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "matrix_shape": [48, 13],
            "max_episode_steps": 1000,
            "total_episodes": 10000,
            "reward_scale": 1.0,
            "confidence_bonus": 0.1,
            "confidence_threshold": 0.65,
            "feature_indices": {
                "mlmi_expert": [0, 1, 9, 10],
                "nwrqk_expert": [2, 3, 4, 5],
                "regime_expert": [10, 11, 12]
            },
            "agents": {
                "mlmi_expert": {
                    "hidden_dims": [256, 128, 64],
                    "dropout_rate": 0.1
                },
                "nwrqk_expert": {
                    "hidden_dims": [256, 128, 64],
                    "dropout_rate": 0.1
                },
                "regime_expert": {
                    "hidden_dims": [256, 128, 64],
                    "dropout_rate": 0.15
                }
            },
            "environment": {
                "data_source": "synthetic",
                "noise_level": 0.1,
                "market_conditions": ["normal", "volatile", "trending"]
            }
        }
    
    def _initialize_agents(self):
        """Initialize strategic agents"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize MLMI agent
            self.agents_dict["mlmi_expert"] = MLMIStrategicAgent(
                config=self.config, 
                device=device
            )
            
            # Initialize NWRQK agent  
            self.agents_dict["nwrqk_expert"] = NWRQKStrategicAgent(
                config=self.config,
                device=device
            )
            
            # Initialize Regime agent
            self.agents_dict["regime_expert"] = RegimeDetectionAgent(
                config=self.config,
                device=device
            )
            
            self.logger.info("All strategic agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Action space: probability distribution over [buy, hold, sell]
        self._action_spaces = {
            agent: spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        # Observation spaces: agent-specific features + shared context
        self._observation_spaces = {}
        
        for agent in self.possible_agents:
            agent_feature_dim = len(self.feature_indices[agent])
            
            self._observation_spaces[agent] = spaces.Dict({
                "agent_features": spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(agent_feature_dim,), 
                    dtype=np.float32
                ),
                "shared_context": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(6,),  # Standard shared context size
                    dtype=np.float32
                ),
                "market_matrix": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(48, 13),  # Full matrix for context
                    dtype=np.float32
                ),
                "episode_info": spaces.Dict({
                    "episode_step": spaces.Discrete(self.max_episode_steps),
                    "phase": spaces.Discrete(len(EnvironmentPhase)),
                    "agent_index": spaces.Discrete(len(self.possible_agents))
                })
            })
    
    @property
    def observation_spaces(self):
        """PettingZoo compatibility - return observation spaces dict"""
        return self._observation_spaces
    
    @property  
    def action_spaces(self):
        """PettingZoo compatibility - return action spaces dict"""
        return self._action_spaces
    
    def observation_space(self, agent: str):
        """Return observation space for specific agent"""
        return self._observation_spaces[agent]
    
    def action_space(self, agent: str):
        """Return action space for specific agent"""
        return self._action_spaces[agent]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Reset PettingZoo state
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # Reset environment state
        self.env_state = EnvironmentState(
            phase=EnvironmentPhase.SETUP,
            current_agent_index=0,
            episode_step=0,
            episode_reward=0.0,
            agent_decisions={},
            matrix_data=None,
            shared_context=None,
            timestamp=datetime.now()
        )
        
        # Generate initial market data
        self.env_state.matrix_data = self._generate_market_data()
        self.env_state.shared_context = self._extract_shared_context(self.env_state.matrix_data)
        
        # Reset episode tracking
        self.episode_count += 1
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Initialize agents for new episode
        self._reset_agents()
        
        # Move to agent decision phase
        self.env_state.phase = EnvironmentPhase.AGENT_DECISION
        
        self.logger.info(f"Environment reset for episode {self.episode_count}")
    
    def step(self, action: Union[np.ndarray, Dict]):
        """Execute one step of the environment"""
        if self.env_state.phase == EnvironmentPhase.EPISODE_END:
            return  # Episode already ended
        
        start_time = time.time()
        current_agent = self.agent_selection
        
        # Validate and process action
        action = self._validate_action(action)
        
        # Execute agent decision
        agent_prediction = self._execute_agent_decision(current_agent, action)
        self.env_state.agent_decisions[current_agent] = agent_prediction
        
        # Update agent info
        self.infos[current_agent] = {
            "agent_name": current_agent,
            "action_probabilities": action.tolist(),
            "confidence": agent_prediction.confidence,
            "computation_time_ms": agent_prediction.computation_time_ms,
            "feature_importance": agent_prediction.feature_importance
        }
        
        # Check if all agents have decided
        if len(self.env_state.agent_decisions) == len(self.possible_agents):
            self._process_ensemble_decision()
            self._calculate_rewards()
            self._update_episode_state()
        
        # Move to next agent
        self._advance_agent_selection()
        
        # Update performance metrics
        decision_time = (time.time() - start_time) * 1000
        self.performance_metrics["agent_decision_times"].append(decision_time)
        
        # Check termination conditions
        self._check_termination_conditions()
    
    def _validate_action(self, action: Union[np.ndarray, Dict]) -> np.ndarray:
        """Validate and normalize action"""
        if isinstance(action, dict):
            action = np.array(action.get("action", [0.33, 0.34, 0.33]))
        
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        if action.shape != (3,):
            raise ValueError(f"Action must be shape (3,), got {action.shape}")
        
        # Normalize to probability distribution
        action = np.clip(action, 0.0, 1.0)
        action = action / (action.sum() + 1e-8)
        
        return action.astype(np.float32)
    
    def _execute_agent_decision(self, agent_name: str, action: np.ndarray) -> AgentPrediction:
        """Execute agent decision and return prediction"""
        agent = self.agents_dict[agent_name]
        
        # Create agent prediction from action
        confidence = float(np.max(action))
        
        # Get feature importance (simplified)
        feature_importance = {}
        features = self.feature_indices[agent_name]
        for i, feature_idx in enumerate(features):
            feature_importance[f"feature_{feature_idx}"] = float(1.0 / len(features))
        
        # Create prediction
        prediction = AgentPrediction(
            action_probabilities=action,
            confidence=confidence,
            feature_importance=feature_importance,
            internal_state={
                "agent_name": agent_name,
                "episode_step": self.env_state.episode_step,
                "phase": self.env_state.phase.value
            },
            computation_time_ms=1.0,  # Placeholder
            timestamp=datetime.now()
        )
        
        return prediction
    
    def _process_ensemble_decision(self):
        """Process ensemble decision from all agents"""
        # Calculate ensemble probabilities (simple average)
        agent_probs = [
            pred.action_probabilities 
            for pred in self.env_state.agent_decisions.values()
        ]
        ensemble_probs = np.mean(agent_probs, axis=0)
        
        # Calculate ensemble confidence
        confidences = [
            pred.confidence 
            for pred in self.env_state.agent_decisions.values()
        ]
        ensemble_confidence = np.mean(confidences)
        
        # Store ensemble decision
        self.env_state.shared_context["ensemble_decision"] = {
            "probabilities": ensemble_probs.tolist(),
            "confidence": ensemble_confidence,
            "action": ["buy", "hold", "sell"][np.argmax(ensemble_probs)]
        }
        
        # Update performance metrics
        self.performance_metrics["ensemble_confidences"].append(ensemble_confidence)
        
        # Move to aggregation phase
        self.env_state.phase = EnvironmentPhase.AGGREGATION
    
    def _calculate_rewards(self):
        """Calculate rewards for all agents"""
        ensemble_decision = self.env_state.shared_context["ensemble_decision"]
        base_reward = self._get_market_reward(ensemble_decision)
        
        for agent_name in self.possible_agents:
            agent_pred = self.env_state.agent_decisions[agent_name]
            
            # Base reward scaled by confidence
            reward = base_reward * agent_pred.confidence * self.reward_scale
            
            # Confidence bonus
            if agent_pred.confidence > self.config["confidence_threshold"]:
                reward += self.confidence_bonus
            
            # Ensemble alignment bonus
            alignment = self._calculate_ensemble_alignment(agent_pred, ensemble_decision)
            reward += alignment * 0.1
            
            self.rewards[agent_name] = reward
        
        # Update episode reward
        self.env_state.episode_reward = sum(self.rewards.values())
        
        # Move to reward calculation phase
        self.env_state.phase = EnvironmentPhase.REWARD_CALCULATION
    
    def _get_market_reward(self, ensemble_decision: Dict) -> float:
        """Calculate market-based reward (placeholder)"""
        # This would integrate with actual market data and performance
        # For now, return a reward based on confidence and randomness
        confidence = ensemble_decision["confidence"]
        market_performance = np.random.normal(0, 0.1)  # Simulated market performance
        
        return confidence * market_performance
    
    def _calculate_ensemble_alignment(self, agent_pred: AgentPrediction, ensemble_decision: Dict) -> float:
        """Calculate alignment between agent and ensemble decision"""
        agent_probs = agent_pred.action_probabilities
        ensemble_probs = np.array(ensemble_decision["probabilities"])
        
        # Calculate KL divergence (lower is better alignment)
        kl_div = np.sum(ensemble_probs * np.log((ensemble_probs + 1e-8) / (agent_probs + 1e-8)))
        
        # Convert to alignment score (higher is better)
        alignment = np.exp(-kl_div)
        
        return alignment
    
    def _update_episode_state(self):
        """Update episode state after processing all agents"""
        self.env_state.episode_step += 1
        self.env_state.timestamp = datetime.now()
        
        # Clear agent decisions for next step
        self.env_state.agent_decisions.clear()
        
        # Generate new market data
        self.env_state.matrix_data = self._generate_market_data()
        self.env_state.shared_context = self._extract_shared_context(self.env_state.matrix_data)
        
        # Return to agent decision phase
        self.env_state.phase = EnvironmentPhase.AGENT_DECISION
    
    def _advance_agent_selection(self):
        """Advance to next agent in turn order"""
        if len(self.env_state.agent_decisions) < len(self.possible_agents):
            self.agent_selection = self._agent_selector.next()
            self.env_state.current_agent_index = self.agent_name_mapping[self.agent_selection]
    
    def _check_termination_conditions(self):
        """Check if episode should terminate"""
        # Episode step limit
        if self.env_state.episode_step >= self.max_episode_steps:
            self.terminations = {agent: True for agent in self.agents}
            self.env_state.phase = EnvironmentPhase.EPISODE_END
            self.performance_metrics["episode_lengths"].append(self.env_state.episode_step)
        
        # Maximum episodes reached
        if self.episode_count >= self.total_episodes:
            self.truncations = {agent: True for agent in self.agents}
            self.env_state.phase = EnvironmentPhase.EPISODE_END
        
        # Episode reward threshold (optional)
        if self.env_state.episode_reward > self.config.get("reward_threshold", float('inf')):
            self.terminations = {agent: True for agent in self.agents}
            self.env_state.phase = EnvironmentPhase.EPISODE_END
            self.performance_metrics["successful_episodes"] += 1
    
    def observe(self, agent: str) -> Dict[str, Any]:
        """Get observation for specific agent"""
        if self.env_state.matrix_data is None:
            return self._get_empty_observation(agent)
        
        # Extract agent-specific features
        agent_features = self._extract_agent_features(agent, self.env_state.matrix_data)
        
        # Build observation
        observation = {
            "agent_features": agent_features,
            "shared_context": self._get_shared_context_vector(),
            "market_matrix": self.env_state.matrix_data.copy(),
            "episode_info": {
                "episode_step": self.env_state.episode_step,
                "phase": self.env_state.phase.value,
                "agent_index": self.env_state.current_agent_index
            }
        }
        
        return observation
    
    def _extract_agent_features(self, agent: str, matrix_data: np.ndarray) -> np.ndarray:
        """Extract agent-specific features from matrix"""
        feature_indices = self.feature_indices[agent]
        
        # Average across time dimension (48 bars)
        matrix_avg = np.mean(matrix_data, axis=0)
        agent_features = matrix_avg[feature_indices]
        
        return agent_features.astype(np.float32)
    
    def _get_shared_context_vector(self) -> np.ndarray:
        """Get shared context as vector"""
        if self.env_state.shared_context is None:
            return np.zeros(6, dtype=np.float32)
        
        context = self.env_state.shared_context
        vector = np.array([
            context.get("market_volatility", 0.0),
            context.get("volume_profile", 0.0),
            context.get("momentum_signal", 0.0),
            context.get("trend_strength", 0.0),
            context.get("mmd_score", 0.0),
            context.get("price_trend", 0.0)
        ], dtype=np.float32)
        
        return vector
    
    def _get_empty_observation(self, agent: str) -> Dict[str, Any]:
        """Get empty observation for initialization"""
        agent_feature_dim = len(self.feature_indices[agent])
        
        return {
            "agent_features": np.zeros(agent_feature_dim, dtype=np.float32),
            "shared_context": np.zeros(6, dtype=np.float32),
            "market_matrix": np.zeros((48, 13), dtype=np.float32),
            "episode_info": {
                "episode_step": 0,
                "phase": 0,
                "agent_index": 0
            }
        }
    
    def _generate_market_data(self) -> np.ndarray:
        """Generate market matrix data"""
        if self.matrix_assembler is not None:
            try:
                # Use real matrix assembler if available
                matrix_data = self.matrix_assembler.assemble_matrix()
                return matrix_data
            except Exception as e:
                self.logger.warning(f"Matrix assembler failed: {e}, using synthetic data")
        
        # Generate synthetic market data
        return self._generate_synthetic_market_data()
    
    def _generate_synthetic_market_data(self) -> np.ndarray:
        """Generate synthetic market data for testing"""
        # Generate realistic market data with trends and patterns
        base_data = np.random.randn(48, 13)
        
        # Add trend component
        trend = np.linspace(-0.1, 0.1, 48).reshape(-1, 1)
        base_data += trend
        
        # Add volatility clustering
        volatility = np.random.exponential(0.1, (48, 13))
        base_data *= volatility
        
        # Add some correlation between features
        correlation_matrix = np.random.uniform(0.1, 0.5, (13, 13))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Apply correlation (simplified)
        correlated_data = base_data @ correlation_matrix
        
        return correlated_data.astype(np.float32)
    
    def _extract_shared_context(self, matrix_data: np.ndarray) -> Dict[str, Any]:
        """Extract shared context from matrix data"""
        if matrix_data is None:
            return {}
        
        # Calculate market metrics
        volatility = np.std(matrix_data[:, -1]) if matrix_data.shape[1] > 0 else 0.0
        volume_profile = np.mean(matrix_data[:, -2]) if matrix_data.shape[1] > 1 else 0.0
        momentum_signal = np.mean(matrix_data[:, 9]) if matrix_data.shape[1] > 9 else 0.0
        trend_strength = np.mean(matrix_data[:, 10]) if matrix_data.shape[1] > 10 else 0.0
        mmd_score = np.mean(matrix_data[:, 11]) if matrix_data.shape[1] > 11 else 0.0
        price_trend = np.mean(matrix_data[:, 0]) if matrix_data.shape[1] > 0 else 0.0
        
        return {
            "market_volatility": float(volatility),
            "volume_profile": float(volume_profile),
            "momentum_signal": float(momentum_signal),
            "trend_strength": float(trend_strength),
            "mmd_score": float(mmd_score),
            "price_trend": float(price_trend),
            "market_regime": self._detect_market_regime(matrix_data),
            "timestamp": datetime.now().isoformat()
        }
    
    def _detect_market_regime(self, matrix_data: np.ndarray) -> str:
        """Detect market regime from matrix data"""
        if matrix_data is None or matrix_data.size == 0:
            return "unknown"
        
        # Simple regime detection
        volatility = np.std(matrix_data[:, -1])
        momentum = np.mean(matrix_data[:, 9]) if matrix_data.shape[1] > 9 else 0
        
        if volatility > 0.05:
            return "volatile"
        elif abs(momentum) > 0.01:
            return "trending"
        else:
            return "ranging"
    
    def _reset_agents(self):
        """Reset all agents for new episode"""
        for agent in self.agents_dict.values():
            if hasattr(agent, 'reset'):
                agent.reset()
    
    def last(self, observe: bool = True) -> Tuple:
        """Return last agent's observation, reward, termination, truncation, and info"""
        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        
        return (
            observation,
            self.rewards.get(agent, 0.0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {})
        )
    
    def render(self, mode: str = "human"):
        """Render environment state"""
        if mode == "human":
            self._render_human()
        elif mode == "rgb_array":
            return self._render_rgb_array()
        elif mode == "ansi":
            return self._render_ansi()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _render_human(self):
        """Render environment in human-readable format"""
        print(f"\n=== Strategic MARL Environment ===")
        print(f"Episode: {self.episode_count}")
        print(f"Step: {self.env_state.episode_step}")
        print(f"Phase: {self.env_state.phase.value}")
        print(f"Current Agent: {self.agent_selection}")
        print(f"Episode Reward: {self.env_state.episode_reward:.4f}")
        
        if self.env_state.agent_decisions:
            print(f"\nAgent Decisions:")
            for agent_name, pred in self.env_state.agent_decisions.items():
                print(f"  {agent_name}: {pred.action_probabilities} (conf: {pred.confidence:.2f})")
        
        if self.env_state.shared_context and "ensemble_decision" in self.env_state.shared_context:
            ensemble = self.env_state.shared_context["ensemble_decision"]
            print(f"\nEnsemble Decision: {ensemble['action']} (conf: {ensemble['confidence']:.2f})")
        
        print(f"\nPerformance Metrics:")
        avg_decision_time = np.mean(self.performance_metrics["agent_decision_times"][-10:]) if self.performance_metrics["agent_decision_times"] else 0
        print(f"  Avg Decision Time: {avg_decision_time:.2f}ms")
        print(f"  Successful Episodes: {self.performance_metrics['successful_episodes']}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array (placeholder)"""
        # This would create a visual representation
        # For now, return a placeholder image
        return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def _render_ansi(self) -> str:
        """Render environment as ANSI string"""
        output = []
        output.append("Strategic MARL Environment")
        output.append(f"Episode: {self.episode_count}, Step: {self.env_state.episode_step}")
        output.append(f"Phase: {self.env_state.phase.value}")
        output.append(f"Current Agent: {self.agent_selection}")
        output.append(f"Episode Reward: {self.env_state.episode_reward:.4f}")
        
        return "\n".join(output)
    
    def close(self):
        """Close environment and cleanup resources"""
        self.logger.info("Closing Strategic MARL Environment")
        
        # Cleanup agents
        for agent in self.agents_dict.values():
            if hasattr(agent, 'cleanup'):
                agent.cleanup()
        
        # Cleanup components
        if self.matrix_assembler and hasattr(self.matrix_assembler, 'cleanup'):
            self.matrix_assembler.cleanup()
        
        if self.synergy_detector and hasattr(self.synergy_detector, 'cleanup'):
            self.synergy_detector.cleanup()
        
        # Clear performance metrics
        self.performance_metrics.clear()
        
        self.logger.info("Environment closed successfully")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "episode_count": self.episode_count,
            "avg_episode_reward": np.mean(self.performance_metrics["episode_rewards"]) if self.performance_metrics["episode_rewards"] else 0.0,
            "avg_episode_length": np.mean(self.performance_metrics["episode_lengths"]) if self.performance_metrics["episode_lengths"] else 0.0,
            "avg_decision_time_ms": np.mean(self.performance_metrics["agent_decision_times"]) if self.performance_metrics["agent_decision_times"] else 0.0,
            "avg_ensemble_confidence": np.mean(self.performance_metrics["ensemble_confidences"]) if self.performance_metrics["ensemble_confidences"] else 0.0,
            "successful_episodes": self.performance_metrics["successful_episodes"],
            "failed_episodes": self.performance_metrics["failed_episodes"],
            "success_rate": self.performance_metrics["successful_episodes"] / max(1, self.episode_count)
        }


# Wrapper functions for PettingZoo compatibility
def env(**kwargs):
    """Create unwrapped environment"""
    return StrategicMARLEnvironment(**kwargs)


def raw_env(**kwargs):
    """Create raw environment"""
    return env(**kwargs)


def parallel_env(**kwargs):
    """Create parallel environment (not supported)"""
    raise NotImplementedError("Parallel execution not supported for Strategic MARL")


# Example usage and testing
if __name__ == "__main__":
    # Test environment creation
    config = {
        "max_episode_steps": 100,
        "total_episodes": 10,
        "reward_scale": 1.0,
        "confidence_threshold": 0.6
    }
    
    env = StrategicMARLEnvironment(config=config)
    
    # Test reset
    env.reset()
    print(f"Environment reset. Current agent: {env.agent_selection}")
    
    # Test observation
    obs = env.observe(env.agent_selection)
    print(f"Observation keys: {list(obs.keys())}")
    
    # Test step
    action = np.array([0.4, 0.4, 0.2])  # Buy bias
    env.step(action)
    
    # Test rendering
    env.render()
    
    # Get performance metrics
    metrics = env.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Close environment
    env.close()