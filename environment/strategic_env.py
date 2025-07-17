"""
Strategic Market Environment for MARL

PettingZoo AEC environment implementing the strategic decision-making process
with 3 expert agents (MLMI, NWRQK, Regime) in a turn-based setting.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import time
import logging
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import torch

logger = logging.getLogger(__name__)

# Import existing components
import sys
sys.path.append('/home/QuantNova/GrandModel')

# Use minimal dependencies for standalone operation
from src.core.minimal_dependencies import MinimalComponentBase as ComponentBase

# Import matrix assembler (will use mock for testing)
try:
    from src.matrix.assembler_30m_enhanced import MatrixAssembler30mEnhanced
except ImportError:
    MatrixAssembler30mEnhanced = None


class EnvironmentState(Enum):
    """State machine for managing agent turn sequence"""
    AWAITING_MLMI = "awaiting_mlmi"
    AWAITING_NWRQK = "awaiting_nwrqk" 
    AWAITING_REGIME = "awaiting_regime"
    READY_FOR_AGGREGATION = "ready_for_aggregation"


class StrategicMarketEnv(AECEnv):
    """
    Strategic MARL Environment
    
    Manages turn-based execution of 3 expert agents, each providing
    superposition outputs (probability distributions) over market actions.
    
    Key Features:
    - State machine for correct agent sequencing
    - Agent-specific feature extraction from 48x13 matrix
    - Integration with SynergyDetector for pattern recognition
    - Superposition output storage and aggregation trigger
    """
    
    metadata = {
        "name": "strategic_market_v1",
        "is_parallelizable": False,
        "render_modes": ["human", "rgb_array"],
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 matrix_assembler=None, synergy_detector=None, kernel=None):
        """
        Initialize Strategic Market Environment
        
        Args:
            config: Configuration dictionary with environment parameters
            matrix_assembler: Optional matrix assembler instance
            synergy_detector: Optional synergy detector instance
            kernel: Optional kernel instance
        """
        super().__init__()
        
        self.config = config or self._default_config()
        
        # Store provided components
        self._provided_matrix_assembler = matrix_assembler
        self._provided_synergy_detector = synergy_detector
        self._provided_kernel = kernel
        
        # Define agents - only the 3 expert agents
        self.possible_agents = ["mlmi_expert", "nwrqk_expert", "regime_expert"]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        # Initialize components
        self._init_components()
        
        # Define action and observation spaces
        self._action_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Agent-specific observation dimensions
        obs_dims = {
            "mlmi_expert": 4,    # Features [0, 1, 9, 10]
            "nwrqk_expert": 4,   # Features [2, 3, 4, 5]
            "regime_expert": 3,  # Features [10, 11, 12]
        }
        
        self._observation_spaces = {
            agent: spaces.Dict({
                "features": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dims[agent],), dtype=np.float32
                ),
                "shared_context": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                ),
                "synergy_active": spaces.Discrete(2),
                "synergy_type": spaces.Discrete(5),  # None + 4 types
            })
            for agent in self.possible_agents
        }
        
        # State variables
        self.state = EnvironmentState.AWAITING_MLMI
        self.agent_outputs = {}  # Store superposition outputs
        self.current_matrix = None
        self.synergy_info = None
        self.market_context = None
        
        # Episode tracking
        self.timestep = 0
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        
        # Performance tracking
        self.inference_times = []
        
    @property
    def observation_spaces(self):
        """PettingZoo compatibility - return observation spaces dict"""
        return self._observation_spaces
    
    @property
    def action_spaces(self):
        """PettingZoo compatibility - return action spaces dict"""
        return self._action_spaces
    
    def observation_space(self, agent):
        """Return observation space for specific agent"""
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        """Return action space for specific agent"""
        return self._action_spaces[agent]
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration parameters"""
        return {
            "matrix_shape": [48, 13],
            "time_window": 10,
            "cooldown_bars": 5,
            "max_timesteps": 1000,
            "feature_indices": {
                "mlmi_expert": [0, 1, 9, 10],
                "nwrqk_expert": [2, 3, 4, 5],
                "regime_expert": [10, 11, 12],
            }
        }
    
    def _init_components(self):
        """Initialize integrated components"""
        # Use provided components or defaults
        self.kernel = self._provided_kernel
        self.matrix_assembler = self._provided_matrix_assembler
        self.synergy_detector = self._provided_synergy_detector
        
        # If no matrix assembler provided and the real one is available, create it
        if self.matrix_assembler is None and MatrixAssembler30mEnhanced is not None:
            try:
                from src.utils.feature_store_minimal import MinimalFeatureStore
                feature_store = MinimalFeatureStore()
                self.matrix_assembler = MatrixAssembler30mEnhanced(
                    config=self.config.get('matrix', {}),
                    feature_store=feature_store
                )
            except Exception as e:
                logger.warning(f"Could not create matrix assembler: {e}")
                self.matrix_assembler = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Reset agents
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # Reset state machine
        self.state = EnvironmentState.AWAITING_MLMI
        self.agent_outputs = {}
        
        # Generate initial observation
        self.current_matrix = self._generate_initial_matrix()
        self.synergy_info = self._detect_synergy(self.current_matrix)
        self.market_context = self._get_market_context()
        
        # Reset tracking variables
        self.timestep = 0
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Clear performance tracking
        self.inference_times = []
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
    
    def step(self, action: np.ndarray):
        """
        Process agent action and advance environment state
        
        Enhanced to handle superposition outputs and trigger aggregation
        
        Args:
            action: Probability distribution [p_bullish, p_neutral, p_bearish]
        """
        start_time = time.time()
        
        # Validate action is probability distribution
        if not isinstance(action, np.ndarray) or action.shape != (3,):
            raise ValueError(f"Action must be numpy array of shape (3,), got {type(action)} {action.shape}")
        
        # Normalize to ensure valid probability distribution
        action = action / (action.sum() + 1e-8)
        
        # Store current agent's output
        current_agent = self.agent_selection
        self.agent_outputs[current_agent] = action
        
        # Log agent action
        self.infos[current_agent]["action"] = action.tolist()
        self.infos[current_agent]["confidence"] = float(action.max())
        self.infos[current_agent]["entropy"] = float(-np.sum(action * np.log(action + 1e-8)))
        
        # Advance state machine
        self._advance_state()
        
        # Check if all agents have acted
        if self.state == EnvironmentState.READY_FOR_AGGREGATION:
            # Trigger aggregation and reward calculation
            self._perform_aggregation()
            
            # Reset for next cycle
            self.state = EnvironmentState.AWAITING_MLMI
            self.agent_outputs = {}
            self.timestep += 1
            
            # Check termination conditions
            if self.timestep >= self.config["max_timesteps"]:
                self.terminations = {agent: True for agent in self.agents}
        
        # Select next agent
        self._agent_selector.next()
        self.agent_selection = self._agent_selector.next()
        
        # Track inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # Accumulate rewards (will be set during aggregation)
        self._cumulative_rewards[current_agent] = self.rewards.get(current_agent, 0)
    
    def last(self, observe=True):
        """Return last agent's observation, reward, termination, truncation, and info"""
        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self._cumulative_rewards.get(agent, 0),
            self.terminations.get(agent, False),
            self.truncations.get(agent, False),
            self.infos.get(agent, {})
        )
    
    def observe(self, agent: str) -> Dict[str, Any]:
        """
        Get observation for specific agent
        
        Returns agent-specific features extracted from the matrix
        
        Args:
            agent: Agent identifier
            
        Returns:
            Dictionary containing features, shared context, and synergy info
        """
        if self.current_matrix is None:
            return self._get_empty_observation(agent)
        
        # Extract agent-specific features
        feature_indices = self.config["feature_indices"][agent]
        
        # Average across time dimension (48 bars)
        matrix_avg = np.mean(self.current_matrix, axis=0)
        agent_features = matrix_avg[feature_indices]
        
        # Build shared context
        shared_context = self._build_shared_context()
        
        # Build observation
        observation = {
            "features": agent_features.astype(np.float32),
            "shared_context": shared_context.astype(np.float32),
            "synergy_active": 1 if self.synergy_info is not None else 0,
            "synergy_type": self._encode_synergy_type(self.synergy_info),
        }
        
        return observation
    
    def render(self):
        """Render environment state (optional)"""
        if self.render_mode == "human":
            print(f"\n=== Strategic Market Environment ===")
            print(f"State: {self.state.value}")
            print(f"Current Agent: {self.agent_selection}")
            print(f"Timestep: {self.timestep}")
            
            if self.agent_outputs:
                print(f"\nAgent Outputs:")
                for agent, output in self.agent_outputs.items():
                    print(f"  {agent}: {output}")
            
            if self.synergy_info:
                print(f"\nSynergy: {self.synergy_info.get('type', 'None')}")
            
            if self.inference_times:
                avg_time = np.mean(self.inference_times)
                print(f"\nAvg Inference Time: {avg_time:.2f}ms")
    
    # Private helper methods
    
    def _advance_state(self):
        """Advance state machine based on current state"""
        state_transitions = {
            EnvironmentState.AWAITING_MLMI: EnvironmentState.AWAITING_NWRQK,
            EnvironmentState.AWAITING_NWRQK: EnvironmentState.AWAITING_REGIME,
            EnvironmentState.AWAITING_REGIME: EnvironmentState.READY_FOR_AGGREGATION,
        }
        
        if self.state in state_transitions:
            self.state = state_transitions[self.state]
    
    def _perform_aggregation(self):
        """
        Perform decision aggregation after all agents have acted
        
        This is where the DecisionAggregator and RewardSystem would be called
        For now, using placeholder logic
        """
        # Placeholder for aggregation logic
        # In full implementation, this would call DecisionAggregator
        
        # Calculate ensemble decision (simple average for now)
        ensemble_probs = np.mean(
            [self.agent_outputs[agent] for agent in self.possible_agents], 
            axis=0
        )
        
        # Store aggregated decision info
        for agent in self.agents:
            self.infos[agent]["ensemble_decision"] = ensemble_probs.tolist()
            self.infos[agent]["aggregation_complete"] = True
        
        # Placeholder reward calculation
        # In full implementation, this would call RewardSystem
        base_reward = np.random.normal(0, 1)  # Placeholder
        
        for agent in self.agents:
            self.rewards[agent] = base_reward
    
    def _generate_initial_matrix(self) -> np.ndarray:
        """Generate initial 48x13 feature matrix"""
        # In production, this would come from MatrixAssembler
        # For now, using synthetic data
        return np.random.randn(48, 13).astype(np.float32)
    
    def _detect_synergy(self, matrix: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect synergy patterns in matrix"""
        # In production, this would use SynergyDetector
        # For now, returning mock synergy with 20% probability
        if np.random.random() < 0.2:
            return {
                "type": f"TYPE_{np.random.randint(1, 5)}",
                "confidence": np.random.uniform(0.6, 0.9),
                "direction": np.random.choice([-1, 1]),
            }
        return None
    
    def _get_market_context(self) -> Dict[str, float]:
        """Get current market context"""
        return {
            "volatility_30": np.random.uniform(0.5, 2.0),
            "volume_profile_skew": np.random.uniform(-1, 1),
        }
    
    def _build_shared_context(self) -> np.ndarray:
        """Build shared context features for all agents"""
        context = np.zeros(6, dtype=np.float32)
        
        # Synergy features
        if self.synergy_info:
            context[0] = self.synergy_info.get("confidence", 0.0)
            context[1] = float(self.synergy_info.get("direction", 0))
        
        # Market features
        context[2] = np.log(self.market_context.get("volatility_30", 1.0))
        
        # Time features (simplified)
        hour = (self.timestep % 24) / 24.0
        context[3] = np.sin(2 * np.pi * hour)  # Hour sine
        context[4] = np.cos(2 * np.pi * hour)  # Hour cosine
        context[5] = (self.timestep % 5) / 4.0  # Day of week normalized
        
        return context
    
    def _encode_synergy_type(self, synergy_info: Optional[Dict]) -> int:
        """Encode synergy type as integer"""
        if synergy_info is None:
            return 0
        
        synergy_type = synergy_info.get("type", "")
        if synergy_type.startswith("TYPE_"):
            return int(synergy_type.split("_")[1])
        return 0
    
    def _get_empty_observation(self, agent: str) -> Dict[str, Any]:
        """Get empty observation for initialization"""
        obs_dims = {
            "mlmi_expert": 4,
            "nwrqk_expert": 4,
            "regime_expert": 3,
        }
        
        return {
            "features": np.zeros(obs_dims[agent], dtype=np.float32),
            "shared_context": np.zeros(6, dtype=np.float32),
            "synergy_active": 0,
            "synergy_type": 0,
        }


# Wrapper functions for PettingZoo compatibility
def env(**kwargs):
    """Create unwrapped environment"""
    env = StrategicMarketEnv(**kwargs)
    return env


def raw_env(**kwargs):
    """Create raw environment"""
    return env(**kwargs)