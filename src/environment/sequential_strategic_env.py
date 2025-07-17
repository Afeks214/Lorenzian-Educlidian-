"""
Sequential Strategic MARL Environment - PettingZoo AECEnv Implementation

This module implements the Sequential Strategic MARL Environment where agents execute
in sequence (MLMI → NWRQK → Regime) with each agent producing superpositions and
receiving enriched observations from predecessors.

Key Features:
- Sequential execution with defined agent order
- Superposition output at each step
- Enriched observations using UniversalObservationEnricher
- Perfect PettingZoo AECEnv compliance
- <5ms per agent performance target
- Comprehensive performance monitoring
- Mathematical validation of superposition properties

Agent Sequence: MLMI → NWRQK → Regime → Final Ensemble
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import torch
from collections import deque, OrderedDict

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


@dataclass
class SuperpositionState:
    """Superposition state from an agent"""
    agent_name: str
    action_probabilities: np.ndarray
    confidence: float
    feature_importance: Dict[str, float]
    internal_state: Dict[str, Any]
    computation_time_ms: float
    timestamp: datetime
    superposition_features: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class EnrichedObservation:
    """Enriched observation containing predecessor context"""
    base_observation: Dict[str, Any]
    predecessor_superpositions: List[SuperpositionState]
    enriched_features: Dict[str, Any]
    sequence_position: int
    total_agents: int


class SequentialPhase(Enum):
    """Sequential execution phases"""
    INITIALIZATION = "initialization"
    MLMI_EXECUTION = "mlmi_execution"
    NWRQK_EXECUTION = "nwrqk_execution"
    REGIME_EXECUTION = "regime_execution"
    ENSEMBLE_AGGREGATION = "ensemble_aggregation"
    REWARD_CALCULATION = "reward_calculation"
    EPISODE_END = "episode_end"


class UniversalObservationEnricher:
    """Universal observation enricher for sequential agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ObservationEnricher")
        
    def enrich_observation(
        self,
        base_observation: Dict[str, Any],
        predecessor_superpositions: List[SuperpositionState],
        sequence_position: int,
        total_agents: int
    ) -> EnrichedObservation:
        """
        Enrich observation with predecessor context
        
        Args:
            base_observation: Base observation for current agent
            predecessor_superpositions: List of predecessor superpositions
            sequence_position: Current position in sequence (0-based)
            total_agents: Total number of agents in sequence
            
        Returns:
            EnrichedObservation with additional context
        """
        enriched_features = {}
        
        # Add sequence context
        enriched_features["sequence_position"] = sequence_position
        enriched_features["total_agents"] = total_agents
        enriched_features["completion_ratio"] = sequence_position / max(1, total_agents)
        
        # Add predecessor information
        if predecessor_superpositions:
            # Aggregate predecessor confidences
            predecessor_confidences = [s.confidence for s in predecessor_superpositions]
            enriched_features["predecessor_avg_confidence"] = np.mean(predecessor_confidences)
            enriched_features["predecessor_max_confidence"] = np.max(predecessor_confidences)
            enriched_features["predecessor_min_confidence"] = np.min(predecessor_confidences)
            
            # Aggregate predecessor action probabilities
            predecessor_actions = [s.action_probabilities for s in predecessor_superpositions]
            enriched_features["predecessor_avg_action"] = np.mean(predecessor_actions, axis=0)
            enriched_features["predecessor_action_variance"] = np.var(predecessor_actions, axis=0)
            
            # Add computation time context
            predecessor_times = [s.computation_time_ms for s in predecessor_superpositions]
            enriched_features["predecessor_avg_computation_time"] = np.mean(predecessor_times)
            
            # Add feature importance aggregation
            all_features = {}
            for superposition in predecessor_superpositions:
                for feature, importance in superposition.feature_importance.items():
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
            
            enriched_features["predecessor_feature_importance"] = {
                f: np.mean(values) for f, values in all_features.items()
            }
        else:
            # No predecessors (first agent)
            enriched_features["predecessor_avg_confidence"] = 0.5
            enriched_features["predecessor_max_confidence"] = 0.5
            enriched_features["predecessor_min_confidence"] = 0.5
            enriched_features["predecessor_avg_action"] = np.array([0.33, 0.34, 0.33])
            enriched_features["predecessor_action_variance"] = np.array([0.0, 0.0, 0.0])
            enriched_features["predecessor_avg_computation_time"] = 0.0
            enriched_features["predecessor_feature_importance"] = {}
        
        return EnrichedObservation(
            base_observation=base_observation,
            predecessor_superpositions=predecessor_superpositions,
            enriched_features=enriched_features,
            sequence_position=sequence_position,
            total_agents=total_agents
        )


@dataclass
class SequentialEnvironmentState:
    """Sequential environment state tracking"""
    phase: SequentialPhase
    current_agent_index: int
    episode_step: int
    episode_reward: float
    agent_superpositions: OrderedDict[str, SuperpositionState]
    matrix_data: Optional[np.ndarray]
    shared_context: Optional[Dict[str, Any]]
    timestamp: datetime
    sequence_start_time: float
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class SequentialStrategicEnvironment(AECEnv):
    """
    Sequential Strategic MARL Environment with PettingZoo AECEnv compliance
    
    This environment executes agents in sequence: MLMI → NWRQK → Regime
    Each agent produces superpositions and receives enriched observations from predecessors.
    
    Key Features:
    - Sequential execution with defined order
    - Superposition output at each step
    - Enriched observations with predecessor context
    - Perfect PettingZoo AECEnv compliance
    - <5ms per agent performance target
    - Comprehensive performance monitoring
    """
    
    metadata = {
        "name": "sequential_strategic_marl_v1",
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
        Initialize Sequential Strategic MARL Environment
        
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
        self.name = "SequentialStrategicEnvironment"
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # Sequential agent configuration - ENFORCED ORDER
        self.possible_agents = ["mlmi_expert", "nwrqk_expert", "regime_expert"]
        self.agent_sequence = OrderedDict([
            ("mlmi_expert", 0),
            ("nwrqk_expert", 1), 
            ("regime_expert", 2)
        ])
        
        # Feature indices for each agent
        self.feature_indices = self.config.get("feature_indices", {
            "mlmi_expert": [0, 1, 9, 10],      # MLMI features
            "nwrqk_expert": [2, 3, 4, 5],      # NWRQK features  
            "regime_expert": [10, 11, 12]      # Regime features
        })
        
        # Initialize components
        self.matrix_assembler = matrix_assembler
        self.synergy_detector = synergy_detector
        self.strategic_component = None
        self.observation_enricher = UniversalObservationEnricher(self.config)
        
        # Initialize agents
        self.agents_dict = {}
        self._initialize_agents()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Sequential environment state
        self.env_state = SequentialEnvironmentState(
            phase=SequentialPhase.INITIALIZATION,
            current_agent_index=0,
            episode_step=0,
            episode_reward=0.0,
            agent_superpositions=OrderedDict(),
            matrix_data=None,
            shared_context=None,
            timestamp=datetime.now(),
            sequence_start_time=time.time()
        )
        
        # Episode management
        self.max_episode_steps = self.config.get("max_episode_steps", 1000)
        self.episode_count = 0
        self.total_episodes = self.config.get("total_episodes", 10000)
        
        # Performance tracking
        self.performance_metrics = {
            "episode_rewards": deque(maxlen=100),
            "episode_lengths": deque(maxlen=100),
            "agent_computation_times": deque(maxlen=1000),
            "sequence_execution_times": deque(maxlen=1000),
            "ensemble_confidences": deque(maxlen=1000),
            "superposition_qualities": deque(maxlen=1000),
            "successful_episodes": 0,
            "failed_episodes": 0,
            "per_agent_performance": {
                agent: {
                    "computation_times": deque(maxlen=1000),
                    "confidences": deque(maxlen=1000),
                    "success_rate": 0.0
                } for agent in self.possible_agents
            }
        }
        
        # Performance targets
        self.performance_targets = {
            "max_agent_computation_time_ms": 5.0,
            "max_sequence_execution_time_ms": 15.0,
            "min_ensemble_confidence": 0.6,
            "min_superposition_quality": 0.7
        }
        
        # Reward calculation
        self.reward_scale = self.config.get("reward_scale", 1.0)
        self.confidence_bonus = self.config.get("confidence_bonus", 0.1)
        self.sequence_bonus = self.config.get("sequence_bonus", 0.05)
        
        self.logger.info(f"SequentialStrategicEnvironment initialized with {len(self.possible_agents)} agents")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "matrix_shape": [48, 13],
            "max_episode_steps": 1000,
            "total_episodes": 10000,
            "reward_scale": 1.0,
            "confidence_bonus": 0.1,
            "sequence_bonus": 0.05,
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
                "market_conditions": ["normal", "volatile", "trending"],
                "sequential_execution": True,
                "superposition_enabled": True,
                "observation_enrichment": True
            },
            "performance": {
                "max_agent_computation_time_ms": 5.0,
                "max_sequence_execution_time_ms": 15.0,
                "min_ensemble_confidence": 0.6,
                "min_superposition_quality": 0.7
            }
        }
    
    def _initialize_agents(self):
        """Initialize sequential strategic agents"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Initialize agents in sequential order
            for agent_name in self.possible_agents:
                if agent_name == "mlmi_expert":
                    self.agents_dict[agent_name] = MLMIStrategicAgent(
                        config=self.config, 
                        device=device
                    )
                elif agent_name == "nwrqk_expert":
                    self.agents_dict[agent_name] = NWRQKStrategicAgent(
                        config=self.config,
                        device=device
                    )
                elif agent_name == "regime_expert":
                    self.agents_dict[agent_name] = RegimeDetectionAgent(
                        config=self.config,
                        device=device
                    )
            
            self.logger.info("All sequential strategic agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sequential agents: {e}")
            raise
    
    def _setup_spaces(self):
        """Setup action and observation spaces for sequential execution"""
        # Action space: probability distribution over [buy, hold, sell]
        self._action_spaces = {
            agent: spaces.Box(
                low=0.0, high=1.0, shape=(3,), dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        # Observation spaces: agent-specific features + enriched context
        self._observation_spaces = {}
        
        for agent in self.possible_agents:
            agent_feature_dim = len(self.feature_indices[agent])
            sequence_position = self.agent_sequence[agent]
            
            # Base observation space
            base_obs_space = spaces.Dict({
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
                    "phase": spaces.Discrete(len(SequentialPhase)),
                    "agent_index": spaces.Discrete(len(self.possible_agents))
                })
            })
            
            # Enriched observation space (includes predecessor context)
            enriched_obs_space = spaces.Dict({
                "base_observation": base_obs_space,
                "enriched_features": spaces.Dict({
                    "sequence_position": spaces.Discrete(len(self.possible_agents)),
                    "total_agents": spaces.Discrete(len(self.possible_agents) + 1),
                    "completion_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "predecessor_avg_confidence": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "predecessor_max_confidence": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "predecessor_min_confidence": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "predecessor_avg_action": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                    "predecessor_action_variance": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                    "predecessor_avg_computation_time": spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32)
                }),
                "predecessor_superpositions": spaces.Sequence(
                    spaces.Dict({
                        "agent_name": spaces.Text(256),
                        "action_probabilities": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                        "confidence": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                        "computation_time_ms": spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32)
                    })
                )
            })
            
            self._observation_spaces[agent] = enriched_obs_space
    
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
        
        # Reset sequential environment state
        self.env_state = SequentialEnvironmentState(
            phase=SequentialPhase.INITIALIZATION,
            current_agent_index=0,
            episode_step=0,
            episode_reward=0.0,
            agent_superpositions=OrderedDict(),
            matrix_data=None,
            shared_context=None,
            timestamp=datetime.now(),
            sequence_start_time=time.time()
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
        
        # Start with first agent (MLMI)
        self.env_state.phase = SequentialPhase.MLMI_EXECUTION
        self.env_state.current_agent_index = 0
        self.agent_selection = self.possible_agents[0]
        
        self.logger.info(f"Sequential environment reset for episode {self.episode_count}")
    
    def step(self, action: Union[np.ndarray, Dict]):
        """Execute one step of the sequential environment"""
        if self.env_state.phase == SequentialPhase.EPISODE_END:
            return  # Episode already ended
        
        start_time = time.time()
        current_agent = self.agent_selection
        
        # Validate and process action
        action = self._validate_action(action)
        
        # Execute agent decision and create superposition
        superposition = self._execute_agent_decision(current_agent, action)
        self.env_state.agent_superpositions[current_agent] = superposition
        
        # Update agent performance metrics
        agent_perf = self.performance_metrics["per_agent_performance"][current_agent]
        agent_perf["computation_times"].append(superposition.computation_time_ms)
        agent_perf["confidences"].append(superposition.confidence)
        
        # Update agent info
        self.infos[current_agent] = {
            "agent_name": current_agent,
            "action_probabilities": action.tolist(),
            "confidence": superposition.confidence,
            "computation_time_ms": superposition.computation_time_ms,
            "feature_importance": superposition.feature_importance,
            "sequence_position": self.agent_sequence[current_agent],
            "superposition_created": True
        }
        
        # Check performance targets
        if superposition.computation_time_ms > self.performance_targets["max_agent_computation_time_ms"]:
            self.logger.warning(
                f"Agent {current_agent} exceeded computation time target: "
                f"{superposition.computation_time_ms:.2f}ms > {self.performance_targets['max_agent_computation_time_ms']:.2f}ms"
            )
        
        # Advance to next agent or finish sequence
        self._advance_sequential_execution()
        
        # Update performance metrics
        execution_time = (time.time() - start_time) * 1000
        self.performance_metrics["agent_computation_times"].append(execution_time)
        
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
    
    def _execute_agent_decision(self, agent_name: str, action: np.ndarray) -> SuperpositionState:
        """Execute agent decision and return superposition state"""
        start_time = time.time()
        
        try:
            agent = self.agents_dict[agent_name]
            
            # Create base prediction
            confidence = float(np.max(action))
            
            # Get feature importance
            feature_importance = {}
            features = self.feature_indices[agent_name]
            for i, feature_idx in enumerate(features):
                feature_importance[f"feature_{feature_idx}"] = float(1.0 / len(features))
            
            # Create superposition features (enhanced with quantum-inspired properties)
            superposition_features = self._create_superposition_features(
                agent_name, action, confidence, self.env_state.matrix_data
            )
            
            computation_time_ms = (time.time() - start_time) * 1000
            
            return SuperpositionState(
                agent_name=agent_name,
                action_probabilities=action,
                confidence=confidence,
                feature_importance=feature_importance,
                internal_state={
                    "agent_name": agent_name,
                    "episode_step": self.env_state.episode_step,
                    "phase": self.env_state.phase.value,
                    "sequence_position": self.agent_sequence[agent_name]
                },
                computation_time_ms=computation_time_ms,
                timestamp=datetime.now(),
                superposition_features=superposition_features
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute agent {agent_name} decision: {e}")
            computation_time_ms = (time.time() - start_time) * 1000
            
            # Return fallback superposition
            return SuperpositionState(
                agent_name=agent_name,
                action_probabilities=np.array([0.33, 0.34, 0.33]),
                confidence=0.5,
                feature_importance={'fallback': 1.0},
                internal_state={'fallback_mode': True},
                computation_time_ms=computation_time_ms,
                timestamp=datetime.now(),
                superposition_features={}
            )
    
    def _create_superposition_features(
        self, 
        agent_name: str, 
        action: np.ndarray, 
        confidence: float, 
        matrix_data: np.ndarray
    ) -> Dict[str, Any]:
        """Create superposition features for agent output"""
        superposition_features = {}
        
        # Quantum-inspired superposition properties
        superposition_features["quantum_coherence"] = float(np.sqrt(np.sum(action ** 2)))
        superposition_features["entanglement_measure"] = float(confidence * np.std(action))
        superposition_features["phase_information"] = float(np.arctan2(action[1], action[0]))
        
        # Agent-specific superposition features
        if agent_name == "mlmi_expert":
            superposition_features["liquidity_superposition"] = float(np.mean(matrix_data[:, 0]))
            superposition_features["impact_superposition"] = float(np.mean(matrix_data[:, 1]))
        elif agent_name == "nwrqk_expert":
            superposition_features["network_superposition"] = float(np.mean(matrix_data[:, 2]))
            superposition_features["quality_superposition"] = float(np.mean(matrix_data[:, 3]))
        elif agent_name == "regime_expert":
            superposition_features["regime_superposition"] = float(np.mean(matrix_data[:, 10]))
            superposition_features["volatility_superposition"] = float(np.mean(matrix_data[:, 11]))
        
        # Temporal superposition features
        superposition_features["temporal_coherence"] = float(confidence)
        superposition_features["stability_measure"] = float(1.0 - np.var(action))
        
        return superposition_features
    
    def _advance_sequential_execution(self):
        """Advance to next agent in sequence or finish sequence"""
        current_agent = self.agent_selection
        current_index = self.agent_sequence[current_agent]
        
        # Determine next phase based on current agent
        if current_agent == "mlmi_expert":
            self.env_state.phase = SequentialPhase.NWRQK_EXECUTION
            self.env_state.current_agent_index = 1
            self.agent_selection = "nwrqk_expert"
        elif current_agent == "nwrqk_expert":
            self.env_state.phase = SequentialPhase.REGIME_EXECUTION
            self.env_state.current_agent_index = 2
            self.agent_selection = "regime_expert"
        elif current_agent == "regime_expert":
            # All agents have executed - move to ensemble aggregation
            self.env_state.phase = SequentialPhase.ENSEMBLE_AGGREGATION
            self._process_ensemble_aggregation()
            self._calculate_sequential_rewards()
            self._update_episode_state()
        
        # Update sequence timing
        sequence_time = (time.time() - self.env_state.sequence_start_time) * 1000
        if len(self.env_state.agent_superpositions) == len(self.possible_agents):
            self.performance_metrics["sequence_execution_times"].append(sequence_time)
            
            # Check sequence performance target
            if sequence_time > self.performance_targets["max_sequence_execution_time_ms"]:
                self.logger.warning(
                    f"Sequence execution exceeded time target: "
                    f"{sequence_time:.2f}ms > {self.performance_targets['max_sequence_execution_time_ms']:.2f}ms"
                )
    
    def _process_ensemble_aggregation(self):
        """Process ensemble aggregation from all agent superpositions"""
        if len(self.env_state.agent_superpositions) != len(self.possible_agents):
            self.logger.error("Incomplete agent superpositions for ensemble aggregation")
            return
        
        # Aggregate superpositions in sequence order
        superposition_list = list(self.env_state.agent_superpositions.values())
        
        # Calculate ensemble probabilities (weighted by confidence and sequence position)
        ensemble_probs = np.zeros(3)
        total_weight = 0.0
        
        for i, superposition in enumerate(superposition_list):
            # Weight by confidence and sequence position (later agents get higher weight)
            position_weight = (i + 1) / len(superposition_list)
            confidence_weight = superposition.confidence
            combined_weight = position_weight * confidence_weight
            
            ensemble_probs += combined_weight * superposition.action_probabilities
            total_weight += combined_weight
        
        # Normalize ensemble probabilities
        if total_weight > 0:
            ensemble_probs = ensemble_probs / total_weight
        else:
            ensemble_probs = np.array([0.33, 0.34, 0.33])
        
        # Calculate ensemble confidence (weighted average)
        ensemble_confidence = np.mean([s.confidence for s in superposition_list])
        
        # Create final ensemble superposition
        ensemble_superposition = {
            "probabilities": ensemble_probs.tolist(),
            "confidence": ensemble_confidence,
            "action": ["buy", "hold", "sell"][np.argmax(ensemble_probs)],
            "superposition_quality": self._calculate_superposition_quality(superposition_list),
            "sequence_coherence": self._calculate_sequence_coherence(superposition_list)
        }
        
        # Store ensemble decision
        self.env_state.shared_context["ensemble_decision"] = ensemble_superposition
        
        # Update performance metrics
        self.performance_metrics["ensemble_confidences"].append(ensemble_confidence)
        self.performance_metrics["superposition_qualities"].append(ensemble_superposition["superposition_quality"])
        
        self.logger.info(f"Ensemble aggregation complete: {ensemble_superposition['action']} (conf: {ensemble_confidence:.3f})")
    
    def _calculate_superposition_quality(self, superposition_list: List[SuperpositionState]) -> float:
        """Calculate quality of superposition ensemble"""
        if not superposition_list:
            return 0.0
        
        # Quality based on coherence and confidence
        confidences = [s.confidence for s in superposition_list]
        avg_confidence = np.mean(confidences)
        confidence_stability = 1.0 - np.var(confidences)
        
        # Temporal coherence (how well agents agree)
        actions = [s.action_probabilities for s in superposition_list]
        action_coherence = 1.0 - np.mean([np.var(actions, axis=0)])
        
        # Combined quality score
        quality = (avg_confidence + confidence_stability + action_coherence) / 3.0
        
        return float(np.clip(quality, 0.0, 1.0))
    
    def _calculate_sequence_coherence(self, superposition_list: List[SuperpositionState]) -> float:
        """Calculate coherence of sequential execution"""
        if len(superposition_list) < 2:
            return 1.0
        
        # Measure how well subsequent agents build on previous ones
        coherence_scores = []
        
        for i in range(1, len(superposition_list)):
            prev_action = superposition_list[i-1].action_probabilities
            curr_action = superposition_list[i].action_probabilities
            
            # Calculate alignment (using dot product)
            alignment = np.dot(prev_action, curr_action)
            coherence_scores.append(alignment)
        
        return float(np.mean(coherence_scores))
    
    def _calculate_sequential_rewards(self):
        """Calculate rewards for sequential execution"""
        ensemble_decision = self.env_state.shared_context["ensemble_decision"]
        base_reward = self._get_market_reward(ensemble_decision)
        
        # Calculate individual agent rewards
        for agent_name in self.possible_agents:
            superposition = self.env_state.agent_superpositions[agent_name]
            
            # Base reward scaled by confidence
            reward = base_reward * superposition.confidence * self.reward_scale
            
            # Confidence bonus
            if superposition.confidence > self.config["confidence_threshold"]:
                reward += self.confidence_bonus
            
            # Sequence position bonus (later agents get bonus for building on predecessors)
            sequence_position = self.agent_sequence[agent_name]
            sequence_bonus = self.sequence_bonus * (sequence_position + 1)
            reward += sequence_bonus
            
            # Superposition quality bonus
            superposition_quality = ensemble_decision["superposition_quality"]
            if superposition_quality > self.performance_targets["min_superposition_quality"]:
                reward += 0.05 * superposition_quality
            
            # Performance penalty for slow execution
            if superposition.computation_time_ms > self.performance_targets["max_agent_computation_time_ms"]:
                penalty = 0.01 * (superposition.computation_time_ms - self.performance_targets["max_agent_computation_time_ms"])
                reward -= penalty
            
            self.rewards[agent_name] = reward
        
        # Update episode reward
        self.env_state.episode_reward = sum(self.rewards.values())
        
        # Move to reward calculation phase
        self.env_state.phase = SequentialPhase.REWARD_CALCULATION
    
    def _get_market_reward(self, ensemble_decision: Dict) -> float:
        """Calculate market-based reward"""
        confidence = ensemble_decision["confidence"]
        superposition_quality = ensemble_decision["superposition_quality"]
        sequence_coherence = ensemble_decision["sequence_coherence"]
        
        # Simulated market performance based on ensemble quality
        market_performance = np.random.normal(0, 0.1)
        
        # Reward is based on confidence, quality, and coherence
        reward = confidence * superposition_quality * sequence_coherence * market_performance
        
        return reward
    
    def _update_episode_state(self):
        """Update episode state after processing all agents"""
        self.env_state.episode_step += 1
        self.env_state.timestamp = datetime.now()
        
        # Clear agent superpositions for next step
        self.env_state.agent_superpositions.clear()
        
        # Generate new market data
        self.env_state.matrix_data = self._generate_market_data()
        self.env_state.shared_context = self._extract_shared_context(self.env_state.matrix_data)
        
        # Reset for next sequence
        self.env_state.phase = SequentialPhase.MLMI_EXECUTION
        self.env_state.current_agent_index = 0
        self.env_state.sequence_start_time = time.time()
        self.agent_selection = self.possible_agents[0]
    
    def _check_termination_conditions(self):
        """Check if episode should terminate"""
        # Episode step limit
        if self.env_state.episode_step >= self.max_episode_steps:
            self.terminations = {agent: True for agent in self.agents}
            self.env_state.phase = SequentialPhase.EPISODE_END
            self.performance_metrics["episode_lengths"].append(self.env_state.episode_step)
        
        # Maximum episodes reached
        if self.episode_count >= self.total_episodes:
            self.truncations = {agent: True for agent in self.agents}
            self.env_state.phase = SequentialPhase.EPISODE_END
        
        # Performance-based termination
        if self.env_state.episode_reward > self.config.get("reward_threshold", float('inf')):
            self.terminations = {agent: True for agent in self.agents}
            self.env_state.phase = SequentialPhase.EPISODE_END
            self.performance_metrics["successful_episodes"] += 1
    
    def observe(self, agent: str) -> Dict[str, Any]:
        """Get enriched observation for specific agent"""
        if self.env_state.matrix_data is None:
            return self._get_empty_observation(agent)
        
        # Get base observation
        base_observation = self._get_base_observation(agent)
        
        # Get predecessor superpositions
        predecessor_superpositions = self._get_predecessor_superpositions(agent)
        
        # Create enriched observation
        sequence_position = self.agent_sequence[agent]
        total_agents = len(self.possible_agents)
        
        enriched_obs = self.observation_enricher.enrich_observation(
            base_observation=base_observation,
            predecessor_superpositions=predecessor_superpositions,
            sequence_position=sequence_position,
            total_agents=total_agents
        )
        
        return self._format_observation_for_pettingzoo(enriched_obs)
    
    def _get_base_observation(self, agent: str) -> Dict[str, Any]:
        """Get base observation for agent"""
        # Extract agent-specific features
        agent_features = self._extract_agent_features(agent, self.env_state.matrix_data)
        
        return {
            "agent_features": agent_features,
            "shared_context": self._get_shared_context_vector(),
            "market_matrix": self.env_state.matrix_data.copy(),
            "episode_info": {
                "episode_step": self.env_state.episode_step,
                "phase": self.env_state.phase.value,
                "agent_index": self.env_state.current_agent_index
            }
        }
    
    def _get_predecessor_superpositions(self, agent: str) -> List[SuperpositionState]:
        """Get superpositions from predecessor agents"""
        predecessor_superpositions = []
        current_position = self.agent_sequence[agent]
        
        # Get all superpositions from agents that executed before current agent
        for agent_name, position in self.agent_sequence.items():
            if position < current_position and agent_name in self.env_state.agent_superpositions:
                predecessor_superpositions.append(self.env_state.agent_superpositions[agent_name])
        
        return predecessor_superpositions
    
    def _format_observation_for_pettingzoo(self, enriched_obs: EnrichedObservation) -> Dict[str, Any]:
        """Format enriched observation for PettingZoo compatibility"""
        # Convert predecessor superpositions to dict format
        predecessor_dicts = []
        for superposition in enriched_obs.predecessor_superpositions:
            predecessor_dicts.append({
                "agent_name": superposition.agent_name,
                "action_probabilities": superposition.action_probabilities,
                "confidence": np.array([superposition.confidence], dtype=np.float32),
                "computation_time_ms": np.array([superposition.computation_time_ms], dtype=np.float32)
            })
        
        # Convert enriched features to proper format
        enriched_features = {}
        for key, value in enriched_obs.enriched_features.items():
            if isinstance(value, (int, float)):
                enriched_features[key] = np.array([value], dtype=np.float32)
            else:
                enriched_features[key] = np.array(value, dtype=np.float32)
        
        return {
            "base_observation": enriched_obs.base_observation,
            "enriched_features": enriched_features,
            "predecessor_superpositions": predecessor_dicts
        }
    
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
        
        base_obs = {
            "agent_features": np.zeros(agent_feature_dim, dtype=np.float32),
            "shared_context": np.zeros(6, dtype=np.float32),
            "market_matrix": np.zeros((48, 13), dtype=np.float32),
            "episode_info": {
                "episode_step": 0,
                "phase": 0,
                "agent_index": 0
            }
        }
        
        enriched_features = {
            "sequence_position": np.array([0], dtype=np.float32),
            "total_agents": np.array([len(self.possible_agents)], dtype=np.float32),
            "completion_ratio": np.array([0.0], dtype=np.float32),
            "predecessor_avg_confidence": np.array([0.5], dtype=np.float32),
            "predecessor_max_confidence": np.array([0.5], dtype=np.float32),
            "predecessor_min_confidence": np.array([0.5], dtype=np.float32),
            "predecessor_avg_action": np.array([0.33, 0.34, 0.33], dtype=np.float32),
            "predecessor_action_variance": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "predecessor_avg_computation_time": np.array([0.0], dtype=np.float32)
        }
        
        return {
            "base_observation": base_obs,
            "enriched_features": enriched_features,
            "predecessor_superpositions": []
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
        print(f"\n=== Sequential Strategic MARL Environment ===")
        print(f"Episode: {self.episode_count}")
        print(f"Step: {self.env_state.episode_step}")
        print(f"Phase: {self.env_state.phase.value}")
        print(f"Current Agent: {self.agent_selection}")
        print(f"Episode Reward: {self.env_state.episode_reward:.4f}")
        
        print(f"\nSequence Progress:")
        for agent_name, position in self.agent_sequence.items():
            status = "✓" if agent_name in self.env_state.agent_superpositions else "⏳" if agent_name == self.agent_selection else "⏸"
            print(f"  {position + 1}. {agent_name}: {status}")
        
        if self.env_state.agent_superpositions:
            print(f"\nAgent Superpositions:")
            for agent_name, superposition in self.env_state.agent_superpositions.items():
                print(f"  {agent_name}: {superposition.action_probabilities} (conf: {superposition.confidence:.2f}, time: {superposition.computation_time_ms:.1f}ms)")
        
        if self.env_state.shared_context and "ensemble_decision" in self.env_state.shared_context:
            ensemble = self.env_state.shared_context["ensemble_decision"]
            print(f"\nEnsemble Decision: {ensemble['action']} (conf: {ensemble['confidence']:.2f}, quality: {ensemble['superposition_quality']:.2f})")
        
        # Performance metrics
        print(f"\nPerformance Metrics:")
        if self.performance_metrics["sequence_execution_times"]:
            avg_sequence_time = np.mean(list(self.performance_metrics["sequence_execution_times"])[-10:])
            print(f"  Avg Sequence Time: {avg_sequence_time:.2f}ms")
        
        if self.performance_metrics["ensemble_confidences"]:
            avg_ensemble_conf = np.mean(list(self.performance_metrics["ensemble_confidences"])[-10:])
            print(f"  Avg Ensemble Confidence: {avg_ensemble_conf:.3f}")
        
        print(f"  Successful Episodes: {self.performance_metrics['successful_episodes']}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array (placeholder)"""
        return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def _render_ansi(self) -> str:
        """Render environment as ANSI string"""
        output = []
        output.append("Sequential Strategic MARL Environment")
        output.append(f"Episode: {self.episode_count}, Step: {self.env_state.episode_step}")
        output.append(f"Phase: {self.env_state.phase.value}")
        output.append(f"Current Agent: {self.agent_selection}")
        output.append(f"Episode Reward: {self.env_state.episode_reward:.4f}")
        
        return "\n".join(output)
    
    def close(self):
        """Close environment and cleanup resources"""
        self.logger.info("Closing Sequential Strategic MARL Environment")
        
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
        
        self.logger.info("Sequential environment closed successfully")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            "episode_count": self.episode_count,
            "avg_episode_reward": np.mean(self.performance_metrics["episode_rewards"]) if self.performance_metrics["episode_rewards"] else 0.0,
            "avg_episode_length": np.mean(self.performance_metrics["episode_lengths"]) if self.performance_metrics["episode_lengths"] else 0.0,
            "avg_sequence_execution_time_ms": np.mean(self.performance_metrics["sequence_execution_times"]) if self.performance_metrics["sequence_execution_times"] else 0.0,
            "avg_ensemble_confidence": np.mean(self.performance_metrics["ensemble_confidences"]) if self.performance_metrics["ensemble_confidences"] else 0.0,
            "avg_superposition_quality": np.mean(self.performance_metrics["superposition_qualities"]) if self.performance_metrics["superposition_qualities"] else 0.0,
            "successful_episodes": self.performance_metrics["successful_episodes"],
            "failed_episodes": self.performance_metrics["failed_episodes"],
            "success_rate": self.performance_metrics["successful_episodes"] / max(1, self.episode_count)
        }
        
        # Add per-agent metrics
        for agent_name, agent_metrics in self.performance_metrics["per_agent_performance"].items():
            metrics[f"{agent_name}_avg_computation_time_ms"] = np.mean(agent_metrics["computation_times"]) if agent_metrics["computation_times"] else 0.0
            metrics[f"{agent_name}_avg_confidence"] = np.mean(agent_metrics["confidences"]) if agent_metrics["confidences"] else 0.0
        
        return metrics
    
    def get_mathematical_validation(self) -> Dict[str, Any]:
        """Get mathematical validation of superposition properties"""
        validation = {
            "superposition_coherence": [],
            "quantum_properties": [],
            "sequence_consistency": [],
            "mathematical_stability": True
        }
        
        # Validate recent superpositions
        if self.env_state.agent_superpositions:
            for agent_name, superposition in self.env_state.agent_superpositions.items():
                # Validate probability normalization
                prob_sum = np.sum(superposition.action_probabilities)
                validation["superposition_coherence"].append(abs(prob_sum - 1.0) < 1e-6)
                
                # Validate quantum properties
                quantum_coherence = superposition.superposition_features.get("quantum_coherence", 0.0)
                validation["quantum_properties"].append(0.0 <= quantum_coherence <= 1.0)
        
        # Overall mathematical stability
        validation["mathematical_stability"] = all(validation["superposition_coherence"]) and all(validation["quantum_properties"])
        
        return validation


# Wrapper functions for PettingZoo compatibility
def env(**kwargs):
    """Create unwrapped sequential strategic environment"""
    return SequentialStrategicEnvironment(**kwargs)


def raw_env(**kwargs):
    """Create raw sequential strategic environment"""
    return env(**kwargs)


def parallel_env(**kwargs):
    """Create parallel environment (not supported for sequential execution)"""
    raise NotImplementedError("Parallel execution not supported for Sequential Strategic MARL")


# Example usage and testing
if __name__ == "__main__":
    # Test sequential environment creation
    config = {
        "max_episode_steps": 50,
        "total_episodes": 5,
        "reward_scale": 1.0,
        "confidence_threshold": 0.6,
        "performance": {
            "max_agent_computation_time_ms": 5.0,
            "max_sequence_execution_time_ms": 15.0
        }
    }
    
    env = SequentialStrategicEnvironment(config=config)
    
    # Test reset
    env.reset()
    print(f"Environment reset. Current agent: {env.agent_selection}")
    
    # Test sequential execution
    for step in range(10):
        print(f"\n--- Step {step + 1} ---")
        
        # Get current agent
        current_agent = env.agent_selection
        print(f"Current agent: {current_agent}")
        
        # Test observation
        obs = env.observe(current_agent)
        print(f"Observation keys: {list(obs.keys())}")
        
        # Test action
        action = np.array([0.4, 0.4, 0.2])  # Buy bias
        env.step(action)
        
        # Render current state
        env.render()
        
        # Check if episode is done
        if env.env_state.phase == SequentialPhase.EPISODE_END:
            print("Episode ended")
            break
    
    # Get performance metrics
    metrics = env.get_performance_metrics()
    print(f"\nPerformance metrics: {metrics}")
    
    # Get mathematical validation
    validation = env.get_mathematical_validation()
    print(f"\nMathematical validation: {validation}")
    
    # Close environment
    env.close()
    print("\nSequential Strategic MARL Environment test completed")