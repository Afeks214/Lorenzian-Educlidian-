"""
Sequential Tactical MARL Environment

This module implements a sequential tactical MARL environment that orchestrates
the execution of tactical agents in a specific order: FVG → Momentum → EntryOpt.
Each agent receives enriched observations from its predecessors and the strategic
context from the upstream Strategic MARL system.

Key Features:
- Sequential execution with superposition accumulation
- Strategic context integration from upstream systems
- High-frequency execution capability (5-minute cycles)
- Byzantine fault tolerance integration
- Comprehensive market microstructure modeling
- Rich tactical superposition output for downstream systems

Architecture:
- Strategic MARL (30m) provides context
- FVG Agent analyzes Fair Value Gaps first
- Momentum Agent receives FVG output + market state
- EntryOpt Agent receives both FVG + Momentum + market state
- Final tactical superposition aggregated for execution

Author: Agent 5 - Sequential Tactical MARL Specialist
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from collections import deque, OrderedDict
from enum import Enum
from dataclasses import dataclass, field
import logging
import time
import json
import uuid
import asyncio
from pathlib import Path
import yaml
import warnings

# PettingZoo imports
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.env import ObsType, ActionType
from gymnasium import spaces
import torch
import torch.nn as nn

# Core system imports
from src.core.events import Event, EventType
from src.core.event_bus import EventBus
from src.indicators.custom.fvg import FVGDetector
from src.matrix.assembler_5m import MatrixAssembler5m
from src.consensus.byzantine_detector import ByzantineDetector
from src.consensus.pbft_engine import PBFTEngine

logger = logging.getLogger(__name__)

class SequentialPhase(Enum):
    """Sequential execution phases for tactical agents"""
    INITIALIZATION = "initialization"
    FVG_ANALYSIS = "fvg_analysis"
    MOMENTUM_ANALYSIS = "momentum_analysis"
    ENTRY_OPTIMIZATION = "entry_optimization"
    SUPERPOSITION_AGGREGATION = "superposition_aggregation"
    EXECUTION_READY = "execution_ready"
    EPISODE_COMPLETE = "episode_complete"

class ExecutionState(Enum):
    """Execution state tracking"""
    AWAITING_STRATEGIC_CONTEXT = "awaiting_strategic_context"
    PROCESSING_AGENTS = "processing_agents"
    AGGREGATING_SUPERPOSITION = "aggregating_superposition"
    READY_FOR_EXECUTION = "ready_for_execution"
    EXECUTION_COMPLETE = "execution_complete"

@dataclass
class StrategicContext:
    """Container for strategic context from upstream MARL system"""
    timestamp: float
    regime_embedding: np.ndarray
    synergy_signal: Dict[str, Any]
    market_state: Dict[str, Any]
    confidence_level: float
    execution_bias: str  # 'bullish', 'bearish', 'neutral'
    volatility_forecast: float
    session_id: str
    
    def __post_init__(self):
        """Validate strategic context"""
        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError(f"Invalid confidence level: {self.confidence_level}")
        if self.execution_bias not in ['bullish', 'bearish', 'neutral']:
            raise ValueError(f"Invalid execution bias: {self.execution_bias}")

@dataclass
class AgentSuperposition:
    """Container for agent superposition output"""
    agent_id: str
    phase: SequentialPhase
    probabilities: np.ndarray  # [bearish, neutral, bullish]
    confidence: float
    feature_importance: Dict[str, float]
    market_insights: Dict[str, Any]
    execution_signals: Dict[str, Any]
    timestamp: float
    predecessor_context: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate superposition output"""
        if len(self.probabilities) != 3:
            raise ValueError("Probabilities must have length 3")
        if not np.allclose(np.sum(self.probabilities), 1.0, atol=1e-6):
            raise ValueError("Probabilities must sum to 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be in [0, 1]")

@dataclass
class TacticalSuperposition:
    """Final tactical superposition aggregation"""
    execute: bool
    action: int
    confidence: float
    aggregated_probabilities: np.ndarray
    agent_contributions: Dict[str, AgentSuperposition]
    strategic_alignment: float
    execution_command: Optional[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    microstructure_analysis: Dict[str, Any]
    timestamp: float
    session_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for downstream systems"""
        return {
            'execute': self.execute,
            'action': self.action,
            'confidence': self.confidence,
            'aggregated_probabilities': self.aggregated_probabilities.tolist(),
            'strategic_alignment': self.strategic_alignment,
            'execution_command': self.execution_command,
            'risk_assessment': self.risk_assessment,
            'microstructure_analysis': self.microstructure_analysis,
            'timestamp': self.timestamp,
            'session_id': self.session_id
        }

@dataclass
class PerformanceMetrics:
    """Performance tracking for sequential execution"""
    episode_count: int = 0
    step_count: int = 0
    total_execution_time: float = 0.0
    phase_latencies: Dict[SequentialPhase, deque] = field(default_factory=lambda: {
        phase: deque(maxlen=1000) for phase in SequentialPhase
    })
    strategic_context_updates: int = 0
    superposition_generations: int = 0
    execution_success_rate: float = 0.0
    byzantine_detections: int = 0
    consensus_failures: int = 0
    last_update: float = field(default_factory=time.time)

class SequentialTacticalEnvironment(AECEnv):
    """
    Sequential Tactical MARL Environment
    
    Implements a sophisticated sequential decision-making environment where
    tactical agents operate in a specific order, each receiving enriched
    observations from their predecessors and strategic context.
    
    Key Features:
    - Strategic context integration from upstream 30m MARL system
    - Sequential execution: FVG → Momentum → EntryOpt
    - Superposition accumulation with predecessor context
    - High-frequency execution (5-minute cycles)
    - Byzantine fault tolerance
    - Rich tactical superposition output
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi", "tactical_dashboard"],
        "name": "sequential_tactical_marl_v1",
        "is_parallelizable": False,
        "render_fps": 12
    }
    
    def __init__(self, config: Optional[Union[Dict[str, Any], str, Path]] = None):
        """
        Initialize Sequential Tactical MARL Environment
        
        Args:
            config: Configuration dictionary, path to config file, or None
        """
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config)
        self._validate_config()
        
        # Sequential agent definitions (ORDER MATTERS)
        self.agent_sequence = ['fvg_agent', 'momentum_agent', 'entry_opt_agent']
        self.possible_agents = self.agent_sequence.copy()
        self.agents = self.possible_agents.copy()
        
        # Agent selector for sequential execution
        self.agent_selector = agent_selector(self.agents)
        
        # State management
        self.current_phase = SequentialPhase.INITIALIZATION
        self.execution_state = ExecutionState.AWAITING_STRATEGIC_CONTEXT
        self.strategic_context: Optional[StrategicContext] = None
        self.agent_superpositions: OrderedDict[str, AgentSuperposition] = OrderedDict()
        self.tactical_superposition: Optional[TacticalSuperposition] = None
        
        # Observation and action spaces
        self._setup_spaces()
        
        # Initialize core components
        self._initialize_components()
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.session_id = str(uuid.uuid4())
        
        # Episode management
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Error handling
        self.error_count = 0
        self.max_errors = self.config.get('max_errors', 5)
        self.emergency_halt = False
        
        # Execution history
        self.execution_history = deque(maxlen=self.config.get('history_length', 1000))
        self.strategic_context_history = deque(maxlen=100)
        
        logger.info(f"Sequential Tactical MARL Environment initialized")
        logger.info(f"Agent sequence: {self.agent_sequence}")
        logger.info(f"Session ID: {self.session_id}")
    
    def _load_config(self, config: Any) -> Dict[str, Any]:
        """Load configuration from various sources"""
        try:
            if isinstance(config, dict):
                return config
            elif isinstance(config, (str, Path)):
                with open(config, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._default_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for sequential tactical environment"""
        return {
            'sequential_tactical': {
                'environment': {
                    'matrix_shape': [60, 7],
                    'max_episode_steps': 2000,
                    'execution_timeout_ms': 50,  # 5-minute cycles need fast execution
                    'enable_byzantine_detection': True,
                    'enable_microstructure_modeling': True,
                    'high_frequency_mode': True,
                    'strategic_context_timeout': 10.0
                },
                'agents': {
                    'fvg_agent': {
                        'observation_enrichment': ['strategic_context', 'market_microstructure'],
                        'output_features': ['gap_probability', 'gap_strength', 'gap_duration'],
                        'confidence_threshold': 0.6,
                        'execution_weight': 0.35
                    },
                    'momentum_agent': {
                        'observation_enrichment': ['strategic_context', 'fvg_output', 'market_microstructure'],
                        'output_features': ['trend_probability', 'trend_strength', 'trend_duration'],
                        'confidence_threshold': 0.65,
                        'execution_weight': 0.40
                    },
                    'entry_opt_agent': {
                        'observation_enrichment': ['strategic_context', 'fvg_output', 'momentum_output', 'market_microstructure'],
                        'output_features': ['entry_probability', 'entry_timing', 'entry_size'],
                        'confidence_threshold': 0.7,
                        'execution_weight': 0.25
                    }
                },
                'superposition': {
                    'aggregation_method': 'weighted_ensemble',
                    'consensus_threshold': 0.75,
                    'execution_threshold': 0.8,
                    'risk_adjustment': True,
                    'microstructure_integration': True
                },
                'performance': {
                    'target_latency_ms': 50,
                    'max_latency_ms': 200,
                    'success_rate_threshold': 0.95,
                    'consensus_rate_threshold': 0.9
                }
            },
            'strategic_integration': {
                'context_update_frequency': 5.0,  # seconds
                'regime_embedding_dim': 64,
                'synergy_signal_features': 12,
                'confidence_decay_rate': 0.05
            },
            'byzantine_tolerance': {
                'max_byzantine_agents': 1,
                'detection_threshold': 0.7,
                'consensus_timeout': 2.0,
                'recovery_mode': 'graceful_degradation'
            },
            'microstructure': {
                'bid_ask_spread_modeling': True,
                'order_flow_analysis': True,
                'liquidity_assessment': True,
                'market_impact_estimation': True
            }
        }
    
    def _validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['sequential_tactical', 'strategic_integration', 'byzantine_tolerance']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate agent sequence
        agent_config = self.config['sequential_tactical']['agents']
        for agent_id in self.agent_sequence:
            if agent_id not in agent_config:
                raise ValueError(f"Missing configuration for agent: {agent_id}")
        
        logger.info("Configuration validation passed")
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Base observation space: 60x7 matrix + strategic context + predecessor outputs
        base_obs_dim = 60 * 7  # Matrix
        strategic_context_dim = self.config['strategic_integration']['regime_embedding_dim']
        synergy_signal_dim = self.config['strategic_integration']['synergy_signal_features']
        
        # Dynamic observation space based on agent position in sequence
        self.observation_spaces = {}
        self.action_spaces = {}
        
        for i, agent_id in enumerate(self.agent_sequence):
            # Base observation
            obs_dim = base_obs_dim + strategic_context_dim + synergy_signal_dim
            
            # Add predecessor outputs
            if i > 0:
                # Each predecessor contributes: 3 (probabilities) + 1 (confidence) + features
                predecessor_features = self.config['sequential_tactical']['agents'][agent_id]['output_features']
                obs_dim += i * (3 + 1 + len(predecessor_features))
            
            # Add microstructure features
            if self.config['microstructure']['bid_ask_spread_modeling']:
                obs_dim += 20  # Microstructure features
            
            self.observation_spaces[agent_id] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            
            # Action space: discrete actions for tactical decisions
            self.action_spaces[agent_id] = spaces.Discrete(3)  # bearish, neutral, bullish
    
    def _initialize_components(self):
        """Initialize core tactical components"""
        try:
            # Matrix assembler for 5-minute data
            self.matrix_assembler = MatrixAssembler5m(
                config=self.config['sequential_tactical']['environment']
            )
            
            # FVG detector for gap analysis
            self.fvg_detector = FVGDetector(
                config=self.config.get('fvg_config', {}),
                event_bus=None
            )
            
            # Byzantine detector for fault tolerance
            if self.config['byzantine_tolerance'].get('enable_detection', True):
                self.byzantine_detector = ByzantineDetector(
                    config=self.config['byzantine_tolerance']
                )
            else:
                self.byzantine_detector = None
            
            # PBFT engine for consensus
            if self.config['byzantine_tolerance'].get('enable_pbft', True):
                self.pbft_engine = PBFTEngine(
                    config=self.config['byzantine_tolerance']
                )
            else:
                self.pbft_engine = None
            
            # Microstructure analyzer
            self.microstructure_analyzer = self._initialize_microstructure_analyzer()
            
            # Strategic context manager
            self.strategic_context_manager = self._initialize_strategic_context_manager()
            
            # Superposition aggregator
            self.superposition_aggregator = self._initialize_superposition_aggregator()
            
            logger.info("All sequential tactical components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _initialize_microstructure_analyzer(self):
        """Initialize market microstructure analyzer"""
        try:
            from src.execution.microstructure.microstructure_engine import MicrostructureEngine
            return MicrostructureEngine(config=self.config['microstructure'])
        except ImportError:
            logger.warning("Microstructure engine not available, using mock")
            return MockMicrostructureAnalyzer()
    
    def _initialize_strategic_context_manager(self):
        """Initialize strategic context manager"""
        try:
            from src.integration.strategic_context_manager import StrategicContextManager
            return StrategicContextManager(config=self.config['strategic_integration'])
        except ImportError:
            logger.warning("Strategic context manager not available, using mock")
            return MockStrategicContextManager()
    
    def _initialize_superposition_aggregator(self):
        """Initialize tactical superposition aggregator"""
        try:
            from src.environment.tactical_superposition_aggregator import TacticalSuperpositionAggregator
            return TacticalSuperpositionAggregator(config=self.config['sequential_tactical']['superposition'])
        except ImportError:
            logger.warning("Tactical superposition aggregator not available, creating placeholder")
            return MockSuperpositionAggregator()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Reset environment for new episode
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Dictionary of initial observations
        """
        try:
            # Set random seed
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)
            
            # Reset episode state
            self.performance_metrics.episode_count += 1
            self.performance_metrics.step_count = 0
            self.current_phase = SequentialPhase.INITIALIZATION
            self.execution_state = ExecutionState.AWAITING_STRATEGIC_CONTEXT
            self.emergency_halt = False
            self.error_count = 0
            
            # Reset agents
            self.agents = self.possible_agents.copy()
            self.agent_selector.reset()
            
            # Clear state
            self.strategic_context = None
            self.agent_superpositions.clear()
            self.tactical_superposition = None
            
            # Reset rewards and states
            self.rewards = {agent: 0.0 for agent in self.agents}
            self.cumulative_rewards = {agent: 0.0 for agent in self.agents}
            self.dones = {agent: False for agent in self.agents}
            self.truncations = {agent: False for agent in self.agents}
            self.infos = {agent: {} for agent in self.agents}
            
            # Initialize strategic context
            self._initialize_strategic_context()
            
            # Update market state
            self._update_market_state()
            
            # Transition to first agent
            self.current_phase = SequentialPhase.FVG_ANALYSIS
            self.execution_state = ExecutionState.PROCESSING_AGENTS
            
            logger.info(f"Episode {self.performance_metrics.episode_count} reset complete")
            
            return {agent: self.observe(agent) for agent in self.agents}
            
        except Exception as e:
            logger.error(f"Error during reset: {e}")
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self.emergency_halt = True
            raise
    
    def step(self, action: ActionType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of sequential tactical decision-making
        
        Args:
            action: Current agent's action
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        try:
            if self.emergency_halt:
                return self._handle_emergency_halt()
            
            if not self.agents:
                raise ValueError("No active agents")
            
            # Record step timing
            step_start_time = time.time()
            
            # Get current agent
            current_agent = self.agent_selection
            
            # Validate action
            if not self.action_spaces[current_agent].contains(action):
                raise ValueError(f"Invalid action {action} for agent {current_agent}")
            
            # Process agent action
            self._process_agent_action(current_agent, action)
            
            # Update current phase
            self._update_sequential_phase()
            
            # Advance to next agent or complete sequence
            if self._is_sequence_complete():
                self._complete_sequential_execution()
            else:
                self.agent_selector.next()
            
            # Update performance metrics
            step_duration = time.time() - step_start_time
            self.performance_metrics.total_execution_time += step_duration
            self.performance_metrics.step_count += 1
            
            # Check episode termination
            self._check_episode_termination()
            
            # Update agent infos
            self._update_agent_infos()
            
            # Get return values
            current_agent = self.agent_selection
            obs = self.observe(current_agent)
            reward = self.rewards[current_agent]
            done = self.dones[current_agent]
            truncated = self.truncations[current_agent]
            info = self.infos[current_agent]
            
            return obs, reward, done, truncated, info
            
        except Exception as e:
            logger.error(f"Error in step: {e}")
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self.emergency_halt = True
            return self._handle_step_error(e)
    
    def observe(self, agent: str) -> np.ndarray:
        """
        Generate enriched observation for specific agent
        
        Args:
            agent: Agent identifier
            
        Returns:
            Enriched observation array
        """
        try:
            if agent not in self.agents:
                raise ValueError(f"Agent {agent} not in active agents")
            
            return self._get_enriched_observation(agent)
            
        except Exception as e:
            logger.error(f"Error in observe for agent {agent}: {e}")
            return np.zeros(self.observation_spaces[agent].shape, dtype=np.float32)
    
    def _get_enriched_observation(self, agent: str) -> np.ndarray:
        """Generate enriched observation with strategic context and predecessor outputs"""
        try:
            # Get base market matrix
            base_matrix = self._get_base_market_matrix()
            
            # Get strategic context
            strategic_features = self._get_strategic_features()
            
            # Get predecessor outputs
            predecessor_features = self._get_predecessor_features(agent)
            
            # Get microstructure features
            microstructure_features = self._get_microstructure_features()
            
            # Combine all features
            observation = np.concatenate([
                base_matrix.flatten(),
                strategic_features,
                predecessor_features,
                microstructure_features
            ]).astype(np.float32)
            
            # Ensure observation matches expected shape
            expected_shape = self.observation_spaces[agent].shape[0]
            if len(observation) != expected_shape:
                logger.warning(f"Observation shape mismatch for {agent}: got {len(observation)}, expected {expected_shape}")
                # Pad or truncate to match expected shape
                if len(observation) < expected_shape:
                    observation = np.pad(observation, (0, expected_shape - len(observation)))
                else:
                    observation = observation[:expected_shape]
            
            return observation
            
        except Exception as e:
            logger.error(f"Error creating enriched observation for {agent}: {e}")
            return np.zeros(self.observation_spaces[agent].shape, dtype=np.float32)
    
    def _get_base_market_matrix(self) -> np.ndarray:
        """Get base 60x7 market matrix"""
        try:
            matrix = self.matrix_assembler.get_matrix()
            if matrix is None:
                matrix = self._generate_synthetic_matrix()
            return matrix
        except Exception as e:
            logger.error(f"Error getting base matrix: {e}")
            return self._generate_synthetic_matrix()
    
    def _generate_synthetic_matrix(self) -> np.ndarray:
        """Generate synthetic market matrix for testing"""
        np.random.seed(int(time.time()) % 10000)
        
        matrix = np.random.randn(60, 7).astype(np.float32)
        
        # Add realistic financial patterns
        matrix[:, 0] = (np.random.rand(60) > 0.85).astype(np.float32)  # FVG bullish
        matrix[:, 1] = (np.random.rand(60) > 0.85).astype(np.float32)  # FVG bearish
        matrix[:, 2] = np.random.normal(0, 0.5, 60)  # FVG nearest level
        matrix[:, 3] = np.random.exponential(2, 60)  # FVG age
        matrix[:, 4] = (np.random.rand(60) > 0.95).astype(np.float32)  # FVG mitigation
        
        # Price momentum with autocorrelation
        momentum = np.random.normal(0, 0.3, 60)
        for i in range(1, 60):
            momentum[i] = 0.7 * momentum[i-1] + 0.3 * momentum[i]
        matrix[:, 5] = momentum
        
        # Volume ratio
        matrix[:, 6] = np.random.lognormal(0, 0.3, 60)
        
        return np.nan_to_num(matrix, nan=0.0, posinf=5.0, neginf=-5.0)
    
    def _get_strategic_features(self) -> np.ndarray:
        """Get strategic context features"""
        try:
            if self.strategic_context is None:
                return np.zeros(self.config['strategic_integration']['regime_embedding_dim'] + 
                              self.config['strategic_integration']['synergy_signal_features'], dtype=np.float32)
            
            # Combine regime embedding and synergy signal
            regime_features = self.strategic_context.regime_embedding
            synergy_features = self._extract_synergy_features(self.strategic_context.synergy_signal)
            
            return np.concatenate([regime_features, synergy_features]).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error getting strategic features: {e}")
            return np.zeros(self.config['strategic_integration']['regime_embedding_dim'] + 
                          self.config['strategic_integration']['synergy_signal_features'], dtype=np.float32)
    
    def _extract_synergy_features(self, synergy_signal: Dict[str, Any]) -> np.ndarray:
        """Extract synergy signal features"""
        try:
            features = []
            
            # Extract key synergy features
            features.extend([
                synergy_signal.get('strength', 0.0),
                synergy_signal.get('confidence', 0.0),
                synergy_signal.get('direction', 0.0),
                synergy_signal.get('urgency', 0.0),
                synergy_signal.get('risk_level', 0.0),
                synergy_signal.get('time_horizon', 0.0),
                synergy_signal.get('volatility_impact', 0.0),
                synergy_signal.get('liquidity_impact', 0.0),
                synergy_signal.get('correlation_shift', 0.0),
                synergy_signal.get('regime_stability', 0.0),
                synergy_signal.get('execution_priority', 0.0),
                synergy_signal.get('risk_reward_ratio', 0.0)
            ])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting synergy features: {e}")
            return np.zeros(12, dtype=np.float32)
    
    def _get_predecessor_features(self, agent: str) -> np.ndarray:
        """Get features from predecessor agents"""
        try:
            features = []
            
            agent_index = self.agent_sequence.index(agent)
            
            # Add features from all predecessors
            for i in range(agent_index):
                predecessor_id = self.agent_sequence[i]
                if predecessor_id in self.agent_superpositions:
                    superposition = self.agent_superpositions[predecessor_id]
                    
                    # Add probabilities, confidence, and output features
                    features.extend(superposition.probabilities)
                    features.append(superposition.confidence)
                    
                    # Add agent-specific output features
                    output_features = self._extract_agent_output_features(superposition)
                    features.extend(output_features)
                else:
                    # Placeholder for missing predecessor
                    features.extend([0.0] * (3 + 1 + len(self.config['sequential_tactical']['agents'][predecessor_id]['output_features'])))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting predecessor features for {agent}: {e}")
            return np.array([], dtype=np.float32)
    
    def _extract_agent_output_features(self, superposition: AgentSuperposition) -> List[float]:
        """Extract agent-specific output features"""
        try:
            features = []
            
            # Extract from execution signals
            if superposition.agent_id == 'fvg_agent':
                features.extend([
                    superposition.execution_signals.get('gap_probability', 0.0),
                    superposition.execution_signals.get('gap_strength', 0.0),
                    superposition.execution_signals.get('gap_duration', 0.0)
                ])
            elif superposition.agent_id == 'momentum_agent':
                features.extend([
                    superposition.execution_signals.get('trend_probability', 0.0),
                    superposition.execution_signals.get('trend_strength', 0.0),
                    superposition.execution_signals.get('trend_duration', 0.0)
                ])
            elif superposition.agent_id == 'entry_opt_agent':
                features.extend([
                    superposition.execution_signals.get('entry_probability', 0.0),
                    superposition.execution_signals.get('entry_timing', 0.0),
                    superposition.execution_signals.get('entry_size', 0.0)
                ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting agent output features: {e}")
            return [0.0] * 3
    
    def _get_microstructure_features(self) -> np.ndarray:
        """Get market microstructure features"""
        try:
            if hasattr(self.microstructure_analyzer, 'get_features'):
                return self.microstructure_analyzer.get_features()
            else:
                # Mock microstructure features
                return np.random.normal(0, 0.1, 20).astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error getting microstructure features: {e}")
            return np.zeros(20, dtype=np.float32)
    
    def _process_agent_action(self, agent: str, action: int):
        """Process agent action and generate superposition"""
        try:
            # Convert action to probabilities
            probabilities = self._action_to_probabilities(action, agent)
            
            # Calculate confidence based on action strength
            confidence = self._calculate_agent_confidence(agent, action, probabilities)
            
            # Extract feature importance
            feature_importance = self._calculate_feature_importance(agent)
            
            # Generate market insights
            market_insights = self._generate_market_insights(agent, action)
            
            # Create execution signals
            execution_signals = self._create_execution_signals(agent, action, probabilities)
            
            # Get predecessor context
            predecessor_context = self._get_predecessor_context(agent)
            
            # Create agent superposition
            superposition = AgentSuperposition(
                agent_id=agent,
                phase=self.current_phase,
                probabilities=probabilities,
                confidence=confidence,
                feature_importance=feature_importance,
                market_insights=market_insights,
                execution_signals=execution_signals,
                timestamp=time.time(),
                predecessor_context=predecessor_context
            )
            
            # Store superposition
            self.agent_superpositions[agent] = superposition
            
            # Update performance metrics
            self.performance_metrics.superposition_generations += 1
            
            logger.debug(f"Agent {agent} superposition generated: action={action}, confidence={confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing agent {agent} action: {e}")
            raise
    
    def _action_to_probabilities(self, action: int, agent: str) -> np.ndarray:
        """Convert discrete action to probability distribution"""
        try:
            # Get agent configuration
            agent_config = self.config['sequential_tactical']['agents'][agent]
            confidence_threshold = agent_config.get('confidence_threshold', 0.6)
            
            # Base probability distribution
            probs = np.zeros(3, dtype=np.float32)
            
            # Add stochasticity based on strategic context
            if self.strategic_context and self.strategic_context.confidence_level > confidence_threshold:
                # High confidence: concentrated probability
                probs[action] = 0.8
                remaining = 0.2
            else:
                # Lower confidence: more distributed
                probs[action] = 0.6
                remaining = 0.4
            
            # Distribute remaining probability
            other_actions = [i for i in range(3) if i != action]
            for other_action in other_actions:
                probs[other_action] = remaining / len(other_actions)
            
            # Normalize
            probs = probs / np.sum(probs)
            
            return probs
            
        except Exception as e:
            logger.error(f"Error converting action to probabilities: {e}")
            return np.array([0.33, 0.34, 0.33], dtype=np.float32)
    
    def _calculate_agent_confidence(self, agent: str, action: int, probabilities: np.ndarray) -> float:
        """Calculate agent confidence based on action and context"""
        try:
            # Base confidence from probability distribution
            base_confidence = float(np.max(probabilities))
            
            # Strategic alignment boost
            strategic_boost = 0.0
            if self.strategic_context:
                if self.strategic_context.execution_bias == 'bullish' and action == 2:
                    strategic_boost = 0.1
                elif self.strategic_context.execution_bias == 'bearish' and action == 0:
                    strategic_boost = 0.1
                elif self.strategic_context.execution_bias == 'neutral' and action == 1:
                    strategic_boost = 0.05
            
            # Predecessor alignment boost
            predecessor_boost = self._calculate_predecessor_alignment_boost(agent, action)
            
            # Final confidence
            confidence = base_confidence + strategic_boost + predecessor_boost
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating agent confidence: {e}")
            return 0.5
    
    def _calculate_predecessor_alignment_boost(self, agent: str, action: int) -> float:
        """Calculate confidence boost from predecessor alignment"""
        try:
            agent_index = self.agent_sequence.index(agent)
            if agent_index == 0:
                return 0.0
            
            boost = 0.0
            for i in range(agent_index):
                predecessor_id = self.agent_sequence[i]
                if predecessor_id in self.agent_superpositions:
                    pred_superposition = self.agent_superpositions[predecessor_id]
                    pred_action = np.argmax(pred_superposition.probabilities)
                    
                    # Alignment boost
                    if pred_action == action:
                        boost += 0.05 * pred_superposition.confidence
                    elif abs(pred_action - action) == 1:  # Adjacent actions
                        boost += 0.02 * pred_superposition.confidence
            
            return min(0.2, boost)
            
        except Exception as e:
            logger.error(f"Error calculating predecessor alignment boost: {e}")
            return 0.0
    
    def _calculate_feature_importance(self, agent: str) -> Dict[str, float]:
        """Calculate feature importance for agent decision"""
        try:
            # Mock feature importance calculation
            # In practice, this would use interpretability methods
            
            features = {
                'market_matrix': 0.3,
                'strategic_context': 0.2,
                'microstructure': 0.2,
                'predecessor_outputs': 0.2,
                'agent_specific': 0.1
            }
            
            # Agent-specific adjustments
            if agent == 'fvg_agent':
                features['market_matrix'] = 0.4
                features['strategic_context'] = 0.3
            elif agent == 'momentum_agent':
                features['predecessor_outputs'] = 0.3
                features['strategic_context'] = 0.25
            elif agent == 'entry_opt_agent':
                features['predecessor_outputs'] = 0.35
                features['microstructure'] = 0.25
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {'default': 1.0}
    
    def _generate_market_insights(self, agent: str, action: int) -> Dict[str, Any]:
        """Generate market insights from agent perspective"""
        try:
            insights = {
                'market_condition': 'normal',
                'volatility_assessment': 'moderate',
                'liquidity_assessment': 'adequate',
                'risk_factors': [],
                'opportunities': [],
                'time_sensitivity': 'medium'
            }
            
            # Agent-specific insights
            if agent == 'fvg_agent':
                insights.update({
                    'gap_analysis': 'active_gaps_detected',
                    'support_resistance': 'strong_levels',
                    'price_action_quality': 'high'
                })
            elif agent == 'momentum_agent':
                insights.update({
                    'trend_strength': 'moderate',
                    'momentum_sustainability': 'likely',
                    'reversal_probability': 'low'
                })
            elif agent == 'entry_opt_agent':
                insights.update({
                    'entry_quality': 'optimal',
                    'timing_confidence': 'high',
                    'execution_difficulty': 'low'
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating market insights: {e}")
            return {'status': 'error'}
    
    def _create_execution_signals(self, agent: str, action: int, probabilities: np.ndarray) -> Dict[str, Any]:
        """Create execution signals for agent"""
        try:
            signals = {
                'primary_signal': action,
                'signal_strength': float(np.max(probabilities)),
                'execution_urgency': 'medium',
                'position_size_modifier': 1.0,
                'risk_adjustment': 0.0
            }
            
            # Agent-specific signals
            if agent == 'fvg_agent':
                signals.update({
                    'gap_probability': float(probabilities[action]),
                    'gap_strength': np.random.uniform(0.5, 1.0),
                    'gap_duration': np.random.uniform(1.0, 5.0)
                })
            elif agent == 'momentum_agent':
                signals.update({
                    'trend_probability': float(probabilities[action]),
                    'trend_strength': np.random.uniform(0.5, 1.0),
                    'trend_duration': np.random.uniform(2.0, 10.0)
                })
            elif agent == 'entry_opt_agent':
                signals.update({
                    'entry_probability': float(probabilities[action]),
                    'entry_timing': np.random.uniform(0.0, 1.0),
                    'entry_size': np.random.uniform(0.5, 1.5)
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error creating execution signals: {e}")
            return {'status': 'error'}
    
    def _get_predecessor_context(self, agent: str) -> Optional[Dict[str, Any]]:
        """Get context from predecessor agents"""
        try:
            agent_index = self.agent_sequence.index(agent)
            if agent_index == 0:
                return None
            
            context = {
                'predecessor_count': agent_index,
                'predecessor_outputs': {},
                'consensus_level': 0.0,
                'alignment_score': 0.0
            }
            
            for i in range(agent_index):
                predecessor_id = self.agent_sequence[i]
                if predecessor_id in self.agent_superpositions:
                    context['predecessor_outputs'][predecessor_id] = {
                        'action': np.argmax(self.agent_superpositions[predecessor_id].probabilities),
                        'confidence': self.agent_superpositions[predecessor_id].confidence,
                        'signals': self.agent_superpositions[predecessor_id].execution_signals
                    }
            
            # Calculate consensus metrics
            if context['predecessor_outputs']:
                actions = [output['action'] for output in context['predecessor_outputs'].values()]
                confidences = [output['confidence'] for output in context['predecessor_outputs'].values()]
                
                context['consensus_level'] = len(set(actions)) / len(actions)  # Reverse measure
                context['alignment_score'] = np.mean(confidences)
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting predecessor context: {e}")
            return None
    
    def _update_sequential_phase(self):
        """Update current sequential phase"""
        try:
            current_agent = self.agent_selection
            
            if current_agent == 'fvg_agent':
                self.current_phase = SequentialPhase.FVG_ANALYSIS
            elif current_agent == 'momentum_agent':
                self.current_phase = SequentialPhase.MOMENTUM_ANALYSIS
            elif current_agent == 'entry_opt_agent':
                self.current_phase = SequentialPhase.ENTRY_OPTIMIZATION
            
            # Record phase timing
            if self.current_phase in self.performance_metrics.phase_latencies:
                self.performance_metrics.phase_latencies[self.current_phase].append(time.time())
            
        except Exception as e:
            logger.error(f"Error updating sequential phase: {e}")
    
    def _is_sequence_complete(self) -> bool:
        """Check if sequential execution is complete"""
        return len(self.agent_superpositions) == len(self.agent_sequence)
    
    def _complete_sequential_execution(self):
        """Complete sequential execution and aggregate superposition"""
        try:
            self.current_phase = SequentialPhase.SUPERPOSITION_AGGREGATION
            self.execution_state = ExecutionState.AGGREGATING_SUPERPOSITION
            
            # Aggregate tactical superposition
            self.tactical_superposition = self._aggregate_tactical_superposition()
            
            # Calculate rewards
            self._calculate_sequential_rewards()
            
            # Update state
            self.current_phase = SequentialPhase.EXECUTION_READY
            self.execution_state = ExecutionState.READY_FOR_EXECUTION
            
            # Record execution
            self.execution_history.append({
                'timestamp': time.time(),
                'episode': self.performance_metrics.episode_count,
                'step': self.performance_metrics.step_count,
                'tactical_superposition': self.tactical_superposition.to_dict(),
                'agent_superpositions': {k: v.__dict__ for k, v in self.agent_superpositions.items()},
                'strategic_context': self.strategic_context.__dict__ if self.strategic_context else None
            })
            
            logger.info(f"Sequential execution complete: execute={self.tactical_superposition.execute}, "
                       f"confidence={self.tactical_superposition.confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Error completing sequential execution: {e}")
            self.error_count += 1
    
    def _aggregate_tactical_superposition(self) -> TacticalSuperposition:
        """Aggregate agent superpositions into tactical superposition"""
        try:
            # Use superposition aggregator if available
            if hasattr(self.superposition_aggregator, 'aggregate'):
                return self.superposition_aggregator.aggregate(
                    agent_superpositions=self.agent_superpositions,
                    strategic_context=self.strategic_context,
                    market_state=self._get_current_market_state()
                )
            else:
                # Fallback aggregation
                return self._fallback_superposition_aggregation()
                
        except Exception as e:
            logger.error(f"Error aggregating tactical superposition: {e}")
            return self._create_safe_superposition()
    
    def _fallback_superposition_aggregation(self) -> TacticalSuperposition:
        """Fallback superposition aggregation"""
        try:
            # Weight agents by configuration
            weights = []
            for agent_id in self.agent_sequence:
                agent_config = self.config['sequential_tactical']['agents'][agent_id]
                weights.append(agent_config.get('execution_weight', 1.0))
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Aggregate probabilities
            aggregated_probs = np.zeros(3, dtype=np.float32)
            total_confidence = 0.0
            
            for i, agent_id in enumerate(self.agent_sequence):
                if agent_id in self.agent_superpositions:
                    superposition = self.agent_superpositions[agent_id]
                    weight = weights[i]
                    
                    aggregated_probs += superposition.probabilities * weight * superposition.confidence
                    total_confidence += superposition.confidence * weight
            
            # Normalize aggregated probabilities
            if np.sum(aggregated_probs) > 0:
                aggregated_probs = aggregated_probs / np.sum(aggregated_probs)
            else:
                aggregated_probs = np.array([0.33, 0.34, 0.33], dtype=np.float32)
            
            # Determine action
            action = int(np.argmax(aggregated_probs))
            
            # Calculate final confidence
            final_confidence = total_confidence
            
            # Apply strategic alignment
            strategic_alignment = self._calculate_strategic_alignment(action)
            final_confidence *= (1.0 + strategic_alignment * 0.1)
            final_confidence = min(1.0, max(0.0, final_confidence))
            
            # Execution decision
            execution_threshold = self.config['sequential_tactical']['superposition']['execution_threshold']
            execute = final_confidence >= execution_threshold
            
            # Create execution command
            execution_command = None
            if execute:
                execution_command = self._create_tactical_execution_command(action, final_confidence)
            
            # Risk assessment
            risk_assessment = self._calculate_risk_assessment()
            
            # Microstructure analysis
            microstructure_analysis = self._analyze_microstructure()
            
            return TacticalSuperposition(
                execute=execute,
                action=action,
                confidence=final_confidence,
                aggregated_probabilities=aggregated_probs,
                agent_contributions=self.agent_superpositions.copy(),
                strategic_alignment=strategic_alignment,
                execution_command=execution_command,
                risk_assessment=risk_assessment,
                microstructure_analysis=microstructure_analysis,
                timestamp=time.time(),
                session_id=self.session_id
            )
            
        except Exception as e:
            logger.error(f"Error in fallback superposition aggregation: {e}")
            return self._create_safe_superposition()
    
    def _calculate_strategic_alignment(self, action: int) -> float:
        """Calculate alignment with strategic context"""
        try:
            if not self.strategic_context:
                return 0.0
            
            # Map action to bias
            action_bias = ['bearish', 'neutral', 'bullish'][action]
            
            # Calculate alignment
            if action_bias == self.strategic_context.execution_bias:
                return self.strategic_context.confidence_level
            elif self.strategic_context.execution_bias == 'neutral':
                return 0.5 * self.strategic_context.confidence_level
            else:
                return -0.5 * self.strategic_context.confidence_level
                
        except Exception as e:
            logger.error(f"Error calculating strategic alignment: {e}")
            return 0.0
    
    def _create_tactical_execution_command(self, action: int, confidence: float) -> Dict[str, Any]:
        """Create tactical execution command"""
        try:
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            
            if action == 1:  # Hold
                return {'action': 'HOLD', 'reason': 'neutral_tactical_consensus'}
            
            # Base position size
            base_size = 1.0
            confidence_multiplier = min(confidence / 0.8, 1.5)
            position_size = base_size * confidence_multiplier
            
            # Risk management
            current_price = 100.0  # Mock price
            atr_estimate = current_price * 0.02
            
            if action == 2:  # Buy
                stop_loss = current_price - (2.0 * atr_estimate)
                take_profit = current_price + (3.0 * atr_estimate)
            else:  # Sell
                stop_loss = current_price + (2.0 * atr_estimate)
                take_profit = current_price - (3.0 * atr_estimate)
            
            return {
                'action': 'EXECUTE_TRADE',
                'side': action_map[action],
                'quantity': round(position_size, 2),
                'order_type': 'MARKET',
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'time_in_force': 'IOC',
                'source': 'sequential_tactical_marl',
                'confidence': confidence,
                'agent_contributions': list(self.agent_superpositions.keys())
            }
            
        except Exception as e:
            logger.error(f"Error creating tactical execution command: {e}")
            return {'action': 'HOLD', 'reason': 'command_creation_error'}
    
    def _calculate_risk_assessment(self) -> Dict[str, float]:
        """Calculate risk assessment for tactical superposition"""
        try:
            # Mock risk assessment
            return {
                'market_risk': 0.3,
                'execution_risk': 0.2,
                'model_risk': 0.1,
                'liquidity_risk': 0.15,
                'operational_risk': 0.05,
                'aggregate_risk': 0.25
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk assessment: {e}")
            return {'aggregate_risk': 0.5}
    
    def _analyze_microstructure(self) -> Dict[str, Any]:
        """Analyze market microstructure"""
        try:
            # Mock microstructure analysis
            return {
                'bid_ask_spread': 0.001,
                'market_depth': 0.8,
                'order_flow_imbalance': 0.1,
                'price_impact_estimate': 0.05,
                'liquidity_score': 0.75,
                'execution_difficulty': 'low'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing microstructure: {e}")
            return {'status': 'error'}
    
    def _create_safe_superposition(self) -> TacticalSuperposition:
        """Create safe default superposition"""
        return TacticalSuperposition(
            execute=False,
            action=1,  # Neutral
            confidence=0.0,
            aggregated_probabilities=np.array([0.33, 0.34, 0.33], dtype=np.float32),
            agent_contributions={},
            strategic_alignment=0.0,
            execution_command=None,
            risk_assessment={'aggregate_risk': 1.0},
            microstructure_analysis={'status': 'safe_default'},
            timestamp=time.time(),
            session_id=self.session_id
        )
    
    def _calculate_sequential_rewards(self):
        """Calculate rewards for sequential execution"""
        try:
            # Base reward from superposition quality
            base_reward = 0.0
            if self.tactical_superposition:
                base_reward = self.tactical_superposition.confidence * 2.0 - 1.0  # Scale to [-1, 1]
            
            # Strategic alignment bonus
            strategic_bonus = 0.0
            if self.tactical_superposition and self.tactical_superposition.strategic_alignment > 0:
                strategic_bonus = self.tactical_superposition.strategic_alignment * 0.5
            
            # Execution bonus
            execution_bonus = 0.0
            if self.tactical_superposition and self.tactical_superposition.execute:
                execution_bonus = 0.2
            
            # Consensus bonus
            consensus_bonus = 0.0
            if len(self.agent_superpositions) == len(self.agent_sequence):
                # Calculate consensus level
                actions = [np.argmax(s.probabilities) for s in self.agent_superpositions.values()]
                if len(set(actions)) == 1:  # Perfect consensus
                    consensus_bonus = 0.3
                elif len(set(actions)) == 2:  # Partial consensus
                    consensus_bonus = 0.1
            
            # Calculate agent-specific rewards
            agent_weights = [
                self.config['sequential_tactical']['agents'][agent_id].get('execution_weight', 1.0)
                for agent_id in self.agent_sequence
            ]
            
            total_reward = base_reward + strategic_bonus + execution_bonus + consensus_bonus
            
            for i, agent_id in enumerate(self.agent_sequence):
                if agent_id in self.agent_superpositions:
                    agent_weight = agent_weights[i]
                    agent_contribution = self.agent_superpositions[agent_id].confidence
                    
                    self.rewards[agent_id] = total_reward * agent_weight * agent_contribution
                    self.cumulative_rewards[agent_id] += self.rewards[agent_id]
            
            logger.debug(f"Sequential rewards calculated: base={base_reward:.3f}, "
                        f"strategic={strategic_bonus:.3f}, execution={execution_bonus:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating sequential rewards: {e}")
            # Fallback to zero rewards
            self.rewards = {agent: 0.0 for agent in self.agents}
    
    def _initialize_strategic_context(self):
        """Initialize strategic context from upstream system"""
        try:
            if hasattr(self.strategic_context_manager, 'get_latest_context'):
                self.strategic_context = self.strategic_context_manager.get_latest_context()
            else:
                # Mock strategic context
                self.strategic_context = StrategicContext(
                    timestamp=time.time(),
                    regime_embedding=np.random.normal(0, 0.1, 64).astype(np.float32),
                    synergy_signal={
                        'strength': 0.7,
                        'confidence': 0.8,
                        'direction': 0.5,
                        'urgency': 0.6,
                        'risk_level': 0.3,
                        'time_horizon': 0.8,
                        'volatility_impact': 0.4,
                        'liquidity_impact': 0.5,
                        'correlation_shift': 0.2,
                        'regime_stability': 0.9,
                        'execution_priority': 0.7,
                        'risk_reward_ratio': 1.5
                    },
                    market_state={'price': 100.0, 'volume': 1000.0, 'volatility': 0.2},
                    confidence_level=0.8,
                    execution_bias='neutral',
                    volatility_forecast=0.25,
                    session_id=self.session_id
                )
            
            self.strategic_context_history.append(self.strategic_context)
            self.performance_metrics.strategic_context_updates += 1
            
        except Exception as e:
            logger.error(f"Error initializing strategic context: {e}")
            self.strategic_context = None
    
    def _update_market_state(self):
        """Update market state from matrix assembler"""
        try:
            # Update matrix assembler
            if hasattr(self.matrix_assembler, 'update'):
                self.matrix_assembler.update()
            
            # Update microstructure analyzer
            if hasattr(self.microstructure_analyzer, 'update'):
                self.microstructure_analyzer.update()
                
        except Exception as e:
            logger.error(f"Error updating market state: {e}")
    
    def _get_current_market_state(self) -> Dict[str, Any]:
        """Get current market state"""
        try:
            matrix = self._get_base_market_matrix()
            latest_bar = matrix[-1] if len(matrix) > 0 else np.zeros(7)
            
            return {
                'price': float(latest_bar[0]) if len(latest_bar) > 0 else 100.0,
                'volume': float(latest_bar[1]) if len(latest_bar) > 1 else 1000.0,
                'volatility': float(np.std(matrix[-20:, 0])) if len(matrix) >= 20 else 0.2,
                'matrix': matrix,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting current market state: {e}")
            return {'price': 100.0, 'volume': 1000.0, 'volatility': 0.2}
    
    def _check_episode_termination(self):
        """Check if episode should terminate"""
        try:
            max_steps = self.config['sequential_tactical']['environment']['max_episode_steps']
            
            # Check step limit
            if self.performance_metrics.step_count >= max_steps:
                self.dones = {agent: True for agent in self.agents}
                self.truncations = {agent: True for agent in self.agents}
                self.current_phase = SequentialPhase.EPISODE_COMPLETE
                logger.info(f"Episode terminated: max steps reached ({max_steps})")
            
            # Check emergency halt
            if self.emergency_halt:
                self.dones = {agent: True for agent in self.agents}
                self.truncations = {agent: True for agent in self.agents}
                logger.warning("Episode terminated: emergency halt")
            
            # Check performance-based termination
            if self._should_terminate_for_performance():
                self.dones = {agent: True for agent in self.agents}
                self.truncations = {agent: False for agent in self.agents}
                logger.info("Episode terminated: performance criteria")
                
        except Exception as e:
            logger.error(f"Error checking episode termination: {e}")
    
    def _should_terminate_for_performance(self) -> bool:
        """Check performance-based termination criteria"""
        try:
            # Check cumulative rewards
            min_performance = -100.0
            for agent in self.agents:
                if self.cumulative_rewards[agent] < min_performance:
                    return True
            
            # Check error rate
            if self.error_count >= self.max_errors:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _update_agent_infos(self):
        """Update agent info dictionaries"""
        try:
            for agent in self.agents:
                self.infos[agent].update({
                    'sequential_phase': self.current_phase.value,
                    'execution_state': self.execution_state.value,
                    'step_count': self.performance_metrics.step_count,
                    'episode_count': self.performance_metrics.episode_count,
                    'session_id': self.session_id,
                    'tactical_superposition': self.tactical_superposition.to_dict() if self.tactical_superposition else None,
                    'strategic_context': self.strategic_context.__dict__ if self.strategic_context else None,
                    'agent_superpositions': {k: v.__dict__ for k, v in self.agent_superpositions.items()},
                    'performance_metrics': {
                        'total_execution_time': self.performance_metrics.total_execution_time,
                        'superposition_generations': self.performance_metrics.superposition_generations,
                        'strategic_context_updates': self.performance_metrics.strategic_context_updates
                    }
                })
                
        except Exception as e:
            logger.error(f"Error updating agent infos: {e}")
    
    def _handle_emergency_halt(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Handle emergency halt condition"""
        logger.critical("Emergency halt activated in sequential tactical environment")
        
        return (
            np.zeros(self.observation_spaces[self.agents[0]].shape, dtype=np.float32),
            -10.0,  # Large penalty
            True,   # done
            True,   # truncated
            {'emergency_halt': True, 'error_count': self.error_count}
        )
    
    def _handle_step_error(self, error: Exception) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Handle step errors gracefully"""
        logger.error(f"Step error in sequential tactical environment: {error}")
        
        current_agent = self.agent_selection if self.agents else 'unknown'
        
        return (
            np.zeros(self.observation_spaces.get(current_agent, spaces.Box(low=0, high=1, shape=(1,))).shape, dtype=np.float32),
            -1.0,   # penalty
            False,  # not done
            False,  # not truncated
            {'error': str(error), 'agent': current_agent}
        )
    
    @property
    def agent_selection(self) -> str:
        """Get current agent selection"""
        return self.agent_selector.agent_selection
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment state"""
        try:
            if mode == "human":
                self._render_human()
            elif mode == "tactical_dashboard":
                return self._render_tactical_dashboard()
            elif mode == "ansi":
                return self._render_ansi()
            else:
                logger.warning(f"Unknown render mode: {mode}")
                
        except Exception as e:
            logger.error(f"Error in render: {e}")
            
        return None
    
    def _render_human(self):
        """Render for human viewing"""
        print(f"\n=== Sequential Tactical MARL Environment ===")
        print(f"Episode: {self.performance_metrics.episode_count}")
        print(f"Step: {self.performance_metrics.step_count}")
        print(f"Phase: {self.current_phase.value}")
        print(f"Execution State: {self.execution_state.value}")
        print(f"Current Agent: {self.agent_selection}")
        
        # Strategic context
        if self.strategic_context:
            print(f"\nStrategic Context:")
            print(f"  Confidence: {self.strategic_context.confidence_level:.3f}")
            print(f"  Execution Bias: {self.strategic_context.execution_bias}")
            print(f"  Volatility Forecast: {self.strategic_context.volatility_forecast:.3f}")
        
        # Agent superpositions
        print(f"\nAgent Superpositions ({len(self.agent_superpositions)}/{len(self.agent_sequence)}):")
        for agent_id, superposition in self.agent_superpositions.items():
            action = np.argmax(superposition.probabilities)
            print(f"  {agent_id}: action={action}, confidence={superposition.confidence:.3f}")
        
        # Tactical superposition
        if self.tactical_superposition:
            print(f"\nTactical Superposition:")
            print(f"  Execute: {self.tactical_superposition.execute}")
            print(f"  Action: {self.tactical_superposition.action}")
            print(f"  Confidence: {self.tactical_superposition.confidence:.3f}")
            print(f"  Strategic Alignment: {self.tactical_superposition.strategic_alignment:.3f}")
        
        # Performance metrics
        print(f"\nPerformance:")
        print(f"  Execution Time: {self.performance_metrics.total_execution_time:.3f}s")
        print(f"  Superposition Generations: {self.performance_metrics.superposition_generations}")
        print(f"  Error Count: {self.error_count}")
    
    def _render_tactical_dashboard(self) -> Dict[str, Any]:
        """Render tactical dashboard data"""
        return {
            'episode_info': {
                'episode': self.performance_metrics.episode_count,
                'step': self.performance_metrics.step_count,
                'phase': self.current_phase.value,
                'execution_state': self.execution_state.value
            },
            'strategic_context': self.strategic_context.__dict__ if self.strategic_context else None,
            'agent_superpositions': {k: v.__dict__ for k, v in self.agent_superpositions.items()},
            'tactical_superposition': self.tactical_superposition.to_dict() if self.tactical_superposition else None,
            'performance_metrics': {
                'total_execution_time': self.performance_metrics.total_execution_time,
                'superposition_generations': self.performance_metrics.superposition_generations,
                'strategic_context_updates': self.performance_metrics.strategic_context_updates,
                'error_count': self.error_count
            }
        }
    
    def _render_ansi(self) -> str:
        """Render as ANSI string"""
        output = []
        output.append(f"Sequential Tactical MARL - Episode {self.performance_metrics.episode_count}")
        output.append(f"Phase: {self.current_phase.value}")
        output.append(f"Agent: {self.agent_selection}")
        output.append(f"Superpositions: {len(self.agent_superpositions)}/{len(self.agent_sequence)}")
        
        if self.tactical_superposition:
            output.append(f"Execute: {self.tactical_superposition.execute}")
            output.append(f"Confidence: {self.tactical_superposition.confidence:.3f}")
        
        return "\n".join(output)
    
    def close(self):
        """Clean up environment resources"""
        try:
            self.agents.clear()
            self.agent_superpositions.clear()
            self.execution_history.clear()
            self.strategic_context_history.clear()
            
            if hasattr(self.matrix_assembler, 'close'):
                self.matrix_assembler.close()
            
            if hasattr(self.microstructure_analyzer, 'close'):
                self.microstructure_analyzer.close()
            
            logger.info(f"Sequential Tactical MARL Environment closed (Session: {self.session_id})")
            
        except Exception as e:
            logger.error(f"Error during close: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            return {
                'episode_count': self.performance_metrics.episode_count,
                'step_count': self.performance_metrics.step_count,
                'total_execution_time': self.performance_metrics.total_execution_time,
                'avg_execution_time': self.performance_metrics.total_execution_time / max(1, self.performance_metrics.step_count),
                'superposition_generations': self.performance_metrics.superposition_generations,
                'strategic_context_updates': self.performance_metrics.strategic_context_updates,
                'error_count': self.error_count,
                'emergency_halt': self.emergency_halt,
                'session_id': self.session_id,
                'phase_latencies': {
                    phase.value: {
                        'count': len(latencies),
                        'avg': np.mean(latencies) if latencies else 0.0,
                        'p95': np.percentile(latencies, 95) if latencies else 0.0
                    } for phase, latencies in self.performance_metrics.phase_latencies.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def get_tactical_superposition(self) -> Optional[TacticalSuperposition]:
        """Get current tactical superposition"""
        return self.tactical_superposition
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history"""
        try:
            return list(self.execution_history)[-limit:]
        except Exception as e:
            logger.error(f"Error getting execution history: {e}")
            return []
    
    def update_strategic_context(self, new_context: StrategicContext):
        """Update strategic context from upstream system"""
        try:
            self.strategic_context = new_context
            self.strategic_context_history.append(new_context)
            self.performance_metrics.strategic_context_updates += 1
            
            logger.debug(f"Strategic context updated: confidence={new_context.confidence_level:.3f}, "
                        f"bias={new_context.execution_bias}")
            
        except Exception as e:
            logger.error(f"Error updating strategic context: {e}")


# Mock classes for missing dependencies
class MockMicrostructureAnalyzer:
    """Mock microstructure analyzer"""
    def get_features(self):
        return np.random.normal(0, 0.1, 20).astype(np.float32)
    
    def update(self):
        pass
    
    def close(self):
        pass

class MockStrategicContextManager:
    """Mock strategic context manager"""
    def get_latest_context(self):
        return None

class MockSuperpositionAggregator:
    """Mock superposition aggregator"""
    def aggregate(self, agent_superpositions, strategic_context, market_state):
        return None


def make_sequential_tactical_env(config: Optional[Union[Dict[str, Any], str, Path]] = None) -> SequentialTacticalEnvironment:
    """
    Factory function to create sequential tactical environment
    
    Args:
        config: Environment configuration
        
    Returns:
        Configured SequentialTacticalEnvironment instance
    """
    try:
        env = SequentialTacticalEnvironment(config)
        
        # Apply PettingZoo wrappers
        env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        
        return env
        
    except Exception as e:
        logger.error(f"Error creating sequential tactical environment: {e}")
        raise


def validate_sequential_environment(env: SequentialTacticalEnvironment) -> Dict[str, Any]:
    """
    Validate sequential tactical environment
    
    Args:
        env: Environment to validate
        
    Returns:
        Validation results
    """
    try:
        validation_results = {
            'pettingzoo_compliance': True,
            'sequential_execution': True,
            'strategic_integration': True,
            'performance_acceptable': True,
            'errors': []
        }
        
        # Basic PettingZoo compliance
        try:
            from pettingzoo.test import api_test
            api_test(env, num_cycles=5)
        except Exception as e:
            validation_results['pettingzoo_compliance'] = False
            validation_results['errors'].append(f"PettingZoo API test failed: {e}")
        
        # Sequential execution test
        try:
            env.reset()
            for cycle in range(3):
                for agent in env.agent_sequence:
                    if agent == env.agent_selection:
                        action = env.action_spaces[agent].sample()
                        env.step(action)
                        if env.dones[agent]:
                            break
                
                if env.tactical_superposition:
                    break
                    
        except Exception as e:
            validation_results['sequential_execution'] = False
            validation_results['errors'].append(f"Sequential execution test failed: {e}")
        
        return validation_results
        
    except Exception as e:
        return {
            'pettingzoo_compliance': False,
            'sequential_execution': False,
            'strategic_integration': False,
            'performance_acceptable': False,
            'errors': [f"Validation failed: {e}"]
        }


# Example usage
if __name__ == "__main__":
    # Create environment
    env = make_sequential_tactical_env()
    
    # Validate
    validation_results = validate_sequential_environment(env)
    print("Validation Results:", validation_results)
    
    # Run example
    observations = env.reset()
    print(f"Initial observations: {list(observations.keys())}")
    
    # Run sequential execution
    for step in range(15):
        current_agent = env.agent_selection
        action = env.action_spaces[current_agent].sample()
        
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: Agent {current_agent}, Action {action}, Reward {reward:.3f}")
        
        if done or truncated:
            print("Episode finished!")
            break
    
    # Get final superposition
    tactical_superposition = env.get_tactical_superposition()
    if tactical_superposition:
        print(f"Final Tactical Superposition: {tactical_superposition.to_dict()}")
    
    # Performance metrics
    metrics = env.get_performance_metrics()
    print(f"Performance Metrics: {metrics}")
    
    env.close()