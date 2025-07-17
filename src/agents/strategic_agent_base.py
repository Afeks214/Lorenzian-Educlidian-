"""
Strategic Agent Base Classes - Abstract base classes for the three specialized agents.

This module provides base classes and interfaces that other agents will inherit from:
- StrategicAgentBase: Abstract base class for all strategic agents
- MLMIStrategicAgent: Market Liquidity and Market Impact expert agent
- NWRQKStrategicAgent: Network-weighted risk-quality kernel expert agent  
- RegimeDetectionAgent: Market regime detection expert agent

These classes provide the interface and structure that other agents (2-4) will implement.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AgentPrediction:
    """Result from a strategic agent prediction."""
    action_probabilities: np.ndarray  # [buy, hold, sell] probabilities
    confidence: float                 # Confidence in the prediction [0, 1]
    feature_importance: Dict[str, float]  # Feature importance scores
    internal_state: Dict[str, Any]    # Agent's internal state
    computation_time_ms: float        # Time taken for computation
    timestamp: datetime


class StrategicAgentBase(ABC):
    """
    Abstract base class for all strategic agents in the ensemble.
    
    Each agent specializes in different aspects of market analysis:
    - MLMI: Market liquidity and market impact patterns
    - NWRQK: Network-weighted risk and quality patterns  
    - Regime: Market regime detection and adaptation
    """
    
    def __init__(
        self, 
        name: str,
        feature_indices: List[int],
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """
        Initialize the strategic agent.
        
        Args:
            name: Agent name for identification
            feature_indices: List of feature column indices this agent uses
            config: Agent-specific configuration
            device: Torch device ('cpu' or 'cuda')
        """
        self.name = name
        self.feature_indices = feature_indices
        self.config = config
        self.device = torch.device(device)
        
        # Set up logging
        self.logger = logging.getLogger(f'strategic_agent.{name.lower()}')
        
        # Model components (to be initialized by subclasses)
        self.actor_network: Optional[nn.Module] = None
        self.critic_network: Optional[nn.Module] = None
        
        # Performance tracking
        self.prediction_count = 0
        self.total_computation_time_ms = 0.0
        self.last_prediction_time = None
        
        # Health status
        self.is_healthy = True
        self.last_error = None
        self.consecutive_errors = 0
        
        # Strategy support configuration
        self.strategy_support_enabled = config.get('strategy_support_enabled', True)
        self.strategy_override_threshold = config.get('strategy_override_threshold', 0.7)
        
        self.logger.info(
            f"Strategic agent {name} initialized: "
            f"feature_indices={feature_indices}, device={str(device)}, "
            f"strategy_support={self.strategy_support_enabled}"
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent's models and internal state.
        
        This method should:
        - Load or initialize neural network models
        - Set up any required preprocessing
        - Validate configuration
        - Perform health checks
        """
        pass
    
    @abstractmethod
    async def predict(
        self,
        matrix_data: np.ndarray,
        shared_context: Dict[str, Any]
    ) -> AgentPrediction:
        """
        Make a strategic prediction based on input data.
        
        Args:
            matrix_data: Full matrix data (48, 13)
            shared_context: Shared context from coordinator
            
        Returns:
            AgentPrediction with action probabilities and metadata
        """
        pass
    
    def support_strategy_decision(self, strategy_signal: Dict[str, Any], prediction: AgentPrediction) -> AgentPrediction:
        """
        Modify agent prediction to support strategy decisions instead of overriding them.
        
        Args:
            strategy_signal: Strategy signal from synergy detection
            prediction: Agent's original prediction
            
        Returns:
            Modified prediction that supports strategy
        """
        try:
            # Extract strategy direction and confidence
            strategy_action = strategy_signal.get('action', 'hold')
            strategy_confidence = strategy_signal.get('confidence', 0.5)
            
            # Only modify if strategy has high confidence
            if strategy_confidence > 0.7:
                # Modify action probabilities to support strategy
                new_probs = prediction.action_probabilities.copy()
                
                if strategy_action == 'buy':
                    # Boost buy probabilities
                    new_probs[0] = max(new_probs[0], 0.6)  # Strong boost for buy
                    new_probs[1] = new_probs[1] * 0.5  # Reduce hold
                    new_probs[2] = new_probs[2] * 0.3  # Reduce sell
                elif strategy_action == 'sell':
                    # Boost sell probabilities
                    new_probs[0] = new_probs[0] * 0.3  # Reduce buy
                    new_probs[1] = new_probs[1] * 0.5  # Reduce hold
                    new_probs[2] = max(new_probs[2], 0.6)  # Strong boost for sell
                
                # Normalize probabilities
                new_probs = new_probs / np.sum(new_probs)
                
                # Update prediction
                prediction.action_probabilities = new_probs
                prediction.confidence = min(1.0, prediction.confidence + strategy_confidence * 0.3)
                
                # Add strategy support metadata
                prediction.internal_state['strategy_support_active'] = True
                prediction.internal_state['strategy_action'] = strategy_action
                prediction.internal_state['strategy_confidence'] = strategy_confidence
                
                self.logger.info(f"Agent {self.name} supporting strategy decision: {strategy_action}")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in strategy support for {self.name}: {e}")
            return prediction
    
    @abstractmethod
    def get_feature_importance(
        self,
        matrix_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate feature importance for the current prediction.
        
        Args:
            matrix_data: Input matrix data
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def extract_features(self, matrix_data: np.ndarray) -> np.ndarray:
        """
        Extract agent-specific features from the full matrix.
        
        Args:
            matrix_data: Full matrix data (48, 13)
            
        Returns:
            Agent-specific feature matrix (48, len(feature_indices))
        """
        return matrix_data[:, self.feature_indices]
    
    def preprocess_features(self, features: np.ndarray) -> torch.Tensor:
        """
        Preprocess features for model input.
        
        Args:
            features: Raw feature matrix
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Basic preprocessing - subclasses can override
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Add batch dimension if needed
        if features_tensor.dim() == 2:
            features_tensor = features_tensor.unsqueeze(0)
        
        return features_tensor
    
    def update_health_status(self, success: bool, error: Optional[Exception] = None) -> None:
        """Update agent health status based on operation result."""
        if success:
            self.consecutive_errors = 0
            self.is_healthy = True
            self.last_error = None
        else:
            self.consecutive_errors += 1
            self.last_error = error
            
            # Mark unhealthy after 3 consecutive errors
            if self.consecutive_errors >= 3:
                self.is_healthy = False
                self.logger.error(
                    f"Agent {self.name} marked as unhealthy: "
                    f"consecutive_errors={self.consecutive_errors}, "
                    f"last_error={str(error) if error else 'Unknown'}"
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and performance metrics."""
        avg_computation_time = (
            self.total_computation_time_ms / max(1, self.prediction_count)
        )
        
        return {
            'name': self.name,
            'is_healthy': self.is_healthy,
            'feature_indices': self.feature_indices,
            'prediction_count': self.prediction_count,
            'avg_computation_time_ms': avg_computation_time,
            'consecutive_errors': self.consecutive_errors,
            'last_error': str(self.last_error) if self.last_error else None,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None
        }


class MLMIStrategicAgent(StrategicAgentBase):
    """
    Market Liquidity and Market Impact Strategic Agent.
    
    This agent specializes in:
    - Market liquidity analysis
    - Market impact prediction
    - Order flow dynamics
    - Microstructure patterns
    
    Features used: [0, 1, 9, 10] (mlmi_value, mlmi_signal, momentum_20, momentum_50)
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        feature_indices = config.get('feature_indices', {}).get('mlmi_expert', [0, 1, 9, 10])
        super().__init__("MLMI", feature_indices, config, device)
        
        # MLMI-specific configuration
        self.hidden_dims = config.get('agents', {}).get('mlmi_expert', {}).get('hidden_dims', [256, 128, 64])
        self.dropout_rate = config.get('agents', {}).get('mlmi_expert', {}).get('dropout_rate', 0.1)
    
    async def initialize(self) -> None:
        """Initialize MLMI agent models and preprocessing."""
        try:
            # Initialize actor network for action prediction
            self.actor_network = self._build_actor_network()
            
            # Initialize critic network for value estimation
            self.critic_network = self._build_critic_network()
            
            # Load pre-trained weights if available
            await self._load_pretrained_weights()
            
            self.logger.info("MLMI Strategic Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MLMI agent: {e}")
            self.update_health_status(False, e)
            raise
    
    def _build_actor_network(self) -> nn.Module:
        """Build the actor network for action prediction."""
        input_dim = len(self.feature_indices) * 48  # Flattened features
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer: 3 actions (buy, hold, sell)
        layers.append(nn.Linear(prev_dim, 3))
        layers.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*layers).to(self.device)
    
    def _build_critic_network(self) -> nn.Module:
        """Build the critic network for value estimation."""
        input_dim = len(self.feature_indices) * 48
        
        layers = []
        prev_dim = input_dim
        
        # Use larger network for critic
        critic_dims = [dim * 2 for dim in self.hidden_dims]
        
        for hidden_dim in critic_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer: single value
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers).to(self.device)
    
    async def _load_pretrained_weights(self) -> None:
        """Load pre-trained weights if available."""
        try:
            model_path = "/home/QuantNova/GrandModel/exports/strategic_mappo_model.pth"
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract MLMI-specific weights if they exist
            if 'mlmi_actor' in checkpoint:
                self.actor_network.load_state_dict(checkpoint['mlmi_actor'])
                self.logger.info("Loaded pre-trained MLMI actor weights")
            
            if 'mlmi_critic' in checkpoint:
                self.critic_network.load_state_dict(checkpoint['mlmi_critic'])
                self.logger.info("Loaded pre-trained MLMI critic weights")
                
        except FileNotFoundError:
            self.logger.info("No pre-trained weights found, using random initialization")
            # Initialize with strategy-supporting weights
            self._initialize_strategy_supporting_weights()
        except Exception as e:
            self.logger.warning(f"Failed to load pre-trained weights: {e}")
            # Initialize with strategy-supporting weights
            self._initialize_strategy_supporting_weights()
    
    def _initialize_strategy_supporting_weights(self) -> None:
        """Initialize weights that favor strategy support over independent decisions."""
        try:
            # Initialize networks with bias toward supporting strategy decisions
            for module in self.actor_network.modules():
                if isinstance(module, nn.Linear):
                    # Initialize with smaller weights to reduce agent autonomy
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
            
            # Initialize final layer to favor middle (hold) action initially
            if hasattr(self.actor_network, '_modules'):
                final_layer = None
                for module in reversed(list(self.actor_network.modules())):
                    if isinstance(module, nn.Linear):
                        final_layer = module
                        break
                
                if final_layer is not None:
                    # Bias toward neutral/hold action to allow strategy to dominate
                    with torch.no_grad():
                        final_layer.bias[1] += 0.1  # Slight bias toward hold
            
            self.logger.info(f"Initialized {self.name} with strategy-supporting weights")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy-supporting weights: {e}")
    
    async def predict(
        self,
        matrix_data: np.ndarray,
        shared_context: Dict[str, Any]
    ) -> AgentPrediction:
        """Make MLMI-based strategic prediction."""
        start_time = datetime.now()
        
        try:
            # Extract MLMI-specific features
            features = self.extract_features(matrix_data)
            
            # Preprocess for model input
            features_tensor = self.preprocess_features(features)
            
            # Flatten for neural network
            flattened_features = features_tensor.flatten(start_dim=1)
            
            # Generate action probabilities
            with torch.no_grad():
                action_logits = self.actor_network(flattened_features)
                action_probs = action_logits.cpu().numpy().flatten()
            
            # Calculate confidence based on prediction certainty
            confidence = float(np.max(action_probs))
            
            # Get feature importance
            feature_importance = self.get_feature_importance(matrix_data)
            
            # Calculate computation time
            computation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update tracking
            self.prediction_count += 1
            self.total_computation_time_ms += computation_time_ms
            self.last_prediction_time = datetime.now()
            self.update_health_status(True)
            
            return AgentPrediction(
                action_probabilities=action_probs,
                confidence=confidence,
                feature_importance=feature_importance,
                internal_state={
                    'market_liquidity_score': float(np.mean(features[:, 0])),  # mlmi_value
                    'impact_signal': float(np.mean(features[:, 1])),           # mlmi_signal
                    'momentum_short': float(np.mean(features[:, 2])),          # momentum_20
                    'momentum_long': float(np.mean(features[:, 3]))            # momentum_50
                },
                computation_time_ms=computation_time_ms,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"MLMI prediction failed: {e}")
            self.update_health_status(False, e)
            
            # Return fallback prediction
            return self._get_fallback_prediction(start_time)
    
    def get_feature_importance(self, matrix_data: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for MLMI features."""
        features = self.extract_features(matrix_data)
        
        # Simple variance-based importance (can be enhanced with gradients)
        feature_names = ['mlmi_value', 'mlmi_signal', 'momentum_20', 'momentum_50']
        importance = {}
        
        for i, name in enumerate(feature_names):
            variance = float(np.var(features[:, i]))
            importance[name] = variance
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _get_fallback_prediction(self, start_time: datetime) -> AgentPrediction:
        """Generate fallback prediction when main prediction fails."""
        computation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return AgentPrediction(
            action_probabilities=np.array([0.33, 0.34, 0.33]),  # Neutral
            confidence=0.5,
            feature_importance={'fallback': 1.0},
            internal_state={'fallback_mode': True},
            computation_time_ms=computation_time_ms,
            timestamp=datetime.now()
        )


class NWRQKStrategicAgent(StrategicAgentBase):
    """
    Network-Weighted Risk-Quality Kernel Strategic Agent.
    
    This agent specializes in:
    - Network topology analysis
    - Risk-quality kernel computations
    - Inter-market relationships
    - Quality-adjusted risk metrics
    
    Features used: [2, 3, 4, 5] (nwrqk_value, nwrqk_slope, lvn_distance, lvn_strength)
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        feature_indices = config.get('feature_indices', {}).get('nwrqk_expert', [2, 3, 4, 5])
        super().__init__("NWRQK", feature_indices, config, device)
        
        # NWRQK-specific configuration
        self.hidden_dims = config.get('agents', {}).get('nwrqk_expert', {}).get('hidden_dims', [256, 128, 64])
        self.dropout_rate = config.get('agents', {}).get('nwrqk_expert', {}).get('dropout_rate', 0.1)
    
    async def initialize(self) -> None:
        """Initialize NWRQK agent models and preprocessing."""
        try:
            # Initialize networks (similar structure to MLMI)
            self.actor_network = self._build_actor_network()
            self.critic_network = self._build_critic_network()
            
            # Load pre-trained weights if available
            await self._load_pretrained_weights()
            
            self.logger.info("NWRQK Strategic Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NWRQK agent: {e}")
            self.update_health_status(False, e)
            raise
    
    def _build_actor_network(self) -> nn.Module:
        """Build the actor network for NWRQK predictions."""
        input_dim = len(self.feature_indices) * 48
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 3))
        layers.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*layers).to(self.device)
    
    def _build_critic_network(self) -> nn.Module:
        """Build the critic network for value estimation."""
        input_dim = len(self.feature_indices) * 48
        
        layers = []
        prev_dim = input_dim
        
        critic_dims = [dim * 2 for dim in self.hidden_dims]
        
        for hidden_dim in critic_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers).to(self.device)
    
    async def _load_pretrained_weights(self) -> None:
        """Load pre-trained weights if available."""
        try:
            model_path = "/home/QuantNova/GrandModel/exports/strategic_mappo_model.pth"
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'nwrqk_actor' in checkpoint:
                self.actor_network.load_state_dict(checkpoint['nwrqk_actor'])
                self.logger.info("Loaded pre-trained NWRQK actor weights")
            
            if 'nwrqk_critic' in checkpoint:
                self.critic_network.load_state_dict(checkpoint['nwrqk_critic'])
                self.logger.info("Loaded pre-trained NWRQK critic weights")
                
        except FileNotFoundError:
            self.logger.info("No pre-trained weights found, using random initialization")
            # Initialize with strategy-supporting weights
            self._initialize_strategy_supporting_weights()
        except Exception as e:
            self.logger.warning(f"Failed to load pre-trained weights: {e}")
            # Initialize with strategy-supporting weights
            self._initialize_strategy_supporting_weights()
    
    async def predict(
        self,
        matrix_data: np.ndarray,
        shared_context: Dict[str, Any]
    ) -> AgentPrediction:
        """Make NWRQK-based strategic prediction."""
        start_time = datetime.now()
        
        try:
            # Extract NWRQK-specific features
            features = self.extract_features(matrix_data)
            
            # Preprocess for model input
            features_tensor = self.preprocess_features(features)
            flattened_features = features_tensor.flatten(start_dim=1)
            
            # Generate action probabilities
            with torch.no_grad():
                action_logits = self.actor_network(flattened_features)
                action_probs = action_logits.cpu().numpy().flatten()
            
            confidence = float(np.max(action_probs))
            feature_importance = self.get_feature_importance(matrix_data)
            
            computation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update tracking
            self.prediction_count += 1
            self.total_computation_time_ms += computation_time_ms
            self.last_prediction_time = datetime.now()
            self.update_health_status(True)
            
            return AgentPrediction(
                action_probabilities=action_probs,
                confidence=confidence,
                feature_importance=feature_importance,
                internal_state={
                    'nwrqk_value': float(np.mean(features[:, 0])),
                    'nwrqk_slope': float(np.mean(features[:, 1])),
                    'lvn_distance': float(np.mean(features[:, 2])),
                    'lvn_strength': float(np.mean(features[:, 3]))
                },
                computation_time_ms=computation_time_ms,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"NWRQK prediction failed: {e}")
            self.update_health_status(False, e)
            return self._get_fallback_prediction(start_time)
    
    def get_feature_importance(self, matrix_data: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for NWRQK features."""
        features = self.extract_features(matrix_data)
        feature_names = ['nwrqk_value', 'nwrqk_slope', 'lvn_distance', 'lvn_strength']
        
        importance = {}
        for i, name in enumerate(feature_names):
            variance = float(np.var(features[:, i]))
            importance[name] = variance
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _get_fallback_prediction(self, start_time: datetime) -> AgentPrediction:
        """Generate fallback prediction when main prediction fails."""
        computation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return AgentPrediction(
            action_probabilities=np.array([0.33, 0.34, 0.33]),
            confidence=0.5,
            feature_importance={'fallback': 1.0},
            internal_state={'fallback_mode': True},
            computation_time_ms=computation_time_ms,
            timestamp=datetime.now()
        )


class RegimeDetectionAgent(StrategicAgentBase):
    """
    Market Regime Detection Strategic Agent.
    
    This agent specializes in:
    - Market regime identification
    - Volatility regime analysis
    - Structural break detection
    - Adaptive strategy selection
    
    Features used: [10, 11, 12] (mmd_score, volatility_30, volume_profile_skew)
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        feature_indices = config.get('feature_indices', {}).get('regime_expert', [10, 11, 12])
        super().__init__("Regime", feature_indices, config, device)
        
        # Regime-specific configuration
        self.hidden_dims = config.get('agents', {}).get('regime_expert', {}).get('hidden_dims', [256, 128, 64])
        self.dropout_rate = config.get('agents', {}).get('regime_expert', {}).get('dropout_rate', 0.15)
    
    async def initialize(self) -> None:
        """Initialize Regime Detection agent models."""
        try:
            self.actor_network = self._build_actor_network()
            self.critic_network = self._build_critic_network()
            
            await self._load_pretrained_weights()
            
            self.logger.info("Regime Detection Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Regime agent: {e}")
            self.update_health_status(False, e)
            raise
    
    def _build_actor_network(self) -> nn.Module:
        """Build the actor network for regime-based predictions."""
        input_dim = len(self.feature_indices) * 48
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 3))
        layers.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*layers).to(self.device)
    
    def _build_critic_network(self) -> nn.Module:
        """Build the critic network for value estimation."""
        input_dim = len(self.feature_indices) * 48
        
        layers = []
        prev_dim = input_dim
        
        critic_dims = [dim * 2 for dim in self.hidden_dims]
        
        for hidden_dim in critic_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers).to(self.device)
    
    async def _load_pretrained_weights(self) -> None:
        """Load pre-trained weights if available."""
        try:
            model_path = "/home/QuantNova/GrandModel/exports/strategic_mappo_model.pth"
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'regime_actor' in checkpoint:
                self.actor_network.load_state_dict(checkpoint['regime_actor'])
                self.logger.info("Loaded pre-trained Regime actor weights")
            
            if 'regime_critic' in checkpoint:
                self.critic_network.load_state_dict(checkpoint['regime_critic'])
                self.logger.info("Loaded pre-trained Regime critic weights")
                
        except FileNotFoundError:
            self.logger.info("No pre-trained weights found, using random initialization")
            # Initialize with strategy-supporting weights
            self._initialize_strategy_supporting_weights()
        except Exception as e:
            self.logger.warning(f"Failed to load pre-trained weights: {e}")
            # Initialize with strategy-supporting weights
            self._initialize_strategy_supporting_weights()
    
    async def predict(
        self,
        matrix_data: np.ndarray,
        shared_context: Dict[str, Any]
    ) -> AgentPrediction:
        """Make regime-based strategic prediction."""
        start_time = datetime.now()
        
        try:
            # Extract regime-specific features
            features = self.extract_features(matrix_data)
            
            # Preprocess for model input
            features_tensor = self.preprocess_features(features)
            flattened_features = features_tensor.flatten(start_dim=1)
            
            # Generate action probabilities
            with torch.no_grad():
                action_logits = self.actor_network(flattened_features)
                action_probs = action_logits.cpu().numpy().flatten()
            
            confidence = float(np.max(action_probs))
            feature_importance = self.get_feature_importance(matrix_data)
            
            computation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update tracking
            self.prediction_count += 1
            self.total_computation_time_ms += computation_time_ms
            self.last_prediction_time = datetime.now()
            self.update_health_status(True)
            
            return AgentPrediction(
                action_probabilities=action_probs,
                confidence=confidence,
                feature_importance=feature_importance,
                internal_state={
                    'mmd_score': float(np.mean(features[:, 0])),
                    'volatility_30': float(np.mean(features[:, 1])),
                    'volume_profile_skew': float(np.mean(features[:, 2])),
                    'detected_regime': shared_context.get('market_regime', 'unknown')
                },
                computation_time_ms=computation_time_ms,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Regime prediction failed: {e}")
            self.update_health_status(False, e)
            return self._get_fallback_prediction(start_time)
    
    def get_feature_importance(self, matrix_data: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for regime features."""
        features = self.extract_features(matrix_data)
        feature_names = ['mmd_score', 'volatility_30', 'volume_profile_skew']
        
        importance = {}
        for i, name in enumerate(feature_names):
            variance = float(np.var(features[:, i]))
            importance[name] = variance
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _get_fallback_prediction(self, start_time: datetime) -> AgentPrediction:
        """Generate fallback prediction when main prediction fails."""
        computation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return AgentPrediction(
            action_probabilities=np.array([0.33, 0.34, 0.33]),
            confidence=0.5,
            feature_importance={'fallback': 1.0},
            internal_state={'fallback_mode': True},
            computation_time_ms=computation_time_ms,
            timestamp=datetime.now()
        )