"""
Sequential Tactical Agents for MARL Environment

This module implements sequential-aware tactical agents that operate in a specific order:
FVG Agent → Momentum Agent → Entry Optimization Agent. Each agent receives enriched
observations from its predecessors and strategic context from the upstream system.

Key Features:
- Sequential execution with predecessor context integration
- Strategic context awareness from upstream 30m MARL system
- Enriched observations with market microstructure
- Agent-specific specialization for different market aspects
- Byzantine fault tolerance and cryptographic validation
- High-frequency execution capability (5-minute cycles)

Architecture:
- FVG Agent: Analyzes Fair Value Gaps and price imbalances
- Momentum Agent: Evaluates trend continuation with FVG context
- Entry Optimization Agent: Optimizes timing with full predecessor context

Author: Agent 5 - Sequential Tactical MARL Specialist
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
import uuid
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import warnings

# Core imports
from src.agents.base.agent import BaseAgent
from src.core.events import Event, EventType
from src.core.event_bus import EventBus
from src.consensus.byzantine_detector import ByzantineDetector

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Sequential agent roles"""
    FVG_ANALYZER = "fvg_analyzer"
    MOMENTUM_EVALUATOR = "momentum_evaluator"
    ENTRY_OPTIMIZER = "entry_optimizer"

class ExecutionMode(Enum):
    """Agent execution modes"""
    TRAINING = "training"
    INFERENCE = "inference"
    VALIDATION = "validation"

@dataclass
class StrategicContext:
    """Strategic context from upstream MARL system"""
    regime_embedding: np.ndarray
    synergy_signal: Dict[str, Any]
    market_state: Dict[str, Any]
    confidence_level: float
    execution_bias: str
    volatility_forecast: float
    timestamp: float

@dataclass
class PredecessorContext:
    """Context from predecessor agents"""
    agent_outputs: Dict[str, Any]
    consensus_level: float
    alignment_score: float
    execution_signals: Dict[str, Any]
    feature_importance: Dict[str, float]
    timestamp: float

@dataclass
class AgentOutput:
    """Agent output with enhanced information"""
    agent_id: str
    role: AgentRole
    action: int
    probabilities: np.ndarray
    confidence: float
    feature_importance: Dict[str, float]
    market_insights: Dict[str, Any]
    execution_signals: Dict[str, Any]
    processing_time: float
    timestamp: float
    signature: Optional[str] = None

class SequentialTacticalAgent(BaseAgent):
    """
    Base class for sequential tactical agents
    
    Provides common functionality for all tactical agents including:
    - Strategic context integration
    - Predecessor context processing
    - Byzantine fault tolerance
    - Performance monitoring
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        observation_dim: int,
        action_dim: int = 3,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize sequential tactical agent
        
        Args:
            agent_id: Unique agent identifier
            role: Agent role in sequential execution
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space (default: 3 for bearish/neutral/bullish)
            config: Agent configuration
        """
        super().__init__(agent_id, config)
        
        self.role = role
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = config or self._default_config()
        
        # Execution state
        self.execution_mode = ExecutionMode.TRAINING
        self.strategic_context: Optional[StrategicContext] = None
        self.predecessor_context: Optional[PredecessorContext] = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_decisions': 0,
            'processing_times': deque(maxlen=1000),
            'confidence_scores': deque(maxlen=1000),
            'strategic_alignments': deque(maxlen=1000),
            'predecessor_alignments': deque(maxlen=1000)
        }
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.last_output: Optional[AgentOutput] = None
        
        # Initialize neural networks
        self._initialize_networks()
        
        # Initialize cryptographic components
        self._initialize_crypto()
        
        logger.info(f"Sequential tactical agent {agent_id} ({role.value}) initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for tactical agents"""
        return {
            'network': {
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.2,
                'activation': 'relu',
                'use_batch_norm': True
            },
            'training': {
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'gradient_clip_norm': 1.0,
                'target_update_frequency': 100
            },
            'execution': {
                'confidence_threshold': 0.6,
                'strategic_alignment_weight': 0.3,
                'predecessor_alignment_weight': 0.2,
                'risk_tolerance': 0.02
            },
            'performance': {
                'target_processing_time_ms': 10,
                'max_processing_time_ms': 50,
                'min_confidence_threshold': 0.4
            },
            'byzantine': {
                'enable_detection': True,
                'signature_validation': True,
                'consensus_threshold': 0.8
            }
        }
    
    def _initialize_networks(self):
        """Initialize neural networks for the agent"""
        try:
            # Main policy network
            self.policy_network = self._create_policy_network()
            
            # Value network for TD learning
            self.value_network = self._create_value_network()
            
            # Attention network for feature importance
            self.attention_network = self._create_attention_network()
            
            # Optimizers
            self.policy_optimizer = torch.optim.Adam(
                self.policy_network.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            self.value_optimizer = torch.optim.Adam(
                self.value_network.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            logger.info(f"Neural networks initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error initializing networks for agent {self.agent_id}: {e}")
            raise
    
    def _create_policy_network(self) -> nn.Module:
        """Create policy network"""
        hidden_dims = self.config['network']['hidden_dims']
        dropout_rate = self.config['network']['dropout_rate']
        use_batch_norm = self.config['network']['use_batch_norm']
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.observation_dim, hidden_dims[0]))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], self.action_dim))
        layers.append(nn.Softmax(dim=-1))
        
        return nn.Sequential(*layers)
    
    def _create_value_network(self) -> nn.Module:
        """Create value network"""
        hidden_dims = self.config['network']['hidden_dims']
        dropout_rate = self.config['network']['dropout_rate']
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.observation_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (single value)
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        return nn.Sequential(*layers)
    
    def _create_attention_network(self) -> nn.Module:
        """Create attention network for feature importance"""
        hidden_dim = self.config['network']['hidden_dims'][0]
        
        return nn.Sequential(
            nn.Linear(self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.observation_dim),
            nn.Softmax(dim=-1)
        )
    
    def _initialize_crypto(self):
        """Initialize cryptographic components"""
        try:
            if self.config['byzantine']['enable_detection']:
                # Initialize Byzantine detector
                self.byzantine_detector = ByzantineDetector(
                    config=self.config['byzantine']
                )
                
                # Initialize signature system
                self.signature_key = self._generate_signature_key()
            else:
                self.byzantine_detector = None
                self.signature_key = None
                
        except Exception as e:
            logger.warning(f"Could not initialize crypto for agent {self.agent_id}: {e}")
            self.byzantine_detector = None
            self.signature_key = None
    
    def _generate_signature_key(self) -> str:
        """Generate cryptographic signature key"""
        import secrets
        return secrets.token_hex(32)
    
    def select_action(
        self,
        observation: np.ndarray,
        strategic_context: Optional[StrategicContext] = None,
        predecessor_context: Optional[PredecessorContext] = None
    ) -> AgentOutput:
        """
        Select action based on observation and contexts
        
        Args:
            observation: Current observation
            strategic_context: Strategic context from upstream system
            predecessor_context: Context from predecessor agents
            
        Returns:
            AgentOutput with action and additional information
        """
        start_time = time.time()
        
        try:
            # Store contexts
            self.strategic_context = strategic_context
            self.predecessor_context = predecessor_context
            
            # Preprocess observation
            processed_obs = self._preprocess_observation(observation)
            
            # Forward pass through networks
            with torch.no_grad():
                # Policy network
                obs_tensor = torch.FloatTensor(processed_obs).unsqueeze(0)
                action_probabilities = self.policy_network(obs_tensor).squeeze(0)
                
                # Value network
                state_value = self.value_network(obs_tensor).squeeze(0).item()
                
                # Attention network
                attention_weights = self.attention_network(obs_tensor).squeeze(0)
            
            # Sample action
            if self.execution_mode == ExecutionMode.TRAINING:
                action_dist = Categorical(action_probabilities)
                action = action_dist.sample().item()
            else:
                action = torch.argmax(action_probabilities).item()
            
            # Calculate confidence
            confidence = self._calculate_confidence(action_probabilities, state_value)
            
            # Apply strategic and predecessor alignments
            confidence = self._apply_context_alignments(confidence, action)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(attention_weights)
            
            # Generate market insights
            market_insights = self._generate_market_insights(processed_obs, action)
            
            # Create execution signals
            execution_signals = self._create_execution_signals(action, confidence)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Create output
            output = AgentOutput(
                agent_id=self.agent_id,
                role=self.role,
                action=action,
                probabilities=action_probabilities.numpy(),
                confidence=confidence,
                feature_importance=feature_importance,
                market_insights=market_insights,
                execution_signals=execution_signals,
                processing_time=processing_time,
                timestamp=time.time()
            )
            
            # Add cryptographic signature
            if self.config['byzantine']['signature_validation']:
                output.signature = self._sign_output(output)
            
            # Update performance metrics
            self._update_performance_metrics(output)
            
            # Store last output
            self.last_output = output
            
            return output
            
        except Exception as e:
            logger.error(f"Error in action selection for agent {self.agent_id}: {e}")
            return self._create_safe_output()
    
    def _preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        """Preprocess observation for neural network"""
        try:
            # Normalize observation
            obs_normalized = (observation - np.mean(observation)) / (np.std(observation) + 1e-8)
            
            # Clip extreme values
            obs_clipped = np.clip(obs_normalized, -5.0, 5.0)
            
            # Handle NaN values
            obs_clean = np.nan_to_num(obs_clipped, nan=0.0, posinf=5.0, neginf=-5.0)
            
            return obs_clean.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preprocessing observation: {e}")
            return np.zeros(self.observation_dim, dtype=np.float32)
    
    def _calculate_confidence(self, action_probabilities: torch.Tensor, state_value: float) -> float:
        """Calculate confidence score"""
        try:
            # Entropy-based confidence
            entropy = -torch.sum(action_probabilities * torch.log(action_probabilities + 1e-8))
            max_entropy = np.log(self.action_dim)
            entropy_confidence = 1.0 - (entropy / max_entropy)
            
            # Probability-based confidence
            prob_confidence = torch.max(action_probabilities).item()
            
            # Value-based confidence
            value_confidence = torch.sigmoid(torch.tensor(state_value)).item()
            
            # Combined confidence
            confidence = (
                0.5 * entropy_confidence +
                0.3 * prob_confidence +
                0.2 * value_confidence
            )
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _apply_context_alignments(self, base_confidence: float, action: int) -> float:
        """Apply strategic and predecessor alignment adjustments"""
        try:
            adjusted_confidence = base_confidence
            
            # Strategic alignment
            if self.strategic_context:
                strategic_boost = self._calculate_strategic_alignment(action)
                weight = self.config['execution']['strategic_alignment_weight']
                adjusted_confidence += strategic_boost * weight
            
            # Predecessor alignment
            if self.predecessor_context:
                predecessor_boost = self._calculate_predecessor_alignment(action)
                weight = self.config['execution']['predecessor_alignment_weight']
                adjusted_confidence += predecessor_boost * weight
            
            return float(np.clip(adjusted_confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error applying context alignments: {e}")
            return base_confidence
    
    def _calculate_strategic_alignment(self, action: int) -> float:
        """Calculate strategic alignment boost"""
        try:
            if not self.strategic_context:
                return 0.0
            
            # Map action to bias
            action_bias = ['bearish', 'neutral', 'bullish'][action]
            
            # Calculate alignment
            if action_bias == self.strategic_context.execution_bias:
                return 0.2 * self.strategic_context.confidence_level
            elif self.strategic_context.execution_bias == 'neutral':
                return 0.1 * self.strategic_context.confidence_level
            else:
                return -0.1 * self.strategic_context.confidence_level
                
        except Exception as e:
            logger.error(f"Error calculating strategic alignment: {e}")
            return 0.0
    
    def _calculate_predecessor_alignment(self, action: int) -> float:
        """Calculate predecessor alignment boost"""
        try:
            if not self.predecessor_context:
                return 0.0
            
            alignment_boost = 0.0
            
            for agent_id, output in self.predecessor_context.agent_outputs.items():
                pred_action = output.get('action', 1)
                pred_confidence = output.get('confidence', 0.5)
                
                if pred_action == action:
                    alignment_boost += 0.1 * pred_confidence
                elif abs(pred_action - action) == 1:  # Adjacent actions
                    alignment_boost += 0.05 * pred_confidence
            
            return min(0.2, alignment_boost)
            
        except Exception as e:
            logger.error(f"Error calculating predecessor alignment: {e}")
            return 0.0
    
    def _calculate_feature_importance(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """Calculate feature importance from attention weights"""
        try:
            importance = {}
            
            # Extract attention weights for different feature groups
            total_dim = len(attention_weights)
            
            # Assuming feature structure: [market_matrix, strategic_context, predecessor_context, microstructure]
            market_matrix_dim = 60 * 7  # 420
            strategic_dim = 64 + 12  # regime_embedding + synergy_signal
            
            if total_dim >= market_matrix_dim:
                importance['market_matrix'] = float(torch.sum(attention_weights[:market_matrix_dim]))
                
            if total_dim >= market_matrix_dim + strategic_dim:
                importance['strategic_context'] = float(torch.sum(attention_weights[market_matrix_dim:market_matrix_dim + strategic_dim]))
                
            if total_dim > market_matrix_dim + strategic_dim:
                importance['predecessor_context'] = float(torch.sum(attention_weights[market_matrix_dim + strategic_dim:market_matrix_dim + strategic_dim + 100]))
                
            if total_dim > market_matrix_dim + strategic_dim + 100:
                importance['microstructure'] = float(torch.sum(attention_weights[market_matrix_dim + strategic_dim + 100:]))
            
            # Normalize importance scores
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {'default': 1.0}
    
    def _generate_market_insights(self, observation: np.ndarray, action: int) -> Dict[str, Any]:
        """Generate market insights based on observation and action"""
        # This is agent-specific and will be overridden in subclasses
        return {
            'market_condition': 'normal',
            'action_rationale': f'Selected action {action}',
            'confidence_factors': ['observation_quality', 'strategic_alignment'],
            'risk_assessment': 'moderate'
        }
    
    def _create_execution_signals(self, action: int, confidence: float) -> Dict[str, Any]:
        """Create execution signals"""
        # This is agent-specific and will be overridden in subclasses
        return {
            'primary_signal': action,
            'signal_strength': confidence,
            'execution_urgency': 'medium',
            'position_size_modifier': 1.0
        }
    
    def _sign_output(self, output: AgentOutput) -> str:
        """Create cryptographic signature for output"""
        try:
            if not self.signature_key:
                return ""
            
            # Create message content
            message_content = {
                'agent_id': output.agent_id,
                'action': output.action,
                'confidence': output.confidence,
                'timestamp': output.timestamp
            }
            
            # Simple signature (in practice, use proper cryptographic signing)
            import hashlib
            import hmac
            
            message_str = json.dumps(message_content, sort_keys=True)
            signature = hmac.new(
                self.signature_key.encode(),
                message_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"Error signing output: {e}")
            return ""
    
    def _update_performance_metrics(self, output: AgentOutput):
        """Update performance metrics"""
        try:
            self.performance_metrics['total_decisions'] += 1
            self.performance_metrics['processing_times'].append(output.processing_time)
            self.performance_metrics['confidence_scores'].append(output.confidence)
            
            # Strategic alignment
            if self.strategic_context:
                alignment = self._calculate_strategic_alignment(output.action)
                self.performance_metrics['strategic_alignments'].append(alignment)
            
            # Predecessor alignment
            if self.predecessor_context:
                alignment = self._calculate_predecessor_alignment(output.action)
                self.performance_metrics['predecessor_alignments'].append(alignment)
                
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _create_safe_output(self) -> AgentOutput:
        """Create safe default output"""
        return AgentOutput(
            agent_id=self.agent_id,
            role=self.role,
            action=1,  # Neutral
            probabilities=np.array([0.33, 0.34, 0.33], dtype=np.float32),
            confidence=0.0,
            feature_importance={'default': 1.0},
            market_insights={'status': 'safe_default'},
            execution_signals={'status': 'safe_default'},
            processing_time=0.0,
            timestamp=time.time()
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            metrics = {
                'total_decisions': self.performance_metrics['total_decisions'],
                'avg_processing_time': np.mean(self.performance_metrics['processing_times']) if self.performance_metrics['processing_times'] else 0.0,
                'p95_processing_time': np.percentile(self.performance_metrics['processing_times'], 95) if self.performance_metrics['processing_times'] else 0.0,
                'avg_confidence': np.mean(self.performance_metrics['confidence_scores']) if self.performance_metrics['confidence_scores'] else 0.0,
                'avg_strategic_alignment': np.mean(self.performance_metrics['strategic_alignments']) if self.performance_metrics['strategic_alignments'] else 0.0,
                'avg_predecessor_alignment': np.mean(self.performance_metrics['predecessor_alignments']) if self.performance_metrics['predecessor_alignments'] else 0.0,
                'session_id': self.session_id
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def train_step(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Single training step"""
        try:
            # Extract experience
            observation = torch.FloatTensor(experience['observation'])
            action = torch.LongTensor([experience['action']])
            reward = torch.FloatTensor([experience['reward']])
            next_observation = torch.FloatTensor(experience['next_observation'])
            done = torch.BoolTensor([experience['done']])
            
            # Policy network forward pass
            action_probs = self.policy_network(observation.unsqueeze(0))
            action_dist = Categorical(action_probs)
            
            # Value network forward pass
            state_value = self.value_network(observation.unsqueeze(0))
            next_state_value = self.value_network(next_observation.unsqueeze(0))
            
            # Calculate targets
            td_target = reward + 0.99 * next_state_value * (~done)
            advantage = td_target - state_value
            
            # Policy loss
            policy_loss = -action_dist.log_prob(action) * advantage.detach()
            
            # Value loss
            value_loss = F.mse_loss(state_value, td_target.detach())
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(),
                self.config['training']['gradient_clip_norm']
            )
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_network.parameters(),
                self.config['training']['gradient_clip_norm']
            )
            self.value_optimizer.step()
            
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'advantage': advantage.item()
            }
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'advantage': 0.0}
    
    def set_execution_mode(self, mode: ExecutionMode):
        """Set execution mode"""
        self.execution_mode = mode
        
        if mode == ExecutionMode.TRAINING:
            self.policy_network.train()
            self.value_network.train()
        else:
            self.policy_network.eval()
            self.value_network.eval()
    
    def save_model(self, filepath: str):
        """Save model state"""
        try:
            state = {
                'agent_id': self.agent_id,
                'role': self.role.value,
                'policy_network': self.policy_network.state_dict(),
                'value_network': self.value_network.state_dict(),
                'attention_network': self.attention_network.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict(),
                'config': self.config,
                'performance_metrics': self.performance_metrics
            }
            
            torch.save(state, filepath)
            logger.info(f"Model saved for agent {self.agent_id} at {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load model state"""
        try:
            state = torch.load(filepath, map_location='cpu')
            
            self.policy_network.load_state_dict(state['policy_network'])
            self.value_network.load_state_dict(state['value_network'])
            self.attention_network.load_state_dict(state['attention_network'])
            self.policy_optimizer.load_state_dict(state['policy_optimizer'])
            self.value_optimizer.load_state_dict(state['value_optimizer'])
            
            logger.info(f"Model loaded for agent {self.agent_id} from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")


class FVGTacticalAgent(SequentialTacticalAgent):
    """
    FVG (Fair Value Gap) Tactical Agent
    
    Specializes in analyzing Fair Value Gaps and price imbalances.
    First agent in the sequential execution chain.
    """
    
    def __init__(self, observation_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="fvg_agent",
            role=AgentRole.FVG_ANALYZER,
            observation_dim=observation_dim,
            config=config
        )
    
    def _generate_market_insights(self, observation: np.ndarray, action: int) -> Dict[str, Any]:
        """Generate FVG-specific market insights"""
        try:
            # Extract FVG-related features from observation
            # Assuming first part of observation is the market matrix
            market_matrix = observation[:420].reshape(60, 7)  # 60x7 matrix
            
            # FVG analysis
            fvg_bullish = np.mean(market_matrix[:, 0])  # FVG bullish active
            fvg_bearish = np.mean(market_matrix[:, 1])  # FVG bearish active
            fvg_nearest_level = np.mean(market_matrix[:, 2])  # FVG nearest level
            fvg_age = np.mean(market_matrix[:, 3])  # FVG age
            
            # Gap strength analysis
            gap_strength = abs(fvg_nearest_level)
            gap_direction = "bullish" if fvg_bullish > fvg_bearish else "bearish"
            gap_quality = "high" if gap_strength > 0.5 else "medium" if gap_strength > 0.2 else "low"
            
            insights = {
                'fvg_analysis': {
                    'bullish_gaps': float(fvg_bullish),
                    'bearish_gaps': float(fvg_bearish),
                    'nearest_level': float(fvg_nearest_level),
                    'average_age': float(fvg_age),
                    'gap_strength': float(gap_strength),
                    'gap_direction': gap_direction,
                    'gap_quality': gap_quality
                },
                'market_condition': gap_quality,
                'action_rationale': f'Gap analysis suggests {["bearish", "neutral", "bullish"][action]} bias',
                'confidence_factors': ['gap_strength', 'gap_age', 'gap_direction'],
                'risk_assessment': 'low' if gap_quality == 'high' else 'moderate'
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating FVG market insights: {e}")
            return super()._generate_market_insights(observation, action)
    
    def _create_execution_signals(self, action: int, confidence: float) -> Dict[str, Any]:
        """Create FVG-specific execution signals"""
        try:
            # Base signals
            signals = {
                'primary_signal': action,
                'signal_strength': confidence,
                'execution_urgency': 'medium',
                'position_size_modifier': 1.0
            }
            
            # FVG-specific signals
            if self.last_output and 'fvg_analysis' in self.last_output.market_insights:
                fvg_data = self.last_output.market_insights['fvg_analysis']
                
                signals.update({
                    'gap_probability': confidence,
                    'gap_strength': fvg_data.get('gap_strength', 0.5),
                    'gap_duration': max(1.0, fvg_data.get('average_age', 2.0)),
                    'gap_direction': fvg_data.get('gap_direction', 'neutral'),
                    'gap_quality': fvg_data.get('gap_quality', 'medium')
                })
                
                # Adjust urgency based on gap quality
                if fvg_data.get('gap_quality') == 'high':
                    signals['execution_urgency'] = 'high'
                    signals['position_size_modifier'] = 1.2
                elif fvg_data.get('gap_quality') == 'low':
                    signals['execution_urgency'] = 'low'
                    signals['position_size_modifier'] = 0.8
            
            return signals
            
        except Exception as e:
            logger.error(f"Error creating FVG execution signals: {e}")
            return super()._create_execution_signals(action, confidence)


class MomentumTacticalAgent(SequentialTacticalAgent):
    """
    Momentum Tactical Agent
    
    Specializes in evaluating price momentum and trend continuation.
    Second agent in the sequential execution chain.
    Receives FVG context from predecessor.
    """
    
    def __init__(self, observation_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="momentum_agent",
            role=AgentRole.MOMENTUM_EVALUATOR,
            observation_dim=observation_dim,
            config=config
        )
    
    def _generate_market_insights(self, observation: np.ndarray, action: int) -> Dict[str, Any]:
        """Generate momentum-specific market insights"""
        try:
            # Extract momentum-related features from observation
            market_matrix = observation[:420].reshape(60, 7)  # 60x7 matrix
            
            # Momentum analysis
            price_momentum = market_matrix[:, 5]  # Price momentum column
            volume_ratio = market_matrix[:, 6]  # Volume ratio column
            
            # Calculate momentum indicators
            momentum_strength = np.std(price_momentum)
            momentum_direction = "bullish" if np.mean(price_momentum[-10:]) > 0 else "bearish"
            momentum_persistence = self._calculate_momentum_persistence(price_momentum)
            trend_quality = self._assess_trend_quality(price_momentum, volume_ratio)
            
            insights = {
                'momentum_analysis': {
                    'strength': float(momentum_strength),
                    'direction': momentum_direction,
                    'persistence': float(momentum_persistence),
                    'recent_change': float(np.mean(price_momentum[-5:])),
                    'volatility': float(np.std(price_momentum)),
                    'trend_quality': trend_quality
                },
                'volume_analysis': {
                    'average_ratio': float(np.mean(volume_ratio)),
                    'volume_trend': "increasing" if np.mean(volume_ratio[-5:]) > np.mean(volume_ratio[-10:-5]) else "decreasing",
                    'volume_momentum_alignment': self._check_volume_momentum_alignment(price_momentum, volume_ratio)
                },
                'market_condition': trend_quality,
                'action_rationale': f'Momentum analysis suggests {["bearish", "neutral", "bullish"][action]} continuation',
                'confidence_factors': ['momentum_strength', 'trend_quality', 'volume_alignment'],
                'risk_assessment': 'low' if trend_quality == 'strong' else 'moderate'
            }
            
            # Integrate FVG context if available
            if self.predecessor_context and 'fvg_agent' in self.predecessor_context.agent_outputs:
                fvg_output = self.predecessor_context.agent_outputs['fvg_agent']
                insights['fvg_momentum_alignment'] = self._assess_fvg_momentum_alignment(
                    fvg_output, momentum_direction
                )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating momentum market insights: {e}")
            return super()._generate_market_insights(observation, action)
    
    def _calculate_momentum_persistence(self, momentum_data: np.ndarray) -> float:
        """Calculate momentum persistence score"""
        try:
            if len(momentum_data) < 10:
                return 0.5
            
            # Calculate directional consistency
            positive_periods = np.sum(momentum_data > 0)
            negative_periods = np.sum(momentum_data < 0)
            total_periods = len(momentum_data)
            
            persistence = max(positive_periods, negative_periods) / total_periods
            return persistence
            
        except Exception:
            return 0.5
    
    def _assess_trend_quality(self, momentum_data: np.ndarray, volume_data: np.ndarray) -> str:
        """Assess overall trend quality"""
        try:
            persistence = self._calculate_momentum_persistence(momentum_data)
            strength = np.std(momentum_data)
            volume_support = np.corrcoef(np.abs(momentum_data), volume_data)[0, 1]
            
            if persistence > 0.7 and strength > 0.3 and volume_support > 0.3:
                return "strong"
            elif persistence > 0.6 and strength > 0.2:
                return "moderate"
            else:
                return "weak"
                
        except Exception:
            return "weak"
    
    def _check_volume_momentum_alignment(self, momentum_data: np.ndarray, volume_data: np.ndarray) -> str:
        """Check if volume supports momentum"""
        try:
            correlation = np.corrcoef(np.abs(momentum_data), volume_data)[0, 1]
            
            if correlation > 0.5:
                return "strong_alignment"
            elif correlation > 0.2:
                return "moderate_alignment"
            else:
                return "weak_alignment"
                
        except Exception:
            return "weak_alignment"
    
    def _assess_fvg_momentum_alignment(self, fvg_output: Dict[str, Any], momentum_direction: str) -> str:
        """Assess alignment between FVG and momentum signals"""
        try:
            fvg_action = fvg_output.get('action', 1)
            fvg_direction = ['bearish', 'neutral', 'bullish'][fvg_action]
            
            if fvg_direction == momentum_direction:
                return "strong_alignment"
            elif fvg_direction == 'neutral' or momentum_direction == 'neutral':
                return "moderate_alignment"
            else:
                return "divergence"
                
        except Exception:
            return "unknown"
    
    def _create_execution_signals(self, action: int, confidence: float) -> Dict[str, Any]:
        """Create momentum-specific execution signals"""
        try:
            signals = {
                'primary_signal': action,
                'signal_strength': confidence,
                'execution_urgency': 'medium',
                'position_size_modifier': 1.0
            }
            
            # Momentum-specific signals
            if self.last_output and 'momentum_analysis' in self.last_output.market_insights:
                momentum_data = self.last_output.market_insights['momentum_analysis']
                
                signals.update({
                    'trend_probability': confidence,
                    'trend_strength': momentum_data.get('strength', 0.5),
                    'trend_duration': max(2.0, momentum_data.get('persistence', 0.5) * 10.0),
                    'trend_direction': momentum_data.get('direction', 'neutral'),
                    'trend_quality': momentum_data.get('trend_quality', 'moderate')
                })
                
                # Adjust based on trend quality
                if momentum_data.get('trend_quality') == 'strong':
                    signals['execution_urgency'] = 'high'
                    signals['position_size_modifier'] = 1.3
                elif momentum_data.get('trend_quality') == 'weak':
                    signals['execution_urgency'] = 'low'
                    signals['position_size_modifier'] = 0.7
            
            return signals
            
        except Exception as e:
            logger.error(f"Error creating momentum execution signals: {e}")
            return super()._create_execution_signals(action, confidence)


class EntryOptimizationAgent(SequentialTacticalAgent):
    """
    Entry Optimization Agent
    
    Specializes in optimizing entry timing and execution parameters.
    Third and final agent in the sequential execution chain.
    Receives context from both FVG and Momentum agents.
    """
    
    def __init__(self, observation_dim: int, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="entry_opt_agent",
            role=AgentRole.ENTRY_OPTIMIZER,
            observation_dim=observation_dim,
            config=config
        )
    
    def _generate_market_insights(self, observation: np.ndarray, action: int) -> Dict[str, Any]:
        """Generate entry optimization-specific market insights"""
        try:
            # Extract microstructure and timing features
            market_matrix = observation[:420].reshape(60, 7)  # 60x7 matrix
            
            # Entry timing analysis
            recent_volatility = np.std(market_matrix[-10:, 5])  # Recent momentum volatility
            price_stability = 1.0 / (1.0 + recent_volatility)  # Inverse of volatility
            
            # Market microstructure considerations
            entry_quality = self._assess_entry_quality(market_matrix)
            timing_score = self._calculate_timing_score(market_matrix)
            execution_difficulty = self._assess_execution_difficulty(market_matrix)
            
            insights = {
                'entry_analysis': {
                    'entry_quality': entry_quality,
                    'timing_score': float(timing_score),
                    'price_stability': float(price_stability),
                    'execution_difficulty': execution_difficulty,
                    'optimal_timing': timing_score > 0.7,
                    'risk_reward_ratio': self._calculate_risk_reward_ratio(market_matrix)
                },
                'microstructure_analysis': {
                    'bid_ask_impact': self._estimate_bid_ask_impact(),
                    'market_depth': self._estimate_market_depth(),
                    'slippage_estimate': self._estimate_slippage(),
                    'execution_cost': self._estimate_execution_cost()
                },
                'market_condition': entry_quality,
                'action_rationale': f'Entry optimization suggests {["bearish", "neutral", "bullish"][action]} entry',
                'confidence_factors': ['entry_quality', 'timing_score', 'predecessor_alignment'],
                'risk_assessment': 'low' if entry_quality == 'optimal' else 'moderate'
            }
            
            # Integrate predecessor contexts
            if self.predecessor_context:
                insights['predecessor_alignment'] = self._assess_predecessor_alignment()
                insights['consensus_level'] = self.predecessor_context.consensus_level
                insights['execution_recommendation'] = self._generate_execution_recommendation()
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating entry optimization market insights: {e}")
            return super()._generate_market_insights(observation, action)
    
    def _assess_entry_quality(self, market_matrix: np.ndarray) -> str:
        """Assess entry quality based on market conditions"""
        try:
            # Analyze recent price action
            recent_momentum = market_matrix[-5:, 5]  # Last 5 periods
            momentum_consistency = 1.0 - np.std(recent_momentum) / (np.mean(np.abs(recent_momentum)) + 1e-8)
            
            # Analyze volume support
            recent_volume = market_matrix[-5:, 6]
            volume_trend = np.mean(recent_volume[-3:]) - np.mean(recent_volume[:2])
            
            # Combined quality score
            if momentum_consistency > 0.7 and volume_trend > 0:
                return "optimal"
            elif momentum_consistency > 0.5:
                return "good"
            elif momentum_consistency > 0.3:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "fair"
    
    def _calculate_timing_score(self, market_matrix: np.ndarray) -> float:
        """Calculate entry timing score"""
        try:
            # Recent price momentum
            momentum = market_matrix[-10:, 5]
            momentum_score = np.mean(np.abs(momentum))
            
            # Volume confirmation
            volume = market_matrix[-10:, 6]
            volume_score = np.mean(volume)
            
            # Volatility consideration
            volatility = np.std(market_matrix[-10:, 5])
            volatility_score = 1.0 / (1.0 + volatility)
            
            # Combined timing score
            timing_score = 0.4 * momentum_score + 0.3 * volume_score + 0.3 * volatility_score
            
            return min(1.0, timing_score)
            
        except Exception:
            return 0.5
    
    def _assess_execution_difficulty(self, market_matrix: np.ndarray) -> str:
        """Assess execution difficulty"""
        try:
            # Volatility factor
            volatility = np.std(market_matrix[-10:, 5])
            
            # Volume factor
            volume = np.mean(market_matrix[-10:, 6])
            
            if volatility < 0.2 and volume > 0.8:
                return "easy"
            elif volatility < 0.5 and volume > 0.5:
                return "moderate"
            else:
                return "difficult"
                
        except Exception:
            return "moderate"
    
    def _calculate_risk_reward_ratio(self, market_matrix: np.ndarray) -> float:
        """Calculate risk-reward ratio"""
        try:
            # Estimate potential reward from momentum
            momentum = market_matrix[-10:, 5]
            potential_reward = np.mean(np.abs(momentum))
            
            # Estimate risk from volatility
            risk = np.std(market_matrix[-10:, 5])
            
            ratio = potential_reward / (risk + 1e-8)
            return min(5.0, ratio)
            
        except Exception:
            return 1.0
    
    def _estimate_bid_ask_impact(self) -> float:
        """Estimate bid-ask spread impact"""
        # Mock implementation
        return 0.0002  # 2 basis points
    
    def _estimate_market_depth(self) -> float:
        """Estimate market depth"""
        # Mock implementation
        return 0.8  # 80% depth score
    
    def _estimate_slippage(self) -> float:
        """Estimate execution slippage"""
        # Mock implementation
        return 0.0003  # 3 basis points
    
    def _estimate_execution_cost(self) -> float:
        """Estimate total execution cost"""
        # Mock implementation
        return 0.0005  # 5 basis points
    
    def _assess_predecessor_alignment(self) -> Dict[str, Any]:
        """Assess alignment with predecessor agents"""
        try:
            if not self.predecessor_context:
                return {'status': 'no_predecessors'}
            
            alignment_scores = {}
            actions = []
            confidences = []
            
            for agent_id, output in self.predecessor_context.agent_outputs.items():
                action = output.get('action', 1)
                confidence = output.get('confidence', 0.5)
                
                actions.append(action)
                confidences.append(confidence)
                alignment_scores[agent_id] = confidence
            
            # Calculate consensus
            if len(set(actions)) == 1:
                consensus = "strong"
            elif len(set(actions)) == 2:
                consensus = "moderate"
            else:
                consensus = "weak"
            
            return {
                'consensus_level': consensus,
                'avg_confidence': np.mean(confidences),
                'action_distribution': {str(i): actions.count(i) for i in range(3)},
                'alignment_scores': alignment_scores
            }
            
        except Exception as e:
            logger.error(f"Error assessing predecessor alignment: {e}")
            return {'status': 'error'}
    
    def _generate_execution_recommendation(self) -> Dict[str, Any]:
        """Generate execution recommendation based on all contexts"""
        try:
            recommendation = {
                'execute': False,
                'confidence': 0.5,
                'position_size': 1.0,
                'urgency': 'medium',
                'risk_level': 'moderate'
            }
            
            # Analyze predecessor alignment
            if self.predecessor_context:
                alignment = self._assess_predecessor_alignment()
                
                if alignment['consensus_level'] == 'strong' and alignment['avg_confidence'] > 0.7:
                    recommendation['execute'] = True
                    recommendation['confidence'] = alignment['avg_confidence']
                    recommendation['position_size'] = 1.2
                    recommendation['urgency'] = 'high'
                    recommendation['risk_level'] = 'low'
                elif alignment['consensus_level'] == 'moderate' and alignment['avg_confidence'] > 0.6:
                    recommendation['execute'] = True
                    recommendation['confidence'] = alignment['avg_confidence'] * 0.8
                    recommendation['position_size'] = 1.0
                    recommendation['urgency'] = 'medium'
                    recommendation['risk_level'] = 'moderate'
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating execution recommendation: {e}")
            return {'execute': False, 'confidence': 0.0}
    
    def _create_execution_signals(self, action: int, confidence: float) -> Dict[str, Any]:
        """Create entry optimization-specific execution signals"""
        try:
            signals = {
                'primary_signal': action,
                'signal_strength': confidence,
                'execution_urgency': 'medium',
                'position_size_modifier': 1.0
            }
            
            # Entry optimization-specific signals
            if self.last_output and 'entry_analysis' in self.last_output.market_insights:
                entry_data = self.last_output.market_insights['entry_analysis']
                
                signals.update({
                    'entry_probability': confidence,
                    'entry_timing': entry_data.get('timing_score', 0.5),
                    'entry_size': entry_data.get('risk_reward_ratio', 1.0),
                    'entry_quality': entry_data.get('entry_quality', 'fair'),
                    'execution_difficulty': entry_data.get('execution_difficulty', 'moderate')
                })
                
                # Adjust based on entry quality
                if entry_data.get('entry_quality') == 'optimal':
                    signals['execution_urgency'] = 'high'
                    signals['position_size_modifier'] = 1.5
                elif entry_data.get('entry_quality') == 'poor':
                    signals['execution_urgency'] = 'low'
                    signals['position_size_modifier'] = 0.5
            
            # Integrate microstructure analysis
            if self.last_output and 'microstructure_analysis' in self.last_output.market_insights:
                microstructure_data = self.last_output.market_insights['microstructure_analysis']
                
                signals.update({
                    'slippage_estimate': microstructure_data.get('slippage_estimate', 0.0005),
                    'execution_cost': microstructure_data.get('execution_cost', 0.0005),
                    'market_depth': microstructure_data.get('market_depth', 0.8)
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error creating entry optimization execution signals: {e}")
            return super()._create_execution_signals(action, confidence)


def create_sequential_tactical_agents(observation_dims: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Dict[str, SequentialTacticalAgent]:
    """
    Create a set of sequential tactical agents
    
    Args:
        observation_dims: Dictionary mapping agent IDs to observation dimensions
        config: Configuration for all agents
        
    Returns:
        Dictionary of configured agents
    """
    try:
        agents = {}
        
        # Create FVG agent
        fvg_obs_dim = observation_dims.get('fvg_agent', 500)
        agents['fvg_agent'] = FVGTacticalAgent(
            observation_dim=fvg_obs_dim,
            config=config.get('fvg_agent', {}) if config else None
        )
        
        # Create Momentum agent
        momentum_obs_dim = observation_dims.get('momentum_agent', 600)
        agents['momentum_agent'] = MomentumTacticalAgent(
            observation_dim=momentum_obs_dim,
            config=config.get('momentum_agent', {}) if config else None
        )
        
        # Create Entry Optimization agent
        entry_obs_dim = observation_dims.get('entry_opt_agent', 700)
        agents['entry_opt_agent'] = EntryOptimizationAgent(
            observation_dim=entry_obs_dim,
            config=config.get('entry_opt_agent', {}) if config else None
        )
        
        logger.info(f"Created {len(agents)} sequential tactical agents")
        return agents
        
    except Exception as e:
        logger.error(f"Error creating sequential tactical agents: {e}")
        raise


def validate_sequential_agents(agents: Dict[str, SequentialTacticalAgent]) -> Dict[str, Any]:
    """
    Validate sequential tactical agents
    
    Args:
        agents: Dictionary of agents to validate
        
    Returns:
        Validation results
    """
    try:
        validation_results = {
            'agent_count': len(agents),
            'all_agents_valid': True,
            'agent_validations': {},
            'errors': []
        }
        
        # Expected agent sequence
        expected_agents = ['fvg_agent', 'momentum_agent', 'entry_opt_agent']
        
        # Check all expected agents are present
        for agent_id in expected_agents:
            if agent_id not in agents:
                validation_results['all_agents_valid'] = False
                validation_results['errors'].append(f"Missing agent: {agent_id}")
                continue
            
            agent = agents[agent_id]
            
            # Validate agent
            agent_validation = {
                'agent_id': agent.agent_id,
                'role': agent.role.value,
                'observation_dim': agent.observation_dim,
                'action_dim': agent.action_dim,
                'networks_initialized': hasattr(agent, 'policy_network') and agent.policy_network is not None,
                'valid': True
            }
            
            # Test action selection
            try:
                dummy_obs = np.random.normal(0, 1, agent.observation_dim)
                output = agent.select_action(dummy_obs)
                agent_validation['action_selection_test'] = True
                agent_validation['last_output_valid'] = isinstance(output, AgentOutput)
            except Exception as e:
                agent_validation['action_selection_test'] = False
                agent_validation['valid'] = False
                validation_results['errors'].append(f"Agent {agent_id} action selection failed: {e}")
            
            validation_results['agent_validations'][agent_id] = agent_validation
            
            if not agent_validation['valid']:
                validation_results['all_agents_valid'] = False
        
        return validation_results
        
    except Exception as e:
        return {
            'agent_count': 0,
            'all_agents_valid': False,
            'agent_validations': {},
            'errors': [f"Validation failed: {e}"]
        }


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'fvg_agent': {
            'network': {'hidden_dims': [256, 128, 64]},
            'execution': {'confidence_threshold': 0.6}
        },
        'momentum_agent': {
            'network': {'hidden_dims': [256, 128, 64]},
            'execution': {'confidence_threshold': 0.65}
        },
        'entry_opt_agent': {
            'network': {'hidden_dims': [256, 128, 64]},
            'execution': {'confidence_threshold': 0.7}
        }
    }
    
    # Create agents
    observation_dims = {
        'fvg_agent': 500,
        'momentum_agent': 600,
        'entry_opt_agent': 700
    }
    
    agents = create_sequential_tactical_agents(observation_dims, config)
    
    # Validate agents
    validation_results = validate_sequential_agents(agents)
    print("Validation Results:", validation_results)
    
    # Test sequential execution
    print("\nTesting sequential execution...")
    
    # Mock strategic context
    strategic_context = StrategicContext(
        regime_embedding=np.random.normal(0, 0.1, 64),
        synergy_signal={'strength': 0.7, 'confidence': 0.8},
        market_state={'price': 100.0, 'volume': 1000.0},
        confidence_level=0.8,
        execution_bias='bullish',
        volatility_forecast=0.25,
        timestamp=time.time()
    )
    
    # Sequential execution
    agent_outputs = {}
    predecessor_context = None
    
    for agent_id in ['fvg_agent', 'momentum_agent', 'entry_opt_agent']:
        agent = agents[agent_id]
        
        # Generate dummy observation
        dummy_obs = np.random.normal(0, 1, agent.observation_dim)
        
        # Select action
        output = agent.select_action(
            observation=dummy_obs,
            strategic_context=strategic_context,
            predecessor_context=predecessor_context
        )
        
        agent_outputs[agent_id] = output
        
        print(f"Agent {agent_id}:")
        print(f"  Action: {output.action}")
        print(f"  Confidence: {output.confidence:.3f}")
        print(f"  Processing Time: {output.processing_time:.2f}ms")
        
        # Update predecessor context for next agent
        if predecessor_context is None:
            predecessor_context = PredecessorContext(
                agent_outputs={agent_id: output.__dict__},
                consensus_level=1.0,
                alignment_score=1.0,
                execution_signals={agent_id: output.execution_signals},
                feature_importance={agent_id: output.feature_importance},
                timestamp=time.time()
            )
        else:
            predecessor_context.agent_outputs[agent_id] = output.__dict__
            predecessor_context.execution_signals[agent_id] = output.execution_signals
            predecessor_context.feature_importance[agent_id] = output.feature_importance
    
    print("\nSequential execution completed!")
    
    # Performance metrics
    print("\nPerformance Metrics:")
    for agent_id, agent in agents.items():
        metrics = agent.get_performance_metrics()
        print(f"{agent_id}: {metrics}")