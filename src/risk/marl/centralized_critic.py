"""
Centralized Critic for Risk Management MARL System

Provides global portfolio risk evaluation V(s) for coordinated multi-agent
risk management decisions across all 4 risk agents.

Features:
- Global portfolio risk state aggregation
- Risk-adjusted value function evaluation  
- Multi-agent gradient computation
- Real-time risk assessment
- Emergency risk escalation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import structlog
from datetime import datetime

from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


class RiskCriticMode(Enum):
    """Operating modes for risk critic"""
    NORMAL = "normal"
    STRESS = "stress" 
    EMERGENCY = "emergency"


@dataclass
class GlobalRiskState:
    """Aggregated global risk state for centralized critic"""
    
    # Individual agent risk vectors (4 agents x 10 dimensions)
    position_sizing_risk: np.ndarray      # π₁ risk vector
    stop_target_risk: np.ndarray          # π₂ risk vector 
    risk_monitor_risk: np.ndarray         # π₃ risk vector
    portfolio_optimizer_risk: np.ndarray  # π₄ risk vector
    
    # Global portfolio metrics
    total_portfolio_var: float            # Aggregate portfolio VaR
    portfolio_correlation_max: float      # Maximum pairwise correlation
    aggregate_leverage: float             # Total leverage across positions
    liquidity_risk_score: float          # Aggregate liquidity risk
    systemic_risk_level: float           # Market-wide systemic risk
    
    # Temporal factors
    timestamp: datetime
    market_hours_factor: float            # 0-1, higher during market hours
    
    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert to tensor for neural network processing"""
        
        # Flatten individual agent vectors (4 x 10 = 40 dimensions)
        agent_vectors = np.concatenate([
            self.position_sizing_risk,
            self.stop_target_risk,
            self.risk_monitor_risk,
            self.portfolio_optimizer_risk
        ])
        
        # Global risk metrics (5 dimensions)
        global_metrics = np.array([
            self.total_portfolio_var,
            self.portfolio_correlation_max,
            self.aggregate_leverage,
            self.liquidity_risk_score,
            self.systemic_risk_level
        ])
        
        # Temporal factors (1 dimension)
        temporal_factors = np.array([self.market_hours_factor])
        
        # Concatenate all features (40 + 5 + 1 = 46 dimensions)
        full_state = np.concatenate([agent_vectors, global_metrics, temporal_factors])
        
        tensor = torch.FloatTensor(full_state)
        if device:
            tensor = tensor.to(device)
            
        return tensor
    
    @property
    def feature_dim(self) -> int:
        """Total feature dimension for critic input"""
        return 46  # 4*10 + 5 + 1


class CentralizedCriticNetwork(nn.Module):
    """Neural network for centralized risk value function"""
    
    def __init__(self, 
                 input_dim: int = 46,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build network layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Final value head
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network
        
        Args:
            state: Global risk state tensor (batch_size, 46)
            
        Returns:
            Value estimates (batch_size, 1)
        """
        return self.network(state)


class CentralizedCritic:
    """
    Centralized critic for multi-agent risk management
    
    Provides global portfolio risk evaluation V(s) that considers:
    - All 4 agent risk states simultaneously
    - Global portfolio correlations and dependencies  
    - Systemic risk factors
    - Emergency risk escalation
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 event_bus: Optional[EventBus] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize centralized critic
        
        Args:
            config: Critic configuration parameters
            event_bus: Event bus for risk communication
            device: PyTorch device for computation
        """
        self.config = config
        self.event_bus = event_bus
        self.device = device or torch.device('cpu')
        
        # Network configuration
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Risk thresholds
        self.stress_threshold = config.get('stress_threshold', 0.15)  # 15% portfolio VaR
        self.emergency_threshold = config.get('emergency_threshold', 0.25)  # 25% portfolio VaR
        self.correlation_alert_threshold = config.get('correlation_alert_threshold', 0.8)
        
        # Initialize network
        self.network = CentralizedCriticNetwork(
            input_dim=46,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            dropout_rate=config.get('dropout_rate', 0.1),
            use_batch_norm=config.get('use_batch_norm', True)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Performance tracking
        self.evaluation_count = 0
        self.stress_events = 0
        self.emergency_events = 0
        self.current_mode = RiskCriticMode.NORMAL
        
        # Value function statistics
        self.value_history = []
        self.loss_history = []
        
        logger.info("Centralized critic initialized",
                   device=str(self.device),
                   parameters=sum(p.numel() for p in self.network.parameters()))
    
    def evaluate_global_risk(self, global_state: GlobalRiskState) -> Tuple[float, RiskCriticMode]:
        """
        Evaluate global portfolio risk value function V(s)
        
        Args:
            global_state: Aggregated global risk state
            
        Returns:
            Tuple of (risk_value, operating_mode)
            - risk_value: Global risk assessment (-1 to 1, lower is riskier)
            - operating_mode: Current risk operating mode
        """
        start_time = datetime.now()
        
        try:
            # Convert to tensor
            state_tensor = global_state.to_tensor(self.device).unsqueeze(0)
            
            # Forward pass through critic network
            with torch.no_grad():
                value = self.network(state_tensor).item()
            
            # Determine operating mode based on risk indicators
            mode = self._determine_operating_mode(global_state, value)
            
            # Track performance
            self.evaluation_count += 1
            self.value_history.append(value)
            
            # Check for risk events
            if mode == RiskCriticMode.STRESS:
                self.stress_events += 1
            elif mode == RiskCriticMode.EMERGENCY:
                self.emergency_events += 1
                
            self.current_mode = mode
            
            # Publish risk evaluation event
            if self.event_bus:
                self._publish_risk_evaluation(global_state, value, mode)
            
            # Performance check
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            if response_time > 10.0:  # 10ms target
                logger.warning("Critic evaluation exceeded target latency",
                             response_time=response_time,
                             target=10.0)
            
            return value, mode
            
        except Exception as e:
            logger.error("Error in global risk evaluation", error=str(e))
            return -1.0, RiskCriticMode.EMERGENCY  # Conservative fallback
    
    def _determine_operating_mode(self, global_state: GlobalRiskState, value: float) -> RiskCriticMode:
        """Determine operating mode based on risk indicators"""
        
        # Check emergency conditions
        if (global_state.total_portfolio_var > self.emergency_threshold or
            global_state.portfolio_correlation_max > 0.95 or
            global_state.aggregate_leverage > 10.0 or
            value < -0.8):
            return RiskCriticMode.EMERGENCY
        
        # Check stress conditions
        if (global_state.total_portfolio_var > self.stress_threshold or
            global_state.portfolio_correlation_max > self.correlation_alert_threshold or
            global_state.systemic_risk_level > 0.7 or
            value < -0.5):
            return RiskCriticMode.STRESS
        
        return RiskCriticMode.NORMAL
    
    def _publish_risk_evaluation(self, global_state: GlobalRiskState, value: float, mode: RiskCriticMode):
        """Publish risk evaluation results via event bus"""
        if not self.event_bus:
            return
        
        event_data = {
            'global_risk_value': value,
            'operating_mode': mode.value,
            'portfolio_var': global_state.total_portfolio_var,
            'max_correlation': global_state.portfolio_correlation_max,
            'leverage': global_state.aggregate_leverage,
            'evaluation_count': self.evaluation_count,
            'timestamp': global_state.timestamp
        }
        
        # Choose event type based on mode
        if mode == RiskCriticMode.EMERGENCY:
            event_type = EventType.EMERGENCY_STOP
        elif mode == RiskCriticMode.STRESS:
            event_type = EventType.RISK_BREACH
        else:
            event_type = EventType.RISK_UPDATE
        
        event = self.event_bus.create_event(event_type, event_data, "centralized_critic")
        self.event_bus.publish(event)
    
    def compute_agent_gradients(self, 
                               global_state: GlobalRiskState,
                               target_value: float,
                               agent_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute gradients for individual agents based on global critic
        
        Args:
            global_state: Current global risk state
            target_value: Target value for training
            agent_weights: Optional weights for different agents
            
        Returns:
            Dictionary of agent gradients
        """
        if agent_weights is None:
            agent_weights = {'π1': 1.0, 'π2': 1.0, 'π3': 1.0, 'π4': 1.0}
        
        state_tensor = global_state.to_tensor(self.device).unsqueeze(0)
        state_tensor.requires_grad_(True)
        
        # Forward pass
        predicted_value = self.network(state_tensor)
        
        # Compute loss
        target_tensor = torch.FloatTensor([target_value]).to(self.device)
        loss = F.mse_loss(predicted_value, target_tensor)
        
        # Backward pass
        loss.backward()
        
        # Extract gradients for each agent's state components
        gradients = {}
        if state_tensor.grad is not None:
            grad = state_tensor.grad.squeeze(0)
            
            # Agent state ranges: [0:10], [10:20], [20:30], [30:40]
            gradients['π1'] = grad[0:10] * agent_weights.get('π1', 1.0)
            gradients['π2'] = grad[10:20] * agent_weights.get('π2', 1.0)
            gradients['π3'] = grad[20:30] * agent_weights.get('π3', 1.0)
            gradients['π4'] = grad[30:40] * agent_weights.get('π4', 1.0)
        
        return gradients
    
    def update_critic(self, 
                     global_state: GlobalRiskState,
                     target_value: float) -> float:
        """
        Update critic network with new data point
        
        Args:
            global_state: Global risk state
            target_value: Target value for training
            
        Returns:
            Training loss
        """
        state_tensor = global_state.to_tensor(self.device).unsqueeze(0)
        target_tensor = torch.FloatTensor([target_value]).to(self.device)
        
        # Forward pass
        predicted_value = self.network(state_tensor)
        
        # Compute loss
        loss = F.mse_loss(predicted_value, target_tensor) * self.value_loss_coef
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        # Track loss
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get critic performance metrics"""
        avg_value = np.mean(self.value_history) if self.value_history else 0.0
        avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
        
        return {
            'evaluation_count': self.evaluation_count,
            'stress_events': self.stress_events,
            'emergency_events': self.emergency_events,
            'current_mode': self.current_mode.value,
            'average_value': avg_value,
            'recent_loss': avg_loss,
            'stress_rate': self.stress_events / max(1, self.evaluation_count),
            'emergency_rate': self.emergency_events / max(1, self.evaluation_count)
        }
    
    def save_model(self, path: str):
        """Save critic model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.get_performance_metrics()
        }, path)
        logger.info("Critic model saved", path=path)
    
    def load_model(self, path: str):
        """Load critic model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Critic model loaded", path=path)
    
    def reset(self):
        """Reset critic state"""
        self.evaluation_count = 0
        self.stress_events = 0
        self.emergency_events = 0
        self.current_mode = RiskCriticMode.NORMAL
        self.value_history.clear()
        self.loss_history.clear()
        logger.info("Centralized critic reset")