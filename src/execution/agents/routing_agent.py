"""
Routing Agent (π₅) - The Arbitrageur
===================================

The 5th MARL agent responsible for intelligent venue selection and adaptive order routing.
Transforms execution quality through real-time broker optimization and learning.

Mission: Optimal venue selection with measurable QoE (Quality of Execution) improvement
- Real-time broker performance monitoring
- Adaptive venue selection based on execution quality metrics
- Multi-agent coordination for optimal routing decisions
- <100μs routing decision target

Agent Architecture:
- State Space: Order features + broker performance vectors (latency, commissions, fill rates)
- Action Space: Discrete(N) for available brokers {IB, Alpaca, etc.}
- Neural Network: 15D+ input → 256→128→64→N_brokers output
- Learning Rate: 2e-4 for adaptive learning

QoE Reward System:
Reward = (1.0 - normalized_slippage) + (1.0 - normalized_commission) + fill_rate

Author: Agent 1 - The Arbitrageur Implementation
Date: 2025-07-13
Mission Status: Implementation in Progress
"""

import asyncio
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import structlog

from src.core.events import EventBus, Event, EventType
from src.execution.brokers.base_broker import BaseBrokerClient, BrokerOrder, BrokerExecution

logger = structlog.get_logger()


class BrokerType(Enum):
    """Available broker types"""
    INTERACTIVE_BROKERS = "IB"
    ALPACA = "ALPACA"
    TD_AMERITRADE = "TDA"
    CHARLES_SCHWAB = "SCHWAB"
    ETRADE = "ETRADE"


@dataclass
class BrokerPerformanceMetrics:
    """Real-time broker performance metrics"""
    
    broker_id: str
    broker_type: BrokerType
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Execution quality metrics
    fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    commission_per_share: float = 0.0
    rebate_per_share: float = 0.0
    
    # Market impact metrics
    market_impact_bps: float = 0.0
    price_improvement_bps: float = 0.0
    
    # Reliability metrics
    uptime_percentage: float = 0.0
    error_rate: float = 0.0
    
    # Volume and liquidity
    daily_volume: float = 0.0
    available_liquidity: float = 0.0
    
    # Time metrics
    last_updated: datetime = field(default_factory=datetime.now)
    measurement_window_minutes: int = 60
    
    # Quality score (0-1)
    quality_score: float = 0.0


@dataclass
class RoutingState:
    """State representation for routing agent"""
    
    # Order characteristics (5D)
    order_size: float = 0.0
    order_value: float = 0.0  
    order_urgency: float = 0.0  # 0-1 scale
    order_side: float = 0.0  # 0=BUY, 1=SELL
    order_type: float = 0.0  # 0=MARKET, 1=LIMIT, 2=STOP
    
    # Market conditions (5D)
    volatility: float = 0.0
    spread_bps: float = 0.0
    volume_ratio: float = 0.0  # Current/Average volume
    time_of_day: float = 0.0  # Normalized 0-1
    market_stress: float = 0.0  # Market stress indicator
    
    # Broker performance vectors (N_brokers * 5D each)
    broker_latencies: np.ndarray = field(default_factory=lambda: np.zeros(8))
    broker_fill_rates: np.ndarray = field(default_factory=lambda: np.zeros(8))
    broker_costs: np.ndarray = field(default_factory=lambda: np.zeros(8))
    broker_reliabilities: np.ndarray = field(default_factory=lambda: np.zeros(8))
    broker_availabilities: np.ndarray = field(default_factory=lambda: np.zeros(8))
    
    # Portfolio context (5D)
    portfolio_value: float = 0.0
    position_concentration: float = 0.0
    risk_budget_used: float = 0.0
    recent_pnl: float = 0.0
    correlation_risk: float = 0.0
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor representation"""
        
        # Order features (5D)
        order_features = torch.tensor([
            self.order_size,
            self.order_value,
            self.order_urgency,
            self.order_side,
            self.order_type
        ], dtype=torch.float32)
        
        # Market features (5D)
        market_features = torch.tensor([
            self.volatility,
            self.spread_bps,
            self.volume_ratio,
            self.time_of_day,
            self.market_stress
        ], dtype=torch.float32)
        
        # Broker features (40D = 8 brokers * 5 features each)
        broker_features = torch.cat([
            torch.from_numpy(self.broker_latencies).float(),
            torch.from_numpy(self.broker_fill_rates).float(),
            torch.from_numpy(self.broker_costs).float(),
            torch.from_numpy(self.broker_reliabilities).float(),
            torch.from_numpy(self.broker_availabilities).float()
        ])
        
        # Portfolio features (5D)
        portfolio_features = torch.tensor([
            self.portfolio_value,
            self.position_concentration,
            self.risk_budget_used,
            self.recent_pnl,
            self.correlation_risk
        ], dtype=torch.float32)
        
        # Combine all features (55D total)
        return torch.cat([
            order_features,
            market_features,
            broker_features,
            portfolio_features
        ])
    
    @property
    def dimension(self) -> int:
        """Get state dimension"""
        return 55  # 5 + 5 + 40 + 5


@dataclass
class RoutingAction:
    """Routing action representation"""
    
    broker_id: str
    broker_type: BrokerType
    confidence: float
    expected_qoe: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QoEMetrics:
    """Quality of Execution metrics"""
    
    execution_id: str
    broker_id: str
    
    # Core QoE components
    fill_rate: float = 0.0
    slippage_bps: float = 0.0
    commission_cost: float = 0.0
    
    # Additional quality metrics
    latency_ms: float = 0.0
    market_impact_bps: float = 0.0
    price_improvement_bps: float = 0.0
    
    # Composite QoE score (0-1)
    qoe_score: float = 0.0
    
    # Timing
    execution_time: datetime = field(default_factory=datetime.now)
    
    def calculate_qoe_score(self) -> float:
        """Calculate composite QoE score"""
        
        # Normalize components (0-1 scale, higher is better)
        fill_rate_score = self.fill_rate
        
        # Slippage score (lower slippage is better)
        slippage_score = max(0.0, 1.0 - (self.slippage_bps / 50.0))  # 50bps = 0 score
        
        # Commission score (lower commission is better)
        commission_score = max(0.0, 1.0 - (self.commission_cost / 0.10))  # $0.10 = 0 score
        
        # Composite QoE score
        self.qoe_score = (fill_rate_score + slippage_score + commission_score) / 3.0
        return self.qoe_score


class RoutingNetwork(nn.Module):
    """Neural network for routing decisions"""
    
    def __init__(self, state_dim: int, num_brokers: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_brokers = num_brokers
        
        # Build network layers
        layers = []
        current_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, num_brokers))
        
        self.network = nn.Sequential(*layers)
        
        # Value head for critic training
        self.value_head = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        logger.info("RoutingNetwork initialized", 
                   state_dim=state_dim, 
                   num_brokers=num_brokers,
                   total_params=sum(p.numel() for p in self.parameters()))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and state value"""
        
        # Ensure proper input shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Forward through main network
        features = state
        for layer in self.network[:-1]:
            features = layer(features)
        
        # Action logits
        action_logits = self.network[-1](features)
        
        # State value
        state_value = self.value_head(features)
        
        return action_logits, state_value
    
    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities"""
        action_logits, _ = self.forward(state)
        return torch.softmax(action_logits, dim=-1)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float]:
        """Get action and log probability"""
        action_probs = self.get_action_probabilities(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(-1, action.unsqueeze(-1)))
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item()


class BrokerPerformanceMonitor:
    """Real-time broker performance monitoring system"""
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.metrics: Dict[str, BrokerPerformanceMetrics] = {}
        self.execution_history: Dict[str, deque] = {}
        self.latency_history: Dict[str, deque] = {}
        
        # Performance calculation cache
        self._last_update = datetime.now()
        self._update_interval = timedelta(seconds=30)
        
        logger.info("BrokerPerformanceMonitor initialized", window_minutes=window_minutes)
    
    def add_execution(self, broker_id: str, execution: BrokerExecution, 
                     latency_ms: float, slippage_bps: float, commission: float):
        """Add execution data for performance tracking"""
        
        if broker_id not in self.execution_history:
            self.execution_history[broker_id] = deque(maxlen=1000)
            self.latency_history[broker_id] = deque(maxlen=1000)
        
        # Store execution data
        execution_data = {
            'timestamp': execution.timestamp,
            'quantity': execution.quantity,
            'price': execution.price,
            'latency_ms': latency_ms,
            'slippage_bps': slippage_bps,
            'commission': commission,
            'fees': execution.fees
        }
        
        self.execution_history[broker_id].append(execution_data)
        self.latency_history[broker_id].append(latency_ms)
        
        # Update metrics if needed
        self._maybe_update_metrics()
    
    def _maybe_update_metrics(self):
        """Update metrics if enough time has passed"""
        if datetime.now() - self._last_update > self._update_interval:
            self._update_all_metrics()
            self._last_update = datetime.now()
    
    def _update_all_metrics(self):
        """Update all broker metrics"""
        for broker_id in self.execution_history.keys():
            self._update_broker_metrics(broker_id)
    
    def _update_broker_metrics(self, broker_id: str):
        """Update metrics for specific broker"""
        
        if broker_id not in self.execution_history:
            return
        
        executions = list(self.execution_history[broker_id])
        latencies = list(self.latency_history[broker_id])
        
        if not executions:
            return
        
        # Filter to time window
        cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes)
        recent_executions = [e for e in executions if e['timestamp'] > cutoff_time]
        recent_latencies = [e['latency_ms'] for e in recent_executions]
        
        if not recent_executions:
            return
        
        # Calculate metrics
        metrics = BrokerPerformanceMetrics(
            broker_id=broker_id,
            broker_type=BrokerType.INTERACTIVE_BROKERS,  # Default, should be set properly
            measurement_window_minutes=self.window_minutes
        )
        
        # Latency metrics
        if recent_latencies:
            metrics.avg_latency_ms = np.mean(recent_latencies)
            metrics.p95_latency_ms = np.percentile(recent_latencies, 95)
            metrics.p99_latency_ms = np.percentile(recent_latencies, 99)
        
        # Execution quality metrics
        total_quantity = sum(e['quantity'] for e in recent_executions)
        filled_quantity = total_quantity  # Assume all filled for now
        metrics.fill_rate = filled_quantity / max(1, total_quantity)
        
        # Average slippage and commission
        if recent_executions:
            metrics.avg_slippage_bps = np.mean([e['slippage_bps'] for e in recent_executions])
            total_commission = sum(e['commission'] for e in recent_executions)
            metrics.commission_per_share = total_commission / max(1, filled_quantity)
        
        # Volume metrics
        metrics.daily_volume = filled_quantity
        
        # Reliability (simplified)
        metrics.uptime_percentage = 0.99  # Assume high uptime
        metrics.error_rate = max(0.0, 1.0 - metrics.fill_rate)
        
        # Calculate quality score
        metrics.quality_score = self._calculate_quality_score(metrics)
        
        self.metrics[broker_id] = metrics
    
    def _calculate_quality_score(self, metrics: BrokerPerformanceMetrics) -> float:
        """Calculate composite quality score (0-1)"""
        
        # Latency score (lower is better)
        latency_score = max(0.0, 1.0 - (metrics.avg_latency_ms / 1000.0))  # 1s = 0 score
        
        # Fill rate score
        fill_score = metrics.fill_rate
        
        # Slippage score (lower is better)
        slippage_score = max(0.0, 1.0 - (metrics.avg_slippage_bps / 50.0))  # 50bps = 0 score
        
        # Commission score (lower is better)
        commission_score = max(0.0, 1.0 - (metrics.commission_per_share / 0.10))  # $0.10 = 0 score
        
        # Reliability score
        reliability_score = metrics.uptime_percentage * (1.0 - metrics.error_rate)
        
        # Weighted average
        return (latency_score * 0.25 + 
                fill_score * 0.25 + 
                slippage_score * 0.20 + 
                commission_score * 0.15 + 
                reliability_score * 0.15)
    
    def get_broker_metrics(self, broker_id: str) -> Optional[BrokerPerformanceMetrics]:
        """Get current metrics for broker"""
        return self.metrics.get(broker_id)
    
    def get_all_metrics(self) -> Dict[str, BrokerPerformanceMetrics]:
        """Get all broker metrics"""
        self._update_all_metrics()
        return self.metrics.copy()
    
    def get_broker_state_vector(self, broker_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get broker state vectors for neural network input"""
        
        latencies = np.zeros(len(broker_ids))
        fill_rates = np.zeros(len(broker_ids))
        costs = np.zeros(len(broker_ids))
        reliabilities = np.zeros(len(broker_ids))
        availabilities = np.zeros(len(broker_ids))
        
        for i, broker_id in enumerate(broker_ids):
            metrics = self.metrics.get(broker_id)
            if metrics:
                # Normalize latency (0-1, lower is better)
                latencies[i] = max(0.0, 1.0 - (metrics.avg_latency_ms / 1000.0))
                fill_rates[i] = metrics.fill_rate
                costs[i] = max(0.0, 1.0 - (metrics.commission_per_share / 0.10))
                reliabilities[i] = metrics.uptime_percentage
                availabilities[i] = 1.0 if metrics.last_updated > datetime.now() - timedelta(minutes=5) else 0.0
            else:
                # Default values for unknown brokers
                latencies[i] = 0.5
                fill_rates[i] = 0.8
                costs[i] = 0.5
                reliabilities[i] = 0.9
                availabilities[i] = 0.0
        
        return latencies, fill_rates, costs, reliabilities, availabilities


class RoutingAgent:
    """
    MARL Agent π₅: The Arbitrageur - Adaptive Order Routing Agent
    
    Intelligent venue selection system that transforms execution quality through
    real-time broker optimization and continuous learning.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        
        # Broker configuration
        self.broker_ids = config.get('broker_ids', ['IB', 'ALPACA', 'TDA', 'SCHWAB'])
        self.num_brokers = len(self.broker_ids)
        
        # Neural network
        state_dim = 55  # From RoutingState.dimension
        self.network = RoutingNetwork(state_dim, self.num_brokers)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.get('learning_rate', 2e-4))
        
        # Performance monitoring
        self.performance_monitor = BrokerPerformanceMonitor(
            window_minutes=config.get('monitoring_window_minutes', 60)
        )
        
        # Reward calculation
        self.qoe_history: deque = deque(maxlen=1000)
        self.reward_history: deque = deque(maxlen=1000)
        
        # Action tracking
        self.action_history: deque = deque(maxlen=1000)
        self.routing_stats = {
            'total_routes': 0,
            'successful_routes': 0,
            'avg_qoe_score': 0.0,
            'avg_routing_time_us': 0.0,
            'broker_usage': {broker_id: 0 for broker_id in self.broker_ids}
        }
        
        # Training state
        self.training_mode = config.get('training_mode', True)
        self.exploration_epsilon = config.get('exploration_epsilon', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.min_epsilon = config.get('min_epsilon', 0.01)
        
        # Performance targets
        self.target_qoe_score = config.get('target_qoe_score', 0.85)
        self.max_routing_latency_us = config.get('max_routing_latency_us', 100.0)
        
        logger.info("RoutingAgent (π₅) initialized",
                   num_brokers=self.num_brokers,
                   state_dim=state_dim,
                   training_mode=self.training_mode,
                   target_qoe=self.target_qoe_score)
    
    async def act(self, state: RoutingState, deterministic: bool = False) -> RoutingAction:
        """
        Execute routing decision with <100μs target latency
        
        Args:
            state: Current routing state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Routing action with broker selection and confidence
        """
        start_time = time.perf_counter()
        
        try:
            # Convert state to tensor
            state_tensor = state.to_tensor()
            
            # Get action from network
            if self.training_mode and not deterministic and np.random.random() < self.exploration_epsilon:
                # Exploration: random action
                action_idx = np.random.randint(0, self.num_brokers)
                confidence = 0.5
            else:
                # Exploitation: network action
                with torch.no_grad():
                    action_idx, log_prob = self.network.get_action(state_tensor, deterministic)
                    action_probs = self.network.get_action_probabilities(state_tensor)
                    confidence = action_probs[action_idx].item()
            
            # Get selected broker
            broker_id = self.broker_ids[action_idx]
            broker_metrics = self.performance_monitor.get_broker_metrics(broker_id)
            
            # Calculate expected QoE
            expected_qoe = self._calculate_expected_qoe(broker_id, state)
            
            # Create routing action
            routing_action = RoutingAction(
                broker_id=broker_id,
                broker_type=BrokerType.INTERACTIVE_BROKERS,  # Should map properly
                confidence=confidence,
                expected_qoe=expected_qoe,
                reasoning=f"MARL routing to {broker_id} (confidence: {confidence:.3f})"
            )
            
            # Track performance
            routing_time_us = (time.perf_counter() - start_time) * 1_000_000
            self._update_routing_stats(routing_action, routing_time_us)
            
            # Decay exploration
            if self.training_mode:
                self.exploration_epsilon = max(self.min_epsilon, 
                                             self.exploration_epsilon * self.epsilon_decay)
            
            logger.debug("Routing action executed",
                        broker=broker_id,
                        confidence=confidence,
                        expected_qoe=expected_qoe,
                        routing_time_us=routing_time_us)
            
            return routing_action
            
        except Exception as e:
            logger.error("Routing action failed", error=str(e))
            
            # Fallback to first available broker
            return RoutingAction(
                broker_id=self.broker_ids[0],
                broker_type=BrokerType.INTERACTIVE_BROKERS,
                confidence=0.1,
                expected_qoe=0.5,
                reasoning=f"Fallback routing due to error: {str(e)}"
            )
    
    def _calculate_expected_qoe(self, broker_id: str, state: RoutingState) -> float:
        """Calculate expected Quality of Execution for broker"""
        
        metrics = self.performance_monitor.get_broker_metrics(broker_id)
        if not metrics:
            return 0.5  # Default conservative estimate
        
        # QoE components
        fill_rate_component = metrics.fill_rate
        slippage_component = max(0.0, 1.0 - (metrics.avg_slippage_bps / 50.0))
        commission_component = max(0.0, 1.0 - (metrics.commission_per_share / 0.10))
        
        # Weighted QoE score
        expected_qoe = (fill_rate_component + slippage_component + commission_component) / 3.0
        
        # Adjust for market conditions
        if state.market_stress > 0.7:
            expected_qoe *= 0.9  # Reduce QoE expectation in stressed markets
        
        if state.order_urgency > 0.8:
            expected_qoe *= 0.95  # High urgency may compromise QoE slightly
        
        return np.clip(expected_qoe, 0.0, 1.0)
    
    def _update_routing_stats(self, action: RoutingAction, routing_time_us: float):
        """Update routing statistics"""
        
        self.routing_stats['total_routes'] += 1
        self.routing_stats['broker_usage'][action.broker_id] += 1
        
        # Update average routing time
        current_avg = self.routing_stats['avg_routing_time_us']
        total_routes = self.routing_stats['total_routes']
        self.routing_stats['avg_routing_time_us'] = (
            (current_avg * (total_routes - 1) + routing_time_us) / total_routes
        )
        
        # Store action for training
        self.action_history.append({
            'timestamp': action.timestamp,
            'broker_id': action.broker_id,
            'confidence': action.confidence,
            'expected_qoe': action.expected_qoe,
            'routing_time_us': routing_time_us
        })
    
    def add_execution_feedback(self, broker_id: str, execution: BrokerExecution,
                             latency_ms: float, slippage_bps: float, commission: float) -> QoEMetrics:
        """Add execution feedback for learning"""
        
        # Update performance monitor
        self.performance_monitor.add_execution(broker_id, execution, latency_ms, slippage_bps, commission)
        
        # Calculate QoE metrics
        qoe_metrics = QoEMetrics(
            execution_id=execution.execution_id,
            broker_id=broker_id,
            fill_rate=1.0,  # Assume filled if we have execution
            slippage_bps=slippage_bps,
            commission_cost=commission,
            latency_ms=latency_ms
        )
        qoe_metrics.calculate_qoe_score()
        
        # Calculate reward for MARL training
        reward = self._calculate_qoe_reward(qoe_metrics)
        
        # Store for training
        self.qoe_history.append(qoe_metrics)
        self.reward_history.append(reward)
        
        # Update stats
        if qoe_metrics.qoe_score >= self.target_qoe_score:
            self.routing_stats['successful_routes'] += 1
        
        # Update average QoE
        total_qoe = sum(q.qoe_score for q in self.qoe_history)
        self.routing_stats['avg_qoe_score'] = total_qoe / len(self.qoe_history)
        
        logger.debug("Execution feedback added",
                    broker=broker_id,
                    qoe_score=qoe_metrics.qoe_score,
                    reward=reward,
                    avg_qoe=self.routing_stats['avg_qoe_score'])
        
        return qoe_metrics
    
    def _calculate_qoe_reward(self, qoe_metrics: QoEMetrics) -> float:
        """
        Calculate MARL reward based on QoE metrics
        
        Reward = (1.0 - normalized_slippage) + (1.0 - normalized_commission) + fill_rate
        """
        
        # Component rewards
        fill_rate_reward = qoe_metrics.fill_rate
        
        # Slippage reward (lower is better)
        slippage_reward = max(0.0, 1.0 - (qoe_metrics.slippage_bps / 50.0))
        
        # Commission reward (lower is better)
        commission_reward = max(0.0, 1.0 - (qoe_metrics.commission_cost / 0.10))
        
        # Total reward
        total_reward = fill_rate_reward + slippage_reward + commission_reward
        
        # Bonus for exceptional performance
        if qoe_metrics.qoe_score > 0.9:
            total_reward += 0.5  # Bonus for excellent execution
        
        # Penalty for poor performance
        if qoe_metrics.qoe_score < 0.5:
            total_reward -= 0.5  # Penalty for poor execution
        
        return total_reward
    
    def create_routing_state(self, order_data: Dict[str, Any], 
                           market_data: Dict[str, Any],
                           portfolio_data: Dict[str, Any]) -> RoutingState:
        """Create routing state from current data"""
        
        # Get broker performance vectors
        latencies, fill_rates, costs, reliabilities, availabilities = (
            self.performance_monitor.get_broker_state_vector(self.broker_ids)
        )
        
        # Create state
        state = RoutingState(
            # Order characteristics
            order_size=order_data.get('quantity', 0) / 10000.0,  # Normalize
            order_value=order_data.get('notional_value', 0) / 1000000.0,  # Normalize to millions
            order_urgency=order_data.get('urgency', 0.5),
            order_side=1.0 if order_data.get('side') == 'SELL' else 0.0,
            order_type=order_data.get('order_type_code', 0.0),
            
            # Market conditions
            volatility=market_data.get('volatility', 0.15),
            spread_bps=market_data.get('spread_bps', 5.0),
            volume_ratio=market_data.get('volume_ratio', 1.0),
            time_of_day=market_data.get('time_of_day_normalized', 0.5),
            market_stress=market_data.get('stress_indicator', 0.0),
            
            # Broker performance vectors
            broker_latencies=latencies,
            broker_fill_rates=fill_rates,
            broker_costs=costs,
            broker_reliabilities=reliabilities,
            broker_availabilities=availabilities,
            
            # Portfolio context
            portfolio_value=portfolio_data.get('portfolio_value', 100000) / 1000000.0,
            position_concentration=portfolio_data.get('concentration', 0.1),
            risk_budget_used=portfolio_data.get('risk_budget_used', 0.3),
            recent_pnl=portfolio_data.get('recent_pnl', 0.0),
            correlation_risk=portfolio_data.get('correlation_risk', 0.2)
        )
        
        return state
    
    def get_broker_rankings(self) -> List[Tuple[str, float]]:
        """Get current broker rankings by quality score"""
        
        rankings = []
        for broker_id in self.broker_ids:
            metrics = self.performance_monitor.get_broker_metrics(broker_id)
            quality_score = metrics.quality_score if metrics else 0.0
            rankings.append((broker_id, quality_score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        return {
            'routing_stats': self.routing_stats.copy(),
            'broker_metrics': {
                broker_id: metrics.__dict__ if metrics else None
                for broker_id, metrics in self.performance_monitor.get_all_metrics().items()
            },
            'recent_qoe_scores': [q.qoe_score for q in list(self.qoe_history)[-10:]],
            'recent_rewards': list(self.reward_history)[-10:],
            'broker_rankings': self.get_broker_rankings(),
            'network_parameters': {
                'total_params': sum(p.numel() for p in self.network.parameters()),
                'exploration_epsilon': self.exploration_epsilon,
                'training_mode': self.training_mode
            },
            'performance_targets': {
                'target_qoe_score': self.target_qoe_score,
                'current_avg_qoe': self.routing_stats['avg_qoe_score'],
                'qoe_target_met': self.routing_stats['avg_qoe_score'] >= self.target_qoe_score,
                'max_routing_latency_us': self.max_routing_latency_us,
                'current_avg_latency_us': self.routing_stats['avg_routing_time_us'],
                'latency_target_met': self.routing_stats['avg_routing_time_us'] <= self.max_routing_latency_us
            }
        }
    
    def update_network(self, states: List[torch.Tensor], actions: List[int], 
                      rewards: List[float], next_states: List[torch.Tensor]) -> Dict[str, float]:
        """Update neural network with training batch"""
        
        if not self.training_mode:
            return {}
        
        # Convert to tensors
        states_tensor = torch.stack(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.stack(next_states)
        
        # Forward pass
        action_logits, state_values = self.network(states_tensor)
        next_action_logits, next_state_values = self.network(next_states_tensor)
        
        # Calculate policy loss
        action_probs = torch.softmax(action_logits, dim=-1)
        selected_action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        policy_loss = -torch.mean(torch.log(selected_action_probs) * rewards_tensor)
        
        # Calculate value loss
        value_loss = torch.mean((state_values.squeeze() - rewards_tensor) ** 2)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    async def shutdown(self):
        """Shutdown routing agent"""
        logger.info("Shutting down RoutingAgent (π₅)")
        
        # Save final performance report
        final_report = self.get_performance_report()
        logger.info("Final routing performance", **final_report['routing_stats'])
        
        logger.info("RoutingAgent shutdown complete")


def create_routing_agent(config: Dict[str, Any], event_bus: EventBus) -> RoutingAgent:
    """Factory function to create routing agent"""
    return RoutingAgent(config, event_bus)


# Default configuration
DEFAULT_ROUTING_CONFIG = {
    'broker_ids': ['IB', 'ALPACA', 'TDA', 'SCHWAB'],
    'learning_rate': 2e-4,
    'training_mode': True,
    'exploration_epsilon': 0.1,
    'epsilon_decay': 0.995,
    'min_epsilon': 0.01,
    'target_qoe_score': 0.85,
    'max_routing_latency_us': 100.0,
    'monitoring_window_minutes': 60
}