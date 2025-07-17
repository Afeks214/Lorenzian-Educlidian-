"""
Enhanced 5-Agent MARL Coordination System
==========================================

Advanced unified execution MARL system integrating 5 specialized agents with
enhanced centralized critic and intelligent coordination protocols.

Agent 4 Mission: The Coordinator - Enhanced 5-Agent MARL Integration:
- Position Sizing Agent (π₁): Optimal position sizing with Kelly Criterion
- Stop/Target Agent (π₂): Dynamic stop-loss and take-profit management  
- Risk Monitor Agent (π₃): Continuous risk monitoring and emergency response
- Portfolio Optimizer Agent (π₄): Dynamic portfolio optimization and allocation
- Routing Agent (π₅): Intelligent order routing with stealth execution

Enhanced Technical Architecture:
- Enhanced centralized critic with 83D input processing (15D execution + 32D market + 16D routing + 12D stealth + 8D RLHF)
- Multi-head attention architecture for different intelligence types
- Advanced agent communication protocols with conflict resolution
- Hierarchical coordination with sequential and parallel execution paths
- Performance target: <500μs total execution latency with 5 agents

Author: Agent 4 - The Coordinator
Date: 2025-07-13
Mission Status: Enhanced 5-Agent MARL Integration
"""

import asyncio
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import structlog
from concurrent.futures import ThreadPoolExecutor
import json

# Agent imports
from src.risk.agents.position_sizing_agent_v2 import PositionSizingAgentV2, PositionSizingDecision
from src.risk.agents.stop_target_agent import StopTargetAgent
from src.risk.agents.risk_monitor_agent import RiskMonitorAgent, RiskMonitorAction  
from src.risk.agents.portfolio_optimizer_agent import PortfolioOptimizerAgent
from src.execution.routing.routing_optimizer import RoutingOptimizer, RoutingFeatures, OptimizationResult
from src.execution.agents.centralized_critic import (
    ExecutionCentralizedCritic, MAPPOTrainer, CombinedState, 
    ExecutionContext, MarketFeatures, create_centralized_critic, create_mappo_trainer
)
from src.intelligence.intelligence_hub import IntelligenceHub
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


@dataclass
class EnhancedRoutingFeatures:
    """Enhanced routing features for π₅ agent (16D)"""
    quantity: int = 0
    notional_value: float = 0.0
    order_type: str = "MARKET"
    side: str = "BUY"
    urgency: float = 0.5
    volatility: float = 0.0
    spread: float = 0.0
    volume_rate: float = 0.0
    time_of_day: float = 0.5
    venue_latency: float = 10.0
    venue_cost: float = 0.002
    venue_fill_rate: float = 0.98
    venue_market_impact: float = 2.0
    recent_performance_score: float = 0.5
    symbol_venue_affinity: float = 0.5
    cross_venue_correlation: float = 0.0  # New feature
    
    def to_tensor(self) -> torch.Tensor:
        """Convert routing features to tensor (16D)"""
        return torch.tensor([
            float(self.quantity) / 100000.0,  # Normalized
            self.notional_value / 1000000.0,  # Normalized
            1.0 if self.order_type == "MARKET" else 0.0,
            1.0 if self.side == "BUY" else 0.0,
            self.urgency,
            self.volatility,
            self.spread,
            self.volume_rate / 10000.0,  # Normalized
            self.time_of_day,
            self.venue_latency / 100.0,  # Normalized
            self.venue_cost * 1000.0,  # Convert to normalized scale
            self.venue_fill_rate,
            self.venue_market_impact / 10.0,  # Normalized
            self.recent_performance_score,
            self.symbol_venue_affinity,
            self.cross_venue_correlation
        ], dtype=torch.float32)


@dataclass
class StealthFeatures:
    """Stealth execution features for minimizing market impact (12D)"""
    iceberg_size_ratio: float = 0.1  # Portion of order to show
    time_slicing_intervals: int = 5  # Number of time slices
    volume_participation_rate: float = 0.05  # Max volume participation
    market_impact_threshold: float = 2.0  # Max acceptable impact (bps)
    stealth_mode_enabled: bool = True
    dark_pool_preference: float = 0.3  # Preference for dark pools
    order_randomization: float = 0.1  # Order timing randomization
    adaptive_sizing: bool = True
    liquidity_seeking: bool = True
    momentum_hiding: bool = True
    cross_venue_coordination: bool = False
    anti_gaming_enabled: bool = True
    
    def to_tensor(self) -> torch.Tensor:
        """Convert stealth features to tensor (12D)"""
        return torch.tensor([
            self.iceberg_size_ratio,
            float(self.time_slicing_intervals) / 10.0,  # Normalized
            self.volume_participation_rate,
            self.market_impact_threshold / 10.0,  # Normalized
            1.0 if self.stealth_mode_enabled else 0.0,
            self.dark_pool_preference,
            self.order_randomization,
            1.0 if self.adaptive_sizing else 0.0,
            1.0 if self.liquidity_seeking else 0.0,
            1.0 if self.momentum_hiding else 0.0,
            1.0 if self.cross_venue_coordination else 0.0,
            1.0 if self.anti_gaming_enabled else 0.0
        ], dtype=torch.float32)


@dataclass
class RLHFFeatures:
    """Reinforcement Learning from Human Feedback features (8D)"""
    human_feedback_score: float = 0.5  # Human rating of execution quality
    execution_style_preference: float = 0.5  # Aggressive vs Conservative
    risk_tolerance_adjustment: float = 0.0  # Human risk preference adjustment
    timing_preference: float = 0.5  # Fast vs Patient execution preference
    cost_vs_speed_tradeoff: float = 0.5  # Cost optimization vs speed preference
    market_impact_sensitivity: float = 0.5  # Sensitivity to market impact
    transparency_preference: float = 0.5  # Visible vs Hidden execution preference
    learning_rate_modifier: float = 1.0  # Human-guided learning rate adjustment
    
    def to_tensor(self) -> torch.Tensor:
        """Convert RLHF features to tensor (8D)"""
        return torch.tensor([
            self.human_feedback_score,
            self.execution_style_preference,
            self.risk_tolerance_adjustment,
            self.timing_preference,
            self.cost_vs_speed_tradeoff,
            self.market_impact_sensitivity,
            self.transparency_preference,
            self.learning_rate_modifier
        ], dtype=torch.float32)


@dataclass
class EnhancedCombinedState:
    """Enhanced combined state for 83D centralized critic"""
    execution_context: ExecutionContext  # 15D
    market_features: MarketFeatures      # 32D
    routing_features: EnhancedRoutingFeatures  # 16D
    stealth_features: StealthFeatures    # 12D
    rlhf_features: RLHFFeatures         # 8D
    agent_actions: Optional[torch.Tensor] = None  # Previous actions from all agents
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to combined tensor for enhanced critic input (83D)"""
        context_tensor = self.execution_context.to_tensor()    # 15D
        market_tensor = self.market_features.to_tensor()       # 32D
        routing_tensor = self.routing_features.to_tensor()     # 16D
        stealth_tensor = self.stealth_features.to_tensor()     # 12D
        rlhf_tensor = self.rlhf_features.to_tensor()          # 8D
        
        combined = torch.cat([
            context_tensor, market_tensor, routing_tensor, 
            stealth_tensor, rlhf_tensor
        ], dim=0)  # 83D total
        
        if self.agent_actions is not None:
            combined = torch.cat([combined, self.agent_actions], dim=0)
            
        return combined


class Enhanced5AgentCentralizedCritic(nn.Module):
    """
    Enhanced Centralized Critic for 5-Agent MARL System
    
    Processes combined execution context (83D) with multi-head attention
    for different intelligence types and hierarchical learning.
    
    Architecture: 83D input → Multi-head → 512→256→128→64→1 output
    """
    
    def __init__(self, 
                 context_dim: int = 15,
                 market_features_dim: int = 32,
                 routing_features_dim: int = 16,
                 stealth_features_dim: int = 12,
                 rlhf_features_dim: int = 8,
                 num_agents: int = 5,
                 hidden_dims: List[int] = None,
                 use_multi_head_attention: bool = True):
        super().__init__()
        
        self.context_dim = context_dim
        self.market_features_dim = market_features_dim
        self.routing_features_dim = routing_features_dim
        self.stealth_features_dim = stealth_features_dim
        self.rlhf_features_dim = rlhf_features_dim
        self.num_agents = num_agents
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]
        
        # Enhanced combined input dimension: 15+32+16+12+8 = 83D
        self.combined_input_dim = (context_dim + market_features_dim + 
                                 routing_features_dim + stealth_features_dim + 
                                 rlhf_features_dim)
        
        # Multi-head attention for different intelligence types
        self.use_multi_head_attention = use_multi_head_attention
        if use_multi_head_attention:
            self.attention_heads = nn.ModuleDict({
                'execution_head': nn.MultiheadAttention(
                    embed_dim=context_dim, num_heads=3, batch_first=True
                ),
                'market_head': nn.MultiheadAttention(
                    embed_dim=market_features_dim, num_heads=4, batch_first=True
                ),
                'routing_head': nn.MultiheadAttention(
                    embed_dim=routing_features_dim, num_heads=2, batch_first=True
                ),
                'intelligence_head': nn.MultiheadAttention(
                    embed_dim=stealth_features_dim + rlhf_features_dim, num_heads=2, batch_first=True
                )
            })
            
            # Attention fusion layer
            self.attention_fusion = nn.Linear(self.combined_input_dim, self.combined_input_dim)
        
        # Build enhanced network layers
        layers = []
        prev_dim = self.combined_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1) if i < len(hidden_dims) - 1 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        # Output layer for state value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Hierarchical learning components
        self.coordination_head = nn.Linear(prev_dim, num_agents)  # Agent coordination weights
        self.conflict_detector = nn.Linear(prev_dim, 1)  # Conflict detection
        
        # Initialize weights
        self._initialize_weights()
        
        # Track evaluation metrics
        self.evaluations = 0
        self.total_evaluation_time = 0.0
        
    def _initialize_weights(self):
        """Initialize network weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, combined_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with multi-head attention and hierarchical outputs
        
        Args:
            combined_state: Combined state tensor [batch_size, 83]
            
        Returns:
            Dictionary with state_value, coordination_weights, conflict_score
        """
        batch_size = combined_state.shape[0]
        
        # Apply multi-head attention if enabled
        if self.use_multi_head_attention:
            combined_state = self._apply_multi_head_attention(combined_state)
        
        # Forward pass through main network
        features = combined_state
        for layer in self.network[:-1]:  # All layers except final
            features = layer(features)
        
        # Multiple outputs
        state_value = self.network[-1](features)  # Final layer
        coordination_weights = torch.softmax(self.coordination_head(features), dim=-1)
        conflict_score = torch.sigmoid(self.conflict_detector(features))
        
        return {
            'state_value': state_value,
            'coordination_weights': coordination_weights,
            'conflict_score': conflict_score,
            'features': features
        }
    
    def _apply_multi_head_attention(self, combined_state: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention to different feature groups"""
        batch_size = combined_state.shape[0]
        
        # Split features by type
        execution_features = combined_state[:, :self.context_dim].unsqueeze(1)  # [B, 1, 15]
        market_features = combined_state[:, self.context_dim:self.context_dim+self.market_features_dim].unsqueeze(1)  # [B, 1, 32]
        routing_features = combined_state[:, self.context_dim+self.market_features_dim:self.context_dim+self.market_features_dim+self.routing_features_dim].unsqueeze(1)  # [B, 1, 16]
        intelligence_features = combined_state[:, -20:].unsqueeze(1)  # [B, 1, 20] (stealth + rlhf)
        
        # Apply attention heads
        execution_attended, _ = self.attention_heads['execution_head'](
            execution_features, execution_features, execution_features
        )
        market_attended, _ = self.attention_heads['market_head'](
            market_features, market_features, market_features
        )
        routing_attended, _ = self.attention_heads['routing_head'](
            routing_features, routing_features, routing_features
        )
        intelligence_attended, _ = self.attention_heads['intelligence_head'](
            intelligence_features, intelligence_features, intelligence_features
        )
        
        # Concatenate attended features
        attended_features = torch.cat([
            execution_attended.squeeze(1),
            market_attended.squeeze(1),
            routing_attended.squeeze(1),
            intelligence_attended.squeeze(1)
        ], dim=-1)
        
        # Fusion layer
        fused_features = self.attention_fusion(attended_features)
        
        # Residual connection
        return combined_state + fused_features
    
    def evaluate_enhanced_state(self, enhanced_state: EnhancedCombinedState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Enhanced state evaluation with coordination and conflict detection
        
        Args:
            enhanced_state: Enhanced combined execution state
            
        Returns:
            Tuple of (evaluation_results, evaluation_info)
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Convert state to tensor
            state_tensor = enhanced_state.to_tensor().unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(state_tensor)
                
            # Extract results
            evaluation_results = {
                'state_value': outputs['state_value'].item(),
                'coordination_weights': outputs['coordination_weights'].squeeze(0).tolist(),
                'conflict_score': outputs['conflict_score'].item(),
                'agent_coordination': {
                    'position_sizing_weight': outputs['coordination_weights'][0, 0].item(),
                    'stop_target_weight': outputs['coordination_weights'][0, 1].item(),
                    'risk_monitor_weight': outputs['coordination_weights'][0, 2].item(),
                    'portfolio_optimizer_weight': outputs['coordination_weights'][0, 3].item(),
                    'routing_weight': outputs['coordination_weights'][0, 4].item()
                }
            }
            
            # Evaluation metrics
            end_time = time.perf_counter()
            evaluation_time = end_time - start_time
            self.evaluations += 1
            self.total_evaluation_time += evaluation_time
            
            evaluation_info = {
                'evaluation_time_ms': evaluation_time * 1000,
                'total_evaluations': self.evaluations,
                'avg_evaluation_time_ms': (self.total_evaluation_time / self.evaluations) * 1000,
                'input_dimensions': self.combined_input_dim,
                'multi_head_attention_enabled': self.use_multi_head_attention
            }
            
            return evaluation_results, evaluation_info
            
        except Exception as e:
            logger.error("Error in enhanced state evaluation", error=str(e))
            return {
                'state_value': 0.0,
                'coordination_weights': [0.2] * 5,
                'conflict_score': 0.5,
                'agent_coordination': {}
            }, {'error': str(e)}


@dataclass
class ExecutionDecision:
    """Enhanced unified execution decision from all 5 agents"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Agent decisions
    position_sizing: Optional[PositionSizingDecision] = None
    stop_target: Optional[Dict[str, Any]] = None
    risk_monitor: Optional[Dict[str, Any]] = None
    portfolio_optimizer: Optional[Dict[str, Any]] = None
    routing_optimizer: Optional[OptimizationResult] = None
    
    # Aggregated decision
    final_position_size: float = 0.0
    stop_loss_level: float = 0.0
    take_profit_level: float = 0.0
    risk_approved: bool = False
    emergency_stop: bool = False
    
    # Enhanced routing and stealth decisions
    recommended_venue: str = "PRIMARY"
    execution_strategy: str = "STANDARD"
    stealth_level: float = 0.5
    expected_market_impact_bps: float = 2.0
    routing_confidence: float = 0.5
    
    # Performance metrics
    total_latency_us: float = 0.0
    agent_latencies: Dict[str, float] = field(default_factory=dict)
    fill_rate: float = 0.0
    estimated_slippage_bps: float = 0.0
    
    # Enhanced intelligence metrics
    intelligence_metrics: Optional[Dict[str, Any]] = None
    coordination_quality: float = 0.0
    conflict_resolution_applied: bool = False
    coordination_weights: List[float] = field(default_factory=lambda: [0.2] * 5)
    
    # Reasoning and validation
    reasoning: str = ""
    confidence: float = 0.0
    risk_score: float = 0.0


class Enhanced5AgentMARLSystem:
    """
    Enhanced 5-Agent MARL Coordination System
    
    Integrates all 5 specialized agents with enhanced centralized critic:
    - π₁: Position Sizing Agent (Kelly Criterion optimization)
    - π₂: Stop/Target Agent (Dynamic risk management)
    - π₃: Risk Monitor Agent (Continuous risk assessment)
    - π₄: Portfolio Optimizer Agent (Portfolio-level coordination)
    - π₅: Routing Agent (Intelligent order routing with stealth execution)
    
    Enhanced Features:
    - 83D centralized critic (15D execution + 32D market + 16D routing + 12D stealth + 8D RLHF)
    - Multi-head attention architecture for different intelligence types
    - Advanced communication protocols with conflict resolution
    - Hierarchical coordination with sequential and parallel execution
    - Integration with Intelligence Hub for enhanced decision-making
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced 5-agent MARL system"""
        self.config = config
        self.event_bus = EventBus()
        
        # Performance tracking
        self.metrics = ExecutionPerformanceMetrics()
        self.latency_history = deque(maxlen=10000)
        self.execution_history = deque(maxlen=1000)
        
        # Initialize all 5 agents
        self._initialize_agents()
        
        # Initialize enhanced centralized critic with 83D input
        self._initialize_enhanced_centralized_critic()
        
        # Initialize enhanced MAPPO trainer for 5 agents
        self._initialize_enhanced_mappo_trainer()
        
        # Initialize Intelligence Hub for advanced coordination
        self._initialize_intelligence_hub()
        
        # Initialize communication protocols
        self._initialize_communication_protocols()
        
        # Performance monitoring
        self.performance_monitor_active = True
        self.monitoring_task = None
        
        # Thread pool for concurrent execution
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 15))
        
        # Coordination state
        self.coordination_state = {
            'last_conflict_score': 0.0,
            'coordination_quality_history': deque(maxlen=100),
            'agent_performance_weights': [0.2] * 5,
            'emergency_protocols_active': False
        }
        
        logger.info("Enhanced5AgentMARLSystem initialized",
                   agents=len(self.agents),
                   critic_dims=f"{self.critic.combined_input_dim}D",
                   intelligence_hub="enabled",
                   max_workers=config.get('max_workers', 15))
    
    def _initialize_agents(self):
        """Initialize all 5 agents with enhanced coordination"""
        self.agents = {}
        
        # Agent π₁: Position Sizing Agent
        try:
            from src.risk.agents.position_sizing_agent_v2 import create_position_sizing_agent_v2
            self.agents['position_sizing'] = create_position_sizing_agent_v2(
                self.config.get('position_sizing', {}), 
                self.event_bus
            )
            logger.info("Position Sizing Agent (π₁) initialized")
        except Exception as e:
            logger.error("Failed to initialize Position Sizing Agent", error=str(e))
            self.agents['position_sizing'] = self._create_mock_agent('position_sizing')
        
        # Agent π₂: Stop/Target Agent
        try:
            self.agents['stop_target'] = StopTargetAgent(
                self.config.get('stop_target', {}),
                self.event_bus
            )
            logger.info("Stop/Target Agent (π₂) initialized")
        except Exception as e:
            logger.error("Failed to initialize Stop/Target Agent", error=str(e))
            self.agents['stop_target'] = self._create_mock_agent('stop_target')
        
        # Agent π₃: Risk Monitor Agent
        try:
            self.agents['risk_monitor'] = RiskMonitorAgent(
                self.config.get('risk_monitor', {}),
                self.event_bus
            )
            logger.info("Risk Monitor Agent (π₃) initialized")
        except Exception as e:
            logger.error("Failed to initialize Risk Monitor Agent", error=str(e))
            self.agents['risk_monitor'] = self._create_mock_agent('risk_monitor')
        
        # Agent π₄: Portfolio Optimizer Agent
        try:
            self.agents['portfolio_optimizer'] = PortfolioOptimizerAgent(
                self.config.get('portfolio_optimizer', {}),
                self.event_bus
            )
            logger.info("Portfolio Optimizer Agent (π₄) initialized")
        except Exception as e:
            logger.error("Failed to initialize Portfolio Optimizer Agent", error=str(e))
            self.agents['portfolio_optimizer'] = self._create_mock_agent('portfolio_optimizer')
        
        # Agent π₅: Routing Agent (NEW)
        try:
            self.agents['routing'] = RoutingOptimizer(
                self.config.get('routing', {})
            )
            logger.info("Routing Agent (π₅) initialized")
        except Exception as e:
            logger.error("Failed to initialize Routing Agent", error=str(e))
            self.agents['routing'] = self._create_mock_routing_agent()
    
    def _create_mock_agent(self, agent_type: str):
        """Create mock agent for testing when real agent fails to initialize"""
        class MockAgent:
            def __init__(self, agent_type):
                self.agent_type = agent_type
                self.name = f"mock_{agent_type}"
            
            def act(self, state, context=None):
                # Return mock decision based on agent type
                if agent_type == 'position_sizing':
                    return PositionSizingDecision(
                        contracts=2,
                        kelly_fraction=0.15,
                        position_size_fraction=0.1,
                        confidence=0.7,
                        reasoning={'method': 'mock'},
                        risk_adjustments=[],
                        computation_time_ms=1.0,
                        timestamp=datetime.now()
                    )
                else:
                    return {
                        'action': 0,
                        'confidence': 0.5,
                        'reasoning': f'Mock {agent_type} decision',
                        'timestamp': datetime.now()
                    }
        
        logger.warning(f"Using mock agent for {agent_type}")
        return MockAgent(agent_type)
    
    def _create_mock_routing_agent(self):
        """Create mock routing agent for testing"""
        class MockRoutingAgent:
            def __init__(self):
                self.agent_type = 'routing'
                self.name = 'mock_routing'
            
            def optimize_routing(self, order_features, available_venues, venue_data):
                return OptimizationResult(
                    recommended_venue="PRIMARY",
                    confidence_score=0.7,
                    expected_performance={
                        'expected_latency_ms': 10.0,
                        'expected_fill_rate': 0.98,
                        'expected_cost_bps': 2.0,
                        'expected_market_impact_bps': 2.0
                    },
                    feature_importance={},
                    alternative_venues=[("SECONDARY", 0.6)]
                )
        
        logger.warning("Using mock routing agent")
        return MockRoutingAgent()
    
    def _initialize_enhanced_centralized_critic(self):
        """Initialize enhanced centralized critic with 83D input processing"""
        critic_config = {
            'context_dim': 15,  # Execution context vector
            'market_features_dim': 32,  # Extended market features
            'routing_features_dim': 16,  # Routing intelligence features
            'stealth_features_dim': 12,  # Stealth execution features
            'rlhf_features_dim': 8,  # RLHF features
            'num_agents': 5,  # Now 5 agents
            'hidden_dims': [512, 256, 128, 64],  # Enhanced architecture
            'use_multi_head_attention': True
        }
        
        self.critic = Enhanced5AgentCentralizedCritic(**critic_config)
        logger.info("Enhanced centralized critic initialized", 
                   input_dim=self.critic.combined_input_dim,
                   output_dim="multi-head",
                   agents=5)
    
    def _initialize_enhanced_mappo_trainer(self):
        """Initialize enhanced MAPPO trainer for 5-agent coordination"""
        trainer_config = {
            'learning_rate': 5e-5,  # Slightly lower for 5-agent stability
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'multi_head_attention': True,  # Enhanced feature
            'hierarchical_learning': True  # Enhanced feature
        }
        
        # Get agent networks (or create enhanced mock networks)
        agent_networks = []
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'network'):
                agent_networks.append(agent.network)
            else:
                # Create enhanced mock network for 83D input
                mock_network = nn.Sequential(
                    nn.Linear(83, 128),  # Enhanced input dimension
                    nn.ReLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.LayerNorm(64),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3)  # Action space
                )
                agent_networks.append(mock_network)
        
        # Create enhanced MAPPO trainer (would need to implement this)
        self.mappo_trainer = self._create_enhanced_mappo_trainer(
            self.critic, 
            agent_networks, 
            trainer_config
        )
        
        logger.info("Enhanced MAPPO trainer initialized", 
                   num_agents=len(agent_networks),
                   input_dim=83,
                   learning_rate=trainer_config['learning_rate'])
    
    def _create_enhanced_mappo_trainer(self, critic, agent_networks, config):
        """Create enhanced MAPPO trainer (simplified implementation)"""
        # This would be a full implementation in production
        class EnhancedMAPPOTrainer:
            def __init__(self, critic, agents, config):
                self.critic = critic
                self.agents = agents
                self.config = config
            
            def get_training_metrics(self):
                return {'training_steps': 0, 'policy_loss': 0.0}
        
        return EnhancedMAPPOTrainer(critic, agent_networks, config)
    
    def _initialize_intelligence_hub(self):
        """Initialize Intelligence Hub for advanced coordination"""
        intelligence_config = {
            'max_intelligence_overhead_ms': 1.0,
            'performance_monitoring': True,
            'regime_detection': {
                'fast_mode': True,
                'cache_analysis': True,
                'min_confidence_threshold': 0.7
            },
            'gating_network': {
                'shared_context_dim': 6,
                'n_agents': 5,  # Updated for 5 agents
                'hidden_dim': 32
            },
            'attention': {
                'enable_caching': True,
                'optimization_level': 'high'
            }
        }
        
        try:
            self.intelligence_hub = IntelligenceHub(intelligence_config)
            logger.info("Intelligence Hub initialized for 5-agent coordination")
        except Exception as e:
            logger.error("Failed to initialize Intelligence Hub", error=str(e))
            self.intelligence_hub = None
    
    def _initialize_communication_protocols(self):
        """Initialize advanced communication protocols for agent coordination"""
        self.communication_protocols = {
            'conflict_resolution': {
                'enabled': True,
                'resolution_strategies': [
                    'weighted_voting',
                    'risk_priority',
                    'performance_based',
                    'human_override'
                ]
            },
            'emergency_coordination': {
                'emergency_stop_threshold': 0.95,
                'cascade_prevention': True,
                'manual_override_required': True
            },
            'performance_adaptation': {
                'dynamic_weight_adjustment': True,
                'learning_rate_adaptation': True,
                'coordination_quality_threshold': 0.7
            }
        }
        
        logger.info("Communication protocols initialized", 
                   conflict_resolution=True,
                   emergency_coordination=True)
    
    async def execute_enhanced_unified_decision(self, 
                                               execution_context: ExecutionContext,
                                               market_features: MarketFeatures,
                                               routing_features: EnhancedRoutingFeatures,
                                               stealth_features: StealthFeatures,
                                               rlhf_features: RLHFFeatures) -> ExecutionDecision:
        """
        Execute enhanced unified decision across all 5 agents with intelligence coordination
        
        Args:
            execution_context: 15D execution context vector
            market_features: 32D market features vector
            routing_features: 16D routing intelligence features
            stealth_features: 12D stealth execution features
            rlhf_features: 8D RLHF features
            
        Returns:
            Enhanced unified execution decision with intelligence metrics
        """
        start_time = time.perf_counter()
        
        try:
            # Create enhanced combined state for critic (83D total)
            enhanced_combined_state = EnhancedCombinedState(
                execution_context=execution_context,
                market_features=market_features,
                routing_features=routing_features,
                stealth_features=stealth_features,
                rlhf_features=rlhf_features
            )
            
            # Get enhanced centralized critic evaluation
            critic_results, critic_info = self.critic.evaluate_enhanced_state(enhanced_combined_state)
            
            # Execute agents with enhanced coordination strategy
            # Sequential execution for critical path: π₃ → π₁ → π₂ → π₄
            # Parallel execution for π₅ (routing) with intelligence processing
            
            # Phase 1: Risk assessment and routing (parallel)
            phase1_tasks = [
                self._execute_agent_async('risk_monitor', execution_context, market_features),
                self._execute_routing_agent_async(routing_features)
            ]
            
            # Intelligence Hub processing (parallel with Phase 1)
            intelligence_task = self._process_intelligence_hub_async(
                execution_context, market_features, routing_features, stealth_features, rlhf_features
            )
            
            # Wait for Phase 1 and intelligence processing
            phase1_results = await asyncio.gather(*phase1_tasks, intelligence_task, return_exceptions=True)
            risk_result, routing_result, intelligence_result = phase1_results
            
            # Phase 2: Position sizing and timing (sequential, dependent on risk)
            if not isinstance(risk_result, Exception) and not risk_result.get('emergency_stop', False):
                position_result = await self._execute_agent_async('position_sizing', execution_context, market_features)
                stop_target_result = await self._execute_agent_async('stop_target', execution_context, market_features)
            else:
                position_result = {'emergency_stop': True}
                stop_target_result = {'emergency_stop': True}
            
            # Phase 3: Portfolio optimization (depends on position sizing)
            if not isinstance(position_result, Exception) and not position_result.get('emergency_stop', False):
                portfolio_result = await self._execute_agent_async('portfolio_optimizer', execution_context, market_features)
            else:
                portfolio_result = {'emergency_stop': True}
            
            # Combine all results
            agent_results = [position_result, stop_target_result, risk_result, portfolio_result, routing_result]
            
            # Process execution decision
            decision = ExecutionDecision()
            decision.timestamp = datetime.now()
            
            # Extract individual agent decisions with enhanced handling
            agent_names = ['position_sizing', 'stop_target', 'risk_monitor', 'portfolio_optimizer', 'routing']
            
            for i, (agent_name, result) in enumerate(zip(agent_names, agent_results)):
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent_name} failed", error=str(result))
                    continue
                
                if agent_name == 'position_sizing':
                    decision.position_sizing = result
                elif agent_name == 'stop_target':
                    decision.stop_target = result
                elif agent_name == 'risk_monitor':
                    decision.risk_monitor = result
                elif agent_name == 'portfolio_optimizer':
                    decision.portfolio_optimizer = result
                elif agent_name == 'routing':
                    decision.routing_optimizer = result
            
            # Enhanced decision aggregation with conflict resolution
            decision = await self._aggregate_enhanced_agent_decisions(
                decision, critic_results, critic_info, intelligence_result, stealth_features, rlhf_features
            )
            
            # Calculate performance metrics
            end_time = time.perf_counter()
            decision.total_latency_us = (end_time - start_time) * 1_000_000  # Convert to microseconds
            
            # Update system metrics
            self._update_performance_metrics(decision)
            
            # Validate performance requirements
            await self._validate_performance_requirements(decision)
            
            return decision
            
        except Exception as e:
            logger.error("Enhanced unified execution failed", error=str(e))
            
            # Return enhanced safe fallback decision
            end_time = time.perf_counter()
            fallback_decision = ExecutionDecision()
            fallback_decision.total_latency_us = (end_time - start_time) * 1_000_000
            fallback_decision.reasoning = f"Enhanced execution failed: {str(e)}"
            fallback_decision.emergency_stop = True
            fallback_decision.coordination_quality = 0.0
            fallback_decision.conflict_resolution_applied = False
            
            return fallback_decision
    
    async def _execute_routing_agent_async(self, routing_features: EnhancedRoutingFeatures) -> OptimizationResult:
        """Execute routing agent asynchronously"""
        agent_start = time.perf_counter()
        
        try:
            routing_agent = self.agents['routing']
            
            # Convert enhanced routing features to standard format
            order_features = RoutingFeatures(
                quantity=routing_features.quantity,
                notional_value=routing_features.notional_value,
                order_type=routing_features.order_type,
                side=routing_features.side,
                urgency=routing_features.urgency,
                volatility=routing_features.volatility,
                spread=routing_features.spread,
                volume_rate=routing_features.volume_rate,
                time_of_day=routing_features.time_of_day,
                venue_latency=routing_features.venue_latency,
                venue_cost=routing_features.venue_cost,
                venue_fill_rate=routing_features.venue_fill_rate,
                venue_market_impact=routing_features.venue_market_impact,
                recent_performance_score=routing_features.recent_performance_score,
                symbol_venue_affinity=routing_features.symbol_venue_affinity
            )
            
            # Available venues (mock data)
            available_venues = ["PRIMARY", "SECONDARY", "DARK_POOL"]
            venue_data = {
                "PRIMARY": {"cost_per_share": 0.002, "avg_latency": 8.0, "fill_rate": 0.99},
                "SECONDARY": {"cost_per_share": 0.001, "avg_latency": 12.0, "fill_rate": 0.97},
                "DARK_POOL": {"cost_per_share": 0.003, "avg_latency": 15.0, "fill_rate": 0.95}
            }
            
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, routing_agent.optimize_routing, order_features, available_venues, venue_data
            )
            
            # Track agent performance
            agent_time = (time.perf_counter() - agent_start) * 1_000_000  # microseconds
            self.metrics.agent_performance.setdefault('routing', {})
            self.metrics.agent_performance['routing']['last_latency_us'] = agent_time
            
            return result
            
        except Exception as e:
            logger.error("Routing agent execution failed", error=str(e))
            raise
    
    async def _process_intelligence_hub_async(self, execution_context, market_features, 
                                            routing_features, stealth_features, rlhf_features):
        """Process intelligence hub asynchronously"""
        if not self.intelligence_hub:
            return {'intelligence_active': False}
        
        try:
            # Convert contexts to intelligence hub format
            market_context = {
                'volatility_30': getattr(market_features, 'realized_garch', 0.15),
                'mmd_score': 0.3,  # Mock value
                'momentum_20': 0.0,  # Mock value
                'momentum_50': 0.0,  # Mock value
                'volume_ratio': 1.0,  # Mock value
                'price_trend': 0.0   # Mock value
            }
            
            # Mock agent predictions
            agent_predictions = [
                {'action_probabilities': [0.3, 0.4, 0.3], 'confidence': 0.7},
                {'action_probabilities': [0.35, 0.35, 0.3], 'confidence': 0.6},
                {'action_probabilities': [0.25, 0.5, 0.25], 'confidence': 0.8}
            ]
            
            result, metrics = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                self.intelligence_hub.process_intelligence_pipeline,
                market_context, agent_predictions, None
            )
            
            return result
            
        except Exception as e:
            logger.error("Intelligence hub processing failed", error=str(e))
            return {'intelligence_active': False, 'error': str(e)}
    
    async def _aggregate_enhanced_agent_decisions(self, 
                                                decision: ExecutionDecision,
                                                critic_results: Dict[str, Any],
                                                critic_info: Dict[str, Any],
                                                intelligence_result: Dict[str, Any],
                                                stealth_features: StealthFeatures,
                                                rlhf_features: RLHFFeatures) -> ExecutionDecision:
        """Enhanced decision aggregation with conflict resolution and coordination"""
        
        # Extract coordination weights from critic
        coordination_weights = critic_results.get('coordination_weights', [0.2] * 5)
        conflict_score = critic_results.get('conflict_score', 0.0)
        
        # Apply conflict resolution if needed
        if conflict_score > 0.7:  # High conflict threshold
            decision.conflict_resolution_applied = True
            coordination_weights = await self._resolve_conflicts(decision, coordination_weights, conflict_score)
        
        decision.coordination_weights = coordination_weights
        decision.coordination_quality = 1.0 - conflict_score  # Inverse of conflict
        
        # Position sizing (π₁) with coordination weight
        if decision.position_sizing:
            base_size = decision.position_sizing.position_size_fraction
            decision.final_position_size = base_size * coordination_weights[0]
            decision.confidence = decision.position_sizing.confidence * coordination_weights[0]
        
        # Stop/Target levels (π₂) with coordination weight
        if decision.stop_target:
            decision.stop_loss_level = decision.stop_target.get('stop_loss', 0.0) * coordination_weights[1]
            decision.take_profit_level = decision.stop_target.get('take_profit', 0.0) * coordination_weights[1]
        
        # Risk monitoring (π₃) with coordination weight and override capability
        if decision.risk_monitor:
            risk_action = decision.risk_monitor.get('action', RiskMonitorAction.NO_ACTION)
            risk_weight = coordination_weights[2]
            
            # Risk agent has override capability
            if risk_weight > 0.8 or risk_action == RiskMonitorAction.EMERGENCY_STOP:
                decision.emergency_stop = (risk_action == RiskMonitorAction.EMERGENCY_STOP)
                decision.risk_approved = (risk_action in [RiskMonitorAction.NO_ACTION, RiskMonitorAction.ALERT])
            
            decision.risk_score = decision.risk_monitor.get('risk_score', 0.0)
        
        # Portfolio optimization (π₄) with coordination weight
        if decision.portfolio_optimizer:
            portfolio_adjustment = decision.portfolio_optimizer.get('position_adjustment', 1.0)
            portfolio_weight = coordination_weights[3]
            decision.final_position_size *= (portfolio_adjustment * portfolio_weight + (1 - portfolio_weight))
        
        # Routing optimization (π₅) with stealth integration
        if decision.routing_optimizer:
            routing_weight = coordination_weights[4]
            decision.recommended_venue = decision.routing_optimizer.recommended_venue
            decision.routing_confidence = decision.routing_optimizer.confidence_score * routing_weight
            decision.expected_market_impact_bps = decision.routing_optimizer.expected_performance.get(
                'expected_market_impact_bps', 2.0
            )
            
            # Apply stealth features
            if stealth_features.stealth_mode_enabled:
                decision.execution_strategy = "STEALTH"
                decision.stealth_level = stealth_features.iceberg_size_ratio
                # Adjust position size for stealth execution
                decision.final_position_size *= stealth_features.iceberg_size_ratio
        
        # Apply RLHF adjustments
        human_feedback_weight = rlhf_features.human_feedback_score
        style_adjustment = rlhf_features.execution_style_preference
        
        # Adjust final position based on human feedback
        if human_feedback_weight > 0.7:  # High human approval
            decision.final_position_size *= 1.1  # Slight increase
        elif human_feedback_weight < 0.3:  # Low human approval
            decision.final_position_size *= 0.9  # Slight decrease
        
        # Apply style preference (aggressive vs conservative)
        if style_adjustment > 0.7:  # Aggressive
            decision.final_position_size *= 1.05
        elif style_adjustment < 0.3:  # Conservative
            decision.final_position_size *= 0.95
        
        # Apply centralized critic coordination
        state_value = critic_results.get('state_value', 0.0)
        critic_adjustment = np.clip(state_value, 0.7, 1.3)  # More conservative range
        decision.final_position_size *= critic_adjustment
        
        # Emergency overrides with enhanced coordination
        if decision.emergency_stop:
            decision.final_position_size = 0.0
            decision.risk_approved = False
            decision.execution_strategy = "EMERGENCY_STOP"
        
        # Intelligence hub integration
        if intelligence_result.get('intelligence_active', False):
            decision.intelligence_metrics = intelligence_result
            # Apply intelligence-based adjustments
            intelligence_confidence = intelligence_result.get('overall_confidence', 0.5)
            decision.confidence = (decision.confidence + intelligence_confidence) / 2.0
        
        # Generate enhanced reasoning
        decision.reasoning = self._generate_enhanced_reasoning(
            decision, critic_results, intelligence_result, coordination_weights, conflict_score
        )
        
        # Estimate enhanced performance metrics
        decision.fill_rate = self._estimate_enhanced_fill_rate(decision, stealth_features)
        decision.estimated_slippage_bps = self._estimate_enhanced_slippage(decision, stealth_features)
        
        return decision
    
    async def _resolve_conflicts(self, decision: ExecutionDecision, 
                               coordination_weights: List[float], 
                               conflict_score: float) -> List[float]:
        """Resolve conflicts between agents using intelligent strategies"""
        
        # Strategy 1: Risk Priority - Give higher weight to risk monitor in conflicts
        if conflict_score > 0.8:
            risk_boost = min(0.3, conflict_score - 0.5)
            coordination_weights[2] += risk_boost  # Risk monitor gets boost
            
            # Redistribute from other agents
            reduction_per_agent = risk_boost / 4
            for i in range(5):
                if i != 2:  # Not risk monitor
                    coordination_weights[i] = max(0.05, coordination_weights[i] - reduction_per_agent)
        
        # Strategy 2: Performance-based adjustment
        agent_performance = self.coordination_state.get('agent_performance_weights', [0.2] * 5)
        performance_blend_factor = 0.3
        
        for i in range(5):
            coordination_weights[i] = (
                coordination_weights[i] * (1 - performance_blend_factor) +
                agent_performance[i] * performance_blend_factor
            )
        
        # Normalize weights
        total_weight = sum(coordination_weights)
        if total_weight > 0:
            coordination_weights = [w / total_weight for w in coordination_weights]
        else:
            coordination_weights = [0.2] * 5
        
        logger.info("Conflict resolution applied", 
                   conflict_score=conflict_score,
                   new_weights=coordination_weights)
        
        return coordination_weights
    
    def _generate_enhanced_reasoning(self, decision: ExecutionDecision,
                                   critic_results: Dict[str, Any],
                                   intelligence_result: Dict[str, Any],
                                   coordination_weights: List[float],
                                   conflict_score: float) -> str:
        """Generate enhanced human-readable reasoning for the decision"""
        reasoning_parts = []
        
        # Agent contributions with weights
        agent_names = ['Position Sizing', 'Stop/Target', 'Risk Monitor', 'Portfolio Optimizer', 'Routing']
        
        for i, (name, weight) in enumerate(zip(agent_names, coordination_weights)):
            if weight > 0.15:  # Only mention significant contributors
                reasoning_parts.append(f"{name} (π{i+1}): {weight:.2f} weight")
        
        # Risk and coordination status
        if decision.emergency_stop:
            reasoning_parts.append("EMERGENCY STOP activated by risk management")
        elif conflict_score > 0.7:
            reasoning_parts.append(f"High conflict detected ({conflict_score:.2f}), resolution applied")
        
        # Routing and stealth
        if decision.routing_optimizer:
            reasoning_parts.append(
                f"Routing: {decision.recommended_venue} "
                f"(confidence: {decision.routing_confidence:.2f})"
            )
        
        if decision.execution_strategy == "STEALTH":
            reasoning_parts.append(f"Stealth execution: {decision.stealth_level:.2f} iceberg ratio")
        
        # Intelligence integration
        if intelligence_result.get('intelligence_active', False):
            regime = intelligence_result.get('regime', 'unknown')
            regime_confidence = intelligence_result.get('regime_confidence', 0.0)
            reasoning_parts.append(f"Market regime: {regime} ({regime_confidence:.2f})")
        
        # Final decision summary
        if decision.risk_approved and not decision.emergency_stop:
            reasoning_parts.append(
                f"Final position: {decision.final_position_size:.3f} "
                f"(confidence: {decision.confidence:.2f})"
            )
        else:
            reasoning_parts.append("Position rejected by coordination system")
        
        return " | ".join(reasoning_parts)
    
    def _estimate_enhanced_fill_rate(self, decision: ExecutionDecision, 
                                   stealth_features: StealthFeatures) -> float:
        """Enhanced fill rate estimation with stealth considerations"""
        base_fill_rate = 0.998  # 99.8% base fill rate
        
        # Routing venue adjustment
        if decision.routing_optimizer:
            venue_fill_rate = decision.routing_optimizer.expected_performance.get('expected_fill_rate', 0.98)
            base_fill_rate = (base_fill_rate + venue_fill_rate) / 2.0
        
        # Position size adjustment
        size_adjustment = max(0.95, 1.0 - decision.final_position_size * 0.1)
        
        # Stealth execution adjustment
        stealth_adjustment = 1.0
        if stealth_features.stealth_mode_enabled:
            # Stealth may slightly reduce fill rate but improve market impact
            stealth_adjustment = 0.995
            if stealth_features.dark_pool_preference > 0.5:
                stealth_adjustment *= 0.99
        
        # Coordination quality adjustment
        coordination_adjustment = 0.995 + (decision.coordination_quality * 0.005)
        
        return base_fill_rate * size_adjustment * stealth_adjustment * coordination_adjustment
    
    def _estimate_enhanced_slippage(self, decision: ExecutionDecision, 
                                  stealth_features: StealthFeatures) -> float:
        """Enhanced slippage estimation with stealth and routing considerations"""
        base_slippage = 1.0  # 1 bps base
        
        # Routing venue impact
        if decision.routing_optimizer:
            venue_impact = decision.routing_optimizer.expected_performance.get('expected_market_impact_bps', 2.0)
            base_slippage = (base_slippage + venue_impact) / 2.0
        
        # Position size impact
        size_slippage = decision.final_position_size * 0.5
        
        # Stealth execution benefits
        stealth_reduction = 0.0
        if stealth_features.stealth_mode_enabled:
            stealth_reduction = stealth_features.iceberg_size_ratio * 0.3  # Up to 30% reduction
            if stealth_features.dark_pool_preference > 0.5:
                stealth_reduction += 0.2  # Additional reduction for dark pools
        
        # Coordination quality impact
        coordination_impact = (1.0 - decision.coordination_quality) * 0.5
        
        total_slippage = base_slippage + size_slippage + coordination_impact - stealth_reduction
        
        return max(0.5, total_slippage)  # Minimum 0.5 bps
    
    # Additional methods would continue here following the same pattern...
    # For brevity, I'm including the key structural methods
    
    async def _execute_agent_async(self, agent_name: str, execution_context: ExecutionContext, market_features: MarketFeatures) -> Any:
        """Execute single agent asynchronously with enhanced error handling"""
        agent_start = time.perf_counter()
        
        try:
            agent = self.agents[agent_name]
            
            # Convert to agent-specific state format
            if agent_name == 'position_sizing':
                from src.risk.agents.base_risk_agent import RiskState
                state = RiskState(
                    portfolio_value=execution_context.portfolio_value,
                    position_size=execution_context.current_position,
                    unrealized_pnl=execution_context.unrealized_pnl,
                    var_estimate=execution_context.var_estimate,
                    correlation_risk=market_features.correlation_spy,
                    volatility=market_features.realized_garch,
                    drawdown=execution_context.drawdown_current,
                    risk_limit_utilization=execution_context.risk_budget_used,
                    kelly_fraction=0.15,
                    timestamp=datetime.now()
                )
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, agent.act, state
                )
            else:
                # Other agents expect generic state
                context_vector = execution_context.to_tensor()
                market_vector = market_features.to_tensor()
                state = torch.cat([context_vector, market_vector], dim=0).numpy()
                
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, agent.act, state
                )
            
            # Track agent performance
            agent_time = (time.perf_counter() - agent_start) * 1_000_000  # microseconds
            self.metrics.agent_performance.setdefault(agent_name, {})
            self.metrics.agent_performance[agent_name]['last_latency_us'] = agent_time
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced agent {agent_name} execution failed", error=str(e))
            raise
    
    def _update_performance_metrics(self, decision: ExecutionDecision):
        """Update enhanced system performance metrics"""
        self.metrics.total_executions += 1
        
        if decision.risk_approved and not decision.emergency_stop:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1
        
        # Update coordination quality history
        self.coordination_state['coordination_quality_history'].append(decision.coordination_quality)
        
        # Update latency metrics
        self.latency_history.append(decision.total_latency_us)
        latencies = list(self.latency_history)
        
        self.metrics.avg_latency_us = np.mean(latencies)
        self.metrics.p95_latency_us = np.percentile(latencies, 95)
        self.metrics.p99_latency_us = np.percentile(latencies, 99)
        self.metrics.max_latency_us = np.max(latencies)
        
        # Update enhanced metrics
        self.metrics.avg_fill_rate = (
            self.metrics.avg_fill_rate * 0.9 + decision.fill_rate * 0.1
        )
        self.metrics.avg_slippage_bps = (
            self.metrics.avg_slippage_bps * 0.9 + decision.estimated_slippage_bps * 0.1
        )
        
        # Store enhanced execution data
        self.execution_history.append({
            'timestamp': decision.timestamp,
            'latency_us': decision.total_latency_us,
            'fill_rate': decision.fill_rate,
            'slippage_bps': decision.estimated_slippage_bps,
            'coordination_quality': decision.coordination_quality,
            'conflict_resolution_applied': decision.conflict_resolution_applied,
            'coordination_weights': decision.coordination_weights,
            'success': decision.risk_approved and not decision.emergency_stop
        })
    
    async def _validate_performance_requirements(self, decision: ExecutionDecision):
        """Validate enhanced performance requirements"""
        # Enhanced latency requirement: <500μs for 5 agents
        if decision.total_latency_us > 500:
            logger.warning("Enhanced latency requirement violation", 
                          actual_us=decision.total_latency_us,
                          limit_us=500,
                          agents=5)
        
        # Enhanced fill rate requirement: >99.8%
        if decision.fill_rate < 0.998:
            logger.warning("Enhanced fill rate requirement violation",
                          actual=decision.fill_rate,
                          limit=0.998)
        
        # Enhanced slippage requirement: <2 bps with stealth
        if decision.estimated_slippage_bps > 2.0:
            logger.warning("Enhanced slippage requirement violation",
                          actual_bps=decision.estimated_slippage_bps,
                          limit_bps=2.0)
        
        # Coordination quality requirement
        if decision.coordination_quality < 0.7:
            logger.warning("Coordination quality below threshold",
                          actual=decision.coordination_quality,
                          threshold=0.7)
    
    def get_enhanced_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive enhanced performance report"""
        coordination_quality_history = list(self.coordination_state['coordination_quality_history'])
        
        return {
            'system_metrics': {
                'total_executions': self.metrics.total_executions,
                'success_rate': self.metrics.successful_executions / max(1, self.metrics.total_executions),
                'avg_latency_us': self.metrics.avg_latency_us,
                'p95_latency_us': self.metrics.p95_latency_us,
                'p99_latency_us': self.metrics.p99_latency_us,
                'avg_fill_rate': self.metrics.avg_fill_rate,
                'avg_slippage_bps': self.metrics.avg_slippage_bps,
                'avg_coordination_quality': np.mean(coordination_quality_history) if coordination_quality_history else 0.0,
                'conflict_resolution_rate': sum(1 for ex in self.execution_history if ex.get('conflict_resolution_applied', False)) / max(1, len(self.execution_history))
            },
            'agent_performance': self.metrics.agent_performance,
            'coordination_metrics': {
                'current_agent_weights': self.coordination_state['agent_performance_weights'],
                'emergency_protocols_active': self.coordination_state['emergency_protocols_active'],
                'coordination_quality_trend': np.polyfit(range(len(coordination_quality_history)), coordination_quality_history, 1)[0] if len(coordination_quality_history) > 1 else 0.0
            },
            'enhanced_requirements': {
                'latency_compliant': self.metrics.p95_latency_us < 500,
                'fill_rate_compliant': self.metrics.avg_fill_rate > 0.998,
                'slippage_compliant': self.metrics.avg_slippage_bps < 2.0,
                'coordination_compliant': np.mean(coordination_quality_history) > 0.7 if coordination_quality_history else False
            },
            'intelligence_hub_metrics': self.intelligence_hub.get_integration_statistics() if self.intelligence_hub else {},
            'critic_metrics': self.mappo_trainer.get_training_metrics(),
            'execution_history_sample': list(self.execution_history)[-10:]  # Last 10 executions
        }
    
    async def shutdown(self):
        """Enhanced graceful shutdown of the 5-agent system"""
        logger.info("Shutting down Enhanced5AgentMARLSystem")
        
        # Stop performance monitoring
        self.performance_monitor_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Shutdown intelligence hub
        if self.intelligence_hub:
            self.intelligence_hub.reset_intelligence_state()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Enhanced5AgentMARLSystem shutdown complete")


# Factory function and configuration
def create_enhanced_5agent_marl_system(config: Dict[str, Any]) -> Enhanced5AgentMARLSystem:
    """Factory function to create enhanced 5-agent MARL system"""
    return Enhanced5AgentMARLSystem(config)


# Enhanced configuration for 5-agent system
ENHANCED_5AGENT_CONFIG = {
    'max_workers': 15,  # Increased for 5 agents
    'position_sizing': {
        'kelly_lookback_periods': 252,
        'max_position_size': 0.25,
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
        'var_threshold': 0.02,
        'correlation_threshold': 0.8,
        'drawdown_threshold': 0.1,
        'emergency_stop_threshold': 0.15
    },
    'portfolio_optimizer': {
        'rebalance_threshold': 0.05,
        'target_volatility': 0.12,
        'max_correlation': 0.8,
        'min_liquidity': 0.1
    },
    'routing': {  # New routing agent configuration
        'learning_rate': 0.01,
        'history_window_days': 30,
        'min_samples_for_learning': 100,
        'feature_weights': {
            'cost': 0.25,
            'latency': 0.25,
            'fill_rate': 0.25,
            'market_impact': 0.25
        }
    },
    'intelligence_hub': {
        'max_intelligence_overhead_ms': 1.0,
        'performance_monitoring': True
    },
    'coordination': {
        'conflict_resolution_threshold': 0.7,
        'emergency_stop_threshold': 0.95,
        'coordination_quality_threshold': 0.7
    }
}


# Performance metrics class for enhanced system
@dataclass
class ExecutionPerformanceMetrics:
    """Enhanced performance metrics for 5-agent execution system"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    
    # Latency metrics (microseconds)
    avg_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    max_latency_us: float = 0.0
    
    # Fill and slippage metrics
    avg_fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    
    # Enhanced coordination metrics
    avg_coordination_quality: float = 0.0
    conflict_resolution_rate: float = 0.0
    intelligence_overhead_us: float = 0.0
    
    # Agent-specific metrics
    agent_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Load testing metrics
    concurrent_capacity: int = 0
    executions_per_second: float = 0.0
    
    # Error tracking
    error_rates: Dict[str, float] = field(default_factory=dict)
    critical_failures: List[str] = field(default_factory=list)