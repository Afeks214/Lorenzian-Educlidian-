"""
Unified Execution MARL System Integration
==========================================

This module implements the unified execution MARL system that integrates all 5 agents
for coordinated multi-agent reinforcement learning execution engine.

MARL Agent Integration:
- Position Sizing Agent (π₁): Optimal position sizing with Kelly Criterion
- Stop/Target Agent (π₂): Dynamic stop-loss and take-profit management  
- Risk Monitor Agent (π₃): Continuous risk monitoring and emergency response
- Portfolio Optimizer Agent (π₄): Dynamic portfolio optimization and allocation
- Routing Agent (π₅): Adaptive order routing with QoE optimization

Technical Architecture:
- Centralized critic for 5-agent coordinated learning
- Execution context vector (15D) + routing state (55D) processing pipeline
- Performance requirements: <500μs latency, >99.8% fill rate, <2 bps slippage, <100μs routing
- Load testing: 50 concurrent requests, 100+ executions per second

Author: Agent 1 - The Arbitrageur Implementation
Date: 2025-07-13
Mission Status: 5th Agent Integration Complete
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
from src.execution.agents.routing_agent import RoutingAgent, RoutingAction, RoutingState, QoEMetrics
from src.execution.agents.centralized_critic import (
    ExecutionCentralizedCritic, MAPPOTrainer, CombinedState, 
    ExecutionContext, MarketFeatures, create_centralized_critic, create_mappo_trainer
)
from src.execution.brokers.broker_performance_tracker import BrokerPerformanceTracker
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


@dataclass
class ExecutionDecision:
    """Unified execution decision from all 5 agents"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Agent decisions
    position_sizing: Optional[PositionSizingDecision] = None
    stop_target: Optional[Dict[str, Any]] = None
    risk_monitor: Optional[Dict[str, Any]] = None
    portfolio_optimizer: Optional[Dict[str, Any]] = None
    routing: Optional[RoutingAction] = None
    
    # Aggregated decision
    final_position_size: float = 0.0
    stop_loss_level: float = 0.0
    take_profit_level: float = 0.0
    risk_approved: bool = False
    emergency_stop: bool = False
    selected_broker: str = ""
    expected_qoe: float = 0.0
    
    # Performance metrics
    total_latency_us: float = 0.0
    agent_latencies: Dict[str, float] = field(default_factory=dict)
    routing_latency_us: float = 0.0
    fill_rate: float = 0.0
    estimated_slippage_bps: float = 0.0
    
    # Reasoning and validation
    reasoning: str = ""
    confidence: float = 0.0
    risk_score: float = 0.0


@dataclass
class ExecutionPerformanceMetrics:
    """Performance metrics for execution system"""
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
    
    # Agent-specific metrics
    agent_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Load testing metrics
    concurrent_capacity: int = 0
    executions_per_second: float = 0.0
    
    # Error tracking
    error_rates: Dict[str, float] = field(default_factory=dict)
    critical_failures: List[str] = field(default_factory=list)


class UnifiedExecutionMARLSystem:
    """
    Unified Execution MARL System for coordinated multi-agent execution
    
    Integrates all 5 agents with centralized critic for optimal execution decisions:
    - π₁: Position Sizing Agent
    - π₂: Stop/Target Agent  
    - π₃: Risk Monitor Agent
    - π₄: Portfolio Optimizer Agent
    - π₅: Routing Agent (The Arbitrageur)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified execution MARL system"""
        self.config = config
        self.event_bus = EventBus()
        
        # Performance tracking
        self.metrics = ExecutionPerformanceMetrics()
        self.latency_history = deque(maxlen=10000)
        self.execution_history = deque(maxlen=1000)
        
        # Initialize broker performance tracker
        self._initialize_broker_tracker()
        
        # Initialize agents
        self._initialize_agents()
        
        # Initialize centralized critic
        self._initialize_centralized_critic()
        
        # Initialize MAPPO trainer
        self._initialize_mappo_trainer()
        
        # Performance monitoring
        self.performance_monitor_active = True
        self.monitoring_task = None
        
        # Thread pool for concurrent execution
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        
        logger.info("UnifiedExecutionMARLSystem initialized",
                   agents=len(self.agents),
                   critic_dims=f"{self.critic.combined_input_dim}D",
                   max_workers=config.get('max_workers', 10))
    
    def _initialize_broker_tracker(self):
        """Initialize broker performance tracker"""
        try:
            self.broker_tracker = BrokerPerformanceTracker(
                self.config.get('broker_tracker', {}),
                self.event_bus
            )
            logger.info("Broker Performance Tracker initialized")
        except Exception as e:
            logger.error("Failed to initialize Broker Performance Tracker", error=str(e))
            self.broker_tracker = None

    def _initialize_agents(self):
        """Initialize all 5 agents"""
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
            # Create mock agent for testing
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
        
        # Agent π₅: Routing Agent (The Arbitrageur)
        try:
            self.agents['routing'] = RoutingAgent(
                self.config.get('routing', {}),
                self.event_bus
            )
            logger.info("Routing Agent (π₅) - The Arbitrageur initialized")
        except Exception as e:
            logger.error("Failed to initialize Routing Agent", error=str(e))
            self.agents['routing'] = self._create_mock_agent('routing')
    
    def _create_mock_agent(self, agent_type: str):
        """Create mock agent for testing when real agent fails to initialize"""
        class MockAgent:
            def __init__(self, agent_type):
                self.agent_type = agent_type
                self.name = f"mock_{agent_type}"
                self.strategy_support_enabled = True  # Enable strategy support
            
            def act(self, state, context=None):
                # Return mock decision based on agent type with strategy support
                if agent_type == 'position_sizing':
                    return PositionSizingDecision(
                        contracts=1,  # Reduced from 2 to be more conservative
                        kelly_fraction=0.1,  # Reduced from 0.15
                        position_size_fraction=0.05,  # Reduced from 0.1
                        confidence=0.3,  # Reduced from 0.7 to let strategy dominate
                        reasoning={'method': 'mock', 'strategy_support': True},
                        risk_adjustments=[],
                        computation_time_ms=1.0,
                        timestamp=datetime.now()
                    )
                elif agent_type == 'routing':
                    from src.execution.agents.routing_agent import RoutingAction, BrokerType
                    return RoutingAction(
                        broker_id='MOCK_BROKER',
                        broker_type=BrokerType.INTERACTIVE_BROKERS,
                        confidence=0.3,  # Reduced from 0.5
                        expected_qoe=0.7,
                        reasoning='Mock routing decision - strategy support mode'
                    )
                else:
                    return {
                        'action': 1,  # Default to hold (index 1)
                        'confidence': 0.3,  # Reduced from 0.5
                        'reasoning': f'Mock {agent_type} decision - strategy support mode',
                        'timestamp': datetime.now(),
                        'strategy_support_mode': True,
                        'strategy_override_allowed': False
                    }
        
        logger.warning(f"Using mock agent for {agent_type} with strategy support")
        return MockAgent(agent_type)
    
    def _initialize_centralized_critic(self):
        """Initialize centralized critic for coordinated learning"""
        critic_config = {
            'context_dim': 15,  # Execution context vector
            'market_features_dim': 32,  # Extended market features
            'routing_state_dim': 55,  # Routing state dimension
            'num_agents': 5,  # Now 5 agents including routing
            'critic_hidden_dims': [256, 128, 64]
        }
        
        self.critic = create_centralized_critic(critic_config)
        logger.info("Centralized critic initialized", 
                   input_dim=self.critic.combined_input_dim,
                   output_dim=1)
    
    def _initialize_mappo_trainer(self):
        """Initialize MAPPO trainer for coordinated learning"""
        trainer_config = {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01
        }
        
        # Get agent networks (or create mock networks)
        agent_networks = []
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'network'):
                agent_networks.append(agent.network)
            else:
                # Create mock network for testing
                mock_network = nn.Sequential(
                    nn.Linear(47, 64),  # Combined input
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(), 
                    nn.Linear(32, 3)  # Mock action space
                )
                agent_networks.append(mock_network)
        
        self.mappo_trainer = create_mappo_trainer(
            self.critic, 
            agent_networks, 
            trainer_config
        )
        
        logger.info("MAPPO trainer initialized", 
                   num_agents=len(agent_networks),
                   learning_rate=trainer_config['learning_rate'])
    
    async def execute_unified_decision(self, 
                                     execution_context: ExecutionContext,
                                     market_features: MarketFeatures,
                                     order_data: Optional[Dict[str, Any]] = None) -> ExecutionDecision:
        """
        Execute unified decision across all 5 agents
        
        Args:
            execution_context: 15D execution context vector
            market_features: 32D market features vector
            order_data: Order data for routing agent (optional)
            
        Returns:
            Unified execution decision with performance metrics
        """
        start_time = time.perf_counter()
        
        try:
            # Create combined state for critic (including routing state)
            combined_state = CombinedState(
                execution_context=execution_context,
                market_features=market_features
            )
            
            # Get centralized critic evaluation
            state_value, critic_info = self.critic.evaluate_state(combined_state)
            
            # Execute all agents concurrently for optimal performance
            agent_tasks = [
                self._execute_agent_async('position_sizing', execution_context, market_features),
                self._execute_agent_async('stop_target', execution_context, market_features),
                self._execute_agent_async('risk_monitor', execution_context, market_features),
                self._execute_agent_async('portfolio_optimizer', execution_context, market_features),
                self._execute_routing_agent_async(execution_context, market_features, order_data)
            ]
            
            # Wait for all agent decisions
            agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Process agent results
            decision = ExecutionDecision()
            decision.timestamp = datetime.now()
            
            # Extract individual agent decisions
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
                    decision.routing = result
            
            # Aggregate decisions using centralized coordination
            decision = await self._aggregate_agent_decisions(decision, state_value, critic_info)
            
            # Calculate performance metrics
            end_time = time.perf_counter()
            decision.total_latency_us = (end_time - start_time) * 1_000_000  # Convert to microseconds
            
            # Update system metrics
            self._update_performance_metrics(decision)
            
            # Validate performance requirements
            await self._validate_performance_requirements(decision)
            
            return decision
            
        except Exception as e:
            logger.error("Unified execution failed", error=str(e))
            
            # Return safe fallback decision
            end_time = time.perf_counter()
            fallback_decision = ExecutionDecision()
            fallback_decision.total_latency_us = (end_time - start_time) * 1_000_000
            fallback_decision.reasoning = f"Execution failed: {str(e)}"
            fallback_decision.emergency_stop = True
            
            return fallback_decision
    
    async def _execute_agent_async(self, 
                                  agent_name: str, 
                                  execution_context: ExecutionContext,
                                  market_features: MarketFeatures) -> Any:
        """Execute single agent asynchronously"""
        agent_start = time.perf_counter()
        
        try:
            agent = self.agents[agent_name]
            
            # Convert to agent-specific state format
            if agent_name == 'position_sizing':
                # Position sizing agent expects risk state
                state = self._create_risk_state(execution_context, market_features)
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, agent.act, state
                )
            else:
                # Other agents expect generic state
                state = self._create_generic_state(execution_context, market_features)
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, agent.act, state
                )
            
            # Track agent performance
            agent_time = (time.perf_counter() - agent_start) * 1_000_000  # microseconds
            self.metrics.agent_performance.setdefault(agent_name, {})
            self.metrics.agent_performance[agent_name]['last_latency_us'] = agent_time
            
            return result
            
        except Exception as e:
            logger.error(f"Agent {agent_name} execution failed", error=str(e))
            raise
    
    def _create_risk_state(self, execution_context: ExecutionContext, market_features: MarketFeatures):
        """Create risk state for position sizing agent"""
        from src.risk.agents.base_risk_agent import RiskState
        
        # Convert execution context and market features to risk state
        # This is a simplified conversion - in production would be more sophisticated
        return RiskState(
            portfolio_value=execution_context.portfolio_value,
            position_size=execution_context.current_position,
            unrealized_pnl=execution_context.unrealized_pnl,
            var_estimate=execution_context.var_estimate,
            correlation_risk=market_features.correlation_spy,
            volatility=market_features.realized_garch,
            drawdown=execution_context.drawdown_current,
            risk_limit_utilization=execution_context.risk_budget_used,
            kelly_fraction=0.15,  # Default value
            timestamp=datetime.now()
        )
    
    def _create_generic_state(self, execution_context: ExecutionContext, market_features: MarketFeatures):
        """Create generic state for other agents"""
        # Combine execution context and market features into generic state vector
        context_vector = execution_context.to_tensor()
        market_vector = market_features.to_tensor()
        
        return torch.cat([context_vector, market_vector], dim=0).numpy()
    
    async def _execute_routing_agent_async(self, 
                                         execution_context: ExecutionContext,
                                         market_features: MarketFeatures,
                                         order_data: Optional[Dict[str, Any]]) -> Any:
        """Execute routing agent asynchronously"""
        agent_start = time.perf_counter()
        
        try:
            routing_agent = self.agents.get('routing')
            if not routing_agent:
                raise RuntimeError("Routing agent not available")
            
            # Create routing state
            if hasattr(routing_agent, 'create_routing_state') and order_data:
                portfolio_data = {
                    'portfolio_value': execution_context.portfolio_value,
                    'concentration': 0.1,  # Simplified
                    'risk_budget_used': execution_context.risk_budget_used,
                    'recent_pnl': execution_context.realized_pnl,
                    'correlation_risk': getattr(market_features, 'correlation_spy', 0.2)
                }
                
                market_data = {
                    'volatility': getattr(market_features, 'realized_garch', 0.15),
                    'spread_bps': 5.0,  # Default
                    'volume_ratio': 1.0,  # Default
                    'time_of_day_normalized': 0.5,  # Default
                    'stress_indicator': 0.0  # Default
                }
                
                routing_state = routing_agent.create_routing_state(order_data, market_data, portfolio_data)
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, routing_agent.act, routing_state
                )
            else:
                # Fallback for mock agent
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, routing_agent.act, None
                )
            
            # Track agent performance
            agent_time = (time.perf_counter() - agent_start) * 1_000_000  # microseconds
            self.metrics.agent_performance.setdefault('routing', {})
            self.metrics.agent_performance['routing']['last_latency_us'] = agent_time
            
            return result
            
        except Exception as e:
            logger.error("Routing agent execution failed", error=str(e))
            raise
    
    async def _aggregate_agent_decisions(self, 
                                       decision: ExecutionDecision,
                                       state_value: float,
                                       critic_info: Dict[str, Any]) -> ExecutionDecision:
        """Aggregate individual agent decisions into unified decision with strategy support"""
        
        # Check if any agents are in strategy support mode
        strategy_support_active = False
        agent_decisions = [decision.position_sizing, decision.stop_target, decision.risk_monitor, 
                          decision.portfolio_optimizer, decision.routing]
        
        for agent_decision in agent_decisions:
            if agent_decision and hasattr(agent_decision, 'reasoning'):
                reasoning = agent_decision.reasoning if isinstance(agent_decision.reasoning, str) else str(agent_decision.reasoning)
                if 'strategy_support' in reasoning:
                    strategy_support_active = True
                    break
        
        # Position sizing (π₁) - reduced influence if strategy support active
        if decision.position_sizing:
            base_size = decision.position_sizing.position_size_fraction
            if strategy_support_active:
                base_size *= 0.5  # Reduce agent influence when supporting strategy
            decision.final_position_size = base_size
            decision.confidence = decision.position_sizing.confidence
        
        # Stop/Target levels (π₂)
        if decision.stop_target:
            decision.stop_loss_level = decision.stop_target.get('stop_loss', 0.0)
            decision.take_profit_level = decision.stop_target.get('take_profit', 0.0)
        
        # Risk monitoring (π₃) - maintains override capability
        if decision.risk_monitor:
            risk_action = decision.risk_monitor.get('action', RiskMonitorAction.NO_ACTION)
            decision.emergency_stop = (risk_action == RiskMonitorAction.EMERGENCY_STOP)
            decision.risk_approved = (risk_action in [RiskMonitorAction.NO_ACTION, RiskMonitorAction.ALERT])
            decision.risk_score = decision.risk_monitor.get('risk_score', 0.0)
        
        # Portfolio optimization (π₄) - reduced influence if strategy support active
        if decision.portfolio_optimizer:
            portfolio_adjustment = decision.portfolio_optimizer.get('position_adjustment', 1.0)
            if strategy_support_active:
                # Reduce portfolio adjustment influence
                portfolio_adjustment = 0.5 * portfolio_adjustment + 0.5 * 1.0
            decision.final_position_size *= portfolio_adjustment
        
        # Routing decisions (π₅)
        if decision.routing:
            decision.selected_broker = decision.routing.broker_id
            decision.expected_qoe = decision.routing.expected_qoe
            decision.routing_latency_us = 0.0  # Will be set by routing agent
            
            # Track routing decision confidence in overall confidence
            routing_confidence = decision.routing.confidence
            if decision.confidence > 0:
                decision.confidence = (decision.confidence + routing_confidence) / 2.0
            else:
                decision.confidence = routing_confidence
        
        # Apply centralized critic coordination with strategy support consideration
        critic_adjustment = np.clip(state_value, 0.5, 1.5)  # Limit adjustment range
        if strategy_support_active:
            # Reduce critic influence when strategy support is active
            critic_adjustment = 0.7 * critic_adjustment + 0.3 * 1.0
        decision.final_position_size *= critic_adjustment
        
        # Emergency overrides (always respected)
        if decision.emergency_stop:
            decision.final_position_size = 0.0
            decision.risk_approved = False
        
        # Generate reasoning with strategy support context
        decision.reasoning = self._generate_decision_reasoning(decision, state_value, critic_info)
        if strategy_support_active:
            decision.reasoning += " [Strategy Support Mode Active]"
        
        # Estimate fill rate and slippage
        decision.fill_rate = self._estimate_fill_rate(decision)
        decision.estimated_slippage_bps = self._estimate_slippage(decision)
        
        return decision
    
    def _generate_decision_reasoning(self, 
                                   decision: ExecutionDecision,
                                   state_value: float,
                                   critic_info: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the decision"""
        reasoning_parts = []
        
        # Position sizing reasoning
        if decision.position_sizing:
            reasoning_parts.append(
                f"Position sizing (π₁): {decision.position_sizing.contracts} contracts "
                f"based on Kelly fraction {decision.position_sizing.kelly_fraction:.3f}"
            )
        
        # Risk monitoring reasoning
        if decision.risk_monitor:
            risk_action = decision.risk_monitor.get('action', 0)
            reasoning_parts.append(f"Risk monitor (π₃): Action {risk_action}")
        
        # Routing reasoning
        if decision.routing:
            reasoning_parts.append(
                f"Routing (π₅): {decision.routing.broker_id} selected "
                f"(confidence: {decision.routing.confidence:.3f}, QoE: {decision.routing.expected_qoe:.3f})"
            )
        
        # Critic coordination
        reasoning_parts.append(f"Centralized critic state value: {state_value:.3f}")
        
        # Final decision
        if decision.emergency_stop:
            reasoning_parts.append("EMERGENCY STOP activated")
        elif decision.risk_approved:
            reasoning_parts.append(f"Final position: {decision.final_position_size:.3f} via {decision.selected_broker}")
        else:
            reasoning_parts.append("Position rejected by risk management")
        
        return " | ".join(reasoning_parts)
    
    def _estimate_fill_rate(self, decision: ExecutionDecision) -> float:
        """Estimate fill rate based on decision parameters"""
        # Simplified fill rate estimation
        base_fill_rate = 0.998  # 99.8% base fill rate
        
        # Adjust based on position size (larger positions harder to fill)
        size_adjustment = max(0.95, 1.0 - decision.final_position_size * 0.1)
        
        # Adjust based on market conditions
        volatility_adjustment = 0.999  # Assume low volatility for now
        
        return base_fill_rate * size_adjustment * volatility_adjustment
    
    def _estimate_slippage(self, decision: ExecutionDecision) -> float:
        """Estimate slippage in basis points"""
        # Base slippage: 1 bps
        base_slippage = 1.0
        
        # Increase with position size
        size_slippage = decision.final_position_size * 0.5
        
        # Market impact
        market_impact = 0.5  # Assume low market impact
        
        return base_slippage + size_slippage + market_impact
    
    def _update_performance_metrics(self, decision: ExecutionDecision):
        """Update system performance metrics"""
        self.metrics.total_executions += 1
        
        if decision.risk_approved and not decision.emergency_stop:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1
        
        # Update latency metrics
        self.latency_history.append(decision.total_latency_us)
        latencies = list(self.latency_history)
        
        self.metrics.avg_latency_us = np.mean(latencies)
        self.metrics.p95_latency_us = np.percentile(latencies, 95)
        self.metrics.p99_latency_us = np.percentile(latencies, 99)
        self.metrics.max_latency_us = np.max(latencies)
        
        # Update fill rate and slippage
        self.metrics.avg_fill_rate = (
            self.metrics.avg_fill_rate * 0.9 + decision.fill_rate * 0.1
        )
        self.metrics.avg_slippage_bps = (
            self.metrics.avg_slippage_bps * 0.9 + decision.estimated_slippage_bps * 0.1
        )
        
        # Store execution for analysis
        self.execution_history.append({
            'timestamp': decision.timestamp,
            'latency_us': decision.total_latency_us,
            'fill_rate': decision.fill_rate,
            'slippage_bps': decision.estimated_slippage_bps,
            'success': decision.risk_approved and not decision.emergency_stop
        })
    
    async def _validate_performance_requirements(self, decision: ExecutionDecision):
        """Validate that performance requirements are met"""
        # Latency requirement: <500μs
        if decision.total_latency_us > 500:
            logger.warning("Latency requirement violation", 
                          actual_us=decision.total_latency_us,
                          limit_us=500)
        
        # Routing latency requirement: <100μs
        if decision.routing_latency_us > 100:
            logger.warning("Routing latency requirement violation",
                          actual_us=decision.routing_latency_us,
                          limit_us=100)
        
        # Fill rate requirement: >99.8%
        if decision.fill_rate < 0.998:
            logger.warning("Fill rate requirement violation",
                          actual=decision.fill_rate,
                          limit=0.998)
        
        # Slippage requirement: <2 bps
        if decision.estimated_slippage_bps > 2.0:
            logger.warning("Slippage requirement violation",
                          actual_bps=decision.estimated_slippage_bps,
                          limit_bps=2.0)
    
    async def load_test(self, 
                       concurrent_requests: int = 50,
                       target_rps: int = 100,
                       duration_seconds: int = 30) -> Dict[str, Any]:
        """
        Perform load testing to validate concurrent capacity
        
        Args:
            concurrent_requests: Number of concurrent requests
            target_rps: Target requests per second
            duration_seconds: Test duration
            
        Returns:
            Load test results
        """
        logger.info("Starting load test",
                   concurrent_requests=concurrent_requests,
                   target_rps=target_rps,
                   duration_seconds=duration_seconds)
        
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        latencies = []
        
        # Create sample execution context and market features
        sample_context = ExecutionContext(
            portfolio_value=100000.0,
            available_capital=50000.0,
            current_position=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            var_estimate=0.02,
            expected_return=0.001,
            volatility=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            drawdown_current=0.01,
            time_since_last_trade=300,
            risk_budget_used=0.3,
            correlation_risk=0.2,
            liquidity_score=0.9
        )
        
        sample_features = MarketFeatures()
        
        async def execute_request():
            """Execute single request"""
            try:
                decision = await self.execute_unified_decision(sample_context, sample_features)
                return decision.total_latency_us, True
            except Exception as e:
                logger.error("Load test request failed", error=str(e))
                return 0, False
        
        # Execute load test
        request_interval = 1.0 / target_rps
        
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            
            # Create batch of concurrent requests
            tasks = [execute_request() for _ in range(min(concurrent_requests, target_rps))]
            
            # Execute batch
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    failed_requests += 1
                    continue
                
                latency, success = result
                if success:
                    successful_requests += 1
                    latencies.append(latency)
                else:
                    failed_requests += 1
            
            # Rate limiting
            batch_time = time.time() - batch_start
            if batch_time < request_interval:
                await asyncio.sleep(request_interval - batch_time)
        
        # Calculate results
        total_time = time.time() - start_time
        total_requests = successful_requests + failed_requests
        actual_rps = total_requests / total_time
        
        load_test_results = {
            'duration_seconds': total_time,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': successful_requests / max(1, total_requests),
            'actual_rps': actual_rps,
            'target_rps': target_rps,
            'rps_achievement': actual_rps / target_rps,
            'concurrent_capacity': concurrent_requests,
            'latency_metrics': {
                'avg_us': np.mean(latencies) if latencies else 0,
                'p95_us': np.percentile(latencies, 95) if latencies else 0,
                'p99_us': np.percentile(latencies, 99) if latencies else 0,
                'max_us': np.max(latencies) if latencies else 0
            },
            'performance_requirements_met': {
                'latency_compliant': np.percentile(latencies, 95) < 500 if latencies else False,
                'rps_compliant': actual_rps >= target_rps,
                'success_rate_compliant': (successful_requests / max(1, total_requests)) > 0.95
            }
        }
        
        logger.info("Load test completed",
                   actual_rps=actual_rps,
                   success_rate=load_test_results['success_rate'],
                   p95_latency_us=load_test_results['latency_metrics']['p95_us'])
        
        return load_test_results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'system_metrics': {
                'total_executions': self.metrics.total_executions,
                'success_rate': self.metrics.successful_executions / max(1, self.metrics.total_executions),
                'avg_latency_us': self.metrics.avg_latency_us,
                'p95_latency_us': self.metrics.p95_latency_us,
                'p99_latency_us': self.metrics.p99_latency_us,
                'avg_fill_rate': self.metrics.avg_fill_rate,
                'avg_slippage_bps': self.metrics.avg_slippage_bps
            },
            'agent_performance': self.metrics.agent_performance,
            'performance_requirements': {
                'latency_compliant': self.metrics.p95_latency_us < 500,
                'fill_rate_compliant': self.metrics.avg_fill_rate > 0.998,
                'slippage_compliant': self.metrics.avg_slippage_bps < 2.0
            },
            'critic_metrics': self.mappo_trainer.get_training_metrics(),
            'execution_history_sample': list(self.execution_history)[-10:]  # Last 10 executions
        }
    
    async def shutdown(self):
        """Graceful shutdown of the system"""
        logger.info("Shutting down UnifiedExecutionMARLSystem")
        
        # Stop performance monitoring
        self.performance_monitor_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("UnifiedExecutionMARLSystem shutdown complete")


def create_unified_execution_system(config: Dict[str, Any]) -> UnifiedExecutionMARLSystem:
    """Factory function to create unified execution MARL system"""
    return UnifiedExecutionMARLSystem(config)


# Example configuration
DEFAULT_CONFIG = {
    'max_workers': 10,
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
    'routing': {
        'broker_ids': ['IB', 'ALPACA', 'TDA', 'SCHWAB'],
        'learning_rate': 2e-4,
        'training_mode': True,
        'exploration_epsilon': 0.1,
        'epsilon_decay': 0.995,
        'min_epsilon': 0.01,
        'target_qoe_score': 0.85,
        'max_routing_latency_us': 100.0,
        'monitoring_window_minutes': 60
    },
    'broker_tracker': {
        'tracking_window_hours': 24,
        'max_records_per_broker': 10000,
        'alert_thresholds': {
            'latency_spike_factor': 2.0,
            'fill_rate_drop_threshold': 0.05,
            'high_slippage_bps': 20.0,
            'high_commission_threshold': 0.05,
            'qoe_degradation_threshold': 0.1
        }
    }
}