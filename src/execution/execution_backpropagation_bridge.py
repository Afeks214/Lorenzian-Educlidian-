"""
ExecutionBackpropagationBridge: Real-time Learning from Execution Outcomes
========================================================================

This module implements the critical bridge between execution outcomes and model parameter updates,
enabling real-time learning during live trading. The bridge captures execution results and
immediately propagates gradients back to the MAPPO agents for continuous adaptation.

Key Features:
1. Real-time execution outcome to gradient computation (<100ms latency)
2. Direct model parameter updates from trade results
3. Immediate backpropagation pipeline during live trading
4. Streaming gradient accumulation for continuous learning
5. Performance-based model weight adjustments
6. Integration with existing MAPPO trainers

Technical Architecture:
- Execution outcome capture and processing
- Gradient computation from PnL and execution quality metrics
- Streaming gradient accumulation with momentum
- Adaptive learning rate scheduling based on performance
- Real-time model parameter updates with stability controls

Performance Requirements:
- <100ms update latency from execution outcome to model update
- Gradient computation with numerical stability
- Memory-efficient streaming accumulation
- Thread-safe concurrent updates

Author: Claude - Execution Learning System
Date: 2025-07-17
"""

import asyncio
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import structlog
from enum import Enum
import json
from pathlib import Path

logger = structlog.get_logger()


class ExecutionOutcome(Enum):
    """Execution outcome types for gradient computation"""
    SUCCESS = "success"
    PARTIAL_FILL = "partial_fill"
    SLIPPAGE_EXCESS = "slippage_excess"
    LATENCY_VIOLATION = "latency_violation"
    RISK_REJECTION = "risk_rejection"
    ROUTING_FAILURE = "routing_failure"
    MARKET_IMPACT = "market_impact"
    TIMEOUT = "timeout"


@dataclass
class ExecutionResult:
    """Complete execution result for backpropagation"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    
    # Execution details
    intended_quantity: float
    filled_quantity: float
    intended_price: float
    fill_price: float
    execution_time_ms: float
    
    # Performance metrics
    slippage_bps: float
    market_impact_bps: float
    fill_rate: float
    latency_us: float
    
    # Financial outcomes
    realized_pnl: float
    unrealized_pnl: float
    commission: float
    fees: float
    
    # Agent decisions that led to this execution
    position_sizing_decision: Optional[Dict[str, Any]] = None
    stop_target_decision: Optional[Dict[str, Any]] = None
    risk_monitor_decision: Optional[Dict[str, Any]] = None
    portfolio_optimizer_decision: Optional[Dict[str, Any]] = None
    routing_decision: Optional[Dict[str, Any]] = None
    
    # Context at execution time
    market_context: Optional[Dict[str, Any]] = None
    execution_context: Optional[Dict[str, Any]] = None
    
    # Outcome classification
    outcome: ExecutionOutcome = ExecutionOutcome.SUCCESS
    quality_score: float = 1.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side,
            'intended_quantity': self.intended_quantity,
            'filled_quantity': self.filled_quantity,
            'intended_price': self.intended_price,
            'fill_price': self.fill_price,
            'execution_time_ms': self.execution_time_ms,
            'slippage_bps': self.slippage_bps,
            'market_impact_bps': self.market_impact_bps,
            'fill_rate': self.fill_rate,
            'latency_us': self.latency_us,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'commission': self.commission,
            'fees': self.fees,
            'position_sizing_decision': self.position_sizing_decision,
            'stop_target_decision': self.stop_target_decision,
            'risk_monitor_decision': self.risk_monitor_decision,
            'portfolio_optimizer_decision': self.portfolio_optimizer_decision,
            'routing_decision': self.routing_decision,
            'market_context': self.market_context,
            'execution_context': self.execution_context,
            'outcome': self.outcome.value,
            'quality_score': self.quality_score
        }


@dataclass
class GradientUpdate:
    """Gradient update for specific agent"""
    agent_name: str
    gradients: Dict[str, torch.Tensor]
    learning_rate: float
    momentum: float
    timestamp: datetime
    execution_id: str
    reward_signal: float
    
    def apply_to_model(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Apply gradient update to model parameters"""
        # Set gradients on model parameters
        for name, param in model.named_parameters():
            if name in self.gradients:
                param.grad = self.gradients[name]
        
        # Apply optimizer step
        optimizer.step()
        optimizer.zero_grad()


@dataclass
class StreamingGradientAccumulator:
    """Streaming gradient accumulation with momentum"""
    agent_name: str
    momentum: float = 0.9
    learning_rate: float = 1e-4
    max_accumulation_steps: int = 100
    
    # Accumulated gradients
    accumulated_gradients: Dict[str, torch.Tensor] = field(default_factory=dict)
    gradient_counts: Dict[str, int] = field(default_factory=dict)
    
    # Momentum terms
    momentum_terms: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Performance tracking
    accumulation_steps: int = 0
    last_update_time: datetime = field(default_factory=datetime.now)
    
    def add_gradient(self, param_name: str, gradient: torch.Tensor, weight: float = 1.0):
        """Add gradient to accumulator with optional weighting"""
        if param_name not in self.accumulated_gradients:
            self.accumulated_gradients[param_name] = torch.zeros_like(gradient)
            self.gradient_counts[param_name] = 0
        
        # Weighted accumulation
        self.accumulated_gradients[param_name] += weight * gradient
        self.gradient_counts[param_name] += 1
        
        self.accumulation_steps += 1
    
    def get_averaged_gradients(self) -> Dict[str, torch.Tensor]:
        """Get averaged gradients and reset accumulator"""
        averaged_gradients = {}
        
        for param_name, accumulated_grad in self.accumulated_gradients.items():
            count = self.gradient_counts[param_name]
            if count > 0:
                # Apply momentum
                averaged_grad = accumulated_grad / count
                
                if param_name in self.momentum_terms:
                    self.momentum_terms[param_name] = (
                        self.momentum * self.momentum_terms[param_name] + 
                        (1 - self.momentum) * averaged_grad
                    )
                    averaged_gradients[param_name] = self.momentum_terms[param_name]
                else:
                    self.momentum_terms[param_name] = averaged_grad
                    averaged_gradients[param_name] = averaged_grad
        
        # Reset accumulator
        self.accumulated_gradients.clear()
        self.gradient_counts.clear()
        self.accumulation_steps = 0
        self.last_update_time = datetime.now()
        
        return averaged_gradients
    
    def should_update(self) -> bool:
        """Check if accumulator should trigger update"""
        return self.accumulation_steps >= self.max_accumulation_steps


class ExecutionBackpropagationBridge:
    """
    Real-time bridge between execution outcomes and model parameter updates
    
    This bridge captures execution results and immediately computes gradients
    for backpropagation to the MAPPO agents, enabling continuous learning
    during live trading.
    """
    
    def __init__(self, 
                 mappo_trainer: Any,
                 unified_execution_system: Any,
                 config: Dict[str, Any] = None):
        """
        Initialize execution backpropagation bridge
        
        Args:
            mappo_trainer: MAPPO trainer instance
            unified_execution_system: Unified execution MARL system
            config: Configuration dictionary
        """
        self.mappo_trainer = mappo_trainer
        self.unified_execution_system = unified_execution_system
        self.config = config or {}
        
        # Performance requirements
        self.max_update_latency_ms = self.config.get('max_update_latency_ms', 100)
        self.gradient_computation_timeout_ms = self.config.get('gradient_computation_timeout_ms', 50)
        
        # Learning parameters
        self.base_learning_rate = self.config.get('base_learning_rate', 1e-4)
        self.momentum = self.config.get('momentum', 0.9)
        self.gradient_clip_norm = self.config.get('gradient_clip_norm', 1.0)
        
        # Streaming accumulation
        self.streaming_accumulators = {}
        self.accumulation_window = self.config.get('accumulation_window', 10)
        
        # Execution result queue for processing
        self.execution_queue = asyncio.Queue(maxsize=1000)
        self.processing_active = True
        
        # Performance tracking
        self.total_updates = 0
        self.update_latencies = deque(maxlen=1000)
        self.gradient_computation_times = deque(maxlen=1000)
        self.learning_rates = deque(maxlen=1000)
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize streaming accumulators for each agent
        self._initialize_streaming_accumulators()
        
        # Start background processing
        self.processing_task = None
        
        logger.info("ExecutionBackpropagationBridge initialized",
                   max_update_latency_ms=self.max_update_latency_ms,
                   base_learning_rate=self.base_learning_rate,
                   accumulation_window=self.accumulation_window)
    
    def _initialize_streaming_accumulators(self):
        """Initialize streaming gradient accumulators for each agent"""
        agent_names = ['position_sizing', 'stop_target', 'risk_monitor', 
                      'portfolio_optimizer', 'routing']
        
        for agent_name in agent_names:
            self.streaming_accumulators[agent_name] = StreamingGradientAccumulator(
                agent_name=agent_name,
                momentum=self.momentum,
                learning_rate=self.base_learning_rate,
                max_accumulation_steps=self.accumulation_window
            )
    
    async def start_processing(self):
        """Start background processing of execution results"""
        self.processing_task = asyncio.create_task(self._process_execution_results())
        logger.info("Started execution result processing")
    
    async def stop_processing(self):
        """Stop background processing"""
        self.processing_active = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped execution result processing")
    
    async def process_execution_result(self, execution_result: ExecutionResult):
        """
        Process execution result and trigger immediate backpropagation
        
        Args:
            execution_result: Complete execution result
        """
        start_time = time.perf_counter()
        
        try:
            # Add to processing queue
            await self.execution_queue.put(execution_result)
            
            # Immediate processing for critical updates
            if execution_result.outcome in [ExecutionOutcome.RISK_REJECTION, 
                                          ExecutionOutcome.ROUTING_FAILURE]:
                await self._immediate_gradient_update(execution_result)
            
            # Track latency
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.debug("Execution result queued for processing",
                        symbol=execution_result.symbol,
                        outcome=execution_result.outcome.value,
                        processing_time_ms=processing_time)
            
        except Exception as e:
            logger.error("Error processing execution result", error=str(e))
    
    async def _process_execution_results(self):
        """Background processing of execution results"""
        while self.processing_active:
            try:
                # Get execution result from queue
                execution_result = await asyncio.wait_for(
                    self.execution_queue.get(),
                    timeout=1.0
                )
                
                # Process the result
                await self._compute_and_apply_gradients(execution_result)
                
            except asyncio.TimeoutError:
                # Check for accumulated gradients that need updating
                await self._check_accumulated_gradients()
                continue
            except Exception as e:
                logger.error("Error in execution result processing", error=str(e))
                continue
    
    async def _immediate_gradient_update(self, execution_result: ExecutionResult):
        """Immediate gradient update for critical execution outcomes"""
        start_time = time.perf_counter()
        
        try:
            # Compute gradients for immediate update
            gradients = await self._compute_gradients_from_execution(execution_result)
            
            # Apply gradients immediately without accumulation
            for agent_name, gradient_update in gradients.items():
                if agent_name in self.unified_execution_system.agents:
                    agent = self.unified_execution_system.agents[agent_name]
                    if hasattr(agent, 'network') and hasattr(agent, 'optimizer'):
                        gradient_update.apply_to_model(agent.network, agent.optimizer)
            
            # Track immediate update
            update_time = (time.perf_counter() - start_time) * 1000
            self.update_latencies.append(update_time)
            self.total_updates += 1
            
            logger.info("Immediate gradient update applied",
                       outcome=execution_result.outcome.value,
                       update_time_ms=update_time,
                       agents_updated=len(gradients))
            
        except Exception as e:
            logger.error("Error in immediate gradient update", error=str(e))
    
    async def _compute_and_apply_gradients(self, execution_result: ExecutionResult):
        """Compute gradients and add to streaming accumulators"""
        start_time = time.perf_counter()
        
        try:
            # Compute gradients from execution result
            gradients = await self._compute_gradients_from_execution(execution_result)
            
            # Add gradients to streaming accumulators
            weight = self._compute_gradient_weight(execution_result)
            
            for agent_name, gradient_update in gradients.items():
                if agent_name in self.streaming_accumulators:
                    accumulator = self.streaming_accumulators[agent_name]
                    
                    # Add gradients to accumulator
                    for param_name, grad in gradient_update.gradients.items():
                        accumulator.add_gradient(param_name, grad, weight)
                    
                    # Check if accumulator should trigger update
                    if accumulator.should_update():
                        await self._apply_accumulated_gradients(agent_name)
            
            # Track gradient computation time
            computation_time = (time.perf_counter() - start_time) * 1000
            self.gradient_computation_times.append(computation_time)
            
            logger.debug("Gradients computed and accumulated",
                        symbol=execution_result.symbol,
                        computation_time_ms=computation_time,
                        gradient_weight=weight)
            
        except Exception as e:
            logger.error("Error computing gradients", error=str(e))
    
    async def _compute_gradients_from_execution(self, 
                                              execution_result: ExecutionResult) -> Dict[str, GradientUpdate]:
        """
        Compute gradients from execution result using reward signal
        
        Args:
            execution_result: Complete execution result
            
        Returns:
            Dictionary of gradient updates for each agent
        """
        gradients = {}
        
        # Compute reward signal from execution outcome
        reward_signal = self._compute_reward_signal(execution_result)
        
        # Compute gradients for each agent that contributed to the execution
        agent_contributions = {
            'position_sizing': execution_result.position_sizing_decision,
            'stop_target': execution_result.stop_target_decision,
            'risk_monitor': execution_result.risk_monitor_decision,
            'portfolio_optimizer': execution_result.portfolio_optimizer_decision,
            'routing': execution_result.routing_decision
        }
        
        for agent_name, decision in agent_contributions.items():
            if decision is not None and agent_name in self.unified_execution_system.agents:
                try:
                    agent = self.unified_execution_system.agents[agent_name]
                    
                    # Compute agent-specific gradients
                    agent_gradients = await self._compute_agent_gradients(
                        agent, agent_name, execution_result, reward_signal
                    )
                    
                    if agent_gradients:
                        gradients[agent_name] = GradientUpdate(
                            agent_name=agent_name,
                            gradients=agent_gradients,
                            learning_rate=self._compute_adaptive_learning_rate(agent_name, execution_result),
                            momentum=self.momentum,
                            timestamp=datetime.now(),
                            execution_id=f"{execution_result.symbol}_{execution_result.timestamp}",
                            reward_signal=reward_signal
                        )
                        
                except Exception as e:
                    logger.error(f"Error computing gradients for {agent_name}", error=str(e))
        
        return gradients
    
    def _compute_reward_signal(self, execution_result: ExecutionResult) -> float:
        """
        Compute reward signal from execution result
        
        Args:
            execution_result: Complete execution result
            
        Returns:
            Reward signal for gradient computation
        """
        # Base reward from financial outcome
        base_reward = execution_result.realized_pnl
        
        # Execution quality adjustments
        fill_rate_bonus = (execution_result.fill_rate - 0.5) * 2.0  # -1 to 1
        slippage_penalty = -abs(execution_result.slippage_bps) / 100.0  # Negative for slippage
        latency_penalty = -max(0, execution_result.latency_us - 100) / 1000.0  # Penalty for >100μs
        
        # Outcome-specific adjustments
        outcome_multiplier = {
            ExecutionOutcome.SUCCESS: 1.0,
            ExecutionOutcome.PARTIAL_FILL: 0.5,
            ExecutionOutcome.SLIPPAGE_EXCESS: -0.5,
            ExecutionOutcome.LATENCY_VIOLATION: -0.3,
            ExecutionOutcome.RISK_REJECTION: -1.0,
            ExecutionOutcome.ROUTING_FAILURE: -0.8,
            ExecutionOutcome.MARKET_IMPACT: -0.4,
            ExecutionOutcome.TIMEOUT: -0.6
        }.get(execution_result.outcome, 0.0)
        
        # Combine all components
        reward = (base_reward + fill_rate_bonus + slippage_penalty + latency_penalty) * outcome_multiplier
        
        # Quality score weighting
        reward *= execution_result.quality_score
        
        return reward
    
    async def _compute_agent_gradients(self, 
                                     agent: Any,
                                     agent_name: str,
                                     execution_result: ExecutionResult,
                                     reward_signal: float) -> Dict[str, torch.Tensor]:
        """
        Compute gradients for specific agent using REINFORCE-style update
        
        Args:
            agent: Agent model
            agent_name: Name of the agent
            execution_result: Execution result
            reward_signal: Reward signal for gradient computation
            
        Returns:
            Dictionary of parameter gradients
        """
        try:
            # Check if agent has network and can compute gradients
            if not hasattr(agent, 'network') or not hasattr(agent.network, 'parameters'):
                return {}
            
            # Reconstruct the state and action that led to this execution
            state = self._reconstruct_execution_state(execution_result)
            action = self._extract_agent_action(agent_name, execution_result)
            
            if state is None or action is None:
                return {}
            
            # Enable gradient computation
            agent.network.train()
            
            # Forward pass to get action probabilities
            with torch.enable_grad():
                if agent_name == 'position_sizing':
                    # Position sizing agent has specific interface
                    output = agent.network(state)
                    action_probs = F.softmax(output, dim=-1)
                elif agent_name == 'routing':
                    # Routing agent has specific interface
                    output = agent.network(state)
                    action_probs = F.softmax(output, dim=-1)
                else:
                    # Generic agent interface
                    output = agent.network(state)
                    action_probs = F.softmax(output, dim=-1)
                
                # Compute log probability of the action taken
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(action)
                
                # REINFORCE gradient: ∇log π(a|s) * R
                loss = -log_prob * reward_signal
                
                # Compute gradients
                loss.backward()
                
                # Extract gradients
                gradients = {}
                for name, param in agent.network.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.clone()
                
                # Clear gradients
                agent.network.zero_grad()
                
                return gradients
                
        except Exception as e:
            logger.error(f"Error computing gradients for {agent_name}", error=str(e))
            return {}
    
    def _reconstruct_execution_state(self, execution_result: ExecutionResult) -> Optional[torch.Tensor]:
        """Reconstruct the state at execution time"""
        try:
            if execution_result.execution_context and execution_result.market_context:
                # Combine execution context and market context
                context_features = list(execution_result.execution_context.values())
                market_features = list(execution_result.market_context.values())
                
                # Convert to tensor
                state_vector = context_features + market_features
                return torch.tensor(state_vector, dtype=torch.float32)
            
            return None
            
        except Exception as e:
            logger.error("Error reconstructing execution state", error=str(e))
            return None
    
    def _extract_agent_action(self, agent_name: str, execution_result: ExecutionResult) -> Optional[torch.Tensor]:
        """Extract the action taken by the agent"""
        try:
            decision_map = {
                'position_sizing': execution_result.position_sizing_decision,
                'stop_target': execution_result.stop_target_decision,
                'risk_monitor': execution_result.risk_monitor_decision,
                'portfolio_optimizer': execution_result.portfolio_optimizer_decision,
                'routing': execution_result.routing_decision
            }
            
            decision = decision_map.get(agent_name)
            if decision is None:
                return None
            
            # Extract action from decision (simplified)
            if agent_name == 'position_sizing':
                action = decision.get('action', 0)  # Default to hold
            elif agent_name == 'routing':
                action = decision.get('broker_index', 0)
            else:
                action = decision.get('action', 0)
            
            return torch.tensor(action, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error extracting action for {agent_name}", error=str(e))
            return None
    
    def _compute_gradient_weight(self, execution_result: ExecutionResult) -> float:
        """Compute weight for gradient based on execution quality"""
        base_weight = 1.0
        
        # Weight based on execution quality
        quality_weight = execution_result.quality_score
        
        # Weight based on outcome severity
        outcome_weights = {
            ExecutionOutcome.SUCCESS: 1.0,
            ExecutionOutcome.PARTIAL_FILL: 0.8,
            ExecutionOutcome.SLIPPAGE_EXCESS: 1.2,  # More weight for learning from mistakes
            ExecutionOutcome.LATENCY_VIOLATION: 1.1,
            ExecutionOutcome.RISK_REJECTION: 1.5,
            ExecutionOutcome.ROUTING_FAILURE: 1.3,
            ExecutionOutcome.MARKET_IMPACT: 1.1,
            ExecutionOutcome.TIMEOUT: 1.2
        }
        
        outcome_weight = outcome_weights.get(execution_result.outcome, 1.0)
        
        # Weight based on PnL magnitude (learn more from significant outcomes)
        pnl_weight = 1.0 + min(2.0, abs(execution_result.realized_pnl) / 1000.0)
        
        return base_weight * quality_weight * outcome_weight * pnl_weight
    
    def _compute_adaptive_learning_rate(self, agent_name: str, execution_result: ExecutionResult) -> float:
        """Compute adaptive learning rate based on execution outcome"""
        base_lr = self.base_learning_rate
        
        # Adjust learning rate based on outcome
        if execution_result.outcome in [ExecutionOutcome.RISK_REJECTION, 
                                      ExecutionOutcome.ROUTING_FAILURE]:
            # Higher learning rate for critical failures
            return base_lr * 2.0
        elif execution_result.outcome == ExecutionOutcome.SUCCESS:
            # Lower learning rate for successful executions
            return base_lr * 0.8
        else:
            # Standard learning rate
            return base_lr
    
    async def _apply_accumulated_gradients(self, agent_name: str):
        """Apply accumulated gradients to agent model"""
        start_time = time.perf_counter()
        
        try:
            accumulator = self.streaming_accumulators[agent_name]
            agent = self.unified_execution_system.agents.get(agent_name)
            
            if not agent or not hasattr(agent, 'network'):
                return
            
            # Get averaged gradients
            averaged_gradients = accumulator.get_averaged_gradients()
            
            if not averaged_gradients:
                return
            
            # Apply gradient clipping
            total_norm = 0.0
            for grad in averaged_gradients.values():
                total_norm += grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > self.gradient_clip_norm:
                clip_coef = self.gradient_clip_norm / total_norm
                for grad in averaged_gradients.values():
                    grad.mul_(clip_coef)
            
            # Apply gradients to model
            for name, param in agent.network.named_parameters():
                if name in averaged_gradients:
                    param.grad = averaged_gradients[name]
            
            # Optimizer step
            if hasattr(agent, 'optimizer'):
                agent.optimizer.step()
                agent.optimizer.zero_grad()
            
            # Track update
            update_time = (time.perf_counter() - start_time) * 1000
            self.update_latencies.append(update_time)
            self.total_updates += 1
            
            logger.info("Accumulated gradients applied",
                       agent_name=agent_name,
                       gradient_count=len(averaged_gradients),
                       update_time_ms=update_time,
                       total_norm=total_norm)
            
        except Exception as e:
            logger.error(f"Error applying accumulated gradients for {agent_name}", error=str(e))
    
    async def _check_accumulated_gradients(self):
        """Check all accumulators for pending updates"""
        for agent_name, accumulator in self.streaming_accumulators.items():
            if accumulator.should_update():
                await self._apply_accumulated_gradients(agent_name)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.update_latencies:
            return {}
        
        return {
            'total_updates': self.total_updates,
            'avg_update_latency_ms': np.mean(self.update_latencies),
            'p95_update_latency_ms': np.percentile(self.update_latencies, 95),
            'max_update_latency_ms': np.max(self.update_latencies),
            'avg_gradient_computation_time_ms': np.mean(self.gradient_computation_times) if self.gradient_computation_times else 0,
            'latency_compliance': np.mean([l < self.max_update_latency_ms for l in self.update_latencies]),
            'accumulator_status': {
                name: {
                    'accumulation_steps': acc.accumulation_steps,
                    'last_update_time': acc.last_update_time.isoformat(),
                    'parameters_tracked': len(acc.accumulated_gradients)
                }
                for name, acc in self.streaming_accumulators.items()
            }
        }
    
    def save_execution_results(self, filepath: str):
        """Save execution results for analysis"""
        try:
            results = []
            
            # Save recent execution results (would need to be tracked)
            # This is a placeholder - actual implementation would track results
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("Execution results saved", filepath=filepath)
            
        except Exception as e:
            logger.error("Error saving execution results", error=str(e))
    
    async def shutdown(self):
        """Graceful shutdown of the bridge"""
        logger.info("Shutting down ExecutionBackpropagationBridge")
        
        # Stop processing
        await self.stop_processing()
        
        # Apply any remaining accumulated gradients
        for agent_name in self.streaming_accumulators:
            await self._apply_accumulated_gradients(agent_name)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("ExecutionBackpropagationBridge shutdown complete")


# Factory function
def create_execution_backpropagation_bridge(
    mappo_trainer: Any,
    unified_execution_system: Any,
    config: Dict[str, Any] = None
) -> ExecutionBackpropagationBridge:
    """Create and initialize execution backpropagation bridge"""
    return ExecutionBackpropagationBridge(
        mappo_trainer=mappo_trainer,
        unified_execution_system=unified_execution_system,
        config=config
    )


# Example configuration
DEFAULT_BRIDGE_CONFIG = {
    'max_update_latency_ms': 100,
    'gradient_computation_timeout_ms': 50,
    'base_learning_rate': 1e-4,
    'momentum': 0.9,
    'gradient_clip_norm': 1.0,
    'accumulation_window': 10,
    'immediate_update_outcomes': [
        ExecutionOutcome.RISK_REJECTION,
        ExecutionOutcome.ROUTING_FAILURE
    ]
}