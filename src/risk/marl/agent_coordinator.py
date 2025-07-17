"""
Agent Coordinator for Multi-Agent Risk Management

Coordinates the 4 risk agents for consensus-based decision making,
emergency protocols, and conflict resolution.

Features:
- Multi-agent consensus mechanisms
- Emergency override protocols
- Inter-agent communication framework
- Conflict resolution and priority management
- Real-time coordination with <10ms response time
"""

import asyncio
import threading
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import structlog
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
import queue
from collections import defaultdict

from src.core.events import EventBus, Event, EventType
from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.risk.marl.centralized_critic import CentralizedCritic, GlobalRiskState, RiskCriticMode

logger = structlog.get_logger()


class ConsensusMethod(Enum):
    """Methods for reaching consensus among agents"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average" 
    HIERARCHICAL = "hierarchical"
    COMMITTEE = "committee"
    EMERGENCY_OVERRIDE = "emergency_override"


class CoordinationMode(Enum):
    """Operating modes for agent coordination"""
    NORMAL = "normal"
    STRESS = "stress"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class AgentPriority(Enum):
    """Agent priority levels for conflict resolution"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentDecision:
    """Individual agent decision with metadata"""
    agent_name: str
    action: Union[int, np.ndarray]
    confidence: float
    priority: AgentPriority
    timestamp: datetime
    reasoning: Optional[str] = None
    risk_assessment: Optional[float] = None


@dataclass
class ConsensusResult:
    """Result of consensus mechanism"""
    consensus_action: Union[int, np.ndarray]
    confidence_score: float
    participating_agents: List[str]
    method_used: ConsensusMethod
    execution_time_ms: float
    conflicts_detected: List[str]
    overrides_applied: List[str]


@dataclass
class CoordinatorConfig:
    """Configuration for agent coordinator"""
    max_response_time_ms: float = 10.0
    consensus_timeout_ms: float = 5.0
    emergency_threshold: float = 0.8
    stress_threshold: float = 0.6
    
    # Agent weights for weighted consensus
    agent_weights: Dict[str, float] = field(default_factory=lambda: {
        'position_sizing': 1.0,
        'stop_target': 1.0, 
        'risk_monitor': 1.5,  # Higher weight for risk monitor
        'portfolio_optimizer': 1.0
    })
    
    # Priority hierarchy
    agent_priorities: Dict[str, AgentPriority] = field(default_factory=lambda: {
        'position_sizing': AgentPriority.MEDIUM,
        'stop_target': AgentPriority.MEDIUM,
        'risk_monitor': AgentPriority.HIGH,
        'portfolio_optimizer': AgentPriority.MEDIUM
    })
    
    # Emergency overrides
    enable_emergency_override: bool = True
    emergency_agent_authority: str = 'risk_monitor'


class AgentCoordinator:
    """
    Coordinates multiple risk agents for consensus-based decisions
    
    Manages:
    - Real-time agent communication
    - Consensus mechanisms and conflict resolution
    - Emergency protocol execution
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 config: CoordinatorConfig,
                 centralized_critic: CentralizedCritic,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize agent coordinator
        
        Args:
            config: Coordinator configuration
            centralized_critic: Centralized critic for global risk assessment
            event_bus: Event bus for communication
        """
        self.config = config
        self.centralized_critic = centralized_critic
        self.event_bus = event_bus
        
        # Registered agents
        self.agents: Dict[str, BaseRiskAgent] = {}
        self.agent_status: Dict[str, str] = {}  # 'active', 'inactive', 'error'
        
        # Coordination state
        self.current_mode = CoordinationMode.NORMAL
        self.decision_queue = queue.Queue(maxsize=100)
        self.consensus_history: List[ConsensusResult] = []
        
        # Performance tracking
        self.coordination_count = 0
        self.consensus_failures = 0
        self.emergency_activations = 0
        self.response_times = []
        
        # Threading and async execution
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="coord")
        self.coordination_lock = threading.RLock()
        self.running = False
        
        # Communication channels
        self.agent_channels: Dict[str, queue.Queue] = {}
        self.broadcast_channel = queue.Queue()
        
        logger.info("Agent coordinator initialized",
                   max_response_time=config.max_response_time_ms,
                   agents_expected=len(config.agent_weights))
    
    def register_agent(self, agent: BaseRiskAgent) -> bool:
        """
        Register a risk agent with the coordinator
        
        Args:
            agent: Risk agent to register
            
        Returns:
            True if registration successful
        """
        try:
            with self.coordination_lock:
                agent_name = agent.name
                
                if agent_name in self.agents:
                    logger.warning("Agent already registered", agent=agent_name)
                    return False
                
                self.agents[agent_name] = agent
                self.agent_status[agent_name] = 'active'
                self.agent_channels[agent_name] = queue.Queue(maxsize=50)
                
                # Subscribe to agent events
                if self.event_bus and hasattr(agent, 'event_bus'):
                    agent.event_bus = self.event_bus
                
                logger.info("Agent registered", agent=agent_name, total_agents=len(self.agents))
                return True
                
        except Exception as e:
            logger.error("Failed to register agent", agent=agent.name, error=str(e))
            return False
    
    def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent"""
        try:
            with self.coordination_lock:
                if agent_name not in self.agents:
                    return False
                
                del self.agents[agent_name]
                del self.agent_status[agent_name]
                if agent_name in self.agent_channels:
                    del self.agent_channels[agent_name]
                
                logger.info("Agent unregistered", agent=agent_name)
                return True
                
        except Exception as e:
            logger.error("Failed to unregister agent", agent=agent_name, error=str(e))
            return False
    
    def coordinate_decision(self, risk_state: RiskState) -> Dict[str, ConsensusResult]:
        """
        Coordinate decision-making across all agents
        
        Args:
            risk_state: Current risk state
            
        Returns:
            Dictionary of consensus results by action type
        """
        start_time = datetime.now()
        
        try:
            # Collect decisions from all active agents
            agent_decisions = self._collect_agent_decisions(risk_state)
            
            if not agent_decisions:
                logger.error("No agent decisions received")
                return {}
            
            # Group decisions by action type/space
            decision_groups = self._group_decisions_by_type(agent_decisions)
            
            # Reach consensus for each decision group
            consensus_results = {}
            for action_type, decisions in decision_groups.items():
                consensus = self._reach_consensus(action_type, decisions)
                consensus_results[action_type] = consensus
            
            # Check for emergency conditions
            self._check_emergency_conditions(consensus_results, risk_state)
            
            # Track performance
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            self.response_times.append(response_time)
            self.coordination_count += 1
            
            if response_time > self.config.max_response_time_ms:
                logger.warning("Coordination exceeded target response time",
                             response_time=response_time,
                             target=self.config.max_response_time_ms)
            
            # Store consensus history
            for consensus in consensus_results.values():
                self.consensus_history.append(consensus)
                if len(self.consensus_history) > 1000:  # Keep last 1000
                    self.consensus_history.pop(0)
            
            # Publish coordination results
            if self.event_bus:
                self._publish_coordination_results(consensus_results, risk_state)
            
            return consensus_results
            
        except Exception as e:
            logger.error("Error in coordination", error=str(e))
            self.consensus_failures += 1
            return {}
    
    def _collect_agent_decisions(self, risk_state: RiskState) -> List[AgentDecision]:
        """Collect decisions from all active agents"""
        
        decisions = []
        futures = {}
        
        # Submit agent decision tasks
        for agent_name, agent in self.agents.items():
            if self.agent_status.get(agent_name) == 'active':
                future = self.executor.submit(self._get_agent_decision, agent, risk_state)
                futures[agent_name] = future
        
        # Collect results with timeout
        timeout = self.config.consensus_timeout_ms / 1000.0
        
        for agent_name, future in futures.items():
            try:
                decision = future.result(timeout=timeout)
                if decision:
                    decisions.append(decision)
            except Exception as e:
                logger.warning("Agent decision failed", agent=agent_name, error=str(e))
                self.agent_status[agent_name] = 'error'
        
        return decisions
    
    def _get_agent_decision(self, agent: BaseRiskAgent, risk_state: RiskState) -> Optional[AgentDecision]:
        """Get decision from individual agent"""
        try:
            action, confidence = agent.make_decision(risk_state.to_vector())
            
            return AgentDecision(
                agent_name=agent.name,
                action=action,
                confidence=confidence,
                priority=self.config.agent_priorities.get(agent.name, AgentPriority.MEDIUM),
                timestamp=datetime.now(),
                reasoning=f"Risk assessment by {agent.name}",
                risk_assessment=confidence
            )
            
        except Exception as e:
            logger.error("Agent decision error", agent=agent.name, error=str(e))
            return None
    
    def _group_decisions_by_type(self, decisions: List[AgentDecision]) -> Dict[str, List[AgentDecision]]:
        """Group agent decisions by action type"""
        
        # For simplicity, group by agent name since each has different action space
        groups = {}
        for decision in decisions:
            # Each agent type gets its own group
            if decision.agent_name == 'position_sizing':
                groups['position_sizing'] = groups.get('position_sizing', []) + [decision]
            elif decision.agent_name == 'stop_target':
                groups['stop_target'] = groups.get('stop_target', []) + [decision]
            elif decision.agent_name == 'risk_monitor':
                groups['risk_monitor'] = groups.get('risk_monitor', []) + [decision]
            elif decision.agent_name == 'portfolio_optimizer':
                groups['portfolio_optimizer'] = groups.get('portfolio_optimizer', []) + [decision]
        
        return groups
    
    def _reach_consensus(self, action_type: str, decisions: List[AgentDecision]) -> ConsensusResult:
        """Reach consensus for a group of decisions"""
        
        start_time = datetime.now()
        conflicts = []
        overrides = []
        
        if not decisions:
            return ConsensusResult(
                consensus_action=0,
                confidence_score=0.0,
                participating_agents=[],
                method_used=ConsensusMethod.MAJORITY_VOTE,
                execution_time_ms=0.0,
                conflicts_detected=[],
                overrides_applied=[]
            )
        
        # For single decision, return as-is
        if len(decisions) == 1:
            decision = decisions[0]
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ConsensusResult(
                consensus_action=decision.action,
                confidence_score=decision.confidence,
                participating_agents=[decision.agent_name],
                method_used=ConsensusMethod.MAJORITY_VOTE,
                execution_time_ms=execution_time,
                conflicts_detected=conflicts,
                overrides_applied=overrides
            )
        
        # Check for emergency override
        emergency_decisions = [d for d in decisions if d.priority == AgentPriority.CRITICAL]
        if emergency_decisions and self.config.enable_emergency_override:
            decision = emergency_decisions[0]  # Take first emergency decision
            overrides.append(f"Emergency override by {decision.agent_name}")
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return ConsensusResult(
                consensus_action=decision.action,
                confidence_score=decision.confidence,
                participating_agents=[d.agent_name for d in decisions],
                method_used=ConsensusMethod.EMERGENCY_OVERRIDE,
                execution_time_ms=execution_time,
                conflicts_detected=conflicts,
                overrides_applied=overrides
            )
        
        # Weighted average consensus for continuous actions
        if isinstance(decisions[0].action, np.ndarray):
            consensus_action, confidence = self._weighted_average_consensus(decisions)
            method = ConsensusMethod.WEIGHTED_AVERAGE
        else:
            # Majority vote for discrete actions
            consensus_action, confidence = self._majority_vote_consensus(decisions)
            method = ConsensusMethod.MAJORITY_VOTE
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ConsensusResult(
            consensus_action=consensus_action,
            confidence_score=confidence,
            participating_agents=[d.agent_name for d in decisions],
            method_used=method,
            execution_time_ms=execution_time,
            conflicts_detected=conflicts,
            overrides_applied=overrides
        )
    
    def _weighted_average_consensus(self, decisions: List[AgentDecision]) -> Tuple[np.ndarray, float]:
        """Compute weighted average consensus for continuous actions"""
        
        if not decisions:
            return np.array([0.0]), 0.0
        
        # Get weights for participating agents
        total_weight = 0.0
        weighted_sum = None
        confidence_sum = 0.0
        
        for decision in decisions:
            weight = self.config.agent_weights.get(decision.agent_name, 1.0)
            confidence_weight = weight * decision.confidence
            
            if weighted_sum is None:
                weighted_sum = decision.action * confidence_weight
            else:
                weighted_sum += decision.action * confidence_weight
            
            total_weight += confidence_weight
            confidence_sum += decision.confidence
        
        if total_weight > 0:
            consensus_action = weighted_sum / total_weight
            avg_confidence = confidence_sum / len(decisions)
        else:
            consensus_action = decisions[0].action
            avg_confidence = decisions[0].confidence
        
        return consensus_action, avg_confidence
    
    def _majority_vote_consensus(self, decisions: List[AgentDecision]) -> Tuple[int, float]:
        """Compute majority vote consensus for discrete actions"""
        
        if not decisions:
            return 0, 0.0
        
        # Count votes with confidence weighting
        vote_scores = defaultdict(float)
        
        for decision in decisions:
            action = int(decision.action)
            weight = self.config.agent_weights.get(decision.agent_name, 1.0)
            confidence_weight = weight * decision.confidence
            vote_scores[action] += confidence_weight
        
        # Find action with highest score
        if vote_scores:
            consensus_action = max(vote_scores, key=vote_scores.get)
            max_score = vote_scores[consensus_action]
            total_score = sum(vote_scores.values())
            consensus_confidence = max_score / total_score if total_score > 0 else 0.0
        else:
            consensus_action = int(decisions[0].action)
            consensus_confidence = decisions[0].confidence
        
        return consensus_action, consensus_confidence
    
    def _check_emergency_conditions(self, consensus_results: Dict[str, ConsensusResult], risk_state: RiskState):
        """Check for emergency conditions and update coordination mode"""
        
        # Check risk monitor decisions for emergency actions
        if 'risk_monitor' in consensus_results:
            risk_action = consensus_results['risk_monitor'].consensus_action
            if isinstance(risk_action, (int, np.integer)) and risk_action == 3:  # Emergency stop
                self.current_mode = CoordinationMode.EMERGENCY
                self.emergency_activations += 1
                logger.critical("EMERGENCY MODE ACTIVATED", reason="risk_monitor_emergency_stop")
                
                if self.event_bus:
                    self.event_bus.publish(
                        self.event_bus.create_event(
                            EventType.EMERGENCY_STOP,
                            {
                                'coordinator_mode': self.current_mode.value,
                                'trigger_agent': 'risk_monitor',
                                'risk_state': risk_state.to_vector().tolist(),
                                'timestamp': datetime.now()
                            },
                            "agent_coordinator"
                        )
                    )
        
        # Check global risk level from centralized critic
        try:
            # Create global risk state for evaluation
            base_vector = risk_state.to_vector()
            global_risk_state = GlobalRiskState(
                position_sizing_risk=base_vector,
                stop_target_risk=base_vector,
                risk_monitor_risk=base_vector,
                portfolio_optimizer_risk=base_vector,
                total_portfolio_var=risk_state.var_estimate_5pct,
                portfolio_correlation_max=risk_state.correlation_risk,
                aggregate_leverage=risk_state.margin_usage_pct * 4.0,  # Approximate leverage
                liquidity_risk_score=1.0 - risk_state.liquidity_conditions,
                systemic_risk_level=risk_state.market_stress_level,
                timestamp=datetime.now(),
                market_hours_factor=risk_state.time_of_day_risk
            )
            
            global_risk_value, critic_mode = self.centralized_critic.evaluate_global_risk(global_risk_state)
            
            if critic_mode == RiskCriticMode.EMERGENCY and self.current_mode != CoordinationMode.EMERGENCY:
                self.current_mode = CoordinationMode.EMERGENCY
                self.emergency_activations += 1
                logger.critical("EMERGENCY MODE ACTIVATED", reason="centralized_critic_assessment")
            elif critic_mode == RiskCriticMode.STRESS and self.current_mode == CoordinationMode.NORMAL:
                self.current_mode = CoordinationMode.STRESS
                logger.warning("STRESS MODE ACTIVATED", reason="centralized_critic_assessment")
            
        except Exception as e:
            logger.error("Error checking emergency conditions", error=str(e))
    
    def _publish_coordination_results(self, consensus_results: Dict[str, ConsensusResult], risk_state: RiskState):
        """Publish coordination results via event bus"""
        if not self.event_bus:
            return
        
        event_data = {
            'coordination_mode': self.current_mode.value,
            'consensus_count': len(consensus_results),
            'risk_state': risk_state.to_vector().tolist(),
            'consensus_summary': {
                action_type: {
                    'action': result.consensus_action.tolist() if isinstance(result.consensus_action, np.ndarray) else result.consensus_action,
                    'confidence': result.confidence_score,
                    'method': result.method_used.value,
                    'agents': result.participating_agents
                }
                for action_type, result in consensus_results.items()
            },
            'performance': {
                'coordination_count': self.coordination_count,
                'consensus_failures': self.consensus_failures,
                'emergency_activations': self.emergency_activations,
                'avg_response_time_ms': np.mean(self.response_times[-100:]) if self.response_times else 0.0
            }
        }
        
        event = self.event_bus.create_event(
            EventType.COORDINATION_UPDATE,
            event_data,
            "agent_coordinator"
        )
        self.event_bus.publish(event)
    
    def execute_emergency_protocol(self, reason: str) -> bool:
        """Execute emergency protocol across all agents"""
        try:
            logger.critical("EXECUTING EMERGENCY PROTOCOL", reason=reason)
            
            self.current_mode = CoordinationMode.EMERGENCY
            self.emergency_activations += 1
            
            # Send emergency stop to all agents
            emergency_futures = []
            for agent_name, agent in self.agents.items():
                if self.agent_status.get(agent_name) == 'active':
                    future = self.executor.submit(agent.emergency_stop, reason)
                    emergency_futures.append((agent_name, future))
            
            # Wait for all emergency stops to complete
            success_count = 0
            for agent_name, future in emergency_futures:
                try:
                    result = future.result(timeout=1.0)  # 1 second timeout
                    if result:
                        success_count += 1
                except Exception as e:
                    logger.error("Emergency stop failed", agent=agent_name, error=str(e))
            
            # Publish emergency event
            if self.event_bus:
                self.event_bus.publish(
                    self.event_bus.create_event(
                        EventType.EMERGENCY_STOP,
                        {
                            'reason': reason,
                            'success_count': success_count,
                            'total_agents': len(self.agents),
                            'timestamp': datetime.now()
                        },
                        "agent_coordinator"
                    )
                )
            
            return success_count == len(self.agents)
            
        except Exception as e:
            logger.error("Error executing emergency protocol", error=str(e))
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get coordinator performance metrics"""
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        max_response_time = np.max(self.response_times) if self.response_times else 0.0
        
        active_agents = sum(1 for status in self.agent_status.values() if status == 'active')
        
        return {
            'coordination_count': self.coordination_count,
            'consensus_failures': self.consensus_failures,
            'emergency_activations': self.emergency_activations,
            'current_mode': self.current_mode.value,
            'active_agents': active_agents,
            'total_agents': len(self.agents),
            'avg_response_time_ms': avg_response_time,
            'max_response_time_ms': max_response_time,
            'consensus_failure_rate': self.consensus_failures / max(1, self.coordination_count),
            'agent_status': dict(self.agent_status)
        }
    
    def reset_coordination(self):
        """Reset coordinator state"""
        with self.coordination_lock:
            self.current_mode = CoordinationMode.NORMAL
            self.coordination_count = 0
            self.consensus_failures = 0
            self.emergency_activations = 0
            self.response_times.clear()
            self.consensus_history.clear()
            
            # Reset agent status
            for agent_name in self.agent_status:
                self.agent_status[agent_name] = 'active'
            
            logger.info("Agent coordinator reset")
    
    def shutdown(self):
        """Shutdown coordinator and cleanup resources"""
        self.running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear queues
        while not self.decision_queue.empty():
            try:
                self.decision_queue.get_nowait()
            except queue.Empty:
                break
        
        for channel in self.agent_channels.values():
            while not channel.empty():
                try:
                    channel.get_nowait()
                except queue.Empty:
                    break
        
        logger.info("Agent coordinator shutdown")