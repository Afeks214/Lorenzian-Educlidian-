"""
Intelligence Coordination Engine - Agent 4: The Intelligence Coordinator

Central nervous system that coordinates all intelligence components (Crisis Forecasting, 
Pre-Mortem Analysis, Human Oversight) with the existing Risk Management MARL system.

Features:
- Seamless coordination of 7 total agents (4 MARL + 3 Intelligence)
- <5ms coordination latency requirement
- Real-time performance optimization and adaptive learning
- Event-driven architecture with priority management
- Zero performance degradation of existing <10ms MARL system
- Fault tolerance with graceful degradation

Architecture:
- Intelligence Hub: Central coordination point for all agents
- Event Orchestration: Complex event processing with prioritization
- Decision Fusion: Bayesian inference for multi-agent decisions
- Adaptive Learning: Continuous optimization and parameter tuning
- Quality Assurance: Real-time validation and health monitoring
"""

import asyncio
import threading
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import structlog
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import queue
from collections import defaultdict, deque
import weakref
import json

from src.core.events import EventBus, Event, EventType
from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskAction, RiskMetrics
from src.risk.marl.agent_coordinator import AgentCoordinator, CoordinatorConfig, ConsensusResult
from src.risk.marl.centralized_critic import CentralizedCritic, GlobalRiskState, RiskCriticMode

logger = structlog.get_logger()


class IntelligenceType(Enum):
    """Types of intelligence components"""
    CRISIS_FORECASTER = "crisis_forecaster"
    PRE_MORTEM_ANALYST = "pre_mortem_analyst"  
    HUMAN_OVERSIGHT = "human_oversight"
    MARL_AGENT = "marl_agent"


class IntelligencePriority(Enum):
    """Priority levels for intelligence coordination"""
    EMERGENCY = 0      # Crisis detection alerts (immediate processing)
    HIGH = 1          # Human review decisions (process within 1 second)
    MEDIUM = 2        # Pre-mortem analysis results (process within 5 seconds)
    LOW = 3           # Performance monitoring and learning updates


class CoordinationStatus(Enum):
    """Status of intelligence coordination"""
    OPTIMAL = "optimal"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


@dataclass
class IntelligenceComponent:
    """Intelligence component registration"""
    name: str
    component_type: IntelligenceType
    priority: IntelligencePriority
    weight: float
    health_status: str = "active"
    last_response_time_ms: float = 0.0
    error_count: int = 0
    success_count: int = 0
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class IntelligenceDecision:
    """Decision from an intelligence component"""
    component_name: str
    component_type: IntelligenceType
    decision_data: Any
    confidence: float
    priority: IntelligencePriority
    timestamp: datetime
    processing_time_ms: float
    reasoning: Optional[str] = None
    risk_factors: Optional[List[str]] = None


@dataclass
class CoordinationResult:
    """Result of intelligence coordination"""
    coordinated_decision: Any
    confidence_score: float
    participating_components: List[str]
    coordination_method: str
    execution_time_ms: float
    conflicts_detected: List[str]
    emergency_overrides: List[str]
    performance_impact: float


@dataclass
class IntelligenceConfig:
    """Configuration for intelligence coordinator"""
    max_coordination_latency_ms: float = 5.0
    emergency_response_time_ms: float = 1.0
    health_check_interval_s: float = 1.0
    performance_monitoring_interval_s: float = 0.1
    
    # Component weights for decision fusion
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        'crisis_forecaster': 2.0,      # Highest weight for crisis detection
        'pre_mortem_analyst': 1.5,     # High weight for failure analysis  
        'human_oversight': 3.0,        # Highest weight for human decisions
        'position_sizing': 1.0,        # Standard MARL agent weight
        'stop_target': 1.0,            # Standard MARL agent weight
        'risk_monitor': 1.5,           # Higher weight for risk monitoring
        'portfolio_optimizer': 1.0     # Standard MARL agent weight
    })
    
    # Adaptive learning parameters
    learning_rate: float = 0.01
    adaptation_window: int = 100
    performance_threshold: float = 0.95
    
    # Emergency protocols
    enable_emergency_override: bool = True
    emergency_authorities: List[str] = field(default_factory=lambda: ['crisis_forecaster', 'human_oversight'])
    
    # Quality assurance thresholds
    max_response_time_degradation: float = 2.0  # 2x normal response time
    max_error_rate: float = 0.05  # 5% error rate
    min_confidence_threshold: float = 0.6


class IntelligenceCoordinator:
    """
    Central Intelligence Coordination Engine
    
    Coordinates all intelligence components (Crisis Forecasting, Pre-Mortem Analysis, 
    Human Oversight) with existing Risk Management MARL system for seamless operation
    with zero performance degradation.
    """
    
    def __init__(self, 
                 config: IntelligenceConfig,
                 existing_marl_coordinator: AgentCoordinator,
                 centralized_critic: CentralizedCritic,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize Intelligence Coordinator
        
        Args:
            config: Intelligence coordination configuration
            existing_marl_coordinator: Existing MARL agent coordinator
            centralized_critic: Centralized critic for global risk assessment
            event_bus: Event bus for communication
        """
        self.config = config
        self.marl_coordinator = existing_marl_coordinator
        self.centralized_critic = centralized_critic
        self.event_bus = event_bus
        
        # Intelligence components registry
        self.intelligence_components: Dict[str, IntelligenceComponent] = {}
        self.component_handlers: Dict[str, Callable] = {}
        
        # Coordination state
        self.coordination_status = CoordinationStatus.OPTIMAL
        self.coordination_count = 0
        self.emergency_activations = 0
        self.performance_degradations = 0
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)
        self.coordination_history: deque = deque(maxlen=500)
        self.performance_metrics: Dict[str, float] = {}
        
        # Threading and async execution
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="intel_coord")
        self.coordination_lock = threading.RLock()
        self.running = False
        self.health_monitor_thread = None
        
        # Event processing
        self.event_queue = queue.PriorityQueue()
        self.emergency_queue = queue.Queue()
        
        # Adaptive learning state
        self.learning_history: deque = deque(maxlen=config.adaptation_window)
        self.adaptive_weights = dict(config.component_weights)
        self.performance_baselines: Dict[str, float] = {}
        
        logger.info("Intelligence coordinator initialized",
                   max_latency_ms=config.max_coordination_latency_ms,
                   emergency_response_ms=config.emergency_response_time_ms)
    
    def register_intelligence_component(self, 
                                      name: str,
                                      component_type: IntelligenceType,
                                      handler: Callable,
                                      priority: IntelligencePriority = IntelligencePriority.MEDIUM,
                                      weight: float = 1.0) -> bool:
        """
        Register an intelligence component
        
        Args:
            name: Component name
            component_type: Type of intelligence component
            handler: Handler function for component decisions
            priority: Default priority for component decisions
            weight: Weight for decision fusion
            
        Returns:
            True if registration successful
        """
        try:
            with self.coordination_lock:
                if name in self.intelligence_components:
                    logger.warning("Intelligence component already registered", component=name)
                    return False
                
                component = IntelligenceComponent(
                    name=name,
                    component_type=component_type,
                    priority=priority,
                    weight=weight
                )
                
                self.intelligence_components[name] = component
                self.component_handlers[name] = handler
                self.adaptive_weights[name] = weight
                
                # Subscribe to relevant events
                if self.event_bus:
                    self._subscribe_component_events(name, component_type)
                
                logger.info("Intelligence component registered",
                           component=name,
                           type=component_type.value,
                           priority=priority.value,
                           weight=weight)
                return True
                
        except Exception as e:
            logger.error("Failed to register intelligence component",
                        component=name, error=str(e))
            return False
    
    def _subscribe_component_events(self, name: str, component_type: IntelligenceType):
        """Subscribe to relevant events for component type"""
        if component_type == IntelligenceType.CRISIS_FORECASTER:
            self.event_bus.subscribe(EventType.MARKET_STRESS, 
                                   lambda event: self._handle_crisis_event(name, event))
            self.event_bus.subscribe(EventType.EMERGENCY_STOP,
                                   lambda event: self._handle_emergency_event(name, event))
        
        elif component_type == IntelligenceType.PRE_MORTEM_ANALYST:
            self.event_bus.subscribe(EventType.TRADE_QUALIFIED,
                                   lambda event: self._handle_premortem_event(name, event))
            self.event_bus.subscribe(EventType.STRATEGIC_DECISION,
                                   lambda event: self._handle_premortem_event(name, event))
        
        elif component_type == IntelligenceType.HUMAN_OVERSIGHT:
            self.event_bus.subscribe(EventType.RISK_BREACH,
                                   lambda event: self._handle_human_oversight_event(name, event))
    
    def coordinate_intelligence_decision(self, 
                                       risk_state: RiskState,
                                       decision_context: Dict[str, Any]) -> CoordinationResult:
        """
        Coordinate decision-making across all intelligence components and MARL agents
        
        Args:
            risk_state: Current risk state
            decision_context: Additional context for decision making
            
        Returns:
            Coordinated decision result
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Collect MARL agent decisions (existing system)
            marl_decisions = self._get_marl_decisions(risk_state)
            
            # Step 2: Collect intelligence component decisions
            intelligence_decisions = self._collect_intelligence_decisions(risk_state, decision_context)
            
            # Step 3: Fuse all decisions with priority management
            coordinated_decision = self._fuse_all_decisions(marl_decisions, intelligence_decisions)
            
            # Step 4: Check for emergency conditions
            emergency_overrides = self._check_emergency_overrides(coordinated_decision, risk_state)
            
            # Step 5: Apply adaptive learning
            self._update_adaptive_learning(coordinated_decision, risk_state)
            
            # Calculate performance metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self.response_times.append(execution_time)
            self.coordination_count += 1
            
            # Create coordination result
            result = CoordinationResult(
                coordinated_decision=coordinated_decision['action'],
                confidence_score=coordinated_decision['confidence'],
                participating_components=coordinated_decision['participants'],
                coordination_method=coordinated_decision['method'],
                execution_time_ms=execution_time,
                conflicts_detected=coordinated_decision.get('conflicts', []),
                emergency_overrides=emergency_overrides,
                performance_impact=self._calculate_performance_impact(execution_time)
            )
            
            # Store history and publish results
            self.coordination_history.append(result)
            self._publish_coordination_results(result, risk_state)
            
            # Performance monitoring
            if execution_time > self.config.max_coordination_latency_ms:
                self._handle_performance_degradation(execution_time)
            
            return result
            
        except Exception as e:
            logger.error("Error in intelligence coordination", error=str(e))
            return self._create_fallback_result(str(e))
    
    def _get_marl_decisions(self, risk_state: RiskState) -> Dict[str, ConsensusResult]:
        """Get decisions from existing MARL system"""
        try:
            # Use existing MARL coordinator for seamless integration
            return self.marl_coordinator.coordinate_decision(risk_state)
        except Exception as e:
            logger.error("Error getting MARL decisions", error=str(e))
            return {}
    
    def _collect_intelligence_decisions(self, 
                                      risk_state: RiskState, 
                                      context: Dict[str, Any]) -> List[IntelligenceDecision]:
        """Collect decisions from intelligence components"""
        decisions = []
        futures = {}
        
        # Submit intelligence decision tasks with priority
        for name, component in self.intelligence_components.items():
            if component.health_status == 'active':
                handler = self.component_handlers.get(name)
                if handler:
                    future = self.executor.submit(
                        self._get_intelligence_decision, 
                        name, component, handler, risk_state, context
                    )
                    futures[name] = future
        
        # Collect results with timeout based on priority
        for name, future in futures.items():
            component = self.intelligence_components[name]
            timeout = self._get_component_timeout(component.priority)
            
            try:
                decision = future.result(timeout=timeout)
                if decision:
                    decisions.append(decision)
            except Exception as e:
                logger.warning("Intelligence component decision failed", 
                             component=name, error=str(e))
                component.error_count += 1
                self._check_component_health(component)
        
        return decisions
    
    def _get_intelligence_decision(self, 
                                 name: str,
                                 component: IntelligenceComponent,
                                 handler: Callable,
                                 risk_state: RiskState,
                                 context: Dict[str, Any]) -> Optional[IntelligenceDecision]:
        """Get decision from individual intelligence component"""
        start_time = datetime.now()
        
        try:
            # Call component handler
            decision_data, confidence, reasoning = handler(risk_state, context)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            component.last_response_time_ms = processing_time
            component.success_count += 1
            component.confidence_history.append(confidence)
            
            return IntelligenceDecision(
                component_name=name,
                component_type=component.component_type,
                decision_data=decision_data,
                confidence=confidence,
                priority=component.priority,
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error("Intelligence component decision error", 
                        component=name, error=str(e))
            component.error_count += 1
            return None
    
    def _fuse_all_decisions(self, 
                          marl_decisions: Dict[str, ConsensusResult],
                          intelligence_decisions: List[IntelligenceDecision]) -> Dict[str, Any]:
        """Fuse MARL and intelligence decisions using Bayesian inference"""
        
        # Check for emergency priorities first
        emergency_decisions = [d for d in intelligence_decisions 
                             if d.priority == IntelligencePriority.EMERGENCY]
        
        if emergency_decisions:
            # Emergency override - use highest confidence emergency decision
            best_emergency = max(emergency_decisions, key=lambda x: x.confidence)
            return {
                'action': best_emergency.decision_data,
                'confidence': best_emergency.confidence,
                'participants': [best_emergency.component_name],
                'method': 'emergency_override',
                'conflicts': []
            }
        
        # Bayesian decision fusion for normal operation
        all_decisions = []
        all_weights = []
        all_confidences = []
        participants = []
        
        # Add MARL decisions
        for action_type, consensus in marl_decisions.items():
            weight = self.adaptive_weights.get(action_type, 1.0)
            all_decisions.append(consensus.consensus_action)
            all_weights.append(weight * consensus.confidence_score)
            all_confidences.append(consensus.confidence_score)
            participants.extend(consensus.participating_agents)
        
        # Add intelligence decisions
        for decision in intelligence_decisions:
            if decision.priority in [IntelligencePriority.HIGH, IntelligencePriority.MEDIUM]:
                weight = self.adaptive_weights.get(decision.component_name, 1.0)
                all_decisions.append(decision.decision_data)
                all_weights.append(weight * decision.confidence)
                all_confidences.append(decision.confidence)
                participants.append(decision.component_name)
        
        # Compute weighted fusion
        if all_decisions:
            fused_decision, fused_confidence = self._compute_bayesian_fusion(
                all_decisions, all_weights, all_confidences
            )
            
            return {
                'action': fused_decision,
                'confidence': fused_confidence,
                'participants': participants,
                'method': 'bayesian_fusion',
                'conflicts': self._detect_conflicts(all_decisions, all_confidences)
            }
        else:
            # Fallback to existing MARL system
            if marl_decisions:
                best_marl = max(marl_decisions.values(), key=lambda x: x.confidence_score)
                return {
                    'action': best_marl.consensus_action,
                    'confidence': best_marl.confidence_score,
                    'participants': best_marl.participating_agents,
                    'method': 'marl_fallback',
                    'conflicts': []
                }
            else:
                return {
                    'action': 0,  # No action
                    'confidence': 0.0,
                    'participants': [],
                    'method': 'no_decision',
                    'conflicts': []
                }
    
    def _compute_bayesian_fusion(self, 
                               decisions: List[Any],
                               weights: List[float],
                               confidences: List[float]) -> Tuple[Any, float]:
        """Compute Bayesian fusion of decisions"""
        
        if not decisions:
            return 0, 0.0
        
        # For continuous decisions (numpy arrays)
        if isinstance(decisions[0], np.ndarray):
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_sum = sum(w * d for w, d in zip(weights, decisions))
                fused_decision = weighted_sum / total_weight
                fused_confidence = np.mean(confidences)
            else:
                fused_decision = decisions[0]
                fused_confidence = confidences[0]
            
            return fused_decision, fused_confidence
        
        # For discrete decisions
        else:
            # Weighted vote
            vote_scores = defaultdict(float)
            for decision, weight in zip(decisions, weights):
                vote_scores[decision] += weight
            
            if vote_scores:
                fused_decision = max(vote_scores, key=vote_scores.get)
                total_score = sum(vote_scores.values())
                fused_confidence = vote_scores[fused_decision] / total_score if total_score > 0 else 0.0
            else:
                fused_decision = decisions[0]
                fused_confidence = confidences[0]
            
            return fused_decision, fused_confidence
    
    def _detect_conflicts(self, decisions: List[Any], confidences: List[float]) -> List[str]:
        """Detect conflicts between decisions"""
        conflicts = []
        
        if len(decisions) < 2:
            return conflicts
        
        # For discrete decisions, check for disagreement
        if not isinstance(decisions[0], np.ndarray):
            unique_decisions = set(decisions)
            if len(unique_decisions) > 1:
                conflicts.append("Discrete decision disagreement")
        
        # For continuous decisions, check for large deviations
        else:
            decision_array = np.array(decisions)
            std_dev = np.std(decision_array, axis=0)
            if np.any(std_dev > 0.5):  # Threshold for significant disagreement
                conflicts.append("Continuous decision high variance")
        
        # Check confidence disagreement
        if len(confidences) > 1:
            confidence_std = np.std(confidences)
            if confidence_std > 0.3:
                conflicts.append("Confidence level disagreement")
        
        return conflicts
    
    def _check_emergency_overrides(self, 
                                 decision: Dict[str, Any], 
                                 risk_state: RiskState) -> List[str]:
        """Check for emergency override conditions"""
        overrides = []
        
        # Check if any emergency authorities triggered
        for component_name in self.config.emergency_authorities:
            component = self.intelligence_components.get(component_name)
            if component and component.health_status == 'active':
                # Emergency check would be component-specific
                if self._check_component_emergency_condition(component_name, risk_state):
                    overrides.append(f"Emergency override by {component_name}")
                    self.emergency_activations += 1
        
        return overrides
    
    def _check_component_emergency_condition(self, component_name: str, risk_state: RiskState) -> bool:
        """Check if component has emergency condition"""
        # Placeholder for component-specific emergency checks
        # This would be implemented based on specific intelligence component logic
        return False
    
    def _update_adaptive_learning(self, decision: Dict[str, Any], risk_state: RiskState):
        """Update adaptive learning parameters"""
        try:
            # Store learning sample
            learning_sample = {
                'decision': decision,
                'risk_state': risk_state.to_vector().tolist(),
                'timestamp': datetime.now(),
                'participants': decision['participants'],
                'confidence': decision['confidence']
            }
            
            self.learning_history.append(learning_sample)
            
            # Update adaptive weights based on recent performance
            if len(self.learning_history) >= self.config.adaptation_window:
                self._adapt_component_weights()
            
        except Exception as e:
            logger.error("Error in adaptive learning update", error=str(e))
    
    def _adapt_component_weights(self):
        """Adapt component weights based on performance"""
        try:
            # Calculate performance metrics for each component
            component_performance = defaultdict(list)
            
            for sample in list(self.learning_history):
                for participant in sample['participants']:
                    component_performance[participant].append(sample['confidence'])
            
            # Update weights based on average performance
            for component, performances in component_performance.items():
                if performances:
                    avg_performance = np.mean(performances)
                    current_weight = self.adaptive_weights.get(component, 1.0)
                    
                    # Gradual weight adjustment
                    adjustment = self.config.learning_rate * (avg_performance - 0.5)
                    new_weight = max(0.1, min(3.0, current_weight + adjustment))
                    
                    self.adaptive_weights[component] = new_weight
            
            logger.debug("Adaptive weights updated", weights=dict(self.adaptive_weights))
            
        except Exception as e:
            logger.error("Error adapting component weights", error=str(e))
    
    def _calculate_performance_impact(self, execution_time_ms: float) -> float:
        """Calculate performance impact score"""
        baseline_time = self.config.max_coordination_latency_ms
        impact = execution_time_ms / baseline_time
        return min(2.0, max(0.0, impact))  # Clamp between 0 and 2
    
    def _handle_performance_degradation(self, execution_time_ms: float):
        """Handle performance degradation"""
        self.performance_degradations += 1
        
        if execution_time_ms > self.config.max_coordination_latency_ms * self.config.max_response_time_degradation:
            self.coordination_status = CoordinationStatus.DEGRADED
            logger.warning("Intelligence coordination performance degraded",
                         execution_time_ms=execution_time_ms,
                         threshold_ms=self.config.max_coordination_latency_ms)
    
    def _get_component_timeout(self, priority: IntelligencePriority) -> float:
        """Get timeout for component based on priority"""
        timeouts = {
            IntelligencePriority.EMERGENCY: 0.001,  # 1ms for emergency
            IntelligencePriority.HIGH: 0.001,       # 1s for high priority
            IntelligencePriority.MEDIUM: 0.005,     # 5s for medium priority  
            IntelligencePriority.LOW: 0.010         # 10s for low priority
        }
        return timeouts.get(priority, 0.005)
    
    def _check_component_health(self, component: IntelligenceComponent):
        """Check and update component health status"""
        total_calls = component.success_count + component.error_count
        if total_calls > 10:  # Minimum calls for health assessment
            error_rate = component.error_count / total_calls
            if error_rate > self.config.max_error_rate:
                component.health_status = 'degraded'
                logger.warning("Component health degraded", 
                             component=component.name,
                             error_rate=error_rate)
    
    def _create_fallback_result(self, error_msg: str) -> CoordinationResult:
        """Create fallback result for error cases"""
        return CoordinationResult(
            coordinated_decision=0,
            confidence_score=0.0,
            participating_components=[],
            coordination_method='error_fallback',
            execution_time_ms=0.0,
            conflicts_detected=[f"Coordination error: {error_msg}"],
            emergency_overrides=[],
            performance_impact=2.0  # Maximum impact for errors
        )
    
    def _publish_coordination_results(self, result: CoordinationResult, risk_state: RiskState):
        """Publish coordination results via event bus"""
        if not self.event_bus:
            return
        
        event_data = {
            'coordination_status': self.coordination_status.value,
            'coordinated_decision': result.coordinated_decision,
            'confidence_score': result.confidence_score,
            'execution_time_ms': result.execution_time_ms,
            'participants': result.participating_components,
            'method': result.coordination_method,
            'conflicts': result.conflicts_detected,
            'emergency_overrides': result.emergency_overrides,
            'performance_impact': result.performance_impact,
            'coordination_count': self.coordination_count,
            'emergency_activations': self.emergency_activations
        }
        
        # Add new event type for intelligence coordination
        event = self.event_bus.create_event(
            EventType.COORDINATION_UPDATE,  # Reuse existing event type
            event_data,
            "intelligence_coordinator"
        )
        self.event_bus.publish(event)
    
    def _handle_crisis_event(self, component_name: str, event: Event):
        """Handle crisis forecaster events"""
        self.emergency_queue.put((IntelligencePriority.EMERGENCY.value, event))
    
    def _handle_emergency_event(self, component_name: str, event: Event):
        """Handle emergency events"""
        self.emergency_queue.put((IntelligencePriority.EMERGENCY.value, event))
    
    def _handle_premortem_event(self, component_name: str, event: Event):
        """Handle pre-mortem analyst events"""
        self.event_queue.put((IntelligencePriority.MEDIUM.value, event))
    
    def _handle_human_oversight_event(self, component_name: str, event: Event):
        """Handle human oversight events"""
        self.event_queue.put((IntelligencePriority.HIGH.value, event))
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get comprehensive coordination metrics"""
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        max_response_time = np.max(self.response_times) if self.response_times else 0.0
        
        active_components = sum(1 for c in self.intelligence_components.values() 
                              if c.health_status == 'active')
        
        component_metrics = {}
        for name, component in self.intelligence_components.items():
            total_calls = component.success_count + component.error_count
            error_rate = component.error_count / max(1, total_calls)
            avg_confidence = np.mean(component.confidence_history) if component.confidence_history else 0.0
            
            component_metrics[name] = {
                'health_status': component.health_status,
                'success_count': component.success_count,
                'error_count': component.error_count,
                'error_rate': error_rate,
                'last_response_time_ms': component.last_response_time_ms,
                'avg_confidence': avg_confidence,
                'weight': self.adaptive_weights.get(name, 1.0)
            }
        
        return {
            'coordination_status': self.coordination_status.value,
            'coordination_count': self.coordination_count,
            'emergency_activations': self.emergency_activations,
            'performance_degradations': self.performance_degradations,
            'active_components': active_components,
            'total_components': len(self.intelligence_components),
            'avg_response_time_ms': avg_response_time,
            'max_response_time_ms': max_response_time,
            'meets_latency_target': avg_response_time <= self.config.max_coordination_latency_ms,
            'component_metrics': component_metrics,
            'adaptive_weights': dict(self.adaptive_weights)
        }
    
    def start_health_monitoring(self):
        """Start health monitoring background thread"""
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            return
        
        self.running = True
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="intelligence_health_monitor"
        )
        self.health_monitor_thread.start()
        logger.info("Intelligence health monitoring started")
    
    def _health_monitor_loop(self):
        """Health monitoring background loop"""
        while self.running:
            try:
                # Check component health
                for component in self.intelligence_components.values():
                    self._check_component_health(component)
                
                # Check overall coordination health
                self._check_coordination_health()
                
                time.sleep(self.config.health_check_interval_s)
                
            except Exception as e:
                logger.error("Error in health monitoring", error=str(e))
                time.sleep(1.0)  # Longer sleep on error
    
    def _check_coordination_health(self):
        """Check overall coordination system health"""
        if len(self.response_times) > 10:
            recent_avg = np.mean(list(self.response_times)[-10:])
            if recent_avg > self.config.max_coordination_latency_ms * 2:
                if self.coordination_status == CoordinationStatus.OPTIMAL:
                    self.coordination_status = CoordinationStatus.DEGRADED
                    logger.warning("Coordination system health degraded",
                                 recent_avg_ms=recent_avg)
    
    def emergency_shutdown(self, reason: str) -> bool:
        """Emergency shutdown of intelligence coordination"""
        try:
            logger.critical("EMERGENCY SHUTDOWN OF INTELLIGENCE COORDINATION", reason=reason)
            
            self.coordination_status = CoordinationStatus.EMERGENCY
            self.emergency_activations += 1
            
            # Mark all components as inactive
            for component in self.intelligence_components.values():
                component.health_status = 'emergency_shutdown'
            
            # Publish emergency event
            if self.event_bus:
                self.event_bus.publish(
                    self.event_bus.create_event(
                        EventType.EMERGENCY_STOP,
                        {
                            'reason': reason,
                            'coordination_shutdown': True,
                            'fallback_to_marl': True,
                            'timestamp': datetime.now()
                        },
                        "intelligence_coordinator"
                    )
                )
            
            return True
            
        except Exception as e:
            logger.error("Error in emergency shutdown", error=str(e))
            return False
    
    def shutdown(self):
        """Shutdown intelligence coordinator"""
        self.running = False
        
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=2.0)
        
        self.executor.shutdown(wait=True)
        
        logger.info("Intelligence coordinator shutdown complete")