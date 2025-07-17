"""
Decision Interception System for Pre-Mortem Analysis

Automatically intercepts significant trading decisions from MARL agents and
routes them through pre-mortem analysis before execution.

Key Features:
- Real-time decision monitoring and filtering
- Threshold-based interception criteria
- Integration with all MARL trading agents
- Emergency bypass for crisis situations
- Parallel processing for multiple decisions
- Complete audit trail and logging
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import structlog
from threading import Lock, RLock
from queue import Queue, PriorityQueue, Empty
import uuid

from src.core.events import EventBus, Event, EventType
from src.risk.agents.base_risk_agent import RiskState

logger = structlog.get_logger()


class DecisionType(Enum):
    """Types of trading decisions that can be intercepted"""
    POSITION_SIZING = "position_sizing"
    STOP_TARGET_ADJUSTMENT = "stop_target_adjustment"
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"
    RISK_REDUCTION = "risk_reduction"
    EMERGENCY_ACTION = "emergency_action"


class DecisionPriority(Enum):
    """Priority levels for decision processing"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class InterceptionStatus(Enum):
    """Status of decision interception"""
    PENDING = "pending"
    IN_ANALYSIS = "in_analysis"
    APPROVED = "approved"
    REJECTED = "rejected"
    BYPASSED = "bypassed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class DecisionContext:
    """Context information for a trading decision"""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    decision_type: DecisionType = DecisionType.POSITION_SIZING
    priority: DecisionPriority = DecisionPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Decision parameters
    current_position_size: float = 0.0
    proposed_position_size: float = 0.0
    position_change_amount: float = 0.0
    position_change_percent: float = 0.0
    portfolio_impact_percent: float = 0.0
    
    # Market context
    current_price: float = 0.0
    symbol: str = ""
    market_volatility: float = 0.0
    
    # Risk context
    current_risk_state: Optional[RiskState] = None
    var_impact: float = 0.0
    
    # Additional metadata
    reasoning: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_significance_score(self) -> float:
        """Calculate decision significance score for filtering"""
        score = 0.0
        
        # Position size impact
        score += abs(self.position_change_percent) * 10
        
        # Portfolio impact
        score += abs(self.portfolio_impact_percent) * 20
        
        # VaR impact
        score += abs(self.var_impact) * 15
        
        # Priority weighting
        priority_weights = {
            DecisionPriority.LOW: 0.5,
            DecisionPriority.NORMAL: 1.0,
            DecisionPriority.HIGH: 2.0,
            DecisionPriority.CRITICAL: 4.0,
            DecisionPriority.EMERGENCY: 8.0
        }
        score *= priority_weights.get(self.priority, 1.0)
        
        return score


@dataclass
class InterceptionResult:
    """Result of decision interception and analysis"""
    decision_id: str
    status: InterceptionStatus
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Analysis results
    failure_probability: float = 0.0
    recommendation: str = "UNKNOWN"  # GO, CAUTION, NO_GO
    confidence: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown_prob: float = 0.0
    
    # Processing metadata
    analysis_time_ms: float = 0.0
    bypass_reason: Optional[str] = None
    error_message: Optional[str] = None
    
    # Human review
    requires_human_review: bool = False
    human_review_reason: str = ""


@dataclass
class InterceptionConfig:
    """Configuration for decision interception"""
    # Filtering thresholds
    min_position_size_threshold: float = 10000.0     # $10,000 minimum
    min_portfolio_impact_threshold: float = 0.20     # 20% portfolio impact
    min_significance_score: float = 15.0             # Minimum significance
    
    # Processing limits
    max_concurrent_analyses: int = 5                 # Max parallel analyses
    analysis_timeout_seconds: float = 2.0           # 2 second timeout
    queue_size_limit: int = 100                     # Max queued decisions
    
    # Bypass conditions
    enable_emergency_bypass: bool = True             # Allow emergency bypass
    crisis_mode_enabled: bool = False               # Crisis mode override
    max_response_time_ms: float = 100.0             # Max response time
    
    # Agent filtering
    monitored_agents: List[str] = field(default_factory=lambda: [
        "position_sizing_agent", "stop_target_agent", 
        "portfolio_optimizer", "risk_monitor_agent"
    ])
    
    # Decision type filtering
    monitored_decision_types: List[DecisionType] = field(default_factory=lambda: [
        DecisionType.POSITION_SIZING,
        DecisionType.STOP_TARGET_ADJUSTMENT,
        DecisionType.PORTFOLIO_REBALANCING
    ])


class DecisionInterceptor:
    """
    Decision Interception System for Pre-Mortem Analysis
    
    Monitors trading decisions from MARL agents and automatically routes
    significant decisions through pre-mortem analysis before execution.
    
    Features:
    - Real-time decision monitoring
    - Threshold-based filtering
    - Parallel processing
    - Emergency bypass capability
    - Complete audit trail
    """
    
    def __init__(self, 
                 config: InterceptionConfig,
                 event_bus: EventBus,
                 premortem_analyzer: Callable[[DecisionContext], InterceptionResult]):
        """
        Initialize decision interceptor
        
        Args:
            config: Interception configuration
            event_bus: Event bus for agent communication
            premortem_analyzer: Pre-mortem analysis function
        """
        self.config = config
        self.event_bus = event_bus
        self.premortem_analyzer = premortem_analyzer
        
        # Processing infrastructure
        self.decision_queue = PriorityQueue(maxsize=config.queue_size_limit)
        self.pending_decisions: Dict[str, DecisionContext] = {}
        self.completed_analyses: Dict[str, InterceptionResult] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_concurrent_analyses)
        
        # Thread safety
        self.decisions_lock = RLock()
        self.results_lock = RLock()
        
        # Performance tracking
        self.interception_stats = {
            'total_decisions': 0,
            'intercepted_decisions': 0,
            'bypassed_decisions': 0,
            'avg_analysis_time_ms': 0.0,
            'analysis_timeouts': 0,
            'analysis_errors': 0
        }
        
        # Active analysis futures
        self.active_analyses: Dict[str, Future] = {}
        self.analysis_futures_lock = Lock()
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("Decision interceptor initialized",
                   monitored_agents=config.monitored_agents,
                   max_concurrent=config.max_concurrent_analyses)
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for decision monitoring"""
        # Subscribe to relevant trading events
        self.event_bus.subscribe(EventType.POSITION_SIZE_UPDATE, self._handle_position_sizing)
        self.event_bus.subscribe(EventType.TRADE_QUALIFIED, self._handle_trade_decision)
        self.event_bus.subscribe(EventType.RISK_UPDATE, self._handle_risk_decision)
        
        logger.info("Event subscriptions configured for decision interception")
    
    def _handle_position_sizing(self, event: Event) -> None:
        """Handle position sizing decisions"""
        try:
            payload = event.payload
            
            context = DecisionContext(
                agent_name=event.source,
                decision_type=DecisionType.POSITION_SIZING,
                current_position_size=payload.get('current_size', 0.0),
                proposed_position_size=payload.get('proposed_size', 0.0),
                symbol=payload.get('symbol', ''),
                reasoning=payload.get('reasoning', ''),
                confidence=payload.get('confidence', 0.0)
            )
            
            # Calculate derived metrics
            context.position_change_amount = abs(
                context.proposed_position_size - context.current_position_size
            )
            context.position_change_percent = (
                context.position_change_amount / max(abs(context.current_position_size), 1.0) * 100
            )
            
            self._process_decision(context)
            
        except Exception as e:
            logger.error("Error handling position sizing decision", error=str(e))
    
    def _handle_trade_decision(self, event: Event) -> None:
        """Handle general trade decisions"""
        try:
            payload = event.payload
            
            context = DecisionContext(
                agent_name=event.source,
                decision_type=DecisionType.POSITION_SIZING,  # Default
                current_price=payload.get('price', 0.0),
                symbol=payload.get('symbol', ''),
                reasoning=payload.get('reasoning', ''),
                confidence=payload.get('confidence', 0.0),
                metadata=payload
            )
            
            self._process_decision(context)
            
        except Exception as e:
            logger.error("Error handling trade decision", error=str(e))
    
    def _handle_risk_decision(self, event: Event) -> None:
        """Handle risk management decisions"""
        try:
            payload = event.payload
            
            context = DecisionContext(
                agent_name=event.source,
                decision_type=DecisionType.RISK_REDUCTION,
                priority=DecisionPriority.HIGH,
                reasoning=payload.get('reasoning', ''),
                confidence=payload.get('confidence', 0.0),
                current_risk_state=payload.get('risk_state'),
                metadata=payload
            )
            
            self._process_decision(context)
            
        except Exception as e:
            logger.error("Error handling risk decision", error=str(e))
    
    def intercept_decision(self, decision_context: DecisionContext) -> Optional[InterceptionResult]:
        """
        Manually intercept a trading decision for analysis
        
        Args:
            decision_context: Decision context to analyze
            
        Returns:
            Interception result if processed, None if bypassed
        """
        return self._process_decision(decision_context)
    
    def _process_decision(self, context: DecisionContext) -> Optional[InterceptionResult]:
        """Process a trading decision through the interception pipeline"""
        self.interception_stats['total_decisions'] += 1
        
        try:
            # Check if decision should be intercepted
            if not self._should_intercept_decision(context):
                self.interception_stats['bypassed_decisions'] += 1
                logger.debug("Decision bypassed", 
                           decision_id=context.decision_id,
                           agent=context.agent_name)
                return None
            
            # Check for emergency bypass conditions
            if self._should_bypass_emergency(context):
                result = InterceptionResult(
                    decision_id=context.decision_id,
                    status=InterceptionStatus.BYPASSED,
                    bypass_reason="Emergency bypass",
                    recommendation="GO"
                )
                self._store_result(result)
                self.interception_stats['bypassed_decisions'] += 1
                return result
            
            # Queue decision for analysis
            self._queue_decision_for_analysis(context)
            self.interception_stats['intercepted_decisions'] += 1
            
            # Wait for analysis completion or timeout
            return self._wait_for_analysis_result(context)
            
        except Exception as e:
            logger.error("Error processing decision", 
                        decision_id=context.decision_id,
                        error=str(e))
            
            # Return error result
            result = InterceptionResult(
                decision_id=context.decision_id,
                status=InterceptionStatus.ERROR,
                error_message=str(e),
                recommendation="CAUTION"
            )
            self._store_result(result)
            return result
    
    def _should_intercept_decision(self, context: DecisionContext) -> bool:
        """Determine if a decision should be intercepted for analysis"""
        # Check agent filtering
        if context.agent_name not in self.config.monitored_agents:
            return False
        
        # Check decision type filtering
        if context.decision_type not in self.config.monitored_decision_types:
            return False
        
        # Check significance thresholds
        significance_score = context.get_significance_score()
        if significance_score < self.config.min_significance_score:
            return False
        
        # Check position size threshold
        if abs(context.position_change_amount) < self.config.min_position_size_threshold:
            return False
        
        # Check portfolio impact threshold
        if abs(context.portfolio_impact_percent) < self.config.min_portfolio_impact_threshold:
            return False
        
        return True
    
    def _should_bypass_emergency(self, context: DecisionContext) -> bool:
        """Check if decision should bypass analysis due to emergency conditions"""
        if not self.config.enable_emergency_bypass:
            return False
        
        # Crisis mode override
        if self.config.crisis_mode_enabled:
            return True
        
        # Emergency priority decisions
        if context.priority == DecisionPriority.EMERGENCY:
            return True
        
        # Queue overflow protection
        if self.decision_queue.qsize() >= self.config.queue_size_limit * 0.9:
            logger.warning("Decision queue near capacity, bypassing non-critical decisions")
            return context.priority != DecisionPriority.CRITICAL
        
        return False
    
    def _queue_decision_for_analysis(self, context: DecisionContext) -> None:
        """Queue decision for pre-mortem analysis"""
        try:
            # Priority queue ordering (lower number = higher priority)
            priority = (6 - context.priority.value, time.time())
            
            self.decision_queue.put((priority, context), timeout=1.0)
            
            with self.decisions_lock:
                self.pending_decisions[context.decision_id] = context
            
            # Submit analysis task
            future = self.thread_pool.submit(self._analyze_decision, context)
            
            with self.analysis_futures_lock:
                self.active_analyses[context.decision_id] = future
            
            logger.debug("Decision queued for analysis",
                        decision_id=context.decision_id,
                        queue_size=self.decision_queue.qsize())
                        
        except Exception as e:
            logger.error("Failed to queue decision", 
                        decision_id=context.decision_id,
                        error=str(e))
            raise
    
    def _analyze_decision(self, context: DecisionContext) -> InterceptionResult:
        """Analyze decision using pre-mortem analysis"""
        start_time = time.perf_counter()
        
        try:
            # Run pre-mortem analysis
            result = self.premortem_analyzer(context)
            
            # Calculate analysis time
            analysis_time = (time.perf_counter() - start_time) * 1000
            result.analysis_time_ms = analysis_time
            
            # Update stats
            self.interception_stats['avg_analysis_time_ms'] = (
                (self.interception_stats['avg_analysis_time_ms'] * 
                 (self.interception_stats['intercepted_decisions'] - 1) + analysis_time) /
                self.interception_stats['intercepted_decisions']
            )
            
            # Store result
            self._store_result(result)
            
            logger.info("Decision analysis completed",
                       decision_id=context.decision_id,
                       recommendation=result.recommendation,
                       analysis_time_ms=f"{analysis_time:.2f}")
            
            return result
            
        except Exception as e:
            logger.error("Pre-mortem analysis failed", 
                        decision_id=context.decision_id,
                        error=str(e))
            
            # Return error result
            result = InterceptionResult(
                decision_id=context.decision_id,
                status=InterceptionStatus.ERROR,
                error_message=str(e),
                recommendation="NO_GO"
            )
            
            self._store_result(result)
            self.interception_stats['analysis_errors'] += 1
            
            return result
        
        finally:
            # Cleanup
            with self.decisions_lock:
                self.pending_decisions.pop(context.decision_id, None)
            
            with self.analysis_futures_lock:
                self.active_analyses.pop(context.decision_id, None)
    
    def _wait_for_analysis_result(self, context: DecisionContext) -> InterceptionResult:
        """Wait for analysis result with timeout"""
        try:
            with self.analysis_futures_lock:
                future = self.active_analyses.get(context.decision_id)
            
            if future is None:
                raise RuntimeError("Analysis future not found")
            
            # Wait for completion with timeout
            result = future.result(timeout=self.config.analysis_timeout_seconds)
            return result
            
        except TimeoutError:
            logger.warning("Analysis timeout", 
                          decision_id=context.decision_id,
                          timeout=self.config.analysis_timeout_seconds)
            
            # Return timeout result
            result = InterceptionResult(
                decision_id=context.decision_id,
                status=InterceptionStatus.TIMEOUT,
                recommendation="CAUTION"
            )
            
            self._store_result(result)
            self.interception_stats['analysis_timeouts'] += 1
            
            return result
        
        except Exception as e:
            logger.error("Error waiting for analysis result",
                        decision_id=context.decision_id,
                        error=str(e))
            
            result = InterceptionResult(
                decision_id=context.decision_id,
                status=InterceptionStatus.ERROR,
                error_message=str(e),
                recommendation="NO_GO"
            )
            
            self._store_result(result)
            return result
    
    def _store_result(self, result: InterceptionResult) -> None:
        """Store analysis result"""
        with self.results_lock:
            self.completed_analyses[result.decision_id] = result
        
        # Publish result event
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.RISK_UPDATE,
                {
                    'type': 'premortem_result',
                    'decision_id': result.decision_id,
                    'status': result.status.value,
                    'recommendation': result.recommendation,
                    'requires_human_review': result.requires_human_review
                },
                'decision_interceptor'
            )
        )
    
    def get_result(self, decision_id: str) -> Optional[InterceptionResult]:
        """Get analysis result for a decision"""
        with self.results_lock:
            return self.completed_analyses.get(decision_id)
    
    def get_pending_decisions(self) -> List[DecisionContext]:
        """Get list of pending decisions"""
        with self.decisions_lock:
            return list(self.pending_decisions.values())
    
    def get_interception_stats(self) -> Dict[str, Any]:
        """Get interception statistics"""
        stats = self.interception_stats.copy()
        stats.update({
            'queue_size': self.decision_queue.qsize(),
            'pending_analyses': len(self.pending_decisions),
            'completed_analyses': len(self.completed_analyses),
            'active_analyses': len(self.active_analyses)
        })
        return stats
    
    def enable_crisis_mode(self) -> None:
        """Enable crisis mode (bypass all non-critical decisions)"""
        self.config.crisis_mode_enabled = True
        logger.warning("Crisis mode ENABLED - most decisions will bypass analysis")
    
    def disable_crisis_mode(self) -> None:
        """Disable crisis mode"""
        self.config.crisis_mode_enabled = False
        logger.info("Crisis mode DISABLED - normal analysis resumed")
    
    def shutdown(self) -> None:
        """Shutdown the decision interceptor"""
        logger.info("Shutting down decision interceptor")
        
        # Cancel pending analyses
        with self.analysis_futures_lock:
            for future in self.active_analyses.values():
                future.cancel()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Decision interceptor shutdown complete")