"""
Routing Decision Analytics and Logging Infrastructure
====================================================

Comprehensive analytics and logging system for routing decisions from the 5th MARL agent.
Tracks routing patterns, broker selection efficiency, and provides insights for optimization.

Features:
- Real-time routing decision logging
- Broker selection pattern analysis
- Performance correlation analysis
- Decision tree analysis for routing logic
- A/B testing framework for routing strategies
- Routing efficiency metrics and reporting

Author: Agent 1 - The Arbitrageur Implementation  
Date: 2025-07-13
"""

import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from enum import Enum
import structlog
from statistics import mean, median, mode, stdev
import pandas as pd

from src.core.events import EventBus, Event, EventType
from src.execution.agents.routing_agent import RoutingAction, QoEMetrics

logger = structlog.get_logger()


class RoutingPattern(Enum):
    """Common routing patterns"""
    STICKY_BROKER = "STICKY_BROKER"  # Consistently uses same broker
    ROUND_ROBIN = "ROUND_ROBIN"      # Rotates between brokers
    PERFORMANCE_BASED = "PERFORMANCE_BASED"  # Routes based on performance
    RANDOM = "RANDOM"                # Random selection
    MARKET_ADAPTIVE = "MARKET_ADAPTIVE"  # Changes with market conditions
    TIME_BASED = "TIME_BASED"        # Changes with time of day


@dataclass
class RoutingDecisionLog:
    """Individual routing decision log entry"""
    
    # Decision metadata
    decision_id: str
    timestamp: datetime
    agent_version: str = "v1.0"
    
    # Order context
    symbol: str = ""
    order_size: int = 0
    order_value: float = 0.0
    order_urgency: float = 0.0
    order_side: str = ""  # BUY/SELL
    order_type: str = ""  # MARKET/LIMIT
    
    # Market context
    market_volatility: float = 0.0
    market_stress: float = 0.0
    spread_bps: float = 0.0
    volume_ratio: float = 0.0
    time_of_day: float = 0.0
    
    # Broker options and performance
    available_brokers: List[str] = field(default_factory=list)
    broker_latencies: Dict[str, float] = field(default_factory=dict)
    broker_fill_rates: Dict[str, float] = field(default_factory=dict)
    broker_costs: Dict[str, float] = field(default_factory=dict)
    broker_qoe_scores: Dict[str, float] = field(default_factory=dict)
    
    # Routing decision
    selected_broker: str = ""
    confidence: float = 0.0
    expected_qoe: float = 0.0
    decision_latency_us: float = 0.0
    
    # Decision reasoning
    routing_strategy: str = ""
    primary_factors: List[str] = field(default_factory=list)
    reasoning: str = ""
    
    # Neural network state
    state_vector_hash: str = ""  # Hash of input state vector
    action_probabilities: Dict[str, float] = field(default_factory=dict)
    exploration_factor: float = 0.0
    
    # Outcome (populated later)
    actual_qoe: Optional[float] = None
    actual_latency_ms: Optional[float] = None
    actual_slippage_bps: Optional[float] = None
    actual_fill_rate: Optional[float] = None
    prediction_error: Optional[float] = None


@dataclass
class RoutingSessionAnalytics:
    """Analytics for a routing session"""
    
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Session metrics
    total_decisions: int = 0
    total_orders_routed: int = 0
    total_value_routed: float = 0.0
    
    # Broker usage
    broker_usage_count: Dict[str, int] = field(default_factory=dict)
    broker_usage_percentage: Dict[str, float] = field(default_factory=dict)
    broker_value_percentage: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    avg_decision_latency_us: float = 0.0
    avg_confidence: float = 0.0
    avg_expected_qoe: float = 0.0
    avg_actual_qoe: Optional[float] = None
    
    # Prediction accuracy
    qoe_prediction_accuracy: Optional[float] = None
    latency_prediction_accuracy: Optional[float] = None
    
    # Pattern analysis
    dominant_pattern: Optional[RoutingPattern] = None
    pattern_confidence: float = 0.0
    
    # Market adaptation
    volatility_adaptation_score: float = 0.0
    stress_adaptation_score: float = 0.0
    
    # Strategy effectiveness
    strategy_performance: Dict[str, float] = field(default_factory=dict)


class RoutingAnalytics:
    """
    Comprehensive routing analytics and logging system
    
    Provides detailed analysis of routing decisions, broker selection patterns,
    and performance optimization insights.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        
        # Configuration
        self.log_retention_days = config.get('log_retention_days', 30)
        self.analytics_window_hours = config.get('analytics_window_hours', 24)
        self.enable_detailed_logging = config.get('enable_detailed_logging', True)
        self.enable_performance_tracking = config.get('enable_performance_tracking', True)
        
        # Data storage
        self.decision_logs: deque = deque(maxlen=100000)  # Recent decisions
        self.session_analytics: Dict[str, RoutingSessionAnalytics] = {}
        self.current_session_id = f"session_{int(time.time())}"
        
        # Real-time tracking
        self.broker_selection_history: deque = deque(maxlen=1000)
        self.pattern_detection_window: deque = deque(maxlen=100)
        self.performance_correlation_data: deque = deque(maxlen=5000)
        
        # Analytics state
        self.routing_patterns: Dict[RoutingPattern, int] = defaultdict(int)
        self.broker_performance_correlation: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.feature_importance: Dict[str, float] = {}
        
        # Background tasks
        self.analytics_task: Optional[asyncio.Task] = None
        self.pattern_analysis_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.total_decisions_logged: int = 0
        self.analytics_calculation_times: deque = deque(maxlen=1000)
        
        logger.info("RoutingAnalytics initialized",
                   retention_days=self.log_retention_days,
                   analytics_window_hours=self.analytics_window_hours,
                   detailed_logging=self.enable_detailed_logging)
        
        # Initialize current session
        self._initialize_current_session()
        
        # Start background processing
        asyncio.create_task(self._start_background_tasks())
    
    def _initialize_current_session(self):
        """Initialize current routing session"""
        self.session_analytics[self.current_session_id] = RoutingSessionAnalytics(
            session_id=self.current_session_id,
            start_time=datetime.now()
        )
    
    async def _start_background_tasks(self):
        """Start background analytics tasks"""
        self.analytics_task = asyncio.create_task(self._analytics_loop())
        self.pattern_analysis_task = asyncio.create_task(self._pattern_analysis_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _analytics_loop(self):
        """Background analytics processing loop"""
        while True:
            try:
                await self._update_analytics()
                await asyncio.sleep(60)  # Update every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Analytics loop error", error=str(e))
                await asyncio.sleep(60)
    
    async def _pattern_analysis_loop(self):
        """Background pattern analysis loop"""
        while True:
            try:
                await self._analyze_routing_patterns()
                await asyncio.sleep(300)  # Analyze every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Pattern analysis loop error", error=str(e))
                await asyncio.sleep(300)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(3600)
    
    def log_routing_decision(self, 
                           routing_action: RoutingAction,
                           decision_context: Dict[str, Any],
                           broker_states: Dict[str, Dict[str, float]]) -> RoutingDecisionLog:
        """
        Log a routing decision with full context
        
        Args:
            routing_action: The routing action taken
            decision_context: Context data for the decision
            broker_states: Current state of all brokers
            
        Returns:
            RoutingDecisionLog entry
        """
        start_time = time.perf_counter()
        
        try:
            # Create decision log
            decision_log = RoutingDecisionLog(
                decision_id=f"decision_{int(time.time() * 1000000)}",
                timestamp=routing_action.timestamp,
                
                # Order context
                symbol=decision_context.get('symbol', ''),
                order_size=decision_context.get('order_size', 0),
                order_value=decision_context.get('order_value', 0.0),
                order_urgency=decision_context.get('order_urgency', 0.0),
                order_side=decision_context.get('order_side', ''),
                order_type=decision_context.get('order_type', ''),
                
                # Market context
                market_volatility=decision_context.get('market_volatility', 0.0),
                market_stress=decision_context.get('market_stress', 0.0),
                spread_bps=decision_context.get('spread_bps', 0.0),
                volume_ratio=decision_context.get('volume_ratio', 1.0),
                time_of_day=decision_context.get('time_of_day', 0.5),
                
                # Broker states
                available_brokers=list(broker_states.keys()),
                broker_latencies={k: v.get('latency', 0.0) for k, v in broker_states.items()},
                broker_fill_rates={k: v.get('fill_rate', 0.0) for k, v in broker_states.items()},
                broker_costs={k: v.get('cost', 0.0) for k, v in broker_states.items()},
                broker_qoe_scores={k: v.get('qoe_score', 0.0) for k, v in broker_states.items()},
                
                # Routing decision
                selected_broker=routing_action.broker_id,
                confidence=routing_action.confidence,
                expected_qoe=routing_action.expected_qoe,
                decision_latency_us=decision_context.get('decision_latency_us', 0.0),
                
                # Decision reasoning
                routing_strategy=decision_context.get('routing_strategy', ''),
                primary_factors=decision_context.get('primary_factors', []),
                reasoning=routing_action.reasoning,
                
                # Neural network state
                state_vector_hash=decision_context.get('state_vector_hash', ''),
                action_probabilities=decision_context.get('action_probabilities', {}),
                exploration_factor=decision_context.get('exploration_factor', 0.0)
            )
            
            # Store decision log
            self.decision_logs.append(decision_log)
            self.total_decisions_logged += 1
            
            # Update session analytics
            self._update_session_analytics(decision_log)
            
            # Track broker selection
            self.broker_selection_history.append({
                'timestamp': decision_log.timestamp,
                'broker': decision_log.selected_broker,
                'confidence': decision_log.confidence,
                'context': decision_context
            })
            
            # Track for pattern detection
            self.pattern_detection_window.append(decision_log.selected_broker)
            
            # Log calculation time
            calc_time = (time.perf_counter() - start_time) * 1000
            self.analytics_calculation_times.append(calc_time)
            
            # Emit analytics event
            if self.enable_detailed_logging:
                self.event_bus.emit(Event(
                    type=EventType.EXECUTION_COMPLETE,
                    data={
                        'routing_decision_logged': True,
                        'decision_id': decision_log.decision_id,
                        'selected_broker': decision_log.selected_broker,
                        'confidence': decision_log.confidence,
                        'expected_qoe': decision_log.expected_qoe
                    }
                ))
            
            logger.debug("Routing decision logged",
                        decision_id=decision_log.decision_id,
                        broker=decision_log.selected_broker,
                        confidence=decision_log.confidence,
                        calc_time_ms=calc_time)
            
            return decision_log
            
        except Exception as e:
            logger.error("Failed to log routing decision", error=str(e))
            raise
    
    def update_decision_outcome(self, decision_id: str, qoe_metrics: QoEMetrics):
        """Update decision log with actual outcome"""
        
        # Find the decision log
        decision_log = None
        for log_entry in reversed(self.decision_logs):
            if log_entry.decision_id == decision_id:
                decision_log = log_entry
                break
        
        if not decision_log:
            logger.warning("Decision log not found for outcome update", decision_id=decision_id)
            return
        
        # Update with actual outcomes
        decision_log.actual_qoe = qoe_metrics.qoe_score
        decision_log.actual_latency_ms = qoe_metrics.latency_ms
        decision_log.actual_slippage_bps = qoe_metrics.slippage_bps
        decision_log.actual_fill_rate = qoe_metrics.fill_rate
        
        # Calculate prediction error
        if decision_log.expected_qoe > 0:
            decision_log.prediction_error = abs(decision_log.expected_qoe - qoe_metrics.qoe_score)
        
        # Store for correlation analysis
        self.performance_correlation_data.append({
            'decision_log': decision_log,
            'qoe_metrics': qoe_metrics,
            'timestamp': datetime.now()
        })
        
        logger.debug("Decision outcome updated",
                    decision_id=decision_id,
                    expected_qoe=decision_log.expected_qoe,
                    actual_qoe=qoe_metrics.qoe_score,
                    prediction_error=decision_log.prediction_error)
    
    def _update_session_analytics(self, decision_log: RoutingDecisionLog):
        """Update current session analytics"""
        
        session = self.session_analytics[self.current_session_id]
        
        # Update counts
        session.total_decisions += 1
        session.total_orders_routed += 1
        session.total_value_routed += decision_log.order_value
        
        # Update broker usage
        broker = decision_log.selected_broker
        session.broker_usage_count[broker] = session.broker_usage_count.get(broker, 0) + 1
        
        # Update averages
        all_decisions = [log for log in self.decision_logs 
                        if log.timestamp >= session.start_time]
        
        if all_decisions:
            session.avg_decision_latency_us = mean(log.decision_latency_us for log in all_decisions)
            session.avg_confidence = mean(log.confidence for log in all_decisions)
            session.avg_expected_qoe = mean(log.expected_qoe for log in all_decisions)
            
            # Calculate broker usage percentages
            total_usage = sum(session.broker_usage_count.values())
            for broker, count in session.broker_usage_count.items():
                session.broker_usage_percentage[broker] = count / total_usage * 100
    
    async def _update_analytics(self):
        """Update comprehensive analytics"""
        
        # Update feature importance
        await self._calculate_feature_importance()
        
        # Update broker performance correlations
        await self._calculate_performance_correlations()
        
        # Update session analytics with outcomes
        await self._update_session_outcomes()
    
    async def _calculate_feature_importance(self):
        """Calculate feature importance for routing decisions"""
        
        recent_decisions = [log for log in self.decision_logs 
                          if log.timestamp > datetime.now() - timedelta(hours=self.analytics_window_hours)]
        
        if len(recent_decisions) < 10:
            return
        
        # Simple correlation-based feature importance
        features = {
            'market_volatility': [log.market_volatility for log in recent_decisions],
            'market_stress': [log.market_stress for log in recent_decisions],
            'order_urgency': [log.order_urgency for log in recent_decisions],
            'order_size': [log.order_size for log in recent_decisions],
            'spread_bps': [log.spread_bps for log in recent_decisions],
            'time_of_day': [log.time_of_day for log in recent_decisions]
        }
        
        # Calculate correlation with broker selection outcome
        broker_scores = []
        for log in recent_decisions:
            if log.selected_broker in log.broker_qoe_scores:
                broker_scores.append(log.broker_qoe_scores[log.selected_broker])
            else:
                broker_scores.append(0.5)  # Default
        
        if len(set(broker_scores)) > 1:  # Need variance to calculate correlation
            for feature_name, feature_values in features.items():
                if len(set(feature_values)) > 1:
                    correlation = np.corrcoef(feature_values, broker_scores)[0, 1]
                    self.feature_importance[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
    
    async def _calculate_performance_correlations(self):
        """Calculate correlations between broker features and performance"""
        
        correlation_data = list(self.performance_correlation_data)
        
        if len(correlation_data) < 20:
            return
        
        # Group by broker
        broker_data = defaultdict(list)
        for data in correlation_data:
            broker = data['decision_log'].selected_broker
            broker_data[broker].append(data)
        
        # Calculate correlations for each broker
        for broker, data_points in broker_data.items():
            if len(data_points) < 10:
                continue
            
            # Extract features and outcomes
            latencies = [dp['decision_log'].actual_latency_ms for dp in data_points if dp['decision_log'].actual_latency_ms is not None]
            qoe_scores = [dp['qoe_metrics'].qoe_score for dp in data_points]
            market_stress = [dp['decision_log'].market_stress for dp in data_points]
            order_sizes = [dp['decision_log'].order_size for dp in data_points]
            
            # Calculate correlations
            correlations = {}
            if len(latencies) > 5 and len(set(latencies)) > 1:
                correlations['latency_qoe'] = np.corrcoef(latencies, qoe_scores[:len(latencies)])[0, 1]
            
            if len(set(market_stress)) > 1:
                correlations['stress_qoe'] = np.corrcoef(market_stress, qoe_scores)[0, 1]
            
            if len(set(order_sizes)) > 1:
                correlations['size_qoe'] = np.corrcoef(order_sizes, qoe_scores)[0, 1]
            
            # Store correlations
            for corr_name, corr_value in correlations.items():
                if not np.isnan(corr_value):
                    self.broker_performance_correlation[broker][corr_name] = corr_value
    
    async def _update_session_outcomes(self):
        """Update session analytics with actual outcomes"""
        
        session = self.session_analytics[self.current_session_id]
        
        # Find decisions with outcomes
        decisions_with_outcomes = [
            log for log in self.decision_logs 
            if log.timestamp >= session.start_time and log.actual_qoe is not None
        ]
        
        if decisions_with_outcomes:
            session.avg_actual_qoe = mean(log.actual_qoe for log in decisions_with_outcomes)
            
            # Calculate prediction accuracy
            prediction_errors = [log.prediction_error for log in decisions_with_outcomes 
                               if log.prediction_error is not None]
            
            if prediction_errors:
                avg_error = mean(prediction_errors)
                session.qoe_prediction_accuracy = max(0.0, 1.0 - avg_error)
    
    async def _analyze_routing_patterns(self):
        """Analyze routing patterns from recent decisions"""
        
        if len(self.pattern_detection_window) < 20:
            return
        
        recent_brokers = list(self.pattern_detection_window)
        
        # Detect patterns
        pattern_scores = {
            RoutingPattern.STICKY_BROKER: self._calculate_stickiness_score(recent_brokers),
            RoutingPattern.ROUND_ROBIN: self._calculate_round_robin_score(recent_brokers),
            RoutingPattern.RANDOM: self._calculate_randomness_score(recent_brokers),
        }
        
        # Find dominant pattern
        dominant_pattern = max(pattern_scores, key=pattern_scores.get)
        pattern_confidence = pattern_scores[dominant_pattern]
        
        # Update session analytics
        session = self.session_analytics[self.current_session_id]
        session.dominant_pattern = dominant_pattern
        session.pattern_confidence = pattern_confidence
        
        # Update pattern counts
        self.routing_patterns[dominant_pattern] += 1
    
    def _calculate_stickiness_score(self, broker_sequence: List[str]) -> float:
        """Calculate how sticky (consistent) the broker selection is"""
        if not broker_sequence:
            return 0.0
        
        # Count consecutive same-broker selections
        consecutive_runs = []
        current_run = 1
        
        for i in range(1, len(broker_sequence)):
            if broker_sequence[i] == broker_sequence[i-1]:
                current_run += 1
            else:
                consecutive_runs.append(current_run)
                current_run = 1
        consecutive_runs.append(current_run)
        
        # Higher score for longer consecutive runs
        avg_run_length = mean(consecutive_runs)
        return min(1.0, avg_run_length / 5.0)  # Normalize to 0-1
    
    def _calculate_round_robin_score(self, broker_sequence: List[str]) -> float:
        """Calculate how round-robin the broker selection is"""
        if len(broker_sequence) < 6:
            return 0.0
        
        unique_brokers = list(set(broker_sequence))
        if len(unique_brokers) < 2:
            return 0.0
        
        # Check for round-robin pattern
        pattern_matches = 0
        window_size = len(unique_brokers)
        
        for i in range(len(broker_sequence) - window_size + 1):
            window = broker_sequence[i:i + window_size]
            if len(set(window)) == len(unique_brokers):  # All brokers used once
                pattern_matches += 1
        
        max_possible_matches = len(broker_sequence) - window_size + 1
        return pattern_matches / max_possible_matches if max_possible_matches > 0 else 0.0
    
    def _calculate_randomness_score(self, broker_sequence: List[str]) -> float:
        """Calculate how random the broker selection is"""
        if not broker_sequence:
            return 0.0
        
        # Calculate entropy
        broker_counts = Counter(broker_sequence)
        total_selections = len(broker_sequence)
        
        entropy = 0.0
        for count in broker_counts.values():
            probability = count / total_selections
            entropy -= probability * np.log2(probability)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(broker_counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    async def _cleanup_old_data(self):
        """Clean up old data beyond retention period"""
        
        cutoff_time = datetime.now() - timedelta(days=self.log_retention_days)
        
        # Clean decision logs
        filtered_logs = deque()
        for log in self.decision_logs:
            if log.timestamp > cutoff_time:
                filtered_logs.append(log)
        self.decision_logs = filtered_logs
        
        # Clean performance correlation data
        filtered_correlation_data = deque()
        for data in self.performance_correlation_data:
            if data['timestamp'] > cutoff_time:
                filtered_correlation_data.append(data)
        self.performance_correlation_data = filtered_correlation_data
        
        # Clean old sessions
        sessions_to_remove = []
        for session_id, session in self.session_analytics.items():
            if session.start_time < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.session_analytics[session_id]
        
        logger.debug("Cleaned up old analytics data",
                    logs_remaining=len(self.decision_logs),
                    correlation_data_remaining=len(self.performance_correlation_data),
                    sessions_remaining=len(self.session_analytics))
    
    def get_routing_analytics_summary(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive routing analytics summary"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
        recent_decisions = [log for log in self.decision_logs if log.timestamp > cutoff_time]
        
        if not recent_decisions:
            return {'error': 'No routing decisions in specified time period'}
        
        # Basic statistics
        summary = {
            'time_period_hours': time_period_hours,
            'total_decisions': len(recent_decisions),
            'unique_brokers_used': len(set(log.selected_broker for log in recent_decisions)),
            
            # Decision quality
            'avg_confidence': mean(log.confidence for log in recent_decisions),
            'avg_expected_qoe': mean(log.expected_qoe for log in recent_decisions),
            'avg_decision_latency_us': mean(log.decision_latency_us for log in recent_decisions),
            
            # Broker usage distribution
            'broker_usage': dict(Counter(log.selected_broker for log in recent_decisions)),
            
            # Pattern analysis
            'dominant_patterns': dict(self.routing_patterns),
            'feature_importance': self.feature_importance,
            
            # Performance correlations
            'broker_correlations': dict(self.broker_performance_correlation),
            
            # Current session
            'current_session': asdict(self.session_analytics[self.current_session_id]) if self.current_session_id in self.session_analytics else None
        }
        
        # Add outcome analysis if available
        decisions_with_outcomes = [log for log in recent_decisions if log.actual_qoe is not None]
        
        if decisions_with_outcomes:
            summary['outcome_analysis'] = {
                'decisions_with_outcomes': len(decisions_with_outcomes),
                'avg_actual_qoe': mean(log.actual_qoe for log in decisions_with_outcomes),
                'avg_prediction_error': mean(log.prediction_error for log in decisions_with_outcomes if log.prediction_error is not None),
                'qoe_prediction_accuracy': summary['current_session']['qoe_prediction_accuracy'] if summary['current_session'] else None
            }
        
        return summary
    
    def get_broker_analytics(self, broker_id: str, time_period_hours: int = 24) -> Dict[str, Any]:
        """Get analytics for specific broker"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
        broker_decisions = [
            log for log in self.decision_logs 
            if log.timestamp > cutoff_time and log.selected_broker == broker_id
        ]
        
        if not broker_decisions:
            return {'error': f'No decisions for broker {broker_id} in specified time period'}
        
        # Broker-specific analytics
        analytics = {
            'broker_id': broker_id,
            'time_period_hours': time_period_hours,
            'total_decisions': len(broker_decisions),
            'usage_percentage': len(broker_decisions) / len([log for log in self.decision_logs if log.timestamp > cutoff_time]) * 100,
            
            # Quality metrics
            'avg_confidence': mean(log.confidence for log in broker_decisions),
            'avg_expected_qoe': mean(log.expected_qoe for log in broker_decisions),
            'avg_decision_latency_us': mean(log.decision_latency_us for log in broker_decisions),
            
            # Market context analysis
            'avg_market_stress_when_selected': mean(log.market_stress for log in broker_decisions),
            'avg_order_urgency_when_selected': mean(log.order_urgency for log in broker_decisions),
            'avg_order_size_when_selected': mean(log.order_size for log in broker_decisions),
            
            # Performance correlations
            'performance_correlations': self.broker_performance_correlation.get(broker_id, {})
        }
        
        # Add outcome analysis if available
        decisions_with_outcomes = [log for log in broker_decisions if log.actual_qoe is not None]
        
        if decisions_with_outcomes:
            analytics['outcome_analysis'] = {
                'decisions_with_outcomes': len(decisions_with_outcomes),
                'avg_actual_qoe': mean(log.actual_qoe for log in decisions_with_outcomes),
                'avg_prediction_error': mean(log.prediction_error for log in decisions_with_outcomes if log.prediction_error is not None),
                'qoe_improvement_vs_expected': mean(
                    log.actual_qoe - log.expected_qoe for log in decisions_with_outcomes
                )
            }
        
        return analytics
    
    def export_analytics_data(self, format: str = "json", time_period_hours: int = 24) -> Dict[str, Any]:
        """Export analytics data for external analysis"""
        
        cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
        recent_decisions = [log for log in self.decision_logs if log.timestamp > cutoff_time]
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_period_hours': time_period_hours,
            'total_decisions': len(recent_decisions),
            
            'decision_logs': [asdict(log) for log in recent_decisions],
            'session_analytics': {k: asdict(v) for k, v in self.session_analytics.items()},
            'performance_correlations': dict(self.broker_performance_correlation),
            'feature_importance': self.feature_importance,
            'routing_patterns': {k.value: v for k, v in self.routing_patterns.items()},
            
            'system_metrics': {
                'total_decisions_logged': self.total_decisions_logged,
                'avg_analytics_calc_time_ms': mean(self.analytics_calculation_times) if self.analytics_calculation_times else 0.0,
                'current_session_id': self.current_session_id
            }
        }
        
        return export_data
    
    async def shutdown(self):
        """Shutdown routing analytics"""
        logger.info("Shutting down RoutingAnalytics")
        
        # Cancel background tasks
        for task in [self.analytics_task, self.pattern_analysis_task, self.cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Export final analytics
        final_summary = self.get_routing_analytics_summary()
        logger.info("Final routing analytics summary", **{k: v for k, v in final_summary.items() if k != 'error'})
        
        logger.info("RoutingAnalytics shutdown complete")


def create_routing_analytics(config: Dict[str, Any], event_bus: EventBus) -> RoutingAnalytics:
    """Factory function to create routing analytics"""
    return RoutingAnalytics(config, event_bus)


# Default configuration
DEFAULT_ROUTING_ANALYTICS_CONFIG = {
    'log_retention_days': 30,
    'analytics_window_hours': 24,
    'enable_detailed_logging': True,
    'enable_performance_tracking': True
}