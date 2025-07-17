"""
Broker Performance Tracker
==========================

Real-time broker performance monitoring and analytics system for the Routing Agent.
Tracks execution quality metrics, performance trends, and provides state vectors for MARL.

Features:
- Real-time QoE (Quality of Execution) monitoring
- Multi-dimensional performance tracking (latency, fill rates, costs, reliability)
- Historical performance analysis and trend detection
- State vector generation for neural network input
- Performance alerts and degradation detection

Author: Agent 1 - The Arbitrageur Implementation
Date: 2025-07-13
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import structlog
import json
from statistics import mean, median, stdev

from .base_broker import BrokerExecution, BrokerOrder
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


class PerformanceAlert(Enum):
    """Performance alert types"""
    LATENCY_SPIKE = "LATENCY_SPIKE"
    FILL_RATE_DROP = "FILL_RATE_DROP"
    HIGH_SLIPPAGE = "HIGH_SLIPPAGE"
    HIGH_COMMISSION = "HIGH_COMMISSION"
    CONNECTION_ISSUE = "CONNECTION_ISSUE"
    QUALITY_DEGRADATION = "QUALITY_DEGRADATION"


@dataclass
class ExecutionRecord:
    """Individual execution record for tracking"""
    
    execution_id: str
    broker_id: str
    symbol: str
    side: str
    quantity: int
    
    # Timing
    order_submitted_at: datetime
    execution_time: datetime
    latency_ms: float
    
    # Execution quality
    fill_rate: float  # Filled quantity / Total quantity
    slippage_bps: float
    commission_per_share: float
    fees: float
    
    # Market context
    market_price: float
    execution_price: float
    spread_at_execution: float
    volume_at_execution: float
    
    # Quality metrics
    price_improvement_bps: float = 0.0
    market_impact_bps: float = 0.0
    
    # Metadata
    venue: str = ""
    algorithm_used: str = ""
    order_type: str = ""
    
    def calculate_qoe_score(self) -> float:
        """Calculate Quality of Execution score"""
        
        # Component scores (0-1, higher is better)
        fill_score = self.fill_rate
        
        # Slippage score (lower slippage is better)
        slippage_score = max(0.0, 1.0 - (abs(self.slippage_bps) / 50.0))
        
        # Commission score (lower commission is better)
        commission_score = max(0.0, 1.0 - (self.commission_per_share / 0.10))
        
        # Latency score (lower latency is better)
        latency_score = max(0.0, 1.0 - (self.latency_ms / 1000.0))
        
        # Price improvement bonus
        improvement_bonus = min(0.2, max(0.0, self.price_improvement_bps / 50.0))
        
        # Weighted QoE score
        qoe_score = (
            fill_score * 0.30 +
            slippage_score * 0.25 +
            commission_score * 0.20 +
            latency_score * 0.25 +
            improvement_bonus
        )
        
        return min(1.0, qoe_score)


@dataclass
class BrokerPerformanceSummary:
    """Comprehensive broker performance summary"""
    
    broker_id: str
    measurement_period_start: datetime
    measurement_period_end: datetime
    
    # Volume metrics
    total_executions: int = 0
    total_quantity: int = 0
    total_notional: float = 0.0
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Fill rate metrics
    overall_fill_rate: float = 0.0
    avg_fill_rate_per_order: float = 0.0
    
    # Cost metrics
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    std_slippage_bps: float = 0.0
    avg_commission_per_share: float = 0.0
    total_commission: float = 0.0
    
    # Quality metrics
    avg_qoe_score: float = 0.0
    median_qoe_score: float = 0.0
    min_qoe_score: float = 0.0
    max_qoe_score: float = 0.0
    
    # Market impact
    avg_market_impact_bps: float = 0.0
    avg_price_improvement_bps: float = 0.0
    
    # Reliability metrics
    connection_uptime_pct: float = 0.0
    error_rate_pct: float = 0.0
    
    # Performance trends
    latency_trend: str = "STABLE"  # IMPROVING, DEGRADING, STABLE
    fill_rate_trend: str = "STABLE"
    slippage_trend: str = "STABLE"
    qoe_trend: str = "STABLE"
    
    # Composite scores
    performance_score: float = 0.0  # Overall performance (0-1)
    reliability_score: float = 0.0  # Reliability score (0-1)
    cost_efficiency_score: float = 0.0  # Cost efficiency (0-1)


class BrokerPerformanceTracker:
    """
    Real-time broker performance tracking and analytics system
    
    Provides comprehensive monitoring of broker execution quality,
    performance trends, and state vectors for MARL training.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        
        # Configuration
        self.tracking_window_hours = config.get('tracking_window_hours', 24)
        self.max_records_per_broker = config.get('max_records_per_broker', 10000)
        self.alert_thresholds = config.get('alert_thresholds', {
            'latency_spike_factor': 2.0,
            'fill_rate_drop_threshold': 0.05,
            'high_slippage_bps': 20.0,
            'high_commission_threshold': 0.05,
            'qoe_degradation_threshold': 0.1
        })
        
        # Data storage
        self.execution_records: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_records_per_broker)
        )
        self.connection_events: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
        # Performance summaries (cached)
        self.performance_summaries: Dict[str, BrokerPerformanceSummary] = {}
        self.last_summary_update: Dict[str, datetime] = {}
        self.summary_update_interval = timedelta(seconds=30)
        
        # Alert tracking
        self.recent_alerts: deque = deque(maxlen=100)
        self.alert_counts: Dict[str, Dict[PerformanceAlert, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_task: Optional[asyncio.Task] = None
        
        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        logger.info("BrokerPerformanceTracker initialized",
                   tracking_window_hours=self.tracking_window_hours,
                   max_records_per_broker=self.max_records_per_broker)
        
        # Start background monitoring
        asyncio.create_task(self._start_monitoring())
    
    async def _start_monitoring(self):
        """Start background monitoring tasks"""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.alert_task = asyncio.create_task(self._alert_monitoring_loop())
    
    async def _monitoring_loop(self):
        """Background performance monitoring loop"""
        while True:
            try:
                # Update performance summaries
                await self._update_performance_summaries()
                
                # Update baselines
                self._update_performance_baselines()
                
                # Clean old records
                self._clean_old_records()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _alert_monitoring_loop(self):
        """Background alert monitoring loop"""
        while True:
            try:
                # Check for performance alerts
                await self._check_performance_alerts()
                
                await asyncio.sleep(10)  # Check alerts every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert monitoring loop error", error=str(e))
                await asyncio.sleep(10)
    
    def record_execution(self, broker_id: str, execution: BrokerExecution,
                        order_submitted_at: datetime, market_context: Dict[str, Any]) -> ExecutionRecord:
        """Record a new execution for performance tracking"""
        
        # Calculate metrics
        latency_ms = (execution.timestamp - order_submitted_at).total_seconds() * 1000
        
        # Create execution record
        execution_record = ExecutionRecord(
            execution_id=execution.execution_id,
            broker_id=broker_id,
            symbol=execution.symbol,
            side=execution.side,
            quantity=execution.quantity,
            order_submitted_at=order_submitted_at,
            execution_time=execution.timestamp,
            latency_ms=latency_ms,
            fill_rate=1.0,  # Assume fully filled if we have execution
            slippage_bps=market_context.get('slippage_bps', 0.0),
            commission_per_share=execution.commission / max(1, execution.quantity),
            fees=execution.fees,
            market_price=market_context.get('market_price', execution.price),
            execution_price=execution.price,
            spread_at_execution=market_context.get('spread_bps', 0.0),
            volume_at_execution=market_context.get('volume', 0.0),
            venue=execution.venue,
            order_type=market_context.get('order_type', ''),
            price_improvement_bps=market_context.get('price_improvement_bps', 0.0),
            market_impact_bps=market_context.get('market_impact_bps', 0.0)
        )
        
        # Store record
        self.execution_records[broker_id].append(execution_record)
        
        # Emit event
        self.event_bus.emit(Event(
            type=EventType.EXECUTION_COMPLETE,
            data={
                'broker_id': broker_id,
                'execution_record': asdict(execution_record),
                'qoe_score': execution_record.calculate_qoe_score()
            }
        ))
        
        logger.debug("Execution recorded",
                    broker=broker_id,
                    symbol=execution.symbol,
                    latency_ms=latency_ms,
                    qoe_score=execution_record.calculate_qoe_score())
        
        return execution_record
    
    def record_connection_event(self, broker_id: str, event_type: str, 
                              timestamp: Optional[datetime] = None, details: Optional[Dict] = None):
        """Record broker connection event"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        connection_event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'details': details or {}
        }
        
        self.connection_events[broker_id].append(connection_event)
        
        logger.debug("Connection event recorded",
                    broker=broker_id,
                    event_type=event_type,
                    timestamp=timestamp)
    
    async def _update_performance_summaries(self):
        """Update performance summaries for all brokers"""
        
        for broker_id in self.execution_records.keys():
            # Check if update needed
            last_update = self.last_summary_update.get(broker_id, datetime.min)
            if datetime.now() - last_update < self.summary_update_interval:
                continue
            
            # Update summary
            summary = self._calculate_performance_summary(broker_id)
            self.performance_summaries[broker_id] = summary
            self.last_summary_update[broker_id] = datetime.now()
    
    def _calculate_performance_summary(self, broker_id: str) -> BrokerPerformanceSummary:
        """Calculate comprehensive performance summary for broker"""
        
        records = list(self.execution_records[broker_id])
        if not records:
            return BrokerPerformanceSummary(
                broker_id=broker_id,
                measurement_period_start=datetime.now(),
                measurement_period_end=datetime.now()
            )
        
        # Filter to time window
        cutoff_time = datetime.now() - timedelta(hours=self.tracking_window_hours)
        recent_records = [r for r in records if r.execution_time > cutoff_time]
        
        if not recent_records:
            return BrokerPerformanceSummary(
                broker_id=broker_id,
                measurement_period_start=cutoff_time,
                measurement_period_end=datetime.now()
            )
        
        # Calculate metrics
        summary = BrokerPerformanceSummary(
            broker_id=broker_id,
            measurement_period_start=min(r.execution_time for r in recent_records),
            measurement_period_end=max(r.execution_time for r in recent_records)
        )
        
        # Volume metrics
        summary.total_executions = len(recent_records)
        summary.total_quantity = sum(r.quantity for r in recent_records)
        summary.total_notional = sum(r.quantity * r.execution_price for r in recent_records)
        
        # Latency metrics
        latencies = [r.latency_ms for r in recent_records]
        summary.avg_latency_ms = mean(latencies)
        summary.median_latency_ms = median(latencies)
        summary.p95_latency_ms = np.percentile(latencies, 95)
        summary.p99_latency_ms = np.percentile(latencies, 99)
        summary.max_latency_ms = max(latencies)
        
        # Fill rate metrics
        fill_rates = [r.fill_rate for r in recent_records]
        summary.overall_fill_rate = mean(fill_rates)
        summary.avg_fill_rate_per_order = mean(fill_rates)
        
        # Cost metrics
        slippages = [r.slippage_bps for r in recent_records]
        commissions = [r.commission_per_share for r in recent_records]
        
        summary.avg_slippage_bps = mean(slippages)
        summary.median_slippage_bps = median(slippages)
        summary.std_slippage_bps = stdev(slippages) if len(slippages) > 1 else 0.0
        summary.avg_commission_per_share = mean(commissions)
        summary.total_commission = sum(r.commission_per_share * r.quantity for r in recent_records)
        
        # Quality metrics
        qoe_scores = [r.calculate_qoe_score() for r in recent_records]
        summary.avg_qoe_score = mean(qoe_scores)
        summary.median_qoe_score = median(qoe_scores)
        summary.min_qoe_score = min(qoe_scores)
        summary.max_qoe_score = max(qoe_scores)
        
        # Market impact
        summary.avg_market_impact_bps = mean(r.market_impact_bps for r in recent_records)
        summary.avg_price_improvement_bps = mean(r.price_improvement_bps for r in recent_records)
        
        # Connection reliability
        connection_events = list(self.connection_events[broker_id])
        recent_connections = [e for e in connection_events 
                            if e['timestamp'] > cutoff_time]
        
        if recent_connections:
            total_time = (datetime.now() - cutoff_time).total_seconds()
            connected_time = sum(
                60.0 for e in recent_connections 
                if e['event_type'] == 'CONNECTED'
            )  # Simplified calculation
            summary.connection_uptime_pct = min(1.0, connected_time / total_time) * 100
            
            error_events = [e for e in recent_connections if 'ERROR' in e['event_type']]
            summary.error_rate_pct = (len(error_events) / len(recent_connections)) * 100
        else:
            summary.connection_uptime_pct = 100.0
            summary.error_rate_pct = 0.0
        
        # Calculate trends
        summary.latency_trend = self._calculate_trend(broker_id, 'latency')
        summary.fill_rate_trend = self._calculate_trend(broker_id, 'fill_rate')
        summary.slippage_trend = self._calculate_trend(broker_id, 'slippage')
        summary.qoe_trend = self._calculate_trend(broker_id, 'qoe')
        
        # Calculate composite scores
        summary.performance_score = self._calculate_performance_score(summary)
        summary.reliability_score = (summary.connection_uptime_pct / 100.0) * (1.0 - summary.error_rate_pct / 100.0)
        summary.cost_efficiency_score = self._calculate_cost_efficiency_score(summary)
        
        return summary
    
    def _calculate_trend(self, broker_id: str, metric: str) -> str:
        """Calculate performance trend for specific metric"""
        
        records = list(self.execution_records[broker_id])
        if len(records) < 10:
            return "STABLE"
        
        # Split into two halves for comparison
        mid_point = len(records) // 2
        first_half = records[:mid_point]
        second_half = records[mid_point:]
        
        if metric == 'latency':
            first_avg = mean(r.latency_ms for r in first_half)
            second_avg = mean(r.latency_ms for r in second_half)
            change_pct = (second_avg - first_avg) / first_avg
            
            if change_pct > 0.2:
                return "DEGRADING"
            elif change_pct < -0.2:
                return "IMPROVING"
            
        elif metric == 'fill_rate':
            first_avg = mean(r.fill_rate for r in first_half)
            second_avg = mean(r.fill_rate for r in second_half)
            change_pct = (second_avg - first_avg) / first_avg
            
            if change_pct > 0.05:
                return "IMPROVING"
            elif change_pct < -0.05:
                return "DEGRADING"
        
        elif metric == 'slippage':
            first_avg = mean(abs(r.slippage_bps) for r in first_half)
            second_avg = mean(abs(r.slippage_bps) for r in second_half)
            change_pct = (second_avg - first_avg) / max(0.1, first_avg)
            
            if change_pct > 0.3:
                return "DEGRADING"
            elif change_pct < -0.3:
                return "IMPROVING"
        
        elif metric == 'qoe':
            first_avg = mean(r.calculate_qoe_score() for r in first_half)
            second_avg = mean(r.calculate_qoe_score() for r in second_half)
            change_pct = (second_avg - first_avg) / first_avg
            
            if change_pct > 0.1:
                return "IMPROVING"
            elif change_pct < -0.1:
                return "DEGRADING"
        
        return "STABLE"
    
    def _calculate_performance_score(self, summary: BrokerPerformanceSummary) -> float:
        """Calculate overall performance score (0-1)"""
        
        # Component scores
        latency_score = max(0.0, 1.0 - (summary.avg_latency_ms / 1000.0))
        fill_rate_score = summary.overall_fill_rate
        slippage_score = max(0.0, 1.0 - (abs(summary.avg_slippage_bps) / 50.0))
        qoe_score = summary.avg_qoe_score
        reliability_score = summary.reliability_score
        
        # Weighted average
        performance_score = (
            latency_score * 0.20 +
            fill_rate_score * 0.25 +
            slippage_score * 0.20 +
            qoe_score * 0.20 +
            reliability_score * 0.15
        )
        
        return min(1.0, performance_score)
    
    def _calculate_cost_efficiency_score(self, summary: BrokerPerformanceSummary) -> float:
        """Calculate cost efficiency score (0-1)"""
        
        # Commission efficiency
        commission_score = max(0.0, 1.0 - (summary.avg_commission_per_share / 0.10))
        
        # Slippage efficiency
        slippage_score = max(0.0, 1.0 - (abs(summary.avg_slippage_bps) / 50.0))
        
        # Price improvement bonus
        improvement_bonus = min(0.2, max(0.0, summary.avg_price_improvement_bps / 20.0))
        
        return min(1.0, (commission_score + slippage_score) / 2.0 + improvement_bonus)
    
    def _update_performance_baselines(self):
        """Update performance baselines for trend detection"""
        
        for broker_id, summary in self.performance_summaries.items():
            if broker_id not in self.performance_baselines:
                self.performance_baselines[broker_id] = {}
            
            baselines = self.performance_baselines[broker_id]
            baselines['avg_latency_ms'] = summary.avg_latency_ms
            baselines['overall_fill_rate'] = summary.overall_fill_rate
            baselines['avg_slippage_bps'] = summary.avg_slippage_bps
            baselines['avg_qoe_score'] = summary.avg_qoe_score
    
    async def _check_performance_alerts(self):
        """Check for performance alerts and emit events"""
        
        for broker_id, summary in self.performance_summaries.items():
            baseline = self.performance_baselines.get(broker_id, {})
            
            # Latency spike alert
            baseline_latency = baseline.get('avg_latency_ms', summary.avg_latency_ms)
            if summary.avg_latency_ms > baseline_latency * self.alert_thresholds['latency_spike_factor']:
                await self._emit_alert(broker_id, PerformanceAlert.LATENCY_SPIKE, {
                    'current_latency': summary.avg_latency_ms,
                    'baseline_latency': baseline_latency,
                    'spike_factor': summary.avg_latency_ms / baseline_latency
                })
            
            # Fill rate drop alert
            baseline_fill_rate = baseline.get('overall_fill_rate', summary.overall_fill_rate)
            fill_rate_drop = baseline_fill_rate - summary.overall_fill_rate
            if fill_rate_drop > self.alert_thresholds['fill_rate_drop_threshold']:
                await self._emit_alert(broker_id, PerformanceAlert.FILL_RATE_DROP, {
                    'current_fill_rate': summary.overall_fill_rate,
                    'baseline_fill_rate': baseline_fill_rate,
                    'drop_amount': fill_rate_drop
                })
            
            # High slippage alert
            if abs(summary.avg_slippage_bps) > self.alert_thresholds['high_slippage_bps']:
                await self._emit_alert(broker_id, PerformanceAlert.HIGH_SLIPPAGE, {
                    'avg_slippage_bps': summary.avg_slippage_bps,
                    'threshold': self.alert_thresholds['high_slippage_bps']
                })
            
            # QoE degradation alert
            baseline_qoe = baseline.get('avg_qoe_score', summary.avg_qoe_score)
            qoe_degradation = baseline_qoe - summary.avg_qoe_score
            if qoe_degradation > self.alert_thresholds['qoe_degradation_threshold']:
                await self._emit_alert(broker_id, PerformanceAlert.QUALITY_DEGRADATION, {
                    'current_qoe': summary.avg_qoe_score,
                    'baseline_qoe': baseline_qoe,
                    'degradation_amount': qoe_degradation
                })
    
    async def _emit_alert(self, broker_id: str, alert_type: PerformanceAlert, details: Dict[str, Any]):
        """Emit performance alert"""
        
        alert_data = {
            'broker_id': broker_id,
            'alert_type': alert_type.value,
            'details': details,
            'timestamp': datetime.now(),
            'severity': self._calculate_alert_severity(alert_type, details)
        }
        
        # Store alert
        self.recent_alerts.append(alert_data)
        self.alert_counts[broker_id][alert_type] += 1
        
        # Emit event
        self.event_bus.emit(Event(
            type=EventType.RISK_ALERT,
            data=alert_data
        ))
        
        logger.warning("Performance alert",
                      broker=broker_id,
                      alert_type=alert_type.value,
                      severity=alert_data['severity'],
                      details=details)
    
    def _calculate_alert_severity(self, alert_type: PerformanceAlert, details: Dict[str, Any]) -> str:
        """Calculate alert severity level"""
        
        if alert_type == PerformanceAlert.LATENCY_SPIKE:
            spike_factor = details.get('spike_factor', 1.0)
            if spike_factor > 5.0:
                return "CRITICAL"
            elif spike_factor > 3.0:
                return "HIGH"
            else:
                return "MEDIUM"
        
        elif alert_type == PerformanceAlert.FILL_RATE_DROP:
            drop_amount = details.get('drop_amount', 0.0)
            if drop_amount > 0.20:
                return "CRITICAL"
            elif drop_amount > 0.10:
                return "HIGH"
            else:
                return "MEDIUM"
        
        elif alert_type == PerformanceAlert.HIGH_SLIPPAGE:
            slippage = abs(details.get('avg_slippage_bps', 0.0))
            if slippage > 50.0:
                return "CRITICAL"
            elif slippage > 30.0:
                return "HIGH"
            else:
                return "MEDIUM"
        
        return "LOW"
    
    def _clean_old_records(self):
        """Clean old execution records outside tracking window"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.tracking_window_hours * 2)  # Keep 2x window
        
        for broker_id, records in self.execution_records.items():
            # Convert to list for filtering
            record_list = list(records)
            filtered_records = [r for r in record_list if r.execution_time > cutoff_time]
            
            # Clear and repopulate
            records.clear()
            records.extend(filtered_records)
        
        # Clean connection events
        for broker_id, events in self.connection_events.items():
            event_list = list(events)
            filtered_events = [e for e in event_list if e['timestamp'] > cutoff_time]
            
            events.clear()
            events.extend(filtered_events)
    
    def get_broker_summary(self, broker_id: str) -> Optional[BrokerPerformanceSummary]:
        """Get performance summary for specific broker"""
        return self.performance_summaries.get(broker_id)
    
    def get_all_summaries(self) -> Dict[str, BrokerPerformanceSummary]:
        """Get all broker performance summaries"""
        return self.performance_summaries.copy()
    
    def get_broker_state_vector(self, broker_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get normalized state vectors for MARL input"""
        
        num_brokers = len(broker_ids)
        latencies = np.zeros(num_brokers)
        fill_rates = np.zeros(num_brokers)
        costs = np.zeros(num_brokers)
        reliabilities = np.zeros(num_brokers)
        availabilities = np.zeros(num_brokers)
        
        for i, broker_id in enumerate(broker_ids):
            summary = self.performance_summaries.get(broker_id)
            
            if summary:
                # Normalize latency (0-1, lower is better -> higher score)
                latencies[i] = max(0.0, 1.0 - (summary.avg_latency_ms / 1000.0))
                
                # Fill rate (already 0-1)
                fill_rates[i] = summary.overall_fill_rate
                
                # Cost efficiency
                costs[i] = summary.cost_efficiency_score
                
                # Reliability
                reliabilities[i] = summary.reliability_score
                
                # Availability (based on recent activity)
                last_update = self.last_summary_update.get(broker_id, datetime.min)
                minutes_since_update = (datetime.now() - last_update).total_seconds() / 60
                availabilities[i] = 1.0 if minutes_since_update < 5 else 0.0
            else:
                # Default values for unknown brokers
                latencies[i] = 0.5
                fill_rates[i] = 0.8
                costs[i] = 0.5
                reliabilities[i] = 0.9
                availabilities[i] = 0.0
        
        return latencies, fill_rates, costs, reliabilities, availabilities
    
    def get_broker_rankings(self, criterion: str = "performance") -> List[Tuple[str, float]]:
        """Get broker rankings by specified criterion"""
        
        rankings = []
        
        for broker_id, summary in self.performance_summaries.items():
            if criterion == "performance":
                score = summary.performance_score
            elif criterion == "cost":
                score = summary.cost_efficiency_score
            elif criterion == "reliability":
                score = summary.reliability_score
            elif criterion == "qoe":
                score = summary.avg_qoe_score
            elif criterion == "latency":
                score = max(0.0, 1.0 - (summary.avg_latency_ms / 1000.0))
            else:
                score = summary.performance_score
            
            rankings.append((broker_id, score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        
        all_summaries = list(self.performance_summaries.values())
        if not all_summaries:
            return {'error': 'No performance data available'}
        
        # Cross-broker analytics
        analytics = {
            'broker_count': len(all_summaries),
            'total_executions': sum(s.total_executions for s in all_summaries),
            'total_notional': sum(s.total_notional for s in all_summaries),
            
            # Performance distributions
            'latency_distribution': {
                'min': min(s.avg_latency_ms for s in all_summaries),
                'max': max(s.avg_latency_ms for s in all_summaries),
                'avg': mean(s.avg_latency_ms for s in all_summaries),
                'median': median(s.avg_latency_ms for s in all_summaries)
            },
            
            'qoe_distribution': {
                'min': min(s.avg_qoe_score for s in all_summaries),
                'max': max(s.avg_qoe_score for s in all_summaries),
                'avg': mean(s.avg_qoe_score for s in all_summaries),
                'median': median(s.avg_qoe_score for s in all_summaries)
            },
            
            'cost_distribution': {
                'min': min(s.avg_commission_per_share for s in all_summaries),
                'max': max(s.avg_commission_per_share for s in all_summaries),
                'avg': mean(s.avg_commission_per_share for s in all_summaries)
            },
            
            # Rankings
            'performance_rankings': self.get_broker_rankings('performance'),
            'cost_rankings': self.get_broker_rankings('cost'),
            'reliability_rankings': self.get_broker_rankings('reliability'),
            
            # Alert summary
            'recent_alerts': len(self.recent_alerts),
            'alert_counts_by_type': {
                alert_type.value: sum(
                    broker_counts[alert_type] 
                    for broker_counts in self.alert_counts.values()
                )
                for alert_type in PerformanceAlert
            },
            
            # Trend summary
            'brokers_improving': sum(
                1 for s in all_summaries 
                if s.qoe_trend == "IMPROVING"
            ),
            'brokers_degrading': sum(
                1 for s in all_summaries 
                if s.qoe_trend == "DEGRADING"
            )
        }
        
        return analytics
    
    async def export_performance_data(self, broker_id: Optional[str] = None, 
                                    format: str = "json") -> Dict[str, Any]:
        """Export performance data for analysis"""
        
        if broker_id:
            # Export specific broker data
            records = list(self.execution_records[broker_id])
            summary = self.performance_summaries.get(broker_id)
            
            export_data = {
                'broker_id': broker_id,
                'summary': asdict(summary) if summary else None,
                'execution_records': [asdict(r) for r in records],
                'export_timestamp': datetime.now().isoformat()
            }
        else:
            # Export all broker data
            export_data = {
                'all_brokers': True,
                'summaries': {
                    broker_id: asdict(summary) 
                    for broker_id, summary in self.performance_summaries.items()
                },
                'execution_records': {
                    broker_id: [asdict(r) for r in records]
                    for broker_id, records in self.execution_records.items()
                },
                'analytics': self.get_performance_analytics(),
                'export_timestamp': datetime.now().isoformat()
            }
        
        return export_data
    
    async def shutdown(self):
        """Shutdown performance tracker"""
        logger.info("Shutting down BrokerPerformanceTracker")
        
        # Cancel background tasks
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.alert_task and not self.alert_task.done():
            self.alert_task.cancel()
            try:
                await self.alert_task
            except asyncio.CancelledError:
                pass
        
        # Export final data
        final_analytics = self.get_performance_analytics()
        logger.info("Final performance analytics", **final_analytics)
        
        logger.info("BrokerPerformanceTracker shutdown complete")


def create_broker_performance_tracker(config: Dict[str, Any], event_bus: EventBus) -> BrokerPerformanceTracker:
    """Factory function to create broker performance tracker"""
    return BrokerPerformanceTracker(config, event_bus)


# Default configuration
DEFAULT_PERFORMANCE_TRACKER_CONFIG = {
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