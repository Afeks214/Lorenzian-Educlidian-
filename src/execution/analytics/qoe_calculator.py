"""
Quality of Execution (QoE) Calculator
=====================================

Comprehensive Quality of Execution measurement and feedback system for the Routing Agent.
Calculates real-time QoE metrics, provides feedback loops for MARL training, and tracks
execution quality improvements over time.

Key Features:
- Real-time QoE calculation based on fill rates, slippage, and commissions
- Multi-dimensional quality assessment
- Learning feedback loop for routing agent optimization
- Performance benchmarking and comparison
- Quality trend analysis and alerts

QoE Formula:
QoE = (Fill Rate Score + Slippage Score + Commission Score + Latency Score) / 4 + Improvement Bonus

Author: Agent 1 - The Arbitrageur Implementation
Date: 2025-07-13
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import structlog
from statistics import mean, median, stdev

from src.core.events import EventBus, Event, EventType
from src.execution.brokers.base_broker import BrokerExecution

logger = structlog.get_logger()


class QoEGrade(Enum):
    """QoE quality grades"""
    EXCELLENT = "A+"  # 0.95+
    VERY_GOOD = "A"   # 0.90-0.95
    GOOD = "B+"       # 0.85-0.90
    AVERAGE = "B"     # 0.75-0.85
    BELOW_AVERAGE = "C+"  # 0.65-0.75
    POOR = "C"        # 0.55-0.65
    VERY_POOR = "D"   # 0.45-0.55
    UNACCEPTABLE = "F"  # <0.45


@dataclass
class QoEComponents:
    """Individual QoE component scores"""
    
    fill_rate_score: float = 0.0      # 0-1, higher is better
    slippage_score: float = 0.0       # 0-1, higher is better (lower slippage)
    commission_score: float = 0.0     # 0-1, higher is better (lower commission)
    latency_score: float = 0.0        # 0-1, higher is better (lower latency)
    improvement_bonus: float = 0.0    # 0-0.2, bonus for price improvement
    
    # Quality factors
    venue_quality_score: float = 0.0  # Venue-specific quality
    timing_quality_score: float = 0.0  # Order timing quality
    size_efficiency_score: float = 0.0  # Size-appropriate execution
    
    # Context scores
    market_conditions_adjustment: float = 0.0  # Market stress adjustment
    urgency_adjustment: float = 0.0            # Urgency impact
    
    def calculate_composite_score(self) -> float:
        """Calculate composite QoE score"""
        
        # Core components (weighted average)
        core_score = (
            self.fill_rate_score * 0.30 +
            self.slippage_score * 0.25 +
            self.commission_score * 0.20 +
            self.latency_score * 0.25
        )
        
        # Quality factors (additional weighting)
        quality_factor = (
            self.venue_quality_score * 0.4 +
            self.timing_quality_score * 0.3 +
            self.size_efficiency_score * 0.3
        )
        
        # Combine with context adjustments
        composite_score = (
            core_score * 0.7 +
            quality_factor * 0.3 +
            self.improvement_bonus +
            self.market_conditions_adjustment +
            self.urgency_adjustment
        )
        
        return np.clip(composite_score, 0.0, 1.0)


@dataclass
class QoEMeasurement:
    """Complete QoE measurement for a single execution"""
    
    execution_id: str
    broker_id: str
    symbol: str
    timestamp: datetime
    
    # Raw execution data
    order_quantity: int
    filled_quantity: int
    execution_price: float
    expected_price: float
    commission: float
    latency_ms: float
    
    # Market context
    market_price_at_order: float
    market_price_at_execution: float
    spread_at_execution: float
    volatility: float
    market_stress_level: float
    
    # Order context
    order_urgency: float
    order_size_category: str  # SMALL, MEDIUM, LARGE, JUMBO
    order_complexity: float
    
    # Calculated metrics
    fill_rate: float = 0.0
    slippage_bps: float = 0.0
    price_improvement_bps: float = 0.0
    effective_spread_bps: float = 0.0
    
    # QoE components
    components: QoEComponents = field(default_factory=QoEComponents)
    
    # Final scores
    qoe_score: float = 0.0
    qoe_grade: QoEGrade = QoEGrade.AVERAGE
    
    # Benchmarking
    peer_comparison_score: float = 0.0  # vs other brokers
    historical_comparison_score: float = 0.0  # vs historical performance
    
    def __post_init__(self):
        self.calculate_metrics()
        self.calculate_qoe()
    
    def calculate_metrics(self):
        """Calculate basic execution metrics"""
        
        # Fill rate
        self.fill_rate = self.filled_quantity / max(1, self.order_quantity)
        
        # Slippage (positive = adverse, negative = favorable)
        price_diff = self.execution_price - self.expected_price
        if self.order_quantity > 0:  # Buy order
            self.slippage_bps = (price_diff / self.expected_price) * 10000
        else:  # Sell order
            self.slippage_bps = (-price_diff / self.expected_price) * 10000
        
        # Price improvement (negative slippage)
        self.price_improvement_bps = max(0.0, -self.slippage_bps)
        
        # Effective spread
        mid_price = (self.market_price_at_order + self.market_price_at_execution) / 2
        self.effective_spread_bps = abs(self.execution_price - mid_price) / mid_price * 10000
    
    def calculate_qoe(self):
        """Calculate comprehensive QoE score"""
        
        # Fill rate score
        self.components.fill_rate_score = self.fill_rate
        
        # Slippage score (lower slippage is better)
        max_acceptable_slippage = 50.0  # 50 bps
        self.components.slippage_score = max(0.0, 1.0 - (abs(self.slippage_bps) / max_acceptable_slippage))
        
        # Commission score (lower commission is better)
        commission_per_share = self.commission / max(1, abs(self.order_quantity))
        max_acceptable_commission = 0.10  # $0.10 per share
        self.components.commission_score = max(0.0, 1.0 - (commission_per_share / max_acceptable_commission))
        
        # Latency score (lower latency is better)
        max_acceptable_latency = 1000.0  # 1 second
        self.components.latency_score = max(0.0, 1.0 - (self.latency_ms / max_acceptable_latency))
        
        # Price improvement bonus
        self.components.improvement_bonus = min(0.2, self.price_improvement_bps / 100.0)
        
        # Venue quality score (simplified)
        self.components.venue_quality_score = 0.8  # Default good venue quality
        
        # Timing quality score (based on market conditions)
        if self.market_stress_level < 0.3:
            self.components.timing_quality_score = 0.9  # Good timing in calm markets
        elif self.market_stress_level < 0.7:
            self.components.timing_quality_score = 0.7  # Moderate timing in volatile markets
        else:
            self.components.timing_quality_score = 0.5  # Challenging timing in stressed markets
        
        # Size efficiency score
        if self.order_size_category == "SMALL":
            self.components.size_efficiency_score = 0.9  # Small orders are efficient
        elif self.order_size_category == "MEDIUM":
            self.components.size_efficiency_score = 0.8
        elif self.order_size_category == "LARGE":
            self.components.size_efficiency_score = 0.7
        else:  # JUMBO
            self.components.size_efficiency_score = 0.6
        
        # Market conditions adjustment
        if self.market_stress_level > 0.7:
            self.components.market_conditions_adjustment = -0.05  # Penalty for stressed conditions
        else:
            self.components.market_conditions_adjustment = 0.0
        
        # Urgency adjustment
        if self.order_urgency > 0.8:
            self.components.urgency_adjustment = -0.02  # Small penalty for urgent orders
        else:
            self.components.urgency_adjustment = 0.0
        
        # Calculate final score
        self.qoe_score = self.components.calculate_composite_score()
        self.qoe_grade = self._determine_grade(self.qoe_score)
    
    def _determine_grade(self, score: float) -> QoEGrade:
        """Determine QoE grade based on score"""
        
        if score >= 0.95:
            return QoEGrade.EXCELLENT
        elif score >= 0.90:
            return QoEGrade.VERY_GOOD
        elif score >= 0.85:
            return QoEGrade.GOOD
        elif score >= 0.75:
            return QoEGrade.AVERAGE
        elif score >= 0.65:
            return QoEGrade.BELOW_AVERAGE
        elif score >= 0.55:
            return QoEGrade.POOR
        elif score >= 0.45:
            return QoEGrade.VERY_POOR
        else:
            return QoEGrade.UNACCEPTABLE


@dataclass
class QoEBenchmark:
    """QoE benchmarking data"""
    
    broker_id: str
    time_period: str  # "1H", "1D", "1W", "1M"
    
    # Aggregate metrics
    avg_qoe_score: float = 0.0
    median_qoe_score: float = 0.0
    std_qoe_score: float = 0.0
    min_qoe_score: float = 0.0
    max_qoe_score: float = 0.0
    
    # Grade distribution
    grade_distribution: Dict[QoEGrade, int] = field(default_factory=dict)
    
    # Component performance
    avg_fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_commission_per_share: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Trend indicators
    qoe_trend: str = "STABLE"  # IMPROVING, DEGRADING, STABLE
    trend_strength: float = 0.0  # -1 to 1
    
    # Relative performance
    percentile_rank: float = 0.0  # vs all brokers
    relative_performance: str = "AVERAGE"  # TOP_TIER, ABOVE_AVERAGE, AVERAGE, BELOW_AVERAGE, BOTTOM_TIER


class QoECalculator:
    """
    Comprehensive Quality of Execution calculator and analytics system
    
    Provides real-time QoE measurement, benchmarking, and feedback for routing optimization.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        
        # Configuration
        self.measurement_window_hours = config.get('measurement_window_hours', 24)
        self.min_measurements_for_benchmark = config.get('min_measurements_for_benchmark', 10)
        
        # Data storage
        self.measurements: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.benchmarks: Dict[str, Dict[str, QoEBenchmark]] = defaultdict(dict)
        
        # Real-time tracking
        self.current_session_measurements: deque = deque(maxlen=1000)
        self.broker_session_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance tracking
        self.calculation_times: deque = deque(maxlen=1000)
        self.total_measurements: int = 0
        
        # Background tasks
        self.benchmark_update_task: Optional[asyncio.Task] = None
        self.alert_monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("QoECalculator initialized",
                   measurement_window_hours=self.measurement_window_hours,
                   min_measurements=self.min_measurements_for_benchmark)
        
        # Start background monitoring
        asyncio.create_task(self._start_monitoring())
    
    async def _start_monitoring(self):
        """Start background monitoring tasks"""
        self.benchmark_update_task = asyncio.create_task(self._benchmark_update_loop())
        self.alert_monitoring_task = asyncio.create_task(self._alert_monitoring_loop())
    
    async def _benchmark_update_loop(self):
        """Background loop to update benchmarks"""
        while True:
            try:
                await self._update_all_benchmarks()
                await asyncio.sleep(300)  # Update every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Benchmark update loop error", error=str(e))
                await asyncio.sleep(300)
    
    async def _alert_monitoring_loop(self):
        """Background loop to monitor for QoE alerts"""
        while True:
            try:
                await self._check_qoe_alerts()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert monitoring loop error", error=str(e))
                await asyncio.sleep(60)
    
    def calculate_qoe(self, execution_data: Dict[str, Any]) -> QoEMeasurement:
        """
        Calculate Quality of Execution for a single execution
        
        Args:
            execution_data: Dictionary containing all execution information
            
        Returns:
            QoEMeasurement with complete quality assessment
        """
        start_time = time.perf_counter()
        
        try:
            # Create QoE measurement
            measurement = QoEMeasurement(
                execution_id=execution_data['execution_id'],
                broker_id=execution_data['broker_id'],
                symbol=execution_data['symbol'],
                timestamp=execution_data.get('timestamp', datetime.now()),
                
                # Execution data
                order_quantity=execution_data['order_quantity'],
                filled_quantity=execution_data['filled_quantity'],
                execution_price=execution_data['execution_price'],
                expected_price=execution_data['expected_price'],
                commission=execution_data.get('commission', 0.0),
                latency_ms=execution_data.get('latency_ms', 0.0),
                
                # Market context
                market_price_at_order=execution_data.get('market_price_at_order', execution_data['execution_price']),
                market_price_at_execution=execution_data.get('market_price_at_execution', execution_data['execution_price']),
                spread_at_execution=execution_data.get('spread_at_execution', 0.05),
                volatility=execution_data.get('volatility', 0.15),
                market_stress_level=execution_data.get('market_stress_level', 0.0),
                
                # Order context
                order_urgency=execution_data.get('order_urgency', 0.5),
                order_size_category=self._categorize_order_size(execution_data['order_quantity']),
                order_complexity=execution_data.get('order_complexity', 0.5)
            )
            
            # Add peer comparison
            measurement.peer_comparison_score = self._calculate_peer_comparison(measurement)
            
            # Add historical comparison
            measurement.historical_comparison_score = self._calculate_historical_comparison(measurement)
            
            # Store measurement
            self.measurements[measurement.broker_id].append(measurement)
            self.current_session_measurements.append(measurement)
            self.total_measurements += 1
            
            # Update session stats
            self._update_session_stats(measurement)
            
            # Track calculation time
            calc_time = (time.perf_counter() - start_time) * 1000
            self.calculation_times.append(calc_time)
            
            # Emit QoE event
            self.event_bus.emit(Event(
                type=EventType.EXECUTION_COMPLETE,
                data={
                    'qoe_measurement': measurement,
                    'qoe_score': measurement.qoe_score,
                    'qoe_grade': measurement.qoe_grade.value,
                    'calculation_time_ms': calc_time
                }
            ))
            
            logger.debug("QoE calculated",
                        execution_id=measurement.execution_id,
                        broker=measurement.broker_id,
                        qoe_score=measurement.qoe_score,
                        qoe_grade=measurement.qoe_grade.value,
                        calc_time_ms=calc_time)
            
            return measurement
            
        except Exception as e:
            logger.error("QoE calculation failed", error=str(e), execution_data=execution_data)
            raise
    
    def _categorize_order_size(self, quantity: int) -> str:
        """Categorize order size"""
        abs_quantity = abs(quantity)
        
        if abs_quantity < 100:
            return "SMALL"
        elif abs_quantity < 1000:
            return "MEDIUM"
        elif abs_quantity < 10000:
            return "LARGE"
        else:
            return "JUMBO"
    
    def _calculate_peer_comparison(self, measurement: QoEMeasurement) -> float:
        """Calculate how this execution compares to other brokers"""
        
        # Get recent measurements from other brokers
        peer_scores = []
        cutoff_time = measurement.timestamp - timedelta(hours=1)
        
        for broker_id, broker_measurements in self.measurements.items():
            if broker_id == measurement.broker_id:
                continue
                
            recent_measurements = [
                m for m in broker_measurements 
                if m.timestamp > cutoff_time and m.symbol == measurement.symbol
            ]
            
            if recent_measurements:
                avg_score = mean(m.qoe_score for m in recent_measurements)
                peer_scores.append(avg_score)
        
        if not peer_scores:
            return 0.5  # Neutral if no peer data
        
        peer_avg = mean(peer_scores)
        
        # Return relative performance (0-1)
        if peer_avg == 0:
            return 0.5
        
        return min(1.0, measurement.qoe_score / peer_avg)
    
    def _calculate_historical_comparison(self, measurement: QoEMeasurement) -> float:
        """Calculate how this execution compares to historical performance"""
        
        # Get historical measurements for this broker
        cutoff_time = measurement.timestamp - timedelta(days=7)
        historical_measurements = [
            m for m in self.measurements[measurement.broker_id]
            if m.timestamp > cutoff_time and m.symbol == measurement.symbol
        ]
        
        if len(historical_measurements) < 5:
            return 0.5  # Neutral if insufficient history
        
        historical_avg = mean(m.qoe_score for m in historical_measurements)
        
        if historical_avg == 0:
            return 0.5
        
        return min(1.0, measurement.qoe_score / historical_avg)
    
    def _update_session_stats(self, measurement: QoEMeasurement):
        """Update real-time session statistics"""
        
        broker_stats = self.broker_session_stats[measurement.broker_id]
        
        # Update running averages
        session_measurements = [
            m for m in self.current_session_measurements 
            if m.broker_id == measurement.broker_id
        ]
        
        if session_measurements:
            broker_stats['avg_qoe_score'] = mean(m.qoe_score for m in session_measurements)
            broker_stats['avg_fill_rate'] = mean(m.fill_rate for m in session_measurements)
            broker_stats['avg_slippage_bps'] = mean(m.slippage_bps for m in session_measurements)
            broker_stats['avg_latency_ms'] = mean(m.latency_ms for m in session_measurements)
            broker_stats['total_executions'] = len(session_measurements)
    
    async def _update_all_benchmarks(self):
        """Update benchmarks for all brokers"""
        
        for broker_id in self.measurements.keys():
            await self._update_broker_benchmarks(broker_id)
    
    async def _update_broker_benchmarks(self, broker_id: str):
        """Update benchmarks for specific broker"""
        
        measurements = list(self.measurements[broker_id])
        if len(measurements) < self.min_measurements_for_benchmark:
            return
        
        # Update benchmarks for different time periods
        time_periods = {
            "1H": timedelta(hours=1),
            "1D": timedelta(days=1),
            "1W": timedelta(weeks=1),
            "1M": timedelta(days=30)
        }
        
        for period_name, period_delta in time_periods.items():
            cutoff_time = datetime.now() - period_delta
            period_measurements = [m for m in measurements if m.timestamp > cutoff_time]
            
            if len(period_measurements) >= 5:  # Minimum for meaningful benchmark
                benchmark = self._calculate_benchmark(broker_id, period_name, period_measurements)
                self.benchmarks[broker_id][period_name] = benchmark
    
    def _calculate_benchmark(self, broker_id: str, period: str, measurements: List[QoEMeasurement]) -> QoEBenchmark:
        """Calculate benchmark for given measurements"""
        
        qoe_scores = [m.qoe_score for m in measurements]
        
        benchmark = QoEBenchmark(
            broker_id=broker_id,
            time_period=period,
            avg_qoe_score=mean(qoe_scores),
            median_qoe_score=median(qoe_scores),
            std_qoe_score=stdev(qoe_scores) if len(qoe_scores) > 1 else 0.0,
            min_qoe_score=min(qoe_scores),
            max_qoe_score=max(qoe_scores),
            avg_fill_rate=mean(m.fill_rate for m in measurements),
            avg_slippage_bps=mean(m.slippage_bps for m in measurements),
            avg_commission_per_share=mean(m.commission / max(1, abs(m.order_quantity)) for m in measurements),
            avg_latency_ms=mean(m.latency_ms for m in measurements)
        )
        
        # Calculate grade distribution
        for measurement in measurements:
            grade = measurement.qoe_grade
            benchmark.grade_distribution[grade] = benchmark.grade_distribution.get(grade, 0) + 1
        
        # Calculate trend
        if len(measurements) >= 10:
            recent_scores = qoe_scores[-5:]
            earlier_scores = qoe_scores[-10:-5]
            
            recent_avg = mean(recent_scores)
            earlier_avg = mean(earlier_scores)
            
            change_pct = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
            
            if change_pct > 0.05:
                benchmark.qoe_trend = "IMPROVING"
                benchmark.trend_strength = min(1.0, change_pct)
            elif change_pct < -0.05:
                benchmark.qoe_trend = "DEGRADING"
                benchmark.trend_strength = max(-1.0, change_pct)
            else:
                benchmark.qoe_trend = "STABLE"
                benchmark.trend_strength = 0.0
        
        return benchmark
    
    async def _check_qoe_alerts(self):
        """Check for QoE quality alerts"""
        
        for broker_id in self.measurements.keys():
            recent_measurements = list(self.measurements[broker_id])[-10:]
            
            if len(recent_measurements) >= 5:
                recent_avg_qoe = mean(m.qoe_score for m in recent_measurements)
                
                # Alert for poor performance
                if recent_avg_qoe < 0.6:
                    await self._emit_qoe_alert(broker_id, "POOR_PERFORMANCE", {
                        'avg_qoe_score': recent_avg_qoe,
                        'measurement_count': len(recent_measurements)
                    })
                
                # Alert for degrading performance
                if len(recent_measurements) >= 10:
                    earlier_measurements = recent_measurements[:5]
                    later_measurements = recent_measurements[5:]
                    
                    earlier_avg = mean(m.qoe_score for m in earlier_measurements)
                    later_avg = mean(m.qoe_score for m in later_measurements)
                    
                    if later_avg < earlier_avg * 0.85:  # 15% degradation
                        await self._emit_qoe_alert(broker_id, "PERFORMANCE_DEGRADATION", {
                            'earlier_avg_qoe': earlier_avg,
                            'later_avg_qoe': later_avg,
                            'degradation_pct': (1 - later_avg / earlier_avg) * 100
                        })
    
    async def _emit_qoe_alert(self, broker_id: str, alert_type: str, details: Dict[str, Any]):
        """Emit QoE quality alert"""
        
        alert_data = {
            'broker_id': broker_id,
            'alert_type': alert_type,
            'details': details,
            'timestamp': datetime.now(),
            'severity': 'HIGH' if alert_type == 'POOR_PERFORMANCE' else 'MEDIUM'
        }
        
        self.event_bus.emit(Event(
            type=EventType.RISK_ALERT,
            data=alert_data
        ))
        
        logger.warning("QoE alert emitted",
                      broker=broker_id,
                      alert_type=alert_type,
                      severity=alert_data['severity'],
                      details=details)
    
    def get_broker_qoe_summary(self, broker_id: str, time_period: str = "1D") -> Optional[QoEBenchmark]:
        """Get QoE summary for specific broker"""
        return self.benchmarks.get(broker_id, {}).get(time_period)
    
    def get_all_broker_rankings(self, time_period: str = "1D") -> List[Tuple[str, float]]:
        """Get broker rankings by QoE score"""
        
        rankings = []
        for broker_id, benchmarks in self.benchmarks.items():
            benchmark = benchmarks.get(time_period)
            if benchmark:
                rankings.append((broker_id, benchmark.avg_qoe_score))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_qoe_analytics(self) -> Dict[str, Any]:
        """Get comprehensive QoE analytics"""
        
        all_measurements = []
        for broker_measurements in self.measurements.values():
            all_measurements.extend(broker_measurements)
        
        if not all_measurements:
            return {'error': 'No QoE measurements available'}
        
        # Overall statistics
        all_scores = [m.qoe_score for m in all_measurements]
        
        analytics = {
            'total_measurements': len(all_measurements),
            'measurement_timespan_hours': (
                (max(m.timestamp for m in all_measurements) - 
                 min(m.timestamp for m in all_measurements)).total_seconds() / 3600
            ) if len(all_measurements) > 1 else 0,
            
            'overall_qoe_stats': {
                'avg_score': mean(all_scores),
                'median_score': median(all_scores),
                'std_score': stdev(all_scores) if len(all_scores) > 1 else 0.0,
                'min_score': min(all_scores),
                'max_score': max(all_scores)
            },
            
            'grade_distribution': {},
            'broker_rankings': self.get_all_broker_rankings(),
            'session_stats': dict(self.broker_session_stats),
            
            'performance_metrics': {
                'avg_calculation_time_ms': mean(self.calculation_times) if self.calculation_times else 0.0,
                'calculations_per_second': self.total_measurements / max(1, len(self.calculation_times) * 0.001)
            }
        }
        
        # Grade distribution
        for measurement in all_measurements:
            grade = measurement.qoe_grade.value
            analytics['grade_distribution'][grade] = analytics['grade_distribution'].get(grade, 0) + 1
        
        return analytics
    
    def create_qoe_feedback(self, measurement: QoEMeasurement) -> Dict[str, float]:
        """
        Create feedback signal for MARL routing agent
        
        Returns:
            Dictionary with reward components for routing agent training
        """
        
        # Base reward components
        fill_rate_reward = measurement.fill_rate
        slippage_reward = max(0.0, 1.0 - (abs(measurement.slippage_bps) / 50.0))
        commission_reward = max(0.0, 1.0 - (measurement.commission / max(1, abs(measurement.order_quantity)) / 0.10))
        latency_reward = max(0.0, 1.0 - (measurement.latency_ms / 1000.0))
        
        # Composite QoE reward
        qoe_reward = measurement.qoe_score
        
        # Performance relative to peers
        peer_reward = measurement.peer_comparison_score
        
        # Improvement over historical performance
        improvement_reward = measurement.historical_comparison_score
        
        # Total reward (scaled for MARL training)
        total_reward = (
            fill_rate_reward +
            slippage_reward +
            commission_reward +
            latency_reward * 0.5 +  # Lower weight for latency
            qoe_reward +
            peer_reward * 0.5 +
            improvement_reward * 0.3
        ) / 6.3  # Normalize to ~0-1 range
        
        return {
            'total_reward': total_reward,
            'fill_rate_reward': fill_rate_reward,
            'slippage_reward': slippage_reward,
            'commission_reward': commission_reward,
            'latency_reward': latency_reward,
            'qoe_reward': qoe_reward,
            'peer_reward': peer_reward,
            'improvement_reward': improvement_reward
        }
    
    async def shutdown(self):
        """Shutdown QoE calculator"""
        logger.info("Shutting down QoECalculator")
        
        # Cancel background tasks
        if self.benchmark_update_task and not self.benchmark_update_task.done():
            self.benchmark_update_task.cancel()
            try:
                await self.benchmark_update_task
            except asyncio.CancelledError:
                pass
        
        if self.alert_monitoring_task and not self.alert_monitoring_task.done():
            self.alert_monitoring_task.cancel()
            try:
                await self.alert_monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Final analytics
        final_analytics = self.get_qoe_analytics()
        logger.info("Final QoE analytics", **final_analytics.get('overall_qoe_stats', {}))
        
        logger.info("QoECalculator shutdown complete")


def create_qoe_calculator(config: Dict[str, Any], event_bus: EventBus) -> QoECalculator:
    """Factory function to create QoE calculator"""
    return QoECalculator(config, event_bus)


# Default configuration
DEFAULT_QOE_CONFIG = {
    'measurement_window_hours': 24,
    'min_measurements_for_benchmark': 10
}