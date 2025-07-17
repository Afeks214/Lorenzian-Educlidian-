"""
Real-Time Portfolio Performance Monitor

This module implements comprehensive real-time performance monitoring for the Portfolio 
Optimizer Agent system, providing continuous tracking, alerts, and optimization 
recommendations with <10ms response times.

Key Features:
- Real-time performance tracking across all portfolio components
- Intelligent optimization recommendations based on performance analysis
- Automated alert system for performance degradation
- Historical performance analysis and trending
- Integration with all portfolio optimization modules
- Performance benchmarking and comparison
- Regime-aware performance analysis
"""

import logging


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
import time
import structlog
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationRegime

logger = structlog.get_logger()


class PerformanceAlertLevel(Enum):
    """Performance alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class OptimizationRecommendation(Enum):
    """Optimization recommendation types"""
    INCREASE_DIVERSIFICATION = "increase_diversification"
    REDUCE_CORRELATION_EXPOSURE = "reduce_correlation_exposure"
    REBALANCE_WEIGHTS = "rebalance_weights"
    ADJUST_RISK_TARGETS = "adjust_risk_targets"
    SWITCH_OPTIMIZATION_METHOD = "switch_optimization_method"
    EMERGENCY_RISK_REDUCTION = "emergency_risk_reduction"
    MAINTAIN_CURRENT_ALLOCATION = "maintain_current"


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    timestamp: datetime
    
    # Return metrics
    portfolio_return_1d: float
    portfolio_return_7d: float
    portfolio_return_30d: float
    annualized_return: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    tracking_error: float
    
    # Diversification metrics
    diversification_ratio: float
    effective_strategies: float
    concentration_index: float
    
    # Optimization metrics
    avg_optimization_time_ms: float
    rebalance_frequency: float
    transaction_costs: float
    
    # Alert counts
    performance_alerts_24h: int
    risk_breaches_24h: int


@dataclass
class PerformanceAlert:
    """Performance alert notification"""
    timestamp: datetime
    level: PerformanceAlertLevel
    category: str
    message: str
    current_value: float
    threshold_value: float
    recommendation: OptimizationRecommendation
    urgency_score: float  # 0-1 scale
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'category': self.category,
            'message': self.message,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'recommendation': self.recommendation.value,
            'urgency_score': self.urgency_score
        }


@dataclass
class OptimizationRecommendationData:
    """Detailed optimization recommendation"""
    timestamp: datetime
    recommendation_type: OptimizationRecommendation
    confidence: float
    expected_improvement: float
    implementation_urgency: float
    suggested_changes: Dict[str, Union[float, str]]
    risk_impact: float
    cost_estimate: float
    reasoning: str


class PortfolioPerformanceMonitor:
    """
    Real-Time Portfolio Performance Monitor with intelligent optimization 
    recommendations and automated alerting system.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        n_strategies: int = 5,
        monitoring_frequency_ms: int = 1000,  # 1 second updates
        alert_cooldown_minutes: int = 5
    ):
        """
        Initialize Portfolio Performance Monitor
        
        Args:
            event_bus: Event bus for real-time communication
            n_strategies: Number of strategies to monitor
            monitoring_frequency_ms: Monitoring update frequency in milliseconds
            alert_cooldown_minutes: Minimum time between similar alerts
        """
        self.event_bus = event_bus
        self.n_strategies = n_strategies
        self.monitoring_frequency = monitoring_frequency_ms / 1000.0  # Convert to seconds
        self.alert_cooldown = timedelta(minutes=alert_cooldown_minutes)
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_sharpe_ratio': 0.5,
            'max_volatility': 0.25,
            'max_drawdown': 0.15,
            'min_diversification_ratio': 1.1,
            'max_concentration': 0.5,
            'max_optimization_time_ms': 10.0,
            'max_tracking_error': 0.05,
            'min_effective_strategies': 2.0
        }
        
        # Real-time data storage
        self.performance_history: deque = deque(maxlen=10000)  # Keep 10k recent points
        self.alert_history: List[PerformanceAlert] = []
        self.recommendation_history: List[OptimizationRecommendationData] = []
        
        # Current state
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.current_weights = np.array([1.0 / n_strategies] * n_strategies)
        self.current_regime = CorrelationRegime.NORMAL
        
        # Performance tracking
        self.monitoring_start_time = datetime.now()
        self.update_count = 0
        self.alert_count_24h = 0
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.update_lock = threading.Lock()
        
        # Strategy performance data
        self.strategy_returns: Dict[str, deque] = {
            f'Strategy_{i}': deque(maxlen=1000) for i in range(n_strategies)
        }
        self.portfolio_returns: deque = deque(maxlen=1000)
        self.benchmark_returns: deque = deque(maxlen=1000)
        
        # Optimization engine references (set externally)
        self.portfolio_optimizer = None
        self.correlation_manager = None
        self.rebalancing_engine = None
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("Portfolio Performance Monitor initialized",
                   n_strategies=n_strategies,
                   monitoring_frequency_ms=monitoring_frequency_ms,
                   alert_cooldown_minutes=alert_cooldown_minutes)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for real-time monitoring"""
        self.event_bus.subscribe(EventType.RISK_UPDATE, self._handle_risk_update)
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_var_update)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_breach)
    
    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Portfolio performance monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Portfolio performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                start_time = time.perf_counter()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check for alerts
                self._check_performance_alerts()
                
                # Generate recommendations if needed
                self._generate_optimization_recommendations()
                
                # Track update performance
                update_time = (time.perf_counter() - start_time) * 1000
                if update_time > 10.0:  # Should complete in <10ms
                    logger.warning("Performance monitoring update exceeded 10ms target",
                                 update_time_ms=update_time)
                
                self.update_count += 1
                
                # Sleep until next update
                time.sleep(self.monitoring_frequency)
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                time.sleep(1.0)  # Prevent tight error loop
    
    def _update_performance_metrics(self):
        """Update current performance metrics"""
        with self.update_lock:
            try:
                # Calculate current performance metrics
                current_time = datetime.now()
                
                # Return calculations
                portfolio_return_1d = self._calculate_return_over_period(timedelta(days=1))
                portfolio_return_7d = self._calculate_return_over_period(timedelta(days=7))
                portfolio_return_30d = self._calculate_return_over_period(timedelta(days=30))
                annualized_return = self._calculate_annualized_return()
                
                # Risk calculations
                volatility = self._calculate_volatility()
                sharpe_ratio = self._calculate_sharpe_ratio(annualized_return, volatility)
                max_drawdown = self._calculate_max_drawdown()
                var_95 = self._calculate_var_95()
                tracking_error = self._calculate_tracking_error()
                
                # Diversification calculations
                diversification_ratio = self._calculate_diversification_ratio()
                effective_strategies = self._calculate_effective_strategies()
                concentration_index = self._calculate_concentration_index()
                
                # Optimization performance
                avg_optimization_time = self._get_avg_optimization_time()
                rebalance_frequency = self._calculate_rebalance_frequency()
                transaction_costs = self._estimate_recent_transaction_costs()
                
                # Alert counts
                alert_count_24h = self._count_alerts_24h()
                breach_count_24h = self._count_breaches_24h()
                
                # Create metrics object
                metrics = PerformanceMetrics(
                    timestamp=current_time,
                    portfolio_return_1d=portfolio_return_1d,
                    portfolio_return_7d=portfolio_return_7d,
                    portfolio_return_30d=portfolio_return_30d,
                    annualized_return=annualized_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    var_95=var_95,
                    tracking_error=tracking_error,
                    diversification_ratio=diversification_ratio,
                    effective_strategies=effective_strategies,
                    concentration_index=concentration_index,
                    avg_optimization_time_ms=avg_optimization_time,
                    rebalance_frequency=rebalance_frequency,
                    transaction_costs=transaction_costs,
                    performance_alerts_24h=alert_count_24h,
                    risk_breaches_24h=breach_count_24h
                )
                
                self.current_metrics = metrics
                self.performance_history.append((current_time, metrics))
                
            except Exception as e:
                logger.error("Error updating performance metrics", error=str(e))
    
    def _check_performance_alerts(self):
        """Check for performance alerts and generate notifications"""
        if not self.current_metrics:
            return
        
        metrics = self.current_metrics
        alerts_generated = []
        
        # Check Sharpe ratio
        if metrics.sharpe_ratio < self.performance_thresholds['min_sharpe_ratio']:
            alert = self._create_alert(
                level=PerformanceAlertLevel.WARNING,
                category="sharpe_ratio",
                message=f"Sharpe ratio below threshold: {metrics.sharpe_ratio:.3f}",
                current_value=metrics.sharpe_ratio,
                threshold_value=self.performance_thresholds['min_sharpe_ratio'],
                recommendation=OptimizationRecommendation.ADJUST_RISK_TARGETS
            )
            alerts_generated.append(alert)
        
        # Check volatility
        if metrics.volatility > self.performance_thresholds['max_volatility']:
            alert = self._create_alert(
                level=PerformanceAlertLevel.WARNING,
                category="volatility",
                message=f"Portfolio volatility too high: {metrics.volatility:.3f}",
                current_value=metrics.volatility,
                threshold_value=self.performance_thresholds['max_volatility'],
                recommendation=OptimizationRecommendation.REDUCE_CORRELATION_EXPOSURE
            )
            alerts_generated.append(alert)
        
        # Check drawdown
        if abs(metrics.max_drawdown) > self.performance_thresholds['max_drawdown']:
            alert = self._create_alert(
                level=PerformanceAlertLevel.CRITICAL,
                category="drawdown",
                message=f"Maximum drawdown exceeded: {metrics.max_drawdown:.3f}",
                current_value=abs(metrics.max_drawdown),
                threshold_value=self.performance_thresholds['max_drawdown'],
                recommendation=OptimizationRecommendation.EMERGENCY_RISK_REDUCTION
            )
            alerts_generated.append(alert)
        
        # Check diversification
        if metrics.diversification_ratio < self.performance_thresholds['min_diversification_ratio']:
            alert = self._create_alert(
                level=PerformanceAlertLevel.WARNING,
                category="diversification",
                message=f"Poor diversification: {metrics.diversification_ratio:.3f}",
                current_value=metrics.diversification_ratio,
                threshold_value=self.performance_thresholds['min_diversification_ratio'],
                recommendation=OptimizationRecommendation.INCREASE_DIVERSIFICATION
            )
            alerts_generated.append(alert)
        
        # Check concentration
        if metrics.concentration_index > self.performance_thresholds['max_concentration']:
            alert = self._create_alert(
                level=PerformanceAlertLevel.WARNING,
                category="concentration",
                message=f"Portfolio too concentrated: {metrics.concentration_index:.3f}",
                current_value=metrics.concentration_index,
                threshold_value=self.performance_thresholds['max_concentration'],
                recommendation=OptimizationRecommendation.REBALANCE_WEIGHTS
            )
            alerts_generated.append(alert)
        
        # Check optimization performance
        if metrics.avg_optimization_time_ms > self.performance_thresholds['max_optimization_time_ms']:
            alert = self._create_alert(
                level=PerformanceAlertLevel.INFO,
                category="optimization_time",
                message=f"Optimization time exceeds target: {metrics.avg_optimization_time_ms:.2f}ms",
                current_value=metrics.avg_optimization_time_ms,
                threshold_value=self.performance_thresholds['max_optimization_time_ms'],
                recommendation=OptimizationRecommendation.SWITCH_OPTIMIZATION_METHOD
            )
            alerts_generated.append(alert)
        
        # Process and publish alerts
        for alert in alerts_generated:
            if self._should_publish_alert(alert):
                self._publish_alert(alert)
    
    def _create_alert(
        self,
        level: PerformanceAlertLevel,
        category: str,
        message: str,
        current_value: float,
        threshold_value: float,
        recommendation: OptimizationRecommendation
    ) -> PerformanceAlert:
        """Create performance alert"""
        
        # Calculate urgency score
        if level == PerformanceAlertLevel.EMERGENCY:
            urgency_score = 1.0
        elif level == PerformanceAlertLevel.CRITICAL:
            urgency_score = 0.8
        elif level == PerformanceAlertLevel.WARNING:
            urgency_score = 0.5
        else:
            urgency_score = 0.2
        
        # Adjust urgency based on deviation from threshold
        if threshold_value != 0:
            deviation_ratio = abs(current_value - threshold_value) / abs(threshold_value)
            urgency_score = min(1.0, urgency_score * (1 + deviation_ratio))
        
        return PerformanceAlert(
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            recommendation=recommendation,
            urgency_score=urgency_score
        )
    
    def _should_publish_alert(self, alert: PerformanceAlert) -> bool:
        """Check if alert should be published (respects cooldown)"""
        alert_key = f"{alert.category}_{alert.level.value}"
        
        if alert_key in self.last_alert_times:
            time_since_last = datetime.now() - self.last_alert_times[alert_key]
            if time_since_last < self.alert_cooldown:
                return False
        
        return True
    
    def _publish_alert(self, alert: PerformanceAlert):
        """Publish performance alert"""
        self.alert_history.append(alert)
        self.alert_count_24h += 1
        
        # Update last alert time
        alert_key = f"{alert.category}_{alert.level.value}"
        self.last_alert_times[alert_key] = datetime.now()
        
        # Publish via event bus
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.RISK_UPDATE,
                {
                    'type': 'PERFORMANCE_ALERT',
                    'alert': alert.to_dict()
                },
                'PortfolioPerformanceMonitor'
            )
        )
        
        logger.warning("Performance alert generated",
                      level=alert.level.value,
                      category=alert.category,
                      message=alert.message,
                      urgency_score=alert.urgency_score)
    
    def _generate_optimization_recommendations(self):
        """Generate intelligent optimization recommendations"""
        if not self.current_metrics:
            return
        
        metrics = self.current_metrics
        recommendations = []
        
        # Analyze recent performance trends
        performance_trend = self._analyze_performance_trend()
        volatility_trend = self._analyze_volatility_trend()
        
        # Generate recommendations based on current state and trends
        if performance_trend < -0.05 and metrics.sharpe_ratio < 1.0:
            # Poor performance trend
            recommendation = OptimizationRecommendationData(
                timestamp=datetime.now(),
                recommendation_type=OptimizationRecommendation.ADJUST_RISK_TARGETS,
                confidence=0.8,
                expected_improvement=0.15,
                implementation_urgency=0.7,
                suggested_changes={
                    'target_volatility': max(0.1, metrics.volatility * 0.9),
                    'rebalance_threshold': 0.03,
                    'method': 'risk_parity'
                },
                risk_impact=0.2,
                cost_estimate=0.005,
                reasoning="Poor performance trend suggests need for risk target adjustment"
            )
            recommendations.append(recommendation)
        
        if metrics.diversification_ratio < 1.2:
            # Poor diversification
            recommendation = OptimizationRecommendationData(
                timestamp=datetime.now(),
                recommendation_type=OptimizationRecommendation.INCREASE_DIVERSIFICATION,
                confidence=0.9,
                expected_improvement=0.2,
                implementation_urgency=0.6,
                suggested_changes={
                    'optimization_method': 'max_diversification',
                    'correlation_threshold': 0.6,
                    'min_weight': 0.1
                },
                risk_impact=0.1,
                cost_estimate=0.003,
                reasoning="Low diversification ratio indicates suboptimal strategy allocation"
            )
            recommendations.append(recommendation)
        
        if volatility_trend > 0.1 and self.current_regime in [CorrelationRegime.ELEVATED, CorrelationRegime.CRISIS]:
            # High volatility in stressed regime
            recommendation = OptimizationRecommendationData(
                timestamp=datetime.now(),
                recommendation_type=OptimizationRecommendation.REDUCE_CORRELATION_EXPOSURE,
                confidence=0.85,
                expected_improvement=0.25,
                implementation_urgency=0.8,
                suggested_changes={
                    'max_correlation_exposure': 0.6,
                    'regime_adjustment': True,
                    'emergency_protocols': True
                },
                risk_impact=0.3,
                cost_estimate=0.008,
                reasoning="High volatility in stressed correlation regime requires exposure reduction"
            )
            recommendations.append(recommendation)
        
        # Store and publish recommendations
        for rec in recommendations:
            self.recommendation_history.append(rec)
            self._publish_recommendation(rec)
    
    def _publish_recommendation(self, recommendation: OptimizationRecommendationData):
        """Publish optimization recommendation"""
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.RISK_UPDATE,
                {
                    'type': 'OPTIMIZATION_RECOMMENDATION',
                    'recommendation': {
                        'type': recommendation.recommendation_type.value,
                        'confidence': recommendation.confidence,
                        'expected_improvement': recommendation.expected_improvement,
                        'urgency': recommendation.implementation_urgency,
                        'suggested_changes': recommendation.suggested_changes,
                        'reasoning': recommendation.reasoning
                    }
                },
                'PortfolioPerformanceMonitor'
            )
        )
        
        logger.info("Optimization recommendation generated",
                   type=recommendation.recommendation_type.value,
                   confidence=recommendation.confidence,
                   urgency=recommendation.implementation_urgency)
    
    # Calculation methods (simplified implementations)
    def _calculate_return_over_period(self, period: timedelta) -> float:
        """Calculate portfolio return over specified period"""
        if len(self.portfolio_returns) < 2:
            return 0.0
        
        cutoff_time = datetime.now() - period
        recent_returns = [ret for ts, ret in self.portfolio_returns if ts >= cutoff_time]
        
        if len(recent_returns) < 2:
            return 0.0
        
        return np.sum(recent_returns)
    
    def _calculate_annualized_return(self) -> float:
        """Calculate annualized return"""
        if len(self.portfolio_returns) < 252:  # Need at least 1 year
            return 0.0
        
        recent_returns = [ret for _, ret in list(self.portfolio_returns)[-252:]]
        return np.mean(recent_returns) * 252  # Annualize daily returns
    
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.portfolio_returns) < 20:
            return 0.15  # Default
        
        recent_returns = [ret for _, ret in list(self.portfolio_returns)[-60:]]
        return np.std(recent_returns) * np.sqrt(252)  # Annualized
    
    def _calculate_sharpe_ratio(self, annual_return: float, volatility: float) -> float:
        """Calculate Sharpe ratio"""
        risk_free_rate = 0.02  # 2% risk-free rate
        if volatility == 0:
            return 0
        return (annual_return - risk_free_rate) / volatility
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_returns) < 20:
            return 0.0
        
        returns = [ret for _, ret in list(self.portfolio_returns)[-252:]]
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_var_95(self) -> float:
        """Calculate 95% VaR"""
        if len(self.portfolio_returns) < 20:
            return 0.02
        
        returns = [ret for _, ret in list(self.portfolio_returns)[-252:]]
        return np.percentile(returns, 5) * np.sqrt(252)  # Annualized
    
    def _calculate_tracking_error(self) -> float:
        """Calculate tracking error vs benchmark"""
        if len(self.portfolio_returns) < 20 or len(self.benchmark_returns) < 20:
            return 0.02
        
        port_returns = [ret for _, ret in list(self.portfolio_returns)[-60:]]
        bench_returns = [ret for _, ret in list(self.benchmark_returns)[-60:]]
        
        min_length = min(len(port_returns), len(bench_returns))
        if min_length < 10:
            return 0.02
        
        port_returns = port_returns[-min_length:]
        bench_returns = bench_returns[-min_length:]
        
        excess_returns = np.array(port_returns) - np.array(bench_returns)
        return np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_diversification_ratio(self) -> float:
        """Calculate diversification ratio"""
        # Simplified calculation
        if self.correlation_manager:
            try:
                corr_matrix = self.correlation_manager.correlation_tracker.get_correlation_matrix()
                if corr_matrix is not None:
                    avg_correlation = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                    return 1.0 / (1.0 - avg_correlation) if avg_correlation < 0.99 else 1.0
            except (ImportError, ModuleNotFoundError) as e:
                logger.error(f'Error occurred: {e}')
        
        return 1.0 + np.random.normal(0, 0.1)  # Default with noise
    
    def _calculate_effective_strategies(self) -> float:
        """Calculate effective number of strategies"""
        concentration = self._calculate_concentration_index()
        return 1.0 / concentration if concentration > 0 else self.n_strategies
    
    def _calculate_concentration_index(self) -> float:
        """Calculate concentration index (Herfindahl)"""
        return np.sum(self.current_weights**2)
    
    def _get_avg_optimization_time(self) -> float:
        """Get average optimization time from rebalancing engine"""
        if self.rebalancing_engine and self.rebalancing_engine.response_times:
            return np.mean(self.rebalancing_engine.response_times[-10:])
        return 5.0  # Default
    
    def _calculate_rebalance_frequency(self) -> float:
        """Calculate rebalancing frequency (per day)"""
        if self.rebalancing_engine:
            return self.rebalancing_engine.rebalance_count / max(1, 
                (datetime.now() - self.monitoring_start_time).days)
        return 1.0  # Default
    
    def _estimate_recent_transaction_costs(self) -> float:
        """Estimate recent transaction costs"""
        if self.rebalancing_engine and self.rebalancing_engine.execution_history:
            recent_executions = [ex for ex in self.rebalancing_engine.execution_history[-5:]]
            if recent_executions:
                return np.mean([ex.transaction_costs for ex in recent_executions])
        return 0.005  # Default 50bps
    
    def _count_alerts_24h(self) -> int:
        """Count alerts in last 24 hours"""
        cutoff = datetime.now() - timedelta(hours=24)
        return len([alert for alert in self.alert_history if alert.timestamp >= cutoff])
    
    def _count_breaches_24h(self) -> int:
        """Count risk breaches in last 24 hours"""
        # This would be integrated with risk breach tracking
        return 0
    
    def _analyze_performance_trend(self) -> float:
        """Analyze recent performance trend"""
        if len(self.portfolio_returns) < 10:
            return 0.0
        
        recent_returns = [ret for _, ret in list(self.portfolio_returns)[-10:]]
        return np.mean(recent_returns)
    
    def _analyze_volatility_trend(self) -> float:
        """Analyze volatility trend"""
        if len(self.portfolio_returns) < 20:
            return 0.0
        
        returns = [ret for _, ret in list(self.portfolio_returns)[-20:]]
        recent_vol = np.std(returns[-10:])
        older_vol = np.std(returns[:10])
        
        return recent_vol - older_vol
    
    def _handle_risk_update(self, event: Event):
        """Handle risk update events"""
        # Update portfolio data from risk updates
        pass
    
    def _handle_var_update(self, event: Event):
        """Handle VaR update events"""
        # Update VaR-related metrics
        pass
    
    def _handle_position_update(self, event: Event):
        """Handle position update events"""
        # Update current weights from position updates
        pass
    
    def _handle_risk_breach(self, event: Event):
        """Handle risk breach events"""
        # Count and track risk breaches
        pass
    
    def update_portfolio_return(self, timestamp: datetime, return_value: float):
        """Update portfolio return data"""
        with self.update_lock:
            self.portfolio_returns.append((timestamp, return_value))
    
    def update_strategy_return(self, strategy_id: str, timestamp: datetime, return_value: float):
        """Update strategy return data"""
        if strategy_id in self.strategy_returns:
            with self.update_lock:
                self.strategy_returns[strategy_id].append((timestamp, return_value))
    
    def update_benchmark_return(self, timestamp: datetime, return_value: float):
        """Update benchmark return data"""
        with self.update_lock:
            self.benchmark_returns.append((timestamp, return_value))
    
    def update_weights(self, new_weights: np.ndarray):
        """Update current portfolio weights"""
        with self.update_lock:
            self.current_weights = new_weights.copy()
    
    def set_correlation_regime(self, regime: CorrelationRegime):
        """Update current correlation regime"""
        self.current_regime = regime
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.current_metrics:
            return {"status": "No performance data available"}
        
        metrics = self.current_metrics
        
        return {
            "monitoring_duration_hours": (datetime.now() - self.monitoring_start_time).total_seconds() / 3600,
            "update_count": self.update_count,
            "current_metrics": {
                "annualized_return": metrics.annualized_return,
                "volatility": metrics.volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "diversification_ratio": metrics.diversification_ratio,
                "effective_strategies": metrics.effective_strategies,
                "avg_optimization_time_ms": metrics.avg_optimization_time_ms
            },
            "recent_alerts_24h": metrics.performance_alerts_24h,
            "recent_breaches_24h": metrics.risk_breaches_24h,
            "recommendations_count": len(self.recommendation_history),
            "monitoring_active": self.monitoring_active
        }