#!/usr/bin/env python3
"""
AGENT 6: Real-Time Performance Monitoring System
Comprehensive real-time monitoring with performance degradation detection,
adaptive thresholds, and automated alerting for the GrandModel trading system.
"""

import asyncio
import time
import threading
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from statistics import mean, stdev
import psutil
import redis
from prometheus_client import Counter, Histogram, Gauge, Summary

# Import existing monitoring components
from .health_monitor import HealthMonitor, HealthStatus
from .prometheus_metrics import MetricsCollector, MetricsConfig
from .enhanced_alerting import EnhancedAlertingSystem, EnhancedAlert, AlertPriority, AlertStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance monitoring metrics
PERFORMANCE_DEGRADATION_ALERTS = Counter(
    'performance_degradation_alerts_total',
    'Total performance degradation alerts',
    ['component', 'metric', 'severity']
)

PERFORMANCE_THRESHOLD_BREACHES = Counter(
    'performance_threshold_breaches_total',
    'Total performance threshold breaches',
    ['component', 'metric', 'threshold_type']
)

PERFORMANCE_BASELINE_UPDATES = Counter(
    'performance_baseline_updates_total',
    'Total performance baseline updates',
    ['component', 'metric']
)

PERFORMANCE_ANOMALY_SCORE = Gauge(
    'performance_anomaly_score',
    'Performance anomaly score (0-1)',
    ['component', 'metric']
)

PERFORMANCE_TREND_SCORE = Gauge(
    'performance_trend_score',
    'Performance trend score (-1 to 1)',
    ['component', 'metric']
)

MARKET_REGIME_DETECTION = Gauge(
    'market_regime_detection_score',
    'Market regime detection score',
    ['regime_type', 'confidence']
)

BEAR_MARKET_DEFENSE_STATUS = Gauge(
    'bear_market_defense_status',
    'Bear market defense system status',
    ['defense_mode', 'component']
)

class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    QUEUE_SIZE = "queue_size"
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    PROFIT_LOSS = "profit_loss"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"

class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"

class DefenseMode(Enum):
    """Bear market defense modes."""
    NORMAL = "normal"
    DEFENSIVE = "defensive"
    ULTRA_DEFENSIVE = "ultra_defensive"
    SHUTDOWN = "shutdown"

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    timestamp: datetime
    component: str
    metric_type: PerformanceMetricType
    tags: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    name: str
    component: str
    metric: str
    warning_threshold: float
    critical_threshold: float
    baseline_window: int = 100
    adaptive: bool = True
    trend_analysis: bool = True
    
@dataclass
class PerformanceBaseline:
    """Performance baseline for adaptive thresholds."""
    component: str
    metric: str
    mean_value: float
    std_deviation: float
    p95_value: float
    p99_value: float
    sample_count: int
    last_updated: datetime
    values: deque = field(default_factory=lambda: deque(maxlen=1000))

class PerformanceAnomalyDetector:
    """Detects performance anomalies using statistical methods."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_windows = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines = {}
        
    def add_data_point(self, component: str, metric: str, value: float) -> float:
        """Add data point and return anomaly score."""
        key = f"{component}:{metric}"
        self.data_windows[key].append(value)
        
        if len(self.data_windows[key]) < 10:
            return 0.0  # Not enough data
            
        # Calculate z-score
        values = list(self.data_windows[key])
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0.0
            
        z_score = abs(value - mean_val) / std_val
        
        # Convert to anomaly score (0-1)
        anomaly_score = min(z_score / 3.0, 1.0)  # 3-sigma rule
        
        # Update baseline if needed
        if key not in self.baselines or len(values) % 50 == 0:
            self.baselines[key] = PerformanceBaseline(
                component=component,
                metric=metric,
                mean_value=mean_val,
                std_deviation=std_val,
                p95_value=np.percentile(values, 95),
                p99_value=np.percentile(values, 99),
                sample_count=len(values),
                last_updated=datetime.utcnow()
            )
            
        return anomaly_score
        
    def get_trend_score(self, component: str, metric: str) -> float:
        """Get trend score (-1 to 1) for metric."""
        key = f"{component}:{metric}"
        if key not in self.data_windows or len(self.data_windows[key]) < 20:
            return 0.0
            
        values = list(self.data_windows[key])
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(values)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalize slope to -1 to 1 range
        max_value = max(values)
        min_value = min(values)
        if max_value == min_value:
            return 0.0
            
        normalized_slope = slope / (max_value - min_value) * len(values)
        return max(-1.0, min(1.0, normalized_slope))

class MarketRegimeDetector:
    """Detects market regimes for adaptive monitoring."""
    
    def __init__(self):
        self.price_data = deque(maxlen=100)
        self.volume_data = deque(maxlen=100)
        self.volatility_data = deque(maxlen=100)
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_confidence = 0.5
        
    def add_market_data(self, price: float, volume: float, volatility: float):
        """Add market data for regime detection."""
        self.price_data.append(price)
        self.volume_data.append(volume)
        self.volatility_data.append(volatility)
        
        # Update regime detection
        self._update_regime_detection()
        
    def _update_regime_detection(self):
        """Update market regime detection."""
        if len(self.price_data) < 20:
            return
            
        prices = np.array(list(self.price_data))
        volumes = np.array(list(self.volume_data))
        volatilities = np.array(list(self.volatility_data))
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate trend
        trend = np.mean(returns[-10:])  # Last 10 periods
        
        # Calculate volatility
        current_volatility = np.std(returns[-10:])
        avg_volatility = np.mean(volatilities[-20:])
        
        # Volume analysis
        volume_trend = np.mean(volumes[-10:]) / np.mean(volumes[-20:])
        
        # Regime classification
        if current_volatility > avg_volatility * 1.5:
            if trend < -0.02:
                self.current_regime = MarketRegime.CRISIS
                self.regime_confidence = 0.8
            else:
                self.current_regime = MarketRegime.HIGH_VOLATILITY
                self.regime_confidence = 0.7
        elif current_volatility < avg_volatility * 0.5:
            self.current_regime = MarketRegime.LOW_VOLATILITY
            self.regime_confidence = 0.6
        elif trend > 0.01 and volume_trend > 1.1:
            self.current_regime = MarketRegime.BULL
            self.regime_confidence = 0.7
        elif trend < -0.01 and volume_trend > 1.1:
            self.current_regime = MarketRegime.BEAR
            self.regime_confidence = 0.7
        else:
            self.current_regime = MarketRegime.SIDEWAYS
            self.regime_confidence = 0.5
            
        # Update metrics
        MARKET_REGIME_DETECTION.labels(
            regime_type=self.current_regime.value,
            confidence=f"{self.regime_confidence:.1f}"
        ).set(self.regime_confidence)

class BearMarketDefenseSystem:
    """Bear market defense system with adaptive position sizing."""
    
    def __init__(self):
        self.current_mode = DefenseMode.NORMAL
        self.defense_triggers = {
            DefenseMode.DEFENSIVE: {
                'drawdown_threshold': 0.05,  # 5% drawdown
                'volatility_threshold': 0.02,  # 2% volatility
                'bear_regime_confidence': 0.6
            },
            DefenseMode.ULTRA_DEFENSIVE: {
                'drawdown_threshold': 0.10,  # 10% drawdown
                'volatility_threshold': 0.03,  # 3% volatility
                'bear_regime_confidence': 0.8
            },
            DefenseMode.SHUTDOWN: {
                'drawdown_threshold': 0.15,  # 15% drawdown
                'volatility_threshold': 0.05,  # 5% volatility
                'crisis_regime_confidence': 0.7
            }
        }
        
    def evaluate_defense_mode(self, current_drawdown: float, volatility: float,
                            regime: MarketRegime, regime_confidence: float) -> DefenseMode:
        """Evaluate and update defense mode."""
        new_mode = DefenseMode.NORMAL
        
        # Check for shutdown conditions
        shutdown_trigger = self.defense_triggers[DefenseMode.SHUTDOWN]
        if (current_drawdown >= shutdown_trigger['drawdown_threshold'] or
            volatility >= shutdown_trigger['volatility_threshold'] or
            (regime == MarketRegime.CRISIS and regime_confidence >= shutdown_trigger['crisis_regime_confidence'])):
            new_mode = DefenseMode.SHUTDOWN
            
        # Check for ultra-defensive conditions
        elif (current_drawdown >= self.defense_triggers[DefenseMode.ULTRA_DEFENSIVE]['drawdown_threshold'] or
              volatility >= self.defense_triggers[DefenseMode.ULTRA_DEFENSIVE]['volatility_threshold'] or
              (regime == MarketRegime.BEAR and regime_confidence >= self.defense_triggers[DefenseMode.ULTRA_DEFENSIVE]['bear_regime_confidence'])):
            new_mode = DefenseMode.ULTRA_DEFENSIVE
            
        # Check for defensive conditions
        elif (current_drawdown >= self.defense_triggers[DefenseMode.DEFENSIVE]['drawdown_threshold'] or
              volatility >= self.defense_triggers[DefenseMode.DEFENSIVE]['volatility_threshold'] or
              (regime == MarketRegime.BEAR and regime_confidence >= self.defense_triggers[DefenseMode.DEFENSIVE]['bear_regime_confidence'])):
            new_mode = DefenseMode.DEFENSIVE
            
        if new_mode != self.current_mode:
            logger.warning(f"Defense mode changed from {self.current_mode.value} to {new_mode.value}")
            self.current_mode = new_mode
            
        # Update metrics
        BEAR_MARKET_DEFENSE_STATUS.labels(
            defense_mode=self.current_mode.value,
            component="position_sizing"
        ).set(self.current_mode.value == DefenseMode.SHUTDOWN.value and 1 or 0)
        
        return self.current_mode
        
    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier based on defense mode."""
        multipliers = {
            DefenseMode.NORMAL: 1.0,
            DefenseMode.DEFENSIVE: 0.7,
            DefenseMode.ULTRA_DEFENSIVE: 0.4,
            DefenseMode.SHUTDOWN: 0.0
        }
        return multipliers.get(self.current_mode, 1.0)

class RealTimePerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, redis_client: redis.Redis, alerting_system: EnhancedAlertingSystem):
        self.redis_client = redis_client
        self.alerting_system = alerting_system
        self.health_monitor = HealthMonitor()
        self.metrics_collector = MetricsCollector(MetricsConfig())
        self.anomaly_detector = PerformanceAnomalyDetector()
        self.regime_detector = MarketRegimeDetector()
        self.bear_defense = BearMarketDefenseSystem()
        
        # Performance thresholds
        self.thresholds = self._initialize_thresholds()
        self.performance_data = defaultdict(lambda: deque(maxlen=1000))
        self.alert_cooldowns = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_tasks = []
        
    def _initialize_thresholds(self) -> Dict[str, PerformanceThreshold]:
        """Initialize performance thresholds."""
        return {
            'strategic_agent_latency': PerformanceThreshold(
                name='strategic_agent_latency',
                component='strategic_agent',
                metric='inference_latency_ms',
                warning_threshold=10.0,
                critical_threshold=20.0,
                adaptive=True,
                trend_analysis=True
            ),
            'tactical_agent_latency': PerformanceThreshold(
                name='tactical_agent_latency',
                component='tactical_agent',
                metric='inference_latency_ms',
                warning_threshold=5.0,
                critical_threshold=10.0,
                adaptive=True,
                trend_analysis=True
            ),
            'execution_latency': PerformanceThreshold(
                name='execution_latency',
                component='execution_engine',
                metric='execution_latency_ms',
                warning_threshold=50.0,
                critical_threshold=100.0,
                adaptive=True,
                trend_analysis=True
            ),
            'system_memory': PerformanceThreshold(
                name='system_memory',
                component='system',
                metric='memory_usage_percent',
                warning_threshold=80.0,
                critical_threshold=90.0,
                adaptive=False,
                trend_analysis=True
            ),
            'error_rate': PerformanceThreshold(
                name='error_rate',
                component='system',
                metric='error_rate_percent',
                warning_threshold=1.0,
                critical_threshold=5.0,
                adaptive=True,
                trend_analysis=True
            ),
            'trading_drawdown': PerformanceThreshold(
                name='trading_drawdown',
                component='trading',
                metric='drawdown_percent',
                warning_threshold=3.0,
                critical_threshold=5.0,
                adaptive=True,
                trend_analysis=True
            )
        }
        
    async def start_monitoring(self):
        """Start real-time monitoring."""
        self.monitoring_active = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_system_performance()),
            asyncio.create_task(self._monitor_trading_performance()),
            asyncio.create_task(self._monitor_agent_performance()),
            asyncio.create_task(self._monitor_market_regime()),
            asyncio.create_task(self._monitor_bear_defense())
        ]
        
        self.monitoring_tasks = tasks
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        logger.info("Real-time performance monitoring started")
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        
        # Cancel all tasks
        for task in self.monitoring_tasks:
            task.cancel()
            
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        logger.info("Real-time performance monitoring stopped")
        
    async def _monitor_system_performance(self):
        """Monitor system performance metrics."""
        while self.monitoring_active:
            try:
                # Get system health
                system_health = await self.health_monitor.check_all_components()
                
                # Process each component
                for component_health in system_health.components:
                    if component_health.details:
                        for metric_name, value in component_health.details.items():
                            if isinstance(value, (int, float)):
                                await self._process_performance_metric(
                                    component=component_health.name,
                                    metric=metric_name,
                                    value=float(value),
                                    metric_type=self._get_metric_type(metric_name)
                                )
                
                await asyncio.sleep(1.0)  # 1 second intervals
                
            except Exception as e:
                logger.error(f"Error in system performance monitoring: {e}")
                await asyncio.sleep(5.0)
                
    async def _monitor_trading_performance(self):
        """Monitor trading performance metrics."""
        while self.monitoring_active:
            try:
                # Get trading metrics from Redis
                trading_metrics = await self._get_trading_metrics()
                
                for metric_name, value in trading_metrics.items():
                    await self._process_performance_metric(
                        component='trading',
                        metric=metric_name,
                        value=value,
                        metric_type=self._get_metric_type(metric_name)
                    )
                
                await asyncio.sleep(5.0)  # 5 second intervals
                
            except Exception as e:
                logger.error(f"Error in trading performance monitoring: {e}")
                await asyncio.sleep(10.0)
                
    async def _monitor_agent_performance(self):
        """Monitor MARL agent performance."""
        while self.monitoring_active:
            try:
                # Get agent metrics
                agent_metrics = await self._get_agent_metrics()
                
                for agent_name, metrics in agent_metrics.items():
                    for metric_name, value in metrics.items():
                        await self._process_performance_metric(
                            component=agent_name,
                            metric=metric_name,
                            value=value,
                            metric_type=self._get_metric_type(metric_name)
                        )
                
                await asyncio.sleep(2.0)  # 2 second intervals
                
            except Exception as e:
                logger.error(f"Error in agent performance monitoring: {e}")
                await asyncio.sleep(5.0)
                
    async def _monitor_market_regime(self):
        """Monitor market regime detection."""
        while self.monitoring_active:
            try:
                # Get market data
                market_data = await self._get_market_data()
                
                if market_data:
                    self.regime_detector.add_market_data(
                        price=market_data['price'],
                        volume=market_data['volume'],
                        volatility=market_data['volatility']
                    )
                
                await asyncio.sleep(30.0)  # 30 second intervals
                
            except Exception as e:
                logger.error(f"Error in market regime monitoring: {e}")
                await asyncio.sleep(60.0)
                
    async def _monitor_bear_defense(self):
        """Monitor bear market defense system."""
        while self.monitoring_active:
            try:
                # Get current performance metrics
                current_drawdown = await self._get_current_drawdown()
                volatility = await self._get_current_volatility()
                
                # Update defense mode
                new_mode = self.bear_defense.evaluate_defense_mode(
                    current_drawdown=current_drawdown,
                    volatility=volatility,
                    regime=self.regime_detector.current_regime,
                    regime_confidence=self.regime_detector.regime_confidence
                )
                
                # Generate alert if defense mode changed
                if new_mode != DefenseMode.NORMAL:
                    await self._generate_defense_alert(new_mode, current_drawdown, volatility)
                
                await asyncio.sleep(60.0)  # 1 minute intervals
                
            except Exception as e:
                logger.error(f"Error in bear defense monitoring: {e}")
                await asyncio.sleep(120.0)
                
    async def _process_performance_metric(self, component: str, metric: str, 
                                        value: float, metric_type: PerformanceMetricType):
        """Process a performance metric."""
        # Create metric object
        perf_metric = PerformanceMetric(
            name=metric,
            value=value,
            timestamp=datetime.utcnow(),
            component=component,
            metric_type=metric_type
        )
        
        # Store in performance data
        key = f"{component}:{metric}"
        self.performance_data[key].append(perf_metric)
        
        # Check for anomalies
        anomaly_score = self.anomaly_detector.add_data_point(component, metric, value)
        if anomaly_score > 0.7:  # High anomaly threshold
            await self._generate_anomaly_alert(perf_metric, anomaly_score)
            
        # Update metrics
        PERFORMANCE_ANOMALY_SCORE.labels(
            component=component,
            metric=metric
        ).set(anomaly_score)
        
        # Check trend
        trend_score = self.anomaly_detector.get_trend_score(component, metric)
        PERFORMANCE_TREND_SCORE.labels(
            component=component,
            metric=metric
        ).set(trend_score)
        
        # Check thresholds
        await self._check_performance_thresholds(perf_metric)
        
    async def _check_performance_thresholds(self, metric: PerformanceMetric):
        """Check performance thresholds and generate alerts."""
        threshold_key = f"{metric.component}_{metric.name}"
        
        if threshold_key not in self.thresholds:
            return
            
        threshold = self.thresholds[threshold_key]
        
        # Check cooldown
        cooldown_key = f"{threshold_key}_{metric.value}"
        if cooldown_key in self.alert_cooldowns:
            if datetime.utcnow() - self.alert_cooldowns[cooldown_key] < timedelta(minutes=5):
                return
                
        # Check thresholds
        if metric.value >= threshold.critical_threshold:
            await self._generate_threshold_alert(metric, threshold, 'critical')
            self.alert_cooldowns[cooldown_key] = datetime.utcnow()
            
        elif metric.value >= threshold.warning_threshold:
            await self._generate_threshold_alert(metric, threshold, 'warning')
            self.alert_cooldowns[cooldown_key] = datetime.utcnow()
            
    async def _generate_threshold_alert(self, metric: PerformanceMetric, 
                                      threshold: PerformanceThreshold, severity: str):
        """Generate threshold breach alert."""
        alert = EnhancedAlert(
            id=f"thresh_{metric.component}_{metric.name}_{int(time.time())}",
            timestamp=datetime.utcnow(),
            priority=AlertPriority.CRITICAL if severity == 'critical' else AlertPriority.HIGH,
            status=AlertStatus.ACTIVE,
            source=f"performance_monitor_{metric.component}",
            alert_type="threshold_breach",
            title=f"{metric.component.title()} {metric.name} Threshold Breach",
            message=f"{metric.component} {metric.name} value {metric.value:.2f} exceeded {severity} threshold {threshold.warning_threshold if severity == 'warning' else threshold.critical_threshold}",
            metrics={
                'current_value': metric.value,
                'threshold_value': threshold.warning_threshold if severity == 'warning' else threshold.critical_threshold,
                'component': metric.component,
                'metric_name': metric.name,
                'severity': severity
            },
            tags={f"component:{metric.component}", f"metric:{metric.name}", f"severity:{severity}"}
        )
        
        await self.alerting_system.process_alert(alert)
        
        # Update metrics
        PERFORMANCE_THRESHOLD_BREACHES.labels(
            component=metric.component,
            metric=metric.name,
            threshold_type=severity
        ).inc()
        
    async def _generate_anomaly_alert(self, metric: PerformanceMetric, anomaly_score: float):
        """Generate anomaly detection alert."""
        alert = EnhancedAlert(
            id=f"anom_{metric.component}_{metric.name}_{int(time.time())}",
            timestamp=datetime.utcnow(),
            priority=AlertPriority.HIGH,
            status=AlertStatus.ACTIVE,
            source=f"anomaly_detector_{metric.component}",
            alert_type="performance_anomaly",
            title=f"{metric.component.title()} {metric.name} Anomaly Detected",
            message=f"Performance anomaly detected in {metric.component} {metric.name}. Anomaly score: {anomaly_score:.2f}",
            metrics={
                'anomaly_score': anomaly_score,
                'current_value': metric.value,
                'component': metric.component,
                'metric_name': metric.name
            },
            tags={f"component:{metric.component}", f"metric:{metric.name}", "type:anomaly"}
        )
        
        await self.alerting_system.process_alert(alert)
        
    async def _generate_defense_alert(self, defense_mode: DefenseMode, drawdown: float, volatility: float):
        """Generate bear market defense alert."""
        alert = EnhancedAlert(
            id=f"defense_{defense_mode.value}_{int(time.time())}",
            timestamp=datetime.utcnow(),
            priority=AlertPriority.CRITICAL if defense_mode == DefenseMode.SHUTDOWN else AlertPriority.HIGH,
            status=AlertStatus.ACTIVE,
            source="bear_market_defense",
            alert_type="defense_mode_change",
            title=f"Bear Market Defense Mode: {defense_mode.value.upper()}",
            message=f"Bear market defense activated: {defense_mode.value}. Drawdown: {drawdown:.2%}, Volatility: {volatility:.2%}",
            metrics={
                'defense_mode': defense_mode.value,
                'drawdown': drawdown,
                'volatility': volatility,
                'position_multiplier': self.bear_defense.get_position_size_multiplier()
            },
            tags={f"defense_mode:{defense_mode.value}", "type:bear_defense"}
        )
        
        await self.alerting_system.process_alert(alert)
        
    async def _get_trading_metrics(self) -> Dict[str, float]:
        """Get trading metrics from Redis."""
        try:
            # Get trading metrics from Redis
            metrics = {}
            
            # Get PnL
            pnl_data = await self.redis_client.get('trading:pnl')
            if pnl_data:
                metrics['pnl'] = float(pnl_data)
                
            # Get drawdown
            drawdown_data = await self.redis_client.get('trading:drawdown')
            if drawdown_data:
                metrics['drawdown_percent'] = float(drawdown_data) * 100
                
            # Get error rate
            error_rate_data = await self.redis_client.get('trading:error_rate')
            if error_rate_data:
                metrics['error_rate_percent'] = float(error_rate_data) * 100
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting trading metrics: {e}")
            return {}
            
    async def _get_agent_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get agent performance metrics."""
        try:
            agents = ['strategic_agent', 'tactical_agent', 'execution_engine']
            metrics = {}
            
            for agent in agents:
                agent_metrics = {}
                
                # Get inference latency
                latency_data = await self.redis_client.get(f'{agent}:inference_latency')
                if latency_data:
                    agent_metrics['inference_latency_ms'] = float(latency_data)
                    
                # Get accuracy
                accuracy_data = await self.redis_client.get(f'{agent}:accuracy')
                if accuracy_data:
                    agent_metrics['accuracy_percent'] = float(accuracy_data) * 100
                    
                if agent_metrics:
                    metrics[agent] = agent_metrics
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting agent metrics: {e}")
            return {}
            
    async def _get_market_data(self) -> Optional[Dict[str, float]]:
        """Get current market data."""
        try:
            # Get market data from Redis
            price_data = await self.redis_client.get('market:price')
            volume_data = await self.redis_client.get('market:volume')
            volatility_data = await self.redis_client.get('market:volatility')
            
            if price_data and volume_data and volatility_data:
                return {
                    'price': float(price_data),
                    'volume': float(volume_data),
                    'volatility': float(volatility_data)
                }
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
            
    async def _get_current_drawdown(self) -> float:
        """Get current drawdown."""
        try:
            drawdown_data = await self.redis_client.get('trading:drawdown')
            return float(drawdown_data) if drawdown_data else 0.0
        except Exception:
            return 0.0
            
    async def _get_current_volatility(self) -> float:
        """Get current volatility."""
        try:
            volatility_data = await self.redis_client.get('market:volatility')
            return float(volatility_data) if volatility_data else 0.0
        except Exception:
            return 0.0
            
    def _get_metric_type(self, metric_name: str) -> PerformanceMetricType:
        """Get metric type based on metric name."""
        if 'latency' in metric_name or 'time' in metric_name:
            return PerformanceMetricType.LATENCY
        elif 'memory' in metric_name:
            return PerformanceMetricType.MEMORY_USAGE
        elif 'cpu' in metric_name:
            return PerformanceMetricType.CPU_USAGE
        elif 'error' in metric_name:
            return PerformanceMetricType.ERROR_RATE
        elif 'accuracy' in metric_name:
            return PerformanceMetricType.ACCURACY
        elif 'pnl' in metric_name:
            return PerformanceMetricType.PROFIT_LOSS
        elif 'drawdown' in metric_name:
            return PerformanceMetricType.DRAWDOWN
        elif 'volatility' in metric_name:
            return PerformanceMetricType.VOLATILITY
        else:
            return PerformanceMetricType.RESPONSE_TIME
            
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'monitoring_active': self.monitoring_active,
            'current_market_regime': self.regime_detector.current_regime.value,
            'regime_confidence': self.regime_detector.regime_confidence,
            'defense_mode': self.bear_defense.current_mode.value,
            'position_size_multiplier': self.bear_defense.get_position_size_multiplier(),
            'active_thresholds': len(self.thresholds),
            'performance_data_points': sum(len(data) for data in self.performance_data.values()),
            'anomaly_baselines': len(self.anomaly_detector.baselines),
            'alert_cooldowns': len(self.alert_cooldowns)
        }

# Factory function
def create_performance_monitor(redis_client: redis.Redis, 
                             alerting_system: EnhancedAlertingSystem) -> RealTimePerformanceMonitor:
    """Create real-time performance monitor instance."""
    return RealTimePerformanceMonitor(redis_client, alerting_system)

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Setup
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        alerting_system = EnhancedAlertingSystem(redis_client)
        
        # Create monitor
        monitor = create_performance_monitor(redis_client, alerting_system)
        
        # Start monitoring
        await monitor.start_monitoring()
        
    asyncio.run(main())