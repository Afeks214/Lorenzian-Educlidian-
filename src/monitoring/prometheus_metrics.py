#!/usr/bin/env python3
"""
AGENT 13: Comprehensive Prometheus Metrics Implementation
Custom metrics for all trading components with business metrics and SLA monitoring
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import psutil
import logging
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum,
    CollectorRegistry, multiprocess, generate_latest,
    CONTENT_TYPE_LATEST, start_http_server
)
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST as OPENMETRICS_CONTENT_TYPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trading Performance Metrics
TRADING_SIGNALS_GENERATED = Counter(
    'trading_signals_generated_total',
    'Total number of trading signals generated',
    ['strategy', 'signal_type', 'asset']
)

TRADING_ORDERS_PLACED = Counter(
    'trading_orders_placed_total',
    'Total number of orders placed',
    ['order_type', 'side', 'asset', 'status']
)

TRADING_ORDERS_FILLED = Counter(
    'trading_orders_filled_total',
    'Total number of orders filled',
    ['order_type', 'side', 'asset']
)

TRADING_PNL = Gauge(
    'trading_pnl_usd',
    'Current profit and loss in USD',
    ['strategy', 'asset']
)

TRADING_POSITION_SIZE = Gauge(
    'trading_position_size',
    'Current position size',
    ['asset', 'side']
)

TRADING_EXPOSURE = Gauge(
    'trading_exposure_usd',
    'Current market exposure in USD',
    ['asset']
)

TRADING_DRAWDOWN = Gauge(
    'trading_drawdown_percent',
    'Current drawdown percentage',
    ['strategy']
)

TRADING_SHARPE_RATIO = Gauge(
    'trading_sharpe_ratio',
    'Current Sharpe ratio',
    ['strategy', 'timeframe']
)

TRADING_WIN_RATE = Gauge(
    'trading_win_rate_percent',
    'Trading win rate percentage',
    ['strategy', 'timeframe']
)

TRADING_SLIPPAGE = Histogram(
    'trading_slippage_bps',
    'Trading slippage in basis points',
    ['asset', 'order_type'],
    buckets=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, float('inf')]
)

TRADING_EXECUTION_LATENCY = Histogram(
    'trading_execution_latency_ms',
    'Order execution latency in milliseconds',
    ['order_type', 'venue'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, float('inf')]
)

# MARL Agent Metrics
MARL_AGENT_INFERENCE_TIME = Histogram(
    'marl_agent_inference_time_ms',
    'MARL agent inference time in milliseconds',
    ['agent_type', 'model_name'],
    buckets=[0.1, 0.5, 1, 2, 5, 8, 10, 20, 50, 100, float('inf')]
)

MARL_AGENT_PREDICTIONS = Counter(
    'marl_agent_predictions_total',
    'Total number of agent predictions',
    ['agent_type', 'prediction_type']
)

MARL_AGENT_ACCURACY = Gauge(
    'marl_agent_accuracy_percent',
    'Agent prediction accuracy percentage',
    ['agent_type', 'timeframe']
)

MARL_AGENT_CONSENSUS = Gauge(
    'marl_agent_consensus_score',
    'Agent consensus score (0-1)',
    ['decision_type']
)

MARL_AGENT_ERRORS = Counter(
    'marl_agent_errors_total',
    'Total number of agent errors',
    ['agent_type', 'error_type']
)

MARL_REWARD_SIGNAL = Gauge(
    'marl_reward_signal',
    'Current reward signal value',
    ['agent_type', 'reward_type']
)

# Risk Management Metrics
RISK_VAR_95 = Gauge(
    'risk_var_95_percent',
    'Value at Risk (95th percentile) as percentage',
    ['portfolio', 'timeframe']
)

RISK_VAR_99 = Gauge(
    'risk_var_99_percent',
    'Value at Risk (99th percentile) as percentage',
    ['portfolio', 'timeframe']
)

RISK_CORRELATION_SHOCK = Gauge(
    'risk_correlation_shock_level',
    'Correlation shock level (0-1)',
    ['asset_pair']
)

RISK_KELLY_FRACTION = Gauge(
    'risk_kelly_fraction',
    'Kelly criterion fraction',
    ['strategy', 'asset']
)

RISK_MARGIN_USAGE = Gauge(
    'risk_margin_usage_percent',
    'Margin usage percentage',
    ['account']
)

RISK_POSITION_CONCENTRATION = Gauge(
    'risk_position_concentration_percent',
    'Position concentration percentage',
    ['asset']
)

RISK_VOLATILITY = Gauge(
    'risk_volatility_percent',
    'Asset volatility percentage',
    ['asset', 'timeframe']
)

RISK_ALERTS_TRIGGERED = Counter(
    'risk_alerts_triggered_total',
    'Total number of risk alerts triggered',
    ['alert_type', 'severity']
)

# System Performance Metrics
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    ['instance']
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage',
    ['instance']
)

SYSTEM_DISK_USAGE = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage',
    ['instance', 'mount_point']
)

SYSTEM_NETWORK_BYTES = Counter(
    'system_network_bytes_total',
    'Total network bytes transferred',
    ['instance', 'interface', 'direction']
)

SYSTEM_UPTIME = Gauge(
    'system_uptime_seconds',
    'System uptime in seconds',
    ['instance']
)

# Data Pipeline Metrics
DATA_PIPELINE_MESSAGES = Counter(
    'data_pipeline_messages_total',
    'Total messages processed in data pipeline',
    ['pipeline_stage', 'message_type', 'status']
)

DATA_PIPELINE_LATENCY = Histogram(
    'data_pipeline_latency_ms',
    'Data pipeline processing latency in milliseconds',
    ['pipeline_stage'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, float('inf')]
)

DATA_PIPELINE_QUEUE_SIZE = Gauge(
    'data_pipeline_queue_size',
    'Current queue size in data pipeline',
    ['queue_name']
)

DATA_PIPELINE_THROUGHPUT = Gauge(
    'data_pipeline_throughput_msgs_per_sec',
    'Data pipeline throughput in messages per second',
    ['pipeline_stage']
)

DATA_QUALITY_SCORE = Gauge(
    'data_quality_score',
    'Data quality score (0-1)',
    ['data_source', 'quality_metric']
)

# SLA Metrics
SLA_RESPONSE_TIME = Histogram(
    'sla_response_time_ms',
    'SLA response time in milliseconds',
    ['service', 'endpoint'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, float('inf')]
)

SLA_AVAILABILITY = Gauge(
    'sla_availability_percent',
    'Service availability percentage',
    ['service']
)

SLA_ERROR_RATE = Gauge(
    'sla_error_rate_percent',
    'Service error rate percentage',
    ['service', 'error_type']
)

SLA_THROUGHPUT = Gauge(
    'sla_throughput_ops_per_sec',
    'Service throughput in operations per second',
    ['service']
)

SLA_BREACH_COUNT = Counter(
    'sla_breach_count_total',
    'Total number of SLA breaches',
    ['service', 'sla_type']
)

# Business Metrics
BUSINESS_REVENUE = Gauge(
    'business_revenue_usd',
    'Business revenue in USD',
    ['timeframe', 'revenue_type']
)

BUSINESS_ACTIVE_STRATEGIES = Gauge(
    'business_active_strategies',
    'Number of active trading strategies',
    ['status']
)

BUSINESS_ASSETS_TRADED = Gauge(
    'business_assets_traded',
    'Number of assets currently being traded',
    ['asset_class']
)

BUSINESS_TRADE_VOLUME = Gauge(
    'business_trade_volume_usd',
    'Trading volume in USD',
    ['timeframe', 'asset_class']
)

BUSINESS_CUSTOMER_SATISFACTION = Gauge(
    'business_customer_satisfaction_score',
    'Customer satisfaction score (0-10)',
    ['timeframe']
)

@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enable_system_metrics: bool = True
    enable_business_metrics: bool = True
    enable_sla_metrics: bool = True
    collection_interval: float = 1.0
    metrics_port: int = 8000
    metrics_path: str = '/metrics'

class PerformanceTracker:
    """Performance tracking utility for SLA monitoring."""
    
    def __init__(self):
        self.response_times = {}
        self.error_counts = {}
        self.request_counts = {}
        self.lock = threading.Lock()
        
    @contextmanager
    def track_request(self, service: str, endpoint: str):
        """Context manager for tracking request performance."""
        start_time = time.time()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            self._record_error(service, endpoint, str(type(e).__name__))
            raise
        finally:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            self._record_request(service, endpoint, response_time, error_occurred)
            
    def _record_request(self, service: str, endpoint: str, response_time: float, error: bool):
        """Record request metrics."""
        with self.lock:
            # Record response time
            SLA_RESPONSE_TIME.labels(service=service, endpoint=endpoint).observe(response_time)
            
            # Update request counts
            key = f"{service}:{endpoint}"
            if key not in self.request_counts:
                self.request_counts[key] = {'total': 0, 'errors': 0}
            
            self.request_counts[key]['total'] += 1
            if error:
                self.request_counts[key]['errors'] += 1
                
            # Update error rate
            if self.request_counts[key]['total'] > 0:
                error_rate = (self.request_counts[key]['errors'] / self.request_counts[key]['total']) * 100
                SLA_ERROR_RATE.labels(service=service, error_type='total').set(error_rate)
                
    def _record_error(self, service: str, endpoint: str, error_type: str):
        """Record error metrics."""
        SLA_ERROR_RATE.labels(service=service, error_type=error_type).inc()
        
    def check_sla_breach(self, service: str, sla_threshold_ms: float) -> bool:
        """Check if SLA has been breached."""
        key = f"{service}:*"
        # This is a simplified check - in production, you'd want more sophisticated SLA monitoring
        # Check if average response time exceeds threshold
        return False  # Placeholder

class BusinessMetricsCollector:
    """Collector for business-specific metrics."""
    
    def __init__(self):
        self.daily_revenue = 0.0
        self.monthly_revenue = 0.0
        self.active_strategies = 0
        self.assets_traded = 0
        self.trade_volume = 0.0
        
    def update_revenue(self, amount: float, timeframe: str = 'daily'):
        """Update revenue metrics."""
        BUSINESS_REVENUE.labels(timeframe=timeframe, revenue_type='trading').set(amount)
        
    def update_active_strategies(self, count: int):
        """Update active strategies count."""
        self.active_strategies = count
        BUSINESS_ACTIVE_STRATEGIES.labels(status='active').set(count)
        
    def update_assets_traded(self, count: int, asset_class: str = 'all'):
        """Update assets traded count."""
        BUSINESS_ASSETS_TRADED.labels(asset_class=asset_class).set(count)
        
    def update_trade_volume(self, volume: float, timeframe: str = 'daily', asset_class: str = 'all'):
        """Update trade volume metrics."""
        BUSINESS_TRADE_VOLUME.labels(timeframe=timeframe, asset_class=asset_class).set(volume)
        
    def update_customer_satisfaction(self, score: float, timeframe: str = 'daily'):
        """Update customer satisfaction score."""
        BUSINESS_CUSTOMER_SATISFACTION.labels(timeframe=timeframe).set(score)

class SystemMetricsCollector:
    """Collector for system performance metrics."""
    
    def __init__(self):
        self.instance_id = f"instance_{id(self)}"
        
    def collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            SYSTEM_CPU_USAGE.labels(instance=self.instance_id).set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.labels(instance=self.instance_id).set(memory.percent)
            
            # Disk usage
            for disk in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(disk.mountpoint)
                    SYSTEM_DISK_USAGE.labels(
                        instance=self.instance_id,
                        mount_point=disk.mountpoint
                    ).set(usage.percent)
                except (PermissionError, OSError):
                    continue
                    
            # Network I/O
            network_io = psutil.net_io_counters(pernic=True)
            for interface, counters in network_io.items():
                SYSTEM_NETWORK_BYTES.labels(
                    instance=self.instance_id,
                    interface=interface,
                    direction='sent'
                ).inc(counters.bytes_sent)
                
                SYSTEM_NETWORK_BYTES.labels(
                    instance=self.instance_id,
                    interface=interface,
                    direction='recv'
                ).inc(counters.bytes_recv)
                
            # System uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            SYSTEM_UPTIME.labels(instance=self.instance_id).set(uptime)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

class TradingMetricsCollector:
    """Collector for trading-specific metrics."""
    
    def __init__(self):
        self.positions = {}
        self.pnl_history = []
        self.orders_history = []
        
    def record_signal(self, strategy: str, signal_type: str, asset: str):
        """Record trading signal generation."""
        TRADING_SIGNALS_GENERATED.labels(
            strategy=strategy,
            signal_type=signal_type,
            asset=asset
        ).inc()
        
    def record_order(self, order_type: str, side: str, asset: str, status: str):
        """Record order placement."""
        TRADING_ORDERS_PLACED.labels(
            order_type=order_type,
            side=side,
            asset=asset,
            status=status
        ).inc()
        
        if status == 'filled':
            TRADING_ORDERS_FILLED.labels(
                order_type=order_type,
                side=side,
                asset=asset
            ).inc()
            
    def update_pnl(self, strategy: str, asset: str, pnl: float):
        """Update P&L metrics."""
        TRADING_PNL.labels(strategy=strategy, asset=asset).set(pnl)
        
    def update_position(self, asset: str, side: str, size: float):
        """Update position metrics."""
        TRADING_POSITION_SIZE.labels(asset=asset, side=side).set(size)
        
    def update_exposure(self, asset: str, exposure: float):
        """Update exposure metrics."""
        TRADING_EXPOSURE.labels(asset=asset).set(exposure)
        
    def update_drawdown(self, strategy: str, drawdown: float):
        """Update drawdown metrics."""
        TRADING_DRAWDOWN.labels(strategy=strategy).set(drawdown)
        
    def update_sharpe_ratio(self, strategy: str, timeframe: str, sharpe: float):
        """Update Sharpe ratio metrics."""
        TRADING_SHARPE_RATIO.labels(strategy=strategy, timeframe=timeframe).set(sharpe)
        
    def update_win_rate(self, strategy: str, timeframe: str, win_rate: float):
        """Update win rate metrics."""
        TRADING_WIN_RATE.labels(strategy=strategy, timeframe=timeframe).set(win_rate)
        
    def record_slippage(self, asset: str, order_type: str, slippage_bps: float):
        """Record slippage metrics."""
        TRADING_SLIPPAGE.labels(asset=asset, order_type=order_type).observe(slippage_bps)
        
    def record_execution_latency(self, order_type: str, venue: str, latency_ms: float):
        """Record execution latency metrics."""
        TRADING_EXECUTION_LATENCY.labels(order_type=order_type, venue=venue).observe(latency_ms)

class RiskMetricsCollector:
    """Collector for risk management metrics."""
    
    def __init__(self):
        self.var_history = []
        self.correlation_matrix = {}
        
    def update_var(self, portfolio: str, timeframe: str, var_95: float, var_99: float):
        """Update VaR metrics."""
        RISK_VAR_95.labels(portfolio=portfolio, timeframe=timeframe).set(var_95)
        RISK_VAR_99.labels(portfolio=portfolio, timeframe=timeframe).set(var_99)
        
    def update_correlation_shock(self, asset_pair: str, shock_level: float):
        """Update correlation shock metrics."""
        RISK_CORRELATION_SHOCK.labels(asset_pair=asset_pair).set(shock_level)
        
    def update_kelly_fraction(self, strategy: str, asset: str, kelly_fraction: float):
        """Update Kelly fraction metrics."""
        RISK_KELLY_FRACTION.labels(strategy=strategy, asset=asset).set(kelly_fraction)
        
    def update_margin_usage(self, account: str, usage_percent: float):
        """Update margin usage metrics."""
        RISK_MARGIN_USAGE.labels(account=account).set(usage_percent)
        
    def update_position_concentration(self, asset: str, concentration_percent: float):
        """Update position concentration metrics."""
        RISK_POSITION_CONCENTRATION.labels(asset=asset).set(concentration_percent)
        
    def update_volatility(self, asset: str, timeframe: str, volatility_percent: float):
        """Update volatility metrics."""
        RISK_VOLATILITY.labels(asset=asset, timeframe=timeframe).set(volatility_percent)
        
    def record_risk_alert(self, alert_type: str, severity: str):
        """Record risk alert metrics."""
        RISK_ALERTS_TRIGGERED.labels(alert_type=alert_type, severity=severity).inc()

class MARLMetricsCollector:
    """Collector for MARL agent metrics."""
    
    def __init__(self):
        self.agent_stats = {}
        
    def record_inference_time(self, agent_type: str, model_name: str, inference_time_ms: float):
        """Record agent inference time."""
        MARL_AGENT_INFERENCE_TIME.labels(
            agent_type=agent_type,
            model_name=model_name
        ).observe(inference_time_ms)
        
    def record_prediction(self, agent_type: str, prediction_type: str):
        """Record agent prediction."""
        MARL_AGENT_PREDICTIONS.labels(
            agent_type=agent_type,
            prediction_type=prediction_type
        ).inc()
        
    def update_accuracy(self, agent_type: str, timeframe: str, accuracy_percent: float):
        """Update agent accuracy metrics."""
        MARL_AGENT_ACCURACY.labels(agent_type=agent_type, timeframe=timeframe).set(accuracy_percent)
        
    def update_consensus(self, decision_type: str, consensus_score: float):
        """Update agent consensus metrics."""
        MARL_AGENT_CONSENSUS.labels(decision_type=decision_type).set(consensus_score)
        
    def record_error(self, agent_type: str, error_type: str):
        """Record agent error."""
        MARL_AGENT_ERRORS.labels(agent_type=agent_type, error_type=error_type).inc()
        
    def update_reward_signal(self, agent_type: str, reward_type: str, reward_value: float):
        """Update reward signal metrics."""
        MARL_REWARD_SIGNAL.labels(agent_type=agent_type, reward_type=reward_type).set(reward_value)

class MetricsCollector:
    """Main metrics collector that coordinates all metric collection."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.performance_tracker = PerformanceTracker()
        self.business_metrics = BusinessMetricsCollector()
        self.system_metrics = SystemMetricsCollector()
        self.trading_metrics = TradingMetricsCollector()
        self.risk_metrics = RiskMetricsCollector()
        self.marl_metrics = MARLMetricsCollector()
        
        # Start background collection
        self.collection_thread = None
        self.stop_collection = False
        
    def start_collection(self):
        """Start background metrics collection."""
        if self.collection_thread is None:
            self.collection_thread = threading.Thread(target=self._collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            
    def stop_collection(self):
        """Stop background metrics collection."""
        self.stop_collection = True
        if self.collection_thread:
            self.collection_thread.join()
            
    def _collection_loop(self):
        """Background collection loop."""
        while not self.stop_collection:
            try:
                if self.config.enable_system_metrics:
                    self.system_metrics.collect_system_metrics()
                    
                time.sleep(self.config.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(5)  # Wait before retrying
                
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format."""
        return generate_latest()
        
    def start_metrics_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.config.metrics_port)
            logger.info(f"Metrics server started on port {self.config.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

# Factory function
def create_metrics_collector(config: Optional[MetricsConfig] = None) -> MetricsCollector:
    """Create metrics collector instance."""
    if config is None:
        config = MetricsConfig()
    return MetricsCollector(config)

# Example usage
if __name__ == "__main__":
    # Example of how to use the metrics collector
    config = MetricsConfig(
        enable_system_metrics=True,
        enable_business_metrics=True,
        enable_sla_metrics=True,
        collection_interval=1.0,
        metrics_port=8000
    )
    
    collector = create_metrics_collector(config)
    collector.start_collection()
    collector.start_metrics_server()
    
    # Example of recording metrics
    collector.trading_metrics.record_signal('momentum_strategy', 'buy', 'BTCUSD')
    collector.trading_metrics.record_order('market', 'buy', 'BTCUSD', 'filled')
    collector.trading_metrics.update_pnl('momentum_strategy', 'BTCUSD', 1250.75)
    
    collector.risk_metrics.update_var('main_portfolio', '1d', 0.025, 0.035)
    collector.risk_metrics.update_margin_usage('main_account', 65.5)
    
    collector.marl_metrics.record_inference_time('strategic', 'transformer', 2.5)
    collector.marl_metrics.update_accuracy('strategic', '1h', 78.5)
    
    collector.business_metrics.update_revenue(15000.0, 'daily')
    collector.business_metrics.update_active_strategies(5)
    
    logger.info("Metrics collection started. Check http://localhost:8000/metrics")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        collector.stop_collection()
        logger.info("Metrics collection stopped")
