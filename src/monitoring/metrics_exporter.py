"""
Prometheus metrics exporter for Strategic MARL 30m System.
Implements comprehensive metrics collection with correlation ID support.
"""

import time
import asyncio
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager
from functools import wraps
import logging

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

logger = logging.getLogger(__name__)

# Global metrics registry
metrics_registry = CollectorRegistry()

class MetricsExporter:
    """
    Centralized metrics exporter for the Strategic MARL system.
    Provides Prometheus-compatible metrics with correlation ID tracking.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics with custom or default registry."""
        self.registry = registry or metrics_registry
        self._correlation_ids: Dict[str, float] = {}
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # Inference metrics
        self.inference_latency = Histogram(
            'inference_latency_seconds',
            'Model inference latency in seconds',
            ['model_type', 'agent_name', 'correlation_id'],
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
            registry=self.registry
        )
        
        # Trading metrics
        self.trade_pnl = Gauge(
            'trade_pnl_dollars',
            'Current trade PnL in dollars',
            ['strategy', 'symbol', 'correlation_id'],
            registry=self.registry
        )
        
        self.active_positions = Gauge(
            'active_positions_count',
            'Number of active positions',
            ['strategy', 'symbol'],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            'portfolio_value_dollars',
            'Total portfolio value in dollars',
            registry=self.registry
        )
        
        # Model confidence metrics
        self.model_confidence = Gauge(
            'model_confidence_score',
            'Model confidence score (0-1)',
            ['model_type', 'agent_name'],
            registry=self.registry
        )
        
        # Event processing metrics
        self.synergy_response_rate = Counter(
            'synergy_response_total',
            'Total SYNERGY_DETECTED events processed',
            ['synergy_type', 'response_status', 'correlation_id'],
            registry=self.registry
        )
        
        self.matrix_processing_success = Counter(
            'matrix_processing_total',
            'Matrix processing operations',
            ['matrix_type', 'status'],
            registry=self.registry
        )
        
        self.matrix_processing_latency = Histogram(
            'matrix_processing_seconds',
            'Matrix processing latency',
            ['matrix_type'],
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01),
            registry=self.registry
        )
        
        # API metrics
        self.http_requests = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry
        )
        
        # System metrics
        self.event_bus_throughput = Counter(
            'event_bus_messages_total',
            'Total messages processed by event bus',
            ['event_type', 'status'],
            registry=self.registry
        )
        
        self.redis_operations = Counter(
            'redis_operations_total',
            'Redis operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        # Data pipeline metrics
        self.data_pipeline_lag = Gauge(
            'data_pipeline_lag_seconds',
            'Data pipeline lag in seconds',
            ['pipeline_stage'],
            registry=self.registry
        )
        
        self.data_pipeline_last_update = Gauge(
            'data_pipeline_last_update_timestamp',
            'Timestamp of last data pipeline update',
            ['pipeline_stage'],
            registry=self.registry
        )
        
        # Agent coordination metrics
        self.agent_agreement_rate = Summary(
            'agent_agreement_rate',
            'Rate of agreement between agents',
            ['decision_type'],
            registry=self.registry
        )
        
        self.ensemble_decisions = Counter(
            'ensemble_decisions_total',
            'Total ensemble decisions made',
            ['decision_type', 'consensus_level'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors = Counter(
            'application_errors_total',
            'Total application errors',
            ['error_type', 'component', 'severity'],
            registry=self.registry
        )
        
        # System info
        self.system_info = Info(
            'system_info',
            'System information',
            registry=self.registry
        )
        
    def track_correlation_id(self, correlation_id: str):
        """Track a correlation ID with timestamp."""
        self._correlation_ids[correlation_id] = time.time()
        
    def get_correlation_id_age(self, correlation_id: str) -> Optional[float]:
        """Get age of correlation ID in seconds."""
        if correlation_id in self._correlation_ids:
            return time.time() - self._correlation_ids[correlation_id]
        return None
        
    def cleanup_old_correlation_ids(self, max_age_seconds: float = 3600):
        """Remove correlation IDs older than max_age_seconds."""
        current_time = time.time()
        self._correlation_ids = {
            cid: timestamp 
            for cid, timestamp in self._correlation_ids.items()
            if current_time - timestamp < max_age_seconds
        }
        
    # Context managers for timing
    @asynccontextmanager
    async def measure_inference_latency(self, model_type: str, agent_name: str, 
                                      correlation_id: str):
        """Context manager to measure inference latency."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.inference_latency.labels(
                model_type=model_type,
                agent_name=agent_name,
                correlation_id=correlation_id
            ).observe(duration)
            
    @asynccontextmanager
    async def measure_matrix_processing(self, matrix_type: str):
        """Context manager to measure matrix processing latency."""
        start_time = time.time()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            self.matrix_processing_latency.labels(
                matrix_type=matrix_type
            ).observe(duration)
            self.matrix_processing_success.labels(
                matrix_type=matrix_type,
                status='success' if success else 'failure'
            ).inc()
            
    # Decorator for HTTP request tracking
    def track_http_request(self, method: str, endpoint: str):
        """Decorator to track HTTP requests."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status_code = 500
                try:
                    result = await func(*args, **kwargs)
                    # Extract status code from response if available
                    if hasattr(result, 'status_code'):
                        status_code = result.status_code
                    else:
                        status_code = 200
                    return result
                except Exception as e:
                    if hasattr(e, 'status_code'):
                        status_code = e.status_code
                    raise
                finally:
                    duration = time.time() - start_time
                    self.http_requests.labels(
                        method=method,
                        endpoint=endpoint,
                        status_code=str(status_code)
                    ).inc()
                    self.http_request_duration.labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
            return wrapper
        return decorator
        
    # Update methods for various metrics
    def update_trade_pnl(self, pnl: float, strategy: str, symbol: str, 
                        correlation_id: str):
        """Update trade PnL metric."""
        self.trade_pnl.labels(
            strategy=strategy,
            symbol=symbol,
            correlation_id=correlation_id
        ).set(pnl)
        
    def update_model_confidence(self, confidence: float, model_type: str, 
                               agent_name: str):
        """Update model confidence metric."""
        self.model_confidence.labels(
            model_type=model_type,
            agent_name=agent_name
        ).set(confidence)
        
    def record_synergy_response(self, synergy_type: str, status: str, 
                               correlation_id: str):
        """Record a synergy response event."""
        self.synergy_response_rate.labels(
            synergy_type=synergy_type,
            response_status=status,
            correlation_id=correlation_id
        ).inc()
        
    def update_data_pipeline_status(self, stage: str, lag_seconds: float):
        """Update data pipeline metrics."""
        self.data_pipeline_lag.labels(pipeline_stage=stage).set(lag_seconds)
        self.data_pipeline_last_update.labels(
            pipeline_stage=stage
        ).set(time.time())
        
    def record_error(self, error_type: str, component: str, 
                    severity: str = 'error'):
        """Record an application error."""
        self.errors.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()
        
    def update_system_info(self, info_dict: Dict[str, str]):
        """Update system information."""
        self.system_info.info(info_dict)
        
    def get_metrics(self) -> bytes:
        """Generate metrics in Prometheus format."""
        self.cleanup_old_correlation_ids()
        return generate_latest(self.registry)
        
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary for internal use."""
        metrics = {}
        for collector in self.registry._collector_to_names:
            for metric in collector.collect():
                metrics[metric.name] = {
                    'samples': [
                        {
                            'labels': dict(sample.labels),
                            'value': sample.value
                        }
                        for sample in metric.samples
                    ],
                    'type': metric.type,
                    'documentation': metric.documentation
                }
        return metrics
        
# Global metrics instance
metrics_exporter = MetricsExporter(metrics_registry)

# Convenience functions
def get_metrics() -> bytes:
    """Get Prometheus metrics."""
    return metrics_exporter.get_metrics()

def get_metrics_content_type() -> str:
    """Get Prometheus metrics content type."""
    return CONTENT_TYPE_LATEST