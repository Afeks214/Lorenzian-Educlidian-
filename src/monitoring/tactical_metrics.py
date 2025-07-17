"""
Tactical Metrics Exporter for High-Frequency Trading System

Provides ultra-fine-grained Prometheus metrics specifically designed for
sub-100ms latency requirements of the tactical MARL system.
"""

import time
import asyncio
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager
from functools import wraps
import logging
from collections import defaultdict, deque
import numpy as np

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server
)

logger = logging.getLogger(__name__)

class TacticalMetricsExporter:
    """
    High-frequency metrics exporter for tactical MARL system.
    
    Designed for sub-100ms latency monitoring with ultra-fine granularity.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, port: int = 9091):
        """Initialize tactical metrics with custom registry."""
        self.registry = registry or CollectorRegistry()
        self.port = port
        self._correlation_ids: Dict[str, float] = {}
        self._latency_buffer = deque(maxlen=1000)  # Rolling window for analysis
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize all tactical-specific Prometheus metrics."""
        
        # === LATENCY METRICS (Ultra-fine granularity) ===
        self.tactical_inference_latency = Histogram(
            'tactical_inference_latency_seconds',
            'Tactical model inference latency in seconds',
            ['model_type', 'agent_name', 'correlation_id'],
            # Ultra-fine buckets for sub-100ms monitoring
            buckets=[
                0.001,   # 1ms
                0.005,   # 5ms
                0.01,    # 10ms
                0.025,   # 25ms
                0.05,    # 50ms
                0.075,   # 75ms
                0.1,     # 100ms (target threshold)
                0.25,    # 250ms
                0.5,     # 500ms
                1.0,     # 1s
                float('inf')
            ],
            registry=self.registry
        )
        
        self.tactical_decision_latency = Histogram(
            'tactical_decision_latency_seconds',
            'End-to-end decision latency from event to execution',
            ['synergy_type', 'decision_type'],
            buckets=[
                0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, float('inf')
            ],
            registry=self.registry
        )
        
        self.tactical_pipeline_latency = Histogram(
            'tactical_pipeline_latency_seconds',
            'Individual pipeline component latency',
            ['component', 'operation'],
            buckets=[
                0.0001,  # 0.1ms
                0.0005,  # 0.5ms
                0.001,   # 1ms
                0.005,   # 5ms
                0.01,    # 10ms
                0.025,   # 25ms
                0.05,    # 50ms
                0.1,     # 100ms
                float('inf')
            ],
            registry=self.registry
        )
        
        # === BUSINESS METRICS ===
        self.tactical_active_positions = Gauge(
            'tactical_active_positions',
            'Number of currently active tactical positions',
            ['symbol', 'strategy_type'],
            registry=self.registry
        )
        
        self.tactical_model_confidence = Histogram(
            'tactical_model_confidence',
            'Model confidence scores from decision aggregation',
            ['agent_id', 'decision_type'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.tactical_agent_accuracy = Gauge(
            'tactical_agent_accuracy',
            'Current accuracy rate of each tactical agent',
            ['agent_id', 'timeframe'],
            registry=self.registry
        )
        
        self.tactical_trade_pnl = Histogram(
            'tactical_trade_pnl_dollars',
            'P&L per tactical trade in dollars',
            ['agent_consensus', 'synergy_type'],
            buckets=[-500, -100, -50, -25, -10, -5, 0, 5, 10, 25, 50, 100, 500],
            registry=self.registry
        )
        
        # === EVENT PROCESSING METRICS ===
        self.tactical_events_processed = Counter(
            'tactical_events_processed_total',
            'Total number of tactical events processed',
            ['event_type', 'source', 'status'],
            registry=self.registry
        )
        
        self.tactical_synergy_responses = Counter(
            'tactical_synergy_responses_total',
            'Responses to SYNERGY_DETECTED events',
            ['synergy_type', 'response_action', 'confidence_level'],
            registry=self.registry
        )
        
        self.tactical_decision_consensus = Counter(
            'tactical_decision_consensus_total',
            'Agent consensus in decision making',
            ['consensus_type', 'agent_count', 'execution_status'],
            registry=self.registry
        )
        
        # === PERFORMANCE METRICS ===
        self.tactical_throughput = Gauge(
            'tactical_throughput_decisions_per_minute',
            'Tactical decision throughput rate',
            registry=self.registry
        )
        
        self.tactical_queue_size = Gauge(
            'tactical_queue_size',
            'Size of tactical processing queues',
            ['queue_type', 'priority'],
            registry=self.registry
        )
        
        self.tactical_memory_usage = Gauge(
            'tactical_memory_usage_bytes',
            'Memory usage by tactical components',
            ['component', 'allocation_type'],
            registry=self.registry
        )
        
        # === REDIS STREAM METRICS ===
        self.tactical_stream_lag = Gauge(
            'tactical_stream_lag_seconds',
            'Consumer lag for Redis streams',
            ['stream_name', 'consumer_group'],
            registry=self.registry
        )
        
        self.tactical_stream_pending = Gauge(
            'tactical_stream_pending_messages',
            'Number of pending messages in Redis streams',
            ['stream_name', 'consumer_group'],
            registry=self.registry
        )
        
        self.tactical_stream_throughput = Counter(
            'tactical_stream_throughput_total',
            'Messages processed from Redis streams',
            ['stream_name', 'consumer_group', 'status'],
            registry=self.registry
        )
        
        # === ERROR METRICS ===
        self.tactical_errors = Counter(
            'tactical_errors_total',
            'Tactical system errors',
            ['error_type', 'component', 'severity'],
            registry=self.registry
        )
        
        self.tactical_timeouts = Counter(
            'tactical_timeouts_total',
            'Tactical operation timeouts',
            ['operation', 'timeout_threshold_ms'],
            registry=self.registry
        )
        
        # === SYSTEM INFO ===
        self.tactical_system_info = Info(
            'tactical_system_info',
            'Tactical system information',
            registry=self.registry
        )
        
        # === REAL-TIME PERFORMANCE TRACKING ===
        self.performance_window = deque(maxlen=100)  # Last 100 operations
        self.throughput_tracker = defaultdict(list)
        
        logger.info("Tactical metrics exporter initialized with ultra-fine granularity")
    
    def start_metrics_server(self):
        """Start the Prometheus metrics HTTP server."""
        try:
            start_http_server(self.port, registry=self.registry)
            logger.info(f"Tactical metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    @asynccontextmanager
    async def measure_inference_latency(self, model_type: str, agent_name: str, 
                                      correlation_id: str = ""):
        """Context manager for measuring inference latency."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.tactical_inference_latency.labels(
                model_type=model_type,
                agent_name=agent_name,
                correlation_id=correlation_id
            ).observe(duration)
            
            # Track for performance analysis
            self._latency_buffer.append(duration)
            
            # Alert if exceeding thresholds
            if duration > 0.1:  # 100ms
                self.tactical_errors.labels(
                    error_type="latency_exceeded",
                    component="inference",
                    severity="critical"
                ).inc()
                logger.warning(f"Inference latency exceeded 100ms: {duration*1000:.2f}ms")
            elif duration > 0.05:  # 50ms
                self.tactical_errors.labels(
                    error_type="latency_exceeded",
                    component="inference",
                    severity="warning"
                ).inc()
    
    @asynccontextmanager
    async def measure_decision_latency(self, synergy_type: str, decision_type: str):
        """Context manager for measuring end-to-end decision latency."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.tactical_decision_latency.labels(
                synergy_type=synergy_type,
                decision_type=decision_type
            ).observe(duration)
            
            # Performance tracking
            self.performance_window.append({
                'timestamp': time.time(),
                'duration': duration,
                'type': 'decision'
            })
    
    @asynccontextmanager
    async def measure_pipeline_component(self, component: str, operation: str):
        """Context manager for measuring pipeline component latency."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.tactical_pipeline_latency.labels(
                component=component,
                operation=operation
            ).observe(duration)
    
    def record_synergy_response(self, synergy_type: str, response_action: str, 
                              confidence: float):
        """Record a response to a SYNERGY_DETECTED event."""
        confidence_level = self._categorize_confidence(confidence)
        self.tactical_synergy_responses.labels(
            synergy_type=synergy_type,
            response_action=response_action,
            confidence_level=confidence_level
        ).inc()
        
        # Update confidence histogram
        self.tactical_model_confidence.labels(
            agent_id="ensemble",
            decision_type=response_action
        ).observe(confidence)
    
    def update_active_positions(self, symbol: str, strategy_type: str, count: int):
        """Update active position count."""
        self.tactical_active_positions.labels(
            symbol=symbol,
            strategy_type=strategy_type
        ).set(count)
    
    def record_trade_pnl(self, pnl: float, agent_consensus: str, synergy_type: str):
        """Record trade P&L."""
        self.tactical_trade_pnl.labels(
            agent_consensus=agent_consensus,
            synergy_type=synergy_type
        ).observe(pnl)
    
    def update_agent_accuracy(self, agent_id: str, accuracy: float, timeframe: str = "1h"):
        """Update agent accuracy metrics."""
        self.tactical_agent_accuracy.labels(
            agent_id=agent_id,
            timeframe=timeframe
        ).set(accuracy)
    
    def update_stream_metrics(self, stream_name: str, consumer_group: str, 
                            lag_seconds: float, pending_count: int):
        """Update Redis stream metrics."""
        self.tactical_stream_lag.labels(
            stream_name=stream_name,
            consumer_group=consumer_group
        ).set(lag_seconds)
        
        self.tactical_stream_pending.labels(
            stream_name=stream_name,
            consumer_group=consumer_group
        ).set(pending_count)
    
    def record_stream_processing(self, stream_name: str, consumer_group: str, 
                               status: str = "success"):
        """Record stream message processing."""
        self.tactical_stream_throughput.labels(
            stream_name=stream_name,
            consumer_group=consumer_group,
            status=status
        ).inc()
    
    def record_error(self, error_type: str, component: str, severity: str = "error"):
        """Record system error."""
        self.tactical_errors.labels(
            error_type=error_type,
            component=component,
            severity=severity
        ).inc()
    
    def record_timeout(self, operation: str, timeout_threshold_ms: float):
        """Record operation timeout."""
        self.tactical_timeouts.labels(
            operation=operation,
            timeout_threshold_ms=str(int(timeout_threshold_ms))
        ).inc()
    
    def update_throughput(self):
        """Update throughput metrics based on recent activity."""
        current_time = time.time()
        
        # Calculate decisions per minute
        recent_decisions = [
            op for op in self.performance_window 
            if current_time - op['timestamp'] < 60 and op['type'] == 'decision'
        ]
        
        throughput = len(recent_decisions)
        self.tactical_throughput.set(throughput)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self._latency_buffer:
            return {"status": "no_data"}
        
        latencies = list(self._latency_buffer)
        
        return {
            "latency_stats": {
                "p50": np.percentile(latencies, 50) * 1000,  # ms
                "p95": np.percentile(latencies, 95) * 1000,  # ms
                "p99": np.percentile(latencies, 99) * 1000,  # ms
                "mean": np.mean(latencies) * 1000,  # ms
                "max": np.max(latencies) * 1000,  # ms
            },
            "target_compliance": {
                "under_50ms": sum(1 for l in latencies if l < 0.05) / len(latencies),
                "under_100ms": sum(1 for l in latencies if l < 0.1) / len(latencies),
            },
            "sample_size": len(latencies)
        }
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence score."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def update_system_info(self, info_dict: Dict[str, str]):
        """Update system information."""
        self.tactical_system_info.info(info_dict)
    
    def get_metrics(self) -> bytes:
        """Generate metrics in Prometheus format."""
        # Update real-time calculations
        self.update_throughput()
        return generate_latest(self.registry)
    
    def get_metrics_content_type(self) -> str:
        """Get Prometheus metrics content type."""
        return CONTENT_TYPE_LATEST

# Global tactical metrics instance
tactical_metrics = TacticalMetricsExporter()

# Convenience functions
def get_tactical_metrics() -> bytes:
    """Get tactical Prometheus metrics."""
    return tactical_metrics.get_metrics()

def get_tactical_metrics_content_type() -> str:
    """Get tactical metrics content type."""
    return tactical_metrics.get_metrics_content_type()

def start_tactical_metrics_server(port: int = 9091):
    """Start tactical metrics server."""
    tactical_metrics.start_metrics_server()
    return tactical_metrics