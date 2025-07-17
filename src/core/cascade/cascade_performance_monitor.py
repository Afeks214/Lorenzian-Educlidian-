"""
Cascade Performance Monitor - Real-time performance monitoring for cascade flows

This module provides comprehensive performance monitoring for the cascade system,
ensuring end-to-end latency remains under 100ms and detecting performance degradation.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
import statistics
import structlog
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np

from ..events import EventBus, Event, EventType
from ..performance.performance_monitor import PerformanceMonitor
from .superposition_cascade_manager import SuperpositionPacket, SuperpositionType, CascadeMetrics


class PerformanceAlert(Enum):
    """Performance alert levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "LATENCY"
    THROUGHPUT = "THROUGHPUT"
    ERROR_RATE = "ERROR_RATE"
    QUEUE_DEPTH = "QUEUE_DEPTH"
    SYSTEM_LOAD = "SYSTEM_LOAD"
    MEMORY_USAGE = "MEMORY_USAGE"
    CPU_USAGE = "CPU_USAGE"


@dataclass
class PerformanceMetric:
    """Single performance metric measurement"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    system_id: str
    packet_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    window_size: int = 100  # Number of samples to consider
    evaluation_method: str = "average"  # average, percentile, max


@dataclass
class CascadePerformanceReport:
    """Comprehensive performance report"""
    report_id: str
    timestamp: datetime
    end_to_end_latency_ms: float
    system_latencies: Dict[str, float]
    throughput_metrics: Dict[str, float]
    error_rates: Dict[str, float]
    queue_depths: Dict[str, int]
    system_loads: Dict[str, float]
    performance_score: float
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    trend_analysis: Dict[str, Any]


class CascadePerformanceMonitor:
    """
    Advanced performance monitoring system for cascade operations.
    Ensures end-to-end latency <100ms and provides real-time performance insights.
    """

    def __init__(
        self,
        event_bus: EventBus,
        performance_monitor: Optional[PerformanceMonitor] = None,
        target_end_to_end_latency_ms: float = 100.0,
        sampling_interval: float = 0.1,  # 100ms
        report_interval: float = 5.0,    # 5 seconds
        history_size: int = 10000
    ):
        self.event_bus = event_bus
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.target_end_to_end_latency_ms = target_end_to_end_latency_ms
        self.sampling_interval = sampling_interval
        self.report_interval = report_interval
        self.history_size = history_size
        
        # State management
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Metrics storage
        self._metrics_history: Dict[str, deque] = {}
        self._packet_traces: Dict[str, List[PerformanceMetric]] = {}
        self._system_metrics: Dict[str, Dict[MetricType, deque]] = {}
        
        # Performance thresholds
        self._thresholds: Dict[MetricType, PerformanceThreshold] = {}
        
        # Alerting system
        self._alert_handlers: List[Callable] = []
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        
        # Real-time monitoring
        self._current_flows: Dict[str, Dict[str, Any]] = {}
        self._flow_start_times: Dict[str, float] = {}
        self._system_performance_cache: Dict[str, float] = {}
        
        # Analysis components
        self._performance_analyzer = PerformanceAnalyzer()
        self._trend_analyzer = TrendAnalyzer()
        self._anomaly_detector = AnomalyDetector()
        
        # Threading
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize monitor
        self._initialize_performance_monitor()
        
    def _initialize_performance_monitor(self) -> None:
        """Initialize the performance monitoring system"""
        try:
            # Initialize default thresholds
            self._initialize_default_thresholds()
            
            # Initialize metrics storage
            self._initialize_metrics_storage()
            
            # Start background monitoring tasks
            self._start_monitoring_tasks()
            
            # Register event handlers
            self._register_event_handlers()
            
            self.logger.info("Cascade performance monitor initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitor: {e}")
            raise
            
    def _initialize_default_thresholds(self) -> None:
        """Initialize default performance thresholds"""
        self._thresholds = {
            MetricType.LATENCY: PerformanceThreshold(
                metric_type=MetricType.LATENCY,
                warning_threshold=70.0,   # 70ms
                critical_threshold=85.0,  # 85ms
                emergency_threshold=100.0, # 100ms
                window_size=100,
                evaluation_method="percentile_95"
            ),
            MetricType.THROUGHPUT: PerformanceThreshold(
                metric_type=MetricType.THROUGHPUT,
                warning_threshold=500.0,   # 500 packets/sec
                critical_threshold=300.0,  # 300 packets/sec
                emergency_threshold=100.0, # 100 packets/sec
                window_size=50,
                evaluation_method="average"
            ),
            MetricType.ERROR_RATE: PerformanceThreshold(
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=0.01,   # 1%
                critical_threshold=0.05,  # 5%
                emergency_threshold=0.10, # 10%
                window_size=100,
                evaluation_method="average"
            ),
            MetricType.QUEUE_DEPTH: PerformanceThreshold(
                metric_type=MetricType.QUEUE_DEPTH,
                warning_threshold=500,     # 500 packets
                critical_threshold=800,    # 800 packets
                emergency_threshold=1000,  # 1000 packets
                window_size=20,
                evaluation_method="max"
            ),
            MetricType.SYSTEM_LOAD: PerformanceThreshold(
                metric_type=MetricType.SYSTEM_LOAD,
                warning_threshold=0.7,    # 70%
                critical_threshold=0.8,   # 80%
                emergency_threshold=0.9,  # 90%
                window_size=20,
                evaluation_method="average"
            )
        }
        
    def _initialize_metrics_storage(self) -> None:
        """Initialize metrics storage structures"""
        metric_types = [
            "end_to_end_latency",
            "strategic_latency",
            "tactical_latency", 
            "risk_latency",
            "execution_latency",
            "throughput",
            "error_rate",
            "queue_depth",
            "system_load"
        ]
        
        for metric_type in metric_types:
            self._metrics_history[metric_type] = deque(maxlen=self.history_size)
            
        # Initialize per-system metrics
        systems = ["strategic", "tactical", "risk", "execution"]
        for system_id in systems:
            self._system_metrics[system_id] = {}
            for metric_type in MetricType:
                self._system_metrics[system_id][metric_type] = deque(maxlen=self.history_size)
                
    def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks"""
        threading.Thread(target=self._continuous_monitor, daemon=True).start()
        threading.Thread(target=self._performance_analyzer_task, daemon=True).start()
        threading.Thread(target=self._alert_processor, daemon=True).start()
        threading.Thread(target=self._report_generator, daemon=True).start()
        threading.Thread(target=self._cleanup_task, daemon=True).start()
        
    def _register_event_handlers(self) -> None:
        """Register event handlers for monitoring"""
        self.event_bus.subscribe(EventType.SYSTEM_START, self._handle_system_event)
        self.event_bus.subscribe(EventType.SYSTEM_SHUTDOWN, self._handle_system_event)
        self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_event)
        
    def track_packet_start(self, packet: SuperpositionPacket) -> None:
        """Track the start of packet processing"""
        with self._lock:
            self._flow_start_times[packet.packet_id] = time.time()
            self._current_flows[packet.packet_id] = {
                "packet": packet,
                "start_time": time.time(),
                "system_times": {},
                "current_system": None
            }
            
    def track_system_entry(self, packet_id: str, system_id: str) -> None:
        """Track packet entry into a system"""
        with self._lock:
            if packet_id in self._current_flows:
                flow = self._current_flows[packet_id]
                flow["system_times"][system_id] = {"entry": time.time()}
                flow["current_system"] = system_id
                
    def track_system_exit(self, packet_id: str, system_id: str, success: bool = True) -> None:
        """Track packet exit from a system"""
        with self._lock:
            if packet_id in self._current_flows:
                flow = self._current_flows[packet_id]
                if system_id in flow["system_times"]:
                    system_time = flow["system_times"][system_id]
                    system_time["exit"] = time.time()
                    system_time["duration"] = system_time["exit"] - system_time["entry"]
                    system_time["success"] = success
                    
                    # Record system-level metric
                    self._record_system_metric(
                        system_id, 
                        MetricType.LATENCY,
                        system_time["duration"] * 1000  # Convert to ms
                    )
                    
    def track_packet_completion(self, packet_id: str, success: bool = True) -> None:
        """Track completion of packet processing"""
        with self._lock:
            if packet_id in self._current_flows and packet_id in self._flow_start_times:
                flow = self._current_flows[packet_id]
                end_time = time.time()
                
                # Calculate end-to-end latency
                total_latency = (end_time - flow["start_time"]) * 1000  # Convert to ms
                
                # Record end-to-end metric
                self._record_metric("end_to_end_latency", total_latency)
                
                # Create performance trace
                self._create_performance_trace(packet_id, flow, total_latency, success)
                
                # Check for performance violations
                self._check_performance_violations(packet_id, flow, total_latency)
                
                # Cleanup
                del self._current_flows[packet_id]
                del self._flow_start_times[packet_id]
                
    def _record_metric(self, metric_name: str, value: float) -> None:
        """Record a performance metric"""
        if metric_name in self._metrics_history:
            self._metrics_history[metric_name].append({
                "value": value,
                "timestamp": time.time()
            })
            
    def _record_system_metric(self, system_id: str, metric_type: MetricType, value: float) -> None:
        """Record a system-specific metric"""
        if system_id in self._system_metrics:
            self._system_metrics[system_id][metric_type].append({
                "value": value,
                "timestamp": time.time()
            })
            
    def _create_performance_trace(
        self, 
        packet_id: str, 
        flow: Dict[str, Any], 
        total_latency: float, 
        success: bool
    ) -> None:
        """Create detailed performance trace for packet"""
        trace = {
            "packet_id": packet_id,
            "total_latency_ms": total_latency,
            "success": success,
            "start_time": flow["start_time"],
            "end_time": time.time(),
            "system_breakdown": {},
            "packet_info": {
                "type": flow["packet"].packet_type.value,
                "source": flow["packet"].source_system,
                "priority": flow["packet"].priority
            }
        }
        
        # Add system-level breakdown
        for system_id, system_time in flow["system_times"].items():
            if "duration" in system_time:
                trace["system_breakdown"][system_id] = {
                    "duration_ms": system_time["duration"] * 1000,
                    "success": system_time.get("success", True)
                }
                
        # Store trace
        self._packet_traces[packet_id] = trace
        
        # Keep only recent traces
        if len(self._packet_traces) > 1000:
            oldest_keys = sorted(self._packet_traces.keys())[:100]
            for key in oldest_keys:
                del self._packet_traces[key]
                
    def _check_performance_violations(
        self, 
        packet_id: str, 
        flow: Dict[str, Any], 
        total_latency: float
    ) -> None:
        """Check for performance threshold violations"""
        # Check end-to-end latency
        if total_latency > self.target_end_to_end_latency_ms:
            self._trigger_alert(
                PerformanceAlert.CRITICAL,
                f"End-to-end latency exceeded: {total_latency:.2f}ms > {self.target_end_to_end_latency_ms}ms",
                {
                    "packet_id": packet_id,
                    "latency_ms": total_latency,
                    "target_ms": self.target_end_to_end_latency_ms,
                    "system_breakdown": flow["system_times"]
                }
            )
            
        # Check system-level violations
        for system_id, system_time in flow["system_times"].items():
            if "duration" in system_time:
                system_latency = system_time["duration"] * 1000
                if system_latency > 30:  # 30ms threshold per system
                    self._trigger_alert(
                        PerformanceAlert.WARNING,
                        f"System latency high: {system_id} took {system_latency:.2f}ms",
                        {
                            "system_id": system_id,
                            "latency_ms": system_latency,
                            "packet_id": packet_id
                        }
                    )
                    
    def _trigger_alert(self, level: PerformanceAlert, message: str, context: Dict[str, Any]) -> None:
        """Trigger a performance alert"""
        alert = {
            "id": f"alert_{int(time.time() * 1000000)}",
            "level": level.value,
            "message": message,
            "context": context,
            "timestamp": datetime.now(),
            "acknowledged": False
        }
        
        # Store alert
        self._active_alerts[alert["id"]] = alert
        
        # Log alert
        if level == PerformanceAlert.EMERGENCY:
            self.logger.critical(f"PERFORMANCE EMERGENCY: {message}")
        elif level == PerformanceAlert.CRITICAL:
            self.logger.error(f"PERFORMANCE CRITICAL: {message}")
        elif level == PerformanceAlert.WARNING:
            self.logger.warning(f"PERFORMANCE WARNING: {message}")
        else:
            self.logger.info(f"PERFORMANCE INFO: {message}")
            
        # Notify alert handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
                
        # Publish alert event
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.SYSTEM_ERROR if level in [PerformanceAlert.CRITICAL, PerformanceAlert.EMERGENCY] else EventType.SYSTEM_START,
                alert,
                "performance_monitor"
            )
        )
        
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics"""
        with self._lock:
            current_time = time.time()
            
            # Calculate recent metrics
            recent_window = 30  # 30 seconds
            cutoff_time = current_time - recent_window
            
            # End-to-end latency
            recent_latencies = [
                m["value"] for m in self._metrics_history["end_to_end_latency"]
                if m["timestamp"] > cutoff_time
            ]
            
            # System latencies
            system_latencies = {}
            for system_id, metrics in self._system_metrics.items():
                recent_system_latencies = [
                    m["value"] for m in metrics[MetricType.LATENCY]
                    if m["timestamp"] > cutoff_time
                ]
                if recent_system_latencies:
                    system_latencies[system_id] = {
                        "average": statistics.mean(recent_system_latencies),
                        "p95": np.percentile(recent_system_latencies, 95),
                        "max": max(recent_system_latencies)
                    }
                    
            # Current flows
            active_flows = len(self._current_flows)
            
            # Throughput (packets per second)
            throughput = len(recent_latencies) / recent_window if recent_latencies else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "end_to_end_latency": {
                    "average": statistics.mean(recent_latencies) if recent_latencies else 0,
                    "p95": np.percentile(recent_latencies, 95) if recent_latencies else 0,
                    "max": max(recent_latencies) if recent_latencies else 0,
                    "target": self.target_end_to_end_latency_ms
                },
                "system_latencies": system_latencies,
                "throughput": throughput,
                "active_flows": active_flows,
                "performance_health": self._calculate_performance_health(),
                "active_alerts": len(self._active_alerts),
                "critical_alerts": len([a for a in self._active_alerts.values() if a["level"] == "CRITICAL"])
            }
            
    def _calculate_performance_health(self) -> float:
        """Calculate overall performance health score"""
        factors = []
        
        # End-to-end latency factor
        recent_latencies = [
            m["value"] for m in self._metrics_history["end_to_end_latency"]
            if m["timestamp"] > time.time() - 60  # Last minute
        ]
        
        if recent_latencies:
            avg_latency = statistics.mean(recent_latencies)
            latency_factor = max(0, 100 - (avg_latency / self.target_end_to_end_latency_ms * 100))
            factors.append(latency_factor)
            
        # Alert factor
        critical_alerts = len([a for a in self._active_alerts.values() if a["level"] in ["CRITICAL", "EMERGENCY"]])
        alert_factor = max(0, 100 - (critical_alerts * 20))
        factors.append(alert_factor)
        
        # System availability factor
        active_systems = len([s for s in self._system_metrics.keys() if self._is_system_active(s)])
        availability_factor = (active_systems / len(self._system_metrics)) * 100
        factors.append(availability_factor)
        
        return statistics.mean(factors) if factors else 0
        
    def _is_system_active(self, system_id: str) -> bool:
        """Check if system is active based on recent metrics"""
        if system_id not in self._system_metrics:
            return False
            
        recent_metrics = [
            m for m in self._system_metrics[system_id][MetricType.LATENCY]
            if m["timestamp"] > time.time() - 60
        ]
        
        return len(recent_metrics) > 0
        
    def generate_performance_report(self) -> CascadePerformanceReport:
        """Generate comprehensive performance report"""
        report_id = f"perf_report_{int(time.time() * 1000)}"
        
        with self._lock:
            # Calculate metrics
            real_time_metrics = self.get_real_time_metrics()
            
            # System latencies
            system_latencies = {}
            for system_id, latency_data in real_time_metrics["system_latencies"].items():
                system_latencies[system_id] = latency_data["average"]
                
            # Throughput metrics
            throughput_metrics = {
                "current_tps": real_time_metrics["throughput"],
                "peak_tps": self._calculate_peak_throughput(),
                "average_tps": self._calculate_average_throughput()
            }
            
            # Error rates
            error_rates = self._calculate_error_rates()
            
            # Queue depths
            queue_depths = self._get_current_queue_depths()
            
            # System loads
            system_loads = self._get_system_loads()
            
            # Performance score
            performance_score = real_time_metrics["performance_health"]
            
            # Active alerts
            alerts = [
                {
                    "id": alert["id"],
                    "level": alert["level"],
                    "message": alert["message"],
                    "timestamp": alert["timestamp"].isoformat()
                }
                for alert in self._active_alerts.values()
            ]
            
            # Recommendations
            recommendations = self._generate_recommendations(real_time_metrics)
            
            # Trend analysis
            trend_analysis = self._trend_analyzer.analyze_trends(self._metrics_history)
            
            return CascadePerformanceReport(
                report_id=report_id,
                timestamp=datetime.now(),
                end_to_end_latency_ms=real_time_metrics["end_to_end_latency"]["average"],
                system_latencies=system_latencies,
                throughput_metrics=throughput_metrics,
                error_rates=error_rates,
                queue_depths=queue_depths,
                system_loads=system_loads,
                performance_score=performance_score,
                alerts=alerts,
                recommendations=recommendations,
                trend_analysis=trend_analysis
            )
            
    def _calculate_peak_throughput(self) -> float:
        """Calculate peak throughput over recent period"""
        # Implementation for peak throughput calculation
        return 0.0
        
    def _calculate_average_throughput(self) -> float:
        """Calculate average throughput over recent period"""
        # Implementation for average throughput calculation
        return 0.0
        
    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate error rates by system"""
        # Implementation for error rate calculation
        return {}
        
    def _get_current_queue_depths(self) -> Dict[str, int]:
        """Get current queue depths"""
        # Implementation for queue depth retrieval
        return {}
        
    def _get_system_loads(self) -> Dict[str, float]:
        """Get current system loads"""
        # Implementation for system load retrieval
        return {}
        
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Latency recommendations
        if metrics["end_to_end_latency"]["average"] > self.target_end_to_end_latency_ms * 0.8:
            recommendations.append("Consider optimizing slowest system components")
            
        # Throughput recommendations
        if metrics["throughput"] < 100:
            recommendations.append("Low throughput detected - investigate bottlenecks")
            
        # Alert recommendations
        if metrics["critical_alerts"] > 0:
            recommendations.append("Address critical performance alerts immediately")
            
        return recommendations
        
    def add_alert_handler(self, handler: Callable) -> None:
        """Add alert handler"""
        self._alert_handlers.append(handler)
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id]["acknowledged"] = True
            return True
        return False
        
    def get_packet_trace(self, packet_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed trace for a packet"""
        return self._packet_traces.get(packet_id)
        
    def get_system_performance(self, system_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific system"""
        if system_id not in self._system_metrics:
            return {}
            
        metrics = self._system_metrics[system_id]
        recent_time = time.time() - 300  # 5 minutes
        
        result = {}
        for metric_type, data in metrics.items():
            recent_data = [m["value"] for m in data if m["timestamp"] > recent_time]
            if recent_data:
                result[metric_type.value] = {
                    "average": statistics.mean(recent_data),
                    "min": min(recent_data),
                    "max": max(recent_data),
                    "count": len(recent_data)
                }
                
        return result
        
    # Background tasks
    def _continuous_monitor(self) -> None:
        """Continuous monitoring task"""
        while not self._shutdown_event.is_set():
            try:
                # Monitor current flows for stale packets
                self._monitor_stale_flows()
                
                # Check thresholds
                self._check_thresholds()
                
                # Update system performance cache
                self._update_system_performance_cache()
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in continuous monitor: {e}")
                
    def _monitor_stale_flows(self) -> None:
        """Monitor for stale flows that haven't completed"""
        current_time = time.time()
        stale_threshold = 30  # 30 seconds
        
        stale_flows = []
        with self._lock:
            for packet_id, flow in self._current_flows.items():
                if current_time - flow["start_time"] > stale_threshold:
                    stale_flows.append(packet_id)
                    
        for packet_id in stale_flows:
            self._trigger_alert(
                PerformanceAlert.WARNING,
                f"Stale flow detected: packet {packet_id} processing for {current_time - self._current_flows[packet_id]['start_time']:.1f}s",
                {"packet_id": packet_id}
            )
            
    def _check_thresholds(self) -> None:
        """Check performance thresholds"""
        for metric_type, threshold in self._thresholds.items():
            self._evaluate_threshold(metric_type, threshold)
            
    def _evaluate_threshold(self, metric_type: MetricType, threshold: PerformanceThreshold) -> None:
        """Evaluate a specific threshold"""
        # Implementation for threshold evaluation
        pass
        
    def _update_system_performance_cache(self) -> None:
        """Update system performance cache"""
        # Implementation for performance cache update
        pass
        
    def _performance_analyzer_task(self) -> None:
        """Performance analysis task"""
        while not self._shutdown_event.is_set():
            try:
                # Run performance analysis
                self._performance_analyzer.analyze(self._metrics_history)
                time.sleep(10)  # Run every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in performance analyzer: {e}")
                
    def _alert_processor(self) -> None:
        """Alert processing task"""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old alerts
                self._cleanup_old_alerts()
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Error in alert processor: {e}")
                
    def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        old_alerts = [
            alert_id for alert_id, alert in self._active_alerts.items()
            if alert["timestamp"] < cutoff_time and alert["acknowledged"]
        ]
        
        for alert_id in old_alerts:
            del self._active_alerts[alert_id]
            
    def _report_generator(self) -> None:
        """Report generation task"""
        while not self._shutdown_event.is_set():
            try:
                # Generate periodic report
                report = self.generate_performance_report()
                
                # Log report summary
                self.logger.info(
                    f"Performance Report: {report.end_to_end_latency_ms:.2f}ms latency, "
                    f"{report.performance_score:.1f}% health, {len(report.alerts)} alerts"
                )
                
                time.sleep(self.report_interval)
                
            except Exception as e:
                self.logger.error(f"Error in report generator: {e}")
                
    def _cleanup_task(self) -> None:
        """Cleanup task"""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old traces
                self._cleanup_old_traces()
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                
    def _cleanup_old_traces(self) -> None:
        """Clean up old packet traces"""
        if len(self._packet_traces) > 1000:
            # Keep only most recent 1000 traces
            sorted_traces = sorted(
                self._packet_traces.items(),
                key=lambda x: x[1]["end_time"],
                reverse=True
            )
            
            self._packet_traces = dict(sorted_traces[:1000])
            
    # Event handlers
    def _handle_system_event(self, event: Event) -> None:
        """Handle system events"""
        self.logger.info(f"System event received: {event.event_type}")
        
    def _handle_emergency_event(self, event: Event) -> None:
        """Handle emergency events"""
        self.logger.critical(f"Emergency event received: {event.event_type}")
        
        # Trigger emergency alert
        self._trigger_alert(
            PerformanceAlert.EMERGENCY,
            f"Emergency stop event: {event.payload}",
            {"event": event.payload}
        )
        
    def shutdown(self) -> None:
        """Shutdown performance monitor"""
        self.logger.info("Shutting down performance monitor")
        self._shutdown_event.set()
        self._executor.shutdown(wait=True)
        self.logger.info("Performance monitor shutdown complete")


class PerformanceAnalyzer:
    """Performance analysis component"""
    
    def analyze(self, metrics_history: Dict[str, deque]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        return {}


class TrendAnalyzer:
    """Trend analysis component"""
    
    def analyze_trends(self, metrics_history: Dict[str, deque]) -> Dict[str, Any]:
        """Analyze performance trends"""
        return {}


class AnomalyDetector:
    """Anomaly detection component"""
    
    def detect_anomalies(self, metrics: List[float]) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        return []