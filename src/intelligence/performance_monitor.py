"""
Intelligence Performance Monitor - Real-time performance tracking and optimization.

Monitors performance of intelligence components and provides optimization
recommendations to maintain <5ms total inference requirement.
"""

import time
import psutil
import torch
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
from collections import deque
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""
    timestamp: float
    total_inference_time_ms: float
    intelligence_overhead_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_memory_mb: float
    throughput_qps: float
    component_breakdown: Dict[str, float]

class IntelligencePerformanceMonitor:
    """
    Real-time performance monitoring for intelligence upgrades.
    
    Tracks:
    - Inference time breakdown by component
    - Memory usage patterns
    - Throughput and latency percentiles
    - Performance degradation detection
    - Optimization recommendations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance targets
        self.target_total_inference_ms = config.get('target_total_inference_ms', 5.0)
        self.target_intelligence_overhead_ms = config.get('target_intelligence_overhead_ms', 1.0)
        self.target_memory_mb = config.get('target_memory_mb', 512)
        
        # Performance history
        self.performance_history = deque(maxlen=1000)
        self.component_timings = {
            'attention': deque(maxlen=200),
            'gating': deque(maxlen=200),
            'regime_detection': deque(maxlen=200),
            'integration': deque(maxlen=200),
            'reward_computation': deque(maxlen=200)
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'inference_time_breach': 1.2,      # 20% over target
            'memory_breach': 1.3,              # 30% over target
            'performance_degradation': 0.8,    # 20% degradation from baseline
            'component_time_limit': 0.5        # Max time per component in ms
        }
        
        # Baseline performance (established during startup)
        self.baseline_metrics = None
        self.calibration_samples = []
        self.is_calibrated = False
        
        # Real-time monitoring
        self.monitoring_active = config.get('real_time_monitoring', True)
        self.monitoring_thread = None
        self.monitoring_interval = config.get('monitoring_interval_ms', 100)
        
        # Performance optimization tracking
        self.optimization_history = []
        self.last_optimization_time = 0
        self.optimization_cooldown = config.get('optimization_cooldown_seconds', 30)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Start real-time monitoring if enabled
        if self.monitoring_active:
            self._start_real_time_monitoring()
        
        self.logger.info("Intelligence Performance Monitor initialized")
    
    def start_performance_measurement(self) -> Dict[str, Any]:
        """Start a performance measurement session."""
        return {
            'start_time': time.perf_counter(),
            'start_memory': self._get_memory_usage(),
            'start_cpu': self._get_cpu_usage(),
            'component_times': {},
            'component_start_times': {}
        }
    
    def record_component_start(self, session: Dict[str, Any], component: str):
        """Record the start time for a specific component."""
        session['component_start_times'][component] = time.perf_counter()
    
    def record_component_end(self, session: Dict[str, Any], component: str):
        """Record the end time for a specific component."""
        if component in session['component_start_times']:
            start_time = session['component_start_times'][component]
            component_time = (time.perf_counter() - start_time) * 1000
            session['component_times'][component] = component_time
            
            # Update component timing history
            if component in self.component_timings:
                with self.lock:
                    self.component_timings[component].append(component_time)
            
            # Check for component performance alerts
            if component_time > self.alert_thresholds['component_time_limit']:
                self.logger.warning(
                    f"Component {component} took {component_time:.3f}ms, "
                    f"exceeds limit {self.alert_thresholds['component_time_limit']}ms"
                )
    
    def complete_performance_measurement(
        self, 
        session: Dict[str, Any],
        intelligence_result: Dict[str, Any]
    ) -> PerformanceSnapshot:
        """Complete performance measurement and create snapshot."""
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - session['start_time']) * 1000
        
        # Calculate intelligence overhead
        intelligence_overhead = sum(session['component_times'].values())
        
        # Memory and CPU usage
        end_memory = self._get_memory_usage()
        end_cpu = self._get_cpu_usage()
        
        # GPU memory if available
        gpu_memory = self._get_gpu_memory_usage()
        
        # Calculate throughput (decisions per second)
        throughput = 1000.0 / total_time_ms if total_time_ms > 0 else 0.0
        
        # Create performance snapshot
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            total_inference_time_ms=total_time_ms,
            intelligence_overhead_ms=intelligence_overhead,
            memory_usage_mb=end_memory,
            cpu_usage_percent=end_cpu,
            gpu_memory_mb=gpu_memory,
            throughput_qps=throughput,
            component_breakdown=session['component_times'].copy()
        )
        
        # Add to history with thread safety
        with self.lock:
            self.performance_history.append(snapshot)
        
        # Check for performance issues
        self._check_performance_alerts(snapshot)
        
        # Update calibration if needed
        if not self.is_calibrated:
            self._update_calibration(snapshot)
        
        # Add performance data to intelligence result
        intelligence_result['performance_snapshot'] = snapshot
        
        return snapshot
    
    def get_performance_summary(self, window_size: int = 100) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        with self.lock:
            if not self.performance_history:
                return {'status': 'no_data'}
            
            # Calculate statistics from recent history
            recent_snapshots = list(self.performance_history)[-window_size:]
        
        inference_times = [s.total_inference_time_ms for s in recent_snapshots]
        intelligence_overheads = [s.intelligence_overhead_ms for s in recent_snapshots]
        memory_usage = [s.memory_usage_mb for s in recent_snapshots]
        throughput = [s.throughput_qps for s in recent_snapshots]
        
        # Component timing statistics
        component_stats = {}
        with self.lock:
            for component, timings in self.component_timings.items():
                if timings:
                    timings_list = list(timings)
                    component_stats[component] = {
                        'mean_ms': np.mean(timings_list),
                        'p50_ms': np.percentile(timings_list, 50),
                        'p95_ms': np.percentile(timings_list, 95),
                        'p99_ms': np.percentile(timings_list, 99),
                        'std_ms': np.std(timings_list),
                        'count': len(timings_list)
                    }
        
        summary = {
            'total_inference_time': {
                'mean_ms': np.mean(inference_times),
                'p50_ms': np.percentile(inference_times, 50),
                'p95_ms': np.percentile(inference_times, 95),
                'p99_ms': np.percentile(inference_times, 99),
                'target_ms': self.target_total_inference_ms,
                'target_compliance': np.mean([t <= self.target_total_inference_ms for t in inference_times]),
                'trend': self._calculate_trend(inference_times)
            },
            'intelligence_overhead': {
                'mean_ms': np.mean(intelligence_overheads),
                'p95_ms': np.percentile(intelligence_overheads, 95),
                'p99_ms': np.percentile(intelligence_overheads, 99),
                'target_ms': self.target_intelligence_overhead_ms,
                'overhead_ratio': np.mean(intelligence_overheads) / np.mean(inference_times),
                'target_compliance': np.mean([t <= self.target_intelligence_overhead_ms for t in intelligence_overheads])
            },
            'memory_usage': {
                'mean_mb': np.mean(memory_usage),
                'max_mb': np.max(memory_usage),
                'min_mb': np.min(memory_usage),
                'target_mb': self.target_memory_mb,
                'target_compliance': np.mean([m <= self.target_memory_mb for m in memory_usage]),
                'trend': self._calculate_trend(memory_usage)
            },
            'throughput': {
                'mean_qps': np.mean(throughput),
                'min_qps': np.min(throughput),
                'max_qps': np.max(throughput),
                'target_qps': 1000.0 / self.target_total_inference_ms
            },
            'component_timings': component_stats,
            'performance_score': self._calculate_performance_score(recent_snapshots),
            'calibration_status': {
                'is_calibrated': self.is_calibrated,
                'baseline_available': self.baseline_metrics is not None,
                'samples_collected': len(self.calibration_samples)
            }
        }
        
        return summary
    
    def _check_performance_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance alerts and log warnings."""
        
        alerts = []
        
        # Check inference time
        if snapshot.total_inference_time_ms > self.target_total_inference_ms * self.alert_thresholds['inference_time_breach']:
            alerts.append(
                f"Inference time {snapshot.total_inference_time_ms:.2f}ms exceeds target "
                f"{self.target_total_inference_ms}ms by {self.alert_thresholds['inference_time_breach']*100-100:.0f}%"
            )
        
        # Check intelligence overhead
        if snapshot.intelligence_overhead_ms > self.target_intelligence_overhead_ms * self.alert_thresholds['inference_time_breach']:
            alerts.append(
                f"Intelligence overhead {snapshot.intelligence_overhead_ms:.2f}ms exceeds target "
                f"{self.target_intelligence_overhead_ms}ms"
            )
        
        # Check memory usage
        if snapshot.memory_usage_mb > self.target_memory_mb * self.alert_thresholds['memory_breach']:
            alerts.append(
                f"Memory usage {snapshot.memory_usage_mb:.1f}MB exceeds target "
                f"{self.target_memory_mb}MB by {self.alert_thresholds['memory_breach']*100-100:.0f}%"
            )
        
        # Check for performance degradation
        if self.baseline_metrics and len(self.performance_history) > 10:
            with self.lock:
                recent_avg_time = np.mean([s.total_inference_time_ms for s in list(self.performance_history)[-10:]])
            
            degradation_threshold = self.baseline_metrics['mean_inference_time'] * (2 - self.alert_thresholds['performance_degradation'])
            if recent_avg_time > degradation_threshold:
                alerts.append(
                    f"Performance degradation detected: {recent_avg_time:.2f}ms vs "
                    f"baseline {self.baseline_metrics['mean_inference_time']:.2f}ms"
                )
        
        # Check component-specific alerts
        for component, time_ms in snapshot.component_breakdown.items():
            if time_ms > self.alert_thresholds['component_time_limit']:
                alerts.append(f"Component {component} taking {time_ms:.2f}ms (limit: {self.alert_thresholds['component_time_limit']}ms)")
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"PERFORMANCE ALERT: {alert}")
        
        # Trigger optimization if needed
        if alerts and self._should_trigger_optimization():
            self._trigger_performance_optimization(alerts)
    
    def _calculate_performance_score(self, snapshots: List[PerformanceSnapshot]) -> float:
        """Calculate overall performance score (0-100)."""
        
        if not snapshots:
            return 0.0
        
        # Scoring factors
        inference_times = [s.total_inference_time_ms for s in snapshots]
        memory_usage = [s.memory_usage_mb for s in snapshots]
        intelligence_overheads = [s.intelligence_overhead_ms for s in snapshots]
        
        # Score components (0-1 each)
        time_score = min(1.0, self.target_total_inference_ms / np.mean(inference_times))
        memory_score = min(1.0, self.target_memory_mb / np.mean(memory_usage))
        overhead_score = min(1.0, self.target_intelligence_overhead_ms / np.mean(intelligence_overheads))
        
        # Consistency score (lower std dev = higher score)
        time_consistency = max(0.0, 1.0 - (np.std(inference_times) / np.mean(inference_times)))
        
        # Weighted overall score
        overall_score = (
            0.4 * time_score +
            0.2 * memory_score +
            0.3 * overhead_score +
            0.1 * time_consistency
        ) * 100
        
        return min(100.0, overall_score)
    
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        
        recommendations = []
        
        with self.lock:
            if not self.performance_history:
                return ["Insufficient performance data for recommendations"]
        
        summary = self.get_performance_summary()
        
        # Check inference time recommendations
        if summary['total_inference_time']['p99_ms'] > self.target_total_inference_ms:
            recommendations.append(
                f"Inference time optimization needed: P99 is {summary['total_inference_time']['p99_ms']:.2f}ms, "
                f"target is {self.target_total_inference_ms}ms"
            )
        
        # Check component-specific recommendations
        component_limits = {
            'attention': 0.2,
            'gating': 0.3,
            'regime_detection': 0.2,
            'integration': 0.2,
            'reward_computation': 0.1
        }
        
        for component, stats in summary['component_timings'].items():
            limit = component_limits.get(component, 0.3)
            if stats['p95_ms'] > limit:
                recommendations.append(
                    f"Optimize {component} component: P95 timing is {stats['p95_ms']:.2f}ms "
                    f"(recommended: <{limit}ms)"
                )
        
        # Memory recommendations
        if summary['memory_usage']['mean_mb'] > self.target_memory_mb * 0.8:
            recommendations.append(
                f"Memory optimization recommended: Current usage {summary['memory_usage']['mean_mb']:.1f}MB "
                f"approaching target {self.target_memory_mb}MB"
            )
        
        # Intelligence overhead recommendations
        if summary['intelligence_overhead']['overhead_ratio'] > 0.3:  # >30% overhead
            recommendations.append(
                f"Intelligence overhead is {summary['intelligence_overhead']['overhead_ratio']:.1%} of total time - "
                "consider simplifying attention/gating computations"
            )
        
        # Trend-based recommendations
        if summary['total_inference_time']['trend'] > 0.1:  # Increasing trend
            recommendations.append("Performance degradation trend detected - investigate recent changes")
        
        if summary['memory_usage']['trend'] > 0.1:  # Memory leak trend
            recommendations.append("Memory usage trend increasing - check for memory leaks")
        
        # Component balance recommendations
        component_times = {k: v['mean_ms'] for k, v in summary['component_timings'].items()}
        if component_times:
            max_component = max(component_times, key=component_times.get)
            max_time = component_times[max_component]
            total_component_time = sum(component_times.values())
            
            if max_time > total_component_time * 0.5:  # One component dominates
                recommendations.append(
                    f"Component {max_component} dominates processing time ({max_time:.2f}ms) - "
                    "consider optimizing or load balancing"
                )
        
        return recommendations if recommendations else ["Performance is within acceptable ranges"]
    
    def _calculate_trend(self, values: List[float], window: int = 20) -> float:
        """Calculate trend direction (-1 to 1, negative = decreasing, positive = increasing)."""
        
        if len(values) < window:
            return 0.0
        
        recent_values = values[-window:]
        
        # Simple linear regression slope
        x = np.arange(len(recent_values))
        coeffs = np.polyfit(x, recent_values, 1)
        slope = coeffs[0]
        
        # Normalize by mean value
        mean_value = np.mean(recent_values)
        if mean_value > 0:
            normalized_trend = slope / mean_value
        else:
            normalized_trend = 0.0
        
        # Clamp to [-1, 1]
        return np.clip(normalized_trend, -1.0, 1.0)
    
    def _update_calibration(self, snapshot: PerformanceSnapshot):
        """Update performance calibration with new snapshot."""
        
        self.calibration_samples.append(snapshot)
        
        # Need at least 50 samples for good calibration
        if len(self.calibration_samples) >= 50:
            inference_times = [s.total_inference_time_ms for s in self.calibration_samples]
            memory_usage = [s.memory_usage_mb for s in self.calibration_samples]
            
            self.baseline_metrics = {
                'mean_inference_time': np.mean(inference_times),
                'std_inference_time': np.std(inference_times),
                'mean_memory_usage': np.mean(memory_usage),
                'std_memory_usage': np.std(memory_usage),
                'calibration_timestamp': time.time()
            }
            
            self.is_calibrated = True
            self.logger.info("Performance monitoring calibration completed")
    
    def _should_trigger_optimization(self) -> bool:
        """Check if enough time has passed since last optimization."""
        current_time = time.time()
        return (current_time - self.last_optimization_time) > self.optimization_cooldown
    
    def _trigger_performance_optimization(self, alerts: List[str]):
        """Trigger performance optimization based on alerts."""
        
        self.last_optimization_time = time.time()
        
        optimization_event = {
            'timestamp': self.last_optimization_time,
            'alerts': alerts,
            'recommendations': self.get_optimization_recommendations(),
            'performance_summary': self.get_performance_summary(50)
        }
        
        self.optimization_history.append(optimization_event)
        
        # Limit optimization history
        if len(self.optimization_history) > 20:
            self.optimization_history.pop(0)
        
        self.logger.warning(f"Performance optimization triggered due to {len(alerts)} alerts")
    
    def _start_real_time_monitoring(self):
        """Start real-time performance monitoring thread."""
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    memory_mb = self._get_memory_usage()
                    cpu_percent = self._get_cpu_usage()
                    gpu_memory_mb = self._get_gpu_memory_usage()
                    
                    # Check for system resource alerts
                    if memory_mb > self.target_memory_mb * 1.5:
                        self.logger.warning(f"High memory usage detected: {memory_mb:.1f}MB")
                    
                    if cpu_percent > 90:
                        self.logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
                    
                    time.sleep(self.monitoring_interval / 1000.0)  # Convert ms to seconds
                    
                except Exception as e:
                    self.logger.error(f"Error in real-time monitoring: {e}")
                    time.sleep(1.0)  # Fallback sleep
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Real-time performance monitoring started")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def export_performance_data(self, filepath: str):
        """Export performance data to file for analysis."""
        
        try:
            with self.lock:
                data = {
                    'performance_history': [
                        {
                            'timestamp': s.timestamp,
                            'total_inference_time_ms': s.total_inference_time_ms,
                            'intelligence_overhead_ms': s.intelligence_overhead_ms,
                            'memory_usage_mb': s.memory_usage_mb,
                            'cpu_usage_percent': s.cpu_usage_percent,
                            'gpu_memory_mb': s.gpu_memory_mb,
                            'throughput_qps': s.throughput_qps,
                            'component_breakdown': s.component_breakdown
                        }
                        for s in self.performance_history
                    ],
                    'component_timings': {
                        component: list(timings)
                        for component, timings in self.component_timings.items()
                    },
                    'optimization_history': self.optimization_history,
                    'baseline_metrics': self.baseline_metrics,
                    'config': self.config
                }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Performance data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting performance data: {e}")
    
    def reset_monitoring_state(self):
        """Reset monitoring state for clean restart."""
        
        with self.lock:
            self.performance_history.clear()
            for component in self.component_timings:
                self.component_timings[component].clear()
        
        self.calibration_samples.clear()
        self.optimization_history.clear()
        self.is_calibrated = False
        self.baseline_metrics = None
        self.last_optimization_time = 0
        
        self.logger.info("Performance monitoring state reset")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        
        self.logger.info("Performance monitoring stopped")