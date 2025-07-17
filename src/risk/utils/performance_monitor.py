"""
Performance Monitoring for VaR and Correlation Systems

This module provides comprehensive performance monitoring and benchmarking
for the enhanced VaR correlation system to ensure <5ms calculation targets.

Features:
- Real-time performance tracking
- Memory usage monitoring
- Throughput measurement
- Performance regression detection
- Optimization recommendations
"""

import time
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import structlog
import asyncio
from contextlib import contextmanager

logger = structlog.get_logger()


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: datetime
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    additional_data: Dict = field(default_factory=dict)


@dataclass
class PerformanceSummary:
    """Performance summary statistics"""
    operation: str
    measurement_count: int
    avg_duration_ms: float
    median_duration_ms: float
    p95_duration_ms: float
    max_duration_ms: float
    min_duration_ms: float
    target_met: bool
    target_ms: float
    memory_avg_mb: float
    memory_peak_mb: float


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for VaR calculations.
    
    Tracks timing, memory usage, and system resources to ensure
    the <5ms VaR calculation target is consistently met.
    """
    
    def __init__(
        self,
        default_target_ms: float = 5.0,
        max_history: int = 10000,
        alert_threshold_multiplier: float = 2.0
    ):
        self.default_target_ms = default_target_ms
        self.max_history = max_history
        self.alert_threshold_ms = default_target_ms * alert_threshold_multiplier
        
        # Performance data storage
        self.metrics: deque = deque(maxlen=max_history)
        self.operation_targets: Dict[str, float] = {}
        
        # Real-time monitoring
        self.active_operations: Dict[str, datetime] = {}
        self.process = psutil.Process()
        
        # Performance alerts
        self.performance_alerts: List[Dict] = []
        
        logger.info("PerformanceMonitor initialized",
                   default_target_ms=default_target_ms,
                   max_history=max_history)
    
    def set_operation_target(self, operation: str, target_ms: float):
        """Set specific performance target for an operation"""
        self.operation_targets[operation] = target_ms
        logger.info("Performance target set",
                   operation=operation,
                   target_ms=target_ms)
    
    @contextmanager
    def measure(self, operation: str, additional_data: Optional[Dict] = None):
        """Context manager for measuring operation performance"""
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        operation_id = f"{operation}_{id(self)}"
        self.active_operations[operation_id] = datetime.now()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self.process.cpu_percent()
            
            # Create metric
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                operation=operation,
                duration_ms=duration_ms,
                memory_mb=end_memory,
                cpu_percent=end_cpu,
                additional_data=additional_data or {}
            )
            
            self.metrics.append(metric)
            
            # Check for performance alerts
            self._check_performance_alert(metric)
            
            # Clean up
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
    
    def _check_performance_alert(self, metric: PerformanceMetric):
        """Check if performance metric triggers an alert"""
        target = self.operation_targets.get(metric.operation, self.default_target_ms)
        
        if metric.duration_ms > self.alert_threshold_ms:
            alert = {
                'timestamp': metric.timestamp,
                'operation': metric.operation,
                'duration_ms': metric.duration_ms,
                'target_ms': target,
                'severity': 'CRITICAL' if metric.duration_ms > target * 5 else 'WARNING',
                'memory_mb': metric.memory_mb,
                'cpu_percent': metric.cpu_percent
            }
            
            self.performance_alerts.append(alert)
            
            # Keep only recent alerts
            if len(self.performance_alerts) > 100:
                self.performance_alerts = self.performance_alerts[-100:]
            
            logger.warning("Performance alert triggered",
                          operation=metric.operation,
                          duration_ms=metric.duration_ms,
                          target_ms=target,
                          severity=alert['severity'])
    
    def get_operation_summary(self, operation: str, hours: int = 1) -> Optional[PerformanceSummary]:
        """Get performance summary for a specific operation"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics for this operation and time period
        operation_metrics = [
            m for m in self.metrics 
            if m.operation == operation and m.timestamp >= cutoff_time
        ]
        
        if not operation_metrics:
            return None
        
        durations = [m.duration_ms for m in operation_metrics]
        memories = [m.memory_mb for m in operation_metrics]
        
        target = self.operation_targets.get(operation, self.default_target_ms)
        
        return PerformanceSummary(
            operation=operation,
            measurement_count=len(operation_metrics),
            avg_duration_ms=np.mean(durations),
            median_duration_ms=np.median(durations),
            p95_duration_ms=np.percentile(durations, 95),
            max_duration_ms=np.max(durations),
            min_duration_ms=np.min(durations),
            target_met=np.mean(durations) <= target,
            target_ms=target,
            memory_avg_mb=np.mean(memories),
            memory_peak_mb=np.max(memories)
        )
    
    def get_system_performance(self) -> Dict:
        """Get overall system performance metrics"""
        if not self.metrics:
            return {"status": "No metrics available"}
        
        # Recent metrics (last hour)
        recent_metrics = [
            m for m in self.metrics 
            if m.timestamp >= datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_metrics:
            return {"status": "No recent metrics"}
        
        # Aggregate by operation
        operations = {}
        for metric in recent_metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric.duration_ms)
        
        # Calculate summary statistics
        operation_summaries = {}
        overall_target_met = True
        
        for operation, durations in operations.items():
            target = self.operation_targets.get(operation, self.default_target_ms)
            avg_duration = np.mean(durations)
            target_met = avg_duration <= target
            
            operation_summaries[operation] = {
                'avg_duration_ms': avg_duration,
                'measurement_count': len(durations),
                'target_ms': target,
                'target_met': target_met,
                'p95_duration_ms': np.percentile(durations, 95)
            }
            
            if not target_met:
                overall_target_met = False
        
        # System resource usage
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_target_met': overall_target_met,
            'total_measurements': len(recent_metrics),
            'operations': operation_summaries,
            'system_resources': {
                'memory_mb': memory_info.rss / 1024 / 1024,
                'memory_percent': self.process.memory_percent(),
                'cpu_percent': cpu_percent,
                'num_threads': self.process.num_threads()
            },
            'recent_alerts': len([
                a for a in self.performance_alerts 
                if a['timestamp'] >= datetime.now() - timedelta(hours=1)
            ])
        }
    
    def get_performance_trends(self, operation: str, hours: int = 24) -> Dict:
        """Get performance trends for an operation over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        operation_metrics = [
            m for m in self.metrics 
            if m.operation == operation and m.timestamp >= cutoff_time
        ]
        
        if len(operation_metrics) < 10:
            return {"status": "Insufficient data for trend analysis"}
        
        # Group by hour
        hourly_data = {}
        for metric in operation_metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_data:
                hourly_data[hour_key] = []
            hourly_data[hour_key].append(metric.duration_ms)
        
        # Calculate hourly averages
        hourly_averages = {
            hour: np.mean(durations) 
            for hour, durations in hourly_data.items()
        }
        
        # Trend analysis
        hours_sorted = sorted(hourly_averages.keys())
        durations_trend = [hourly_averages[hour] for hour in hours_sorted]
        
        # Simple linear trend
        if len(durations_trend) >= 3:
            x = np.arange(len(durations_trend))
            trend_slope = np.polyfit(x, durations_trend, 1)[0]
            
            trend_direction = "IMPROVING" if trend_slope < 0 else "DEGRADING"
            trend_strength = abs(trend_slope)
        else:
            trend_direction = "STABLE"
            trend_strength = 0.0
        
        target = self.operation_targets.get(operation, self.default_target_ms)
        
        return {
            'operation': operation,
            'hours_analyzed': hours,
            'data_points': len(operation_metrics),
            'current_avg_ms': np.mean([m.duration_ms for m in operation_metrics[-10:]]),  # Last 10
            'target_ms': target,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'hourly_data': {
                hour.isoformat(): avg_duration 
                for hour, avg_duration in hourly_averages.items()
            }
        }
    
    def benchmark_operation(
        self,
        operation_func,
        operation_name: str,
        iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict:
        """Benchmark a specific operation"""
        
        logger.info("Starting benchmark",
                   operation=operation_name,
                   iterations=iterations,
                   warmup=warmup_iterations)
        
        # Warmup
        for _ in range(warmup_iterations):
            operation_func()
        
        # Actual benchmark
        durations = []
        memory_usage = []
        
        for i in range(iterations):
            start_memory = self.process.memory_info().rss / 1024 / 1024
            
            with self.measure(f"benchmark_{operation_name}"):
                operation_func()
            
            # Get the last metric (just recorded)
            if self.metrics:
                last_metric = self.metrics[-1]
                durations.append(last_metric.duration_ms)
                memory_usage.append(last_metric.memory_mb)
        
        # Calculate statistics
        target = self.operation_targets.get(operation_name, self.default_target_ms)
        
        benchmark_results = {
            'operation': operation_name,
            'iterations': iterations,
            'warmup_iterations': warmup_iterations,
            'target_ms': target,
            'avg_duration_ms': np.mean(durations),
            'median_duration_ms': np.median(durations),
            'min_duration_ms': np.min(durations),
            'max_duration_ms': np.max(durations),
            'p95_duration_ms': np.percentile(durations, 95),
            'p99_duration_ms': np.percentile(durations, 99),
            'std_duration_ms': np.std(durations),
            'target_met': np.mean(durations) <= target,
            'success_rate': len([d for d in durations if d <= target]) / len(durations),
            'memory_avg_mb': np.mean(memory_usage),
            'memory_peak_mb': np.max(memory_usage),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Benchmark completed",
                   operation=operation_name,
                   avg_duration_ms=benchmark_results['avg_duration_ms'],
                   target_met=benchmark_results['target_met'],
                   success_rate=f"{benchmark_results['success_rate']:.1%}")
        
        return benchmark_results
    
    def generate_optimization_recommendations(self) -> List[Dict]:
        """Generate optimization recommendations based on performance data"""
        recommendations = []
        
        # Analyze recent performance
        recent_metrics = [
            m for m in self.metrics 
            if m.timestamp >= datetime.now() - timedelta(hours=1)
        ]
        
        if not recent_metrics:
            return [{"recommendation": "No recent data available for analysis"}]
        
        # Group by operation
        operations = {}
        for metric in recent_metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric)
        
        for operation, metrics in operations.items():
            target = self.operation_targets.get(operation, self.default_target_ms)
            durations = [m.duration_ms for m in metrics]
            avg_duration = np.mean(durations)
            
            if avg_duration > target:
                severity = "HIGH" if avg_duration > target * 2 else "MEDIUM"
                
                # Memory analysis
                memories = [m.memory_mb for m in metrics]
                avg_memory = np.mean(memories)
                
                recommendation = {
                    'operation': operation,
                    'severity': severity,
                    'current_avg_ms': avg_duration,
                    'target_ms': target,
                    'performance_gap': avg_duration - target,
                    'suggestions': []
                }
                
                # Generate specific suggestions
                if avg_duration > target * 3:
                    recommendation['suggestions'].append(
                        "CRITICAL: Consider algorithmic optimization - performance is 3x slower than target"
                    )
                
                if avg_memory > 500:  # 500MB threshold
                    recommendation['suggestions'].append(
                        f"High memory usage ({avg_memory:.0f}MB) - consider memory optimization"
                    )
                
                # Check for variance
                duration_std = np.std(durations)
                if duration_std > avg_duration * 0.5:
                    recommendation['suggestions'].append(
                        "High performance variance - investigate system load or contention issues"
                    )
                
                # Check for trend
                if len(durations) >= 10:
                    recent_avg = np.mean(durations[-5:])
                    earlier_avg = np.mean(durations[:5])
                    if recent_avg > earlier_avg * 1.2:
                        recommendation['suggestions'].append(
                            "Performance degrading over time - investigate memory leaks or resource exhaustion"
                        )
                
                if not recommendation['suggestions']:
                    recommendation['suggestions'].append(
                        "Review algorithm efficiency and consider caching frequently computed values"
                    )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def export_performance_data(self, hours: int = 24) -> Dict:
        """Export performance data for external analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        export_metrics = [
            {
                'timestamp': m.timestamp.isoformat(),
                'operation': m.operation,
                'duration_ms': m.duration_ms,
                'memory_mb': m.memory_mb,
                'cpu_percent': m.cpu_percent,
                'additional_data': m.additional_data
            }
            for m in self.metrics 
            if m.timestamp >= cutoff_time
        ]
        
        return {
            'export_timestamp': datetime.now().isoformat(),
            'hours_exported': hours,
            'total_metrics': len(export_metrics),
            'metrics': export_metrics,
            'operation_targets': self.operation_targets.copy(),
            'performance_alerts': [
                alert for alert in self.performance_alerts
                if alert['timestamp'] >= cutoff_time
            ]
        }


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def measure_performance(operation: str, additional_data: Optional[Dict] = None):
    """Decorator for measuring function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with performance_monitor.measure(operation, additional_data):
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def benchmark_var_system():
    """Comprehensive benchmark of VaR system components"""
    
    logger.info("Starting comprehensive VaR system benchmark")
    
    # Import here to avoid circular imports
    from src.core.events import EventBus
    from src.risk.core.correlation_tracker import CorrelationTracker
    from src.risk.core.var_calculator import VaRCalculator, PositionData
    
    # Setup test system
    event_bus = EventBus()
    correlation_tracker = CorrelationTracker(event_bus)
    var_calculator = VaRCalculator(correlation_tracker, event_bus)
    
    # Initialize with test data
    test_assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    correlation_tracker.initialize_assets(test_assets)
    
    # Setup test portfolio
    test_positions = {
        'AAPL': PositionData('AAPL', 1000, 180000, 180.0, 0.25),
        'GOOGL': PositionData('GOOGL', 500, 150000, 300.0, 0.30),
        'MSFT': PositionData('MSFT', 800, 240000, 300.0, 0.22),
        'TSLA': PositionData('TSLA', 200, 50000, 250.0, 0.45),
        'NVDA': PositionData('NVDA', 300, 120000, 400.0, 0.40)
    }
    
    var_calculator.positions = test_positions
    var_calculator.portfolio_value = sum(pos.market_value for pos in test_positions.values())
    
    # Generate some historical data
    for i in range(100):
        for asset in test_assets:
            return_value = np.random.normal(0, 0.01)
            price = 100 * (1 + return_value)
            correlation_tracker.asset_returns[asset].append((datetime.now(), return_value))
    
    # Update correlation matrix
    correlation_tracker._update_correlation_matrix()
    
    # Benchmark operations
    results = {}
    
    # Benchmark correlation matrix update
    def update_correlation():
        correlation_tracker._update_correlation_matrix()
    
    results['correlation_update'] = performance_monitor.benchmark_operation(
        update_correlation,
        'correlation_matrix_update',
        iterations=100
    )
    
    # Benchmark VaR calculation
    async def calculate_var():
        await var_calculator.calculate_var()
    
    # For sync benchmark, we need a wrapper
    def sync_var_calc():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(calculate_var())
        finally:
            loop.close()
    
    results['var_calculation'] = performance_monitor.benchmark_operation(
        sync_var_calc,
        'var_calculation',
        iterations=50  # Fewer iterations for complex calculation
    )
    
    # Benchmark correlation shock detection
    def shock_detection():
        correlation_tracker._check_correlation_shock()
    
    results['shock_detection'] = performance_monitor.benchmark_operation(
        shock_detection,
        'correlation_shock_detection',
        iterations=200
    )
    
    logger.info("VaR system benchmark completed", results=results)
    return results


if __name__ == "__main__":
    """Run performance benchmarks"""
    
    print("🚀 VaR Performance Monitoring System")
    print("=" * 40)
    
    # Run comprehensive benchmark
    results = asyncio.run(benchmark_var_system())
    
    print("\n📊 Benchmark Results:")
    print("-" * 20)
    
    for operation, result in results.items():
        print(f"\n{operation.replace('_', ' ').title()}:")
        print(f"  Average: {result['avg_duration_ms']:.2f}ms")
        print(f"  Target: {result['target_ms']:.2f}ms")
        print(f"  Target Met: {'✓' if result['target_met'] else '✗'}")
        print(f"  Success Rate: {result['success_rate']:.1%}")
    
    # Generate recommendations
    recommendations = performance_monitor.generate_optimization_recommendations()
    
    if recommendations:
        print("\n🔧 Optimization Recommendations:")
        print("-" * 30)
        for rec in recommendations:
            print(f"\n{rec.get('operation', 'General')} ({rec.get('severity', 'INFO')}):")
            for suggestion in rec.get('suggestions', []):
                print(f"  • {suggestion}")
    
    print("\n✅ Performance monitoring complete!")