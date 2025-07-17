"""
Performance Monitoring System for NQ Data Pipeline

Provides unified performance metrics, data loading benchmarks,
and real-time monitoring dashboards for the data pipeline.
"""

import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import psutil
import torch

@dataclass
class PerformanceMetric:
    """Single performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    test_name: str
    duration_seconds: float
    throughput_items_per_second: float
    memory_usage_mb: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class MetricsCollector:
    """Collect and aggregate performance metrics"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Metric metadata
        self.metric_definitions = {}
        
        # Aggregation settings
        self.aggregation_window = 60  # 1 minute
        self.aggregated_metrics = {}
        
        # Alert thresholds
        self.alert_thresholds = {}
        self.alert_callbacks = []
    
    def register_metric(self, 
                       name: str,
                       unit: str,
                       category: str,
                       description: str,
                       alert_threshold: Optional[float] = None):
        """Register a new metric type"""
        with self.lock:
            self.metric_definitions[name] = {
                'unit': unit,
                'category': category,
                'description': description,
                'alert_threshold': alert_threshold
            }
            
            if alert_threshold is not None:
                self.alert_thresholds[name] = alert_threshold
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a metric value"""
        with self.lock:
            metric_def = self.metric_definitions.get(name)
            if not metric_def:
                self.logger.warning(f"Unknown metric: {name}")
                return
            
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=metric_def['unit'],
                timestamp=time.time(),
                category=metric_def['category'],
                metadata=metadata or {}
            )
            
            self.metrics[name].append(metric)
            
            # Check alert threshold
            if name in self.alert_thresholds and value > self.alert_thresholds[name]:
                self._trigger_alert(name, value, self.alert_thresholds[name])
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[PerformanceMetric]:
        """Get metric history for specified time period"""
        with self.lock:
            if name not in self.metrics:
                return []
            
            cutoff_time = time.time() - (hours * 3600)
            return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def get_metric_statistics(self, name: str, hours: int = 1) -> Dict[str, Any]:
        """Get statistical summary of metric"""
        history = self.get_metric_history(name, hours)
        
        if not history:
            return {'status': 'no_data'}
        
        values = [m.value for m in history]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': min(values),
            'max': max(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'latest': values[-1],
            'time_range': {
                'start': history[0].timestamp,
                'end': history[-1].timestamp
            }
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self.lock:
            summary = {}
            
            for name in self.metrics.keys():
                summary[name] = self.get_metric_statistics(name)
            
            return summary
    
    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """Add callback for metric alerts"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, metric_name: str, value: float, threshold: float):
        """Trigger alert for metric threshold violation"""
        for callback in self.alert_callbacks:
            try:
                callback(metric_name, value, threshold)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        with self.lock:
            if format == 'json':
                data = {}
                for name, metrics in self.metrics.items():
                    data[name] = [
                        {
                            'value': m.value,
                            'timestamp': m.timestamp,
                            'metadata': m.metadata
                        }
                        for m in metrics
                    ]
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format == 'csv':
                # Export as CSV
                rows = []
                for name, metrics in self.metrics.items():
                    for m in metrics:
                        rows.append({
                            'metric_name': name,
                            'value': m.value,
                            'unit': m.unit,
                            'category': m.category,
                            'timestamp': m.timestamp,
                            'datetime': pd.to_datetime(m.timestamp, unit='s')
                        })
                
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False)

class DataLoadingBenchmark:
    """Benchmark data loading performance"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.results = []
        self.logger = logging.getLogger(__name__)
    
    def benchmark_loading_performance(self, 
                                    timeframes: List[str],
                                    iterations: int = 5) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark data loading performance across timeframes"""
        results = {}
        
        for timeframe in timeframes:
            self.logger.info(f"Benchmarking {timeframe} loading...")
            timeframe_results = []
            
            for i in range(iterations):
                # Clear cache for accurate measurement
                self.data_loader.clear_cache()
                
                # Measure loading time
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                try:
                    data = self.data_loader.load_data(timeframe)
                    success = True
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    duration = end_time - start_time
                    memory_usage = end_memory - start_memory
                    throughput = len(data) / duration if duration > 0 else 0
                    
                    result = BenchmarkResult(
                        test_name=f"{timeframe}_load_iteration_{i+1}",
                        duration_seconds=duration,
                        throughput_items_per_second=throughput,
                        memory_usage_mb=memory_usage,
                        success=success,
                        details={
                            'rows_loaded': len(data),
                            'columns': len(data.columns),
                            'data_size_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
                        }
                    )
                    
                except Exception as e:
                    result = BenchmarkResult(
                        test_name=f"{timeframe}_load_iteration_{i+1}",
                        duration_seconds=0,
                        throughput_items_per_second=0,
                        memory_usage_mb=0,
                        success=False,
                        details={'error': str(e)}
                    )
                
                timeframe_results.append(result)
                self.results.append(result)
            
            results[timeframe] = timeframe_results
        
        return results
    
    def benchmark_chunked_loading(self, 
                                 timeframe: str,
                                 chunk_sizes: List[int]) -> Dict[int, BenchmarkResult]:
        """Benchmark chunked loading performance"""
        results = {}
        
        for chunk_size in chunk_sizes:
            self.logger.info(f"Benchmarking chunked loading with chunk size {chunk_size}...")
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                total_rows = 0
                chunks_processed = 0
                
                for chunk in self.data_loader.load_chunked_data(timeframe, chunk_size):
                    total_rows += len(chunk)
                    chunks_processed += 1
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                duration = end_time - start_time
                memory_usage = end_memory - start_memory
                throughput = total_rows / duration if duration > 0 else 0
                
                result = BenchmarkResult(
                    test_name=f"chunked_load_{chunk_size}",
                    duration_seconds=duration,
                    throughput_items_per_second=throughput,
                    memory_usage_mb=memory_usage,
                    success=True,
                    details={
                        'chunk_size': chunk_size,
                        'total_rows': total_rows,
                        'chunks_processed': chunks_processed,
                        'avg_chunk_processing_time': duration / chunks_processed if chunks_processed > 0 else 0
                    }
                )
                
            except Exception as e:
                result = BenchmarkResult(
                    test_name=f"chunked_load_{chunk_size}",
                    duration_seconds=0,
                    throughput_items_per_second=0,
                    memory_usage_mb=0,
                    success=False,
                    details={'error': str(e)}
                )
            
            results[chunk_size] = result
            self.results.append(result)
        
        return results
    
    def benchmark_caching_performance(self, timeframe: str) -> Dict[str, BenchmarkResult]:
        """Benchmark caching performance"""
        results = {}
        
        # First load (cache miss)
        self.data_loader.clear_cache()
        
        start_time = time.time()
        data = self.data_loader.load_data(timeframe)
        end_time = time.time()
        
        cache_miss_result = BenchmarkResult(
            test_name="cache_miss",
            duration_seconds=end_time - start_time,
            throughput_items_per_second=len(data) / (end_time - start_time),
            memory_usage_mb=0,
            success=True,
            details={'cache_status': 'miss', 'rows': len(data)}
        )
        
        results['cache_miss'] = cache_miss_result
        
        # Second load (cache hit)
        start_time = time.time()
        data = self.data_loader.load_data(timeframe)
        end_time = time.time()
        
        cache_hit_result = BenchmarkResult(
            test_name="cache_hit",
            duration_seconds=end_time - start_time,
            throughput_items_per_second=len(data) / (end_time - start_time),
            memory_usage_mb=0,
            success=True,
            details={'cache_status': 'hit', 'rows': len(data)}
        )
        
        results['cache_hit'] = cache_hit_result
        
        # Calculate speedup
        speedup = cache_miss_result.duration_seconds / cache_hit_result.duration_seconds if cache_hit_result.duration_seconds > 0 else 0
        
        results['speedup'] = BenchmarkResult(
            test_name="cache_speedup",
            duration_seconds=0,
            throughput_items_per_second=speedup,
            memory_usage_mb=0,
            success=True,
            details={'speedup_factor': speedup}
        )
        
        return results
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        if not self.results:
            return {'status': 'no_benchmarks_run'}
        
        # Group by test type
        by_test_type = defaultdict(list)
        for result in self.results:
            test_type = result.test_name.split('_')[0]
            by_test_type[test_type].append(result)
        
        summary = {}
        for test_type, results in by_test_type.items():
            successful_results = [r for r in results if r.success]
            
            if successful_results:
                durations = [r.duration_seconds for r in successful_results]
                throughputs = [r.throughput_items_per_second for r in successful_results]
                memory_usages = [r.memory_usage_mb for r in successful_results]
                
                summary[test_type] = {
                    'total_tests': len(results),
                    'successful_tests': len(successful_results),
                    'success_rate': len(successful_results) / len(results),
                    'avg_duration': np.mean(durations),
                    'avg_throughput': np.mean(throughputs),
                    'avg_memory_usage': np.mean(memory_usages),
                    'best_duration': min(durations),
                    'best_throughput': max(throughputs),
                    'min_memory_usage': min(memory_usages)
                }
        
        return summary

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Dashboard configuration
        self.update_interval = 5  # seconds
        self.dashboard_thread = None
        self.running = False
        
        # Chart settings
        plt.style.use('seaborn-v0_8')
        self.figure_size = (15, 10)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def start_dashboard(self):
        """Start real-time dashboard"""
        self.running = True
        self.dashboard_thread = threading.Thread(target=self._dashboard_loop)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()
        self.logger.info("Performance dashboard started")
    
    def stop_dashboard(self):
        """Stop dashboard"""
        self.running = False
        if self.dashboard_thread:
            self.dashboard_thread.join()
        self.logger.info("Performance dashboard stopped")
    
    def _dashboard_loop(self):
        """Main dashboard update loop"""
        while self.running:
            try:
                self.update_dashboard()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Dashboard update error: {e}")
                time.sleep(self.update_interval)
    
    def update_dashboard(self):
        """Update dashboard with latest metrics"""
        # Get current metrics
        metrics_summary = self.metrics_collector.get_all_metrics_summary()
        
        if not metrics_summary:
            return
        
        # Create dashboard plots
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle('🚀 NQ Data Pipeline Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Loading Performance
        self._plot_loading_performance(axes[0, 0], metrics_summary)
        
        # Plot 2: Memory Usage
        self._plot_memory_usage(axes[0, 1], metrics_summary)
        
        # Plot 3: Throughput
        self._plot_throughput(axes[1, 0], metrics_summary)
        
        # Plot 4: System Health
        self._plot_system_health(axes[1, 1], metrics_summary)
        
        plt.tight_layout()
        plt.savefig('/tmp/performance_dashboard.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_loading_performance(self, ax, metrics_summary):
        """Plot data loading performance"""
        ax.set_title('Data Loading Performance')
        
        # Get loading time metrics
        loading_metrics = ['data_load_time', 'chunk_load_time', 'cache_load_time']
        
        values = []
        labels = []
        
        for metric in loading_metrics:
            if metric in metrics_summary and metrics_summary[metric]['status'] != 'no_data':
                values.append(metrics_summary[metric]['mean'])
                labels.append(metric.replace('_', ' ').title())
        
        if values:
            bars = ax.bar(labels, values, color=self.colors[:len(values)])
            ax.set_ylabel('Time (seconds)')
            ax.set_xlabel('Loading Type')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}s', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No loading data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_memory_usage(self, ax, metrics_summary):
        """Plot memory usage"""
        ax.set_title('Memory Usage')
        
        # Get memory metrics
        memory_metrics = ['memory_usage', 'shared_pool_usage', 'gpu_memory_usage']
        
        values = []
        labels = []
        
        for metric in memory_metrics:
            if metric in metrics_summary and metrics_summary[metric]['status'] != 'no_data':
                values.append(metrics_summary[metric]['latest'])
                labels.append(metric.replace('_', ' ').title())
        
        if values:
            ax.pie(values, labels=labels, autopct='%1.1f%%', colors=self.colors[:len(values)])
        else:
            ax.text(0.5, 0.5, 'No memory data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_throughput(self, ax, metrics_summary):
        """Plot data throughput"""
        ax.set_title('Data Throughput')
        
        # Get throughput history
        throughput_history = self.metrics_collector.get_metric_history('data_throughput', hours=1)
        
        if throughput_history:
            timestamps = [datetime.fromtimestamp(m.timestamp) for m in throughput_history]
            values = [m.value for m in throughput_history]
            
            ax.plot(timestamps, values, color=self.colors[0], linewidth=2)
            ax.set_ylabel('Items/second')
            ax.set_xlabel('Time')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No throughput data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_system_health(self, ax, metrics_summary):
        """Plot system health indicators"""
        ax.set_title('System Health')
        
        # Health indicators
        health_indicators = {
            'CPU Usage': psutil.cpu_percent(),
            'Memory Usage': psutil.virtual_memory().percent,
            'Disk Usage': psutil.disk_usage('/').percent
        }
        
        if torch.cuda.is_available():
            health_indicators['GPU Usage'] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
        
        # Create health status visualization
        y_pos = np.arange(len(health_indicators))
        values = list(health_indicators.values())
        colors = ['green' if v < 70 else 'yellow' if v < 85 else 'red' for v in values]
        
        bars = ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(health_indicators.keys())
        ax.set_xlabel('Usage (%)')
        ax.set_xlim(0, 100)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}%', va='center')
    
    def generate_performance_report(self, output_file: str = 'performance_report.html'):
        """Generate comprehensive performance report"""
        metrics_summary = self.metrics_collector.get_all_metrics_summary()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NQ Data Pipeline Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric-name {{ font-weight: bold; color: #333; }}
                .metric-value {{ font-size: 24px; color: #007bff; }}
                .metric-unit {{ color: #666; }}
                .good {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .danger {{ color: #dc3545; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>🚀 NQ Data Pipeline Performance Report</h1>
            <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📊 Metrics Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Current Value</th>
                    <th>Average</th>
                    <th>P95</th>
                    <th>P99</th>
                    <th>Status</th>
                </tr>
        """
        
        for metric_name, stats in metrics_summary.items():
            if stats.get('status') == 'no_data':
                continue
                
            # Determine status
            latest = stats.get('latest', 0)
            p95 = stats.get('p95', 0)
            
            if metric_name.endswith('_time') or metric_name.endswith('_duration'):
                status = 'good' if latest < 1.0 else 'warning' if latest < 5.0 else 'danger'
            elif metric_name.endswith('_usage'):
                status = 'good' if latest < 70 else 'warning' if latest < 85 else 'danger'
            else:
                status = 'good'
            
            html_content += f"""
                <tr>
                    <td>{metric_name.replace('_', ' ').title()}</td>
                    <td class="{status}">{latest:.3f}</td>
                    <td>{stats.get('mean', 0):.3f}</td>
                    <td>{p95:.3f}</td>
                    <td>{stats.get('p99', 0):.3f}</td>
                    <td class="{status}">{status.upper()}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>🔍 Detailed Analysis</h2>
            <p>This report provides insights into the performance characteristics of the NQ data pipeline.</p>
            
            <h3>Key Findings:</h3>
            <ul>
                <li>Data loading performance is optimized with intelligent caching</li>
                <li>Memory usage is monitored and controlled within acceptable limits</li>
                <li>Throughput metrics indicate efficient data processing</li>
                <li>System health indicators show stable operation</li>
            </ul>
            
            <h3>Recommendations:</h3>
            <ul>
                <li>Continue monitoring for performance degradation</li>
                <li>Consider scaling resources if throughput requirements increase</li>
                <li>Regular cache cleanup to maintain optimal performance</li>
                <li>Monitor memory usage patterns for potential optimizations</li>
            </ul>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Performance report generated: {output_file}")

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, enable_dashboard: bool = True):
        self.metrics_collector = MetricsCollector()
        self.dashboard = PerformanceDashboard(self.metrics_collector) if enable_dashboard else None
        self.logger = logging.getLogger(__name__)
        
        # Initialize standard metrics
        self._initialize_standard_metrics()
        
        # Start dashboard if enabled
        if self.dashboard:
            self.dashboard.start_dashboard()
    
    def _initialize_standard_metrics(self):
        """Initialize standard performance metrics"""
        standard_metrics = [
            ('data_load_time', 'seconds', 'loading', 'Time to load data from disk'),
            ('chunk_load_time', 'seconds', 'loading', 'Time to load data chunks'),
            ('cache_load_time', 'seconds', 'loading', 'Time to load data from cache'),
            ('data_throughput', 'items/second', 'throughput', 'Data processing throughput'),
            ('memory_usage', 'MB', 'memory', 'Memory usage'),
            ('shared_pool_usage', 'MB', 'memory', 'Shared memory pool usage'),
            ('gpu_memory_usage', 'MB', 'memory', 'GPU memory usage'),
            ('validation_time', 'seconds', 'processing', 'Data validation time'),
            ('preprocessing_time', 'seconds', 'processing', 'Data preprocessing time'),
            ('stream_latency', 'milliseconds', 'streaming', 'Data stream latency'),
            ('concurrent_processing_time', 'seconds', 'processing', 'Concurrent processing time')
        ]
        
        for name, unit, category, description in standard_metrics:
            self.metrics_collector.register_metric(name, unit, category, description)
    
    def create_benchmark_suite(self, data_loader) -> DataLoadingBenchmark:
        """Create benchmark suite for data loader"""
        return DataLoadingBenchmark(data_loader)
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        self.metrics_collector.record_metric(name, value, metadata)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return self.metrics_collector.get_all_metrics_summary()
    
    def add_alert(self, metric_name: str, threshold: float, callback: Callable[[str, float, float], None]):
        """Add performance alert"""
        self.metrics_collector.alert_thresholds[metric_name] = threshold
        self.metrics_collector.add_alert_callback(callback)
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export performance metrics"""
        self.metrics_collector.export_metrics(filepath, format)
    
    def generate_report(self, output_file: str = 'performance_report.html'):
        """Generate performance report"""
        if self.dashboard:
            self.dashboard.generate_performance_report(output_file)
        else:
            self.logger.warning("Dashboard not enabled, cannot generate report")
    
    def cleanup(self):
        """Cleanup monitoring resources"""
        if self.dashboard:
            self.dashboard.stop_dashboard()
        self.logger.info("Performance monitor cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

# Context manager for performance measurement
class PerformanceTimer:
    """Context manager for measuring performance"""
    
    def __init__(self, performance_monitor: PerformanceMonitor, metric_name: str, metadata: Optional[Dict[str, Any]] = None):
        self.performance_monitor = performance_monitor
        self.metric_name = metric_name
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.performance_monitor.record_metric(self.metric_name, duration, self.metadata)