#!/usr/bin/env python3
"""
AGENT 3 Performance Monitoring Dashboard
Real-time performance monitoring and alerting system
"""

import torch
import time
import json
import threading
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import psutil
import numpy as np
from collections import deque, defaultdict
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceRecord:
    """Single performance record"""
    timestamp: float
    model_name: str
    latency_ms: float
    throughput_qps: float
    memory_mb: float
    cpu_percent: float
    gpu_memory_mb: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PerformanceAlert:
    """Performance alert system"""
    
    def __init__(self, latency_threshold_ms: float = 1.0, error_rate_threshold: float = 0.01):
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.alerts = deque(maxlen=1000)
        self.alert_counts = defaultdict(int)
        
    def check_performance(self, record: PerformanceRecord) -> Optional[Dict[str, Any]]:
        """Check if performance record triggers an alert"""
        alert = None
        
        # Latency alert
        if record.latency_ms > self.latency_threshold_ms:
            alert = {
                'type': 'latency_breach',
                'timestamp': record.timestamp,
                'model_name': record.model_name,
                'actual_latency_ms': record.latency_ms,
                'threshold_ms': self.latency_threshold_ms,
                'severity': 'high' if record.latency_ms > self.latency_threshold_ms * 2 else 'medium'
            }
        
        # Error rate alert
        elif record.error_rate > self.error_rate_threshold:
            alert = {
                'type': 'error_rate_breach',
                'timestamp': record.timestamp,
                'model_name': record.model_name,
                'actual_error_rate': record.error_rate,
                'threshold': self.error_rate_threshold,
                'severity': 'high'
            }
        
        if alert:
            self.alerts.append(alert)
            self.alert_counts[alert['type']] += 1
            logger.warning(f"Performance alert: {alert}")
        
        return alert
    
    def get_recent_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get alerts from the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        return [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]

class PerformanceCollector:
    """Real-time performance data collector"""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.records = deque(maxlen=10000)
        self.model_stats = defaultdict(lambda: {
            'count': 0,
            'total_latency': 0.0,
            'min_latency': float('inf'),
            'max_latency': 0.0,
            'latencies': deque(maxlen=1000)
        })
        self.running = False
        self.thread = None
        self.alerter = PerformanceAlert()
    
    def record_performance(self, model_name: str, latency_ms: float, throughput_qps: float = 0.0, 
                          memory_mb: float = 0.0, error_rate: float = 0.0):
        """Record a performance measurement"""
        record = PerformanceRecord(
            timestamp=time.time(),
            model_name=model_name,
            latency_ms=latency_ms,
            throughput_qps=throughput_qps,
            memory_mb=memory_mb,
            cpu_percent=psutil.cpu_percent(),
            gpu_memory_mb=torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
            error_rate=error_rate
        )
        
        self.records.append(record)
        
        # Update model statistics
        stats = self.model_stats[model_name]
        stats['count'] += 1
        stats['total_latency'] += latency_ms
        stats['min_latency'] = min(stats['min_latency'], latency_ms)
        stats['max_latency'] = max(stats['max_latency'], latency_ms)
        stats['latencies'].append(latency_ms)
        
        # Check for alerts
        self.alerter.check_performance(record)
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get performance summary for a model"""
        stats = self.model_stats[model_name]
        
        if stats['count'] == 0:
            return {'error': 'No data for model'}
        
        latencies = list(stats['latencies'])
        
        return {
            'model_name': model_name,
            'total_requests': stats['count'],
            'avg_latency_ms': stats['total_latency'] / stats['count'],
            'min_latency_ms': stats['min_latency'],
            'max_latency_ms': stats['max_latency'],
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'last_updated': time.time()
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get overall system performance summary"""
        if not self.records:
            return {'error': 'No performance data available'}
        
        recent_records = [r for r in self.records if time.time() - r.timestamp < 300]  # Last 5 minutes
        
        if not recent_records:
            return {'error': 'No recent performance data'}
        
        return {
            'total_records': len(self.records),
            'recent_records': len(recent_records),
            'avg_cpu_percent': np.mean([r.cpu_percent for r in recent_records]),
            'avg_memory_mb': np.mean([r.memory_mb for r in recent_records]),
            'avg_gpu_memory_mb': np.mean([r.gpu_memory_mb for r in recent_records]),
            'models_monitored': len(self.model_stats),
            'alerts_last_hour': len(self.alerter.get_recent_alerts(60)),
            'data_collection_rate': len(recent_records) / 5.0,  # Records per minute
            'timestamp': time.time()
        }

class PerformanceBenchmark:
    """Comprehensive performance benchmarking"""
    
    def __init__(self, model_dir: Path = Path("models/ultra_fast")):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.benchmark_results = {}
        self.collector = PerformanceCollector()
        
    def load_models(self):
        """Load all optimized models"""
        logger.info(f"Loading models from {self.model_dir}")
        
        for model_file in self.model_dir.glob("*.pt"):
            try:
                model_name = model_file.stem
                model = torch.jit.load(str(model_file))
                model.eval()
                self.models[model_name] = model
                logger.info(f"✅ Loaded {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_file}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def run_latency_benchmark(self, model_name: str, num_iterations: int = 1000) -> Dict[str, float]:
        """Run comprehensive latency benchmark"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # Determine input shape
        if 'tactical' in model_name or model_name in ['fvg_agent', 'momentum_agent', 'entry_agent']:
            test_input = torch.randn(1, 60, 7)
        elif model_name == 'strategic_system':
            test_input = (torch.randn(1, 4), torch.randn(1, 6), torch.randn(1, 3))
        elif 'mlmi' in model_name:
            test_input = torch.randn(1, 4)
        elif 'nwrqk' in model_name:
            test_input = torch.randn(1, 6)
        elif 'mmd' in model_name:
            test_input = torch.randn(1, 3)
        else:  # critic
            test_input = torch.randn(1, 13)
        
        # Warm up
        with torch.no_grad():
            for _ in range(100):
                if isinstance(test_input, tuple):
                    _ = model(*test_input)
                else:
                    _ = model(test_input)
        
        # Benchmark
        latencies = []
        errors = 0
        
        with torch.no_grad():
            for i in range(num_iterations):
                try:
                    start_time = time.perf_counter()
                    
                    if isinstance(test_input, tuple):
                        _ = model(*test_input)
                    else:
                        _ = model(test_input)
                    
                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    
                    # Record performance
                    self.collector.record_performance(
                        model_name=model_name,
                        latency_ms=latency_ms,
                        throughput_qps=1000 / latency_ms,
                        memory_mb=psutil.virtual_memory().used / (1024 * 1024)
                    )
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"Error in iteration {i}: {e}")
        
        # Calculate statistics
        results = {
            'model_name': model_name,
            'iterations': num_iterations,
            'errors': errors,
            'error_rate': errors / num_iterations,
            'avg_latency_ms': np.mean(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'p999_latency_ms': np.percentile(latencies, 99.9),
            'std_latency_ms': np.std(latencies),
            'avg_throughput_qps': 1000 / np.mean(latencies),
            'meets_1ms_target': np.percentile(latencies, 99) < 1.0,
            'meets_500us_target': np.percentile(latencies, 99) < 0.5,
            'consistency_score': 1.0 - (np.std(latencies) / np.mean(latencies))
        }
        
        self.benchmark_results[model_name] = results
        return results
    
    def run_stress_test(self, model_name: str, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run stress test for sustained performance"""
        logger.info(f"Starting stress test for {model_name} ({duration_seconds}s)")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # Determine input
        if 'tactical' in model_name or model_name in ['fvg_agent', 'momentum_agent', 'entry_agent']:
            test_input = torch.randn(1, 60, 7)
        elif model_name == 'strategic_system':
            test_input = (torch.randn(1, 4), torch.randn(1, 6), torch.randn(1, 3))
        elif 'mlmi' in model_name:
            test_input = torch.randn(1, 4)
        elif 'nwrqk' in model_name:
            test_input = torch.randn(1, 6)
        elif 'mmd' in model_name:
            test_input = torch.randn(1, 3)
        else:  # critic
            test_input = torch.randn(1, 13)
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        iterations = 0
        errors = 0
        latencies = []
        
        with torch.no_grad():
            while time.time() < end_time:
                try:
                    iter_start = time.perf_counter()
                    
                    if isinstance(test_input, tuple):
                        _ = model(*test_input)
                    else:
                        _ = model(test_input)
                    
                    iter_end = time.perf_counter()
                    latency_ms = (iter_end - iter_start) * 1000
                    latencies.append(latency_ms)
                    
                    iterations += 1
                    
                    # Record performance
                    self.collector.record_performance(
                        model_name=f"{model_name}_stress",
                        latency_ms=latency_ms,
                        throughput_qps=1000 / latency_ms,
                        memory_mb=psutil.virtual_memory().used / (1024 * 1024)
                    )
                    
                except Exception as e:
                    errors += 1
        
        actual_duration = time.time() - start_time
        
        return {
            'model_name': model_name,
            'test_type': 'stress_test',
            'duration_seconds': actual_duration,
            'iterations': iterations,
            'errors': errors,
            'error_rate': errors / max(iterations, 1),
            'avg_throughput_qps': iterations / actual_duration,
            'avg_latency_ms': np.mean(latencies),
            'p99_latency_ms': np.percentile(latencies, 99),
            'degradation_detected': np.percentile(latencies[-100:], 99) > np.percentile(latencies[:100], 99) * 1.2,
            'memory_stable': psutil.virtual_memory().percent < 90
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': time.time(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            'models_benchmarked': len(self.benchmark_results),
            'benchmark_results': self.benchmark_results,
            'system_summary': self.collector.get_system_summary(),
            'alerts_summary': {
                'recent_alerts': len(self.collector.alerter.get_recent_alerts(60)),
                'alert_types': dict(self.collector.alerter.alert_counts)
            }
        }
        
        # Add model summaries
        report['model_summaries'] = {}
        for model_name in self.models.keys():
            report['model_summaries'][model_name] = self.collector.get_model_summary(model_name)
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: Path = Path("performance_report.json")):
        """Save performance report"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Performance report saved to {output_path}")

class PerformanceVisualizer:
    """Performance visualization and dashboard"""
    
    def __init__(self, collector: PerformanceCollector):
        self.collector = collector
        plt.style.use('seaborn-v0_8')
        
    def plot_latency_distribution(self, model_name: str, save_path: Optional[Path] = None):
        """Plot latency distribution for a model"""
        stats = self.collector.model_stats[model_name]
        latencies = list(stats['latencies'])
        
        if not latencies:
            logger.warning(f"No latency data for {model_name}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(latencies), color='red', linestyle='--', label=f'Mean: {np.mean(latencies):.2f}ms')
        ax1.axvline(np.percentile(latencies, 99), color='orange', linestyle='--', label=f'P99: {np.percentile(latencies, 99):.2f}ms')
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Latency Distribution - {model_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time series
        ax2.plot(latencies, alpha=0.7)
        ax2.axhline(np.mean(latencies), color='red', linestyle='--', label=f'Mean: {np.mean(latencies):.2f}ms')
        ax2.axhline(np.percentile(latencies, 99), color='orange', linestyle='--', label=f'P99: {np.percentile(latencies, 99):.2f}ms')
        ax2.set_xlabel('Request Number')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title(f'Latency Time Series - {model_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dashboard(self, save_path: Optional[Path] = None):
        """Create comprehensive performance dashboard"""
        if not self.collector.model_stats:
            logger.warning("No performance data available for dashboard")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Model performance comparison
        ax1 = fig.add_subplot(gs[0, :])
        model_names = list(self.collector.model_stats.keys())
        p99_latencies = [np.percentile(list(stats['latencies']), 99) for stats in self.collector.model_stats.values()]
        
        bars = ax1.bar(model_names, p99_latencies, color=['green' if l < 1.0 else 'orange' if l < 5.0 else 'red' for l in p99_latencies])
        ax1.axhline(1.0, color='red', linestyle='--', label='1ms target')
        ax1.set_ylabel('P99 Latency (ms)')
        ax1.set_title('Model Performance Comparison (P99 Latency)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # System resources
        ax2 = fig.add_subplot(gs[1, 0])
        recent_records = [r for r in self.collector.records if time.time() - r.timestamp < 300]
        if recent_records:
            cpu_usage = [r.cpu_percent for r in recent_records]
            ax2.plot(cpu_usage, color='blue', alpha=0.7)
            ax2.set_ylabel('CPU Usage (%)')
            ax2.set_title('CPU Usage (Last 5 mins)')
            ax2.grid(True, alpha=0.3)
        
        # Memory usage
        ax3 = fig.add_subplot(gs[1, 1])
        if recent_records:
            memory_usage = [r.memory_mb for r in recent_records]
            ax3.plot(memory_usage, color='green', alpha=0.7)
            ax3.set_ylabel('Memory Usage (MB)')
            ax3.set_title('Memory Usage (Last 5 mins)')
            ax3.grid(True, alpha=0.3)
        
        # Throughput
        ax4 = fig.add_subplot(gs[1, 2])
        if recent_records:
            throughput = [r.throughput_qps for r in recent_records]
            ax4.plot(throughput, color='purple', alpha=0.7)
            ax4.set_ylabel('Throughput (QPS)')
            ax4.set_title('Throughput (Last 5 mins)')
            ax4.grid(True, alpha=0.3)
        
        # Latency heatmap
        ax5 = fig.add_subplot(gs[2, :])
        if len(model_names) > 1:
            latency_matrix = []
            for model_name in model_names:
                stats = self.collector.model_stats[model_name]
                latencies = list(stats['latencies'])
                if latencies:
                    latency_matrix.append([
                        np.percentile(latencies, 50),
                        np.percentile(latencies, 95),
                        np.percentile(latencies, 99),
                        np.percentile(latencies, 99.9)
                    ])
                else:
                    latency_matrix.append([0, 0, 0, 0])
            
            im = ax5.imshow(latency_matrix, cmap='RdYlGn_r', aspect='auto')
            ax5.set_xticks(range(4))
            ax5.set_xticklabels(['P50', 'P95', 'P99', 'P99.9'])
            ax5.set_yticks(range(len(model_names)))
            ax5.set_yticklabels(model_names)
            ax5.set_title('Latency Percentiles Heatmap (ms)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax5)
            cbar.set_label('Latency (ms)')
        
        plt.suptitle('GrandModel Performance Dashboard', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main performance monitoring workflow"""
    print("📊 GRAND MODEL PERFORMANCE MONITORING SYSTEM")
    print("=" * 80)
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    benchmark.load_models()
    
    if not benchmark.models:
        print("❌ No models loaded. Please ensure models are available in models/optimized/")
        return
    
    # Run benchmarks
    print("\n🏁 Running performance benchmarks...")
    
    for model_name in benchmark.models.keys():
        print(f"  Benchmarking {model_name}...")
        try:
            result = benchmark.run_latency_benchmark(model_name, num_iterations=1000)
            status = "✅ FAST" if result['meets_1ms_target'] else "❌ SLOW"
            print(f"    {model_name}: {result['p99_latency_ms']:.3f}ms p99 {status}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    # Generate report
    print("\n📋 Generating performance report...")
    report = benchmark.generate_performance_report()
    benchmark.save_report(report, Path("performance_report.json"))
    
    # Create visualizations
    print("\n📊 Creating performance visualizations...")
    visualizer = PerformanceVisualizer(benchmark.collector)
    
    # Create dashboard
    try:
        visualizer.create_dashboard(Path("performance_dashboard.png"))
        print("✅ Dashboard saved to performance_dashboard.png")
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 PERFORMANCE MONITORING SUMMARY")
    print("=" * 80)
    
    fast_models = sum(1 for r in benchmark.benchmark_results.values() if r['meets_1ms_target'])
    total_models = len(benchmark.benchmark_results)
    
    print(f"📈 Models benchmarked: {total_models}")
    print(f"⚡ Fast models (<1ms): {fast_models}")
    print(f"🎯 Success rate: {fast_models/total_models*100:.1f}%")
    
    # Best performing models
    if benchmark.benchmark_results:
        best_model = min(benchmark.benchmark_results.items(), key=lambda x: x[1]['p99_latency_ms'])
        print(f"🏆 Best model: {best_model[0]} ({best_model[1]['p99_latency_ms']:.3f}ms p99)")
    
    print(f"\n💾 Performance report: performance_report.json")
    print(f"📊 Dashboard: performance_dashboard.png")
    print("🔍 Monitoring system ready for production deployment!")

if __name__ == "__main__":
    main()