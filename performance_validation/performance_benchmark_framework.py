#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Framework
===============================================

This framework provides comprehensive performance validation for the MARL system
with large-scale dataset processing capabilities.

Features:
- Training time benchmarks
- Memory usage analysis
- Throughput measurement
- Resource utilization monitoring
- Scalability projections
- Bottleneck identification
"""

import time
import psutil
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import threading
import queue
import traceback
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    gpu_utilization: float = 0.0
    gpu_memory_mb: float = 0.0

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    dataset_size: int
    total_time_seconds: float
    peak_memory_mb: float
    avg_cpu_percent: float
    throughput_records_per_second: float
    success: bool
    error_message: Optional[str] = None
    detailed_metrics: List[PerformanceMetrics] = None

class ResourceMonitor:
    """Real-time resource monitoring system."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.metrics_queue = queue.Queue()
        self.monitoring = False
        self.monitor_thread = None
        
        # Initialize baseline metrics
        self.baseline_io = psutil.disk_io_counters()
        self.baseline_net = psutil.net_io_counters()
    
    def _get_gpu_metrics(self) -> Tuple[float, float]:
        """Get GPU utilization and memory usage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            
            # GPU memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory = mem_info.used / 1024 / 1024  # Convert to MB
            
            return float(gpu_util), float(gpu_memory)
        except:
            return 0.0, 0.0
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get current metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                # Get I/O metrics
                current_io = psutil.disk_io_counters()
                current_net = psutil.net_io_counters()
                
                disk_read = (current_io.read_bytes - self.baseline_io.read_bytes) / 1024 / 1024
                disk_write = (current_io.write_bytes - self.baseline_io.write_bytes) / 1024 / 1024
                net_sent = (current_net.bytes_sent - self.baseline_net.bytes_sent) / 1024 / 1024
                net_recv = (current_net.bytes_recv - self.baseline_net.bytes_recv) / 1024 / 1024
                
                # Get GPU metrics
                gpu_util, gpu_memory = self._get_gpu_metrics()
                
                # Create metrics object
                metrics = PerformanceMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory.used / 1024 / 1024,
                    memory_percent=memory.percent,
                    disk_io_read_mb=disk_read,
                    disk_io_write_mb=disk_write,
                    network_io_sent_mb=net_sent,
                    network_io_recv_mb=net_recv,
                    gpu_utilization=gpu_util,
                    gpu_memory_mb=gpu_memory
                )
                
                self.metrics_queue.put(metrics)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> List[PerformanceMetrics]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Collect all metrics
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        
        return metrics

class PerformanceBenchmarkFramework:
    """
    Comprehensive performance benchmarking framework for MARL system validation.
    """
    
    def __init__(self, output_dir: str = "performance_validation/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.benchmark_results = []
        self.system_info = self._get_system_info()
        
        # Performance thresholds
        self.performance_thresholds = {
            'max_memory_mb': 8000,  # 8GB
            'min_throughput_records_per_second': 1000,
            'max_training_time_per_epoch_seconds': 300,  # 5 minutes
            'max_cpu_percent': 80
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information for benchmarking context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'python_version': sys.version,
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat()
        }
    
    def benchmark_data_loading(self, file_path: str, chunk_size: int = 10000) -> BenchmarkResult:
        """
        Benchmark data loading performance.
        
        Args:
            file_path: Path to the dataset file
            chunk_size: Size of chunks for processing
            
        Returns:
            BenchmarkResult with loading performance metrics
        """
        
        print(f"Benchmarking data loading: {file_path}")
        
        monitor = ResourceMonitor(sampling_interval=0.5)
        monitor.start_monitoring()
        
        start_time = time.time()
        success = True
        error_message = None
        total_records = 0
        
        try:
            # Test chunked loading
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                total_records += len(chunk)
                
                # Simple processing simulation
                _ = chunk.describe()
                
                # Memory pressure check
                if psutil.virtual_memory().percent > 90:
                    print("Warning: Memory usage >90%")
        
        except Exception as e:
            success = False
            error_message = str(e)
            print(f"Error in data loading benchmark: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Stop monitoring and collect metrics
        metrics = monitor.stop_monitoring()
        
        # Calculate performance statistics
        peak_memory = max([m.memory_mb for m in metrics]) if metrics else 0
        avg_cpu = np.mean([m.cpu_percent for m in metrics]) if metrics else 0
        throughput = total_records / total_time if total_time > 0 else 0
        
        result = BenchmarkResult(
            test_name="data_loading",
            dataset_size=total_records,
            total_time_seconds=total_time,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=avg_cpu,
            throughput_records_per_second=throughput,
            success=success,
            error_message=error_message,
            detailed_metrics=metrics
        )
        
        self.benchmark_results.append(result)
        
        print(f"Data Loading Benchmark Complete:")
        print(f"  Records processed: {total_records:,}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Throughput: {throughput:.0f} records/second")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Avg CPU: {avg_cpu:.1f}%")
        
        return result
    
    def benchmark_training_simulation(self, dataset_size: int, epochs: int = 10) -> BenchmarkResult:
        """
        Simulate training performance with synthetic workload.
        
        Args:
            dataset_size: Number of records to simulate
            epochs: Number of training epochs
            
        Returns:
            BenchmarkResult with training performance metrics
        """
        
        print(f"Benchmarking training simulation: {dataset_size:,} records, {epochs} epochs")
        
        monitor = ResourceMonitor(sampling_interval=1.0)
        monitor.start_monitoring()
        
        start_time = time.time()
        success = True
        error_message = None
        
        try:
            # Simulate training workload
            for epoch in range(epochs):
                print(f"  Epoch {epoch + 1}/{epochs}")
                
                # Simulate data processing
                batch_size = 1000
                num_batches = dataset_size // batch_size
                
                for batch in range(num_batches):
                    # Simulate computation
                    data = np.random.random((batch_size, 10))
                    
                    # Simulate forward pass
                    weights = np.random.random((10, 5))
                    output = np.dot(data, weights)
                    
                    # Simulate backward pass
                    gradients = np.random.random((10, 5))
                    weights -= 0.001 * gradients
                    
                    # Simulate memory allocation/deallocation
                    if batch % 100 == 0:
                        temp_data = np.random.random((batch_size * 10, 10))
                        del temp_data
                    
                    # Check for memory pressure
                    if psutil.virtual_memory().percent > 85:
                        print(f"    Warning: Memory usage at {psutil.virtual_memory().percent:.1f}%")
                
                # Simulate checkpoint saving
                time.sleep(0.1)
        
        except Exception as e:
            success = False
            error_message = str(e)
            print(f"Error in training simulation: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Stop monitoring and collect metrics
        metrics = monitor.stop_monitoring()
        
        # Calculate performance statistics
        peak_memory = max([m.memory_mb for m in metrics]) if metrics else 0
        avg_cpu = np.mean([m.cpu_percent for m in metrics]) if metrics else 0
        throughput = (dataset_size * epochs) / total_time if total_time > 0 else 0
        
        result = BenchmarkResult(
            test_name="training_simulation",
            dataset_size=dataset_size,
            total_time_seconds=total_time,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=avg_cpu,
            throughput_records_per_second=throughput,
            success=success,
            error_message=error_message,
            detailed_metrics=metrics
        )
        
        self.benchmark_results.append(result)
        
        print(f"Training Simulation Complete:")
        print(f"  Dataset size: {dataset_size:,}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Time per epoch: {total_time/epochs:.2f} seconds")
        print(f"  Throughput: {throughput:.0f} records/second")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Avg CPU: {avg_cpu:.1f}%")
        
        return result
    
    def benchmark_notebook_execution(self, notebook_path: str, dataset_path: str) -> BenchmarkResult:
        """
        Benchmark notebook execution with large dataset.
        
        Args:
            notebook_path: Path to the notebook
            dataset_path: Path to the dataset
            
        Returns:
            BenchmarkResult with notebook performance metrics
        """
        
        print(f"Benchmarking notebook execution: {notebook_path}")
        
        monitor = ResourceMonitor(sampling_interval=2.0)
        monitor.start_monitoring()
        
        start_time = time.time()
        success = True
        error_message = None
        
        try:
            # Execute notebook using nbconvert
            cmd = [
                'jupyter', 'nbconvert', '--to', 'notebook',
                '--execute', '--allow-errors',
                '--ExecutePreprocessor.timeout=3600',
                '--output', f'executed_{os.path.basename(notebook_path)}',
                notebook_path
            ]
            
            # Set environment variable for dataset path
            env = os.environ.copy()
            env['BENCHMARK_DATASET_PATH'] = dataset_path
            
            # Execute notebook
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
            
            if result.returncode != 0:
                success = False
                error_message = f"Notebook execution failed: {result.stderr}"
        
        except subprocess.TimeoutExpired:
            success = False
            error_message = "Notebook execution timed out (1 hour)"
        
        except Exception as e:
            success = False
            error_message = str(e)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Stop monitoring and collect metrics
        metrics = monitor.stop_monitoring()
        
        # Calculate performance statistics
        peak_memory = max([m.memory_mb for m in metrics]) if metrics else 0
        avg_cpu = np.mean([m.cpu_percent for m in metrics]) if metrics else 0
        
        # Estimate dataset size
        dataset_size = 0
        if os.path.exists(dataset_path):
            dataset_size = sum(1 for _ in open(dataset_path)) - 1  # Exclude header
        
        result = BenchmarkResult(
            test_name="notebook_execution",
            dataset_size=dataset_size,
            total_time_seconds=total_time,
            peak_memory_mb=peak_memory,
            avg_cpu_percent=avg_cpu,
            throughput_records_per_second=dataset_size / total_time if total_time > 0 else 0,
            success=success,
            error_message=error_message,
            detailed_metrics=metrics
        )
        
        self.benchmark_results.append(result)
        
        print(f"Notebook Execution Complete:")
        print(f"  Success: {success}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Avg CPU: {avg_cpu:.1f}%")
        
        return result
    
    def identify_bottlenecks(self) -> Dict[str, List[str]]:
        """
        Identify performance bottlenecks from benchmark results.
        
        Returns:
            Dictionary with bottleneck categories and issues
        """
        
        bottlenecks = {
            'memory': [],
            'cpu': [],
            'throughput': [],
            'stability': []
        }
        
        for result in self.benchmark_results:
            # Memory bottlenecks
            if result.peak_memory_mb > self.performance_thresholds['max_memory_mb']:
                bottlenecks['memory'].append(
                    f"{result.test_name}: Peak memory {result.peak_memory_mb:.1f} MB "
                    f"exceeds threshold {self.performance_thresholds['max_memory_mb']} MB"
                )
            
            # CPU bottlenecks
            if result.avg_cpu_percent > self.performance_thresholds['max_cpu_percent']:
                bottlenecks['cpu'].append(
                    f"{result.test_name}: Average CPU {result.avg_cpu_percent:.1f}% "
                    f"exceeds threshold {self.performance_thresholds['max_cpu_percent']}%"
                )
            
            # Throughput bottlenecks
            if result.throughput_records_per_second < self.performance_thresholds['min_throughput_records_per_second']:
                bottlenecks['throughput'].append(
                    f"{result.test_name}: Throughput {result.throughput_records_per_second:.0f} records/sec "
                    f"below threshold {self.performance_thresholds['min_throughput_records_per_second']} records/sec"
                )
            
            # Stability issues
            if not result.success:
                bottlenecks['stability'].append(
                    f"{result.test_name}: Failed with error: {result.error_message}"
                )
        
        return bottlenecks
    
    def generate_scaling_projections(self) -> Dict[str, Any]:
        """
        Generate scaling projections based on benchmark results.
        
        Returns:
            Dictionary with scaling analysis
        """
        
        # Extract data for projection
        data_points = []
        for result in self.benchmark_results:
            if result.success:
                data_points.append({
                    'dataset_size': result.dataset_size,
                    'total_time': result.total_time_seconds,
                    'peak_memory': result.peak_memory_mb,
                    'throughput': result.throughput_records_per_second
                })
        
        if len(data_points) < 2:
            return {'error': 'Insufficient data for scaling projections'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data_points)
        
        # Calculate scaling factors
        scaling_analysis = {
            'time_complexity': self._calculate_time_complexity(df),
            'memory_scaling': self._calculate_memory_scaling(df),
            'throughput_scaling': self._calculate_throughput_scaling(df),
            'projections': self._generate_size_projections(df)
        }
        
        return scaling_analysis
    
    def _calculate_time_complexity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate time complexity scaling."""
        if len(df) < 2:
            return {'slope': 0, 'r_squared': 0}
        
        # Linear regression on log-log scale
        log_size = np.log(df['dataset_size'])
        log_time = np.log(df['total_time'])
        
        slope, intercept = np.polyfit(log_size, log_time, 1)
        r_squared = np.corrcoef(log_size, log_time)[0, 1] ** 2
        
        return {
            'slope': slope,
            'r_squared': r_squared,
            'complexity_class': self._classify_complexity(slope)
        }
    
    def _calculate_memory_scaling(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate memory scaling."""
        if len(df) < 2:
            return {'slope': 0, 'r_squared': 0}
        
        slope, intercept = np.polyfit(df['dataset_size'], df['peak_memory'], 1)
        r_squared = np.corrcoef(df['dataset_size'], df['peak_memory'])[0, 1] ** 2
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        }
    
    def _calculate_throughput_scaling(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate throughput scaling."""
        if len(df) < 2:
            return {'slope': 0, 'r_squared': 0}
        
        slope, intercept = np.polyfit(df['dataset_size'], df['throughput'], 1)
        r_squared = np.corrcoef(df['dataset_size'], df['throughput'])[0, 1] ** 2
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        }
    
    def _classify_complexity(self, slope: float) -> str:
        """Classify algorithmic complexity based on slope."""
        if slope < 0.5:
            return "Sub-linear (better than O(n))"
        elif slope < 1.5:
            return "Linear (O(n))"
        elif slope < 2.5:
            return "Quadratic (O(n²))"
        else:
            return "Exponential (O(n^k), k>2)"
    
    def _generate_size_projections(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate projections for different dataset sizes."""
        projections = {}
        
        # Target sizes (5-year datasets)
        target_sizes = {
            '5min_1year': 105120,  # 5-min intervals for 1 year
            '5min_5years': 525600,  # 5-min intervals for 5 years
            '30min_1year': 17520,   # 30-min intervals for 1 year
            '30min_5years': 87600   # 30-min intervals for 5 years
        }
        
        for size_name, size_value in target_sizes.items():
            # Use linear extrapolation for projections
            if len(df) >= 2:
                # Time projection
                time_slope, time_intercept = np.polyfit(df['dataset_size'], df['total_time'], 1)
                projected_time = time_slope * size_value + time_intercept
                
                # Memory projection
                memory_slope, memory_intercept = np.polyfit(df['dataset_size'], df['peak_memory'], 1)
                projected_memory = memory_slope * size_value + memory_intercept
                
                projections[size_name] = {
                    'dataset_size': size_value,
                    'projected_time_seconds': max(0, projected_time),
                    'projected_memory_mb': max(0, projected_memory),
                    'projected_time_hours': max(0, projected_time) / 3600,
                    'projected_memory_gb': max(0, projected_memory) / 1024
                }
        
        return projections
    
    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report.
        
        Returns:
            Path to the generated report file
        """
        
        report_path = os.path.join(self.output_dir, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_path, 'w') as f:
            f.write("# Performance Validation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System information
            f.write("## System Information\n\n")
            for key, value in self.system_info.items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Benchmark results
            f.write("## Benchmark Results\n\n")
            for result in self.benchmark_results:
                f.write(f"### {result.test_name}\n\n")
                f.write(f"- **Success**: {result.success}\n")
                f.write(f"- **Dataset Size**: {result.dataset_size:,} records\n")
                f.write(f"- **Total Time**: {result.total_time_seconds:.2f} seconds\n")
                f.write(f"- **Peak Memory**: {result.peak_memory_mb:.1f} MB\n")
                f.write(f"- **Average CPU**: {result.avg_cpu_percent:.1f}%\n")
                f.write(f"- **Throughput**: {result.throughput_records_per_second:.0f} records/second\n")
                
                if result.error_message:
                    f.write(f"- **Error**: {result.error_message}\n")
                
                f.write("\n")
            
            # Bottleneck analysis
            f.write("## Bottleneck Analysis\n\n")
            bottlenecks = self.identify_bottlenecks()
            for category, issues in bottlenecks.items():
                f.write(f"### {category.title()} Issues\n\n")
                if issues:
                    for issue in issues:
                        f.write(f"- {issue}\n")
                else:
                    f.write("- No issues detected\n")
                f.write("\n")
            
            # Scaling projections
            f.write("## Scaling Projections\n\n")
            projections = self.generate_scaling_projections()
            if 'error' not in projections:
                f.write("### Time Complexity\n\n")
                tc = projections['time_complexity']
                f.write(f"- **Slope**: {tc['slope']:.3f}\n")
                f.write(f"- **R²**: {tc['r_squared']:.3f}\n")
                f.write(f"- **Complexity Class**: {tc['complexity_class']}\n\n")
                
                f.write("### 5-Year Dataset Projections\n\n")
                for size_name, proj in projections['projections'].items():
                    f.write(f"#### {size_name}\n\n")
                    f.write(f"- **Dataset Size**: {proj['dataset_size']:,} records\n")
                    f.write(f"- **Projected Time**: {proj['projected_time_hours']:.2f} hours\n")
                    f.write(f"- **Projected Memory**: {proj['projected_memory_gb']:.2f} GB\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write(self._generate_recommendations())
        
        return report_path
    
    def _generate_recommendations(self) -> str:
        """Generate performance recommendations."""
        recommendations = []
        
        # Analyze bottlenecks
        bottlenecks = self.identify_bottlenecks()
        
        if bottlenecks['memory']:
            recommendations.append("**Memory Optimization:**")
            recommendations.append("- Implement data chunking for large datasets")
            recommendations.append("- Use memory-mapped files for large data access")
            recommendations.append("- Consider distributed processing for very large datasets")
            recommendations.append("")
        
        if bottlenecks['cpu']:
            recommendations.append("**CPU Optimization:**")
            recommendations.append("- Implement multiprocessing for CPU-intensive tasks")
            recommendations.append("- Use vectorized operations where possible")
            recommendations.append("- Consider GPU acceleration for matrix operations")
            recommendations.append("")
        
        if bottlenecks['throughput']:
            recommendations.append("**Throughput Optimization:**")
            recommendations.append("- Implement asynchronous I/O operations")
            recommendations.append("- Use batch processing for better efficiency")
            recommendations.append("- Consider caching frequently accessed data")
            recommendations.append("")
        
        if bottlenecks['stability']:
            recommendations.append("**Stability Improvements:**")
            recommendations.append("- Implement error handling and recovery mechanisms")
            recommendations.append("- Add progress monitoring and checkpointing")
            recommendations.append("- Use memory profiling to identify leaks")
            recommendations.append("")
        
        # General recommendations
        recommendations.append("**General Recommendations:**")
        recommendations.append("- Regular performance monitoring in production")
        recommendations.append("- Implement auto-scaling based on workload")
        recommendations.append("- Use performance profiling tools for optimization")
        recommendations.append("- Consider cloud-based solutions for scalability")
        
        return "\n".join(recommendations)
    
    def save_benchmark_data(self) -> str:
        """
        Save benchmark data to JSON file.
        
        Returns:
            Path to the saved data file
        """
        
        data_path = os.path.join(self.output_dir, f"benchmark_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Convert benchmark results to JSON-serializable format
        results_data = []
        for result in self.benchmark_results:
            result_dict = asdict(result)
            if result_dict['detailed_metrics']:
                result_dict['detailed_metrics'] = [asdict(m) for m in result_dict['detailed_metrics']]
            results_data.append(result_dict)
        
        output_data = {
            'system_info': self.system_info,
            'benchmark_results': results_data,
            'bottlenecks': self.identify_bottlenecks(),
            'scaling_projections': self.generate_scaling_projections()
        }
        
        with open(data_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return data_path

def main():
    """Main execution function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run performance benchmarks')
    parser.add_argument('--output-dir', default='performance_validation/results', help='Output directory')
    parser.add_argument('--dataset', help='Dataset file for benchmarking')
    parser.add_argument('--notebook', help='Notebook file for benchmarking')
    parser.add_argument('--training-sizes', nargs='+', type=int, default=[10000, 50000, 100000], 
                        help='Dataset sizes for training simulation')
    
    args = parser.parse_args()
    
    # Create benchmark framework
    framework = PerformanceBenchmarkFramework(output_dir=args.output_dir)
    
    # Run data loading benchmark if dataset provided
    if args.dataset and os.path.exists(args.dataset):
        framework.benchmark_data_loading(args.dataset)
    
    # Run training simulations
    for size in args.training_sizes:
        framework.benchmark_training_simulation(size)
    
    # Run notebook benchmark if provided
    if args.notebook and args.dataset and os.path.exists(args.notebook):
        framework.benchmark_notebook_execution(args.notebook, args.dataset)
    
    # Generate reports
    report_path = framework.generate_performance_report()
    data_path = framework.save_benchmark_data()
    
    print(f"\nPerformance validation complete!")
    print(f"Report saved to: {report_path}")
    print(f"Data saved to: {data_path}")

if __name__ == "__main__":
    main()