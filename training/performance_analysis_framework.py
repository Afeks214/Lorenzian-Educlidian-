"""
Comprehensive Performance Analysis and Benchmarking Framework
Provides detailed analysis, profiling, and optimization recommendations for training systems
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import json
import pickle
import psutil
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Profiling imports
import cProfile
import pstats
import tracemalloc
from line_profiler import LineProfiler
import memory_profiler

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    metric_name: str
    value: float
    unit: str
    timestamp: float
    category: str
    device: str
    additional_info: Dict[str, Any] = None

@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    benchmark_name: str
    configuration: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time: float
    memory_usage: Dict[str, float]
    throughput: float
    efficiency_score: float
    recommendations: List[str]
    raw_data: Dict[str, Any] = None

@dataclass
class ProfilingResult:
    """Profiling result data structure"""
    profile_type: str
    duration: float
    top_functions: List[Dict[str, Any]]
    memory_profile: Dict[str, Any]
    bottlenecks: List[str]
    optimization_suggestions: List[str]

class SystemProfiler:
    """System-level performance profiler"""
    
    def __init__(self):
        self.start_time = None
        self.profiles = {}
        self.memory_snapshots = []
        
    def start_profiling(self, profile_name: str):
        """Start profiling session"""
        self.start_time = time.time()
        
        # Start memory tracing
        tracemalloc.start()
        
        # Initial memory snapshot
        self.memory_snapshots.append({
            'timestamp': self.start_time,
            'memory_mb': psutil.Process().memory_info().rss / 1024**2,
            'cpu_percent': psutil.cpu_percent()
        })
        
        # GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.memory_snapshots[-1]['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024**2
        
        logger.info(f"Started profiling session: {profile_name}")
    
    def capture_snapshot(self, label: str = ""):
        """Capture performance snapshot"""
        if self.start_time is None:
            logger.warning("Profiling not started")
            return
        
        snapshot = {
            'timestamp': time.time() - self.start_time,
            'label': label,
            'memory_mb': psutil.Process().memory_info().rss / 1024**2,
            'cpu_percent': psutil.cpu_percent(),
            'memory_peak_mb': tracemalloc.get_traced_memory()[1] / 1024**2
        }
        
        if torch.cuda.is_available():
            snapshot['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024**2
            snapshot['gpu_memory_peak_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        
        self.memory_snapshots.append(snapshot)
        logger.debug(f"Captured snapshot: {label}")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return results"""
        if self.start_time is None:
            logger.warning("Profiling not started")
            return {}
        
        # Final snapshot
        self.capture_snapshot("final")
        
        # Stop memory tracing
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate statistics
        total_time = time.time() - self.start_time
        
        memory_values = [s['memory_mb'] for s in self.memory_snapshots]
        cpu_values = [s['cpu_percent'] for s in self.memory_snapshots if 'cpu_percent' in s]
        
        results = {
            'total_time': total_time,
            'memory_stats': {
                'peak_mb': max(memory_values),
                'avg_mb': np.mean(memory_values),
                'growth_mb': memory_values[-1] - memory_values[0]
            },
            'cpu_stats': {
                'avg_percent': np.mean(cpu_values) if cpu_values else 0,
                'max_percent': max(cpu_values) if cpu_values else 0
            },
            'snapshots': self.memory_snapshots.copy(),
            'tracemalloc_peak_mb': peak / 1024**2
        }
        
        if torch.cuda.is_available():
            gpu_memory_values = [s.get('gpu_memory_mb', 0) for s in self.memory_snapshots]
            results['gpu_stats'] = {
                'peak_mb': max(gpu_memory_values),
                'avg_mb': np.mean(gpu_memory_values),
                'final_mb': gpu_memory_values[-1]
            }
        
        # Reset state
        self.start_time = None
        self.memory_snapshots = []
        
        return results

class TrainingProfiler:
    """Training-specific performance profiler"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timers = {}
        self.counters = defaultdict(int)
        
    def start_timer(self, name: str):
        """Start timing operation"""
        self.timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str) -> float:
        """Stop timing operation and record"""
        if name not in self.timers:
            logger.warning(f"Timer {name} not started")
            return 0.0
        
        elapsed = time.perf_counter() - self.timers[name]
        self.metrics[f"{name}_time"].append(elapsed)
        del self.timers[name]
        return elapsed
    
    def record_metric(self, name: str, value: float, unit: str = ""):
        """Record performance metric"""
        self.metrics[name].append(value)
        logger.debug(f"Recorded metric {name}: {value} {unit}")
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment counter"""
        self.counters[name] += value
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Profile a function call"""
        # Memory before
        memory_before = psutil.Process().memory_info().rss / 1024**2
        
        # Execute function
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        # Memory after
        memory_after = psutil.Process().memory_info().rss / 1024**2
        
        profile_data = {
            'execution_time': execution_time,
            'memory_delta_mb': memory_after - memory_before,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after
        }
        
        return result, profile_data
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        summary = {
            'metrics': {},
            'counters': dict(self.counters),
            'active_timers': list(self.timers.keys())
        }
        
        # Calculate statistics for each metric
        for metric_name, values in self.metrics.items():
            if values:
                summary['metrics'][metric_name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'total': np.sum(values)
                }
        
        return summary

class ModelComplexityAnalyzer:
    """Analyze model complexity and computational requirements"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.complexity_metrics = {}
        
    def analyze_parameters(self) -> Dict[str, Any]:
        """Analyze model parameters"""
        total_params = 0
        trainable_params = 0
        layer_params = {}
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
            
            layer_params[name] = {
                'shape': list(param.shape),
                'parameters': param_count,
                'trainable': param.requires_grad,
                'dtype': str(param.dtype)
            }
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'layer_parameters': layer_params,
            'parameter_size_mb': total_params * 4 / 1024**2  # Assuming float32
        }
    
    def analyze_flops(self, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Analyze FLOPs (Floating Point Operations)"""
        try:
            from torchprofile import profile_macs
            
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape)
            
            # Profile MACs (Multiply-Accumulate operations)
            macs = profile_macs(self.model, dummy_input)
            
            # Estimate FLOPs (typically 2 * MACs for dense layers)
            flops = macs * 2
            
            return {
                'macs': macs,
                'flops': flops,
                'gflops': flops / 1e9,
                'input_shape': input_shape
            }
            
        except ImportError:
            logger.warning("torchprofile not available, skipping FLOP analysis")
            return {'error': 'torchprofile_not_available'}
    
    def analyze_memory_usage(self, batch_size: int, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Analyze memory usage"""
        # Parameter memory
        param_analysis = self.analyze_parameters()
        param_memory_mb = param_analysis['parameter_size_mb']
        
        # Activation memory estimation
        activation_memory_mb = 0
        
        # Simple estimation based on layer types
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Linear layer activation memory
                activation_memory_mb += module.out_features * batch_size * 4 / 1024**2
            elif isinstance(module, nn.Conv2d):
                # Conv2d layer activation memory (rough estimate)
                activation_memory_mb += module.out_channels * batch_size * 4 / 1024**2
        
        # Gradient memory (same as parameters)
        gradient_memory_mb = param_memory_mb
        
        # Optimizer memory (depends on optimizer, estimate for Adam)
        optimizer_memory_mb = param_memory_mb * 2  # momentum + variance
        
        total_memory_mb = param_memory_mb + activation_memory_mb + gradient_memory_mb + optimizer_memory_mb
        
        return {
            'parameter_memory_mb': param_memory_mb,
            'activation_memory_mb': activation_memory_mb,
            'gradient_memory_mb': gradient_memory_mb,
            'optimizer_memory_mb': optimizer_memory_mb,
            'total_memory_mb': total_memory_mb,
            'batch_size': batch_size
        }
    
    def get_complexity_report(self, input_shape: Tuple[int, ...], batch_size: int = 1) -> Dict[str, Any]:
        """Get comprehensive complexity report"""
        return {
            'parameters': self.analyze_parameters(),
            'flops': self.analyze_flops(input_shape),
            'memory': self.analyze_memory_usage(batch_size, input_shape),
            'model_size_mb': sum(p.numel() * 4 for p in self.model.parameters()) / 1024**2
        }

class TrainingBenchmark:
    """Comprehensive training benchmark suite"""
    
    def __init__(self, model_factory: Callable, optimizer_factory: Callable):
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.benchmark_results = []
        
    def benchmark_batch_sizes(self, 
                            batch_sizes: List[int],
                            input_shape: Tuple[int, ...],
                            num_iterations: int = 100,
                            device: str = 'cuda') -> List[BenchmarkResult]:
        """Benchmark different batch sizes"""
        results = []
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch size: {batch_size}")
            
            try:
                # Create model and optimizer
                model = self.model_factory().to(device)
                optimizer = self.optimizer_factory(model.parameters())
                
                # Warm up
                self._warmup(model, optimizer, batch_size, input_shape, device)
                
                # Benchmark
                metrics = self._benchmark_training_loop(
                    model, optimizer, batch_size, input_shape, num_iterations, device
                )
                
                result = BenchmarkResult(
                    benchmark_name=f"batch_size_{batch_size}",
                    configuration={'batch_size': batch_size, 'input_shape': input_shape},
                    metrics=metrics,
                    execution_time=metrics['total_time'],
                    memory_usage=metrics['memory'],
                    throughput=metrics['throughput'],
                    efficiency_score=self._calculate_efficiency_score(metrics),
                    recommendations=self._generate_batch_size_recommendations(batch_size, metrics)
                )
                
                results.append(result)
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM for batch size {batch_size}")
                result = BenchmarkResult(
                    benchmark_name=f"batch_size_{batch_size}",
                    configuration={'batch_size': batch_size, 'input_shape': input_shape},
                    metrics={'error': 'OOM'},
                    execution_time=0,
                    memory_usage={'error': 'OOM'},
                    throughput=0,
                    efficiency_score=0,
                    recommendations=['Reduce batch size due to OOM']
                )
                results.append(result)
            
            except Exception as e:
                logger.error(f"Benchmark failed for batch size {batch_size}: {e}")
                continue
            
            # Clean up
            del model, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def benchmark_optimizers(self,
                           optimizer_configs: List[Dict[str, Any]],
                           batch_size: int,
                           input_shape: Tuple[int, ...],
                           num_iterations: int = 100,
                           device: str = 'cuda') -> List[BenchmarkResult]:
        """Benchmark different optimizers"""
        results = []
        
        for config in optimizer_configs:
            logger.info(f"Benchmarking optimizer: {config['name']}")
            
            try:
                # Create model
                model = self.model_factory().to(device)
                
                # Create optimizer
                optimizer_class = config['class']
                optimizer_params = config.get('params', {})
                optimizer = optimizer_class(model.parameters(), **optimizer_params)
                
                # Warm up
                self._warmup(model, optimizer, batch_size, input_shape, device)
                
                # Benchmark
                metrics = self._benchmark_training_loop(
                    model, optimizer, batch_size, input_shape, num_iterations, device
                )
                
                result = BenchmarkResult(
                    benchmark_name=f"optimizer_{config['name']}",
                    configuration=config,
                    metrics=metrics,
                    execution_time=metrics['total_time'],
                    memory_usage=metrics['memory'],
                    throughput=metrics['throughput'],
                    efficiency_score=self._calculate_efficiency_score(metrics),
                    recommendations=self._generate_optimizer_recommendations(config, metrics)
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Benchmark failed for optimizer {config['name']}: {e}")
                continue
            
            # Clean up
            del model, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def benchmark_precision_modes(self,
                                 precision_modes: List[str],
                                 batch_size: int,
                                 input_shape: Tuple[int, ...],
                                 num_iterations: int = 100,
                                 device: str = 'cuda') -> List[BenchmarkResult]:
        """Benchmark different precision modes"""
        results = []
        
        for precision_mode in precision_modes:
            logger.info(f"Benchmarking precision mode: {precision_mode}")
            
            try:
                # Create model and optimizer
                model = self.model_factory().to(device)
                optimizer = self.optimizer_factory(model.parameters())
                
                # Setup precision mode
                scaler = None
                if precision_mode == 'mixed':
                    scaler = torch.cuda.amp.GradScaler()
                
                # Warm up
                self._warmup(model, optimizer, batch_size, input_shape, device, precision_mode)
                
                # Benchmark
                metrics = self._benchmark_training_loop(
                    model, optimizer, batch_size, input_shape, num_iterations, device, precision_mode
                )
                
                result = BenchmarkResult(
                    benchmark_name=f"precision_{precision_mode}",
                    configuration={'precision_mode': precision_mode},
                    metrics=metrics,
                    execution_time=metrics['total_time'],
                    memory_usage=metrics['memory'],
                    throughput=metrics['throughput'],
                    efficiency_score=self._calculate_efficiency_score(metrics),
                    recommendations=self._generate_precision_recommendations(precision_mode, metrics)
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Benchmark failed for precision mode {precision_mode}: {e}")
                continue
            
            # Clean up
            del model, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def _warmup(self, model, optimizer, batch_size, input_shape, device, precision_mode='float32'):
        """Warm up model for accurate benchmarking"""
        model.train()
        
        for _ in range(5):  # 5 warmup iterations
            batch = torch.randn(batch_size, *input_shape).to(device)
            target = torch.randn(batch_size, 10).to(device)  # Assuming 10 outputs
            
            optimizer.zero_grad()
            
            if precision_mode == 'mixed':
                with torch.cuda.amp.autocast():
                    output = model(batch)
                    loss = nn.MSELoss()(output, target)
            else:
                output = model(batch)
                loss = nn.MSELoss()(output, target)
            
            loss.backward()
            optimizer.step()
    
    def _benchmark_training_loop(self, model, optimizer, batch_size, input_shape, num_iterations, device, precision_mode='float32'):
        """Benchmark training loop"""
        model.train()
        
        # Initialize profiler
        profiler = TrainingProfiler()
        system_profiler = SystemProfiler()
        
        # Start profiling
        system_profiler.start_profiling("training_loop")
        
        # Training loop
        total_loss = 0.0
        forward_times = []
        backward_times = []
        
        for i in range(num_iterations):
            # Generate batch
            batch = torch.randn(batch_size, *input_shape).to(device)
            target = torch.randn(batch_size, 10).to(device)
            
            # Forward pass
            profiler.start_timer("forward")
            optimizer.zero_grad()
            
            if precision_mode == 'mixed':
                with torch.cuda.amp.autocast():
                    output = model(batch)
                    loss = nn.MSELoss()(output, target)
            else:
                output = model(batch)
                loss = nn.MSELoss()(output, target)
            
            forward_time = profiler.stop_timer("forward")
            forward_times.append(forward_time)
            
            # Backward pass
            profiler.start_timer("backward")
            loss.backward()
            optimizer.step()
            backward_time = profiler.stop_timer("backward")
            backward_times.append(backward_time)
            
            total_loss += loss.item()
            
            # Capture memory snapshot periodically
            if i % 20 == 0:
                system_profiler.capture_snapshot(f"iteration_{i}")
        
        # Stop profiling
        system_results = system_profiler.stop_profiling()
        
        # Calculate metrics
        avg_forward_time = np.mean(forward_times)
        avg_backward_time = np.mean(backward_times)
        total_time = system_results['total_time']
        avg_loss = total_loss / num_iterations
        
        throughput = (num_iterations * batch_size) / total_time
        
        metrics = {
            'total_time': total_time,
            'avg_forward_time': avg_forward_time,
            'avg_backward_time': avg_backward_time,
            'avg_loss': avg_loss,
            'throughput': throughput,
            'samples_per_second': throughput,
            'memory': system_results.get('memory_stats', {}),
            'cpu': system_results.get('cpu_stats', {}),
            'gpu': system_results.get('gpu_stats', {})
        }
        
        return metrics
    
    def _calculate_efficiency_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate efficiency score (higher is better)"""
        # Simple efficiency score based on throughput and memory usage
        throughput = metrics.get('throughput', 0)
        memory_usage = metrics.get('memory', {}).get('peak_mb', 1)
        
        # Normalize and combine
        efficiency = (throughput / 1000) / (memory_usage / 1000)  # samples per second per GB
        
        return min(100, max(0, efficiency))  # Cap between 0 and 100
    
    def _generate_batch_size_recommendations(self, batch_size: int, metrics: Dict[str, Any]) -> List[str]:
        """Generate batch size recommendations"""
        recommendations = []
        
        memory_usage = metrics.get('memory', {}).get('peak_mb', 0)
        throughput = metrics.get('throughput', 0)
        
        if memory_usage > 8000:  # > 8GB
            recommendations.append("Consider reducing batch size to save memory")
        
        if throughput < 100:  # Low throughput
            recommendations.append("Consider increasing batch size for better GPU utilization")
        
        if metrics.get('error') == 'OOM':
            recommendations.append("Reduce batch size or use gradient accumulation")
        
        return recommendations
    
    def _generate_optimizer_recommendations(self, config: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
        """Generate optimizer recommendations"""
        recommendations = []
        
        optimizer_name = config['name']
        convergence_rate = 1.0 / metrics.get('avg_loss', 1.0)
        
        if optimizer_name == 'SGD' and convergence_rate < 0.5:
            recommendations.append("Consider using Adam or AdamW for faster convergence")
        
        if optimizer_name == 'Adam' and metrics.get('memory', {}).get('peak_mb', 0) > 6000:
            recommendations.append("Consider using SGD to reduce memory usage")
        
        return recommendations
    
    def _generate_precision_recommendations(self, precision_mode: str, metrics: Dict[str, Any]) -> List[str]:
        """Generate precision mode recommendations"""
        recommendations = []
        
        if precision_mode == 'float32':
            recommendations.append("Consider mixed precision for faster training and lower memory usage")
        
        if precision_mode == 'mixed' and metrics.get('avg_loss', 0) > 10:
            recommendations.append("Check for numerical instability with mixed precision")
        
        return recommendations

class PerformanceAnalysisFramework:
    """Comprehensive performance analysis framework"""
    
    def __init__(self, results_dir: str = "performance_analysis"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.analysis_results = {}
        self.benchmark_results = []
        self.profiling_results = []
        
    def analyze_model_complexity(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Analyze model complexity"""
        logger.info("Analyzing model complexity")
        
        analyzer = ModelComplexityAnalyzer(model)
        complexity_report = analyzer.get_complexity_report(input_shape)
        
        self.analysis_results['model_complexity'] = complexity_report
        return complexity_report
    
    def run_comprehensive_benchmark(self,
                                  model_factory: Callable,
                                  optimizer_factory: Callable,
                                  input_shape: Tuple[int, ...],
                                  device: str = 'cuda') -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        logger.info("Running comprehensive benchmark suite")
        
        benchmark = TrainingBenchmark(model_factory, optimizer_factory)
        
        # Benchmark batch sizes
        batch_sizes = [16, 32, 64, 128, 256]
        batch_results = benchmark.benchmark_batch_sizes(batch_sizes, input_shape, device=device)
        
        # Benchmark optimizers
        optimizer_configs = [
            {'name': 'SGD', 'class': optim.SGD, 'params': {'lr': 0.01}},
            {'name': 'Adam', 'class': optim.Adam, 'params': {'lr': 0.001}},
            {'name': 'AdamW', 'class': optim.AdamW, 'params': {'lr': 0.001}}
        ]
        optimizer_results = benchmark.benchmark_optimizers(optimizer_configs, 64, input_shape, device=device)
        
        # Benchmark precision modes
        precision_modes = ['float32', 'mixed']
        precision_results = benchmark.benchmark_precision_modes(precision_modes, 64, input_shape, device=device)
        
        benchmark_summary = {
            'batch_sizes': batch_results,
            'optimizers': optimizer_results,
            'precision_modes': precision_results
        }
        
        self.benchmark_results.extend(batch_results + optimizer_results + precision_results)
        self.analysis_results['benchmarks'] = benchmark_summary
        
        return benchmark_summary
    
    def profile_training_session(self,
                               model: nn.Module,
                               optimizer: optim.Optimizer,
                               data_loader,
                               num_epochs: int = 1,
                               device: str = 'cuda') -> Dict[str, Any]:
        """Profile training session"""
        logger.info("Profiling training session")
        
        profiler = TrainingProfiler()
        system_profiler = SystemProfiler()
        
        # Start profiling
        system_profiler.start_profiling("training_session")
        
        model.train()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                profiler.start_timer("forward")
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                profiler.stop_timer("forward")
                
                # Backward pass
                profiler.start_timer("backward")
                loss.backward()
                optimizer.step()
                profiler.stop_timer("backward")
                
                # Record metrics
                profiler.record_metric("loss", loss.item())
                profiler.increment_counter("batches_processed")
                
                # Memory snapshot
                if batch_idx % 100 == 0:
                    system_profiler.capture_snapshot(f"epoch_{epoch}_batch_{batch_idx}")
        
        # Stop profiling
        system_results = system_profiler.stop_profiling()
        training_summary = profiler.get_summary()
        
        profile_result = {
            'system_profile': system_results,
            'training_profile': training_summary,
            'recommendations': self._generate_training_recommendations(system_results, training_summary)
        }
        
        self.profiling_results.append(profile_result)
        self.analysis_results['training_profile'] = profile_result
        
        return profile_result
    
    def analyze_scalability(self,
                          model_factory: Callable,
                          data_sizes: List[int],
                          input_shape: Tuple[int, ...],
                          device: str = 'cuda') -> Dict[str, Any]:
        """Analyze scalability with different data sizes"""
        logger.info("Analyzing scalability")
        
        scalability_results = {}
        
        for data_size in data_sizes:
            logger.info(f"Testing scalability with {data_size} samples")
            
            # Create model
            model = model_factory().to(device)
            optimizer = optim.Adam(model.parameters())
            
            # Generate data
            data = torch.randn(data_size, *input_shape).to(device)
            targets = torch.randint(0, 10, (data_size,)).to(device)
            
            # Measure training time
            start_time = time.time()
            
            # Training loop
            for i in range(0, data_size, 32):  # Batch size 32
                batch_data = data[i:i+32]
                batch_targets = targets[i:i+32]
                
                optimizer.zero_grad()
                output = model(batch_data)
                loss = nn.CrossEntropyLoss()(output, batch_targets)
                loss.backward()
                optimizer.step()
            
            training_time = time.time() - start_time
            
            scalability_results[data_size] = {
                'training_time': training_time,
                'time_per_sample': training_time / data_size,
                'throughput': data_size / training_time
            }
            
            # Clean up
            del model, optimizer, data, targets
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.analysis_results['scalability'] = scalability_results
        return scalability_results
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        logger.info("Generating optimization report")
        
        report = {
            'summary': {
                'total_benchmarks': len(self.benchmark_results),
                'analysis_timestamp': time.time(),
                'device_info': self._get_device_info()
            },
            'analysis_results': self.analysis_results,
            'recommendations': self._generate_comprehensive_recommendations(),
            'performance_summary': self._generate_performance_summary()
        }
        
        # Save report
        report_file = self.results_dir / "optimization_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_visualizations()
        
        logger.info(f"Optimization report generated: {report_file}")
        return report
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        device_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3,
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            device_info['gpu_count'] = torch.cuda.device_count()
            device_info['gpu_name'] = torch.cuda.get_device_name(0)
            device_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return device_info
    
    def _generate_training_recommendations(self, system_results: Dict[str, Any], training_summary: Dict[str, Any]) -> List[str]:
        """Generate training recommendations"""
        recommendations = []
        
        # Memory recommendations
        peak_memory = system_results.get('memory_stats', {}).get('peak_mb', 0)
        if peak_memory > 8000:
            recommendations.append("Consider reducing batch size or using gradient accumulation")
        
        # CPU recommendations
        avg_cpu = system_results.get('cpu_stats', {}).get('avg_percent', 0)
        if avg_cpu < 50:
            recommendations.append("CPU utilization is low, consider increasing data loading workers")
        
        # Training time recommendations
        forward_time = training_summary.get('metrics', {}).get('forward_time', {}).get('mean', 0)
        backward_time = training_summary.get('metrics', {}).get('backward_time', {}).get('mean', 0)
        
        if forward_time > backward_time * 2:
            recommendations.append("Forward pass is slow, consider model optimization")
        
        return recommendations
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Generate comprehensive optimization recommendations"""
        recommendations = []
        
        # Model complexity recommendations
        if 'model_complexity' in self.analysis_results:
            complexity = self.analysis_results['model_complexity']
            total_params = complexity.get('parameters', {}).get('total_parameters', 0)
            
            if total_params > 100_000_000:  # > 100M parameters
                recommendations.append("Large model detected, consider model pruning or quantization")
        
        # Benchmark recommendations
        if self.benchmark_results:
            best_batch_size = self._find_optimal_batch_size()
            if best_batch_size:
                recommendations.append(f"Optimal batch size appears to be {best_batch_size}")
        
        # Scalability recommendations
        if 'scalability' in self.analysis_results:
            scalability = self.analysis_results['scalability']
            times = [result['time_per_sample'] for result in scalability.values()]
            if len(times) > 1 and times[-1] > times[0] * 2:
                recommendations.append("Training time doesn't scale linearly, consider optimization")
        
        return recommendations
    
    def _find_optimal_batch_size(self) -> Optional[int]:
        """Find optimal batch size from benchmark results"""
        batch_size_results = [r for r in self.benchmark_results if 'batch_size' in r.benchmark_name]
        
        if not batch_size_results:
            return None
        
        # Find batch size with best efficiency score
        best_result = max(batch_size_results, key=lambda r: r.efficiency_score)
        
        # Extract batch size from configuration
        return best_result.configuration.get('batch_size')
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {}
        
        if self.benchmark_results:
            # Throughput summary
            throughputs = [r.throughput for r in self.benchmark_results if r.throughput > 0]
            if throughputs:
                summary['throughput'] = {
                    'max': max(throughputs),
                    'min': min(throughputs),
                    'avg': np.mean(throughputs)
                }
            
            # Memory summary
            memory_usages = []
            for result in self.benchmark_results:
                if isinstance(result.memory_usage, dict) and 'peak_mb' in result.memory_usage:
                    memory_usages.append(result.memory_usage['peak_mb'])
            
            if memory_usages:
                summary['memory_usage'] = {
                    'max_mb': max(memory_usages),
                    'min_mb': min(memory_usages),
                    'avg_mb': np.mean(memory_usages)
                }
        
        return summary
    
    def _generate_visualizations(self):
        """Generate performance visualizations"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Throughput comparison
            if self.benchmark_results:
                benchmark_names = [r.benchmark_name for r in self.benchmark_results]
                throughputs = [r.throughput for r in self.benchmark_results]
                
                axes[0, 0].bar(benchmark_names, throughputs)
                axes[0, 0].set_title('Throughput Comparison')
                axes[0, 0].set_ylabel('Samples/Second')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Memory usage
            if self.benchmark_results:
                memory_usages = []
                for result in self.benchmark_results:
                    if isinstance(result.memory_usage, dict) and 'peak_mb' in result.memory_usage:
                        memory_usages.append(result.memory_usage['peak_mb'])
                    else:
                        memory_usages.append(0)
                
                axes[0, 1].bar(benchmark_names, memory_usages)
                axes[0, 1].set_title('Memory Usage')
                axes[0, 1].set_ylabel('Memory (MB)')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Efficiency scores
            if self.benchmark_results:
                efficiency_scores = [r.efficiency_score for r in self.benchmark_results]
                
                axes[1, 0].bar(benchmark_names, efficiency_scores)
                axes[1, 0].set_title('Efficiency Scores')
                axes[1, 0].set_ylabel('Efficiency Score')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Scalability plot
            if 'scalability' in self.analysis_results:
                scalability = self.analysis_results['scalability']
                data_sizes = list(scalability.keys())
                training_times = [scalability[size]['training_time'] for size in data_sizes]
                
                axes[1, 1].plot(data_sizes, training_times, 'o-')
                axes[1, 1].set_title('Scalability Analysis')
                axes[1, 1].set_xlabel('Data Size')
                axes[1, 1].set_ylabel('Training Time (s)')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")


# Example usage functions
def create_example_model():
    """Create example model for testing"""
    return nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def create_example_optimizer(params):
    """Create example optimizer"""
    return optim.Adam(params, lr=0.001)

def run_performance_analysis_example():
    """Run example performance analysis"""
    
    # Initialize framework
    framework = PerformanceAnalysisFramework()
    
    # Create model
    model = create_example_model()
    
    # Analyze model complexity
    complexity_report = framework.analyze_model_complexity(model, (100,))
    print("Model Complexity Analysis:")
    print(f"Total Parameters: {complexity_report['parameters']['total_parameters']:,}")
    print(f"Model Size: {complexity_report['model_size_mb']:.2f} MB")
    
    # Run comprehensive benchmark
    benchmark_results = framework.run_comprehensive_benchmark(
        create_example_model,
        create_example_optimizer,
        (100,),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate optimization report
    report = framework.generate_optimization_report()
    
    print("\nOptimization Report Generated!")
    print(f"Total Benchmarks: {report['summary']['total_benchmarks']}")
    print("Top Recommendations:")
    for rec in report['recommendations'][:3]:
        print(f"- {rec}")
    
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = run_performance_analysis_example()
    print("Performance analysis completed successfully!")