"""
CPU Optimization System for GrandModel

This module implements comprehensive CPU optimization strategies including:
- PyTorch JIT compilation for inference
- Vectorization for mathematical operations
- Thread pool optimization
- CPU affinity settings
- SIMD operations
- Batch processing optimization

Key Performance Targets:
- 30% reduction in CPU usage for inference
- 20% improvement in mathematical operations
- Optimal thread utilization
- Enhanced cache locality
"""

import torch
import torch.jit
import numpy as np
import numba
from numba import jit, vectorize, prange
import threading
import multiprocessing
import concurrent.futures
import psutil
import os
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import logging
from datetime import datetime
import structlog
from collections import defaultdict, deque
import warnings

# Suppress numba warnings
warnings.filterwarnings('ignore', category=numba.NumbaDeprecationWarning)
warnings.filterwarnings('ignore', category=numba.NumbaWarning)

logger = structlog.get_logger()


@dataclass
class CPUStats:
    """CPU performance statistics"""
    timestamp: datetime
    cpu_percent: float
    cpu_count: int
    cpu_freq: float
    load_avg: Tuple[float, float, float]
    context_switches: int
    interrupts: int
    thread_count: int
    jit_compilation_time: float = 0.0
    vectorized_operations: int = 0
    batch_operations: int = 0


class JITCompiler:
    """
    PyTorch JIT compilation system for inference optimization.
    Provides significant speedup for inference operations.
    """
    
    def __init__(self):
        self.compiled_models = {}
        self.compilation_stats = {}
        self.warmup_iterations = 5
        
        logger.info("JITCompiler initialized")
    
    def compile_model(self, model: torch.nn.Module, 
                     example_inputs: torch.Tensor,
                     model_name: str = None) -> torch.jit.ScriptModule:
        """Compile PyTorch model for optimized inference"""
        
        if model_name is None:
            model_name = f"model_{id(model)}"
        
        if model_name in self.compiled_models:
            return self.compiled_models[model_name]
        
        logger.info("Compiling model", model_name=model_name)
        start_time = time.time()
        
        # Set model to eval mode
        model.eval()
        
        # Compile using torch.jit.trace
        with torch.no_grad():
            try:
                compiled_model = torch.jit.trace(model, example_inputs)
                
                # Warmup compiled model
                for _ in range(self.warmup_iterations):
                    _ = compiled_model(example_inputs)
                
                # Optimize for inference
                compiled_model = torch.jit.optimize_for_inference(compiled_model)
                
                compilation_time = time.time() - start_time
                
                self.compiled_models[model_name] = compiled_model
                self.compilation_stats[model_name] = {
                    'compilation_time': compilation_time,
                    'input_shape': example_inputs.shape,
                    'parameter_count': sum(p.numel() for p in model.parameters()),
                    'compiled_at': datetime.now()
                }
                
                logger.info("Model compiled successfully",
                           model_name=model_name,
                           compilation_time=compilation_time)
                
                return compiled_model
                
            except Exception as e:
                logger.error("Model compilation failed",
                            model_name=model_name,
                            error=str(e))
                return model
    
    def compile_function(self, func: Callable, 
                        example_inputs: Tuple,
                        function_name: str = None) -> torch.jit.ScriptFunction:
        """Compile standalone function"""
        
        if function_name is None:
            function_name = f"func_{func.__name__}"
        
        logger.info("Compiling function", function_name=function_name)
        
        try:
            # Use torch.jit.script for function compilation
            compiled_func = torch.jit.script(func)
            
            # Warmup
            for _ in range(self.warmup_iterations):
                _ = compiled_func(*example_inputs)
            
            logger.info("Function compiled successfully", 
                       function_name=function_name)
            
            return compiled_func
            
        except Exception as e:
            logger.error("Function compilation failed",
                        function_name=function_name,
                        error=str(e))
            return func
    
    def get_compiled_model(self, model_name: str) -> Optional[torch.jit.ScriptModule]:
        """Get compiled model by name"""
        return self.compiled_models.get(model_name)
    
    def get_compilation_stats(self) -> Dict:
        """Get compilation statistics"""
        return {
            'total_models': len(self.compiled_models),
            'compilation_stats': self.compilation_stats.copy(),
            'warmup_iterations': self.warmup_iterations
        }


class VectorizedOps:
    """
    Vectorized mathematical operations using NumPy and Numba.
    Provides significant speedup for mathematical computations.
    """
    
    def __init__(self):
        self.operation_cache = {}
        self.performance_stats = defaultdict(list)
        
        logger.info("VectorizedOps initialized")
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_correlation(x: np.ndarray, y: np.ndarray) -> float:
        """Fast correlation calculation using Numba"""
        n = x.shape[0]
        
        # Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate correlation
        numerator = 0.0
        x_var = 0.0
        y_var = 0.0
        
        for i in prange(n):
            x_diff = x[i] - x_mean
            y_diff = y[i] - y_mean
            
            numerator += x_diff * y_diff
            x_var += x_diff * x_diff
            y_var += y_diff * y_diff
        
        return numerator / np.sqrt(x_var * y_var)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Fast moving average calculation"""
        n = len(data)
        result = np.zeros(n)
        
        # Calculate first window
        window_sum = 0.0
        for i in range(window):
            window_sum += data[i]
        result[window - 1] = window_sum / window
        
        # Rolling calculation
        for i in range(window, n):
            window_sum = window_sum - data[i - window] + data[i]
            result[i] = window_sum / window
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_volatility(returns: np.ndarray, window: int) -> np.ndarray:
        """Fast rolling volatility calculation"""
        n = len(returns)
        result = np.zeros(n)
        
        for i in prange(window, n):
            window_data = returns[i - window:i]
            mean_val = np.mean(window_data)
            variance = np.mean((window_data - mean_val) ** 2)
            result[i] = np.sqrt(variance)
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def fast_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fast matrix multiplication using parallel processing"""
        m, k = a.shape
        k2, n = b.shape
        
        if k != k2:
            raise ValueError("Matrix dimensions don't match")
        
        c = np.zeros((m, n))
        
        for i in prange(m):
            for j in range(n):
                for p in range(k):
                    c[i, j] += a[i, p] * b[p, j]
        
        return c
    
    @staticmethod
    @vectorize(['float64(float64, float64)'], nopython=True)
    def fast_element_wise_operations(x: float, y: float) -> float:
        """Fast element-wise operations"""
        return x * y + np.sin(x) * np.cos(y)
    
    def benchmark_operation(self, operation_name: str, 
                          operation_func: Callable,
                          *args, **kwargs) -> Dict:
        """Benchmark a vectorized operation"""
        
        start_time = time.perf_counter()
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = (end_time - start_time) * 1000  # ms
        
        stats = {
            'operation': operation_name,
            'execution_time_ms': execution_time,
            'timestamp': datetime.now(),
            'input_sizes': [arg.shape if hasattr(arg, 'shape') else len(arg) 
                           if hasattr(arg, '__len__') else 1 for arg in args]
        }
        
        self.performance_stats[operation_name].append(stats)
        
        logger.info("Operation benchmarked",
                   operation=operation_name,
                   time_ms=execution_time)
        
        return stats
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for all operations"""
        stats = {}
        
        for operation, measurements in self.performance_stats.items():
            if measurements:
                times = [m['execution_time_ms'] for m in measurements]
                stats[operation] = {
                    'count': len(measurements),
                    'avg_time_ms': np.mean(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'std_time_ms': np.std(times)
                }
        
        return stats


class ThreadPoolOptimizer:
    """
    Thread pool optimization for CPU-bound tasks.
    Optimizes thread usage based on CPU cores and workload.
    """
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_thread_count = self._calculate_optimal_threads()
        self.thread_pools = {}
        self.execution_stats = defaultdict(list)
        
        logger.info("ThreadPoolOptimizer initialized",
                   cpu_count=self.cpu_count,
                   optimal_threads=self.optimal_thread_count)
    
    def _calculate_optimal_threads(self) -> int:
        """Calculate optimal thread count based on CPU characteristics"""
        # For CPU-bound tasks, typically CPU count is optimal
        # For I/O-bound tasks, 2-4x CPU count is better
        
        # Check if hyperthreading is available
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        
        if logical_cores > physical_cores:
            # Hyperthreading available, use physical cores for CPU-bound
            return physical_cores
        else:
            return logical_cores
    
    def create_thread_pool(self, pool_name: str, 
                          max_workers: Optional[int] = None,
                          thread_name_prefix: str = None) -> concurrent.futures.ThreadPoolExecutor:
        """Create optimized thread pool"""
        
        if max_workers is None:
            max_workers = self.optimal_thread_count
        
        if thread_name_prefix is None:
            thread_name_prefix = f"GrandModel-{pool_name}"
        
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix
        )
        
        self.thread_pools[pool_name] = executor
        
        logger.info("Thread pool created",
                   pool_name=pool_name,
                   max_workers=max_workers)
        
        return executor
    
    def execute_parallel(self, pool_name: str, 
                        func: Callable, 
                        args_list: List[Tuple],
                        timeout: Optional[float] = None) -> List[Any]:
        """Execute function in parallel using thread pool"""
        
        if pool_name not in self.thread_pools:
            self.create_thread_pool(pool_name)
        
        executor = self.thread_pools[pool_name]
        
        start_time = time.perf_counter()
        
        # Submit all tasks
        futures = [executor.submit(func, *args) for args in args_list]
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error("Task execution failed", error=str(e))
                results.append(None)
        
        execution_time = time.perf_counter() - start_time
        
        # Record statistics
        stats = {
            'pool_name': pool_name,
            'task_count': len(args_list),
            'execution_time_ms': execution_time * 1000,
            'timestamp': datetime.now(),
            'success_rate': len([r for r in results if r is not None]) / len(results)
        }
        
        self.execution_stats[pool_name].append(stats)
        
        logger.info("Parallel execution completed",
                   pool_name=pool_name,
                   tasks=len(args_list),
                   time_ms=stats['execution_time_ms'])
        
        return results
    
    def get_thread_pool(self, pool_name: str) -> Optional[concurrent.futures.ThreadPoolExecutor]:
        """Get thread pool by name"""
        return self.thread_pools.get(pool_name)
    
    def shutdown_all_pools(self):
        """Shutdown all thread pools"""
        for pool_name, executor in self.thread_pools.items():
            executor.shutdown(wait=True)
            logger.info("Thread pool shutdown", pool_name=pool_name)
        
        self.thread_pools.clear()
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics for all pools"""
        stats = {}
        
        for pool_name, measurements in self.execution_stats.items():
            if measurements:
                times = [m['execution_time_ms'] for m in measurements]
                task_counts = [m['task_count'] for m in measurements]
                success_rates = [m['success_rate'] for m in measurements]
                
                stats[pool_name] = {
                    'executions': len(measurements),
                    'avg_time_ms': np.mean(times),
                    'avg_task_count': np.mean(task_counts),
                    'avg_success_rate': np.mean(success_rates),
                    'total_tasks': sum(task_counts)
                }
        
        return stats


class CPUAffinityManager:
    """
    CPU affinity management for critical processes.
    Optimizes CPU core assignment for better performance.
    """
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.process = psutil.Process()
        self.original_affinity = self.process.cpu_affinity()
        self.affinity_assignments = {}
        
        logger.info("CPUAffinityManager initialized",
                   cpu_count=self.cpu_count,
                   original_affinity=self.original_affinity)
    
    def set_process_affinity(self, cpu_cores: List[int], 
                           process_name: str = "current"):
        """Set CPU affinity for process"""
        
        # Validate CPU cores
        valid_cores = [core for core in cpu_cores if 0 <= core < self.cpu_count]
        
        if not valid_cores:
            logger.warning("No valid CPU cores specified")
            return False
        
        try:
            self.process.cpu_affinity(valid_cores)
            self.affinity_assignments[process_name] = valid_cores
            
            logger.info("CPU affinity set",
                       process=process_name,
                       cores=valid_cores)
            
            return True
            
        except Exception as e:
            logger.error("Failed to set CPU affinity",
                        process=process_name,
                        error=str(e))
            return False
    
    def set_high_performance_affinity(self):
        """Set affinity for high-performance computing"""
        
        # Use first half of CPU cores for main process
        cores = list(range(self.cpu_count // 2))
        return self.set_process_affinity(cores, "high_performance")
    
    def set_inference_affinity(self):
        """Set affinity optimized for inference tasks"""
        
        # Use even-numbered cores for inference
        cores = list(range(0, self.cpu_count, 2))
        return self.set_process_affinity(cores, "inference")
    
    def restore_original_affinity(self):
        """Restore original CPU affinity"""
        
        try:
            self.process.cpu_affinity(self.original_affinity)
            logger.info("Original CPU affinity restored",
                       cores=self.original_affinity)
            return True
            
        except Exception as e:
            logger.error("Failed to restore CPU affinity", error=str(e))
            return False
    
    def get_current_affinity(self) -> List[int]:
        """Get current CPU affinity"""
        return self.process.cpu_affinity()
    
    def get_affinity_stats(self) -> Dict:
        """Get CPU affinity statistics"""
        return {
            'cpu_count': self.cpu_count,
            'original_affinity': self.original_affinity,
            'current_affinity': self.get_current_affinity(),
            'assignments': self.affinity_assignments.copy()
        }


class CPUOptimizer:
    """
    Main CPU optimization coordinator.
    Integrates all CPU optimization components.
    """
    
    def __init__(self):
        self.jit_compiler = JITCompiler()
        self.vectorized_ops = VectorizedOps()
        self.thread_pool_optimizer = ThreadPoolOptimizer()
        self.affinity_manager = CPUAffinityManager()
        
        self.optimization_enabled = False
        self.performance_stats = deque(maxlen=1000)
        
        logger.info("CPUOptimizer initialized")
    
    def enable_optimizations(self):
        """Enable all CPU optimizations"""
        if self.optimization_enabled:
            return
        
        self.optimization_enabled = True
        
        # Set high-performance CPU affinity
        self.affinity_manager.set_high_performance_affinity()
        
        # Create default thread pools
        self.thread_pool_optimizer.create_thread_pool("default")
        self.thread_pool_optimizer.create_thread_pool("inference")
        self.thread_pool_optimizer.create_thread_pool("computation")
        
        logger.info("CPU optimizations enabled")
    
    def disable_optimizations(self):
        """Disable all CPU optimizations"""
        if not self.optimization_enabled:
            return
        
        self.optimization_enabled = False
        
        # Restore original CPU affinity
        self.affinity_manager.restore_original_affinity()
        
        # Shutdown thread pools
        self.thread_pool_optimizer.shutdown_all_pools()
        
        logger.info("CPU optimizations disabled")
    
    def optimize_model_inference(self, model: torch.nn.Module,
                                example_inputs: torch.Tensor,
                                model_name: str = None) -> torch.jit.ScriptModule:
        """Optimize PyTorch model for inference"""
        
        # Compile model with JIT
        compiled_model = self.jit_compiler.compile_model(
            model, example_inputs, model_name
        )
        
        # Set inference affinity
        self.affinity_manager.set_inference_affinity()
        
        return compiled_model
    
    def benchmark_cpu_operations(self) -> Dict:
        """Benchmark CPU-optimized operations"""
        
        results = {}
        
        # Benchmark vectorized operations
        test_data_1 = np.random.randn(10000).astype(np.float64)
        test_data_2 = np.random.randn(10000).astype(np.float64)
        
        # Correlation benchmark
        results['correlation'] = self.vectorized_ops.benchmark_operation(
            'fast_correlation',
            self.vectorized_ops.fast_correlation,
            test_data_1, test_data_2
        )
        
        # Moving average benchmark
        results['moving_average'] = self.vectorized_ops.benchmark_operation(
            'fast_moving_average',
            self.vectorized_ops.fast_moving_average,
            test_data_1, 50
        )
        
        # Volatility benchmark
        results['volatility'] = self.vectorized_ops.benchmark_operation(
            'fast_volatility',
            self.vectorized_ops.fast_volatility,
            test_data_1, 30
        )
        
        # Matrix multiplication benchmark
        matrix_a = np.random.randn(500, 500).astype(np.float64)
        matrix_b = np.random.randn(500, 500).astype(np.float64)
        
        results['matrix_multiply'] = self.vectorized_ops.benchmark_operation(
            'fast_matrix_multiply',
            self.vectorized_ops.fast_matrix_multiply,
            matrix_a, matrix_b
        )
        
        return results
    
    def get_cpu_stats(self) -> CPUStats:
        """Get current CPU statistics"""
        
        cpu_stats = psutil.cpu_stats()
        cpu_freq = psutil.cpu_freq()
        load_avg = os.getloadavg()
        
        return CPUStats(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            cpu_count=psutil.cpu_count(),
            cpu_freq=cpu_freq.current if cpu_freq else 0.0,
            load_avg=load_avg,
            context_switches=cpu_stats.ctx_switches,
            interrupts=cpu_stats.interrupts,
            thread_count=threading.active_count()
        )
    
    def get_optimization_stats(self) -> Dict:
        """Get comprehensive optimization statistics"""
        
        return {
            'optimization_enabled': self.optimization_enabled,
            'cpu_stats': self.get_cpu_stats().__dict__,
            'jit_compiler': self.jit_compiler.get_compilation_stats(),
            'vectorized_ops': self.vectorized_ops.get_performance_stats(),
            'thread_pools': self.thread_pool_optimizer.get_execution_stats(),
            'cpu_affinity': self.affinity_manager.get_affinity_stats()
        }
    
    def generate_recommendations(self) -> List[Dict]:
        """Generate CPU optimization recommendations"""
        
        recommendations = []
        
        # Check CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1.0)
        if cpu_percent > 80:
            recommendations.append({
                'type': 'CPU_UTILIZATION',
                'severity': 'HIGH',
                'message': f'High CPU utilization ({cpu_percent:.1f}%). Consider reducing workload or optimizing algorithms.',
                'current_value': cpu_percent,
                'target_value': 70.0
            })
        
        # Check thread pool efficiency
        thread_stats = self.thread_pool_optimizer.get_execution_stats()
        for pool_name, stats in thread_stats.items():
            if stats['avg_success_rate'] < 0.95:
                recommendations.append({
                    'type': 'THREAD_POOL_EFFICIENCY',
                    'severity': 'MEDIUM',
                    'message': f'Thread pool "{pool_name}" has low success rate ({stats["avg_success_rate"]:.1%}). Check for task failures.',
                    'current_value': stats['avg_success_rate'],
                    'target_value': 0.95
                })
        
        # Check JIT compilation
        jit_stats = self.jit_compiler.get_compilation_stats()
        if jit_stats['total_models'] == 0:
            recommendations.append({
                'type': 'JIT_COMPILATION',
                'severity': 'MEDIUM',
                'message': 'No models have been JIT compiled. Consider compiling frequently used models for better performance.',
                'current_value': 0,
                'target_value': 1
            })
        
        return recommendations
    
    @contextmanager
    def optimized_cpu_context(self):
        """Context manager for CPU-optimized operations"""
        self.enable_optimizations()
        try:
            yield self
        finally:
            pass  # Keep optimizations enabled


# Global CPU optimizer instance
cpu_optimizer = CPUOptimizer()


def optimize_cpu():
    """Decorator to enable CPU optimization for a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cpu_optimizer.enable_optimizations()
            return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    """Demo CPU optimization system"""
    
    print("⚡ CPU Optimization System Demo")
    print("=" * 40)
    
    # Enable optimizations
    cpu_optimizer.enable_optimizations()
    
    # Benchmark operations
    print("\n📊 Benchmarking CPU operations...")
    
    benchmark_results = cpu_optimizer.benchmark_cpu_operations()
    
    print("\n🏆 Benchmark Results:")
    for operation, result in benchmark_results.items():
        print(f"  {operation}: {result['execution_time_ms']:.2f}ms")
    
    # Test thread pool
    print("\n🧵 Testing thread pool optimization...")
    
    def test_task(x):
        return x * x + np.sin(x)
    
    args_list = [(i,) for i in range(100)]
    
    results = cpu_optimizer.thread_pool_optimizer.execute_parallel(
        "test_pool", test_task, args_list
    )
    
    print(f"Processed {len(results)} tasks in parallel")
    
    # Get statistics
    stats = cpu_optimizer.get_optimization_stats()
    
    print("\n📈 CPU Optimization Statistics:")
    print(f"CPU usage: {stats['cpu_stats']['cpu_percent']:.1f}%")
    print(f"Active threads: {stats['cpu_stats']['thread_count']}")
    print(f"JIT compiled models: {stats['jit_compiler']['total_models']}")
    
    # Get recommendations
    recommendations = cpu_optimizer.generate_recommendations()
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec['message']}")
    
    print("\n✅ CPU optimization demo completed!")