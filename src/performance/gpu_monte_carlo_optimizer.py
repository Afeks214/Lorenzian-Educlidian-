#!/usr/bin/env python3
"""
GPU Monte Carlo Optimizer for 1000 Samples with <500μs Latency
Performance Optimization Agent (Agent 6) - Institutional-Grade GPU Optimization

Key Features:
- GPU-accelerated Monte Carlo simulation
- <500μs execution latency for 1000 samples
- Memory optimization for <8GB usage
- Batch processing with memory pooling
- Real-time performance monitoring
- Automatic fallback to CPU optimization
"""

import torch
import torch.nn as nn
import numpy as np
import time
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
from numba import cuda, jit, float32, float64
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPUPerformanceMetrics:
    """GPU performance metrics for monitoring"""
    timestamp: float
    execution_time_us: float
    memory_usage_mb: float
    gpu_utilization: float
    samples_processed: int
    throughput_samples_per_second: float
    latency_target_met: bool
    memory_target_met: bool

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    num_samples: int = 1000
    time_horizon_days: int = 1
    num_assets: int = 3
    batch_size: int = 256
    use_gpu: bool = True
    precision: torch.dtype = torch.float32
    memory_pool_size: int = 1024
    target_latency_us: float = 500.0
    max_memory_gb: float = 8.0

class GPUMemoryManager:
    """Optimized GPU memory management for Monte Carlo simulations"""
    
    def __init__(self, config: MonteCarloConfig):
        self.config = config
        self.device = self._setup_device()
        self.memory_pool = []
        self.allocation_cache = {}
        self.memory_usage_history = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # Initialize memory pool
        self._initialize_memory_pool()
        
        logger.info(f"GPU Memory Manager initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup optimal GPU device"""
        if torch.cuda.is_available() and self.config.use_gpu:
            device = torch.device('cuda')
            
            # Optimize CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable memory pool if available
            if hasattr(torch.cuda, 'memory_pool'):
                torch.cuda.memory_pool.set_per_process_memory_fraction(0.8)
            
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            return device
        else:
            logger.warning("GPU not available, falling back to CPU")
            return torch.device('cpu')
    
    def _initialize_memory_pool(self):
        """Initialize memory pool for tensor reuse"""
        if self.device.type == 'cuda':
            # Pre-allocate tensor pool
            pool_shapes = [
                (self.config.batch_size, self.config.num_assets),  # Price vectors
                (self.config.batch_size, self.config.time_horizon_days, self.config.num_assets),  # Paths
                (self.config.batch_size,),  # Scalar results
                (self.config.num_samples, self.config.num_assets),  # Full sample batch
            ]
            
            for shape in pool_shapes:
                for _ in range(self.config.memory_pool_size // len(pool_shapes)):
                    tensor = torch.empty(shape, dtype=self.config.precision, device=self.device)
                    self.memory_pool.append(tensor)
            
            logger.info(f"Memory pool initialized with {len(self.memory_pool)} tensors")
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = None) -> torch.Tensor:
        """Get tensor from memory pool or allocate new one"""
        if dtype is None:
            dtype = self.config.precision
        
        with self.lock:
            # Try to find matching tensor in pool
            for i, tensor in enumerate(self.memory_pool):
                if tensor.shape == shape and tensor.dtype == dtype:
                    return self.memory_pool.pop(i)
            
            # Allocate new tensor if not found
            return torch.empty(shape, dtype=dtype, device=self.device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to memory pool"""
        with self.lock:
            if len(self.memory_pool) < self.config.memory_pool_size:
                # Clear tensor and return to pool
                tensor.zero_()
                self.memory_pool.append(tensor)
            else:
                # Pool is full, let tensor be garbage collected
                del tensor
    
    def clear_cache(self):
        """Clear GPU cache and collect garbage"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            usage = {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'total_gb': total,
                'utilization_pct': (allocated / total) * 100
            }
        else:
            memory = psutil.virtual_memory()
            usage = {
                'allocated_gb': memory.used / 1e9,
                'cached_gb': 0.0,
                'total_gb': memory.total / 1e9,
                'utilization_pct': memory.percent
            }
        
        self.memory_usage_history.append(usage)
        return usage

class GPUMonteCarloKernel:
    """Optimized CUDA kernel for Monte Carlo simulation"""
    
    def __init__(self, memory_manager: GPUMemoryManager):
        self.memory_manager = memory_manager
        self.device = memory_manager.device
        self.config = memory_manager.config
        
        # Compile CUDA kernels
        self._compile_kernels()
        
        # Performance tracking
        self.kernel_times = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=1000)
    
    def _compile_kernels(self):
        """Compile optimized CUDA kernels"""
        if self.device.type == 'cuda':
            # Compile custom CUDA kernel for Monte Carlo
            self.mc_kernel = self._create_monte_carlo_kernel()
            logger.info("CUDA kernels compiled successfully")
    
    def _create_monte_carlo_kernel(self):
        """Create optimized Monte Carlo CUDA kernel"""
        # Use CuPy for custom kernel compilation
        monte_carlo_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void monte_carlo_simulation(
            float* paths,
            float* randoms,
            float* initial_prices,
            float* drift_rates,
            float* volatilities,
            int num_samples,
            int num_assets,
            int time_steps,
            float dt
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = gridDim.x * blockDim.x;
            
            for (int sample = idx; sample < num_samples; sample += stride) {
                for (int asset = 0; asset < num_assets; asset++) {
                    float price = initial_prices[asset];
                    paths[sample * num_assets + asset] = price;
                    
                    for (int t = 1; t < time_steps; t++) {
                        int rand_idx = sample * num_assets * time_steps + asset * time_steps + t;
                        float dW = randoms[rand_idx];
                        
                        // Geometric Brownian Motion
                        float drift = drift_rates[asset] * price * dt;
                        float diffusion = volatilities[asset] * price * dW * sqrtf(dt);
                        
                        price += drift + diffusion;
                        if (price <= 0) price = 0.01f;  // Ensure positive prices
                        
                        paths[sample * num_assets * time_steps + asset * time_steps + t] = price;
                    }
                }
            }
        }
        ''', 'monte_carlo_simulation')
        
        return monte_carlo_kernel
    
    def execute_monte_carlo(self, 
                          initial_prices: torch.Tensor,
                          drift_rates: torch.Tensor,
                          volatilities: torch.Tensor,
                          num_samples: int = 1000) -> torch.Tensor:
        """Execute Monte Carlo simulation with GPU optimization"""
        
        start_time = time.perf_counter()
        
        # Batch processing for memory efficiency
        batch_size = self.config.batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        results = []
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            # Generate random numbers
            randoms = torch.randn(
                current_batch_size, 
                self.config.time_horizon_days, 
                self.config.num_assets,
                device=self.device, 
                dtype=self.config.precision
            )
            
            # Execute simulation for this batch
            batch_result = self._execute_batch(
                initial_prices, drift_rates, volatilities, 
                randoms, current_batch_size
            )
            
            results.append(batch_result)
            
            # Memory management
            self.memory_manager.return_tensor(randoms)
        
        # Concatenate results
        final_result = torch.cat(results, dim=0)
        
        # Record performance
        execution_time = (time.perf_counter() - start_time) * 1e6  # Convert to microseconds
        self.kernel_times.append(execution_time)
        
        throughput = num_samples / (execution_time / 1e6)  # Samples per second
        self.throughput_history.append(throughput)
        
        logger.info(f"Monte Carlo execution: {execution_time:.2f}μs for {num_samples} samples")
        
        return final_result
    
    def _execute_batch(self, 
                      initial_prices: torch.Tensor,
                      drift_rates: torch.Tensor,
                      volatilities: torch.Tensor,
                      randoms: torch.Tensor,
                      batch_size: int) -> torch.Tensor:
        """Execute Monte Carlo simulation for a single batch"""
        
        if self.device.type == 'cuda':
            return self._execute_cuda_batch(
                initial_prices, drift_rates, volatilities, randoms, batch_size
            )
        else:
            return self._execute_cpu_batch(
                initial_prices, drift_rates, volatilities, randoms, batch_size
            )
    
    def _execute_cuda_batch(self, 
                           initial_prices: torch.Tensor,
                           drift_rates: torch.Tensor,
                           volatilities: torch.Tensor,
                           randoms: torch.Tensor,
                           batch_size: int) -> torch.Tensor:
        """Execute batch using CUDA kernel"""
        
        # Convert to CuPy arrays for kernel execution
        initial_prices_cp = cp.asarray(initial_prices)
        drift_rates_cp = cp.asarray(drift_rates)
        volatilities_cp = cp.asarray(volatilities)
        randoms_cp = cp.asarray(randoms)
        
        # Allocate output
        paths = cp.zeros((batch_size, self.config.time_horizon_days, self.config.num_assets), 
                        dtype=cp.float32)
        
        # Execute CUDA kernel
        dt = 1.0 / 365.0  # Daily time step
        
        block_size = 256
        grid_size = (batch_size + block_size - 1) // block_size
        
        self.mc_kernel(
            (grid_size,), (block_size,),
            (
                paths, randoms_cp, initial_prices_cp, drift_rates_cp, 
                volatilities_cp, batch_size, self.config.num_assets, 
                self.config.time_horizon_days, dt
            )
        )
        
        # Convert back to PyTorch
        result = torch.as_tensor(paths, device=self.device)
        
        return result
    
    def _execute_cpu_batch(self, 
                          initial_prices: torch.Tensor,
                          drift_rates: torch.Tensor,
                          volatilities: torch.Tensor,
                          randoms: torch.Tensor,
                          batch_size: int) -> torch.Tensor:
        """Execute batch using optimized CPU implementation"""
        
        # Vectorized CPU implementation
        dt = 1.0 / 365.0
        
        # Initialize paths
        paths = torch.zeros(
            batch_size, self.config.time_horizon_days, self.config.num_assets,
            device=self.device, dtype=self.config.precision
        )
        
        # Set initial prices
        paths[:, 0, :] = initial_prices
        
        # Generate paths
        for t in range(1, self.config.time_horizon_days):
            # Vectorized GBM update
            S_prev = paths[:, t-1, :]
            dW = randoms[:, t-1, :]
            
            drift = drift_rates * S_prev * dt
            diffusion = volatilities * S_prev * dW * torch.sqrt(torch.tensor(dt, device=self.device))
            
            paths[:, t, :] = S_prev + drift + diffusion
            
            # Ensure positive prices
            paths[:, t, :] = torch.clamp(paths[:, t, :], min=0.01)
        
        return paths
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get kernel performance statistics"""
        if not self.kernel_times:
            return {}
        
        kernel_times = list(self.kernel_times)
        throughput = list(self.throughput_history)
        
        return {
            'avg_execution_time_us': np.mean(kernel_times),
            'p95_execution_time_us': np.percentile(kernel_times, 95),
            'p99_execution_time_us': np.percentile(kernel_times, 99),
            'target_latency_met_pct': np.mean(np.array(kernel_times) < self.config.target_latency_us) * 100,
            'avg_throughput_samples_per_sec': np.mean(throughput),
            'max_throughput_samples_per_sec': np.max(throughput),
            'total_executions': len(kernel_times)
        }

class GPUMonteCarloOptimizer:
    """Main GPU Monte Carlo optimizer with <500μs latency target"""
    
    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()
        self.memory_manager = GPUMemoryManager(self.config)
        self.kernel = GPUMonteCarloKernel(self.memory_manager)
        
        # Performance monitoring
        self.performance_metrics = deque(maxlen=10000)
        self.performance_lock = threading.Lock()
        
        # Warmup
        self._warmup()
        
        logger.info("GPU Monte Carlo Optimizer initialized")
    
    def _warmup(self):
        """Warmup GPU kernels for optimal performance"""
        logger.info("Warming up GPU kernels...")
        
        # Small warmup run
        initial_prices = torch.tensor([100.0, 200.0, 50.0], device=self.memory_manager.device)
        drift_rates = torch.tensor([0.1, 0.08, 0.12], device=self.memory_manager.device)
        volatilities = torch.tensor([0.2, 0.25, 0.18], device=self.memory_manager.device)
        
        for _ in range(5):
            self.simulate_monte_carlo(initial_prices, drift_rates, volatilities, num_samples=100)
        
        # Clear cache after warmup
        self.memory_manager.clear_cache()
        
        logger.info("Warmup completed")
    
    def simulate_monte_carlo(self, 
                           initial_prices: torch.Tensor,
                           drift_rates: torch.Tensor,
                           volatilities: torch.Tensor,
                           num_samples: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation with performance monitoring"""
        
        start_time = time.perf_counter()
        
        # Record initial memory usage
        initial_memory = self.memory_manager.get_memory_usage()
        
        try:
            # Execute simulation
            paths = self.kernel.execute_monte_carlo(
                initial_prices, drift_rates, volatilities, num_samples
            )
            
            # Calculate final portfolio values
            final_values = self._calculate_portfolio_values(paths, initial_prices)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(final_values)
            
            # Record performance
            execution_time = (time.perf_counter() - start_time) * 1e6  # microseconds
            final_memory = self.memory_manager.get_memory_usage()
            
            metrics = GPUPerformanceMetrics(
                timestamp=time.time(),
                execution_time_us=execution_time,
                memory_usage_mb=final_memory['allocated_gb'] * 1024,
                gpu_utilization=final_memory['utilization_pct'],
                samples_processed=num_samples,
                throughput_samples_per_second=num_samples / (execution_time / 1e6),
                latency_target_met=execution_time < self.config.target_latency_us,
                memory_target_met=final_memory['allocated_gb'] < self.config.max_memory_gb
            )
            
            with self.performance_lock:
                self.performance_metrics.append(metrics)
            
            # Log performance
            logger.info(f"Monte Carlo completed: {execution_time:.2f}μs, "
                       f"Memory: {final_memory['allocated_gb']:.2f}GB, "
                       f"Target met: {metrics.latency_target_met}")
            
            return {
                'paths': paths,
                'final_values': final_values,
                'risk_metrics': risk_metrics,
                'performance_metrics': metrics,
                'execution_time_us': execution_time,
                'memory_usage_gb': final_memory['allocated_gb']
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation failed: {e}")
            raise
        finally:
            # Cleanup
            self.memory_manager.clear_cache()
    
    def _calculate_portfolio_values(self, paths: torch.Tensor, initial_prices: torch.Tensor) -> torch.Tensor:
        """Calculate portfolio values from price paths"""
        # Equal weight portfolio
        weights = torch.ones(self.config.num_assets, device=self.memory_manager.device) / self.config.num_assets
        
        # Calculate returns
        returns = (paths[:, -1, :] - initial_prices) / initial_prices
        
        # Portfolio returns
        portfolio_returns = torch.sum(returns * weights, dim=1)
        
        return portfolio_returns
    
    def _calculate_risk_metrics(self, portfolio_returns: torch.Tensor) -> Dict[str, float]:
        """Calculate risk metrics from portfolio returns"""
        returns_np = portfolio_returns.cpu().numpy()
        
        return {
            'var_95': float(np.percentile(returns_np, 5)),
            'var_99': float(np.percentile(returns_np, 1)),
            'expected_shortfall_95': float(np.mean(returns_np[returns_np <= np.percentile(returns_np, 5)])),
            'expected_shortfall_99': float(np.mean(returns_np[returns_np <= np.percentile(returns_np, 1)])),
            'mean_return': float(np.mean(returns_np)),
            'std_return': float(np.std(returns_np)),
            'skewness': float(self._calculate_skewness(returns_np)),
            'kurtosis': float(self._calculate_kurtosis(returns_np))
        }
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        mean = np.mean(returns)
        std = np.std(returns)
        return np.mean(((returns - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        mean = np.mean(returns)
        std = np.std(returns)
        return np.mean(((returns - mean) / std) ** 4) - 3
    
    def benchmark_performance(self, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark performance with multiple iterations"""
        logger.info(f"Running performance benchmark with {iterations} iterations")
        
        # Test parameters
        initial_prices = torch.tensor([100.0, 200.0, 50.0], device=self.memory_manager.device)
        drift_rates = torch.tensor([0.1, 0.08, 0.12], device=self.memory_manager.device)
        volatilities = torch.tensor([0.2, 0.25, 0.18], device=self.memory_manager.device)
        
        execution_times = []
        memory_usage = []
        
        for i in range(iterations):
            result = self.simulate_monte_carlo(
                initial_prices, drift_rates, volatilities, 
                num_samples=self.config.num_samples
            )
            
            execution_times.append(result['execution_time_us'])
            memory_usage.append(result['memory_usage_gb'])
            
            if i % 10 == 0:
                logger.info(f"Benchmark progress: {i}/{iterations}")
        
        # Calculate statistics
        benchmark_stats = {
            'iterations': iterations,
            'avg_execution_time_us': np.mean(execution_times),
            'p95_execution_time_us': np.percentile(execution_times, 95),
            'p99_execution_time_us': np.percentile(execution_times, 99),
            'max_execution_time_us': np.max(execution_times),
            'min_execution_time_us': np.min(execution_times),
            'std_execution_time_us': np.std(execution_times),
            'target_latency_met_pct': np.mean(np.array(execution_times) < self.config.target_latency_us) * 100,
            'avg_memory_usage_gb': np.mean(memory_usage),
            'max_memory_usage_gb': np.max(memory_usage),
            'memory_target_met_pct': np.mean(np.array(memory_usage) < self.config.max_memory_gb) * 100,
            'avg_throughput_samples_per_sec': self.config.num_samples / (np.mean(execution_times) / 1e6),
            'device_type': str(self.memory_manager.device)
        }
        
        logger.info(f"Benchmark completed: {benchmark_stats['avg_execution_time_us']:.2f}μs average, "
                   f"{benchmark_stats['target_latency_met_pct']:.1f}% target met")
        
        return benchmark_stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.performance_metrics:
            return {}
        
        with self.performance_lock:
            metrics = list(self.performance_metrics)
        
        execution_times = [m.execution_time_us for m in metrics]
        memory_usage = [m.memory_usage_mb for m in metrics]
        throughput = [m.throughput_samples_per_second for m in metrics]
        
        return {
            'total_simulations': len(metrics),
            'avg_execution_time_us': np.mean(execution_times),
            'p95_execution_time_us': np.percentile(execution_times, 95),
            'p99_execution_time_us': np.percentile(execution_times, 99),
            'target_latency_met_pct': np.mean([m.latency_target_met for m in metrics]) * 100,
            'avg_memory_usage_mb': np.mean(memory_usage),
            'max_memory_usage_mb': np.max(memory_usage),
            'memory_target_met_pct': np.mean([m.memory_target_met for m in metrics]) * 100,
            'avg_throughput_samples_per_sec': np.mean(throughput),
            'max_throughput_samples_per_sec': np.max(throughput),
            'kernel_stats': self.kernel.get_performance_stats(),
            'memory_stats': self.memory_manager.get_memory_usage(),
            'device_type': str(self.memory_manager.device)
        }
    
    def create_performance_report(self, output_path: Path = None) -> Dict[str, Any]:
        """Create comprehensive performance report"""
        if output_path is None:
            output_path = Path("gpu_monte_carlo_performance_report.json")
        
        report = {
            'timestamp': time.time(),
            'configuration': {
                'num_samples': self.config.num_samples,
                'target_latency_us': self.config.target_latency_us,
                'max_memory_gb': self.config.max_memory_gb,
                'batch_size': self.config.batch_size,
                'precision': str(self.config.precision),
                'device_type': str(self.memory_manager.device)
            },
            'performance_summary': self.get_performance_summary(),
            'benchmark_results': self.benchmark_performance(iterations=50),
            'system_info': {
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
                'cpu_count': psutil.cpu_count(),
                'system_memory_gb': psutil.virtual_memory().total / 1e9
            },
            'optimization_targets': {
                'latency_target_us': self.config.target_latency_us,
                'memory_target_gb': self.config.max_memory_gb,
                'throughput_target_samples_per_sec': 1000 / (self.config.target_latency_us / 1e6)
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {output_path}")
        return report

def main():
    """Main GPU Monte Carlo optimization demo"""
    print("🚀 GPU MONTE CARLO OPTIMIZER - AGENT 6 PERFORMANCE SYSTEM")
    print("=" * 80)
    
    # Configuration for 1000 samples with <500μs latency
    config = MonteCarloConfig(
        num_samples=1000,
        target_latency_us=500.0,
        max_memory_gb=8.0,
        batch_size=256,
        use_gpu=True,
        precision=torch.float32
    )
    
    # Initialize optimizer
    optimizer = GPUMonteCarloOptimizer(config)
    
    # Test parameters
    initial_prices = torch.tensor([100.0, 200.0, 50.0], device=optimizer.memory_manager.device)
    drift_rates = torch.tensor([0.1, 0.08, 0.12], device=optimizer.memory_manager.device)
    volatilities = torch.tensor([0.2, 0.25, 0.18], device=optimizer.memory_manager.device)
    
    # Run single simulation
    print("\n🎯 Running Monte Carlo simulation...")
    result = optimizer.simulate_monte_carlo(
        initial_prices, drift_rates, volatilities, num_samples=1000
    )
    
    print(f"✅ Simulation completed in {result['execution_time_us']:.2f}μs")
    print(f"📊 Memory usage: {result['memory_usage_gb']:.2f}GB")
    print(f"🎯 Target met: {result['performance_metrics'].latency_target_met}")
    
    # Run benchmark
    print("\n📈 Running performance benchmark...")
    benchmark_results = optimizer.benchmark_performance(iterations=100)
    
    print(f"📊 Benchmark Results:")
    print(f"  Average execution time: {benchmark_results['avg_execution_time_us']:.2f}μs")
    print(f"  P99 execution time: {benchmark_results['p99_execution_time_us']:.2f}μs")
    print(f"  Target latency met: {benchmark_results['target_latency_met_pct']:.1f}%")
    print(f"  Average memory usage: {benchmark_results['avg_memory_usage_gb']:.2f}GB")
    print(f"  Memory target met: {benchmark_results['memory_target_met_pct']:.1f}%")
    print(f"  Throughput: {benchmark_results['avg_throughput_samples_per_sec']:.0f} samples/sec")
    
    # Generate comprehensive report
    print("\n📄 Generating performance report...")
    report = optimizer.create_performance_report()
    
    print(f"✅ Performance report generated")
    print(f"📁 Report saved to: gpu_monte_carlo_performance_report.json")
    
    # Summary
    print("\n" + "=" * 80)
    print("🏆 GPU MONTE CARLO OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    target_met = benchmark_results['target_latency_met_pct'] >= 95
    memory_efficient = benchmark_results['memory_target_met_pct'] >= 95
    
    print(f"🎯 Latency Target (<500μs): {'✅ ACHIEVED' if target_met else '❌ MISSED'}")
    print(f"💾 Memory Target (<8GB): {'✅ ACHIEVED' if memory_efficient else '❌ MISSED'}")
    print(f"🚀 Throughput: {benchmark_results['avg_throughput_samples_per_sec']:.0f} samples/sec")
    print(f"⚡ Device: {benchmark_results['device_type']}")
    
    if target_met and memory_efficient:
        print("\n🎉 INSTITUTIONAL-GRADE PERFORMANCE ACHIEVED!")
        print("🔥 System ready for high-frequency trading deployment")
    else:
        print("\n⚠️  Performance optimization needed")
        print("🔧 Consider GPU upgrade or batch size optimization")

if __name__ == "__main__":
    main()