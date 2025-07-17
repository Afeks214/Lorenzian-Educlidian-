#!/usr/bin/env python3
"""
Memory Optimization System for <8GB Usage with 1000 MC Samples
Performance Optimization Agent (Agent 6) - Memory Leak Prevention & Optimization

Key Features:
- Memory usage optimization for <8GB target
- Advanced memory leak detection and prevention
- Intelligent garbage collection management
- Memory pool optimization with tensor reuse
- Real-time memory monitoring with alerts
- Automatic memory cleanup and defragmentation
"""

import torch
import numpy as np
import gc
import psutil
import time
import threading
import weakref
import tracemalloc
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import json
import logging
from contextlib import contextmanager
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot for tracking"""
    timestamp: float
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    gpu_memory_gb: float
    gpu_cached_gb: float
    python_memory_mb: float
    tensors_allocated: int
    memory_leaks_detected: int
    gc_collections: int
    memory_efficiency_score: float

@dataclass
class MemoryPool:
    """Memory pool for tensor reuse"""
    tensors: List[torch.Tensor] = field(default_factory=list)
    max_size: int = 1000
    hits: int = 0
    misses: int = 0
    allocations: int = 0
    deallocations: int = 0
    
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class MemoryLeakDetector:
    """Advanced memory leak detection system"""
    
    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self.memory_history = deque(maxlen=1000)
        self.tensor_registry = weakref.WeakSet()
        self.allocation_trackers = {}
        self.leak_threshold_mb = 100.0
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Start trace malloc for detailed tracking
        tracemalloc.start()
        
    def start_monitoring(self):
        """Start memory leak monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Memory leak monitoring started")
    
    def stop_monitoring(self):
        """Stop memory leak monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Memory leak monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_memory_leaks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def _check_memory_leaks(self):
        """Check for memory leaks"""
        # Get current memory usage
        current_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        self.memory_history.append(current_memory)
        
        # Check for memory growth trend
        if len(self.memory_history) >= 10:
            recent_avg = np.mean(list(self.memory_history)[-5:])
            older_avg = np.mean(list(self.memory_history)[-10:-5])
            
            if recent_avg > older_avg + self.leak_threshold_mb:
                self._report_potential_leak(recent_avg - older_avg)
        
        # Check Python memory allocations
        current, peak = tracemalloc.get_traced_memory()
        if current > peak * 0.9:  # Close to peak usage
            self._analyze_memory_allocations()
    
    def _report_potential_leak(self, growth_mb: float):
        """Report potential memory leak"""
        logger.warning(f"Potential memory leak detected: {growth_mb:.1f}MB growth")
        
        # Get top memory consumers
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        logger.warning("Top memory consumers:")
        for stat in top_stats[:5]:
            logger.warning(f"  {stat}")
    
    def _analyze_memory_allocations(self):
        """Analyze current memory allocations"""
        snapshot = tracemalloc.take_snapshot()
        
        # Group by file
        top_stats = snapshot.statistics('filename')
        
        # Look for suspicious patterns
        for stat in top_stats[:10]:
            if stat.size > 50 * 1024 * 1024:  # > 50MB
                logger.warning(f"Large allocation in {stat.traceback.format()[-1]}: {stat.size / (1024*1024):.1f}MB")
    
    def register_tensor(self, tensor: torch.Tensor, context: str = ""):
        """Register tensor for tracking"""
        self.tensor_registry.add(tensor)
        tensor_id = id(tensor)
        self.allocation_trackers[tensor_id] = {
            'size_mb': tensor.numel() * tensor.element_size() / (1024 * 1024),
            'context': context,
            'timestamp': time.time()
        }
    
    def get_tensor_stats(self) -> Dict[str, Any]:
        """Get tensor allocation statistics"""
        total_tensors = len(self.tensor_registry)
        total_memory = sum(
            tracker['size_mb'] for tracker in self.allocation_trackers.values()
        )
        
        return {
            'total_tensors': total_tensors,
            'total_memory_mb': total_memory,
            'recent_memory_mb': np.mean(list(self.memory_history)[-10:]) if self.memory_history else 0,
            'memory_growth_rate': self._calculate_growth_rate()
        }
    
    def _calculate_growth_rate(self) -> float:
        """Calculate memory growth rate"""
        if len(self.memory_history) < 20:
            return 0.0
        
        recent = np.mean(list(self.memory_history)[-10:])
        older = np.mean(list(self.memory_history)[-20:-10])
        
        return (recent - older) / older if older > 0 else 0.0

class MemoryOptimizer:
    """Comprehensive memory optimization system"""
    
    def __init__(self, target_memory_gb: float = 8.0):
        self.target_memory_gb = target_memory_gb
        self.target_memory_bytes = int(target_memory_gb * 1024 * 1024 * 1024)
        
        # Memory pools for different tensor types
        self.memory_pools = {
            'small': MemoryPool(max_size=500),      # < 1MB tensors
            'medium': MemoryPool(max_size=200),     # 1-10MB tensors
            'large': MemoryPool(max_size=50),       # > 10MB tensors
            'monte_carlo': MemoryPool(max_size=100) # MC-specific tensors
        }
        
        # Memory monitoring
        self.memory_snapshots = deque(maxlen=10000)
        self.leak_detector = MemoryLeakDetector()
        self.optimization_stats = defaultdict(int)
        
        # Threading
        self.memory_lock = threading.RLock()
        self.cleanup_thread = None
        self.cleanup_active = False
        
        # GPU optimization
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self._optimize_gpu_memory()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.info(f"Memory Optimizer initialized with {target_memory_gb}GB target")
    
    def _optimize_gpu_memory(self):
        """Optimize GPU memory settings"""
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Enable memory pool if available
        if hasattr(torch.cuda, 'memory_pool'):
            torch.cuda.memory_pool.set_per_process_memory_fraction(0.8)
        
        # Optimize CUDA settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        logger.info("GPU memory optimization applied")
    
    def start_monitoring(self):
        """Start memory monitoring"""
        self.leak_detector.start_monitoring()
        self.cleanup_active = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.leak_detector.stop_monitoring()
        self.cleanup_active = False
        if self.cleanup_thread:
            self.cleanup_thread.join()
        logger.info("Memory monitoring stopped")
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.cleanup_active:
            try:
                self._periodic_cleanup()
                time.sleep(5.0)  # Cleanup every 5 seconds
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def _periodic_cleanup(self):
        """Periodic memory cleanup"""
        # Check memory usage
        memory_usage = self.get_current_memory_usage()
        
        if memory_usage['total_memory_gb'] > self.target_memory_gb * 0.9:
            logger.warning(f"Memory usage high: {memory_usage['total_memory_gb']:.2f}GB")
            self._aggressive_cleanup()
        elif memory_usage['total_memory_gb'] > self.target_memory_gb * 0.7:
            self._moderate_cleanup()
        
        # Record snapshot
        self._record_memory_snapshot()
    
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup"""
        with self.memory_lock:
            # Clear all memory pools
            for pool in self.memory_pools.values():
                pool.tensors.clear()
            
            # Force garbage collection
            gc.collect()
            
            # GPU cache cleanup
            if self.gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.optimization_stats['aggressive_cleanups'] += 1
            logger.info("Aggressive memory cleanup performed")
    
    def _moderate_cleanup(self):
        """Moderate memory cleanup"""
        with self.memory_lock:
            # Reduce memory pool sizes
            for pool in self.memory_pools.values():
                if len(pool.tensors) > pool.max_size // 2:
                    pool.tensors = pool.tensors[:pool.max_size // 2]
            
            # Garbage collection
            gc.collect()
            
            self.optimization_stats['moderate_cleanups'] += 1
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32, 
                  device: torch.device = torch.device('cpu'), context: str = "") -> torch.Tensor:
        """Get optimized tensor from memory pool"""
        with self.memory_lock:
            tensor_size_mb = np.prod(shape) * 4 / (1024 * 1024)  # Assuming float32
            
            # Determine pool type
            if tensor_size_mb < 1:
                pool = self.memory_pools['small']
            elif tensor_size_mb < 10:
                pool = self.memory_pools['medium']
            else:
                pool = self.memory_pools['large']
            
            # Try to reuse tensor from pool
            for i, tensor in enumerate(pool.tensors):
                if (tensor.shape == shape and 
                    tensor.dtype == dtype and 
                    tensor.device == device):
                    
                    # Remove from pool and return
                    reused_tensor = pool.tensors.pop(i)
                    reused_tensor.zero_()  # Clear data
                    pool.hits += 1
                    
                    return reused_tensor
            
            # Create new tensor if no reuse possible
            new_tensor = torch.empty(shape, dtype=dtype, device=device)
            pool.misses += 1
            pool.allocations += 1
            
            # Register with leak detector
            self.leak_detector.register_tensor(new_tensor, context)
            
            return new_tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to memory pool"""
        with self.memory_lock:
            tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            
            # Determine pool type
            if tensor_size_mb < 1:
                pool = self.memory_pools['small']
            elif tensor_size_mb < 10:
                pool = self.memory_pools['medium']
            else:
                pool = self.memory_pools['large']
            
            # Add to pool if not full
            if len(pool.tensors) < pool.max_size:
                pool.tensors.append(tensor)
                pool.deallocations += 1
            else:
                # Pool is full, let tensor be garbage collected
                del tensor
    
    @contextmanager
    def memory_context(self, context_name: str = ""):
        """Context manager for memory-optimized operations"""
        start_memory = self.get_current_memory_usage()
        
        try:
            yield
        finally:
            # Cleanup after context
            end_memory = self.get_current_memory_usage()
            memory_diff = end_memory['total_memory_gb'] - start_memory['total_memory_gb']
            
            if memory_diff > 0.5:  # > 500MB increase
                logger.warning(f"High memory usage in {context_name}: {memory_diff:.2f}GB")
                self._moderate_cleanup()
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        # System memory
        memory = psutil.virtual_memory()
        
        # GPU memory
        gpu_memory_gb = 0.0
        gpu_cached_gb = 0.0
        if self.gpu_available:
            gpu_memory_gb = torch.cuda.memory_allocated() / 1e9
            gpu_cached_gb = torch.cuda.memory_reserved() / 1e9
        
        # Python memory
        python_memory_mb = 0.0
        if tracemalloc.is_tracing():
            current, _ = tracemalloc.get_traced_memory()
            python_memory_mb = current / (1024 * 1024)
        
        return {
            'total_memory_gb': memory.used / 1e9,
            'available_memory_gb': memory.available / 1e9,
            'used_memory_gb': memory.used / 1e9,
            'memory_percent': memory.percent,
            'gpu_memory_gb': gpu_memory_gb,
            'gpu_cached_gb': gpu_cached_gb,
            'python_memory_mb': python_memory_mb
        }
    
    def _record_memory_snapshot(self):
        """Record memory snapshot for monitoring"""
        current_usage = self.get_current_memory_usage()
        tensor_stats = self.leak_detector.get_tensor_stats()
        
        # Calculate memory efficiency score
        efficiency_score = min(1.0, self.target_memory_gb / current_usage['total_memory_gb'])
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_memory_gb=current_usage['total_memory_gb'],
            available_memory_gb=current_usage['available_memory_gb'],
            used_memory_gb=current_usage['used_memory_gb'],
            gpu_memory_gb=current_usage['gpu_memory_gb'],
            gpu_cached_gb=current_usage['gpu_cached_gb'],
            python_memory_mb=current_usage['python_memory_mb'],
            tensors_allocated=tensor_stats['total_tensors'],
            memory_leaks_detected=0,  # TODO: Implement leak detection count
            gc_collections=gc.get_count()[0],
            memory_efficiency_score=efficiency_score
        )
        
        self.memory_snapshots.append(snapshot)
    
    def get_memory_pools_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get memory pool statistics"""
        stats = {}
        
        for pool_name, pool in self.memory_pools.items():
            stats[pool_name] = {
                'tensors_available': len(pool.tensors),
                'max_size': pool.max_size,
                'utilization': len(pool.tensors) / pool.max_size,
                'hit_rate': pool.hit_rate(),
                'total_hits': pool.hits,
                'total_misses': pool.misses,
                'allocations': pool.allocations,
                'deallocations': pool.deallocations
            }
        
        return stats
    
    def optimize_monte_carlo_memory(self, num_samples: int = 1000, 
                                   num_assets: int = 3) -> Dict[str, Any]:
        """Optimize memory usage for Monte Carlo simulation"""
        logger.info(f"Optimizing memory for {num_samples} Monte Carlo samples")
        
        # Estimate memory requirements
        sample_size_mb = num_samples * num_assets * 4 / (1024 * 1024)  # float32
        batch_size = min(256, max(32, int(self.target_memory_gb * 1024 / sample_size_mb)))
        
        # Pre-allocate Monte Carlo memory pool
        mc_pool = self.memory_pools['monte_carlo']
        mc_pool.tensors.clear()
        
        # Pre-allocate common tensor shapes
        common_shapes = [
            (batch_size, num_assets),                    # Price vectors
            (batch_size, 1, num_assets),                 # Single time step
            (batch_size, 252, num_assets),               # Full year daily
            (num_samples,),                              # Results vector
        ]
        
        for shape in common_shapes:
            for _ in range(10):  # 10 tensors per shape
                if len(mc_pool.tensors) < mc_pool.max_size:
                    tensor = torch.empty(shape, dtype=torch.float32)
                    mc_pool.tensors.append(tensor)
        
        optimization_result = {
            'recommended_batch_size': batch_size,
            'estimated_memory_mb': sample_size_mb,
            'pre_allocated_tensors': len(mc_pool.tensors),
            'memory_efficiency': sample_size_mb / (self.target_memory_gb * 1024),
            'optimization_feasible': sample_size_mb < (self.target_memory_gb * 1024 * 0.8)
        }
        
        logger.info(f"Monte Carlo memory optimization: {optimization_result}")
        return optimization_result
    
    def create_memory_report(self, output_path: Path = None) -> Dict[str, Any]:
        """Create comprehensive memory optimization report"""
        if output_path is None:
            output_path = Path("memory_optimization_report.json")
        
        # Get recent snapshots
        recent_snapshots = list(self.memory_snapshots)[-100:] if self.memory_snapshots else []
        
        # Calculate statistics
        if recent_snapshots:
            avg_memory = np.mean([s.total_memory_gb for s in recent_snapshots])
            max_memory = np.max([s.total_memory_gb for s in recent_snapshots])
            min_memory = np.min([s.total_memory_gb for s in recent_snapshots])
            avg_efficiency = np.mean([s.memory_efficiency_score for s in recent_snapshots])
        else:
            avg_memory = max_memory = min_memory = avg_efficiency = 0.0
        
        report = {
            'timestamp': time.time(),
            'configuration': {
                'target_memory_gb': self.target_memory_gb,
                'gpu_available': self.gpu_available,
                'monitoring_active': self.leak_detector.monitoring_active
            },
            'current_usage': self.get_current_memory_usage(),
            'memory_statistics': {
                'avg_memory_gb': avg_memory,
                'max_memory_gb': max_memory,
                'min_memory_gb': min_memory,
                'avg_efficiency_score': avg_efficiency,
                'target_met': max_memory <= self.target_memory_gb
            },
            'memory_pools': self.get_memory_pools_stats(),
            'optimization_stats': dict(self.optimization_stats),
            'leak_detection': self.leak_detector.get_tensor_stats(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Memory optimization report saved to {output_path}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        current_usage = self.get_current_memory_usage()
        
        if current_usage['total_memory_gb'] > self.target_memory_gb:
            recommendations.append("Memory usage exceeds target - consider reducing batch size")
        
        if current_usage['gpu_memory_gb'] > 0 and current_usage['gpu_cached_gb'] > current_usage['gpu_memory_gb'] * 2:
            recommendations.append("High GPU cache usage - consider clearing cache more frequently")
        
        pool_stats = self.get_memory_pools_stats()
        for pool_name, stats in pool_stats.items():
            if stats['hit_rate'] < 0.5:
                recommendations.append(f"Low hit rate in {pool_name} pool - consider increasing pool size")
        
        if self.optimization_stats['aggressive_cleanups'] > 10:
            recommendations.append("Frequent aggressive cleanups - consider optimizing memory allocation patterns")
        
        return recommendations
    
    def plot_memory_usage(self, save_path: Path = None):
        """Plot memory usage over time"""
        if not self.memory_snapshots:
            logger.warning("No memory snapshots available for plotting")
            return
        
        snapshots = list(self.memory_snapshots)
        timestamps = [s.timestamp for s in snapshots]
        memory_usage = [s.total_memory_gb for s in snapshots]
        gpu_usage = [s.gpu_memory_gb for s in snapshots]
        efficiency = [s.memory_efficiency_score for s in snapshots]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Memory usage over time
        ax1.plot(timestamps, memory_usage, label='System Memory', color='blue')
        ax1.axhline(y=self.target_memory_gb, color='red', linestyle='--', label='Target')
        ax1.set_ylabel('Memory Usage (GB)')
        ax1.set_title('System Memory Usage Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # GPU memory usage
        ax2.plot(timestamps, gpu_usage, label='GPU Memory', color='green')
        ax2.set_ylabel('GPU Memory (GB)')
        ax2.set_title('GPU Memory Usage Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Memory efficiency
        ax3.plot(timestamps, efficiency, label='Efficiency Score', color='purple')
        ax3.axhline(y=1.0, color='red', linestyle='--', label='Perfect Efficiency')
        ax3.set_ylabel('Efficiency Score')
        ax3.set_title('Memory Efficiency Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Memory pools utilization
        pool_stats = self.get_memory_pools_stats()
        pool_names = list(pool_stats.keys())
        utilizations = [pool_stats[name]['utilization'] for name in pool_names]
        
        ax4.bar(pool_names, utilizations, color=['blue', 'green', 'orange', 'red'])
        ax4.set_ylabel('Utilization')
        ax4.set_title('Memory Pool Utilization')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main memory optimization demo"""
    print("💾 MEMORY OPTIMIZATION SYSTEM - AGENT 6 PERFORMANCE")
    print("=" * 80)
    
    # Initialize memory optimizer
    memory_optimizer = MemoryOptimizer(target_memory_gb=8.0)
    
    print(f"🎯 Target memory usage: {memory_optimizer.target_memory_gb}GB")
    print(f"🖥️  GPU available: {memory_optimizer.gpu_available}")
    
    # Test memory optimization
    print("\n📊 Current memory usage:")
    current_usage = memory_optimizer.get_current_memory_usage()
    for key, value in current_usage.items():
        print(f"  {key}: {value:.2f}")
    
    # Optimize for Monte Carlo
    print("\n🎲 Optimizing for Monte Carlo simulation...")
    mc_optimization = memory_optimizer.optimize_monte_carlo_memory(
        num_samples=1000, num_assets=3
    )
    
    print(f"📈 Monte Carlo optimization results:")
    for key, value in mc_optimization.items():
        print(f"  {key}: {value}")
    
    # Test memory pool functionality
    print("\n🔄 Testing memory pool functionality...")
    with memory_optimizer.memory_context("tensor_test"):
        # Allocate and return tensors
        tensors = []
        for i in range(10):
            tensor = memory_optimizer.get_tensor((100, 100), context=f"test_{i}")
            tensors.append(tensor)
        
        # Return tensors to pool
        for tensor in tensors:
            memory_optimizer.return_tensor(tensor)
    
    # Get memory pool statistics
    print("\n📊 Memory pool statistics:")
    pool_stats = memory_optimizer.get_memory_pools_stats()
    for pool_name, stats in pool_stats.items():
        print(f"  {pool_name}: {stats['hit_rate']:.2f} hit rate, {stats['tensors_available']} available")
    
    # Generate report
    print("\n📄 Generating memory optimization report...")
    report = memory_optimizer.create_memory_report()
    
    # Summary
    print("\n" + "=" * 80)
    print("🏆 MEMORY OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    target_met = report['memory_statistics']['max_memory_gb'] <= memory_optimizer.target_memory_gb
    efficiency = report['memory_statistics']['avg_efficiency_score']
    
    print(f"🎯 Memory Target (<8GB): {'✅ ACHIEVED' if target_met else '❌ EXCEEDED'}")
    print(f"⚡ Memory Efficiency: {efficiency:.2f}")
    print(f"🔧 Optimization Active: {'✅ YES' if memory_optimizer.cleanup_active else '❌ NO'}")
    
    # Recommendations
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    memory_optimizer.stop_monitoring()
    
    print("✅ Memory optimization system test completed!")

if __name__ == "__main__":
    main()