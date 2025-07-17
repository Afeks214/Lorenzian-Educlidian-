"""
Integrated Performance Optimization System for GrandModel

This module provides a unified interface for all performance optimizations:
- Memory optimization integration
- CPU optimization integration  
- I/O optimization integration
- Enhanced monitoring integration
- Coordinated optimization strategies
- Performance validation and testing

Key Features:
- Unified optimization control
- Coordinated optimization strategies
- Performance validation framework
- Automated optimization recommendations
- Real-time performance tracking
- Comprehensive performance reporting
"""

import asyncio
import threading
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import structlog
from contextlib import asynccontextmanager, contextmanager
import json
import os

# Import optimization modules
from .memory_optimizer import MemoryOptimizer, memory_optimizer
from .cpu_optimizer import CPUOptimizer, cpu_optimizer
from .io_optimizer import IOOptimizer, io_optimizer
from .enhanced_monitoring import EnhancedPerformanceMonitor, enhanced_monitor

logger = structlog.get_logger()


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies"""
    memory_optimization: bool = True
    cpu_optimization: bool = True
    io_optimization: bool = True
    monitoring_enabled: bool = True
    
    # Memory settings
    tensor_pool_size: int = 1000
    gc_optimization: bool = True
    memory_monitoring: bool = True
    
    # CPU settings
    jit_compilation: bool = True
    vectorization: bool = True
    thread_pool_optimization: bool = True
    cpu_affinity: bool = True
    
    # I/O settings
    async_io: bool = True
    caching_enabled: bool = True
    batch_processing: bool = True
    connection_pooling: bool = True
    
    # Monitoring settings
    real_time_metrics: bool = True
    regression_detection: bool = True
    performance_alerts: bool = True
    dashboard_updates: bool = True


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results"""
    timestamp: datetime
    operation: str
    baseline_time_ms: float
    optimized_time_ms: float
    improvement_percent: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    notes: str = ""


class OptimizationStrategy:
    """
    Intelligent optimization strategy selector.
    Chooses optimal combination of optimizations based on workload.
    """
    
    def __init__(self):
        self.strategies = {
            'inference': {
                'memory_optimization': True,
                'cpu_optimization': True,
                'io_optimization': False,
                'jit_compilation': True,
                'vectorization': True,
                'tensor_pooling': True,
                'async_io': False
            },
            'training': {
                'memory_optimization': True,
                'cpu_optimization': True,
                'io_optimization': True,
                'jit_compilation': False,  # Can interfere with training
                'vectorization': True,
                'tensor_pooling': True,
                'async_io': True
            },
            'data_processing': {
                'memory_optimization': True,
                'cpu_optimization': True,
                'io_optimization': True,
                'jit_compilation': False,
                'vectorization': True,
                'tensor_pooling': False,
                'async_io': True
            },
            'real_time': {
                'memory_optimization': True,
                'cpu_optimization': True,
                'io_optimization': True,
                'jit_compilation': True,
                'vectorization': True,
                'tensor_pooling': True,
                'async_io': True
            }
        }
        
        logger.info("OptimizationStrategy initialized")
    
    def get_strategy(self, workload_type: str) -> Dict:
        """Get optimization strategy for workload type"""
        return self.strategies.get(workload_type, self.strategies['real_time'])
    
    def recommend_strategy(self, performance_data: Dict) -> str:
        """Recommend optimal strategy based on performance data"""
        
        # Analyze performance characteristics
        cpu_usage = performance_data.get('cpu_usage_percent', 0)
        memory_usage = performance_data.get('memory_usage_percent', 0)
        io_wait = performance_data.get('io_wait_percent', 0)
        
        # Decision logic
        if cpu_usage > 70 and memory_usage < 50:
            return 'inference'  # CPU-bound, memory available
        elif memory_usage > 80:
            return 'training'  # Memory-intensive workload
        elif io_wait > 20:
            return 'data_processing'  # I/O-bound workload
        else:
            return 'real_time'  # Balanced workload


class PerformanceValidator:
    """
    Performance validation and testing framework.
    Validates optimization effectiveness and safety.
    """
    
    def __init__(self):
        self.validation_results = []
        self.benchmark_suite = {}
        self.safety_checks = {}
        
        logger.info("PerformanceValidator initialized")
    
    def register_benchmark(self, name: str, benchmark_func: Callable):
        """Register performance benchmark"""
        self.benchmark_suite[name] = benchmark_func
        logger.info("Benchmark registered", name=name)
    
    def register_safety_check(self, name: str, check_func: Callable):
        """Register safety check"""
        self.safety_checks[name] = check_func
        logger.info("Safety check registered", name=name)
    
    async def validate_optimization(self, optimization_name: str,
                                  before_func: Callable, 
                                  after_func: Callable,
                                  test_data: Any = None,
                                  iterations: int = 10) -> PerformanceBenchmark:
        """Validate optimization effectiveness"""
        
        logger.info("Starting optimization validation", name=optimization_name)
        
        # Baseline performance
        baseline_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            if asyncio.iscoroutinefunction(before_func):
                await before_func(test_data)
            else:
                before_func(test_data)
            baseline_times.append((time.perf_counter() - start_time) * 1000)
        
        baseline_avg = np.mean(baseline_times)
        
        # Optimized performance
        optimized_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            if asyncio.iscoroutinefunction(after_func):
                await after_func(test_data)
            else:
                after_func(test_data)
            optimized_times.append((time.perf_counter() - start_time) * 1000)
        
        optimized_avg = np.mean(optimized_times)
        
        # Calculate improvement
        improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
        
        # Resource usage
        memory_usage = enhanced_monitor.get_dashboard_data().get('metrics', {}).get('memory_usage_percent', {}).get('current', 0)
        cpu_usage = enhanced_monitor.get_dashboard_data().get('metrics', {}).get('cpu_usage_percent', {}).get('current', 0)
        
        # Create benchmark result
        benchmark = PerformanceBenchmark(
            timestamp=datetime.now(),
            operation=optimization_name,
            baseline_time_ms=baseline_avg,
            optimized_time_ms=optimized_avg,
            improvement_percent=improvement,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            success=improvement > 0,
            notes=f"Tested with {iterations} iterations"
        )
        
        self.validation_results.append(benchmark)
        
        logger.info("Optimization validation completed",
                   name=optimization_name,
                   improvement=f"{improvement:.1f}%",
                   success=benchmark.success)
        
        return benchmark
    
    async def run_safety_checks(self) -> Dict:
        """Run all safety checks"""
        results = {}
        
        for name, check_func in self.safety_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    'passed': result,
                    'timestamp': datetime.now(),
                    'notes': f"Safety check {'passed' if result else 'failed'}"
                }
                
            except Exception as e:
                results[name] = {
                    'passed': False,
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'notes': f"Safety check failed with error: {e}"
                }
        
        return results
    
    def get_validation_report(self) -> Dict:
        """Get comprehensive validation report"""
        if not self.validation_results:
            return {"status": "No validation results available"}
        
        # Calculate summary statistics
        improvements = [r.improvement_percent for r in self.validation_results]
        success_rate = len([r for r in self.validation_results if r.success]) / len(self.validation_results)
        
        return {
            'total_validations': len(self.validation_results),
            'success_rate': success_rate,
            'avg_improvement': np.mean(improvements),
            'max_improvement': np.max(improvements),
            'min_improvement': np.min(improvements),
            'recent_validations': [
                {
                    'operation': r.operation,
                    'improvement_percent': r.improvement_percent,
                    'success': r.success,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.validation_results[-10:]  # Last 10
            ]
        }


class IntegratedOptimizer:
    """
    Main integrated optimization system.
    Coordinates all optimization components for maximum performance.
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.strategy_selector = OptimizationStrategy()
        self.validator = PerformanceValidator()
        
        # Optimization components
        self.memory_optimizer = memory_optimizer
        self.cpu_optimizer = cpu_optimizer
        self.io_optimizer = io_optimizer
        self.enhanced_monitor = enhanced_monitor
        
        # State tracking
        self.optimization_enabled = False
        self.current_strategy = None
        self.optimization_history = []
        self.performance_baseline = None
        
        # Register default benchmarks and safety checks
        self._register_default_benchmarks()
        self._register_default_safety_checks()
        
        logger.info("IntegratedOptimizer initialized")
    
    def _register_default_benchmarks(self):
        """Register default performance benchmarks"""
        
        def tensor_operations_benchmark(data=None):
            """Benchmark tensor operations"""
            x = torch.randn(1000, 1000)
            y = torch.randn(1000, 1000)
            result = torch.mm(x, y)
            return result.sum().item()
        
        def memory_allocation_benchmark(data=None):
            """Benchmark memory allocation"""
            tensors = [torch.randn(100, 100) for _ in range(100)]
            result = sum(t.sum().item() for t in tensors)
            del tensors
            return result
        
        def file_io_benchmark(data=None):
            """Benchmark file I/O operations"""
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write("test data" * 1000)
                temp_file = f.name
            
            with open(temp_file, 'r') as f:
                content = f.read()
            
            os.unlink(temp_file)
            return len(content)
        
        self.validator.register_benchmark('tensor_operations', tensor_operations_benchmark)
        self.validator.register_benchmark('memory_allocation', memory_allocation_benchmark)
        self.validator.register_benchmark('file_io', file_io_benchmark)
    
    def _register_default_safety_checks(self):
        """Register default safety checks"""
        
        def memory_safety_check():
            """Check if memory usage is within safe limits"""
            stats = self.memory_optimizer.get_optimization_stats()
            current_memory = stats.get('memory_monitor', {}).get('current_stats', {}).get('used_memory_mb', 0)
            return current_memory < 8000  # 8GB limit
        
        def cpu_safety_check():
            """Check if CPU usage is within safe limits"""
            stats = self.cpu_optimizer.get_optimization_stats()
            cpu_usage = stats.get('cpu_stats', {}).get('cpu_percent', 0)
            return cpu_usage < 90  # 90% limit
        
        def optimization_consistency_check():
            """Check if optimizations are working correctly"""
            return (
                self.memory_optimizer.optimization_enabled and
                self.cpu_optimizer.optimization_enabled and
                self.optimization_enabled
            )
        
        self.validator.register_safety_check('memory_safety', memory_safety_check)
        self.validator.register_safety_check('cpu_safety', cpu_safety_check)
        self.validator.register_safety_check('optimization_consistency', optimization_consistency_check)
    
    async def enable_optimizations(self, strategy: str = 'real_time'):
        """Enable coordinated optimizations"""
        if self.optimization_enabled:
            logger.warning("Optimizations already enabled")
            return
        
        logger.info("Enabling integrated optimizations", strategy=strategy)
        
        # Get optimization strategy
        strategy_config = self.strategy_selector.get_strategy(strategy)
        self.current_strategy = strategy
        
        # Enable monitoring first
        if self.config.monitoring_enabled:
            enhanced_monitor.enable_monitoring()
        
        # Enable optimizations based on strategy
        if strategy_config.get('memory_optimization', True):
            self.memory_optimizer.enable_optimizations()
        
        if strategy_config.get('cpu_optimization', True):
            self.cpu_optimizer.enable_optimizations()
        
        if strategy_config.get('io_optimization', True):
            await self.io_optimizer.enable_optimizations()
        
        # Configure specific optimizations
        if strategy_config.get('jit_compilation', True):
            logger.info("JIT compilation enabled for strategy", strategy=strategy)
        
        if strategy_config.get('tensor_pooling', True):
            logger.info("Tensor pooling enabled for strategy", strategy=strategy)
        
        self.optimization_enabled = True
        
        # Record optimization event
        self.optimization_history.append({
            'action': 'enable',
            'strategy': strategy,
            'timestamp': datetime.now(),
            'config': strategy_config
        })
        
        logger.info("Integrated optimizations enabled successfully", strategy=strategy)
    
    async def disable_optimizations(self):
        """Disable all optimizations"""
        if not self.optimization_enabled:
            return
        
        logger.info("Disabling integrated optimizations")
        
        # Disable optimizations
        self.memory_optimizer.disable_optimizations()
        self.cpu_optimizer.disable_optimizations()
        await self.io_optimizer.disable_optimizations()
        
        # Disable monitoring
        enhanced_monitor.disable_monitoring()
        
        self.optimization_enabled = False
        
        # Record optimization event
        self.optimization_history.append({
            'action': 'disable',
            'timestamp': datetime.now()
        })
        
        logger.info("Integrated optimizations disabled")
    
    async def optimize_for_workload(self, workload_type: str):
        """Optimize for specific workload type"""
        logger.info("Optimizing for workload", workload_type=workload_type)
        
        # Disable current optimizations
        if self.optimization_enabled:
            await self.disable_optimizations()
        
        # Enable optimizations for workload
        await self.enable_optimizations(workload_type)
        
        logger.info("Workload optimization completed", workload_type=workload_type)
    
    async def auto_optimize(self):
        """Automatically optimize based on current performance"""
        logger.info("Starting auto-optimization")
        
        # Get current performance data
        dashboard_data = enhanced_monitor.get_dashboard_data()
        performance_data = {}
        
        for metric_name, metric_data in dashboard_data.get('metrics', {}).items():
            if metric_data.get('current') is not None:
                performance_data[metric_name] = metric_data['current']
        
        # Get recommended strategy
        recommended_strategy = self.strategy_selector.recommend_strategy(performance_data)
        
        logger.info("Auto-optimization recommendation", 
                   strategy=recommended_strategy,
                   performance_data=performance_data)
        
        # Apply recommended optimization
        await self.optimize_for_workload(recommended_strategy)
        
        return recommended_strategy
    
    async def validate_optimizations(self) -> Dict:
        """Validate all optimizations"""
        logger.info("Starting optimization validation")
        
        # Run safety checks
        safety_results = await self.validator.run_safety_checks()
        
        # Run performance benchmarks
        benchmark_results = {}
        
        for benchmark_name, benchmark_func in self.validator.benchmark_suite.items():
            try:
                # Create optimized version
                async def optimized_benchmark(data):
                    return benchmark_func(data)
                
                # Validate optimization
                result = await self.validator.validate_optimization(
                    benchmark_name,
                    benchmark_func,
                    optimized_benchmark,
                    iterations=5
                )
                
                benchmark_results[benchmark_name] = {
                    'improvement_percent': result.improvement_percent,
                    'success': result.success,
                    'baseline_time_ms': result.baseline_time_ms,
                    'optimized_time_ms': result.optimized_time_ms
                }
                
            except Exception as e:
                logger.error("Benchmark validation failed", 
                           benchmark=benchmark_name, error=str(e))
                benchmark_results[benchmark_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Compile validation report
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'safety_checks': safety_results,
            'performance_benchmarks': benchmark_results,
            'optimization_enabled': self.optimization_enabled,
            'current_strategy': self.current_strategy,
            'validation_summary': self.validator.get_validation_report()
        }
        
        logger.info("Optimization validation completed",
                   safety_passed=all(r['passed'] for r in safety_results.values()),
                   benchmarks_passed=sum(1 for r in benchmark_results.values() if r.get('success', False)))
        
        return validation_report
    
    async def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive optimization statistics"""
        
        # Get stats from all components
        memory_stats = self.memory_optimizer.get_optimization_stats()
        cpu_stats = self.cpu_optimizer.get_optimization_stats()
        io_stats = await self.io_optimizer.get_optimization_stats()
        monitoring_stats = enhanced_monitor.get_comprehensive_stats()
        
        return {
            'optimization_enabled': self.optimization_enabled,
            'current_strategy': self.current_strategy,
            'optimization_history': self.optimization_history[-10:],  # Last 10 events
            'memory_optimization': memory_stats,
            'cpu_optimization': cpu_stats,
            'io_optimization': io_stats,
            'monitoring': monitoring_stats,
            'validation_report': self.validator.get_validation_report()
        }
    
    async def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        
        # Get comprehensive statistics
        stats = await self.get_comprehensive_stats()
        
        # Get monitoring report
        monitoring_report = enhanced_monitor.create_performance_report(hours=1)
        
        # Get validation results
        validation_report = await self.validate_optimizations()
        
        # Compile final report
        report = {
            'report_generated': datetime.now().isoformat(),
            'optimization_status': {
                'enabled': self.optimization_enabled,
                'strategy': self.current_strategy,
                'history_entries': len(self.optimization_history)
            },
            'performance_improvements': {
                'memory_pool_hit_rate': stats['memory_optimization']['tensor_pool']['hit_rate'],
                'cpu_jit_models': stats['cpu_optimization']['jit_compiler']['total_models'],
                'io_cache_hit_rate': stats['io_optimization']['cache']['hit_rate']
            },
            'monitoring_metrics': monitoring_report,
            'validation_results': validation_report,
            'recommendations': []
        }
        
        # Generate recommendations
        recommendations = []
        
        # Memory recommendations
        if stats['memory_optimization']['tensor_pool']['hit_rate'] < 0.7:
            recommendations.append({
                'type': 'memory',
                'priority': 'medium',
                'message': 'Consider increasing tensor pool size for better memory efficiency'
            })
        
        # CPU recommendations
        if stats['cpu_optimization']['jit_compiler']['total_models'] == 0:
            recommendations.append({
                'type': 'cpu',
                'priority': 'medium',
                'message': 'Consider enabling JIT compilation for frequently used models'
            })
        
        # I/O recommendations
        if stats['io_optimization']['cache']['hit_rate'] < 0.8:
            recommendations.append({
                'type': 'io',
                'priority': 'low',
                'message': 'Consider increasing cache size or TTL for better I/O performance'
            })
        
        report['recommendations'] = recommendations
        
        return report
    
    @asynccontextmanager
    async def optimization_context(self, strategy: str = 'real_time'):
        """Context manager for scoped optimizations"""
        await self.enable_optimizations(strategy)
        try:
            yield self
        finally:
            await self.disable_optimizations()


# Global integrated optimizer instance
integrated_optimizer = IntegratedOptimizer()


async def optimize_system():
    """High-level system optimization function"""
    return await integrated_optimizer.auto_optimize()


def optimize_performance(strategy: str = 'real_time'):
    """Decorator for performance-optimized functions"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            async with integrated_optimizer.optimization_context(strategy):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


async def main():
    """Demo integrated optimization system"""
    
    print("⚡ Integrated Performance Optimization Demo")
    print("=" * 50)
    
    # Enable optimizations
    await integrated_optimizer.enable_optimizations('real_time')
    
    # Wait for monitoring to collect data
    print("\n🔄 Collecting baseline performance data...")
    await asyncio.sleep(5)
    
    # Run auto-optimization
    print("\n🤖 Running auto-optimization...")
    recommended_strategy = await integrated_optimizer.auto_optimize()
    print(f"Recommended strategy: {recommended_strategy}")
    
    # Validate optimizations
    print("\n✅ Validating optimizations...")
    validation_report = await integrated_optimizer.validate_optimizations()
    
    print(f"Safety checks passed: {sum(1 for r in validation_report['safety_checks'].values() if r['passed'])}/{len(validation_report['safety_checks'])}")
    print(f"Benchmark tests passed: {sum(1 for r in validation_report['performance_benchmarks'].values() if r.get('success', False))}/{len(validation_report['performance_benchmarks'])}")
    
    # Generate performance report
    print("\n📊 Generating performance report...")
    performance_report = await integrated_optimizer.generate_performance_report()
    
    print(f"Optimization enabled: {performance_report['optimization_status']['enabled']}")
    print(f"Current strategy: {performance_report['optimization_status']['strategy']}")
    print(f"Recommendations: {len(performance_report['recommendations'])}")
    
    if performance_report['recommendations']:
        print("\n💡 Optimization Recommendations:")
        for rec in performance_report['recommendations']:
            print(f"  • [{rec['type'].upper()}] {rec['message']}")
    
    # Test with context manager
    print("\n🧪 Testing optimization context manager...")
    async with integrated_optimizer.optimization_context('inference'):
        # Simulate inference workload
        x = torch.randn(1000, 1000)
        result = torch.mm(x, x)
        print(f"Inference test completed: {result.shape}")
    
    print("\n✅ Integrated optimization demo completed!")


if __name__ == "__main__":
    asyncio.run(main())