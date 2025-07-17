"""
Performance Benchmark for Strategic MARL Component Parallel Execution.

This script benchmarks the parallel execution optimizations and compares
performance before and after the improvements.
"""

import asyncio
import time
import numpy as np
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# from strategic_marl_component import StrategicMARLComponent


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    name: str
    execution_times_ms: List[float]
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    cache_hit_rate: float
    parallel_speedup: float
    success_rate: float


class PerformanceBenchmark:
    """Performance benchmark for Strategic MARL Component."""
    
    def __init__(self, component: StrategicMARLComponent):
        self.component = component
        self.test_iterations = 50
        self.warmup_iterations = 10
    
    async def run_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run comprehensive performance benchmark."""
        results = {}
        
        # Generate test data
        matrix_data = self._generate_test_matrix()
        shared_context = self._generate_test_context()
        
        # Warmup
        print("Running warmup iterations...")
        await self._run_warmup(matrix_data, shared_context)
        
        # Benchmark parallel execution
        print("Benchmarking parallel execution...")
        results['parallel_execution'] = await self._benchmark_parallel_execution(
            matrix_data, shared_context
        )
        
        # Benchmark cache performance
        print("Benchmarking cache performance...")
        results['cache_performance'] = await self._benchmark_cache_performance(
            matrix_data, shared_context
        )
        
        # Benchmark under load
        print("Benchmarking under load...")
        results['load_testing'] = await self._benchmark_load_testing(
            matrix_data, shared_context
        )
        
        return results
    
    def _generate_test_matrix(self) -> np.ndarray:
        """Generate test matrix data."""
        # Generate realistic market data
        np.random.seed(42)  # Reproducible results
        
        # Base prices
        base_prices = np.random.uniform(100, 1000, 48)
        
        # Features: price, volume, momentum, volatility, etc.
        matrix = np.zeros((48, 13))
        
        # Price trends
        matrix[:, 0] = base_prices * (1 + np.random.normal(0, 0.01, 48))
        matrix[:, 1] = base_prices * np.random.uniform(0.8, 1.2, 48)
        
        # Volume features
        matrix[:, 2] = np.random.exponential(1000, 48)
        matrix[:, 3] = np.random.exponential(1500, 48)
        matrix[:, 4] = np.random.exponential(800, 48)
        matrix[:, 5] = np.random.exponential(1200, 48)
        
        # Technical indicators
        matrix[:, 6] = np.random.normal(0, 0.02, 48)  # RSI-like
        matrix[:, 7] = np.random.normal(0, 0.01, 48)  # MACD-like
        matrix[:, 8] = np.random.normal(0, 0.015, 48)  # Bollinger-like
        
        # Momentum features
        matrix[:, 9] = np.random.normal(0, 0.005, 48)   # 20-day momentum
        matrix[:, 10] = np.random.normal(0, 0.003, 48)  # 50-day momentum
        
        # Volatility
        matrix[:, 11] = np.random.uniform(0.01, 0.05, 48)
        matrix[:, 12] = np.random.uniform(0.5, 2.0, 48)  # Volume ratio
        
        return matrix.astype(np.float32)
    
    def _generate_test_context(self) -> Dict[str, Any]:
        """Generate test shared context."""
        return {
            'market_volatility': 0.025,
            'volume_profile': 1.2,
            'momentum_signal': 0.008,
            'trend_strength': 0.04,
            'mmd_score': 0.015,
            'price_trend': 0.002,
            'market_regime': 'trending',
            'timestamp': datetime.now().isoformat()
        }
    
    async def _run_warmup(self, matrix_data: np.ndarray, shared_context: Dict[str, Any]):
        """Run warmup iterations."""
        for i in range(self.warmup_iterations):
            try:
                await self.component._execute_agents_parallel(matrix_data, shared_context)
            except Exception as e:
                print(f"Warmup iteration {i+1} failed: {e}")
    
    async def _benchmark_parallel_execution(
        self, 
        matrix_data: np.ndarray, 
        shared_context: Dict[str, Any]
    ) -> BenchmarkResult:
        """Benchmark parallel execution performance."""
        execution_times = []
        success_count = 0
        
        for i in range(self.test_iterations):
            start_time = time.time()
            try:
                results = await self.component._execute_agents_parallel(matrix_data, shared_context)
                execution_time_ms = (time.time() - start_time) * 1000
                execution_times.append(execution_time_ms)
                
                # Check for successful execution
                if len(results) == 3 and all(not r.get('fallback', False) for r in results):
                    success_count += 1
                    
            except Exception as e:
                print(f"Iteration {i+1} failed: {e}")
                execution_times.append(float('inf'))
        
        # Filter out failed executions
        valid_times = [t for t in execution_times if t != float('inf')]
        
        if not valid_times:
            return BenchmarkResult(
                name="Parallel Execution",
                execution_times_ms=[],
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                cache_hit_rate=0,
                parallel_speedup=0,
                success_rate=0
            )
        
        return BenchmarkResult(
            name="Parallel Execution",
            execution_times_ms=valid_times,
            avg_time_ms=statistics.mean(valid_times),
            min_time_ms=min(valid_times),
            max_time_ms=max(valid_times),
            std_dev_ms=statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
            cache_hit_rate=self.component._parallel_metrics['cache_hit_rate'],
            parallel_speedup=self.component._parallel_metrics['parallel_speedup'],
            success_rate=success_count / self.test_iterations
        )
    
    async def _benchmark_cache_performance(
        self, 
        matrix_data: np.ndarray, 
        shared_context: Dict[str, Any]
    ) -> BenchmarkResult:
        """Benchmark cache performance."""
        # Clear cache first
        self.component._agent_result_cache.clear()
        
        # First execution (cache miss)
        start_time = time.time()
        await self.component._execute_agents_parallel(matrix_data, shared_context)
        miss_time_ms = (time.time() - start_time) * 1000
        
        # Subsequent executions (cache hits)
        cache_times = []
        for i in range(10):
            start_time = time.time()
            await self.component._execute_agents_parallel(matrix_data, shared_context)
            cache_time_ms = (time.time() - start_time) * 1000
            cache_times.append(cache_time_ms)
        
        cache_speedup = miss_time_ms / statistics.mean(cache_times) if cache_times else 1.0
        
        return BenchmarkResult(
            name="Cache Performance",
            execution_times_ms=cache_times,
            avg_time_ms=statistics.mean(cache_times),
            min_time_ms=min(cache_times),
            max_time_ms=max(cache_times),
            std_dev_ms=statistics.stdev(cache_times) if len(cache_times) > 1 else 0,
            cache_hit_rate=self.component._parallel_metrics['cache_hit_rate'],
            parallel_speedup=cache_speedup,
            success_rate=1.0
        )
    
    async def _benchmark_load_testing(
        self, 
        matrix_data: np.ndarray, 
        shared_context: Dict[str, Any]
    ) -> BenchmarkResult:
        """Benchmark performance under concurrent load."""
        
        async def single_execution():
            start_time = time.time()
            await self.component._execute_agents_parallel(matrix_data, shared_context)
            return (time.time() - start_time) * 1000
        
        # Run concurrent executions
        concurrent_tasks = [single_execution() for _ in range(10)]
        execution_times = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_times = [t for t in execution_times if not isinstance(t, Exception)]
        
        if not valid_times:
            return BenchmarkResult(
                name="Load Testing",
                execution_times_ms=[],
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                std_dev_ms=0,
                cache_hit_rate=0,
                parallel_speedup=0,
                success_rate=0
            )
        
        return BenchmarkResult(
            name="Load Testing",
            execution_times_ms=valid_times,
            avg_time_ms=statistics.mean(valid_times),
            min_time_ms=min(valid_times),
            max_time_ms=max(valid_times),
            std_dev_ms=statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
            cache_hit_rate=self.component._parallel_metrics['cache_hit_rate'],
            parallel_speedup=self.component._parallel_metrics['parallel_speedup'],
            success_rate=len(valid_times) / len(concurrent_tasks)
        )
    
    def print_results(self, results: Dict[str, BenchmarkResult]):
        """Print benchmark results."""
        print("\n" + "="*80)
        print("STRATEGIC MARL COMPONENT PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        target_latency = 10.0  # ms
        
        for name, result in results.items():
            print(f"\n{result.name.upper()} BENCHMARK:")
            print(f"  Average Execution Time: {result.avg_time_ms:.2f}ms")
            print(f"  Min Execution Time: {result.min_time_ms:.2f}ms")
            print(f"  Max Execution Time: {result.max_time_ms:.2f}ms")
            print(f"  Standard Deviation: {result.std_dev_ms:.2f}ms")
            print(f"  Cache Hit Rate: {result.cache_hit_rate:.2%}")
            print(f"  Parallel Speedup: {result.parallel_speedup:.2f}x")
            print(f"  Success Rate: {result.success_rate:.2%}")
            
            # Performance assessment
            if result.avg_time_ms <= target_latency:
                print(f"  ✅ MEETS TARGET (<{target_latency}ms)")
            else:
                print(f"  ❌ EXCEEDS TARGET ({result.avg_time_ms:.2f}ms > {target_latency}ms)")
            
            # Percentiles
            if result.execution_times_ms:
                sorted_times = sorted(result.execution_times_ms)
                p50 = sorted_times[len(sorted_times) // 2]
                p95 = sorted_times[int(len(sorted_times) * 0.95)]
                p99 = sorted_times[int(len(sorted_times) * 0.99)]
                
                print(f"  P50: {p50:.2f}ms, P95: {p95:.2f}ms, P99: {p99:.2f}ms")
        
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT:")
        
        # Check if all benchmarks meet target
        all_meet_target = all(
            result.avg_time_ms <= target_latency 
            for result in results.values()
        )
        
        if all_meet_target:
            print("✅ ALL BENCHMARKS MEET <10ms TARGET")
            print("🚀 PARALLEL EXECUTION OPTIMIZATION SUCCESSFUL!")
        else:
            print("❌ SOME BENCHMARKS EXCEED TARGET")
            print("🔧 FURTHER OPTIMIZATION NEEDED")
        
        print("="*80)


async def main():
    """Run the performance benchmark."""
    print("Strategic MARL Component Performance Benchmark")
    print("Initializing component...")
    
    # Mock kernel for testing
    class MockKernel:
        def __init__(self):
            self.config = MockConfig()
            self.event_bus = MockEventBus()
    
    class MockConfig:
        def get(self, key, default=None):
            return {
                'strategic': {
                    'environment': {
                        'matrix_shape': [48, 13],
                        'feature_indices': {
                            'mlmi_expert': [0, 1, 9, 10],
                            'nwrqk_expert': [2, 3, 4, 5],
                            'regime_expert': [6, 7, 8, 11, 12]
                        }
                    },
                    'ensemble': {
                        'confidence_threshold': 0.65,
                        'weights': [0.33, 0.33, 0.34]
                    },
                    'performance': {
                        'max_inference_latency_ms': 10.0,
                        'agent_timeout_ms': 8.0
                    }
                }
            }.get(key, default)
    
    class MockEventBus:
        async def subscribe(self, event_type, handler):
            pass
    
    # Create component
    kernel = MockKernel()
    component = StrategicMARLComponent(kernel)
    
    # Mock agents for testing
    class MockAgent:
        async def predict(self, matrix_data, shared_context):
            # Simulate realistic agent computation time
            await asyncio.sleep(0.003)  # 3ms
            return {
                'agent_name': 'Mock',
                'action_probabilities': [0.3, 0.4, 0.3],
                'confidence': 0.8,
                'features_used': [0, 1, 2],
                'feature_importance': {'feature_0': 0.5, 'feature_1': 0.3, 'feature_2': 0.2},
                'internal_state': {},
                'computation_time_ms': 3.0,
                'fallback': False
            }
    
    component.mlmi_agent = MockAgent()
    component.nwrqk_agent = MockAgent()
    component.regime_agent = MockAgent()
    
    # Run benchmark
    benchmark = PerformanceBenchmark(component)
    results = await benchmark.run_benchmark()
    
    # Print results
    benchmark.print_results(results)


if __name__ == "__main__":
    asyncio.run(main())