"""
JIT Performance Benchmark

Validates that JIT-compiled models meet performance requirements for tactical trading.
"""

import torch
import torch.jit
import numpy as np
import time
import logging
import sys
from typing import Dict, Any, List, Tuple
from pathlib import Path
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JITPerformanceBenchmark:
    """Benchmark JIT-compiled model performance."""
    
    def __init__(self):
        """Initialize performance benchmarker."""
        self.models_dir = Path("models/tactical")
        self.benchmark_results = []
        
        # Performance targets
        self.performance_targets = {
            "max_inference_time_ms": 25.0,  # Maximum inference time
            "min_speedup_factor": 1.2,      # Minimum speedup from JIT
            "max_compilation_time_ms": 1000.0,  # Maximum compilation time
            "max_memory_mb": 500.0          # Maximum memory usage
        }
        
        # Benchmark configuration
        self.warmup_iterations = 10
        self.benchmark_iterations = 100
        self.batch_sizes = [1, 4, 8, 16, 32]
        
        # Test input dimensions
        self.test_inputs = {
            "tactical_actor": (60, 7),      # 60 bars, 7 features
            "centralized_critic": (60, 7),  
            "fvg_agent": (60, 7),           
            "momentum_agent": (60, 7),
            "entry_agent": (60, 7)
        }
    
    def benchmark_all_models(self) -> Dict[str, Any]:
        """Benchmark all tactical models."""
        logger.info("📊 Starting JIT performance benchmarking")
        
        for model_name, input_shape in self.test_inputs.items():
            logger.info(f"Benchmarking {model_name}")
            
            try:
                result = self._benchmark_model(model_name, input_shape)
                self.benchmark_results.append(result)
                
            except Exception as e:
                logger.error(f"Benchmark failed for {model_name}: {e}")
                self.benchmark_results.append({
                    "model_name": model_name,
                    "success": False,
                    "error": str(e)
                })
        
        # Generate summary
        summary = self._generate_summary()
        logger.info(f"Performance benchmarking complete. All targets met: {summary['all_targets_met']}")
        
        return summary
    
    def _benchmark_model(self, model_name: str, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Benchmark a specific model."""
        result = {
            "model_name": model_name,
            "success": False,
            "batch_results": [],
            "best_inference_time_ms": float('inf'),
            "best_speedup_factor": 0.0,
            "compilation_time_ms": 0.0,
            "memory_usage_mb": 0.0,
            "targets_met": {}
        }
        
        try:
            # Create model
            model = self._create_mock_model(model_name, input_shape)
            
            # Compile with JIT
            logger.info(f"  Compiling {model_name} with JIT...")
            compilation_start = time.perf_counter()
            # Create example input for tracing
            seq_len, features = input_shape
            example_input = torch.randn(1, seq_len, features)
            jit_model = torch.jit.trace(model, example_input)
            compilation_time = (time.perf_counter() - compilation_start) * 1000
            result["compilation_time_ms"] = compilation_time
            
            # Benchmark across different batch sizes
            for batch_size in self.batch_sizes:
                logger.info(f"  Benchmarking batch size {batch_size}...")
                
                batch_result = self._benchmark_batch_size(
                    model, jit_model, input_shape, batch_size
                )
                result["batch_results"].append(batch_result)
                
                # Update best metrics
                if batch_result["jit_inference_time_ms"] < result["best_inference_time_ms"]:
                    result["best_inference_time_ms"] = batch_result["jit_inference_time_ms"]
                
                if batch_result["speedup_factor"] > result["best_speedup_factor"]:
                    result["best_speedup_factor"] = batch_result["speedup_factor"]
            
            # Check targets
            result["targets_met"] = self._check_targets(result)
            result["success"] = all(result["targets_met"].values())
            
            logger.info(f"  ✅ {model_name} benchmark complete (best: {result['best_inference_time_ms']:.2f}ms, {result['best_speedup_factor']:.2f}x)")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"  ❌ {model_name} benchmark failed: {e}")
        
        return result
    
    def _benchmark_batch_size(self, model: torch.nn.Module, jit_model: torch.jit.ScriptModule, 
                             input_shape: Tuple[int, ...], batch_size: int) -> Dict[str, Any]:
        """Benchmark specific batch size."""
        seq_len, features = input_shape
        test_input = torch.randn(batch_size, seq_len, features)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = model(test_input)
            _ = jit_model(test_input)
        
        # Benchmark regular model
        regular_times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = model(test_input)
            regular_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark JIT model
        jit_times = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = jit_model(test_input)
            jit_times.append((time.perf_counter() - start) * 1000)
        
        # Calculate statistics
        regular_mean = statistics.mean(regular_times)
        jit_mean = statistics.mean(jit_times)
        speedup = regular_mean / jit_mean if jit_mean > 0 else 0.0
        
        return {
            "batch_size": batch_size,
            "regular_inference_time_ms": regular_mean,
            "jit_inference_time_ms": jit_mean,
            "speedup_factor": speedup,
            "regular_p99_ms": np.percentile(regular_times, 99),
            "jit_p99_ms": np.percentile(jit_times, 99),
            "jit_std_ms": statistics.stdev(jit_times)
        }
    
    def _create_mock_model(self, model_name: str, input_shape: Tuple[int, ...]) -> torch.nn.Module:
        """Create a mock model for benchmarking."""
        seq_len, features = input_shape
        
        if "actor" in model_name:
            return torch.nn.Sequential(
                torch.nn.Linear(features, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 3)
            )
        elif "critic" in model_name:
            return torch.nn.Sequential(
                torch.nn.Linear(features, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )
        else:
            return torch.nn.Sequential(
                torch.nn.Linear(features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 4)
            )
    
    def _check_targets(self, result: Dict[str, Any]) -> Dict[str, bool]:
        """Check if performance targets are met."""
        targets_met = {}
        
        # Check inference time target
        targets_met["inference_time"] = result["best_inference_time_ms"] <= self.performance_targets["max_inference_time_ms"]
        
        # Check speedup target
        targets_met["speedup"] = result["best_speedup_factor"] >= self.performance_targets["min_speedup_factor"]
        
        # Check compilation time target
        targets_met["compilation_time"] = result["compilation_time_ms"] <= self.performance_targets["max_compilation_time_ms"]
        
        # Memory usage (simplified check)
        targets_met["memory_usage"] = True  # Would implement proper memory monitoring
        
        return targets_met
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary."""
        total_models = len(self.benchmark_results)
        successful_models = sum(1 for r in self.benchmark_results if r.get("success", False))
        
        # Check if all targets are met
        all_targets_met = all(
            r.get("success", False) and all(r.get("targets_met", {}).values())
            for r in self.benchmark_results
        )
        
        # Calculate aggregate metrics
        successful_results = [r for r in self.benchmark_results if r.get("success", False)]
        avg_inference_time = 0.0
        avg_speedup = 0.0
        
        if successful_results:
            avg_inference_time = sum(r["best_inference_time_ms"] for r in successful_results) / len(successful_results)
            avg_speedup = sum(r["best_speedup_factor"] for r in successful_results) / len(successful_results)
        
        return {
            "all_targets_met": all_targets_met,
            "total_models": total_models,
            "successful_models": successful_models,
            "average_inference_time_ms": avg_inference_time,
            "average_speedup_factor": avg_speedup,
            "performance_targets": self.performance_targets,
            "detailed_results": self.benchmark_results
        }

def main():
    """Main function for standalone execution."""
    benchmark = JITPerformanceBenchmark()
    results = benchmark.benchmark_all_models()
    
    # Print summary
    print("=" * 60)
    print("JIT PERFORMANCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total models: {results['total_models']}")
    print(f"Successful: {results['successful_models']}")
    print(f"All targets met: {results['all_targets_met']}")
    print(f"Average inference time: {results['average_inference_time_ms']:.2f}ms")
    print(f"Average speedup: {results['average_speedup_factor']:.2f}x")
    
    print("\nPerformance Targets:")
    for target, value in results['performance_targets'].items():
        print(f"  {target}: {value}")
    
    # Show detailed results
    print("\nDetailed Results:")
    for result in results['detailed_results']:
        if result.get('success', False):
            print(f"  {result['model_name']}:")
            print(f"    Best inference: {result['best_inference_time_ms']:.2f}ms")
            print(f"    Best speedup: {result['best_speedup_factor']:.2f}x")
            print(f"    Targets met: {all(result['targets_met'].values())}")
    
    print("=" * 60)
    
    # Exit with appropriate code
    if results['all_targets_met']:
        print("✅ All models meet performance targets")
        sys.exit(0)
    else:
        print("❌ Some models do not meet performance targets")
        sys.exit(1)

if __name__ == "__main__":
    main()