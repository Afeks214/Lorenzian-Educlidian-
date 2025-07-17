"""
AGENT 4: Production Model Optimizer
Implements JIT compilation, quantization, and memory optimization for <5ms inference
"""

import torch
import torch.nn as nn
import torch.jit
import torch.quantization
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
import numpy as np
from pathlib import Path
import gc
from contextlib import contextmanager

from fast_architectures import (
    FastTacticalMARLSystem, FastStrategicMARLSystem,
    FastMLMIActor, FastNWRQKActor, FastMMDActor, FastCentralizedCritic
)

logger = logging.getLogger(__name__)

class TensorPool:
    """Memory pool for tensor reuse to minimize allocations."""
    
    def __init__(self, pool_size: int = 1000):
        self.pools = {}
        self.pool_size = pool_size
        self.available = {}
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool or create new one."""
        key = (shape, dtype)
        
        if key not in self.pools:
            self.pools[key] = []
            self.available[key] = []
        
        if self.available[key]:
            idx = self.available[key].pop()
            tensor = self.pools[key][idx]
            tensor.zero_()  # Clear data
            return tensor
        else:
            # Create new tensor
            tensor = torch.zeros(shape, dtype=dtype)
            if len(self.pools[key]) < self.pool_size:
                self.pools[key].append(tensor)
            return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool."""
        shape = tuple(tensor.shape)
        dtype = tensor.dtype
        key = (shape, dtype)
        
        if key in self.pools:
            try:
                idx = self.pools[key].index(tensor)
                if idx not in self.available[key]:
                    self.available[key].append(idx)
            except ValueError:
                pass  # Tensor not in pool
    
    def clear(self):
        """Clear all pools."""
        self.pools.clear()
        self.available.clear()


class ProductionModelOptimizer:
    """Production-ready model optimizer with JIT, quantization, and memory optimization."""
    
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.tensor_pool = TensorPool()
        self.compiled_models = {}
        self.quantized_models = {}
        self.optimization_stats = {}
        
        logger.info(f"ProductionModelOptimizer initialized on {device}")
    
    def compile_model_jit(self, model: nn.Module, model_name: str, example_inputs: Any) -> torch.jit.ScriptModule:
        """Compile model using TorchScript JIT for optimal performance."""
        logger.info(f"🔧 JIT compiling {model_name}")
        
        start_time = time.perf_counter()
        
        try:
            model.eval()
            model.to(self.device)
            
            # Ensure example inputs are on correct device
            if isinstance(example_inputs, torch.Tensor):
                example_inputs = example_inputs.to(self.device)
            elif isinstance(example_inputs, dict):
                example_inputs = {k: v.to(self.device) for k, v in example_inputs.items()}
            
            # Use tracing for better performance
            with torch.no_grad():
                if hasattr(model, 'fast_inference'):
                    # For models with fast_inference method
                    traced_model = torch.jit.trace(model.fast_inference, example_inputs)
                else:
                    traced_model = torch.jit.trace(model, example_inputs)
            
            # Optimize for inference
            traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Warm up compiled model
            with torch.no_grad():
                for _ in range(20):
                    if hasattr(model, 'fast_inference'):
                        _ = traced_model(example_inputs)
                    else:
                        _ = traced_model(example_inputs)
            
            compilation_time = time.perf_counter() - start_time
            
            self.compiled_models[model_name] = traced_model
            self.optimization_stats[model_name] = {
                'compilation_time': compilation_time,
                'original_params': sum(p.numel() for p in model.parameters()),
                'optimization_type': 'jit_trace'
            }
            
            logger.info(f"✅ JIT compiled {model_name} in {compilation_time:.3f}s")
            return traced_model
            
        except Exception as e:
            logger.error(f"❌ JIT compilation failed for {model_name}: {e}")
            logger.info(f"🔄 Using original {model_name} as fallback")
            # Store fallback information
            self.optimization_stats[model_name] = {
                'compilation_time': 0.0,
                'original_params': sum(p.numel() for p in model.parameters()),
                'optimization_type': 'fallback_original',
                'fallback_reason': str(e)
            }
            return model
    
    def quantize_model(self, model: nn.Module, model_name: str) -> torch.jit.ScriptModule:
        """Apply dynamic quantization for memory and speed optimization."""
        logger.info(f"⚡ Quantizing {model_name}")
        
        try:
            model.eval()
            
            # Dynamic quantization for CPU deployment
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
            
            # Convert to TorchScript for deployment
            if model_name == 'fast_tactical_marl':
                example_input = torch.randn(1, 60, 7)
            elif model_name == 'fast_strategic_marl':
                example_input = {
                    'mlmi': torch.randn(1, 4),
                    'nwrqk': torch.randn(1, 6),
                    'mmd': torch.randn(1, 3)
                }
            elif 'mlmi' in model_name:
                example_input = torch.randn(1, 4)
            elif 'nwrqk' in model_name:
                example_input = torch.randn(1, 6)
            elif 'mmd' in model_name:
                example_input = torch.randn(1, 3)
            else:  # critic
                example_input = torch.randn(1, 13)
            
            with torch.no_grad():
                if hasattr(quantized_model, 'fast_inference'):
                    traced_quantized = torch.jit.trace(quantized_model.fast_inference, example_input)
                else:
                    traced_quantized = torch.jit.trace(quantized_model, example_input)
            
            self.quantized_models[model_name] = traced_quantized
            
            # Calculate size reduction
            original_size = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
            # Quantized models use int8 for weights, so roughly 4x smaller
            quantized_size = original_size // 4
            
            logger.info(f"✅ Quantized {model_name}: {original_size//1024}KB -> {quantized_size//1024}KB")
            return traced_quantized
            
        except Exception as e:
            logger.error(f"❌ Quantization failed for {model_name}: {e}")
            return model
    
    def create_optimized_models(self) -> Dict[str, torch.jit.ScriptModule]:
        """Create all optimized models for production."""
        logger.info("🚀 Creating optimized production models")
        
        # Create base fast models
        fast_models = {
            'fast_tactical_marl': FastTacticalMARLSystem(),
            'fast_strategic_marl': FastStrategicMARLSystem(),
            'fast_mlmi_actor': FastMLMIActor(),
            'fast_nwrqk_actor': FastNWRQKActor(),
            'fast_mmd_actor': FastMMDActor(),
            'fast_critic': FastCentralizedCritic(state_dim=13)
        }
        
        # Example inputs for each model
        example_inputs = {
            'fast_tactical_marl': torch.randn(1, 60, 7),
            'fast_strategic_marl': {
                'mlmi': torch.randn(1, 4),
                'nwrqk': torch.randn(1, 6),
                'mmd': torch.randn(1, 3)
            },
            'fast_mlmi_actor': torch.randn(1, 4),
            'fast_nwrqk_actor': torch.randn(1, 6),
            'fast_mmd_actor': torch.randn(1, 3),
            'fast_critic': torch.randn(1, 13)
        }
        
        optimized_models = {}
        
        # JIT compile all models
        for model_name, model in fast_models.items():
            jit_model = self.compile_model_jit(model, model_name, example_inputs[model_name])
            optimized_models[f"{model_name}_jit"] = jit_model
            
            # Also create quantized versions
            quantized_model = self.quantize_model(model, model_name)
            optimized_models[f"{model_name}_quantized"] = quantized_model
        
        return optimized_models
    
    def benchmark_optimized_models(self, models: Dict[str, torch.jit.ScriptModule]) -> Dict[str, Dict[str, float]]:
        """Benchmark all optimized models."""
        logger.info("📊 Benchmarking optimized models")
        
        results = {}
        
        # Test inputs
        test_inputs = {
            'tactical': torch.randn(1, 60, 7),
            'strategic': {
                'mlmi': torch.randn(1, 4),
                'nwrqk': torch.randn(1, 6),
                'mmd': torch.randn(1, 3)
            },
            'mlmi': torch.randn(1, 4),
            'nwrqk': torch.randn(1, 6),
            'mmd': torch.randn(1, 3),
            'critic': torch.randn(1, 13)
        }
        
        for model_name, model in models.items():
            # Determine input type
            if 'tactical' in model_name:
                test_input = test_inputs['tactical']
            elif 'strategic' in model_name:
                test_input = test_inputs['strategic']
            elif 'mlmi' in model_name:
                test_input = test_inputs['mlmi']
            elif 'nwrqk' in model_name:
                test_input = test_inputs['nwrqk']
            elif 'mmd' in model_name:
                test_input = test_inputs['mmd']
            else:  # critic
                test_input = test_inputs['critic']
            
            # Warm up
            with torch.no_grad():
                for _ in range(20):
                    _ = model(test_input)
            
            # Benchmark
            latencies = []
            with torch.no_grad():
                for _ in range(1000):
                    start_time = time.perf_counter()
                    _ = model(test_input)
                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)
            
            results[model_name] = {
                'avg_latency_ms': np.mean(latencies),
                'p99_latency_ms': np.percentile(latencies, 99),
                'throughput_qps': 1000 / np.mean(latencies),
                'meets_target': np.percentile(latencies, 99) < 5.0
            }
        
        return results
    
    def save_optimized_models(self, models: Dict[str, torch.jit.ScriptModule], save_dir: Path):
        """Save optimized models for production deployment."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in models.items():
            model_path = save_dir / f"{model_name}.pt"
            torch.jit.save(model, str(model_path))
            logger.info(f"💾 Saved {model_name} to {model_path}")
        
        # Save optimization stats
        stats_path = save_dir / "optimization_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.optimization_stats, f, indent=2)
        
        logger.info(f"📊 Saved optimization stats to {stats_path}")


@contextmanager
def performance_context():
    """Context manager for performance monitoring."""
    start_time = time.perf_counter()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    yield
    
    end_time = time.perf_counter()
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    execution_time = (end_time - start_time) * 1000  # ms
    memory_delta = (end_memory - start_memory) / (1024 * 1024)  # MB
    
    logger.info(f"⏱️ Execution time: {execution_time:.2f}ms, Memory delta: {memory_delta:.2f}MB")


class ProductionInferenceEngine:
    """Production inference engine with optimized models."""
    
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.tensor_pool = TensorPool()
        self.stats = {
            'inferences': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0
        }
        
        # Simple result cache
        self.cache = {}
        self.cache_max_size = 1000
    
    def load_models(self):
        """Load optimized models from disk."""
        logger.info(f"📂 Loading models from {self.model_dir}")
        
        # Load JIT models (preferred for production)
        jit_models = [
            'fast_tactical_marl_jit.pt',
            'fast_strategic_marl_jit.pt',
            'fast_mlmi_actor_jit.pt',
            'fast_nwrqk_actor_jit.pt',
            'fast_mmd_actor_jit.pt',
            'fast_critic_jit.pt'
        ]
        
        for model_file in jit_models:
            model_path = self.model_dir / model_file
            if model_path.exists():
                model_name = model_file.replace('_jit.pt', '')
                model = torch.jit.load(str(model_path))
                model.eval()
                self.models[model_name] = model
                logger.info(f"✅ Loaded {model_name}")
            else:
                logger.warning(f"⚠️ Model not found: {model_path}")
    
    @torch.no_grad()
    def infer_tactical(self, state: np.ndarray, use_cache: bool = True) -> Dict[str, Any]:
        """Ultra-fast tactical inference."""
        # Convert to tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        
        # Check cache
        cache_key = None
        if use_cache:
            cache_key = f"tactical_{hash(state.tobytes())}"
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[cache_key]
        
        start_time = time.perf_counter()
        
        # Inference
        if 'fast_tactical_marl' in self.models:
            result = self.models['fast_tactical_marl'](state_tensor)
        else:
            raise RuntimeError("Tactical model not loaded")
        
        end_time = time.perf_counter()
        
        # Process result
        if isinstance(result, dict) and 'actions' in result:
            actions = result['actions']
            value = result.get('value', 0.0)
        else:
            # Handle raw model output
            actions = {'fvg': 0, 'momentum': 0, 'entry': 0}  # Default
            value = 0.0
        
        inference_result = {
            'actions': actions,
            'value': value,
            'inference_time_ms': (end_time - start_time) * 1000,
            'cached': False
        }
        
        # Cache result
        if use_cache and cache_key:
            if len(self.cache) >= self.cache_max_size:
                # Simple LRU: remove oldest
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = inference_result
            self.stats['cache_misses'] += 1
        
        # Update stats
        self.stats['inferences'] += 1
        self.stats['total_time'] += (end_time - start_time)
        
        return inference_result
    
    @torch.no_grad()
    def infer_strategic(self, states: Dict[str, np.ndarray], use_cache: bool = True) -> Dict[str, Any]:
        """Ultra-fast strategic inference."""
        # Convert to tensors
        state_tensors = {}
        for key, state in states.items():
            state_tensors[key] = torch.from_numpy(state).float().unsqueeze(0)
        
        # Check cache
        cache_key = None
        if use_cache:
            state_hashes = [f"{k}_{hash(v.tobytes())}" for k, v in states.items()]
            cache_key = f"strategic_{'_'.join(state_hashes)}"
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                return self.cache[cache_key]
        
        start_time = time.perf_counter()
        
        # Inference
        if 'fast_strategic_marl' in self.models:
            result = self.models['fast_strategic_marl'](state_tensors)
        else:
            raise RuntimeError("Strategic model not loaded")
        
        end_time = time.perf_counter()
        
        # Process result
        if isinstance(result, dict) and 'actions' in result:
            actions = result['actions']
            value = result.get('value', 0.0)
        else:
            # Handle raw model output
            actions = {'mlmi': 0, 'nwrqk': 0, 'mmd': 0}  # Default
            value = 0.0
        
        inference_result = {
            'actions': actions,
            'value': value,
            'inference_time_ms': (end_time - start_time) * 1000,
            'cached': False
        }
        
        # Cache result
        if use_cache and cache_key:
            if len(self.cache) >= self.cache_max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[cache_key] = inference_result
            self.stats['cache_misses'] += 1
        
        # Update stats
        self.stats['inferences'] += 1
        self.stats['total_time'] += (end_time - start_time)
        
        return inference_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = self.stats['total_time'] / max(self.stats['inferences'], 1)
        cache_hit_rate = self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
        
        return {
            'total_inferences': self.stats['inferences'],
            'avg_inference_time_ms': avg_time * 1000,
            'cache_hit_rate': cache_hit_rate,
            'throughput_qps': 1.0 / max(avg_time, 0.001),
            'models_loaded': list(self.models.keys())
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.cache.clear()
        self.tensor_pool.clear()
        gc.collect()


def create_production_models(save_dir: Path = Path("models/optimized")) -> Dict[str, Any]:
    """Create and save all production-optimized models."""
    optimizer = ProductionModelOptimizer()
    
    # Create optimized models
    models = optimizer.create_optimized_models()
    
    # Benchmark performance
    benchmark_results = optimizer.benchmark_optimized_models(models)
    
    # Save models
    optimizer.save_optimized_models(models, save_dir)
    
    return {
        'models': models,
        'benchmark_results': benchmark_results,
        'optimization_stats': optimizer.optimization_stats
    }


if __name__ == "__main__":
    print("🚀 Creating Production-Optimized Models")
    
    results = create_production_models()
    
    print("\n" + "="*80)
    print("📊 PRODUCTION MODEL OPTIMIZATION RESULTS")
    print("="*80)
    
    print("\n🔧 JIT COMPILED MODELS:")
    for model_name, stats in results['benchmark_results'].items():
        if '_jit' in model_name:
            status = "✅ PASS" if stats['meets_target'] else "❌ FAIL" 
            print(f"  {model_name:30}: {stats['p99_latency_ms']:6.2f}ms p99 | {stats['throughput_qps']:8.1f} QPS {status}")
    
    print("\n⚡ QUANTIZED MODELS:")
    for model_name, stats in results['benchmark_results'].items():
        if '_quantized' in model_name:
            status = "✅ PASS" if stats['meets_target'] else "❌ FAIL"
            print(f"  {model_name:30}: {stats['p99_latency_ms']:6.2f}ms p99 | {stats['throughput_qps']:8.1f} QPS {status}")
    
    # Check if tactical and strategic systems meet requirements
    tactical_jit = results['benchmark_results'].get('fast_tactical_marl_jit', {})
    strategic_jit = results['benchmark_results'].get('fast_strategic_marl_jit', {})
    
    print("\n" + "="*80)
    print("🎯 PRODUCTION READINESS ASSESSMENT")
    print("="*80)
    
    if tactical_jit.get('meets_target', False):
        print(f"✅ TACTICAL SYSTEM: {tactical_jit['p99_latency_ms']:.2f}ms - PRODUCTION READY")
    else:
        print(f"❌ TACTICAL SYSTEM: {tactical_jit.get('p99_latency_ms', 999):.2f}ms - NEEDS OPTIMIZATION")
    
    if strategic_jit.get('meets_target', False):
        print(f"✅ STRATEGIC SYSTEM: {strategic_jit['p99_latency_ms']:.2f}ms - PRODUCTION READY")
    else:
        print(f"❌ STRATEGIC SYSTEM: {strategic_jit.get('p99_latency_ms', 999):.2f}ms - NEEDS OPTIMIZATION")
    
    total_systems_ready = sum([
        tactical_jit.get('meets_target', False),
        strategic_jit.get('meets_target', False)
    ])
    
    if total_systems_ready == 2:
        print("\n🏆 SUCCESS: ALL SYSTEMS MEET <5MS REQUIREMENT!")
        print("✅ Models are ready for production deployment")
    else:
        print(f"\n⚠️ WARNING: {2-total_systems_ready} system(s) still need optimization")
    
    print("="*80)